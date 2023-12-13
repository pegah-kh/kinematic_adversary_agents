from __future__ import annotations

import logging
from typing import List, Dict, Optional
from omegaconf import DictConfig
import torch
import numpy as np
from hydra.utils import instantiate
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import pdb
from scipy.ndimage import zoom
import pdb


from nuplan_gpu_work.planning.simulation.adversary_optimizer.abstract_optimizer import AbstractOptimizer
from nuplan_gpu_work.planning.simulation.adversary_optimizer.agent_tracker.agent_lqr_tracker import LQRTracker 
from nuplan_gpu_work.planning.simulation.motion_model.bicycle_model import BicycleModel
from nuplan_gpu_work.planning.simulation.cost.king_costs import RouteDeviationCostRasterized, BatchedPolygonCollisionCost, DummyCost, DummyCost_FixedPoint

from nuplan.common.maps.maps_datatypes import SemanticMapLayer
from nuplan.common.actor_state.tracked_objects import TrackedObject
from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType
from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.actor_state.waypoint import Waypoint
from nuplan.common.actor_state.oriented_box import OrientedBox
from nuplan.common.actor_state.state_representation import StateSE2, StateVector2D
from nuplan_gpu_work.planning.scenario_builder.nuplan_db_modif.nuplan_scenario import NuPlanScenarioModif
from nuplan.planning.simulation.planner.abstract_planner import AbstractPlanner
from nuplan.planning.simulation.runner.runner_report import RunnerReport
from nuplan.planning.simulation.simulation import Simulation
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from nuplan.planning.simulation.observation.observation_type import DetectionsTracks


logger = logging.getLogger(__name__)
PIXELS_PER_METER = 10
device = 'cuda' if torch.cuda.is_available() else 'cpu'



# Gradient hooks to visualize the gradient flow for throttle and steer separately
# how to do it for each agent?? let's say we have the gradient for each step of the optimization for each of the agents.
# then how to visualize it?? just show the numbers
# but there are throttles and steers for all the steps in the simulation!! how to manage to observe them for all the timesteps??

def dummy(grad):
    print('dummy dummy *************')
    
# from https://stackoverflow.com/questions/17458580/embedding-small-plots-inside-subplots-in-matplotlib
def add_subplot_axes(ax,rect,facecolor='w'):
    fig = plt.gcf()
    box = ax.get_position()
    width = box.width
    height = box.height
    inax_position  = ax.transAxes.transform(rect[0:2])
    transFigure = fig.transFigure.inverted()
    infig_position = transFigure.transform(inax_position)    
    x = infig_position[0]
    y = infig_position[1]
    width *= rect[2]
    height *= rect[3]  # <= Typo was here
    #subax = fig.add_axes([x,y,width,height],facecolor=facecolor)  # matplotlib 2.0+
    subax = fig.add_axes([x,y,width,height], facecolor=facecolor)
    x_labelsize = subax.get_xticklabels()[0].get_size()
    y_labelsize = subax.get_yticklabels()[0].get_size()
    x_labelsize *= rect[2]**0.5
    y_labelsize *= rect[3]**0.5
    subax.xaxis.set_tick_params(labelsize=x_labelsize)
    subax.yaxis.set_tick_params(labelsize=y_labelsize)
    return subax

# the OptimizationKING calls the simulation runner for iterations
class OptimizationKING():
    """
    Manager which executes multiple simulations with the same planner
    """

    def __init__(self, simulation: Simulation, planner: AbstractPlanner, tracker: DictConfig, motion_model: BicycleModel, max_opt_iterations: int):

        # super.__init__(self, simulation, planner)
        self._simulation = simulation
        self._planner = planner
        
        # motion model
        self._motion_model = motion_model

        
        # the maximum number of optimization iterations
        self._max_opt_iterations = max_opt_iterations


        # to check the dimensions later
        self._number_states:int = self._simulation._time_controller.number_of_iterations()
        self._number_actions:int = self._number_states - 1
        # use this trajectory sampling to get the initial observations, and obtain the actions accordingly
        self._observation_trajectory_sampling = TrajectorySampling(num_poses=self._number_actions, time_horizon=self._simulation.scenario.duration_s.time_s)
        self._horizon = self._number_actions
        self._number_agents: int = None

        # tracker
        self._tracker = LQRTracker(**tracker, discretization_time=self._observation_trajectory_sampling.interval_length)

        assert self._observation_trajectory_sampling.interval_length==self._tracker._discretization_time, 'tracker discretization time of the tracker is not equal to interval_length of the sampled trajectory'
        

        tracked_objects: DetectionsTracks = self._simulation._observations.get_observation_at_iteration(0, self._observation_trajectory_sampling)
        agents: List[TrackedObject] = tracked_objects.tracked_objects.get_tracked_objects_of_type(TrackedObjectType.VEHICLE)
        self._agents = agents
        num_agents = len(agents)
        self._number_agents = num_agents



        # the cost functions
        self.col_cost_fn = BatchedPolygonCollisionCost(num_agents=self._number_agents)
        self.adv_rd_cost = RouteDeviationCostRasterized(num_agents=self._number_agents)
        # self.dummy_cost_fn = DummyCost_FixedPoint(num_agents=self._number_agents)
        self.dummy_cost_fn = DummyCost(num_agents=self._number_agents)
        # to accumulate the cost functions
        self._cost_dict = {"ego_col": [], "adv_rd": [], "adv_col": [], "dummy": []}

        # to keep track of the states in one simulation
        self.state_buffer = []
        # to keep track of the evolution of states from one optimization round to the other
        self.whole_state_buffers = []


        self.plot_yaw_gt = []
        self.plot_gt_velocities = []
        self.plot_throttle_track = []
        self.ego_y_direction_vel = []
        self.that_agent_y_velocity = []

    def reset(self) -> None:
        '''
        inherited method.
        '''
        self._current_optimization_on_iteration = 1
        self._simulation.reset()
        self._simulation._ego_controller.reset()
        self._simulation._observations.reset()
        self._optimizer.zero_grad(set_to_none=True)
        self.init_dynamic_states()

        # new costs
        self._cost_dict = {"ego_col": [], "adv_rd": [], "adv_col": [], "dummy": []}
        self.state_buffer = []

        # gradient hooks to keep track of the evolution of the gradients of thottle and steering during the optimization
        self.throttle_gradients = [np.zeros(self._number_actions) for _ in range(self._number_agents)]
        self.steer_gradients = [np.zeros(self._number_actions) for _ in range(self._number_agents)]



    def gradient_hook_throttle(self, row_idx):
        def hook(grad):
            # This hook will be called during the backward pass for the row
            print('hook throttle ', grad.clone().detach().numpy())
            self.throttle_gradients[row_idx] += grad.clone().detach().numpy()
        return hook

    def gradient_hook_steer(self, row_idx):
        def hook(grad):
            # This hook will be called during the backward pass for the row
            print('hook steer ', grad.clone().detach().numpy())
            self.steer_gradients[row_idx] += grad.clone().detach().numpy()
        return hook
    

    def scenario(self) -> NuPlanScenarioModif:
        """
        :return: Get the scenario.
        """
        return self._simulation.scenario

    def planner(self) -> AbstractPlanner:
        """
        :return: Get a planner.
        """
        return self._planner



    def init_dynamic_states(self):

        self._states = {'pos': torch.zeros(self._number_agents, 2).to(device=device), 'yaw': torch.zeros(self._number_agents, 1).to(device=device), 'steering_angle': torch.zeros(self._number_agents, 1).to(device=device), 'vel': torch.zeros(self._number_agents, 2).to(device=device), 'accel': torch.zeros(self._number_agents, 1).to(device=device)}
        for idx, tracked_agent in enumerate(self._agents):

            self._states['pos'][idx] = torch.tensor([tracked_agent.predictions[0].valid_waypoints[0].x, tracked_agent.predictions[0].valid_waypoints[0].y], device=device)
            self._states['yaw'][idx] = torch.tensor([tracked_agent.predictions[0].valid_waypoints[0].heading], device=device)
            # self._states['vel'][idx] = torch.tensor([tracked_agent.predictions[0].valid_waypoints[0].velocity.x, tracked_agent.predictions[0].valid_waypoints[0].velocity.y], device=device)
            self._states['vel'][idx] = torch.tensor([0., 0.], dtype=torch.float64, device=device)
            self._states['accel'][idx] = torch.tensor([0.0], dtype=torch.float64, device=device)
            approx_tire_steering_angle = np.arctan(3.089*(tracked_agent.predictions[0].valid_waypoints[1].heading - tracked_agent.predictions[0].valid_waypoints[0].heading)/(self._observation_trajectory_sampling.interval_length*np.hypot(tracked_agent.velocity.x, tracked_agent.velocity.y)+1e-3))
            self._states['steering_angle'][idx] = torch.clamp(torch.tensor([approx_tire_steering_angle], device=device), min=-torch.pi/3, max=torch.pi/3)
            
    

    def check_map(self):

        '''
            if render_map_raster:
        map_raster, map_translation, map_scale = lidarpc_rec.ego_pose.get_map_crop(  # type: ignore
            maps_db=db.maps_db,  # type: ignore
            xrange=xrange,
            yrange=yrange,
            map_layer_name="drivable_area",
            rotate_face_up=True,
        )
        ax.imshow(map_raster[::-1, :], cmap="gray")
        '''
        print('they hey hey hey ')
        resized_image = zoom(self._data_nondrivable_map, zoom=0.0001/self._map_resolution)
        print('they hey hey hey ')

        plt.figure(figsize = (10,5))
        plt.imshow(resized_image, interpolation='nearest')
        plt.gcf()
        plt.savefig('/home/kpegah/workspace/SIMPLE/map_check.png')
        plt.show()
        plt.clf()






    # the simulation runner should call the optimizer
    # but does optimizer need to call simulation runner??
    def initialize_old_one(self) -> None:
        """
        To obtain the initial actions using the lqr controller, adn from the trajectory.
        Inherited method.
        """

        self._fixed_point = np.array([self._simulation._ego_controller.get_state().center.x,self._simulation._ego_controller.get_state().center.y])

        # setting the trajectory_sampling value of the observation to the correct value
        self._simulation._observations.set_traj_sampling(self._observation_trajectory_sampling)
        
        self._current_optimization_on_iteration = 1
        num_agents = self._number_agents


                
        # defining an indexing for the agents
        self._agent_indexes: Dict[str, int] = {}
        self._agent_tokens: Dict[int, str] = {}

        # initializing the dimension of the states and the actions
        # the actions parameters
        self._throttle = torch.nn.Parameter(
            torch.zeros(
                num_agents, self._horizon, 1 # the horizon parameter is the same as the numer of actions
                # and is one less than the number of iterations in the simulation
            )
        )
        self._throttle.retain_grad()
        self._steer = torch.nn.Parameter(
            torch.zeros(
                num_agents, self._horizon, 1
            )
        )
        self._steer.retain_grad()

        '''
        # resgistering the hooks for throttle and steer
        for i in range(num_agents):
            self._throttle[i].register_hook(dummy)

        for i in range(num_agents):
            self._steer[i].register_hook(self.gradient_hook_steer(i))

        '''

        self._brake = torch.zeros(
                num_agents, 1
            )
        
        self._actions = {'throttle': self._throttle, 'steer': self._steer, 'brake': self._brake}
        # self._states = {'pos': torch.zeros(num_agents, self._horizon + 1, 2), 'yaw': torch.zeros(num_agents, self._horizon + 1, 1), 'steering_angle': torch.zeros(num_agents, self._horizon + 1, 1), 'vel': torch.zeros(num_agents, self._horizon + 1, 2), 'accel': torch.zeros(num_agents, self._horizon + 1, 2)}
        self.init_dynamic_states()

        optimization_params = [self._throttle, self._steer]
        self._optimizer = torch.optim.Adam(optimization_params, lr=0.005)

        # FOR PLOTTING ******************* OPEN
        all_agents_position_x = []
        all_agents_position_y = []
        all_agents_heading = []
        all_agents_velocity = []
        all_agents_xy_velocity = []
        ego_state = self._simulation._ego_controller.get_state()
        ego_position = [ego_state.center.x, ego_state.center.y]
        ego_heading = [np.cos(ego_state.center.heading), np.sin(ego_state.center.heading)]
        ego_velocity = [ego_state.dynamic_car_state.center_velocity_2d.x, ego_state.dynamic_car_state.center_velocity_2d.y]
        ego_xy_velocity = [ego_state.dynamic_car_state.center_velocity_2d.x*np.cos(ego_state.center.heading), ego_state.dynamic_car_state.center_velocity_2d.x*np.sin(ego_state.center.heading)]
        # FOR PLOTTING ******************* CLOSE

        
        
        colors = cm.rainbow(np.linspace(0, 1, self._number_agents))

        for idx, tracked_agent in enumerate(self._agents):

           
            # FOR PLOTTING ******************* OPEN
            all_agents_position_x.append(tracked_agent.predictions[0].valid_waypoints[0].x)
            all_agents_position_y.append(tracked_agent.predictions[0].valid_waypoints[0].y)
            all_agents_heading.append([np.cos(tracked_agent.predictions[0].valid_waypoints[0].heading), np.sin(tracked_agent.predictions[0].valid_waypoints[0].heading)])
            all_agents_velocity.append([tracked_agent.predictions[0].valid_waypoints[0].velocity.x, tracked_agent.predictions[0].valid_waypoints[0].velocity.y])
            all_agents_xy_velocity.append([np.cos(tracked_agent.predictions[0].valid_waypoints[0].heading)*tracked_agent.predictions[0].valid_waypoints[0].velocity.x, np.sin(tracked_agent.predictions[0].valid_waypoints[0].heading)*tracked_agent.predictions[0].valid_waypoints[0].velocity.x])
            agent_velocities = []
            # FOR PLOTTING ******************* CLOSE


            self._agent_indexes[tracked_agent.metadata.track_token] = idx
            self._agent_tokens[idx] = tracked_agent.metadata.track_token
        
            

            initial_waypoint = tracked_agent.predictions[0].valid_waypoints[0]
            initial_tire_steering_angle = np.arctan(3.089*(tracked_agent.predictions[0].valid_waypoints[1].heading - tracked_agent.predictions[0].valid_waypoints[0].heading)/(self._observation_trajectory_sampling.interval_length*np.hypot(tracked_agent.velocity.x, tracked_agent.velocity.y)+1e-3))


            # initializing the dynamic state of the agents AT ALL TIME STEPS steps of the horizon using the lqr controller
            with torch.no_grad():
                for _time_step in range(len(tracked_agent.predictions[0].valid_waypoints) - 1):
                    # using the timepoints of the simulation instead of those of predictions
                    throttle, steer = self._tracker.track_trajectory(tracked_agent.predictions[0].valid_waypoints[_time_step].time_point, tracked_agent.predictions[0].valid_waypoints[_time_step+1].time_point, initial_waypoint, tracked_agent.predictions[0].trajectory, initial_steering_angle=initial_tire_steering_angle)
                    self._throttle[idx,_time_step], self._steer[idx,_time_step] = torch.tensor([throttle]), torch.tensor([steer])
                    # updating the initial_waypoint based on the update from the bicycle model
                    if _time_step==0:
                        beginning_state = {_key: _value.clone().detach() for _key, _value in self.get_adv_state(_time_step, id=idx).items()} # B*N*S
                        next_state = self._motion_model.forward(beginning_state, self.get_adv_actions(_time_step, id=idx), track_token=self._agent_tokens[idx], iter=_time_step+1, plotter=False)
                    else:
                        next_state = self._motion_model.forward(next_state, self.get_adv_actions(_time_step, id=idx), track_token=self._agent_tokens[idx], iter=_time_step+1, plotter=False)

                    # FOR CAPTURING INITIAL STATES *********************** OPEN
                    current_color = colors[idx]
                    plt.quiver(*np.array([next_state['pos'][0,0,0], next_state['pos'][0,0,1]]), np.cos(next_state['yaw'][0,0,0]), np.sin(next_state['yaw'][0,0,0]), color=current_color, scale=60)
                    # FOR CAPTURING INITIAL STATES *********************** CLOSE


                    initial_waypoint = Waypoint(time_point=tracked_agent.predictions[0].valid_waypoints[_time_step+1].time_point, oriented_box=OrientedBox.from_new_pose(tracked_agent.box, StateSE2(next_state['pos'][0,0,0], next_state['pos'][0,0,1], next_state['yaw'][0,0,0])), velocity=StateVector2D(next_state['vel'][0,0,0], next_state['vel'][0,0,1]))
                    initial_tire_steering_angle = next_state['steering_angle']
                    # FOR PLOTTING ******************* OPEN
                    agent_velocities.append({'velocity' : tracked_agent.predictions[0].valid_waypoints[_time_step].velocity, 'heading' : tracked_agent.predictions[0].valid_waypoints[_time_step].heading, 'throttle': throttle, 'steer': steer})
                    if tracked_agent.track_token in ['6e0f1dbf087e570f'] and _time_step>0:
                        self.that_agent_y_velocity.append(tracked_agent.predictions[0].valid_waypoints[_time_step+1].velocity.y)
                        self.plot_yaw_gt.append(tracked_agent.predictions[0].valid_waypoints[_time_step+1].heading)
                        self.plot_gt_velocities.append(np.hypot(tracked_agent.predictions[0].valid_waypoints[_time_step+1].velocity.x, tracked_agent.predictions[0].valid_waypoints[_time_step+1].velocity.y))
                        self.plot_throttle_track.append(throttle)
                    # FOR PLOTTING ******************* CLOSE
                

                # TODO : here we can fill the next time steps for which otherwise the throttle and steer will be zero
        plt.gcf()
        plt.savefig(f'/home/kpegah/workspace/{self._simulation.scenario.scenario_name}_initialized_scenario_bm.png')
        plt.show()
        plt.clf()


        # make two plots one for the heading and the other for the velocities of all the agents and the ego
        all_positions = np.array([all_agents_position_x, all_agents_position_y])
        all_agents_velocity.append(ego_velocity)
        origin = np.array(all_positions) # origin point
        V_all_agents = np.array(all_agents_velocity)

        plt.quiver(*origin, V_all_agents[:-1,0], V_all_agents[:-1,1], color='b', scale=50)
        plt.quiver(*np.array([[ego_position[0]], [ego_position[1]]]), V_all_agents[-1:,0], V_all_agents[-1:,1], color='r', scale=30)
        plt.gcf()
        plt.savefig('/home/kpegah/workspace/agents_velocities.png')
        plt.show()
        plt.clf()
        # how to change the heading??
        all_agents_heading.append(ego_heading)
        V_all_agents = np.array(all_agents_heading)
        plt.quiver(*origin, V_all_agents[:-1,0], V_all_agents[:-1,1], color='b', scale=20)
        plt.quiver(*np.array([[ego_position[0]], [ego_position[1]]]), V_all_agents[-1:,0], V_all_agents[-1:,1], color='r', scale=20)
        plt.gcf()
        plt.savefig('/home/kpegah/workspace/agents_headings.png')
        plt.show()
        plt.clf()
        # how to change the heading??
        all_agents_xy_velocity.append(ego_xy_velocity)
        V_all_agents = np.array(all_agents_xy_velocity)

        plt.quiver(*origin, V_all_agents[:-1,0], V_all_agents[:-1,1], color='b', scale=20)
        plt.quiver(*np.array([[ego_position[0]], [ego_position[1]]]), V_all_agents[-1:,0], V_all_agents[-1:,1], color='r', scale=20)
        plt.gcf()
        plt.savefig('/home/kpegah/workspace/agents_xy_velocities.png')
        plt.show()
        plt.clf()
  
        self.print_initial_actions(id=0)
        # initialize the ego state
        self._ego_state = {'pos': torch.zeros(2).to(device=device), 'yaw': torch.zeros(1).to(device=device), 'vel': torch.zeros(2).to(device=device)}
        
        # initialize the map of the drivable area
        # TODO
        
        # drivable_map_layer = self._simulation.scenario.map_api.get_raster_map_layer(SemanticMapLayer.DRIVABLE_AREA).data
        # self._nondrivable_map_layer = torch.tensor(np.logical_not(drivable_map_layer))
        # print(self._nondrivable_map_layer.shape)
        # self._map_offset = 10


        # initialize the dimension of the vehicles
        self.ego_extent = torch.tensor(
            [(4.049+1.127)/2,
             14.85],
            device=device,
            dtype=torch.float64
        ).view(1, 1, 2).expand(1, 1, 2)

        self.adv_extent = torch.tensor(
            [(4.049+1.127)/2,
             14.85],
            device=device,
            dtype=torch.float64
        ).view(1, 1, 2).expand(1, num_agents, 2)




    # the simulation runner should call the optimizer
    # but does optimizer need to call simulation runner??
    def initialize(self) -> None:
        """
        To obtain the initial actions using the lqr controller, adn from the trajectory.
        Inherited method.
        """


        print(f"Using device ******* ******** ******** *********: {device}")

        # self._fixed_point = np.array([664195.,3996283.])
        self._fixed_point = np.array([self._simulation._ego_controller.get_state().center.x,self._simulation._ego_controller.get_state().center.y])

        self._constructed_map_around_ego = True
        #self._fixed_point = np.array([self._simulation._ego_controller.get_state().center.x,self._simulation._ego_controller.get_state().center.y])


        # setting the trajectory_sampling value of the observation to the correct value
        self._simulation._observations.set_traj_sampling(self._observation_trajectory_sampling)
        
        self._current_optimization_on_iteration = 1
        num_agents = self._number_agents
                
        # defining an indexing for the agents
        self._agent_indexes: Dict[str, int] = {}
        self._agent_tokens: Dict[int, str] = {}

        # initializing the dimension of the states and the actions
        # the actions parameters
        self._throttle = torch.nn.Parameter(
            torch.zeros(
                num_agents, self._horizon, 1, # the horizon parameter is the same as the numer of actions
                # and is one less than the number of iterations in the simulation
                device=device
            ),
        )
        self._throttle.retain_grad()
        self._steer = torch.nn.Parameter(
            torch.zeros(
                num_agents, self._horizon, 1,
                device=device
            )
        )
        self._steer.retain_grad()

        self._brake = torch.zeros(
                num_agents, 1
            )
        print('the device of the parameters ', self._steer.device, '    ', self._throttle.device)
        self._actions = {'throttle': self._throttle, 'steer': self._steer, 'brake': self._brake}
        # self._states = {'pos': torch.zeros(num_agents, self._horizon + 1, 2), 'yaw': torch.zeros(num_agents, self._horizon + 1, 1), 'steering_angle': torch.zeros(num_agents, self._horizon + 1, 1), 'vel': torch.zeros(num_agents, self._horizon + 1, 2), 'accel': torch.zeros(num_agents, self._horizon + 1, 2)}
        self.init_dynamic_states()

        
        colors = cm.rainbow(np.linspace(0, 1, self._number_agents))


        # initializing the parameters by the tracker and updating the current state using the bicycle model
        # bu we should add another forward function to the motion model to update all the states together
        for idx, tracked_agent in enumerate(self._agents):

           
            self._agent_indexes[tracked_agent.metadata.track_token] = idx
            self._agent_tokens[idx] = tracked_agent.metadata.track_token

            print('treating the agent ', tracked_agent.track_token)
        
            

            initial_waypoint = tracked_agent.predictions[0].valid_waypoints[0]
            initial_tire_steering_angle = np.arctan(3.089*(tracked_agent.predictions[0].valid_waypoints[1].heading - tracked_agent.predictions[0].valid_waypoints[0].heading)/(self._observation_trajectory_sampling.interval_length*np.hypot(tracked_agent.velocity.x, tracked_agent.velocity.y)+1e-3))


            # initializing the dynamic state of the agents AT ALL TIME STEPS steps of the horizon using the lqr controller
            with torch.no_grad():
                for _time_step in range(len(tracked_agent.predictions[0].valid_waypoints) - 1):
                    # using the timepoints of the simulation instead of those of predictions
                    throttle, steer = self._tracker.track_trajectory(tracked_agent.predictions[0].valid_waypoints[_time_step].time_point, tracked_agent.predictions[0].valid_waypoints[_time_step+1].time_point, initial_waypoint, tracked_agent.predictions[0].trajectory, initial_steering_angle=initial_tire_steering_angle)
                    # self._throttle[idx,_time_step], self._steer[idx,_time_step] = torch.Tensor([throttle]), torch.Tensor([steer])
                    self._throttle[idx,_time_step], self._steer[idx,_time_step] = torch.tensor([0.0], dtype=torch.float64).to(device), torch.tensor([steer]).to(device)
                    # updating the initial_waypoint based on the update from the bicycle model
                    if _time_step==0:
                        beginning_state = {_key: _value.clone().detach().to(device) for _key, _value in self.get_adv_state(_time_step, id=idx).items()} # B*N*S
                        next_state = self._motion_model.forward(beginning_state, self.get_adv_actions(_time_step, id=idx), track_token=self._agent_tokens[idx], iter=_time_step+1, plotter=False)
                    else:
                        next_state = self._motion_model.forward(next_state, self.get_adv_actions(_time_step, id=idx), track_token=self._agent_tokens[idx], iter=_time_step+1, plotter=False)

                    # FOR CAPTURING INITIAL STATES *********************** OPEN
                    current_color = colors[idx]
                    plt.quiver(*np.array([next_state['pos'].cpu()[0,0,0], next_state['pos'].cpu()[0,0,1]]), np.cos(next_state['yaw'].cpu()[0,0,0]), np.sin(next_state['yaw'].cpu()[0,0,0]), color=current_color, scale=60)
                    # FOR CAPTURING INITIAL STATES *********************** CLOSE


                    initial_waypoint = Waypoint(time_point=tracked_agent.predictions[0].valid_waypoints[_time_step+1].time_point, oriented_box=OrientedBox.from_new_pose(tracked_agent.box, StateSE2(next_state['pos'].cpu()[0,0,0], next_state['pos'].cpu()[0,0,1], next_state['yaw'].cpu()[0,0,0])), velocity=StateVector2D(next_state['vel'].cpu()[0,0,0], next_state['vel'].cpu()[0,0,1]))
                    initial_tire_steering_angle = next_state['steering_angle'].cpu()

                for _time_step in range(len(tracked_agent.predictions[0].valid_waypoints) - 1, self._horizon):
                    # using the timepoints of the simulation instead of those of predictions
                    throttle, steer = self._tracker.track_trajectory(tracked_agent.predictions[0].valid_waypoints[-2].time_point, tracked_agent.predictions[0].valid_waypoints[-1].time_point, initial_waypoint, tracked_agent.predictions[0].trajectory, initial_steering_angle=initial_tire_steering_angle)
                    # self._throttle[idx,_time_step], self._steer[idx,_time_step] = torch.Tensor([throttle]), torch.Tensor([steer])
                    self._throttle[idx,_time_step], self._steer[idx,_time_step] = torch.tensor([0.0], dtype=torch.float64).to(device), torch.tensor([steer]).to(device)
                    # updating the initial_waypoint based on the update from the bicycle model
                    next_state = self._motion_model.forward(next_state, self.get_adv_actions(_time_step, id=idx), track_token=self._agent_tokens[idx], iter=_time_step+1, plotter=False)

                    # FOR CAPTURING INITIAL STATES *********************** OPEN
                    current_color = colors[idx]
                    plt.quiver(*np.array([next_state['pos'].cpu()[0,0,0], next_state['pos'].cpu()[0,0,1]]), np.cos(next_state['yaw'].cpu()[0,0,0]), np.sin(next_state['yaw'].cpu()[0,0,0]), color=current_color, scale=60)
                    # FOR CAPTURING INITIAL STATES *********************** CLOSE


                    initial_waypoint = Waypoint(time_point=tracked_agent.predictions[0].valid_waypoints[-1].time_point, oriented_box=OrientedBox.from_new_pose(tracked_agent.box, StateSE2(next_state['pos'].cpu()[0,0,0], next_state['pos'].cpu()[0,0,1], next_state['yaw'].cpu()[0,0,0])), velocity=StateVector2D(next_state['vel'].cpu()[0,0,0], next_state['vel'].cpu()[0,0,1]))
                    initial_tire_steering_angle = next_state['steering_angle'].cpu()

        

        optimization_params = [self._throttle, self._steer]
        self._optimizer = torch.optim.Adam(optimization_params, lr=0.01)
        # initialize the ego state
        self._ego_state = {'pos': torch.zeros(2).to(device=device), 'yaw': torch.zeros(1).to(device=device), 'vel': torch.zeros(2).to(device=device)}
        
        # initialize the map of the drivable area
        # TODO

        # initialize the dimension of the vehicles
        self.ego_extent = torch.tensor(
            [(4.049+1.127)/2,
             14.85],
            device=device,
            dtype=torch.float64
        ).view(1, 1, 2).expand(1, 1, 2)

        self.adv_extent = torch.tensor(
            [(4.049+1.127)/2,
             14.85],
            device=device,
            dtype=torch.float64
        ).view(1, 1, 2).expand(1, num_agents, 2)


    def print_initial_actions(self, id:int):
        '''
        :params id:int is the agent for which we want to print actions
        '''

        steers = [self._actions['steer'][id,t,0].data for t in range(self._horizon)]
        throttles = [self._actions['throttle'][id,t,0].data for t in range(self._horizon)]
        print(f'the steers for the agent {id} is {steers}')
        print(f'the throttles for the agent {id} is {throttles}')
        
    # function adapted from https://github.com/autonomousvision/king/blob/main/proxy_simulator/simulator.py
    def get_adv_state(self, current_iteration:int = 0, id:Optional(int) = None):
        """
        Fetches the adversarial agent state. If id is None, the state for all
        adversarial agents is returned. If the id is not None, only the state of
        the corresponding agent is returned.

        :param id: Index of a specific adversarial agent to be returned.

        Returns:
            A dictionary holding tensors for each substate (pos, yaw, velocity in our case)
            in king the returned shape is B x N x S
        """
        adv_state = {}
        # for substate in self._states.keys():
        #     if id == None:
        #         adv_state.update({substate: torch.unsqueeze(self._states[substate][:, current_iteration, ...], dim=0)})
        #     else:
        #         # we index for id+1 since id 0 in the tensor is the ego agent
        #         adv_state.update(
        #             {substate: torch.unsqueeze(self._states[substate][id:id+1, current_iteration, ...], dim=0)}
        #         )

        for substate in self._states.keys():
            if id == None:
                adv_state.update({substate: torch.unsqueeze(self._states[substate][:, ...], dim=0)})
            else:
                # we index for id+1 since id 0 in the tensor is the ego agent
                adv_state.update(
                    {substate: torch.unsqueeze(self._states[substate][id:id+1, ...], dim=0)}
                )

        return adv_state
    
    def get_adv_actions(self, current_iteration:int = 0, id:Optional(int) = None):
        """
        the same as the above function, but to get actions of agents

        returns:
                the actions of adversary agents at the current iteration of shape B x N x S
        """
        adv_action = {}
        for substate in self._actions.keys():
            if id == None:
                if not substate=='brake':
                    adv_action.update({substate: torch.unsqueeze(self._actions[substate][:, current_iteration, ...], dim=0)})
                else:
                    adv_action.update({substate:  torch.tanh(torch.unsqueeze(self._actions[substate], dim=0))})
            else:
                # we index for id+1 since id 0 in the tensor is the ego agent
                if not substate=='brake':
                    adv_action.update(
                        {substate: torch.unsqueeze(self._actions[substate][id:id+1, current_iteration, ...], dim=0)}
                    )
                else:
                    adv_action.update(
                        {substate:  torch.tanh(torch.unsqueeze(self._actions[substate][id:id+1, ...], dim=0))}
                    )
        return adv_action

    def set_adv_state(self, next_state: Dict[str, torch.TensorType] = None, next_iteration:int = 1, id=None):
        """
        Set a new adversarial agent state.

        :param next_state: setting the parameters of the next_state, a dict of type[str, Tnesor]
                           where the Tensor contains the state for each agent
        Returns:
            None
        """

        # for substate in self._states.keys():
        #     if not id:
        #         self._states[substate][:, next_iteration, ...] = next_state[substate][0,:, ...]
        #     else:
        #         # we index for id+1 since id 0 in the tensor is the ego agent
        #         self._states[substate][id:id+1, next_iteration,...] = next_state[substate][0,0:1,...]

        for substate in self._states.keys():
            if id==None:
                self._states[substate] = next_state[substate][0]
            else:
                attached_states = []
                if id>0:
                    attached_states.append(self._states[substate][0:id, ...])
                attached_states.append(next_state[substate][0, 0:1, ...])
                if id+1<self._number_agents:
                    attached_states.append(self._states[substate][id+1:, ...])


                # print('hello substate ', substate)
                # for i in range(len(attached_states)):
                #     print('hello to you ', attached_states[i].size())
                self._states[substate] = torch.cat(
                    attached_states,
                    dim=0,
                )


    # Question: what to do with the very first state? the very first state is always the same...
    # so the update should only be performed after the very first round of simulation??
    def step(self, current_iterarion:int) -> Dict[str, ((float, float), (float, float), float)]:
        '''
        to update the state of the agents at the 'current_iterarion' using the actions taken at step 'current_iterarion-1'.

        in the case where the 'current_iterarion' is 0, the state does simply not get updated.

        return:
                a dictionary where the keys are tokens of agents, and the values are (pos, velocity, yaw)
        '''


        if current_iterarion is None:
            return
        
        # the trajectory even from the very start seems not to reach the trajectory of the ego, are we really registering sth before the opetimization?
        # we do the 
        

        # TODO : updating the list of agents at each iteration
        #        some agents may go inactive and untracked, their predicted trajectory should be filled in the untracked spaces?
        #        or maybe we can only work on the agents that remain tracked during the entire simulation
        

        not_differentiable_state = {}  
        if not current_iterarion==0:
            '''
            for id_tracked_agent in range(self._number_agents):
                
                self.set_adv_state(self._motion_model.forward(self.get_adv_state(current_iterarion-1, id=id_tracked_agent), self.get_adv_actions(current_iterarion-1, id=id_tracked_agent), track_token=self._agent_tokens[id_tracked_agent], iter=current_iterarion), next_iteration=current_iterarion, id=id_tracked_agent)

            '''
            # updating all the states together
            self.set_adv_state(self._motion_model.forward_all(self.get_adv_state(current_iterarion-1), self.get_adv_actions(current_iterarion-1)), next_iteration=current_iterarion)


        current_state = self.get_adv_state(current_iterarion) # Dict[str (type of state), torch.Tensor (idx of agent, ...)]
        agents_pos = current_state['pos'][0].clone().detach().cpu().numpy() # adding [0] to get rid of the batch dimension that we don't want to have in nuplan for now
        agents_vel = current_state['vel'][0].clone().detach().cpu().numpy()
        agents_yaw = current_state['yaw'][0].clone().detach().cpu().numpy()

        # adding the state of the ego to the buffere of states
        ego_state = self._simulation._ego_controller.get_state()
        ego_position = np.array([[ego_state.center.x, ego_state.center.y]])
        ego_heading = np.array([[ego_state.center.heading]])
        ego_velocity = np.array([[ego_state.dynamic_car_state.center_velocity_2d.x, ego_state.dynamic_car_state.center_velocity_2d.y]])

        agents_pos = np.concatenate((agents_pos, ego_position), axis=0)      
        agents_yaw = np.concatenate((agents_yaw, ego_heading), axis=0)    
        agents_vel = np.concatenate((agents_vel, ego_velocity), axis=0)    

        self.state_buffer.append({'pos': agents_pos, 'vel': agents_vel, 'yaw': agents_yaw})


        # print('agents_pos ', agents_pos.shape)
        # print('agents_vel ', agents_vel.shape)
        # print('agents_yaw ', agents_yaw.shape)

        for idx in range(self._number_agents):
            not_differentiable_state[self._agent_tokens[idx]] = ((agents_pos[idx, 0], agents_pos[idx, 1]), (agents_vel[idx, 0], agents_vel[idx, 1]), (agents_yaw[idx, 0]))
        

        self.ego_y_direction_vel.append(self._simulation._ego_controller.get_state().dynamic_car_state.rear_axle_velocity_2d.y)
        return not_differentiable_state
        

    def update_ego_state(self, current_ego_state: EgoState, current_iteration: int):
        '''
        update the state of the ego agent using its trajectory
        :param trajectory: the ego trajectory that is taken at the current iteration
        :param next_iteration: the next iteration of simulation
        '''

        # Do we need to keep the state of the ego for subsequent iterations? KING: they keep it in state_detached https://github.com/autonomousvision/king/blob/6acd2154cd689b3121664e60760378ba484659c6/proxy_simulator/simulator.py#L672
        # but we only keep the state of the ego for which we are going to compute the cost
        assert self._current_optimization_on_iteration==current_iteration, 'we are not optimizing on the right iteration of the simulation'


        self._ego_state['pos'] = torch.tensor([current_ego_state.center.x, current_ego_state.center.y], device=device, dtype=torch.float64)
        self._ego_state['vel'] = torch.tensor([current_ego_state.agent.velocity.x, current_ego_state.agent.velocity.y], device=device, dtype=torch.float64)
        self._ego_state['yaw'] = torch.tensor([current_ego_state.center.heading], device=device, dtype=torch.float64)

        self._current_optimization_on_iteration += 1
    

    def world_to_pix(self, pos):
        pos_px = (pos-self._map_offset) * PIXELS_PER_METER

        return pos_px
    
    @staticmethod
    def coord_to_matrix(coord, yaw):

        return np.array(
            [
                [np.cos(yaw[0]), -np.sin(yaw[0]), 0.0, coord[0]],
                [np.sin(yaw[0]), np.cos(yaw[0]), 0.0, coord[1]],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0]
            ]
        )


    @staticmethod
    def from_matrix(matrix):
 
        assert matrix.shape == (4, 4), f"Expected 3x3 transformation matrix, but input matrix has shape {matrix.shape}"

        return matrix[0, 3], matrix[1, 3], np.arctan2(matrix[1, 0], matrix[0, 0])



    def compute_current_cost(self, current_iteration: int):
        '''
        - find the initial actions from the existing trajectories for background agents
        - create variables for these extracted parameters

        '''


        if not self._constructed_map_around_ego:

            
            drivable_map_layer = self._simulation.scenario.map_api.get_raster_map_layer(SemanticMapLayer.DRIVABLE_AREA)
            self._map_resolution = drivable_map_layer.precision
            self._map_transform = drivable_map_layer.transform
            # self._data_nondrivable_map = torch.tensor(zoom(np.logical_not(drivable_map_layer.data), 1/self._map_resolution))
            self._data_nondrivable_map = np.logical_not(drivable_map_layer.data)
            # self._data_nondrivable_map = drivable_map_layer.data
            self._map_offset = 10


            print('the transform is ', self._map_transform)
            print('this is the self._ego_state[pos][0] ', self._ego_state['pos'][0], '   ', self._ego_state['pos'][1], '   ', self._ego_state['yaw'], '    ', self._ego_state['pos'][0].size())
            # pdb.set_trace()

            smaller_map = self._data_nondrivable_map[:20000, 10000:30000]
            smaller_smaller_map = zoom(smaller_map, 0.05)
            plt.figure(figsize = (10,10))
            plt.imshow(smaller_smaller_map, interpolation='nearest')
            plt.gcf()
            plt.savefig('/home/kpegah/workspace/SIMPLE/smaller_smaller1_map.png')
            plt.show()
            plt.clf()


            smaller_map = self._data_nondrivable_map[20000:40000, 10000:30000]
            smaller_smaller_map = zoom(smaller_map, 0.05)
            plt.figure(figsize = (10,10))
            plt.imshow(smaller_smaller_map, interpolation='nearest')
            plt.gcf()
            plt.savefig('/home/kpegah/workspace/SIMPLE/smaller_smaller2_map.png')
            plt.show()
            plt.clf()


            smaller_map = self._data_nondrivable_map[40000:60000, 10000:30000]
            smaller_smaller_map = zoom(smaller_map, 0.05)
            plt.figure(figsize = (10,10))
            plt.imshow(smaller_smaller_map, interpolation='nearest')
            plt.gcf()
            plt.savefig('/home/kpegah/workspace/SIMPLE/smaller_smaller3_map.png')
            plt.show()
            plt.clf()


            smaller_map = self._data_nondrivable_map[:20000, 0:20000]
            smaller_smaller_map = zoom(smaller_map, 0.05)
            plt.figure(figsize = (10,10))
            plt.imshow(smaller_smaller_map, interpolation='nearest')
            plt.gcf()
            plt.savefig('/home/kpegah/workspace/SIMPLE/smaller_smaller4_map.png')
            plt.show()
            plt.clf()


            smaller_map = self._data_nondrivable_map[:20000, 20000:30000]
            smaller_smaller_map = zoom(smaller_map, 0.05)
            plt.figure(figsize = (10,10))
            plt.imshow(smaller_smaller_map, interpolation='nearest')
            plt.gcf()
            plt.savefig('/home/kpegah/workspace/SIMPLE/smaller_smaller5_map.png')
            plt.show()
            plt.clf()


            smaller_map = self._data_nondrivable_map[20000:40000, 0:20000]
            smaller_smaller_map = zoom(smaller_map, 0.05)
            # so, 20000*0.05 = 1000
            # about x 1/5 and y 4/5
            # about x 4000+20000=24000 and y 8000 in the transformed coordinates
            plt.figure(figsize = (10,10))
            plt.imshow(smaller_smaller_map, interpolation='nearest')
            plt.gcf()
            plt.savefig('/home/kpegah/workspace/SIMPLE/smaller_smaller6_map.png')
            plt.show()
            plt.clf()

            # 7 tooye x (tasvir) az 6 barik tare => x va y tasvir va map jabejast!!
            # 


            smaller_map = self._data_nondrivable_map[20000:40000, 0:10000]
            smaller_smaller_map = zoom(smaller_map, 0.05)
            plt.figure(figsize = (10,10))
            plt.imshow(smaller_smaller_map, interpolation='nearest')
            plt.gcf()
            plt.savefig('/home/kpegah/workspace/SIMPLE/smaller_smaller7_map.png')
            plt.show()
            plt.clf()



            transformed_crop_x, transformed_crop_y, transformed_crop_yaw = self.from_matrix(self._map_transform@self.coord_to_matrix(self._ego_state['pos'].detach().cpu().numpy(), self._ego_state['yaw'].detach().cpu().numpy()))
            self.transformed_crop_x = torch.tensor([transformed_crop_x], device=device)
            self.transformed_crop_y = torch.tensor([transformed_crop_y], device=device)
            # print('the size before before ', self._data_nondrivable_map.shape)
            # print('the transformed coords ', transformed_crop_x, '    ', transformed_crop_y)
            self._data_nondrivable_map = self._data_nondrivable_map[int(transformed_crop_y-1000):int(transformed_crop_y+1000), int(transformed_crop_x-1000):int(transformed_crop_x+1000)]
         
            # print('the size before   ', self._data_nondrivable_map.shape)
            # plt.figure(figsize = (10,10))
            # plt.imshow(self._data_nondrivable_map, interpolation='nearest')
            # plt.gcf()
            # plt.savefig('/home/kpegah/workspace/SIMPLE/the_small_map.png')
            # plt.show()
            # plt.clf()
            # print('the size after   ', self._data_nondrivable_map.shape)
            # # self._x_crop_center, self._y_crop_center = self._ego_state['pos'][0], self._ego_state['pos'][1]
            self._constructed_map_around_ego = True
            # # check by positionning the agents the first time on the _data_nondrivable_map
            # plt.figure(figsize = (10,10))
            # plt.imshow(self._data_nondrivable_map, interpolation='nearest')
            # # Add points to the image using plt.scatter
            # points_x, points_y = [], []
            # for idx in range(self._number_agents):
            #     transformed_adv_x, transformed_adv_y, _ = self.from_matrix(self._map_transform@self.coord_to_matrix(self._states['pos'][idx].detach().cpu().numpy(), self._states['yaw'][idx].detach().cpu().numpy()))
            #     points_x.append(transformed_adv_x-transformed_crop_x+100/self._map_resolution)
            #     points_y.append(transformed_adv_y-transformed_crop_y+100/self._map_resolution)
            # plt.scatter(points_x, points_y, c='red', marker='o', s=50)  # Adjust marker style and size as needed
            # plt.gcf()
            # plt.savefig('/home/kpegah/workspace/SIMPLE/the_cropped_map.png')
            # plt.show()
            # plt.clf()

            # cp_map = self._data_nondrivable_map
            # plt.figure(figsize = (10,10))
            # # Add points to the image using plt.scatter
            # points_x, points_y = [], []
            # print('*******************   ', np.unique(cp_map))
            # for idx in range(self._number_agents):
            #     transformed_adv_x, transformed_adv_y, _ = self.from_matrix(self._map_transform@self.coord_to_matrix(self._states['pos'][idx].detach().cpu().numpy(), self._states['yaw'][idx].detach().cpu().numpy()))
            #     init_value = cp_map[int(transformed_adv_y-transformed_crop_y+100/self._map_resolution), int(transformed_adv_x-transformed_crop_x+100/self._map_resolution)]
            #     cp_map[int(transformed_adv_y-transformed_crop_y+100/self._map_resolution), int(transformed_adv_x-transformed_crop_x+100/self._map_resolution)] = not init_value
            #     cp_map[int(transformed_adv_y-transformed_crop_y+100/self._map_resolution)+1, int(transformed_adv_x-transformed_crop_x+100/self._map_resolution)+1] = not init_value
            #     cp_map[int(transformed_adv_y-transformed_crop_y+100/self._map_resolution), int(transformed_adv_x-transformed_crop_x+100/self._map_resolution)+1] = not init_value
            #     cp_map[int(transformed_adv_y-transformed_crop_y+100/self._map_resolution)-1, int(transformed_adv_x-transformed_crop_x+100/self._map_resolution)+1] = not init_value
            #     cp_map[int(transformed_adv_y-transformed_crop_y+100/self._map_resolution)-1, int(transformed_adv_x-transformed_crop_x+100/self._map_resolution)] = not init_value
            #     # points_x.append(transformed_adv_x-transformed_crop_x+100/self._map_resolution)
            #     # points_y.append(transformed_adv_y-transformed_crop_y+100/self._map_resolution)
            
            # plt.imshow(cp_map, interpolation='nearest')
            # plt.scatter(points_x, points_y, c='red', marker='o', s=50)  # Adjust marker style and size as needed
            # plt.gcf()
            # plt.savefig('/home/kpegah/workspace/SIMPLE/the_colored_image.png')
            # plt.show()
            # plt.clf()


            self._map_transform = torch.tensor(self._map_transform, device=device)
            self._map_resolution = torch.tensor(self._map_resolution, device=device)
            self._data_nondrivable_map = torch.tensor(self._data_nondrivable_map, device=device).to(dtype=torch.float64)
            

        input_cost_ego_state = {}
        for substate in self._ego_state.keys():
            input_cost_ego_state[substate] = torch.unsqueeze(torch.unsqueeze(self._ego_state[substate], dim=0), dim=0)

        input_cost_adv_state = self.get_adv_state(current_iteration)

        # write a dummy cost that just makes all the vehicles close to the ego


        '''
        ego_col_cost, adv_col_cost, _ = self.col_cost_fn(
            input_cost_ego_state,
            self.ego_extent,
            input_cost_adv_state,
            self.adv_extent,
        )
        

    
        adv_rd_cost = self.adv_rd_cost(
            self._data_nondrivable_map, 
            self.transformed_crop_x,
            self.transformed_crop_y,
            input_cost_adv_state['pos'],
            input_cost_adv_state['yaw'],
            self._map_resolution,
            self._map_transform
        )
        print('this is the drivable area cost ', adv_rd_cost)

        '''

        
        dummy_cost = self.dummy_cost_fn(
            input_cost_ego_state,
            self.ego_extent,
            input_cost_adv_state,
            self.adv_extent,
        )


        '''
        dummy_cost = self.dummy_cost_fn(
            self.ego_extent,
            input_cost_adv_state,
            self.adv_extent,
            torch.unsqueeze(torch.unsqueeze(torch.tensor(self._fixed_point, device=device, dtype=torch.float64), dim=0), dim=0)
        )
        '''


        # TODO
        # 1. do not propagate any cost, just reproduce the original trajectory of vehicles using the bicycle model
        # 2. a simple cost to attract all the vehicles to the ego without any complication

        '''
        if adv_col_cost.size(-1) == 0:
            adv_col_cost = torch.zeros(1,1).cpu()
            assert adv_col_cost.size(0) == 1, 'This works only for batchsize 1!'

        adv_col_cost = torch.minimum(
            adv_col_cost, torch.tensor([1.25]).float().cpu()
        )

        

        self._cost_dict['ego_col'].append(ego_col_cost)
        self._cost_dict['adv_col'].append(adv_col_cost)
        '''
        self._cost_dict['dummy'].append(dummy_cost)
        # self._cost_dict['adv_rd'].append(adv_rd_cost)
        

    def back_prop(self):
        '''
        - find the initial actions from the existing trajectories for background agents
        - create variables for these extracted parameters

        '''
        self._optimizer.zero_grad(set_to_none=True)
        cost_dict = self._cost_dict
        # aggregate costs and build total objective
        '''
        cost_dict["ego_col"] = torch.min(
            torch.mean(
                torch.stack(cost_dict["ego_col"], dim=1),
                dim=1,
            ),
            dim=1,
        )[0]
        cost_dict["adv_col"] = torch.min(
            torch.min(
                torch.stack(cost_dict["adv_col"], dim=1),
                dim=1,
            )[0],
            dim=1,
        )[0]
        '''
        
        cost_dict["dummy"] = torch.sum(
            torch.stack(cost_dict["dummy"], dim=-1),
        )
        
        
        '''
        cost_dict["adv_rd"] = torch.mean(
            torch.stack(cost_dict["adv_rd"], dim=1),
            dim=1,
        )
        '''

    

        '''
        total_objective = sum([
            1. * cost_dict["ego_col"].mean(),
            23. * cost_dict["adv_rd"].mean(),
            -1*5. * cost_dict["adv_col"].mean()
        ])
        '''

        print('the non drivable area cost and the fixed cost  ', cost_dict["adv_rd"], '  ', cost_dict["adv_rd"])
        # total_objective = self._cost_dict["dummy"] + cost_dict["adv_rd"]
        # total_objective = cost_dict["adv_rd"]*20
        total_objective = self._cost_dict["dummy"]
        # total_objective = cost_dict["dummy"]



        # TODO
        # find how to spot collisions using the king functions
        '''
        collisions = self.simulator.ego_collision[self.simulator.ego_collision == 1.]
        col_metric = len(collisions) / self.args.batch_size
        '''

        # if col_metric != 1.0:
        if True:
            total_objective.backward()
            # for param_group in self._optimizer.param_groups:
            #     for param in param_group['params']:
            #         for small_param in param:
            #             print('this is the param ', small_param.item())
            #             print('this is the gradient ', small_param.grad)
            # let us update the gradients for each agent
            for i_agent in range(self._number_agents):
                self.throttle_gradients[i_agent] = self._throttle.grad[i_agent].clone().cpu().numpy()
                self.steer_gradients[i_agent] = self._steer.grad[i_agent].clone().cpu().numpy()
            self._optimizer.step()



    def check_collision_at_iteration(self, iteration: int):
        pass

    def save_state_buffer(self):

        self.whole_state_buffers.append(self.state_buffer)

    def plots(self):

        # following the evolution of different agents in rounds of optimization, using their position, heading and velocity
        colors = cm.rainbow(np.linspace(0, 1, self._number_agents+1))
        for opt_round in range(len(self.whole_state_buffers)):
            current_opt_round_states = self.whole_state_buffers[opt_round]
            # current_round_states is a list of states at each timestep of the simulation
            for sim_round in range(len(current_opt_round_states)):
                current_sim_round_states = current_opt_round_states[sim_round]
                current_sim_round_positions = current_sim_round_states['pos']
                current_sim_round_headings = current_sim_round_states['yaw']

                for i_agent, i_color in enumerate(colors):
                    plt.quiver(*np.array([current_sim_round_positions[i_agent,0], current_sim_round_positions[i_agent,1]]), np.cos(current_sim_round_headings[i_agent:i_agent+1, 0]), np.sin(current_sim_round_headings[i_agent:i_agent+1,0]), color=i_color, scale=60)

            plt.gcf()
            plt.savefig(f'/home/kpegah/workspace/{self._simulation.scenario.scenario_name}_agents_evolution_{opt_round}.png')
            plt.show()
            plt.clf()



        sin_beta, that_y_vel, longitudinal_vel, vel_x, heading, steering_angle, steering_rate, updated_steering_rate = self._motion_model.that_agent()
        plt.plot(torch.tensor(vel_x).detach().numpy())
        plt.plot(self.plot_gt_velocities*np.cos(self.plot_yaw_gt))
        plt.legend(['x_bm_velocities', 'x_gt_velocities'])
        plt.gcf()
        plt.savefig('/home/kpegah/workspace/x_velocity_bm_gt.png')
        plt.show()
        plt.clf()


        plt.plot(torch.tensor(longitudinal_vel).detach().numpy())
        plt.plot(self.plot_gt_velocities)
        plt.legend(['bm_velocities', 'gt_velocities'])
        plt.gcf()
        plt.savefig('/home/kpegah/workspace/longitudinal_velocity_bm_gt.png')
        plt.show()
        plt.clf()

        plt.plot(torch.tensor(heading).detach().numpy())
        plt.plot(self.plot_yaw_gt)
        plt.legend(['yaw_bm', 'yaw_gt'])
        plt.gcf()
        plt.savefig('/home/kpegah/workspace/yaw_bm_gt.png')
        plt.show()
        plt.clf()

        plt.plot(torch.tensor(that_y_vel).detach().numpy())
        plt.plot(self.that_agent_y_velocity)
        plt.legend(['bm_y_vel', 'y_vel_gt'])
        plt.gcf()
        plt.savefig('/home/kpegah/workspace/_vel.png')
        plt.show()
        plt.clf()

        plt.plot(torch.tensor(updated_steering_rate).detach().numpy())
        plt.plot(torch.tensor(steering_rate).detach().numpy())
        plt.legend(['updated_steer', 'steer'])
        plt.gcf()
        plt.savefig('/home/kpegah/workspace/updated_steer.png')
        plt.show()
        plt.clf()

        plt.plot(torch.tensor(steering_angle).detach().numpy())
        plt.legend(['steering_angle'])
        plt.gcf()
        plt.savefig('/home/kpegah/workspace/steering_angle.png')
        plt.show()
        plt.clf()

        plt.plot(self.plot_throttle_track)
        plt.legend(['throttle'])
        plt.gcf()
        plt.savefig('/home/kpegah/workspace/throttle.png')
        plt.show()
        plt.clf()


        
        plt.plot(torch.tensor(steering_rate).detach().numpy())
        plt.legend(['steering_rate'])
        plt.gcf()
        plt.savefig('/home/kpegah/workspace/steering_rate.png')
        plt.show()
        plt.clf()

        
        plt.plot(self.ego_y_direction_vel)
        plt.legend(['ego_y_direction_vel'])
        plt.gcf()
        plt.savefig('/home/kpegah/workspace/ego_y_direction_vel.png')
        plt.show()
        plt.clf()


         
        plt.plot(torch.tensor(sin_beta).detach().numpy())
        plt.legend(['sin_beta'])
        plt.gcf()
        plt.savefig('/home/kpegah/workspace/sin_beta.png')
        plt.show()
        plt.clf()

    def visualize_grads_throttle(self, optimization_iteration: int):

        '''
        the function to visualize the gradients registered by the hooks for the throttle and steer
        :params optimization_iteration
        '''

        # we have gradients accumulated for each step of the simulation, and for each of the agents, we know its position
        pos_agents = self.whole_state_buffers[0][0]['pos'][:self._number_agents, :]
        smallest_x, largest_x, smallest_y, largest_y = np.min(pos_agents[:,0]), np.max(pos_agents[:,0]), np.min(pos_agents[:,1]), np.max(pos_agents[:,1])
        # height_map, width_map = self._nondrivable_map_layer.shape[0], self._nondrivable_map_layer.shape[1]
        width, height = (largest_x - smallest_x), (largest_y - smallest_y)

        fig_throttle, ax_throttle = plt.subplots(facecolor ='#A0F0CC')       
        ax_throttle.set_xlim(smallest_x, largest_x)
        ax_throttle.set_ylim(smallest_y, largest_y)


        colors = cm.rainbow(np.linspace(0, 1, self._number_agents))

        for i_agent in range(self._number_agents):
            # contsruct the gradient (throttle and steer) for the current agent
            # the position of the agent should be with respect to the figure
            relative_x, relative_y = (pos_agents[i_agent,0]-smallest_x)/width, (pos_agents[i_agent,1]-smallest_y)/height
            # what should be the size of the plot for the gradients?? decide based on the values obtained for each agent
            current_throttle_grads = self.throttle_gradients[i_agent] # this is of the size of length of actions, and for each of the actions, 

            subpos_throttle = [relative_x ,relative_y ,0.15 ,0.1]
            subax1 = add_subplot_axes(ax_throttle,subpos_throttle)

            # should first normalize the gradients in the x direction??
            subax1.plot(current_throttle_grads, color=colors[i_agent])
        
        
        
        plt.gcf()
        plt.savefig(f'/home/kpegah/workspace/SIMPLE/{optimization_iteration}_throttle_grads.png')
        plt.show()
        plt.clf()



    def visualize_grads_steer(self, optimization_iteration: int):

        '''
        the function to visualize the gradients registered by the hooks for the throttle and steer
        :params optimization_iteration
        '''

        # we have gradients accumulated for each step of the simulation, and for each of the agents, we know its position
        pos_agents = self.whole_state_buffers[0][0]['pos'][:self._number_agents, :]
        smallest_x, largest_x, smallest_y, largest_y = np.min(pos_agents[:,0]), np.max(pos_agents[:,0]), np.min(pos_agents[:,1]), np.max(pos_agents[:,1])
        width, height = (largest_x - smallest_x), (largest_y - smallest_y)

        fig_steer, ax_steer = plt.subplots(facecolor ='#A0F0CC')       
        ax_steer.set_xlim(smallest_x, largest_x)
        ax_steer.set_ylim(smallest_y, largest_y)


        colors = cm.rainbow(np.linspace(0, 1, self._number_agents))

        for i_agent in range(self._number_agents):
            # contsruct the gradient (throttle and steer) for the current agent
            # the position of the agent should be with respect to the figure
            relative_x, relative_y = (pos_agents[i_agent,0]-smallest_x)/width, (pos_agents[i_agent,1]-smallest_y)/height
            current_steer_grads = self.steer_gradients[i_agent]

            subpos_steer = [relative_x ,relative_y ,0.15 ,0.1]

            subax2 = add_subplot_axes(ax_steer,subpos_steer)

            subax2.plot(current_steer_grads, color=colors[i_agent])
        
        
        
        plt.gcf()
        plt.savefig(f'/home/kpegah/workspace/SIMPLE/{optimization_iteration}_steer_grads.png')
        plt.show()
        plt.clf()

