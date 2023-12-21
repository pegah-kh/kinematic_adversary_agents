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
import os
from torchviz import make_dot
from PIL import Image
import cv2
from time import perf_counter




from nuplan_king.planning.simulation.adversary_optimizer.abstract_optimizer import AbstractOptimizer
from nuplan_king.planning.simulation.adversary_optimizer.agent_tracker.agent_lqr_tracker import LQRTracker 
from nuplan_king.planning.simulation.motion_model.bicycle_model import BicycleModel
from nuplan_king.planning.simulation.cost.king_costs import RouteDeviationCostRasterized, BatchedPolygonCollisionCost, DummyCost, DummyCost_FixedPoint, DrivableAreaCost

from nuplan.common.maps.maps_datatypes import SemanticMapLayer
from nuplan.common.actor_state.tracked_objects import TrackedObject
from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType
from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.actor_state.waypoint import Waypoint
from nuplan.common.actor_state.oriented_box import OrientedBox
from nuplan.common.actor_state.state_representation import StateSE2, StateVector2D
from nuplan_king.planning.scenario_builder.nuplan_db_modif.nuplan_scenario import NuPlanScenarioModif
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

                
def create_directory_if_not_exists(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)

    
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
    subax = fig.add_axes([x,y,width,height], facecolor=facecolor)
    x_labelsize = subax.get_xticklabels()[0].get_size()
    y_labelsize = subax.get_yticklabels()[0].get_size()
    # x_labelsize *= rect[2]**0.3
    # y_labelsize *= rect[3]**0.3
    subax.xaxis.set_tick_params(labelsize=5)
    subax.yaxis.set_tick_params(labelsize=5)
    # subax.set_ylim([-100,100])
    return subax





class TransformCoordMap():
    """
    Manager which executes multiple simulations with the same planner
    """

    def __init__(self, resoltuion, transform, x_crop, y_crop):
        self._transform = transform
        self._resolution = resoltuion
        self._x_crop = x_crop
        self._y_crop = y_crop

        self._inv_transform = np.linalg.inv(self._transform)


        
    @staticmethod
    def coord_to_matrix(coord, yaw):

        return np.array(
            [
                np.cos(yaw[0]), -np.sin(yaw[0]), 0.0, coord[0],
                np.sin(yaw[0]), np.cos(yaw[0]), 0.0, coord[1],
                0.0, 0.0, 1.0, 0.0,
                0.0, 0.0, 0.0, 1.0,
            ]
        ).reshape(4,4)
    
    @staticmethod
    def from_matrix(matrix):

        assert matrix.shape == (4, 4), f"Expected 3x3 transformation matrix, but input matrix has shape {matrix.shape}"

        return matrix[0, 3], matrix[1, 3], np.arctan2(matrix[1, 0], matrix[0, 0])

    def pos_to_map(self, coord_x, coord_y, coord_yaw):

        transformed_adv_x, transformed_adv_y, transformed_yaw = self.from_matrix(self._transform@self.coord_to_matrix([coord_x, coord_y], [coord_yaw]))

        return transformed_adv_y - self._y_crop + 100/self._resolution, transformed_adv_x - self._x_crop + 100/self._resolution, np.pi/2-transformed_yaw
    

    def pos_to_coord(self, map_x, map_y, map_yaw):


        transformed_adv_x, transformed_adv_y, transformed_yaw = self.from_matrix(self._inv_transform@self.coord_to_matrix([map_y+ self._x_crop - 100/self._resolution, map_x+ self._y_crop - 100/self._resolution], [np.pi/2-map_yaw]))

        return transformed_adv_x, transformed_adv_y, transformed_yaw



# the OptimizationKING calls the simulation runner for iterations
class OptimizationKING():
    """
    Manager which executes multiple simulations with the same planner
    """

    def __init__(self, simulation: Simulation, planner: AbstractPlanner, tracker: DictConfig, motion_model: BicycleModel, opt_iterations: int, max_opt_iterations: int, lrs: List[float], loss_weights: List[float], costs_to_use: List[str] = ['fixed_dummy'], requires_grad_params: List[str] = ['throttle', 'steer'], experiment_name: str = 'map_cost'):


        self._experiment_name = experiment_name
        # super.__init__(self, simulation, planner)
        self._simulation = simulation
        self._planner = planner
        
        # motion model
        self._motion_model = motion_model

        
        # the maximum number of optimization iterations
        self._opt_iterations = opt_iterations
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

        self._ego_state = {'pos': torch.zeros(2).to(device=device), 'yaw': torch.zeros(1).to(device=device), 'vel': torch.zeros(2).to(device=device)}

        self._throttle_requires_grad = False
        self._steer_requires_grad = False
        if 'throttle' in requires_grad_params:
            self._throttle_requires_grad = True
        if 'steer' in requires_grad_params:
            self._steer_requires_grad = True

        self._costs_to_use = costs_to_use
        # learning rates for two actions and losses
        self.lr_throttle = lrs[0]
        self.lr_steer = lrs[1]
        # the weight of different losses
        self.weight_collision = loss_weights[0]
        self.weight_drivable = loss_weights[1]
        # the cost functions
        self.col_cost_fn = BatchedPolygonCollisionCost(num_agents=self._number_agents)
        self.adv_rd_cost = RouteDeviationCostRasterized(num_agents=self._number_agents)
        if 'fixed_dummy' in costs_to_use:
            self.dummy_cost_fn = DummyCost_FixedPoint(num_agents=self._number_agents)
        elif 'moving_dummy' in costs_to_use:
            self.dummy_cost_fn = DummyCost(num_agents=self._number_agents)
        # if 'drivable' in costs_to_use:
        drivable_map_layer = self._simulation.scenario.map_api.get_raster_map_layer(SemanticMapLayer.DRIVABLE_AREA)
        self._map_resolution = drivable_map_layer.precision
        self._map_transform = drivable_map_layer.transform
        self._data_nondrivable_map = np.logical_not(drivable_map_layer.data)

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
        self._accumulated_drivable_loss = 0
        self._accumulated_collision_loss = 0
        self._accumulated_drivable_loss_agents = torch.zeros(self._number_agents).to(device=device)
        self._accumulated_collision_loss_agents = torch.zeros(self._number_agents).to(device=device)
        self._simulation.reset()
        self._simulation._ego_controller.reset()
        self._simulation._observations.reset()
        self._optimizer_throttle.zero_grad()
        self._optimizer_steer.zero_grad()
        self.reset_dynamic_states()

        # new costs
        self._cost_dict = {"ego_col": [], "adv_rd": [], "adv_col": [], "dummy": []}
        self.state_buffer = []

        # gradient hooks to keep track of the evolution of the gradients of thottle and steering during the optimization
        self.throttle_gradients = [torch.zeros(self._number_actions).to(device=device) for _ in range(self._number_agents)]
        self.steer_gradients = [torch.zeros(self._number_actions).to(device=device) for _ in range(self._number_agents)]
        self.throttles = [torch.zeros(self._number_actions).to(device=device) for _ in range(self._number_agents)]
        self.steers = [torch.zeros(self._number_actions).to(device=device) for _ in range(self._number_agents)]
        self._drivable_area_losses_per_step = []
        self._dummy_losses_per_step = []



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


        # transformed_crop_x, transformed_crop_y, transformed_crop_yaw = self.from_matrix(self._map_transform@self.coord_to_matrix(self._ego_state['pos'].detach().cpu().numpy(), self._ego_state['yaw'].detach().cpu().numpy()))
        # new_pos[0,idx,:], new_yaw[0,idx,:] = self.crop_positions(new_pos[0,idx,:], new_yaw[0,idx,:], x_crop, y_crop, resolution, transform)

        self._states = {'pos': torch.zeros(self._number_agents, 2, requires_grad=True).to(device=device), 'yaw': torch.zeros(self._number_agents, 1, requires_grad=True).to(device=device), 'steering_angle': torch.zeros(self._number_agents, 1, requires_grad=True).to(device=device), 'vel': torch.zeros(self._number_agents, 2, requires_grad=True).to(device=device), 'accel': torch.zeros(self._number_agents, 1, requires_grad=True).to(device=device), 'speed': torch.zeros(self._number_agents, 1, requires_grad=True).to(device=device)}
        self._states['pos'].retain_grad()
        self._states['yaw'].retain_grad()
        self._states['speed'].retain_grad()

        self._initial_map_x, self._initial_map_y, self._initial_map_yaw = [], [], []

        before_transforming_x, before_transforming_y, before_transforming_yaw = [], [], []
        after_transforming_x, after_transforming_y, after_transforming_yaw = [], [], []
        for idx, tracked_agent in enumerate(self._agents):

            coord_x, coord_y, coord_yaw = tracked_agent.predictions[0].valid_waypoints[0].x, tracked_agent.predictions[0].valid_waypoints[0].y, tracked_agent.predictions[0].valid_waypoints[0].heading
            before_transforming_x.append(coord_x)
            before_transforming_y.append(coord_y)
            before_transforming_yaw.append(coord_yaw)
            map_x, map_y, map_yaw = self._convert_coord_map.pos_to_map(coord_x, coord_y, coord_yaw)
            after_transforming_x.append(map_x)
            after_transforming_y.append(map_y)
            after_transforming_yaw.append(map_yaw)

            self._initial_map_x.append(map_x)
            self._initial_map_y.append(map_y)
            self._initial_map_yaw.append(map_yaw)

            self._states['pos'][idx] = torch.tensor([map_x, map_y], device=device, dtype=torch.float64, requires_grad=True)
            self._states['pos'][idx].retain_grad()
            self._states['yaw'][idx] = torch.tensor([map_yaw], device=device, dtype=torch.float64, requires_grad=True)
            self._states['yaw'][idx].retain_grad()
            # self._states['vel'][idx] = torch.tensor([tracked_agent.predictions[0].valid_waypoints[0].velocity.x, tracked_agent.predictions[0].valid_waypoints[0].velocity.y], device=device)
            self._states['vel'][idx] = torch.tensor([0., 0.], dtype=torch.float64, device=device, requires_grad=True)
            self._states['speed'][idx] = torch.tensor([0.0], dtype=torch.float64, device=device, requires_grad=True)
            self._states['accel'][idx] = torch.tensor([0.0], dtype=torch.float64, device=device, requires_grad=True)
            # approx_tire_steering_angle = np.arctan(3.089*(tracked_agent.predictions[0].valid_waypoints[1].heading - tracked_agent.predictions[0].valid_waypoints[0].heading)/(self._observation_trajectory_sampling.interval_length*np.hypot(tracked_agent.velocity.x, tracked_agent.velocity.y)+1e-3))
            # self._states['steering_angle'][idx] = torch.clamp(torch.tensor([approx_tire_steering_angle], device=device, dtype=torch.float64), min=-torch.pi/3, max=torch.pi/3)
            self._states['steering_angle'][idx] = torch.clamp(torch.tensor([0.0], device=device, dtype=torch.float64, requires_grad=True), min=-torch.pi/3, max=torch.pi/3)
        
        
        # create_directory_if_not_exists('/home/kpegah/workspace/DEBUG_TWOSIDE')
        # create_directory_if_not_exists(f'/home/kpegah/workspace/DEBUG_TWOSIDE/{self._simulation.scenario.scenario_name}')
        # create_directory_if_not_exists(f'/home/kpegah/workspace/DEBUG_TWOSIDE/{self._simulation.scenario.scenario_name}/{self._experiment_name}')
        
        # plt.quiver(before_transforming_x, before_transforming_y, np.cos(before_transforming_yaw), np.sin(before_transforming_yaw), scale=10)


        # plt.gcf()
        # plt.savefig(f'/home/kpegah/workspace/DEBUG_TWOSIDE/{self._simulation.scenario.scenario_name}/{self._experiment_name}/pos_agents_before1.png')
        # plt.show()
        # plt.clf()


        # plt.quiver(after_transforming_x, after_transforming_y, np.cos(after_transforming_yaw), np.sin(after_transforming_yaw), scale=10)


        # plt.gcf()
        # plt.savefig(f'/home/kpegah/workspace/DEBUG_TWOSIDE/{self._simulation.scenario.scenario_name}/{self._experiment_name}/pos_agents_after1.png')
        # plt.show()
        # plt.clf()


    def reset_dynamic_states(self):


        # transformed_crop_x, transformed_crop_y, transformed_crop_yaw = self.from_matrix(self._map_transform@self.coord_to_matrix(self._ego_state['pos'].detach().cpu().numpy(), self._ego_state['yaw'].detach().cpu().numpy()))
        # new_pos[0,idx,:], new_yaw[0,idx,:] = self.crop_positions(new_pos[0,idx,:], new_yaw[0,idx,:], x_crop, y_crop, resolution, transform)

        self._states = {'pos': torch.zeros(self._number_agents, 2, requires_grad=True).to(device=device), 'yaw': torch.zeros(self._number_agents, 1, requires_grad=True).to(device=device), 'steering_angle': torch.zeros(self._number_agents, 1, requires_grad=True).to(device=device), 'vel': torch.zeros(self._number_agents, 2, requires_grad=True).to(device=device), 'accel': torch.zeros(self._number_agents, 1, requires_grad=True).to(device=device), 'speed': torch.zeros(self._number_agents, 1, requires_grad=True).to(device=device)}
        self._states['pos'].retain_grad()
        self._states['yaw'].retain_grad()
        self._states['speed'].retain_grad()
        for idx, _ in enumerate(self._agents):

            map_x, map_y, map_yaw = self._initial_map_x[idx], self._initial_map_y[idx], self._initial_map_yaw[idx]

            self._states['pos'][idx] = torch.tensor([map_x, map_y], device=device, dtype=torch.float64, requires_grad=True)
            self._states['pos'][idx].retain_grad()
            self._states['yaw'][idx] = torch.tensor([map_yaw], device=device, dtype=torch.float64, requires_grad=True)
            self._states['yaw'][idx].retain_grad()
            # self._states['vel'][idx] = torch.tensor([tracked_agent.predictions[0].valid_waypoints[0].velocity.x, tracked_agent.predictions[0].valid_waypoints[0].velocity.y], device=device)
            self._states['vel'][idx] = torch.tensor([0., 0.], dtype=torch.float64, device=device, requires_grad=True)
            self._states['speed'][idx] = torch.tensor([0.0], dtype=torch.float64, device=device, requires_grad=True)
            self._states['accel'][idx] = torch.tensor([0.0], dtype=torch.float64, device=device, requires_grad=True)
            # approx_tire_steering_angle = np.arctan(3.089*(tracked_agent.predictions[0].valid_waypoints[1].heading - tracked_agent.predictions[0].valid_waypoints[0].heading)/(self._observation_trajectory_sampling.interval_length*np.hypot(tracked_agent.velocity.x, tracked_agent.velocity.y)+1e-3))
            # self._states['steering_angle'][idx] = torch.clamp(torch.tensor([approx_tire_steering_angle], device=device, dtype=torch.float64), min=-torch.pi/3, max=torch.pi/3)
            self._states['steering_angle'][idx] = torch.clamp(torch.tensor([0.0], device=device, dtype=torch.float64, requires_grad=True), min=-torch.pi/3, max=torch.pi/3)
        


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
    def initialize(self) -> None:
        """
        To obtain the initial actions using the lqr controller, adn from the trajectory.
        Inherited method.
        """

        # The initial coordinate of the ego that we will use to transform from coord space to map space
        transformed_crop_x, transformed_crop_y, transformed_crop_yaw = self.from_matrix(self._map_transform@self.coord_to_matrix(self._ego_state['pos'].detach().cpu().numpy(), self._ego_state['yaw'].detach().cpu().numpy()))
        self.transformed_crop_x = transformed_crop_x
        self.transformed_crop_y = transformed_crop_y
        self._data_nondrivable_map = self._data_nondrivable_map[int(transformed_crop_y-1000):int(transformed_crop_y+1000), int(transformed_crop_x-1000):int(transformed_crop_x+1000)]
        self.test_distance_map = DrivableAreaCost(self._data_nondrivable_map)
            
        self._convert_coord_map = TransformCoordMap(self._map_resolution, self._map_transform, self.transformed_crop_x, self.transformed_crop_y)

        self._data_nondrivable_map = torch.tensor(self._data_nondrivable_map, device=device, dtype=torch.float64, requires_grad=True)
        self._transpose_data_nondrivable_map = torch.transpose((1-self._data_nondrivable_map), 1, 0)

        self.drivable_loss_agents = []
        self.dummy_loss_agents = []

        print(f"Using device ******* ******** ******** *********: {device}")
        self._drivable_area_losses_per_opt = []
        self._drivable_area_losses_per_step = []
        self._dummy_losses_per_step = []
        self._dummy_losses_per_opt = []

        
        self.drivable_loss_agents_per_opt = []
        self.dummy_loss_agents_per_opt = []

        # self._fixed_point = np.array([664195.,3996283.])
        # self._fixed_point = np.array([self._simulation._ego_controller.get_state().center.x,self._simulation._ego_controller.get_state().center.y])
        self._fixed_point = np.array([1000, 1000])

        # self._constructed_map_around_ego = False
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
                device=device,
                dtype=torch.float64
            ),
            requires_grad=self._throttle_requires_grad
        )
        if self._throttle_requires_grad:
            self._throttle.retain_grad()
        self._steer = torch.nn.Parameter(
            torch.zeros(
                num_agents, self._horizon, 1,
                device=device,
                dtype=torch.float64
            ),
            requires_grad=self._steer_requires_grad
        )
        if self._steer_requires_grad:
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

            continue
            print('treating the agent ', tracked_agent.track_token)
        
            

            initial_waypoint = tracked_agent.predictions[0].valid_waypoints[0]
            initial_tire_steering_angle = np.arctan(3.089*(tracked_agent.predictions[0].valid_waypoints[1].heading - tracked_agent.predictions[0].valid_waypoints[0].heading)/(self._observation_trajectory_sampling.interval_length*np.hypot(tracked_agent.velocity.x, tracked_agent.velocity.y)+1e-3))


            # initializing the dynamic state of the agents AT ALL TIME STEPS steps of the horizon using the lqr controller
            with torch.no_grad():
                for _time_step in range(len(tracked_agent.predictions[0].valid_waypoints) - 1):
                    # using the timepoints of the simulation instead of those of predictions
                    throttle, steer = self._tracker.track_trajectory(tracked_agent.predictions[0].valid_waypoints[_time_step].time_point, tracked_agent.predictions[0].valid_waypoints[_time_step+1].time_point, initial_waypoint, tracked_agent.predictions[0].trajectory, initial_steering_angle=initial_tire_steering_angle)
                    # self._throttle[idx,_time_step], self._steer[idx,_time_step] = torch.Tensor([throttle]), torch.Tensor([steer])
                    self._throttle[idx,_time_step], self._steer[idx,_time_step] = torch.tensor([0.0], dtype=torch.float64).to(device), torch.tensor([0.0], dtype=torch.float64).to(device)
                    # updating the initial_waypoint based on the update from the bicycle model
                    if _time_step==0:
                        beginning_state = {_key: _value.clone().detach().to(device) for _key, _value in self.get_adv_state(_time_step, id=idx).items()} # B*N*S
                        next_state = self._motion_model.forward(beginning_state, self.get_adv_actions(_time_step, id=idx), track_token=self._agent_tokens[idx], iter=_time_step+1, plotter=False)
                    else:
                        next_state = self._motion_model.forward(next_state, self.get_adv_actions(_time_step, id=idx), track_token=self._agent_tokens[idx], iter=_time_step+1, plotter=False)

                    # FOR CAPTURING INITIAL STATES *********************** OPEN
                    current_color = colors[idx]
                    plt.quiver(*np.array([next_state['pos'].cpu()[0,0,0], next_state['pos'].cpu()[0,0,1]]), np.cos(next_state['yaw'].cpu()[0,0,0]), np.sin(next_state['yaw'].cpu()[0,0,0]), color=current_color, scale=40)
                    # FOR CAPTURING INITIAL STATES *********************** CLOSE


                    initial_waypoint = Waypoint(time_point=tracked_agent.predictions[0].valid_waypoints[_time_step+1].time_point, oriented_box=OrientedBox.from_new_pose(tracked_agent.box, StateSE2(next_state['pos'].cpu()[0,0,0], next_state['pos'].cpu()[0,0,1], next_state['yaw'].cpu()[0,0,0])), velocity=StateVector2D(next_state['vel'].cpu()[0,0,0], next_state['vel'].cpu()[0,0,1]))
                    initial_tire_steering_angle = next_state['steering_angle'].cpu()

                for _time_step in range(len(tracked_agent.predictions[0].valid_waypoints) - 1, self._horizon):
                    # using the timepoints of the simulation instead of those of predictions
                    throttle, steer = self._tracker.track_trajectory(tracked_agent.predictions[0].valid_waypoints[-2].time_point, tracked_agent.predictions[0].valid_waypoints[-1].time_point, initial_waypoint, tracked_agent.predictions[0].trajectory, initial_steering_angle=initial_tire_steering_angle)
                    # self._throttle[idx,_time_step], self._steer[idx,_time_step] = torch.Tensor([throttle]), torch.Tensor([steer])
                    self._throttle[idx,_time_step], self._steer[idx,_time_step] = torch.tensor([0.0], dtype=torch.float64).to(device), torch.tensor([0.0], dtype=torch.float64).to(device)
                    # updating the initial_waypoint based on the update from the bicycle model
                    next_state = self._motion_model.forward(next_state, self.get_adv_actions(_time_step, id=idx), track_token=self._agent_tokens[idx], iter=_time_step+1, plotter=False)

                    # FOR CAPTURING INITIAL STATES *********************** OPEN
                    current_color = colors[idx]
                    plt.quiver(*np.array([next_state['pos'].cpu()[0,0,0], next_state['pos'].cpu()[0,0,1]]), np.cos(next_state['yaw'].cpu()[0,0,0]), np.sin(next_state['yaw'].cpu()[0,0,0]), color=current_color, scale=40)
                    # FOR CAPTURING INITIAL STATES *********************** CLOSE


                    initial_waypoint = Waypoint(time_point=tracked_agent.predictions[0].valid_waypoints[-1].time_point, oriented_box=OrientedBox.from_new_pose(tracked_agent.box, StateSE2(next_state['pos'].cpu()[0,0,0], next_state['pos'].cpu()[0,0,1], next_state['yaw'].cpu()[0,0,0])), velocity=StateVector2D(next_state['vel'].cpu()[0,0,0], next_state['vel'].cpu()[0,0,1]))
                    initial_tire_steering_angle = next_state['steering_angle'].cpu()

        

        # optimization_params = [self._throttle, self._steer]
        self._optimizer_throttle = torch.optim.Adam([self._throttle], lr=self.lr_throttle, betas=(0.8, 0.99))
        self._optimizer_steer = torch.optim.Adam([self._steer], lr=self.lr_steer, betas=(0.8, 0.99))
        # self._optimizer_collision = torch.optim.SGD(optimization_params, lr=0.1)
        # self._optimizer_drivable_throttle = torch.optim.Adam([self._throttle], lr=self.lr_drivable_throttle, betas=(0.8, 0.99))
        # self._optimizer_drivable_steer = torch.optim.Adam([self._steer], lr=self.lr_drivable_steer, betas=(0.8, 0.99))
        # initialize the ego state
        
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

   
        # self.adv_extent = torch.tensor(
        #     [(4.049+1.127)/2,
        #      14.85],
        #     device=device,
        #     dtype=torch.float64
        # ).view(1, 1, 2).expand(1, 1, 2)


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
            adv_state[substate].retain_grad()

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
            self._states[substate].retain_grad()


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


        # for idx in range(self._number_agents):
        #     not_differentiable_state[self._agent_tokens[idx]] = ((agents_pos[idx, 0], agents_pos[idx, 1]), (agents_vel[idx, 0], agents_vel[idx, 1]), (agents_yaw[idx, 0]))
        



        after_transforming_x, after_transforming_y, after_transforming_yaw = [], [], []
        for idx in range(self._number_agents):
            coord_vel_x, coord_vel_y, _ = self._convert_coord_map.pos_to_coord(agents_vel[idx, 0], agents_vel[idx, 1], agents_yaw[idx, 0])
            coord_pos_x, coord_pos_y, coord_pos_yaw = self._convert_coord_map.pos_to_coord(agents_pos[idx, 0], agents_pos[idx, 1], agents_yaw[idx, 0])
            after_transforming_x.append(coord_pos_x)
            after_transforming_y.append(coord_pos_y)
            after_transforming_yaw.append(coord_pos_yaw)
            # print('the coords ', coord_pos_x, '  ', coord_pos_y)
            not_differentiable_state[self._agent_tokens[idx]] = ((coord_pos_x, coord_pos_y), (coord_vel_x, coord_vel_y), (coord_pos_yaw))
        
        # plt.quiver(after_transforming_x, after_transforming_y, np.cos(after_transforming_yaw), np.sin(after_transforming_yaw), scale=10)


        # plt.gcf()
        # plt.savefig(f'/home/kpegah/workspace/DEBUG_TWOSIDE/{self._simulation.scenario.scenario_name}/{self._experiment_name}/pos_agents_after2.png')
        # plt.show()
        # plt.clf()


        self.ego_y_direction_vel.append(self._simulation._ego_controller.get_state().dynamic_car_state.rear_axle_velocity_2d.y)
        return not_differentiable_state
        

    def update_ego_state(self, current_ego_state: EgoState, current_iteration: int):
        '''
        update the state of the ego agent using its trajectory
        :param trajectory: the ego trajectory that is taken at the current iteration
        :param next_iteration: the next iteration of simulation
        '''

        assert self._current_optimization_on_iteration==current_iteration, 'we are not optimizing on the right iteration of the simulation'


        self._ego_state['pos'] = torch.tensor([current_ego_state.center.x, current_ego_state.center.y], device=device, dtype=torch.float64)
        self._ego_state['vel'] = torch.tensor([current_ego_state.agent.velocity.x, current_ego_state.agent.velocity.y], device=device, dtype=torch.float64)
        self._ego_state['yaw'] = torch.tensor([current_ego_state.center.heading], device=device, dtype=torch.float64)

        self._current_optimization_on_iteration += 1

    def init_ego_state(self, current_ego_state: EgoState):
      

        self._ego_state['pos'] = torch.tensor([current_ego_state.center.x, current_ego_state.center.y], device=device, dtype=torch.float64)
        self._ego_state['vel'] = torch.tensor([current_ego_state.agent.velocity.x, current_ego_state.agent.velocity.y], device=device, dtype=torch.float64)
        self._ego_state['yaw'] = torch.tensor([current_ego_state.center.heading], device=device, dtype=torch.float64)
    

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
        drivable_computation_time = 0
        
        input_cost_ego_state = {}
        for substate in self._ego_state.keys():
            input_cost_ego_state[substate] = torch.unsqueeze(torch.unsqueeze(self._ego_state[substate], dim=0), dim=0)

        input_cost_adv_state = self.get_adv_state(current_iteration)

        # write a dummy cost that just makes all the vehicles close to the ego


        if 'collision' in self._costs_to_use:
            ego_col_cost, adv_col_cost, _ = self.col_cost_fn(
                input_cost_ego_state,
                self.ego_extent,
                input_cost_adv_state,
                self.adv_extent,
            )
            if adv_col_cost.size(-1) == 0:
                adv_col_cost = torch.zeros(1,1).cpu()
                assert adv_col_cost.size(0) == 1, 'This works only for batchsize 1!'

            adv_col_cost = torch.minimum(
                adv_col_cost, torch.tensor([1.25]).float().cpu()
            )

            

            self._cost_dict['ego_col'].append(ego_col_cost)
            self._cost_dict['adv_col'].append(adv_col_cost)
                


        if 'drivable_us' in self._costs_to_use:

            # self.routedeviation_heatmap()
            # self.routedeviation_heatmap_agents(input_cost_adv_state['pos'], input_cost_adv_state['yaw'])
            # self.routedeviation_agents(input_cost_adv_state['pos'], input_cost_adv_state['yaw'])
            
            input_cost_adv_state['pos'].retain_grad()
            drivable_computation_time = perf_counter()
            gradients, current_cost_agents = self.test_distance_map(input_cost_adv_state['pos'])
            (input_cost_adv_state['pos'][0]).backward(retain_graph=True, gradient=gradients*self.weight_drivable)
            drivable_computation_time = perf_counter() - drivable_computation_time
            self._accumulated_drivable_loss_agents += current_cost_agents
            current_cost = torch.sum(current_cost_agents)
            self._accumulated_drivable_loss += current_cost
            self._drivable_area_losses_per_step.append(current_cost)
            # (pos_prime).backward(retain_graph=True, gradient=gradients)
            # (new_pos[0]).backward(retain_graph=True, gradient=gradients)
            


            # print('gradient of the longitudinal 0 ', self._states['pos'].grad)
            # print('gradient of the longitudinal 1 ', input_cost_adv_state['pos'][0].grad)
            # print('gradient of the longitudinal 2 ', input_cost_adv_state['pos'].grad)
            # print('gradient of the longitudinal 3 ', self._states['speed'].grad)
            # print('gradient of the longitudinal 4 ', self._states['vel'].grad)
            # print('gradient of the longitudinal 5 ', torch.sum(self._actions['throttle'].grad))

        

        if 'drivable_king' in self._costs_to_use:
            
            drivable_computation_time = perf_counter()
            adv_rd_cost = self.adv_rd_cost(
                self._transpose_data_nondrivable_map,
                input_cost_adv_state['pos'],
                input_cost_adv_state['yaw'],
            )
            self._cost_dict['adv_rd'].append(adv_rd_cost*self.weight_drivable)
            drivable_computation_time = perf_counter() - drivable_computation_time
            current_cost = torch.sum(adv_rd_cost).detach().cpu().numpy()
            self._accumulated_drivable_loss += current_cost
            self._drivable_area_losses_per_step.append(current_cost)

            

        if 'fixed_dummy' in self._costs_to_use:
            dummy_cost, agents_costs = self.dummy_cost_fn(
                self.ego_extent,
                input_cost_adv_state,
                self.adv_extent,
                torch.unsqueeze(torch.unsqueeze(torch.tensor(self._fixed_point, device=device, dtype=torch.float64), dim=0), dim=0)
            )
            self._accumulated_collision_loss_agents += agents_costs
            self._cost_dict['dummy'].append(dummy_cost*self.weight_collision)
            current_cost =  torch.sum(dummy_cost).detach().cpu().numpy()
            self._accumulated_collision_loss += current_cost
            self._dummy_losses_per_step.append(current_cost)




        if 'moving_dummy' in self._costs_to_use:
            dummy_cost = self.dummy_cost_fn(
                input_cost_ego_state,
                self.ego_extent,
                input_cost_adv_state,
                self.adv_extent,
            )
            self._cost_dict['dummy'].append(dummy_cost*self.weight_collision)

        return drivable_computation_time

        

    def back_prop(self, last_optimization_step:bool):
        '''
        - find the initial actions from the existing trajectories for background agents
        - create variables for these extracted parameters

        '''
        cost_dict = self._cost_dict
        # aggregate costs and build total objective

        total_objective = 0

        if 'collision' in self._costs_to_use:
            
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
        
        
        if 'fixed_dummy' in self._costs_to_use or 'moving_dummy' in self._costs_to_use:

            cost_dict["dummy"] = torch.sum(
                torch.stack(cost_dict["dummy"], dim=-1),
            )

            total_objective = total_objective + cost_dict["dummy"]
        
        
        if 'drivable_us' in self._costs_to_use:

            self._optimizer_throttle.step()
            self._optimizer_steer.step()
            for i_agent in range(self._number_agents):
                if self._throttle_requires_grad:
                    self.throttle_gradients[i_agent] += self._throttle.grad[i_agent][:,0].detach()
                if self._steer_requires_grad:
                    self.steer_gradients[i_agent] += self._steer.grad[i_agent][:,0].detach()
                self.throttles[i_agent] = self._throttle[i_agent].detach()
                self.steers[i_agent] = self._steer[i_agent].detach()

            
            self._optimizer_throttle.zero_grad()
            self._optimizer_steer.zero_grad()

        if 'drivable_king' in self._costs_to_use:
        
            cost_dict["adv_rd"] = torch.mean(
                torch.stack(self._cost_dict["adv_rd"], dim=1),
                dim=1,
            )

            total_objective = total_objective + cost_dict["adv_rd"]
   

        # if col_metric != 1.0:
        # for param_group in self._optimizer.param_groups:
        #     for param in param_group['params']:
        #         for small_param in param:
        #             print('this is the param ', small_param.item())
        #             print('this is the gradient ', small_param.grad)
        # let us update the gradients for each agent

        print('the whole throttle ', self._throttle.grad)
        if 'fixed_dummy' in self._costs_to_use or 'moving_dummy' in self._costs_to_use or 'collision' in self._costs_to_use or 'drivable_king' in self._costs_to_use:
            total_objective.backward()
            self._optimizer_throttle.step()
            self._optimizer_steer.step()

        if 'drivable_king' in self._costs_to_use:
            self._optimizer_throttle.step()
            self._optimizer_steer.step()

        for i_agent in range(self._number_agents):
            if self._throttle_requires_grad:
                self.throttle_gradients[i_agent] += self._throttle.grad[i_agent][:,0].detach()
            if self._steer_requires_grad:
                self.steer_gradients[i_agent] += self._steer.grad[i_agent][:,0].detach()
            self.throttles[i_agent] = self._throttle[i_agent].detach()
            self.steers[i_agent] = self._steer[i_agent].detach()


        # for i_agent in range(self._number_agents):
        #     if self._throttle_requires_grad:
        #         self.throttle_gradients[i_agent] = self._throttle.grad[i_agent].detach().clone().cpu().numpy()
        #     if self._steer_requires_grad:
        #         self.steer_gradients[i_agent] = self._steer.grad[i_agent].detach().clone().cpu().numpy()
        #     self.throttles[i_agent] = self._throttle[i_agent].detach().clone().cpu().numpy()
        #     self.steers[i_agent] = self._steer[i_agent].detach().clone().cpu().numpy()

        

        self._optimizer_throttle.zero_grad()
        self._optimizer_steer.zero_grad()
        self._drivable_area_losses_per_opt.append(self._accumulated_drivable_loss)
        self._dummy_losses_per_opt.append(self._accumulated_collision_loss)
        self.drivable_loss_agents_per_opt.append(self._accumulated_drivable_loss_agents)
        self.dummy_loss_agents_per_opt.append(self._accumulated_collision_loss_agents)

        if last_optimization_step:

            self.plot_losses()




    def plot_losses(self):

        create_directory_if_not_exists('/home/kpegah/workspace/EXPERIMENTS')
        create_directory_if_not_exists(f'/home/kpegah/workspace/EXPERIMENTS/{self._simulation.scenario.scenario_name}')
        create_directory_if_not_exists(f'/home/kpegah/workspace/EXPERIMENTS/{self._simulation.scenario.scenario_name}/{self._experiment_name}')

       

        if 'drivable_king' in self._costs_to_use or 'drivable_us' in self._costs_to_use:


            plt.plot(self._drivable_area_losses_per_opt.detach().cpu().numpy())
            plt.gcf()
            plt.savefig(f'/home/kpegah/workspace/EXPERIMENTS/{self._simulation.scenario.scenario_name}/{self._experiment_name}/drivable_accumulated_loss.png')
            plt.show()
            plt.clf()


            print('this is the _drivable_area_losses_per_step ', self._drivable_area_losses_per_step)
            plt.plot(self._drivable_area_losses_per_step.detach().cpu().numpy(), label='cost')
            # plt.plot(self.throttles[3], label='throttle val')
            plt.legend() 
            plt.gcf()
            plt.savefig(f'/home/kpegah/workspace/EXPERIMENTS/{self._simulation.scenario.scenario_name}/{self._experiment_name}/drivable_cost_sim_steps.png')
            plt.show()
            plt.clf()


        if 'fixed_dummy' in self._costs_to_use or 'moving_dummy' in self._costs_to_use or 'collision' in self._costs_to_use:

            
            plt.plot(self._dummy_losses_per_opt.detach().cpu().numpy())
            plt.gcf()
            plt.savefig(f'/home/kpegah/workspace/EXPERIMENTS/{self._simulation.scenario.scenario_name}/{self._experiment_name}/dummy_accumulated_loss.png')
            plt.show()
            plt.clf()

  
            plt.plot(self._dummy_losses_per_step.detach().cpu().numpy(), label='cost')
            plt.legend() 
            plt.gcf()
            plt.savefig(f'/home/kpegah/workspace/EXPERIMENTS/{self._simulation.scenario.scenario_name}/{self._experiment_name}/fixeddummy_cost_sim_steps.png')
            plt.show()
            plt.clf()

        

        self.plot_loss_per_agent(self.drivable_loss_agents_per_opt.detach().cpu().numpy(), self.dummy_loss_agents_per_opt.detach().cpu().numpy())
    

    
    def plot_loss_per_agent(self, drivable_loss_per_opt, collision_loss_per_opt):

        '''
        the function to visualize the gradients registered by the hooks for the throttle and steer
        :params optimization_iteration
        '''

        drivable_loss_per_opt = np.array(drivable_loss_per_opt).transpose(1,0)
        collision_loss_per_opt = np.array(collision_loss_per_opt).transpose(1,0)

        create_directory_if_not_exists('/home/kpegah/workspace/EXPERIMENTS')
        create_directory_if_not_exists(f'/home/kpegah/workspace/EXPERIMENTS/{self._simulation.scenario.scenario_name}')
        create_directory_if_not_exists(f'/home/kpegah/workspace/EXPERIMENTS/{self._simulation.scenario.scenario_name}/{self._experiment_name}')

        plt.figure(figsize=(50, 30))

        # we have gradients accumulated for each step of the simulation, and for each of the agents, we know its position
        pos_agents = self.whole_state_buffers[0][0]['pos'][:self._number_agents, :]
        smallest_x, largest_x, smallest_y, largest_y = np.min(pos_agents[:,0]), np.max(pos_agents[:,0]), np.min(pos_agents[:,1]), np.max(pos_agents[:,1])
        width, height = (largest_x - smallest_x), (largest_y - smallest_y)

        fig_steer, ax_steer = plt.subplots(facecolor ='#a0d9f0')       
        ax_steer.set_xlim(smallest_x, largest_x)
        ax_steer.set_ylim(smallest_y, largest_y)


        colors = cm.rainbow(np.linspace(0, 1, self._number_agents))

        for i_agent in range(self._number_agents):
            # if not i_agent==3:
            #     continue
            # contsruct the gradient (throttle and steer) for the current agent
            # the position of the agent should be with respect to the figure
            relative_x, relative_y = (pos_agents[i_agent,0]-smallest_x)/width, (pos_agents[i_agent,1]-smallest_y)/height
            agent_losses = drivable_loss_per_opt[i_agent]

            subpos_steer = [relative_y, relative_x ,0.12 ,0.08]

            subax2 = add_subplot_axes(ax_steer,subpos_steer)

            subax2.plot(agent_losses, color=colors[i_agent])
        
        
        
        plt.gcf()
        plt.savefig(f'/home/kpegah/workspace/EXPERIMENTS/{self._simulation.scenario.scenario_name}/{self._experiment_name}/drivable_loss_evolution_agents_per_opt.png')
        plt.show()
        plt.clf()




        
        plt.figure(figsize=(50, 30))

        # we have gradients accumulated for each step of the simulation, and for each of the agents, we know its position
        pos_agents = self.whole_state_buffers[0][0]['pos'][:self._number_agents, :]
        smallest_x, largest_x, smallest_y, largest_y = np.min(pos_agents[:,0]), np.max(pos_agents[:,0]), np.min(pos_agents[:,1]), np.max(pos_agents[:,1])
        width, height = (largest_x - smallest_x), (largest_y - smallest_y)

        fig_steer, ax_steer = plt.subplots(facecolor ='#a0d9f0')       
        ax_steer.set_xlim(smallest_x, largest_x)
        ax_steer.set_ylim(smallest_y, largest_y)


        colors = cm.rainbow(np.linspace(0, 1, self._number_agents))

        for i_agent in range(self._number_agents):
            # if not i_agent==3:
            #     continue
            # contsruct the gradient (throttle and steer) for the current agent
            # the position of the agent should be with respect to the figure
            relative_x, relative_y = (pos_agents[i_agent,0]-smallest_x)/width, (pos_agents[i_agent,1]-smallest_y)/height
            agent_losses = collision_loss_per_opt[i_agent]

            subpos_steer = [relative_y, relative_x ,0.12 ,0.08]

            subax2 = add_subplot_axes(ax_steer,subpos_steer)

            subax2.plot(agent_losses, color=colors[i_agent])
        
        
        
        plt.gcf()
        plt.savefig(f'/home/kpegah/workspace/EXPERIMENTS/{self._simulation.scenario.scenario_name}/{self._experiment_name}/collition_loss_evolution_agents_per_opt.png')
        plt.show()
        plt.clf()




    def check_collision_at_iteration(self, iteration: int):
        pass

    def save_state_buffer(self):

        self.whole_state_buffers.append(self.state_buffer)

    def plots(self):


        create_directory_if_not_exists('/home/kpegah/workspace/EXPERIMENTS')
        create_directory_if_not_exists(f'/home/kpegah/workspace/EXPERIMENTS/{self._simulation.scenario.scenario_name}')
        create_directory_if_not_exists(f'/home/kpegah/workspace/EXPERIMENTS/{self._simulation.scenario.scenario_name}/{self._experiment_name}')
        create_directory_if_not_exists(f'/home/kpegah/workspace/EXPERIMENTS/{self._simulation.scenario.scenario_name}/{self._experiment_name}/Position_Evolution/')

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
            plt.savefig(f'/home/kpegah/workspace/EXPERIMENTS/{self._simulation.scenario.scenario_name}/{self._experiment_name}/Position_Evolution/pos_evolution_{opt_round}.png')
            plt.show()
            plt.clf()
      

    def visualize_grads_throttle(self, optimization_iteration: int):

        '''
        :params optimization_iteration
        '''

        self.throttle_gradients = torch.cat(self.throttle_gradients, dim=-1).cpu().numpy().reshape(self._number_agents,-1)


        create_directory_if_not_exists('/home/kpegah/workspace/EXPERIMENTS')
        create_directory_if_not_exists(f'/home/kpegah/workspace/EXPERIMENTS/{self._simulation.scenario.scenario_name}')
        create_directory_if_not_exists(f'/home/kpegah/workspace/EXPERIMENTS/{self._simulation.scenario.scenario_name}/{self._experiment_name}')
        create_directory_if_not_exists(f'/home/kpegah/workspace/EXPERIMENTS/{self._simulation.scenario.scenario_name}/{self._experiment_name}/Throttle_Grad_Evolution')

        plt.figure(figsize=(50, 30))

        # we have gradients accumulated for each step of the simulation, and for each of the agents, we know its position
        pos_agents = self.whole_state_buffers[0][0]['pos'][:self._number_agents, :]
        smallest_x, largest_x, smallest_y, largest_y = np.min(pos_agents[:,0]), np.max(pos_agents[:,0]), np.min(pos_agents[:,1]), np.max(pos_agents[:,1])
        # height_map, width_map = self._nondrivable_map_layer.shape[0], self._nondrivable_map_layer.shape[1]
        width, height = (largest_x - smallest_x), (largest_y - smallest_y)

        fig_throttle, ax_throttle = plt.subplots(facecolor ='#a0d9f0')       
        ax_throttle.set_xlim(smallest_x, largest_x)
        ax_throttle.set_ylim(smallest_y, largest_y)


        colors = cm.rainbow(np.linspace(0, 1, self._number_agents))

        for i_agent in range(self._number_agents):
            # if not i_agent==3:
            #     continue
            # contsruct the gradient (throttle and steer) for the current agent
            # the position of the agent should be with respect to the figure
            relative_x, relative_y = (pos_agents[i_agent,0]-smallest_x)/width, (pos_agents[i_agent,1]-smallest_y)/height
            # what should be the size of the plot for the gradients?? decide based on the values obtained for each agent
            current_throttle_grads = self.throttle_gradients[i_agent] # this is of the size of length of actions, and for each of the actions, 

            subpos_throttle = [relative_y, relative_x ,0.12 ,0.08]
            subax1 = add_subplot_axes(ax_throttle,subpos_throttle)

            # should first normalize the gradients in the x direction??
            subax1.plot(current_throttle_grads, color=colors[i_agent])
        
        
        
        plt.gcf()
        plt.savefig(f'/home/kpegah/workspace/EXPERIMENTS/{self._simulation.scenario.scenario_name}/{self._experiment_name}/Throttle_Grad_Evolution/{optimization_iteration}.png')
        plt.show()
        plt.clf()



    def visualize_grads_steer(self, optimization_iteration: int):

        '''
        the function to visualize the gradients registered by the hooks for the throttle and steer
        :params optimization_iteration
        '''

        self.steer_gradients = torch.cat(self.steer_gradients, dim=-1).cpu().numpy().reshape(self._number_agents,-1)

        create_directory_if_not_exists('/home/kpegah/workspace/EXPERIMENTS')
        create_directory_if_not_exists(f'/home/kpegah/workspace/EXPERIMENTS/{self._simulation.scenario.scenario_name}')
        create_directory_if_not_exists(f'/home/kpegah/workspace/EXPERIMENTS/{self._simulation.scenario.scenario_name}/{self._experiment_name}')
        create_directory_if_not_exists(f'/home/kpegah/workspace/EXPERIMENTS/{self._simulation.scenario.scenario_name}/{self._experiment_name}/Steer_Grad_Evolution')

        plt.figure(figsize=(50, 30))

        # we have gradients accumulated for each step of the simulation, and for each of the agents, we know its position
        pos_agents = self.whole_state_buffers[0][0]['pos'][:self._number_agents, :]
        smallest_x, largest_x, smallest_y, largest_y = np.min(pos_agents[:,0]), np.max(pos_agents[:,0]), np.min(pos_agents[:,1]), np.max(pos_agents[:,1])
        width, height = (largest_x - smallest_x), (largest_y - smallest_y)

        fig_steer, ax_steer = plt.subplots(facecolor ='#a0d9f0')       
        ax_steer.set_xlim(smallest_x, largest_x)
        ax_steer.set_ylim(smallest_y, largest_y)


        colors = cm.rainbow(np.linspace(0, 1, self._number_agents))

        for i_agent in range(self._number_agents):
            # if not i_agent==3:
            #     continue
            # contsruct the gradient (throttle and steer) for the current agent
            # the position of the agent should be with respect to the figure
            relative_x, relative_y = (pos_agents[i_agent,0]-smallest_x)/width, (pos_agents[i_agent,1]-smallest_y)/height
            current_steer_grads = self.steer_gradients[i_agent]

            subpos_steer = [relative_y, relative_x ,0.12 ,0.08]

            subax2 = add_subplot_axes(ax_steer,subpos_steer)

            subax2.plot(current_steer_grads, color=colors[i_agent])
        
        
        
        plt.gcf()
        plt.savefig(f'/home/kpegah/workspace/EXPERIMENTS/{self._simulation.scenario.scenario_name}/{self._experiment_name}/Steer_Grad_Evolution/{optimization_iteration}.png')
        plt.show()
        plt.clf()





    def visualize_throttle(self, optimization_iteration: int):

        '''
        :params optimization_iteration
        '''

        self.throttles = torch.cat(self.throttles, dim=-1).cpu().numpy().reshape(self._number_agents,-1)


        create_directory_if_not_exists('/home/kpegah/workspace/EXPERIMENTS')
        create_directory_if_not_exists(f'/home/kpegah/workspace/EXPERIMENTS/{self._simulation.scenario.scenario_name}')
        create_directory_if_not_exists(f'/home/kpegah/workspace/EXPERIMENTS/{self._simulation.scenario.scenario_name}/{self._experiment_name}')
        create_directory_if_not_exists(f'/home/kpegah/workspace/EXPERIMENTS/{self._simulation.scenario.scenario_name}/{self._experiment_name}/Throttle_Evolution')

        plt.figure(figsize=(50, 30))

        # we have gradients accumulated for each step of the simulation, and for each of the agents, we know its position
        pos_agents = self.whole_state_buffers[0][0]['pos'][:self._number_agents, :]
        smallest_x, largest_x, smallest_y, largest_y = np.min(pos_agents[:,0]), np.max(pos_agents[:,0]), np.min(pos_agents[:,1]), np.max(pos_agents[:,1])
        # height_map, width_map = self._nondrivable_map_layer.shape[0], self._nondrivable_map_layer.shape[1]
        width, height = (largest_x - smallest_x), (largest_y - smallest_y)

        fig_throttle, ax_throttle = plt.subplots(facecolor ='#a0d9f0')       
        ax_throttle.set_xlim(smallest_x, largest_x)
        ax_throttle.set_ylim(smallest_y, largest_y)


        colors = cm.rainbow(np.linspace(0, 1, self._number_agents))

        for i_agent in range(self._number_agents):
            # if not i_agent==3:
            #     continue
            # contsruct the gradient (throttle and steer) for the current agent
            # the position of the agent should be with respect to the figure
            relative_x, relative_y = (pos_agents[i_agent,0]-smallest_x)/width, (pos_agents[i_agent,1]-smallest_y)/height
            # what should be the size of the plot for the gradients?? decide based on the values obtained for each agent
            current_throttles = self.throttles[i_agent] # this is of the size of length of actions, and for each of the actions, 

            subpos_throttle = [relative_y, relative_x ,0.12 ,0.08]
            subax1 = add_subplot_axes(ax_throttle,subpos_throttle)

            # should first normalize the gradients in the x direction??
            subax1.plot(current_throttles, color=colors[i_agent])
        
        
        
        plt.gcf()
        plt.savefig(f'/home/kpegah/workspace/EXPERIMENTS/{self._simulation.scenario.scenario_name}/{self._experiment_name}/Throttle_Evolution/{optimization_iteration}.png')
        plt.show()
        plt.clf()



    def visualize_steer(self, optimization_iteration: int):

        '''
        the function to visualize the gradients registered by the hooks for the throttle and steer
        :params optimization_iteration
        '''

        self.steers = torch.cat(self.steers, dim=-1).cpu().numpy().reshape(self._number_agents,-1)

        create_directory_if_not_exists('/home/kpegah/workspace/EXPERIMENTS')
        create_directory_if_not_exists(f'/home/kpegah/workspace/EXPERIMENTS/{self._simulation.scenario.scenario_name}')
        create_directory_if_not_exists(f'/home/kpegah/workspace/EXPERIMENTS/{self._simulation.scenario.scenario_name}/{self._experiment_name}')
        create_directory_if_not_exists(f'/home/kpegah/workspace/EXPERIMENTS/{self._simulation.scenario.scenario_name}/{self._experiment_name}/Steer_Evolution')

        plt.figure(figsize=(50, 30))

        # we have gradients accumulated for each step of the simulation, and for each of the agents, we know its position
        pos_agents = self.whole_state_buffers[0][0]['pos'][:self._number_agents, :]
        smallest_x, largest_x, smallest_y, largest_y = np.min(pos_agents[:,0]), np.max(pos_agents[:,0]), np.min(pos_agents[:,1]), np.max(pos_agents[:,1])
        width, height = (largest_x - smallest_x), (largest_y - smallest_y)

        fig_steer, ax_steer = plt.subplots(facecolor ='#a0d9f0')       
        ax_steer.set_xlim(smallest_x, largest_x)
        ax_steer.set_ylim(smallest_y, largest_y)


        colors = cm.rainbow(np.linspace(0, 1, self._number_agents))

        for i_agent in range(self._number_agents):
            # if not i_agent==3:
            #     continue
            # contsruct the gradient (throttle and steer) for the current agent
            # the position of the agent should be with respect to the figure
            relative_x, relative_y = (pos_agents[i_agent,0]-smallest_x)/width, (pos_agents[i_agent,1]-smallest_y)/height
            current_steers = self.steers[i_agent]

            subpos_steer = [relative_y, relative_x ,0.12 ,0.08]

            subax2 = add_subplot_axes(ax_steer,subpos_steer)

            subax2.plot(current_steers, color=colors[i_agent])
        
        
        
        plt.gcf()
        plt.savefig(f'/home/kpegah/workspace/EXPERIMENTS/{self._simulation.scenario.scenario_name}/{self._experiment_name}/Steer_Evolution/{optimization_iteration}.png')
        plt.show()
        plt.clf()


    def routedeviation_king_heatmap(self):

        # B x N x S
        x_size = self._data_nondrivable_map.size()[0]
        y_size = self._data_nondrivable_map.size()[1]

        coords = torch.stack(
            torch.meshgrid(
                torch.linspace(0, x_size - 1, x_size),
                torch.linspace(0, y_size - 1, y_size)
            ),
            -1
        ).to(device=device)  # (H, W, 2)

        coords = coords[100:-100:100, 100:-100:100]

        coords = coords.reshape(-1, 2)
        coords = coords.unsqueeze(0)
        created_yaw = torch.zeros(1, coords.size()[1], 1, device=device)
        fn_cost = RouteDeviationCostRasterized(num_agents=coords.size()[1])
        adv_rd_cost = fn_cost.king_heatmap(
                self._transpose_data_nondrivable_map, 
                coords,
                created_yaw
            )
        
        print('this is the size of the adv_rd_cost ', adv_rd_cost.size())

        adv_rd_cost = adv_rd_cost.reshape(18,18)
        adv_rd_cost = adv_rd_cost.cpu().numpy()

        heatmap = np.zeros((200, 200))

        # Determine the center region to fill
        center_start = 10
        center_end = 190

        print('this is the final size ', adv_rd_cost.shape)
        scaled_array = zoom(adv_rd_cost, (10, 10), order=1)
        
        heatmap[center_start:center_end, center_start:center_end] = scaled_array

        # Display the heatmap
        plt.imshow(heatmap, cmap='viridis', interpolation='nearest')

        
        plt.gcf()
        plt.savefig(f'/home/kpegah/workspace/EXPERIMENTS/heatmap_king_driving_cost.png')
        plt.colorbar()
        plt.show()
        plt.clf()


    def routedeviation_ours_heatmap(self):

        # B x N x S
        x_size = self._data_nondrivable_map.size()[0]
        y_size = self._data_nondrivable_map.size()[1]

        coords = torch.stack(
            torch.meshgrid(
                torch.linspace(0, x_size - 1, x_size),
                torch.linspace(0, y_size - 1, y_size)
            ),
            -1
        ).to(device=device)  # (H, W, 2)

        coords = coords[100:-100:100, 100:-100:100]

        coords = coords.reshape(-1, 2)
        coords = coords.unsqueeze(0)
        adv_rd_cost = self.test_distance_map.ours_heatmap(coords)
        
        print('this is the size of the adv_rd_cost ', adv_rd_cost.size())

        adv_rd_cost = adv_rd_cost.reshape(18,18)
        adv_rd_cost = adv_rd_cost.cpu().numpy()

        heatmap = np.zeros((200, 200))

        # Determine the center region to fill
        center_start = 10
        center_end = 190

        print('this is the final size ', adv_rd_cost.shape)
        scaled_array = zoom(adv_rd_cost, (10, 10), order=1)
        
        heatmap[center_start:center_end, center_start:center_end] = scaled_array

        # Display the heatmap
        plt.imshow(heatmap, cmap='viridis', interpolation='nearest')

        
        plt.gcf()
        plt.savefig(f'/home/kpegah/workspace/EXPERIMENTS/heatmap_ours_driving_cost.png')
        plt.colorbar()
        plt.show()
        plt.clf()



    def routedeviation_king_heatmap_agents(self):

        
        input_cost_adv_state = self.get_adv_state(0)
        pos = input_cost_adv_state['pos'].detach()
        yaw = input_cost_adv_state['yaw'].detach()

       
        adv_rd_cost = self.adv_rd_cost.king_heatmap(self._transpose_data_nondrivable_map, pos, yaw)
        
        adv_rd_cost = adv_rd_cost.cpu().numpy()

        print('king cost agents ', adv_rd_cost)

    
        cp_map = self._data_nondrivable_map.detach().cpu().numpy()
        plt.figure(figsize = (10,10))

        plt.imshow(cp_map, interpolation='nearest')

        for idx in range(self._number_agents):
            transformed_adv_x, transformed_adv_y = self._states['pos'][idx].detach().cpu().numpy()[0], self._states['pos'][idx].detach().cpu().numpy()[1]
            plt.text(int(transformed_adv_y), int(transformed_adv_x), str("%.2f" % adv_rd_cost[idx]))
        
        plt.gcf()
        plt.savefig(f'/home/kpegah/workspace/EXPERIMENTS/map_king_drivingcost_agents.png')
        plt.colorbar()
        plt.show()
        plt.clf()


    def routedeviation_ours_heatmap_agents(self):

          
        input_cost_adv_state = self.get_adv_state(0)
        pos = input_cost_adv_state['pos'].detach()

       
       
        adv_rd_cost = self.test_distance_map.ours_heatmap(pos)
        
        adv_rd_cost = adv_rd_cost.cpu().numpy()
        print('agents ours cost ', adv_rd_cost)
    
        cp_map = self._data_nondrivable_map.detach().cpu().numpy()
        plt.figure(figsize = (10,10))

        plt.imshow(cp_map, interpolation='nearest')

        for idx in range(self._number_agents):
            transformed_adv_x, transformed_adv_y = self._states['pos'][idx].detach().cpu().numpy()[0], self._states['pos'][idx].detach().cpu().numpy()[1]
            plt.text(int(transformed_adv_y), int(transformed_adv_x), str("%.2f" % adv_rd_cost[idx]))
        
        plt.gcf()
        plt.savefig(f'/home/kpegah/workspace/EXPERIMENTS/map_ours_drivingcost_agents.png')
        plt.colorbar()
        plt.show()
        plt.clf()
