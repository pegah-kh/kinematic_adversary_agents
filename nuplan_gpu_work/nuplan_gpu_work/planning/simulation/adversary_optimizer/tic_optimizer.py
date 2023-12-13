# The collision cost affects all the agents in the scene (present in the first frame of the scenario)
# After the very first collision, the collided agent (with ego) is detected, and the trajectory of other
# agents is kept as log replay

from __future__ import annotations

import logging
from typing import List, Dict, Optional
from omegaconf import DictConfig
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
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
import wandb
import seaborn as sns
import pandas as pd




from nuplan_gpu_work.planning.simulation.adversary_optimizer.abstract_optimizer import AbstractOptimizer
from nuplan_gpu_work.planning.simulation.adversary_optimizer.agent_tracker.agent_lqr_tracker import LQRTracker 
from nuplan_gpu_work.planning.simulation.motion_model.bicycle_model import BicycleModel
from nuplan_gpu_work.planning.simulation.cost.king_costs import RouteDeviationCostRasterized, BatchedPolygonCollisionCost, DummyCost, DummyCost_FixedPoint, DrivableAreaCost

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
from nuplan.planning.simulation.trajectory.interpolated_trajectory import InterpolatedTrajectory
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
    def coord_to_matrix_vel(coord, yaw):

        return np.array(
            [
                np.cos(yaw[0]), -np.sin(yaw[0]), 0.0, coord[0],
                np.sin(yaw[0]), np.cos(yaw[0]), 0.0, coord[1],
                0.0, 0.0, 1.0, 0.0,
                0.0, 0.0, 0.0, 0.0,
            ]
        ).reshape(4,4).astype(np.float64)
    
    @staticmethod
    def from_matrix(matrix):

        assert matrix.shape == (4, 4), f"Expected 3x3 transformation matrix, but input matrix has shape {matrix.shape}"

        return matrix[0, 3], matrix[1, 3], np.arctan2(matrix[1, 0], matrix[0, 0])

    def pos_to_map(self, coord_x, coord_y, coord_yaw):

        transformed_adv_x, transformed_adv_y, transformed_yaw = self.from_matrix(self._transform@self.coord_to_matrix([coord_x, coord_y], [coord_yaw]))

        return transformed_adv_y - self._y_crop + 100/self._resolution, transformed_adv_x - self._x_crop + 100/self._resolution, np.pi/2-transformed_yaw
    
    def pos_to_map_vel(self, coord_x, coord_y, coord_yaw):
        
        transformed_adv_x, transformed_adv_y, transformed_yaw = self.from_matrix(self._transform@self.coord_to_matrix_vel([coord_x, coord_y], [coord_yaw]))

        return transformed_adv_y, transformed_adv_x, np.pi/2-transformed_yaw
    

    def pos_to_coord(self, map_x, map_y, map_yaw):


        transformed_adv_x, transformed_adv_y, transformed_yaw = self.from_matrix(self._inv_transform@self.coord_to_matrix([map_y+ self._x_crop - 100/self._resolution, map_x+ self._y_crop - 100/self._resolution], [np.pi/2-map_yaw]))

        return transformed_adv_x, transformed_adv_y, transformed_yaw


    def pos_to_coord_vel(self, map_x, map_y, map_yaw):


        transformed_adv_x, transformed_adv_y, transformed_yaw = self.from_matrix(self._inv_transform@self.coord_to_matrix_vel([map_y+ self._x_crop - 100/self._resolution, map_x+ self._y_crop - 100/self._resolution], [np.pi/2-map_yaw]))

        return transformed_adv_x, transformed_adv_y, transformed_yaw



# the OptimizationKING calls the simulation runner for iterations
class OptimizationKING():
    """
    Manager which executes multiple simulations with the same planner
    """

    def __init__(self, simulation: Simulation, 
                 planner: AbstractPlanner, 
                 tracker: DictConfig, 
                 motion_model: BicycleModel,
                opt_iterations: int, 
                max_opt_iterations: int, 
                lrs: List[float], 
                loss_weights: List[float], 
                costs_to_use: List[str] = ['fixed_dummy'], 
                requires_grad_params: List[str] = ['throttle', 'steer'], 
                experiment_name: str = 'map_cost', 
                project_name: str = 'finding_hyperparameter', 
                opt_jump: int = 0, 
                collision_strat: str = 'stopping_collider', 
                metric_report_path: str = '/home/kpegah/workspace',
                dense_opt_rounds:int = 0):


        self._experiment_name = experiment_name
        self._project_name = project_name
        self._metric_report_path = metric_report_path
        # super.__init__(self, simulation, planner)
        self._simulation = simulation
        self._planner = planner
        self._collision_strat = collision_strat
        self._use_original_states = False
       

        # densely estimating the dynamic parameters variables
        self._freq_multiplier = 1
        self.bm_iteration = 1
        self._dense_opt_rounds = dense_opt_rounds


        
        # the maximum number of optimization iterations
        self._opt_iterations = opt_iterations
        self._max_opt_iterations = max_opt_iterations
        self._optimization_jump = opt_jump


        # to check the dimensions later
        self._number_states:int = self._simulation._time_controller.number_of_iterations()
        # self._number_actions:int = self._number_states - 1
        self._number_actions:int = (self._number_states - 1) * self._freq_multiplier
        # use this trajectory sampling to get the initial observations, and obtain the actions accordingly
        self._observation_trajectory_sampling = TrajectorySampling(num_poses=self._number_actions, time_horizon=self._simulation.scenario.duration_s.time_s)
        self._horizon = self._number_actions
        self._number_agents: int = None

        print('hey hey hey ', self._observation_trajectory_sampling.interval_length)

        # tracker
        self._tracker = LQRTracker(**tracker, discretization_time=self._observation_trajectory_sampling.interval_length)
        print('interval ', self._observation_trajectory_sampling.interval_length)
         
        # motion model
        # self._motion_model = BicycleModel(self._observation_trajectory_sampling.interval_length)
        self._motion_model = motion_model

        assert self._observation_trajectory_sampling.interval_length==self._tracker._discretization_time, 'tracker discretization time of the tracker is not equal to interval_length of the sampled trajectory'
        

        tracked_objects: DetectionsTracks = self._simulation._observations.get_observation_at_iteration(0, self._observation_trajectory_sampling)
        agents: List[TrackedObject] = tracked_objects.tracked_objects.get_tracked_objects_of_type(TrackedObjectType.VEHICLE)
        self._agents = agents
        num_agents = len(agents)
        self._number_agents = num_agents

        self._ego_state = {'pos': torch.zeros(2).to(device=device), 'yaw': torch.zeros(1).to(device=device), 'vel': torch.zeros(2).to(device=device)}
        self._ego_state_whole = {'pos': torch.zeros(self._number_states, 2).to(device=device), 'yaw': torch.zeros(self._number_states, 1).to(device=device), 'vel': torch.zeros(self._number_states, 2).to(device=device)}
        self._ego_state_whole_np = {'pos': np.zeros((1,2), dtype=np.float64), 'yaw': np.zeros((1,1), dtype=np.float64), 'vel': np.zeros((1,2), dtype=np.float64)}

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


        # whether we are calling the planner during the round of optimization or it's fixed
        self._planner_on = True
        # to accumulate the states of the ego whenever the planner state gets updated => _ego_state_whole defined and started in init_ego_state

        self._original_corners = np.array([[1.15, 2.88], [1.15, -2.88], [-1.15, -2.88], [-1.15, 2.88]])/self._map_resolution
        self._collision_occurred = False
        self._collision_index = -1
        self._collision_iteration = 0
        self._collision_not_differentiable_state = None
        self._collided_agent_drivable_cost = 0
        self._adversary_collisions = 0
        self._collision_after_trajectory_changes = False
        



    def reset(self) -> None:
        '''
        inherited method.
        '''
        self._current_optimization_on_iteration = 1
        self.bm_iteration = 1
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

        self.throttle_gradients = [torch.zeros(self._number_actions).to(device=device) for _ in range(self._number_agents)]
        self.steer_gradients = [torch.zeros(self._number_actions).to(device=device) for _ in range(self._number_agents)]
        self.throttles = [torch.zeros(self._number_actions).to(device=device) for _ in range(self._number_agents)]
        self.steers = [torch.zeros(self._number_actions).to(device=device) for _ in range(self._number_agents)]
        self._drivable_area_losses_per_step = []
        self._dummy_losses_per_step = []


        self._first_forward = True



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

        self._states = {'pos': torch.zeros(self._number_agents, 2, dtype=torch.float64, requires_grad=True).to(device=device), 
                        'yaw': torch.zeros(self._number_agents, 1, dtype=torch.float64, requires_grad=True).to(device=device), 
                        'steering_angle': torch.zeros(self._number_agents, 1, dtype=torch.float64, requires_grad=True).to(device=device), 
                        'vel': torch.zeros(self._number_agents, 2, dtype=torch.float64, requires_grad=True).to(device=device), 
                        'accel': torch.zeros(self._number_agents, 1, dtype=torch.float64, requires_grad=True).to(device=device), 
                        'speed': torch.zeros(self._number_agents, 1, dtype=torch.float64, requires_grad=True).to(device=device)}
        

        self._states_original = {'pos': torch.zeros(self._number_agents, self._number_actions+1, 2, requires_grad=True).to(device=device), 
                                 'yaw': torch.zeros(self._number_agents,self._number_actions+1, 1, requires_grad=True).to(device=device),
                                 'vel': torch.zeros(self._number_agents, self._number_actions+1, 2, requires_grad=True).to(device=device), 
                                 'steering_angle': torch.zeros(self._number_agents, self._number_actions+1, 1, requires_grad=True).to(device=device), 
                                 'accel': torch.zeros(self._number_agents, self._number_actions+1, 1, requires_grad=True).to(device=device), 
                                 'speed': torch.zeros(self._number_agents, self._number_actions+1, 1, requires_grad=True).to(device=device)}
        
        # for key in self._states:
        #     self._states[key] = self._states[key] + 0

        # self._states['pos'].retain_grad()
        # self._states['yaw'].retain_grad()
        # self._states['speed'].retain_grad()

        self._initial_map_x, self._initial_map_y, self._initial_map_yaw = [], [], []
        self._initial_steering_rate = []
        self._initial_accel = []
        self._initial_map_vel_x, self._initial_map_vel_y = [], []

        before_transforming_x, before_transforming_y, before_transforming_yaw = [], [], []
        # after_transforming_x, after_transforming_y, after_transforming_yaw = [], [], []
        for idx, tracked_agent in enumerate(self._agents):

            coord_x, coord_y, coord_yaw = tracked_agent.predictions[0].valid_waypoints[0].x, tracked_agent.predictions[0].valid_waypoints[0].y, tracked_agent.predictions[0].valid_waypoints[0].heading
            coord_vel_x, coord_vel_y = tracked_agent.predictions[0].valid_waypoints[0].velocity.x, tracked_agent.predictions[0].valid_waypoints[0].velocity.y
            before_transforming_x.append(coord_x)
            before_transforming_y.append(coord_y)
            before_transforming_yaw.append(coord_yaw)
            map_x, map_y, map_yaw = self._convert_coord_map.pos_to_map(coord_x, coord_y, coord_yaw)
            _, _, second_point_map_yaw = self._convert_coord_map.pos_to_map(tracked_agent.predictions[0].valid_waypoints[1].x, tracked_agent.predictions[0].valid_waypoints[1].y, tracked_agent.predictions[0].valid_waypoints[1].heading)
            map_vel_x, map_vel_y, _ = self._convert_coord_map.pos_to_map_vel(coord_vel_x, coord_vel_y, coord_yaw)
            second_point_map_vel_x, second_point_map_vel_y, _ = self._convert_coord_map.pos_to_map_vel(tracked_agent.predictions[0].valid_waypoints[1].velocity.x, tracked_agent.predictions[0].valid_waypoints[1].velocity.y, tracked_agent.predictions[0].valid_waypoints[1].heading)

            # after_transforming_x.append(map_x)
            # after_transforming_y.append(map_y)
            # after_transforming_yaw.append(map_yaw)

            self._initial_map_x.append(map_x)
            self._initial_map_y.append(map_y)
            self._initial_map_yaw.append(map_yaw)
            self._initial_map_vel_x.append(map_vel_x)
            self._initial_map_vel_y.append(map_vel_y)

            # the important change
            self._states['pos'][idx] = torch.tensor([coord_x, coord_y], device=device, dtype=torch.float64, requires_grad=True)
            # self._states['pos'][idx].retain_grad()
            self._states['yaw'][idx] = torch.tensor([coord_yaw], device=device, dtype=torch.float64, requires_grad=True)
            # self._states['yaw'][idx].retain_grad()
            self._states['vel'][idx] = torch.tensor([coord_vel_x, coord_vel_y], device=device, requires_grad=True, dtype=torch.float64,)
            # self._states['vel'][idx] = torch.tensor([0., 0.], dtype=torch.float64, device=device, requires_grad=True)
            # self._states['speed'][idx] = torch.tensor([0.0], dtype=torch.float64, device=device, requires_grad=True)
            self._states['speed'][idx] = torch.tensor([np.linalg.norm(np.array([coord_vel_x, coord_vel_y]))], dtype=torch.float64, device=device, requires_grad=True)
            # self._states['accel'][idx] = torch.tensor([0.0], dtype=torch.float64, device=device, requires_grad=True)
            approx_initi_accel = (np.linalg.norm(np.array([coord_vel_x, coord_vel_y])) - np.linalg.norm(np.array([tracked_agent.predictions[0].valid_waypoints[1].velocity.x, tracked_agent.predictions[0].valid_waypoints[1].velocity.y])))/self._observation_trajectory_sampling.interval_length
            self._states['accel'][idx] = torch.tensor([approx_initi_accel], dtype=torch.float64, device=device, requires_grad=True)
            self._initial_accel.append(approx_initi_accel)
            approx_tire_steering_angle = np.arctan(3.089*(tracked_agent.predictions[0].valid_waypoints[1].heading - coord_yaw)/(self._observation_trajectory_sampling.interval_length*np.hypot(coord_vel_x, coord_vel_y)+1e-3))
            self._initial_steering_rate.append(approx_tire_steering_angle)
            self._states['steering_angle'][idx] = torch.clamp(torch.tensor([approx_tire_steering_angle], device=device, dtype=torch.float64, requires_grad=True), min=-torch.pi/3, max=torch.pi/3)
            # self._states['steering_angle'][idx] = torch.clamp(torch.tensor([0.0], device=device, dtype=torch.float64, requires_grad=True), min=-torch.pi/3, max=torch.pi/3)
        
        
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

        for key in self._states:
            self._states[key].requires_grad_(True).retain_grad()


    def reset_dynamic_states(self):


        # transformed_crop_x, transformed_crop_y, transformed_crop_yaw = self.from_matrix(self._map_transform@self.coord_to_matrix(self._ego_state['pos'].detach().cpu().numpy(), self._ego_state['yaw'].detach().cpu().numpy()))
        # new_pos[0,idx,:], new_yaw[0,idx,:] = self.crop_positions(new_pos[0,idx,:], new_yaw[0,idx,:], x_crop, y_crop, resolution, transform)

        self._states = {'pos': torch.zeros(self._number_agents, 2, dtype=torch.float64, requires_grad=True).to(device=device), 
                        'yaw': torch.zeros(self._number_agents, 1, dtype=torch.float64, requires_grad=True).to(device=device), 
                        'steering_angle': torch.zeros(self._number_agents, 1, dtype=torch.float64, requires_grad=True).to(device=device), 
                        'vel': torch.zeros(self._number_agents, 2, dtype=torch.float64, requires_grad=True).to(device=device), 
                        'accel': torch.zeros(self._number_agents, 1, dtype=torch.float64, requires_grad=True).to(device=device), 
                        'speed': torch.zeros(self._number_agents, 1, dtype=torch.float64, requires_grad=True).to(device=device)}
        
        # self._states['pos'].retain_grad()
        # self._states['yaw'].retain_grad()
        # self._states['speed'].retain_grad()
        for idx, _ in enumerate(self._agents):

            map_x, map_y, map_yaw = self._initial_map_x[idx], self._initial_map_y[idx], self._initial_map_yaw[idx]
            map_vel_x, map_vel_y = self._initial_map_vel_x[idx], self._initial_map_vel_y[idx]
            approx_tire_steering_angle = self._initial_steering_rate[idx]
            approx_accel = self._initial_accel[idx]

            self._states['pos'][idx] = torch.tensor([map_x, map_y], device=device, dtype=torch.float64, requires_grad=True)
            # self._states['pos'][idx].retain_grad()
            self._states['yaw'][idx] = torch.tensor([map_yaw], device=device, dtype=torch.float64, requires_grad=True)
            # self._states['yaw'][idx].retain_grad()
            self._states['vel'][idx] = torch.tensor([map_vel_x, map_vel_y], device=device, dtype=torch.float64, requires_grad=True)
            # self._states['vel'][idx] = torch.tensor([0., 0.], dtype=torch.float64, device=device, requires_grad=True)
            # self._states['speed'][idx] = torch.tensor([0.0], dtype=torch.float64, device=device, requires_grad=True)
            self._states['speed'][idx] = torch.tensor([np.linalg.norm(np.array([map_vel_x, map_vel_y]))], dtype=torch.float64, device=device, requires_grad=True)
            # self._states['accel'][idx] = torch.tensor([0.0], dtype=torch.float64, device=device, requires_grad=True)
            self._states['accel'][idx] = torch.tensor([approx_accel], dtype=torch.float64, device=device, requires_grad=True)
            # approx_tire_steering_angle = np.arctan(3.089*(tracked_agent.predictions[0].valid_waypoints[1].heading - tracked_agent.predictions[0].valid_waypoints[0].heading)/(self._observation_trajectory_sampling.interval_length*np.hypot(tracked_agent.velocity.x, tracked_agent.velocity.y)+1e-3))
            self._states['steering_angle'][idx] = torch.clamp(torch.tensor([approx_tire_steering_angle], device=device, dtype=torch.float64, requires_grad=True), min=-torch.pi/3, max=torch.pi/3)
            # self._states['steering_angle'][idx] = torch.clamp(torch.tensor([0.0], device=device, dtype=torch.float64, requires_grad=True), min=-torch.pi/3, max=torch.pi/3)
        
        
        for key in self._states:
            self._states[key].requires_grad_(True).retain_grad()

        
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

        wandb.init(
        # set the wandb project where this run will be logged
        project=self._project_name,
        name=self._experiment_name,
        
        # track hyperparameters and run metadata
        config={
            "lr1": self.lr_throttle,
            "lr2": self.lr_steer,
            "exp_name": self._experiment_name,
            "opt_iters": self._opt_iterations,
        }
    )

        # The initial coordinate of the ego that we will use to transform from coord space to map space
        # # transformed_crop_x, transformed_crop_y, transformed_crop_yaw = self.from_matrix(self._map_transform@self.coord_to_matrix(self._ego_state['pos'].detach().cpu().numpy(), self._ego_state['yaw'].detach().cpu().numpy()))
        # transformed_crop_x, transformed_crop_y, transformed_crop_yaw = self._ego_state_whole['pos']
        # self.transformed_crop_x = transformed_crop_x
        # self.transformed_crop_y = transformed_crop_y

            
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

        self._throttle_temp = [torch.nn.Parameter(
            torch.zeros(
                num_agents, 1, # the horizon parameter is the same as the numer of actions
                # and is one less than the number of iterations in the simulation
                device=device, 
                dtype=torch.float64
            ),
            requires_grad=False
        ) for _ in range(self._horizon)]

        self._steer_temp = [torch.nn.Parameter(
            torch.zeros(
                num_agents, 1,
                device=device, 
                dtype=torch.float64
            ),
            requires_grad=False
        ) for _ in range(self._horizon)]

        # if self._throttle_requires_grad:
        #     self._throttle.retain_grad()


        self._brake = torch.zeros(
                num_agents, 1
            )
        
        self._actions_temp = {'throttle': self._throttle_temp, 'steer': self._steer_temp, 'brake': self._brake}
        # self._states = {'pos': torch.zeros(num_agents, self._horizon + 1, 2), 'yaw': torch.zeros(num_agents, self._horizon + 1, 1), 'steering_angle': torch.zeros(num_agents, self._horizon + 1, 1), 'vel': torch.zeros(num_agents, self._horizon + 1, 2), 'accel': torch.zeros(num_agents, self._horizon + 1, 2)}
        self.init_dynamic_states()

        
        # colors = cm.rainbow(np.linspace(0, 1, self._number_agents))


        # initializing the parameters by the tracker and updating the current state using the bicycle model
        # bu we should add another forward function to the motion model to update all the states together

        # to plot the trajectory of each of the agents
        traj_agents_controller = [[] for _ in range(self._number_agents)]
        self.traj_agents_logged = [[] for _ in range(self._number_agents)]
        self.difference = [[] for _ in range(self._number_agents)]
                
        # to save the valid waypoints for each agent in the map coordinates
        self._map_valid_waypoints = [[] for i in range(self._number_agents)]
        for idx, tracked_agent in enumerate(self._agents):

           
            self._agent_indexes[tracked_agent.metadata.track_token] = idx
            self._agent_tokens[idx] = tracked_agent.metadata.track_token

            print('treating the agent ', tracked_agent.track_token)


            # initial_waypoint = tracked_agent.predictions[0].valid_waypoints[0]
            # initial_tire_steering_angle = np.arctan(3.089*(tracked_agent.predictions[0].valid_waypoints[1].heading - tracked_agent.predictions[0].valid_waypoints[0].heading)/(self._observation_trajectory_sampling.interval_length*np.hypot(tracked_agent.velocity.x, tracked_agent.velocity.y)+1e-3))

            

            # CORRECTED
            waypoints: List[Waypoint] = []
            for _time_step in range(len(tracked_agent.predictions[0].valid_waypoints)):
                coord_x, coord_y, coord_yaw = tracked_agent.predictions[0].valid_waypoints[_time_step].x, tracked_agent.predictions[0].valid_waypoints[_time_step].y, tracked_agent.predictions[0].valid_waypoints[_time_step].heading
                coord_vel_x, coord_vel_y = tracked_agent.predictions[0].valid_waypoints[_time_step].velocity.x, tracked_agent.predictions[0].valid_waypoints[_time_step].velocity.y
                map_x, map_y, map_yaw = self._convert_coord_map.pos_to_map(coord_x, coord_y, coord_yaw)
                map_vel_x, map_vel_y, _ = self._convert_coord_map.pos_to_map_vel(coord_vel_x, coord_vel_y, coord_yaw)
                current_waypoint = Waypoint(tracked_agent.predictions[0].valid_waypoints[_time_step].time_point, OrientedBox.from_new_pose(tracked_agent.box, StateSE2(map_x, map_y, map_yaw)), StateVector2D(map_vel_x, map_vel_y))

                # waypoints.append(current_waypoint) the change
                waypoints.append(tracked_agent.predictions[0].valid_waypoints[_time_step])


                self._states_original['pos'][idx, _time_step, :] = torch.tensor([map_x, map_y], dtype=torch.float64)
                self._states_original['vel'][idx, _time_step, :] = torch.tensor([map_vel_x, map_vel_y], dtype=torch.float64)
                self._states_original['yaw'][idx, _time_step, :] = torch.tensor([map_yaw], dtype=torch.float64)


                self.traj_agents_logged[idx].append(np.array([map_x, map_y]))

            # what would be state for the time steps that the vehicle idx is invisible to the ego
            # for _time_step in range(len(tracked_agent.predictions[0].valid_waypoints) - 1, self._horizon):
            #     self._states_original['pos'][idx, _time_step, :] = torch.tensor([map_x, map_y], dtype=torch.float64)
            #     self._states_original['vel'][idx, _time_step, :] = torch.tensor([map_vel_x, map_vel_y], dtype=torch.float64)
            #     self._states_original['yaw'][idx, _time_step, :] = torch.tensor([map_yaw], dtype=torch.float64)

            self._map_valid_waypoints[idx].append(waypoints)


            # transformed_initial_waypoint = waypoints[0]
            # transformed_initial_tire_steering_angle = np.arctan(3.089*(waypoints[1].heading - waypoints[0].heading)/(self._observation_trajectory_sampling.interval_length*np.hypot(transformed_initial_waypoint.velocity.x, transformed_initial_waypoint.velocity.y)+1e-3))
            # transformed_trajectory = InterpolatedTrajectory(waypoints)

            initial_waypoint = waypoints[0]
            initial_tire_steering_angle = np.arctan(3.089*(waypoints[1].heading - waypoints[0].heading)/(self._observation_trajectory_sampling.interval_length*np.hypot(initial_waypoint.velocity.x, initial_waypoint.velocity.y)+1e-3))


            # maybe the actions will have to be multiplied by the resolution?? when used on map coordinates, while the tracker is on real coordinates
            # why would the throttle be different? it should be /resolution but then it would probably not be stable.
            #even the collision occures when using the estimated throttle on real coordinates on map coordinates. Let's see the simulation.
            # what would be the next solution?? just do it on map coordinates and try to find the problem?? in the last optimizer.

            # initializing the dynamic state of the agents AT ALL TIME STEPS steps of the horizon using the lqr controller
            with torch.no_grad():
                for _time_step in range(len(tracked_agent.predictions[0].valid_waypoints) - 1):
                    # traj_agents_controller[idx].append(np.array([transformed_initial_waypoint.center.x, transformed_initial_waypoint.center.y]))
                    # using the timepoints of the simulation instead of those of predictions
                    throttle, steer = self._tracker.track_trajectory(tracked_agent.predictions[0].valid_waypoints[_time_step].time_point, tracked_agent.predictions[0].valid_waypoints[_time_step+1].time_point, initial_waypoint, tracked_agent.predictions[0].trajectory, initial_steering_angle=initial_tire_steering_angle)
                    # throttle, steer = self._tracker.track_trajectory(waypoints[_time_step].time_point, waypoints[_time_step+1].time_point, transformed_initial_waypoint, transformed_trajectory, initial_steering_angle=transformed_initial_tire_steering_angle)

                    # self._throttle[idx,_time_step], self._steer[idx,_time_step] = torch.Tensor([throttle]), torch.Tensor([steer])
                    self._throttle_temp[_time_step][idx], self._steer_temp[_time_step][idx] = torch.tensor([throttle], dtype=torch.float64).to(device), torch.tensor([steer], dtype=torch.float64).to(device)
                    # self._throttle_original_temp[_time_step][idx], self._steer_original_temp[_time_step][idx] = torch.tensor([throttle], dtype=torch.float64).to(device), torch.tensor([steer], dtype=torch.float64).to(device)
                    # updating the initial_waypoint based on the update from the bicycle model
                    if _time_step==0:
                        beginning_state = {_key: _value.clone().detach().to(device) for _key, _value in self.get_adv_state(id=idx).items()} # B*N*S
                        next_state = self._motion_model.forward_all(beginning_state, self.get_adv_actions_temp(_time_step, id=idx))
                    else:
                        next_state = self._motion_model.forward_all(next_state, self.get_adv_actions_temp(_time_step, id=idx))

                    # # FOR CAPTURING INITIAL STATES *********************** OPEN
                    # current_color = colors[idx]
                    # plt.quiver(*np.array([next_state['pos'].cpu()[0,0,0], next_state['pos'].cpu()[0,0,1]]), np.cos(next_state['yaw'].cpu()[0,0,0]), np.sin(next_state['yaw'].cpu()[0,0,0]), color=current_color, scale=40)
                    # # FOR CAPTURING INITIAL STATES *********************** CLOSE

                    # the difference is considered from downwards
                    possible_next_yaws = np.array([-2*np.pi, 0, 2*np.pi]) + next_state['yaw'].cpu().numpy()[0,0,0]
                    best_next_yaw_offset = np.argmin(np.abs(possible_next_yaws  - waypoints[_time_step+1].heading))
                    next_yaw = possible_next_yaws[best_next_yaw_offset]


                    # transformed_initial_waypoint = Waypoint(waypoints[_time_step+1].time_point, oriented_box=OrientedBox.from_new_pose(tracked_agent.box, StateSE2(next_state['pos'].cpu()[0,0,0], next_state['pos'].cpu()[0,0,1], next_yaw)), velocity=StateVector2D(next_state['vel'].cpu()[0,0,0], next_state['vel'].cpu()[0,0,1]))
                    # transformed_initial_tire_steering_angle = next_state['steering_angle'].cpu()

                    
                    initial_waypoint = Waypoint(waypoints[_time_step+1].time_point, oriented_box=OrientedBox.from_new_pose(tracked_agent.box, StateSE2(next_state['pos'].cpu()[0,0,0], next_state['pos'].cpu()[0,0,1], next_yaw)), velocity=StateVector2D(next_state['vel'].cpu()[0,0,0], next_state['vel'].cpu()[0,0,1]))
                    initial_tire_steering_angle = next_state['steering_angle'].cpu()

                    # 7, 8, 10, 12, 16, 17, 18, 19
                    if idx==10:
                        print('hi1 ', next_state['pos'].cpu().numpy()[0,0,0], '  ', next_state['pos'].cpu().numpy()[0,0,1], '   ', next_yaw)
                        print('hi2 ', waypoints[_time_step+1].x, '  ',  waypoints[_time_step+1].y, '  ',  waypoints[_time_step+1].heading)
                        print()
                        

                # traj_agents_controller[idx].append(np.array([transformed_initial_waypoint.center.x, transformed_initial_waypoint.center.y]))
                # for _time_step in range(len(waypoints) - 1, self._horizon):
                #     # using the timepoints of the simulation instead of those of predictions
                #     # throttle, steer = self._tracker.track_trajectory(tracked_agent.predictions[0].valid_waypoints[-2].time_point, tracked_agent.predictions[0].valid_waypoints[-1].time_point, initial_waypoint, tracked_agent.predictions[0].trajectory, initial_steering_angle=initial_tire_steering_angle)
                #     throttle, steer = self._tracker.track_trajectory(waypoints[-2].time_point, waypoints[-1].time_point, transformed_initial_waypoint, transformed_trajectory, initial_steering_angle=transformed_initial_tire_steering_angle)
                #     # self._throttle[idx,_time_step], self._steer[idx,_time_step] = torch.Tensor([throttle]), torch.Tensor([steer])
                #     self._throttle_temp[_time_step][idx], self._steer_temp[_time_step][idx] = torch.tensor([throttle], dtype=torch.float64).to(device), torch.tensor([steer], dtype=torch.float64).to(device)
                #     # self._throttle_original[_time_step][idx], self._steer_original[_time_step][idx] = torch.tensor([throttle], dtype=torch.float64).to(device), torch.tensor([steer], dtype=torch.float64).to(device)
                #     # updating the initial_waypoint based on the update from the bicycle model
                #     next_state = self._motion_model.forward_all(next_state, self.get_adv_actions_temp(_time_step, id=idx))


                #     # for the states in which the agent is not in the scene, use the bm to guess its next states, instead of staying stationary
                #     self._states_original['pos'][idx, _time_step, :] = next_state['pos']
                #     self._states_original['vel'][idx, _time_step, :] = next_state['vel']
                #     self._states_original['yaw'][idx, _time_step, :] = next_state['yaw']


                #     # # FOR CAPTURING INITIAL STATES *********************** OPEN
                #     # current_color = colors[idx]
                #     # plt.quiver(*np.array([next_state['pos'].cpu()[0,0,0], next_state['pos'].cpu()[0,0,1]]), np.cos(next_state['yaw'].cpu()[0,0,0]), np.sin(next_state['yaw'].cpu()[0,0,0]), color=current_color, scale=40)
                #     # # FOR CAPTURING INITIAL STATES *********************** CLOSE


                #     transformed_initial_waypoint = Waypoint(time_point=waypoints[-1].time_point, oriented_box=OrientedBox.from_new_pose(tracked_agent.box, StateSE2(next_state['pos'].cpu()[0,0,0], next_state['pos'].cpu()[0,0,1], next_state['yaw'].cpu()[0,0,0])), velocity=StateVector2D(next_state['vel'].cpu()[0,0,0], next_state['vel'].cpu()[0,0,1]))
                #     transformed_initial_tire_steering_angle = next_state['steering_angle'].cpu()

                #     if idx == 10:
                #         print('hi2 ',next_state['pos'].cpu()[0,0,0], '  ', next_state['pos'].cpu()[0,0,1])
            

            # # plotting the logged and the controlled trajectories
            # create_directory_if_not_exists(f'/home/kpegah/workspace/EXPERIMENTS')
            # create_directory_if_not_exists(f'/home/kpegah/workspace/EXPERIMENTS/{self._simulation.scenario.scenario_name}')
            # create_directory_if_not_exists(f'/home/kpegah/workspace/EXPERIMENTS/{self._simulation.scenario.scenario_name}/{self._experiment_name}')
            # create_directory_if_not_exists(f'/home/kpegah/workspace/EXPERIMENTS/{self._simulation.scenario.scenario_name}/{self._experiment_name}/Controller')

            # _difference = np.linalg.norm(np.array(traj_agents_controller[idx]) - np.array(self.traj_agents_logged[idx]), axis=-1)
            # self.difference[idx].append(_difference)

            
            # plt.plot(difference)
            # plt.gcf()
            # plt.savefig(f'/home/kpegah/workspace/EXPERIMENTS/{self._simulation.scenario.scenario_name}/{self._experiment_name}/Controller/MSE_{idx}.png')
            # plt.show()
            # plt.clf()

    

        

        # optimization_params = [self._throttle, self._steer]
        # self._optimizer_throttle = torch.optim.Adam([self._throttle], lr=self.lr_throttle, betas=(0.8, 0.99))
        # self._optimizer_steer = torch.optim.Adam([self._steer], lr=self.lr_steer, betas=(0.8, 0.99))
    
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

    
    def reshape_actions(self):
        # reshaping the actions to bring the horizon component into the tensor, so that we can use a single optimizer for all the actions in time


        self._throttle = torch.nn.Parameter(
            torch.zeros(
                self._number_agents, self._horizon, 1, # the horizon parameter is the same as the numer of actions
                # and is one less than the number of iterations in the simulation
                device=device,
                dtype=torch.float64
            ),
            requires_grad=self._throttle_requires_grad
        )
        if self._throttle_requires_grad:
            self._throttle.retain_grad()
        self._throttle_original = torch.nn.Parameter(
            torch.zeros(
                self._number_agents, self._horizon, 1, # the horizon parameter is the same as the numer of actions
                # and is one less than the number of iterations in the simulation
                device=device,
                dtype=torch.float64
            ),
            requires_grad=False
        )
        self._steer = torch.nn.Parameter(
            torch.zeros(
                self._number_agents, self._horizon, 1,
                device=device,
                dtype=torch.float64
            ),
            requires_grad=self._steer_requires_grad
        )
        if self._steer_requires_grad:
            self._steer.retain_grad()
        self._steer_original = torch.nn.Parameter(
            torch.zeros(
                self._number_agents, self._horizon, 1,
                device=device,
                dtype=torch.float64
            ),
            requires_grad=False
        )

        with torch.no_grad():

            for time_step in range(self._number_actions):
                self._throttle[:, time_step, 0] = self._throttle_temp[time_step][:,0]*self._map_resolution
                self._throttle_original[:, time_step, 0] = self._throttle_temp[time_step][:,0]*self._map_resolution
                self._steer[:, time_step, 0] = self._steer_temp[time_step][:,0]
                self._steer_original[:, time_step, 0] = self._steer_temp[time_step][:,0]

            

        self._actions = {'throttle': self._throttle, 'steer': self._steer, 'brake': self._brake}
        
        self._optimizer_throttle = torch.optim.Adam([self._throttle], lr=self.lr_throttle, betas=(0.8, 0.99))
        self._optimizer_steer = torch.optim.Adam([self._steer], lr=self.lr_steer, betas=(0.8, 0.99))

    

    

    def optimize_all_actions(self):
        '''
        This function optimizes the dynamic parameters, throttle and steer, in order to get closer to the dense trajectory points when passed through the bicycle model

        the intialization of the actions is given by tracking the interpolated trajectory.
        From these initilized actions, we should make the trajectory closer to the interpolated one, by optimizing the taken actions at each step.
        optimizing the actions for now only for ['6e0f1dbf087e570f']

        1. The optimized actions of step t pass through the bicycle model to give the state t+1 [for now for only the selected agent]
        2. From the state t+1 the loss is calculated [which is the MSE with the interpolated location of the agent at the step t+1]
        3. all the parameters, except those of that step are frozen, and the loss that gets backpropagated updates the actions at step t+1
        4. the next state is computed from the updated action, and .....


        difficulty: the actions of all the time steps are encoded as one. just create new variables from the the initial ones, and after optimization, put them into the actual one.
        difficulty: the function get_adv_state and actions will no longer be of use and the correctness of dimensions should be handled manually.
        '''

        n_rounds = self._dense_opt_rounds

        error_reductions = []
        movement_extents = []

        err_before_opt = []
        err_after_opt = []
        err_after_filtered_opt = []

        filtering = True

        
        def get_optimize_state(current_state):
            new_state = {key:value.detach().requires_grad_(True) for key,value in current_state.items()}
            return new_state
        
        def detached_clone_of_dict(current_state):
            new_state = {key:value.detach().clone() for key,value in current_state.items()}
            return new_state
        
        def clone_of_dict(current_state):
            new_state = {key:value.clone() for key,value in current_state.items()}
            return new_state

        def compute_total_distance(waypoints):

            differences = np.array(waypoints[1:]) - np.array(waypoints[:-1])
            distances = np.linalg.norm(differences, axis=1)
            print('this is the size of the distances ', distances.shape)
            total_distance = np.sum(distances)

            return total_distance
        
        traj_agents_controller = [[] for _ in range(self._number_agents)]
        for idx, tracked_agent in enumerate(self._agents):

            
            target_throttle = [self._throttle_temp[t][idx,:].detach().requires_grad_(True) for t in range(self._horizon)]
            target_steer = [self._steer_temp[t][idx,:].detach().requires_grad_(True) for t in range(self._horizon)]


            # optimizers_throttle = [torch.optim.RMSprop([throttle_param], lr=0.01) for throttle_param in target_throttle]
            # optimizers_steer = [torch.optim.RMSprop([steer_param], lr=0.01) for steer_param in target_steer]

            
            optimizers_throttle = [torch.optim.Adam([throttle_param], lr=0.0001) for throttle_param in target_throttle]
            optimizers_steer = [torch.optim.Adam([steer_param], lr=0.00005) for steer_param in target_steer]
            schedulers_throttle = [ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=False) for optimizer in optimizers_throttle]
            schedulers_steer = [ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=False) for optimizer in optimizers_steer]
            loss = torch.nn.MSELoss()

            def get_optimize_actions(iteration: int):
                return {'throttle': torch.unsqueeze(torch.unsqueeze(target_throttle[iteration], dim=0), dim=0),
                        'steer': torch.unsqueeze(torch.unsqueeze(target_steer[iteration], dim=0), dim=0),
                        'brake': torch.unsqueeze(self._brake, dim=0)}
            

            first_state:Dict[str, torch.TensorType] = {'pos': torch.zeros(1, 1, 2, requires_grad=True).to(device=device), 
                                                       'yaw': torch.zeros(1, 1, 1, requires_grad=True).to(device=device), 
                                                       'steering_angle': torch.zeros(1, 1, 1, requires_grad=True).to(device=device), 
                                                       'vel': torch.zeros(1, 1, 2, requires_grad=True).to(device=device), 
                                                       'accel': torch.zeros(1, 1, 1, requires_grad=True).to(device=device), 
                                                       'speed': torch.zeros(1, 1, 1, requires_grad=True).to(device=device)}


            tensor_waypoints_pos = [torch.unsqueeze(torch.unsqueeze(torch.tensor([waypoint.x, waypoint.y], dtype=torch.float64), dim=0), dim=0).to(device) for waypoint in self._map_valid_waypoints[idx][0][1:]]
            tensor_waypoints_yaw = [torch.unsqueeze(torch.unsqueeze(torch.tensor([waypoint.heading], dtype=torch.float64), dim=0), dim=0).to(device) for waypoint in self._map_valid_waypoints[idx][0][1:]]
            waypoints_pos = np.array([[waypoint.x, waypoint.y] for waypoint in self._map_valid_waypoints[idx][0]]).reshape(len(self._map_valid_waypoints[idx][0]),2)
            
            if float(compute_total_distance(waypoints_pos))<400 or not filtering:
                continue
            movement_extents.append(float(compute_total_distance(waypoints_pos)))
            # tensor_waypoints_yaw = [torch.unsqueeze(torch.unsqueeze(torch.tensor([waypoint.heading], dtype=torch.float64), dim=0), dim=0).to(device) for waypoint in tracked_agent.predictions[0].valid_waypoints]

            for n_round in range(n_rounds):

                traj_agents_controller[idx] = []
                with torch.no_grad():
                    # agent predicted waypoints in map coordinates
                    agent_initial_state: Waypoint = self._map_valid_waypoints[idx][0][0]
                    first_state['pos'] = torch.unsqueeze(torch.unsqueeze(torch.tensor([agent_initial_state.x, agent_initial_state.y], device=device, dtype=torch.float64), dim=0), dim=0)
                    first_state['yaw'] = torch.unsqueeze(torch.unsqueeze(torch.tensor([agent_initial_state.heading], device=device, dtype=torch.float64), dim=0), dim=0)
                    first_state['vel'] = torch.unsqueeze(torch.unsqueeze(torch.tensor([agent_initial_state.velocity.x, agent_initial_state.velocity.y], device=device, dtype=torch.float64), dim=0), dim=0)
                    first_state['accel'] = torch.unsqueeze(torch.unsqueeze(torch.tensor([0.], device=device, dtype=torch.float64), dim=0), dim=0)
                    approx_tire_steering_angle = np.arctan(3.089*(self._map_valid_waypoints[idx][0][1].heading - agent_initial_state.heading)/(self._observation_trajectory_sampling.interval_length*np.hypot(agent_initial_state.velocity.x, agent_initial_state.velocity.y)+1e-3))
                    first_state['steering_angle'] = torch.clamp(torch.unsqueeze(torch.unsqueeze(torch.tensor([approx_tire_steering_angle], device=device, dtype=torch.float64), dim=0), dim=0), min=-torch.pi/3, max=torch.pi/3)
                    first_state['speed'] = torch.unsqueeze(torch.unsqueeze(torch.tensor([np.linalg.norm(np.array([agent_initial_state.velocity.x, agent_initial_state.velocity.y]))], dtype=torch.float64, device=device), dim=0), dim=0)

                
                
                accumulated_loss = 0
                traj_len = len(self._map_valid_waypoints[idx][0]) - 1

                # traj_agents_controller[idx].append([first_state['pos'].cpu().detach().numpy()[0,0,0], first_state['pos'].cpu().detach().numpy()[0,0,1]])
                # for _time_step in range(traj_len):

                #     next_state = self._motion_model.forward(get_optimize_state(first_state), get_optimize_actions(iteration=_time_step), track_token=tracked_agent.track_token, iter=_time_step, plotter=False)
                #     # computing the MSE loss between the position component of the state and tensor_waypoints

                #     current_loss = loss(next_state['pos'], tensor_waypoints_pos[_time_step])
                    
                #     accumulated_loss += current_loss.detach().cpu().numpy()
                #     current_loss.backward(retain_graph=True)
                #     optimizers_throttle[_time_step].step()
                #     optimizers_throttle[_time_step].zero_grad()
                #     # schedulers_throttle[_time_step].step(current_loss)
                #     # schedulers_steer[_time_step].step(current_loss)

                #     current_loss = loss(next_state['yaw'], tensor_waypoints_yaw[_time_step])
                    
                #     accumulated_loss += current_loss.detach().cpu().numpy()
                #     current_loss.backward()
                #     optimizers_steer[_time_step].step()
                #     optimizers_steer[_time_step].zero_grad()
                    

                #     # the first_state should now be updated based on the optimized actions, using the bicycle model again but without any gradients
                #     with torch.no_grad():
                #         next_state = self._motion_model.forward(get_optimize_state(first_state), get_optimize_actions(iteration=_time_step), track_token=tracked_agent.track_token, iter=_time_step, plotter=False)

                #     first_state = {key:value.detach().requires_grad_(True) for key,value in next_state.items()}
                #     traj_agents_controller[idx].append([first_state['pos'].cpu().detach().numpy()[0,0,0], first_state['pos'].cpu().detach().numpy()[0,0,1]])


                last_optimized_time_step = -1
                bm_steps_before_opt = 1000
                _time_step = 0
                dummy_counter = 0
                while _time_step<traj_len:
                    bm_step = 0
                    cp_time_step = _time_step
                    cp_first_state = detached_clone_of_dict(first_state)
                    current_loss = 0
                    first_state = get_optimize_state(first_state)
                    losses_pos = []
                    losses_yaw = []
                    
                    while bm_step<bm_steps_before_opt and _time_step<traj_len:
                        # traj_agents_controller[idx].append([first_state['pos'].cpu().detach().numpy()[0,0,0], first_state['pos'].cpu().detach().numpy()[0,0,1]])
                        next_state = self._motion_model.forward(first_state, get_optimize_actions(iteration=_time_step), track_token=tracked_agent.track_token, iter=_time_step, plotter=False)
                        # computing the MSE loss between the position component of the state and tensor_waypoints
                        current_loss_pos = loss(next_state['pos'], tensor_waypoints_pos[_time_step])
                        current_loss_yaw = loss(next_state['yaw'], tensor_waypoints_yaw[_time_step])
                        # print('compare ', next_state['pos'], '\n', tensor_waypoints_pos[_time_step], '\n', current_loss)
                        losses_pos.append(current_loss_pos)
                        losses_yaw.append(current_loss_yaw)
                        accumulated_loss += current_loss_pos.clone().detach().cpu().numpy() + current_loss_yaw.clone().detach().cpu().numpy()
                        # first_state = next_state
                        first_state = clone_of_dict(next_state)

                        _time_step += 1
                        bm_step += 1
                    

                    current_loss_pos = torch.sum(torch.stack(losses_pos))
                    current_loss_yaw = torch.sum(torch.stack(losses_yaw))
                    print('this is the time step ', _time_step, ' and round ', n_round)
                    current_loss_pos.backward(retain_graph=True)

                    for optimize_indice in range(last_optimized_time_step+1, _time_step):
                        
                        optimizers_throttle[optimize_indice].step()
                        # optimizers_steer[optimize_indice].step()

                        # print('the gradient value for that time step ', target_throttle[optimize_indice].grad)
                        schedulers_throttle[optimize_indice].step(current_loss)
                        # schedulers_steer[optimize_indice].step(current_loss)
                        optimizers_throttle[optimize_indice].zero_grad()
                        # optimizers_steer[optimize_indice].zero_grad()


                        dummy_counter += 1

                    # if we want to optimize the steer and the throttle separately
                    current_loss_yaw.backward()
                    for optimize_indice in range(last_optimized_time_step+1, _time_step):
                        
                        optimizers_steer[optimize_indice].step()

                        # print('the gradient value for that time step ', target_throttle[optimize_indice].grad)
                        schedulers_steer[optimize_indice].step(current_loss)
                        optimizers_steer[optimize_indice].zero_grad()
                        
                    

                    # the first_state should now be updated based on the optimized actions, using the bicycle model again but without any gradients
                    with torch.no_grad():
                        bm_step = 0
                        while bm_step<bm_steps_before_opt and cp_time_step<traj_len:

                            traj_agents_controller[idx].append([cp_first_state['pos'].cpu().detach().numpy()[0,0,0], cp_first_state['pos'].cpu().detach().numpy()[0,0,1]])

                            next_state = self._motion_model.forward(get_optimize_state(cp_first_state), get_optimize_actions(iteration=cp_time_step), track_token=tracked_agent.track_token, iter=cp_time_step, plotter=False)
                            cp_first_state = next_state

                            cp_time_step += 1
                            bm_step += 1


                    last_optimized_time_step = _time_step-1
                    first_state = {key:value.detach().requires_grad_(True) for key,value in next_state.items()}
                traj_agents_controller[idx].append([first_state['pos'].cpu().detach().numpy()[0,0,0], first_state['pos'].cpu().detach().numpy()[0,0,1]])


                print('this is is the size ', np.array(traj_agents_controller[idx]).shape)
                _difference = np.linalg.norm(np.array(traj_agents_controller[idx]) - np.array(self.traj_agents_logged[idx]), axis=-1)
                self.difference[idx].append(_difference)
            

                print('********************************************************************')
                print('*************** this is the accumulated loss ', n_round, '  ', accumulated_loss)
                if n_round==0:
                    error_reduction = accumulated_loss
                    err_before_opt.append(float(error_reduction))
                    # for those for which we don't want to optimize, we consider the very first loss
                    if not float(compute_total_distance(waypoints_pos))<20:
                        err_after_filtered_opt.append(accumulated_loss)
                if n_round==(n_rounds-1):
                    error_reduction = error_reduction - accumulated_loss
                    error_reductions.append(float(error_reduction))
                    err_after_opt.append(float(accumulated_loss))
                    if float(compute_total_distance(waypoints_pos))<20:
                        err_after_filtered_opt.append(accumulated_loss)


            # just replace the real actions with the improved ones
            with torch.no_grad():
                for _time_step in range(len(tracked_agent.predictions[0].valid_waypoints) - 1):
                    self._throttle_temp[_time_step][idx], self._steer_temp[_time_step][idx] = target_throttle[_time_step].detach().to(device), target_steer[_time_step].detach().to(device)

            
            # plotting the logged and the controlled trajectories
            create_directory_if_not_exists(f'/home/kpegah/workspace/EXPERIMENTS')
            create_directory_if_not_exists(f'/home/kpegah/workspace/EXPERIMENTS/{self._simulation.scenario.scenario_name}')
            create_directory_if_not_exists(f'/home/kpegah/workspace/EXPERIMENTS/{self._simulation.scenario.scenario_name}/{self._experiment_name}')
            create_directory_if_not_exists(f'/home/kpegah/workspace/EXPERIMENTS/{self._simulation.scenario.scenario_name}/{self._experiment_name}/Controller')
            for plot_idx, plot_difference in enumerate(self.difference[idx]):
                plt.plot(plot_difference, label=plot_idx)

            plt.legend()
            plt.gcf()
            plt.savefig(f'/home/kpegah/workspace/EXPERIMENTS/{self._simulation.scenario.scenario_name}/{self._experiment_name}/Controller/MSE_{idx}.png')
            plt.show()
            plt.clf()


                

        create_directory_if_not_exists('/home/kpegah/workspace/DENSE')
        create_directory_if_not_exists(f'/home/kpegah/workspace/DENSE/{self._simulation.scenario.scenario_name}')
        create_directory_if_not_exists(f'/home/kpegah/workspace/DENSE/{self._simulation.scenario.scenario_name}/{self._experiment_name}')

        plt.figure(figsize=(10, 10))
        plt.scatter(movement_extents, error_reductions)
        plt.xlabel('Movement Extents')
        plt.ylabel('Error Reductions')
        
        plt.gcf()
        plt.savefig(f'/home/kpegah/workspace/DENSE/{self._simulation.scenario.scenario_name}/{self._experiment_name}/err_red_vs_ext.png')
        plt.show()
        plt.clf()

        # Err reduction vs the movement extent
        # dirstribution of the error before and after the optimization

        df = pd.DataFrame({
            'Values': err_before_opt + err_after_opt + err_after_filtered_opt,
            'Group': ['Before Optimization'] * len(err_before_opt) + ['After Optimization'] * len(err_after_opt) + ['Filtered Optimization'] * len(err_after_filtered_opt)
        })

        # Plot kernel density plots
        plt.figure(figsize=(8, 5))
        sns.kdeplot(data=df, x='Values', hue='Group', fill=True, common_norm=False, palette="husl")
        plt.xlim(left=0)
        plt.title('Kernel Density Plot Before and After Optimization')

        
        plt.gcf()
        plt.savefig(f'/home/kpegah/workspace/DENSE/{self._simulation.scenario.scenario_name}/{self._experiment_name}/err_dist_before_after.png')
        plt.show()
        plt.clf()





    def print_initial_actions(self, id:int):
        '''
        :params id:int is the agent for which we want to print actions
        '''

        steers = [self._actions['steer'][id,t,0].data for t in range(self._horizon)]
        throttles = [self._actions['throttle'][id,t,0].data for t in range(self._horizon)]
        print(f'the steers for the agent {id} is {steers}')
        print(f'the throttles for the agent {id} is {throttles}')
        
    # function adapted from https://github.com/autonomousvision/king/blob/main/proxy_simulator/simulator.py
    def get_adv_state(self, id:Optional(int) = None):
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


            adv_state[substate].requires_grad_(True).retain_grad()

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

    def get_adv_actions_temp(self, current_iteration:int = 0, id:Optional(int) = None):
        """
        the same as the above function, but to get actions of agents

        returns:
                the actions of adversary agents at the current iteration of shape B x N x S
        """
        adv_action = {}
        for substate in self._actions_temp.keys():
            if id == None:
                if not substate=='brake':
                    adv_action.update({substate: torch.unsqueeze(self._actions_temp[substate][current_iteration][:, ...], dim=0)})
                else:
                    adv_action.update({substate:  torch.unsqueeze(self._actions_temp[substate], dim=0)})
            else:
                # we index for id+1 since id 0 in the tensor is the ego agent
                if not substate=='brake':
                    adv_action.update(
                        {substate: torch.unsqueeze(self._actions_temp[substate][current_iteration][id:id+1, ...], dim=0)}
                    )
                else:
                    adv_action.update(
                        {substate:  torch.unsqueeze(self._actions_temp[substate][id:id+1, ...], dim=0)}
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
            self._states[substate].requires_grad_(True).retain_grad()

    def set_adv_state_to_original(self, next_state: Dict[str, torch.TensorType] = None, next_iteration:int = 1, exception_agent_index:int = 0):
        """
        Set the current adversarial state to the original one extracted during the initialization

        :param next_state: setting the parameters of the next_state, a dict of type[str, Tnesor]
                           where the Tensor contains the state for each agent
        Returns:
            None
        """


        for substate in self._states.keys():
            attached_states = []
            if exception_agent_index>0:
                attached_states.append(self._states_original[substate][0:exception_agent_index, next_iteration, ...])
            attached_states.append(next_state[substate][0, exception_agent_index:exception_agent_index+1, ...])
            if exception_agent_index+1<self._number_agents:
                attached_states.append(self._states_original[substate][exception_agent_index+1:, next_iteration, ...])


            self._states[substate] = torch.cat(
                attached_states,
                dim=0,
            )
            self._states[substate]


    # Question: what to do with the very first state? the very first state is always the same...
    # so the update should only be performed after the very first round of simulation??
    def step(self, current_iteration:int) -> Dict[str, ((float, float), (float, float), float)]:
        '''
        to update the state of the agents at the 'current_iterarion' using the actions taken at step 'current_iterarion-1'.

        in the case where the 'current_iterarion' is 0, the state does simply not get updated.

        return:
                a dictionary where the keys are tokens of agents, and the values are (pos, velocity, yaw)
        '''


        if self._collision_occurred and self._collision_strat=='back_to_after_bm':
            if current_iteration==1:
                self.actions_to_original()
                self.stopping_collider()
            return self.step_after_collision(current_iteration)
        elif self._collision_occurred and self._collision_strat=='back_to_before_bm':
            if current_iteration==1:
                self.use_original_states()
                self.stopping_collider()
            return self.step_after_collision(current_iteration)
        elif self._collision_occurred and self._collision_strat=='stopping_collider':
            if current_iteration==1:
                self.stopping_collider()
            return self.step_after_collision(current_iteration)
        
                
        if current_iteration is None:
            return
        
        # the trajectory even from the very start seems not to reach the trajectory of the ego, are we really registering sth before the opetimization?
        # we do the 
        

        # TODO : updating the list of agents at each iteration
        #        some agents may go inactive and untracked, their predicted trajectory should be filled in the untracked spaces?
        #        or maybe we can only work on the agents that remain tracked during the entire simulation
        

        not_differentiable_state = {}  
        # if not current_iteration==0:
        #     '''
        #     for id_tracked_agent in range(self._number_agents):
                
        #         self.set_adv_state(self._motion_model.forward(self.get_adv_state(current_iterarion-1, id=id_tracked_agent), self.get_adv_actions(current_iterarion-1, id=id_tracked_agent), track_token=self._agent_tokens[id_tracked_agent], iter=current_iterarion), next_iteration=current_iterarion, id=id_tracked_agent)

        #     '''
        #     self.set_adv_state(self._motion_model.forward_all(self.get_adv_state(), self.get_adv_actions(current_iteration-1)), next_iteration=current_iteration)
        

        if not current_iteration==0:
            temp_bm_iter = self.bm_iteration
            while temp_bm_iter <= current_iteration*self._freq_multiplier: 
                print('hi')
                self.set_adv_state(self._motion_model.forward_all(self.get_adv_state(), self.get_adv_actions(temp_bm_iter-1)))
                temp_bm_iter += 1
            print()
            
            self.bm_iteration += self._freq_multiplier

        current_state = self.get_adv_state() # Dict[str (type of state), torch.Tensor (idx of agent, ...)]
        agents_pos = current_state['pos'][0].clone().detach().cpu().numpy() # adding [0] to get rid of the batch dimension that we don't want to have in nuplan for now
        agents_vel = current_state['vel'][0].clone().detach().cpu().numpy()
        agents_yaw = current_state['yaw'][0].clone().detach().cpu().numpy()


        # # adding the state of the ego to the buffere of states
        # ego_state = self._simulation._ego_controller.get_state()
        # ego_position = np.array([[ego_state.center.x, ego_state.center.y]])
        # ego_heading = np.array([[ego_state.center.heading]])
        # ego_velocity = np.array([[ego_state.dynamic_car_state.center_velocity_2d.x, ego_state.dynamic_car_state.center_velocity_2d.y]])


        ego_position = self._ego_state_whole_np['pos']
        ego_heading = self._ego_state_whole_np['yaw']
        ego_velocity = self._ego_state_whole_np['vel']
        
        # print('agents_yaw ', agents_yaw[20])


        # print('ego_heading  ' , ego_heading)


        # # check the collision between adversar agents and the ego vehicle
        # if self.check_collision_at_iteration(agents_pos, agents_yaw, ego_position, ego_heading):
        #     # send a flag to the simulation runner to save the simulation?
        #     # what does it mean to save the simulation? to keep the state of adversary agents
        #     pass

        agents_pos_ = np.concatenate((agents_pos, ego_position), axis=0)      
        agents_yaw_ = np.concatenate((agents_yaw, ego_heading), axis=0)    
        agents_vel_ = np.concatenate((agents_vel, ego_velocity), axis=0)    

        self.state_buffer.append({'pos': agents_pos_, 'vel': agents_vel_, 'yaw': agents_yaw_})




        after_transforming_x, after_transforming_y, after_transforming_yaw = [], [], []
        for idx in range(self._number_agents):
            coord_vel_x, coord_vel_y, _ = self._convert_coord_map.pos_to_coord_vel(agents_vel[idx, 0], agents_vel[idx, 1], agents_yaw[idx, 0])
            coord_pos_x, coord_pos_y, coord_pos_yaw = self._convert_coord_map.pos_to_coord(agents_pos[idx, 0], agents_pos[idx, 1], agents_yaw[idx, 0])
            after_transforming_x.append(coord_pos_x)
            after_transforming_y.append(coord_pos_y)
            after_transforming_yaw.append(coord_pos_yaw)
            not_differentiable_state[self._agent_tokens[idx]] = ((coord_pos_x, coord_pos_y), (coord_vel_x, coord_vel_y), (coord_pos_yaw))
        

        if not self._collision_occurred and self.check_collision_simple(agents_pos, ego_position):
            collision, collision_index = self.check_collision(agents_pos, agents_yaw, ego_position, ego_heading)
            if collision:
                # should make the next states in the same place, and the same state as now
                # keep the current not_differentiable_state, and from now on just return it for the iterations after the current one, without enrolling bm
                self._collision_occurred = True
                self._collision_index = collision_index
                self._collision_iteration = current_iteration*self._freq_multiplier
                self._collision_not_differentiable_state = not_differentiable_state
                print('collision occurred **********************')
                print('collision occurred **********************')
                print('collision occurred **********************')
                print('collision occurred **********************')
                print('collision occurred **********************')
                print('collision occurred **********************')
                return True, None
  
        return False, not_differentiable_state
    
    
    def step_after_collision(self, current_iteration):

     
        if current_iteration is None:
            return
   

        not_differentiable_state = {}

        
        if not current_iteration==0:
            temp_bm_iter = self.bm_iteration
            while temp_bm_iter <= current_iteration*self._freq_multiplier:
                if not self._use_original_states:
                    self.set_adv_state(self._motion_model.forward_all(self.get_adv_state(), self.get_adv_actions(temp_bm_iter-1)))
                else:
                    self.set_adv_state_to_original(self._motion_model.forward_all(self.get_adv_state(), self.get_adv_actions(temp_bm_iter-1)), next_iteration=temp_bm_iter, exception_agent_index=self._collision_index)

                temp_bm_iter += 1

            self.bm_iteration += self._freq_multiplier
           

        if current_iteration*self._freq_multiplier==self._collision_iteration:
            with torch.no_grad():
                self._states['vel'][self._collision_index] = torch.tensor([0., 0.], dtype=torch.float64, device=device)
                self._states['speed'][self._collision_index] = torch.tensor([0.0], dtype=torch.float64, device=device)
                self._states['accel'][self._collision_index] = torch.tensor([0.0], dtype=torch.float64, device=device)
                # the steering angle should be set to the one at the last step before collision => already set by set_adv_sate
            

        current_state = self.get_adv_state() # Dict[str (type of state), torch.Tensor (idx of agent, ...)]
        agents_pos = current_state['pos'][0].clone().detach().cpu().numpy() # adding [0] to get rid of the batch dimension that we don't want to have in nuplan for now
        agents_vel = current_state['vel'][0].clone().detach().cpu().numpy()
        agents_yaw = current_state['yaw'][0].clone().detach().cpu().numpy()


        self._collided_agent_drivable_cost += self.drivable_area_metric()
        # checking for collision between adversary agents
        collided_indices = self.check_adversary_collision(agents_pos, agents_yaw)
        for collided_idx in collided_indices:
            self.stopping_agent(collided_idx, current_iteration*self._freq_multiplier)
            self._adversary_collisions += 1


        # this part is wrong!! it should be ego_state_whole
        ego_position = self._ego_state_whole['pos'][current_iteration].cpu().detach().numpy()[None, ...]
        ego_heading = self._ego_state_whole['yaw'][current_iteration].cpu().detach().numpy()[None, ...]
        ego_velocity = self._ego_state_whole['vel'][current_iteration].cpu().detach().numpy()[None, ...]
        
        agents_pos_ = np.concatenate((agents_pos, ego_position), axis=0)      
        agents_yaw_ = np.concatenate((agents_yaw, ego_heading), axis=0)    
        agents_vel_ = np.concatenate((agents_vel, ego_velocity), axis=0)    

        self.state_buffer.append({'pos': agents_pos_, 'vel': agents_vel_, 'yaw': agents_yaw_})

        after_transforming_x, after_transforming_y, after_transforming_yaw = [], [], []
        for idx in range(self._number_agents):
            coord_vel_x, coord_vel_y, _ = self._convert_coord_map.pos_to_coord_vel(agents_vel[idx, 0], agents_vel[idx, 1], agents_yaw[idx, 0])
            coord_pos_x, coord_pos_y, coord_pos_yaw = self._convert_coord_map.pos_to_coord(agents_pos[idx, 0], agents_pos[idx, 1], agents_yaw[idx, 0])
            after_transforming_x.append(coord_pos_x)
            after_transforming_y.append(coord_pos_y)
            after_transforming_yaw.append(coord_pos_yaw)
            not_differentiable_state[self._agent_tokens[idx]] = ((coord_pos_x, coord_pos_y), (coord_vel_x, coord_vel_y), (coord_pos_yaw))

        if not self._collision_occurred and self.check_collision_simple(agents_pos, ego_position):
            collision, _ = self.check_collision(agents_pos, agents_yaw, ego_position, ego_heading)
            if collision:
                self._collision_after_trajectory_changes = True

        return False, not_differentiable_state
    
    def actions_to_original(self):
        # put all the actions to their original value, except the one collided with the ego
        with torch.no_grad():
            self._throttle[:][:self._collision_index, :] = self._throttle_original[:][:self._collision_index, :]
            self._throttle[:][self._collision_index+1:, :] = self._throttle_original[:][self._collision_index+1:, :]

            self._steer[:][:self._collision_index, :] = self._steer_original[:][:self._collision_index, :]
            self._steer[:][self._collision_index+1:, :] = self._steer_original[:][self._collision_index+1:, :]

    
    def states_to_original(self):
        # put all the actions to their original value, except the one collided with the ego
        with torch.no_grad():
            self._throttle[:self._collision_index, :] = self._throttle_original[:self._collision_index, :]
            self._throttle[self._collision_index+1:, :] = self._throttle_original[self._collision_index+1:, :]

            self._steer[:self._collision_index, :] = self._steer_original[:self._collision_index, :]
            self._steer[self._collision_index+1:, :] = self._steer_original[self._collision_index+1:, :]

    def use_original_states(self):

        self._use_original_states = True

    def stopping_collider(self):

        with torch.no_grad():

            self._throttle[self._collision_index, self._collision_iteration:] = torch.zeros((self._number_actions-self._collision_iteration), 1)
            self._steer[self._collision_index, self._collision_iteration:] = torch.zeros((self._number_actions-self._collision_iteration), 1)
            # should also make the velocity and speed zero for this agent => do it in the step_after_collision

    def stopping_agent(self, idx, step):
        with torch.no_grad():
            
            self._throttle[idx, step:] = torch.zeros((self._number_actions-step), 1)
            self._steer[idx, step:] = torch.zeros((self._number_actions-step), 1)
            # should also make the velocity and speed zero for this agent => do it in the step_after_collision
            self._states['vel'][idx] = torch.tensor([0., 0.], dtype=torch.float64, device=device)
            self._states['speed'][idx] = torch.tensor([0.0], dtype=torch.float64, device=device)
            self._states['accel'][idx] = torch.tensor([0.0], dtype=torch.float64, device=device)
            


    def drivable_area_metric(self):
        # for the collided agent, computes its current drivable area cost => higher if out of drivable area
        return self.test_distance_map.collided_agent_cost(self._states['pos'][self._collision_index, :])
    
    def get_collided_agent_drivable_cost(self):

        return self._collided_agent_drivable_cost
    
    def get_number_adversary_collisions(self):

        return self._adversary_collisions
    
    def get_collision_after_traj_changes(self):

        return self._collision_after_trajectory_changes

    def update_ego_state(self, current_ego_state: EgoState, current_iteration: int):
        '''
        update the state of the ego agent using its trajectory
        :param trajectory: the ego trajectory that is taken at the current iteration
        :param next_iteration: the next iteration of simulation
        '''

        assert self._current_optimization_on_iteration==current_iteration, 'we are not optimizing on the right iteration of the simulation'




        ego_state = current_ego_state
        ego_x, ego_y, ego_heading = ego_state.center.x, ego_state.center.y, ego_state.center.heading
        ego_vel_x, ego_vel_y = ego_state.agent.velocity.x, ego_state.agent.velocity.y
        ego_x_map, ego_y_map, ego_heading_map = self._convert_coord_map.pos_to_map(ego_x, ego_y, ego_heading)
        ego_vel_x_map, ego_vel_y_map, _ = self._convert_coord_map.pos_to_map_vel(ego_vel_x, ego_vel_y, ego_heading)

        self._ego_state['pos'] = torch.tensor([ego_x, ego_y], device=device, dtype=torch.float64)
        self._ego_state['vel'] = torch.tensor([ego_vel_x, ego_vel_y], device=device, dtype=torch.float64)
        self._ego_state['yaw'] = torch.tensor([ego_heading], device=device, dtype=torch.float64)



        self._ego_state_whole['pos'][current_iteration, :] = torch.tensor([ego_x_map, ego_y_map], device=device, dtype=torch.float64)
        self._ego_state_whole['vel'][current_iteration, :] = torch.tensor([ego_vel_x_map, ego_vel_y_map], device=device, dtype=torch.float64)
        self._ego_state_whole['yaw'][current_iteration, :] = torch.tensor([ego_heading_map], device=device, dtype=torch.float64)

        
        self._ego_state_whole_np['pos'][0,:] = np.array([ego_x_map, ego_y_map], dtype=np.float64)
        self._ego_state_whole_np['vel'][0,:] = np.array([ego_vel_x_map, ego_vel_y_map], dtype=np.float64)
        self._ego_state_whole_np['yaw'][0,:] = np.array([ego_heading_map], dtype=np.float64)


        self._current_optimization_on_iteration += 1

    def init_ego_state(self, current_ego_state: EgoState):
      

        ego_state = current_ego_state
        ego_x, ego_y, ego_heading = ego_state.center.x, ego_state.center.y, ego_state.center.heading
        ego_vel_x, ego_vel_y = ego_state.agent.velocity.x, ego_state.agent.velocity.y
        ego_x_map, ego_y_map, ego_heading_map =  self.from_matrix(self._map_transform@self.coord_to_matrix(np.array([ego_x, ego_y]), np.array([ego_heading])))
        ego_vel_x_map, ego_vel_y_map, _ = self.from_matrix(self._map_transform@self.coord_to_matrix_vel(np.array([ego_vel_x, ego_vel_y]), np.array([ego_heading])))


        self.transformed_crop_x = ego_x_map
        self.transformed_crop_y = ego_y_map

        self._ego_state['pos'] = torch.tensor([ego_x, ego_y], device=device, dtype=torch.float64)
        self._ego_state['vel'] = torch.tensor([ego_vel_x, ego_vel_y], device=device, dtype=torch.float64)
        self._ego_state['yaw'] = torch.tensor([ego_heading], device=device, dtype=torch.float64)



        self._ego_state_whole['pos'][0, :] = torch.tensor([ego_x_map, ego_y_map], device=device, dtype=torch.float64)
        self._ego_state_whole['vel'][0, :] = torch.tensor([ego_vel_x_map, ego_vel_y_map], device=device, dtype=torch.float64)
        self._ego_state_whole['yaw'][0, :] = torch.tensor([ego_heading_map], device=device, dtype=torch.float64)

        
        
        self._ego_state_whole_np['pos'][0,:] = np.array([ego_x_map, ego_y_map], dtype=np.float64)
        self._ego_state_whole_np['vel'][0,:] = np.array([ego_vel_x_map, ego_vel_y_map], dtype=np.float64)
        self._ego_state_whole_np['yaw'][0,:] = np.array([ego_heading_map], dtype=np.float64)

        self.init_map()


    def init_map(self):
        self._data_nondrivable_map = self._data_nondrivable_map[int(self.transformed_crop_y-1000):int(self.transformed_crop_y+1000), int(self.transformed_crop_x-1000):int(self.transformed_crop_x+1000)]
        self.test_distance_map = DrivableAreaCost(self._data_nondrivable_map)
        self._convert_coord_map = TransformCoordMap(self._map_resolution, self._map_transform, self.transformed_crop_x, self.transformed_crop_y)
        self._data_nondrivable_map = torch.tensor(self._data_nondrivable_map, device=device, dtype=torch.float64, requires_grad=True)
        self._transpose_data_nondrivable_map = torch.transpose((1-self._data_nondrivable_map), 1, 0)


    
    @staticmethod
    def coord_to_matrix(coord, yaw):

        return np.array(
            [
                [np.cos(yaw[0]), -np.sin(yaw[0]), 0.0, coord[0]],
                [np.sin(yaw[0]), np.cos(yaw[0]), 0.0, coord[1]],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0]
            ]
        ).astype(np.float64)
    
    @staticmethod
    def coord_to_matrix_vel(coord, yaw):

        return np.array(
            [
                [np.cos(yaw[0]), -np.sin(yaw[0]), 0.0, coord[0]],
                [np.sin(yaw[0]), np.cos(yaw[0]), 0.0, coord[1]],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 0.0]
            ]
        ).astype(np.float64)



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
        # we use ego_state_whole to be able to compute the cost even when not calling the planner but just optimizing
        for substate in self._ego_state.keys():
            input_cost_ego_state[substate] = torch.unsqueeze(torch.unsqueeze(self._ego_state_whole[substate][current_iteration,:], dim=0), dim=0)

        input_cost_adv_state = self.get_adv_state()

        # write a dummy cost that just makes all the vehicles close to the ego


        if 'collision' in self._costs_to_use:
            ego_col_cost, adv_col_cost, agents_costs = self.col_cost_fn(
                input_cost_ego_state,
                self.ego_extent,
                input_cost_adv_state,
                self.adv_extent,
            )
            if adv_col_cost.size(-1) == 0:
                adv_col_cost = torch.zeros(1,1).to(device)
                assert adv_col_cost.size(0) == 1, 'This works only for batchsize 1!'

            adv_col_cost = torch.minimum(
                adv_col_cost, torch.tensor([1.25]).float().to(device)
            )

            

            self._cost_dict['ego_col'].append(ego_col_cost)
            self._cost_dict['adv_col'].append(adv_col_cost)
            self._accumulated_collision_loss_agents += agents_costs.detach()
            current_cost =  torch.sum(ego_col_cost).detach()
            self._accumulated_collision_loss += current_cost
            self._dummy_losses_per_step.append(current_cost)
                


        if 'drivable_us' in self._costs_to_use:

            # self.routedeviation_heatmap()
            # self.routedeviation_heatmap_agents(input_cost_adv_state['pos'], input_cost_adv_state['yaw'])
            # self.routedeviation_agents(input_cost_adv_state['pos'], input_cost_adv_state['yaw'])
            
            input_cost_adv_state['pos'].retain_grad()
            drivable_computation_time = perf_counter()
            gradients, current_cost_agents = self.test_distance_map(input_cost_adv_state['pos'])
            # (input_cost_adv_state['pos'][0]).backward(retain_graph=True, gradient=gradients*self.weight_drivable)

            if self._first_forward:
                self._drivable_backward_pos = torch.unsqueeze(input_cost_adv_state['pos'][0], dim=0)
                self._drivable_backward_grad = torch.unsqueeze(gradients*self.weight_drivable, dim=0)
                self._first_forward = False
            else:
                
                self._drivable_backward_pos = torch.cat((torch.unsqueeze(input_cost_adv_state['pos'][0], dim=0), self._drivable_backward_pos), dim=0)
                self._drivable_backward_grad = torch.cat((torch.unsqueeze(gradients*self.weight_drivable, dim=0), self._drivable_backward_grad), dim=0)

                # print('checking the size of the self._drivable_backward_pos ', (self._drivable_backward_pos).size())
                # print('checking the size of the self._drivable_backward_grad ', (self._drivable_backward_grad).size())


            # accumulate all the input_cost_adv_state['pos'][0] and gradients*self.weight_drivable to do the backpropagation later
            drivable_computation_time = perf_counter() - drivable_computation_time
            self._accumulated_drivable_loss_agents += current_cost_agents.detach()
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
            self._accumulated_collision_loss_agents += agents_costs.detach()
            self._cost_dict['dummy'].append(dummy_cost*self.weight_collision)
            current_cost =  torch.sum(dummy_cost).detach()
            self._accumulated_collision_loss += current_cost
            self._dummy_losses_per_step.append(current_cost)




        if 'moving_dummy' in self._costs_to_use:
            dummy_cost, agents_costs = self.dummy_cost_fn(
                input_cost_ego_state,
                self.ego_extent,
                input_cost_adv_state,
                self.adv_extent,
            )
            self._cost_dict['dummy'].append(dummy_cost*self.weight_collision)
            self._accumulated_collision_loss_agents += agents_costs.detach()
            current_cost =  torch.sum(dummy_cost).detach()
            self._accumulated_collision_loss += current_cost
            self._dummy_losses_per_step.append(current_cost)


        return drivable_computation_time

        

    def back_prop(self):
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


            total_objective = total_objective + cost_dict["ego_col"]
            # FOR NOW IGNORING THE ADVERSARY COST
        
        
        if 'fixed_dummy' in self._costs_to_use or 'moving_dummy' in self._costs_to_use:

            cost_dict["dummy"] = torch.sum(
                torch.stack(cost_dict["dummy"], dim=-1),
            )

            total_objective = total_objective + cost_dict["dummy"]
        
        
        if 'drivable_us' in self._costs_to_use:

            (self._drivable_backward_pos).backward(retain_graph=True, gradient=self._drivable_backward_grad)
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


        if not 'nothing' in self._costs_to_use:
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




    def plot_losses(self):

        create_directory_if_not_exists('/home/kpegah/workspace/EXPERIMENTS')
        create_directory_if_not_exists(f'/home/kpegah/workspace/EXPERIMENTS/{self._simulation.scenario.scenario_name}')
        create_directory_if_not_exists(f'/home/kpegah/workspace/EXPERIMENTS/{self._simulation.scenario.scenario_name}/{self._experiment_name}')

        if len(self._drivable_area_losses_per_opt)==0:
            return

        if 'drivable_king' in self._costs_to_use or 'drivable_us' in self._costs_to_use:


            plt.plot(torch.stack(self._drivable_area_losses_per_opt).cpu().numpy())
            plt.gcf()
            plt.savefig(f'/home/kpegah/workspace/EXPERIMENTS/{self._simulation.scenario.scenario_name}/{self._experiment_name}/drivable_accumulated_loss.png')
            wandb.log({"drivable_accumulated_loss": wandb.Image(plt)})
            plt.show()
            plt.clf()


            print('this is the _drivable_area_losses_per_step ', self._drivable_area_losses_per_step)
            plt.plot(torch.stack(self._drivable_area_losses_per_step).cpu().numpy(), label='cost')
            # plt.plot(self.throttles[3], label='throttle val')
            plt.legend() 
            plt.gcf()
            plt.savefig(f'/home/kpegah/workspace/EXPERIMENTS/{self._simulation.scenario.scenario_name}/{self._experiment_name}/drivable_cost_sim_steps.png')
            wandb.log({"drivable_cost_sim_steps": wandb.Image(plt)})
            plt.show()
            plt.clf()


        if 'fixed_dummy' in self._costs_to_use or 'moving_dummy' in self._costs_to_use or 'collision' in self._costs_to_use:

            
            plt.plot(torch.stack(self._dummy_losses_per_opt).cpu().numpy())
            plt.gcf()
            plt.savefig(f'/home/kpegah/workspace/EXPERIMENTS/{self._simulation.scenario.scenario_name}/{self._experiment_name}/dummy_accumulated_loss.png')
            wandb.log({"dummy_accumulated_loss": wandb.Image(plt)})
            plt.show()
            plt.clf()

  
            plt.plot(torch.stack(self._dummy_losses_per_step).cpu().numpy(), label='cost')
            plt.legend() 
            plt.gcf()
            plt.savefig(f'/home/kpegah/workspace/EXPERIMENTS/{self._simulation.scenario.scenario_name}/{self._experiment_name}/fixeddummy_cost_sim_steps.png')
            wandb.log({"fixeddummy_cost_sim_steps": wandb.Image(plt)})
            plt.show()
            plt.clf()

        
        # just to visualize one of the scnes where the colision happens at the very beginning
        # self.plot_loss_per_agent(self.drivable_loss_agents_per_opt, self.dummy_loss_agents_per_opt)
    

    
    def plot_loss_per_agent(self, drivable_loss_per_opt, collision_loss_per_opt):

        '''
        the function to visualize the gradients registered by the hooks for the throttle and steer
        :params optimization_iteration
        '''

        drivable_loss_per_opt = [item.cpu().numpy() for item in drivable_loss_per_opt]
        collision_loss_per_opt = [item.cpu().numpy() for item in collision_loss_per_opt]

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
        wandb.log({"drivable_loss_evolution_agents_per_opt": wandb.Image(plt)})
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



    def get_corners(self, pos, yaw):
        """
        Obtain agent corners given the position and yaw.
        """
        n_agnets = yaw.shape[0]
        # yaw =  np.pi/2 - yaw
        yaw =  yaw - np.pi/2

        rot_mat = np.concatenate(
            [
                np.cos(yaw), -np.sin(yaw),
                np.sin(yaw), np.cos(yaw),
            ],
            axis=-1,
        ).reshape(n_agnets, 1, 2, 2)
        rot_mat = np.broadcast_to(rot_mat, (n_agnets, 4, 2, 2)) # n_agents, 4, 2, 2

        rotated_corners = rot_mat @ np.expand_dims(self._original_corners, axis=-1)

        rotated_corners = rotated_corners.reshape(n_agnets, 4, 2) + np.expand_dims(pos, axis=1)

        return rotated_corners.reshape(1, -1, 2)

    def check_collision(self, agents_pos, agents_yaw, ego_pos, ego_yaw):
        # we have the orientation of the vahicle, its extent, and its orientation, all in numpy
        adv_corners = self.get_corners(agents_pos, agents_yaw) # corners of all agents in the scene of size (1, num_agents*4, 2)
        adv_boxes = adv_corners.reshape(self._number_agents, 4, 2)
        ego_corners = self.get_corners(ego_pos, ego_yaw)
        ego_box = ego_corners.reshape(4, 2)
        # now we should check the collision using the corners
        return self.check_collision_boxes(adv_boxes, ego_box)
    
    def check_adversary_collision(self, agents_pos, agents_yaw):
        adv_corners = self.get_corners(agents_pos, agents_yaw) # corners of all agents in the scene of size (1, num_agents*4, 2)
        adv_boxes = adv_corners.reshape(self._number_agents, 4, 2)
        collided_indices = []
        for idx1, reference_box in enumerate(adv_boxes[:-1,...]):
            to_check_boxes = adv_boxes[idx1+1:, ...]
            for idx2, check_box in enumerate(to_check_boxes):
                check_box = check_box[None, ...]
                colllision, _ = self.check_collision_boxes(check_box, reference_box)
                if colllision:
                    collided_indices.append(idx1)
                    collided_indices.append(idx2)


        return list(set(collided_indices))
    

    def check_collision_simple(self, agents_pos, ego_pos):
        print(ego_pos.shape)
        print(agents_pos.shape)
        distances = np.linalg.norm(agents_pos - ego_pos, axis=1)
        print(np.min(distances))
        return np.any(distances < 50)


    def check_collision_boxes(self, boxes, reference_box):
        # Assuming vehicle_coordinates is a list of (x, y) coordinates for n vehicles
        # and new_vehicle_coordinates is a tuple of (x, y) coordinates for the (n+1)th vehicle

        for idx, box in enumerate(boxes): # box should be of shape (4,2)
            if self.overlap_complex(box, reference_box):
                return True, idx  # Collision detected
        return False, -1  # No collision detected

    def overlap(self, box1, box2):
        # Check if two bounding boxes overlap
        print('box1 ', box1)
        print('box2 ', box2)
        return not (box1[:, 0].min() > box2[:, 0].max() or
                    box1[:, 0].max() < box2[:, 0].min() or
                    box1[:, 1].min() > box2[:, 1].max() or
                    box1[:, 1].max() < box2[:, 1].min())

    def overlap_complex(self, box1, box2):
        def project(axis, vertices):
            # Project the vertices onto the axis
            return np.dot(vertices, axis)

        def axis(vertices):
            # Get the axes formed by the edges of the rectangle
            return np.array([vertices[1] - vertices[0], vertices[2] - vertices[1]])

        # Project rectangles onto potential separating axes
        axes1 = axis(box1)
        axes2 = axis(box2)

        # # Check for corner-to-edge contact
        # for vertex in box1:
        #     # Project the vertex onto potential separating axes
        #     for ax in axes2:
        #         projection_vertex = np.dot(vertex, ax)
        #         projection2 = project(ax, box2)
        #         if projection_vertex <= np.max(projection2) and projection_vertex >= np.min(projection2):
        #             return True  # Corner-to-edge contact

        # for vertex in box2:
        #     # Project the vertex onto potential separating axes
        #     for ax in axes1:
        #         projection_vertex = np.dot(vertex, ax)
        #         projection1 = project(ax, box1)
        #         if projection_vertex <= np.max(projection1) and projection_vertex >= np.min(projection1):
        #             return True  # Corner-to-edge contact

        axes = np.vstack([axes1, axes2])
        for ax in axes:
            projection1 = project(ax, box1)
            projection2 = project(ax, box2)

            # Check for overlap on this axis
            if np.max(projection1) < np.min(projection2) or np.min(projection1) > np.max(projection2):
                return False


        # print('these are the boxes \n', box1, '\n', box2)
        return True

   

    def save_state_buffer(self):

        self.whole_state_buffers.append(self.state_buffer)
      

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

        
        input_cost_adv_state = self.get_adv_state()
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

          
        input_cost_adv_state = self.get_adv_state()
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


    


    def optimize_without_simulation(self, optimization_rounds: int):

        for _ in range (optimization_rounds):
            self.reset()

            # do steps of simulation  by calling the 
            for iter in range(self._number_states-1):
                
                self.step_cost_without_simulation(iter)
            
            self.back_prop()


    def step_cost_without_simulation(self, current_iteration: int):

        if not current_iteration==0:
            temp_bm_iter = self.bm_iteration
            while temp_bm_iter <= current_iteration*self._freq_multiplier: 
                self.set_adv_state(self._motion_model.forward_all(self.get_adv_state(), self.get_adv_actions(temp_bm_iter-1)))
                temp_bm_iter += 1
            
            self.bm_iteration += self._freq_multiplier

        self.compute_current_cost(current_iteration)

        return
        


