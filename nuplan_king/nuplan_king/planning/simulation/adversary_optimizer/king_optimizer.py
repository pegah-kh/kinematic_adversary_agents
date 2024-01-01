"""
parts of the code are no longer used, and are indicated by CAN_BE_DELETED.
parts of the code are not optimized enough, indicated by TO_BE_OPTIMIZED.
they are still kept for the sake keeping track of the changes made.
"""

from __future__ import annotations

import logging
from typing import List, Dict, Optional
from omegaconf import DictConfig
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.autograd.profiler as profiler
import numpy as np
from hydra.utils import instantiate
import matplotlib.pyplot as plt
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
import random
import csv




from nuplan_gpu_work.planning.simulation.adversary_optimizer.abstract_optimizer import AbstractOptimizer
from nuplan_gpu_work.planning.simulation.adversary_optimizer.agent_tracker.agent_lqr_tracker import LQRTracker 
from nuplan_gpu_work.planning.simulation.motion_model.bicycle_model import BicycleModel
from nuplan_gpu_work.planning.simulation.cost.king_costs import RouteDeviationCostRasterized, BatchedPolygonCollisionCost, DummyCost, DummyCost_FixedPoint, DrivableAreaCost
from nuplan_gpu_work.planning.simulation.adversary_optimizer.trajectory_reconstructor import TrajectoryReconstructor
from nuplan_gpu_work.planning.simulation.adversary_optimizer.debug_visualizations import before_after_transform, agents_position_before_transformation, visualize_whole_map, routedeviation_efficient_heatmap, routedeviation_king_heatmap, routedeviation_efficient_agents_map, routedeviation_king_agents_map
from nuplan_gpu_work.planning.simulation.adversary_optimizer.visualizations_plots import visualize_grads_steer, visualize_grads_throttle, visualize_steer, visualize_throttle, plot_losses, plot_loss_per_agent

from nuplan.common.maps.maps_datatypes import SemanticMapLayer
from nuplan.common.actor_state.tracked_objects import TrackedObject
from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType
from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.actor_state.waypoint import Waypoint
from nuplan.common.actor_state.oriented_box import OrientedBox
from nuplan.common.actor_state.state_representation import StateSE2, StateVector2D, TimePoint
from nuplan_gpu_work.planning.scenario_builder.nuplan_db_modif.nuplan_scenario import NuPlanScenarioModif
from nuplan.planning.simulation.planner.abstract_planner import AbstractPlanner
from nuplan.planning.simulation.runner.runner_report import RunnerReport
from nuplan.planning.simulation.simulation import Simulation
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from nuplan.planning.simulation.trajectory.interpolated_trajectory import InterpolatedTrajectory
from nuplan.planning.simulation.observation.observation_type import DetectionsTracks


logger = logging.getLogger(__name__)
device = 'cuda' if torch.cuda.is_available() else 'cpu'


class TransformCoordMap():
    """
        Helper class to transform the coordiantes from and to the world and map coordinate systems.
    """
    def __init__(self, resoltuion, transform, x_crop, y_crop):
        """
        :param resoltuion : the resolution of the map with respect coordinates compared to world coordinates
                            0.1 value for resolution means each unit would represent 0.1 meters in map coordinates
                            this value is given in the logs
        :param transform : the transformation matrix (4x4) to convert world coordinates to map coordinates
        :param x_crop : the x position of the ego in the MAP COORDINATES to crop around
        :param y_crop : same as x_crop for y axis
        """
        self._transform = transform.astype(np.float64)
        self._resolution = resoltuion
        self._x_crop = x_crop
        self._y_crop = y_crop

        self._inv_transform = np.linalg.inv(self._transform)


        
    @staticmethod
    def coord_to_matrix(coord, yaw):
        """
        Static method to construct a 4x4 transformation matrix using yaw angle and coordinates.
        """
        return np.array(
            [
                np.cos(yaw[0]), -np.sin(yaw[0]), 0.0, coord[0],
                np.sin(yaw[0]), np.cos(yaw[0]), 0.0, coord[1],
                0.0, 0.0, 1.0, 0.0,
                0.0, 0.0, 0.0, 1.0,
            ]
        ).reshape(4,4).astype(np.float64)
    
    @staticmethod
    def coord_to_matrix_vel(coord, yaw):
        """
        Static method to convert coordinates and yaw angle to a 4x4 transformation matrix.

            this function is different from the above 'coord_to_matrix' function:
            the translation component of the 'transform' will be ignored when multiplied by this matrix (as the last row consists of only zeros).
        """

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
        """
        Static method to extract coordinates and yaw angle from a 4x4 matrix.
        """

        assert matrix.shape == (4, 4), f"Expected 3x3 transformation matrix, but input matrix has shape {matrix.shape}"

        return matrix[0, 3], matrix[1, 3], np.arctan2(matrix[1, 0], matrix[0, 0])

    def pos_to_map(self, coord_x, coord_y, coord_yaw):
        """
        Transform world coordinates to map coordinates.

        :param coord_x, coord_y: Coordinates of the position in the world coordinate system.
        :param coord_yaw: yaw angle in the world coordinate system.

        Returns:
        - Transformed map coordinates adjusted for cropping and resolution.
          1. x and y axes are flipped
          2. coordinates are cropped around the ego (x_crop and y_crop), within a square of 200 meters (100/resolution).
        """

        transformed_adv_x, transformed_adv_y, transformed_yaw = self.from_matrix(self._transform@self.coord_to_matrix([coord_x, coord_y], [coord_yaw]))

        return transformed_adv_y - self._y_crop + 100/self._resolution, transformed_adv_x - self._x_crop + 100/self._resolution, np.pi/2-transformed_yaw
    
    def pos_to_map_vel(self, coord_x, coord_y, coord_yaw):

        """
        Similar to the above function 'pos_to_map', only without translation.
        """
        
        transformed_adv_x, transformed_adv_y, transformed_yaw = self.from_matrix(self._transform@self.coord_to_matrix_vel([coord_x, coord_y], [coord_yaw]))

        return transformed_adv_y, transformed_adv_x, np.pi/2-transformed_yaw
    

    def pos_to_coord(self, map_x, map_y, map_yaw):
        """
        Inverse of the 'pos_to_map' function
        Transform map coordinates to position coordinates, using the inverse of transformation matrix.

        :param map_x, map_y: Coordinates in the map coordinate system.
        :param map_yaw: Yaw angle in the map coordinate system.

        Returns:
        - Transformed position coordinates.
        """

        transformed_adv_x, transformed_adv_y, transformed_yaw = self.from_matrix(self._inv_transform@self.coord_to_matrix([map_y+ self._x_crop - 100/self._resolution, map_x+ self._y_crop - 100/self._resolution], [np.pi/2-map_yaw]))

        return transformed_adv_x, transformed_adv_y, transformed_yaw


    def pos_to_coord_vel(self, map_x, map_y, map_yaw):
        """
        Inverse of the 'pos_to_map_vel' function
        """

        transformed_adv_x, transformed_adv_y, transformed_yaw = self.from_matrix(self._inv_transform@self.coord_to_matrix_vel([map_y+ self._x_crop - 100/self._resolution, map_x+ self._y_crop - 100/self._resolution], [np.pi/2-map_yaw]))

        return transformed_adv_x, transformed_adv_y, transformed_yaw



class OptimizationKING(AbstractOptimizer):
    """
    This class contains all the functions related to optimizing the actions using the adversary losses.
    Have a look at the abstract class to find an overall view over the functions here.
    """

    def __init__(self, simulation: Simulation, 
                 tracker: DictConfig,
                 tracker1: DictConfig,
                 tracker2: DictConfig, 
                 tracker3: DictConfig, 
                 tracker4: DictConfig, 
                 tracker5: DictConfig, 
                 tracker6: DictConfig, 
                 tracker7: DictConfig, 
                 tracker8: DictConfig, 
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
                dense_opt_rounds:int = 0):
        """
        initializes the components used in optimizing the actions:

        :param simulation : an instant of the simulation class,
                            we mainly access : 
                            1. the information on sampling frequency of the scenario,
                            2. the scenario name,
                            3. the 'observation' class associated to this simulation

        :param tracker : the controller used as input to TrajectoryReconstructor, in our case LQR
        :param tracker_i, i from 1 to 8 : different controllers (with different Q and R parameters), used in previous versions of the code (CAN_BE_DELETED)
        :param motion_model : the kinematics model, that is differentiable, in our case bicycle model (bm)
        :param opt_iterations : number of optimization iterations in adversary optimization
        :param max_opt_iterations : max for opt_iterations
        :param lrs : [lr_throttle, lr_steer] when optimizing the actions adversarily
        :param loss_weights : [weight_collision, weight_drivable]
                              - weight_collision is the weight given to the collision loss in the total loss
                              - weight_drivable is the weight given to the deviation loss in the total loss
        :param costs_to_use : Options for different losses
                              - 'nothing' : no optimization
                              - ['collision', 'moving_dummy', 'fixed_dummy'] : 
                                    - collision is the loss in king, attraction between ego and the closest adversary
                                    - moving_dummy is the loss that attracts all the adversaries to the moving ego
                                    - fiexd_dummy is the loss that attracts all the adversaries to the starting position of the ego
                              - ['drivable_efficient', 'drivable_king']
                                    - drivable_efficient is our modified version of the deviation loss
                                    - drivable_king is the king's version of the deviation loss
        :param requires_grad_params : choose the actions to optimize : 'steer', 'throttle', or both
        :param experiment_name : the nameof the current experiment, that can be chosen with respect to the current set of params
        :param project_name : the name of the project, that will be used in wandb
        :param opt_jump : jumps in the optimization iterations:
                          - if n, after each optimization iteration with calling the ego planner (and storing its planned trajectory), we optimize for n rounds, while using the stored trajectory of the ego
        :param collision_strat : decides on the trajectory of non-colliding vehicles, after there is a collision with the ego vehicle
                                 options between ['back_to_after_bm', 'back_to_before_bm', 'stopping_collider']
                                 - back_to_after_bm : converts the trajectory of adversary agents to their version after extracting their actions and enrolling the bm
                                 - back_to_before_bm : using their trajectories from the logs directly
                                 - stopping_collider : only stopping the collider adversary, without modifying the optimized trajectory of other agents
        :param dense_opt_rounds CAN_BE_DELETED
        
        """


        self._experiment_name = experiment_name
        self._project_name = project_name
        self._simulation = simulation
        self._collision_strat = collision_strat
        self._use_original_states = False
       

        # densely estimating the dynamic parameters variables
        self._freq_multiplier = 1
        self._dense_opt_rounds = dense_opt_rounds # CAN_BE_DELETED


        
        # the maximum number of optimization iterations
        self._opt_iterations = opt_iterations
        self._max_opt_iterations = max_opt_iterations
        self._optimization_jump = opt_jump


        # to check the dimensions later
        self._number_states:int = self._simulation._time_controller.number_of_iterations()
        self._number_actions:int = (self._number_states - 1) * self._freq_multiplier
        # use this trajectory sampling to get the initial observations, and obtain the actions accordingly
        self._observation_trajectory_sampling = TrajectorySampling(num_poses=self._number_actions, time_horizon=self._simulation.scenario.duration_s.time_s)
        self._horizon = self._number_actions
        self._number_agents: int = None



        self._tracker = LQRTracker(**tracker, discretization_time=self._observation_trajectory_sampling.interval_length)

        self._tracker1 = LQRTracker(**tracker1, discretization_time=self._observation_trajectory_sampling.interval_length)
        self._tracker2 = LQRTracker(**tracker2, discretization_time=self._observation_trajectory_sampling.interval_length)
        self._tracker3 = LQRTracker(**tracker3, discretization_time=self._observation_trajectory_sampling.interval_length)
        self._tracker4 = LQRTracker(**tracker4, discretization_time=self._observation_trajectory_sampling.interval_length)
        self._tracker5 = LQRTracker(**tracker5, discretization_time=self._observation_trajectory_sampling.interval_length)
        self._tracker6 = LQRTracker(**tracker6, discretization_time=self._observation_trajectory_sampling.interval_length)
        self._tracker7 = LQRTracker(**tracker7, discretization_time=self._observation_trajectory_sampling.interval_length)
        self._tracker8 = LQRTracker(**tracker8, discretization_time=self._observation_trajectory_sampling.interval_length)
        self._trackers = [self._tracker1, self._tracker2, self._tracker3, self._tracker4, self._tracker5, self._tracker6, self._tracker7, self._tracker8]
        print('interval ', self._observation_trajectory_sampling.interval_length)
         
        # motion model
        self._motion_model = BicycleModel(delta_t=self._observation_trajectory_sampling.interval_length)
        # self._motion_model = motion_model

        assert self._observation_trajectory_sampling.interval_length==self._tracker1._discretization_time, 'tracker discretization time of the tracker is not equal to interval_length of the sampled trajectory'
        

        tracked_objects: DetectionsTracks = self._simulation._observations.get_observation_at_iteration(0, self._observation_trajectory_sampling)
        agents: List[TrackedObject] = tracked_objects.tracked_objects.get_tracked_objects_of_type(TrackedObjectType.VEHICLE)
        self._agents = agents
        self._interval_timepoint = TimePoint(int(self._observation_trajectory_sampling.interval_length*1e6))
        num_agents = len(agents)
        self._number_agents = num_agents

        # the current state of the ego in torch
        self._ego_state = {'pos': torch.zeros(2).to(device=device), 'yaw': torch.zeros(1).to(device=device), 'vel': torch.zeros(2).to(device=device)}
        # storing the state of the ego during the whole simulation in torch
        self._ego_state_whole = {'pos': torch.zeros(self._number_states, 2).to(device=device), 'yaw': torch.zeros(self._number_states, 1).to(device=device), 'vel': torch.zeros(self._number_states, 2).to(device=device)}
        # the current state of ego in numpy
        self._ego_state_np = {'pos': np.zeros((1,2), dtype=np.float64), 'yaw': np.zeros((1,1), dtype=np.float64), 'vel': np.zeros((1,2), dtype=np.float64)}

        self._throttle_requires_grad = False
        self._steer_requires_grad = False
        if 'throttle' in requires_grad_params:
            self._throttle_requires_grad = True
        if 'steer' in requires_grad_params:
            self._steer_requires_grad = True

        self._costs_to_use = costs_to_use
        self.lr_throttle = lrs[0]
        self.lr_steer = lrs[1]
        self.weight_collision = loss_weights[0]
        self.weight_drivable = loss_weights[1]


        # the cost functions
        self.col_cost_fn = BatchedPolygonCollisionCost(num_agents=self._number_agents)
        self.adv_rd_cost = RouteDeviationCostRasterized(num_agents=self._number_agents)
        if 'fixed_dummy' in costs_to_use:
            self.dummy_cost_fn = DummyCost_FixedPoint(num_agents=self._number_agents)
        elif 'moving_dummy' in costs_to_use:
            self.dummy_cost_fn = DummyCost(num_agents=self._number_agents)


        drivable_map_layer = self._simulation.scenario.map_api.get_raster_map_layer(SemanticMapLayer.DRIVABLE_AREA)
        self._map_resolution = drivable_map_layer.precision
        self._map_transform = drivable_map_layer.transform
        self._data_nondrivable_map = np.logical_not(drivable_map_layer.data)

        # to accumulate the cost functions, over the new simulation (or optimization without simulation)
        # CAN_BE_DELETED since also in 'reset' function
        self._cost_dict = {"ego_col": [], "adv_rd": [], "adv_col": [], "dummy": []}

        # to keep track of the states in one simulation
        # CAN_BE_DELETED since also in 'reset' function
        self.state_buffer = []
        # to keep track of the evolution of states from one optimization round to the other
        self.whole_state_buffers = []

        # whether we are calling the planner during the round of optimization or it's fixed
        self._planner_on = True

        self._original_corners = np.array([[1.15, 2.88], [1.15, -2.88], [-1.15, -2.88], [-1.15, 2.88]])/self._map_resolution
        self._collision_occurred = False
        self._collision_index = -1
        self._collision_iteration = 0
        self._collision_not_differentiable_state = None # the state of different agents at the time step when collision has happened
        self._collided_agent_drivable_cost = 0
        self._adversary_collisions = 0
        # if the collision still happens after updating trajectories according to 'collision_strat' after collision 
        self._collision_after_trajectory_changes = False

        
    def reset(self) -> None:
        """
        Reseting variables at the beginning of each simulation (or optimization without simulation)

            - current_optimization_on_iteration : 
        """
        self.update_ego_state_on_iteration = 1 # to count the iteration of the simulation from the 'update_ego_state' function
        self.bm_iteration = 1 # to count the iteration of the simulation from the 'step' function

        # variables that will be used to report the evolution of OVERALL and PER AGENT losses:

        # the loss accumulated at each optimization iteration, and will be accumulated to demonstrate the decrease across iterations
        self._accumulated_drivable_loss = 0
        self._accumulated_collision_loss = 0

        # the loss for each agent across optimization iterations (like accumulated_*_loss, but individually)
        self._accumulated_drivable_loss_agents = torch.zeros(self._number_agents).to(device=device)
        self._accumulated_collision_loss_agents = torch.zeros(self._number_agents).to(device=device)
        
        # this per_step loss will be used to show the evolution of the loss across steps of the last optimization iteration
        self.routedeviation_losses_per_step = []
        self.collision_losses_per_step = []

        self._simulation.reset()
        self._simulation._ego_controller.reset()
        self._simulation._observations.reset()

        # this is not necessary, and CAN_BE_DELETED
        self._optimizer_throttle.zero_grad()
        self._optimizer_steer.zero_grad()


        self.reset_dynamic_states()

        # to accumulate the cost functions, over the new simulation (or optimization without simulation)
        self._cost_dict = {"ego_col": [], "adv_rd": [], "adv_col": [], "dummy": []}
        self.state_buffer = []

        self.throttle_gradients = [torch.zeros(self._number_actions).to(device=device) for _ in range(self._number_agents)]
        self.steer_gradients = [torch.zeros(self._number_actions).to(device=device) for _ in range(self._number_agents)]
        
        self.throttles = [torch.zeros(self._number_actions).to(device=device) for _ in range(self._number_agents)]
        self.steers = [torch.zeros(self._number_actions).to(device=device) for _ in range(self._number_agents)]
        

        self._first_forward = True


    # CAN_BE_DELETED
    """
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
    """
    

    def scenario(self) -> NuPlanScenarioModif:
        """
        :return: Get the scenario.
        """
        return self._simulation.scenario


    def init_dynamic_states(self):

        self._states = {'pos': torch.zeros(self._number_agents, 2, requires_grad=True).to(device=device), 
                        'yaw': torch.zeros(self._number_agents, 1, requires_grad=True).to(device=device), 
                        'steering_angle': torch.zeros(self._number_agents, 1, requires_grad=True).to(device=device), 
                        'vel': torch.zeros(self._number_agents, 2, requires_grad=True).to(device=device), 
                        'accel': torch.zeros(self._number_agents, 1, requires_grad=True).to(device=device), 
                        'speed': torch.zeros(self._number_agents, 1, requires_grad=True).to(device=device)}
        

        self._states_original = {'pos': torch.zeros(self._number_agents, self._number_actions+1, 2, requires_grad=True).to(device=device), 
                                 'yaw': torch.zeros(self._number_agents,self._number_actions+1, 1, requires_grad=True).to(device=device),
                                 'vel': torch.zeros(self._number_agents, self._number_actions+1, 2, requires_grad=True).to(device=device), 
                                 'steering_angle': torch.zeros(self._number_agents, self._number_actions+1, 1, requires_grad=True).to(device=device), 
                                 'accel': torch.zeros(self._number_agents, self._number_actions+1, 1, requires_grad=True).to(device=device), 
                                 'speed': torch.zeros(self._number_agents, self._number_actions+1, 1, requires_grad=True).to(device=device)}
        
      
        self._initial_map_x, self._initial_map_y, self._initial_map_yaw = [], [], []
        self._initial_steering_rate = []
        self._initial_map_vel_x, self._initial_map_vel_y = [], []

        before_transforming_x, before_transforming_y, before_transforming_yaw = [], [], []
        after_transforming_x, after_transforming_y, after_transforming_yaw = [], [], []
        for idx, tracked_agent in enumerate(self._agents):

            coord_x, coord_y, coord_yaw = tracked_agent.predictions[0].valid_waypoints[0].x, tracked_agent.predictions[0].valid_waypoints[0].y, tracked_agent.predictions[0].valid_waypoints[0].heading
            coord_vel_x, coord_vel_y = tracked_agent.predictions[0].valid_waypoints[0].velocity.x, tracked_agent.predictions[0].valid_waypoints[0].velocity.y
            
            # for visualizing the position of agents + comparing before and after transformation position
            before_transforming_x.append(coord_x)
            before_transforming_y.append(coord_y)
            before_transforming_yaw.append(coord_yaw)
            map_x, map_y, map_yaw = self._convert_coord_map.pos_to_map(coord_x, coord_y, coord_yaw)
            _, _, second_point_map_yaw = self._convert_coord_map.pos_to_map(tracked_agent.predictions[0].valid_waypoints[1].x, tracked_agent.predictions[0].valid_waypoints[1].y, tracked_agent.predictions[0].valid_waypoints[1].heading)
            map_vel_x, map_vel_y, _ = self._convert_coord_map.pos_to_map_vel(coord_vel_x, coord_vel_y, coord_yaw)
            second_point_map_vel_x, second_point_map_vel_y, _ = self._convert_coord_map.pos_to_map_vel(tracked_agent.predictions[0].valid_waypoints[1].velocity.x, tracked_agent.predictions[0].valid_waypoints[1].velocity.y, tracked_agent.predictions[0].valid_waypoints[1].heading)

            # for comparing before and after transformation position
            after_transforming_x.append(map_x)
            after_transforming_y.append(map_y)
            after_transforming_yaw.append(map_yaw)

            self._initial_map_x.append(map_x)
            self._initial_map_y.append(map_y)
            self._initial_map_yaw.append(map_yaw)
            self._initial_map_vel_x.append(map_vel_x)
            self._initial_map_vel_y.append(map_vel_y)


            # the commented initialization are for when starting from stationary states and optimizing the actions
            self._states['pos'][idx] = torch.tensor([map_x, map_y], device=device, dtype=torch.float64, requires_grad=True)
            self._states['yaw'][idx] = torch.tensor([map_yaw], device=device, dtype=torch.float64, requires_grad=True)
            self._states['vel'][idx] = torch.tensor([map_vel_x, map_vel_y], device=device, requires_grad=True, dtype=torch.float64)
            # self._states['vel'][idx] = torch.tensor([0., 0.], dtype=torch.float64, device=device, requires_grad=True)
            # self._states['speed'][idx] = torch.tensor([0.0], dtype=torch.float64, device=device, requires_grad=True)
            self._states['speed'][idx] = torch.tensor([np.linalg.norm(np.array([map_vel_x, map_vel_y]))], dtype=torch.float64, device=device, requires_grad=True)
            # self._states['accel'][idx] = torch.tensor([0.0], dtype=torch.float64, device=device, requires_grad=True)
            # approx_initi_accel = (np.linalg.norm(np.array([map_vel_x, map_vel_y])) - np.linalg.norm(np.array([second_point_map_vel_x, second_point_map_vel_y])))/self._observation_trajectory_sampling.interval_length
            self._states['accel'][idx] = torch.tensor([0.0], dtype=torch.float64, device=device, requires_grad=True)
            approx_tire_steering_angle = np.arctan(3.089*(second_point_map_yaw - map_yaw)/(self._observation_trajectory_sampling.interval_length*np.hypot(map_vel_x, map_vel_y)+1e-3))
            self._initial_steering_rate.append(approx_tire_steering_angle)
            self._states['steering_angle'][idx] = torch.clamp(torch.tensor([approx_tire_steering_angle], device=device, dtype=torch.float64, requires_grad=True), min=-torch.pi/3, max=torch.pi/3)
            # self._states['steering_angle'][idx] = torch.tensor([0.0], device=device, dtype=torch.float64, requires_grad=True)
        

        
        # using before_after_transform, agents_position_before_transformation functions for visualization
        # before_after_transform(self._simulation.scenario.scenario_name, self._experiment_name, before_transforming_x, before_transforming_y, before_transforming_yaw, after_transforming_x, after_transforming_y, after_transforming_yaw)
        # agents_position_before_transformation(self._simulation.scenario.scenario_name, self._experiment_name, before_transforming_x, before_transforming_y, before_transforming_yaw)

        for key in self._states:
            self._states[key].requires_grad_(True).retain_grad()
    


    def reset_dynamic_states(self):

        """
        Function to reset the state of adversary agents back to their initial state (first time point in the simulation)
        """

        self._states = {'pos': torch.zeros(self._number_agents, 2, requires_grad=True).to(device=device), 
                        'yaw': torch.zeros(self._number_agents, 1, requires_grad=True).to(device=device), 
                        'steering_angle': torch.zeros(self._number_agents, 1, requires_grad=True).to(device=device), 
                        'vel': torch.zeros(self._number_agents, 2, requires_grad=True).to(device=device), 
                        'accel': torch.zeros(self._number_agents, 1, requires_grad=True).to(device=device), 
                        'speed': torch.zeros(self._number_agents, 1, requires_grad=True).to(device=device)}

        for idx, _ in enumerate(self._agents):

            map_x, map_y, map_yaw = self._initial_map_x[idx], self._initial_map_y[idx], self._initial_map_yaw[idx]
            map_vel_x, map_vel_y = self._initial_map_vel_x[idx], self._initial_map_vel_y[idx]
            approx_tire_steering_angle = self._initial_steering_rate[idx]

            self._states['pos'][idx] = torch.tensor([map_x, map_y], device=device, dtype=torch.float64, requires_grad=True)
            # self._states['pos'][idx].retain_grad()
            self._states['yaw'][idx] = torch.tensor([map_yaw], device=device, dtype=torch.float64, requires_grad=True)
            # self._states['yaw'][idx].retain_grad()
            self._states['vel'][idx] = torch.tensor([map_vel_x, map_vel_y], device=device, dtype=torch.float64, requires_grad=True)
            # self._states['vel'][idx] = torch.tensor([0., 0.], dtype=torch.float64, device=device, requires_grad=True)
            # self._states['speed'][idx] = torch.tensor([0.0], dtype=torch.float64, device=device, requires_grad=True)
            self._states['speed'][idx] = torch.tensor([np.linalg.norm(np.array([map_vel_x, map_vel_y]))], dtype=torch.float64, device=device, requires_grad=True)
            self._states['accel'][idx] = torch.tensor([0.0], dtype=torch.float64, device=device, requires_grad=True)
            # approx_tire_steering_angle = np.arctan(3.089*(tracked_agent.predictions[0].valid_waypoints[1].heading - tracked_agent.predictions[0].valid_waypoints[0].heading)/(self._observation_trajectory_sampling.interval_length*np.hypot(tracked_agent.velocity.x, tracked_agent.velocity.y)+1e-3))
            self._states['steering_angle'][idx] = torch.clamp(torch.tensor([approx_tire_steering_angle], device=device, dtype=torch.float64, requires_grad=True), min=-torch.pi/3, max=torch.pi/3)
            # self._states['steering_angle'][idx] = torch.clamp(torch.tensor([0.0], device=device, dtype=torch.float64, requires_grad=True), min=-torch.pi/3, max=torch.pi/3)
        
        
        for key in self._states:
            self._states[key].requires_grad_(True).retain_grad()




    def initialize(self) -> None:
        """
        To obtain the initial actions using the lqr controller from the logged trajectory.
        with the option of optimizing the extracted actions to better fit the logged trajectory after erolling the bm.
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

        # the losses to be accumulated overall/per-agent across optimization iterations
        self.routedeviation_losses_per_opt = []
        self.collision_losses_per_opt = []
        self.routedeviation_loss_agents_per_opt = []
        self.collision_loss_agents_per_opt = []


        # different options for chossing the fixed point when using fixed_dummy as a cost
        if 'fixed_dummy' in self._costs_to_use:
            # self._fixed_point = np.array([self._simulation._ego_controller.get_state().center.x,self._simulation._ego_controller.get_state().center.y])
            self._fixed_point = np.array([1000, 1000])


        # setting the trajectory_sampling value of the observation to the correct value
        # it should be set correctly without this call is freq_multiplier = 1
        self._simulation._observations.set_traj_sampling(self._observation_trajectory_sampling)
        
        num_agents = self._number_agents
                
        # defining an indexing for the agents, storing their corresponding tokens
        self._agent_indexes: Dict[str, int] = {}
        self._agent_tokens: Dict[int, str] = {}


        # defining the actions: throttle and steer
        # we consider a nn.Parameter for all the actions taken at each time step of the simulation => to be able to optimize all the actions for all the agents at a certain time step in parallel
        # later the actions are changed in shape in 'reshape_actions' function to be of the shape 'self._number_agents x self._horizon x 1'
        self._throttle_temp = [torch.nn.Parameter(
            torch.zeros(
                num_agents, 1, # the horizon parameter is the same as the numer of actions
                # and is one less than the number of iterations in the simulation
                device=device, 
                dtype=torch.float64
            ),
            requires_grad=True
        ) for _ in range(self._horizon)]

        self._steer_temp = [torch.nn.Parameter(
            torch.zeros(
                num_agents, 1,
                device=device, 
                dtype=torch.float64
            ),
            requires_grad=True
        ) for _ in range(self._horizon)]

        self._actions_temp = {'throttle': self._throttle_temp, 'steer': self._steer_temp}
        self.init_dynamic_states()



        # logged position, yaw, and velocity
        self._positions = torch.zeros(num_agents, self._horizon+1, 2, dtype=torch.float64, requires_grad=True, device=device)
        self._velocities = torch.zeros(num_agents, self._horizon+1, 2, dtype=torch.float64, requires_grad=True, device=device)
        self._headings = torch.zeros(num_agents, self._horizon+1, 1, dtype=torch.float64, requires_grad=True, device=device)
        # a mask to later optimize the actions of all the agents in parallel, but only for valid actions (we do not want to optimize the action of an agent at time step T (or higher), when it has stopped being in the scene at time step T-1)
        self._action_mask = torch.zeros(num_agents, self._horizon+1, dtype=torch.float64, requires_grad=False, device=device)


        # the helper class that calls the controller and have different options for optimizing the actions
        self._trajectory_reconstructor = TrajectoryReconstructor(self._motion_model, self._tracker, self._horizon, self._throttle_temp, self._steer_temp, self._map_resolution)
        
    
        trajectories = []
        endtimepoints = []

        # Extracting for each agentc its entire trajectory and convert it to map coordinate system
        for idx, tracked_agent in enumerate(self._agents):

           
            self._agent_indexes[tracked_agent.metadata.track_token] = idx
            self._agent_tokens[idx] = tracked_agent.metadata.track_token


            print('this is the agent ', tracked_agent.metadata.track_token, '  ', idx)


            waypoints: List[Waypoint] = []
            interpolated_traj = tracked_agent.predictions[0].trajectory
            start_timepoint, end_timepoint = interpolated_traj.start_time, interpolated_traj.end_time
            current_timepoint = start_timepoint
            counter_steps = 0

            # while a waypoint can still be extracted from the trajectory
            while current_timepoint < end_timepoint:
                # this might be a bit costly because of the get_state_at_time function
                # alternative way: intepolate the trajectory at first and the extrating its waypoints directly 
                current_state: Waypoint = interpolated_traj.get_state_at_time(current_timepoint)
                coord_x, coord_y, coord_yaw = current_state.x, current_state.y, current_state.heading
                coord_vel_x, coord_vel_y = current_state.velocity.x, current_state.velocity.y
                map_x, map_y, map_yaw = self._convert_coord_map.pos_to_map(coord_x, coord_y, coord_yaw)
                map_vel_x, map_vel_y, _ = self._convert_coord_map.pos_to_map_vel(coord_vel_x, coord_vel_y, coord_yaw)

                current_waypoint = Waypoint(current_timepoint, OrientedBox.from_new_pose(tracked_agent.box, StateSE2(map_x, map_y, map_yaw)), StateVector2D(map_vel_x, map_vel_y))
                waypoints.append(current_waypoint)

                with torch.no_grad():
                    self._positions[idx, counter_steps, :] = torch.tensor([map_x, map_y], dtype=torch.float64)
                    self._velocities[idx, counter_steps, :] = torch.tensor([map_vel_x, map_vel_y], dtype=torch.float64)
                    self._headings[idx, counter_steps, :] = torch.tensor([map_yaw], dtype=torch.float64)
                    self._action_mask[idx, counter_steps] = 1.0


                current_timepoint = current_timepoint + self._interval_timepoint
                counter_steps += 1

            trajectories.append(waypoints)
            endtimepoints.append(current_timepoint)

        # providing the _trajectory_reconstructor with the logged postions, yaws, and velocities
        self._trajectory_reconstructor.initialize_optimization(self._action_mask, self._positions, self._headings, self._velocities)
        self._trajectory_reconstructor.reset_error_losses()
        optimization_time = 0
        for idx, tracked_agent in enumerate(self._agents):

            print('here we are ', idx)

            waypoints = trajectories[idx]
            print(len(waypoints))
            if len(waypoints) > 1:
                current_timepoint = endtimepoints[idx]
                end_timepoint = current_timepoint - self._interval_timepoint
                current_state = {_key: _value.clone().detach().to(device) for _key, _value in self.get_adv_state(id=idx).items()} # B*N*S
                # converting the trajectory to InterpolatedTrajectory class, for compatibility reasons in controller functions
                transformed_trajectory = InterpolatedTrajectory(waypoints)

                # providing _trajectory_reconstructor with information on the current trajectory and agent
                self._trajectory_reconstructor.set_current_trajectory(idx, tracked_agent.box, transformed_trajectory, waypoints, current_state, end_timepoint, self._interval_timepoint, storing_path=f'/home/kpegah/workspace/Recontruction/{self._simulation.scenario.scenario_name}')
                # extracting the actions using only the controller
                self._trajectory_reconstructor.extract_actions_only_controller()
                # resetting the time and state back to initial before optimizing the extracted actions
                self._trajectory_reconstructor.reset_time_state()

                # different options can be chosen for optimizing the actions in a step-by-step manner
                # have a look at the individual_step_by_step_optimization from trajectory_reconstructor
                start_time = perf_counter()
                self._trajectory_reconstructor.individual_step_by_step_optimization('controller')
                optimization_time += (perf_counter() - start_time)

                self._trajectory_reconstructor.report(idx, len(waypoints)-1, current_state)

        # writing the losses accumulated for all the agents (for each agent on a separate line, the loss for positon, yaw, and velocity), and the last line being the optimization time
        self._trajectory_reconstructor.write_losses('step_by_step_losses', optimization_time)

        

        # the extension of the agents that will be used in collision cost
        # the value of the parameters are extracted coordinates in https://github.dev/motional/nuplan-devkit/tree/master/nuplan/planning : 
        self.ego_extent = torch.tensor(
            [5.176/2.,
             2.297/2.],
            device=device,
            dtype=torch.float64
        ).view(1, 1, 2).expand(1, 1, 2)

        self.adv_extent = torch.tensor(
            [5.176/2.,
             2.297/2.],
            device=device,
            dtype=torch.float64
        ).view(1, 1, 2).expand(1, num_agents, 2)



    
    def reshape_actions(self):
        """
        reshaping the actions to bring the horizon component into the tensor,
        so that we can use a single optimizer for all the actions in time.
        """

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
                self._throttle[:, time_step, :] = self._throttle_temp[time_step][:,:]
                self._throttle_original[:, time_step, :] = self._throttle_temp[time_step][:,:]
                self._steer[:, time_step, :] = self._steer_temp[time_step][:,:]
                self._steer_original[:, time_step, :] = self._steer_temp[time_step][:,:]

            

        self._actions = {'throttle': self._throttle, 'steer': self._steer}
        
        # TO_BE_OPTIMIZED : the beta parameters of Adam are taken from the King's code
        self._optimizer_throttle = torch.optim.Adam([self._throttle], lr=self.lr_throttle, betas=(0.8, 0.99))
        self._optimizer_steer = torch.optim.Adam([self._steer], lr=self.lr_steer, betas=(0.8, 0.99))

    def print_initial_actions(self, id:int):
        '''
        :param id: is the agent for which we want to print actions
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
                adv_action.update({substate: torch.unsqueeze(self._actions[substate][:, current_iteration, ...], dim=0)})
            else:
                adv_action.update(
                    {substate: torch.unsqueeze(self._actions[substate][id:id+1, current_iteration, ...], dim=0)}
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
                adv_action.update({substate: torch.unsqueeze(self._actions_temp[substate][current_iteration][:, ...], dim=0)})
            else:
                # we index for id+1 since id 0 in the tensor is the ego agent
                adv_action.update(
                    {substate: torch.unsqueeze(self._actions_temp[substate][current_iteration][id:id+1, ...], dim=0)}
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

                self._states[substate] = torch.cat(
                    attached_states,
                    dim=0,
                )
            self._states[substate].requires_grad_(True).retain_grad()

    def set_adv_state_to_original(self, next_state: Dict[str, torch.TensorType] = None, next_iteration:int = 1, exception_agent_index:int = 0):
        """
        Set the current adversarial state to the original one extracted during the initialization
        using the input next_state only for the agent that has collided with the ego.

        :param next_state: setting the parameters of the next_state, a dict of type[str, Tensor]
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


    def step(self, current_iteration:int) -> Dict[str, ((float, float), (float, float), float)]:
        '''
        to update the state of the agents at the 'current_iterarion' using the actions taken at step 'current_iterarion-1'.

        in the case where the 'current_iterarion' is 0, the state does not get updated.

        return:
            a dictionary where the keys are tokens of agents, and the values are (pos, velocity, yaw)
            this will be used to update the 'observation' in 'simulation_runner'
        '''


        if self._collision_occurred and self._collision_strat=='back_to_after_bm':
            # updating the actions back to original in the very first step of the simulation
            if current_iteration==1:
                self.actions_to_original()
                self.stopping_collider()
            return self.step_after_collision(current_iteration)
        elif self._collision_occurred and self._collision_strat=='back_to_before_bm':
            # updating the flag to use the original states instead of enrolling actions
            # for all the agents except the one that has collided with the ego
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


        # TODO : updating the list of agents at each iteration
        #        some agents may go inactive and untracked, their predicted trajectory should be filled in the untracked spaces?
        #        or maybe we can only work on the agents that remain tracked during the entire simulation
        

        not_differentiable_state = {}  
      
        if not current_iteration==0:
            temp_bm_iter = self.bm_iteration
            while temp_bm_iter <= current_iteration*self._freq_multiplier: 
                self.set_adv_state(self._motion_model.forward_all(self.get_adv_state(), self.get_adv_actions(temp_bm_iter-1)))
                temp_bm_iter += 1
            
            self.bm_iteration += self._freq_multiplier

        current_state = self.get_adv_state() # Dict[str (type of state), torch.Tensor (idx of agent, ...)]
        agents_pos = current_state['pos'][0].clone().detach().cpu().numpy() # adding [0] to get rid of the batch dimension
        agents_vel = current_state['vel'][0].clone().detach().cpu().numpy()
        agents_yaw = current_state['yaw'][0].clone().detach().cpu().numpy()


        ego_position = self._ego_state_np['pos']
        ego_heading = self._ego_state_np['yaw']
        ego_velocity = self._ego_state_np['vel']
        


        # check the collision between adversary agents and the ego vehicle
        # TODO: completing the following
        # if self.check_collision_at_iteration(agents_pos, agents_yaw, ego_position, ego_heading):
        #     pass

        agents_pos_ = np.concatenate((agents_pos, ego_position), axis=0)      
        agents_yaw_ = np.concatenate((agents_yaw, ego_heading), axis=0)    
        agents_vel_ = np.concatenate((agents_vel, ego_velocity), axis=0)    

        self.state_buffer.append({'pos': agents_pos_, 'vel': agents_vel_, 'yaw': agents_yaw_})


        # after_transforming_x, after_transforming_y, after_transforming_yaw = [], [], []
        for idx in range(self._number_agents):
            coord_vel_x, coord_vel_y, _ = self._convert_coord_map.pos_to_coord_vel(agents_vel[idx, 0], agents_vel[idx, 1], agents_yaw[idx, 0])
            coord_pos_x, coord_pos_y, coord_pos_yaw = self._convert_coord_map.pos_to_coord(agents_pos[idx, 0], agents_pos[idx, 1], agents_yaw[idx, 0])
            # after_transforming_x.append(coord_pos_x)
            # after_transforming_y.append(coord_pos_y)
            # after_transforming_yaw.append(coord_pos_yaw)
            not_differentiable_state[self._agent_tokens[idx]] = ((coord_pos_x, coord_pos_y), (coord_vel_x, coord_vel_y), (coord_pos_yaw))
        

        if not self._collision_occurred and self.check_collision_simple(agents_pos, ego_position):
            collision, collision_index = self.check_collision(agents_pos, agents_yaw, ego_position, ego_heading)
            if collision:
                # should make the next states the same state as the current one
                # keep the current not_differentiable_state, and from now on just return it for the iterations after the current one, without enrolling bm
                self._collision_occurred = True
                self._collision_index = collision_index
                self._collision_iteration = current_iteration*self._freq_multiplier
                self._collision_not_differentiable_state = not_differentiable_state
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
           

        if current_iteration*self._freq_multiplier>=self._collision_iteration:
            with torch.no_grad():
                self._states['vel'][self._collision_index] = torch.tensor([0., 0.], dtype=torch.float64, device=device)
                self._states['speed'][self._collision_index] = torch.tensor([0.0], dtype=torch.float64, device=device)
                self._states['accel'][self._collision_index] = torch.tensor([0.0], dtype=torch.float64, device=device)
                # the steering angle should be set to the one at the last step before collision => already set by set_adv_sate
            

        current_state = self.get_adv_state() # Dict[str (type of state), torch.Tensor (idx of agent, ...)]
        agents_pos = current_state['pos'][0].clone().detach().cpu().numpy() # adding [0] to get rid of the batch dimension that we don't want to have in nuplan for now
        agents_vel = current_state['vel'][0].clone().detach().cpu().numpy()
        agents_yaw = current_state['yaw'][0].clone().detach().cpu().numpy()


        # accumulating the out-of-drivable area penalty for the colloded vehicle at each step of its trajectory
        self._collided_agent_drivable_cost += self.drivable_area_metric()
        # checking for collision between adversary agents
        collided_indices = self.check_adversary_collision(agents_pos, agents_yaw)
        for collided_idx in collided_indices:
            self.stopping_agent(collided_idx, current_iteration*self._freq_multiplier)
            self._adversary_collisions += 1


        ego_position = self._ego_state_whole['pos'][current_iteration].cpu().detach().numpy()[None, ...]
        ego_heading = self._ego_state_whole['yaw'][current_iteration].cpu().detach().numpy()[None, ...]
        ego_velocity = self._ego_state_whole['vel'][current_iteration].cpu().detach().numpy()[None, ...]
        
        agents_pos_ = np.concatenate((agents_pos, ego_position), axis=0)      
        agents_yaw_ = np.concatenate((agents_yaw, ego_heading), axis=0)    
        agents_vel_ = np.concatenate((agents_vel, ego_velocity), axis=0)    

        self.state_buffer.append({'pos': agents_pos_, 'vel': agents_vel_, 'yaw': agents_yaw_})

        # after_transforming_x, after_transforming_y, after_transforming_yaw = [], [], []
        for idx in range(self._number_agents):
            coord_vel_x, coord_vel_y, _ = self._convert_coord_map.pos_to_coord_vel(agents_vel[idx, 0], agents_vel[idx, 1], agents_yaw[idx, 0])
            coord_pos_x, coord_pos_y, coord_pos_yaw = self._convert_coord_map.pos_to_coord(agents_pos[idx, 0], agents_pos[idx, 1], agents_yaw[idx, 0])
            # after_transforming_x.append(coord_pos_x)
            # after_transforming_y.append(coord_pos_y)
            # after_transforming_yaw.append(coord_pos_yaw)
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

    
    def use_original_states(self):

        self._use_original_states = True

    def stopping_collider(self):
        """
        Stopping the agent that has collided with the ego vehicle.
        """

        with torch.no_grad():

            self._throttle[self._collision_index, self._collision_iteration:] = torch.zeros((self._number_actions-self._collision_iteration), 1)
            self._steer[self._collision_index, self._collision_iteration:] = torch.zeros((self._number_actions-self._collision_iteration), 1)
            # should also make the velocity and speed zero for this agent => do it in the step_after_collision

    def stopping_agent(self, idx, step):
        """
        Stopping the agent 'idx' at step 'step'
        """

        with torch.no_grad():
            
            self._throttle[idx, step:] = torch.zeros((self._number_actions-step), 1)
            self._steer[idx, step:] = torch.zeros((self._number_actions-step), 1)
            # should also make the velocity and speed zero for this agent => do it in the step_after_collision
            self._states['vel'][idx] = torch.tensor([0., 0.], dtype=torch.float64, device=device)
            self._states['speed'][idx] = torch.tensor([0.0], dtype=torch.float64, device=device)
            self._states['accel'][idx] = torch.tensor([0.0], dtype=torch.float64, device=device)
            

    def drivable_area_metric(self):
        # for the collided agent, computes its current drivable area cost => higher if out of drivable area
        return self.effient_deviation_cost.collided_agent_cost(self._states['pos'][self._collision_index, :])
    
    def get_collided_agent_drivable_cost(self):

        return self._collided_agent_drivable_cost
    
    def get_number_adversary_collisions(self):
        """
        number of agents that have collided with each other during the simulation, after collision
        """

        return self._adversary_collisions
    
    def get_collision_after_traj_changes(self):
        """
        Returns True is the collision still happens after changing the non-colliding trajectories [back to their original states] or [back to after extracting actions + enrolling bm] or [not modifying the optimized trajectories]
        """
        return self._collision_after_trajectory_changes

    def update_ego_state(self, current_ego_state: EgoState, current_iteration: int):
        '''
        update the state of the ego agent using the given current state (and converts its position to map coordinate system)

        :param trajectory: the ego trajectory that is taken at the current iteration
        :param next_iteration: the next iteration of simulation

        '''

        assert self.update_ego_state_on_iteration==current_iteration, 'we are not optimizing on the right iteration of the simulation'

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

        
        self._ego_state_np['pos'][0,:] = np.array([ego_x_map, ego_y_map], dtype=np.float64)
        self._ego_state_np['vel'][0,:] = np.array([ego_vel_x_map, ego_vel_y_map], dtype=np.float64)
        self._ego_state_np['yaw'][0,:] = np.array([ego_heading_map], dtype=np.float64)


        self.update_ego_state_on_iteration += 1

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

        
        
        self._ego_state_np['pos'][0,:] = np.array([ego_x_map, ego_y_map], dtype=np.float64)
        self._ego_state_np['vel'][0,:] = np.array([ego_vel_x_map, ego_vel_y_map], dtype=np.float64)
        self._ego_state_np['yaw'][0,:] = np.array([ego_heading_map], dtype=np.float64)

        self.init_map()


    def init_map(self):
        """
        Function to initialize the cropped map,
        using the 'self.transformed_crop_x' and 'self.transformed_crop_y' set in 'init_ego_state' function
        """
        self._data_nondrivable_map = self._data_nondrivable_map[int(self.transformed_crop_y-1000):int(self.transformed_crop_y+1000), int(self.transformed_crop_x-1000):int(self.transformed_crop_x+1000)]
        self.effient_deviation_cost = DrivableAreaCost(self._data_nondrivable_map)
        self._convert_coord_map = TransformCoordMap(self._map_resolution, self._map_transform, self.transformed_crop_x, self.transformed_crop_y)
        self._data_nondrivable_map = torch.tensor(self._data_nondrivable_map, device=device, dtype=torch.float64, requires_grad=True)
        self._transpose_data_drivable_map = torch.transpose((1-self._data_nondrivable_map), 1, 0)


    
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
        """
        Function to compute the cost at each step of the simulation (or optimization without simulation)
        """
        drivable_computation_time = 0
        
        input_cost_ego_state = {}
        # we use ego_state_whole to be able to compute the cost even when not calling the planner but just optimizing
        for substate in self._ego_state.keys():
            input_cost_ego_state[substate] = torch.unsqueeze(torch.unsqueeze(self._ego_state_whole[substate][current_iteration,:], dim=0), dim=0)

        input_cost_adv_state = self.get_adv_state()


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
            self.collision_losses_per_step.append(current_cost)
                


        if 'drivable_efficient' in self._costs_to_use:

            # TODO: check these visualization functions
            # self.routedeviation_heatmap()
            # self.routedeviation_heatmap_agents(input_cost_adv_state['pos'], input_cost_adv_state['yaw'])
            # self.routedeviation_agents(input_cost_adv_state['pos'], input_cost_adv_state['yaw'])
            
            input_cost_adv_state['pos'].retain_grad()
            drivable_computation_time = perf_counter()
            gradients, current_cost_agents = self.effient_deviation_cost(input_cost_adv_state['pos'])
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
            self.routedeviation_losses_per_step.append(current_cost)
          

        if 'drivable_king' in self._costs_to_use:
            
            drivable_computation_time = perf_counter()
            adv_rd_cost = self.adv_rd_cost(
                self._transpose_data_drivable_map,
                input_cost_adv_state['pos'],
                input_cost_adv_state['yaw'],
            )
            self._cost_dict['adv_rd'].append(adv_rd_cost*self.weight_drivable)
            drivable_computation_time = perf_counter() - drivable_computation_time
            current_cost = torch.sum(adv_rd_cost).detach().cpu().numpy()
            self._accumulated_drivable_loss += current_cost
            self.routedeviation_losses_per_step.append(current_cost)

            

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
            self.collision_losses_per_step.append(current_cost)




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
            self.collision_losses_per_step.append(current_cost)


        return drivable_computation_time

        

    def back_prop(self):
        """
        Back-propagating the costs computed druing the entire simulation.
        """
        cost_dict = self._cost_dict
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
        
        
        if 'drivable_efficient' in self._costs_to_use:

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


        self._optimizer_throttle.zero_grad()
        self._optimizer_steer.zero_grad()
        self.routedeviation_losses_per_opt.append(self._accumulated_drivable_loss)
        self.collision_losses_per_opt.append(self._accumulated_collision_loss)

        self.routedeviation_loss_agents_per_opt.append(self._accumulated_drivable_loss_agents)
        self.collision_loss_agents_per_opt.append(self._accumulated_collision_loss_agents)



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
    
        for idx, box in enumerate(boxes): # box should be of shape (4,2)
            if self.overlap_complex(box, reference_box):
                return True, idx  # Collision detected
        return False, -1  # No collision detected

    def overlap(self, box1, box2):
        # Check if two bounding boxes overlap
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


        return True

   

    def save_state_buffer(self):

        self.whole_state_buffers.append(self.state_buffer)
      

    def optimize_without_simulation(self):

        optimization_rounds = self._optimization_jump
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
    



    # VISUALIZATION FUNCTIONS SHORTCUTS

    def visualize_grads_steer(self, optimization_iteration):
        visualize_grads_steer(self._simulation.scenario.scenario_name,
                              self._experiment_name,
                              optimization_iteration,
                              self._number_agents,
                              self.whole_state_buffers,
                              self.steer_gradients)
            
    def visualize_grads_throttle(self, optimization_iteration):
        visualize_grads_throttle(self._simulation.scenario.scenario_name,
                              self._experiment_name,
                              optimization_iteration,
                              self._number_agents,
                              self.whole_state_buffers,
                              self.throttle_gradients)
        
    def visualize_steer(self, optimization_iteration):
        visualize_steer(self._simulation.scenario.scenario_name,
                              self._experiment_name,
                              optimization_iteration,
                              self._number_agents,
                              self.whole_state_buffers,
                              self.steers)

    def visualize_throttle(self, optimization_iteration): 
        visualize_throttle(self._simulation.scenario.scenario_name,
                              self._experiment_name,
                              optimization_iteration,
                              self._number_agents,
                              self.whole_state_buffers,
                              self.throttles)
    
    def compare_king_efficient_deviation_cost_heatmaps(self,):
        """
        Generates two heatmaps:
        one corresponding to the deviation cost computed by king's function.
        the other corresponding to the efficient deviation cost function.
        """
        
        routedeviation_king_heatmap(self._simulation.scenario.scenario_name, self._experiment_name, self._transpose_data_drivable_map)

        routedeviation_efficient_heatmap(self._simulation.scenario.scenario_name, self._experiment_name, self._data_nondrivable_map, self.effient_deviation_cost)


    def compare_king_efficient_deviation_cost_agents(self,):
        """
        Generates two maps where at the position of each agent, its deviation cost is indicated:
        One map using king's deviation cost.
        The other using the efficient deviation cost.
        """
        
        routedeviation_king_agents_map(self._simulation.scenario.scenario_name, self._experiment_name, self._number_agents, self.get_adv_state(), self.adv_rd_cost, self._transpose_data_drivable_map, self._data_nondrivable_map)

        routedeviation_efficient_agents_map(self._simulation.scenario.scenario_name, self._experiment_name, self._number_agents, self.get_adv_state(), self.effient_deviation_cost, self._data_nondrivable_map)
    

    def plot_losses(self,): # losses_per_step_and_per_opt
        """
        plots drivable and collision losees:
        1. accumulated for all the agents and steps, across optimization iterations
        2. accumulated for all the agents, across the simulation steps of the last optimization iteration
        """
        plot_losses(self._simulation.scenario.scenario_name, self._experiment_name, self._costs_to_use, self.routedeviation_losses_per_opt, self.routedeviation_losses_per_step, self.collision_losses_per_opt, self.collision_losses_per_step)

    def plot_loss_per_agent(self,): # losses_per_agent_per_opt
        """
        To visualize the evolution of collision and drivable losses, per agent, across the optimization iterations.
        Visualizes a plot at the position of each agent of the evolution of its collision and drivable loss.
        """
        plot_loss_per_agent(self._simulation.scenario.scenario_name, self._experiment_name, self._number_agents, self.whole_state_buffers, self.routedeviation_loss_agents_per_opt.detach().cpu().numpy(), self.collision_loss_agents_per_opt.detach().cpu().numpy())
