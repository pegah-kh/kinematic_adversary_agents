"""
there are different options to extract the actions:

1. to use the controller on each of the steps of the trajectory
2. individual agents: initialize the actions of step 't+1' with optimized actions of step 't', and optimize them further
3. ensemble of agents: optimize the actions of the whole trajectory for all the agents afetr enrolling the bm for all the steps of trajectory


Not added options:

4. individual agents: optimize all the actions after enrolling the whole trajectory
5. ensemble of agents: optimizing the actions immediately after enrolling the bm and obtaining next state


Methods compared in the report:

1. only using the controller
2. immediate optimization for each agent
3. optimizing the actions of all agents after obtaining all the states of simulation
4. combination of 2 and 3
"""

from __future__ import annotations

import logging
from typing import List, Dict, Optional, Literal
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
import wandb
import pandas as pd
import random
import csv
import numpy as np



from nuplan.common.actor_state.waypoint import Waypoint
from nuplan.common.actor_state.oriented_box import OrientedBox
from nuplan.common.actor_state.state_representation import StateSE2, StateVector2D, TimePoint

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def create_directory_if_not_exists(directory_path):
    """
        Creating a dirctory at the given directory path
        :param directory_path
    """
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)



class TrajectoryReconstructor():
    """
    reconstructs the trajectory through controller and bicycle model, from the given logged trajectory.
    + options to optimize the initially estimated actions.
    + stores the actions for the next calls of the optimizer.
    """

    def __init__(self, username, motion_model, controller, horizon, throttles, steers, map_resolution, storing_path):
        self._username = username

        self._motion_model = motion_model
        self._controller = controller

        self._horizon = horizon

        self._throttles = throttles
        self._steers = steers
        self._actions = {'throttle': self._throttles, 'steer': self._steers}

        self._map_resolution = map_resolution

        # to store the estimated actions
        self._storing_path = storing_path
        create_directory_if_not_exists(self._storing_path)


    # helper functions to construct from numpy variables a dict of actions
    def get_actions(self, throttle, steer):
        return {'throttle': torch.unsqueeze(torch.unsqueeze(throttle, dim=0), dim=0),
                'steer': torch.unsqueeze(torch.unsqueeze(steer, dim=0), dim=0)}

    def get_optimize_state(self, current_state):
        new_state = {key:value.clone().detach().requires_grad_(True) for key,value in current_state.items()}
        return new_state
    
    
    def get_adv_actions_temp(self, current_iteration:int = 0, id:Optional(int) = None):
        """
        the same as the above function, but to get actions of agents

        returns:
                the actions of adversary agents at the current iteration of shape B x N x S
        """
        adv_action = {}
        for substate in self._actions.keys():
            if id == None:
                adv_action.update({substate: torch.unsqueeze(self._actions[substate][current_iteration], dim=0)})
             
            else:
                adv_action.update(
                    {substate: torch.unsqueeze(self._actions[substate][current_iteration][id:id+1, ...], dim=0)}
                )
            
        return adv_action
    
    # 1. SIMPLE FUNCTIONS TO EXTRACT THE ACTIONS FROM A SINGLE TRAJECTORY
    def set_current_trajectory(self, idx: int, agent_box: OrientedBox, trajectory, waypoints: List[Waypoint], initial_state: Dict[str, torch.Tensor], end_timepoint: TimePoint, interval_timepoint: TimePoint):
        self._idx = idx # the index of the agent
        self._agent_box = agent_box
        self._trajectory = trajectory
        self._waypoints = waypoints


        # timepoints of the start of the trajectory, its end, and the interval time between two consequent step
        self._start_timepoint = TimePoint(waypoints[0].time_point.time_us+100)
        self._end_timepoint = end_timepoint
        self._current_timepoint = self._start_timepoint
        self._next_timepoint = self._current_timepoint + interval_timepoint
        self._interval_timepoint = interval_timepoint


        # initialization
        # to determine initial estimation of the steering angle
        self._current_waypoint = self._waypoints[0]
        self._current_tire_steering_angle = np.arctan(3.089*(self._waypoints[1].heading - self._current_waypoint.heading)/(self._interval_timepoint.time_s/self._map_resolution*np.hypot(self._current_waypoint.velocity.x, self._current_waypoint.velocity.y)+1e-3))
        
        self._current_state = self.get_optimize_state(initial_state)
        self._initial_state = self.get_optimize_state(initial_state)
    

    def reset_time_state(self):
        self._current_state = self.get_optimize_state(self._initial_state)
        self._current_timepoint = self._start_timepoint
        self._next_timepoint = self._current_timepoint + self._interval_timepoint

        self._current_waypoint = self._waypoints[0]
        self._current_tire_steering_angle = np.arctan(3.089*(self._waypoints[1].heading - self._current_waypoint.heading)/(self._interval_timepoint.time_s/self._map_resolution*np.hypot(self._current_waypoint.velocity.x, self._current_waypoint.velocity.y)+1e-3))

        self._trajectory_ended = False
        
    

    def call_controller(self, step:int):

        throttle, steer = self._controller.track_trajectory(self._current_timepoint, self._next_timepoint, self._current_waypoint, self._trajectory, initial_steering_angle=self._current_tire_steering_angle)

        with torch.no_grad():
            self._throttles[step][self._idx,:] = throttle
            self._steers[step][self._idx, :] = steer
            
            self._current_state = self._motion_model.forward_reconstruction(self._current_state, self.get_actions(torch.tensor([throttle], dtype=torch.float64, device=device), torch.tensor([steer], dtype=torch.float64, device=device)))
            self._current_waypoint = Waypoint(self._next_timepoint, oriented_box=OrientedBox.from_new_pose(self._agent_box, StateSE2(self._current_state['pos'].cpu()[0,0,0], self._current_state['pos'].cpu()[0,0,1], self._current_state['yaw'].cpu()[0,0,0])), velocity=StateVector2D(self._current_state['vel'].cpu()[0,0,0], self._current_state['vel'].cpu()[0,0,1]))
            self._current_tire_steering_angle = self._current_state['steering_angle'].cpu()

        self.update_timepoint()
    

    def update_timepoint(self):
        self._current_timepoint = self._current_timepoint + self._interval_timepoint
        self._next_timepoint = self._next_timepoint + self._interval_timepoint
        if self._next_timepoint > self._end_timepoint:
            self._trajectory_ended = True

    def store_actions(self, idx, file_name):
        '''
        Store the estimated actions in storing_path
        '''
        actions = [[self._throttles[step].cpu().detach().numpy()[idx,:], self._steers[step].cpu().detach().numpy()[idx,:]] for step in range(self._horizon)]
        actions_array = np.array(actions)
        np.save(os.path.join(self._storing_path, f'{file_name}_{idx}.npy'), actions_array)

    def recover_actions(self, idx, file_name):
        '''
        Recover the stored actions from storing_path
        '''
        actions_file = os.path.join(self._storing_path, f'{file_name}_{idx}.npy')
        print(actions_file)
        if os.path.exists(actions_file):
            actions = np.load(actions_file)
            return True, actions
        
        return False, None


    def extract_actions_only_controller(self):
        '''
        applying the controller over all the steps of the trajectory to determine the actions
        '''

        idx = self._idx

        # check if these actions have already been calculated and saved
        saved_actions, actions = self.recover_actions(idx, 'only_controller')
        actions = np.array(actions)
        if saved_actions:
            for step in range(self._horizon):
                with torch.no_grad():
                    self._throttles[step][idx,:] = torch.tensor(actions[step,0], dtype=torch.float64, device=device)
                    self._steers[step][idx,:] = torch.tensor(actions[step,1], dtype=torch.float64, device=device)
            return

        # wether the scenario is finished or not for the current agent
        self._trajectory_ended = False
        step = 0
        while not self._trajectory_ended:
            self.call_controller(step)
            step += 1
        self.store_actions(idx, 'only_controller')
    

    def extract_actions_only_controller_step(self, step):
        '''
        applying the controller over the 'step' step of the trajectory
        '''

        idx = self._idx

        # check if these actions have already been calculated and saved
        saved_actions, actions = self.recover_actions(idx, 'only_controller')
        actions = np.array(actions)
        if saved_actions:
            with torch.no_grad():
                self._throttles[step][idx,:] = torch.tensor(actions[step,0], dtype=torch.float64, device=device)
                self._steers[step][idx,:] = torch.tensor(actions[step,1], dtype=torch.float64, device=device)
            return

        # wether the scenario is finished or not for the current agent
        self._trajectory_ended = False
        step_counter = 0
        while not self._trajectory_ended and step_counter <= step:
            self.call_controller(step)
            step_counter += 1
    


    # 2. FUNCTIONS TO OPTIMIZE THE ACTIONS OF A SINGLE TRAJECTORY
    def immediate_action_optimize(self, counter_step, previous_state, goal_waypoint):
        '''
        optimizes the actions at a given step, starting from an initialization already stored in self._throttles[step][idx,:] and self._steers[step][idx,:]
        '''

        idx = self._idx
        previous_state = self.get_optimize_state(previous_state)

        # optimizes the actions immediately afetr the controller

        tensor_waypoints_pos = torch.unsqueeze(torch.unsqueeze(torch.tensor([goal_waypoint.x, goal_waypoint.y], dtype=torch.float64), dim=0), dim=0).to(device)
        tensor_waypoints_yaw = torch.unsqueeze(torch.unsqueeze(torch.tensor([goal_waypoint.heading], dtype=torch.float64), dim=0), dim=0).to(device)
        tensor_waypoints_vel_x = torch.unsqueeze(torch.tensor([goal_waypoint.velocity.x], dtype=torch.float64), dim=0).to(device)
        tensor_waypoints_vel_y = torch.unsqueeze(torch.tensor([goal_waypoint.velocity.y], dtype=torch.float64), dim=0).to(device)

        throttle_param = self._throttles[counter_step][idx,:].detach().requires_grad_(True).to(device)
        steer_param = self._steers[counter_step][idx,:].detach().requires_grad_(True).to(device)

        optimizer_throttle = torch.optim.Adam([throttle_param], lr=0.005)
        optimizer_steer = torch.optim.Adam([steer_param], lr=0.005)
        
        # scheduler_throttle = ReduceLROnPlateau(optimizer_throttle, mode='min', factor=0.5, patience=1000, verbose=False)
        # scheduler_steer = ReduceLROnPlateau(optimizer_steer, mode='min', factor=0.5, patience=1000, verbose=False)
            
        loss = torch.nn.MSELoss()

        # def detached_clone_of_dict(current_state):
        #     new_state = {key:value.detach().clone().to(device) for key,value in current_state.items()}
        #     return new_state
        

        opt_idx = 0
        current_loss = 5
        loss_yaw = torch.tensor([1])
        loss_pos, loss_speed = torch.tensor([20]), torch.tensor([20])
        while (loss_speed > 0.1 or loss_pos > 0.1) or loss_yaw > 1e-6:
           
            predicted_state = self._motion_model.forward_reconstruction(previous_state, self.get_actions(throttle_param, steer_param))
           


            loss_yaw = loss(torch.tan(predicted_state['yaw']/2.), torch.tan(tensor_waypoints_yaw/2.))/torch.abs(torch.tan(tensor_waypoints_yaw/2.))
            loss_pos = loss(predicted_state['pos'], tensor_waypoints_pos)
            loss_speed = loss(predicted_state['vel'][:,:,0], tensor_waypoints_vel_x) + loss(predicted_state['vel'][:,:,1], tensor_waypoints_vel_y)
            current_loss = loss_yaw + loss_pos + loss_speed
            # current_loss = loss_yaw + loss_pos
            # current_loss = loss_pos
           

            # to penalize for high steering and throttling
            # TODO: find the good weight for throttle_param and steer_param
            (current_loss+throttle_param**2*0.01+steer_param**2).backward()
            # current_loss.backward()
            optimizer_throttle.step()
            optimizer_steer.step()


            if opt_idx%300==0:
                print()
                print(current_loss.item(), '   ', loss_pos.item(), '   ', loss_yaw.item(), '   ', loss_speed.item())
                print()
                print(counter_step, ' ********** ', predicted_state['pos'].detach().cpu().numpy(), '   *****     ', predicted_state['yaw'].detach().cpu().numpy(), '  ****  ', predicted_state['speed'].detach().cpu().numpy())
                print(counter_step, ' ********** ', tensor_waypoints_pos.cpu().numpy(), '   ****     ', tensor_waypoints_yaw.cpu().numpy(), '  ****  ', np.hypot(goal_waypoint.velocity.x, goal_waypoint.velocity.y))
                
            opt_idx += 1

            optimizer_steer.zero_grad()
            optimizer_throttle.zero_grad()
            previous_state = self.get_optimize_state(previous_state)

            # self._optimization_rounds[idx] += opt_idx
            if opt_idx > 1000:
                
                # self._position_loss[idx] +=  loss_pos
                # self._heading_loss[idx] += loss_yaw
                # self._speed_loss[idx] += loss_speed

                with torch.no_grad():
                    self._throttles[counter_step][idx], self._steers[counter_step][idx] = torch.tensor([throttle_param], dtype=torch.float64).to(device), torch.tensor([steer_param], dtype=torch.float64).to(device)
                    print()
                    print(current_loss.item(), '   ', loss_pos.item(), '   ', loss_yaw.item(), '   ', loss_speed.item())
                    print()
                    print(counter_step, ' ********** ', predicted_state['pos'].detach().cpu().numpy(), '   *****     ', predicted_state['yaw'].detach().cpu().numpy(), '  ****  ', predicted_state['speed'].detach().cpu().numpy())
                    print(counter_step, ' ********** ', tensor_waypoints_pos.cpu().numpy(), '   ****     ', tensor_waypoints_yaw.cpu().numpy(), '  ****  ', np.hypot(goal_waypoint.velocity.x, goal_waypoint.velocity.y))
                    


                return False, loss_pos
            # if opt_idx==1:
            #     self._init_position_loss[idx] += loss_pos
            #     self._init_heading_loss[idx] += loss_yaw
            #     self._init_speed_loss[idx] += loss_speed
                
        
        with torch.no_grad():
            self._throttles[counter_step][idx], self._steers[counter_step][idx] = torch.tensor([throttle_param], dtype=torch.float64).to(device), torch.tensor([steer_param], dtype=torch.float64).to(device)

            print()
            print(current_loss.item(), '   ', loss_pos.item(), '   ', loss_yaw.item(), '   ', loss_speed.item())
            print()
            print(counter_step, ' ********** ', predicted_state['pos'].detach().cpu().numpy(), '   *****     ', predicted_state['yaw'].detach().cpu().numpy(), '  ****  ', predicted_state['speed'].detach().cpu().numpy())
            print(counter_step, ' ********** ', tensor_waypoints_pos.cpu().numpy(), '   ****     ', tensor_waypoints_yaw.cpu().numpy(), '  ****  ', np.hypot(goal_waypoint.velocity.x, goal_waypoint.velocity.y))
            

        # self._position_loss[idx] += loss_pos
        # self._heading_loss[idx] += loss_yaw
        # self._speed_loss[idx] += loss_speed 
                
      
        return True, loss_pos
    
    

    def individual_step_by_step_optimization(self, initialization_type: Optional[Literal['initial_controller', 'controller', 'last_optimal']] = 'controller'):
        '''
        optimizes the actions of the idx-th agents using a step by step strategy:
        after every step of enrolling the bm, the actions taken in the previous step are optimized.

        what should be the initialization fo the actions? 
        1. initialize using the actions from the tracker on all the steps
        2. call the controller at each time step and optimize its given actions
        3. initialize the actions at step 't+1' by the optimized actions at step 't' (when t==0, we use the tracker)
        '''
        idx = self._idx
        if initialization_type=='last_optimal':

            # check if these actions have already been calculated and saved
            saved_actions, actions = self.recover_actions(idx, 'last_optimal')
            actions = np.array(actions)
            if saved_actions:
                for step in range(actions.shape[0]):
                    with torch.no_grad():
                        self._throttles[step][idx,:] = torch.tensor(actions[step,0], dtype=torch.float64, device=device)
                        self._steers[step][idx,:] = torch.tensor(actions[step,1], dtype=torch.float64, device=device)
                return

            
            self.extract_actions_only_controller_step(0)
            # optimize the initialized actions step by step using immediate_action_optimize
            for step in range(len(self._waypoints)-1):
                if step > 0:
                    with torch.no_grad():
                        self._throttles[step][idx], self._steers[step][idx] = self._throttles[step-1][idx], self._steers[step-1][idx]
                
                self.immediate_action_optimize(step, self._current_state, self._waypoints[step+1])

                with torch.no_grad():
                    self._current_state = self._motion_model.forward_reconstruction(self._current_state, self.get_adv_actions_temp(step, id=idx))
            

            self.store_actions(idx, 'last_optimal')

             

        if initialization_type=='initial_controller':
            # check if these actions have already been calculated and saved
            saved_actions, actions = self.recover_actions(idx, 'initial_controller')
            actions = np.array(actions)
            if saved_actions:
                for step in range(actions.shape[0]):
                    with torch.no_grad():
                        self._throttles[step][idx,:] = torch.tensor(actions[step,0], dtype=torch.float64, device=device)
                        self._steers[step][idx,:] = torch.tensor(actions[step,1], dtype=torch.float64, device=device)
                return

            # the controler is only applied on the initial states and not on the optimized ones
            self.extract_actions_only_controller()
            # optimize the initialized actions step by step using immediate_action_optimize
            for step in range(len(self._waypoints)-1):
                self.immediate_action_optimize(step, self._current_state, self._waypoints[step+1])

                with torch.no_grad():
                    self._current_state = self._motion_model.forward_reconstruction(self._current_state, self.get_adv_actions_temp(step, id=idx))
            

            self.store_actions(idx, 'initial_controller')

        if initialization_type=='controller':
            # check if these actions have already been calculated and saved
            saved_actions, actions = self.recover_actions(idx, 'controller')
            actions = np.array(actions)
            
            if saved_actions:
                for step in range(actions.shape[0]):
                    with torch.no_grad():
                        self._throttles[step][idx,:] = torch.tensor(actions[step,0], dtype=torch.float64, device=device)
                        self._steers[step][idx,:] = torch.tensor(actions[step,1], dtype=torch.float64, device=device)
                return
            


            # optimize the initialized actions step by step using immediate_action_optimize
            # for step in range(len(self._waypoints)-1):
            step = 0
            while not self._trajectory_ended:
                throttle, steer = self._controller.track_trajectory(self._current_timepoint, self._next_timepoint, self._current_waypoint, self._trajectory, initial_steering_angle=self._current_tire_steering_angle)

                with torch.no_grad():
                    self._throttles[step][idx,:] = throttle
                    self._steers[step][idx, :] = steer

                self.immediate_action_optimize(step, self._current_state, self._waypoints[step+1])

                with torch.no_grad():
                    self._current_state = self._motion_model.forward_reconstruction(self._current_state, self.get_adv_actions_temp(step, id=idx))

                    self._current_waypoint = Waypoint(self._next_timepoint, oriented_box=OrientedBox.from_new_pose(self._agent_box, StateSE2(self._current_state['pos'].cpu()[0,0,0], self._current_state['pos'].cpu()[0,0,1], self._current_state['yaw'].cpu()[0,0,0])), velocity=StateVector2D(self._current_state['vel'].cpu()[0,0,0], self._current_state['vel'].cpu()[0,0,1]))
                    self._current_tire_steering_angle = self._current_state['steering_angle'].cpu()
                
                self.update_timepoint()
                step += 1

            self.store_actions(idx, 'controller')


    def initialize_optimization(self, actions_mask, positions, headings, velocities):
        self._actions_mask = actions_mask
        self._positions = positions
        self._headings = headings
        self._velocities = velocities

        self._num_agents = positions.size()[0]

    


    # 3. FUNCTIONS TO OPTIMIZE THE ACTIONS OF ALL THE TRAJECTORIES TOGETHER
    def parallel_step_by_step_optimization(self, initial_state):
        '''
        optimizes the actions of all the agents using a step by step strategy:
        after every step of enrolling the bm, the actions taken in the previous step are optimized.
        '''
        reloaded_actions = False
        for idx in range(self._num_agents):
            saved_actions, actions = self.recover_actions(idx, 'parallel_step_optimized')
            actions = np.array(actions)
            if saved_actions:
                reloaded_actions = True
                for step in range(actions.shape[0]):
                    with torch.no_grad():
                        self._throttles[step][idx,:] = torch.tensor(actions[step,0], dtype=torch.float64, device=device)
                        self._steers[step][idx,:] = torch.tensor(actions[step,1], dtype=torch.float64, device=device)

        if reloaded_actions:
            return

        loss = torch.nn.MSELoss(reduction='sum')

        optimizers_throttle, optimizers_steer = [], []
        for  step in range(self._horizon):

            optimizers_throttle.append(torch.optim.Adam([self._throttles[step]], lr=0.005))
            optimizers_steer.append(torch.optim.Adam([self._steers[step]], lr=0.005))


        current_state = self.get_optimize_state(initial_state)
        # with torch.autograd.profiler.profile(use_cuda=True, with_stack=True, profile_memory=True) as prof:
        for step in range(self._horizon):

            cp_current_state = self.get_optimize_state(current_state)
            for opt_step in range(2000):

                # pass the agents forward using the bm
                # with profiler.record_function("BM forward"):
                next_state = self._motion_model.forward_reconstruction(cp_current_state, self.get_adv_actions_temp(step))

                # compute the cost over the agents in this state
                # with profiler.record_function("Loss computation"):
                position_loss = loss(self._positions[:,step+1,:]*self._actions_mask[:,step].unsqueeze(1), next_state['pos'][0, ...]*self._actions_mask[:,step].unsqueeze(1))

                heading_loss = loss(torch.tan(self._headings[:,step+1,:]/2.0)*self._actions_mask[:,step].unsqueeze(1), torch.tan(next_state['yaw'][0, ...]/2.0)*self._actions_mask[:,step].unsqueeze(1))

                vel_loss = loss(self._velocities[:,step+1, 0:1]*self._actions_mask[:,step].unsqueeze(1), next_state['vel'][0,:,0:1]*self._actions_mask[:,step].unsqueeze(1)) + loss(self._velocities[:,step+1, 1:]*self._actions_mask[:,step].unsqueeze(1), next_state['vel'][0,:,1:]*self._actions_mask[:,step].unsqueeze(1))


                overall_loss = position_loss + heading_loss + vel_loss

                overall_loss.backward()
                optimizers_throttle[step].step()
                optimizers_steer[step].step()
                optimizers_throttle[step].zero_grad()
                optimizers_steer[step].zero_grad()
                print(f' step {step}, opt step {opt_step} : {overall_loss}')
                cp_current_state = self.get_optimize_state(current_state)
                


            with torch.no_grad():
                next_state = self._motion_model.forward_reconstruction(current_state, self.get_adv_actions_temp(step))

            # contniuing the process
            current_state = self.get_optimize_state(next_state)


        for  idx in range(self._num_agents):
            self.store_actions(idx, 'parallel_step_optimized')



    def parallel_all_actions_optimization(self, initial_state):


        '''
        optimizing the actions of all the agents through the entire trajectories
        '''
        

        reloaded_actions = False
        for idx in range(self._num_agents):
            saved_actions, actions = self.recover_actions(idx, 'parallel_all_optimized')
            actions = np.array(actions)
            if saved_actions:
                reloaded_actions = True
                for step in range(actions.shape[0]):
                    with torch.no_grad():
                        self._throttles[step][idx,:] = torch.tensor(actions[step,0], dtype=torch.float64, device=device)
                        self._steers[step][idx,:] = torch.tensor(actions[step,1], dtype=torch.float64, device=device)
        

        if reloaded_actions:
            return

        loss = torch.nn.MSELoss(reduction='sum')
        
        optimizer_throttle = torch.optim.Adam(self._throttles, lr=0.0005)
        optimizer_steer = torch.optim.Adam(self._steers, lr=0.0001)

        # start_time = perf_counter()

        for opt_step in range(2000):
            current_state = self.get_optimize_state(initial_state)
            position_loss, heading_loss, vel_loss = [], [], []
            # with torch.autograd.profiler.profile(use_cuda=True, with_stack=True, profile_memory=True) as prof:
            for step in range(self._horizon):

                # pass the agents forward using the bm
                # with profiler.record_function("BM forward"):
                next_state = self._motion_model.forward_reconstruction(current_state, self.get_adv_actions_temp(step))

                # compute the cost over the agents in this state
                # with profiler.record_function("Loss computation"):
                position_loss.append(loss(self._positions[:,step+1,:]*self._actions_mask[:,step].unsqueeze(1), next_state['pos'][0, ...]*self._actions_mask[:,step].unsqueeze(1)))

                heading_loss.append(loss(torch.tan(self._headings[:,step+1,:]/2.0)*self._actions_mask[:,step].unsqueeze(1), torch.tan(next_state['yaw'][0, ...]/2.0)*self._actions_mask[:,step].unsqueeze(1)))

                vel_loss.append( loss(self._velocities[:,step+1, 0:1]*self._actions_mask[:,step].unsqueeze(1), next_state['vel'][0,:,0:1]*self._actions_mask[:,step].unsqueeze(1)) + 
                                loss(self._velocities[:,step+1, 1:]*self._actions_mask[:,step].unsqueeze(1), next_state['vel'][0,:,1:]*self._actions_mask[:,step].unsqueeze(1)))


                # contniuing the process
                current_state = next_state

            # backpropagating the loss through all the actions
            # with profiler.record_function("Backprop"):
            print(position_loss[0].size())
            overall_loss = torch.sum(torch.stack(position_loss)) + torch.sum(torch.stack(heading_loss)) + torch.sum(torch.stack(vel_loss))
            print('the device is ', overall_loss.device)
            overall_loss.backward()
            optimizer_throttle.step()
            optimizer_steer.step()
            optimizer_throttle.zero_grad()
            optimizer_steer.zero_grad()


            print(f' step {opt_step} : {overall_loss}')
        

        # parallel_opt_time = perf_counter() - start_time
        # create_directory_if_not_exists(f'/home/{self._username}/workspace/Recontruction')        
        # create_directory_if_not_exists(f'/home/{self._username}/workspace/Recontruction/{self._simulation.scenario.scenario_name}')
        # create_directory_if_not_exists(f'/home/{self._username}/workspace/Recontruction/{self._simulation.scenario.scenario_name}/{self._experiment_name}')
        # file_path = f'/home/{self._username}/workspace/Recontruction/{self._simulation.scenario.scenario_name}/{self._experiment_name}/traj_opt.csv'
        # with open(file_path, mode='w', newline='') as file:
        #     writer = csv.writer(file)
        #     writer.writerow([parallel_opt_time])
            

        for  idx in range(self._num_agents):
            self.store_actions(idx, 'parallel_all_optimized')

    

    # FUNCTIONS TO REPORT THE SCORES

    def reset_error_losses(self):
        self._position_error = []
        self._heading_error = []
        self._velocity_error = []

    def write_losses(self, file_name:str, optimization_time):
        
        write_path = os.path.join(self._storing_path, f'{file_name}.csv')
        with open(write_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            for idx in range(len(self._position_error)):
                writer.writerow([self._position_error[idx].item(), self._heading_error[idx].item(), self._velocity_error[idx].item()])
            
            writer.writerow([optimization_time])



    def report(self, idx: int, n_steps: int, initial_state):
        '''
        for a single agent:
        enrolls the bm with the current actions and computes the loss in
            1. position
            2. heading
            3. velocity
        '''

        loss = torch.nn.MSELoss()
        position_error, heading_error, vel_error = [], [], []
        current_state = self.get_optimize_state(initial_state)
        for step in range(n_steps): # till the last step of the trajectory
            
            with torch.no_grad():
                current_state = self._motion_model.forward_reconstruction(current_state, self.get_adv_actions_temp(step, idx))

            position_error.append(loss(self._positions[idx,step+1,:], current_state['pos'][0, 0, ...]))

            heading_error.append(loss(self._headings[idx,step+1,:], current_state['yaw'][0, 0,...]))

            vel_error.append( loss(self._velocities[idx,step+1, 0:1], current_state['vel'][0,0,0:1]) + 
                             loss(self._velocities[idx,step+1, 1:], current_state['vel'][0,0,1:]))
        

        if n_steps>0:
            position_error = torch.sum(torch.stack(position_error))
            heading_error = torch.sum(torch.stack(heading_error))
            vel_error = torch.sum(torch.stack(vel_error))
            print('this is pos error ', position_error.item())

            self._position_error.append(position_error)
            self._heading_error.append(heading_error)
            self._velocity_error.append(vel_error)


    def report_all(self, initial_state, method_name: str):
        '''
        writes a single row: the overall losses for all the agents accumulated
        '''

        loss = torch.nn.MSELoss()
        position_error, heading_error, vel_error = [], [], []
        current_state = initial_state
        for step in range(self._horizon): # till the last step of the trajectory
            
            with torch.no_grad():
                current_state = self._motion_model.forward_reconstruction(current_state, self.get_adv_actions_temp(step))

            position_error.append(loss(self._positions[:,step+1,:]*self._actions_mask[:,step].unsqueeze(1), current_state['pos'][0, ...]*self._actions_mask[:,step].unsqueeze(1)))

            heading_error.append(loss(self._headings[:,step+1,:]*self._actions_mask[:,step].unsqueeze(1), current_state['yaw'][0, ...]*self._actions_mask[:,step].unsqueeze(1)))

            vel_error.append(loss(self._velocities[:,step+1, 0:1]*self._actions_mask[:,step].unsqueeze(1), current_state['vel'][0,:,0:1]*self._actions_mask[:,step].unsqueeze(1)) + 
                            loss(self._velocities[:,step+1, 1:]*self._actions_mask[:,step].unsqueeze(1), current_state['vel'][0,:,1:]*self._actions_mask[:,step].unsqueeze(1)))

        

        position_error = torch.sum(torch.stack(position_error))
        heading_error = torch.sum(torch.stack(heading_error))
        vel_error = torch.sum(torch.stack(vel_error))




        write_path = os.path.join(self._storing_path, f'{method_name}.csv')
        with open(write_path, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([position_error, heading_error, vel_error])
