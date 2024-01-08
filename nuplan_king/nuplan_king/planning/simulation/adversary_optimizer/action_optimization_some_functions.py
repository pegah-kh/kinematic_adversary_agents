



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
        inner_optimization_rounds = 100

        error_reductions = []
        movement_extents = []

        err_before_opt = []
        err_after_opt = []
        err_after_filtered_opt = []

        filtering = False

        
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
        loss_decrease_agents = [[] for _ in range(self._number_agents)]
        for idx, tracked_agent in enumerate(self._agents):

            
            target_throttle = [self._throttle_temp[t][idx,:].detach().requires_grad_(True) for t in range(self._horizon)]
            target_steer = [self._steer_temp[t][idx,:].detach().requires_grad_(True) for t in range(self._horizon)]


            # optimizers_throttle = [torch.optim.RMSprop([throttle_param], lr=0.01) for throttle_param in target_throttle]
            # optimizers_steer = [torch.optim.RMSprop([steer_param], lr=0.01) for steer_param in target_steer]

            
            optimizers_throttle = [torch.optim.Adam([throttle_param], lr=0.0005) for throttle_param in target_throttle]
            optimizers_steer = [torch.optim.Adam([steer_param], lr=0.0001) for steer_param in target_steer]
            schedulers_throttle = [ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=False) for optimizer in optimizers_throttle]
            schedulers_steer = [ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=False) for optimizer in optimizers_steer]
            loss = torch.nn.MSELoss()

            def get_optimize_actions(iteration: int):
                return {'throttle': torch.unsqueeze(torch.unsqueeze(target_throttle[iteration], dim=0), dim=0),
                        'steer': torch.unsqueeze(torch.unsqueeze(target_steer[iteration], dim=0), dim=0)}
            

            first_state:Dict[str, torch.TensorType] = {'pos': torch.zeros(1, 1, 2, requires_grad=True).to(device=device), 
                                                       'yaw': torch.zeros(1, 1, 1, requires_grad=True).to(device=device), 
                                                       'steering_angle': torch.zeros(1, 1, 1, requires_grad=True).to(device=device), 
                                                       'vel': torch.zeros(1, 1, 2, requires_grad=True).to(device=device), 
                                                       'accel': torch.zeros(1, 1, 1, requires_grad=True).to(device=device), 
                                                       'speed': torch.zeros(1, 1, 1, requires_grad=True).to(device=device)}


            tensor_waypoints_pos = [torch.unsqueeze(torch.unsqueeze(torch.tensor([waypoint.x, waypoint.y], dtype=torch.float64), dim=0), dim=0).to(device) for waypoint in self._map_valid_waypoints[idx][0][1:]]
            tensor_waypoints_yaw = [torch.unsqueeze(torch.unsqueeze(torch.tensor([waypoint.heading], dtype=torch.float64), dim=0), dim=0).to(device) for waypoint in self._map_valid_waypoints[idx][0][1:]]
            waypoints_pos = np.array([[waypoint.x, waypoint.y] for waypoint in self._map_valid_waypoints[idx][0]]).reshape(len(self._map_valid_waypoints[idx][0]),2)
            
            # if float(compute_total_distance(waypoints_pos))<400 and filtering:
            #     continue
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

                traj_agents_controller[idx].append([first_state['pos'].cpu().detach().numpy()[0,0,0], first_state['pos'].cpu().detach().numpy()[0,0,1]])
                for _time_step in range(traj_len):

                    for _ in range(inner_optimization_rounds):

                        next_state = self._motion_model.forward(get_optimize_state(first_state), get_optimize_actions(iteration=_time_step), track_token=tracked_agent.track_token, iter=_time_step, plotter=False)
                        # computing the MSE loss between the position component of the state and tensor_waypoints

                        current_loss = loss(next_state['pos'], tensor_waypoints_pos[_time_step])
                        
                        accumulated_loss += current_loss.detach().cpu().numpy()
                        current_loss.backward(retain_graph=True)
                        optimizers_throttle[_time_step].step()
                        optimizers_throttle[_time_step].zero_grad()
                        # schedulers_throttle[_time_step].step(current_loss)
                        # schedulers_steer[_time_step].step(current_loss)

                        current_loss = loss(next_state['yaw'], tensor_waypoints_yaw[_time_step])
                        
                        accumulated_loss += current_loss.detach().cpu().numpy()
                        current_loss.backward()
                        optimizers_steer[_time_step].step()
                        optimizers_steer[_time_step].zero_grad()
                        

                    # the first_state should now be updated based on the optimized actions, using the bicycle model again but without any gradients
                    with torch.no_grad():
                        next_state = self._motion_model.forward(get_optimize_state(first_state), get_optimize_actions(iteration=_time_step), track_token=tracked_agent.track_token, iter=_time_step, plotter=False)

                    first_state = {key:value.detach().requires_grad_(True) for key,value in next_state.items()}
                    traj_agents_controller[idx].append([first_state['pos'].cpu().detach().numpy()[0,0,0], first_state['pos'].cpu().detach().numpy()[0,0,1]])


                # last_optimized_time_step = -1
                # bm_steps_before_opt = 1000
                # _time_step = 0
                # dummy_counter = 0
                # while _time_step<traj_len:
                #     bm_step = 0
                #     cp_time_step = _time_step
                #     cp_first_state = detached_clone_of_dict(first_state)
                #     current_loss = 0
                #     first_state = get_optimize_state(first_state)
                #     losses_pos = []
                #     losses_yaw = []
                    
                #     while bm_step<bm_steps_before_opt and _time_step<traj_len:
                #         # traj_agents_controller[idx].append([first_state['pos'].cpu().detach().numpy()[0,0,0], first_state['pos'].cpu().detach().numpy()[0,0,1]])
                #         next_state = self._motion_model.forward(first_state, get_optimize_actions(iteration=_time_step), track_token=tracked_agent.track_token, iter=_time_step, plotter=False)
                #         # computing the MSE loss between the position component of the state and tensor_waypoints
                #         current_loss_pos = loss(next_state['pos'], tensor_waypoints_pos[_time_step])
                #         current_loss_yaw = loss(next_state['yaw'], tensor_waypoints_yaw[_time_step])
                #         # print('compare ', next_state['pos'], '\n', tensor_waypoints_pos[_time_step], '\n', current_loss)
                #         losses_pos.append(current_loss_pos)
                #         losses_yaw.append(current_loss_yaw)
                #         accumulated_loss += current_loss_pos.clone().detach().cpu().numpy() + current_loss_yaw.clone().detach().cpu().numpy()
                #         # first_state = next_state
                #         first_state = clone_of_dict(next_state)

                #         _time_step += 1
                #         bm_step += 1
                    

                #     current_loss_pos = torch.sum(torch.stack(losses_pos))
                #     current_loss_yaw = torch.sum(torch.stack(losses_yaw))
                #     print('this is the time step ', _time_step, ' and round ', n_round)
                #     current_loss_pos.backward(retain_graph=True)

                #     for optimize_indice in range(last_optimized_time_step+1, _time_step):
                        
                #         optimizers_throttle[optimize_indice].step()
                #         # optimizers_steer[optimize_indice].step()

                #         # print('the gradient value for that time step ', target_throttle[optimize_indice].grad)
                #         schedulers_throttle[optimize_indice].step(current_loss)
                #         # schedulers_steer[optimize_indice].step(current_loss)
                #         optimizers_throttle[optimize_indice].zero_grad()
                #         # optimizers_steer[optimize_indice].zero_grad()


                #         dummy_counter += 1

                #     # if we want to optimize the steer and the throttle separately
                #     current_loss_yaw.backward()
                #     for optimize_indice in range(last_optimized_time_step+1, _time_step):
                        
                #         optimizers_steer[optimize_indice].step()

                #         # print('the gradient value for that time step ', target_throttle[optimize_indice].grad)
                #         schedulers_steer[optimize_indice].step(current_loss)
                #         optimizers_steer[optimize_indice].zero_grad()
                        
                    

                #     # the first_state should now be updated based on the optimized actions, using the bicycle model again but without any gradients
                #     with torch.no_grad():
                #         bm_step = 0
                #         while bm_step<bm_steps_before_opt and cp_time_step<traj_len:

                #             traj_agents_controller[idx].append([cp_first_state['pos'].cpu().detach().numpy()[0,0,0], cp_first_state['pos'].cpu().detach().numpy()[0,0,1]])

                #             next_state = self._motion_model.forward(get_optimize_state(cp_first_state), get_optimize_actions(iteration=cp_time_step), track_token=tracked_agent.track_token, iter=cp_time_step, plotter=False)
                #             cp_first_state = next_state

                #             cp_time_step += 1
                #             bm_step += 1


                #     last_optimized_time_step = _time_step-1
                #     first_state = {key:value.detach().requires_grad_(True) for key,value in next_state.items()}
                # traj_agents_controller[idx].append([first_state['pos'].cpu().detach().numpy()[0,0,0], first_state['pos'].cpu().detach().numpy()[0,0,1]])


                print('this is is the size ', np.array(traj_agents_controller[idx]).shape)
                _difference = np.linalg.norm(np.array(traj_agents_controller[idx]) - np.array(self.traj_agents_logged[idx]), axis=-1)
                self.difference[idx].append(_difference)
            

                print('********************************************************************')
                print('*************** this is the accumulated loss ', n_round, '  ', accumulated_loss)
                for _time_step in range(traj_len):
                    
                    print('hi1 ', traj_agents_controller[idx][_time_step])
                    print('hi2 ', self.traj_agents_logged[idx][_time_step])
                
                if n_round==0:
                    last_accumulated_loss = accumulated_loss + 20
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
                    
                if np.abs(accumulated_loss-last_accumulated_loss) < 10:
                    break
                last_accumulated_loss = accumulated_loss
                loss_decrease_agents[idx].append(accumulated_loss)


            # just replace the real actions with the improved ones
            with torch.no_grad():
                for _time_step in range(traj_len):
                    
                    print('hi1 ', traj_agents_controller[idx][_time_step])
                    print('hi2 ', self.traj_agents_logged[idx][_time_step])
                    self._throttle_temp[_time_step][idx], self._steer_temp[_time_step][idx] = target_throttle[_time_step].detach().to(device), target_steer[_time_step].detach().to(device)

            
            # plotting the logged and the controlled trajectories
            create_directory_if_not_exists(f'/home/{username}/workspace/EXPERIMENTS')
            create_directory_if_not_exists(f'/home/{username}/workspace/EXPERIMENTS/{self._simulation.scenario.scenario_name}')
            create_directory_if_not_exists(f'/home/{username}/workspace/EXPERIMENTS/{self._simulation.scenario.scenario_name}/{self._experiment_name}')
            create_directory_if_not_exists(f'/home/{username}/workspace/EXPERIMENTS/{self._simulation.scenario.scenario_name}/{self._experiment_name}/Controller')
            for plot_idx, plot_difference in enumerate(self.difference[idx]):
                plt.plot(plot_difference, label=plot_idx)

            plt.legend()
            plt.gcf()
            plt.savefig(f'/home/{username}/workspace/EXPERIMENTS/{self._simulation.scenario.scenario_name}/{self._experiment_name}/Controller/MSE_{idx}.png')
            plt.show()
            plt.clf()



                

        create_directory_if_not_exists('/home/{username}/workspace/DENSE')
        create_directory_if_not_exists(f'/home/{username}/workspace/DENSE/{self._simulation.scenario.scenario_name}')
        create_directory_if_not_exists(f'/home/{username}/workspace/DENSE/{self._simulation.scenario.scenario_name}/{self._experiment_name}')
        create_directory_if_not_exists(f'/home/{username}/workspace/DENSE/{self._simulation.scenario.scenario_name}/{self._experiment_name}/Controller')

        for i_agent in range(self._number_agents):
            plt.plot(loss_decrease_agents[i_agent], label=i_agent)
        plt.legend()
        plt.gcf()
        plt.savefig(f'/home/{username}/workspace/EXPERIMENTS/{self._simulation.scenario.scenario_name}/{self._experiment_name}/Controller/loss_decrease.png')
        plt.show()
        plt.clf()


        # plt.figure(figsize=(10, 10))
        # plt.scatter(movement_extents, error_reductions)
        # plt.xlabel('Movement Extents')
        # plt.ylabel('Error Reductions')
        
        # plt.gcf()
        # plt.savefig(f'/home/{username}/workspace/DENSE/{self._simulation.scenario.scenario_name}/{self._experiment_name}/err_red_vs_ext.png')
        # plt.show()
        # plt.clf()

        # Err reduction vs the movement extent
        # dirstribution of the error before and after the optimization

        # df = pd.DataFrame({
        #     'Values': err_before_opt + err_after_opt + err_after_filtered_opt,
        #     'Group': ['Before Optimization'] * len(err_before_opt) + ['After Optimization'] * len(err_after_opt) + ['Filtered Optimization'] * len(err_after_filtered_opt)
        # })

        # # Plot kernel density plots
        # plt.figure(figsize=(8, 5))
        # sns.kdeplot(data=df, x='Values', hue='Group', fill=True, common_norm=False, palette="husl")
        # plt.xlim(left=0)
        # plt.title('Kernel Density Plot Before and After Optimization')

        
        # plt.gcf()
        # plt.savefig(f'/home/{username}/workspace/DENSE/{self._simulation.scenario.scenario_name}/{self._experiment_name}/err_dist_before_after.png')
        # plt.show()
        # plt.clf()



    def write_report_optimization(self, init_position_loss, init_heading_loss, init_speed_loss, position_loss, heading_loss, speed_loss, optimization_time, optimization_rounds):
        
        file_path = f'/home/{username}/workspace/Recontruction/{self._simulation.scenario.scenario_name}/{self._experiment_name}/traj_opt.csv'
        with open(file_path, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['init pos err', 'init heading err', 'init speed err', 'pos err', 'heading err', 'speed err', 'opt time', 'opt rounds'])

        with open(file_path, mode='a', newline='') as file:
            writer = csv.writer(file)
            for idx in range(self._number_agents):
                writer.writerow([init_position_loss[idx].item(), init_heading_loss[idx].item(), init_speed_loss[idx].item(), position_loss[idx].item(), heading_loss[idx].item(), speed_loss[idx].item(), optimization_time[idx], optimization_rounds[idx]])



                        

    def immediate_action_optimize(self, idx, counter_step, previous_state, goal_waypoint):

        # optimizes the actions immediately afetr the controller

        tensor_waypoints_pos = torch.unsqueeze(torch.unsqueeze(torch.tensor([goal_waypoint.x, goal_waypoint.y], dtype=torch.float64), dim=0), dim=0).to(device)
        tensor_waypoints_yaw = torch.unsqueeze(torch.unsqueeze(torch.tensor([goal_waypoint.heading], dtype=torch.float64), dim=0), dim=0).to(device)
        tensor_waypoints_vel_x = torch.unsqueeze(torch.tensor([goal_waypoint.velocity.x], dtype=torch.float64), dim=0).to(device)
        tensor_waypoints_vel_y = torch.unsqueeze(torch.tensor([goal_waypoint.velocity.y], dtype=torch.float64), dim=0).to(device)

        throttle_param = self._throttle_temp[counter_step][idx,:].detach().requires_grad_(True).to(device)
        steer_param = self._steer_temp[counter_step][idx,:].detach().requires_grad_(True).to(device)

        optimizer_throttle = torch.optim.Adam([throttle_param], lr=0.005)
        optimizer_steer = torch.optim.Adam([steer_param], lr=0.001)
        
        scheduler_throttle = ReduceLROnPlateau(optimizer_throttle, mode='min', factor=0.5, patience=1000, verbose=False)
        scheduler_steer = ReduceLROnPlateau(optimizer_steer, mode='min', factor=0.5, patience=1000, verbose=False)
            
        loss = torch.nn.MSELoss()

        def get_optimize_actions():
                return {'throttle': torch.unsqueeze(torch.unsqueeze(throttle_param, dim=0), dim=0),
                        'steer': torch.unsqueeze(torch.unsqueeze(steer_param, dim=0), dim=0)}
        def detached_clone_of_dict(current_state):
            new_state = {key:value.detach().clone().to(device) for key,value in current_state.items()}
            return new_state
        

        opt_idx = 0
        # pos_optimized, yaw_optimized, speed_optimized = False, False, False
        current_loss = 5
        loss_yaw = torch.tensor([1])
        # previous_loss = None
        loss_pos, loss_speed = torch.tensor([20]), torch.tensor([20])
        while (loss_speed > 2 or loss_pos > 2) or loss_yaw > 1e-4:
            # just try it on cpu
            # do this again loss_change_threshold = 1e-5
            # maybe there are some points are not really fittable, and it is taking too much time => maybe a global optimization
            # initialize from sth dummy like a straight traj btw the start and the ending points to avoid difficult points

            detached_loss_throttle = 0
            detached_loss_steer = 0
            predicted_state = self._motion_model.forward_all(previous_state, get_optimize_actions())
           
            # if current_loss < 2 and loss_yaw > 1e-4:
            #     loss_yaw.backward()
            #     optimizer_steer.step()
            # else:
            #     current_loss.backward()
            #     optimizer_throttle.step()
            #     optimizer_steer.step()
            # # scheduler_throttle.step(current_loss)
            # # scheduler_steer.step(current_loss)


            # loss_change_threshold = 1e-5

            loss_yaw = loss(torch.tan(predicted_state['yaw']/2.), torch.tan(tensor_waypoints_yaw/2.))
            loss_pos = loss(predicted_state['pos'], tensor_waypoints_pos)
            loss_speed = loss(predicted_state['vel'][:,:,0], tensor_waypoints_vel_x) + loss(predicted_state['vel'][:,:,1], tensor_waypoints_vel_y)
            current_loss = loss_yaw + loss_pos + loss_speed
            # if previous_loss is not None:
            #     print(abs(current_loss - previous_loss))
            # if (previous_loss is not None and abs(previous_loss-current_loss)<loss_change_threshold) or (current_loss < 2 and loss_yaw > 1e-4):
            # if loss_pos < 10 and loss_speed < 10 and loss_yaw > 1e-4:
            #     (loss_yaw+throttle_param**2*0.01+steer_param**2).backward()
            #     optimizer_steer.step()
            #     optimizer_throttle.step()
            # else:
            (current_loss+throttle_param**2*0.01+steer_param**2).backward()
            optimizer_throttle.step()
            optimizer_steer.step()

            # Update the previous_loss for the next iteration
            # previous_loss = current_loss

            previous_state = detached_clone_of_dict(previous_state)

            if opt_idx%100==0:
                print(counter_step, ' ********** ', predicted_state['pos'].detach().cpu().numpy(), '   *****     ', predicted_state['yaw'].detach().cpu().numpy(), '  ****  ', predicted_state['speed'].detach().cpu().numpy())
                print(counter_step, ' ********** ', tensor_waypoints_pos.cpu().numpy(), '   ****     ', tensor_waypoints_yaw.cpu().numpy(), '  ****  ', np.hypot(goal_waypoint.velocity.x, goal_waypoint.velocity.y))

                # print('the loss ', detached_loss_steer, '   ', detached_loss_throttle)
                if current_loss > 2 or loss_yaw > 1e-4:
                    print('the grads ', steer_param._grad.item(), '   ', throttle_param._grad.item())
                    print('the current steer ', steer_param.item(), '  and throttle ', throttle_param.item())
                    print('loss ', loss_yaw.item(), '   ', loss_pos.item(), '  ', loss_speed.item())
                    print(loss_pos.device)
                    print()
                
            opt_idx += 1

            optimizer_steer.zero_grad()
            optimizer_throttle.zero_grad()

            self._optimization_rounds[idx] += opt_idx
            if opt_idx > 200:
                
                print('final losses ')
                print(loss_pos.item())
                print(loss_yaw.item())
                self._position_loss[idx] +=  loss_pos
                self._heading_loss[idx] += loss_yaw
                self._speed_loss[idx] += loss_speed

                with torch.no_grad():
                    self._throttle_temp[counter_step][idx], self._steer_temp[counter_step][idx] = torch.tensor([throttle_param], dtype=torch.float64).to(device), torch.tensor([steer_param], dtype=torch.float64).to(device)
        
                return False, loss_pos
            if opt_idx==1:
                print('first losses ')
                print(loss_pos.item())
                print(loss_yaw.item())
                self._init_position_loss[idx] += loss_pos
                self._init_heading_loss[idx] += loss_yaw
                self._init_speed_loss[idx] += loss_speed


        self._position_loss[idx] += loss_pos
        self._heading_loss[idx] += loss_yaw
        self._speed_loss[idx] += loss_speed 
        print(counter_step, ' ********** ', predicted_state['pos'].detach().cpu().numpy(), '   *****     ', predicted_state['yaw'].detach().cpu().numpy(), '  ****  ', predicted_state['speed'].detach().cpu().numpy())
        print(counter_step, ' ********** ', tensor_waypoints_pos.cpu().numpy(), '   ****     ', tensor_waypoints_yaw.cpu().numpy(), '  ****  ', np.hypot(goal_waypoint.velocity.x, goal_waypoint.velocity.y))

        print('the selected steer and throttle ', steer_param.item(), '  ', throttle_param.item())
        print()

        with torch.no_grad():
            self._throttle_temp[counter_step][idx], self._steer_temp[counter_step][idx] = torch.tensor([throttle_param], dtype=torch.float64).to(device), torch.tensor([steer_param], dtype=torch.float64).to(device)
        
        return True, loss_pos
    

    def optimize_the_straight_path(self, waypoints: List[Waypoint], initial_timepoint: TimePoint, last_timepoint: TimePoint):
        '''
        Having the starting and ending points of a trajectory, and their corresponding time points,
        we intepolate a path between them, and initialize the whole set of actions.
        How to initialize the actions? we use the controller over the points of the interpolated trajectory.

        The actions are then optimized to make the interpolated trajectory closer to the ground truth one.
        '''

        # 1. find the interpolated traj
        return
    def extract_actions_bm_inverse(self, idx: int, waypoints: List[Waypoint], previous_state, starting_timepoint):
        '''
        Find the actions by solving the inverse of the bm
        '''

        traj_len = len(waypoints)
        candidate_shifts_heading = np.array([-2*np.pi, 0, 2*np.pi])

        current_waypoint = waypoints[0]
        current_speed = np.linalg.norm([current_waypoint.velocity.x, current_waypoint.velocity.y])
        current_timepoint = starting_timepoint

        bm_delta_t = self._observation_trajectory_sampling.interval_length

        for step in range(1, traj_len):
            next_waypoint = waypoints[step]
            delta_x, delta_y = (next_waypoint.x - current_waypoint.x), (next_waypoint.y - current_waypoint.y)
            print(delta_x, '   ', delta_y)
            delta_pos = np.linalg.norm([delta_x, delta_y])
            print('delta_pos ', delta_pos)

            next_speed = delta_pos/bm_delta_t
            print('compare the speed ', next_speed, '   ', np.linalg.norm([next_waypoint.velocity.x, next_waypoint.velocity.y]))
            throttle = (next_speed - current_speed)/bm_delta_t
            print('thrott ', throttle)



            print('har har ', (delta_x/bm_delta_t)/next_speed)
            print('har har ', (delta_y/bm_delta_t)/next_speed)
            print('possible heading ', np.arcsin((delta_y/bm_delta_t)/next_speed))
            print('possible heading  ', np.arccos((delta_x/bm_delta_t)/next_speed))
            print('possible heading  ', np.arcsin(-(delta_y/bm_delta_t)/next_speed))
            print('possible heading  ', np.arccos(-(delta_x/bm_delta_t)/next_speed))
            # we cannot be sure that this value is the right one
            next_heading1 = np.arccos((delta_x/bm_delta_t)/next_speed)
            next_heading2 = np.arccos(-(delta_x/bm_delta_t)/next_speed)
            # print('next heading ', next_heading1, '     ', next_heading2, '   ', current_waypoint.heading)
            candidate_next_heading_error = np.concatenate([np.abs(candidate_shifts_heading + next_heading1 - current_waypoint.heading), np.abs(candidate_shifts_heading - next_heading1 - current_waypoint.heading), np.abs(candidate_shifts_heading + next_heading2 - current_waypoint.heading), np.abs(candidate_shifts_heading - next_heading2 - current_waypoint.heading)], axis=0)
            next_heading = candidate_shifts_heading[np.argmin(candidate_next_heading_error)] + next_heading
            # should find the closest next_heading to the last heading
            # candidate_heading_delta = np.abs(candidate_shifts_heading + next_heading - current_waypoint.heading)
            delta_heading = next_heading - current_waypoint.heading
         
            steer = np.arcsin(308.9*delta_heading/(next_speed*bm_delta_t))
            # is the value of the arcsin valid for use?


            with torch.no_grad():
                self._throttle_temp[step-1][idx], self._steer_temp[step-1][idx] = torch.tensor([throttle], dtype=torch.float64).to(device), torch.tensor([steer], dtype=torch.float64).to(device)
                # run the bm and obtain the next point
                predicted_state = self._motion_model.forward_all(previous_state, self.get_adv_actions_temp(step-1, idx))            

            # convert the next state to a way point, but also keep it for the next round input to bm
            current_timepoint += self._time_points[step-1]
            current_waypoint = Waypoint(current_timepoint,  OrientedBox.from_new_pose(self._agents[idx].box, StateSE2(predicted_state['pos'].cpu().numpy()[0,0,0], predicted_state['pos'].cpu().numpy()[0,0,1], predicted_state['yaw'].cpu().numpy()[0,0,0])), StateVector2D(predicted_state['vel'].cpu().numpy()[0,0,0], predicted_state['vel'].cpu().numpy()[0,0,1]))
            previous_state = predicted_state

            print('step and idx ', step, '  ', idx)
            print('gt   ', waypoints[step].x, '  ', waypoints[step].y)
            print('pred ', current_waypoint.x, '  ', current_waypoint.y)



            
    def write_report_optimization(self, init_position_loss, init_heading_loss, init_speed_loss, position_loss, heading_loss, speed_loss, optimization_time, optimization_rounds):
        
        file_path = f'/home/{username}/workspace/Recontruction/{self._simulation.scenario.scenario_name}/{self._experiment_name}/traj_opt.csv'
        with open(file_path, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['init pos err', 'init heading err', 'init speed err', 'pos err', 'heading err', 'speed err', 'opt time', 'opt rounds'])

        with open(file_path, mode='a', newline='') as file:
            writer = csv.writer(file)
            for idx in range(self._number_agents):
                writer.writerow([init_position_loss[idx].item(), init_heading_loss[idx].item(), init_speed_loss[idx].item(), position_loss[idx].item(), heading_loss[idx].item(), speed_loss[idx].item(), optimization_time[idx], optimization_rounds[idx]])

                    


   
    def optimize_actions_parallel(self):


        '''
        optimizing the actions of all the agents through the entire trajectories
        '''

        loss = torch.nn.MSELoss()
        
        optimizer_throttle = torch.optim.Adam(self._throttle_temp, lr=0.005)
        optimizer_steer = torch.optim.Adam(self._steer_temp, lr=0.001)

        start_time = perf_counter()

        for opt_step in range(400):
            self.reset_dynamic_states()
            current_state = self.get_adv_state()
            position_loss, heading_loss, vel_loss = [], [], []
            # with torch.autograd.profiler.profile(use_cuda=True, with_stack=True, profile_memory=True) as prof:
            for step in range(self._horizon):

                # pass the agents forward using the bm
                # with profiler.record_function("BM forward"):
                next_state = self._motion_model.forward_all(current_state, self.get_adv_actions_temp(step))

                # compute the cost over the agents in this state
                # with profiler.record_function("Loss computation"):
                position_loss.append(loss(self._positions[:,step+1,:]*self._action_mask[:,step].unsqueeze(1), next_state['pos'][0, ...]*self._action_mask[:,step].unsqueeze(1)))

                heading_loss.append(loss(torch.tan(self._headings[:,step+1,:]/2.0)*self._action_mask[:,step].unsqueeze(1), torch.tan(next_state['yaw'][0, ...]/2.0)*self._action_mask[:,step].unsqueeze(1)))

                vel_loss.append( loss(self._velocities[:,step+1, 0:1]*self._action_mask[:,step].unsqueeze(1), next_state['vel'][0,:,0:1]*self._action_mask[:,step].unsqueeze(1)) + 
                                loss(self._velocities[:,step+1, 1:]*self._action_mask[:,step].unsqueeze(1), next_state['vel'][0,:,1:]*self._action_mask[:,step].unsqueeze(1)))


                # contniuing the process
                current_state = next_state

            # backpropagating the loss through all the actions
            # with profiler.record_function("Backprop"):
            overall_loss = torch.sum(torch.stack(position_loss)) + torch.sum(torch.stack(heading_loss)) + torch.sum(torch.stack(vel_loss))
            print('device type ', overall_loss.device)
            overall_loss.backward()
            optimizer_throttle.step()
            optimizer_steer.step()
            optimizer_throttle.zero_grad()
            optimizer_steer.zero_grad()

            # print(prof.key_averages().table(sort_by='self_cpu_time_total', row_limit=100))
            # print(prof.key_averages().table(sort_by='self_cpu_time_total', row_limit=100))
            # cpu_time_total


            print(f' step {opt_step} : {overall_loss}')
        

        parallel_opt_time = perf_counter() - start_time
        create_directory_if_not_exists(f'/home/{username}/workspace/Recontruction')        
        create_directory_if_not_exists(f'/home/{username}/workspace/Recontruction/{self._simulation.scenario.scenario_name}')
        create_directory_if_not_exists(f'/home/{username}/workspace/Recontruction/{self._simulation.scenario.scenario_name}/{self._experiment_name}')
        file_path = f'/home/{username}/workspace/Recontruction/{self._simulation.scenario.scenario_name}/{self._experiment_name}/traj_opt.csv'
        with open(file_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([parallel_opt_time])

            

    def trying_multiple_trackers(self, idx, counter_step, previous_state, goal_waypoint, current_timepoint, next_timepoint, transformed_initial_waypoint, transformed_trajectory, transformed_initial_tire_steering_angle):

        if counter_step > 0:
            throttle, steer = self._throttle_temp[counter_step-1][idx].detach(), self._steer_temp[counter_step-1][idx].detach()
            print()
            print()
            print('**** hey   ', idx, '   ', throttle, '   ', steer)
            print()
            print()
            with torch.no_grad():
                self._throttle_temp[counter_step][idx], self._steer_temp[counter_step][idx] = torch.tensor([throttle], dtype=torch.float64).to(device), torch.tensor([steer], dtype=torch.float64).to(device)
                

            optimized, _ = self.immediate_action_optimize(idx, counter_step, previous_state, goal_waypoint)
            # if optimized:
            #     return
            return

        tracker_loss, optimized_throttles, optimized_steers = [], [], []
        for i, tracker in enumerate(self._trackers):

            throttle, steer = tracker.track_trajectory(current_timepoint, next_timepoint, transformed_initial_waypoint, transformed_trajectory, initial_steering_angle=transformed_initial_tire_steering_angle)
            print()
            print()
            print('**** ', i, '   ', idx, '   ', throttle, '   ', steer)
            print()
            print()
            with torch.no_grad():
                self._throttle_temp[counter_step][idx], self._steer_temp[counter_step][idx] = torch.tensor([throttle], dtype=torch.float64).to(device), torch.tensor([steer], dtype=torch.float64).to(device)
                

            optimized, tracker_pos_loss = self.immediate_action_optimize(idx, counter_step, previous_state, goal_waypoint)
            tracker_loss.append(tracker_pos_loss.detach().cpu().numpy())
            optimized_throttles.append(self._throttle_temp[counter_step][idx].detach().cpu().numpy())
            optimized_steers.append(self._steer_temp[counter_step][idx].detach().cpu().numpy())
        

        with torch.no_grad():
            best_tracker_idx = np.argmin(np.array(tracker_loss))
            self._throttle_temp[counter_step][idx], self._steer_temp[counter_step][idx] = torch.tensor([optimized_throttles[best_tracker_idx]], dtype=torch.float64).to(device), torch.tensor([optimized_steers[best_tracker_idx]], dtype=torch.float64).to(device)
                

        
        return
        
