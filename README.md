# Kinematic Adversary Agents


- [Dataset](#nuplan_structure)
- [Environment Setup](#Environment_setup)
- [Dependencies](#nuplan_devkit_garage)
- [Testing Installations](#testing_install)


<a name="nuplan_structure"></a>
## Dataset

### Download the Dataset
Download the mini split and the maps from [NuPlan Website](https://www.nuscenes.org/nuplan).

### Directory Structure
This order of folders works for me.

```plaintext
.
├── nuplan
│   ├── maps
│   │   ├── nuplan-maps-v1.0.json
│   │   └── sg-one-nort
│   │   └── us-ma-boston
│   │   └── ...
│   ├── nuplan-v1.1
│   │   ├── mini
│   │   │   ├── 2021.05.12.22.00.38_veh-35_01008_01518.db
│   │   │   └── 2021.06.23.20.43.31_veh-16_03607_04007.db 
│   │   │   └── ...
```
### Setting the Environment Variables
You should set the path different parts of the dataset in the `.bashrc`. Conveniently, the paths can be added to the `bashrc-docker` file on your Windows distribution, and then coppied to the `.bashrc` (on the cluster) via a copy command in `Dockerfile`; to detail the process:

1. add the path to the variables to the `bashrc-docker`:
```
USERNAME="{Username}"

# Replace placeholders with actual values
export NUPLAN_DEVKIT_ROOT="/home/${USERNAME}/workspace/nuplan-devkit/"
export NUPLAN_DATA_ROOT="/datasets_local/nuplan"
export NUPLAN_MAPS_ROOT="/datasets_local/nuplan/maps"
export NUPLAN_EXP_ROOT="/home/${USERNAME}/workspace/exp"
export NUPLAN_SIMULATION_ALLOW_ANY_BUILDER=1
export HYDRA_FULL_ERROR=1

```

2. add to the copying command to the `Dockerfile`:

```
USERNAME="{Username}"

COPY bashrc_docker.sh /home/${USERNAME}/bashrc_temp_docker.sh
RUN cat /home/${USERNAME}/bashrc_temp_docker.sh >> /home/${USERNAME}/.bashrc && \
    rm /home/${USERNAME}/bashrc_temp_docker.sh
```

<a name="Environment_setup"></a>
## Environment Setup


(from NuPlan Repository)
1. Install Python
```
sudo apt install python-pip
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt-get update
sudo apt-get install python3.9
sudo apt-get install python3.9-dev
```

2. Installing miniconda

```
mkdir -p ~/workspace/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/workspace/miniconda3/miniconda.sh
bash ~/workspace/miniconda3/miniconda.sh -b -u -p ~/workspace/miniconda3
rm -rf ~/workspace/miniconda3/miniconda.sh
```
 and then initialize the miniconda:

```
~/workspace/miniconda3/bin/conda init bash
```

3. Clone this repository:

```
cd ~/workspace/ && git clone https://github.com/pegah-kh/kinematic_adversary_agents.git && cd kinematic_adversary_agents
```

(not polished list of packages...)
4. Creeating `nuplan` Conda Environment
creathe a conda environment called `nuplan` (you do not have to do anything to set the name, it is indicated the `environment.yml` file.):

```
conda env create -f environment.yml
```

To activate the environment:

```
conda activate nuplan 
```

or 

```
conda activate ~/workspace/miniconda3/envs/nuplan 
```




<a name="nuplan_devkit_garage"></a>
## Dependencies

The current repository is dependent on two other projects: *nuplan-devkit* and *tuplan-garage*.
Here we detail the installation process of these two repositories.

Install both of these packages in your workspace (you can install them elsewhere, but the rest of the instructions are assuming the installation in workspace.)

### nuplan-devkit

1. Clone the nuplan-devkit repository
```
cd ~/workspace && git clone https://github.com/motional/nuplan-devkit.git && cd nuplan-devkit
```
2. Change the path to data when using mini version of NuPlan:

Change data paths in `nuplan/planning/script/config/common/scenario_builder/nuplan_mini.yaml` to:
```
data_root: ${oc.env:NUPLAN_DATA_ROOT}/nuplan-v1.1/mini
map_root: ${oc.env:NUPLAN_MAPS_ROOT}
sensor_root: ${oc.env:NUPLAN_DATA_ROOT}/nuplan-v1.1/sensor_blobs
```
(Change them accordingly to how you have structured your dataset folders).

3. install the nuplan devkit locally, in editable mode:

```
cd ~/workspace/nuplan-devkit
python -m pip install -e .
```

### tuplan-garage

This repository provides a number of planners that can be used for the ego agent in the nuplan simulator. They also provide the pretrained models.

(Is it necessary to shift back to previous versions??)
1. Clone the tuplan-garage repository
```
cd ~/workspace && git clone https://github.com/autonomousvision/tuplan_garage.git && cd tuplan_garage
```


(Is it necessary to shift back to previous versions??)
2. install the package locally, in editable mode:
```
cd ~/workspace/tuplan-garage
python -m pip install -e .
```

3. Download the pretrained models from the [this link](https://drive.google.com/drive/folders/1LLdunqyvQQuBuknzmf7KMIJiA2grLYB2); save the path to these models for later.


<a name="testing_install"></a>
## Testing Installations

To test the installation, we can try simulating a scenario, using one the planners from `tuplan-garage`; open an interactive session on cluster, activate the `nuplan` environmenet, and run the following script:
```
python $NUPLAN_DEVKIT_ROOT/nuplan/planning/script/run_simulation.py \
    +simulation=open_loop_boxes \
    planner=pdm_open_planner \ planner.pdm_open_planner.checkpoint_path=${path to pdm open model} \
    scenario_filter=all_scenarios \
    scenario_filter.scenario_types="[starting_unprotected_cross_turn]" \
    scenario_filter.num_scenarios_per_type=1 \
    scenario_builder=nuplan_mini \
    worker=sequential \
    'hydra.searchpath=["pkg://tuplan_garage.planning.script.config.common", "pkg://tuplan_garage.planning.script.config.simulation", "pkg://nuplan.planning.script.config.common", "pkg://nuplan.planning.script.experiments"]' \
    
```


The result of the simulation should be saved in `NUPLAN_EXP_ROOT`.
Under the experiment folder you can find the subfolder called `code/hydra`, under which you can check the hydra configuration, and importantly your overrides.


<a name="nuplan_king"></a>
## Installing Kinematic Adversary Agents


To install the current repository:

```
cd ~/workspace/kinematic_adversary_agents/nuplan_king
python -m pip install -e .
```


We can now try to launch the nuboard html page by running this command in `vscode terminal`:
```
conda activate workspace/miniconda3/envs/nuplan/
python /home/kpegah/workspace/nuplan_king/nuplan/planning/script/run_nuboard.py
```


<a name="arguments"></a>
## Important parts of the code

The code consists of two main parts:

1. Reconstructing the trajectory by using the `tracker` and the `motion model`.
2. Adversarially optimizing the extracted actions to induce a collision with the ego vehicle.

Here, we give a short explanation on the most important functions of each component of the code:

<a name="optimizer"></a>
#### Optimizer

To reset the state of differen components + preparing buffers to store the losses, actions, and their gradients (per step, and per optimization iteration).
```python
def reset(self):
    self.bm_iteration = 1
    ...


    self._simulation.reset()
    self._simulation._ego_controller.reset()
    self._simulation._observations.reset()

    self.reset_dynamic_states()
```


To initialize the position of the agents, and more precisely their state.
To see how different components of the state are related: check out `motion_model`.
There are lines commented out: some are to visualize the position of the agents before and after transforming them to map coordinates; other comments are change the initialization state, in case we want to start from stationary position (in this case you should also set the `throttle` and `steer` in `initialize` function to zero instead of using `tracker`.)
```python
def init_dynamic_states(self):
    self._states = {'pos': torch.zeros(self._number_agents, 2,  requires_grad=True).to(device=device), 
                        'yaw': torch.zeros(self._number_agents, 1, requires_grad=True).to(device=device), 
                        'steering_angle': torch.zeros(self._number_agents, 1, requires_grad=True).to(device=device), 
                        'vel': torch.zeros(self._number_agents, 2, requires_grad=True).to(device=device), 
                        'accel': torch.zeros(self._number_agents, 1, requires_grad=True).to(device=device), 
                        'speed': torch.zeros(self._number_agents, 1, requires_grad=True).to(device=device)}
        

    self._states_original = ...


    ...

    for idx, tracked_agent in enumerate(self._agents):

            coord_x, coord_y, coord_yaw = tracked_agent.predictions[0].valid_waypoints[0].x, tracked_agent.predictions[0].valid_waypoints[0].y, 

            map_x, map_y, map_yaw = self._convert_coord_map.pos_to_map(coord_x, coord_y, coord_yaw)
           

```


`initialize` function is the part that calls `trajectory reconstructor`, which extracts and optimizes the actions.
One important detail is the dimension of the `throttle_temp` and `steer_temp` which is a list of size `horizon` of tensors of size `num agents`; the reason is to enable optimizing the actions of all agents at a certain step in parallel. We later use `reshape_actions` to change the shape of these tensors.
We also save the extracted (and converted to map coordinate system) positions, so that we can later have the choice to use their original states instead of enrolling the `motion model` on their actions.
`complete_states_original` is to put the state of the agent in its unpresence time steps equal to its states in last presence time instead of zero (since we are keeping all the agents present in the first frame, during the whole simulation).

```python
def initialize(self):

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

    for idx, tracked_agent in enumerate(self._agents):

        while current_timepoint < end_timepoint:
            
            map_x, map_y, map_yaw = self._convert_coord_map.pos_to_map(coord_x, coord_y, coord_yaw)
            

            with torch.no_grad():
                self._positions[idx, counter_steps, :] = torch.tensor([map_x, map_y], dtype=torch.float64)
                self._velocities[idx, counter_steps, :] = torch.tensor([map_vel_x, map_vel_y], dtype=torch.float64)
                self._headings[idx, counter_steps, :] = torch.tensor([map_yaw], dtype=torch.float64)
                self._action_mask[idx, counter_steps] = 1.0
                
            current_timepoint = current_timepoint + self._interval_timepoint
            
        self.complete_states_original(counter_steps-1, idx)

          
    self._trajectory_reconstructor.initialize_optimization(self._action_mask, self._positions, self._headings, self._velocities)
    self._trajectory_reconstructor.reset_error_losses()
    
    for idx, tracked_agent in enumerate(self._agents):

        self._trajectory_reconstructor.set_current_trajectory(idx, tracked_agent.box, transformed_trajectory, waypoints, current_state, end_timepoint, self._interval_timepoint)
                
        self._trajectory_reconstructor.individual_step_by_step_optimization(self._actions_to_use)
        self._trajectory_reconstructor.report(idx, len(waypoints)-1, current_state)


    self._trajectory_reconstructor.parallel_step_by_step_optimization(self.get_adv_state())
    # self._trajectory_reconstructor.parallel_all_actions_optimization(self.get_adv_state())
```

To get the state and actions of an agent or all of them.
```python
def get_adv_state(self, id:Optional(int) = None)
```
```python
def get_adv_actions(self, current_iteration:int = 0, id:Optional(int) = None)
```

Chaging the state of an agent or all of them. `set_adv_state_to_original` is used to set the state of all agents, except the one indicated by its index, to their original extracted state (stored in `self._states_original`).

```python
def set_adv_state(self, next_state: Dict[str, torch.TensorType] = None, next_iteration:int = 1, id=None)
```
```python
def set_adv_state_to_original(self, next_state: Dict[str, torch.TensorType] = None, next_iteration:int = 1, exception_agent_index:int = 0)
```

`step` function enrolls the bicycle model and changes the state of agents over one step of simulation (if using a denser sampling of actions than the sampling time of simulation, we take few actions in one step of simulation).

```python
def step(self, current_iteration:int) -> Dict[str, ((float, float), (float, float), float)]:
  

    if self._collision_occurred and self._collision_strat=='back_to_after_bm':
        if current_iteration==1:
            self.actions_to_original()
            self.stopping_collider()
        return self.step_after_collision(current_iteration)
    ...

    not_differentiable_state = {}  
      
    self.set_adv_state(self._motion_model.forward_all(self.get_adv_state(), self.get_adv_actions(temp_bm_iter-1)))
            


    for idx in range(self._number_agents):
        coord_vel_x, coord_vel_y, _ = self._convert_coord_map.pos_to_coord_vel(agents_vel[idx, 0], agents_vel[idx, 1], agents_yaw[idx, 0])
        coord_pos_x, coord_pos_y, coord_pos_yaw = self._convert_coord_map.pos_to_coord(agents_pos[idx, 0], agents_pos[idx, 1], agents_yaw[idx, 0])

        not_differentiable_state[self._agent_tokens[idx]] = ((coord_pos_x, coord_pos_y), (coord_vel_x, coord_vel_y), (coord_pos_yaw))
        

    
    if not self._collision_occurred and self.check_collision_simple(agents_pos, ego_position):
        collision, collision_index = self.check_collision(agents_pos, agents_yaw, ego_position, ego_heading)
        if collision:
            self._collision_occurred = True
            self._collision_index = collision_index
            self._collision_iteration = ...
            return True, None

    return False, not_differentiable_state
```

Other types of `step`:
```python
def step_after_collision(self, current_iteration)
```
```python
def step_cost_without_simulation(self, current_iteration: int)
```

Important detail on `compute cost` is that it uses a buffer of ego state that get updated in every call of the planner. But in case we want to optimize without calling the simulator, this has saved the state of the ego in its last call.
```python
def compute_current_cost(self, current_iteration: int):
        
        
    input_cost_ego_state = {}
       
    for substate in self._ego_state.keys():
        input_cost_ego_state[substate] = torch.unsqueeze(torch.unsqueeze(self._ego_state_whole[substate][current_iteration,:], dim=0), dim=0)

    input_cost_adv_state = self.get_adv_state()

    ...
```

Back-propagation function for all the losses coputed during the whole simulation.
```python
def back_prop(self)
```


<a name="traj_recons"></a>
#### Trajectory Reconstructor


<a name="arguments"></a>
## The scripts and Configurations



