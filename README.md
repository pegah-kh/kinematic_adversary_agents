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
<a name="arguments"></a>
#### Optimizer
```python
def calculate_sum(a, b):
    return a + b
    

<a name="arguments"></a>
#### Optimizer


<a name="arguments"></a>
## The scripts and Configurations



