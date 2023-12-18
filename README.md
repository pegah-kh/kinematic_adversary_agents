# Knematic Adversary Agents


- [Dataset Steucture](#nuplan_structure)
- [An illustration](#Anillustration)

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
COPY bashrc_docker.sh /home/kpegah/bashrc_temp_docker.sh
RUN cat /home/kpegah/bashrc_temp_docker.sh >> /home/kpegah/.bashrc && \
    rm /home/kpegah/bashrc_temp_docker.sh
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

### Nuplan-Devkit







This repository is the code for [*Category classification and landmark localization for a fashion dataset*](https://drive.google.com/drive/folders/1jqvd6CmmyKQaodJAdwPNVwH92M9YC9tg?usp=sharing).
The project is a part of an internship I did at VITA Lab, EPFL.

![Alt Text](visualizations/clothing_landmark.gif)


The above is a short illustration of predictions made by one of the trained models. The numbers shown on the second image of each pair are the normalized errors.


### -Requirements for converting deepfashion to a coco style dataset

First of all, download deepfashion-c (category classification and attribute prediction) dataset from [dedicated drive](https://drive.google.com/drive/folders/0B7EVK8r0v71pQ2FuZ0k0QnhBQnc?resourcekey=0-NWldFxSChFuCpK4nzAIGsg). In our training we used low resolution images. Create a folder for all the dataset files and put in it:

1. unzipped img folder
2. an Eval folder containing list_eval_partition.txt
3. an Anno folder containing list_landmarks.txt and list_bbox.txt from Anno_coarse

The next step is to change the running script `convert_coco_separate/convert_separate.sh` (in case you are running the code on a cluster); change *--dataset-root* to the path of the folder you created for dataset and *--root-save* to the path of the folder where you want to save your coco style converted deepfashion. At the end, you will have three folders and three json files that correspond to different subsets of the dataset (train, eval, and test).

### -Requirements for training

Python 3,  openpifpaf (tested with version 0.13.4):

```
pip3 install openpifpaf
```

Find a FAQ page and other useful information at [Openpifpaf guide](https://openpifpaf.github.io/intro.html).

to train the model you can use a similar script to `shufflenetv2k16_scratch_0001.sh`. However, you should either indicate the path to dataset either in the script, either directly in the code `openpifpaf_deepfashion/deepfashion_datamodule`.

### -Requirements for evaluation

Your trained model should be in `outputs` folder in the main directory (same level as the script used for training). 
To evaluate, change the script `evaluation/shufflenetv2k16_scratch_evaluation.sh` accordingly to the path of your model. You can also have tensorboard logs if not using `single_epoch` option and changing other parameters in the script accordingly.

One example of such curves can be found [here](https://wandb.ai/pekhpekhpekh/uncategorized/runs/9ar9kssd/overview?workspace=user-pekhpekhpekh).






