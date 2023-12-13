import logging
import os
from pathlib import Path
from typing import List

import pandas as pd
from omegaconf import DictConfig

from nuplan.planning.simulation.simulation import Simulation
from nuplan.planning.simulation.planner.abstract_planner import AbstractPlanner
from nuplan_gpu_work.planning.simulation.adversary_optimizer.king_optimizer import OptimizationKING
from nuplan.planning.script.utils import CommonBuilder, save_runner_reports 


logger = logging.getLogger(__name__)


def run_optimization(
    my_simulation: Simulation, my_planner: AbstractPlanner, common_builder: CommonBuilder, profiler_name: str, cfg: DictConfig
) -> None:
    """
    Run the optimizer.
    :param my_simulation: The simulation to be optimized (optimizing the actions in king for example).
    :param my_planner: The planner of the go.
    :param profiler_name: Profiler name.
    :param cfg: Hydra config.
    """


    # this shuold call the optimizer that then calls the simulation
    # the common_builder should be changed to a one that is for optimization and not starting a simulation
    if common_builder.profiler:
        common_builder.profiler.start_profiler(profiler_name)

    logger.info('Initializing the optimizer...')
    '''
    The initialization of the optimizer
    Inputs are:
        - the simulation
        - the planner
    the optimizer outputs at each iteration if the collision has happened or not.
    '''

    optimizer = OptimizationKING(my_simulation, my_planner)
    
    # The loop of optimization:
    optimization_round = 0
    reports = []
    while optimization_round < cfg.max_optimization_rounds and not optimizer.collision_happened():
        '''
        A step should be taken in optimizer to:
            - run the simulation step by step
                - having the current actions, and states, go to the next set of states, using the bm model
                - update the state of the agents in the scene from the new simulations_runner (change the observation, and hence the get_observation function)
                - pass the go trajectory to the optimizer
                - compute the loss in the optimizer, and backpropagate it on the actions taken beforehand
            - check the collision in this while loop
            - reset the simulation and its runner
        '''
        optimizer.simulation.reset()
        reports.append(optimizer.step())
        
    if optimizer.collision_happened():
        logger.info('Found a scenario with collision!')
    else:
        logger.info('No collision has been found in this scenario!')

    # TODO: change the format of the reports
    
    # Save RunnerReports as parquet file
    save_runner_reports(reports, common_builder.output_dir, cfg.runner_report_file)

    # Save profiler
    if common_builder.profiler:
        common_builder.profiler.save_profiler(profiler_name)
