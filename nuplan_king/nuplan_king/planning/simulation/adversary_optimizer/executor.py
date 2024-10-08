import logging 
import time
import traceback
from typing import Optional
from dataclasses import dataclass
from omegaconf import DictConfig
from hydra.utils import instantiate
import torch
import os
import csv
import wandb


from nuplan.planning.simulation.runner.abstract_runner import AbstractRunner
from nuplan.planning.utils.multithreading.worker_pool import WorkerPool
from nuplan.planning.script.utils import CommonBuilder
from nuplan.planning.simulation.planner.abstract_planner import AbstractPlanner


from nuplan_king.planning.simulation.adversary_optimizer.abstract_optimizer import AbstractOptimizer
from nuplan_king.planning.simulation.adversary_optimizer.optimizer_report import RunnerReport
from nuplan_king.planning.simulation.adversary_simulation_runner.simulation_runner import AdvSimulationRunner
from nuplan_king.planning.simulation.adversary_optimizer.king_optimizer import OptimizationKING



logger = logging.getLogger(__name__)


def create_directory_if_not_exists(directory_path):
    """
        Creating a dirctory at the given directory path
        :param directory_path
    """
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)

def run_optimization(simulation: AdvSimulationRunner, planner: AbstractPlanner, cfg: Optional[DictConfig], exit_on_failure: bool=False, common_builder: Optional[CommonBuilder] = None, profiler_name: Optional[str] = None) -> RunnerReport:

    """
    proxy to call the optimizer.
    """

    start_time = time.perf_counter()
    
    # the tracker and the cost functions should be given in the hydra config file of the optimizer
    # TODO: the optimizer should output the report
    torch.autograd.set_detect_anomaly(True)
        
    # optimizer :OptimizationKING = instantiate(cfg.adversary_optimizer, tracker=cfg.adversary_optimizer.tracker,simulation=simulation, planner=planner)
    optimizer :OptimizationKING = instantiate(cfg.adversary_optimizer, simulation=simulation)
    simulation_runner = AdvSimulationRunner(simulation, planner, optimizer)
    optimizer.initialize()
    # optimizer.save_state_buffer()

    # # temporary
    # return

    torch.set_default_dtype(torch.float64)
    total_time_to_collision = 0
    try:
        current_iteration = 0
        while current_iteration < optimizer._opt_iterations:
            time_to_collision, collision, _ = run(optimizer, simulation_runner, current_iteration, first_iteration=current_iteration==0, final_iteration=current_iteration==(optimizer._opt_iterations-1))
            total_time_to_collision += time_to_collision
            if collision:

                if 'moving_dummy' in optimizer._costs_to_use:
                    collision_loss = 'moving_dummy'
                else:
                    collision_loss = 'king_collision'
                

                optimizer.reset()
                simulation_runner._initialize(True)
                if current_iteration==0:
                    optimizer.reset_number_collision_after_bm()

                simulation_runner.run(first_iteration=current_iteration==0, final_iteration=True)

                write_metrics(iteration_jump=optimizer._optimization_jump,
                          number_iterations=current_iteration,
                          time_to_collision=total_time_to_collision,
                          collided_agent_drivable_cost=optimizer.get_collided_agent_drivable_cost(),
                          number_adversary_collisions=optimizer.get_number_adversary_collisions(),
                          number_initial_adversary_collisions=optimizer.number_collision_original_states(),
                          number_after_bm_adversary_collisions=optimizer.get_number_collision_after_bm(),
                          collision_after_traj_changes=optimizer.get_collision_after_traj_changes(),
                          scenario_log_name=str(optimizer._simulation.scenario.scenario_name)+str(optimizer._experiment_name), 
                          collision_strat=optimizer._collision_strat, 
                          collision_loss=collision_loss, 
                          report_path=os.path.join(optimizer._metric_report_path, optimizer._simulation.scenario.scenario_name, optimizer._experiment_name))
                
                
                return
            current_iteration += 1
        # optimizer.plots()
        write_simple_metrics(number_initial_adversary_collisions=optimizer.number_collision_original_states(),
                             number_after_bm_adversary_collisions=optimizer.get_number_collision_after_bm(),
                             scenario_log_name=str(optimizer._simulation.scenario.scenario_name)+str(optimizer._experiment_name),
                             report_path=os.path.join(optimizer._metric_report_path, optimizer._simulation.scenario.scenario_name, optimizer._experiment_name))
        
        logger.info(f"The optimization was successful")
  
    except Exception as e:
        error = traceback.format_exc()


        logger.warning("Optimization failed with the folllowing trace:")
        traceback.print_exc()
        logger.warning(f"Optimization failed with the error: {e}")

        if exit_on_failure:
            raise RuntimeError('Optimization failed')
        
        end_time = time.perf_counter()


        report = RunnerReport(
            succeeded=False,
            error_message=error,
            start_time=start_time,
            end_time=end_time,
            planner_report=None,
            scenario_name=optimizer._simulation.scenario.scenario_name,
            log_name=optimizer._simulation.scenario.log_name
        )
        logger.info(f"The optimization failed")
        return report
    return



def run(optimizer: OptimizationKING, 
        simulation_run: AdvSimulationRunner, 
        current_iteration: int, 
        first_iteration: bool,
        final_iteration: bool) -> RunnerReport:
    """
    one round of optimization (the current_ietration step)
    """

    if current_iteration < optimizer._max_opt_iterations:
        # optimizer.reset()
        simulation_run._initialize(final_iteration)

        '''
        # type sanity checks:
        logger.info(f'the simulation of the optimizer is {optimizer._simulation}')
        logger.info(f'the planner of the optimizer is {optimizer._planner}')
        
        # at starting point sanity checks
        logger.info(f'the current time point in the simulation is {optimizer._simulation._time_controller.get_iteration().time_point.time_s}')
        logger.info(f'the first iteration time in the scenario {optimizer._simulation._scenario.get_time_point(0).time_s}')

        # intervals sanity checks
        logger.info(f'the discretization time in tracker {optimizer._tracker._discretization_time}')
        logger.info(f'the interval length in trajectory sampling in tracker {optimizer._observation_trajectory_sampling.interval_length}')

        
        # there is a simulation in the simulation runner and there is a simulation parameter of the optimizer
        # are they the same simulation??
        logger.info(f'time of sim in runner {simulation_run.simulation._time_controller.get_iteration().time_point.time_s} and in the sim in optimizer {optimizer._simulation._time_controller.get_iteration().time_point.time_s}')
        # calling the simulation_runner from after here

        '''

        # optimizer.check_map()
        if current_iteration==0:
            optimizer.compare_king_efficient_deviation_cost_heatmaps()
            optimizer.compare_king_efficient_deviation_cost_agents()
        if current_iteration==0:
            # optimizer.optimize_all_actions()
            optimizer.reshape_actions()

        optimizer.reset()
        time_to_collision, collision_occurred, report = simulation_run.run(first_iteration, final_iteration)
        if collision_occurred:
            return time_to_collision, True, report
        if not final_iteration:
            optimizer.optimize_without_simulation()
        if current_iteration%10==0:
            optimizer.visualize_grads_steer(optimization_iteration=current_iteration)
            optimizer.visualize_grads_throttle(optimization_iteration=current_iteration)
            optimizer.visualize_steer(optimization_iteration=current_iteration)
            optimizer.visualize_throttle(optimization_iteration=current_iteration)  
    return time_to_collision, False, report 


def write_metrics(iteration_jump,
                  number_iterations,
                  time_to_collision, 
                  collided_agent_drivable_cost,
                  number_adversary_collisions,
                  number_initial_adversary_collisions,
                  number_after_bm_adversary_collisions,
                  collision_after_traj_changes,
                  scenario_log_name, 
                  collision_strat, 
                  collision_loss,
                  report_path):
    

    # wandb.log({
    #     'Time to Collision': time_to_collision,
    #     'Drivable Cost': collided_agent_drivable_cost, # drivable cost of the collided agent
    #     'Scenario Log Name': scenario_log_name,
    #     'Collision Strategy': collision_strat,
    #     'Collision Loss': collision_loss,
    # })


    # to locally also log the results
    create_directory_if_not_exists(report_path)
    file_path =  os.path.join(report_path, 'collision_report.csv')
    with open(file_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Iteration jump', 'Number of iterations', 'Time to Collision', 'Drivable Cost', 'Adversary Collision Count', 'Initial Adversary Collision Count', 'After BM Adversary Collision Count', 'Collision happens after change', 'Scenario Log Name', 'Collision Strategy', 'Collision Loss'])
        writer.writerow([iteration_jump, number_iterations, time_to_collision, collided_agent_drivable_cost, number_adversary_collisions, number_initial_adversary_collisions, number_after_bm_adversary_collisions, collision_after_traj_changes, scenario_log_name, collision_strat, collision_loss])






def write_simple_metrics(number_initial_adversary_collisions,
                  number_after_bm_adversary_collisions,
                  scenario_log_name, 
                  report_path):
    

 
    # to locally also log the results
    create_directory_if_not_exists(report_path)
    file_path =  os.path.join(report_path, 'reconstruction_report.csv')
    with open(file_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Initial Adversary Collision Count', 'After BM Adversary Collision Count', 'Scenario Log Name'])
        writer.writerow([number_initial_adversary_collisions, number_after_bm_adversary_collisions, scenario_log_name])
