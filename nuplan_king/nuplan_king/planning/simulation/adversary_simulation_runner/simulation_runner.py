from __future__ import annotations

import logging
import time
from typing import Optional
from time import perf_counter


from nuplan.planning.simulation.planner.abstract_planner import AbstractPlanner
from nuplan.planning.simulation.runner.simulations_runner import SimulationRunner
from nuplan.planning.simulation.runner.runner_report import RunnerReport
from nuplan.planning.simulation.simulation import Simulation
from nuplan_king.planning.simulation.adversary_optimizer.king_optimizer import OptimizationKING

logger = logging.getLogger(__name__)


class AdvSimulationRunner(SimulationRunner):
    """
    Manager which executes multiple simulations with the same planner
    """

    def __init__(self, simulation: Simulation, planner: AbstractPlanner, optimizer: Optional[OptimizationKING]= None):
        """
        Initialize the simulations manager
        :param simulation: Simulation which will be executed
        :param planner: to be used to compute the desired ego's trajectory
        """
        super().__init__(simulation=simulation, planner=planner)
        self._optimizer = optimizer

        # Initialize Planner
        self.planner.initialize(self._simulation.initialize())
        self._optimizer.init_ego_state(self.simulation._ego_controller.get_state())




    
    def _initialize(self, final_iteration: bool) -> None:
        """
        Initialize the planner
        """
        # Execute specific callback
        if final_iteration:
            self._simulation.callback.on_initialization_start(self._simulation.setup, self.planner)

        # Initialize Planner
        self.planner.initialize(self._simulation.initialize())

        # Execute specific callback
        if final_iteration:
            self._simulation.callback.on_initialization_end(self._simulation.setup, self.planner)

    def run(self, first_iteration: bool, final_iteration: bool) -> RunnerReport:
        """
        Run the scenario and interact with the optimizer. The steps of execution follow:
         - update the states in the optimizer part, and then update the states in observation
         - output the trajectory of the ego to the optmizer at each step to compute the cost
         - backpropagate the cost on prior actions in 
        :return: A list of a single SimulationReports containing the result of the simulation.
        """
        start_time = perf_counter()
        time_to_collision = 0

        # Initialize reports for all the simulations that will run
        report = RunnerReport(
            succeeded=True,
            error_message=None,
            start_time=start_time,
            end_time=None,
            planner_report=None,
            scenario_name=self._simulation.scenario.scenario_name,
            planner_name=self.planner.name(),
            log_name=self._simulation.scenario.log_name,
        )

        # Execute specific callback
        if final_iteration:
            self.simulation.callback.on_simulation_start(self.simulation.setup)

        # Initialize all simulations
        self._initialize(final_iteration)


        # Calculating the extra time taken by the added modules
        time_amount_enroll_bm = 0
        time_amount_compute_cost = 0
        time_amount_whole_propagate = 0
        time_amount_cost_backprop = 0
        while self.simulation.is_simulation_running():

            
            # Execute specific callback
            if final_iteration:
                self.simulation.callback.on_step_start(self.simulation.setup, self.planner)

            # Perform step
            planner_input = self._simulation.get_planner_input() # this calls the history_buffer, initilized with the very first state
            logger.debug("Simulation iterations: %s" % planner_input.iteration.index)

            # Execute specific callback
            if final_iteration:
                self._simulation.callback.on_planner_start(self.simulation.setup, self.planner)

            # Plan path based on all planner's inputs
            trajectory = self.planner.compute_trajectory(planner_input)

            '''
            1. in the optimizer go from the current state to the next, considering the obtained actions at this iteration, using the bicycle model
            '''
            next_iteration_idx = self._simulation._time_controller.get_iteration().index+1

            start_time = perf_counter()
            if next_iteration_idx <= self._simulation._time_controller.number_of_iterations():
                # what happens in the last step of the simulation, when the history gets updated anyway, but the 
                # get_planner_input is never used? 
                # then just do a dummy kind of filling
                # next_states: Dict[str, Tuple[Tuple[float, float], Tuple[float, float], float]]
                collision_occurred, next_states = self._optimizer.step(next_iteration_idx)
                time_amount_enroll_bm += perf_counter() - start_time
                time_to_collision += perf_counter() - start_time
                if collision_occurred:
                    return time_to_collision, True, report
                if first_iteration:
                    self._optimizer.number_collision_after_bm()


            '''
            2. the obtained states should update the current state and available_iteration in the observation
            '''
            
            self._simulation._observations.update_available_state(next_iteration_idx, next_states)
            
            
            # Propagate simulation based on planner trajectory
            if final_iteration:
                self._simulation.callback.on_planner_end(self.simulation.setup, self.planner, trajectory)
            
            start_time = perf_counter()
            self.simulation.propagate(trajectory) # the state of the ego_controller in the simulation gets updated in the simulation using the trajectory
            time_amount_whole_propagate += perf_counter() - start_time
            

            
            
            if next_iteration_idx:
                '''
                3. update the ego trajectory to the optimizer, this 
                '''
                self._optimizer.update_ego_state(self.simulation._ego_controller.get_state(), next_iteration_idx)
                '''
                4. compute the cost in the optimizer for the current state, and accumulate the costs in optimizer
                '''
                # we should compute the cost for the next iteration, that we have already the states for in the king optimizer
                start_time = perf_counter()
                time_drivable_compute = self._optimizer.compute_current_cost(next_iteration_idx)
                time_amount_compute_cost += time_drivable_compute
                time_to_collision += perf_counter()-start_time


            # Execute specific callback
            if final_iteration:
                self.simulation.callback.on_step_end(self.simulation.setup, self.planner, self.simulation.history.last())

            # Store reports for simulations which just finished running
            current_time = perf_counter()
            if not self.simulation.is_simulation_running():
                report.end_time = current_time
                print('hey!! this is the last step of the simulation!!')
                # self._optimizer.compute_current_cost(next_iteration_idx)

        '''
        updating the actions in the optimizer
            - backpropagate the computed cost on the taken actions
        '''
        start_time = perf_counter()
        if not final_iteration:
            self._optimizer.back_prop()
            time_to_collision += perf_counter() - start_time
        else:
            self._optimizer.plot_losses()

        time_amount_cost_backprop += perf_counter() - start_time
        self._optimizer.save_state_buffer()
        # Execute specific callback
        if final_iteration:
            start_time = perf_counter()
            self.simulation.callback.on_simulation_end(self.simulation.setup, self.planner, self.simulation.history)
            end_time = perf_counter()
            print(f'storing the simulation took {end_time-start_time}')
        
        print(f'******** forward bm took {time_amount_enroll_bm} and cost computation {time_amount_compute_cost}  and backprop took {time_amount_cost_backprop} and propagation took {time_amount_whole_propagate} ********')

        planner_report = self.planner.generate_planner_report()
        report.planner_report = planner_report

        return time_to_collision, False, report