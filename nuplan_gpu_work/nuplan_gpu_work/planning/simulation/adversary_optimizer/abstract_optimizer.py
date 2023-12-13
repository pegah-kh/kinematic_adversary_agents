from abc import ABCMeta, abstractmethod


from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from nuplan.planning.simulation.planner.abstract_planner import AbstractPlanner
from nuplan.planning.simulation.runner.runner_report import RunnerReport
from nuplan.planning.simulation.runner.abstract_runner import AbstractRunner
from nuplan.planning.simulation.simulation import Simulation




# what is the difference between this runner and an abstract runner?? it does also have access to a simulator runner

class AbstractOptimizer(metaclass=ABCMeta):
    """Interface for a generic runner."""


    # def __init__(self, simulation: Simulation, planner: AbstractPlanner):

    #     self._simulation = simulation
    #     self._planner = planner

    #     # accumulating the reports from each round of optimization
    #     self._reports = []

    @abstractmethod
    def reset(self) -> None:
        '''
        to reset the simulation after each round of optimization
        '''
        pass

    @abstractmethod
    def initialize(self):
        '''
        - find the initial actions from the existing trajectories for background agents
        - create variables for these extracted parameters
        '''
        pass


    @property
    @abstractmethod
    def scenario(self) -> AbstractScenario:
        """
        :return: Get the scenario.
        """
        return self._simulation.scenario

    @property
    @abstractmethod
    def planner(self) -> AbstractPlanner:
        """
        :return: Get a planner.
        """
        return self._planner