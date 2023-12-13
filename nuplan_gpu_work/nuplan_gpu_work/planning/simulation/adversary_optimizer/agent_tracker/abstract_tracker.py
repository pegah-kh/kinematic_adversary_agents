import abc

from nuplan.common.actor_state.dynamic_car_state import DynamicCarState
from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.actor_state.state_representation import TimePoint
from nuplan.common.actor_state.tracked_objects import TrackedObject
from nuplan.planning.simulation.simulation_time_controller.simulation_iteration import SimulationIteration
from nuplan.planning.simulation.trajectory.abstract_trajectory import AbstractTrajectory


class AbstractTracker(abc.ABC):
    """
    Interface for a generic tracker.
    """

    @abc.abstractmethod
    def track_trajectory(
        self,
        current_timepoint: TimePoint,
        next_timepoint: TimePoint,
        initial_state: TrackedObject,
        trajectory: AbstractTrajectory,
    ) -> DynamicCarState:
        """
        Return the dynamic state of the agent with the given trajectory at the initial state, in the next iteration.
        :param current_iteration: The current timepoint.
        :param next_iteration: The desired next timepoint.
        :param initial_state: The current simulation state of the agent.
        :param trajectory: The reference trajectory to track.
        :return: The ego state to be propagated
        """
        pass