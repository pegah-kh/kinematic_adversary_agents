from typing import Type, Dict, List, Tuple, Set
import numpy as np

from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType
from nuplan.common.actor_state.tracked_objects import TrackedObject
from nuplan.common.actor_state.oriented_box import OrientedBox
from nuplan.common.actor_state.state_representation import StateSE2
from nuplan.common.actor_state.waypoint import Waypoint
from nuplan.common.actor_state.agent import Agent
from nuplan.common.actor_state.tracked_objects import TrackedObject, TrackedObjects
from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from nuplan.planning.simulation.history.simulation_history_buffer import SimulationHistoryBuffer
from nuplan.planning.simulation.observation.abstract_observation import AbstractObservation
from nuplan.planning.simulation.observation.observation_type import DetectionsTracks, Observation
from nuplan.planning.simulation.simulation_time_controller.simulation_iteration import SimulationIteration
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from nuplan.planning.simulation.trajectory.predicted_trajectory import PredictedTrajectory
from nuplan.common.geometry.interpolate_state import interpolate_future_waypoints
from nuplan.database.nuplan_db.nuplan_db_utils import get_lidarpc_sensor_data


from nuplan.database.nuplan_db.nuplan_scenario_queries import (
    get_future_waypoints_for_agents_from_db,
    get_sensor_data_token_timestamp_from_db,
    get_tracked_objects_for_lidarpc_token_from_db,
)


class TracksObservation(AbstractObservation):
    """
    Replay detections from samples.
    """

    def __init__(self, scenario: AbstractScenario):
        """
        :param scenario: The scenario
        """
        self._scenario = scenario
        self.current_iteration = 0
        self._trajectory_sampling = None
        self._agents_states: Dict[str, Tuple[Tuple[float, float], Tuple[float, float], float]] = None
        self._available_iteration: int = None

    def set_traj_sampling(self, trajectory_sampling: TrajectorySampling):
        self._trajectory_sampling = trajectory_sampling

    def reset(self) -> None:
        """Inherited, see superclass."""
        self.current_iteration = 0
        self._available_iteration = 0

    def observation_type(self) -> Type[Observation]:
        """Inherited, see superclass."""
        return DetectionsTracks  # type: ignore

    def initialize(self) -> None:
        """Inherited, see superclass."""
        self._initialy_tracked_vehicles = self._scenario.get_tracked_objects_at_iteration(self.current_iteration, self._trajectory_sampling).tracked_objects.get_tracked_objects_of_type(TrackedObjectType.VEHICLE)
        self._tokento_idx: Dict[str, int] = {}
        for i, agent in enumerate(self._initialy_tracked_vehicles):
            track_token = agent.metadata.track_token
            self._tokento_idx[track_token] = i
    
    def perturbed_get_observation(self) -> DetectionsTracks:
        """Inherited, see superclass."""
        tracked_objects = self._scenario.get_tracked_objects_at_iteration(self.current_iteration, self._trajectory_sampling)
        agents: List[TrackedObject] = tracked_objects.tracked_objects.get_tracked_objects_of_type(TrackedObjectType.VEHICLE)
        # other_than_vehicle: List[TrackedObject] = [tracked_object for tracked_object in tracked_objects.tracked_objects if tracked_object not in agents]
        changed_prediction: Dict[str, List[Waypoint]] = {}
        for agent in agents:
            track_token = agent.metadata.track_token
            waypoint_modified = []
            for w in agent.predictions[0].valid_waypoints:
                w_box = w._oriented_box
                rand_perturb = np.random.randint(8, size=2)
                w._oriented_box = OrientedBox.from_new_pose(w_box, StateSE2(w_box.center.x+rand_perturb[0], w_box.center.y+rand_perturb[1], w_box.center.heading))
                waypoint_modified.append(w)
            changed_prediction[track_token] = waypoint_modified
        return self._scenario.modified_get_tracked_objects_at_iteration(self.current_iteration, changed_prediction, self._trajectory_sampling) # should add changed_prediction: Dict[str, List[Waypoint]]
    
    def update_available_state(self, current_iteration:int, updated_states:Dict[str, Tuple[Tuple[float, float], Tuple[float, float], float]]):
        '''
        update the available state number and also the available states
        these available states will later be used in the get_observation function to change the state at that iteration
        '''

        if current_iteration is None:
            return
        

        self._available_iteration = current_iteration
        self._agents_states = updated_states

    
    def get_observation_old(self) -> DetectionsTracks:
        """Inherited, see superclass."""

        # assert not np.isnan(self._trajectory_sampling), 'the trajectory sampling is not yet set'
        # assert self._available_iteration==self.current_iteration, 'Not having the state for the current iteration'

        if self.current_iteration==self._available_iteration:
            if self.current_iteration==0:
                return self._scenario.get_tracked_objects_at_iteration(self.current_iteration, self._trajectory_sampling)
            

            tracked_objects = self._scenario.get_tracked_objects_at_iteration(self.current_iteration, self._trajectory_sampling)
            agents: List[TrackedObject] = tracked_objects.tracked_objects.get_tracked_objects_of_type(TrackedObjectType.VEHICLE)
            changed_current_center: Dict[str, StateSE2] = {}
            # other_than_vehicle: List[TrackedObject] = [tracked_object for tracked_object in tracked_objects.tracked_objects if tracked_object not in agents
            # using the updated states to change the      
            for agent in agents:
                track_token = agent.metadata.track_token
                if track_token in self._agents_states.keys(
                ):
                    (pos_x, pos_y), (_, _), yaw = self._agents_states[track_token]
                    new_center = StateSE2(pos_x, pos_y, yaw)
                    changed_current_center[track_token] = new_center
            return self._scenario.modified_get_tracked_objects_at_iteration(self.current_iteration, changed_current_center, self._trajectory_sampling) # should add changed_prediction: Dict[str, List[Waypoint]]

        else:
            return self._scenario.get_tracked_objects_at_iteration(self.current_iteration, self._trajectory_sampling)


    def get_observation_at_iteration(self, iteration: int, trajectory_sampling: TrajectorySampling) -> DetectionsTracks:

        return self._scenario.get_tracked_objects_at_iteration(iteration, trajectory_sampling)
    
    def update_observation(
        self, iteration: SimulationIteration, next_iteration: SimulationIteration, history: SimulationHistoryBuffer
    ) -> None:
        """Inherited, see superclass."""
        self.current_iteration = next_iteration.index

        
    def get_observation(self) -> DetectionsTracks:
        """Inherited, see superclass."""


        """ 
        Keep the agents alive even after they are no more in the get_tracked_objects_at_iteration
            what would be their position in that case? there is a position predicted in the agents_states
        """


        # print('the current iteration is ', self.current_iteration)
        # assert not np.isnan(self._trajectory_sampling), 'the trajectory sampling is not yet set'
        # assert self._available_iteration==self.current_iteration, 'Not having the state for the current iteration'

        if self.current_iteration==self._available_iteration:
            if self.current_iteration==0:
                return self._scenario.get_tracked_objects_at_iteration(self.current_iteration, self._trajectory_sampling)
            

            # tracked_objects = self._scenario.get_tracked_objects_at_iteration(self.current_iteration, self._trajectory_sampling)
            # agents: List[TrackedObject] = tracked_objects.tracked_objects.get_tracked_objects_of_type(TrackedObjectType.VEHICLE)
            # ************
            tracked_objects: List[TrackedObject] = []
            agent_indexes: Dict[str, int] = {}
            agent_future_trajectories: Dict[str, List[Waypoint]] = {}
            agents_waypoint_counter: Dict[str, int] = {}
            seen_tokens: List[str] = []

            token = self._scenario._lidarpc_tokens[self.current_iteration]
            log_file = self._scenario._log_file
            assert isinstance(token, str), 'token is not string'
            assert isinstance(log_file, str), 'log_file is not string'

            for idx, tracked_object in enumerate(get_tracked_objects_for_lidarpc_token_from_db(log_file, token)):
                if self._trajectory_sampling and isinstance(tracked_object, Agent):
                    agent_indexes[tracked_object.metadata.track_token] = idx
                    agent_future_trajectories[tracked_object.metadata.track_token] = []
                # print(f'the changed_prediction {changed_prediction}')
                if tracked_object.track_token in  self._agents_states.keys():
                    (pos_x, pos_y), (_, _), yaw = self._agents_states[tracked_object.track_token]
                    new_center = StateSE2(pos_x, pos_y, yaw)
                    tracked_object = Agent.from_new_pose(tracked_object, new_center)
                    seen_tokens.append(tracked_object.track_token)
                tracked_objects.append(tracked_object)
                agents_waypoint_counter[tracked_object.metadata.track_token] = 0

            # for the agents no more in the observation, but that were tracked at first
            seen_tokens = list(set(self._agents_states.keys()) - set(seen_tokens))
            for initialy_tracked in seen_tokens:
                (pos_x, pos_y), (_, _), yaw = self._agents_states[initialy_tracked]
                new_center = StateSE2(pos_x, pos_y, yaw)
                # how do we get the very first objects that were tracked?
                tracked_object = self._initialy_tracked_vehicles[self._tokento_idx[initialy_tracked]]
                tracked_object = Agent.from_new_pose(tracked_object, new_center)
                tracked_objects.append(tracked_object)

            if self._trajectory_sampling and len(tracked_objects) > 0:
                timestamp_time = get_sensor_data_token_timestamp_from_db(log_file, get_lidarpc_sensor_data(), token)
                end_time = timestamp_time + int(
                    1e6 * (self._trajectory_sampling.time_horizon + self._trajectory_sampling.interval_length)
                )

                # TODO:(taken from nuplan code) This is somewhat inefficient because the resampling should happen in SQL layer
                for track_token, waypoint in get_future_waypoints_for_agents_from_db(
                    log_file, list(agent_indexes.keys()), timestamp_time, end_time
                ):  
                    
                    # if the agent is from those tracked from the first frame and not having added any waypoint to its trajectory
                    # this second constraint is to ensure that we do not go through one of the followed agents twice
                    # as we do only have one waypoint into future
                    if track_token in self._agents_states.keys() and agents_waypoint_counter[track_token]==0:
                        counter = agents_waypoint_counter[track_token]
                        (pos_x, pos_y), (_, _), yaw = self._agents_states[track_token]
                        new_box = OrientedBox.from_new_pose(waypoint.oriented_box, StateSE2(pos_x, pos_y, yaw))
                        waypoint._oriented_box = new_box
                        agent_future_trajectories[track_token].append(waypoint) # this should be a Waypoint, but we have the state that waypoing should be constructed from
                        agents_waypoint_counter[track_token] = counter+1
                    else:
                        agent_future_trajectories[track_token].append(waypoint)


                for key in agent_future_trajectories:
                    # We can only interpolate waypoints if there is more than one in the future.
                    if len(agent_future_trajectories[key]) == 1:
                        tracked_objects[agent_indexes[key]]._predictions = [
                            PredictedTrajectory(1.0, agent_future_trajectories[key])
                        ]
                    elif len(agent_future_trajectories[key]) > 1:
                        tracked_objects[agent_indexes[key]]._predictions = [
                            PredictedTrajectory(
                                1.0,
                                interpolate_future_waypoints(
                                    agent_future_trajectories[key],
                                    self._trajectory_sampling.time_horizon,
                                    self._trajectory_sampling.interval_length,
                                ),
                            )
                        ]

            return DetectionsTracks(TrackedObjects(tracked_objects=tracked_objects))
        else:
            return self._scenario.get_tracked_objects_at_iteration(self.current_iteration, self._trajectory_sampling)


