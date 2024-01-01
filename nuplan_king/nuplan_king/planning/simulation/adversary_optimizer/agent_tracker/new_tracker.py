import logging
from enum import IntEnum
from typing import List, Tuple

import numpy as np
import numpy.typing as npt

from nuplan.common.actor_state.dynamic_car_state import DynamicCarState
from nuplan.common.actor_state.state_representation import StateVector2D
from nuplan.common.actor_state.state_representation import TimePoint
from nuplan.common.actor_state.waypoint import Waypoint
from nuplan.common.actor_state.vehicle_parameters import VehicleParameters, get_pacifica_parameters
from nuplan.common.actor_state.tracked_objects import TrackedObject
from nuplan.database.utils.measure import angle_diff
from nuplan_king.planning.simulation.adversary_optimizer.agent_tracker.abstract_tracker import AbstractTracker
from nuplan.planning.simulation.controller.tracker.tracker_utils import (
    _generate_profile_from_initial_condition_and_derivatives,
    get_velocity_curvature_profiles_with_derivatives_from_poses,
)
from nuplan.planning.simulation.trajectory.abstract_trajectory import AbstractTrajectory
from nuplan.common.actor_state.state_representation import StateSE2
from nuplan.common.actor_state.dynamic_car_state import get_velocity_shifted, get_acceleration_shifted
from nuplan.common.geometry.transform import translate_longitudinally_and_laterally, translate_longitudinally

DoubleMatrix = npt.NDArray[np.float64]

logger = logging.getLogger(__name__)


def from_center_to_rear(pose: StateSE2) -> StateSE2:
    _rear_length = 14.61
    return translate_longitudinally(pose, -_rear_length)

def from_rear_to_center(pose: StateSE2) -> StateSE2:
    _rear_length = 14.61
    return translate_longitudinally(pose, _rear_length)

# the discretization_time should be the same as interval_time considered in the tracker
def get_interpolated_reference_trajectory_poses_from_center_to_rear(
    trajectory: AbstractTrajectory,
    discretization_time: float,
) -> Tuple[DoubleMatrix, DoubleMatrix]:
    """
    Resamples the reference trajectory at discretization_time resolution.
    It will return N times and poses, where N is a function of the trajectory duration and the discretization time.
    :param trajectory: The full trajectory from which we perform pose interpolation.
    :param discretization_time: [s] The discretization time for resampling the trajectory.
    :return An array of times in seconds (N) and an array of associated poses (N,3), sampled at the discretization time.
    """
    start_time_point = trajectory.start_time
    end_time_point = trajectory.end_time

    delta_time_point = TimePoint(int(discretization_time * 1e6))

    interpolation_times_us = np.arange(start_time_point.time_us+100, end_time_point.time_us, delta_time_point.time_us)

    # Adds extra state if it aligns with discretization time
    if interpolation_times_us[-1] + delta_time_point.time_us <= end_time_point.time_us:
        interpolation_times_us = np.append(
            interpolation_times_us, interpolation_times_us[-1] + delta_time_point.time_us
        )

    interpolation_time_points = [TimePoint(t_us) for t_us in interpolation_times_us]

    states: List[Waypoint] = trajectory.get_state_at_times(interpolation_time_points)

    poses_interp = [[*from_center_to_rear(state.center)] for state in states]
    # poses_interp = [[*state.center] for state in states]

    return interpolation_times_us / 1e6, np.array(poses_interp)




class LateralStateIndex(IntEnum):
    """
    Index mapping for the lateral dynamics state vector.
    """

    LATERAL_ERROR = 0  # [m] The lateral error with respect to the planner centerline at the vehicle's rear axle center.
    HEADING_ERROR = 1  # [rad] The heading error "".
    STEERING_ANGLE = 2  # [rad] The wheel angle relative to the longitudinal axis of the vehicle.


class LQRTracker(AbstractTracker):
    """
    Implements an LQR tracker for a kinematic bicycle model.

    We decouple into two subsystems, longitudinal and lateral, with small angle approximations for linearization.
    We then solve two sequential LQR subproblems to find acceleration and steering rate inputs.

    Longitudinal Subsystem:
        States: [velocity]
        Inputs: [acceleration]
        Dynamics (continuous time):
            velocity_dot = acceleration

    Lateral Subsystem (After Linearization/Small Angle Approximation):
        States: [lateral_error, heading_error, steering_angle]
        Inputs: [steering_rate]
        Parameters: [velocity, curvature]
        Dynamics (continuous time):
            lateral_error_dot  = velocity * heading_error
            heading_error_dot  = velocity * (steering_angle / wheelbase_length - curvature)
            steering_angle_dot = steering_rate

    The continuous time dynamics are discretized using Euler integration and zero-order-hold on the input.
    In case of a stopping reference, we use a simplified stopping P controller instead of LQR.

    The final control inputs passed on to the motion model are:
        - acceleration
        - steering_rate
    """

    def __init__(
        self,
        q_longitudinal: npt.NDArray[np.float64],
        r_longitudinal: npt.NDArray[np.float64],
        q_lateral: npt.NDArray[np.float64],
        r_lateral: npt.NDArray[np.float64],
        discretization_time: float,
        tracking_horizon: int,
        jerk_penalty: float,
        curvature_rate_penalty: float,
        stopping_proportional_gain: float,
        stopping_velocity: float,
        vehicle: VehicleParameters = get_pacifica_parameters(),
    ):
        """
        Constructor for LQR controller
        :param q_longitudinal: The weights for the Q matrix for the longitudinal subystem.
        :param r_longitudinal: The weights for the R matrix for the longitudinal subystem.
        :param q_lateral: The weights for the Q matrix for the lateral subystem.
        :param r_lateral: The weights for the R matrix for the lateral subystem.
        :param discretization_time: [s] The time interval used for discretizing the continuous time dynamics.
        :param tracking_horizon: How many discrete time steps ahead to consider for the LQR objective.
        :param stopping_proportional_gain: The proportional_gain term for the P controller when coming to a stop.
        :param stopping_velocity: [m/s] The velocity below which we are deemed to be stopping and we don't use LQR.
        :param vehicle: Vehicle parameters
        """
        # Longitudinal LQR Parameters
        assert len(q_longitudinal) == 1, "q_longitudinal should have 1 element (velocity)."
        assert len(r_longitudinal) == 1, "r_longitudinal should have 1 element (acceleration)."
        self._q_longitudinal: npt.NDArray[np.float64] = np.diag(q_longitudinal)
        self._r_longitudinal: npt.NDArray[np.float64] = np.diag(r_longitudinal)

        # Lateral LQR Parameters
        assert len(q_lateral) == 3, "q_lateral should have 3 elements (lateral_error, heading_error, steering_angle)."
        assert len(r_lateral) == 1, "r_lateral should have 1 element (steering_rate)."
        self._q_lateral: npt.NDArray[np.float64] = np.diag(q_lateral)
        self._r_lateral: npt.NDArray[np.float64] = np.diag(r_lateral)

        # Validate cost matrices for longitudinal and lateral LQR.
        for attr in ["_q_lateral", "_q_longitudinal"]:
            assert np.all(np.diag(getattr(self, attr)) >= 0.0), f"self.{attr} must be positive semidefinite."

        for attr in ["_r_lateral", "_r_longitudinal"]:
            assert np.all(np.diag(getattr(self, attr)) > 0.0), f"self.{attr} must be positive definite."

        # Common LQR Parameters
        # Note we want a horizon > 1 so that steering rate actually can impact lateral/heading error in discrete time.
        assert discretization_time > 0.0, "The discretization_time should be positive."
        assert (
            tracking_horizon > 1
        ), "We expect the horizon to be greater than 1 - else steering_rate has no impact with Euler integration."
        self._discretization_time = discretization_time
        self._tracking_horizon = tracking_horizon
        self._wheel_base = vehicle.wheel_base

        # Velocity/Curvature Estimation Parameters
        assert jerk_penalty > 0.0, "The jerk penalty must be positive."
        assert curvature_rate_penalty > 0.0, "The curvature rate penalty must be positive."
        self._jerk_penalty = jerk_penalty
        self._curvature_rate_penalty = curvature_rate_penalty

        # Stopping Controller Parameters
        assert stopping_proportional_gain > 0, "stopping_proportional_gain has to be greater than 0."
        assert stopping_velocity > 0, "stopping_velocity has to be greater than 0."
        self._stopping_proportional_gain = stopping_proportional_gain
        self._stopping_velocity = stopping_velocity

    def track_trajectory(
        self,
        current_timepoint: TimePoint,
        next_timepoint: TimePoint,
        initial_state: Waypoint,
        trajectory: AbstractTrajectory,
        initial_steering_angle: float,
    ) -> DynamicCarState:
        """Inherited, see superclass."""
        initial_velocity, initial_lateral_state_vector = self._compute_initial_velocity_and_lateral_state(
            current_timepoint, next_timepoint, initial_state, trajectory, initial_steering_angle
        )


        reference_velocity, curvature_profile = self._compute_reference_velocity_and_curvature_profile(
            current_timepoint, trajectory
        )

        should_stop = reference_velocity <= self._stopping_velocity and initial_velocity <= self._stopping_velocity

        if should_stop:
            accel_cmd, steering_rate_cmd = self._stopping_controller(initial_velocity, reference_velocity)
        else:
            accel_cmd = self._longitudinal_lqr_controller(initial_velocity, reference_velocity)
            velocity_profile = _generate_profile_from_initial_condition_and_derivatives(
                initial_condition=initial_velocity,
                derivatives=np.ones(self._tracking_horizon, dtype=np.float64) * accel_cmd,
                discretization_time=self._discretization_time,
            )[: self._tracking_horizon]
            steering_rate_cmd = self._lateral_lqr_controller(
                initial_lateral_state_vector, velocity_profile, curvature_profile
            )
        
        throttle = accel_cmd # for the rear axle
        steer = steering_rate_cmd # for the rear axle
        # TODO
        # converting to throttle and steer for the center

        '''
        _rear_length = 1.127
        displacement = StateVector2D(-_rear_length, 0.0)
        throttle = get_acceleration_shifted(
            displacement, throttle, initial_state.angular_velocity, angular_acceleration
        )
        steer = ....
        '''

        return throttle, steer
        

    def _compute_initial_velocity_and_lateral_state(
        self,
        current_timepoint: TimePoint,
        next_timepoint: TimePoint,
        initial_state: Waypoint,
        trajectory: AbstractTrajectory,
        initial_steering_angle: float,
    ) -> Tuple[float, npt.NDArray[np.float64]]:
        """
        This method projects the initial tracking error into vehicle/Frenet frame.  It also extracts initial velocity.
        :param current_timepoint: Used to get the current time.
        :param initial_state: The current state for agent.
        :param trajectory: The reference trajectory we are tracking.
        :return: Initial velocity [m/s] and initial lateral state.
        """

        
        # Get initial trajectory state => of type interpolatable_state
        initial_trajectory_state: Waypoint = trajectory.get_state_at_time(current_timepoint)
        next_trajectory_state: Waypoint = trajectory.get_state_at_time(next_timepoint)
        # Determine initial error state.
        # derieving the x and y of the rear axle from the center of the bounding box of the vehicle


        _rear_length = 1.127
        _rear_axle_to_center_dist = 14.61
        rear_initial_state:StateSE2 = from_center_to_rear(initial_state.center)
        rear_initial_trajectory_state:StateSE2 = from_center_to_rear(initial_trajectory_state.center)
        rear_next_trajectory_state:StateSE2 = from_center_to_rear(next_trajectory_state.center)
        x_error = rear_initial_state.x - rear_initial_trajectory_state.x
        y_error = rear_initial_state.y - rear_initial_trajectory_state.y
        heading_reference = rear_initial_trajectory_state.heading # the interpolatable states of the predicted trajectory may be of type waypoint and not ego state

        # we only have rear_axle.heading for the ego_state and not the waypoint
        '''
        x_error = initial_state.center.x - initial_trajectory_state.x
        y_error = initial_state.center.y - initial_trajectory_state.y
        heading_reference = initial_trajectory_state.heading
        '''
        
        
        lateral_error = -x_error * np.sin(heading_reference) + y_error * np.cos(heading_reference)
        heading_error = angle_diff(rear_initial_state.heading, heading_reference, 2 * np.pi)
        # print('heading error ', heading_error, '  ', rear_initial_state.heading, '   ', heading_reference)
        # heading_error = angle_diff(initial_state.center.heading, heading_reference, 2*np.pi)
        

        '''
        if np.isnan(initial_state._angular_velocity):
            # approximating the angular velocity of the agent because it is nan
            initial_state._angular_velocity = (next_trajectory_state.heading - initial_trajectory_state.heading)/(next_timepoint.time_s - current_timepoint.time_s        # Return initial velocity and lateral state vector.
        # original : initial_velocity = initial_state.dynamic_car_state.rear_axle_velocity_2d.x
        '''
        
        _angular_velocity = (next_trajectory_state.heading - initial_state.heading)/(next_timepoint.time_s - current_timepoint.time_s)
        # displacement: The displacement vector from the reference to the query point
        displacement_array = np.array([-_rear_axle_to_center_dist * np.cos(initial_state.heading), -_rear_axle_to_center_dist * np.sin(initial_state.heading)])
        displacement = StateVector2D(displacement_array[0], displacement_array[1])
        initial_velocity_state = get_velocity_shifted(displacement, initial_state.velocity, _angular_velocity)
        initial_velocity = np.hypot(initial_velocity_state.x, initial_velocity_state.y)

        # initial_velocity = np.hypot(initial_state.velocity.x, initial_state.velocity.y)

        # the tire_steering_angle is considered to be the heading of the vehicle at the next timestep.
        # # steering_angle = tan-1(wheel_base*(h2-h1)/(t*vel_x))
        # approx_tire_steering_angle = np.arctan(3.089*(next_trajectory_state.center.heading-initial_trajectory_state.center.heading)/((next_timepoint.time_s - current_timepoint.time_s)*initial_velocity+1e-6))
        # approx_tire_steering_angle = np.arctan(3.089*(rear_next_trajectory_state.heading-rear_initial_trajectory_state.heading)/((next_timepoint.time_s - current_timepoint.time_s)*initial_velocity+1e-6))
        # print(f'******** {approx_tire_steering_angle} approx_tire_steering_angle ********')
        initial_lateral_state_vector: npt.NDArray[np.float64] = np.array(
            [
                lateral_error,
                heading_error,
                initial_steering_angle,
                # (final_trajectory_state.center.heading-initial_trajectory_state.center.heading) 
                # original : initial_state.tire_steering_angle,
            ],
            dtype=np.float64,
        )

        return initial_velocity, initial_lateral_state_vector

    def _compute_reference_velocity_and_curvature_profile(
        self,
        current_timepoint: TimePoint,
        trajectory: AbstractTrajectory,
    ) -> Tuple[float, npt.NDArray[np.float64]]:
        """
        This method computes reference velocity and curvature profile based on the reference trajectory.
        We use a lookahead time equal to self._tracking_horizon * self._discretization_time.
        :param current_timepoint: Used to get the current time.
        :param trajectory: The reference trajectory we are tracking.
        :return: The reference velocity [m/s] and curvature profile [rad] to track.
        """
        times_s, poses = get_interpolated_reference_trajectory_poses_from_center_to_rear(trajectory, self._discretization_time)

        (
            velocity_profile,
            acceleration_profile,
            curvature_profile,
            curvature_rate_profile,
        ) = get_velocity_curvature_profiles_with_derivatives_from_poses(
            discretization_time=self._discretization_time,
            poses=poses,
            jerk_penalty=self._jerk_penalty,
            curvature_rate_penalty=self._curvature_rate_penalty,
        )

        # what is really the reference time?
        reference_time = current_timepoint.time_s + self._tracking_horizon * self._discretization_time
        reference_velocity = np.interp(reference_time, times_s[:-1], velocity_profile)

        profile_times = [
            current_timepoint.time_s + x * self._discretization_time for x in range(self._tracking_horizon)
        ]
        reference_curvature_profile = np.interp(profile_times, times_s[:-1], curvature_profile)

        return float(reference_velocity), reference_curvature_profile

    def _stopping_controller(
        self,
        initial_velocity: float,
        reference_velocity: float,
    ) -> Tuple[float, float]:
        """
        Apply proportional controller when at near-stop conditions.
        :param initial_velocity: [m/s] The current velocity of agent.
        :param reference_velocity: [m/s] The reference velocity to track.
        :return: Acceleration [m/s^2] and zero steering_rate [rad/s] command.
        """
        accel = -self._stopping_proportional_gain * (initial_velocity - reference_velocity)
        return accel, 0.0

    def _longitudinal_lqr_controller(
        self,
        initial_velocity: float,
        reference_velocity: float,
    ) -> float:
        """
        This longitudinal controller determines an acceleration input to minimize velocity error at a lookahead time.
        :param initial_velocity: [m/s] The current velocity of agent.
        :param reference_velocity: [m/s] The reference_velocity to track at a lookahead time.
        :return: Acceleration [m/s^2] command based on LQR.
        """
        # We assume that we hold the acceleration constant for the entire tracking horizon.
        # Given this, we can show the following where N = self._tracking_horizon and dt = self._discretization_time:
        # velocity_N = velocity_0 + (N * dt) * acceleration
        A: npt.NDArray[np.float64] = np.array([1.0], dtype=np.float64)
        B: npt.NDArray[np.float64] = np.array([self._tracking_horizon * self._discretization_time], dtype=np.float64)

        accel_cmd = self._solve_one_step_lqr(
            initial_state=np.array([initial_velocity], dtype=np.float64),
            reference_state=np.array([reference_velocity], dtype=np.float64),
            Q=self._q_longitudinal,
            R=self._r_longitudinal,
            A=A,
            B=B,
            g=np.zeros(1, dtype=np.float64),
            angle_diff_indices=[],
        )

        return float(accel_cmd)

    def _lateral_lqr_controller(
        self,
        initial_lateral_state_vector: npt.NDArray[np.float64],
        velocity_profile: npt.NDArray[np.float64],
        curvature_profile: npt.NDArray[np.float64],
    ) -> float:
        """
        This lateral controller determines a steering_rate input to minimize lateral errors at a lookahead time.
        It requires a velocity sequence as a parameter to ensure linear time-varying lateral dynamics.
        :param initial_lateral_state_vector: The current lateral state of agent.
        :param velocity_profile: [m/s] The velocity over the entire self._tracking_horizon-step lookahead.
        :param curvature_profile: [rad] The curvature over the entire self._tracking_horizon-step lookahead..
        :return: Steering rate [rad/s] command based on LQR.
        """
        assert len(velocity_profile) == self._tracking_horizon, (
            f"The linearization velocity sequence should have length {self._tracking_horizon} "
            f"but is {len(velocity_profile)}."
        )
        assert len(curvature_profile) == self._tracking_horizon, (
            f"The linearization curvature sequence should have length {self._tracking_horizon} "
            f"but is {len(curvature_profile)}."
        )

        # Set up the lateral LQR problem using the constituent linear time-varying (affine) system dynamics.
        # Ultimately, we'll end up with the following problem structure where N = self._tracking_horizon:
        # lateral_error_N = A @ lateral_error_0 + B @ steering_rate + g
        n_lateral_states = len(LateralStateIndex)
        I: npt.NDArray[np.float64] = np.eye(n_lateral_states, dtype=np.float64)

        A: npt.NDArray[np.float64] = I
        B: npt.NDArray[np.float64] = np.zeros((n_lateral_states, 1), dtype=np.float64)
        g: npt.NDArray[np.float64] = np.zeros(n_lateral_states, dtype=np.float64)

        # Convenience aliases for brevity.
        idx_lateral_error = LateralStateIndex.LATERAL_ERROR
        idx_heading_error = LateralStateIndex.HEADING_ERROR
        idx_steering_angle = LateralStateIndex.STEERING_ANGLE

        input_matrix: npt.NDArray[np.float64] = np.zeros((n_lateral_states, 1), np.float64)
        input_matrix[idx_steering_angle] = self._discretization_time

        for index_step, (velocity, curvature) in enumerate(zip(velocity_profile, curvature_profile)):
            state_matrix_at_step: npt.NDArray[np.float64] = np.eye(n_lateral_states, dtype=np.float64)
            state_matrix_at_step[idx_lateral_error, idx_heading_error] = velocity * self._discretization_time
            state_matrix_at_step[idx_heading_error, idx_steering_angle] = (
                velocity * self._discretization_time / self._wheel_base
            )

            affine_term: npt.NDArray[np.float64] = np.zeros(n_lateral_states, dtype=np.float64)
            affine_term[idx_heading_error] = -velocity * curvature * self._discretization_time

            A = state_matrix_at_step @ A
            B = state_matrix_at_step @ B + input_matrix
            g = state_matrix_at_step @ g + affine_term

        steering_rate_cmd = self._solve_one_step_lqr(
            initial_state=initial_lateral_state_vector,
            reference_state=np.zeros(n_lateral_states, dtype=np.float64),
            Q=self._q_lateral,
            R=self._r_lateral,
            A=A,
            B=B,
            g=g,
            angle_diff_indices=[idx_heading_error, idx_steering_angle],
        )

        return float(steering_rate_cmd)

    @staticmethod
    def _solve_one_step_lqr(
        initial_state: npt.NDArray[np.float64],
        reference_state: npt.NDArray[np.float64],
        Q: npt.NDArray[np.float64],
        R: npt.NDArray[np.float64],
        A: npt.NDArray[np.float64],
        B: npt.NDArray[np.float64],
        g: npt.NDArray[np.float64],
        angle_diff_indices: List[int] = [],
    ) -> npt.NDArray[np.float64]:
        """
        This function uses LQR to find an optimal input to minimize tracking error in one step of dynamics.
        The dynamics are next_state = A @ initial_state + B @ input + g and our target is the reference_state.
        :param initial_state: The current state.
        :param reference_state: The desired state in 1 step (according to A,B,g dynamics).
        :param Q: The state tracking 2-norm cost matrix.
        :param R: The input 2-norm cost matrix.
        :param A: The state dynamics matrix.
        :param B: The input dynamics matrix.
        :param g: The offset/affine dynamics term.
        :param angle_diff_indices: The set of state indices for which we need to apply angle differences, if defined.
        :return: LQR optimal input for the 1-step problem.
        """
        state_error_zero_input = A @ initial_state + g - reference_state

        for angle_diff_index in angle_diff_indices:
            state_error_zero_input[angle_diff_index] = angle_diff(
                state_error_zero_input[angle_diff_index], 0.0, 2 * np.pi
            )

        lqr_input = -np.linalg.inv(B.T @ Q @ B + R) @ B.T @ Q @ state_error_zero_input
        return lqr_input