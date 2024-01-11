import torch
from torch.nn import functional as F
from typing import Dict

# Builds on the implentation from "World on Rails", see king/external_code/wor.
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class BicycleModel(torch.nn.Module):
    """
    """
    def __init__(self, delta_t):
        super().__init__()
        self.delta_t = torch.tensor([delta_t], device=device)
        
        self.accel_time_constant: float = 0.2 # torch.tensor([0.2], device=device)
        self.steering_angle_time_constant: float = 0.05 # torch.tensor([0.05], device=device)

    # there are some variables that are requires grad false in the calculation of the states, but then why the steer grad was not zero?
    def forward_reconstruction(self, state: Dict[str, torch.TensorType], actions: Dict[str, torch.TensorType]): # discarding the is_terminated option for now
        """
        :param state: state of the agnets at the current iteration
        :param actions: actions taken by different agents at the current iteration

        Computes the next from the current state given associated actions using
        the bicycle model tuned for CARLA from the "World On Rails" paper

        https://arxiv.org/abs/2105.00636

        Args:
            -
        Returns:
            -
        """

        state_return = {}

    
        # updating the steering angle of the state, due to the delay in control
        ideal_steering_angle = state["steering_angle"][:,:,0] + self.delta_t * actions['steer'][:,:,0]
        # updated_steering_angle = (
        #     self.delta_t / (self.delta_t + self.steering_angle_time_constant) * (ideal_steering_angle - state["steering_angle"][:,:,0])
        #     + state["steering_angle"][:,:,0]
        # )
        updated_steering_angle = ideal_steering_angle

        updated_steering_rate = (ideal_steering_angle - state["steering_angle"][:,:,0]) / self.delta_t

      
        # updating the throttle, due to the delay in control
        ideal_accel = actions['throttle'][:,:,0]
        # ideal_accel = accel
        
        # updated_accel = self.delta_t / (self.delta_t + self.accel_time_constant) * (ideal_accel - state["accel"][:,:,0]) + state["accel"][:,:,0]
        updated_accel = ideal_accel

        # START OF DIRTY CODE
        # longitudinal_speed = torch.norm(state["vel"][:,:,:], dim=-1)
        longitudinal_speed = state["speed"][:,:,0]


        beta = updated_steering_rate
        # beta = torch.atan(torch.tan(updated_steering_rate)/2.0)

        wheel_base = torch.tensor([3.89], dtype=torch.float64).to(device=device)

        min_ = - torch.pi
        # longitudinal_speed = F.softplus(longitudinal_speed + updated_accel*self.delta_t, beta=7)
        # longitudinal_speed = longitudinal_speed + updated_accel * self.delta_t
        # lhs = (state['yaw'][:,:,0] + longitudinal_speed/wheel_base*self.delta_t*torch.tan(updated_steering_angle) - min_) % (2 * torch.pi) + min_ # as in nuplan version of bm

        longitudinal_speed = longitudinal_speed + updated_accel * self.delta_t

        # lhs = (state['yaw'][:,:,0] + longitudinal_speed/wheel_base*self.delta_t*torch.sin(beta)/10.0 - min_) % (2 * torch.pi) + min_ # as in king's version of bm
        lhs = (state['yaw'][:,:,0] + longitudinal_speed/wheel_base*self.delta_t*torch.sin(beta)) # as in king's version of bm
        # print('the alone yaw ', state['yaw'].item(), '   ', (longitudinal_speed/wheel_base*self.delta_t*torch.sin(beta)).item())

        state_return['vel'] = torch.cat([torch.unsqueeze(longitudinal_speed*torch.cos(lhs), dim=-1), torch.unsqueeze(longitudinal_speed*torch.sin(lhs), dim=-1)], dim=-1)        
        state_return['yaw'] = torch.unsqueeze(lhs, dim=-1)
        # print('the yaw in bm ***** **** **** **** ', state_return['yaw'].item(), '  ', state['yaw'].item())
     
        # x_dot = torch.cos(state["yaw"][:,:,0]+beta)*longitudinal_speed
        # y_dot = torch.sin(state["yaw"][:,:,0]+beta)*longitudinal_speed

        
        
        
        # x_dot = torch.cos(state_return["yaw"][:,:,0]+beta)*longitudinal_speed
        # y_dot = torch.sin(state_return["yaw"][:,:,0]+beta)*longitudinal_speed


        x_dot = torch.cos(state_return["yaw"][:,:,0])*longitudinal_speed # of size B*N
        y_dot = torch.sin(state_return["yaw"][:,:,0])*longitudinal_speed
        longitudinal_speed = torch.norm(torch.stack([x_dot, y_dot], dim=-1), dim=-1)
        state_return['pos'] = torch.cat([torch.unsqueeze(state['pos'][:,:,0] + x_dot*self.delta_t, dim=-1), torch.unsqueeze(state['pos'][:,:,1] + y_dot*self.delta_t, dim=-1)], dim=-1)


        

        state_return['steering_angle'] = torch.unsqueeze(torch.clamp(updated_steering_angle, min=-torch.pi/3, max=torch.pi/3), dim=-1)
        state_return['accel'] = torch.unsqueeze(updated_accel, dim=-1)
        state_return['speed'] = torch.unsqueeze(longitudinal_speed, dim=-1)



        return state_return
    # there are some variables that are requires grad false in the calculation of the states, but then why the steer grad was not zero?
    def forward_all(self, state: Dict[str, torch.TensorType], actions: Dict[str, torch.TensorType]): # discarding the is_terminated option for now
        """
        :param state: state of the agnets at the current iteration
        :param actions: actions taken by different agents at the current iteration

        Computes the next from the current state given associated actions using
        the bicycle model tuned for CARLA from the "World On Rails" paper

        https://arxiv.org/abs/2105.00636

        Args:
            -
        Returns:
            -
        """

        state_return = {}

    
        # updating the steering angle of the state, due to the delay in control
        ideal_steering_angle = state["steering_angle"][:,:,0] + self.delta_t * actions['steer'][:,:,0]
        # updated_steering_angle = (
        #     self.delta_t / (self.delta_t + self.steering_angle_time_constant) * (ideal_steering_angle - state["steering_angle"][:,:,0])
        #     + state["steering_angle"][:,:,0]
        # )
        updated_steering_angle = ideal_steering_angle

        updated_steering_rate = (ideal_steering_angle - state["steering_angle"][:,:,0]) / self.delta_t

      
        # updating the throttle, due to the delay in control
        ideal_accel = actions['throttle'][:,:,0]
        # ideal_accel = accel
        
        # updated_accel = self.delta_t / (self.delta_t + self.accel_time_constant) * (ideal_accel - state["accel"][:,:,0]) + state["accel"][:,:,0]
        updated_accel = ideal_accel

        # START OF DIRTY CODE
        # longitudinal_speed = torch.norm(state["vel"][:,:,:], dim=-1)
        longitudinal_speed = state["speed"][:,:,0]


        beta = updated_steering_rate
        # beta = torch.atan(torch.tan(updated_steering_rate)/2.0)

        wheel_base = torch.tensor([3.89], dtype=torch.float64).to(device=device)

        min_ = - torch.pi
        # longitudinal_speed = F.softplus(longitudinal_speed + updated_accel*self.delta_t, beta=7)
        # longitudinal_speed = longitudinal_speed + updated_accel * self.delta_t
        # lhs = (state['yaw'][:,:,0] + longitudinal_speed/wheel_base*self.delta_t*torch.tan(updated_steering_angle) - min_) % (2 * torch.pi) + min_ # as in nuplan version of bm

        longitudinal_speed = longitudinal_speed + updated_accel * self.delta_t

        # lhs = (state['yaw'][:,:,0] + longitudinal_speed/wheel_base*self.delta_t*torch.sin(beta)/10.0 - min_) % (2 * torch.pi) + min_ # as in king's version of bm
        lhs = (state['yaw'][:,:,0] + longitudinal_speed/wheel_base*self.delta_t*torch.sin(beta)) # as in king's version of bm
        # print('the alone yaw ', state['yaw'].item(), '   ', (longitudinal_speed/wheel_base*self.delta_t*torch.sin(beta)).item())

        state_return['vel'] = torch.cat([torch.unsqueeze(longitudinal_speed*torch.cos(lhs), dim=-1), torch.unsqueeze(longitudinal_speed*torch.sin(lhs), dim=-1)], dim=-1)        
        state_return['yaw'] = torch.unsqueeze(lhs, dim=-1)
        # print('the yaw in bm ***** **** **** **** ', state_return['yaw'].item(), '  ', state['yaw'].item())
     
        # x_dot = torch.cos(state["yaw"][:,:,0]+beta)*longitudinal_speed
        # y_dot = torch.sin(state["yaw"][:,:,0]+beta)*longitudinal_speed

        
        
        
        # x_dot = torch.cos(state_return["yaw"][:,:,0]+beta)*longitudinal_speed
        # y_dot = torch.sin(state_return["yaw"][:,:,0]+beta)*longitudinal_speed


        x_dot = torch.cos(state_return["yaw"][:,:,0])*longitudinal_speed # of size B*N
        y_dot = torch.sin(state_return["yaw"][:,:,0])*longitudinal_speed
        longitudinal_speed = torch.norm(torch.stack([x_dot, y_dot], dim=-1), dim=-1) # comment this line to enable going backwards
        state_return['pos'] = torch.cat([torch.unsqueeze(state['pos'][:,:,0] + x_dot*self.delta_t, dim=-1), torch.unsqueeze(state['pos'][:,:,1] + y_dot*self.delta_t, dim=-1)], dim=-1)


        

        state_return['steering_angle'] = torch.unsqueeze(torch.clamp(updated_steering_angle, min=-torch.pi/3, max=torch.pi/3), dim=-1)
        state_return['accel'] = torch.unsqueeze(updated_accel, dim=-1)
        state_return['speed'] = torch.unsqueeze(longitudinal_speed, dim=-1)



        return state_return
    
    def that_agent(self):
        return self.sin_beta, self.that_agent_y_vel, self.that_agent_longitudinal_vel, self.that_agent_vel_x, self.that_agent_heading, self.that_agent_steering_angle, self.that_agent_steering_rate, self.that_agent_updated_steering_rate