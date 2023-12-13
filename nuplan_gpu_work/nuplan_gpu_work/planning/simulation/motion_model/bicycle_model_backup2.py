import torch
from torch.nn import functional as F
from typing import Dict

# Builds on the implentation from "World on Rails", see king/external_code/wor.
device = 'cuda' if torch.cuda.is_available() else 'cpu'


# even whent the throttle and all is zero, the vehicle still goes forward, why?
class BicycleModel(torch.nn.Module):
    """
    """
    def __init__(self, delta_t):
        super().__init__()
        self.delta_t = torch.tensor([delta_t], device=device)
        
        self.accel_time_constant: float = 0.2 # torch.tensor([0.2], device=device)
        self.steering_angle_time_constant: float = 0.05 # torch.tensor([0.05], device=device)
        self.that_agent_vel_x = []
        self.that_agent_heading = []
        self.that_agent_steering_rate = []
        self.that_agent_steering_angle = []
        self.that_agent_longitudinal_vel = []
        self.that_agent_y_vel = []
        self.that_agent_updated_steering_rate = []
        self.sin_beta = []

        # values taken from "World On Rails"
        self.register_buffer("front_wb", torch.tensor([-0.090769015], device=device))
        self.register_buffer("rear_wb", torch.tensor([1.4178275], device=device))
        self.register_buffer("steer_gain", torch.tensor([0.36848336], device=device))
        self.register_buffer("brake_accel", torch.tensor([-4.952399], device=device))
        self.register_buffer("throt_accel", torch.tensor([[0.5633837]], device=device))

    def forward(self, state: Dict[str, torch.TensorType], actions: Dict[str, torch.TensorType], track_token:str, iter:int, plotter=True): # discarding the is_terminated option for now
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
        
        ideal_steering_angle = state["steering_angle"][:,:,0] + self.delta_t * actions['steer'][:,:,0]
        # updated_steering_angle = (
        #     self.delta_t / (self.delta_t + self.steering_angle_time_constant) * (ideal_steering_angle - state["steering_angle"][:,:,0])
        #     + state["steering_angle"][:,:,0]
        # )

        updated_steering_angle = ideal_steering_angle
        updated_steering_rate = (updated_steering_angle - state["steering_angle"][:,:,0]) / self.delta_t

      
        # updating the throttle, due to the delay in control
        ideal_accel = actions['throttle'][:,:,0]
        # ideal_accel = accel
        
        # updated_accel = self.delta_t / (self.delta_t + self.accel_time_constant) * (ideal_accel - state["accel"][:,:,0]) + state["accel"][:,:,0]
        updated_accel = ideal_accel

        # START OF DIRTY CODE
        # longitudinal_speed = torch.norm(state["vel"][:,:,:], dim=-1)
        longitudinal_speed = state["speed"][:,:,:]

        beta = updated_steering_rate


        wheel_base = torch.tensor([3.089]).to(device=device).to(dtype=torch.float64)

        min_ = - torch.pi
        # why cannot go backwards!!
        # longitudinal_speed = F.softplus(longitudinal_speed + updated_accel*self.delta_t, beta=7)
        # longitudinal_speed = F.softplus(longitudinal_speed, beta=7) + updated_accel * self.delta_t
        longitudinal_speed = longitudinal_speed + updated_accel * self.delta_t
        lhs = (state['yaw'][:,:,0] + longitudinal_speed/wheel_base*self.delta_t*torch.sin(beta) - min_) % (2 * torch.pi) + min_

        state['vel'] = torch.cat([torch.unsqueeze(longitudinal_speed*torch.cos(lhs), dim=-1), torch.unsqueeze(longitudinal_speed*torch.sin(lhs), dim=-1)], dim=-1)
        state['yaw'] = torch.unsqueeze(lhs, dim=-1)
        

        # longitudinal_speed = torch.norm(state["vel"][:,:,:], dim=-1) hiiiiiii
     
        # x_dot = torch.cos(state["yaw"][:,:,0]+beta)*longitudinal_speed hiiiiii
        # y_dot = torch.sin(state["yaw"][:,:,0]+beta)*longitudinal_speed hiiiiiii

        # we did not "use the velocity in the above way, by considering
        # the velocity was never used, but only the speed
        
        x_dot = state['vel'][:,:,0]
        y_dot = state['vel'][:,:,1]
        state['pos'] = torch.cat([torch.unsqueeze(state['pos'][:,:,0] + x_dot*self.delta_t, dim=-1), torch.unsqueeze(state['pos'][:,:,1] + y_dot*self.delta_t, dim=-1)], dim=-1)

        
        print('longitudinal ', longitudinal_speed[:,3,:], ' and accel ', updated_accel[:,3])
        state['speed'][:,:,:] = longitudinal_speed
        state['steering_angle'] = torch.unsqueeze(torch.clamp(updated_steering_angle, min=-torch.pi/3, max=torch.pi/3), dim=-1)
        state['accel'] = torch.unsqueeze(updated_accel, dim=-1)



        return state
    

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

    
        # updating the steering angle of the state, due to the delay in control
        ideal_steering_angle = state["steering_angle"][:,:,0] + self.delta_t * actions['steer'][:,:,0]
        # updated_steering_angle = (
        #     self.delta_t / (self.delta_t + self.steering_angle_time_constant) * (ideal_steering_angle - state["steering_angle"][:,:,0])
        #     + state["steering_angle"][:,:,0]
        # )
        updated_steering_angle = ideal_steering_angle

        updated_steering_rate = (updated_steering_angle - state["steering_angle"][:,:,0]) / self.delta_t

      
        # updating the throttle, due to the delay in control
        ideal_accel = actions['throttle'][:,:,0]
        # ideal_accel = accel
        
        # updated_accel = self.delta_t / (self.delta_t + self.accel_time_constant) * (ideal_accel - state["accel"][:,:,0]) + state["accel"][:,:,0]
        updated_accel = ideal_accel

        # START OF DIRTY CODE
        # longitudinal_speed = torch.norm(state["vel"][:,:,:], dim=-1)
        longitudinal_speed = state["speed"][:,:,0]

        beta = updated_steering_rate


        wheel_base = torch.tensor([3.089]).to(device=device).to(dtype=torch.float64)

        min_ = - torch.pi
        # longitudinal_speed = F.softplus(longitudinal_speed + updated_accel*self.delta_t, beta=7)
        # longitudinal_speed = F.softplus(longitudinal_speed, beta=7) + updated_accel * self.delta_t
        longitudinal_speed = longitudinal_speed + updated_accel * self.delta_t

        lhs = (state['yaw'][:,:,0] + longitudinal_speed/wheel_base*self.delta_t*torch.sin(beta) - min_) % (2 * torch.pi) + min_

        state['vel'] = torch.cat([torch.unsqueeze(longitudinal_speed*torch.cos(lhs), dim=-1), torch.unsqueeze(longitudinal_speed*torch.sin(lhs), dim=-1)], dim=-1)
        state['yaw'] = torch.unsqueeze(lhs, dim=-1)
        

        # longitudinal_speed = torch.norm(state["vel"][:,:,:], dim=-1)  hiiiiiiiii
     
        x_dot = torch.cos(state["yaw"][:,:,0]+beta)*longitudinal_speed
        y_dot = torch.sin(state["yaw"][:,:,0]+beta)*longitudinal_speed
        state['pos'] = torch.cat([torch.unsqueeze(state['pos'][:,:,0] + x_dot*self.delta_t, dim=-1), torch.unsqueeze(state['pos'][:,:,1] + y_dot*self.delta_t, dim=-1)], dim=-1)

        
        # print('longitudinal ', longitudinal_speed[:,20], ' and accel ', updated_accel[:,3])
        state['speed'][:,:,0] = longitudinal_speed
        state['steering_angle'] = torch.unsqueeze(torch.clamp(updated_steering_angle, min=-torch.pi/3, max=torch.pi/3), dim=-1)
        state['accel'] = torch.unsqueeze(updated_accel, dim=-1)



        return state
    
    def that_agent(self):
        return self.sin_beta, self.that_agent_y_vel, self.that_agent_longitudinal_vel, self.that_agent_vel_x, self.that_agent_heading, self.that_agent_steering_angle, self.that_agent_steering_rate, self.that_agent_updated_steering_rate