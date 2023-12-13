import torch
from torch.nn import functional as F
from typing import Dict
import pdb


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

        '''
        is_terminated = torch.zeros(
            [1],
            dtype=torch.bool,
            device='cpu',
        )
        braking_ego = actions["brake"]
        braking_adv = torch.lt(actions["throttle"], torch.zeros_like(actions["throttle"]))
        accel = braking_ego * self.brake_accel + \
            braking_adv * -self.brake_accel * actions["throttle"] + \
            ~braking_adv * self.throt_accel * actions["throttle"]

        wheel = self.steer_gain * actions["steer"]
        beta = torch.atan(
            self.rear_wb/(self.front_wb+self.rear_wb) * torch.tan(wheel)
        )

        speed = torch.norm(state["vel"], dim=-1, keepdim=True)
        motion_components = torch.cat(
            [torch.cos(state["yaw"]+beta), torch.sin(state["yaw"]+beta)],
            dim=-1,
        )

        update_mask = ~is_terminated.view(-1, 1, 1)
        state["pos"] = state["pos"] + speed * motion_components * self.delta_t * update_mask
        state["yaw"] = state["yaw"] + speed / self.rear_wb * torch.sin(beta) * self.delta_t * update_mask

        speed = F.softplus(speed + accel * self.delta_t, beta=7)
        print('the dimension of speed * motion_components ', (speed * motion_components).shape)
        print('is the problem this ', ((speed * motion_components - state["vel"])).shape)
        state["vel"] = state["vel"] + (speed * motion_components - state["vel"]) * update_mask

        return state
        '''

        # updating the steering angle of the state, due to the delay in control
        # pdb.set_trace()
        ideal_steering_angle = state["steering_angle"][:,:,0] + self.delta_t * actions['steer'][:,:,0]
        # print('hi and hooy ******** ******** ******* ******* ')
        # print('hi and hooy ******** ******** ******* ******* ')
        # print('hi and hooy ******** ******** ******* ******* ')
        # print('hi and hooy ******** ******** ******* ******* ')
        # print('hi and hooy ******** ******** ******* ******* ')
        updated_steering_angle = (
            self.delta_t / (self.delta_t + self.steering_angle_time_constant) * (ideal_steering_angle - state["steering_angle"][:,:,0])
            + state["steering_angle"][:,:,0]
        )
        updated_steering_rate = (updated_steering_angle - state["steering_angle"][:,:,0]) / self.delta_t

        # adding the braking to the throttle, should change the formulas considering that we are only handling one agent
        '''
        braking_ego = actions["brake"][:,:,0]
        braking_adv = torch.lt(actions["throttle"][:,:,0], torch.zeros_like(actions["throttle"][:,:,0]))
        accel = braking_ego * self.brake_accel + \
            braking_adv * -self.brake_accel * actions["throttle"][:,:,0] + \
            ~braking_adv * self.throt_accel * actions["throttle"][:,:,0]
        '''
        # updating the throttle, due to the delay in control
        ideal_accel = actions['throttle'][:,:,0]
        # ideal_accel = accel
        updated_accel = self.delta_t / (self.delta_t + self.accel_time_constant) * (ideal_accel - state["accel"][:,:,0]) + state["accel"][:,:,0]

        
        braking_ego = actions["brake"][:,:,0]
        braking_adv = torch.lt(updated_accel, torch.zeros_like(updated_accel))
        updated_accel = braking_ego * self.brake_accel + \
            braking_adv * -self.brake_accel * updated_accel + \
            ~braking_adv * self.throt_accel * updated_accel
        
        # longitudinal_speed = torch.hypot(state["vel"][:,:,0], state["vel"][:,:,1])


        longitudinal_speed = torch.norm(state["vel"][:,:,:])
        # using the parameters from the king
        # beta = torch.atan(
        #     self.rear_wb/(self.front_wb+self.rear_wb) * torch.tan(updated_steering_rate*self.steer_gain)
        # )

        beta = updated_steering_rate*self.steer_gain
        # beta = updated_steering_rate

        # beta = torch.atan(
        #      0.5 * torch.tan(updated_steering_rate*self.steer_gain)
        # )

        wheel_base = torch.tensor([3.089]).to(device=device)
        '''
        longitudinal_speed = state["vel"][:,:,0]
        x_dot = longitudinal_speed * torch.cos(state["yaw"][:,:,0])
        y_dot = longitudinal_speed * torch.sin(state["yaw"][:,:,0])
        '''
        x_dot = torch.cos(state["yaw"][:,:,0]+beta)*longitudinal_speed
        y_dot = torch.sin(state["yaw"][:,:,0]+beta)*longitudinal_speed

        motion_components = torch.cat(
            [torch.cos(state["yaw"][:,:,0]+beta), torch.sin(state["yaw"][:,:,0]+beta)],
            dim=-1,
        )
        
        # yaw_dot = longitudinal_speed * torch.tan(state["steering_angle"][:,:,0]) / wheel_base
        # yaw_dot = longitudinal_speed * torch.tan(updated_steering_angle) / wheel_base
        # ['1f935c796c4e5258', '6e0f1dbf087e570f', '105f4ecbf9d05853', '4e86a1acfa645707', '60ca1d120e62535e']
        with torch.no_grad():
            if track_token in ['6e0f1dbf087e570f'] and plotter:
                self.that_agent_heading.append(state["yaw"][:,:,0])
                self.that_agent_updated_steering_rate.append(updated_steering_rate)
                self.that_agent_steering_rate.append(actions['steer'][:,:,0])
                self.that_agent_steering_angle.append(updated_steering_angle)
                self.that_agent_vel_x.append(x_dot)
                self.that_agent_longitudinal_vel.append(longitudinal_speed)
                self.that_agent_y_vel.append(state['vel'][:,:,1])
                print(f'iter {iter} the track token {track_token}, and the x_dot is {x_dot.data} and the y_dot is {y_dot.data} and updated_steering_angle {updated_steering_angle} and longitudinal_speed {longitudinal_speed}')
            


        # so, to obtain the tire steering angle from two headings h1 and h2:
        # (h2-h1)/t = vel_x * tan(steering_angle) / wheel_base  =>
        # steering_angle = tan-1(wheel_base*(h2-h1)/(t*vel_x))

        # state['pos'] = torch.cat([torch.unsqueeze(state['pos'][:,:,0] + x_dot*self.delta_t, dim=-1), torch.unsqueeze(state['pos'][:,:,1] + y_dot*self.delta_t, dim=-1)], dim=-1)
        state['pos'] = torch.unsqueeze(state['pos'][:,:,:] + motion_components*longitudinal_speed*self.delta_t, dim=-1)
        
        min_ = - torch.pi
        # lhs = (state['yaw'][:,:,0] + yaw_dot*self.delta_t - min_) % (2 * torch.pi) + min_

        with torch.no_grad():
            if track_token in ['6e0f1dbf087e570f'] and plotter:
                self.sin_beta.append(torch.sin(beta))
        # prev_yaw = state['yaw'][:,:,0]

        # print(f'from {prev_yaw} the change to be applied on the yaw is {longitudinal_speed/self.rear_wb*self.delta_t*torch.sin(beta)}')
        # lhs = (state['yaw'][:,:,0] + longitudinal_speed/self.rear_wb*self.delta_t*torch.sin(beta) - min_) % (2 * torch.pi) + min_
        lhs = (state['yaw'][:,:,0] + longitudinal_speed/wheel_base*self.delta_t*torch.sin(beta) - min_) % (2 * torch.pi) + min_
        # state['yaw'][:,:,0] = lhs
        state['yaw'] = torch.unsqueeze(lhs, dim=-1)
        # print('now the yaw is ', lhs)

        # assumption : the throttle contributes to the velocity in the direction of the updated yaw .....
        # longitudinal_speed = longitudinal_speed + actions['throttle'][:,:,0]*self.delta_t
        
        longitudinal_speed = F.softplus(longitudinal_speed + updated_accel*self.delta_t, beta=7)
        # state['vel'][:,:,0] = longitudinal_speed*torch.cos(lhs)
        # state['vel'][:,:,1] = longitudinal_speed*torch.sin(lhs)
        # state['vel'] = torch.cat([torch.unsqueeze(longitudinal_speed*torch.cos(lhs), dim=-1), torch.unsqueeze(longitudinal_speed*torch.sin(lhs), dim=-1)], dim=-1)
        state['vel'] = torch.cat([torch.unsqueeze(longitudinal_speed*torch.cos(lhs), dim=-1), torch.unsqueeze(longitudinal_speed*torch.sin(lhs), dim=-1)], dim=-1)


        # state['steering_angle'][:,:,0] = torch.clamp(state['steering_angle'][:,:,0] + actions['steer'][:,:,0]*self.delta_t, min=-torch.pi/3, max=torch.pi/3)
        # state['steering_angle'][:,:,0] = torch.clamp(updated_steering_angle + updated_steering_rate*self.delta_t, min=-torch.pi/3, max=torch.pi/3)
        # state['accel'][:,:,0] = updated_accel

        state['steering_angle'] = torch.unsqueeze(torch.clamp(updated_steering_angle + updated_steering_rate*self.delta_t, min=-torch.pi/3, max=torch.pi/3), dim=-1)
        state['accel'] = torch.unsqueeze(updated_accel, dim=-1)



        return state
    

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
        updated_steering_angle = (
            self.delta_t / (self.delta_t + self.steering_angle_time_constant) * (ideal_steering_angle - state["steering_angle"][:,:,0])
            + state["steering_angle"][:,:,0]
        )
        updated_steering_rate = (updated_steering_angle - state["steering_angle"][:,:,0]) / self.delta_t

      
        # updating the throttle, due to the delay in control
        ideal_accel = actions['throttle'][:,:,0]
        # ideal_accel = accel
        updated_accel = self.delta_t / (self.delta_t + self.accel_time_constant) * (ideal_accel - state["accel"][:,:,0]) + state["accel"][:,:,0]


        
        braking_ego = actions["brake"]
        braking_adv = torch.lt(updated_accel, torch.zeros_like(updated_accel))
        updated_accel = braking_ego * self.brake_accel + \
            braking_adv * -self.brake_accel * updated_accel + \
            ~braking_adv * self.throt_accel * updated_accel

        longitudinal_speed = torch.norm(state["vel"][:,:,:], dim=-1)
       
        beta = updated_steering_rate


        wheel_base = torch.tensor([3.089]).to(device=device)
     
        x_dot = torch.cos(state["yaw"][:,:,0]+beta)*longitudinal_speed
        y_dot = torch.sin(state["yaw"][:,:,0]+beta)*longitudinal_speed
    
        

        state['pos'] = torch.cat([torch.unsqueeze(state['pos'][:,:,0] + x_dot*self.delta_t, dim=-1), torch.unsqueeze(state['pos'][:,:,1] + y_dot*self.delta_t, dim=-1)], dim=-1)

        min_ = - torch.pi
        # lhs = (state['yaw'][:,:,0] + yaw_dot*self.delta_t - min_) % (2 * torch.pi) + min_

        # prev_yaw = state['yaw'][:,:,0]
        lhs = (state['yaw'][:,:,0] + longitudinal_speed/wheel_base*self.delta_t*torch.sin(beta) - min_) % (2 * torch.pi) + min_
        state['yaw'] = torch.unsqueeze(lhs, dim=-1)

        longitudinal_speed = F.softplus(longitudinal_speed + updated_accel*self.delta_t, beta=7)
   
        state['vel'] = torch.cat([torch.unsqueeze(longitudinal_speed*torch.cos(lhs), dim=-1), torch.unsqueeze(longitudinal_speed*torch.sin(lhs), dim=-1)], dim=-1)


      
        state['steering_angle'] = torch.unsqueeze(torch.clamp(updated_steering_angle + updated_steering_rate*self.delta_t, min=-torch.pi/3, max=torch.pi/3), dim=-1)
        state['accel'] = torch.unsqueeze(updated_accel, dim=-1)



        return state
    
    def that_agent(self):
        return self.sin_beta, self.that_agent_y_vel, self.that_agent_longitudinal_vel, self.that_agent_vel_x, self.that_agent_heading, self.that_agent_steering_angle, self.that_agent_steering_rate, self.that_agent_updated_steering_rate