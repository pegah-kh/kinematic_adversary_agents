import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import zoom
import os
from matplotlib.pyplot import cm
import torch
import wandb


device = 'cuda' if torch.cuda.is_available() else 'cpu'
                
def create_directory_if_not_exists(directory_path):
    """
        Creating a dirctory at the given directory path
        :param directory_path
    """
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)




# from https://stackoverflow.com/questions/17458580/embedding-small-plots-inside-subplots-in-matplotlib
def add_subplot_axes(ax,rect,facecolor='w'):
    """
        Outputting an axis to plot on, at a given point and scale
        :param ax: the main axis to add the subplot to.
        :param rect:
               rect[:2] : position of the axis (relatively to the ax)
               rect[2:] : scale of the axis (relatively to the ax)
    """
    fig = plt.gcf()
    box = ax.get_position()
    width = box.width
    height = box.height
    inax_position  = ax.transAxes.transform(rect[0:2])
    transFigure = fig.transFigure.inverted()
    infig_position = transFigure.transform(inax_position)    
    x = infig_position[0]
    y = infig_position[1]
    width *= rect[2]
    height *= rect[3]
    subax = fig.add_axes([x,y,width,height], facecolor=facecolor)
    # x_labelsize = subax.get_xticklabels()[0].get_size()
    # y_labelsize = subax.get_yticklabels()[0].get_size()
    # x_labelsize *= rect[2]**0.3
    # y_labelsize *= rect[3]**0.3
    subax.xaxis.set_tick_params(labelsize=5)
    subax.yaxis.set_tick_params(labelsize=5)
    # subax.set_ylim([-100,100])
    return subax


def visualize_grads_throttle(scenario_name, experiment_name, optimization_iteration, number_agents, whole_state_buffers, throttle_gradients):

    """
    the function to visualize the gradients registered for the throttle
    for each agent at its position on the map, in the given optimization iteration
    at each agent position : a plot of the evolution of throttle gradient across steps of the simulation

    :params scenario_name
    :params experiment_name
    :params optimization_iteration
    :params number_agents
    :params whole_state_buffers : to determine the position of each agent at the given optimization_iteration
    :params throttle_gradients
    """
    throttle_gradients = torch.cat(throttle_gradients, dim=-1).cpu().numpy().reshape(number_agents,-1)


    create_directory_if_not_exists('/home/kpegah/workspace/EXPERIMENTS')
    create_directory_if_not_exists(f'/home/kpegah/workspace/EXPERIMENTS/{scenario_name}')
    create_directory_if_not_exists(f'/home/kpegah/workspace/EXPERIMENTS/{scenario_name}/{experiment_name}')
    create_directory_if_not_exists(f'/home/kpegah/workspace/EXPERIMENTS/{scenario_name}/{experiment_name}/Throttle_Grad_Evolution')

    plt.figure(figsize=(50, 30))

    # we have gradients accumulated for each step of the simulation, and for each of the agents, we know its position
    pos_agents = whole_state_buffers[0][0]['pos'][:number_agents, :]
    smallest_x, largest_x, smallest_y, largest_y = np.min(pos_agents[:,0]), np.max(pos_agents[:,0]), np.min(pos_agents[:,1]), np.max(pos_agents[:,1])
    width, height = (largest_x - smallest_x), (largest_y - smallest_y)

    fig_throttle, ax_throttle = plt.subplots(facecolor ='#a0d9f0')       
    ax_throttle.set_xlim(smallest_x, largest_x)
    ax_throttle.set_ylim(smallest_y, largest_y)


    colors = cm.rainbow(np.linspace(0, 1, number_agents))

    for i_agent in range(number_agents):
        
        # the position of the agent should be with respect to the figure
        relative_x, relative_y = (pos_agents[i_agent,0]-smallest_x)/width, (pos_agents[i_agent,1]-smallest_y)/height
        # what should be the size of the plot for the gradients?? decide based on the values obtained for each agent
        current_throttle_grads = throttle_gradients[i_agent] # this is of the size of length of actions, and for each of the actions, 

        subpos_throttle = [relative_y, relative_x ,0.12 ,0.08]
        subax1 = add_subplot_axes(ax_throttle,subpos_throttle)

        # should first normalize the gradients in the x direction??
        subax1.plot(current_throttle_grads, color=colors[i_agent])
    
    
    
    plt.gcf()
    plt.savefig(f'/home/kpegah/workspace/EXPERIMENTS/{scenario_name}/{experiment_name}/Throttle_Grad_Evolution/{optimization_iteration}.png')
    plt.show()
    plt.clf()



def visualize_grads_steer(scenario_name, experiment_name, optimization_iteration, number_agents, whole_state_buffers, steer_gradients):

    """
    the function to visualize the gradients registered for the steer
    for each agent at its position on the map, in the given optimization iteration
    at each agent position : a plot of the evolution of steer gradient across steps of the simulation

    :params scenario_name
    :params experiment_name
    :params optimization_iteration
    :params number_agents
    :params whole_state_buffers : to determine the position of each agent at the given optimization_iteration
    :params throttle_gradients
    """

    steer_gradients = torch.cat(steer_gradients, dim=-1).cpu().numpy().reshape(number_agents,-1)

    create_directory_if_not_exists('/home/kpegah/workspace/EXPERIMENTS')
    create_directory_if_not_exists(f'/home/kpegah/workspace/EXPERIMENTS/{scenario_name}')
    create_directory_if_not_exists(f'/home/kpegah/workspace/EXPERIMENTS/{scenario_name}/{experiment_name}')
    create_directory_if_not_exists(f'/home/kpegah/workspace/EXPERIMENTS/{scenario_name}/{experiment_name}/Steer_Grad_Evolution')

    plt.figure(figsize=(50, 30))

    # we have gradients accumulated for each step of the simulation, and for each of the agents, we know its position
    pos_agents = whole_state_buffers[0][0]['pos'][:number_agents, :]
    smallest_x, largest_x, smallest_y, largest_y = np.min(pos_agents[:,0]), np.max(pos_agents[:,0]), np.min(pos_agents[:,1]), np.max(pos_agents[:,1])
    width, height = (largest_x - smallest_x), (largest_y - smallest_y)

    fig_steer, ax_steer = plt.subplots(facecolor ='#a0d9f0')       
    ax_steer.set_xlim(smallest_x, largest_x)
    ax_steer.set_ylim(smallest_y, largest_y)


    colors = cm.rainbow(np.linspace(0, 1, number_agents))

    for i_agent in range(number_agents):
        # if not i_agent==3:
        #     continue
        # contsruct the gradient (throttle and steer) for the current agent
        # the position of the agent should be with respect to the figure
        relative_x, relative_y = (pos_agents[i_agent,0]-smallest_x)/width, (pos_agents[i_agent,1]-smallest_y)/height
        current_steer_grads = steer_gradients[i_agent]

        subpos_steer = [relative_y, relative_x ,0.12 ,0.08]

        subax2 = add_subplot_axes(ax_steer,subpos_steer)

        subax2.plot(current_steer_grads, color=colors[i_agent])
    
    
    
    plt.gcf()
    plt.savefig(f'/home/kpegah/workspace/EXPERIMENTS/{scenario_name}/{experiment_name}/Steer_Grad_Evolution/{optimization_iteration}.png')
    plt.show()
    plt.clf()





def visualize_throttle(scenario_name, experiment_name, optimization_iteration, number_agents, whole_state_buffers, throttles):

    """
    the function to visualize the throttle
    for each agent at its position on the map, in the given optimization iteration
    at each agent position : a plot of the evolution of throttle across steps of the simulation


    :params scenario_name
    :params experiment_name
    :params optimization_iteration
    :params number_agents
    :params whole_state_buffers : to determine the position of each agent at the given optimization_iteration
    :params throttle_gradients
    """
    throttles = torch.cat(throttles, dim=-1).cpu().numpy().reshape(number_agents,-1)


    create_directory_if_not_exists('/home/kpegah/workspace/EXPERIMENTS')
    create_directory_if_not_exists(f'/home/kpegah/workspace/EXPERIMENTS/{scenario_name}')
    create_directory_if_not_exists(f'/home/kpegah/workspace/EXPERIMENTS/{scenario_name}/{experiment_name}')
    create_directory_if_not_exists(f'/home/kpegah/workspace/EXPERIMENTS/{scenario_name}/{experiment_name}/Throttle_Evolution')

    plt.figure(figsize=(50, 30))

    # we have gradients accumulated for each step of the simulation, and for each of the agents, we know its position
    pos_agents = whole_state_buffers[0][0]['pos'][:number_agents, :]
    smallest_x, largest_x, smallest_y, largest_y = np.min(pos_agents[:,0]), np.max(pos_agents[:,0]), np.min(pos_agents[:,1]), np.max(pos_agents[:,1])
    # height_map, width_map = self._nondrivable_map_layer.shape[0], self._nondrivable_map_layer.shape[1]
    width, height = (largest_x - smallest_x), (largest_y - smallest_y)

    fig_throttle, ax_throttle = plt.subplots(facecolor ='#a0d9f0')       
    ax_throttle.set_xlim(smallest_x, largest_x)
    ax_throttle.set_ylim(smallest_y, largest_y)


    colors = cm.rainbow(np.linspace(0, 1, number_agents))

    for i_agent in range(number_agents):
        # if not i_agent==3:
        #     continue
        # contsruct the gradient (throttle and steer) for the current agent
        # the position of the agent should be with respect to the figure
        relative_x, relative_y = (pos_agents[i_agent,0]-smallest_x)/width, (pos_agents[i_agent,1]-smallest_y)/height
        # what should be the size of the plot for the gradients?? decide based on the values obtained for each agent
        current_throttles = throttles[i_agent] # this is of the size of length of actions, and for each of the actions, 

        subpos_throttle = [relative_y, relative_x ,0.12 ,0.08]
        subax1 = add_subplot_axes(ax_throttle,subpos_throttle)

        # should first normalize the gradients in the x direction??
        subax1.plot(current_throttles, color=colors[i_agent])
    
    
    
    plt.gcf()
    plt.savefig(f'/home/kpegah/workspace/EXPERIMENTS/{scenario_name}/{experiment_name}/Throttle_Evolution/{optimization_iteration}.png')
    plt.show()
    plt.clf()



def visualize_steer(scenario_name, experiment_name, optimization_iteration, number_agents, whole_state_buffers, steers):

    """
    the function to visualize the gradients registered for the throttle
    for each agent at its position on the map, in the given optimization iteration
    at each agent position : a plot of the evolution of steer across steps of the simulation


    :params scenario_name
    :params experiment_name
    :params optimization_iteration
    :params number_agents
    :params whole_state_buffers : to determine the position of each agent at the given optimization_iteration
    :params throttle_gradients
    """

    steers = torch.cat(steers, dim=-1).cpu().numpy().reshape(number_agents,-1)

    create_directory_if_not_exists('/home/kpegah/workspace/EXPERIMENTS')
    create_directory_if_not_exists(f'/home/kpegah/workspace/EXPERIMENTS/{scenario_name}')
    create_directory_if_not_exists(f'/home/kpegah/workspace/EXPERIMENTS/{scenario_name}/{experiment_name}')
    create_directory_if_not_exists(f'/home/kpegah/workspace/EXPERIMENTS/{scenario_name}/{experiment_name}/Steer_Evolution')

    plt.figure(figsize=(50, 30))

    # we have gradients accumulated for each step of the simulation, and for each of the agents, we know its position
    pos_agents = whole_state_buffers[0][0]['pos'][:number_agents, :]
    smallest_x, largest_x, smallest_y, largest_y = np.min(pos_agents[:,0]), np.max(pos_agents[:,0]), np.min(pos_agents[:,1]), np.max(pos_agents[:,1])
    width, height = (largest_x - smallest_x), (largest_y - smallest_y)

    fig_steer, ax_steer = plt.subplots(facecolor ='#a0d9f0')       
    ax_steer.set_xlim(smallest_x, largest_x)
    ax_steer.set_ylim(smallest_y, largest_y)


    colors = cm.rainbow(np.linspace(0, 1, number_agents))

    for i_agent in range(number_agents):
        # if not i_agent==3:
        #     continue
        # contsruct the gradient (throttle and steer) for the current agent
        # the position of the agent should be with respect to the figure
        relative_x, relative_y = (pos_agents[i_agent,0]-smallest_x)/width, (pos_agents[i_agent,1]-smallest_y)/height
        current_steers = steers[i_agent]

        subpos_steer = [relative_y, relative_x ,0.12 ,0.08]

        subax2 = add_subplot_axes(ax_steer,subpos_steer)

        subax2.plot(current_steers, color=colors[i_agent])
    
    
    
    plt.gcf()
    plt.savefig(f'/home/kpegah/workspace/EXPERIMENTS/{scenario_name}/{experiment_name}/Steer_Evolution/{optimization_iteration}.png')
    plt.show()
    plt.clf()


def plot_loss_per_agent(scenario_name, experiment_name, number_agents, whole_state_buffers, drivable_loss_per_opt, collision_loss_per_opt):

    """
    To visualize the evolution of collision and drivable losses, per agent, across the optimization iterations.
    Visualizes a plot at the position of each agent of the evolution of its collision and drivable loss.
    """

    drivable_loss_per_opt = [item.cpu().numpy() for item in drivable_loss_per_opt]
    collision_loss_per_opt = [item.cpu().numpy() for item in collision_loss_per_opt]

    drivable_loss_per_opt = np.array(drivable_loss_per_opt).transpose(1,0)
    collision_loss_per_opt = np.array(collision_loss_per_opt).transpose(1,0)

    create_directory_if_not_exists('/home/kpegah/workspace/EXPERIMENTS')
    create_directory_if_not_exists(f'/home/kpegah/workspace/EXPERIMENTS/{scenario_name}')
    create_directory_if_not_exists(f'/home/kpegah/workspace/EXPERIMENTS/{scenario_name}/{experiment_name}')

    plt.figure(figsize=(50, 30))

    pos_agents = whole_state_buffers[0][0]['pos'][:number_agents, :]
    smallest_x, largest_x, smallest_y, largest_y = np.min(pos_agents[:,0]), np.max(pos_agents[:,0]), np.min(pos_agents[:,1]), np.max(pos_agents[:,1])
    width, height = (largest_x - smallest_x), (largest_y - smallest_y)

    fig_steer, ax_steer = plt.subplots(facecolor ='#a0d9f0')       
    ax_steer.set_xlim(smallest_x, largest_x)
    ax_steer.set_ylim(smallest_y, largest_y)


    colors = cm.rainbow(np.linspace(0, 1, number_agents))

    for i_agent in range(number_agents):
        relative_x, relative_y = (pos_agents[i_agent,0]-smallest_x)/width, (pos_agents[i_agent,1]-smallest_y)/height
        agent_losses = drivable_loss_per_opt[i_agent]

        subpos = [relative_y, relative_x ,0.12 ,0.08]

        subax2 = add_subplot_axes(ax_steer,subpos)

        subax2.plot(agent_losses, color=colors[i_agent])
    
    
    
    plt.gcf()
    plt.savefig(f'/home/kpegah/workspace/EXPERIMENTS/{scenario_name}/{experiment_name}/drivable_loss_evolution_agents_per_opt.png')
    wandb.log({"drivable_loss_evolution_agents_per_opt": wandb.Image(plt)})
    plt.show()
    plt.clf()
    
    plt.figure(figsize=(50, 30))

    # we have gradients accumulated for each step of the simulation, and for each of the agents, we know its position
    pos_agents = whole_state_buffers[0][0]['pos'][:number_agents, :]
    smallest_x, largest_x, smallest_y, largest_y = np.min(pos_agents[:,0]), np.max(pos_agents[:,0]), np.min(pos_agents[:,1]), np.max(pos_agents[:,1])
    width, height = (largest_x - smallest_x), (largest_y - smallest_y)

    fig_steer, ax_steer = plt.subplots(facecolor ='#a0d9f0')       
    ax_steer.set_xlim(smallest_x, largest_x)
    ax_steer.set_ylim(smallest_y, largest_y)


    colors = cm.rainbow(np.linspace(0, 1, number_agents))

    for i_agent in range(number_agents):
        relative_x, relative_y = (pos_agents[i_agent,0]-smallest_x)/width, (pos_agents[i_agent,1]-smallest_y)/height
        agent_losses = collision_loss_per_opt[i_agent]

        subpos = [relative_y, relative_x ,0.12 ,0.08]

        subax2 = add_subplot_axes(ax_steer,subpos)

        subax2.plot(agent_losses, color=colors[i_agent])
    
    
    
    plt.gcf()
    plt.savefig(f'/home/kpegah/workspace/EXPERIMENTS/{scenario_name}/{experiment_name}/collition_loss_evolution_agents_per_opt.png')
    plt.show()
    plt.clf()



def plot_losses(scenario_name, experiment_name, costs_to_use, deviation_losses_per_opt, deviation_loss_per_step, collision_loss_per_opt, collision_loss_per_step):
    """
    plots drivable and collision losees:
    1. accumulated for all the agents and steps, across optimization iterations
    2. accumulated for all the agents, across the simulation steps of the last optimization iteration
    """
    create_directory_if_not_exists('/home/kpegah/workspace/EXPERIMENTS')
    create_directory_if_not_exists(f'/home/kpegah/workspace/EXPERIMENTS/{scenario_name}')
    create_directory_if_not_exists(f'/home/kpegah/workspace/EXPERIMENTS/{scenario_name}/{experiment_name}')

    if len(deviation_losses_per_opt)==0:
        return

    if 'drivable_king' in costs_to_use or 'drivable_efficient' in costs_to_use:


        plt.plot(torch.stack(deviation_losses_per_opt).cpu().numpy())
        plt.gcf()
        plt.savefig(f'/home/kpegah/workspace/EXPERIMENTS/{scenario_name}/{experiment_name}/deviation_losses_per_opt.png')
        wandb.log({"drivable_accumulated_loss": wandb.Image(plt)})
        plt.show()
        plt.clf()


        print('this is the _drivable_area_losses_per_step ', deviation_loss_per_step)
        plt.plot(torch.stack(deviation_loss_per_step).cpu().numpy(), label='cost')
        # plt.plot(self.throttles[3], label='throttle val')
        plt.legend() 
        plt.gcf()
        plt.savefig(f'/home/kpegah/workspace/EXPERIMENTS/{scenario_name}/{experiment_name}/deviation_loss_per_step.png')
        wandb.log({"drivable_cost_sim_steps": wandb.Image(plt)})
        plt.show()
        plt.clf()


    if 'fixed_dummy' in costs_to_use or 'moving_dummy' in costs_to_use or 'collision' in costs_to_use:

        
        plt.plot(torch.stack(collision_loss_per_opt).cpu().numpy())
        plt.gcf()
        plt.savefig(f'/home/kpegah/workspace/EXPERIMENTS/{scenario_name}/{experiment_name}/collision_loss_per_opt.png')
        wandb.log({"dummy_accumulated_loss": wandb.Image(plt)})
        plt.show()
        plt.clf()


        plt.plot(torch.stack(collision_loss_per_step).cpu().numpy(), label='cost')
        plt.legend() 
        plt.gcf()
        plt.savefig(f'/home/kpegah/workspace/EXPERIMENTS/{scenario_name}/{experiment_name}/collision_loss_per_step.png')
        wandb.log({"fixeddummy_cost_sim_steps": wandb.Image(plt)})
        plt.show()
        plt.clf()

