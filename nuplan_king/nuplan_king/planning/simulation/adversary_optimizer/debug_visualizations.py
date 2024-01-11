import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import zoom
import os
import torch

from nuplan_king.planning.simulation.cost.king_costs import RouteDeviationCostRasterized

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


class DebugginVisualization():
    """
        Helper class to transform the coordiantes from and to the world and map coordinate systems.
    """
    def __init__(self, scenario_name, experiment_name, username, plot_path=None):
        self._username = username
        self._scenario_name = scenario_name
        self._experiment_name = experiment_name
        if plot_path==None:
            plot_path = f'/home/{username}/workspace/EXPERIMENTS_map_new/{scenario_name}/{experiment_name}'

        create_directory_if_not_exists(plot_path)
        self._plot_path = plot_path
        
        

    def before_after_transform(self, before_transforming_x, before_transforming_y, before_transforming_yaw, after_transforming_x, after_transforming_y, after_transforming_yaw):
        
        plt.quiver(before_transforming_x, before_transforming_y, np.cos(before_transforming_yaw), np.sin(before_transforming_yaw), scale=10)
        plt.gcf()
        plt.savefig(f'{self._plot_path}/pos_agents_before_transformation.png')
        plt.show()
        plt.clf()
        plt.quiver(after_transforming_x, after_transforming_y, np.cos(after_transforming_yaw), np.sin(after_transforming_yaw), scale=10)
        plt.gcf()
        plt.savefig(f'{self._plot_path}/pos_agents_after_transformation.png')
        plt.show()
        plt.clf()

    def agents_position_before_transformation(self, before_transforming_x, before_transforming_y,):

        
        number_agents = before_transforming_x.shape[0]
        for idx in range(number_agents):
            plt.scatter(before_transforming_x[idx], before_transforming_y[idx])
            plt.text(before_transforming_x[idx], before_transforming_y[idx], str(idx))


        plt.gcf()
        plt.savefig(f'{self._plot_path}/agents_pos_idx.png')
        plt.show()
        plt.clf()


            
    def visualize_whole_map(self, map_resolution, data_nondrivable_map):

        

        resized_image = zoom(data_nondrivable_map, zoom=0.0001/map_resolution)
        plt.figure(figsize = (10,5))
        plt.imshow(resized_image, interpolation='nearest')
        plt.gcf()
        plt.savefig(f'{self._plot_path}/whole_map.png')
        plt.show()
        plt.clf()

        

    def routedeviation_king_heatmap(self, transpose_data_nondrivable_map):

        """
        Heatmap of the king deviation from drivable area cost
        """
        

        x_size = transpose_data_nondrivable_map.size()[1]
        y_size = transpose_data_nondrivable_map.size()[0]

        coords = torch.stack(
            torch.meshgrid(
                torch.linspace(0, x_size - 1, x_size),
                torch.linspace(0, y_size - 1, y_size)
            ),
            -1
        ).to(device=device)  # (H, W, 2)

        coords = coords[100:-100:100, 100:-100:100]

        coords = coords.reshape(-1, 2)
        coords = coords.unsqueeze(0)
        created_yaw = torch.zeros(1, coords.size()[1], 1, device=device)
        fn_cost = RouteDeviationCostRasterized(num_agents=coords.size()[1])
        adv_rd_cost = fn_cost.king_heatmap(
                transpose_data_nondrivable_map, 
                coords,
                created_yaw
            )
        
        adv_rd_cost = adv_rd_cost.reshape(18,18)
        adv_rd_cost = adv_rd_cost.cpu().numpy()

        heatmap = np.zeros((200, 200))

        # Determine the center region to fill
        center_start = 10
        center_end = 190

        scaled_array = zoom(adv_rd_cost, (10, 10), order=1)
        
        heatmap[center_start:center_end, center_start:center_end] = scaled_array

        # Display the heatmap
        plt.imshow(heatmap, cmap='viridis', interpolation='nearest')
        
        plt.gcf()
        plt.savefig(f'{self._plot_path}/heatmap_king_driving_cost.png')
        plt.colorbar()
        plt.show()
        plt.clf()


    def routedeviation_efficient_heatmap(self, data_nondrivable_map, effient_deviation_cost):

        """
        Heatmap of our efficient version of deviation from drivable area cost
        """        

        x_size = data_nondrivable_map.size()[0]
        y_size = data_nondrivable_map.size()[1]

        coords = torch.stack(
            torch.meshgrid(
                torch.linspace(0, x_size - 1, x_size),
                torch.linspace(0, y_size - 1, y_size)
            ),
            -1
        ).to(device=device)  # (H, W, 2)

        coords = coords[100:-100:100, 100:-100:100]

        coords = coords.reshape(-1, 2)
        coords = coords.unsqueeze(0)
        adv_rd_cost = effient_deviation_cost.heatmap(coords)
        
        print('this is the size of the adv_rd_cost ', adv_rd_cost.size())

        adv_rd_cost = adv_rd_cost.reshape(18,18)
        adv_rd_cost = adv_rd_cost.cpu().numpy()

        heatmap = np.zeros((200, 200))

        # Determine the center region to fill
        center_start = 10
        center_end = 190

        scaled_array = zoom(adv_rd_cost, (10, 10), order=1)
        
        heatmap[center_start:center_end, center_start:center_end] = scaled_array

        # Display the heatmap
        plt.imshow(heatmap, cmap='viridis', interpolation='nearest')

        
        plt.gcf()
        plt.savefig(f'{self._plot_path}/heatmap_efficient_driving_cost.png')
        plt.colorbar()
        plt.show()
        plt.clf()


    def routedeviation_king_agents_map(self, number_agents, current_state, king_deviation_cost, transpose_data_drivable_map, data_nondrivable_map):
        """
        writing down the deviation cost of each agent on the map
        the deviation cost calculated by king's deviation cost function
        """
        
        pos = current_state['pos'].detach().clone()
        yaw = current_state['yaw'].detach().clone()

        
        adv_rd_cost = king_deviation_cost.king_heatmap(transpose_data_drivable_map, pos, yaw)
        
        adv_rd_cost = adv_rd_cost.cpu().numpy()


        cp_map = data_nondrivable_map.detach().cpu().numpy()
        plt.figure(figsize = (10,10))

        plt.imshow(cp_map, interpolation='nearest')

        for idx in range(number_agents):
            position = current_state['pos'][0, idx].detach().cpu().numpy()
            transformed_adv_x, transformed_adv_y = position[0], position[1]
            plt.text(int(transformed_adv_y), int(transformed_adv_x), str("%.2f" % adv_rd_cost[idx]))
        
        plt.gcf()
        plt.savefig(f'{self._plot_path}/routedeviation_king_agents_map.png')
        plt.colorbar()
        plt.show()
        plt.clf()


    def routedeviation_efficient_agents_map(self, number_agents, current_state, efficient_deviation_cost, data_nondrivable_map):
        """
        writing down the deviation cost of each agent on the map
        the deviation cost calculated by the efficient version of deviation cost function
        """
        
            
        pos = current_state['pos'].detach().clone()

        adv_rd_cost = efficient_deviation_cost.heatmap(pos)
        
        adv_rd_cost = adv_rd_cost.cpu().numpy()
        print('agents ours cost ', adv_rd_cost)

        cp_map = data_nondrivable_map.detach().cpu().numpy()
        plt.figure(figsize = (10,10))

        plt.imshow(cp_map, interpolation='nearest')

        for idx in range(number_agents):
            position = current_state['pos'][0, idx].detach().cpu().numpy()
            transformed_adv_x, transformed_adv_y = position[0], position[1]
            plt.text(int(transformed_adv_y), int(transformed_adv_x), str("%.2f" % adv_rd_cost[idx]))
        
        plt.gcf()
        plt.savefig(f'{self._plot_path}/routedeviation_efficient_agents_map.png')
        plt.colorbar()
        plt.show()
        plt.clf()


    def routedeviation_efficient_agents_quiver_map(self, number_agents, current_state, efficient_deviation_cost, data_nondrivable_map):
        """
        writing down the deviation cost of each agent on the map
        the deviation cost calculated by the efficient version of deviation cost function
        """
        
            
        pos = current_state['pos'].detach().clone()
        
        adv_rd_cost_quiver = efficient_deviation_cost.heatmap_quiver(pos)
        
        adv_rd_cost_quiver = adv_rd_cost_quiver.cpu().numpy()
        print('agents ours cost ', adv_rd_cost_quiver)

        cp_map = data_nondrivable_map.detach().cpu().numpy()
        plt.figure(figsize = (10,10))

        plt.imshow(cp_map, interpolation='nearest')

        for idx in range(number_agents):
            position = pos[0, idx].detach().cpu().numpy()
            # print(adv_rd_cost_quiver[0,idx,0], adv_rd_cost_quiver[0,idx,1])
            plt.quiver(int(position[1]), int(position[0]), adv_rd_cost_quiver[0,idx,1], -adv_rd_cost_quiver[0,idx,0])
        
        
        plt.gcf()
        plt.savefig(f'{self._plot_path}/routedeviation_efficient_agents_quiver_map.png')
        plt.colorbar()
        plt.show()
        plt.clf()

