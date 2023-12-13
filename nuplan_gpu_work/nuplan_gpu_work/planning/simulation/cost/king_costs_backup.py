import torch
import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt


device = 'cuda' if torch.cuda.is_available() else 'cpu'
GPU_PI = torch.tensor([np.pi], device=device, dtype=torch.float64) # cuda


# do a fixed cost, where the gradients at each point is fixed. 
# the min distance at each point to zeros, gives the function distance_transform_edt
# how to find the directions of the gradient?? it also provides the indices of the pixels




class DrivableAreaCost():
    """
    """
    def __init__(self, road_rasterized: torch.Tensor):
        """
        This cost measures the overlap between an agent and
        the non-drivable area using a Gaussian Kernel.
        """
        self.batch_size = 1
        self.sigma_x = 5*128
        self.sigma_y = 2*128
        self.variance_x = self.sigma_x**2.
        self.variance_y = self.sigma_y**2.


        self._road_rasterized = road_rasterized.astype(int)
        self._side_dim = road_rasterized.shape[0]
        edt_road, inds_road = ndimage.distance_transform_edt(self._road_rasterized, return_indices=True)
        self._edt_road = edt_road
        height_road, width_road = road_rasterized.shape[0], road_rasterized.shape[1]
        coords = np.stack(
            np.meshgrid(
                np.linspace(0, height_road - 1, height_road),
                np.linspace(0, width_road - 1, width_road)
            ),
            -1
        )
        target_x_indices, target_y_indices = inds_road[0,:,:], inds_road[1,:,:]
        x_indices, y_indices = coords[:,:,1], coords[:,:,0]
        # tensor_edt_road = torch.tensor(edt_road, dtype=torch.float64, device=device, requires_grad=True)
        # grad_xy = torch.autograd.grad(tensor_edt_road.sum(), (tensor_edt_road,), create_graph=True)[0]
        # plt.imshow(edt_road, cmap='viridis', interpolation='nearest')
        # plt.gcf()
        # plt.savefig(f'/home/kpegah/workspace/VISUALIZE/distance_heatmap.png')
        # plt.colorbar()
        # plt.show()
        # plt.clf()

        # why also the agents that are in the drivable start moving?


        norm_grads = np.sqrt((x_indices-target_x_indices)*(x_indices-target_x_indices) + (y_indices-target_y_indices)*(y_indices-target_y_indices))+10e-3
        # just visualize the heatmap of these gradients   
        self._grad_direction_x, self._grad_direction_y = torch.tensor((x_indices-target_x_indices)/norm_grads/5, dtype=torch.float64, device=device), torch.tensor((y_indices-target_y_indices)/norm_grads/5, dtype=torch.float64, device=device)
        # print(self._grad_direction_x)
        # k=100
        # x_indices_d = x_indices[::k, ::k]
        # y_indices_d = y_indices[::k, ::k]
        # plt.imshow(self._road_rasterized, interpolation='nearest')

        # plt.quiver(x_indices_d, y_indices_d, self._grad_direction_x.detach().cpu().numpy()[::k,::k], self._grad_direction_y.detach().cpu().numpy()[::k, ::k], scale=20)
        # # plt.scatter(points_x, points_y, c='red', marker='o', s=50)  # Adjust marker style and size as needed
        # plt.gcf()
        # plt.savefig('/home/kpegah/workspace/VISUALIZE/grad_direction_in_cost1.png')
        # plt.show()
        # plt.clf()
        # # print('the mean value for the y gradient ', torch.mean(self._grad_direction_y))
        # print('the mean value of the x gradient ', torch.mean(self._grad_direction_x))


        # plt.imshow(self._road_rasterized, interpolation='nearest')

        # plt.quiver(x_indices_d, y_indices_d, self._grad_direction_y.detach().cpu().numpy()[::k, ::k], self._grad_direction_x.detach().cpu().numpy()[::k,::k], scale=20)
        # # plt.scatter(points_x, points_y, c='red', marker='o', s=50)  # Adjust marker style and size as needed
        # plt.gcf()
        # plt.savefig('/home/kpegah/workspace/VISUALIZE/grad_direction_in_cost2.png')
        # plt.show()
        # plt.clf()
        # # print('the mean value for the y gradient ', torch.mean(self._grad_direction_y))


        # # just in indices where the drivable area is not
        # road_d = self._road_rasterized[::k,::k]
        # x_indices_d = x_indices_d[road_d==1]
        # y_indices_d = y_indices_d[road_d==1]
        # plt.imshow(self._road_rasterized, interpolation='nearest')

        # plt.quiver(x_indices_d, y_indices_d, self._grad_direction_y.detach().cpu().numpy()[::k, ::k][road_d==1], self._grad_direction_x.detach().cpu().numpy()[::k,::k][road_d==1], scale=20)
        # # plt.scatter(points_x, points_y, c='red', marker='o', s=50)  # Adjust marker style and size as needed
        # plt.gcf()
        # plt.savefig('/home/kpegah/workspace/VISUALIZE/grad_direction_in_cost3.png')
        # plt.show()
        # plt.clf()
        # # print('the mean value for the y gradient ', torch.mean(self._grad_direction_y))


        # plt.imshow(self._road_rasterized, interpolation='nearest')

        # plt.quiver(x_indices_d, y_indices_d, self._grad_direction_x.detach().cpu().numpy()[::k, ::k][road_d==1], self._grad_direction_y.detach().cpu().numpy()[::k,::k][road_d==1], scale=20)
        # # plt.scatter(points_x, points_y, c='red', marker='o', s=50)  # Adjust marker style and size as needed
        # plt.gcf()
        # plt.savefig('/home/kpegah/workspace/VISUALIZE/grad_direction_in_cost4.png')
        # plt.show()
        # plt.clf()
        # # print('the mean value for the y gradient ', torch.mean(self._grad_direction_y))



        # print(self._grad_direction_y.detach().cpu().numpy()[::200])
        # print(self._grad_direction_x.detach().cpu().numpy()[::200])
        # print('hey')
        # print(target_x_indices[950:980,0:50])
        # print(target_y_indices[0:50,950:980])

        

    def crop_positions(self, pos, yaw, x_crop, y_crop, resolution, map_transform):
        pos_prime, transformed_adv_x, transformed_adv_y, transformed_yaw = self.from_matrix(map_transform@self.coord_to_matrix(pos, yaw))


        # return torch.tensor([transformed_adv_x, transformed_adv_y], device=device, dtype=torch.float64, requires_grad=True), torch.tensor([transformed_adv_y - y_crop + 100/resolution, transformed_adv_x - x_crop + 100/resolution], device=device, dtype=torch.float64, requires_grad=True), torch.tensor([transformed_yaw], device=device, dtype=torch.float64, requires_grad=True)
        # return pos_prime, torch.tensor([transformed_adv_y - y_crop + 100/resolution, transformed_adv_x - x_crop + 100/resolution], device=device, dtype=torch.float64, requires_grad=True), torch.tensor([transformed_yaw], device=device, dtype=torch.float64, requires_grad=True)
        return torch.tensor([transformed_adv_y - y_crop + 100/resolution, transformed_adv_x - x_crop + 100/resolution], device=device, dtype=torch.float64, requires_grad=True), torch.tensor([transformed_yaw], device=device, dtype=torch.float64, requires_grad=True)



    @staticmethod
    def coord_to_matrix(coord, yaw):

        return torch.tensor(
            [
                torch.cos(yaw[0]), -torch.sin(yaw[0]), 0.0, coord[0],
                torch.sin(yaw[0]), torch.cos(yaw[0]), 0.0, coord[1],
                0.0, 0.0, 1.0, 0.0,
                0.0, 0.0, 0.0, 1.0,
            ], requires_grad=True  
        ).to(device=device).to(dtype=torch.float64).view(4,4)


    @staticmethod
    def from_matrix(matrix):
 
        assert matrix.shape == (4, 4), f"Expected 3x3 transformation matrix, but input matrix has shape {matrix.shape}"

        return matrix[:2,3], matrix[0, 3], matrix[1, 3], torch.arctan2(matrix[1, 0], matrix[0, 0])
    

    def change_the_pos(self, pos, yaw, x_crop, y_crop, resolution, transform):
     
        new_pos = pos.clone()
        new_yaw = yaw.clone()
        for idx in range(pos.shape[1]):
            new_pos[0,idx,:], new_yaw[0,idx,:] = self.crop_positions(pos[0,idx,:], yaw[0,idx,:], x_crop, y_crop, resolution, transform)
            # pos[0,idx,:], yaw[0,idx,:] = new_pos[0,idx,:], new_yaw[0,idx,:]


        return new_pos, new_yaw


    def change_back_the_pos(self, grad, yaw, transform):


        # inv_transform = torch.linalg.inv(transform)
        new_grad = grad.clone()
        new_yaw = yaw.clone()
        for idx in range(grad.shape[1]):

            new_grad[0,idx,:], new_yaw[0,idx,:], _ = self.from_matrix(transform@self.coord_to_matrix(grad[0,idx,:], yaw[0,idx,:]))


        return new_grad, new_yaw
    

    # def prepare_cropped_positions(self, pos):

    #     smallest_x, largest_x = torch.max(0, torch.min(pos[:,:,0])-500), torch.min(torch.max(pos[:,:,0])+500, self._grad_direction_x.size()[1])     
    #     smallest_y, largest_y = torch.max(0, torch.min(pos[:,:,1])-500), torch.min(torch.max(pos[:,:,1])+500, self._grad_direction_x.size()[1])
    #     # compute the cropped positions in a sparser tensor, but how to keep the mapping??
    #     coords = torch.stack(
    #         torch.meshgrid(
    #             torch.linspace(smallest_x, largest_x-1, largest_x-smallest_x),
    #             torch.linspace(smallest_y, largest_y-1, largest_y-smallest_y)
    #         ),
    #         -1
    #     )

    #     # but the transformation should also be differentiable from the original position
    #     # and when finding the equivalent of the actual position in the equivalent matrix, how could it be differentiable??
    #     coords = np.array(coords[::5,::5,:]).reshape(-1, 2)
    #     for coord in coords:
    #         new_pos[0,idx,:], new_yaw[0,idx,:] = self.crop_positions(new_pos[0,idx,:], new_yaw[0,idx,:], x_crop, y_crop, resolution, transform)


        
    def __call__(self, x_crop, y_crop, pos, yaw, resolution, transform):
        """
        Computes the cost.
        """
        # the gradient of the position should then flow back to change the throttle and the steer
        # and also should have a mask, to not backpropagate the cost when in the drivable, but that is already included.
        current_cost = 0
        new_pos = pos.clone()
        new_yaw = yaw.clone()
        # gradients = torch.zeros_like(new_pos, dtype=torch.float64, device=device)
        gradients = torch.zeros_like(new_pos, dtype=torch.float64, device=device)
        plt.figure(figsize = (10,10))
        cp_map = self._road_rasterized
        for idx in range(pos.shape[1]):

            if not idx==20:
                continue
            
            new_pos[0,idx,:], new_yaw[0,idx,:] = self.crop_positions(new_pos[0,idx,:], new_yaw[0,idx,:], x_crop, y_crop, resolution, transform)
           
            # print('this is the grad ', pos[0,idx,0].grad)
            # the transfomed pos may be not in the map, so just consider the closest pixelto it
            x_pixel = int(new_pos[0,idx,0]) if int(new_pos[0,idx,0]) > 0 else 0
            x_pixel = x_pixel if x_pixel < self._side_dim else self._side_dim-1
            y_pixel = int(new_pos[0,idx,1]) if int(new_pos[0,idx,1]) > 0 else 0
            y_pixel = y_pixel if y_pixel < self._side_dim else self._side_dim-1
            if self._road_rasterized[x_pixel, y_pixel] == 1:
                # gradients[0,idx,0], gradients[0,idx,1] = self._grad_direction_x[x_pixel, y_pixel], self._grad_direction_y[x_pixel, y_pixel]
                gradients[0,idx,1], gradients[0,idx,0] = -self._grad_direction_x[x_pixel, y_pixel], self._grad_direction_y[x_pixel, y_pixel]
                current_cost += self._edt_road[x_pixel, y_pixel]
                # hard write the gradient direction on the map itself

                # for i in range(60):
                #     if i%5==0:
                #         plt.scatter(y_pixel+i*gradients[0,idx,1].detach().cpu().numpy()/20, x_pixel+i*gradients[0,idx,0].detach().cpu().numpy()/20, c='red', marker='o', s=2*i)  # Adjust marker style and size as needed
                #         plt.text(y_pixel, x_pixel, str(idx))

                # theta = np.arctan(gradients[0,idx,1].detach().cpu().numpy()/gradients[0,idx,0].detach().cpu().numpy())
                # plt.quiver(*np.array([y_pixel, x_pixel]), np.cos(-(theta-np.pi/2)), np.sin(-(theta-np.pi/2)), scale=60)
                # plt.quiver(*np.array([y_pixel, x_pixel]), gradients[0,idx,1].detach().cpu().numpy(), gradients[0,idx,0].detach().cpu().numpy(), scale=60)
                # plt.text(y_pixel, x_pixel, str("%.2f" % theta))
                # print(str("%.2f" % theta))

        # new_gradient = self.change_back_the_pos(gradients, new_yaw, transform)
        # gradients, _ = self.change_back_the_pos(gradients, new_yaw, transform)

            
        # plt.imshow(cp_map, interpolation='nearest')
        # # plt.scatter(points_x, points_y, c='red', marker='o', s=50)  # Adjust marker style and size as needed
        # plt.gcf()
        # plt.savefig('/home/kpegah/workspace/VISUALIZE/index_agents.png')
        # plt.show()
        # plt.clf()


        # pos.grad = gradients


        return gradients[0], current_cost

class RouteDeviationCostRasterized():
    """
    """
    def __init__(self, num_agents:int):
        """
        This cost measures the overlap between an agent and
        the non-drivable area using a Gaussian Kernel.
        """
        self.batch_size = 1
        self.num_agents = num_agents
        self.sigma_x = 5*128
        self.sigma_y = 2*128
        self.variance_x = self.sigma_x**2.
        self.variance_y = self.sigma_y**2.

        # default vehicle corners given an agent at the origin
        self.original_corners = torch.tensor(
            [[1.0, 2.5], [1.0, -2.5], [-1.0, -2.5], [-1.0, 2.5]],
            device=device,
            dtype=torch.float64
        )
        # .cuda()

    def get_corners(self, pos, yaw):
        """
        Obtain agent corners given the position and yaw.
        """
        yaw = GPU_PI/2 - yaw

        # rot_mat = torch.cat(
        #     [
        #         torch.cos(yaw), -torch.sin(yaw),
        #         torch.sin(yaw), torch.cos(yaw),
        #     ],
        #     dim=-1,
        # ).view(yaw.size(1), 1, 2, 2).expand(yaw.size(1), 4, 2, 2).to(device=device).to(dtype=torch.float64)

        
        rot_mat = torch.cat(
            [
                torch.sin(yaw), -torch.cos(yaw),
                torch.cos(yaw), torch.sin(yaw),
            ],
            dim=-1,
        ).view(yaw.size(1), 1, 2, 2).expand(yaw.size(1), 4, 2, 2).to(device=device).to(dtype=torch.float64)

        rotated_corners = rot_mat @ self.original_corners.unsqueeze(-1)

        rotated_corners = rotated_corners.view(yaw.size(1), 4, 2) + pos[0].unsqueeze(1)

        return rotated_corners.view(1, -1, 2)

    def crop_map(self, j, i, y_extent, x_extent, road_rasterized):
        i_min, i_max = int(max(0, i - 32)), int(min(i + 32, x_extent))
        j_min, j_max = int(max(0, j - 32)), int(min(j + 32, y_extent))
        road_rasterized = road_rasterized[i_min:i_max:2, j_min:j_max:2]
        return road_rasterized

    def get_pixel_grid(self, i, j, x_extent, y_extent):

        i_min, i_max = int(max(0, i - 32)), int(min(i + 32, x_extent))
        j_min, j_max = int(max(0, j - 32)), int(min(j + 32, y_extent))

        # print('this is (i_max - i_min) ', (i_max - i_min), '  ', i_max, '  ', i_min, '  ', i)
        coords = torch.stack(
            torch.meshgrid(
                torch.linspace(i_min, i_max - 1, (i_max - i_min)),
                torch.linspace(j_min, j_max - 1, (j_max - j_min))
            ),
            -1
        ).to(device=device)  # (H, W, 2)

        coords = coords[::2, ::2]
        # return coords.float()
        return coords.double()

    def apply_gauss_kernels(self, coords, pos):
        sigma = 5.
        pos = pos[0, :, :]
        coords = torch.cat(coords, dim=0)

        gk = torch.mean(((coords - pos[: ,None, None, :])/sigma)**2, dim=-1)
        gk = (1./(2*GPU_PI*sigma*sigma))*torch.exp(-gk)

        return gk
    
    def crop_positions(self, pos, yaw, x_crop, y_crop, resolution, map_transform):
        transformed_adv_x, transformed_adv_y, transformed_yaw = self.from_matrix(map_transform@self.coord_to_matrix(pos, yaw))
 
        # pos[0] = transformed_adv_y - y_crop + 100/resolution
        # pos[1] = transformed_adv_x - x_crop + 100/resolution

        # print('the position x and y are ', pos[0], '    ', pos[1])

        return torch.tensor([transformed_adv_y - y_crop + 100/resolution, transformed_adv_x - x_crop + 100/resolution], device=device, requires_grad=True), torch.tensor([transformed_yaw], device=device, requires_grad=True)



    @staticmethod
    def coord_to_matrix(coord, yaw):

        return torch.tensor(
            [
                torch.cos(yaw[0]), -torch.sin(yaw[0]), 0.0, coord[0],
                torch.sin(yaw[0]), torch.cos(yaw[0]), 0.0, coord[1],
                0.0, 0.0, 1.0, 0.0,
                0.0, 0.0, 0.0, 1.0,
            ], requires_grad=True  
        ).to(device=device).to(dtype=torch.float64).view(4,4)


    @staticmethod
    def from_matrix(matrix):
 
        assert matrix.shape == (4, 4), f"Expected 3x3 transformation matrix, but input matrix has shape {matrix.shape}"

        return matrix[0, 3], matrix[1, 3], torch.arctan2(matrix[1, 0], matrix[0, 0])
    

    def change_the_pos(self, pos, yaw, x_crop, y_crop, resolution, transform):
     
        new_pos = pos.clone()
        new_yaw = yaw.clone()
        for idx in range(pos.shape[1]):
            new_pos[0,idx,:], new_yaw[0,idx,:] = self.crop_positions(pos[0,idx,:], yaw[0,idx,:], x_crop, y_crop, resolution, transform)
            # pos[0,idx,:], yaw[0,idx,:] = new_pos[0,idx,:], new_yaw[0,idx,:]


        return new_pos, new_yaw

        

    def __call__(self, road_rasterized, x_crop, y_crop, pos, yaw, resolution, transform):
        """
        Computes the cost.
        """

        # you should not change the pos by itself!!
        # new_pos = pos.clone()
        # new_yaw = yaw.clone()
        # for idx in range(pos.shape[1]):
        #     new_pos[0,idx,:], new_yaw[0,idx,:] = self.crop_positions(new_pos[0,idx,:], new_yaw[0,idx,:], x_crop, y_crop, resolution, transform)
        #     # pos[0,idx,:], yaw[0,idx,:] = new_pos[0,idx,:], new_yaw[0,idx,:]

        pos, yaw = self.change_the_pos(pos, yaw, x_crop, y_crop, resolution, transform)

        pos = self.get_corners(pos, yaw)
        # crop_center = pos_w2m(crop_center[None])[0]
        pos = pos.view(-1, 2)
        pos = pos.view(self.batch_size, self.num_agents * 4, 2)

        x_extent, y_extent = road_rasterized.size(0), road_rasterized.size(1)

        roads_rasterized = []
        for i in range(pos.size(1)):
            crop_center = pos[0, i, :]
            crop_road_rasterized = self.crop_map(
                crop_center[0].item(),
                crop_center[1].item(),
                x_extent,
                y_extent,
                road_rasterized
            )
            crop_road_rasterized = 1. - crop_road_rasterized
            roads_rasterized.append(crop_road_rasterized)

        coords = []
        for i in range(pos.size(1)):
            crop_center = pos[0,i,:]
            coords.append(
                self.get_pixel_grid(
                    crop_center[0].item(),
                    crop_center[1].item(),
                    x_extent,
                    y_extent
                ).unsqueeze(0)
                # .cuda() before the last operation
            )

        gks = self.apply_gauss_kernels(coords, pos)
        roads_rasterized = torch.cat([road_rasterized[None] for road_rasterized in roads_rasterized],dim=0)
        costs = torch.sum(roads_rasterized*torch.transpose(gks[:, :, :], 1, 2))[None, None]
        return costs


class BatchedPolygonCollisionCost():
    """
    """
    def __init__(self, num_agents:int):
        """
        """
        self.batch_size = 1
        self.num_agents = num_agents

        self.unit_square = torch.tensor(
            [
                [ 1.,  1.],  # back right corner
                [-1.,  1.],  # back left corner
                [-1., -1.],  # front left corner
                [ 1., -1.],  # front right corner
            ],
            device=device,
            dtype=torch.float64
        ).view(1, 1, 4, 2).expand(self.batch_size, self.num_agents+1, 4, 2)

        self.segment_start_transform = torch.tensor(
            [
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ],
            dtype=torch.float64,
            device=device,
        ).reshape(1, 4, 4)

        self.segment_end_transform = torch.tensor(
            [
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
                [1, 0, 0, 0],
            ],
            dtype=torch.float64,
            device=device,
        ).reshape(1, 4, 4)

    def vertices_to_edges_vectorized(self, vertices):
        """
        """
        segment_start = self.segment_start_transform @ vertices
        segment_end = self.segment_end_transform @ vertices
        return segment_start, segment_end

    def __call__(self, ego_state, ego_extent, adv_state, adv_extent):
        """
        """
        ego_pos = ego_state["pos"]
        ego_yaw = ego_state["yaw"]
        ego_extent = torch.diag_embed(ego_extent)

        adv_pos = adv_state["pos"]
        adv_yaw = adv_state["yaw"]
        adv_extent = torch.diag_embed(adv_extent)

        pos = torch.cat([ego_pos, adv_pos], dim=1)
        yaw = torch.cat([ego_yaw, adv_yaw], dim=1)
        extent = torch.cat([ego_extent, adv_extent], dim=1)

        rot_mat = torch.cat(
            [
                torch.cos(yaw), -torch.sin(yaw),
                torch.sin(yaw), torch.cos(yaw),
            ],
            dim=-1,
        ).view(self.batch_size, self.num_agents+1, 2, 2).to(dtype=torch.float64)

        corners = self.unit_square @ extent

        corners = corners @ rot_mat.permute(0, 1, 3, 2)

        corners = corners + pos.unsqueeze(-2)

        segment_starts, segment_ends = self.vertices_to_edges_vectorized(corners)
        segments = segment_ends - segment_starts

        corners = corners.repeat_interleave(self.num_agents+1, dim=1)
        segment_starts = segment_starts.repeat(1, self.num_agents+1, 1, 1)
        segment_ends = segment_ends.repeat(1, self.num_agents+1, 1, 1)
        segments = segments.repeat(1, self.num_agents+1, 1, 1)

        corners = corners.repeat_interleave(4, dim=2)
        segment_starts = segment_starts.repeat(1, 1, 4, 1)
        segment_ends = segment_ends.repeat(1, 1, 4, 1)
        segments = segments.repeat(1, 1, 4, 1)

        projections = torch.matmul(
            (corners - segment_starts).unsqueeze(-2),
            segments.unsqueeze(-1)
        ).squeeze(-1)

        projections = projections / torch.sum(segments**2,dim=-1, keepdim=True)

        projections = torch.clamp(projections, 0., 1.)

        closest_points = segment_starts + segments * projections

        distances = torch.norm(corners - closest_points, dim=-1, keepdim=True)
        closest_points_list = closest_points.view(-1,2).clone()

        distances, distances_idxs = torch.min(distances, dim=-2)

        distances_idxs = distances_idxs.unsqueeze(-1).repeat(1, 1, 1, 2)

        distances = distances.view(self.batch_size, self.num_agents + 1, self.num_agents + 1, 1)

        n = self.num_agents + 1
        distances = distances[0, :, :, 0].flatten()[1:].view(n-1, n+1)[:, :-1].reshape(n, n-1)

        ego_cost = torch.min(distances[0])[None, None]

        if distances.size(0) > 2:
            distances_adv = distances[1:, 1:]
            adv_cost = torch.min(distances_adv, dim=-1)[0][None]
        else:
            adv_cost = torch.zeros(1, 0)
        # .cuda()

        return ego_cost, adv_cost, closest_points_list





class DummyCost():
    """
    """
    def __init__(self, num_agents:int):
        """
        """
        self.batch_size = 1
        self.num_agents = num_agents

        self.unit_square = torch.tensor(
            [
                [ 1.,  1.],  # back right corner
                [-1.,  1.],  # back left corner
                [-1., -1.],  # front left corner
                [ 1., -1.],  # front right corner
            ],
            device=device,
            dtype=torch.float64
        ).view(1, 1, 4, 2).expand(self.batch_size, self.num_agents+1, 4, 2)

        self.segment_start_transform = torch.tensor(
            [
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ],
            dtype=torch.float64,
            device=device, # there was a cuda here :))
        ).reshape(1, 4, 4)

        self.segment_end_transform = torch.tensor(
            [
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
                [1, 0, 0, 0],
            ],
            dtype=torch.float64,
            device=device, # there was a cuda here :))
        ).reshape(1, 4, 4)

    def vertices_to_edges_vectorized(self, vertices):
        """
        """
        segment_start = self.segment_start_transform @ vertices
        segment_end = self.segment_end_transform @ vertices
        return segment_start, segment_end

    def __call__(self, ego_state, ego_extent, adv_state, adv_extent):
        """
        """
        ego_pos = ego_state["pos"]
        ego_yaw = ego_state["yaw"]

        ego_extent = torch.diag_embed(ego_extent)

        adv_pos = adv_state["pos"]
        adv_yaw = adv_state["yaw"]
        adv_extent = torch.diag_embed(adv_extent)

        pos = torch.cat([ego_pos, adv_pos], dim=1)
        yaw = torch.cat([ego_yaw, adv_yaw], dim=1)
        extent = torch.cat([ego_extent, adv_extent], dim=1)

        rot_mat = torch.cat(
            [
                torch.cos(yaw), -torch.sin(yaw),
                torch.sin(yaw), torch.cos(yaw),
            ],
            dim=-1,
        ).view(self.batch_size, self.num_agents+1, 2, 2)

        corners = self.unit_square @ extent

        corners = corners @ rot_mat.permute(0, 1, 3, 2)

        corners = corners + pos.unsqueeze(-2)

        segment_starts, segment_ends = self.vertices_to_edges_vectorized(corners)
        segments = segment_ends - segment_starts

        corners = corners.repeat_interleave(self.num_agents+1, dim=1)
        segment_starts = segment_starts.repeat(1, self.num_agents+1, 1, 1)
        segment_ends = segment_ends.repeat(1, self.num_agents+1, 1, 1)
        segments = segments.repeat(1, self.num_agents+1, 1, 1)

        corners = corners.repeat_interleave(4, dim=2)
        segment_starts = segment_starts.repeat(1, 1, 4, 1)
        segment_ends = segment_ends.repeat(1, 1, 4, 1)
        segments = segments.repeat(1, 1, 4, 1)

        projections = torch.matmul(
            (corners - segment_starts).unsqueeze(-2),
            segments.unsqueeze(-1)
        ).squeeze(-1)

        projections = projections / torch.sum(segments**2,dim=-1, keepdim=True)

        projections = torch.clamp(projections, 0., 1.)

        closest_points = segment_starts + segments * projections

        distances = torch.norm(corners - closest_points, dim=-1, keepdim=True)
        closest_points_list = closest_points.view(-1,2).clone()

        distances, distances_idxs = torch.min(distances, dim=-2)

        distances_idxs = distances_idxs.unsqueeze(-1).repeat(1, 1, 1, 2)

        distances = distances.view(self.batch_size, self.num_agents + 1, self.num_agents + 1, 1)

        n = self.num_agents + 1
        distances = distances[0, :, :, 0].flatten()[1:].view(n-1, n+1)[:, :-1].reshape(n, n-1)

        ego_cost = torch.mean(distances[0])

        return ego_cost
    





class DummyCost_FixedPoint():
    """
    """
    def __init__(self, num_agents:int):
        """
        """
        self.batch_size = 1
        self.num_agents = 1 # manually changed

        self.unit_square = torch.tensor(
            [
                [ 1.,  1.],  # back right corner
                [-1.,  1.],  # back left corner
                [-1., -1.],  # front left corner
                [ 1., -1.],  # front right corner
            ],
            device=device,
            dtype=torch.float64
        ).view(1, 1, 4, 2).expand(self.batch_size, self.num_agents+1, 4, 2)

        self.segment_start_transform = torch.tensor(
            [
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ],
            dtype=torch.float64,
            device=device,
        ).reshape(1, 4, 4)

        self.segment_end_transform = torch.tensor(
            [
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
                [1, 0, 0, 0],
            ],
            dtype=torch.float64,
            device=device,
        ).reshape(1, 4, 4)

    def vertices_to_edges_vectorized(self, vertices):
        """
        """
        segment_start = self.segment_start_transform @ vertices
        segment_end = self.segment_end_transform @ vertices
        return segment_start, segment_end

    def __call__(self, ego_extent, adv_state, adv_extent, fixed_point):
        """
        """
       
        fixed_pos = fixed_point
        ego_yaw = torch.unsqueeze(torch.unsqueeze(torch.tensor([0.], device=device, dtype=torch.float64), dim=0), dim=0)
        ego_extent = torch.diag_embed(ego_extent)

        # adv_pos = adv_state["pos"]
        # adv_yaw = adv_state["yaw"]

        adv_pos = adv_state["pos"][:,20:21,:]
        adv_yaw = adv_state["yaw"][:,20:21,:]

        adv_extent = torch.diag_embed(adv_extent)

        pos = torch.cat([fixed_pos, adv_pos], dim=1)
        yaw = torch.cat([ego_yaw, adv_yaw], dim=1)
        extent = torch.cat([ego_extent, adv_extent], dim=1)

        rot_mat = torch.cat(
            [
                torch.cos(yaw), -torch.sin(yaw),
                torch.sin(yaw), torch.cos(yaw),
            ],
            dim=-1,
        ).view(self.batch_size, self.num_agents+1, 2, 2).to(dtype=torch.float64)

        corners = self.unit_square @ extent

        corners = corners @ rot_mat.permute(0, 1, 3, 2)

        corners = corners + pos.unsqueeze(-2)

        segment_starts, segment_ends = self.vertices_to_edges_vectorized(corners)
        segments = segment_ends - segment_starts

        corners = corners.repeat_interleave(self.num_agents+1, dim=1)
        segment_starts = segment_starts.repeat(1, self.num_agents+1, 1, 1)
        segment_ends = segment_ends.repeat(1, self.num_agents+1, 1, 1)
        segments = segments.repeat(1, self.num_agents+1, 1, 1)

        corners = corners.repeat_interleave(4, dim=2)
        segment_starts = segment_starts.repeat(1, 1, 4, 1)
        segment_ends = segment_ends.repeat(1, 1, 4, 1)
        segments = segments.repeat(1, 1, 4, 1)

        projections = torch.matmul(
            (corners - segment_starts).unsqueeze(-2),
            segments.unsqueeze(-1)
        ).squeeze(-1)

        projections = projections / torch.sum(segments**2,dim=-1, keepdim=True)

        projections = torch.clamp(projections, 0., 1.)

        closest_points = segment_starts + segments * projections

        distances = torch.norm(corners - closest_points, dim=-1, keepdim=True)
        closest_points_list = closest_points.view(-1,2).clone()

        distances, distances_idxs = torch.min(distances, dim=-2)

        distances_idxs = distances_idxs.unsqueeze(-1).repeat(1, 1, 1, 2)

        distances = distances.view(self.batch_size, self.num_agents + 1, self.num_agents + 1, 1)

        n = self.num_agents + 1
        distances = distances[0, :, :, 0].flatten()[1:].view(n-1, n+1)[:, :-1].reshape(n, n-1)

        ego_cost = torch.mean(distances[0])

        return ego_cost







class RouteDeviationHeatmap():
    """
    """
    def __init__(self, num_agents:int):
        """
        This cost measures the overlap between an agent and
        the non-drivable area using a Gaussian Kernel.
        """
        self.batch_size = 1
        self.num_agents = num_agents
        self.sigma_x = 5*128
        self.sigma_y = 2*128
        self.variance_x = self.sigma_x**2.
        self.variance_y = self.sigma_y**2.

        # default vehicle corners given an agent at the origin
        self.original_corners = torch.tensor(
            [[1.0, 2.5], [1.0, -2.5], [-1.0, 2.5], [-1.0, -2.5]],
            device=device,
            dtype=torch.float64
        )
        # .cuda()

    def get_corners(self, pos, yaw):
        """
        Obtain agent corners given the position and yaw.
        """
        yaw = GPU_PI/2 - yaw

        rot_mat = torch.cat(
            [
                torch.cos(yaw), -torch.sin(yaw),
                torch.sin(yaw), torch.cos(yaw),
            ],
            dim=-1,
        ).view(yaw.size(1), 1, 2, 2).expand(yaw.size(1), 4, 2, 2).to(device=device).to(dtype=torch.float64)

        rotated_corners = rot_mat @ self.original_corners.unsqueeze(-1)

        rotated_corners = rotated_corners.view(yaw.size(1), 4, 2) + pos[0].unsqueeze(1)

        return rotated_corners.view(1, -1, 2)

    def crop_map(self, j, i, y_extent, x_extent, road_rasterized):
        i_min, i_max = int(max(0, i - 32)), int(min(i + 32, x_extent))
        j_min, j_max = int(max(0, j - 32)), int(min(j + 32, y_extent))
        road_rasterized = road_rasterized[i_min:i_max:2, j_min:j_max:2]
        return road_rasterized

    def get_pixel_grid(self, i, j, x_extent, y_extent):

        i_min, i_max = int(max(0, i - 32)), int(min(i + 32, x_extent))
        j_min, j_max = int(max(0, j - 32)), int(min(j + 32, y_extent))

        # print('this is (i_max - i_min) ', (i_max - i_min), '  ', i_max, '  ', i_min, '  ', i)
        coords = torch.stack(
            torch.meshgrid(
                torch.linspace(i_min, i_max - 1, (i_max - i_min)),
                torch.linspace(j_min, j_max - 1, (j_max - j_min))
            ),
            -1
        ).to(device=device)  # (H, W, 2)

        print('hey ', (i_max - i_min), '   ', (j_max - j_min))

        coords = coords[::2, ::2]
        # return coords.float()
        return coords.double()

    def apply_gauss_kernels(self, coords, pos):
        sigma = 5.
        pos = pos[0, :, :]
        coords = torch.cat(coords, dim=0)

        gk = torch.mean(((coords - pos[: ,None, None, :])/sigma)**2, dim=-1)
        gk = (1./(2*GPU_PI*sigma*sigma))*torch.exp(-gk)

        return gk


    def __call__(self, road_rasterized, pos, yaw):
        """
        Computes the cost.
        """


        '''
        for idx in range(pos.shape[1]):
            new_pos[0,idx,:] = self.crop_positions(new_pos[0,idx,:], yaw[0,idx,:], x_crop, y_crop, resolution, transform)
        '''

        pos = self.get_corners(pos, yaw)
        # crop_center = pos_w2m(crop_center[None])[0]
        pos = pos.view(-1, 2)
        pos = pos.view(self.batch_size, self.num_agents * 4, 2)

        x_extent, y_extent = road_rasterized.size(0), road_rasterized.size(1)

        roads_rasterized = []
        for i in range(pos.size(1)):
            crop_center = pos[0, i, :]
            crop_road_rasterized = self.crop_map(
                crop_center[0].item(),
                crop_center[1].item(),
                x_extent,
                y_extent,
                road_rasterized
            )
            crop_road_rasterized = 1. - crop_road_rasterized
            roads_rasterized.append(crop_road_rasterized)

        coords = []
        for i in range(pos.size(1)):
            crop_center = pos[0,i,:]
            coords.append(
                self.get_pixel_grid(
                    crop_center[0].item(),
                    crop_center[1].item(),
                    x_extent,
                    y_extent
                ).unsqueeze(0)
                # .cuda() before the last operation
            )

        gks = self.apply_gauss_kernels(coords, pos)
        roads_rasterized = torch.cat([road_rasterized[None] for road_rasterized in roads_rasterized],dim=0)
        # the size of the loss is (84*32*32), around of each corner of each vehicle
        reshaped_tensor = (roads_rasterized*torch.transpose(gks[:, :, :], 1, 2)).detach().reshape(self.num_agents, 4, 32, 32)
        summed_tensor = reshaped_tensor.sum(axis=(1,2,3))
        return summed_tensor
    









class RouteDeviationHeatmap_agents():
    """
    """
    def __init__(self, num_agents:int):
        """
        This cost measures the overlap between an agent and
        the non-drivable area using a Gaussian Kernel.
        """
        self.batch_size = 1
        self.num_agents = num_agents
        self.sigma_x = 5*128
        self.sigma_y = 2*128
        self.variance_x = self.sigma_x**2.
        self.variance_y = self.sigma_y**2.

        # default vehicle corners given an agent at the origin
        self.original_corners = torch.tensor(
            [[1.0, 2.5], [1.0, -2.5], [-1.0, 2.5], [-1.0, -2.5]],
            device=device,
            dtype=torch.float64
        )
        # .cuda()

    def get_corners(self, pos, yaw):
        """
        Obtain agent corners given the position and yaw.
        """
        yaw = GPU_PI/2 - yaw

        rot_mat = torch.cat(
            [
                torch.cos(yaw), -torch.sin(yaw),
                torch.sin(yaw), torch.cos(yaw),
            ],
            dim=-1,
        ).view(yaw.size(1), 1, 2, 2).expand(yaw.size(1), 4, 2, 2).to(device=device).to(dtype=torch.float64)

        rotated_corners = rot_mat @ self.original_corners.unsqueeze(-1)

        rotated_corners = rotated_corners.view(yaw.size(1), 4, 2) + pos[0].unsqueeze(1)

        return rotated_corners.view(1, -1, 2)

    def crop_map(self, j, i, y_extent, x_extent, road_rasterized):
        i_min, i_max = int(max(0, i - 32)), int(min(i + 32, x_extent))
        j_min, j_max = int(max(0, j - 32)), int(min(j + 32, y_extent))
        road_rasterized = road_rasterized[i_min:i_max:2, j_min:j_max:2]
        return road_rasterized

    def get_pixel_grid(self, i, j, x_extent, y_extent):

        i_min, i_max = int(max(0, i - 32)), int(min(i + 32, x_extent))
        j_min, j_max = int(max(0, j - 32)), int(min(j + 32, y_extent))

        # print('this is (i_max - i_min) ', (i_max - i_min), '  ', i_max, '  ', i_min, '  ', i)
        coords = torch.stack(
            torch.meshgrid(
                torch.linspace(i_min, i_max - 1, (i_max - i_min)),
                torch.linspace(j_min, j_max - 1, (j_max - j_min))
            ),
            -1
        ).to(device=device)  # (H, W, 2)

        print('hey ', (i_max - i_min), '   ', (j_max - j_min))

        coords = coords[::2, ::2]
        # return coords.float()
        return coords.double()

    def apply_gauss_kernels(self, coords, pos):
        sigma = 5.
        pos = pos[0, :, :]
        coords = torch.cat(coords, dim=0)

        gk = torch.mean(((coords - pos[: ,None, None, :])/sigma)**2, dim=-1)
        gk = (1./(2*GPU_PI*sigma*sigma))*torch.exp(-gk)

        return gk

    def crop_positions(self, pos, yaw, x_crop, y_crop, resolution, map_transform):
        transformed_adv_x, transformed_adv_y, _ = self.from_matrix(map_transform@self.coord_to_matrix(pos, yaw))
 
        # pos[0] = transformed_adv_y - y_crop + 100/resolution
        # pos[1] = transformed_adv_x - x_crop + 100/resolution

        # print('the position x and y are ', pos[0], '    ', pos[1])

        # return torch.tensor([transformed_adv_y - y_crop + 100/resolution, transformed_adv_x - x_crop + 100/resolution], device=device)
        return torch.tensor([transformed_adv_x - x_crop + 100/resolution, transformed_adv_y - y_crop + 100/resolution], device=device)



    @staticmethod
    def coord_to_matrix(coord, yaw):

        return torch.tensor(
            [
                [torch.cos(yaw[0]), -torch.sin(yaw[0]), 0.0, coord[0]],
                [torch.sin(yaw[0]), torch.cos(yaw[0]), 0.0, coord[1]],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            device=device,
            dtype=torch.float64,
        )


    @staticmethod
    def from_matrix(matrix):
 
        assert matrix.shape == (4, 4), f"Expected 3x3 transformation matrix, but input matrix has shape {matrix.shape}"

        return matrix[0, 3], matrix[1, 3], torch.arctan2(matrix[1, 0], matrix[0, 0])

        

    def __call__(self, road_rasterized, x_crop, y_crop, pos, yaw, resolution, transform):
        """
        Computes the cost.
        """

        # you should not change the pos by itself!!
        new_pos = pos.clone()
        for idx in range(pos.shape[1]):
            new_pos[0,idx,:] = self.crop_positions(new_pos[0,idx,:], yaw[0,idx,:], x_crop, y_crop, resolution, transform)
        
        new_pos = self.get_corners(new_pos, yaw)
        # crop_center = pos_w2m(crop_center[None])[0]
        new_pos = new_pos.view(-1, 2)
        new_pos = new_pos.view(self.batch_size, self.num_agents * 4, 2)

        x_extent, y_extent = road_rasterized.size(0), road_rasterized.size(1)

        roads_rasterized = []
        for i in range(new_pos.size(1)):
            crop_center = new_pos[0, i, :]
            crop_road_rasterized = self.crop_map(
                crop_center[0].item(),
                crop_center[1].item(),
                x_extent,
                y_extent,
                road_rasterized
            )
            crop_road_rasterized = 1. - crop_road_rasterized
            roads_rasterized.append(crop_road_rasterized)

        coords = []
        for i in range(new_pos.size(1)):
            crop_center = new_pos[0,i,:]
            coords.append(
                self.get_pixel_grid(
                    crop_center[0].item(),
                    crop_center[1].item(),
                    x_extent,
                    y_extent
                ).unsqueeze(0)
                # .cuda() before the last operation
            )

        gks = self.apply_gauss_kernels(coords, new_pos)
        roads_rasterized = torch.cat([road_rasterized[None] for road_rasterized in roads_rasterized],dim=0)
        # the size of the loss is (84*32*32), around of each corner of each vehicle
        reshaped_tensor = (roads_rasterized*torch.transpose(gks[:, :, :], 1, 2)).detach().reshape(self.num_agents, 4, 32, 32)
        summed_tensor = reshaped_tensor.sum(axis=(1,2,3))
        return summed_tensor
    


