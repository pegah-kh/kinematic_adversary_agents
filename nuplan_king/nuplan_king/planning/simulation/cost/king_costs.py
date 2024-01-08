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
    def __init__(self, road_rasterized):
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
        self._edt_road_np = edt_road
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

        norm_grads = np.sqrt((x_indices-target_x_indices)*(x_indices-target_x_indices) + (y_indices-target_y_indices)*(y_indices-target_y_indices))+10e-3
        # just visualize the heatmap of these gradients   
        self._norm_grads = norm_grads
        self._grad_direction_x, self._grad_direction_y = torch.tensor((x_indices-target_x_indices)/norm_grads, dtype=torch.float64, device=device), torch.tensor((y_indices-target_y_indices)/norm_grads, dtype=torch.float64, device=device)
        self._gradients_value = torch.cat((torch.unsqueeze(self._grad_direction_x, dim=-1), torch.unsqueeze(self._grad_direction_y, dim=-1)), dim=-1)

        self._road_rasterized = torch.tensor(self._road_rasterized).to(device=device)
        self._edt_road = torch.tensor(self._edt_road_np).to(device=device)


    def __call__(self, pos):
        """
        Computes the cost.
        """
        # the gradient of the position should then flow back to change the throttle and the steer
        # and also should have a mask, to not backpropagate the cost when in the drivable, but that is already included.
        current_cost = 0
        new_pos = pos.clone()
        # new_yaw = yaw.clone()
        # gradients = torch.zeros_like(new_pos, dtype=torch.float64, device=device)
        gradients = torch.zeros_like(new_pos, dtype=torch.float64, device=device)
        new_pos[new_pos[:,:,:]<0] = 0
        new_pos[new_pos[:,:,:]>=self._side_dim] = self._side_dim-1
        new_pos = new_pos.to(torch.int32)
        condition = self._road_rasterized[new_pos[0,:,0], new_pos[0,:,1]] == 1
        gradients[0,condition,:] = self._gradients_value[new_pos[0,:,0], new_pos[0,:,1]][condition]
        current_cost += self._edt_road[new_pos[0,:,0], new_pos[0,:,1]]

        return gradients[0], current_cost
    
    def heatmap(self, pos):
       
        new_pos = pos.clone()
        # new_yaw = yaw.clone()
        # gradients = torch.zeros_like(new_pos, dtype=torch.float64, device=device)
        gradient_size = torch.zeros(new_pos.size()[1], 1, dtype=torch.float64, device=device)
        new_pos[new_pos[:,:,:]<0] = 0
        new_pos[new_pos[:,:,:]>=self._side_dim] = self._side_dim-1
        new_pos = new_pos.to(torch.int32)
        condition = self._road_rasterized[new_pos[0,:,0], new_pos[0,:,1]] == 1
        norm_values = torch.norm(self._gradients_value[new_pos[0,:,0], new_pos[0,:,1]], dim=-1)
        # gradient_size[condition, 0] =  norm_values[condition]
        gradient_size = norm_values


        return gradient_size
    
    def heatmap_quiver(self, pos):
       
        new_pos = pos.clone().detach()
        gradients = torch.zeros_like(new_pos, dtype=torch.float64, device=device)

        # gradients = torch.zeros_like(new_pos, dtype=torch.float64, device=device)
        new_pos[new_pos[:,:,:]<0] = 0
        new_pos[new_pos[:,:,:]>=self._side_dim] = self._side_dim-1
        new_pos = new_pos.to(torch.int32)
        print('the gradients before size ', gradients.size())
        print('the size of the  ', self._gradients_value.size())
        gradients[0,:,:] = self._gradients_value[new_pos[0,:,0], new_pos[0,:,1]]


        return gradients

    def collided_agent_cost(self, pos):

        new_pos = pos.clone().to(torch.int32)
        new_pos[new_pos[:]<0] = 0
        new_pos[new_pos[:]>=self._side_dim] = self._side_dim-1
        return self._norm_grads[new_pos[0], new_pos[1]]
    

# implementing a different version of the drivable loss


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
            [[1.0, 2.5], [1.0, -2.5], [-1.0, 2.5], [-1.0, -2.5]],
            device=device,
            dtype=torch.float64
        )

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
        i_min, i_max = int(max(0, i - 16)), int(min(i + 16, x_extent))
        j_min, j_max = int(max(0, j - 16)), int(min(j + 16, y_extent))
        # if not (i_max - i_min)==32:
        #     print('this is (i_max - i_min) ', (i_max - i_min))
        
        # if not (j_max - j_min)==32:
        #     print('this is (j_max - j_min) ', (j_max - j_min))
        road_rasterized = road_rasterized[i_min:i_max:2, j_min:j_max:2]
        return road_rasterized

    def get_pixel_grid(self, i, j, x_extent, y_extent):

        i_min, i_max = int(max(0, i - 16)), int(min(i + 16, x_extent))
        j_min, j_max = int(max(0, j - 16)), int(min(j + 16, y_extent))
        # if not (i_max - i_min)==32:
        #     print('this is (i_max - i_min) ', (i_max - i_min))
                
        # if not (j_max - j_min)==32:
        #     print('this is (j_max - j_min) ', (j_max - j_min))
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


    def __call__(self, road_rasterized, pos, yaw):
        """
        Computes the cost.
        """

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
            )

        gks = self.apply_gauss_kernels(coords, pos)
        roads_rasterized = torch.cat([road_rasterized[None] for road_rasterized in roads_rasterized],dim=0)
        costs = torch.sum(roads_rasterized*torch.transpose(gks[:, :, :], 1, 2))[None, None]

        return costs
    

    def king_heatmap(self, road_rasterized, pos, yaw):
        pos = self.get_corners(pos, yaw)
        # crop_center = pos_w2m(crop_center[None])[0]
        pos = pos.view(-1, 2)
        pos = pos.view(self.batch_size, self.num_agents * 4, 2)

        x_extent, y_extent = road_rasterized.size(0), road_rasterized.size(1)
        print('extents ', x_extent, '   ', y_extent)

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
        reshaped_tensor = (roads_rasterized*torch.transpose(gks[:, :, :], 1, 2)).detach().reshape(-1, 4, 16, 16)
        print('size ', reshaped_tensor.size())
        summed_tensor = reshaped_tensor.sum(axis=(1,2,3))
        return summed_tensor



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
        # closest_points_list = closest_points.view(-1,2).clone()

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

        # return ego_cost, adv_cost, closest_points_list
        return ego_cost, adv_cost, distances[0]





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
        # closest_points_list = closest_points.view(-1,2).clone()

        distances, distances_idxs = torch.min(distances, dim=-2)

        distances_idxs = distances_idxs.unsqueeze(-1).repeat(1, 1, 1, 2)

        distances = distances.view(self.batch_size, self.num_agents + 1, self.num_agents + 1, 1)

        n = self.num_agents + 1
        distances = distances[0, :, :, 0].flatten()[1:].view(n-1, n+1)[:, :-1].reshape(n, n-1)

        ego_cost = torch.mean(distances[0])

        return ego_cost, distances[0]
    





class DummyCost_FixedPoint():
    """
    """
    def __init__(self, num_agents:int):
        """
        """
        self.batch_size = 1
        self.num_agents = num_agents # manually changed

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

        adv_pos = adv_state["pos"]
        adv_yaw = adv_state["yaw"]

        # adv_pos = adv_state["pos"][:,20:21,:]
        # adv_yaw = adv_state["yaw"][:,20:21,:]

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

        return ego_cost, distances[0]


