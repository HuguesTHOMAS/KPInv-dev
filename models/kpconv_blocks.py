#
#
#      0=================================0
#      |    Kernel Point Convolutions    |
#      0=================================0
#
#
# ----------------------------------------------------------------------------------------------------------------------
#
#      Define network blocks
#
# ----------------------------------------------------------------------------------------------------------------------
#
#      Hugues THOMAS - 06/03/2020
#

import math
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.init import kaiming_uniform_


from kernels.kernel_points import load_kernels
from models.generic_blocks import index_select, radius_gaussian, local_maxpool, UnaryBlock, BatchNormBlock, GroupNormBlock

# ----------------------------------------------------------------------------------------------------------------------
#
#           Simple functions
#       \**********************/
#



# ----------------------------------------------------------------------------------------------------------------------
#
#           KPConv class
#       \******************/
#


class KPConv(nn.Module):

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 radius: float,
                 sigma: float,
                 modulated: bool = False,
                 groups: int = 1,
                 dimension: int = 3,
                 influence_mode: str = 'linear',
                 aggregation_mode: str = 'sum',
                 fixed_kernel_points: str = 'center',
                 inf: float = 1e6):
        """
        Rigid KPConv.
        Paper: https://arxiv.org/abs/1904.08889.
        Args:
            in_channels (int): The number of the input channels.
            out_channels (int): The number of the output channels.
            kernel_size (int): The number of kernel points.
            radius (float): The radius used for kernel point init.
            sigma (float): The influence radius of each kernel point.
            modulated (bool=False): Use modulations (self-attention)
            groups (int=1): Groups in convolution (=in_channels for depthwise conv).
            dimension (int=3): The dimension of the point space.
            influence_mode (str='linear'): Influence function ('constant', 'linear', 'gaussian').
            aggregation_mode (str='sum'): Aggregation mode ('nearest', 'sum').
            fixed_kernel_points (str='center'): kernel points whose position is fixed ('none', 'center' or 'verticals').
            inf (float=1e6): The value of infinity to generate the padding point.
        """
        super(KPConv, self).__init__()

        # Verification of group parameter

        assert in_channels % groups == 0, "in_channels must be divisible by groups."
        assert out_channels % groups == 0, "out_channels must be divisible by groups."
        in_channels_per_group = in_channels // groups
        out_channels_per_group = out_channels // groups

        # Save parameters
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.radius = radius
        self.sigma = sigma
        self.groups = groups
        self.dimension = dimension
        self.influence_mode = influence_mode
        self.aggregation_mode = aggregation_mode
        self.fixed_kernel_points = fixed_kernel_points
        self.inf = inf
        self.in_channels_per_group = in_channels_per_group
        self.out_channels_per_group = out_channels_per_group

        # Initialize weights
        if self.groups == 1:
            weights = torch.zeros(size=(kernel_size, in_channels, out_channels))
        else:
            weights = torch.zeros(size=(kernel_size, groups, in_channels_per_group, out_channels_per_group))
        self.weights = nn.Parameter(weights, requires_grad=True)
        
        # modulations, MLP
        self.modulated = modulated
        if self.modulated:
            self.gen_mlp = nn.Linear(in_channels, self.kernel_size, bias=True)

        # Reset parameters
        self.reset_parameters()

        # Initialize kernel points
        kernel_points = self.initialize_kernel_points()
        self.register_buffer("kernel_points", kernel_points)

        return

    def reset_parameters(self):
        kaiming_uniform_(self.weights, a=math.sqrt(5))
        return

    def initialize_kernel_points(self) -> Tensor:
        """
        Initialize the kernel point positions in a sphere
        :return: the tensor of kernel points
        """
        kernel_points = load_kernels(self.radius, self.kernel_size, dimension=self.dimension, fixed=self.fixed_kernel_points)
        return torch.from_numpy(kernel_points).float()

    def get_neighbors_influences(self, q_pts: Tensor,
                                 s_pts: Tensor,
                                 neighb_inds: Tensor) -> Tensor: 
        """
        Influence function of kernel points on neighbors.
        Args:
            q_points (Tensor): query points (M, 3).
            s_points (Tensor): support points carrying input features (N, 3).
            neighb_inds (LongTensor): neighbor indices of query points among support points (M, H).
        Returns:
            neighbor_weights (Tensor): the influence weight of each kernel point on each neighbors point (M, K, H).
        """

        # TODO share neighbor_weights between convolutions at same layer (need to share kernel point location as well)

        with torch.no_grad():

            # Add a fake point in the last row for shadow neighbors
            s_pts = torch.cat((s_pts, torch.zeros_like(s_pts[:1, :]) + self.inf), 0)   # (N, 3) -> (N+1, 3)

            # Get neighbor points [n_points, n_neighbors, dim]
            # neighbors = s_pts[neighb_inds, :]  # (N+1, 3) -> (M, H, 3)
            neighbors = index_select(s_pts, neighb_inds, dim=0)  # (N+1, 3) -> (M, H, 3)

            # Center every neighborhood
            neighbors = neighbors - q_pts.unsqueeze(1)  # (M, H, 3)
        
            # Get Kernel point distances to neigbors
            neighbors = neighbors.unsqueeze(2)  # (M, H, 3) -> (M, H, 1, 3)
            differences = neighbors - self.kernel_points  # (M, H, 1, 3) x (K, 3) -> (M, H, K, 3)
            sq_distances = torch.sum(differences ** 2, dim=3)  # (M, H, K)

            # Get Kernel point influences
            if self.influence_mode == 'constant':
                # Every point get an influence of 1.
                neighbor_weights = torch.ones_like(sq_distances)

            elif self.influence_mode == 'linear':
                # Influence decrease linearly with the distance, and get to zero when d = sigma.
                neighbor_weights = torch.clamp(1 - torch.sqrt(sq_distances) / self.sigma, min=0.0)  # (M, H, K)

            elif self.influence_mode == 'gaussian':
                # Influence in gaussian of the distance.
                gaussian_sigma = self.sigma * 0.3
                neighbor_weights = radius_gaussian(sq_distances, gaussian_sigma)
            else:
                raise ValueError("Unknown influence mode: : '{:s}'.  Should be 'constant', 'linear', or 'gaussian'".format(self.aggregation_mode))
            neighbor_weights = torch.transpose(neighbor_weights, 1, 2)  # (M, H, K) -> (M, K, H)

            # In case of nearest mode, only the nearest KP can influence each point
            if self.aggregation_mode == 'nearest':
                neighbors_1nn = torch.argmin(sq_distances, dim=2)
                neighbor_weights *= torch.transpose(nn.functional.one_hot(neighbors_1nn, self.kernel_size), 1, 2)

            elif self.aggregation_mode != 'sum':
                raise ValueError("Unknown aggregation mode: '{:s}'. Should be 'nearest' or 'sum'".format(self.aggregation_mode))

        return neighbor_weights

    def forward(self, q_pts: Tensor,
                s_pts: Tensor,
                s_feats: Tensor,
                neighb_inds: Tensor) -> Tensor:
        """
        KPConv forward.
        Args:
            q_points (Tensor): query points (M, 3).
            s_points (Tensor): support points carrying input features (N, 3).
            s_feats (Tensor): input features values (N, C_in).
            neighb_inds (LongTensor): neighbor indices of query points among support points (M, H).
        Returns:
            q_feats (Tensor): output features carried by query points (M, C_out).
        """

        # Get features for each neighbor
        # ******************************

        # Add a zero feature for shadow neighbors
        padded_s_feats = torch.cat((s_feats, torch.zeros_like(s_feats[:1, :])), 0)  # (N, C) -> (N+1, C)

        # Get the features of each neighborhood
        # neighbor_feats = gather(padded_s_feats, neighb_inds)  # (N+1, C) -> (M, H, C)
        neighbor_feats = index_select(padded_s_feats, neighb_inds, dim=0) 

        
        # Transfer features to kernel points
        # **********************************
        
        # Get Kernel point influences (M, K, H)
        neighbor_weights = self.get_neighbors_influences(q_pts, s_pts, neighb_inds)

        # Apply distance weights
        weighted_feats = torch.matmul(neighbor_weights, neighbor_feats)  # (M, K, H) x (M, H, C) -> (M, K, C)

        # Apply modulations
        if self.modulated:
            
            # In case M == N, we can assume this is an in-place convolution.
            if q_pts.shape[0] == s_pts.shape[0]:
                pooled_feats = s_feats
            else:
                # pooled_feats = neighbor_feats[:, 0, :]  # nearest pool (M, H, C) -> (M, C)
                # pooled_feats = torch.max(neighbor_feats, dim=1)  # max pool (M, H, C) -> (M, C)
                pooled_feats = torch.mean(neighbor_feats, dim=1)  # avg pool (M, H, C) -> (M, C)
            self.offset_features = self.gen_mlp(pooled_feats)  # (M, K)
            modulations = 2 * torch.sigmoid(self.offset_features)  # (M, K)

            weighted_feats *= modulations.unsqueeze(2)

        # Apply convolutional weights
        # ***************************
        
        if self.groups == 1:

            # # standard conv
            # weighted_feats = weighted_feats.permute((1, 0, 2))  # (M, K, C) -> (K, M, C)
            # kernel_outputs = torch.matmul(weighted_feats, self.weights)  # (K, M, C) x (K, C, O) -> (K, M, O)
            # output_feats = torch.sum(kernel_outputs, dim=0)  # (K, M, O) -> (M, O)

            output_feats = torch.einsum("mkc,kcd->md", weighted_feats, self.weights)  # (M, K, C) x (K, C, O) -> (M, O)

        else:
            # group conv
            weighted_feats = weighted_feats.view(-1, self.kernel_size, self.groups, self.in_channels_per_group)  # (M, K, C) -> (M, K, G, C//G)
            output_feats = torch.einsum("mkgc,kgcd->mgd", weighted_feats, self.weights)  # (M, K, G, C//G) * (K, G, C//G, O//G) -> (M, G, O//G)
            output_feats = output_feats.view(-1, self.out_channels)  # (M, G, O//G) -> (M, O)

            # weighted_feats = weighted_feats.view(-1, self.kernel_size, self.groups, self.in_channels_per_group)  # (M, K, C) -> (M, K, G, C//G)
            # weighted_feats = weighted_feats.permute((1, 2, 0, 3))  # (M, K, G, C//G) -> (K, G, M, C//G)
            # kernel_outputs = torch.matmul(weighted_feats, self.weights)  # (K, G, M, C//G) x (K, G, C//G, O//G) -> (K, G, M, O//G)
            # kernel_outputs = torch.sum(kernel_outputs, dim=0)  # (K, G, M, O//G) -> (G, M, O//G)
            # kernel_outputs = kernel_outputs.permute((1, 0, 2))  # (G, M, O//G) -> (M, G, O//G)
            # kernel_outputs = kernel_outputs.view(-1, self.out_channels)  # (M, G, O//G) -> (M, O)

        # # density normalization (divide output features by the sum of neighbor positive features)
        # neighbor_feats_sum = torch.sum(neighbor_feats, dim=-1)  # (M, H)
        # neighbor_num = torch.sum(torch.gt(neighbor_feats_sum, 0.0), dim=-1)  # (M,)
        # neighbor_num = torch.max(neighbor_num, torch.ones_like(neighbor_num))  # (M,)
        # output_feats = output_feats / neighbor_num.unsqueeze(1)

        return output_feats

    def __repr__(self):

        repr_str = 'KPConv'
        repr_str += '(K: {:d}'.format(self.kernel_size)
        repr_str += ', in_C: {:d}'.format(self.in_channels)
        repr_str += ', out_C: {:d}'.format(self.out_channels)
        repr_str += ', r: {:.2f}'.format(self.radius)
        repr_str += ', sigma: {:d})'.format(self.sigma)

        return repr_str


class KPDef(nn.Module):

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 radius: float,
                 sigma: float,
                 modulated: bool = False,
                 groups: int = 1,
                 dimension: int = 3,
                 influence_mode: str = 'linear',
                 aggregation_mode: str = 'sum',
                 fixed_kernel_points: str = 'center',
                 inf: float = 1e6):
        """
        Deformable KPConv.
        Paper: https://arxiv.org/abs/1904.08889.
        Args:
            in_channels (int): The number of the input channels.
            out_channels (int): The number of the output channels.
            kernel_size (int): The number of kernel points.
            radius (float): The radius used for kernel point init.
            sigma (float): The influence radius of each kernel point.
            modulated (bool=False): Use modulations (self-attention)
            groups (int=1): Groups in convolution (=in_channels for depthwise conv).
            dimension (int=3): The dimension of the point space.
            influence_mode (str='linear'): Influence function ('constant', 'linear', 'gaussian').
            aggregation_mode (str='sum'): Aggregation mode ('nearest', 'sum').
            fixed_kernel_points (str='center'): kernel points whose position is fixed ('none', 'center' or 'verticals').
            inf (float=1e6): The value of infinity to generate the padding point.
        """
        super(KPDef, self).__init__()

        # Verification of group parameter
        assert in_channels % groups == 0, "in_channels must be divisible by groups."
        assert out_channels % groups == 0, "out_channels must be divisible by groups."
        in_channels_per_group = in_channels // groups
        out_channels_per_group = out_channels // groups

        # Save parameters
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.radius = radius
        self.sigma = sigma
        self.groups = groups
        self.dimension = dimension
        self.influence_mode = influence_mode
        self.aggregation_mode = aggregation_mode
        self.fixed_kernel_points = fixed_kernel_points
        self.inf = inf
        self.in_channels_per_group = in_channels_per_group
        self.out_channels_per_group = out_channels_per_group

        # Running variable containing deformed KP distance to input points. (used in regularization loss)
        self.min_d2 = None
        self.deformed_KP = None
        self.offset_features = None

        # Initialize weights
        if self.groups == 1:
            weights = torch.zeros(size=(kernel_size, in_channels, out_channels))
        else:
            weights = torch.zeros(size=(kernel_size, groups, in_channels_per_group, out_channels_per_group))
        self.weights = nn.Parameter(weights, requires_grad=True)

        # Deformation generation (temporary make some test then keep only the best implem)
        self.modulated = modulated
        if self.modulated:
            self.offset_dim = (self.dimension + 1) * self.kernel_size
        else:
            self.offset_dim = self.dimension * self.kernel_size

        self.version = 'v1'
        if self.version == 'v1':
            # MLP
            self.offset_mlp = nn.Linear(in_channels, self.offset_dim, bias=True)

        elif self.version == 'v2':
            # KPConv
            self.offset_conv = KPConv(in_channels,
                                      self.offset_dim,
                                      kernel_size,
                                      radius,
                                      sigma,
                                      modulated=False,
                                      groups=groups,
                                      dimension=dimension,
                                      influence_mode=influence_mode,
                                      aggregation_mode=aggregation_mode,
                                      fixed_kernel_points=fixed_kernel_points)
            self.offset_bias = nn.Parameter(torch.zeros(self.offset_dim, dtype=torch.float32), requires_grad=True)
        else:
            raise ValueError('temp')

        # Reset parameters
        self.reset_parameters()

        # Initialize kernel points
        kernel_points = self.initialize_kernel_points()
        self.register_buffer("kernel_points", kernel_points)

        return

    def reset_parameters(self):
        kaiming_uniform_(self.weights, a=math.sqrt(5))
        return

    def initialize_kernel_points(self) -> Tensor:
        """
        Initialize the kernel point positions in a sphere
        :return: the tensor of kernel points
        """
        kernel_points = load_kernels(self.radius, self.kernel_size, dimension=self.dimension, fixed=self.fixed_kernel_points)
        return torch.from_numpy(kernel_points).float()

    def get_neighbors_influences(self, q_pts: Tensor,
                                 s_pts: Tensor,
                                 neighb_inds: Tensor,
                                 deformed_K_points: Tensor) -> Tensor: 
        """
        Influence function of kernel points on neighbors.
        Args:
            q_points (Tensor): query points (M, 3).
            s_points (Tensor): support points carrying input features (N, 3).
            neighb_inds (LongTensor): neighbor indices of query points among support points (M, H).
        Returns:
            neighbor_weights (Tensor): the influence weight of each kernel point on each neighbors point (M, K, H).
        """

        with torch.no_grad():

            # Add a fake point in the last row for shadow neighbors
            s_pts = torch.cat((s_pts, torch.zeros_like(s_pts[:1, :]) + self.inf), 0)   # (N, 3) -> (N+1, 3)

            # Get neighbor points [n_points, n_neighbors, dim]
            # neighbors = s_pts[neighb_inds, :]  # (N+1, 3) -> (M, H, 3)
            neighbors = index_select(s_pts, neighb_inds, dim=0)  # (N+1, 3) -> (M, H, 3)

            # Center every neighborhood
            neighbors = neighbors - q_pts.unsqueeze(1)  # (M, H, 3)
        
            # Get Kernel point distances to neigbors
            neighbors = neighbors.unsqueeze(2)  # (M, H, 3) -> (M, H, 1, 3)

        differences = neighbors - deformed_K_points  # (M, H, 1, 3) x (M, 1, K, 3) -> (M, H, K, 3)
        sq_distances = torch.sum(differences ** 2, dim=3)  # (M, H, K)

        # Save min distances for loss
        self.min_d2, _ = torch.min(sq_distances, dim=1)   # (M, K)

        # TODO: Should we use grad at this stage or not ???
        with torch.no_grad():

            # Get Kernel point influences
            if self.influence_mode == 'constant':
                # Every point get an influence of 1.
                neighbor_weights = torch.ones_like(sq_distances.detach())

            elif self.influence_mode == 'linear':
                # Influence decrease linearly with the distance, and get to zero when d = sigma.
                neighbor_weights = torch.clamp(1 - torch.sqrt(sq_distances.detach()) / self.sigma, min=0.0)  # (M, H, K)

            elif self.influence_mode == 'gaussian':
                # Influence in gaussian of the distance.
                gaussian_sigma = self.sigma * 0.3
                neighbor_weights = radius_gaussian(sq_distances.detach(), gaussian_sigma)
            else:
                raise ValueError("Unknown influence mode: : '{:s}'.  Should be 'constant', 'linear', or 'gaussian'".format(self.aggregation_mode))
            neighbor_weights = torch.transpose(neighbor_weights, 1, 2)  # (M, H, K) -> (M, K, H)

            # In case of nearest mode, only the nearest KP can influence each point
            if self.aggregation_mode == 'nearest':
                neighbors_1nn = torch.argmin(sq_distances.detach(), dim=2)
                neighbor_weights *= torch.transpose(nn.functional.one_hot(neighbors_1nn, self.kernel_size), 1, 2)

            elif self.aggregation_mode != 'sum':
                raise ValueError("Unknown aggregation mode: '{:s}'. Should be 'nearest' or 'sum'".format(self.aggregation_mode))

        return neighbor_weights

    def forward(self, q_pts: Tensor,
                s_pts: Tensor,
                s_feats: Tensor,
                neighb_inds: Tensor) -> Tensor:
        """
        KPConv forward.
        Args:
            q_points (Tensor): query points (M, 3).
            s_points (Tensor): support points carrying input features (N, 3).
            s_feats (Tensor): input features values (N, C_in).
            neighb_inds (LongTensor): neighbor indices of query points among support points (M, H).
        Returns:
            q_feats (Tensor): output features carried by query points (M, C_out).
        """
        
        # Get features for each neighbor
        # ******************************

        # Add a zero feature for shadow neighbors
        padded_s_feats = torch.cat((s_feats, torch.zeros_like(s_feats[:1, :])), 0)  # (N, C) -> (N+1, C)

        # Get the features of each neighborhood
        # neighbor_feats = gather(padded_s_feats, neighb_inds)  # (N+1, C) -> (M, H, C)
        neighbor_feats = index_select(padded_s_feats, neighb_inds, dim=0) 

        
        # Get deformable convolution offsets
        # **********************************

        if self.version == 'v1':

            # In case M == N, we can assume this is an in-place convolution.
            if q_pts.shape[0] == s_pts.shape[0]:
                pooled_feats = s_feats
            else:
                # pooled_feats = neighbor_feats[:, 0, :]  # nearest pool (M, H, C) -> (M, C)
                # pooled_feats = torch.max(neighbor_feats, dim=1)  # max pool (M, H, C) -> (M, C)
                pooled_feats = torch.mean(neighbor_feats, dim=1)  # avg pool (M, H, C) -> (M, C)
            self.offset_features = self.offset_mlp(pooled_feats)  # (M, K*D)

        elif self.version == 'v2':
            self.offset_features = self.offset_conv(q_pts, s_pts, s_feats, neighb_inds) + self.offset_bias

        if self.modulated:
            # Get offset (in normalized scale) from features
            unscaled_offsets = self.offset_features[:, :self.dimension * self.kernel_size]
            unscaled_offsets = unscaled_offsets.view(-1, self.kernel_size, self.dimension)  # (M, K, D)

            # Get modulations
            modulations = 2 * torch.sigmoid(self.offset_features[:, self.dimension * self.kernel_size:])   # (M, K)

        else:
            unscaled_offsets = self.offset_features.view(-1, self.kernel_size, self.dimension)  # (M, K, D)

        # Rescale offset for this layer
        offsets = unscaled_offsets * self.radius

        # Apply offsets to kernel points [n_points, n_kpoints, dim]
        self.deformed_KP = offsets + self.kernel_points  # (M, K, D)
        deformed_K_points = self.deformed_KP.unsqueeze(1)  # (M, 1, K, D)

        
        # Transfer features to kernel points
        # **********************************

        # Get Kernel point influences (M, K, H)
        neighbor_weights = self.get_neighbors_influences(q_pts, s_pts, neighb_inds, deformed_K_points)

        # Apply distance weights
        weighted_feats = torch.matmul(neighbor_weights, neighbor_feats)  # (M, K, H) x (M, H, C) -> (M, K, C)
        
        # Apply modulations
        if self.modulated:
            weighted_feats *= modulations.unsqueeze(2)


        # Apply convolutional weights
        # ***************************

        # apply convolutional weights
        if self.groups == 1:

            # # standard conv
            # weighted_feats = weighted_feats.permute((1, 0, 2))  # (M, K, C) -> (K, M, C)
            # kernel_outputs = torch.matmul(weighted_feats, self.weights)  # (K, M, C) x (K, C, O) -> (K, M, O)
            # output_feats = torch.sum(kernel_outputs, dim=0)  # (K, M, O) -> (M, O)

            output_feats = torch.einsum("mkc,kcd->md", weighted_feats, self.weights)  # (M, K, C) x (K, C, O) -> (M, O)

        else:
            # group conv
            weighted_feats = weighted_feats.view(-1, self.kernel_size, self.groups, self.in_channels_per_group)  # (M, K, C) -> (M, K, G, C//G)
            output_feats = torch.einsum("mkgc,kgcd->mgd", weighted_feats, self.weights)  # (M, K, G, C//G) * (K, G, C//G, O//G) -> (M, G, O//G)
            output_feats = output_feats.view(-1, self.out_channels)  # (M, G, O//G) -> (M, O)

            # weighted_feats = weighted_feats.view(-1, self.kernel_size, self.groups, self.in_channels_per_group)  # (M, K, C) -> (M, K, G, C//G)
            # weighted_feats = weighted_feats.permute((1, 2, 0, 3))  # (M, K, G, C//G) -> (K, G, M, C//G)
            # kernel_outputs = torch.matmul(weighted_feats, self.weights)  # (K, G, M, C//G) x (K, G, C//G, O//G) -> (K, G, M, O//G)
            # kernel_outputs = torch.sum(kernel_outputs, dim=0)  # (K, G, M, O//G) -> (G, M, O//G)
            # kernel_outputs = kernel_outputs.permute((1, 0, 2))  # (G, M, O//G) -> (M, G, O//G)
            # kernel_outputs = kernel_outputs.view(-1, self.out_channels)  # (M, G, O//G) -> (M, O)

        # # density normalization (divide output features by the sum of neighbor positive features)
        # neighbor_feats_sum = torch.sum(neighbor_feats, dim=-1)  # (M, H)
        # neighbor_num = torch.sum(torch.gt(neighbor_feats_sum, 0.0), dim=-1)  # (M,)
        # neighbor_num = torch.max(neighbor_num, torch.ones_like(neighbor_num))  # (M,)
        # output_feats = output_feats / neighbor_num.unsqueeze(1)



        return output_feats

    def __repr__(self):

        repr_str = 'KPConv'
        repr_str += '(K: {:d}'.format(self.kernel_size)
        repr_str += ', in_C: {:d}'.format(self.in_channels)
        repr_str += ', out_C: {:d}'.format(self.out_channels)
        repr_str += ', r: {:.2f}'.format(self.radius)
        repr_str += ', sigma: {:d})'.format(self.sigma)

        return repr_str


# ----------------------------------------------------------------------------------------------------------------------
#
#           Complex blocks
#       \********************/
#



class KPConvBlock(nn.Module):

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 radius: float,
                 sigma: float,
                 modulated: bool = False,
                 deformable: bool = False,
                 influence_mode: str = 'linear',
                 aggregation_mode: str = 'sum',
                 dimension: int = 3,
                 groups: int = 1,
                 norm_type: str = 'batch',
                 bn_momentum: float = 0.98,
                 activation: nn.Module = nn.LeakyReLU(0.1)):
        """
        KPConv block with normalization and activation.  
        Args:
            in_channels (int): dimension input features
            out_channels (int): dimension input features
            kernel_size (int): number of kernel points
            radius (float): convolution radius
            sigma (float): influence radius of each kernel point
            modulated (bool=False): Use modulations (self-attention)
            deformable (bool=False): Use deformable KPConv
            influence_mode (str='linear'): Influence function ('constant', 'linear', 'gaussian')
            aggregation_mode (str='sum'): Aggregation mode ('nearest', 'sum')
            dimension (int=3): dimension of input
            groups (int=1): Number of groups in KPConv
            norm_type (str='batch'): type of normalization used in layer ('group', 'batch', 'none')
            bn_momentum (float=0.98): Momentum for batch normalization
            activation (nn.Module|None=nn.LeakyReLU(0.1)): Activation function. Use None for no activation.
        """
        super(KPConvBlock, self).__init__()

        # Define parameters
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.radius = radius
        self.sigma = sigma
        self.influence_mode = influence_mode
        self.aggregation_mode = aggregation_mode
        self.dimension = dimension
        self.groups = groups
        self.norm_type = norm_type
        self.bn_momentum = bn_momentum

        # Define modules
        self.activation = activation

        if norm_type == 'none':
            self.norm = BatchNormBlock(out_channels, -1)
        elif norm_type == 'batch':
            self.norm = BatchNormBlock(out_channels, bn_momentum)
        elif norm_type == 'group':
            self.norm = GroupNormBlock(out_channels)
        else:
            raise ValueError('Unknown normalization type: {:s}. Must be in (\'group\', \'batch\', \'none\')'.format(norm_type))

        if deformable:
            self.conv = KPDef(in_channels,
                              out_channels,
                              kernel_size,
                              radius,
                              sigma,
                              modulated=modulated,
                              groups=groups,
                              dimension=dimension,
                              influence_mode=influence_mode,
                              aggregation_mode=aggregation_mode)
        else:
            self.conv = KPConv(in_channels,
                               out_channels,
                               kernel_size,
                               radius,
                               sigma,
                               modulated=modulated,
                               groups=groups,
                               dimension=dimension,
                               influence_mode=influence_mode,
                               aggregation_mode=aggregation_mode)

        return
        
    def forward(self, q_pts, s_pts, s_feats, neighbor_indices):
        q_feats = self.conv(q_pts, s_pts, s_feats, neighbor_indices)
        q_feats = self.norm(q_feats)
        q_feats = self.activation(q_feats)
        return q_feats

     
    def __repr__(self):
        return 'KPConvBlock(in_C: {:d}, out_C: {:d}, r: {:.2f}, modes: {:s}+{:s})'.format(self.in_channels,
                                                                                          self.out_channels,
                                                                                          self.radius,
                                                                                          self.influence_mode,
                                                                                          self.aggregation_mode)


class KPConvResidualBlock(nn.Module):

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 radius: float,
                 sigma: float,
                 modulated: bool = False,
                 deformable: bool = False,
                 influence_mode: str = 'linear',
                 aggregation_mode: str = 'sum',
                 dimension: int = 3,
                 groups: int = 1,
                 strided: bool = False,
                 norm_type: str = 'batch',
                 bn_momentum: float = 0.98,
                 activation: nn.Module = nn.LeakyReLU(0.1)):
        """
        KPConv residual bottleneck block.
        Args:
            in_channels (int): dimension input features
            out_channels (int): dimension input features
            kernel_size (int): number of kernel points
            radius (float): convolution radius
            sigma (float): influence radius of each kernel point
            modulated (bool=False): Use modulations (self-attention)
            deformable (bool=False): Use deformable KPConv
            influence_mode (str='linear'): Influence function ('constant', 'linear', 'gaussian')
            aggregation_mode (str='sum'): Aggregation mode ('nearest', 'sum')
            dimension (int=3): dimension of input
            groups (int=1): Number of groups in KPConv
            strided (bool=False): strided or not
            norm_type (str='batch'): type of normalization used in layer ('group', 'batch', 'none')
            bn_momentum (float=0.98): Momentum for batch normalization
            activation (nn.Module|None=nn.LeakyReLU(0.1)): Activation function. Use None for no activation.
        """
        super(KPConvResidualBlock, self).__init__()

        # Define parameters
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.strided = strided
        self.bn_momentum = bn_momentum
        self.norm_type = norm_type

        mid_channels = out_channels // 4
        
        # First downscaling mlp
        if in_channels != mid_channels:
            self.unary1 = UnaryBlock(in_channels, mid_channels, norm_type, bn_momentum)
        else:
            self.unary1 = nn.Identity()

        # KPConv block with normalizatio and activation
        self.conv = KPConvBlock(mid_channels,
                                mid_channels,
                                kernel_size,
                                radius,
                                sigma,
                                modulated=modulated,
                                deformable=deformable,
                                influence_mode=influence_mode,
                                aggregation_mode=aggregation_mode,
                                dimension=dimension,
                                groups=groups,
                                norm_type=norm_type,
                                bn_momentum=bn_momentum)

        # Second upscaling mlp
        self.unary2 = UnaryBlock(mid_channels, out_channels, norm_type, bn_momentum, activation=None)

        # Shortcut optional mpl
        if in_channels != out_channels:
            self.unary_shortcut = UnaryBlock(in_channels, out_channels, norm_type, bn_momentum, activation=None)
        else:
            self.unary_shortcut = nn.Identity()

        # Final activation function
        self.activation = activation

        return

    def forward(self, q_pts, s_pts, s_feats, neighbor_indices):
        
        # First downscaling mlp
        x = self.unary1(s_feats)

        # Convolution
        x = self.conv(q_pts, s_pts, x, neighbor_indices)

        # Second upscaling mlp
        x = self.unary2(x)

        # Shortcut
        if self.strided:
            shortcut = local_maxpool(s_feats, neighbor_indices)
        else:
            shortcut = s_feats
        shortcut = self.unary_shortcut(shortcut)

        # Final activation
        q_feats = x + shortcut
        q_feats = self.activation(q_feats)

        return q_feats

