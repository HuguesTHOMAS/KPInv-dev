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
from typing import Callable

from kernels.kernel_points import load_kernels
from models.generic_blocks import index_select, radius_gaussian, local_maxpool, UnaryBlock, BatchNormBlock, GroupNormBlock

# ----------------------------------------------------------------------------------------------------------------------
#
#           Simple functions
#       \**********************/
#



# ----------------------------------------------------------------------------------------------------------------------
#
#           KPInv class
#       \*****************/
#


class point_involution_v1(nn.Module):

    def __init__(self,
                 channels: int,
                 neighborhood_size: int,
                 alpha_layers: int = 2,
                 alpha_reduction: int = 1,
                 channels_per_group: int = 8,
                 stride_mode: str = 'nearest',
                 dimension: int = 3,
                 norm_type: str = 'batch',
                 bn_momentum: float = 0.98):
        """
        Naive implementation of point involution. 
        Has the following problems:
            > No geometric encodings  
            > attention weights are assigned to neighbors according to 

        N.B. Compared to the original paper 
            our "channels_per_group" = their     "groups" 
            our      "groups"        = their "group_channels"
        We changed for clarity and consistency with previous works like PointTransformers.

        Args:
            channels (int): The number of the input=output channels.
            neighborhood_size (int): The number of knn neighbors (has to be fixed in advance).

            alpha_layers (int=16): number of layers in MLP alpha.
            alpha_reduction (int=16): number of layers in MLP alpha.

            sigma (float): The influence radius of each kernel point.
            channels_per_group (int=16): number of channels per group in convolution.
            reduction_ratio (int=4): Reduction ratio for generating KPinv conv weights.
            influence_mode (str='linear'): Influence function ('constant', 'linear', 'gaussian').
            aggregation_mode (str='sum'): Aggregation mode ('nearest', 'sum').
            dimension (int=3): The dimension of the point space.
            norm_type (str='batch'): type of normalization used in layer ('group', 'batch', 'none')
            bn_momentum (float=0.98): Momentum for batch normalization
            fixed_kernel_points (str='center'): kernel points whose position is fixed ('none', 'center' or 'verticals').
            inf (float=1e6): The value of infinity to generate the padding point.
        """
        super(KPInv, self).__init__()

        # Verification of group parameter
        assert channels % channels_per_group == 0, "channels must be divisible by channels_per_group."
        assert channels % reduction_ratio == 0, "channels must be divisible by reduction_ratio."
        groups = channels // channels_per_group

        # Save parameters
        self.channels = channels
        self.kernel_size = kernel_size
        self.radius = radius
        self.sigma = sigma
        self.groups = groups
        self.channels_per_group = channels_per_group
        self.reduction_ratio = reduction_ratio
        self.dimension = dimension
        self.influence_mode = influence_mode
        self.aggregation_mode = aggregation_mode
        self.fixed_kernel_points = fixed_kernel_points
        self.inf = inf
        
        # MLP for kernel weights generation (first ones reduces the feature dimension for efficiency)
        self.reduce_mlp = UnaryBlock(channels, channels // reduction_ratio, norm_type, bn_momentum)

        # MLP for kernel weights generation (second one doe not have activation as it need to predict any value)
        self.gen_mlp = nn.Linear(channels // reduction_ratio, self.kernel_size * self.groups, bias=True)

        # Initialize kernel points
        kernel_points = self.initialize_kernel_points()
        self.register_buffer("kernel_points", kernel_points)

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
        KPInv forward.
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


        # Get adaptive convolutional weights
        # **********************************

        # In case M == N, we can assume this is an in-place convolution.
        if q_pts.shape[0] == s_pts.shape[0]:
            conv_weights = self.gen_mlp(self.reduce_mlp(s_feats)) # (M, C) -> (M, C//r) -> (M, K*G)
        else:
            pooled_feats = neighbor_feats[:, 0, :]  # nearest pool (M, H, C) -> (M, C)
            # pooled_feats = torch.max(neighbor_feats, dim=1)  # max pool (M, H, C) -> (M, C)
            # pooled_feats = torch.mean(neighbor_feats, dim=1)  # avg pool (M, H, C) -> (M, C)
            conv_weights = self.gen_mlp(self.reduce_mlp(pooled_feats)) # (M, C) -> (M, C//r) -> (M, K*G)


        # Transfer features to kernel points
        # **********************************

        # Get Kernel point influences 
        neighbor_weights = self.get_neighbors_influences(q_pts, s_pts, neighb_inds)  # (M, K, H)

        # Apply influence weights
        weighted_feats = torch.matmul(neighbor_weights, neighbor_feats)  # (M, K, H) x (M, H, C) -> (M, K, C)


        # Apply convolutional weights
        # ***************************
        
        # Separate features in groups
        weighted_feats2 = weighted_feats.view(-1, self.kernel_size, self.groups, self.channels_per_group)  # (M, K, C) -> (M, K, G, C//G)
        conv_weights2 = conv_weights.view(-1, self.kernel_size, self.groups)  # (M, K*G) -> (M, K, G)

        # (M, K, G, C//G) x (M, K, G) -> (M, G, C//G)
        output_feats = torch.sum(weighted_feats2 * conv_weights2.unsqueeze(-1), dim=1)
        # test_einsum = torch.einsum("mkgc,mkg->mgc", weighted_feats, conv_weights)
        output_feats = output_feats.view(-1, self.channels)  # (M, G, O//G) -> (M, O)

        # # density normalization (divide output features by the sum of neighbor positive features)
        # neighbor_feats_sum = torch.sum(neighbor_feats, dim=-1)  # (M, H)
        # neighbor_num = torch.sum(torch.gt(neighbor_feats_sum, 0.0), dim=-1)  # (M,)
        # neighbor_num = torch.max(neighbor_num, torch.ones_like(neighbor_num))  # (M,)
        # output_feats = output_feats / neighbor_num.unsqueeze(1)

        return output_feats

    def __repr__(self):

        repr_str = 'KPInv'
        repr_str += '(K: {:d}'.format(self.kernel_size)
        repr_str += ', C: {:d}'.format(self.channels)
        repr_str += ', r: {:.2f}'.format(self.radius)
        repr_str += ', sigma: {:d})'.format(self.sigma)

        return repr_str



class point_involution_v1(nn.Module):

    def __init__(self,
                 channels: int,
                 neighborhood_size: int,
                 radius: float,
                 alpha_layers: int = 2,
                 alpha_reduction: int = 1,
                 channels_per_group: int = 8,
                 stride_mode: str = 'nearest',

                 
                 influence_mode: str = 'linear',
                 aggregation_mode: str = 'sum',
                 dimension: int = 3,
                 norm_type: str = 'batch',
                 bn_momentum: float = 0.98,
                 fixed_kernel_points: str = 'center',
                 inf: float = 1e6):
        """
        General implementation of point involution. Has full control over a bunch of parameters.
        N.B. Compared to the original paper 
            > our "channels_per_group" = their "groups" 
            > our "groups+ = their "group_channels"
        We changed for clarity and consistency with previous works like PointTransformers

        Args:
            channels (int): The number of the input=output channels.
            kernel_size (int): The number of kernel points.
            radius (float): The radius used for kernel point init.
            sigma (float): The influence radius of each kernel point.
            channels_per_group (int=16): number of channels per group in convolution.
            reduction_ratio (int=4): Reduction ratio for generating KPinv conv weights.
            influence_mode (str='linear'): Influence function ('constant', 'linear', 'gaussian').
            aggregation_mode (str='sum'): Aggregation mode ('nearest', 'sum').
            dimension (int=3): The dimension of the point space.
            norm_type (str='batch'): type of normalization used in layer ('group', 'batch', 'none')
            bn_momentum (float=0.98): Momentum for batch normalization
            fixed_kernel_points (str='center'): kernel points whose position is fixed ('none', 'center' or 'verticals').
            inf (float=1e6): The value of infinity to generate the padding point.
        """
        super(KPInv, self).__init__()

        # Verification of group parameter
        assert channels % channels_per_group == 0, "channels must be divisible by channels_per_group."
        assert channels % reduction_ratio == 0, "channels must be divisible by reduction_ratio."
        groups = channels // channels_per_group

        # Save parameters
        self.channels = channels
        self.kernel_size = kernel_size
        self.radius = radius
        self.sigma = sigma
        self.groups = groups
        self.channels_per_group = channels_per_group
        self.reduction_ratio = reduction_ratio
        self.dimension = dimension
        self.influence_mode = influence_mode
        self.aggregation_mode = aggregation_mode
        self.fixed_kernel_points = fixed_kernel_points
        self.inf = inf
        
        # MLP for kernel weights generation (first ones reduces the feature dimension for efficiency)
        self.reduce_mlp = UnaryBlock(channels, channels // reduction_ratio, norm_type, bn_momentum)

        # MLP for kernel weights generation (second one doe not have activation as it need to predict any value)
        self.gen_mlp = nn.Linear(channels // reduction_ratio, self.kernel_size * self.groups, bias=True)

        # Initialize kernel points
        kernel_points = self.initialize_kernel_points()
        self.register_buffer("kernel_points", kernel_points)

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
        KPInv forward.
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


        # Get adaptive convolutional weights
        # **********************************

        # In case M == N, we can assume this is an in-place convolution.
        if q_pts.shape[0] == s_pts.shape[0]:
            conv_weights = self.gen_mlp(self.reduce_mlp(s_feats)) # (M, C) -> (M, C//r) -> (M, K*G)
        else:
            pooled_feats = neighbor_feats[:, 0, :]  # nearest pool (M, H, C) -> (M, C)
            # pooled_feats = torch.max(neighbor_feats, dim=1)  # max pool (M, H, C) -> (M, C)
            # pooled_feats = torch.mean(neighbor_feats, dim=1)  # avg pool (M, H, C) -> (M, C)
            conv_weights = self.gen_mlp(self.reduce_mlp(pooled_feats)) # (M, C) -> (M, C//r) -> (M, K*G)


        # Transfer features to kernel points
        # **********************************

        # Get Kernel point influences 
        neighbor_weights = self.get_neighbors_influences(q_pts, s_pts, neighb_inds)  # (M, K, H)

        # Apply influence weights
        weighted_feats = torch.matmul(neighbor_weights, neighbor_feats)  # (M, K, H) x (M, H, C) -> (M, K, C)


        # Apply convolutional weights
        # ***************************
        
        # Separate features in groups
        weighted_feats2 = weighted_feats.view(-1, self.kernel_size, self.groups, self.channels_per_group)  # (M, K, C) -> (M, K, G, C//G)
        conv_weights2 = conv_weights.view(-1, self.kernel_size, self.groups)  # (M, K*G) -> (M, K, G)

        # (M, K, G, C//G) x (M, K, G) -> (M, G, C//G)
        output_feats = torch.sum(weighted_feats2 * conv_weights2.unsqueeze(-1), dim=1)
        # test_einsum = torch.einsum("mkgc,mkg->mgc", weighted_feats, conv_weights)
        output_feats = output_feats.view(-1, self.channels)  # (M, G, O//G) -> (M, O)

        # # density normalization (divide output features by the sum of neighbor positive features)
        # neighbor_feats_sum = torch.sum(neighbor_feats, dim=-1)  # (M, H)
        # neighbor_num = torch.sum(torch.gt(neighbor_feats_sum, 0.0), dim=-1)  # (M,)
        # neighbor_num = torch.max(neighbor_num, torch.ones_like(neighbor_num))  # (M,)
        # output_feats = output_feats / neighbor_num.unsqueeze(1)

        return output_feats

    def __repr__(self):

        repr_str = 'KPInv'
        repr_str += '(K: {:d}'.format(self.kernel_size)
        repr_str += ', C: {:d}'.format(self.channels)
        repr_str += ', r: {:.2f}'.format(self.radius)
        repr_str += ', sigma: {:d})'.format(self.sigma)

        return repr_str




# ----------------------------------------------------------------------------------------------------------------------
#
#           Complex blocks
#       \********************/
#

class KPInvBlock(nn.Module):

    def __init__(self,
                 channels: int,
                 kernel_size: int,
                 radius: float,
                 sigma: float,
                 channels_per_group: int = 16,
                 reduction_ratio: int = 4,
                 influence_mode: str = 'linear',
                 aggregation_mode: str = 'sum',
                 dimension: int = 3,
                 norm_type: str = 'batch',
                 bn_momentum: float = 0.98,
                 activation: nn.Module = nn.LeakyReLU(0.1)):
        """
        KPInv block with normalization and activation.  
        Args:
            channels (int): dimension input=output features
            kernel_size (int): number of kernel points
            radius (float): convolution radius
            sigma (float): influence radius of each kernel point
            channels_per_group (int=16): number of channels per group in convolution.
            reduction_ratio (int=4): Reduction ratio for generating KPinv conv weights.
            influence_mode (str='linear'): Influence function ('constant', 'linear', 'gaussian')
            aggregation_mode (str='sum'): Aggregation mode ('nearest', 'sum')
            dimension (int=3): dimension of input
            norm_type (str='batch'): type of normalization used in layer ('group', 'batch', 'none')
            bn_momentum (float=0.98): Momentum for batch normalization
            activation (nn.Module|None=nn.LeakyReLU(0.1)): Activation function. Use None for no activation.
        """
        super(KPInvBlock, self).__init__()

        # Define parameters
        self.channels = channels
        self.kernel_size = kernel_size
        self.radius = radius
        self.sigma = sigma
        self.channels_per_group = channels_per_group
        self.influence_mode = influence_mode
        self.aggregation_mode = aggregation_mode
        self.dimension = dimension
        self.norm_type = norm_type
        self.bn_momentum = bn_momentum

        # Define modules
        self.activation = activation

        if norm_type == 'none':
            self.norm = BatchNormBlock(channels, -1)
        elif norm_type == 'batch':
            self.norm = BatchNormBlock(channels, bn_momentum)
        elif norm_type == 'group':
            self.norm = GroupNormBlock(channels)
        else:
            raise ValueError('Unknown normalization type: {:s}. Must be in (\'group\', \'batch\', \'none\')'.format(norm_type))

        self.conv = KPInv(channels,
                          kernel_size,
                          radius,
                          sigma,
                          channels_per_group=channels_per_group,
                          reduction_ratio=reduction_ratio,
                          influence_mode=influence_mode,
                          aggregation_mode=aggregation_mode,
                          dimension=dimension,
                          norm_type=norm_type,
                          bn_momentum=bn_momentum)

        return
        
    def forward(self, q_pts, s_pts, s_feats, neighbor_indices):
        q_feats = self.conv(q_pts, s_pts, s_feats, neighbor_indices)
        q_feats = self.norm(q_feats)
        q_feats = self.activation(q_feats)
        return q_feats

     
    def __repr__(self):
        return 'KPInvBlock(C: {:d}, r: {:.2f}, modes: {:s}+{:s})'.format(self.channels,
                                                                         self.radius,
                                                                         self.influence_mode,
                                                                         self.aggregation_mode)


class KPInvResidualBlock(nn.Module):

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 radius: float,
                 sigma: float,
                 channels_per_group: int = 16,
                 reduction_ratio: int = 4,
                 influence_mode: str = 'linear',
                 aggregation_mode: str = 'sum',
                 dimension: int = 3,
                 strided: bool = False,
                 norm_type: str = 'batch',
                 bn_momentum: float = 0.98,
                 activation: nn.Module = nn.LeakyReLU(0.1)):
        """
        KPInv residual bottleneck block.
        Args:
            in_channels (int): dimension input features
            out_channels (int): dimension input features
            kernel_size (int): number of kernel points
            radius (float): convolution radius
            sigma (float): influence radius of each kernel point
            channels_per_group (int=16): number of channels per group in convolution.
            reduction_ratio (int=4): Reduction ratio for generating KPinv conv weights.
            influence_mode (str='linear'): Influence function ('constant', 'linear', 'gaussian')
            aggregation_mode (str='sum'): Aggregation mode ('nearest', 'sum')
            dimension (int=3): dimension of input
            strided (bool=False): strided or not
            norm_type (str='batch'): type of normalization used in layer ('group', 'batch', 'none')
            bn_momentum (float=0.98): Momentum for batch normalization
            activation (nn.Module|None=nn.LeakyReLU(0.1)): Activation function. Use None for no activation.
        """
        super(KPInvResidualBlock, self).__init__()

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


        # KPInv block with normalization and activation
        self.conv = KPInvBlock(mid_channels,
                               kernel_size,
                               radius,
                               sigma,
                               channels_per_group=channels_per_group,
                               reduction_ratio=reduction_ratio,
                               influence_mode=influence_mode,
                               aggregation_mode=aggregation_mode,
                               dimension=dimension,
                               norm_type=norm_type,
                               bn_momentum=bn_momentum,
                               activation=activation)

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
