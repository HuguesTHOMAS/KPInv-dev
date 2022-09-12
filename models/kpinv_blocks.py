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
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.init import kaiming_uniform_

from kernels.kernel_points import load_kernels
from models.generic_blocks import gather, index_select, radius_gaussian, local_maxpool, UnaryBlock, build_mlp

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


class KPInv(nn.Module):

    def __init__(self,
                 channels: int,
                 shell_sizes: list,
                 radius: float,
                 sigma: float,
                 reduction_ratio: int = 1,
                 weight_act: bool = False,
                 shared_kp_data = None,
                 dimension: int = 3,
                 influence_mode: str = 'linear',
                 fixed_kernel_points: str = 'center',
                 norm_type: str = 'batch',
                 bn_momentum: float = 0.98,
                 activation: nn.Module = nn.LeakyReLU(0.1),
                 inf: float = 1e6):
        """
        KPInv block. Very similar to KPMini, but with the weights being generated from the center feature
        Args:
            channels           (int): The number of channels.
            shell_sizes       (list): The number of kernel points per shell.
            radius           (float): The radius used for kernel point init.
            sigma            (float): The influence radius of each kernel point.
            reduction_ratio  (int=1): Reduction ratio for generating KPinv conv weights.
            weight_act  (bool=False): Activate the weight with sigmoid, softmax or tanh.
            shared_kp_data    (None): Optional data dict shared across the layer
            use_geom    (bool=False): Use geometric encodings
            groups           (int=1): Groups in convolution (=in_channels for depthwise conv).
            dimension        (int=3): The dimension of the point space.
            influence_mode      (str='linear'): Influence function ('constant', 'linear', 'gaussian').
            fixed_kernel_points (str='center'): kernel points whose position is fixed ('none', 'center' or 'verticals').
            norm_type            (str='batch'): type of normalization used in layer ('group', 'batch', 'none')
            bn_momentum           (float=0.98): Momentum for batch normalization
            activation (nn.Module|None=nn.LeakyReLU(0.1)): Activation function. Use None for no activation.
            inf (float=1e6): The value of infinity to generate the padding point.
        """
        super(KPInv, self).__init__()

        # Save parameters
        self.channels = channels
        self.shell_sizes = shell_sizes
        self.K = int(np.sum(shell_sizes))
        self.radius = radius
        self.sigma = sigma
        self.dimension = dimension
        self.influence_mode = influence_mode
        self.fixed_kernel_points = fixed_kernel_points
        self.inf = inf

        # Initialize kernel points
        self.share_kp = shared_kp_data is not None
        self.first_kp = False

        if self.share_kp:
            self.first_kp = 'k_pts' not in shared_kp_data
            self.shared_kp_data = shared_kp_data
            if self.first_kp:
                self.shared_kp_data['k_pts'] = self.initialize_kernel_points()
            self.register_buffer("kernel_points", self.shared_kp_data['k_pts'])
        else:
            self.shared_kp_data = {}
            kernel_points = self.initialize_kernel_points()
            self.register_buffer("kernel_points", kernel_points)
            self.shared_kp_data['k_pts'] = kernel_points

        # Merge and aggregation function
        self.merge_op = torch.mul
        self.aggr_op = torch.sum

        # Weight generation function
        reduction_ratio = 1
        self.alpha_mlp = build_mlp(n_layers=2,
                                   Cin=channels,
                                   Cmid=channels // reduction_ratio,
                                   Cout=self.K * self.channels,
                                   norm_type=norm_type,
                                   bn_momentum=bn_momentum,
                                   activation=activation)

        # Weight activation
        self.weight_activation = nn.Identity()
        if weight_act:
            self.weight_activation = torch.tanh()
            # self.weight_activation = torch.sigmoid()
            # self.weight_activation = torch.softmax(dim=1)

        return

    def initialize_kernel_points(self) -> Tensor:
        """
        Initialize the kernel point positions in a sphere
        :return: the tensor of kernel points
        """
        kernel_points = load_kernels(self.radius, self.shell_sizes, dimension=self.dimension, fixed=self.fixed_kernel_points)
        return torch.from_numpy(kernel_points).float()

    @torch.no_grad()
    def get_neighbors_influences(self, q_pts: Tensor,
                                 s_pts: Tensor,
                                 neighb_inds: Tensor) -> Tensor:
        """
        Influence function of kernel points on neighbors.
        Args:
            q_points (Tensor): query points (M, 3).
            s_points (Tensor): support points carrying input features (N, 3).
            neighb_inds (LongTensor): neighbor indices of query points among support points (M, H).
        """

        if self.share_kp and not self.first_kp:

            # We use data already computed from the first KPConv of the layer
            influence_weights = self.shared_kp_data['infl_w']
            neighbors = self.shared_kp_data['neighb_p']
            neighbors_1nn = self.shared_kp_data['neighb_1nn']

        else:

            # Add a fake point in the last row for shadow neighbors
            s_pts = torch.cat((s_pts, torch.zeros_like(s_pts[:1, :]) + self.inf), 0)   # (N, 3) -> (N+1, 3)

            # Get neighbor points [n_points, n_neighbors, dim]
            # neighbors = s_pts[neighb_inds, :]  # (N+1, 3) -> (M, H, 3)
            neighbors = index_select(s_pts, neighb_inds, dim=0)  # (N+1, 3) -> (M, H, 3)

            # Center every neighborhood
            neighbors = neighbors - q_pts.unsqueeze(1)  # (M, H, 3)

            if self.influence_mode == 'mlp':
                neighbors *= 1 / self.radius   # -> (M, H, 3)
                neighbors_1nn = None
                influence_weights = None

            else:

                # Get Kernel point distances to neigbors
                differences = neighbors.unsqueeze(2) - self.kernel_points  # (M, H, 1, 3) x (K, 3) -> (M, H, K, 3)
                sq_distances = torch.sum(differences ** 2, dim=3)  # (M, H, K)

                # Get nearest kernel point (M, H), values < K
                nn_sq_dists, neighbors_1nn = torch.min(sq_distances, dim=2)

                influence_weights = None
                if self.influence_mode != 'constant':

                    # Get Kernel point influences
                    if self.influence_mode == 'linear':
                        # Influence decrease linearly with the distance, and get to zero when d = sigma.
                        influence_weights = torch.clamp(1 - torch.sqrt(nn_sq_dists) / self.sigma, min=0.0)  # (M, H)

                    elif self.influence_mode == 'gaussian':
                        # Influence in gaussian of the distance.
                        gaussian_sigma = self.sigma * 0.3
                        influence_weights = radius_gaussian(nn_sq_dists, gaussian_sigma)  # (M, H)
                    else:
                        raise ValueError("Unknown influence mode: : '{:s}'.  Should be 'constant', 'linear', or 'gaussian'".format(self.influence_mode))

            # Share with next kernels if necessary
            if self.share_kp:

                self.shared_kp_data['neighb_1nn'] = neighbors_1nn
                self.shared_kp_data['neighb_p'] = neighbors
                self.shared_kp_data['infl_w'] = influence_weights

        return influence_weights, neighbors, neighbors_1nn

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

        # Get adaptive convolutional weights
        # **********************************

        # In case M == N, we can assume this is an in-place convolution.
        if q_pts.shape[0] == s_pts.shape[0]:
            pooled_feats = s_feats # (M, C) -> (M, C//r) -> (M, K*G)
        else:
            pooled_feats = neighbor_feats[:, 0, :]  # nearest pool (M, H, C) -> (M, C)
            # pooled_feats = torch.max(neighbor_feats, dim=1)  # max pool (M, H, C) -> (M, C)
            # pooled_feats = torch.mean(neighbor_feats, dim=1)  # avg pool (M, H, C) -> (M, C)
            
        conv_weights = self.alpha_mlp(pooled_feats) # (M, C) -> (M, C//r) -> (M, K*G)


        # Get neighbor/kernel influences
        # ******************************

        # Add a zero feature for shadow neighbors
        padded_s_feats = torch.cat((s_feats, torch.zeros_like(s_feats[:1, :])), 0)  # (N, C) -> (N+1, C)

        # Get the features of each neighborhood
        # neighbor_feats = gather(padded_s_feats, neighb_inds)  # (N+1, C) -> (M, H, C)
        neighbor_feats = index_select(padded_s_feats, neighb_inds, dim=0)

        # Get nearest kernel point (M, H) and weights applied to each neighbors (M, H)
        influence_weights, neighbors, neighbors_1nn = self.get_neighbors_influences(q_pts, s_pts, neighb_inds)


        # Apply convolution weights
        # *************************
        
        # Apply optional activation
        # conv_weights = self.weight_activation(conv_weights.view(-1, self.K, self.channels))
        conv_weights = conv_weights.view(-1, self.K, self.channels)

        # Collect nearest kernel point weights -> (M, H, C)
        neighbors_weights = gather(conv_weights, neighbors_1nn)

        # Apply influence weights
        if self.influence_mode != 'constant':
            neighbors_weights *= influence_weights.unsqueeze(2)

        # Apply optional activation
        neighbors_weights = self.weight_activation(neighbors_weights)

        # Apply weights and summation
        output_feats = self.aggr_op(self.merge_op(neighbor_feats, neighbors_weights), dim=1)  # (M, H, C) -> (M, C)


        return output_feats

    def __repr__(self):

        repr_str = 'KPInv'
        repr_str += '(K: {:d}'.format(self.K)
        repr_str += ', C: {:d}'.format(self.channels)
        repr_str += ', r: {:.2f}'.format(self.radius)
        repr_str += ', sigma: {:.2f})'.format(self.sigma)

        return repr_str







class KPInvOLD(nn.Module):

    def __init__(self,
                 channels: int,
                 shell_sizes: int,
                 radius: float,
                 sigma: float,
                 channels_per_group: int = 16,
                 reduction_ratio: int = 4,
                 influence_mode: str = 'linear',
                 aggregation_mode: str = 'sum',
                 dimension: int = 3,
                 norm_type: str = 'batch',
                 bn_momentum: float = 0.98,
                 fixed_kernel_points: str = 'center',
                 inf: float = 1e6):
        """
        Rigid KPInvOLD.
        Args:
            channels (int): The number of the input=output channels.
            shell_sizes (int): The number of kernel points.
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
        super(KPInvOLD, self).__init__()

        # Verification of group parameter
        assert channels % channels_per_group == 0, "channels must be divisible by channels_per_group."
        assert channels % reduction_ratio == 0, "channels must be divisible by reduction_ratio."
        groups = channels // channels_per_group

        # Save parameters
        self.channels = channels
        self.shell_sizes = shell_sizes
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
        self.gen_mlp = nn.Linear(channels // reduction_ratio, self.shell_sizes * self.groups, bias=True)

        # Initialize kernel points
        kernel_points = self.initialize_kernel_points()
        self.register_buffer("kernel_points", kernel_points)

        return

    def initialize_kernel_points(self) -> Tensor:
        """
        Initialize the kernel point positions in a sphere
        :return: the tensor of kernel points
        """
        kernel_points = load_kernels(self.radius, self.shell_sizes, dimension=self.dimension, fixed=self.fixed_kernel_points)
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
                neighbor_weights *= torch.transpose(nn.functional.one_hot(neighbors_1nn, self.shell_sizes), 1, 2)

            elif self.aggregation_mode != 'sum':
                raise ValueError("Unknown aggregation mode: '{:s}'. Should be 'nearest' or 'sum'".format(self.aggregation_mode))

        return neighbor_weights

    def forward(self, q_pts: Tensor,
                s_pts: Tensor,
                s_feats: Tensor,
                neighb_inds: Tensor) -> Tensor:
        """
        KPInvOLD forward.
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
        weighted_feats2 = weighted_feats.view(-1, self.shell_sizes, self.groups, self.channels_per_group)  # (M, K, C) -> (M, K, G, C//G)
        conv_weights2 = conv_weights.view(-1, self.shell_sizes, self.groups)  # (M, K*G) -> (M, K, G)

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
        repr_str += '(K: {:d}'.format(self.shell_sizes)
        repr_str += ', C: {:d}'.format(self.channels)
        repr_str += ', r: {:.2f}'.format(self.radius)
        repr_str += ', sigma: {:.2f})'.format(self.sigma)

        return repr_str




# ----------------------------------------------------------------------------------------------------------------------
#
#           Complex blocks
#       \********************/
#

class KPInvBlock(nn.Module):

    def __init__(self,
                 channels: int,
                 shell_sizes: int,
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
            shell_sizes (int): number of kernel points
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
        self.shell_sizes = shell_sizes
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
        self.norm = NormBlock(out_channels, norm_type, bn_momentum)


        self.conv = KPInv(channels,
                          shell_sizes,
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
                 shell_sizes: int,
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
            shell_sizes (int): number of kernel points
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
                               shell_sizes,
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
