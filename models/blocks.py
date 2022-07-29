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

import time
import math
import torch
import torch.nn as nn
from easydict import EasyDict

from typing import List

from torch import Tensor

from torch.nn.init import kaiming_uniform_
from kernels.kernel_points import load_kernels

from utils.gpu_neigbors import radius_search_pack_mode
from utils.gpu_subsampling import subsample_list_mode
from utils.ply import write_ply

# ----------------------------------------------------------------------------------------------------------------------
#
#           Simple functions
#       \**********************/
#


def index_select(inputs: Tensor, indices: Tensor, dim: int) -> Tensor:
    """Advanced indices select.
    Returns a tensor `output` which indexes the `inputs` tensor along dimension `dim` using the entries in `indices`
    which is a `LongTensor`.
    Different from `torch.indices_select`, `indices` does not have to be 1-D. The `dim`-th dimension of `inputs` will
    be expanded to the number of dimensions in `indices`.
    For example, suppose the shape `inputs` is $(a_0, a_1, ..., a_{n-1})$, the shape of `indices` is
    $(b_0, b_1, ..., b_{m-1})$, and `dim` is $i$, then `output` is $(n+m-1)$-d tensor, whose shape is
    $(a_0, ..., a_{i-1}, b_0, b_1, ..., b_{m-1}, a_{i+1}, ..., a_{n-1})$.
    Args:
        inputs (Tensor): (a_0, a_1, ..., a_{n-1})
        indices (LongTensor): (b_0, b_1, ..., b_{m-1})
        dim (int): The dimension to index.
    Returns:
        output (Tensor): (a_0, ..., a_{dim-1}, b_0, ..., b_{m-1}, a_{dim+1}, ..., a_{n-1})
    """
    outputs = inputs.index_select(dim, indices.view(-1))

    if indices.dim() > 1:
        if dim < 0:
            dim += inputs.dim()
        output_shape = inputs.shape[:dim] + indices.shape + inputs.shape[dim + 1 :]
        outputs = outputs.view(*output_shape)

    return outputs


def gather(x, idx, method=2):
    """
    implementation of a custom gather operation for faster backwards.
    :param x: input with shape [N, D_1, ... D_d]
    :param idx: indexing with shape [n_1, ..., n_m]
    :param method: Choice of the method
    :return: x[idx] with shape [n_1, ..., n_m, D_1, ... D_d]
    """

    if method == 0:
        return x[idx]
    elif method == 1:
        x = x.unsqueeze(1)
        x = x.expand((-1, idx.shape[-1], -1))
        idx = idx.unsqueeze(2)
        idx = idx.expand((-1, -1, x.shape[-1]))
        return x.gather(0, idx)
    elif method == 2:
        for i, ni in enumerate(idx.size()[1:]):
            x = x.unsqueeze(i+1)
            new_s = list(x.size())
            new_s[i+1] = ni
            x = x.expand(new_s)
        n = len(idx.size())
        for i, di in enumerate(x.size()[n:]):
            idx = idx.unsqueeze(i+n)
            new_s = list(idx.size())
            new_s[i+n] = di
            idx = idx.expand(new_s)
        return x.gather(0, idx)
    else:
        raise ValueError('Unkown method')


def radius_gaussian(sq_r, sig, eps=1e-9):
    """
    Compute a radius gaussian (gaussian of distance)
    :param sq_r: input radiuses [dn, ..., d1, d0]
    :param sig: extents of gaussians [d1, d0] or [d0] or float
    :return: gaussian of sq_r [dn, ..., d1, d0]
    """
    return torch.exp(-sq_r / (2 * sig**2 + eps))


def local_nearest_pool(x, inds):
    """
    Pools features from the nearest neighbors.
    WARNING: this function assumes the neighbors are ordered.
    Args:
        x (Tensor): The input features in the shape of (N, C).
        inds (LongTensor): The neighbor indices in the shape of (M, K).
    Returns:
        pooled_feats (Tensor): The pooled features in the shape of (M, C).
    """

    # Add a last row with minimum features for shadow pools
    x = torch.cat((x, torch.zeros_like(x[:1, :])), dim=0)

    # Get features for each pooling location
    outputs = gather(x, inds[:, 0])
    #outputs = index_select(x, inds[:, 0], dim=0)

    return outputs


def local_maxpool(x, inds):
    """
    Max pooling from neighbors in pack mode.
    Args:
        x (Tensor): The input features in the shape of (N, C).
        inds (LongTensor): The neighbor indices in the shape of (M, K).
    Returns:
        pooled_feats (Tensor): The pooled features in the shape of (M, C).
    """

    # Add a last row with minimum features for shadow pools (N+1, C)
    x = torch.cat((x, torch.zeros_like(x[:1, :])), 0)

    # Get all features for each pooling location (M, K, C)
    pool_features = gather(x, inds)
    # pool_features = index_select(x, inds, dim=0)

    # Pool the maximum (M, K)
    max_features, _ = torch.max(pool_features, 1)

    return max_features


def global_avgpool(x, lengths):
    """
    Global average pooling over batch.
    Args:
        x (Tensor): The input features in the shape of (N, C).
        lengths (LongTensor): The length of each sample in the batch in the shape of (B).
    Returns:
        feats (Tensor): The pooled features in the shape of (B, C).
    """

    # Loop over the clouds of the batch
    averaged_features = []
    i0 = 0
    for b_i, length in enumerate(lengths):

        # Average features for each batch cloud
        averaged_features.append(torch.mean(x[i0:i0 + length.item()], dim=0))

        # Increment for next cloud
        i0 += length

    # Average features in each batch
    return torch.stack(averaged_features)


@torch.no_grad()
def build_graph_pyramid(points: Tensor,
                        lengths: Tensor,
                        num_layers: int,
                        sub_size: float,
                        search_radius: float,
                        neighbor_limits: List[int],
                        sub_mode: str = 'grid'):
    """
    Build the graph pyramid, consisting of:
        > The subampled points for each layer, in pack mode.
        > The lengths of the pack.
        > The neigbors indices for convolutions.
        > The pooling indices (neighbors from one layer to another).
        > The upsampling indices (opposite of pooling indices).
    """

    # Results lists
    pyramid = EasyDict()
    pyramid.points = []
    pyramid.lengths = []
    pyramid.neighbors = []
    pyramid.pools = []
    pyramid.upsamples = []

    # Subsample all point clouds on GPU
    for i in range(num_layers):
        if i > 0:
            sub_points, sub_lengths = subsample_list_mode(points, lengths, sub_size, method=sub_mode)
        pyramid.points.append(sub_points)
        pyramid.lengths.append(sub_lengths)
        sub_size *= 2.0

    # Find all neighbors
    for i in range(num_layers):

        # Get current points
        cur_points = pyramid.points[i]
        cur_lengths = pyramid.lengths[i]

        # Get convolution indices
        neighbors = radius_search_pack_mode(cur_points, cur_points, cur_lengths, cur_lengths, search_radius, neighbor_limits[i])
        pyramid.neighbors.append(neighbors)

        # Relation with next layer 
        if i < num_layers - 1:
            sub_points = pyramid.points[i + 1]
            sub_lengths = pyramid.lengths[i + 1]

            # Get pooling indices
            subsampling_inds = radius_search_pack_mode(sub_points, cur_points, sub_lengths, cur_lengths, search_radius, neighbor_limits[i])
            pyramid.pools.append(subsampling_inds)

            upsampling_inds = radius_search_pack_mode(cur_points, sub_points, cur_lengths, sub_lengths, search_radius * 2, neighbor_limits[i + 1])
            pyramid.upsamples.append(upsampling_inds)

        # Increase radius for next layer
        search_radius *= 2

    return pyramid

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

    def forward(self, q_pts: Tensor,
                s_pts: Tensor,
                s_feats: Tensor,
                neighb_inds: Tensor) -> Tensor:
        """KPConv forward.
        Args:
            q_points (Tensor): query points (M, 3).
            s_points (Tensor): support points carrying input features (N, 3).
            s_feats (Tensor): input features values (N, C_in).
            neighb_inds (LongTensor): neighbor indices of query points among support points (M, H).
        Returns:
            q_feats (Tensor): output features carried by query points (M, C_out).
        """

        # Add a fake point in the last row for shadow neighbors
        s_pts = torch.cat((s_pts, torch.zeros_like(s_pts[:1, :]) + self.inf), 0)   # (N, 3) -> (N+1, 3)

        # Get neighbor points [n_points, n_neighbors, dim]
        neighbors = s_pts[neighb_inds, :]  # (N+1, 3) -> (M, H, 3)
        # neighbors = index_select(padded_s_points, neighb_inds, dim=0)  # (N+1, 3) -> (M, H, 3)

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

        # Add a zero feature for shadow neighbors
        padded_s_feats = torch.cat((s_feats, torch.zeros_like(s_feats[:1, :])), 0)  # (N, C) -> (N+1, C)

        # Get the features of each neighborhood
        neighbor_feats = gather(padded_s_feats, neighb_inds)  # (N+1, C) -> (M, H, C)
        # neighbor_feats = index_select(padded_s_feats, neighb_inds, dim=0) 

        # Apply distance weights
        weighted_feats = torch.matmul(neighbor_weights, neighbor_feats)  # (M, K, H) x (M, H, C) -> (M, K, C)


        # apply convolutional weights
        if self.groups == 1:

            # standard conv
            weighted_feats = weighted_feats.permute((1, 0, 2))  # (M, K, C) -> (K, M, C)
            kernel_outputs = torch.matmul(weighted_feats, self.weights)  # (K, M, C) x (K, C, O) -> (K, M, O)
            output_feats = torch.sum(kernel_outputs, dim=0)  # (K, M, O) -> (M, O)

            # output_feats = torch.einsum("mkc,kcd->md", weighted_feats, self.weights)  # (M, K, C) x (K, C, O) -> (M, O)

        else:
            # group conv
            weighted_feats = weighted_feats.view(-1, self.kernel_size, self.groups, self.in_channels_per_group)  # (M, K, C) -> (M/G, K, G, C)
            output_feats = torch.einsum("mkgc,kgcd->mgd", weighted_feats, self.weights)  # (M/G, K, G, C) -> (M/G, G, O)
            output_feats = output_feats.view(-1, self.out_channels)  # (M/G, G, O) -> (M, O)

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


class BatchNormBlock(nn.Module):

    def __init__(self,
                 num_channels: int,
                 bn_momentum: float = 0.98):
        """
        Initialize a batch normalization block. If network does not use batch normalization, replace with biases.
        Args:
            in_channels (int): dimension input features
            bn_momentum (float=0.98): Momentum for batch normalization. < 0 to avoid using it.
        """
        super(BatchNormBlock, self).__init__()
        
        # Define parameters
        self.num_channels = num_channels
        self.bn_momentum = bn_momentum

        if bn_momentum > 0:
            self.batch_norm = nn.BatchNorm1d(num_channels, momentum=bn_momentum)
        else:
            self.bias = Parameter(torch.zeros(num_channels, dtype=torch.float32), requires_grad=True)
        return

    def reset_parameters(self):
        nn.init.zeros_(self.bias)

    def forward(self, x):

        if self.bn_momentum > 0:
            x = x.transpose(0, 1).unsqueeze(0)  # (N, C) -> (B=1, C, N)
            x = self.batch_norm(x)
            x = x.squeeze(0).transpose(0, 1)  # (B=1, C, N) -> (N, C)
            return x

        else:
            return x + self.bias

    def __repr__(self):
        return 'BatchNormBlock(num_C: {:d}, momentum: {:.2f}, only_bias: {:s})'.format(self.in_dim,
                                                                                       self.bn_momentum,
                                                                                       str(self.bn_momentum <= 0))


class GroupNormBlock(nn.Module):

    def __init__(self,
                 num_channels: int,
                 num_groups: int = 32,
                 eps: float = 1e-5, 
                 affine: bool = True):
        """
        Initialize a group normalization block. If network does not use batch normalization, replace with biases.
        Args:
            num_channels (int): dimension input features
            num_groups (float=0.98): Momentum for batch normalization. < 0 to avoid using it.
            eps (float=1e-5): a value added to the denominator for numerical stability.
            affine (bool=True): Should the module have learnable per-channel affine parameters.
        """
        super(GroupNormBlock, self).__init__()

        # Define parameters
        self.num_channels = num_channels
        self.num_groups = num_groups
        
        # Adjust number of groups
        while self.num_groups > 1:
            if num_channels % self.num_groups == 0:
                num_channels_per_group = num_channels // self.num_groups
                if num_channels_per_group >= 8:
                    break
            self.num_groups = self.num_groups // 2

        # Verify that the best number of groups have ben found
        assert self.num_groups != 1, (
            f"Cannot find 'num_groups' in GroupNorm with 'num_channels={num_channels}' automatically. "
            "Please manually specify 'num_groups'."
        )

        # Define layer
        self.norm = nn.GroupNorm(self.num_groups, num_channels, eps=eps, affine=affine)
        
        return

    def reset_parameters(self):
        nn.init.zeros_(self.bias)
         
    def forward(self, x):
        x = x.transpose(0, 1).unsqueeze(0)  # (N, C) -> (B=1, C, N)
        x = self.norm(x)
        x = x.squeeze(0).transpose(0, 1)  # (B=1, C, N) -> (N, C)
        return x

    def __repr__(self):
        return 'GroupNormBlock(num_C: {:d}, groups: {:d})'.format(self.in_dim,
                                                                  self.num_groups)


class UnaryBlock(nn.Module):

    
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 norm_type: str = 'batch',
                 bn_momentum: float = 0.98,
                 activation: nn.Module = nn.LeakyReLU(0.1)):
        """
        Unary block with normalization and activation in pack mode.
        Args:
            in_channels (int): dimension input features
            out_channels (int): dimension input features
            norm_type (str='batch'): type of normalization used in layer ('group', 'batch', 'none')
            bn_momentum (float=0.98): Momentum for batch normalization. < 0 to avoid using it.
            activation (nn.Module|None=nn.LeakyReLU(0.1)): Activation function. Use None for no activation.
        """
        super(UnaryBlock, self).__init__()

        # Define parameters
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.norm_type = norm_type
        self.bn_momentum = bn_momentum

        # Define modules
        self.mlp = nn.Linear(in_channels, out_channels, bias=False)
        self.activation = activation

        if norm_type == 'none':
            self.norm = BatchNormBlock(out_channels, -1)
        elif norm_type == 'batch':
            self.norm = BatchNormBlock(out_channels, bn_momentum)
        elif norm_type == 'group':
            self.norm = GroupNormBlock(out_channels)
        else:
            raise ValueError('Unknown normalization type: {:s}. Must be in (\'group\', \'batch\', \'none\')'.format(norm_type))

        return

    def forward(self, x):
        x = self.mlp(x)
        x = self.norm(x)
        if self.activation is not None:
            x = self.activation(x)
        return x

    def __repr__(self):
        return 'UnaryBlock(in_C: {:d}, out_C: {:d}, norm: {:s})'.format(self.in_channels,
                                                                        self.out_channels,
                                                                        self.norm_type)


class KPConvBlock(nn.Module):

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 radius: float,
                 sigma: float,
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

        self.conv = KPConv(in_channels,
                           out_channels,
                           kernel_size,
                           radius,
                           sigma,
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


class KPResidualBlock(nn.Module):

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 radius: float,
                 sigma: float,
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
            influence_mode (str='linear'): Influence function ('constant', 'linear', 'gaussian')
            aggregation_mode (str='sum'): Aggregation mode ('nearest', 'sum')
            dimension (int=3): dimension of input
            groups (int=1): Number of groups in KPConv
            strided (bool=False): strided or not
            norm_type (str='batch'): type of normalization used in layer ('group', 'batch', 'none')
            bn_momentum (float=0.98): Momentum for batch normalization
            activation (nn.Module|None=nn.LeakyReLU(0.1)): Activation function. Use None for no activation.
        """
        super(KPResidualBlock, self).__init__()

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


class GlobalAverageBlock(nn.Module):

    def __init__(self):
        """
        Initialize a global average block with its ReLU and BatchNorm.
        """
        super(GlobalAverageBlock, self).__init__()
        return

    def forward(self, x, lengths):
        return global_avgpool(x, lengths)


class NearestUpsampleBlock(nn.Module):

    def __init__(self):
        """
        Initialize a nearest upsampling block with its ReLU and BatchNorm.
        """
        super(NearestUpsampleBlock, self).__init__()
        return

    def forward(self, x, upsample_inds):
        return local_nearest_pool(x, upsample_inds)


class MaxPoolBlock(nn.Module):

    def __init__(self):
        """
        Initialize a max pooling block with its ReLU and BatchNorm.
        """
        super(MaxPoolBlock, self).__init__()
        return

    def forward(self, x, neighb_inds):
        return local_maxpool(x, neighb_inds)

