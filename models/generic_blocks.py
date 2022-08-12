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

import torch
import torch.nn as nn
from torch import Tensor

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
    # outputs = gather(x, inds[:, 0])
    outputs = index_select(x, inds[:, 0], dim=0)

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
    # pool_features = gather(x, inds)
    pool_features = index_select(x, inds, dim=0)

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
            self.bias = nn.Parameter(torch.zeros(num_channels, dtype=torch.float32), requires_grad=True)
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
