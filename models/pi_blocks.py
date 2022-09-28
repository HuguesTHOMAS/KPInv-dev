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
from collections import OrderedDict



from kernels.kernel_points import load_kernels
from models.generic_blocks import index_select, radius_gaussian, local_maxpool, UnaryBlock, BatchNormBlock, GroupNormBlock, NormBlock

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
                 radius: float,
                 groups: int = 8,
                 channels_per_group: int = -1,
                 alpha_layers: int = 2,
                 alpha_reduction: int = 1,
                 delta_layers: int = 2,
                 delta_reduction: int = 1,
                 normalize_p: bool = False,
                 geom_mode: str = 'sub',
                 stride_mode: str = 'nearest',
                 dimension: int = 3,
                 norm_type: str = 'batch',
                 bn_momentum: float = 0.1,
                 activation: nn.Module = nn.LeakyReLU(0.1)):
        """
        v1 of point involution (naive and naiv+geom). It can includes geometric encodings
        Has the following problems:
            > attention weights are assigned to neighbors according to knn order

        N.B. Compared to the original paper 
            our "channels_per_group" = their     "groups" 
            our      "groups"        = their "group_channels"
        We changed for clarity and consistency with previous works like PointTransformers.

        Args:
            channels              (int): The number of the input=output channels.
            radius              (float): The radius used for geometric normalization.
            neighborhood_size     (int): The number of neighbors to be able to generate weights for each.
            groups              (int=8): number of groups in involution.
            channels_per_group (int=32): number of channels per group in involution. Ignored if group is specified
            alpha_layers        (int=2): number of layers in MLP alpha.
            alpha_reduction     (int=1): Reduction ratio for MLP alpha.
            double_delta (bool = False): Are we using double delta network (v4)
            normalize_p  (bool = False): Are we normalizing geometric data for encodings
            geom_mode       (str='add'): Mode for geometric encoding merge ('add', 'sub', 'mul', 'cat')
            stride_mode (str='nearest'): Mode for strided attention ('nearest', 'avg', 'max')
            dimension           (int=3): The dimension of the point space.
            norm_type     (str='batch'): type of normalization used in layer ('group', 'batch', 'none')
            bn_momentum    (float=0.10): Momentum for batch normalization
            activation (nn.Module|None=nn.LeakyReLU(0.1)): Activation function. Use None for no activation.
        """
        super(point_involution_v1, self).__init__()

        # Verification of group parameter
        if groups > 0:
            assert channels % groups == 0, "channels must be divisible by channels_per_group."
            channels_per_group = channels // groups
        else:
            assert channels % channels_per_group == 0, "channels must be divisible by channels_per_group."
            groups = channels // channels_per_group
        assert channels % alpha_reduction == 0, "channels must be divisible by reduction_ratio."

        # Save parameters
        self.channels = channels
        self.radius = radius
        self.neighborhood_size = neighborhood_size
        self.channels_per_group = channels_per_group
        self.groups = groups
        self.alpha_layers = alpha_layers
        self.alpha_reduction = alpha_reduction
        self.delta_layers = delta_layers
        self.delta_reduction = delta_reduction
        self.normalize_p = normalize_p
        self.geom_mode = geom_mode
        self.stride_mode = stride_mode
        self.dimension = dimension

        # Define MLP alpha
        C = channels
        R = alpha_reduction
        Cout = self.neighborhood_size * channels_per_group
        if alpha_layers < 2:
            self.alpha_mlp = nn.Sequential(NormBlock(C),
                                           activation,
                                           nn.Linear(C, Cout))

        else:
            self.alpha_mlp = nn.Sequential(NormBlock(C),
                                           activation,
                                           UnaryBlock(C, C // R, norm_type, bn_momentum, activation))
            for _ in range(alpha_layers - 2):
                self.alpha_mlp.append(UnaryBlock(C // R, C // R, norm_type, bn_momentum, activation))
            self.alpha_mlp.append(nn.Linear(C // R, Cout))

        # Define MLP delta
        D = self.dimension
        R = delta_reduction
        if delta_layers < 2:
            self.delta_mlp = nn.Linear(D, C)
        else:
            self.delta_mlp = nn.Sequential(UnaryBlock(D, C // R, norm_type, bn_momentum, activation))
            for _ in range(delta_layers - 2):
                self.delta_mlp.append(UnaryBlock(C // R, C // R, norm_type, bn_momentum, activation))
            self.delta_mlp.append(nn.Linear(C // R, C))
            
        # Define MLP gamma
        use_gamma_mlp = True
        if geom_mode == 'cat':
            self.gamma_mlp = nn.Sequential(NormBlock(2 * C),
                                           activation,
                                           nn.Linear(2 * C, C))
        elif use_gamma_mlp:
            self.gamma_mlp = nn.Sequential(NormBlock(C),
                                           activation,
                                           nn.Linear(C, C))
        else:
            self.gamma_mlp = nn.Identity()
        
        self.softmax = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()

        self.use_geom = geom_mode in ['add', 'sub', 'mul', 'cat']

        return

    def forward(self, q_pts: Tensor,
                s_pts: Tensor,
                s_feats: Tensor,
                neighb_inds: Tensor) -> Tensor:
        """
        Args:
            q_points (Tensor): query points (M, 3).
            s_points (Tensor): support points carrying input features (N, 3).
            s_feats (Tensor): input features values (N, C_in).
            neighb_inds (LongTensor): neighbor indices of query points among support points (M, H).
        Returns:
            q_feats (Tensor): output features carried by query points (M, C_out).
        """

        # Shadow neighbors have to be handled via a mask
        # **********************************************

        with torch.no_grad():
            valid_mask = neighb_inds < int(s_feats.shape[0])
            shadow_bool = not torch.all(valid_mask).item()


        # Get features for each neighbor
        # ******************************

        # Get the features of each neighborhood
        padded_s_feats = torch.cat((s_feats, torch.zeros_like(s_feats[:1, :])), 0)  # (N, C) -> (N+1, C)
        neighb_v_feats = index_select(padded_s_feats, neighb_inds, dim=0)  # -> (M, H, C)


        # Get geometric encoding features
        # *******************************

        if self.use_geom:
        
            with torch.no_grad():

                # Add a fake point in the last row for shadow neighbors
                s_pts = torch.cat((s_pts, torch.zeros_like(s_pts[:1, :])), 0)   # (N, 3) -> (N+1, 3)

                # Get neighbor points [n_points, n_neighbors, dim]
                neighbors = index_select(s_pts, neighb_inds, dim=0)  # (N+1, 3) -> (M, H, 3)

                # Center every neighborhood
                neighbors = (neighbors - q_pts.unsqueeze(1))

                # Rescale for normalization
                if self.normalize_p:
                    neighbors *= 1 / self.radius   # -> (M, H, 3)
            
            # Generate geometric encodings
            geom_encodings = self.delta_mlp(neighbors) # (M, H, 3) -> (M, H, C)
                
            # Merge with features
            if self.geom_mode == 'add':
                neighb_v_feats += geom_encodings  # -> (M, H, C)
            elif self.geom_mode == 'sub':
                neighb_v_feats -= geom_encodings  # -> (M, H, C)
            elif self.geom_mode == 'mul':
                neighb_v_feats *= geom_encodings  # -> (M, H, C)
            elif self.geom_mode == 'cat':
                neighb_v_feats = torch.cat((neighb_v_feats, geom_encodings), dim=2)  # -> (M, H, 2C)

            # Final linear transform
            neighb_v_feats = self.gamma_mlp(neighb_v_feats) # (M, H, C) -> (M, H, C)


        # Get attention weights
        # *********************

        # Get query features from the center point
        if q_pts.shape[0] == s_pts.shape[0]:
            q_feats = s_feats  # In case M == N, supports and queries are the same
        elif self.stride_mode == 'nearest':
            q_feats = neighb_v_feats[:, 0, :]  # nearest pool (M, H, C) -> (M, C)
        elif self.stride_mode == 'max':
            q_feats = torch.max(neighb_v_feats, dim=1)  # max pool (M, H, C) -> (M, C)
        elif self.stride_mode == 'avg':
            q_feats = torch.mean(neighb_v_feats, dim=1)  # avg pool (M, H, C) -> (M, C)
        

        # Generate attention weights
        attention_weights = self.alpha_mlp(q_feats)  # (M, C) -> (M, H*CpG)
        H = int(neighb_inds.shape[1])
        attention_weights = attention_weights.view(-1, H, self.channels_per_group, 1)  # (M, H*CpG) -> (M, H, CpG, 1)
        attention_weights = self.softmax(attention_weights)

        

        # Apply attention weights
        # ************************

        # Separate features in groups
        neighb_v_feats = neighb_v_feats.view(-1, H, self.channels_per_group, self.groups)  # (M, H, C) -> (M, H, CpG, G)

        # Multiply features with attention
        neighb_v_feats *= attention_weights  # -> (M, H, CpG, G)

        # Apply shadow mask (every gradient for shadow neighbors will be zero)
        if shadow_bool:
            neighb_v_feats *= valid_mask.type(torch.float32).unsqueeze(2).unsqueeze(3)

        # Sum over neighbors
        output_feats = torch.sum(neighb_v_feats, dim=1)  # -> (M, CpG, G)
        
        # Reshape
        output_feats = output_feats.view(-1, self.channels)  # -> (M, C)

        return output_feats

    def __repr__(self):

        repr_str = 'point_involution_v1'
        repr_str += '(C: {:d}'.format(self.channels)
        repr_str += ', G: {:d})'.format(self.groups)

        return repr_str


class point_involution_v3(nn.Module):

    def __init__(self,
                 channels: int,
                 radius: float,
                 groups: int = 8,
                 channels_per_group: int = -1,
                 alpha_layers: int = 2,
                 alpha_reduction: int = 1,
                 delta_layers: int = 2,
                 delta_reduction: int = 1,
                 double_delta: bool = False,
                 normalize_p: bool = False,
                 geom_mode: str = 'sub',
                 stride_mode: str = 'nearest',
                 dimension: int = 3,
                 norm_type: str = 'batch',
                 bn_momentum: float = 0.1,
                 activation: nn.Module = nn.LeakyReLU(0.1)):
        """
        v3 of point involution. It includes geometric encodings in both features and attention branches

        N.B. Compared to the original paper 
            our "channels_per_group" = their     "groups" 
            our      "groups"        = their "group_channels"
        We changed for clarity and consistency with previous works like PointTransformers.

        Args:
            channels              (int): The number of the input=output channels.
            radius              (float): The radius used for geometric normalization.
            groups              (int=8): number of groups in involution.
            channels_per_group (int=32): number of channels per group in involution. Ignored if group is specified
            alpha_layers        (int=2): number of layers in MLP alpha.
            alpha_reduction     (int=1): Reduction ratio for MLP alpha.
            double_delta (bool = False): Are we using double delta network (v4)
            normalize_p  (bool = False): Are we normalizing geometric data for encodings
            geom_mode       (str='add'): Mode for geometric encoding merge ('add', 'sub', 'mul', 'cat')
            stride_mode (str='nearest'): Mode for strided attention ('nearest', 'avg', 'max')
            dimension           (int=3): The dimension of the point space.
            norm_type     (str='batch'): type of normalization used in layer ('group', 'batch', 'none')
            bn_momentum    (float=0.10): Momentum for batch normalization
            activation (nn.Module|None=nn.LeakyReLU(0.1)): Activation function. Use None for no activation.
        """
        super(point_involution_v3, self).__init__()

        # Verification of group parameter
        if groups > 0:
            assert channels % groups == 0, "channels must be divisible by channels_per_group."
            channels_per_group = channels // groups
        else:
            assert channels % channels_per_group == 0, "channels must be divisible by channels_per_group."
            groups = channels // channels_per_group
        assert channels % alpha_reduction == 0, "channels must be divisible by reduction_ratio."

        # Save parameters
        self.channels = channels
        self.radius = radius
        self.channels_per_group = channels_per_group
        self.groups = groups
        self.alpha_layers = alpha_layers
        self.alpha_reduction = alpha_reduction
        self.delta_layers = delta_layers
        self.delta_reduction = delta_reduction
        self.double_delta = double_delta
        self.normalize_p = normalize_p
        self.geom_mode = geom_mode
        self.stride_mode = stride_mode
        self.dimension = dimension

        # Define MLP alpha
        C = channels
        R = alpha_reduction
        CpG = channels_per_group
        if geom_mode == 'cat':
            Cin = 2 * C
        else:
            Cin = C

        if alpha_layers < 2:
            self.alpha_mlp = nn.Sequential(NormBlock(Cin),
                                           activation,
                                           nn.Linear(Cin, CpG))

        else:
            self.alpha_mlp = nn.Sequential(NormBlock(Cin),
                                           activation,
                                           UnaryBlock(Cin, C // R, norm_type, bn_momentum, activation))
            for _ in range(alpha_layers - 2):
                self.alpha_mlp.append(UnaryBlock(C // R, C // R, norm_type, bn_momentum, activation))
            self.alpha_mlp.append(nn.Linear(C // R, CpG))

        # Define MLP delta
        D = self.dimension
        R = delta_reduction
        if delta_layers < 2:
            self.delta_mlp = nn.Linear(D, C)
        else:
            self.delta_mlp = nn.Sequential(UnaryBlock(D, C // R, norm_type, bn_momentum, activation))
            for _ in range(delta_layers - 2):
                self.delta_mlp.append(UnaryBlock(C // R, C // R, norm_type, bn_momentum, activation))
            self.delta_mlp.append(nn.Linear(C // R, C))
        if double_delta:
            if delta_layers < 2:
                self.delta2_mlp = nn.Linear(D, C)
            else:
                self.delta2_mlp = nn.Sequential(UnaryBlock(D, C // R, norm_type, bn_momentum, activation))
                for _ in range(delta_layers - 2):
                    self.delta2_mlp.append(UnaryBlock(C // R, C // R, norm_type, bn_momentum, activation))
                self.delta2_mlp.append(nn.Linear(C // R, C))


        # Define MLP gamma
        use_gamma_mlp = True
        if geom_mode == 'cat':
            self.gamma_mlp = nn.Sequential(NormBlock(2 * C),
                                           activation,
                                           nn.Linear(2 * C, C))
        elif use_gamma_mlp:
            self.gamma_mlp = nn.Sequential(NormBlock(C),
                                           activation,
                                           nn.Linear(C, C))
        else:
            self.gamma_mlp = nn.Identity()
        
        self.softmax = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()
        self.activation =activation

        return

    def forward(self, q_pts: Tensor,
                s_pts: Tensor,
                s_feats: Tensor,
                neighb_inds: Tensor) -> Tensor:
        """
        Args:
            q_points (Tensor): query points (M, 3).
            s_points (Tensor): support points carrying input features (N, 3).
            s_feats (Tensor): input features values (N, C_in).
            neighb_inds (LongTensor): neighbor indices of query points among support points (M, H).
        Returns:
            q_feats (Tensor): output features carried by query points (M, C_out).
        """

        # Shadow neighbors have to be handled via a mask
        # **********************************************

        with torch.no_grad():
            valid_mask = neighb_inds < int(s_feats.shape[0])
            shadow_bool = not torch.all(valid_mask).item()


        # Get features for each neighbor
        # ******************************

        # Get the features of each neighborhood
        padded_s_feats = torch.cat((s_feats, torch.zeros_like(s_feats[:1, :])), 0)  # (N, C) -> (N+1, C)
        neighb_v_feats = index_select(padded_s_feats, neighb_inds, dim=0)  # -> (M, H, C)


        # Get geometric encoding features
        # *******************************
        
        with torch.no_grad():

            # Add a fake point in the last row for shadow neighbors
            s_pts = torch.cat((s_pts, torch.zeros_like(s_pts[:1, :])), 0)   # (N, 3) -> (N+1, 3)

            # Get neighbor points [n_points, n_neighbors, dim]
            neighbors = index_select(s_pts, neighb_inds, dim=0)  # (N+1, 3) -> (M, H, 3)

            # Center every neighborhood
            neighbors = (neighbors - q_pts.unsqueeze(1))

            # Rescale for normalization
            if self.normalize_p:
                neighbors *= 1 / self.radius   # -> (M, H, 3)
        
        # Generate geometric encodings
        geom_encodings = self.delta_mlp(neighbors) # (M, H, 3) -> (M, H, C)
        if self.double_delta:
            geom_encodings2 = self.delta2_mlp(neighbors) # (M, H, 3) -> (M, H, C)
        else:
            geom_encodings2 = geom_encodings
            
        # Merge with features
        if self.geom_mode == 'add':
            neighb_v_feats += geom_encodings  # -> (M, H, C)
        elif self.geom_mode == 'sub':
            neighb_v_feats -= geom_encodings  # -> (M, H, C)
        elif self.geom_mode == 'mul':
            neighb_v_feats *= geom_encodings  # -> (M, H, C)
        elif self.geom_mode == 'cat':
            neighb_v_feats = torch.cat((neighb_v_feats, geom_encodings), dim=2)  # -> (M, H, 2C)

        # Final linear transform
        neighb_v_feats = self.gamma_mlp(neighb_v_feats) # (M, H, C) -> (M, H, C)


        # Get attention weights
        # *********************

        # Get query features from the center point
        if q_pts.shape[0] == s_pts.shape[0]:
            q_feats = s_feats  # In case M == N, supports and queries are the same
        elif self.stride_mode == 'nearest':
            q_feats = neighb_v_feats[:, 0, :]  # nearest pool (M, H, C) -> (M, C)
        elif self.stride_mode == 'max':
            q_feats = torch.max(neighb_v_feats, dim=1)  # max pool (M, H, C) -> (M, C)
        elif self.stride_mode == 'avg':
            q_feats = torch.mean(neighb_v_feats, dim=1)  # avg pool (M, H, C) -> (M, C)
        
        # Merge geometric encodings with feature
        q_feats = q_feats.unsqueeze(1)    # -> (M, 1, C)
        if self.geom_mode == 'add':
            q_feats = q_feats + geom_encodings2  # -> (M, H, C)
        elif self.geom_mode == 'sub':
            q_feats = q_feats - geom_encodings2  # -> (M, H, C)
        elif self.geom_mode == 'mul':
            q_feats = q_feats * geom_encodings2  # -> (M, H, C)
        elif self.geom_mode == 'cat':
            q_feats = torch.cat((q_feats, geom_encodings2), dim=2)  # -> (M, H, 2C)

        # Generate attention weights
        attention_weights = self.alpha_mlp(q_feats)  # (M, H, C) -> (M, H, CpG)
        attention_weights = self.softmax(attention_weights)

        
        # Apply attention weights
        # ************************

        # Separate features in groups
        H = int(neighb_inds.shape[1])
        neighb_v_feats = neighb_v_feats.view(-1, H, self.channels_per_group, self.groups)  # (M, H, C) -> (M, H, CpG, G)
        attention_weights = attention_weights.view(-1, H, self.channels_per_group, 1)  # (M, H, CpG) -> (M, H, CpG, 1)

        # Multiply features with attention
        neighb_v_feats *= attention_weights  # -> (M, H, CpG, G)

        # Apply shadow mask (every gradient for shadow neighbors will be zero)
        if shadow_bool:
            neighb_v_feats *= valid_mask.type(torch.float32).unsqueeze(2).unsqueeze(3)

        # Sum over neighbors
        output_feats = torch.sum(neighb_v_feats, dim=1)  # -> (M, CpG, G)
        
        # Reshape
        output_feats = output_feats.view(-1, self.channels)  # -> (M, C)

        return output_feats

    def __repr__(self):

        repr_str = 'point_involution_v3'
        repr_str += '(C: {:d}'.format(self.channels)
        repr_str += ', G: {:d})'.format(self.groups)

        return repr_str


class point_transformer(nn.Module):

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 radius: float,
                 groups: int = 8,
                 channels_per_group: int = -1,
                 alpha_layers: int = 2,
                 delta_layers: int = 2,
                 delta_reduction: int = 4,
                 double_delta: bool = False,
                 normalize_p: bool = False,
                 geom_mode: str = 'sub',
                 stride_mode: str = 'nearest',
                 dimension: int = 3,
                 norm_type: str = 'batch',
                 bn_momentum: float = 0.1,
                 activation: nn.Module = nn.LeakyReLU(0.1)):
        """
        Reimplementation of point_transformer. Basically a point_involution_v3, with keys that introduce features in the attention process.

        List of difference:
            - delta MLP: we have 3 -> C/4 -> C  instead of  3 -> 3 -> C
            - possibility of double delta
            - possibility of normalize p_c

        Args:
            in_channels           (int): The number of input channels.
            out_channels          (int): The number of output channels.
            radius              (float): The radius used for geometric normalization.
            groups              (int=8): number of groups in involution.
            channels_per_group (int=32): number of channels per group in involution. Ignored if group is specified
            alpha_layers        (int=2): number of layers in MLP alpha.
            double_delta (bool = False): Are we using double delta network (v4)
            normalize_p  (bool = False): Are we normalizing geometric data for encodings
            geom_mode       (str='add'): Mode for geometric encoding merge ('add', 'sub', 'mul', 'cat')
            stride_mode (str='nearest'): Mode for strided attention ('nearest', 'avg', 'max')
            dimension           (int=3): The dimension of the point space.
            norm_type     (str='batch'): type of normalization used in layer ('group', 'batch', 'none')
            bn_momentum    (float=0.10): Momentum for batch normalization
        """
        super(point_transformer, self).__init__()

        # Verification of group parameter
        if groups > 0:
            assert out_channels % groups == 0, "channels must be divisible by channels_per_group."
            channels_per_group = out_channels // groups
        else:
            assert out_channels % channels_per_group == 0, "channels must be divisible by channels_per_group."
            groups = out_channels // channels_per_group

        # Save parameters
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.radius = radius
        self.channels_per_group = channels_per_group
        self.groups = groups
        self.alpha_layers = alpha_layers
        self.delta_layers = delta_layers
        self.delta_reduction = delta_reduction
        self.double_delta = double_delta
        self.normalize_p = normalize_p
        self.geom_mode = geom_mode
        self.stride_mode = stride_mode
        self.dimension = dimension
        self.activation = activation

        # Define first linear transforms
        Cin = in_channels
        C = out_channels
        self.linear_q = nn.Linear(Cin, C)
        self.linear_k = nn.Linear(Cin, C)
        self.linear_v = nn.Linear(Cin, C)

        # Define MLP alpha
        CpG = channels_per_group
        if geom_mode == 'cat':
            Cin = 2 * C
        else:
            Cin = C
        if alpha_layers < 2:
            self.alpha_mlp = nn.Sequential(NormBlock(Cin),
                                           activation,
                                           nn.Linear(Cin, CpG))
        else:
            self.alpha_mlp = nn.Sequential(NormBlock(Cin),
                                           activation,
                                           UnaryBlock(Cin, CpG, norm_type, bn_momentum, activation))
            for _ in range(alpha_layers - 2):
                self.alpha_mlp.append(UnaryBlock(CpG, CpG, norm_type, bn_momentum, activation))
            self.alpha_mlp.append(nn.Linear(CpG, CpG))

        # Define MLP delta
        D = self.dimension
        R = delta_reduction
        if delta_layers < 2:
            self.delta_mlp = nn.Linear(D, C)
        else:
            self.delta_mlp = nn.Sequential(UnaryBlock(D, C // R, norm_type, bn_momentum, activation))
            for _ in range(delta_layers - 2):
                self.delta_mlp.append(UnaryBlock(C // R, C // R, norm_type, bn_momentum, activation))
            self.delta_mlp.append(nn.Linear(C // R, C))
        if double_delta:
            if delta_layers < 2:
                self.delta2_mlp = nn.Linear(D, C)
            else:
                self.delta2_mlp = nn.Sequential(UnaryBlock(D, C // R, norm_type, bn_momentum, activation))
                for _ in range(delta_layers - 2):
                    self.delta2_mlp.append(UnaryBlock(C // R, C // R, norm_type, bn_momentum, activation))
                self.delta2_mlp.append(nn.Linear(C // R, C))

        # Define MLP gamma
        use_gamma_mlp = False
        if geom_mode == 'cat':
            self.gamma_mlp = nn.Linear(2 * C, C)
        elif use_gamma_mlp:
            self.gamma_mlp = nn.Linear(C, C)
        else:
            self.gamma_mlp = nn.Identity()
        
        self.softmax = nn.Softmax(dim=1)
        
        # Set this to false to have something very similar to involution (some additional linears)
        self.use_k_feats = True

        return

    def forward(self, q_pts: Tensor,
                s_pts: Tensor,
                s_feats: Tensor,
                neighb_inds: Tensor) -> Tensor:
        """
        Args:
            q_points (Tensor): query points (M, 3).
            s_points (Tensor): support points carrying input features (N, 3).
            s_feats (Tensor): input features values (N, C_in).
            neighb_inds (LongTensor): neighbor indices of query points among support points (M, H).
        Returns:
            q_feats (Tensor): output features carried by query points (M, C_out).
        """

        # Shadow neighbors have to be handled via a mask
        # **********************************************

        with torch.no_grad():
            valid_mask = neighb_inds < int(s_feats.shape[0])
            shadow_bool = not torch.all(valid_mask).item()


        # Get features for each neighbor
        # ******************************

        # Get the features of each neighborhood
        v_feats = self.linear_v(s_feats)
        padded_v_feats = torch.cat((v_feats, torch.zeros_like(v_feats[:1, :])), 0)  # (N, C) -> (N+1, C)
        neighb_v_feats = index_select(padded_v_feats, neighb_inds, dim=0)  # -> (M, H, C)

        # Get keys features from neighbors
        if self.use_k_feats:
            k_feats = self.linear_k(s_feats)
            padded_k_feats = torch.cat((k_feats, torch.zeros_like(k_feats[:1, :])), 0)  # (N, C) -> (N+1, C)
            neighb_k_feats = index_select(padded_k_feats, neighb_inds, dim=0)  # -> (M, H, C)

        # Get geometric encoding features
        # *******************************
        
        with torch.no_grad():

            # Add a fake point in the last row for shadow neighbors
            s_pts = torch.cat((s_pts, torch.zeros_like(s_pts[:1, :])), 0)   # (N, 3) -> (N+1, 3)

            # Get neighbor points [n_points, n_neighbors, dim]
            neighbors = index_select(s_pts, neighb_inds, dim=0)  # (N+1, 3) -> (M, H, 3)

            # Center every neighborhood
            neighbors = (neighbors - q_pts.unsqueeze(1))

            # Rescale for normalization
            if self.normalize_p:
                neighbors *= 1 / self.radius   # -> (M, H, 3)
        
        # Generate geometric encodings
        geom_encodings = self.delta_mlp(neighbors) # (M, H, 3) -> (M, H, C)
        if self.double_delta:
            geom_encodings2 = self.delta2_mlp(neighbors) # (M, H, 3) -> (M, H, C)
        else:
            geom_encodings2 = geom_encodings
        
        # Merge with features
        if self.geom_mode == 'add':
            neighb_v_feats += geom_encodings  # -> (M, H, C)
        elif self.geom_mode == 'sub':
            neighb_v_feats -= geom_encodings  # -> (M, H, C)
        elif self.geom_mode == 'mul':
            neighb_v_feats *= geom_encodings  # -> (M, H, C)
        elif self.geom_mode == 'cat':
            neighb_v_feats = torch.cat((neighb_v_feats, geom_encodings), dim=2)  # -> (M, H, 2C)

        # Final linear transform
        neighb_v_feats = self.gamma_mlp(neighb_v_feats) # (M, H, C) -> (M, H, C)


        # Get attention weights
        # *********************

        # Get query features from the center point
        q_feats = self.linear_q(s_feats)

        # In case M != N, pool features to query positions
        if q_pts.shape[0] != s_pts.shape[0]:
            padded_q_feats = torch.cat((q_feats, torch.zeros_like(q_feats[:1, :])), 0)  # (N, C) -> (N+1, C)
            if self.stride_mode == 'nearest':
                q_feats = index_select(padded_q_feats, neighb_inds[:, 0], dim=0)  # nearest pool -> (M, C)
            elif self.stride_mode == 'max':
                q_feats = torch.max(index_select(padded_q_feats, neighb_inds, dim=0), dim=1)  # max pool (M, H, C) -> (M, C)
            elif self.stride_mode == 'avg':
                q_feats = torch.mean(index_select(padded_q_feats, neighb_inds, dim=0), dim=1)  # avg pool (M, H, C) -> (M, C)

        # Merge with keys
        qk_feats = q_feats.unsqueeze(1)    # -> (M, 1, C)
        if self.use_k_feats:
            qk_feats = qk_feats - neighb_k_feats  # -> (M, H, C)

        # Merge geometric encodings with feature
        if self.geom_mode == 'add':
            qk_feats = qk_feats + geom_encodings2  # -> (M, H, C)
        elif self.geom_mode == 'sub':
            qk_feats = qk_feats - geom_encodings2  # -> (M, H, C)
        elif self.geom_mode == 'mul':
            qk_feats = qk_feats * geom_encodings2  # -> (M, H, C)
        elif self.geom_mode == 'cat':
            qk_feats = torch.cat((qk_feats, geom_encodings2), dim=2)  # -> (M, H, 2C)

        # Generate attention weights
        attention_weights = self.alpha_mlp(qk_feats) # (M, H, C) -> (M, H, CpG)
        attention_weights = self.softmax(attention_weights)

        # Apply attention weights
        # ************************

        # Separate features in groups
        H = int(neighb_inds.shape[1])
        neighb_v_feats = neighb_v_feats.view(-1, H, self.channels_per_group, self.groups)  # (M, H, C) -> (M, H, CpG, G)
        attention_weights = attention_weights.view(-1, H, self.channels_per_group, 1)  # (M, H*CpG) -> (M, H, CpG, 1)

        # Multiply features with attention
        neighb_v_feats *= attention_weights  # -> (M, H, CpG, G)

        # Apply shadow mask (every gradient for shadow neighbors will be zero)
        if shadow_bool:
            # print('shadow neighbors present', s_feats.shape)
            neighb_v_feats *= valid_mask.type(torch.float32).unsqueeze(2).unsqueeze(3)

        # Sum over neighbors
        output_feats = torch.sum(neighb_v_feats, dim=1)  # -> (M, CpG, G)
        
        # Reshape
        output_feats = output_feats.view(-1, self.out_channels)  # -> (M, C)

        return output_feats

    def __repr__(self):

        repr_str = 'point_transformer'
        repr_str += '(Cin: {:d}'.format(self.in_channels)
        repr_str += ', Cout: {:d}'.format(self.out_channels)
        repr_str += ', G: {:d})'.format(self.groups)

        return repr_str




# ----------------------------------------------------------------------------------------------------------------------
#
#           Complex blocks
#       \********************/
#

class InvolutionBlock(nn.Module):

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 radius: float,
                 neighborhood_size: int,
                 inv_mode: str,
                 groups: int = 8,
                 dimension: int = 3,
                 norm_type: str = 'batch',
                 bn_momentum: float = 0.1,
                 activation: nn.Module = nn.LeakyReLU(0.1)):
        """
        InvolutionBlock block with normalization and activation.  
        Args:
            in_channels           (int): The number of input channels.
            out_channels          (int): The number of output channels.
            radius              (float): convolution radius
            neighborhood_size     (int): number of neighbor points
            inv_mode              (str): type of involution used in layer
            groups              (int=8): number of groups in involution.
            dimension           (int=3): dimension of input
            norm_type     (str='batch'): type of normalization used in layer ('group', 'batch', 'none')
            bn_momentum    (float=0.10): Momentum for batch normalization
            activation (nn.Module|None=nn.LeakyReLU(0.1)): Activation function. Use None for no activation.
        """
        super(InvolutionBlock, self).__init__()

        # Define parameters
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.radius = radius
        self.neighborhood_size = neighborhood_size
        self.inv_mode = inv_mode

        # Define modules
        self.activation = activation
        self.norm = NormBlock(out_channels, norm_type, bn_momentum)

        if inv_mode == 'inv_v1':
            assert in_channels == out_channels
            self.involution = point_involution_v1(out_channels,
                                                  neighborhood_size,
                                                  radius,
                                                  groups=groups,
                                                  geom_mode='none',
                                                  dimension=dimension,
                                                  norm_type=norm_type,
                                                  bn_momentum=bn_momentum)

        elif inv_mode == 'inv_v2':
            assert in_channels == out_channels
            self.involution = point_involution_v1(out_channels,
                                                  neighborhood_size,
                                                  radius,
                                                  groups=groups,
                                                  dimension=dimension,
                                                  norm_type=norm_type,
                                                  bn_momentum=bn_momentum)

        elif inv_mode == 'inv_v3':
            assert in_channels == out_channels
            self.involution = point_involution_v3(out_channels,
                                                  radius,
                                                  groups=groups,
                                                  double_delta=False,
                                                  dimension=dimension,
                                                  norm_type=norm_type,
                                                  bn_momentum=bn_momentum)

        elif inv_mode == 'inv_v4':
            self.involution = point_involution_v3(out_channels,
                                                  radius,
                                                  groups=groups,
                                                  double_delta=True,
                                                  dimension=dimension,
                                                  norm_type=norm_type,
                                                  bn_momentum=bn_momentum)

        elif inv_mode == 'transformer':
            self.involution = point_transformer(in_channels,
                                                out_channels,
                                                radius,
                                                groups=groups,
                                                dimension=dimension,
                                                norm_type=norm_type,
                                                bn_momentum=bn_momentum)


        return
        
    def forward(self, q_pts, s_pts, s_feats, neighbor_indices):
        q_feats = self.involution(q_pts, s_pts, s_feats, neighbor_indices)
        q_feats = self.norm(q_feats)
        q_feats = self.activation(q_feats)
        return q_feats

     
    def __repr__(self):
        return 'KPInvBlock(C: {:d} -> {:d}, r: {:.2f})'.format(self.in_channels,
                                                                         self.out_channels,
                                                                         self.radius)


class InvolutionResidualBlock(nn.Module):

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 radius: float,
                 neighborhood_size: int,
                 inv_mode: str,
                 groups: int = 16,
                 strided: bool = False,
                 dimension: int = 3,
                 norm_type: str = 'batch',
                 bn_momentum: float = 0.1,
                 activation: nn.Module = nn.LeakyReLU(0.1)):
        """
        Involution residual bottleneck block.
        Args:
            in_channels           (int): The number of input channels.
            out_channels          (int): The number of output channels.
            radius              (float): convolution radius
            neighborhood_size     (int): number of neighbor points
            inv_mode              (str): type of involution used in layer
            groups              (int=8): number of groups in involution.
            strided (bool=False): strided or not
            dimension           (int=3): dimension of input
            norm_type     (str='batch'): type of normalization used in layer ('group', 'batch', 'none')
            bn_momentum    (float=0.10): Momentum for batch normalization
            activation (nn.Module|None=nn.LeakyReLU(0.1)): Activation function. Use None for no activation.
        """
        super(InvolutionResidualBlock, self).__init__()

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
        self.conv = InvolutionBlock(mid_channels,
                                    mid_channels,
                                    radius,
                                    neighborhood_size,
                                    inv_mode=inv_mode,
                                    groups=groups,
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


















###############################################################################################################################################################

test = False
if test:

    class LocalAggregation(nn.Module):
        def __init__(self,
                    channels: List[int],
                    norm_args={'norm': 'bn1d'},
                    act_args={'act': 'relu'},
                    group_args={'NAME': 'ballquery',
                                'radius': 0.1, 'nsample': 16},
                    conv_args=None,
                    feature_type='dp_fj',
                    reduction='max',
                    last_act=True,
                    **kwargs
                    ):
            super().__init__()
            if kwargs:
                logging.warning(
                    f"kwargs: {kwargs} are not used in {__class__.__name__}")


            channels[0] = CHANNEL_MAP[feature_type](channels[0]) # C[0] = C[0] + 3


            convs = []
            for i in range(len(channels) - 1):  # #layers in each blocks
                convs.append(create_convblock2d(channels[i], channels[i + 1],
                                                norm_args=norm_args,
                                                act_args=None if i == (len(channels) - 2) and not last_act else act_args,
                                                **conv_args)
                            )
            self.convs = nn.Sequential(*convs)
            self.grouper = create_grouper(group_args)

            reduction = 'mean' if reduction.lower() == 'avg' else reduction.lower()
            self.reduction = reduction
            assert reduction in ['sum', 'max', 'mean']
            if reduction == 'max':
                self.pool = lambda x: torch.max(x, dim=-1, keepdim=False)[0]
            elif reduction == 'mean':
                self.pool = lambda x: torch.mean(x, dim=-1, keepdim=False)
            elif reduction == 'sum':
                self.pool = lambda x: torch.sum(x, dim=-1, keepdim=False)

        def forward(self, px) -> torch.Tensor:
            # p: position, x: feature
            p, x = px
            # neighborhood_features
            dp, xj = self.grouper(p, p, x)
            # dp (B, 3, M, H)
            # xj (B, C, M, H)
            x = torch.cat((dp, xj), dim=1)  # (B, C+3, M, H)
            x = self.convs(x)  # -> (B, C, M, H)
            x = self.pool(x)
            """ DEBUG neighbor numbers. 
            if x.shape[-1] != 1:
                query_xyz, support_xyz = p, p
                radius = self.grouper.radius
                dist = torch.cdist(query_xyz.cpu(), support_xyz.cpu())
                points = len(dist[dist < radius]) / (dist.shape[0] * dist.shape[1])
                logging.info(
                    f'query size: {query_xyz.shape}, support size: {support_xyz.shape}, radius: {radius}, num_neighbors: {points}')
            DEBUG end """
            return x


    class SetAbstraction(nn.Module):
        """The modified set abstraction module in PointNet++ with residual connection support
        """
        def __init__(self,
                    in_channels, out_channels,
                    layers=1,
                    stride=1,
                    group_args={'NAME': 'ballquery',
                                'radius': 0.1, 'nsample': 16},
                    norm_args={'norm': 'bn1d'},
                    act_args={'act': 'relu'},
                    conv_args=None,
                    sample_method='fps',
                    use_res=False,
                    is_head=False,
                    ):
            super().__init__()

            self.stride = stride
            self.is_head = is_head

            # current blocks aggregates all spatial information.
            self.all_aggr = not is_head and stride == 1  # False everytime stride = 4
            self.use_res = use_res and not self.all_aggr and not self.is_head

            mid_channel = out_channels // 2 if stride > 1 else out_channels
            channels = [in_channels] + [mid_channel] * (layers - 1) + [out_channels]
            channels[0] = in_channels + 3 * (not is_head)

            if self.use_res:
                self.skipconv = create_convblock1d(in_channels,
                                                channels[-1],
                                                norm_args=None,
                                                act_args=None) if in_channels != channels[-1] else nn.Identity()
                self.act = create_act(act_args)


            create_conv = create_convblock1d if is_head else create_convblock2d
            convs = []
            for i in range(len(channels) - 1):
                convs.append(create_conv(channels[i],
                                        channels[i + 1],
                                        norm_args=norm_args if not is_head else None,
                                        act_args=None if i == len(channels) - 2 and (self.use_res or is_head) else act_args,
                                        **conv_args))
            self.convs = nn.Sequential(*convs)

            if not is_head:
                if self.all_aggr:
                    group_args.nsample = None
                    group_args.radius = None
                self.grouper = create_grouper(group_args)
                self.pool = lambda x: torch.max(x, dim=-1, keepdim=False)[0]
                if sample_method.lower() == 'fps':
                    self.sample_fn = furthest_point_sample
                elif sample_method.lower() == 'random':
                    self.sample_fn = random_sample


        def forward(self, px):
            p, x = px
            if self.is_head:
                x = self.convs(x)  # (n, c)
            else:
                if not self.all_aggr:
                    idx = self.sample_fn(p, p.shape[1] // self.stride).long()
                    new_p = torch.gather(p, 1, idx.unsqueeze(-1).expand(-1, -1, 3))
                else:
                    new_p = p
                """ DEBUG neighbor numbers. 
                query_xyz, support_xyz = new_p, p
                radius = self.grouper.radius
                dist = torch.cdist(query_xyz.cpu(), support_xyz.cpu())
                points = len(dist[dist < radius]) / (dist.shape[0] * dist.shape[1])
                logging.info(f'query size: {query_xyz.shape}, support size: {support_xyz.shape}, radius: {radius}, num_neighbors: {points}')
                DEBUG end """
                if self.use_res:
                    identity = torch.gather(x, -1, idx.unsqueeze(1).expand(-1, x.shape[1], -1))
                    identity = self.skipconv(identity)
                dp, xj = self.grouper(new_p, p, x)

                # dp (B, 3, M, H)
                # xj (B, C, M, H)

                # Concat geom then multiple mlps

                x = self.pool(self.convs(torch.cat((dp, xj), dim=1)))
                if self.use_res:
                    x = self.act(x + identity)
                p = new_p
            return p, x


    class FeaturePropogation(nn.Module):
        """The Feature Propogation module in PointNet++
        """

        def __init__(self, mlp,
                    upsample=True,
                    norm_args={'norm': 'bn1d'},
                    act_args={'act': 'relu'}
                    ):
            """
            Args:
                mlp: [current_channels, next_channels, next_channels]
                out_channels:
                norm_args:
                act_args:
            """
            super().__init__()
            if not upsample:
                self.linear2 = nn.Sequential(
                    nn.Linear(mlp[0], mlp[1]), nn.ReLU(inplace=True))
                mlp[1] *= 2
                linear1 = []
                for i in range(1, len(mlp) - 1):
                    linear1.append(create_convblock1d(mlp[i], mlp[i + 1],
                                                    norm_args=norm_args, act_args=act_args
                                                    ))
                self.linear1 = nn.Sequential(*linear1)
            else:
                convs = []
                for i in range(len(mlp) - 1):
                    convs.append(create_convblock1d(mlp[i], mlp[i + 1],
                                                    norm_args=norm_args, act_args=act_args
                                                    ))
                self.convs = nn.Sequential(*convs)

            self.pool = lambda x: torch.mean(x, dim=-1, keepdim=False)

        def forward(self, px1, px2=None):
            # pxb1 is with the same size of upsampled points
            if px2 is None:
                _, x = px1  # (B, N, 3), (B, C, N)
                x_global = self.pool(x)
                x = torch.cat(
                    (x, self.linear2(x_global).unsqueeze(-1).expand(-1, -1, x.shape[-1])), dim=1)
                x = self.linear1(x)
            else:
                p1, x1 = px1
                p2, x2 = px2
                x = self.convs(
                    torch.cat((x1, three_interpolation(p1, p2, x2)), dim=1))
            return x


    class InvResMLP(nn.Module):
        def __init__(self,
                    in_channels,
                    norm_args=None,
                    act_args=None,
                    aggr_args={'feature_type': 'dp_fj', "reduction": 'max'},
                    group_args={'NAME': 'ballquery'},
                    conv_args=None,
                    expansion=1,
                    use_res=True,
                    num_posconvs=2,
                    less_act=False,
                    **kwargs
                    ):
            super().__init__()
            self.use_res = use_res
            mid_channels = in_channels * expansion
            self.convs = LocalAggregation([in_channels, in_channels],
                                        norm_args=norm_args, act_args=act_args if num_posconvs > 0 else None,
                                        group_args=group_args, conv_args=conv_args,
                                        **aggr_args, **kwargs)
            if num_posconvs < 1:
                channels = []
            elif num_posconvs == 1:
                channels = [in_channels, in_channels]
            else:
                channels = [in_channels, mid_channels, in_channels]
            pwconv = []
            # point wise after depth wise conv (without last layer)
            for i in range(len(channels) - 1):
                pwconv.append(create_convblock1d(channels[i], channels[i + 1],
                                                norm_args=norm_args,
                                                act_args=act_args if (i != len(channels) - 2) and not less_act else None,
                                                **conv_args))
            self.pwconv = nn.Sequential(*pwconv)
            self.act = create_act(act_args)

        def forward(self, px):
            p, x = px
            identity = x
            x = self.convs([p, x])
            x = self.pwconv(x)
            if x.shape[-1] == identity.shape[-1] and self.use_res:
                x += identity
            x = self.act(x)
            return [p, x]


    class ResBlock(nn.Module):
        def __init__(self,
                    in_channels,
                    norm_args=None,
                    act_args=None,
                    aggr_args={'feature_type': 'dp_fj', "reduction": 'max'},
                    group_args={'NAME': 'ballquery'},
                    conv_args=None,
                    expansion=1,
                    use_res=True,
                    **kwargs
                    ):
            super().__init__()
            self.use_res = use_res
            mid_channels = in_channels * expansion
            self.convs = LocalAggregation([in_channels, in_channels, mid_channels, in_channels],
                                        norm_args=norm_args, act_args=None,
                                        group_args=group_args, conv_args=conv_args,
                                        **aggr_args, **kwargs)
            self.act = create_act(act_args)

        def forward(self, px):
            p, x = px
            identity = x
            x = self.convs([p, x])
            if x.shape[-1] == identity.shape[-1] and self.use_res:
                x += identity
            x = self.act(x)
            return [p, x]





    class PointNextEncoder(nn.Module):
        r"""The Encoder for PointNext 
        `"PointNeXt: Revisiting PointNet++ with Improved Training and Scaling Strategies".
        <https://arxiv.org/abs/2206.04670>`_.
        .. note::
            For an example of using :obj:`PointNextEncoder`, see
            `examples/segmentation/main.py <https://github.com/guochengqian/PointNeXt/blob/master/cfgs/s3dis/README.md>`_.
        Args:
            in_channels (int, optional): input channels . Defaults to 4.
            width (int, optional): width of network, the output mlp of the stem MLP. Defaults to 32.
            blocks (List[int], optional): # of blocks per stage (including the SA block). Defaults to [1, 4, 7, 4, 4].
            strides (List[int], optional): the downsampling ratio of each stage. Defaults to [4, 4, 4, 4].
            block (strorType[InvResMLP], optional): the block to use for depth scaling. Defaults to 'InvResMLP'.
            nsample (intorList[int], optional): the number of neighbors to query for each block. Defaults to 32.
            radius (floatorList[float], optional): the initial radius. Defaults to 0.1.
            aggr_args (_type_, optional): the args for local aggregataion. Defaults to {'feature_type': 'dp_fj', "reduction": 'max'}.
            group_args (_type_, optional): the args for grouping. Defaults to {'NAME': 'ballquery'}.
            norm_args (_type_, optional): the args for normalization layer. Defaults to {'norm': 'bn'}.
            act_args (_type_, optional): the args for activation layer. Defaults to {'act': 'relu'}.
            expansion (int, optional): the expansion ratio of the InvResMLP block. Defaults to 4.
            sa_layers (int, optional): the number of MLP layers to use in the SA block. Defaults to 1.
            sa_use_res (bool, optional): wheter to use residual connection in SA block. Set to True only for PointNeXt-S. 
        """

        def __init__(self,
                    in_channels: int = 4,
                    width: int = 32,
                    blocks: List[int] = [1, 4, 7, 4, 4],
                    strides: List[int] = [4, 4, 4, 4],
                    block: str or Type[InvResMLP] = 'InvResMLP',
                    nsample: int or List[int] = 32,
                    radius: float or List[float] = 0.1,
                    aggr_args: dict = {'feature_type': 'dp_fj', "reduction": 'max'},
                    group_args: dict = {'NAME': 'ballquery'},
                    norm_args: dict = {'norm': 'bn'},
                    act_args: dict = {'act': 'relu'},
                    expansion: int = 4,
                    sa_layers: int = 1,
                    sa_use_res: bool = False,
                    **kwargs
                    ):
            super().__init__()
            if isinstance(block, str):
                block = eval(block)
            self.blocks = blocks
            self.strides = strides
            self.in_channels = in_channels
            self.aggr_args = aggr_args
            self.norm_args = norm_args
            self.act_args = act_args
            self.conv_args = kwargs.get('conv_args', None)
            self.sample_method = kwargs.get('sample_method', 'fps')
            self.expansion = expansion
            self.sa_layers = sa_layers
            self.sa_use_res = sa_use_res
            self.use_res = kwargs.get('use_res', True)
            radius_scaling = kwargs.get('radius_scaling', 2)
            nsample_scaling = kwargs.get('nsample_scaling', 1)

            self.radii = self._to_full_list(radius, radius_scaling)
            self.nsample = self._to_full_list(nsample, nsample_scaling)
            logging.info(f'radius: {self.radii},\n nsample: {self.nsample}')

            # double width after downsampling.
            channels = []
            for stride in strides:
                if stride != 1:
                    width *= 2
                channels.append(width)
            encoder = []
            for i in range(len(blocks)):
                group_args.radius = self.radii[i]
                group_args.nsample = self.nsample[i]
                encoder.append(self._make_enc(block,
                                            channels[i],
                                            blocks[i],
                                            stride=strides[i],
                                            group_args=group_args,
                                            is_head=i == 0 and strides[i] == 1 ))
            self.encoder = nn.Sequential(*encoder)
            self.out_channels = channels[-1]
            self.channel_list = channels

        def _to_full_list(self, param, param_scaling=1):
            # param can be: radius, nsample
            param_list = []
            if isinstance(param, List):
                # make param a full list
                for i, value in enumerate(param):
                    value = [value] if not isinstance(value, List) else value
                    if len(value) != self.blocks[i]:
                        value += [value[-1]] * (self.blocks[i] - len(value))
                    param_list.append(value)
            else:  # radius is a scalar (in this case, only initial raidus is provide), then create a list (radius for each block)
                for i, stride in enumerate(self.strides):
                    if stride == 1:
                        param_list.append([param] * self.blocks[i])
                    else:
                        param_list.append(
                            [param] + [param * param_scaling] * (self.blocks[i] - 1))
                        param *= param_scaling
            return param_list

        def _make_enc(self, block, channels, blocks, stride, group_args, is_head=False):
            layers = []
            radii = group_args.radius
            nsample = group_args.nsample
            group_args.radius = radii[0]
            group_args.nsample = nsample[0]
            layers.append(SetAbstraction(self.in_channels,
                                        channels,
                                        self.sa_layers if not is_head else 1,
                                        stride,
                                        group_args=group_args,
                                        sample_method=self.sample_method,
                                        norm_args=self.norm_args, act_args=self.act_args, conv_args=self.conv_args,
                                        is_head=is_head, use_res=self.sa_use_res
                                        ))
            self.in_channels = channels
            for i in range(1, blocks):
                group_args.radius = radii[i]
                group_args.nsample = nsample[i]
                layers.append(block(self.in_channels,
                                    aggr_args=self.aggr_args,
                                    norm_args=self.norm_args, act_args=self.act_args, group_args=group_args,
                                    conv_args=self.conv_args, expansion=self.expansion,
                                    use_res=self.use_res
                                    ))
            return nn.Sequential(*layers)

        def forward_cls_feat(self, p0, f0=None):
            if hasattr(p0, 'keys'):
                p0, f0 = p0['pos'], p0['x']
            if f0 is None:
                f0 = p0.clone().transpose(1, 2).contiguous()
            for i in range(0, len(self.encoder)):
                p0, f0 = self.encoder[i]([p0, f0])
            return f0.squeeze(-1)

        def forward_all_features(self, p0, x0=None):
            if hasattr(p0, 'keys'):
                p0, x0 = p0['pos'], p0['x']
            if x0 is None:
                x0 = p0.clone().transpose(1, 2).contiguous()
            p, x = [p0], [x0]
            for i in range(0, len(self.encoder)):
                _p, _x = self.encoder[i]([p[-1], x[-1]])
                p.append(_p)
                x.append(_x)
            return p, x

        def forward(self, p0, x0=None):
            self.forward_all_features(p0, x0)


    class PointNextDecoder(nn.Module):
        def __init__(self,
                    encoder_channel_list: List[int], 
                    decoder_layers: int = 2, 
                    **kwargs
                    ):
            super().__init__()
            self.decoder_layers = decoder_layers
            self.in_channels = encoder_channel_list[-1]
            skip_channels = encoder_channel_list[:-1]
            # the output channel after interpolation
            fp_channels = encoder_channel_list[:-1]
            
            n_decoder_stages = len(fp_channels) 
            decoder = [[] for _ in range(n_decoder_stages)]
            for i in range(-1, -n_decoder_stages-1, -1):
                decoder[i] = self._make_dec(
                    skip_channels[i], fp_channels[i])
            self.decoder = nn.Sequential(*decoder)
            self.out_channels = fp_channels[-n_decoder_stages]

        def _make_dec(self, skip_channels, fp_channels):
            layers = []
            mlp = [skip_channels + self.in_channels] + \
                    [fp_channels] * self.decoder_layers
            layers.append(FeaturePropogation(mlp))
            self.in_channels = fp_channels
            return nn.Sequential(*layers)

        def forward(self, p, f):
            for i in range(-1, -len(self.decoder) - 1, -1):
                f[i - 1] = self.decoder[i][1:](
                    [p[i], self.decoder[i][0]([p[i - 1], f[i - 1]], [p[i], f[i]])])[1]
            return f[-len(self.decoder) - 1]


    class PointNextPartDecoder(nn.Module):
        """PointNextSeg for point cloud segmentation with inputs of variable sizes
        """

        def __init__(self,
                    block,
                    decoder_blocks=[1, 1, 1, 1],  # depth
                    decoder_layers=2,
                    in_channels=6,
                    width=32,
                    strides=[1, 4, 4, 4, 4],
                    nsample=[8, 16, 16, 16, 16],
                    radius=0.1,
                    radius_scaling=2,
                    nsample_scaling=1,
                    aggr_args={'feature_type': 'dp_fj', "reduction": 'max'},
                    group_args={'NAME': 'ballquery'},
                    norm_args={'norm': 'bn'},
                    act_args={'act': 'relu'},
                    conv_args=None,
                    mid_res=False,
                    expansion=1,
                    cls_map='PointNet2',
                    **kwargs
                    ):
            super().__init__()
            if kwargs:
                logging.warning(
                    f"kwargs: {kwargs} are not used in {__class__.__name__}")
            if isinstance(block, str):
                block = eval(block)
            self.blocks = decoder_blocks
            self.cls_map = cls_map.lower()
            self.decoder_layers = decoder_layers
            self.strides = strides[:-1]
            self.mid_res = mid_res
            self.aggr_args = aggr_args
            self.norm_args = norm_args
            self.act_args = act_args
            self.conv_args = conv_args
            self.in_channels = in_channels
            self.expansion = expansion

            # self.radii = self._to_full_list(radius, radius_scaling)
            # self.nsample = self._to_full_list(nsample, nsample_scaling)
            # logging.info(f'radius: {self.radii},\n nsample: {self.nsample}')

            # width *2 after downsampling.

            channels = []
            initial_width = width
            for stride in strides:
                if stride != 1:
                    width *= 2
                channels.append(width)

            self.in_channels = channels[-1]
            skip_channels = [in_channels] + channels[:-1]
            fp_channels = [initial_width] + channels[:-1]
            decoder = [[] for _ in range(len(decoder_blocks))]
            # conv embeding of shapes class

            if self.cls_map == 'curvenet':
                # global features
                self.global_conv2 = nn.Sequential(
                    create_convblock1d(fp_channels[-1] * 2, 128,
                                    norm_args=None,
                                    act_args=act_args,
                                    **conv_args))
                self.global_conv1 = nn.Sequential(
                    create_convblock1d(fp_channels[-2] * 2, 64,
                                    norm_args=None,
                                    act_args=act_args,
                                    **conv_args))

                # self.convc = nn.Sequential()
                skip_channels[1] += 64 + 128 + 16  # shape categories labels
            else:
                self.convc = nn.Sequential(create_convblock1d(16, 64,
                                                            norm_args=None,
                                                            act_args=act_args,
                                                            **conv_args))
                skip_channels[1] += 64  # shape categories labels

            for i in range(-1, -len(decoder_blocks) - 1, -1):
                # group_args.radius = self.radii[i]
                # group_args.nsample = self.nsample[i]
                decoder[i] = self._make_dec(
                    skip_channels[i], fp_channels[i], block, decoder_blocks[i])
            self.decoder = nn.Sequential(*decoder)
            self.out_channels = fp_channels[0]

        def _to_full_list(self, param, param_scaling=1):
            # param can be: radius, nsample
            param_list = []
            if isinstance(param, List):
                # make param a full list
                for i, value in enumerate(param):
                    value = [value] if not isinstance(value, List) else value
                    if len(value) != self.blocks[i]:
                        value += [value[-1]] * (self.blocks[i] - len(value))
                    param_list.append(value)
            else:  # radius is a scalar, then create a list
                for i, stride in enumerate(self.strides):
                    if stride == 1:
                        param_list.append([param] * self.blocks[i])
                    else:
                        param_list.append(
                            [param] + [param * param_scaling] * (self.blocks[i] - 1))
                        param *= param_scaling
            return param_list

        def _make_dec(self, skip_channels, fp_channels, block, blocks, group_args=None, is_head=False):
            """_summary_
            Args:
                skip_channels (int): channels for the incomming upsampled features
                fp_channels (_type_): channels for the output upsampled features
                block (_type_): _description_
                blocks (_type_): _description_
                group_args (_type_, optional): _description_. Defaults to None.
                is_head (bool, optional): _description_. Defaults to False.
            Returns:
                _type_: _description_
            """
            layers = []
            if is_head:
                mlp = [skip_channels] + [fp_channels] * self.decoder_layers
            else:
                mlp = [skip_channels + self.in_channels] + \
                    [fp_channels] * self.decoder_layers
            layers.append(FeaturePropogation(mlp, not is_head))
            self.in_channels = fp_channels

            # radii = group_args.radius
            # nsample = group_args.nsample
            # for i in range(1, blocks):
            #     group_args.radius = radii[i]
            #     group_args.nsample = nsample[i]
            #     layers.append(block(self.in_channels, self.in_channels,
            #                         aggr_args=self.aggr_args,
            #                         norm_args=self.norm_args, act_args=self.act_args, group_args=group_args,
            #                         conv_args=self.conv_args, mid_res=self.mid_res))
            return nn.Sequential(*layers)

        def forward(self, p, f, cls_label):
            B, N = p[0].shape[0:2]

            if self.cls_map == 'curvenet':
                emb1 = self.global_conv1(f[-2])
                emb1 = emb1.max(dim=-1, keepdim=True)[0]  # bs, 64, 1
                emb2 = self.global_conv2(f[-1])
                emb2 = emb2.max(dim=-1, keepdim=True)[0]  # bs, 128, 1
                cls_one_hot = torch.zeros((B, 16), device=p[0].device)
                cls_one_hot = cls_one_hot.scatter_(1, cls_label, 1).unsqueeze(-1)
                cls_one_hot = torch.cat((emb1, emb2, cls_one_hot), dim=1)
                cls_one_hot = cls_one_hot.expand(-1, -1, N)
                # x = torch.cat((l1_xyz, l1_points, l), dim=1)
            else:
                cls_one_hot = torch.zeros((B, 16), device=p[0].device)
                cls_one_hot = cls_one_hot.scatter_(
                    1, cls_label, 1).unsqueeze(-1).repeat(1, 1, N)
                cls_one_hot = self.convc(cls_one_hot)

            for i in range(-1, -len(self.decoder), -1):
                f[i - 1] = self.decoder[i][1:](
                    [p[i], self.decoder[i][0]([p[i - 1], f[i - 1]], [p[i], f[i]])])[1]

            f[-len(self.decoder) - 1] = self.decoder[0][1:](
                [p[2], self.decoder[0][0]([p[1], torch.cat([cls_one_hot, f[1]], 1)], [p[2], f[2]])])[1]

            return f[-len(self.decoder) - 1]





    def sample_and_group(npoint, nsample, xyz, points, density_scale = None):
        """
        Input:
            npoint:
            nsample:
            xyz: input points position data, [B, N, C]
            points: input points data, [B, N, D]
        Return:
            new_xyz: sampled points position data, [B, 1, C]
            new_points: sampled points data, [B, 1, N, C+D]
            grouped_xyz_norm: 
        """
        B, N, C = xyz.shape
        S = npoint
        fps_idx = farthest_point_sample(xyz, npoint) # [B, npoint, C]
        new_xyz = index_points(xyz, fps_idx)
        idx = knn_point(nsample, xyz, new_xyz)
        grouped_xyz = index_points(xyz, idx) # [B, npoint, nsample, C]
        grouped_xyz_norm = grouped_xyz - new_xyz.view(B, S, 1, C)
        if points is not None:
            grouped_points = index_points(points, idx)
            new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1) # [B, npoint, nsample, C+D]
        else:
            new_points = grouped_xyz_norm

        if density_scale is None:
            return new_xyz, new_points, grouped_xyz_norm, idx
        else:
            grouped_density = index_points(density_scale, idx)
            return new_xyz, new_points, grouped_xyz_norm, idx, grouped_density


class WeightNet(nn.Module):

    def __init__(self, in_channel, out_channel, hidden_unit = [8, 8]):
        super(WeightNet, self).__init__()

        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        if hidden_unit is None or len(hidden_unit) == 0:
            self.mlp_convs.append(nn.Conv2d(in_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
        else:
            self.mlp_convs.append(nn.Conv2d(in_channel, hidden_unit[0], 1))
            self.mlp_bns.append(nn.BatchNorm2d(hidden_unit[0]))
            for i in range(1, len(hidden_unit)):
                self.mlp_convs.append(nn.Conv2d(hidden_unit[i - 1], hidden_unit[i], 1))
                self.mlp_bns.append(nn.BatchNorm2d(hidden_unit[i]))
            self.mlp_convs.append(nn.Conv2d(hidden_unit[-1], out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
        
    def forward(self, localized_xyz):
        #xyz : BxCxKxN

        weights = localized_xyz
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            weights =  F.relu(bn(conv(weights)))

        return weights


    class PointConvSetAbstraction(nn.Module):
        def __init__(self, npoint, nsample, in_channel, mlp, group_all):
            super(PointConvSetAbstraction, self).__init__()
            self.npoint = npoint
            self.nsample = nsample
            self.mlp_convs = nn.ModuleList()
            self.mlp_bns = nn.ModuleList()
            last_channel = in_channel
            for out_channel in mlp:
                self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
                self.mlp_bns.append(nn.BatchNorm2d(out_channel))
                last_channel = out_channel

            self.weightnet = WeightNet(3, 16)
            self.linear = nn.Linear(16 * mlp[-1], mlp[-1])
            self.bn_linear = nn.BatchNorm1d(mlp[-1])
            self.group_all = group_all

        def forward(self, xyz, points):
            """
            Input:
                xyz: input points position data, [B, C, N]
                points: input points data, [B, D, N]
            Return:
                new_xyz: sampled points position data, [B, C, S]
                new_points_concat: sample points feature data, [B, D', S]
            """
            B = xyz.shape[0]
            xyz = xyz.permute(0, 2, 1)
            if points is not None:
                points = points.permute(0, 2, 1)

            if self.group_all:
                new_xyz, new_points, grouped_xyz_norm = sample_and_group_all(xyz, points)
            else:
                new_xyz, new_points, grouped_xyz_norm, _ = sample_and_group(self.npoint, self.nsample, xyz, points)
            # new_xyz: sampled points position data, [B, npoint, C]
            # new_points: sampled points data, [B, npoint, nsample, C+D]
            new_points = new_points.permute(0, 3, 2, 1) # [B, C+D, nsample,npoint]
            for i, conv in enumerate(self.mlp_convs):
                bn = self.mlp_bns[i]
                new_points =  F.relu(bn(conv(new_points)))
            # new_points are the features -> [B, C, H, N,]


            # Point differences [B, N, H, 3] -> [B, 3, H, N] 
            grouped_xyz = grouped_xyz_norm.permute(0, 3, 2, 1)
            # function w [B, 3, H, N] -> [B, Cmid, H, N] (Cmid=16)
            weights = self.weightnet(grouped_xyz)


            # Matmul [B, N, C, H] x [B, N, H, Cmid] ->  [B, N, C, Cmid] -> [B, N, C*Cmid]
            new_points = torch.matmul(input=new_points.permute(0, 3, 1, 2), other = weights.permute(0, 3, 2, 1)).view(B, self.npoint, -1)

            # Final linear (optimized conv) [B, N, C*Cmid] -> [B, N, Cout]
            new_points = self.linear(new_points)

            # Activation
            new_points = self.bn_linear(new_points.permute(0, 2, 1))
            new_points = F.relu(new_points)
            new_xyz = new_xyz.permute(0, 2, 1)
            # -> [B, N, Cout]

            return new_xyz, new_points









