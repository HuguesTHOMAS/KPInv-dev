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
import numpy as np
from torch import Tensor
from torch.nn.init import kaiming_uniform_


from kernels.kernel_points import load_kernels
from models.generic_blocks import gather, index_select, radius_gaussian, local_maxpool, UnaryBlock, \
    NormBlock, DropPathPack, build_mlp, mlp_from_list

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


class KPMiniMod(nn.Module):

    def __init__(self,
                 channels: int,
                 shell_sizes: list,
                 radius: float,
                 sigma: float,
                 attention_groups: int = 8,
                 attention_act: str = 'sigmoid',
                 mod_grp_norm: bool = False,
                 shared_kp_data = None,
                 dimension: int = 3,
                 influence_mode: str = 'linear',
                 fixed_kernel_points: str = 'center',
                 norm_type: str = 'batch',
                 bn_momentum: float = 0.1,
                 activation: nn.Module = nn.LeakyReLU(0.1),
                 inf: float = 1e6):
        """
        V2 of Mini KPConv with incoporated attention modulations. 
        Args:
            channels                  (int): The number of input channels.
            shell_sizes                 (list): The number of kernel points per shell.
            radius                     (float): The radius used for kernel point init.
            sigma                      (float): The influence radius of each kernel point.
            attention_groups           (int=8): number of groups in attention (negative value for ch_per_grp).
            attention_act                (str): Activate the weight with 'none', 'sigmoid', 'softmax' or 'tanh'.
            mod_grp_norm          (bool=False): Use group norm for modulations or not.
            shared_kp_data              (None): Optional data dict shared across the layer
            dimension                  (int=3): The dimension of the point space.
            influence_mode      (str='linear'): Influence function ('constant', 'linear', 'gaussian').
            fixed_kernel_points (str='center'): kernel points whose position is fixed ('none', 'center' or 'verticals').
            norm_type            (str='batch'): type of normalization used in layer ('group', 'batch', 'none')
            bn_momentum           (float=0.10): Momentum for batch normalization
            activation              (nn.Module: Activation function. Use None for no activation.
            inf (float=1e6): The value of infinity to generate the padding point.
        """
        super(KPMiniMod, self).__init__()

        # Verification of group parameter
        if attention_groups > 0:
            assert channels % attention_groups == 0, "channels must be divisible by ch_per_grp."
            ch_per_grp = channels // attention_groups
        else:
            ch_per_grp = -attention_groups
            assert channels % ch_per_grp == 0, "channels must be divisible by ch_per_grp."
            attention_groups = channels // ch_per_grp

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
        self.ch_per_grp = ch_per_grp
        self.groups = attention_groups
        self.attention_act = attention_act
        self.mod_grp_norm = mod_grp_norm


        # Depthwise conv parameters
        # *************************

        # Initialize weights
        self.weights = nn.Parameter(torch.zeros(size=(self.K, channels)), requires_grad=True)
        kaiming_uniform_(self.weights, a=math.sqrt(5))

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


        # Attention parameters
        # ********************

        # Attention mlp
        Cout = self.K * self.ch_per_grp
        # alpha_list = [Cout]
        alpha_list = [channels, 'NA', Cout]
        self.alpha_mlp = mlp_from_list(channels,
                                       alpha_list,
                                       final_bias=False,
                                       norm_type='none',
                                       bn_momentum=-1,
                                       activation=activation)
                                       
        # Optional final group norm for each kernel weights
        self.grpnorm = nn.GroupNorm(self.K, self.K * self.ch_per_grp)
        # self.grpnorm = nn.BatchNorm1d(self.K * self.ch_per_grp, momentum=bn_momentum)
        

        # Weight activation
        if attention_act == 'sigmoid':
            self.attention_act = torch.sigmoid
        elif attention_act == 'tanh':
            self.attention_act = torch.tanh
        elif attention_act == 'softmax':
            self.attention_act = nn.Softmax(dim=1)
        else:
            self.attention_act = nn.Identity()

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
        KPTransformer forward.
        Args:
            q_points (Tensor): query points (M, 3).
            s_points (Tensor): support points carrying input features (N, 3).
            s_feats (Tensor): input features values (N, C_in).
            neighb_inds (LongTensor): neighbor indices of query points among support points (M, H).
        Returns:
            q_feats (Tensor): output features carried by query points (M, C_out).
        """

        # Get Neighbor features
        # *********************
        
        # Add a zero feature for shadow neighbors
        padded_s_feats = torch.cat((s_feats, torch.zeros_like(s_feats[:1, :])), 0)  # (N, C) -> (N+1, C)

        # Get the features of each neighborhood
        neighbor_feats = index_select(padded_s_feats, neighb_inds, dim=0)  # -> (M, H, C)
   

        # Get modulations
        # ***************

        # In case M == N, we can assume this is an in-place convolution.
        if q_pts.shape[0] == s_pts.shape[0]:
            pooled_feats = s_feats  # (M, C)
        else:
            pooled_feats = neighbor_feats[:, 0, :]  # nearest pool (M, H, C) -> (M, C)
            # pooled_feats = torch.max(neighbor_feats, dim=1)  # max pool (M, H, C) -> (M, C)
            # pooled_feats = torch.mean(neighbor_feats, dim=1)  # avg pool (M, H, C) -> (M, C)

        # MLP to get weights
        modulations = self.alpha_mlp(pooled_feats)  # (M, C) -> (M, C//r) -> (M, K*CpG)

        # Optional normalization per kernel
        if self.mod_grp_norm:
            modulations = modulations.transpose(0, 1).unsqueeze(0)  # (M, K*CpG) -> (B=1, K*CpG, M)
            modulations = self.grpnorm(modulations)
            modulations = modulations.squeeze(0).transpose(0, 1)  # (B=1, K*CpG, M) -> (M, K*CpG)

        # Activation
        modulations = self.attention_act(modulations)

        # Apply modulations
        # *****************

        # Reshapes
        modulations = modulations.view(-1, self.K, self.ch_per_grp, 1)  # -> (M, K, CpG, 1)
        conv_weights = self.weights.view(1, self.K, self.ch_per_grp, self.groups)  # -> (1, K, CpG, G)

        # Modulate convolution weights at each location (M, K, CpG, G)
        conv_weights = conv_weights * modulations

        # Reshape
        conv_weights = conv_weights.reshape(-1, self.K, self.channels)  # -> (M, K, C)


        # Depthwise convolution
        # *********************

        # Get nearest kernel point (M, H) and weights applied to each neighbors (M, H)
        influence_weights, neighbors, neighbors_1nn = self.get_neighbors_influences(q_pts, s_pts, neighb_inds)

        # Collect nearest kernel point weights (M, K, C) -> (M, H, C)
        neighbors_weights = torch.gather(conv_weights, 1, neighbors_1nn.unsqueeze(2).expand(-1, -1, self.channels))

        # Adjust weights with influence
        if self.influence_mode != 'constant':
            neighbors_weights *= influence_weights.unsqueeze(2)

        # Apply convolution weights
        neighbor_feats = self.merge_op(neighbor_feats, neighbors_weights)  # (M, H, C)


        # Output
        # ******

        # Final summation
        output_feats = self.aggr_op(neighbor_feats, dim=1)  # (M, H, C) -> (M, C)

        return output_feats

    def __repr__(self):

        repr_str = 'KPMiniMod'
        repr_str += '(K: {:d}'.format(self.K)
        repr_str += ', C: {:d}'.format(self.channels)
        repr_str += ', r: {:.2f}'.format(self.radius)
        repr_str += ', sigma: {:.2f})'.format(self.sigma)

        return repr_str


class KPTransformer(nn.Module):

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 shell_sizes: list,
                 radius: float,
                 sigma: float,
                 attention_groups: int = 8,
                 attention_act: str = 'sigmoid',
                 shared_kp_data = None,
                 dimension: int = 3,
                 influence_mode: str = 'linear',
                 fixed_kernel_points: str = 'center',
                 norm_type: str = 'batch',
                 bn_momentum: float = 0.1,
                 activation: nn.Module = nn.LeakyReLU(0.1),
                 inf: float = 1e6):
        """
        Design similar to Mini KPConv but with incoporated transformer attention. 
        Args:
            in_channels                  (int): The number of input channels.
            out_channels                 (int): The number of output channels.
            shell_sizes                 (list): The number of kernel points per shell.
            radius                     (float): The radius used for kernel point init.
            sigma                      (float): The influence radius of each kernel point.
            attention_groups           (int=8): number of groups in attention (negative value for ch_per_grp).
            attention_act                (str): Activate the weight with 'none', 'sigmoid', 'softmax' or 'tanh'.
            shared_kp_data              (None): Optional data dict shared across the layer
            dimension                  (int=3): The dimension of the point space.
            influence_mode      (str='linear'): Influence function ('constant', 'linear', 'gaussian').
            fixed_kernel_points (str='center'): kernel points whose position is fixed ('none', 'center' or 'verticals').
            norm_type            (str='batch'): type of normalization used in layer ('group', 'batch', 'none')
            bn_momentum           (float=0.10): Momentum for batch normalization
            activation              (nn.Module: Activation function. Use None for no activation.
            inf (float=1e6): The value of infinity to generate the padding point.
        """
        super(KPTransformer, self).__init__()

        # Verification of group parameter
        if attention_groups > 0:
            assert out_channels % attention_groups == 0, "channels must be divisible by ch_per_grp."
            ch_per_grp = out_channels // attention_groups
        else:
            ch_per_grp = -attention_groups
            assert out_channels % ch_per_grp == 0, "channels must be divisible by ch_per_grp."
            attention_groups = out_channels // ch_per_grp

        # Save parameters
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.shell_sizes = shell_sizes
        self.K = int(np.sum(shell_sizes))
        self.radius = radius
        self.sigma = sigma
        self.dimension = dimension
        self.influence_mode = influence_mode
        self.fixed_kernel_points = fixed_kernel_points
        self.inf = inf
        self.ch_per_grp = ch_per_grp
        self.groups = attention_groups
        self.attention_act = attention_act


        # Depthwise conv parameters
        # *************************

        # Initialize weights
        self.weights = nn.Parameter(torch.zeros(size=(self.K, out_channels)), requires_grad=True)
        kaiming_uniform_(self.weights, a=math.sqrt(5))

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
        # self.merge_op = torch.add
        self.merge_op = torch.mul
        # self.aggr_op = lambda x, dim=0: torch.max(x, dim=dim)[0]
        self.aggr_op = torch.sum


        # Attention parameters
        # ********************

        # First linear transform for features
        self.use_v_linear = in_channels != out_channels  # or always True
        if self.use_v_linear:
            self.linear_v = nn.Linear(in_channels, out_channels)
        else:
            self.linear_v = nn.Identity()

        # First linear transform for queries and keys
        self.use_qk_linear = in_channels != out_channels or True  # or always True
        if self.use_qk_linear:
            self.linear_q = nn.Linear(in_channels, out_channels)
            self.linear_k = nn.Linear(in_channels, out_channels)

        # Stride mode (this op should not be used in stride mode)
        self.stride_mode = 'nearest'

        # Attention mlp
        CpG = ch_per_grp
        Cin = out_channels
        if self.use_qk_linear:
            alpha_list = ['NA', CpG, 'NA', CpG]
        else:
            alpha_list = [CpG, 'NA', CpG, 'NA', CpG]
        self.alpha_mlp = mlp_from_list(Cin,
                                       alpha_list,
                                       final_bias=True,
                                       norm_type=norm_type,
                                       bn_momentum=bn_momentum,
                                       activation=activation)

        # Weight activation
        if attention_act == 'sigmoid':
            self.attention_act = torch.sigmoid
        elif attention_act == 'tanh':
            self.attention_act = torch.tanh
        elif attention_act == 'softmax':
            self.attention_act = nn.Softmax(dim=1)
        else:
            self.attention_act = nn.Identity()

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
        KPTransformer forward.
        Args:
            q_points (Tensor): query points (M, 3).
            s_points (Tensor): support points carrying input features (N, 3).
            s_feats (Tensor): input features values (N, C_in).
            neighb_inds (LongTensor): neighbor indices of query points among support points (M, H).
        Returns:
            q_feats (Tensor): output features carried by query points (M, C_out).
        """

        # Note on shadow neighbors
        # ************************
        #
        #   For transformers, we need to handle shadow neighbors with a mask
        #   Here, because we multiply by the kernel influence, it is already handled.
        #

        with torch.no_grad():
            valid_mask = neighb_inds < int(s_feats.shape[0])
            shadow_bool = not torch.all(valid_mask).item()


        # Get Neighbor features
        # *********************

        # Optional first linear layer
        v_feats = self.linear_v(s_feats)
        
        # Add a zero feature for shadow neighbors
        padded_v_feats = torch.cat((v_feats, torch.zeros_like(v_feats[:1, :])), 0)  # (N, C) -> (N+1, C)

        # Get the features of each neighborhood
        neighb_v_feats = index_select(padded_v_feats, neighb_inds, dim=0)  # -> (M, H, C)


        use_conv = True
        if use_conv:

            # Depthwise convolution
            # *********************

            # Get nearest kernel point (M, H) and weights applied to each neighbors (M, H)
            influence_weights, neighbors, neighbors_1nn = self.get_neighbors_influences(q_pts, s_pts, neighb_inds)

            # Collect nearest kernel point weights -> (M, H, C)
            neighbors_weights = gather(self.weights, neighbors_1nn)

            # Adjust weights with influence
            if self.influence_mode != 'constant':
                neighbors_weights *= influence_weights.unsqueeze(2)

            # Apply convolution weights
            neighb_v_feats = self.merge_op(neighb_v_feats, neighbors_weights)  # (M, H, C)

            
        use_tran = True
        if use_tran:


            # Get transformers keys and values
            # ********************************

            if self.use_qk_linear:

                # Get keys features from neighbors
                k_feats = self.linear_k(s_feats)
                padded_k_feats = torch.cat((k_feats, torch.zeros_like(k_feats[:1, :])), 0)  # (N, C) -> (N+1, C)
                neighb_k_feats = index_select(padded_k_feats, neighb_inds, dim=0)  # -> (M, H, C)
                
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

            else:
                neighb_k_feats = neighb_v_feats
                if q_pts.shape[0] == s_pts.shape[0]:
                    q_feats = v_feats
                else:
                    if self.stride_mode == 'nearest':
                        q_feats = neighb_v_feats[:, 0]  # nearest pool -> (M, C)
                    elif self.stride_mode == 'max':
                        q_feats = torch.max(neighb_v_feats, dim=1)  # max pool (M, H, C) -> (M, C)
                    elif self.stride_mode == 'avg':
                        q_feats = torch.mean(neighb_v_feats, dim=1)  # avg pool (M, H, C) -> (M, C)


            # Get attention weights
            # *********************

            # Merge queries with keys
            qk_feats = q_feats.unsqueeze(1) - neighb_k_feats  # (M, 1, C) -> (M, H, C)

            # Generate attention weights
            attention_weights = self.alpha_mlp(qk_feats) # (M, H, C) -> (M, H, CpG)
            attention_weights = self.attention_act(attention_weights)

            # Apply attention weights
            # ************************

            # Separate features in groups
            H = int(neighb_inds.shape[1])
            neighb_v_feats = neighb_v_feats.view(-1, H, self.ch_per_grp, self.groups)  # (M, H, C) -> (M, H, CpG, G)
            attention_weights = attention_weights.view(-1, H, self.ch_per_grp, 1)  # (M, H*CpG) -> (M, H, CpG, 1)

            # Multiply features with attention
            neighb_v_feats *= attention_weights  # -> (M, H, CpG, G)
                
            # Apply shadow mask (every gradient for shadow neighbors will be zero)
            if shadow_bool:
                # print('shadow neighbors present', s_feats.shape)
                neighb_v_feats *= valid_mask.type(torch.float32).unsqueeze(2).unsqueeze(3)


        # Output
        # ******

        # Final summation
        output_feats = self.aggr_op(neighb_v_feats, dim=1)  # (M, H, CpG, G) -> (M, CpG, G)
        
        # Reshape
        output_feats = output_feats.view(-1, self.out_channels)  # -> (M, C)

        return output_feats

    def __repr__(self):

        repr_str = 'KPTransformer'
        repr_str += '(K: {:d}'.format(self.K)
        repr_str += ', C: {:d}'.format(self.out_channels)
        repr_str += ', r: {:.2f}'.format(self.radius)
        repr_str += ', sigma: {:.2f})'.format(self.sigma)

        return repr_str


# ----------------------------------------------------------------------------------------------------------------------
#
#           Complex blocks
#       \********************/
#



class KPTransformerBlock(nn.Module):

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 shell_sizes: list,
                 radius: float,
                 sigma: float,
                 attention_groups: int = 8,
                 attention_act: str = 'sigmoid',
                 shared_kp_data = None,
                 influence_mode: str = 'linear',
                 dimension: int = 3,
                 norm_type: str = 'batch',
                 bn_momentum: float = 0.1,
                 activation: nn.Module = nn.LeakyReLU(0.1)):
        """
        KPConv block with normalization and activation.
        Args:
            in_channels             (int): dimension input features
            out_channels            (int): dimension input features
            shell_sizes            (list): The number of kernel points per shell.
            radius                (float): convolution radius
            sigma                 (float): influence radius of each kernel point
            attention_groups      (int=8): number of groups in attention (negative value for ch_per_grp).
            attention_act           (str): Activate the weight with 'none', 'sigmoid', 'softmax' or 'tanh'.
            shared_kp_data      (None): Optional data dict shared across the layer
            influence_mode (str='linear'): Influence function ('constant', 'linear', 'gaussian')
            dimension             (int=3): dimension of input
            norm_type       (str='batch'): type of normalization used in layer ('group', 'batch', 'none')
            bn_momentum      (float=0.10): Momentum for batch normalization
            activation (nn.Module|None=nn.LeakyReLU(0.1)): Activation function. Use None for no activation.
        """
        super(KPTransformerBlock, self).__init__()

        # Define parameters
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.shell_sizes = shell_sizes
        self.radius = radius
        self.sigma = sigma
        self.influence_mode = influence_mode
        self.dimension = dimension
        self.norm_type = norm_type
        self.bn_momentum = bn_momentum

        # Define modules
        self.activation = activation
        self.norm = NormBlock(out_channels, norm_type, bn_momentum)

        self.conv = KPTransformer(in_channels,
                                  out_channels,
                                  shell_sizes,
                                  radius,
                                  sigma,
                                  attention_groups=attention_groups,
                                  attention_act=attention_act,
                                  shared_kp_data=shared_kp_data,
                                  dimension=dimension,
                                  influence_mode=influence_mode,
                                  norm_type=norm_type,
                                  bn_momentum=bn_momentum,
                                  activation=activation)
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


class KPTransformerResidualBlock(nn.Module):

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 shell_sizes: list,
                 radius: float,
                 sigma: float,
                 attention_groups: int = 8,
                 attention_act: str = 'sigmoid',
                 mod_grp_norm: bool = False,
                 minimod: bool = False,
                 shared_kp_data = None,
                 influence_mode: str = 'linear',
                 dimension: int = 3,
                 strided: bool = False,
                 norm_type: str = 'batch',
                 bn_momentum: float = 0.1,
                 activation: nn.Module = nn.LeakyReLU(0.1)):
        """
        KPConv residual bottleneck block.
        Args:
            in_channels             (int): dimension input features
            out_channels            (int): dimension input features
            shell_sizes            (list): The number of kernel points per shell.
            radius                (float): convolution radius
            sigma                 (float): influence radius of each kernel point
            attention_groups      (int=8): number of groups in attention (negative value for ch_per_grp).
            attention_act           (str): Activate the weight with 'none', 'sigmoid', 'softmax' or 'tanh'.
            mod_grp_norm     (bool=False): Use group norm for modulations or not.
            minimod          (bool=False): Use KPMiniMod instead of transformer
            shared_kp_data      (None): Optional data dict shared across the layer
            influence_mode (str='linear'): Influence function ('constant', 'linear', 'gaussian')
            dimension             (int=3): dimension of input
            strided          (bool=False): strided or not
            norm_type       (str='batch'): type of normalization used in layer ('group', 'batch', 'none')
            bn_momentum      (float=0.10): Momentum for batch normalization
            activation (nn.Module|None=nn.LeakyReLU(0.1)): Activation function. Use None for no activation.
        """
        super(KPTransformerResidualBlock, self).__init__()

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


        if minimod:
            self.conv = KPMiniMod(mid_channels,
                                shell_sizes,
                                radius,
                                sigma,
                                attention_groups=attention_groups,
                                attention_act=attention_act,
                                mod_grp_norm=mod_grp_norm,
                                shared_kp_data=shared_kp_data,
                                dimension=dimension,
                                influence_mode=influence_mode,
                                norm_type=norm_type,
                                bn_momentum=bn_momentum,
                                activation=activation)

        else:
            self.conv = KPTransformer(mid_channels,
                                    mid_channels,
                                    shell_sizes,
                                    radius,
                                    sigma,
                                    attention_groups=attention_groups,
                                    attention_act=attention_act,
                                    shared_kp_data=shared_kp_data,
                                    dimension=dimension,
                                    influence_mode=influence_mode,
                                    norm_type=norm_type,
                                    bn_momentum=bn_momentum,
                                    activation=activation)

        # Define modules
        self.activation = activation
        self.norm = NormBlock(mid_channels, norm_type, bn_momentum)

        # Second upscaling mlp
        self.unary2 = UnaryBlock(mid_channels, out_channels, norm_type, bn_momentum, activation=None)

        # Shortcut optional mpl
        if in_channels != out_channels:
            self.unary_shortcut = UnaryBlock(in_channels, out_channels, norm_type, bn_momentum, activation=None)
        else:
            self.unary_shortcut = nn.Identity()

        return

    def forward(self, q_pts, s_pts, s_feats, neighbor_indices):

        # First downscaling mlp
        x = self.unary1(s_feats)

        # Convolution
        x = self.conv(q_pts, s_pts, x, neighbor_indices)
        x = self.activation(self.norm(x))

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


class KPTransformerInvertedBlock(nn.Module):

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 shell_sizes: list,
                 radius: float,
                 sigma: float,
                 attention_groups: int = 8,
                 attention_act: str = 'sigmoid',
                 expansion: int = 4,
                 drop_path: float = -1.,
                 layer_scale_init_v: float = -1.,
                 shared_kp_data = None,
                 influence_mode: str = 'linear',
                 dimension: int = 3,
                 strided: bool = False,
                 norm_type: str = 'batch',
                 bn_momentum: float = 0.1,
                 activation: nn.Module = nn.LeakyReLU(0.1)):
        """
        KPConv inverted block as in ConvNext (and PointNext).
        Args:
            in_channels             (int): dimension input features
            out_channels            (int): dimension input features
            shell_sizes            (list): The number of kernel points per shell.
            radius                (float): convolution radius
            sigma                 (float): influence radius of each kernel point
            attention_groups      (int=8): number of groups in attention (negative value for ch_per_grp).
            attention_act           (str): Activate the weight with 'none', 'sigmoid', 'softmax' or 'tanh'.
            expansion               (int): Factor for linear layer expansion
            drop_path             (float): Proba to drop convolution paths (for stochastic network depth)
            layer_scale_init_v    (float): Value for initialization of layer scales
            shared_kp_data      (None): Optional data dict shared across the layer
            influence_mode (str='linear'): Influence function ('constant', 'linear', 'gaussian')
            dimension             (int=3): dimension of input
            strided          (bool=False): strided or not
            norm_type       (str='batch'): type of normalization used in layer ('group', 'batch', 'none')
            bn_momentum      (float=0.10): Momentum for batch normalization
            activation (nn.Module|None=nn.LeakyReLU(0.1)): Activation function. Use None for no activation.
        """
        super(KPTransformerInvertedBlock, self).__init__()

        # Define parameters
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.expansion = expansion
        self.strided = strided
        self.bn_momentum = bn_momentum
        self.norm_type = norm_type
        mid_channels = out_channels * expansion

        # learnable parameters
        self.linear1 = nn.Linear(in_channels, mid_channels, bias=False)
        self.linear2 = nn.Linear(mid_channels, out_channels, bias=False)

        # KPTransformer
        self.conv = KPTransformer(in_channels,
                                  in_channels,
                                  shell_sizes,
                                  radius,
                                  sigma,
                                  attention_groups=attention_groups,
                                  attention_act=attention_act,
                                  shared_kp_data=shared_kp_data,
                                  dimension=dimension,
                                  influence_mode=influence_mode,
                                  norm_type=norm_type,
                                  bn_momentum=bn_momentum,
                                  activation=activation)

        if in_channels != out_channels:
            self.linear_shortcut =nn.Linear(in_channels, out_channels, bias=False)
        else:
            self.linear_shortcut = nn.Identity()

        # Other parameters
        self.norm0 = NormBlock(in_channels, norm_type, bn_momentum)
        self.norm1 = NormBlock(in_channels, norm_type, bn_momentum)
        self.norm2 = NormBlock(in_channels, norm_type, bn_momentum)
        self.norm3 = NormBlock(in_channels, norm_type, bn_momentum)
        self.activation = activation

        # Optimizations
        if layer_scale_init_v > 0:
            self.gamma = nn.Parameter(layer_scale_init_v * torch.ones((out_channels)), requires_grad=True)
        else:
            self.gamma = None
        self.drop_path = DropPathPack(drop_path) if drop_path > 0. else nn.Identity()

        return

    def forward(self, q_pts, s_pts, s_feats, neighbor_indices):

        # First depthwise convolution
        x = self.conv(q_pts, s_pts, s_feats, neighbor_indices)
        x = self.norm0(x)

        # Activation
        x = self.activation(x)

        # Upscale features
        x = self.linear1(x)
        x = self.norm1(x)

        # Activation
        x = self.activation(x)

        # Downscale features
        x = self.linear2(x)
        x = self.norm2(x)

        # LayerScale
        if self.gamma is not None:
            x = self.gamma * x

        # Adapt shortcut in case block is strided
        if self.strided:
            shortcut = local_maxpool(s_feats, neighbor_indices)
        else:
            shortcut = s_feats

        # Adapt shortcut in case this is a upscaling layer (only happens when strided)
        shortcut = self.linear_shortcut(shortcut)
        shortcut = self.norm3(shortcut)

        # Apply shortcut with stochastic depth
        q_feats = shortcut + self.drop_path(x)

        return q_feats

