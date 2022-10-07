#
#
#      0==============================0
#      |    KPConvX network blocks    |
#      0==============================0
#
#
# ----------------------------------------------------------------------------------------------------------------------
#
#      Here we define the network blocks using kpconvx
#
# ----------------------------------------------------------------------------------------------------------------------
#
#      Hugues THOMAS - 06/03/2020
#


import math
from symbol import return_stmt
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
#           KPConvD Class
#       \*******************/
#


class KPConvD(nn.Module):

    def __init__(self,
                 channels: int,
                 shell_sizes: list,
                 radius: float,
                 sigma: float,
                 Cmid: int = 0,
                 shared_kp_data = None,
                 dimension: int = 3,
                 influence_mode: str = 'linear',
                 fixed_kernel_points: str = 'center',
                 norm_type: str = 'batch',
                 bn_momentum: float = 0.1,
                 activation: nn.Module = nn.LeakyReLU(0.1),
                 inf: float = 1e6):
        """
        Mini KPConv. Basically a depthwise KPConv with nearest aggregation optimized to run faster.
        Option to use MLP instead of kernel to get neighbor weights (then we are similar to PointNeXt)
        Args:
            channels                     (int): The number of channels.
            shell_sizes                 (list): The number of kernel points per shell.
            radius                     (float): The radius used for kernel point init.
            sigma                      (float): The influence radius of each kernel point.
            Cmid                         (int): Dimension of mid f. 0 for depthwise conv, > 0 for PointConv style
            shared_kp_data              (None): Optional data dict shared across the layer
            dimension                  (int=3): The dimension of the point space.
            influence_mode      (str='linear'): Influence function ('constant', 'linear', 'gaussian').
            fixed_kernel_points (str='center'): kernel points whose position is fixed ('none', 'center' or 'verticals').
            norm_type            (str='batch'): type of normalization used in layer ('group', 'batch', 'none')
            bn_momentum           (float=0.10): Momentum for batch normalization
            activation              (nn.Module: Activation function. Use None for no activation.
            inf (float=1e6): The value of infinity to generate the padding point.
        """
        super(KPConvD, self).__init__()

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
        self.Cmid = Cmid

        # Initialize weights
        if Cmid > 0:
            self.weights = nn.Parameter(torch.zeros(size=(self.K, Cmid)), requires_grad=True)
            self.out_mlp = nn.Linear(Cmid * channels, channels)
        else:
            self.weights = nn.Parameter(torch.zeros(size=(self.K, channels)), requires_grad=True)

        # Reset parameters
        self.reset_parameters()

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

        # Handle mlp case 
        if self.influence_mode == 'mlp':
            if Cmid > 0:
                # MLP does not have it final linear layer. It is the same as point conv
                self.delta_mlp = UnaryBlock(self.dimension, Cmid, norm_type, bn_momentum, activation)

            else:
                # Complete mlp
                self.delta_mlp = build_mlp(n_layers=2,
                                           Cin=self.dimension,
                                           Cmid=16,
                                           Cout=channels,
                                           norm_type=norm_type,
                                           bn_momentum=bn_momentum,
                                           activation=activation)


        return

    def reset_parameters(self):
        kaiming_uniform_(self.weights, a=math.sqrt(5))
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

        # Add a zero feature for shadow neighbors
        padded_s_feats = torch.cat((s_feats, torch.zeros_like(s_feats[:1, :])), 0)  # (N, C) -> (N+1, C)

        # Get the features of each neighborhood
        # neighbor_feats = gather(padded_s_feats, neighb_inds)  # (N+1, C) -> (M, H, C)
        neighbor_feats = index_select(padded_s_feats, neighb_inds, dim=0)

        # Get nearest kernel point (M, H) and weights applied to each neighbors (M, H)
        influence_weights, neighbors, neighbors_1nn = self.get_neighbors_influences(q_pts, s_pts, neighb_inds)

        if self.influence_mode == 'mlp':

            # Generate geometric encodings
            neighbors_weights = self.delta_mlp(neighbors) # (M, H, 3) -> (M, H, C)

        else:

            # Collect nearest kernel point weights -> (M, H, C or Cmid)
            neighbors_weights = gather(self.weights, neighbors_1nn)

            # Apply influence weights
            if self.influence_mode != 'constant':
                neighbors_weights *= influence_weights.unsqueeze(2)
        

        if self.Cmid > 0:

            # Apply weights via matmul
            intermediate_feats = torch.matmul(neighbors_weights.transpose(1, 2), neighbor_feats)  # (M, Cmid, H) x (M, H, C) -> (M, Cmid, C)

            # Final linear combination
            output_feats = self.out_mlp(intermediate_feats.view(-1, self.Cmid * self.channels))

        else:

            # Apply weights and summation
            output_feats = self.aggr_op(self.merge_op(neighbor_feats, neighbors_weights), dim=1)  # (M, H, C) -> (M, C)


        return output_feats

    def __repr__(self):

        repr_str = 'KPConvD'
        repr_str += '(K: {:d}'.format(self.K)
        repr_str += ', C: {:d}'.format(self.channels)
        repr_str += ', r: {:.2f}'.format(self.radius)
        repr_str += ', sigma: {:.2f})'.format(self.sigma)

        return repr_str


# ----------------------------------------------------------------------------------------------------------------------
#
#           KPConvX Class
#       \*******************/
#


class KPConvX(nn.Module):

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
        Next generation KPConv. Basically a depthwise KPConv with nearest aggregation optimized to run faster, 
        and self-generated modulation for attention.
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
        super(KPConvX, self).__init__()

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

        repr_str = 'KPConvX'
        repr_str += '(K: {:d}'.format(self.K)
        repr_str += ', C: {:d}'.format(self.channels)
        repr_str += ', r: {:.2f}'.format(self.radius)
        repr_str += ', sigma: {:.2f})'.format(self.sigma)

        return repr_str


# ----------------------------------------------------------------------------------------------------------------------
#
#           Network blocks
#       \********************/
#


class KPNextBlock(nn.Module):

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 shell_sizes: list,
                 radius: float,
                 sigma: float,
                 attention_groups: int = 8,
                 attention_act: str = 'sigmoid',
                 mod_grp_norm: bool = False,
                 mlp_first: bool= True,
                 shared_kp_data = None,
                 influence_mode: str = 'linear',
                 dimension: int = 3,
                 norm_type: str = 'batch',
                 bn_momentum: float = 0.1,
                 activation: nn.Module = nn.LeakyReLU(0.1)):
        """
        KPNext block where we can choose KPConvX or KPConvD. 
        If features dimension is changed, we combine with a linear layer.
        Args:
            in_channels             (int): dimension input features
            out_channels            (int): dimension input features
            shell_sizes            (list): The number of kernel points per shell.
            radius                (float): convolution radius
            sigma                 (float): influence radius of each kernel point
            attention_groups      (int=8): number of groups in attention (negative value for ch_per_grp).
            attention_act           (str): Activate the weight with 'none', 'sigmoid', 'softmax' or 'tanh'.
            mod_grp_norm     (bool=False): Use group norm for modulations or not.
            mlp_first        (bool=False): If we need mlp, use it first or last.
            shared_kp_data      (None): Optional data dict shared across the layer
            influence_mode (str='linear'): Influence function ('constant', 'linear', 'gaussian')
            dimension             (int=3): dimension of input
            norm_type       (str='batch'): type of normalization used in layer ('group', 'batch', 'none')
            bn_momentum      (float=0.10): Momentum for batch normalization
            activation (nn.Module|None=nn.LeakyReLU(0.1)): Activation function. Use None for no activation.
        """
        super(KPNextBlock, self).__init__()

        # Define parameters
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.influence_mode = influence_mode
        self.radius = radius
        self.bn_momentum = bn_momentum
        self.norm_type = norm_type
        self.mlp_first = mlp_first
        
        if mlp_first:
            conv_channels = out_channels
        else:
            conv_channels = in_channels

        # Define modules
        self.activation = activation
        self.conv_norm = NormBlock(conv_channels, norm_type, bn_momentum)
        
        # KPConvX or KPConvD
        if attention_groups == 0:
            self.conv = KPConvD(conv_channels,
                                shell_sizes,
                                radius,
                                sigma,
                                Cmid=0,
                                shared_kp_data=shared_kp_data,
                                dimension=dimension,
                                influence_mode=influence_mode,
                                norm_type=norm_type,
                                bn_momentum=bn_momentum,
                                activation=activation)
        else:
            self.conv = KPConvX(conv_channels,
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

        # Optional mlp to up feature
        if in_channels != out_channels:         
            self.up_mlp = nn.Sequential(nn.Linear(in_channels, out_channels, bias=False),
                                        NormBlock(out_channels, norm_type, bn_momentum),
                                        activation)

        return


    def forward(self, q_pts, s_pts, x, neighbor_indices):

        # Option 1: Upscale features first
        if self.mlp_first:
            if self.in_channels != self.out_channels:    
                x = self.up_mlp(x)
            x = self.conv(q_pts, s_pts, x, neighbor_indices)
            x = self.conv_norm(x)
            x = self.activation(x)

        #  Option 2: Upscale features last
        else:
            x = self.conv(q_pts, s_pts, x, neighbor_indices)
            x = self.conv_norm(x)
            x = self.activation(x)
            if self.in_channels != self.out_channels:    
                x = self.up_mlp(x)

        return x

    def __repr__(self):
        return 'KPNextBlock(in_C: {:d}, out_C: {:d}, r: {:.2f}, mode: {:s})'.format(self.in_channels,
                                                                                          self.out_channels,
                                                                                          self.radius,
                                                                                          self.influence_mode)


class KPNextResidualBlock(nn.Module):

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 shell_sizes: list,
                 radius: float,
                 sigma: float,
                 attention_groups: int = 8,
                 attention_act: str = 'sigmoid',
                 mod_grp_norm: bool = False,
                 shared_kp_data = None,
                 influence_mode: str = 'linear',
                 dimension: int = 3,
                 strided: bool = False,
                 norm_type: str = 'batch',
                 bn_momentum: float = 0.1,
                 activation: nn.Module = nn.LeakyReLU(0.1)):
        """
        Old design of bottleneck residual block. With the option to use KPConvX or KPConvD.
        Args:
            in_channels             (int): dimension input features
            out_channels            (int): dimension input features
            shell_sizes            (list): The number of kernel points per shell.
            radius                (float): convolution radius
            sigma                 (float): influence radius of each kernel point
            attention_groups      (int=8): groups in KPConvX modulations / <0  for ch_per_grp / 0 for KPConvD.
            attention_act           (str): Activate the weight with 'none', 'sigmoid', 'softmax' or 'tanh'.
            mod_grp_norm     (bool=False): Use group norm for modulations or not.
            shared_kp_data      (None): Optional data dict shared across the layer
            influence_mode (str='linear'): Influence function ('constant', 'linear', 'gaussian')
            dimension             (int=3): dimension of input
            strided          (bool=False): strided or not
            norm_type       (str='batch'): type of normalization used in layer ('group', 'batch', 'none')
            bn_momentum      (float=0.10): Momentum for batch normalization
            activation (nn.Module|None=nn.LeakyReLU(0.1)): Activation function. Use None for no activation.
        """
        super(KPNextResidualBlock, self).__init__()

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

        if attention_groups == 0:
            self.conv = KPConvD(mid_channels,
                                shell_sizes,
                                radius,
                                sigma,
                                Cmid=0,
                                shared_kp_data=shared_kp_data,
                                dimension=dimension,
                                influence_mode=influence_mode,
                                norm_type=norm_type,
                                bn_momentum=bn_momentum,
                                activation=activation)
        else:
            self.conv = KPConvX(mid_channels,
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


class KPNextInvertedBlock(nn.Module):

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 shell_sizes: list,
                 radius: float,
                 sigma: float,
                 attention_groups: int = 8,
                 attention_act: str = 'sigmoid',
                 mod_grp_norm: bool = False,
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
        KPNext inverted block as in ConvNext (and PointNext).
        Args:
            in_channels             (int): dimension input features
            out_channels            (int): dimension input features
            shell_sizes            (list): The number of kernel points per shell.
            radius                (float): convolution radius
            sigma                 (float): influence radius of each kernel point
            attention_groups      (int=8): number of groups in attention (negative value for ch_per_grp).
            attention_act           (str): Activate the weight with 'none', 'sigmoid', 'softmax' or 'tanh'.
            mod_grp_norm     (bool=False): Use group norm for modulations or not.
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
        super(KPNextInvertedBlock, self).__init__()

        # Define parameters
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.expansion = expansion
        self.strided = strided
        self.bn_momentum = bn_momentum
        self.norm_type = norm_type
        mid_channels = out_channels * expansion


        # KPTransformer
        if attention_groups == 0:
            self.conv = KPConvD(in_channels,
                                shell_sizes,
                                radius,
                                sigma,
                                Cmid=0,
                                shared_kp_data=shared_kp_data,
                                dimension=dimension,
                                influence_mode=influence_mode,
                                norm_type=norm_type,
                                bn_momentum=bn_momentum,
                                activation=activation)
        else:
            self.conv = KPConvX(in_channels,
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

        # learnable parameters
        self.linear1 = nn.Linear(in_channels, mid_channels, bias=False)
        self.linear2 = nn.Linear(mid_channels, out_channels, bias=False)

        if in_channels != out_channels:
            self.linear_shortcut =nn.Linear(in_channels, out_channels, bias=False)
        else:
            self.linear_shortcut = nn.Identity()

        # Other parameters
        self.norm0 = NormBlock(in_channels, norm_type, bn_momentum)
        self.norm1 = NormBlock(mid_channels, norm_type, bn_momentum)
        self.norm2 = NormBlock(out_channels, norm_type, bn_momentum)
        self.norm3 = NormBlock(out_channels, norm_type, bn_momentum)
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
        x = self.activation(x)

        # Upscale features
        x = self.linear1(x)
        x = self.norm1(x)
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

        # Adapt shortcut in case this is a upscaling layer
        shortcut = self.linear_shortcut(shortcut)
        shortcut = self.norm3(shortcut)

        # Apply shortcut with stochastic depth
        q_feats = shortcut + self.drop_path(x)

        return q_feats


class KPNextMultiShortcutBlock(nn.Module):

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 shell_sizes: list,
                 radius: float,
                 sigma: float,
                 attention_groups: int = 8,
                 attention_act: str = 'sigmoid',
                 mod_grp_norm: bool = False,
                 expansion: int = 4,
                 drop_path_p: float = -1.,
                 layer_scale_init_v: float = -1.,
                 use_upcut: bool = False,
                 shared_kp_data = None,
                 influence_mode: str = 'linear',
                 dimension: int = 3,
                 norm_type: str = 'batch',
                 bn_momentum: float = 0.1,
                 activation: nn.Module = nn.LeakyReLU(0.1)):
        """
        KPNext block to define layers with multiple interconnected shortcuts. 
        
        Traditionally we define:
            > Resnet:   --- downMLP --- Conv --- upMLP --- + --- downMLP --- Conv --- upMLP --- + --- 
                         ┕---------------->----------------┙  ┕---------------->----------------┙

            > Inverted:             --- Conv --- upMLP --- downMLP --- + --- Conv --- upMLP --- downMLP --- + ---
                                     ┕---------------->----------------┙  ┕---------------->----------------┙

        The only difference is where the shortcut is placed. Here we wonder, what if we place both shortcut types:

                                            ┍------------------>--------------------┑  ┍----------->-------
            > KPNext:   --- Conv --- upMLP --- downMLP --- + --- Conv --- upMLP --- + --- downMLP --- + ---   ...
                         ┕---------------->----------------┙  ┕------------------>--------------------┙
        
        We call the shortcut after upMLP: "upcut" and the shortcut after downmlp the "downcut"
        Because of the multiple shortcuts we cannot define this layer block by block.
        Args:
            in_channels             (int): dimension input features
            out_channels            (int): dimension input features
            shell_sizes            (list): The number of kernel points per shell.
            radius                (float): convolution radius
            sigma                 (float): influence radius of each kernel point
            attention_groups      (int=8): number of groups in attention (negative value for ch_per_grp).
            attention_act           (str): Activate the weight with 'none', 'sigmoid', 'softmax' or 'tanh'.
            mod_grp_norm     (bool=False): Use group norm for modulations or not.
            expansion               (int): Factor for linear layer expansion
            drop_path             (float): Proba to drop convolution paths (for stochastic network depth)
            layer_scale_init_v    (float): Value for initialization of layer scales
            shared_kp_data      (None): Optional data dict shared across the layer
            influence_mode (str='linear'): Influence function ('constant', 'linear', 'gaussian')
            dimension             (int=3): dimension of input
            norm_type       (str='batch'): type of normalization used in layer ('group', 'batch', 'none')
            bn_momentum      (float=0.10): Momentum for batch normalization
            activation (nn.Module|None=nn.LeakyReLU(0.1)): Activation function. Use None for no activation.
        """
        super(KPNextMultiShortcutBlock, self).__init__()

        # Define parameters
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.expansion = expansion
        self.bn_momentum = bn_momentum
        self.norm_type = norm_type
        self.use_upcut = use_upcut
        mid_channels = out_channels * expansion


        # KPConvX or KPConvD
        if attention_groups == 0:
            self.conv = KPConvD(in_channels,
                                shell_sizes,
                                radius,
                                sigma,
                                Cmid=0,
                                shared_kp_data=shared_kp_data,
                                dimension=dimension,
                                influence_mode=influence_mode,
                                norm_type=norm_type,
                                bn_momentum=bn_momentum,
                                activation=activation)
        else:
            self.conv = KPConvX(in_channels,
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

        # Convolution normalization
        self.conv_norm = NormBlock(in_channels, norm_type, bn_momentum)

        # learnable parameters
        self.activation = activation
        self.up_mlp = nn.Sequential(nn.Linear(in_channels, mid_channels, bias=False),
                                    NormBlock(mid_channels, norm_type, bn_momentum))
        self.down_mlp = nn.Sequential(nn.Linear(mid_channels, out_channels, bias=False),
                                    NormBlock(out_channels, norm_type, bn_momentum))

        if in_channels != out_channels:
            self.mlp_downcut = nn.Sequential(nn.Linear(in_channels, out_channels, bias=False),
                                             NormBlock(out_channels, norm_type, bn_momentum))
        else:
            self.mlp_downcut = nn.Identity()



        # Optimizations
        if layer_scale_init_v > 0:
            self.gamma = nn.Parameter(layer_scale_init_v * torch.ones((out_channels)), requires_grad=True)
        else:
            self.gamma = None
        self.drop_path_p = drop_path_p
        if self.drop_path_p > 0:
            self.drop_path = DropPathPack(drop_path_p)

        return

    def forward(self, q_pts, s_pts, s_feats, neighbor_indices, upcut=None):

        # Get downcut here
        downcut = s_feats

        # First depthwise convolution
        x = self.conv(q_pts, s_pts, s_feats, neighbor_indices)
        x = self.conv_norm(x)
        x = self.activation(x)

        # Upscale features
        x = self.up_mlp(x)

        # Optional drop path
        if self.drop_path_p > 0:
            x, drop_mask = self.drop_path(x, return_mask=True)

        # Apply incoming upcut (we never need linear on this shortcut)
        if upcut is not None:
            x = upcut + x
            
        # Post-shortcut activation
        x = self.activation(x)

        # Get upcut here, we need to return it to be used in the next block
        if self.use_upcut:
            upcut = x
        else:
            upcut = None

        # Downscale features
        x = self.down_mlp(x)

        # LayerScale
        if self.gamma is not None:
            x = self.gamma * x

        # Same drop path should be applied here
        if self.drop_path_p > 0:
            x = drop_mask * x

        # Apply downcut
        if self.in_channels != self.out_channels:
            downcut = self.mlp_downcut(downcut)
        x = downcut + x

        # Final activation
        q_feats = self.activation(x)

        return q_feats, upcut

