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
    NormBlock, DropPathPack

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



class KPMini(nn.Module):

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
                 bn_momentum: float = 0.98,
                 activation: nn.Module = nn.LeakyReLU(0.1),
                 inf: float = 1e6):
        """
        Mini KPConv. Basically a depthwise KPConv with nearest aggregation optimized to run faster.
        Option to switch final multiplication with another operation
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
            bn_momentum           (float=0.98): Momentum for batch normalization
            activation              (nn.Module: Activation function. Use None for no activation.
            inf (float=1e6): The value of infinity to generate the padding point.
        """
        super(KPMini, self).__init__()

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

        if self.influence_mode == 'mlp':
            # Define MLP delta
            delta_layers = 2
            C = channels
            D = self.dimension
            Cmid = 8
            if delta_layers < 2:
                self.delta_mlp = nn.Linear(D, C)
            else:
                self.delta_mlp = nn.Sequential(UnaryBlock(D, Cmid, norm_type, bn_momentum, activation))
                for _ in range(delta_layers - 2):
                    self.delta_mlp.append(UnaryBlock(Cmid, Cmid, norm_type, bn_momentum, activation))
                self.delta_mlp.append(nn.Linear(Cmid, C, bias=False))

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

            # Apply weights and summation
            output_feats = self.aggr_op(self.merge_op(neighbor_feats, neighbors_weights), dim=1)  # (M, H, C) -> (M, C)

        else:

            # Choose between two mode (summation or gathering). Gathering is only worth if K > 20
            if self.K > 25 or self.Cmid > 0:

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

            else:

                # WARNING THIS ONLY WORKS WITH merge_op=mul AND aggr_op=sum

                # Create 1-hot weights from 1nn -> (M, K, H)
                one_hot_w = torch.transpose(nn.functional.one_hot(neighbors_1nn, self.K), 1, 2).type(torch.float32)

                # Apply influence weights
                if self.influence_mode != 'constant':
                    one_hot_w *= influence_weights.unsqueeze(1)

                # Apply 1-hot weights to neihbor feature (projection of features on kenrel points)
                weighted_feats = torch.matmul(one_hot_w, neighbor_feats)  # (M, K, H) x (M, H, C) -> (M, K, C)

                # Sum features over the kenrel points
                output_feats = torch.sum(weighted_feats * self.weights.unsqueeze(0), dim=1)  # (M, K, C) -> (M, C)

        return output_feats

    def __repr__(self):

        repr_str = 'KPMini'
        repr_str += '(K: {:d}'.format(self.K)
        repr_str += ', C: {:d}'.format(self.channels)
        repr_str += ', r: {:.2f}'.format(self.radius)
        repr_str += ', sigma: {:.2f})'.format(self.sigma)

        return repr_str

class KPConv(nn.Module):

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 shell_sizes: list,
                 radius: float,
                 sigma: float,
                 shared_kp_data = None,
                 modulated: bool = False,
                 use_geom: bool = False,
                 groups: int = 1,
                 dimension: int = 3,
                 influence_mode: str = 'linear',
                 aggregation_mode: str = 'sum',
                 fixed_kernel_points: str = 'center',
                 norm_type: str = 'batch',
                 bn_momentum: float = 0.98,
                 activation: nn.Module = nn.LeakyReLU(0.1),
                 inf: float = 1e6):
        """
        Rigid KPConv.
        Paper: https://arxiv.org/abs/1904.08889.
        Args:
            in_channels        (int): The number of the input channels.
            out_channels       (int): The number of the output channels.
            shell_sizes       (list): The number of kernel points per shell.
            radius           (float): The radius used for kernel point init.
            sigma            (float): The influence radius of each kernel point.
            shared_kp_data (None): Optional data dict shared across the layer
            modulated   (bool=False): Use modulations (self-attention)
            use_geom    (bool=False): Use geometric encodings
            groups           (int=1): Groups in convolution (=-1 for depthwise conv).
            dimension        (int=3): The dimension of the point space.
            influence_mode      (str='linear'): Influence function ('constant', 'linear', 'gaussian').
            aggregation_mode       (str='sum'): Aggregation mode ('nearest', 'sum').
            fixed_kernel_points (str='center'): kernel points whose position is fixed ('none', 'center' or 'verticals').
            norm_type            (str='batch'): type of normalization used in layer ('group', 'batch', 'none')
            bn_momentum           (float=0.98): Momentum for batch normalization
            activation (nn.Module|None=nn.LeakyReLU(0.1)): Activation function. Use None for no activation.
            inf (float=1e6): The value of infinity to generate the padding point.
        """
        super(KPConv, self).__init__()

        # Verification of group parameter
        if groups < 0:
            groups = in_channels
        assert in_channels % groups == 0, "in_channels must be divisible by groups."
        assert out_channels % groups == 0, "out_channels must be divisible by groups."
        in_channels_per_group = in_channels // groups
        out_channels_per_group = out_channels // groups

        # Save parameters
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.shell_sizes = shell_sizes
        self.K = np.sum(shell_sizes).item()
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
        self.use_geom = use_geom
        self.normalize_p = True
        self.gather_mode = (groups == in_channels and self.aggregation_mode == 'nearest' and self.K > 25)

        # Initialize weights
        weights = torch.zeros(size=(self.K, groups, in_channels_per_group, out_channels_per_group))
        self.weights = nn.Parameter(weights, requires_grad=True)

        # modulations, MLP
        self.modulated = modulated
        if self.modulated:
            self.gen_mlp = nn.Linear(in_channels, self.K, bias=True)

        # Define MLP delta
        delta_layers = 2
        delta_reduction = 4
        C = in_channels
        D = self.dimension
        R = delta_reduction
        self.use_gamma_mlp = False
        if use_geom:
            if delta_layers < 2:
                self.delta_mlp = nn.Linear(D, C)
            else:
                self.delta_mlp = nn.Sequential(UnaryBlock(D, C // R, norm_type, bn_momentum, activation))
                for _ in range(delta_layers - 2):
                    self.delta_mlp.append(UnaryBlock(C // R, C // R, norm_type, bn_momentum, activation))
                self.delta_mlp.append(nn.Linear(C // R, C, bias=False))

            # Define MLP gamma
            self.use_gamma_mlp = True
            if self.use_gamma_mlp:
                self.init_linear = nn.Linear(C, C, bias=False)
                self.gamma_mlp = nn.Sequential(NormBlock(C),
                                            activation)

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

            # Get Kernel point distances to neigbors
            differences = neighbors.unsqueeze(2) - self.kernel_points  # (M, H, 1, 3) x (K, 3) -> (M, H, K, 3)
            sq_distances = torch.sum(differences ** 2, dim=3)  # (M, H, K)

            # In case of nearest mode, only the nearest KP can influence each point
            if self.aggregation_mode == 'nearest':
                nn_sq_dists, neighbors_1nn = torch.min(sq_distances, dim=2)
                if self.gather_mode:
                    sq_distances = nn_sq_dists

            elif self.aggregation_mode != 'sum':
                raise ValueError("Unknown aggregation mode: '{:s}'. Should be 'nearest' or 'sum'".format(self.aggregation_mode))

            # Get Kernel point influences
            if self.influence_mode == 'constant':
                # Every point get an influence of 1.
                influence_weights = torch.ones_like(sq_distances)

            elif self.influence_mode == 'linear':
                # Influence decrease linearly with the distance, and get to zero when d = sigma.
                influence_weights = torch.clamp(1 - torch.sqrt(sq_distances) / self.sigma, min=0.0)  # (M, H, K)

            elif self.influence_mode == 'gaussian':
                # Influence in gaussian of the distance.
                gaussian_sigma = self.sigma * 0.3
                influence_weights = radius_gaussian(sq_distances, gaussian_sigma)
            else:
                raise ValueError("Unknown influence mode: : '{:s}'.  Should be 'constant', 'linear', or 'gaussian'".format(self.aggregation_mode))

            if not self.gather_mode:
                influence_weights = torch.transpose(influence_weights, 1, 2)  # (M, H, K) -> (M, K, H)
                if self.aggregation_mode == 'nearest':
                    influence_weights *= torch.transpose(nn.functional.one_hot(neighbors_1nn, self.K), 1, 2)

            # Share with next kernels if necessary
            if self.share_kp:
                self.shared_kp_data['infl_w'] = influence_weights
                self.shared_kp_data['neighb_p'] = neighbors
                self.shared_kp_data['neighb_1nn'] = neighbors_1nn

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

        # Get features for each neighbor
        # ******************************

        if self.use_gamma_mlp:
            # Init linear transform
            s_feats = self.init_linear(s_feats) # (N, C) -> (N, C)

        # Add a zero feature for shadow neighbors
        padded_s_feats = torch.cat((s_feats, torch.zeros_like(s_feats[:1, :])), 0)  # (N, C) -> (N+1, C)

        # Get the features of each neighborhood
        # neighbor_feats = gather(padded_s_feats, neighb_inds)  # (N+1, C) -> (M, H, C)
        neighbor_feats = index_select(padded_s_feats, neighb_inds, dim=0)


        # Get geometric encoding features
        # *******************************

        # Get Kernel point influences (M, K, H)
        influence_weights, neighbors, neighbors_1nn = self.get_neighbors_influences(q_pts, s_pts, neighb_inds)

        if self.use_geom:

            # Rescale for normalization
            if self.normalize_p:
                neighbors *= 1 / self.radius   # -> (M, H, 3)

            # Generate geometric encodings
            geom_encodings = self.delta_mlp(neighbors) # (M, H, 3) -> (M, H, C)

            # Merge with features (we use add or sub mode, both are equivalent)
            neighbor_feats += geom_encodings  # -> (M, H, C)

            if self.use_gamma_mlp:
                # Final activation (+optional mlp)
                neighbor_feats = self.gamma_mlp(neighbor_feats) # (M, H, C) -> (M, H, C)


        # Special mode: gather nearest neighbor for large kernels
        # *******************************************************

        # Choose between two mode (summation or gathering).
        # Gathering only worth if K > 25, nearest mode and depthwise_conv
        if self.gather_mode:

            # Collect nearest kernel point weights -> (M, H, G, C//G, 0//G)
            neighbors_weights = gather(self.weights, neighbors_1nn)

            # Apply influence weights
            H = int(influence_weights.shape[1])
            if self.influence_mode != 'constant':
                neighbors_weights *= influence_weights.view(-1, H, 1, 1, 1)

            # Depthwise
            neighbors_weights = neighbors_weights.view(-1, H, self.groups, self.out_channels_per_group)  # (M, H, G=C, O//G)
            neighbor_feats = neighbors_weights.view(-1, H, self.groups, 1)  # (M, H, C, 1)

            # Apply weights and summation
            output_feats = torch.sum(neighbor_feats * neighbors_weights, dim=1)  # -> (M, G, 0//G)
            output_feats = output_feats.reshape((-1, self.out_channels))  # -> (M, O)


        # Normal mode: summation over the kernel points
        # *********************************************

        else:

            # Apply distance weights
            weighted_feats = torch.matmul(influence_weights, neighbor_feats)  # (M, K, H) x (M, H, C) -> (M, K, C)

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
            if self.groups == self.in_channels and self.out_channels == self.in_channels:

                # Depthwise conv
                weights = self.weights.view(1, self.K, self.groups) # (K, C, 1, 1) -> (1, K, C)
                output_feats = torch.sum(weighted_feats * weights, dim=1)  # (M, K, C) -> (M, C)

            else:
                # group conv
                weighted_feats = weighted_feats.view(-1, self.K, self.groups, self.in_channels_per_group)  # (M, K, C) -> (M, K, G, C//G)
                output_feats = torch.einsum("mkgc,kgcd->mgd", weighted_feats, self.weights)  # (M, K, G, C//G) * (K, G, C//G, O//G) -> (M, G, O//G)
                output_feats = output_feats.reshape((-1, self.out_channels))  # (M, G, O//G) -> (M, O)

        # # density normalization (divide output features by the sum of neighbor positive features)
        # neighbor_feats_sum = torch.sum(neighbor_feats, dim=-1)  # (M, H)
        # neighbor_num = torch.sum(torch.gt(neighbor_feats_sum, 0.0), dim=-1)  # (M,)
        # neighbor_num = torch.max(neighbor_num, torch.ones_like(neighbor_num))  # (M,)
        # output_feats = output_feats / neighbor_num.unsqueeze(1)

        return output_feats

    def __repr__(self):

        repr_str = 'KPConv'
        repr_str += '(K: {:d}'.format(self.K)
        repr_str += ', in_C: {:d}'.format(self.in_channels)
        repr_str += ', out_C: {:d}'.format(self.out_channels)
        repr_str += ', r: {:.2f}'.format(self.radius)
        repr_str += ', sigma: {:.2f})'.format(self.sigma)

        return repr_str


class KPDef(nn.Module):

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 shell_sizes: list,
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
            shell_sizes (list): The number of kernel points per shell.
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
        self.shell_sizes = shell_sizes
        self.K = np.sum(shell_sizes).item()
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
            weights = torch.zeros(size=(self.K, in_channels, out_channels))
        else:
            weights = torch.zeros(size=(self.K, groups, in_channels_per_group, out_channels_per_group))
        self.weights = nn.Parameter(weights, requires_grad=True)

        # Deformation generation (temporary make some test then keep only the best implem)
        self.modulated = modulated
        if self.modulated:
            self.offset_dim = (self.dimension + 1) * self.K
        else:
            self.offset_dim = self.dimension * self.K

        self.version = 'v1'
        if self.version == 'v1':
            # MLP
            self.offset_mlp = nn.Linear(in_channels, self.offset_dim, bias=True)

        elif self.version == 'v2':
            # KPConv
            self.offset_conv = KPConv(in_channels,
                                      self.offset_dim,
                                      shell_sizes,
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
        kernel_points = load_kernels(self.radius, self.shell_sizes, dimension=self.dimension, fixed=self.fixed_kernel_points)
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
            influence_weights (Tensor): the influence weight of each kernel point on each neighbors point (M, K, H).
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
                influence_weights = torch.ones_like(sq_distances.detach())

            elif self.influence_mode == 'linear':
                # Influence decrease linearly with the distance, and get to zero when d = sigma.
                influence_weights = torch.clamp(1 - torch.sqrt(sq_distances.detach()) / self.sigma, min=0.0)  # (M, H, K)

            elif self.influence_mode == 'gaussian':
                # Influence in gaussian of the distance.
                gaussian_sigma = self.sigma * 0.3
                influence_weights = radius_gaussian(sq_distances.detach(), gaussian_sigma)
            else:
                raise ValueError("Unknown influence mode: : '{:s}'.  Should be 'constant', 'linear', or 'gaussian'".format(self.aggregation_mode))
            influence_weights = torch.transpose(influence_weights, 1, 2)  # (M, H, K) -> (M, K, H)

            # In case of nearest mode, only the nearest KP can influence each point
            if self.aggregation_mode == 'nearest':
                neighbors_1nn = torch.argmin(sq_distances.detach(), dim=2)
                influence_weights *= torch.transpose(nn.functional.one_hot(neighbors_1nn, self.K), 1, 2)

            elif self.aggregation_mode != 'sum':
                raise ValueError("Unknown aggregation mode: '{:s}'. Should be 'nearest' or 'sum'".format(self.aggregation_mode))

        return influence_weights

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
            unscaled_offsets = self.offset_features[:, :self.dimension * self.K]
            unscaled_offsets = unscaled_offsets.view(-1, self.K, self.dimension)  # (M, K, D)

            # Get modulations
            modulations = 2 * torch.sigmoid(self.offset_features[:, self.dimension * self.K:])   # (M, K)

        else:
            unscaled_offsets = self.offset_features.view(-1, self.K, self.dimension)  # (M, K, D)

        # Rescale offset for this layer
        offsets = unscaled_offsets * self.radius

        # Apply offsets to kernel points [n_points, n_kpoints, dim]
        self.deformed_KP = offsets + self.kernel_points  # (M, K, D)
        deformed_K_points = self.deformed_KP.unsqueeze(1)  # (M, 1, K, D)


        # Transfer features to kernel points
        # **********************************

        # Get Kernel point influences (M, K, H)
        influence_weights = self.get_neighbors_influences(q_pts, s_pts, neighb_inds, deformed_K_points)

        # Apply distance weights
        weighted_feats = torch.matmul(influence_weights, neighbor_feats)  # (M, K, H) x (M, H, C) -> (M, K, C)

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
            weighted_feats = weighted_feats.view(-1, self.K, self.groups, self.in_channels_per_group)  # (M, K, C) -> (M, K, G, C//G)
            output_feats = torch.einsum("mkgc,kgcd->mgd", weighted_feats, self.weights)  # (M, K, G, C//G) * (K, G, C//G, O//G) -> (M, G, O//G)
            output_feats = output_feats.view(-1, self.out_channels)  # (M, G, O//G) -> (M, O)

            # weighted_feats = weighted_feats.view(-1, self.K, self.groups, self.in_channels_per_group)  # (M, K, C) -> (M, K, G, C//G)
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
        repr_str += '(K: {:d}'.format(self.K)
        repr_str += ', in_C: {:d}'.format(self.in_channels)
        repr_str += ', out_C: {:d}'.format(self.out_channels)
        repr_str += ', r: {:.2f}'.format(self.radius)
        repr_str += ', sigma: {:.2f})'.format(self.sigma)

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
                 shell_sizes: list,
                 radius: float,
                 sigma: float,
                 shared_kp_data = None,
                 modulated: bool = False,
                 deformable: bool = False,
                 use_geom: bool = False,
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
            in_channels             (int): dimension input features
            out_channels            (int): dimension input features
            shell_sizes            (list): The number of kernel points per shell.
            radius                (float): convolution radius
            sigma                 (float): influence radius of each kernel point
            shared_kp_data      (None): Optional data dict shared across the layer
            modulated        (bool=False): Use modulations (self-attention)
            use_geom         (bool=False): Use geometric encodings
            deformable       (bool=False): Use deformable KPConv
            influence_mode (str='linear'): Influence function ('constant', 'linear', 'gaussian')
            aggregation_mode  (str='sum'): Aggregation mode ('nearest', 'sum')
            dimension             (int=3): dimension of input
            groups                (int=1): Number of groups in KPConv
            norm_type       (str='batch'): type of normalization used in layer ('group', 'batch', 'none')
            bn_momentum      (float=0.98): Momentum for batch normalization
            activation (nn.Module|None=nn.LeakyReLU(0.1)): Activation function. Use None for no activation.
        """
        super(KPConvBlock, self).__init__()

        # Define parameters
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.shell_sizes = shell_sizes
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
        self.norm = NormBlock(out_channels, norm_type, bn_momentum)

        if deformable:
            self.conv = KPDef(in_channels,
                              out_channels,
                              shell_sizes,
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
                               shell_sizes,
                               radius,
                               sigma,
                               shared_kp_data=shared_kp_data,
                               modulated=modulated,
                               use_geom=use_geom,
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
                 shell_sizes: list,
                 radius: float,
                 sigma: float,
                 shared_kp_data = None,
                 modulated: bool = False,
                 mini: bool = False,
                 use_geom: bool = False,
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
            in_channels             (int): dimension input features
            out_channels            (int): dimension input features
            shell_sizes            (list): The number of kernel points per shell.
            radius                (float): convolution radius
            sigma                 (float): influence radius of each kernel point
            shared_kp_data      (None): Optional data dict shared across the layer
            modulated        (bool=False): Use modulations (self-attention)
            mini             (bool=False): Use mini KPConv
            use_geom         (bool=False): Use geometric encodings
            influence_mode (str='linear'): Influence function ('constant', 'linear', 'gaussian')
            aggregation_mode  (str='sum'): Aggregation mode ('nearest', 'sum')
            dimension             (int=3): dimension of input
            groups                (int=1): Number of groups in KPConv
            strided          (bool=False): strided or not
            norm_type       (str='batch'): type of normalization used in layer ('group', 'batch', 'none')
            bn_momentum      (float=0.98): Momentum for batch normalization
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

        # KPConv block with normalizatiom and activation
        if mini:
            self.conv = KPMini(mid_channels,
                                shell_sizes,
                                radius,
                                sigma,
                                shared_kp_data=shared_kp_data,
                                dimension=dimension,
                                influence_mode=influence_mode,
                                norm_type=norm_type,
                                bn_momentum=bn_momentum,
                                activation=activation)
        else:
            self.conv = KPConv(mid_channels,
                                mid_channels,
                                shell_sizes,
                                radius,
                                sigma,
                                shared_kp_data=shared_kp_data,
                                modulated=modulated,
                                use_geom=use_geom,
                                groups=groups,
                                dimension=dimension,
                                influence_mode=influence_mode,
                                aggregation_mode=aggregation_mode)

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


class KPConvInvertedBlock(nn.Module):

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 shell_sizes: list,
                 radius: float,
                 sigma: float,
                 shared_kp_data = None,
                 drop_path: float = 0.,
                 layer_scale_init_value: float = 1e-6,
                 modulated: bool = False,
                 deformable: bool = False,
                 use_geom: bool = False,
                 influence_mode: str = 'linear',
                 aggregation_mode: str = 'sum',
                 dimension: int = 3,
                 strided: bool = False,
                 norm_type: str = 'layer',
                 bn_momentum: float = 0.98,
                 activation: nn.Module = nn.GELU()):
        """
        KPConv inverted block as in ConvNext (and PointNext).
        Args:
            in_channels             (int): dimension input features
            out_channels            (int): dimension input features
            shell_sizes            (list): The number of kernel points per shell.
            radius                (float): convolution radius
            sigma                 (float): influence radius of each kernel point
            shared_kp_data      (None): Optional data dict shared across the layer
            modulated        (bool=False): Use modulations (self-attention)
            deformable       (bool=False): Use deformable KPConv
            use_geom         (bool=False): Use geometric encodings
            influence_mode (str='linear'): Influence function ('constant', 'linear', 'gaussian')
            aggregation_mode  (str='sum'): Aggregation mode ('nearest', 'sum')
            dimension             (int=3): dimension of input
            groups                (int=1): Number of groups in KPConv
            strided          (bool=False): strided or not
            norm_type       (str='batch'): type of normalization used in layer ('group', 'batch', 'none')
            bn_momentum      (float=0.98): Momentum for batch normalization
            activation (nn.Module|None=nn.LeakyReLU(0.1)): Activation function. Use None for no activation.
        """
        super(KPConvInvertedBlock, self).__init__()

        # Define parameters
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.strided = strided
        self.bn_momentum = bn_momentum
        self.norm_type = norm_type
        mid_channels = out_channels * 4

        # learnable parameters
        self.linear1 = nn.Linear(in_channels, mid_channels)
        self.linear2 = nn.Linear(mid_channels, out_channels)

        self.conv = KPConv(in_channels,
                           in_channels,
                           shell_sizes,
                           radius,
                           sigma,
                           shared_kp_data=shared_kp_data,
                           modulated=modulated,
                           use_geom=use_geom,
                           groups=in_channels,
                           dimension=dimension,
                           influence_mode=influence_mode,
                           aggregation_mode=aggregation_mode)
        if in_channels != out_channels:
            self.linear_shortcut =nn.Linear(in_channels, out_channels)
        else:
            self.linear_shortcut = nn.Identity()

        # Other parameters
        self.norm = NormBlock(in_channels, norm_type, bn_momentum)
        self.activation = activation

        # Optimizations
        if layer_scale_init_value > 0:
            self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((out_channels)), requires_grad=True)
        else:
            self.gamma = None
        self.drop_path = DropPathPack(drop_path) if drop_path > 0. else nn.Identity()

        return

    def forward(self, q_pts, s_pts, s_feats, neighbor_indices):

        # First depthwise convolution
        x = self.conv(q_pts, s_pts, s_feats, neighbor_indices)

        # Normalization
        x = self.norm(x)

        # Upscale features
        x = self.linear1(x)

        # Activation
        x = self.activation(x)

        # Downscale features
        x = self.linear2(x)

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

        # Apply shortcut with stochastic depth
        q_feats = shortcut + self.drop_path(x)

        return q_feats

