
from models.blocks import *
import numpy as np


def p2p_fitting_regularizer(net):

    fitting_loss = 0
    repulsive_loss = 0

    for m in net.modules():

        if isinstance(m, KPConv) and m.deformable:

            ##############
            # Fitting loss
            ##############

            # Get the distance to nearest input point and normalize to be independant from layers
            KP_min_d2 = m.min_d2 / (m.KP_extent ** 2)

            # Loss will be the square distance to nearest input point. We use L1 because dist is already squared
            fitting_loss += net.l1(KP_min_d2, torch.zeros_like(KP_min_d2))

            ################
            # Repulsive loss
            ################

            # Normalized KP locations
            KP_locs = m.deformed_KP / m.KP_extent

            # Point should not be close to each other
            for i in range(net.K):
                other_KP = torch.cat([KP_locs[:, :i, :], KP_locs[:, i + 1:, :]], dim=1).detach()
                distances = torch.sqrt(torch.sum((other_KP - KP_locs[:, i:i + 1, :]) ** 2, dim=2))
                rep_loss = torch.sum(torch.clamp_max(distances - net.repulse_extent, max=0.0) ** 2, dim=1)
                repulsive_loss += net.l1(rep_loss, torch.zeros_like(rep_loss)) / net.K

    return net.deform_fitting_power * (2 * fitting_loss + repulsive_loss)


class KPCNN(nn.Module):
    """
    Class defining KPCNN
    """

    def __init__(self, config):
        super(KPCNN, self).__init__()

        #####################
        # Network opperations
        #####################

        # Current radius of convolution and feature dimension
        layer = 0
        r = config.first_subsampling_dl * config.conv_radius
        in_dim = config.in_features_dim
        out_dim = config.first_features_dim
        self.K = config.num_kernel_points

        # Save all block operations in a list of modules
        self.block_ops = nn.ModuleList()

        # Loop over consecutive blocks
        block_in_layer = 0
        for block_i, block in enumerate(config.architecture):

            # Check equivariance
            if ('equivariant' in block) and (not out_dim % 3 == 0):
                raise ValueError('Equivariant block but features dimension is not a factor of 3')

            # Detect upsampling block to stop
            if 'upsample' in block:
                break

            # Apply the good block function defining tf ops
            self.block_ops.append(block_decider(block,
                                                r,
                                                in_dim,
                                                out_dim,
                                                layer,
                                                config))


            # Index of block in this layer
            block_in_layer += 1

            # Update dimension of input from output
            if 'simple' in block:
                in_dim = out_dim // 2
            else:
                in_dim = out_dim


            # Detect change to a subsampled layer
            if 'pool' in block or 'strided' in block:
                # Update radius and feature dimension for next layer
                layer += 1
                r *= 2
                out_dim *= 2
                block_in_layer = 0

        self.head_mlp = UnaryBlock(out_dim, 1024, False, 0)
        self.head_softmax = UnaryBlock(1024, config.num_classes, False, 0, no_relu=True)

        ################
        # Network Losses
        ################

        self.criterion = torch.nn.CrossEntropyLoss()
        self.deform_fitting_mode = config.deform_fitting_mode
        self.deform_fitting_power = config.deform_fitting_power
        self.deform_lr_factor = config.deform_lr_factor
        self.repulse_extent = config.repulse_extent
        self.output_loss = 0
        self.reg_loss = 0
        self.l1 = nn.L1Loss()

        return

    def forward(self, batch, config):

        # Save all block operations in a list of modules
        x = batch.features.clone().detach()

        # Loop over consecutive blocks
        for block_op in self.block_ops:
            x = block_op(x, batch)

        # Head of network
        x = self.head_mlp(x, batch)
        x = self.head_softmax(x, batch)

        return x

    def loss(self, outputs, labels):
        """
        Runs the loss on outputs of the model
        :param outputs: logits
        :param labels: labels
        :return: loss
        """

        # Cross entropy loss
        self.output_loss = self.criterion(outputs, labels)

        # Regularization of deformable offsets
        if self.deform_fitting_mode == 'point2point':
            self.reg_loss = p2p_fitting_regularizer(self)
        elif self.deform_fitting_mode == 'point2plane':
            raise ValueError('point2plane fitting mode not implemented yet.')
        else:
            raise ValueError('Unknown fitting mode: ' + self.deform_fitting_mode)

        # Combined loss
        return self.output_loss + self.reg_loss

    @staticmethod
    def accuracy(outputs, labels):
        """
        Computes accuracy of the current batch
        :param outputs: logits predicted by the network
        :param labels: labels
        :return: accuracy value
        """

        predicted = torch.argmax(outputs.data, dim=1)
        total = labels.size(0)
        correct = (predicted == labels).sum().item()

        return correct / total


class KPFCNN(nn.Module):

    def __init__(self, cfg):
        """
        Class defining KPFCNN, in a more readable way. The number of block at each layer can be chosen.
        For KPConv Paper architecture, use cfg.model.layer_blocks = (2, 1, 1, 1, 1).
        Args:
            cfg (EasyDict): configuration dictionary
        """
        super(KPFCNN, self).__init__()

        ############
        # Parameters
        ############

        # Parameters
        self.subsample_size = cfg.model.init_sub_size
        self.sub_mode = cfg.model.sub_mode
        self.kp_radius = cfg.model.kp_radius
        self.kp_sigma = cfg.model.kp_sigma
        self.neighbor_limits = cfg.model.neighbor_limits
        self.first_radius = self.subsample_size * self.kp_radius
        self.first_sigma = self.subsample_size * self.kp_sigma
        self.layer_blocks = cfg.model.layer_blocks
        self.num_layers = len(self.layer_blocks)

        # Varaibles
        in_C = cfg.model.input_channels
        first_C = cfg.model.init_channels
        C = first_C
        conv_r = self.first_radius
        conv_sig = self.first_sigma

        # Verify the architecture validity
        if self.layer_blocks[0] < 2:
            raise ValueError('First layer must contain at least 2 convolutional layers')
        if np.min(self.layer_blocks) < 1:
            raise ValueError('Each layer must contain at least 1 convolutional layers')
        
        #####################
        # List Encoder blocks
        #####################
        
        # ------ Layers 1 ------

        # Initial convolution 
        self.encoder_1 = nn.ModuleList()
        self.encoder_1.append(self.get_conv_block(in_C, C, conv_r, conv_sig, cfg))
        self.encoder_1.append(self.get_residual_block(C, C * 2, conv_r, conv_sig, cfg))

        # Next blocks
        for _ in range(self.layer_blocks[0] - 2):
            self.encoder_1.append(self.get_residual_block(C * 2, C * 2, conv_r, conv_sig))

        # Pooling block
        self.pooling_1 = self.get_residual_block(C * 2, C * 2, conv_r, conv_sig, cfg, strided=True)


        # ------ Layers [2, 3, 4, 5] ------
        for layer in range(2, self.num_layers + 1):

            # Update features, radius, sigma for this layer
            C *= 2; conv_r *= 2; conv_sig *= 2

            # First block takes features to new dimension.
            encoder_i = nn.ModuleList()
            encoder_i.append(self.get_residual_block(C, C * 2, conv_r, conv_sig, cfg))

            # Next blocks
            for _ in range(self.layer_blocks[layer - 1] - 1):
                encoder_i.append(self.get_residual_block(C * 2, C * 2, conv_r, conv_sig))
            setattr(self, 'encoder_{:d}'.format(layer), encoder_i)

            # Pooling block (not for the last layer)
            if layer < self.num_layers:
                pooling_i = self.get_residual_block(C * 2, C * 2, conv_r, conv_sig, cfg, strided=True)
                setattr(self, 'pooling_{:d}'.format(layer), pooling_i)


        #####################
        # List Decoder blocks
        #####################
        
        # ------ Layers [4, 3, 2, 1] ------
        for layer in range(self.num_layers - 1, 0, -1):
            
            decoder_i = self.get_unary_block(C * 3, C, cfg)
            upsampling_i = NearestUpsampleBlock()
            
            setattr(self, 'decoder_{:d}'.format(layer), decoder_i)
            setattr(self, 'upsampling_{:d}'.format(layer), upsampling_i)

            C = C // 2

        #  ------ Head ------
        
        # New head
        self.head = nn.Sequential(self.get_unary_block(first_C * 2, first_C, cfg),
                                  nn.Linear(first_C, cfg.data.num_classes))
        # Easy KPConv Head
        # self.head = nn.Sequential(nn.Linear(first_C * 2, first_C),
        #                           nn.GroupNorm(8, first_C),
        #                           nn.ReLU(),
        #                           nn.Linear(first_C, cfg.data.num_classes))

        # My old head
        # self.head = nn.Sequential(self.get_unary_block(first_C * 2, first_C, cfg, norm_type='none'),
        #                           nn.Linear(first_C, cfg.data.num_classes))



        # ################
        # # Network Losses
        # ################

        # # List of valid labels (those not ignored in loss)
        # self.valid_labels = np.sort([c for c in lbl_values if c not in ign_lbls])

        # # Choose segmentation loss
        # if len(config.class_w) > 0:
        #     class_w = torch.from_numpy(np.array(config.class_w, dtype=np.float32))
        #     self.criterion = torch.nn.CrossEntropyLoss(weight=class_w, ignore_index=-1)
        # else:
        #     self.criterion = torch.nn.CrossEntropyLoss(ignore_index=-1)
        # self.deform_fitting_mode = config.deform_fitting_mode
        # self.deform_fitting_power = config.deform_fitting_power
        # self.deform_lr_factor = config.deform_lr_factor
        # self.repulse_extent = config.repulse_extent
        # self.output_loss = 0
        # self.reg_loss = 0
        # self.l1 = nn.L1Loss()

        return

    def get_unary_block(self, in_C, out_C, cfg, norm_type=None):

        if norm_type is None:
            norm_type = cfg.model.norm

        return UnaryBlock(in_C,
                           out_C,
                           norm_type=norm_type,
                           bn_momentum=cfg.model.bn_momentum)

    def get_conv_block(self, in_C, out_C, radius, sigma, cfg):

        return KPConvBlock(in_C,
                           out_C,
                           cfg.model.kernel_size,
                           radius,
                           sigma,
                           influence_mode=cfg.model.kp_influence,
                           aggregation_mode=cfg.model.kp_aggregation,
                           dimension=cfg.data.dim,
                           norm_type=cfg.model.norm,
                           bn_momentum=cfg.model.bn_momentum)


    def get_residual_block(self, in_C, out_C, radius, sigma, cfg, strided=False):

        return KPResidualBlock(in_C,
                               out_C,
                               cfg.model.kernel_size,
                               radius,
                               sigma,
                               influence_mode=cfg.model.kp_influence,
                               aggregation_mode=cfg.model.kp_aggregation,
                               dimension=cfg.data.dim,
                               groups=cfg.model.conv_groups,
                               strided=strided,
                               norm_type=cfg.model.norm,
                               bn_momentum=cfg.model.bn_momentum)

    def forward(self, batch, cfg):

        #  ------ Init ------

        # First prepare the pyramid graph structure
        pyramid = build_graph_pyramid(batch.points,
                                      batch.lengths,
                                      self.num_layers,
                                      self.subsample_size,
                                      self.first_radius,
                                      self.neighbor_limits,
                                      sub_mode=self.sub_mode)

        # Get input features
        feats = batch.features.clone().detach()


        #  ------ Encoder ------

        skip_feats = []
        for layer in range(1, self.num_layers + 1):

            # Get layer blocks
            l = layer -1
            block_list = getattr(self, 'encoder_{:d}'.format(layer))

            # Layer blocks
            for block in block_list:
                feats = block(pyramid.points[l], pyramid.points[l], feats, pyramid.neighbors[l])
            
            if layer < self.num_layers:

                # Skip features
                skip_feats.append(feats)

                # Pooling
                layer_pool = getattr(self, 'pooling_{:d}'.format(layer))
                feats = layer_pool(pyramid.points[l+1], pyramid.points[l], feats, pyramid.pools[l])


        #  ------ Decoder ------

        for layer in range(self.num_layers - 1, 0, -1):

            # Get layer blocks
            l = layer -1    # 3, 2, 1, 0
            unary = getattr(self, 'decoder_{:d}'.format(layer))

            # Upsample
            feats = local_nearest_pool(feats, pyramid.upsamples[l])

            # Concat with skip features
            feats = torch.cat([feats, skip_feats[l]], dim=1)

            # MLP
            feats = unary(feats)


        #  ------ Head ------

        logits = self.head(feats)

        return logits

    def loss(self, outputs, labels):
        """
        Runs the loss on outputs of the model
        :param outputs: logits
        :param labels: labels
        :return: loss
        """

        # Set all ignored labels to -1 and correct the other label to be in [0, C-1] range
        target = - torch.ones_like(labels)
        for i, c in enumerate(self.valid_labels):
            target[labels == c] = i

        # Reshape to have a minibatch size of 1
        outputs = torch.transpose(outputs, 0, 1)
        outputs = outputs.unsqueeze(0)
        target = target.unsqueeze(0)

        # Cross entropy loss
        self.output_loss = self.criterion(outputs, target)

        # Regularization of deformable offsets
        if self.deform_fitting_mode == 'point2point':
            self.reg_loss = p2p_fitting_regularizer(self)
        elif self.deform_fitting_mode == 'point2plane':
            raise ValueError('point2plane fitting mode not implemented yet.')
        else:
            raise ValueError('Unknown fitting mode: ' + self.deform_fitting_mode)

        # Combined loss
        return self.output_loss + self.reg_loss

    def accuracy(self, outputs, labels):
        """
        Computes accuracy of the current batch
        :param outputs: logits predicted by the network
        :param labels: labels
        :return: accuracy value
        """

        # Set all ignored labels to -1 and correct the other label to be in [0, C-1] range
        target = - torch.ones_like(labels)
        for i, c in enumerate(self.valid_labels):
            target[labels == c] = i

        predicted = torch.argmax(outputs.data, dim=1)
        total = target.size(0)
        correct = (predicted == target).sum().item()

        return correct / total





















