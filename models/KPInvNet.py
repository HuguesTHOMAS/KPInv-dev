
import time
import torch
import torch.nn as nn
import numpy as np

from models.generic_blocks import LinearUpsampleBlock, UnaryBlock, local_nearest_pool
from models.kpinv_blocks import KPInvResidualBlock, KPInvXBottleNeckBlock
from models.kpconv_blocks import KPConvBlock, KPConvResidualBlock
from models.kptran_blocks import KPTransformerResidualBlock


from utils.torch_pyramid import fill_pyramid


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


class KPInvFCNN(nn.Module):

    def __init__(self, cfg):
        """
        Class defining KPFCNN, in a more readable way. The number of block at each layer can be chosen.
        For KPConv Paper architecture, use cfg.model.layer_blocks = (2, 1, 1, 1, 1).
        Args:
            cfg (EasyDict): configuration dictionary
        """
        super(KPInvFCNN, self).__init__()

        ############
        # Parameters
        ############

        # Parameters
        self.subsample_size = cfg.model.in_sub_size
        if self.subsample_size < 0:
            self.subsample_size = cfg.data.init_sub_size
        self.in_sub_mode = cfg.model.in_sub_mode
        self.kp_radius = cfg.model.kp_radius
        self.kp_sigma = cfg.model.kp_sigma
        self.neighbor_limits = cfg.model.neighbor_limits
        if cfg.model.in_sub_size > cfg.data.init_sub_size * 1.01:
            radius0 = cfg.model.in_sub_size * cfg.model.kp_radius
        else:
            radius0 = cfg.data.init_sub_size * cfg.model.kp_radius
        self.first_radius = radius0 * self.kp_radius
        self.first_sigma = radius0 * self.kp_sigma
        self.layer_blocks = cfg.model.layer_blocks
        self.num_layers = len(self.layer_blocks)
        self.upsample_n = cfg.model.upsample_n
        self.share_kp = cfg.model.share_kp
        self.kp_mode = cfg.model.kp_mode

        # List of valid labels (those not ignored in loss)
        self.valid_labels = np.sort([c for c in cfg.data.label_values if c not in cfg.data.ignored_labels])
        self.num_logits = len(self.valid_labels)

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
        if cfg.model.share_kp:
            self.shared_kp = [{} for _ in range(self.num_layers)]
        else:
            self.shared_kp = [None for _ in range(self.num_layers)]

        # Initial convolution
        use_conv = cfg.model.first_inv_layer >= 1
        self.encoder_1 = nn.ModuleList()
        self.encoder_1.append(self.get_conv_block(in_C, C, conv_r, conv_sig, cfg))
        self.encoder_1.append(self.get_residual_block(C, C * 2, conv_r, conv_sig, cfg,
                                                      shared_kp_data=self.shared_kp[0],
                                                      conv_layer=use_conv))

        # Next blocks
        for _ in range(self.layer_blocks[0] - 2):
            self.encoder_1.append(self.get_residual_block(C * 2, C * 2, conv_r, conv_sig, cfg,
                                                          shared_kp_data=self.shared_kp[0],
                                                          conv_layer=use_conv))

        # Pooling block
        self.pooling_1 = self.get_residual_block(C * 2, C * 2, conv_r, conv_sig, cfg,
                                                 strided=True,
                                                 conv_layer=cfg.model.use_strided_conv or use_conv)

        # ------ Layers [2, 3, 4, 5] ------
        for layer in range(2, self.num_layers + 1):

            # Update features, radius, sigma for this layer
            C *= 2
            conv_r *= 2
            conv_sig *= 2

            # First block takes features to new dimension.
            use_conv = cfg.model.first_inv_layer >= layer
            encoder_i = nn.ModuleList()
            encoder_i.append(self.get_residual_block(C, C * 2, conv_r, conv_sig, cfg,
                                                     shared_kp_data=self.shared_kp[layer - 1],
                                                     conv_layer=use_conv))

            # Next blocks
            for _ in range(self.layer_blocks[layer - 1] - 1):
                encoder_i.append(self.get_residual_block(C * 2, C * 2, conv_r, conv_sig, cfg,
                                                         shared_kp_data=self.shared_kp[layer - 1],
                                                         conv_layer=use_conv))
            setattr(self, 'encoder_{:d}'.format(layer), encoder_i)

            # Pooling block (not for the last layer)
            if layer < self.num_layers:
                pooling_i = self.get_residual_block(C * 2, C * 2, conv_r, conv_sig, cfg,
                                                    strided=True,
                                                    conv_layer=cfg.model.use_strided_conv or use_conv)
                setattr(self, 'pooling_{:d}'.format(layer), pooling_i)

        #####################
        # List Decoder blocks
        #####################

        # ------ Layers [4, 3, 2, 1] ------
        for layer in range(self.num_layers - 1, 0, -1):

            decoder_i = self.get_unary_block(C * 3, C, cfg)
            upsampling_i = LinearUpsampleBlock(self.upsample_n)

            setattr(self, 'decoder_{:d}'.format(layer), decoder_i)
            setattr(self, 'upsampling_{:d}'.format(layer), upsampling_i)

            C = C // 2

        #  ------ Head ------

        # New head
        self.head = nn.Sequential(self.get_unary_block(first_C * 2, first_C, cfg),
                                  nn.Linear(first_C, self.num_logits))
        # Easy KPConv Head
        # self.head = nn.Sequential(nn.Linear(first_C * 2, first_C),
        #                           nn.GroupNorm(8, first_C),
        #                           nn.ReLU(),
        #                           nn.Linear(first_C, self.num_logits))

        # My old head
        # self.head = nn.Sequential(self.get_unary_block(first_C * 2, first_C, cfg, norm_type='none'),
        #                           nn.Linear(first_C, self.num_logits))

        ################
        # Network Losses
        ################

        # Choose segmentation loss
        if len(cfg.train.class_w) > 0:
            class_w = torch.from_numpy(np.array(cfg.train.class_w, dtype=np.float32))
            self.criterion = torch.nn.CrossEntropyLoss(weight=class_w, ignore_index=-1)
        else:
            self.criterion = torch.nn.CrossEntropyLoss(ignore_index=-1)

        # self.deform_fitting_mode = config.deform_fitting_mode
        # self.deform_fitting_power = config.deform_fitting_power
        # self.deform_lr_factor = config.deform_lr_factor
        # self.repulse_extent = config.repulse_extent

        self.output_loss = 0
        self.deform_loss = 0
        self.l1 = nn.L1Loss()

        return

    def get_unary_block(self, in_C, out_C, cfg, norm_type=None):

        if norm_type is None:
            norm_type = cfg.model.norm

        return UnaryBlock(in_C,
                          out_C,
                          norm_type=norm_type,
                          bn_momentum=cfg.model.bn_momentum)

    def get_conv_block(self, in_C, out_C, radius, sigma, cfg, shared_kp_data=None):

        return KPConvBlock(in_C,
                           out_C,
                           cfg.model.shell_sizes,
                           radius,
                           sigma,
                           shared_kp_data=shared_kp_data,
                           influence_mode=cfg.model.kp_influence,
                           aggregation_mode=cfg.model.kp_aggregation,
                           dimension=cfg.data.dim,
                           norm_type=cfg.model.norm,
                           bn_momentum=cfg.model.bn_momentum)

    def get_conv_residual_block(self, in_C, out_C, radius, sigma, cfg, strided=False, shared_kp_data=None):

        if 'kpinvx' in self.kp_mode:
            minix_C = cfg.model.kpx_expansion
        else:
            minix_C = 0

        return KPConvResidualBlock(in_C,
                                   out_C,
                                   cfg.model.shell_sizes,
                                   radius,
                                   sigma,
                                   shared_kp_data=shared_kp_data,
                                   minix_C=minix_C,
                                   influence_mode=cfg.model.kp_influence,
                                   aggregation_mode=cfg.model.kp_aggregation,
                                   dimension=cfg.data.dim,
                                   strided=strided,
                                   norm_type=cfg.model.norm,
                                   bn_momentum=cfg.model.bn_momentum)

    def get_residual_block(self, in_C, out_C, radius, sigma, cfg, strided=False, shared_kp_data=None, conv_layer=False):

        if conv_layer:
            return self.get_conv_residual_block(in_C, out_C, radius, sigma, cfg,
                                                strided=strided,
                                                shared_kp_data=shared_kp_data)

        # 'none', 'sigmoid', 'softmax', 'tanh' or 'tanh2'.
        weight_act = 'tanh'

        # Warning when testing be sure this is the same as when test
        if 'kptran' in self.kp_mode:

            return KPTransformerResidualBlock(in_C,
                                              out_C,
                                              cfg.model.shell_sizes,
                                              radius,
                                              sigma,
                                              attention_groups=cfg.model.inv_groups,
                                              attention_act='softmax',
                                              shared_kp_data=shared_kp_data,
                                              influence_mode=cfg.model.kp_influence,
                                              dimension=cfg.data.dim,
                                              strided=strided,
                                              norm_type=cfg.model.norm,
                                              bn_momentum=cfg.model.bn_momentum)

        elif 'kpinvx' in self.kp_mode:

            return KPInvXBottleNeckBlock(in_C,
                                         out_C,
                                         cfg.model.shell_sizes,
                                         radius,
                                         sigma,
                                         expansion=cfg.model.kpx_expansion,
                                         reduction_ratio=cfg.model.kpinv_reduc,
                                         weight_act=weight_act,
                                         shared_kp_data=shared_kp_data,
                                         influence_mode=cfg.model.kp_influence,
                                         dimension=cfg.data.dim,
                                         strided=strided,
                                         norm_type=cfg.model.norm,
                                         bn_momentum=cfg.model.bn_momentum)

        else:

            return KPInvResidualBlock(in_C,
                                      out_C,
                                      cfg.model.shell_sizes,
                                      radius,
                                      sigma,
                                      groups=cfg.model.inv_groups,
                                      reduction_ratio=cfg.model.kpinv_reduc,
                                      weight_act=weight_act,
                                      shared_kp_data=shared_kp_data,
                                      influence_mode=cfg.model.kp_influence,
                                      dimension=cfg.data.dim,
                                      strided=strided,
                                      norm_type=cfg.model.norm,
                                      bn_momentum=cfg.model.bn_momentum)

    def forward(self, batch, verbose=False):

        #  ------ Init ------

        if verbose:
            torch.cuda.synchronize(batch.device())
            t = [time.time()]

        # First complete the input pyramid if not already done
        if len(batch.in_dict.neighbors) < 1:
            fill_pyramid(batch.in_dict,
                         self.num_layers,
                         self.subsample_size,
                         self.first_radius,
                         self.neighbor_limits,
                         self.upsample_n,
                         sub_mode=self.in_sub_mode)

        if verbose:
            torch.cuda.synchronize(batch.device())
            t += [time.time()]

        # Get input features
        feats = batch.in_dict.features.clone().detach()

        if verbose:
            torch.cuda.synchronize(batch.device())
            t += [time.time()]

        #  ------ Encoder ------

        skip_feats = []
        for layer in range(1, self.num_layers + 1):

            # Get layer blocks
            l = layer - 1
            block_list = getattr(self, 'encoder_{:d}'.format(layer))

            # Layer blocks
            for block in block_list:
                feats = block(batch.in_dict.points[l], batch.in_dict.points[l], feats, batch.in_dict.neighbors[l])

            if layer < self.num_layers:

                # Skip features
                skip_feats.append(feats)

                # Pooling
                layer_pool = getattr(self, 'pooling_{:d}'.format(layer))
                feats = layer_pool(batch.in_dict.points[l + 1], batch.in_dict.points[l], feats, batch.in_dict.pools[l])

        # Remove shared data
        if self.share_kp:
            for l in range(self.num_layers):
                self.shared_kp[l].pop('infl_w', None)
                self.shared_kp[l].pop('neighb_p', None)
                self.shared_kp[l].pop('neighb_1nn', None)

        if verbose:
            torch.cuda.synchronize(batch.device())
            t += [time.time()]

        #  ------ Decoder ------

        for layer in range(self.num_layers - 1, 0, -1):

            # Get layer blocks
            l = layer - 1    # 3, 2, 1, 0
            unary = getattr(self, 'decoder_{:d}'.format(layer))
            upsample = getattr(self, 'upsampling_{:d}'.format(layer))

            # Upsample
            feats = upsample(feats, batch.in_dict.upsamples[l], batch.in_dict.up_distances[l])

            # Concat with skip features
            feats = torch.cat([feats, skip_feats[l]], dim=1)

            # MLP
            feats = unary(feats)

        #  ------ Head ------

        logits = self.head(feats)

        if verbose:
            torch.cuda.synchronize(batch.device())
            t += [time.time()]
            mean_dt = 1000 * (np.array(t[1:]) - np.array(t[:-1]))
            message = ' ' * 75 + 'net (ms):'
            for dt in mean_dt:
                message += ' {:5.1f}'.format(dt)
            print(message)

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
        target = target.squeeze().unsqueeze(0)

        # Cross entropy loss
        self.output_loss = self.criterion(outputs, target)

        # # Regularization of deformable offsets
        # if self.deform_fitting_mode == 'point2point':
        #     self.reg_loss = p2p_fitting_regularizer(self)
        # elif self.deform_fitting_mode == 'point2plane':
        #     raise ValueError('point2plane fitting mode not implemented yet.')
        # else:
        #     raise ValueError('Unknown fitting mode: ' + self.deform_fitting_mode)

        self.deform_loss = 0

        # Combined loss
        return self.output_loss

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


class KPResNet(nn.Module):

    def __init__(self, cfg):
        """
        Class defining KPResNet similar to resnet-xxx architectures.

        stem (first two layers):
        ************************
        
                Resnet                    LR-ResNet                   Involution
        7x7conv - 64 - stride2           1x1mlp - 64            3x3conv - 32 - stride2
        3x3maxp - 64 - stride2      7x7conv - 64 - stride2            7x7inv - 32 
                                       3x3maxp - stride2              3x3conv - 64
                                                                3x3maxp - 64 - stride2

        following:
        **********

        D = 64, 128, 256, 512

            Resnet                LR-ResNet              Involution
         1x1 mlp - D             1x1 mlp - D             1x1 mlp - D
         3x3 conv - D            7x7 conv - D            7x7 inv - D
        1x1 mlp - D*4           1x1 mlp - D*4           1x1 mlp - D*4

        Depths:
        *******

        Resnet-26:  (Bottleneck, (1,  2,  4,  1)),
        Resnet-38:  (Bottleneck, (2,  3,  5,  2)),
        Resnet-50:  (Bottleneck, (3,  4,  6,  3)),
        Resnet-101: (Bottleneck, (3,  4, 23,  3)),
        Resnet-152: (Bottleneck, (3,  8, 36,  3))




        Args:
            cfg (EasyDict): configuration dictionary
        """
        super(KPFCNN, self).__init__()

        ############
        # Parameters
        ############

        # Parameters
        self.subsample_size = cfg.model.in_sub_size
        self.in_sub_mode = cfg.model.in_sub_mode
        self.kp_radius = cfg.model.kp_radius
        self.kp_sigma = cfg.model.kp_sigma
        self.neighbor_limits = cfg.model.neighbor_limits
        self.first_radius = self.subsample_size * self.kp_radius
        self.first_sigma = self.subsample_size * self.kp_sigma
        self.layer_blocks = cfg.model.layer_blocks
        self.num_layers = len(self.layer_blocks)

        return

    def forward(self, batch, verbose=False):

        return
