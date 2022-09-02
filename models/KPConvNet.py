
import time
import torch
import torch.nn as nn
import numpy as np

from models.generic_blocks import LinearUpsampleBlock, UnaryBlock, local_nearest_pool
from models.kpconv_blocks import KPDef, KPConvBlock, KPConvResidualBlock, KPConvInvertedBlock

from utils.torch_pyramid import fill_pyramid


def p2p_fit_rep_loss(net):
    """
    Explore a network parameters to find deformable convolutions and get fitting and repulsives losses for all of them.
    """

    fitting_loss = 0
    repulsive_loss = 0

    for m in net.modules():

        if isinstance(m, KPDef):

            ##############
            # Fitting loss
            ##############

            # Get the distance to nearest input point and normalize to be independant from layers
            KP_min_d2 = m.min_d2 / (m.radius ** 2)

            # Loss will be the square distance to nearest input point. We use L1 because dist is already squared
            fitting_loss += net.l1(KP_min_d2, torch.zeros_like(KP_min_d2))

            ################
            # Repulsive loss
            ################

            # Normalized KP locations
            KP_locs = m.deformed_KP / m.radius

            # Point should not be close to each other
            for i in range(m.kernel_size):
                other_KP = torch.cat([KP_locs[:, :i, :], KP_locs[:, i + 1:, :]], dim=1).detach()
                distances = torch.sqrt(torch.sum((other_KP - KP_locs[:, i:i + 1, :]) ** 2, dim=2))
                rep_loss = torch.sum(torch.clamp_max(distances - m.sigma, max=0.0) ** 2, dim=1)
                repulsive_loss += net.l1(rep_loss, torch.zeros_like(rep_loss)) / m.kernel_size

    return fitting_loss, repulsive_loss


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
        self.deform_loss = 0
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
            self.deform_loss = p2p_fitting_regularizer(self)
        elif self.deform_fitting_mode == 'point2plane':
            raise ValueError('point2plane fitting mode not implemented yet.')
        else:
            raise ValueError('Unknown fitting mode: ' + self.deform_fitting_mode)

        # Combined loss
        return self.output_loss + self.deform_loss

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

    def __init__(self, cfg, modulated=False, deformable=False):
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
        self.deformable = deformable
        self.modulated = modulated
        self.upsample_n = cfg.model.upsample_n
        
        
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

        # Initial convolution 
        self.encoder_1 = nn.ModuleList()
        self.encoder_1.append(self.get_conv_block(in_C, C, conv_r, conv_sig, cfg))
        self.encoder_1.append(self.get_residual_block(C, C * 2, conv_r, conv_sig, cfg))

        # Next blocks
        for _ in range(self.layer_blocks[0] - 2):
            self.encoder_1.append(self.get_residual_block(C * 2, C * 2, conv_r, conv_sig, cfg))

        # Pooling block
        self.pooling_1 = self.get_residual_block(C * 2, C * 2, conv_r, conv_sig, cfg, strided=True)


        # ------ Layers [2, 3, 4, 5] ------
        for layer in range(2, self.num_layers + 1):

            # Update features, radius, sigma for this layer
            C *= 2; conv_r *= 2; conv_sig *= 2

            # First block takes features to new dimension.
            encoder_i = nn.ModuleList()
            encoder_i.append(self.get_residual_block(C, C * 2, conv_r, conv_sig, cfg, deformable=self.deformable))

            # Next blocks
            for _ in range(self.layer_blocks[layer - 1] - 1):
                encoder_i.append(self.get_residual_block(C * 2, C * 2, conv_r, conv_sig, cfg, deformable=self.deformable))
            setattr(self, 'encoder_{:d}'.format(layer), encoder_i)

            # Pooling block (not for the last layer)
            if layer < self.num_layers:
                pooling_i = self.get_residual_block(C * 2, C * 2, conv_r, conv_sig, cfg, deformable=self.deformable, strided=True)
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

        self.deform_loss_factor = cfg.train.deform_loss_factor
        self.fit_rep_ratio = cfg.train.deform_fit_rep_ratio
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

    def get_conv_block(self, in_C, out_C, radius, sigma, cfg, deformable=False):

        # First layer is the most simple convolution possible
        return KPConvBlock(in_C,
                           out_C,
                           cfg.model.kernel_size,
                           radius,
                           sigma,
                           modulated=False,
                           deformable=False,
                           use_geom=False,
                           influence_mode=cfg.model.kp_influence,
                           aggregation_mode=cfg.model.kp_aggregation,
                           dimension=cfg.data.dim,
                           norm_type=cfg.model.norm,
                           bn_momentum=cfg.model.bn_momentum)

    def get_residual_block(self, in_C, out_C, radius, sigma, cfg, deformable=False, strided=False):

        use_geom = 'geom' in cfg.model.kp_mode

        return KPConvResidualBlock(in_C,
                               out_C,
                               cfg.model.kernel_size,
                               radius,
                               sigma,
                               modulated=self.modulated,
                               deformable=deformable,
                               use_geom=use_geom,
                               influence_mode=cfg.model.kp_influence,
                               aggregation_mode=cfg.model.kp_aggregation,
                               dimension=cfg.data.dim,
                               groups=cfg.model.conv_groups,
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
            l = layer -1
            block_list = getattr(self, 'encoder_{:d}'.format(layer))

            # Layer blocks
            for block in block_list:
                feats = block(batch.in_dict.points[l], batch.in_dict.points[l], feats, batch.in_dict.neighbors[l])
            
            if layer < self.num_layers:

                # Skip features
                skip_feats.append(feats)

                # Pooling
                layer_pool = getattr(self, 'pooling_{:d}'.format(layer))
                feats = layer_pool(batch.in_dict.points[l+1], batch.in_dict.points[l], feats, batch.in_dict.pools[l])

         
        if verbose:    
            torch.cuda.synchronize(batch.device())                         
            t += [time.time()]

        #  ------ Decoder ------

        for layer in range(self.num_layers - 1, 0, -1):

            # Get layer blocks
            l = layer -1    # 3, 2, 1, 0
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

        # Regularization of deformable offsets (=0 if no deformable conv in network)
        fitting_loss, repulsive_loss = p2p_fit_rep_loss(self)
        self.deform_loss = self.deform_loss_factor * (self.fit_rep_ratio * fitting_loss + repulsive_loss)

        # Combined loss
        return self.output_loss + self.deform_loss

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




class KPNeXt(nn.Module):

    def __init__(self, cfg, modulated=False, deformable=False):
        """
        Class defining KPNeXt, a modern architecture inspired from ConvNext.
        Standard drop_path_rate: 0
        Standard layer_scale_init_value: 1e-6
        Standard head_init_scale: 1

        depth/features: 
        ConvNeXt-T  [3, 3,  9, 3] / [ 96, 192,  384,  768]
        ConvNeXt-S  [3, 3, 27, 3] / [ 96, 192,  384,  768]
        ConvNeXt-B  [3, 3, 27, 3] / [128, 256,  512, 1024]
        ConvNeXt-L  [3, 3, 27, 3] / [192, 384,  768, 1536]
        ConvNeXt-XL [3, 3, 27, 3] / [256, 512, 1024, 2048]

        Args:
            cfg (EasyDict): configuration dictionary
        """
        super(KPNeXt, self).__init__()

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
        self.deformable = deformable
        self.modulated = modulated
        self.upsample_n = cfg.model.upsample_n
        
        
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

        # Initial convolution 
        self.encoder_1 = nn.ModuleList()
        self.encoder_1.append(self.get_conv_block(in_C, C, conv_r, conv_sig, cfg))

        # Next blocks
        for _ in range(self.layer_blocks[0] - 1):
            self.encoder_1.append(self.get_residual_block(C, C, conv_r, conv_sig, cfg))

        # Pooling block
        self.pooling_1 = self.get_residual_block(C, C * 2, conv_r, conv_sig, cfg, strided=True)


        # ------ Layers [2, 3, 4, 5] ------
        for layer in range(2, self.num_layers + 1):

            # Update features, radius, sigma for this layer
            C *= 2; conv_r *= 2; conv_sig *= 2

            # First block takes features to new dimension.
            encoder_i = nn.ModuleList()

            # Next blocks
            for _ in range(self.layer_blocks[layer - 1] - 1):
                encoder_i.append(self.get_residual_block(C, C, conv_r, conv_sig, cfg, deformable=self.deformable))
            setattr(self, 'encoder_{:d}'.format(layer), encoder_i)

            # Pooling block (not for the last layer)
            if layer < self.num_layers:
                pooling_i = self.get_residual_block(C, C * 2, conv_r, conv_sig, cfg, deformable=self.deformable, strided=True)
                setattr(self, 'pooling_{:d}'.format(layer), pooling_i)


        #####################
        # List Decoder blocks
        #####################
        
        # ------ Layers [4, 3, 2, 1] ------
        for layer in range(self.num_layers - 1, 0, -1):
            
            C = C // 2
            
            decoder_i = self.get_unary_block(C * 3, C, cfg)
            upsampling_i = LinearUpsampleBlock(self.upsample_n)
            
            setattr(self, 'decoder_{:d}'.format(layer), decoder_i)
            setattr(self, 'upsampling_{:d}'.format(layer), upsampling_i)


        #  ------ Head ------
        
        # New head
        self.head = nn.Sequential(self.get_unary_block(first_C, first_C // 2, cfg),
                                  nn.Linear(first_C // 2, self.num_logits))
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

        self.deform_loss_factor = cfg.train.deform_loss_factor
        self.fit_rep_ratio = cfg.train.deform_fit_rep_ratio
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

    def get_conv_block(self, in_C, out_C, radius, sigma, cfg, deformable=False):

        # First layer is the most simple convolution possible
        return KPConvBlock(in_C,
                           out_C,
                           cfg.model.kernel_size,
                           radius,
                           sigma,
                           modulated=False,
                           deformable=False,
                           use_geom=False,
                           influence_mode=cfg.model.kp_influence,
                           aggregation_mode=cfg.model.kp_aggregation,
                           dimension=cfg.data.dim,
                           norm_type=cfg.model.norm,
                           bn_momentum=cfg.model.bn_momentum)

    def get_residual_block(self, in_C, out_C, radius, sigma, cfg, deformable=False, strided=False):

        use_geom = 'geom' in cfg.model.kp_mode

        return KPConvInvertedBlock(in_C,
                                   out_C,
                                   cfg.model.kernel_size,
                                   radius,
                                   sigma,
                                   drop_path=0.,
                                   modulated=self.modulated,
                                   deformable=deformable,
                                   use_geom=use_geom,
                                   influence_mode=cfg.model.kp_influence,
                                   aggregation_mode=cfg.model.kp_aggregation,
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
            l = layer -1
            block_list = getattr(self, 'encoder_{:d}'.format(layer))

            # Layer blocks
            for block in block_list:
                feats = block(batch.in_dict.points[l], batch.in_dict.points[l], feats, batch.in_dict.neighbors[l])
            
            if layer < self.num_layers:

                # Skip features
                skip_feats.append(feats)

                # Pooling
                layer_pool = getattr(self, 'pooling_{:d}'.format(layer))
                feats = layer_pool(batch.in_dict.points[l+1], batch.in_dict.points[l], feats, batch.in_dict.pools[l])

         
        if verbose:    
            torch.cuda.synchronize(batch.device())                         
            t += [time.time()]

        #  ------ Decoder ------

        for layer in range(self.num_layers - 1, 0, -1):

            # Get layer blocks
            l = layer -1    # 3, 2, 1, 0
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

        # Regularization of deformable offsets (=0 if no deformable conv in network)
        fitting_loss, repulsive_loss = p2p_fit_rep_loss(self)
        self.deform_loss = self.deform_loss_factor * (self.fit_rep_ratio * fitting_loss + repulsive_loss)

        # Combined loss
        return self.output_loss + self.deform_loss

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














