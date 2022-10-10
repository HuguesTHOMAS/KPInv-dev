
import time
import torch
import torch.nn as nn
import numpy as np

from models.generic_blocks import LinearUpsampleBlock, UnaryBlock, local_nearest_pool, GlobalAverageBlock, SmoothCrossEntropyLoss
from models.kpconv_blocks import KPConvBlock, KPConvResidualBlock, KPConvInvertedBlock
from models.kpnext_blocks import KPNextResidualBlock, KPNextInvertedBlock, KPNextMultiShortcutBlock, KPNextBlock

from utils.torch_pyramid import fill_pyramid




class KPCNN_old(nn.Module):

    def __init__(self, cfg):
        """
        Class defining KPFCNN, in a more readable way. The number of block at each layer can be chosen.
        For KPConv Paper architecture, use cfg.model.layer_blocks = (2, 1, 1, 1, 1).
        Args:
            cfg (EasyDict): configuration dictionary
        """
        super(KPCNN_old, self).__init__()

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

        # Variables
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

    def get_residual_block(self, in_C, out_C, radius, sigma, cfg, strided=False, shared_kp_data=None, conv_layer=False):

        attention_groups = cfg.model.inv_groups
        if conv_layer or 'kpconvd' in self.kp_mode:
            attention_groups = 0

        inverted_block = False
        if inverted_block:
            return KPNextInvertedBlock(in_C,
                                       out_C,
                                       cfg.model.shell_sizes,
                                       radius,
                                       sigma,
                                       attention_groups=attention_groups,
                                       attention_act=cfg.model.inv_act,
                                       mod_grp_norm=cfg.model.inv_grp_norm,
                                       shared_kp_data=shared_kp_data,
                                       influence_mode=cfg.model.kp_influence,
                                       dimension=cfg.data.dim,
                                       strided=strided,
                                       norm_type=cfg.model.norm,
                                       bn_momentum=cfg.model.bn_momentum)
        else:
            return KPNextResidualBlock(in_C,
                                       out_C,
                                       cfg.model.shell_sizes,
                                       radius,
                                       sigma,
                                       attention_groups=attention_groups,
                                       attention_act=cfg.model.inv_act,
                                       mod_grp_norm=cfg.model.inv_grp_norm,
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
                         self.radius_scaling,
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


class KPNeXt(nn.Module):

    def __init__(self, cfg):
        """
        Class defining KPNeXt, a modern architecture inspired from ConvNext.
        Standard drop_path_rate: 0
        Standard layer_scale_init_value: 1e-6
        Standard head_init_scale: 1

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
        self.radius_scaling = cfg.model.radius_scaling
        self.first_radius = radius0 * self.kp_radius
        self.first_sigma = radius0 * self.kp_sigma
        self.layer_blocks = cfg.model.layer_blocks
        self.num_layers = len(self.layer_blocks)
        self.upsample_n = cfg.model.upsample_n
        self.share_kp = cfg.model.share_kp
        self.kp_mode = cfg.model.kp_mode
        self.task = cfg.data.task
        
        # List of valid labels (those not ignored in loss)
        self.valid_labels = np.sort([c for c in cfg.data.label_values if c not in cfg.data.ignored_labels])
        self.num_logits = len(self.valid_labels)

        # Variables
        in_C = cfg.model.input_channels
        first_C = cfg.model.init_channels
        conv_r = self.first_radius
        conv_sig = self.first_sigma
        channel_scaling = 2
        if 'channel_scaling' in cfg.model:
            channel_scaling = cfg.model.channel_scaling

        # Get channels at each layer
        layer_C = []
        for l in range(self.num_layers):
            target_C = first_C * channel_scaling ** l     # Scale channels
            layer_C.append(int(np.ceil(target_C / 16)) * 16)             # Ensure it is divisible by 16 (even the first one)

        # Verify the architecture validity
        if self.layer_blocks[0] < 1:
            raise ValueError('First layer must contain at least 1 convolutional layers')
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

        # Initial convolution or MLP
        C = layer_C[0]
        self.stem = self.get_conv_block(in_C, C, conv_r, conv_sig, cfg)
        # self.stem = self.get_unary_block(in_C, C, cfg)

        # Next blocks
        self.encoder_1 = nn.ModuleList()
        use_conv = cfg.model.first_inv_layer >= 1
        for _ in range(self.layer_blocks[0]):
            self.encoder_1.append(self.get_residual_block(C, C, conv_r, conv_sig, cfg,
                                                          shared_kp_data=self.shared_kp[0],
                                                          conv_layer=use_conv))

        # Pooling block
        self.pooling_1 = self.get_pooling_block(C, layer_C[1], conv_r, conv_sig, cfg,
                                                use_mod=(not use_conv))

        # ------ Layers [2, 3, 4, 5] ------
        for layer in range(2, self.num_layers + 1):
            l = layer - 1

            # Update features, radius, sigma for this layer
            C = layer_C[l]
            conv_r *= self.radius_scaling
            conv_sig *= self.radius_scaling

            # Layer blocks
            use_conv = cfg.model.first_inv_layer >= layer
            encoder_i = nn.ModuleList()
            for _ in range(self.layer_blocks[l]):
                encoder_i.append(self.get_residual_block(C, C, conv_r, conv_sig, cfg,
                                                         shared_kp_data=self.shared_kp[l],
                                                         conv_layer=use_conv))
            setattr(self, 'encoder_{:d}'.format(layer), encoder_i)

            # Pooling block (not for the last layer)
            if layer < self.num_layers:
                pooling_i = self.get_pooling_block(C, layer_C[l+1], conv_r, conv_sig, cfg,
                                                   use_mod=(not use_conv))

                setattr(self, 'pooling_{:d}'.format(layer), pooling_i)

        #####################
        # List Decoder blocks
        #####################

        if cfg.data.task == 'classification':

            #  ------ Head ------

            # Global pooling
            self.global_pooling = GlobalAverageBlock()

            # New head
            self.head = nn.Sequential(self.get_unary_block(layer_C[-1], 256, cfg, norm_type='none'),
                                      nn.Dropout(0.5),
                                      nn.Linear(256, self.num_logits))

            # # Old head
            # self.head = nn.Sequential(self.get_unary_block(layer_C[-1], layer_C[-1], cfg),
            #                           nn.Linear(layer_C[-1], self.num_logits))

        elif cfg.data.task == 'cloud_segmentation':

            # ------ Layers [4, 3, 2, 1] ------
            for layer in range(self.num_layers - 1, 0, -1):

                C = layer_C[layer - 1]
                C1 = layer_C[layer]
                decoder_i = self.get_unary_block(C + C1, C, cfg)
                upsampling_i = LinearUpsampleBlock(self.upsample_n)

                setattr(self, 'decoder_{:d}'.format(layer), decoder_i)
                setattr(self, 'upsampling_{:d}'.format(layer), upsampling_i)

            #  ------ Head ------
            
            # New head
            self.head = nn.Sequential(self.get_unary_block(layer_C[0], layer_C[0], cfg),
                                    nn.Linear(layer_C[0], self.num_logits))
            # Easy KPConv Head
            # self.head = nn.Sequential(nn.Linear(layer_C[0] * 2, layer_C[0]),
            #                           nn.GroupNorm(8, layer_C[0]),
            #                           nn.ReLU(),
            #                           nn.Linear(layer_C[0], self.num_logits))

            # My old head
            # self.head = nn.Sequential(self.get_unary_block(layer_C[0] * 2, layer_C[0], cfg, norm_type='none'),
            #                           nn.Linear(layer_C[0], self.num_logits))



        ################
        # Network Losses
        ################

        # Choose between normal cross entropy and smoothed labels
        if cfg.train.smooth_labels:
            CrossEntropy = SmoothCrossEntropyLoss
        else:
            CrossEntropy = torch.nn.CrossEntropyLoss

        if cfg.data.task == 'classification':
            self.criterion = CrossEntropy()
            
        elif cfg.data.task == 'cloud_segmentation':
            if len(cfg.train.class_w) > 0:
                class_w = torch.from_numpy(np.array(cfg.train.class_w, dtype=np.float32))
                self.criterion = CrossEntropy(weight=class_w, ignore_index=-1)
            else:
                self.criterion = CrossEntropy(ignore_index=-1)

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

    def get_conv_block(self, in_C, out_C, radius, sigma, cfg):

        # First layer is the most simple convolution possible
        return KPConvBlock(in_C,
                           out_C,
                           cfg.model.shell_sizes,
                           radius,
                           sigma,
                           influence_mode=cfg.model.kp_influence,
                           aggregation_mode=cfg.model.kp_aggregation,
                           dimension=cfg.data.dim,
                           norm_type=cfg.model.norm,
                           bn_momentum=cfg.model.bn_momentum)

    def get_pooling_block(self, in_C, out_C, radius, sigma, cfg, use_mod=False):

        # Depthwise conv 
        if cfg.model.use_strided_conv:
            return KPConvBlock(in_C,
                            out_C,
                            cfg.model.shell_sizes,
                            radius,
                            sigma,
                            influence_mode=cfg.model.kp_influence,
                            aggregation_mode=cfg.model.kp_aggregation,
                            dimension=cfg.data.dim,
                            norm_type=cfg.model.norm,
                            bn_momentum=cfg.model.bn_momentum)

        else:
            attention_groups = cfg.model.inv_groups
            if 'kpconvd' in self.kp_mode or not use_mod:
                attention_groups = 0
            return KPNextBlock(in_C,
                            out_C,
                            cfg.model.shell_sizes,
                            radius,
                            sigma,
                            attention_groups=attention_groups,
                            attention_act=cfg.model.inv_act,
                            mod_grp_norm=cfg.model.inv_grp_norm,
                            influence_mode=cfg.model.kp_influence,
                            dimension=cfg.data.dim,
                            norm_type=cfg.model.norm,
                            bn_momentum=cfg.model.bn_momentum)
                           
    def get_residual_block(self, in_C, out_C, radius, sigma, cfg, shared_kp_data=None, conv_layer=False):

        attention_groups = cfg.model.inv_groups
        if conv_layer or 'kpconvd' in self.kp_mode:
            attention_groups = 0

        return KPNextMultiShortcutBlock(in_C,
                                        out_C,
                                        cfg.model.shell_sizes,
                                        radius,
                                        sigma,
                                        attention_groups=attention_groups,
                                        attention_act=cfg.model.inv_act,
                                        mod_grp_norm=cfg.model.inv_grp_norm,
                                        expansion=4,
                                        drop_path_p=-1.,
                                        layer_scale_init_v=-1.,
                                        use_upcut=False,
                                        shared_kp_data=shared_kp_data,
                                        influence_mode=cfg.model.kp_influence,
                                        dimension=cfg.data.dim,
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
                         self.radius_scaling,
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

        
        #  ------ Stem ------
        feats = self.stem(batch.in_dict.points[0], batch.in_dict.points[0], feats, batch.in_dict.neighbors[0])
        # feats = self.stem(feats)


        #  ------ Encoder ------

        skip_feats = []
        for layer in range(1, self.num_layers + 1):

            # Get layer blocks
            l = layer -1
            block_list = getattr(self, 'encoder_{:d}'.format(layer))

            # Layer blocks
            upcut = None
            for block in block_list:
                feats, upcut = block(batch.in_dict.points[l], batch.in_dict.points[l], feats, batch.in_dict.neighbors[l], upcut=upcut)
            
            if layer < self.num_layers:

                # Skip features
                skip_feats.append(feats)

                # Pooling
                layer_pool = getattr(self, 'pooling_{:d}'.format(layer))
                feats = layer_pool(batch.in_dict.points[l+1], batch.in_dict.points[l], feats, batch.in_dict.pools[l])

         
        if verbose:    
            torch.cuda.synchronize(batch.device())                         
            t += [time.time()]

        if self.task == 'classification':
            
            # Global pooling
            feats = self.global_pooling(feats, batch.in_dict.lengths[-1])

            
        elif self.task == 'cloud_segmentation':

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

        # Reshape to have size [1, C, N]
        outputs = torch.transpose(outputs, 0, 1)
        outputs = outputs.unsqueeze(0)
        target = target.squeeze().unsqueeze(0)

        # Cross entropy loss
        self.output_loss = self.criterion(outputs, target)

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














