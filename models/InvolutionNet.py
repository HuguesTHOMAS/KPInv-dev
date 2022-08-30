
import time
import torch
import torch.nn as nn
import numpy as np

from models.generic_blocks import LinearUpsampleBlock, UnaryBlock, local_nearest_pool
from models.kpconv_blocks import KPConvBlock, KPConvResidualBlock
from models.pi_blocks import InvolutionResidualBlock

from utils.torch_pyramid import fill_pyramid


class InvolutionFCNN(nn.Module):

    def __init__(self, cfg):
        """
        Class defining InvolutionFCNN, in a more readable way. The number of block at each layer can be chosen.
        Args:
            cfg (EasyDict): configuration dictionary
        """
        super(InvolutionFCNN, self).__init__()

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
        use_conv = cfg.model.first_inv_layer > 0
        self.encoder_1 = nn.ModuleList()
        self.encoder_1.append(self.get_conv_block(in_C, C, conv_r, conv_sig, cfg))
        self.encoder_1.append(self.get_residual_block(C, C * 2, 0, conv_r, conv_sig, cfg, conv_layer=use_conv))

        
        # Next blocks
        for _ in range(self.layer_blocks[0] - 2):
            self.encoder_1.append(self.get_residual_block(C * 2, C * 2, 0, conv_r, conv_sig, cfg, conv_layer=use_conv))

        # Pooling block
        use_conv = cfg.model.use_strided_conv or use_conv
        self.pooling_1 = self.get_residual_block(C * 2, C * 2, 0, conv_r, conv_sig, cfg, strided=True, conv_layer=use_conv)


        # ------ Layers [2, 3, 4, 5] ------
        for layer in range(2, self.num_layers + 1):

            # Update features, radius, sigma for this layer
            C *= 2; conv_r *= 2; conv_sig *= 2

            # First block takes features to new dimension.
            use_conv = cfg.model.first_inv_layer > layer - 1
            encoder_i = nn.ModuleList()
            encoder_i.append(self.get_residual_block(C, C * 2, layer - 1, conv_r, conv_sig, cfg, conv_layer=use_conv))

            # Next blocks
            for _ in range(self.layer_blocks[layer - 1] - 1):
                encoder_i.append(self.get_residual_block(C * 2, C * 2, layer - 1, conv_r, conv_sig, cfg, conv_layer=use_conv))
            setattr(self, 'encoder_{:d}'.format(layer), encoder_i)

            # Pooling block (not for the last layer)
            if layer < self.num_layers:
                use_conv = cfg.model.use_strided_conv or use_conv
                pooling_i = self.get_residual_block(C * 2, C * 2, layer - 1, conv_r, conv_sig, cfg, strided=True, conv_layer=use_conv)
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

    def get_residual_block(self, in_C, out_C, layer, radius, sigma, cfg, strided=False, conv_layer=False):
        
        if conv_layer:
            return self.get_conv_residual_block(in_C, out_C, radius, sigma, cfg, strided=strided)
        else:
            return InvolutionResidualBlock(in_C,
                                           out_C,
                                           radius,
                                           cfg.model.neighbor_limits[layer],
                                           cfg.model.kp_mode,
                                           groups=cfg.model.inv_groups,
                                           strided=strided,
                                           dimension=cfg.data.dim,
                                           norm_type=cfg.model.norm,
                                           bn_momentum=cfg.model.bn_momentum)

    def get_conv_residual_block(self, in_C, out_C, radius, sigma, cfg, deformable=False, strided=False):

        return KPConvResidualBlock(in_C,
                                   out_C,
                                   cfg.model.kernel_size,
                                   radius,
                                   sigma,
                                   modulated=False,
                                   deformable=deformable,
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














