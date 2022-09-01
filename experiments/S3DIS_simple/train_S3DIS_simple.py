
# ----------------------------------------------------------------------------------------------------------------------
#
#           Script Intro
#       \******************/
#
#
#   Use this script to train a network on S3DIS using the simple input pipeline 
#   (no neighbors computation in the dataloader)
#
#


# ----------------------------------------------------------------------------------------------------------------------
#
#           Imports and global variables
#       \**********************************/
#

# Common libs
from decimal import MAX_PREC
from operator import mod
import os
import sys
import time
import signal
import argparse
import numpy as np
from torch.utils.data import DataLoader

# Local libs
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(os.path.dirname(current))
sys.path.append(parent)

from utils.config import init_cfg, save_cfg, get_directories
from utils.printing import frame_lines_1, underline

from models.KPConvNet import KPFCNN as KPConvFCNN
from models.KPInvNet import KPFCNN as KPInvFCNN
from models.InvolutionNet import InvolutionFCNN

from datasets.scene_seg import SceneSegSampler, SceneSegCollate

from experiments.S3DIS_simple.S3DIS import S3DIS_cfg, S3DISDataset

from tasks.trainval import train_and_validate


# ----------------------------------------------------------------------------------------------------------------------
#
#           Config Class
#       \******************/
#


def my_config():
    """
    Override the parameters you want to modify for this dataset
    """

    cfg = init_cfg()

    # Network parameters
    # ------------------

    # cfg.model.layer_blocks = (2, 1, 1, 1, 1)    # KPConv paper architecture. Can be changed for a deeper network
    # cfg.model.layer_blocks = (2, 2, 2, 4, 2)
    # cfg.model.layer_blocks = (2, 3, 4, 8, 4)
    # cfg.model.layer_blocks = (2, 3, 4, 16, 4)
    # cfg.model.layer_blocks = (2, 3, 8, 32, 4)

    
    # cfg.model.layer_blocks = (2,  3,  4,  6,  3)    # Same as point transformers
    # cfg.model.layer_blocks = (3,  4,  6,  8,  4)
    # cfg.model.layer_blocks = (4,  6,  8,  8,  6)
    # cfg.model.layer_blocks = (4,  6,  8, 12,  6)  # Strong architecture

    # cfg.model.layer_blocks = (3,  5,  7,  9,  9,  3)  # For large block inputs
    cfg.model.layer_blocks = (3,  5,  7,  9,  7)  # For large block inputs

    cfg.model.kp_mode = 'transformer'       # Choose ['kpconv', 'kpdef', 'kpinv']. And ['kpconv-mod', 'kpdef-mod', 'kpconv-geom'] for modulations
                                            # Choose ['inv_v1', 'inv_v2', 'inv_v3', 'inv_v4', 'transformer']
    cfg.model.kernel_size = 15
    cfg.model.kp_radius = 2.5
    cfg.model.kp_influence = 'linear'
    cfg.model.kp_aggregation = 'sum'
    cfg.model.conv_groups = 1

    cfg.data.init_sub_size = 0.04          # -1.0 so that dataset point clouds are not initially subsampled
    cfg.data.init_sub_mode = 'grid'        # Mode for initial subsampling of data
    cfg.model.in_sub_size = -4             # Adapt this with train.in_radius. Try to keep a ratio of ~50 (*0.67 if fps). If negative, and fps, it is stride
    cfg.model.in_sub_mode = 'fps'          # Mode for input subsampling

    cfg.model.upsample_n = 3          # Number of neighbors used for nearest neighbor linear interpolation

    cfg.model.input_channels = 5    # This value has to be compatible with one of the dataset input features definition

    # cfg.model.neighbor_limits = []                      # Use empty list to let calibration get the values
    # cfg.model.neighbor_limits = [35, 40, 50, 50, 50]    # Use empty list to let calibration get the values
    cfg.model.neighbor_limits = [16, 16, 16, 16, 16]    # List for point_transformer


    # Specific parameters for involution and transformers
    cfg.model.use_strided_conv = True           # Use convolution op for strided layers instead of involution
    cfg.model.first_inv_layer = 1               # Use involution layers only from this layer index
    cfg.model.inv_groups = 1


    # Training parameters
    # -------------------
    
    # Are we using spheres/cubes/cylinders/cubic_cylinders as input
    cfg.data.use_cubes = False
    cfg.data.cylindric_input = False

    # Input threads
    cfg.train.num_workers = 16

    # Input spheres radius. Adapt this with model.in_sub_size. Try to keep a ratio of ~50
    cfg.train.in_radius = 1.8

    # Batch related_parames
    cfg.train.batch_size = 2                # Target batch size. If you don't want calibration, you can directly set train.batch_limit
    cfg.train.accum_batch = 8               # Accumulate batches for an effective batch size of batch_size * accum_batch.
    cfg.train.steps_per_epoch = 250
    
    # Training length
    cfg.train.max_epoch = 180
    
    # Deformations
    cfg.train.deform_loss_factor = 0.1      # Reduce to reduce influence for deformation on overall features
    cfg.train.deform_lr_factor = 1.0        # Higher so that deformation are learned faster (especially if deform_loss_factor is low)

    # Optimizer
    cfg.train.optimizer = 'AdamW'
    cfg.train.adam_b = (0.9, 0.999)
    cfg.train.adam_eps = 1e-08
    cfg.train.weight_decay = 0.01     # for KPConv
    # cfg.train.weight_decay = 0.0001     # for transformer

    # Cyclic lr 
    cfg.train.cyc_lr0 = 5e-4                # Float, Start (minimum) learning rate of 1cycle decay
    cfg.train.cyc_lr1 = 5e-3                # Float, Maximum learning rate of 1cycle decay
    cfg.train.cyc_raise_n = 1               #   Int, Raise rate for first part of 1cycle = number of epoch to multiply lr by 10
    cfg.train.cyc_decrease10 = 80           #   Int, Decrease rate for second part of 1cycle = number of epoch to divide lr by 10
    cfg.train.cyc_plateau = 1               #   Int, Number of epoch for plateau at maximum lr

    # import matplotlib.pyplot as plt
    # fig = plt.figure('lr')
    # y = [init_lr]
    # for i in range(cfg.train.max_epoch):
    #     y.append(y[-1])
    #     if str(i) in cfg.train.lr_decays:
    #         y[-1] *= cfg.train.lr_decays[str(i)]
    # plt.plot(y)
    # plt.xlabel('epochs')
    # plt.ylabel('lr')
    # plt.yscale('log')
    # ax = fig.gca()
    # ax.grid(linestyle='-.', which='both')
    # plt.show()
    # a = 1/0

    # Train Augmentations
    cfg.augment_train.anisotropic = False
    cfg.augment_train.scale = [0.9, 1.1]
    cfg.augment_train.flips = [0.5, 0, 0]
    cfg.augment_train.rotations = 'all'
    cfg.augment_train.jitter = 0.005
    cfg.augment_train.color_drop = 0.3
    cfg.augment_train.chromatic_contrast = True
    cfg.augment_train.chromatic_all = False
    cfg.augment_train.chromatic_norm = True
    cfg.augment_train.height_norm = False

    
    # Test parameters
    # ---------------

    cfg.test.max_steps_per_epoch = 50    # Size of one validation epoch (should be small)
    cfg.test.batch_limit = 1
    cfg.test.batch_size = 1

    cfg.test.val_momentum = 0.95

    # Test Augmentations
    cfg.augment_test.anisotropic = False
    cfg.augment_test.scale = [0.99, 1.01]
    cfg.augment_test.flips = [0.5, 0, 0]
    cfg.augment_test.rotations = 'vertical'
    cfg.augment_test.jitter = 0
    cfg.augment_test.color_drop = 0.0
    cfg.augment_test.chromatic_contrast = False
    cfg.augment_test.chromatic_all = False

    return cfg


def adjust_config(cfg):

    # Model
    cfg.model.kp_sigma = 0.7 * cfg.model.kp_radius

    # Train
    if cfg.data.use_cubes:
        if cfg.data.cylindric_input:
            cfg.train.in_radius *= np.pi**(1/2) / 2  # ratio between square and circle area
        else:
            cfg.train.in_radius *= (4 / 3 * np.pi)**(1/3) / 2  # ratio between cube and sphere volume   
    cfg.train.checkpoint_gap = cfg.train.max_epoch // 5

    # Learning rate
    raise_rate = (cfg.train.cyc_lr1 / cfg.train.cyc_lr0)**(1/cfg.train.cyc_raise_n)
    decrease_rate = 0.1**(1 / cfg.train.cyc_decrease10)
    cfg.train.lr = cfg.train.cyc_lr0
    cfg.train.lr_decays = {str(i): raise_rate for i in range(1, cfg.train.cyc_raise_n + 1)}
    for i in range(cfg.train.cyc_raise_n + 1 + cfg.train.cyc_plateau, cfg.train.max_epoch):
        cfg.train.lr_decays[str(i)] = decrease_rate

    # Test
    cfg.test.in_radius = cfg.train.in_radius * 2
    cfg.augment_test.chromatic_norm = cfg.augment_train.chromatic_norm
    cfg.augment_test.height_norm = cfg.augment_train.height_norm
    cfg.test.num_workers = cfg.train.num_workers
    cfg.test.max_points = cfg.train.max_points


    return cfg


# ----------------------------------------------------------------------------------------------------------------------
#
#           Main Call
#       \***************/
#
if __name__ == '__main__':


    ###################
    # Define parameters
    ###################

    # Add argument here to handle it
    str_args = ['model.kp_mode']

    float_args = ['train.weight_decay',
                  'model.kp_radius',
                  'model.kp_sigma']

    int_args = ['model.kernel_size',
                'model.conv_groups',
                'model.inv_groups',
                'model.first_inv_layer',
                'train.cyc_decrease10',
                'train.max_epoch']

    bool_args = ['model.use_strided_conv',
                 'data.use_cubes',
                 'data.cylindric_input',
                 'augment_train.chromatic_contrast',
                 'augment_train.chromatic_all',
                 'augment_train.chromatic_norm',
                 'augment_train.height_norm']
                 

    list_args = ['model.layer_blocks',
                 'model.neighbor_limits']
    
    parser = argparse.ArgumentParser()
    for str_arg_name in str_args:
        parser_name = '--' + str_arg_name.split('.')[-1]
        parser.add_argument(parser_name, type=str)

    for float_arg_name in float_args:
        parser_name = '--' + float_arg_name.split('.')[-1]
        parser.add_argument(parser_name, type=float)

    for int_arg_name in int_args:
        parser_name = '--' + int_arg_name.split('.')[-1]
        parser.add_argument(parser_name, type=int)

    for bool_arg_name in bool_args:
        parser_name = '--' + bool_arg_name.split('.')[-1]
        parser.add_argument(parser_name, type=int)

    for list_arg_name in list_args:
        parser_name = '--' + list_arg_name.split('.')[-1]
        parser.add_argument(parser_name, nargs='+', type=int)

    # Log path special arg
    parser.add_argument('--log_path', type=str)
    args = parser.parse_args()

    # Configuration parameters
    cfg = my_config()

    # Load data parameters
    cfg.data.update(S3DIS_cfg(cfg).data)

    # Load experiment parameters
    if args.log_path is not None:
        get_directories(cfg, date=args.log_path)
    else:
        get_directories(cfg)

    # Update parameters
    for all_args in [str_args, float_args, int_args, list_args, bool_args]:
        for arg_name in all_args:
            key1, key2 = arg_name.split('.')
            new_arg = getattr(args, key2)
            if new_arg is not None:
                cfg[key1][key2] = new_arg

    # Sepcial boolean handling
    for arg_name in bool_args:
        key1, key2 = arg_name.split('.')
        new_arg = getattr(args, key2)
        if new_arg is not None:
            cfg[key1][key2] = bool(new_arg)

    # Adjust config after parameters have been changed
    cfg = adjust_config(cfg)

    
    ##############
    # Prepare Data
    ##############

    print('\n')
    frame_lines_1(['Data Preparation'])

    # Load dataset
    underline('Loading training dataset')
    training_dataset = S3DISDataset(cfg,
                                    chosen_set='training',
                                    precompute_pyramid=True)
    underline('Loading validation dataset')
    test_dataset = S3DISDataset(cfg,
                                chosen_set='validation',
                                regular_sampling=True,
                                precompute_pyramid=True)
    
    # Calib from training data
    training_dataset.calib_batch(cfg, update_test=False)
    training_dataset.calib_neighbors(cfg)
    test_dataset.b_n = cfg.test.batch_size
    test_dataset.b_lim = cfg.test.batch_limit

    # Save configuration now that it is complete
    save_cfg(cfg)
    
    # Initialize samplers
    training_sampler = SceneSegSampler(training_dataset)
    test_sampler = SceneSegSampler(test_dataset)

    # Initialize the dataloader
    training_loader = DataLoader(training_dataset,
                                 batch_size=1,
                                 sampler=training_sampler,
                                 collate_fn=SceneSegCollate,
                                 num_workers=cfg.train.num_workers,
                                 pin_memory=True)
    test_loader = DataLoader(test_dataset,
                             batch_size=1,
                             sampler=test_sampler,
                             collate_fn=SceneSegCollate,
                             num_workers=cfg.test.num_workers,
                             pin_memory=True)


    ###############
    # Build network
    ###############

    print()
    frame_lines_1(['Model preparation'])

    underline('Loading network')

    # Define network model
    t1 = time.time()


    modulated = False
    if 'mod' in cfg.model.kp_mode:
        modulated = True

    if cfg.model.kp_mode.startswith('kpconv'):
        net = KPConvFCNN(cfg, modulated=modulated, deformable=False)
    elif cfg.model.kp_mode.startswith('kpdef'):
        net = KPConvFCNN(cfg, modulated=modulated, deformable=True)
    elif cfg.model.kp_mode.startswith('kpinv'):
        net = KPInvFCNN(cfg)
    elif cfg.model.kp_mode.startswith('transformer') or cfg.model.kp_mode.startswith('inv_'):
        net = InvolutionFCNN(cfg)

    print()
    print(net)
    print("Model size %i" % sum(param.numel() for param in net.parameters() if param.requires_grad))

    debug = False
    if debug:
        print('\n*************************************\n')
        print(net.state_dict().keys())
        print('\n*************************************\n')
        for param in net.parameters():
            if param.requires_grad:
                print(param.shape)
        print('\n*************************************\n')
        print("Model size %i" % sum(param.numel() for param in net.parameters() if param.requires_grad))
        print('\n*************************************\n')

    print()
    
    ################
    # Start training
    ################

    # TODO:
    #
    #       00. KPDef List of experiments to do:
    #           > Test with param that allow kpdef-mod v2 to run. Compare v1 v2, def, conv mod, nomod
    #           > Test if we propagate gradient with neighbor influence
    #           > Test values of deform loss and deform lr
    #           > Test having deform only on later layers
    #           > Test using groups to reduce computation cost
    #           > Test inception style block with def and conv
    #           > Study if modulation = self-attention
    #           > Study relation with KPInv
    #           > Replace modulation with self-attention
    #
    #       0. KPInv does not work why??? Do we need specific learning rate for it?
    #
    #           > TODO: Implement and test all the designs in our powerpoint
    #                       - Point-involution-naive            OK
    #                       - Point-involution-v2               OK
    #                       - Point-involution-v3               OK
    #                       - Point-involution-v4               OK
    #                       - Point-transformers                OK
    #                       - KP-involution (verif si bug)
    #                       - KPConv-group modulations
    #                       - KPConv-inv
    #                       - Add geometric encoding to KPConv and related designs
    #                       TODO - see what Point COnv Former does? https://arxiv.org/pdf/2208.02879.pdf
    #                       TODO - Also SPNet https://arxiv.org/pdf/2109.11610.pdf
    #                       TODO - Also Fast Transformer and their geometric encoding (more efficient) https://arxiv.org/pdf/2112.04702.pdf
    #                               it also use cosine similarity instead of softmax
    #
    #           > TODO: Kernel point verification by measurinf chamfer distance with neighbors given different radiuses => Get optimal radius value
    #
    #       1. Go implement other datasets (NPM3D, Semantic3D, Scannetv2)
    #          Also other task: ModelNet40, ShapeNetPart, SemanticKitti
    #          Add code for completely different tasks??? Invariance??
    #           New classif dataset: ScanObjectNN
    #           Revisiting point cloud classification: A new benchmark dataset 
    #           and classification model on real-world data
    #
    #       3. Optimize operation
    #           > verify group conv results
    #           > check the effect of normalization in conv
    #           > use keops lazytensor in convolution ?
    #
    #       4. Optimize network
    #           > Test heads
    #           > Compare deeper architectures
    #           > Test subsampling ph
    #           > Number of parameters. Use groups, reduce some of the mlp operations
    #           > See optimization here:
    #               TODO - https://spell.ml/blog/pytorch-training-tricks-YAnJqBEAACkARhgD
    #               TODO - https://efficientdl.com/faster-deep-learning-in-pytorch-a-guide/#2-use-multiple-workers-and-pinned-memory-in-dataloader
    #               TODO - https://www.fast.ai/2018/07/02/adam-weight-decay/
    #               TODO - https://arxiv.org/pdf/2206.04670v1.pdf
    #               TODO - https://arxiv.org/pdf/2205.05740v2.pdf
    #               TODO - https://arxiv.org/pdf/2201.03545.pdf  MODERN RESNET
    #               TODO - https://arxiv.org/pdf/2109.11610.pdf  SPNet shows that Poisson Disc sampling  better (so FPS also) and Trilinear interp for upsampling as well
    #
    #           > State of the art agmentation technique:
    #               https://arxiv.org/pdf/2110.02210.pdf
    #
    #           > dont hesitate to train ensemble of models to score on Scannetv2
    #
    #           > Other state of the art technique to incorporate in code: border learning
    #               https://openaccess.thecvf.com/content/CVPR2022/papers/Tang_Contrastive_Boundary_Learning_for_Point_Cloud_Segmentation_CVPR_2022_paper.pdf
    #
    #           > Investigate cosine annealing (cosine decay).
    #
    #
    #       5. Explore
    #           > For benchmarking purpose, use multiscale dataset: introduce another scaling parameter
    #               in addtion to the anysotropic one, pick random value just before getting sphere and
    #               pick a sphere with the according size. then scale the sphere so that we have spheres 
    #               of the same scale eveytime, just the object in it will be "zoomed" or "dezoomed"
    #
    #           > Use multidataset, multihead segmentation and test deeper and deeper networks
    #
    #           > New task instance seg: look at mask group and soft group
    #
    #           > Study stronger downsampling at first layer like stems in RedNet101
    #
    #           > Study the batch size accumulation
    #
    #

    print('\n')
    frame_lines_1(['Training and Validation'])

    # Go
    train_and_validate(net, training_loader, test_loader, cfg, on_gpu=True)

    print('Forcing exit now')
    os.kill(os.getpid(), signal.SIGINT)




