
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

    #
    #   +-------------------------------------------------------------------------------------------+
    #   |   How to Benchmark:                                                                       |
    #   |                                                                                           |
    #   |       1. Modify parameters as you wish below this.                                        |
    #   |                                                                                           |
    #   |       2. cd to the directory [...]/KPInv-dev                                              |
    #   |                                                                                           |
    #   |       3. Run:                                                                             |
    #   |           python3 experiments/S3DIS_simple/benchmark_S3DIS.py --param_name param_value    |
    #   |                                                                                           |
    #   |       4. After a few minutes you should be able to plot where the training is at with:    |
    #   |           python3 experiments/S3DIS_simple/plot_benchmarks.py                             |
    #   +-------------------------------------------------------------------------------------------+
    #
    #
    #       List of benchmarkings that we need to perform:
    #           
    #               1. KPConv radius/sigma study
    #                   The radius of KPConv determines the nubmer of neighbors, but also the size 
    #                   of the kernel. We want to try different values of radius and sigma:
    #                       - radius in [1.4, 1.9, 2.4, 2.9, 3.4, 3.9]
    #                       - sigma = radius * 0.7
    #                       
    #                   (args: --kp_radius XXXX --kp_sigma XXXX)
    #                   N.B. Start with the biggest radiuses so that you are sure that the next 
    #                        experiments will fit in the GPU memory
    #           
    #               2. KPConv radius/sigma study around 1.9
    #                   Same but focus on small radiuses.
    #                       - radius in [1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3]
    #                       - sigma = radius * 0.7
    #
    #               3. Network number of channels.
    #                   See if we can get better results with more channels.
    #                       - init_channels in [16, 32, 48, 64, 80, 96, 112, 128]
    #
    #               4. Test group convolution. TODO
    #
    #               5. check the effect of normalization in conv TODO
    #
    #               6. Study stronger downsampling at first layer like stems in RedNet101 TODO
    #


    cfg.model.init_channels = 64
    cfg.model.layer_blocks = (2,  3,  4,  6,  3)
    # cfg.model.layer_blocks = (3,  4,  6,  8,  4)
    # cfg.model.layer_blocks = (4,  6,  8,  8,  6)
    # cfg.model.layer_blocks = (4,  6,  8, 12,  6)


    cfg.model.kp_mode = 'kpconv'        # Choose ['kpconv', 'kpdef', 'kpinv']. And ['kpconv-mod', 'kpdef-mod'] for modulations
    cfg.model.shell_sizes = [15]
    cfg.model.kp_radius = 2.5
    cfg.model.kp_sigma = 1.2
    cfg.model.kp_influence = 'linear'
    cfg.model.kp_aggregation = 'sum'

    cfg.data.init_sub_size = 0.02          # -1.0 so that dataset point clouds are not initially subsampled
    cfg.model.in_sub_size = 0.04    # Adapt this with train.in_radius. Try to keep a ratio of ~50
    cfg.model.in_sub_mode = 'grid'

    cfg.model.input_channels = 5    # This value has to be compatible with one of the dataset input features definition

    # cfg.model.neighbor_limits = [35, 40, 50, 50, 50]    # Use empty list to let calibration get the values
    cfg.model.neighbor_limits = []

    cfg.model.kpinv_grp_ch = 1          #   Int, number of channels per group in involutions
    cfg.model.kpinv_reduc = 1           #   Int, reduction ration for kpinv gen mlp



    # Training parameters
    # -------------------

    # Input threads
    cfg.train.num_workers = 16

    # Input spheres radius. Adapt this with model.in_sub_size. Try to keep a ratio of ~50
    cfg.train.in_radius = 1.8

    # Batch related_parames
    cfg.train.batch_size = 8            # Target batch size. If you don't want calibration, you can directly set train.batch_limit
    cfg.train.accum_batch = 5           # Accumulate batches for an effective batch size of batch_size * accum_batch.
    cfg.train.steps_per_epoch = 125
    
    # Training length
    cfg.train.max_epoch = 180
    cfg.train.checkpoint_gap = cfg.train.max_epoch // 5
    
    # Deformations
    cfg.train.deform_loss_factor = 0.1      # Reduce to reduce influence for deformation on overall features
    cfg.train.deform_lr_factor = 1.0        # Higher so that deformation are learned faster (especially if deform_loss_factor is low)

    # Optimizer
    cfg.train.optimizer = 'AdamW'
    cfg.train.adam_b = (0.9, 0.999)
    cfg.train.adam_eps = 1e-08
    cfg.train.weight_decay = 0.01

    # Cyclic lr 
    cfg.train.cyc_lr0 = 1e-4                # Float, Start (minimum) learning rate of 1cycle decay
    cfg.train.cyc_lr1 = 1e-2                # Float, Maximum learning rate of 1cycle decay
    cfg.train.cyc_raise10 = 5               #   Int, Raise rate for first part of 1cycle = number of epoch to multiply lr by 10
    cfg.train.cyc_decrease10 = 80           #   Int, Decrease rate for second part of 1cycle = number of epoch to divide lr by 10
    cfg.train.cyc_plateau = 20              #   Int, Number of epoch for plateau at maximum lr
    raise_rate = 10**(1 / cfg.train.cyc_raise10)
    decrease_rate = 0.1**(1 / cfg.train.cyc_decrease10)
    cfg.train.lr = cfg.train.cyc_lr0
    n_raise = int(np.ceil(np.log(cfg.train.cyc_lr1/cfg.train.cyc_lr0) / np.log(raise_rate)))
    cfg.train.lr_decays = {str(i): raise_rate for i in range(1, n_raise)}
    for i in range(n_raise + cfg.train.cyc_plateau, cfg.train.max_epoch):
        cfg.train.lr_decays[str(i)] = decrease_rate

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

    # Augmentations
    cfg.augment_train.anisotropic = True
    cfg.augment_train.scale = [0.8, 1.2]
    cfg.augment_train.flips = [0.5, 0, 0]
    cfg.augment_train.rotations = 'vertical'
    cfg.augment_train.jitter = 0.005
    cfg.augment_train.color_drop = 0.2

    
    # Test parameters
    # ---------------

    cfg.test.val_momentum = 0.5  # momentum for averaging predictions during validation. 0 for no averaging at all

    cfg.test.max_steps_per_epoch = 50    # Size of one validation epoch (should be small)
    
    cfg.test.max_votes = 10

    cfg.test.in_radius = cfg.train.in_radius
    cfg.test.num_workers = cfg.train.num_workers
    cfg.test.batch_size = cfg.train.batch_size
    cfg.test.batch_limit = cfg.train.batch_limit

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

    int_args = []

    list_args = ['model.layer_blocks',
                 'model.shell_sizes']

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
    for all_args in [str_args, float_args, int_args, list_args]:
        for arg_name in all_args:
            key1, key2 =arg_name.split('.')
            new_arg = getattr(args, key2)
            if new_arg is not None:
                cfg[key1][key2] = new_arg

    
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
                                precompute_pyramid=True)
    
    # Calib from training data
    training_dataset.calib_batch(cfg)
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

    print('\n')
    frame_lines_1(['Training and Validation'])

    # Go
    train_and_validate(net, training_loader, test_loader, cfg, on_gpu=True)

    print('Forcing exit now')
    os.kill(os.getpid(), signal.SIGINT)




