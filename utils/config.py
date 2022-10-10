#
#
#      0=================================0
#      |    Kernel Point Convolutions    |
#      0=================================0
#
#
# ----------------------------------------------------------------------------------------------------------------------
#
#      Configuration class
#
# ----------------------------------------------------------------------------------------------------------------------
#
#      Hugues THOMAS - 11/06/2018
#

import os
from os import makedirs, getcwd
from os.path import join, basename, dirname, realpath, exists
import numpy as np
import json
import time

from easydict import EasyDict




def init_cfg():

    # Use easydict for parameters
    cfg = EasyDict()

    # Experiment specification
    # ------------------------

    cfg.exp = EasyDict()
    cfg.exp.working_dir = ''            #   Str, current working directory
    cfg.exp.date = ''                   #   Str, date of the strat of the experiment
    cfg.exp.results_dir = ''            #   Str, directory for all results
    cfg.exp.log_dir = ''                #   Str, directory where this experiemnt is saved
    cfg.exp.seed = 42                   #   Int, seed for random stuff
    cfg.exp.saving = True               #  Bool, is the experiment saved or not


    # Data parameters
    # ---------------

    cfg.data = EasyDict()
    cfg.data.name = ''                  #   Str, name of the dataset
    cfg.data.path = ''                  #   Str, path of the dataset
    cfg.data.task = ''                  #   Str, name of the task performed

    cfg.data.dim = 3                    #   Int, dimension of data points
    cfg.data.num_classes = 2            #   Int, Number of classes

    cfg.data.label_and_names = []       #  List, value and name of each label
    cfg.data.label_names = []           #  List, name of each label
    cfg.data.label_values = []          #  List, value of each label

    cfg.data.name_to_label = {}         #  Dict, get label values from names
    cfg.data.name_to_idx = {}           #  Dict, get label index from values

    cfg.data.ignored_labels = []        #  List, value of ignored label values

    cfg.data.init_sub_size = 0.02       # Float, data subampling size, negative value means we use original data
    cfg.data.init_sub_mode = 'grid'     # Mode for initial subsampling of data

    cfg.data.use_cubes = False          #  Bool, do we use cube sampling instead of sphere samplings
    cfg.data.cylindric_input = False         #  Bool, do we use infinite height for cube sampling


    # Network parameters
    # ------------------

    cfg.model = EasyDict()
    cfg.model.kp_mode = 'kpconv'        #   Str, choice of basic operator ('kpconv', 'kpinv', 'kpdef')
    cfg.model.shell_sizes = [15]        #  List, number of kernel points
    cfg.model.kp_radius = 2.5           # Float, radius of the basic operator
    cfg.model.kp_sigma = 1.0            # Float, radius of the kernel point influence
    cfg.model.kp_influence = 'linear'   #   Str, Influence function when d < KP_extent. ('constant', 'linear', 'gaussian')
    cfg.model.kp_aggregation = 'sum'    #   Str, Aggregation mode ('nearest', 'sum')
    cfg.model.kp_fixed = 'center'       #   Str, Fixed points in the kernel ('none', 'center', 'verticals')
    cfg.model.conv_groups = 1           #   Int, number of groups for groups in convolution (-1 for depthwise)
    cfg.model.share_kp = False          #  Bool, option to share the kenrel point (and thus neighbor influences and weights) 
                                        #        across the KPConv of the same layer
                   
    cfg.model.inv_groups = 1            #   Int, number of groups for groups in involution                     
    cfg.model.inv_grp_norm = False      #  Bool, Choose to use group norm for involution weights or kpminimod modulations
    cfg.model.inv_act = 'sigmoid'       #   Str, activation function for involution weights, kpminimod modulations etc.

    cfg.model.kpinv_reduc = 1           #   Int, reduction ration for kpinv gen mlp
    cfg.model.kpx_expansion = 8         #   Int, expansion parameter for kpinvX

    cfg.model.use_strided_conv = True   #  Bool, Use convolution op for strided layers instead of involution
    cfg.model.first_inv_layer = 0       #   Int, Use involution layers only from this layer index

    cfg.model.kp_deform_w = 1.0         # Float, multiplier for deformation the fitting/repulsive loss
    cfg.model.kp_deform_grad = 0.1      # Float, multiplier for deformation gradient
    cfg.model.kp_repulse_dist = 1.0     # Float, distance of repulsion for deformed kernel points
    
    cfg.model.in_sub_size = 0.04        # Float, initial subampling size, (voxel size, lattice size of fps minimum distance)
    cfg.model.in_sub_mode = 'grid'      #   Str, subsampling mode ('grid', 'ph', 'fps')
    cfg.model.radius_scaling = 2.0      # Float, scaling of the radius at each layer.
    
    cfg.model.upsample_n = 1            #   Int, Number of neighbors used for nearest neighbor linear interpolation

    cfg.model.input_channels = 1        #   Int, dimension of input feaures
    cfg.model.init_channels = 64        #   Int, dimension of first network features
    cfg.model.norm = 'batch'            #   Str, type of normalization in the network ('group', 'batch', 'layer' 'none')
    cfg.model.bn_momentum = 0.1         # Float, Momentum for batch normalization (inverse of what is usually done , 0.01 is strong momentum).

    cfg.model.layer_blocks = (2, 1, 1)  # Tuple, number of blocks in each layers (in addition to the strided ones in between).
    cfg.model.neighbor_limits = []      #  List, maximum number of neigbors per layers

    cfg.model.process_ratio = 1.0       # Float, ratio between the radius of processed volume and the radius of the input volume
    cfg.model.n_frames = 1              #   Int, number of frames used (Specific to SLAM)


    # Training parameters
    # -------------------

    # train
    cfg.train = EasyDict()
    cfg.train.num_workers = 16          #   Int, number of parallel workers for input dataset
    cfg.train.checkpoint_gap = 50       #   Int, gap between each saved checkpoint

    cfg.train.max_epoch = 300           #   Int, number of training epochs
    cfg.train.steps_per_epoch = 1000    #   Int, number of steps per epoch

    cfg.train.in_radius = 1.0           # Float, radius of the input sphere
    cfg.train.batch_size = 16           #   Int, number of input point cloud per batch
    cfg.train.accum_batch = 1           #   Int, number of batches accumulated before performing an optimizer step batch size.
    cfg.train.batch_limit = -1          #   Int, maximum number of points in total in a batch

    cfg.train.data_sampler = 'c-random' #   Str, Data sampling mode to choose input spheres ('regular', 'random', 'c-random')
    cfg.train.max_points = -1           #   Int, maximum number of points per element (obsolete)
    
    cfg.train.optimizer = 'SGD'         #   Str, optimizer ('SGD', 'Adam' or 'AdamW')
    cfg.train.lr = 1e-2                 # Float, initial learning rate
    cfg.train.sgd_momentum = 0.95       # Float, learning rate momentum for sgd optimizer
    cfg.train.adam_b = (0.9, 0.999)     # Float, betas for Adam optimizer
    cfg.train.adam_eps = 1e-08          # Float, eps for Adam optimizer
    cfg.train.weight_decay = 1e-2       # Float, weight decay
    cfg.train.lr_decays = {'10': 0.1}   #  Dict, decay values with their epoch {epoch: decay}
    cfg.train.warmup = True             #  Bool, should the first epoch be a warmup
    cfg.train.grad_clip = 100.0         #   Int, gradient clipping value (negative means no clipping)
    cfg.train.class_w = []              #  List, weight for each class in the segmentation loss
    cfg.train.smooth_labels = False     #  Bool, should smooth labels for cross entropy loss?

    cfg.train.segloss_balance = 'none'      #   Str, Respectively each point, class, or cloud in the batch has
                                            #        the same loss contribution ('none', 'class', 'batch'). 
                                            
    cfg.train.cyc_lr0 = 1e-4                # Float, Start (minimum) learning rate of 1cycle decay
    cfg.train.cyc_lr1 = 1e-2                # Float, Maximum learning rate of 1cycle decay
    cfg.train.cyc_raise_n = 5               #   Int, Raise rate for first part of 1cycle = number of epoch to multiply lr by 10
    cfg.train.cyc_decrease10 = 50           #   Int, Decrease rate for second part of 1cycle = number of epoch to divide lr by 10
    cfg.train.cyc_plateau = 20              #   Int, Number of epoch for plateau at maximum lr

    cfg.train.deform_loss_factor = 0.1      # Float, multiplier for deformation loss. Reduce to reduce influence for deformation on overall features
    cfg.train.deform_lr_factor = 1.0        # Float, multiplier for deformation lr. Higher so that feformation are learned faster (especially if deform_loss_factor i low)
    cfg.train.deform_fit_rep_ratio = 2.0    # Float, ratio between fitting loss and regularization loss
    

    # Test parameters
    # ---------------

    # test
    cfg.test = EasyDict()
    cfg.test.num_workers = 16           #   Int, number of parallel workers for input dataset.

    cfg.test.max_votes = 10             #   Int, number of training epochs
    cfg.test.max_steps_per_epoch = 50   #   Int, number of steps per epoch

    cfg.test.in_radius = 1.0            # Float, radius of the input sphere
    cfg.test.batch_size = 8             #   Int, number of input point cloud per batch
    cfg.test.batch_limit = -1           #   Int, maximum number of points in total in a batch

    cfg.test.data_sampler = 'regular'   #   Str, Data sampling mode to choose input spheres ('regular', 'random', 'c-random')
    cfg.test.max_points = -1            #   Int, maximum number of points per element (obsolete)

    cfg.test.val_momentum = 0.95        # Float, momentum for averaging predictions during validation.
    cfg.test.test_momentum = 0.95       # Float, momentum for averaging predictions during test.
    cfg.test.chkp_idx = None            #   Int, index of the checkpoint used for test


    # Augmentation parameters
    # -----------------------
    
    # Train Augmentations
    cfg.augment_train = EasyDict()
    cfg.augment_train.anisotropic = True
    cfg.augment_train.scale = [0.9, 1.1]
    cfg.augment_train.flips = [0.5, 0, 0]
    cfg.augment_train.rotations = 'vertical'
    cfg.augment_train.jitter = 0.005
    cfg.augment_train.color_drop = 0.2
    cfg.augment_train.chromatic_contrast = False
    cfg.augment_train.chromatic_all = False
    cfg.augment_train.chromatic_norm = False
    cfg.augment_train.height_norm = False

    # Test Augmentations
    cfg.augment_test = EasyDict()
    cfg.augment_test.anisotropic = False
    cfg.augment_test.scale = [0.99, 1.01]
    cfg.augment_test.flips = [0.5, 0, 0]
    cfg.augment_test.rotations = 'vertical'
    cfg.augment_test.jitter = 0
    cfg.augment_test.color_drop = 0
    cfg.augment_test.chromatic_contrast = False
    cfg.augment_test.chromatic_all = False
    cfg.augment_test.chromatic_norm = False
    cfg.augment_test.height_norm = False


    return cfg


def get_directories(cfg, date=None, seed=None):
    
    # Get date unless it is given
    if date is None:
        cfg.exp.date = time.strftime('Log_%Y-%m-%d_%H-%M-%S')
    else:
        cfg.exp.date = date
    cfg.exp.working_dir = getcwd() # dirname(realpath(__file__))
    cfg.exp.results_dir = join(cfg.exp.working_dir, "results")
    cfg.exp.log_dir = join(cfg.exp.results_dir, cfg.exp.date)
    if seed is not None:
        cfg.exp.seed = seed

    if not exists(cfg.exp.log_dir):
        makedirs(cfg.exp.log_dir)

    return

def save_cfg(cfg, path=None):

    if path is None:
        path = cfg.exp.log_dir

    # Serialize data into file:
    with open(join(path, 'parameters.json'), "w") as jsonfile:
        json.dump(cfg, jsonfile, indent=4, sort_keys=False)
    return


def load_cfg(log_path):

    # Create an empty cfg
    cfg = init_cfg()

    # Read data from file:
    with open(join(log_path, 'parameters.json'), "r") as jsonfile:
        cfg2 =json.load(jsonfile)

    for k, v in cfg.items():
        if k in cfg2:
            cfg[k].update(cfg2[k])

    return cfg

