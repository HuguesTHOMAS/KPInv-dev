
# ----------------------------------------------------------------------------------------------------------------------
#
#           Script Intro
#       \******************/
#
#
#   Use this script to display CUDA memory consumption depending on the batch size (and thus nubmer of points)
#   Adapt the configuration to match your experiment and this script will give you insight on the maximum value 
#   of batch limit you should be able to use.
#
#


# ----------------------------------------------------------------------------------------------------------------------
#
#           Imports and global variables
#       \**********************************/
#

# Common libs
import os
import sys
import time
import signal
from torch.utils.data import DataLoader

# Local libs
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(os.path.dirname(current))
sys.path.append(parent)

from utils.config import init_cfg, save_cfg, get_directories
from utils.printing import frame_lines_1, underline

from models.architectures import KPFCNN

from datasets.scene_seg import SceneSegSampler, SceneSegCollate

from experiments.S3DIS_simple.S3DIS import S3DIS_cfg, S3DISDataset

from tasks.trainval import train_and_debug_cuda


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
    cfg.model.layer_blocks = (3, 4, 8, 8, 4)    

    cfg.model.kp_mode = 'kpconv'
    cfg.model.kernel_size = 15
    cfg.model.kp_radius = 2.5
    cfg.model.kp_sigma = 1.0
    cfg.model.kp_influence = 'linear'
    cfg.model.kp_aggregation = 'sum'

    cfg.data.sub_size = 0.02          # -1.0 so that dataset point clouds are not initially subsampled
    cfg.model.init_sub_size = 0.04    # Adapt this with train.in_radius. Try to keep a ratio of ~50
    cfg.model.sub_mode = 'grid'

    cfg.model.input_channels = 5    # This value has to be compatible with one of the dataset input features definition

    cfg.model.neighbor_limits = [35, 40, 50, 50, 50]    # Use empty list to let calibration get the values
    # cfg.model.neighbor_limits = []


    # Training parameters
    # -------------------

    cfg.train.num_workers = 0

    cfg.train.in_radius = 2.0    # Adapt this with model.init_sub_size. Try to keep a ratio of ~50
    cfg.train.batch_size = 6     # Target batch size. If you don't want calibration, you can directly set train.batch_limit

    cfg.train.max_epoch = 300
    cfg.train.steps_per_epoch = 10000
    cfg.train.checkpoint_gap = 50

    cfg.train.optimizer = 'SGD'
    cfg.train.sgd_momentum = 0.98

    cfg.train.lr = 1e-2
    cfg.train.lr_decays = {str(i): 0.1**(1 / 50) for i in range(1, cfg.train.max_epoch)}

    cfg.train.class_w = []

    cfg.train.augment_anisotropic = True
    cfg.train.augment_min_scale = 0.8
    cfg.train.augment_max_scale = 1.2
    cfg.train.augment_symmetries =  [True, False, False]
    cfg.train.augment_rotation = 'vertical'
    cfg.train.augment_noise = 0.005
    cfg.train.augment_color = 0.7


    # Test parameters
    # ---------------

    cfg.test.max_votes = 10
    cfg.test.steps_per_epoch = 50    # Size of one validation epoch (should be small)

    cfg.test.in_radius = cfg.train.in_radius
    cfg.test.num_workers = cfg.train.num_workers
    cfg.test.batch_size = cfg.train.batch_size
    cfg.test.batch_limit = cfg.train.batch_limit
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

    # Configuration parameters
    cfg = my_config()

    # Load data parameters
    cfg.data.update(S3DIS_cfg(cfg).data)

    # Load experiment parameters
    if len(sys.argv) > 1:
        get_directories(cfg, date=sys.argv[1])
    else:
        get_directories(cfg)
    
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
                                chosen_set='test',
                                regular_sampling=True,
                                precompute_pyramid=True)

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
    net = KPFCNN(cfg)

    print()
    print(net)


    print('\n')
    frame_lines_1(['Training and Validation'])

    # Go
    train_and_debug_cuda(net, training_loader, cfg)

    print('Forcing exit now')
    os.kill(os.getpid(), signal.SIGINT)




