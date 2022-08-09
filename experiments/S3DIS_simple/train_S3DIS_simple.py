
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

    # cfg.model.layer_blocks = (2, 1, 1, 1, 1)    # KPConv paper architecture. Can be changed for a deeper network
    # cfg.model.layer_blocks = (2, 2, 2, 4, 2)
    # cfg.model.layer_blocks = (2, 3, 4, 8, 4)
    # cfg.model.layer_blocks = (2, 3, 4, 16, 4)
    # cfg.model.layer_blocks = (2, 3, 8, 32, 4)

    
    # cfg.model.layer_blocks = (2,  3,  4,  6,  3)
    # cfg.model.layer_blocks = (3,  4,  6,  8,  4)
    # cfg.model.layer_blocks = (4,  6,  8,  8,  6)
    cfg.model.layer_blocks = (4,  6,  8, 12,  6)

    cfg.model.kp_mode = 'kpconv'
    cfg.model.kernel_size = 15
    cfg.model.kp_radius = 2.9
    cfg.model.kp_sigma = 1.7
    # cfg.model.kp_influence = 'linear'
    # cfg.model.kp_aggregation = 'sum'
    cfg.model.kp_influence = 'constant'
    cfg.model.kp_aggregation = 'nearest'

    cfg.data.sub_size = 0.02          # -1.0 so that dataset point clouds are not initially subsampled
    cfg.model.init_sub_size = 0.04    # Adapt this with train.in_radius. Try to keep a ratio of ~50
    cfg.model.sub_mode = 'grid'

    cfg.model.input_channels = 5    # This value has to be compatible with one of the dataset input features definition

    # cfg.model.neighbor_limits = [35, 40, 50, 50, 50]    # Use empty list to let calibration get the values
    cfg.model.neighbor_limits = []

    cfg.model.kpinv_grp_ch = 16         #   Int, number of channels per group in involutions
    cfg.model.kpinv_reduc = 4           #   Int, reduction ration for kpinv gen mlp


    # Training parameters
    # -------------------

    cfg.train.num_workers = 16

    cfg.train.in_radius = 1.8       # Adapt this with model.init_sub_size. Try to keep a ratio of ~50
    cfg.train.batch_size = 8        # Target batch size. If you don't want calibration, you can directly set train.batch_limit
    cfg.train.accum_batch = 5       # Accumulate batches for an effective batch size of batch_size * accum_batch.

    cfg.train.max_epoch = 200
    cfg.train.steps_per_epoch = 1000
    cfg.train.checkpoint_gap = 25

    cfg.train.optimizer = 'SGD'
    cfg.train.sgd_momentum = 0.9

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

    cfg.test.steps_per_epoch = 50    # Size of one validation epoch (should be small)
    
    cfg.test.max_votes = 10

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
                                chosen_set='validation',
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

    if cfg.model.kp_mode == 'kpconv':
        net = KPConvFCNN(cfg)
    elif cfg.model.kp_mode == 'kpinv':
        net = KPInvFCNN(cfg)


    print()
    print(net)

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
    #       0. KPInv does not work why??? Do we need specific learning rate for it?
    #           > Investigate why kpInv does not work
    #           > In plot consider the accumulation of batch. Search other places where we need to take care of that
    #           > Validation save only yhe susampled cloud, do reproj when testing the score
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
    #           > use einsum for the whole conv
    #           > use keops lazytensor
    #
    #       4. Optimize network
    #           > Test heads
    #           > Compare deeper architectures
    #           > Test subsampling ph
    #           > Number of parameters. Use groups, reduce some of the mlp operations
    #           > See optimization here:
    #                 OK - https://spell.ml/blog/pytorch-training-tricks-YAnJqBEAACkARhgD
    #               TODO - https://efficientdl.com/faster-deep-learning-in-pytorch-a-guide/#2-use-multiple-workers-and-pinned-memory-in-dataloader
    #               TODO - https://www.fast.ai/2018/07/02/adam-weight-decay/
    #               TODO - https://arxiv.org/pdf/2206.04670v1.pdf
    #               TODO - https://arxiv.org/pdf/2205.05740v2.pdf
    #
    #           > State of the art agmentation technique:
    #               https://arxiv.org/pdf/2110.02210.pdf
    #
    #           > dont hesitate to train ensemble of models to score on Scannetv2
    #
    #           > Other state of the art technique to incorporate in code: border learning
    #               https://openaccess.thecvf.com/content/CVPR2022/papers/Tang_Contrastive_Boundary_Learning_for_Point_Cloud_Segmentation_CVPR_2022_paper.pdf
    #
    #
    #       5. Explore
    #           > For benchmarking purpose, use multiscale dataset: introduce another scaling parameter
    #               in addtion to the anysotropic one, pick random value just before getting sphere and
    #               pick a sphere with the according size. then scale the sphere so that we have spheres 
    #               of the same scale eveytime, justthe object in it will be "zoomed" or "dezoomed"
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




