
# ----------------------------------------------------------------------------------------------------------------------
#
#           Script Intro
#       \******************/
#
#
#   Use this script to test a network on S3DIS using the simple input pipeline 
#   (no neighbors computation in the dataloader)
#
#


# ----------------------------------------------------------------------------------------------------------------------
#
#           Imports and global variables
#       \**********************************/
#

# Common libs
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

from utils.config import load_cfg, save_cfg, get_directories
from utils.printing import frame_lines_1, underline

from models.KPConvNet import KPFCNN as KPConvFCNN
from models.KPInvNet import KPFCNN as KPInvFCNN
from models.InvolutionNet import InvolutionFCNN

from datasets.scene_seg import SceneSegSampler, SceneSegCollate

from experiments.S3DIS_simple.S3DIS import S3DIS_cfg, S3DISDataset

from tasks.trainval import train_and_validate


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
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_path', type=str)
    args = parser.parse_args()

    # Get log to test
    log = 'results/Log_2022-08-17_16-52-30'
    if args.log_path is not None:
        log = 'results/' + args.log_path

    # Configuration parameters
    cfg = load_cfg(log)

    # Change some parameters
    cfg.test.in_radius = 5.0
    cfg.test.batch_size = 4
    cfg.test.batch_limit = -1
    

    ##############
    # Prepare Data
    ##############

    print('\n')
    frame_lines_1(['Data Preparation'])

    # Load dataset
    underline('Loading validation dataset')
    test_dataset = S3DISDataset(cfg,
                                chosen_set='validation',
                                regular_sampling=True,
                                precompute_pyramid=True)
    
    # Calib from training data
    test_dataset.calib_batch(cfg)
    test_dataset.calib_neighbors(cfg)

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




