
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
import torch
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

from tasks.test import test_model


# ----------------------------------------------------------------------------------------------------------------------
#
#           Main Function
#       \*******************/
#


def test_log(chosen_log, new_cfg, save_visu=False):

    ##############
    # Prepare Data
    ##############

    print('\n')
    frame_lines_1(['Data Preparation'])

    # Load dataset
    underline('Loading validation dataset')
    test_dataset = S3DISDataset(new_cfg,
                                chosen_set='test',
                                regular_sampling=True,
                                precompute_pyramid=True)
    
    # Calib from training data
    # test_dataset.calib_batch(new_cfg)
    # test_dataset.calib_neighbors(new_cfg)
    
    # Initialize samplers
    test_sampler = SceneSegSampler(test_dataset)
    test_loader = DataLoader(test_dataset,
                             batch_size=1,
                             sampler=test_sampler,
                             collate_fn=SceneSegCollate,
                             num_workers=new_cfg.test.num_workers,
                             pin_memory=True)


    ###############
    # Build network
    ###############

    print()
    frame_lines_1(['Model preparation'])

    underline('Loading network')

    modulated = False
    if 'mod' in new_cfg.model.kp_mode:
        modulated = True

    if new_cfg.model.kp_mode.startswith('kpconv'):
        net = KPConvFCNN(new_cfg, modulated=modulated, deformable=False)
    elif new_cfg.model.kp_mode.startswith('kpdef'):
        net = KPConvFCNN(new_cfg, modulated=modulated, deformable=True)
    elif new_cfg.model.kp_mode.startswith('kpinv'):
        net = KPInvFCNN(new_cfg)
    elif new_cfg.model.kp_mode.startswith('transformer') or new_cfg.model.kp_mode.startswith('inv_'):
        net = InvolutionFCNN(new_cfg)

        
    #########################
    # Load pretrained weights
    #########################

    # Find all checkpoints in the chosen training folder
    chkp_path = os.path.join(chosen_log, 'checkpoints')
    chkps = [f for f in os.listdir(chkp_path) if f[:4] == 'chkp']

    # Find which snapshot to restore
    if new_cfg.test.chkp_idx is None:
        chosen_chkp = 'current_chkp.tar'
    else:
        chosen_chkp = np.sort(chkps)[new_cfg.test.chkp_idx]
    chosen_chkp = os.path.join(chosen_log, 'checkpoints', chosen_chkp)

    # Load previous checkpoint
    checkpoint = torch.load(chosen_chkp)
    net.load_state_dict(checkpoint['model_state_dict'])
    net.eval()
    print("\nModel and training state restored from:")
    print(chosen_chkp)
    print()
    
    ############
    # Start test
    ############

    print('\n')
    frame_lines_1(['Training and Validation'])

    # Go
    test_model(net, test_loader, new_cfg, save_visu=save_visu)

    return


# ----------------------------------------------------------------------------------------------------------------------
#
#           Main Call
#       \***************/
#


if __name__ == '__main__':

    ###############################
    # Choose the model to visualize
    ###############################

    #   Here you can choose which model you want to test with the variable test_model. Here are the possible values :
    #
    #       > 'last_XXX': Automatically retrieve the last trained model on dataset XXX
    #       > '(old_)results/Log_YYYY-MM-DD_HH-MM-SS': Directly provide the path of a trained model

    chosen_log = 'results/Log_2022-08-17_16-52-30'

    # Add argument here to handle it
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_path', type=str)
    args = parser.parse_args()

    # Get log to test
    if args.log_path is not None:
        chosen_log = 'results/' + args.log_path

    #############
    # Load config
    #############

    # Configuration parameters
    new_cfg = load_cfg(chosen_log)


    ###################
    # Define parameters
    ###################

    # Change some parameters
    new_cfg.test.in_radius = 4.0
    new_cfg.test.batch_limit = 1
    new_cfg.test.steps_per_epoch = 9999999
    new_cfg.test.max_votes = 3
    new_cfg.test.chkp_idx = None

    # Augmentations
    new_cfg.train.augment_anisotropic = True
    new_cfg.train.augment_min_scale = 0.99
    new_cfg.train.augment_max_scale = 1.01
    new_cfg.train.augment_symmetries =  [True, False, False]
    new_cfg.train.augment_rotation = 'vertical'
    new_cfg.train.augment_noise = 0.0001
    new_cfg.train.augment_color = 0.1


    test_log(chosen_log, new_cfg)
    

    print('Forcing exit now')
    os.kill(os.getpid(), signal.SIGINT)




