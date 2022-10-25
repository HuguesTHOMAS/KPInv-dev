
# ----------------------------------------------------------------------------------------------------------------------
#
#           Script Intro
#       \******************/
#
#
#   This script is used to plot the results of our networks. YOu can run it while a network is still training to see  
#   if it has converged yet.
#
#


# ----------------------------------------------------------------------------------------------------------------------
#
#           Imports and global variables
#       \**********************************/
#

# Common libs
from signal import raise_signal
import sys
import os
import time
import numpy as np
import matplotlib.pyplot as plt
from os.path import isfile, join, exists
from os import listdir, remove, getcwd, makedirs


# Common libs
import matplotlib.pyplot as plt
from os.path import isfile, join, exists
from os import listdir, remove, getcwd

# Local libs
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(os.path.dirname(current))
sys.path.append(parent)


# My libs
from utils.ply import read_ply


# My libs
from utils.printing import frame_lines_1, underline
from utils.ply import read_ply, write_ply
from utils.config import load_cfg

from utils.plot_utilities import listdir_str, print_cfg_diffs, compare_trainings, compare_convergences_classif, \
    compare_on_test_set, cleanup


# ----------------------------------------------------------------------------------------------------------------------
#
#           Experiments
#       \*****************/
#


def experiment_name_1():
    """
    In this function you choose the results you want to plot together, to compare them as an experiment.
    Just return the list of log paths (like 'results/Log_2020-04-04_10-04-42' for example), and the associated names
    of these logs.
    Below an example of how to automatically gather all logs between two dates, and name them.
    """

    # Using the dates of the logs, you can easily gather consecutive ones. All logs should be of the same dataset.
    start = 'Log_2020-04-22_11-52-58'
    end = 'Log_2023-07-29_12-40-27'

    # Name of the result path
    res_path = 'results'

    # Gather logs and sort by date
    logs = np.sort([join(res_path, l) for l in listdir_str(res_path) if start <= l <= end])

    # Give names to the logs (for plot legends)
    logs_names = ['name_log_1',
                  'name_log_2',
                  'name_log_3',
                  'name_log_4']

    # safe check log names
    logs_names = np.array(logs_names[:len(logs)])

    return logs, logs_names


def experiment_name_2():
    """
    In this function you choose the results you want to plot together, to compare them as an experiment.
    Just return the list of log paths (like 'results/Log_2020-04-04_10-04-42' for example), and the associated names
    of these logs.
    Below an example of how to automatically gather all logs between two dates, and name them.
    """

    # Using the dates of the logs, you can easily gather consecutive ones. All logs should be of the same dataset.
    start = 'Log_2020-04-22_11-52-58'
    end = 'Log_2020-05-22_11-52-58'

    # Name of the result path
    res_path = 'results'

    # Gather logs and sort by date
    logs = np.sort([join(res_path, l) for l in listdir_str(res_path) if start <= l <= end])

    # Optionally add a specific log at a specific place in the log list
    logs = logs.astype('<U50')
    logs = np.insert(logs, 0, 'results/Log_2020-04-04_10-04-42')

    # Give names to the logs (for plot legends)
    logs_names = ['name_log_inserted',
                  'name_log_1',
                  'name_log_2',
                  'name_log_3']

    # safe check log names
    logs_names = np.array(logs_names[:len(logs)])

    return logs, logs_names


def test_kpnext_1():
    """
    GOGO KPNext experiments
    """

    # Using the dates of the logs, you can easily gather consecutive ones. All logs should be of the same dataset.
    start = 'Log_2022-10-21_08-00-57'
    end = 'Log_2022-12-29_23-43-08'

    # Name of the result path
    res_path = 'results'

    # Gather logs and sort by date
    logs = np.sort([join(res_path, l) for l in listdir_str(res_path) if start <= l <= end])

    # # Optionally add a specific log at a specific place in the log list
    # logs = logs.astype('<U50')
    # logs = np.insert(logs, 0, 'results/Log_2022-10-13_17-07-58')

    # Give names to the logs (for plot legends)
    logs_names = ['KPNextBig48 grid0.02 + upcut',
                  'KPNextBig48 grid0.02',
                  'KPNextDouble48 grid0.02 + upcut',
                  'KPNextDouble48 grid0.02',
                  'KPNextBig48 grid0.025']

    # safe check log names
    if len(logs) > len(logs_names):
        logs = logs[:len(logs_names)]
    logs_names = np.array(logs_names[:len(logs)])

    # logs = logs[[-1, -3, -4]]
    # logs_names = logs_names[[-1, -3, -4]]

    return logs, logs_names


def test_kpnext_2():
    """
    GOGO KPNext experiments
    """

    # Using the dates of the logs, you can easily gather consecutive ones. All logs should be of the same dataset.
    start = 'Log_2022-10-21_21-47-10'
    end = 'Log_2022-12-29_23-43-08'

    # Name of the result path
    res_path = 'results'

    # Gather logs and sort by date
    logs = np.sort([join(res_path, l) for l in listdir_str(res_path) if start <= l <= end])

    # # Optionally add a specific log at a specific place in the log list
    # logs = logs.astype('<U50')
    # logs = np.insert(logs, 0, 'results/Log_2022-10-13_17-07-58')

    # Give names to the logs (for plot legends)
    logs_names = ['KPNextBig48 grid0.010',
                  'KPNextBig48 grid0.011',
                  'KPNextBig48 grid0.012',
                  'KPNextBig48 grid0.013',
                  'KPNextBig48 grid0.014',
                  'KPNextBig48 grid0.015',
                  'KPNextBig48 grid0.016',
                  'KPNextBig48 grid0.017',
                  'KPNextBig48 grid0.018',
                  'KPNextBig48 grid0.019',
                  'KPNextBig48 grid0.020',
                  'KPNextBig48 grid0.021',
                  'KPNextBig48 grid0.022',
                  'KPNextBig48 grid0.024']

    # Write KPNExt architecture. Note: it is very similar to resnet, just the shortcuts 
    # are not in the same place otherwise everythong is similar. SO write KPNext and then 
    # rewrite KPFCNN with equivalent nubmer of params. KPFCNN should be biggger because of 
    # shortcut mlp being bigger.
    # In the end if we do not use shortcut during downsampling, I think resnet should be 
    # better than Inverted because the shortcut is on the bigger features. Just think about 
    # how you want to place your downsampling because between to shortcuts the dim will be x4.
    # 
    # For KPNext, we need 
    #   OK - a better stem, convolution that output direcly the right number of features. 
    #                    Also directly downsample to next layer to avoid having inverted block on large layers
    #   OK - better strided blocks. Use pure KPConv?
    #   ?  - new stuff for optimization, DropPath etc
    #   ?  - test heads
    #   ?  - test number of channels vs number of layer (depth vs width)


    # TODO:
    #
    #       1. New architecture 
    #           > Test heads
    #           > Test stems
    #           > Convnext, DropPath etc
    #           > Number of parameters.
    #           > See optimization here:
    #               TODO - https://spell.ml/blog/pytorch-training-tricks-YAnJqBEAACkARhgD
    #               TODO - https://efficientdl.com/faster-deep-learning-in-pytorch-a-guide/#2-use-multiple-workers-and-pinned-memory-in-dataloader
    #               TODO - https://www.fast.ai/2018/07/02/adam-weight-decay/
    #               TODO - https://arxiv.org/pdf/2206.04670v1.pdf
    #               TODO - https://arxiv.org/pdf/2205.05740v2.pdf
    #               TODO - https://arxiv.org/pdf/2201.03545.pdf  MODERN RESNET
    #               TODO - https://arxiv.org/pdf/2109.11610.pdf  SPNet shows that Poisson Disc sampling  better (so FPS also) and Trilinear interp for upsampling as well
    #
    #       2. Poisson disk sampling
    #
    #       3. (Border repulsive loss) + (Mix3D) + (model ensemble) and submit to Scannetv2
    #
    #       4. Go implement other datasets (NPM3D, Semantic3D, Scannetv2)
    #          Also other task: ModelNet40, ShapeNetPart, SemanticKitti
    #          Add code for completely different tasks??? Invariance??
    #           New classif dataset: ScanObjectNN
    #           Revisiting point cloud classification: A new benchmark dataset 
    #           and classification model on real-world data
    #
    #       5. Parameters to play with at the end
    #           > color drop
    #           > init_feature_dim
    #           > layers
    #           > radius (sphere or cylinder)
    #           > knn
    #           > kp radius (for kp) and K and shells
    #           > trainer
    #           > KPConvX vs KPConvD vs KPInv
    #           > groups in KPConvX
    #           > n_layers in KPConvX
    #


    # safe check log names
    if len(logs) > len(logs_names):
        logs = logs[:len(logs_names)]
    logs_names = np.array(logs_names[:len(logs)])

    # logs = logs[[-1, -3, -4]]
    # logs_names = logs_names[[-1, -3, -4]]

    return logs, logs_names


def test_kpnext_3():
    """
    GOGO KPNext experiments
    """

    # Using the dates of the logs, you can easily gather consecutive ones. All logs should be of the same dataset.
    start = 'Log_2022-10-24_11-13-42'
    end = 'Log_2022-12-29_23-43-08'

    # Name of the result path
    res_path = 'results'

    # Gather logs and sort by date
    logs = np.sort([join(res_path, l) for l in listdir_str(res_path) if start <= l <= end])

    # Optionally add a specific log at a specific place in the log list
    logs = logs.astype('<U50')
    logs = np.insert(logs, 0, 'results/Log_2022-10-23_04-15-22')

    # Give names to the logs (for plot legends)
    logs_names = ['none + grid0.019',
                  'fps1024 + grid0.014',
                  'fps1024 + grid0.015',
                  'fps1024 + grid0.016',
                  'fps1024 + grid0.017',
                  'fps1024 + grid0.018',
                  'fps1024 + grid0.019 12',
                  'fps1024 + grid0.020',
                  'fps1200 + grid0.019',
                  'fps1024 + grid0.019 10',
                  'grid1024 + grid0.019 10']

    # safe check log names
    if len(logs) > len(logs_names):
        logs = logs[:len(logs_names)]
    logs_names = np.array(logs_names[:len(logs)])

    # logs = logs[[-1, -3, -4]]
    # logs_names = logs_names[[-1, -3, -4]]

    return logs, logs_names


def test_grid019():
    """
    GOGO KPNext experiments
    """

    # Using the dates of the logs, you can easily gather consecutive ones. All logs should be of the same dataset.
    start = 'Log_2022-10-25_16-41-25'
    end = 'Log_2022-12-29_23-43-08'

    # Name of the result path
    res_path = 'results'

    # Gather logs and sort by date
    logs = np.sort([join(res_path, l) for l in listdir_str(res_path) if start <= l <= end])

    # Optionally add a specific log at a specific place in the log list
    logs = logs.astype('<U50')
    logs = np.insert(logs, 0, 'results/Log_2022-10-25_14-14-19')  # 'grid1024 + grid0.019 10'
    logs = np.insert(logs, 1, 'results/Log_2022-10-25_14-13-17')  # 'fps1024 + grid0.019 10'
    logs = np.insert(logs, 2, 'results/Log_2022-10-25_09-15-27')  # 'fps1024 + grid0.019 12'

    # Give names to the logs (for plot legends)
    logs_names = ['grid1024 + grid0.019 10',
                  'fps1024 + grid0.019 10',
                  'fps1024 + grid0.019 12',
                  'fps1024 + grid0.019 12',
                  'fps1024 + grid0.019 14']

    # TODO: next: [16, 17, 18, 18, 18]
    # TODO: then: best neighbors with grid

    # TODO: Next exp: radius scaling = 2.5 test different input grid size
    # TODO: Test other architectures lighter and faster??? Can we reach 4000 instances per second at test time?
    # TODO: Add other stuff from convnext like droppath etc

    # TODO:
    #
    #       2. Poisson disk sampling
    #
    #       3. (Border repulsive loss) + (Mix3D) + (model ensemble) and submit to Scannetv2
    #
    #       Need to cite point transformer v2: https://arxiv.org/pdf/2210.05666.pdf
    #
    #       4. Go implement other datasets (NPM3D, Semantic3D, Scannetv2)
    #          Also other task: ModelNet40, ShapeNetPart, SemanticKitti
    #          Add code for completely different tasks??? Invariance??
    #          New classif dataset: ScanObjectNN
    #          Revisiting point cloud classification: A new benchmark dataset 
    #          and classification model on real-world data
    #
    #       5. Parameters to play with at the end
    #           > color drop
    #           > init_feature_dim
    #           > layers
    #           > radius (sphere or cylinder)
    #           > knn
    #           > kp radius (for kp) and K and shells
    #           > trainer
    #           > KPConvX vs KPConvD vs KPInv
    #           > groups in KPConvX
    #           > n_layers in KPConvX
    #


    # safe check log names
    if len(logs) > len(logs_names):
        logs = logs[:len(logs_names)]
    logs_names = np.array(logs_names[:len(logs)])

    # logs = logs[[-1, -3, -4]]
    # logs_names = logs_names[[-1, -3, -4]]

    return logs, logs_names


# ----------------------------------------------------------------------------------------------------------------------
#
#           Main Call
#       \***************/
#


if __name__ == '__main__':

    ##########
    # Clean-up
    ##########

    # Optional. Do it to save space but you will lose some data:
    cleaning = False
    res_path = 'results'
    if cleaning:
        cleanup(res_path, 'Log_2022-09-16_17-04-53', keep_val_ply=False, keep_last_ckpt=False)
        cleanup(res_path, 'Log_2022-10-03_17-16-09')

    ######################################################
    # Choose a list of log to plot together for comparison
    ######################################################

    # My logs: choose the logs to show
    logs, logs_names = test_grid019()

    frame_lines_1(["Plot ScanObjectNN experiments"])

    #################
    # Compare configs
    #################

    # Load all cfg
    all_cfgs = []
    for log in logs:
        all_cfgs.append(load_cfg(log))

    # Verify that we are dealing with S3DIS logs
    for cfg in all_cfgs:
        if cfg.data.name != "ScanObjectNN":
            err_mess = '\nTrying to plot ScanObjectNN experiments, but {:s} was trained on {:s} dataset.'
            raise ValueError(err_mess.format(cfg.exp.date, cfg.data.name))
            
    # Print differences in a nice table
    print_cfg_diffs(logs_names,
                    all_cfgs,
                    # show_params=['model.in_sub_size',
                    #              'train.in_radius'],
                    hide_params=['test.batch_limit',
                                 'train.batch_limit',
                                 'test.batch_size',
                                 'augment_test.height_norm',
                                 'augment_test.chromatic_norm',
                                 'augment_test.chromatic_all',
                                 'augment_test.chromatic_contrast',
                                 'train.max_epoch',
                                 'train.checkpoint_gap',
                                 'train.lr_decays'])


    ################
    # Plot functions
    ################

    print()
    underline("Ploting training info")

    # Plot the training loss and accuracy
    compare_trainings(all_cfgs, logs, logs_names)


    # Test the network or show validation
    perform_test = False
    if perform_test:

        print()
        underline("Test networks")
        print()

        # Plot the validation
        compare_on_test_set(all_cfgs, logs, logs_names)


    else:
        print()
        underline("Ploting validation info")
        print()

        # Plot the validation
        compare_convergences_classif(all_cfgs, logs, logs_names)

