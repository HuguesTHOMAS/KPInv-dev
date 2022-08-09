
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
from utils.metrics import IoU_from_confusions, smooth_metrics, fast_confusion
from utils.ply import read_ply


# My libs
from utils.printing import frame_lines_1, underline
from utils.metrics import IoU_from_confusions, smooth_metrics, fast_confusion
from utils.ply import read_ply, write_ply
from utils.config import load_cfg

from utils.plot_utilities import listdir_str, print_cfg_diffs, compare_trainings, compare_convergences_segment


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


def exp_neighbors_cropping():
    """
    This experiment is a study of the impact of cropping radius neighbors with neighbors limits.
    It is not clear yet what gives the best score, we will need futher study. 
    In any case, lower neighbor limits means faster network.
    """

    # Using the dates of the logs, you can easily gather consecutive ones. All logs should be of the same dataset.
    start = 'Log_2022-08-04_22-43-08'
    end = 'Log_2022-08-06_23-43-08'

    # Name of the result path
    res_path = 'results'

    # Gather logs and sort by date
    logs = np.sort([join(res_path, l) for l in listdir_str(res_path) if start <= l <= end])

    # Give names to the logs (for plot legends)
    logs_names = ['Full_neigh',
                  'Less neighb',
                  'Low neighb',
                  'Very low neighb',
                  'etc']

    # safe check log names
    logs_names = np.array(logs_names[:len(logs)])

    return logs, logs_names


def exp_architecture():
    """
    In this experiment we have a first try at deeper architectures.
    They are clearly better. We will need to go further and see if we can duplicate resnet101 
    and other very deep architectures.
    """

    # Using the dates of the logs, you can easily gather consecutive ones. All logs should be of the same dataset.
    start = 'Log_2022-08-06_22-43-08'
    end = 'Log_2022-08-07_03-43-08'

    # Name of the result path
    res_path = 'results'

    # Gather logs and sort by date
    logs = np.sort([join(res_path, l) for l in listdir_str(res_path) if start <= l <= end])

    # Optionally add a specific log at a specific place in the log list
    logs = logs.astype('<U50')
    logs = np.insert(logs, 1, 'results/Log_2022-08-05_14-14-50')

    # Give names to the logs (for plot legends)
    logs_names = ['Small net',
                  'Med net',
                  'Big net',
                  'etc']

    # safe check log names
    logs_names = np.array(logs_names[:len(logs)])

    return logs, logs_names


def exp_smaller_conv_radius():
    """
    Using a smaller convolution radius increases network speed but not in a direct manner.
    It just means that the corresponding neighbors limits are smaller, which is the reason why the network is faster.
    But convolution radius is crucial to control the alignment between conv points and subsample input neighbors.
    We compare our medium conv size (roughly a 5x5 conv) to a smaller conv size more similar to 3x3 convolutions.
    """

    # Using the dates of the logs, you can easily gather consecutive ones. All logs should be of the same dataset.
    start = 'Log_2022-08-07_00-27-37'
    end = 'Log_2022-08-08_17-53-44'

    # Name of the result path
    res_path = 'results'

    # Gather logs and sort by date
    logs = np.sort([join(res_path, l) for l in listdir_str(res_path) if start <= l <= end])

    # Give names to the logs (for plot legends)
    logs_names = ['BigNet - conv=2.5',
                  'MedNet - conv=1.9',
                  'BigNet - conv=1.9',
                  'CustomNet - conv=1.9',
                  'etc']

    # safe check log names
    logs_names = np.array(logs_names[:len(logs)])

    return logs, logs_names


def exp_kpinv():
    """
    Using a smaller convolution radius increases network speed but not in a direct manner.
    It just means that the corresponding neighbors limits are smaller, which is the reason why the network is faster.
    But convolution radius is crucial to control the alignment between conv points and subsample input neighbors.
    We compare our medium conv size (roughly a 5x5 conv) to a smaller conv size more similar to 3x3 convolutions.
    """

    # Using the dates of the logs, you can easily gather consecutive ones. All logs should be of the same dataset.
    start = 'Log_2022-08-08_17-53-43'
    end = 'Log_2022-08-09_23-43-08'

    # Name of the result path
    res_path = 'results'

    # Gather logs and sort by date
    logs = np.sort([join(res_path, l) for l in listdir_str(res_path) if start <= l <= end])

    # Give names to the logs (for plot legends)
    logs_names = ['kpconv',
                  'kpinv',
                  'etc']

    # safe check log names
    logs_names = np.array(logs_names[:len(logs)])

    return logs, logs_names


def test_input_pipeline():
    """
    Sort these runs
    """

    # Using the dates of the logs, you can easily gather consecutive ones. All logs should be of the same dataset.
    start = 'Log_2022-08-04_22-43-08'
    end = 'Log_2022-08-09_23-43-08'

    # Name of the result path
    res_path = 'results'

    # Gather logs and sort by date
    logs = np.sort([join(res_path, l) for l in listdir_str(res_path) if start <= l <= end])

    # Give names to the logs (for plot legends)
    logs_names = ['Full_neigh',
                  'Less neighb',
                  'Low neighb',
                  'Very low neighb',
                  'Small net',
                  'Big net',
                  'Med net / smaller conv',
                  'Big net / smaller conv',
                  'Mega net / smaller conv',
                  'etc']

    # safe check log names
    logs_names = np.array(logs_names[:len(logs)])

    return logs, logs_names



# ----------------------------------------------------------------------------------------------------------------------
#
#           Main Call
#       \***************/
#

if __name__ == '__main__':

    ######################################################
    # Choose a list of log to plot together for comparison
    ######################################################

    # My logs: choose the logs to show
    logs, logs_names = exp_kpinv()

    frame_lines_1(["Plot S3DIS experiments"])

    #################
    # Compare configs
    #################

    # Load all cfg
    all_cfgs = []
    for log in logs:
        all_cfgs.append(load_cfg(log))

    # Verify that we are dealing with S3DIS logs
    for cfg in all_cfgs:
        if cfg.data.name != "S3DIS":
            err_mess = '\nTrying to plot S3DIS experiments, but {:s} was trained on {:s} dataset.'
            raise ValueError(err_mess.format(cfg.exp.date, cfg.data.name))
            
    # Print differences in a nice table
    print_cfg_diffs(logs_names,
                    all_cfgs,
                    # show_params=['model.init_sub_size',
                    #              'train.in_radius'],
                    hide_params=['test.batch_limit',
                                 'train.batch_limit'])


    ################
    # Plot functions
    ################

    print()
    underline("Ploting training info")

    # Plot the training loss and accuracy
    compare_trainings(all_cfgs, logs, logs_names)

    print()
    underline("Ploting validation info")

    # Plot the validation
    compare_convergences_segment(all_cfgs, logs, logs_names)

