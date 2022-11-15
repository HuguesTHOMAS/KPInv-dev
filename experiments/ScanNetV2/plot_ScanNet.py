
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

from utils.plot_utilities import listdir_str, print_cfg_diffs, compare_trainings, compare_convergences_segment, \
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


def test_initial():
    """
    GOGO KPNext experiments
    """

    # Using the dates of the logs, you can easily gather consecutive ones. All logs should be of the same dataset.
    start = 'Log_2022-11-08_13-13-49'
    end = 'Log_2022-11-09_11-32-59'

    # Name of the result path
    res_path = 'results'

    # Gather logs and sort by date
    logs = np.sort([join(res_path, l) for l in listdir_str(res_path) if start <= l <= end])

    # Optionally add a specific log at a specific place in the log list
    logs = logs.astype('<U50')
    logs = np.insert(logs, 0, 'results/Log_2022-11-07_09-57-34')

    # Give names to the logs (for plot legends)
    logs_names = ['test 0.04/2.0',
                  'test 0.02/1.2',
                  'test 0.02/1.2',
                  'test 0.02/2.0 f48',
                  'bis  0.02/2.0']

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
    logs, logs_names = test_initial()

    frame_lines_1(["Plot Scannetv2 experiments"])

    #################
    # Compare configs
    #################

    # Load all cfg
    all_cfgs = []
    for log in logs:
        all_cfgs.append(load_cfg(log))

    # Verify that we are dealing with Scannetv2 logs
    for cfg in all_cfgs:
        if cfg.data.name != "ScanNetV2":
            err_mess = '\nTrying to plot Scannetv2 experiments, but {:s} was trained on {:s} dataset.'
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
        compare_convergences_segment(all_cfgs, logs, logs_names)

