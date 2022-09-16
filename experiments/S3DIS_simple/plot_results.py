
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

from utils.plot_utilities import listdir_str, print_cfg_diffs, compare_trainings, compare_convergences_segment, compare_on_test_set


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
                  'etc']

    # safe check log names
    logs_names = np.array(logs_names[:len(logs)])

    return logs, logs_names


def exp_training_strat():
    """
    Here we look at what Cyclic learning rate and AdamW can bring to the table.
    Also what happens if we use accumulation to simulate very large batch (but keep 
    the same amount of data seen by reducing the number of epoch steps).
    """

    # Using the dates of the logs, you can easily gather consecutive ones. All logs should be of the same dataset.
    start = 'Log_2022-08-10_16-59-52'
    end = 'Log_2022-08-19_23-43-08'

    # Name of the result path
    res_path = 'results'

    # Gather logs and sort by date
    logs = np.sort([join(res_path, l) for l in listdir_str(res_path) if start <= l <= end])

    # Optionally add a specific log at a specific place in the log list
    logs = logs.astype('<U50')
    logs = np.insert(logs, 0, 'results/Log_2022-08-07_00-27-37')
    
    # Give names to the logs (for plot legends)
    logs_names = ['B=10_Accum=1_SGD',
                  'B=10_Accum=1_Cyclic_AdamW',
                  'B=10_Accum=5_Cyclic_AdamW',]

    # safe check log names
    if len(logs) > len(logs_names):
        logs = logs[:len(logs_names)]
    logs_names = np.array(logs_names[:len(logs)])

    return logs, logs_names


def exp_LR_range_test():
    """
    Here we fo a LR range test. REsult the biggest lr should be 1e-2
    """

    # Using the dates of the logs, you can easily gather consecutive ones. All logs should be of the same dataset.
    start = 'Log_2022-08-11_15-10-04'
    end = 'Log_2022-08-19_23-43-08'

    # Name of the result path
    res_path = 'results'

    # Gather logs and sort by date
    logs = np.sort([join(res_path, l) for l in listdir_str(res_path) if start <= l <= end])
    
    # Give names to the logs (for plot legends)
    logs_names = ['LR_range_test']

    # safe check log names
    if len(logs) > len(logs_names):
        logs = logs[:len(logs_names)]
    logs_names = np.array(logs_names[:len(logs)])

    return logs, logs_names


def exp_training_strat2():
    """
    Here we look at what Cyclic learning rate and AdamW can bring to the table.
    Also what happens if we use accumulation to simulate very large batch (but keep 
    the same amount of data seen by reducing the number of epoch steps).
    """

    # Using the dates of the logs, you can easily gather consecutive ones. All logs should be of the same dataset.
    start = 'Log_2022-08-12_12-49-02'
    end = 'Log_2022-08-19_23-43-08'

    # Name of the result path
    res_path = 'results'

    # Gather logs and sort by date
    logs = np.sort([join(res_path, l) for l in listdir_str(res_path) if start <= l <= end])

    # Optionally add a specific log at a specific place in the log list
    logs = logs.astype('<U50')
    logs = np.insert(logs, 0, 'results/Log_2022-08-10_17-03-50')
    
    # Give names to the logs (for plot legends)
    logs_names = ['e-4:20:e-2:40:-50(old)',
                  'e-4: 5:e-2:30:-50',
                  'e-4: 5:e-2:30:-80']

    # safe check log names
    if len(logs) > len(logs_names):
        logs = logs[:len(logs_names)]
    logs_names = np.array(logs_names[:len(logs)])

    return logs, logs_names


def exp_weight_decay():
    """
    w=0.01 is the best value.
    """

    # Using the dates of the logs, you can easily gather consecutive ones. All logs should be of the same dataset.
    start = 'Log_2022-08-12_12-49-29'
    end = 'Log_2022-08-19_23-43-08'

    # Name of the result path
    res_path = 'results'

    # Gather logs and sort by date
    logs = np.sort([join(res_path, l) for l in listdir_str(res_path) if start <= l <= end])
    
    # Give names to the logs (for plot legends)
    logs_names = ['w=0.01',
                  'w=0.1',
                  'w=0.001',
                  'w=0.0001',
                  'w=1.0',]

    # safe check log names
    if len(logs) > len(logs_names):
        logs = logs[:len(logs_names)]
    logs_names = np.array(logs_names[:len(logs)])

    # Reorder
    logs = logs[[3, 2, 0, 1, 4]]
    logs_names = logs_names[[3, 2, 0, 1, 4]]

    return logs, logs_names


def exp_deformable_modulations():
    """
     
    """

    # Using the dates of the logs, you can easily gather consecutive ones. All logs should be of the same dataset.
    start = 'Log_2022-08-15_11-48-16'
    end = 'Log_2022-08-19_23-43-08'

    # Name of the result path
    res_path = 'results'

    # Gather logs and sort by date
    logs = np.sort([join(res_path, l) for l in listdir_str(res_path) if start <= l <= end])

    
    # Optionally add a specific log at a specific place in the log list
    logs = logs.astype('<U50')
    logs = np.insert(logs, 0, 'results/Log_2022-08-12_12-49-29')
    
    # Give names to the logs (for plot legends)
    logs_names = ['baseline',
                  'KPConv',
                  'KPConv-mod',
                  'KPDef',
                  'KPDef-mod',
                  'KPInv',
                  'KPConv-mod-again',
                  'KPConv-mod-bigger-batch',
                  'KPConv-mod-G4',
                  'KPConv-mod-G8']

    # safe check log names
    if len(logs) > len(logs_names):
        logs = logs[:len(logs_names)]
    logs_names = np.array(logs_names[:len(logs)])

    return logs, logs_names


def exp_kp_groups():
    """
     
    """

    # Using the dates of the logs, you can easily gather consecutive ones. All logs should be of the same dataset.
    start = 'Log_2022-08-18_16-24-47'
    end = 'Log_2022-08-19_23-43-08'

    # Name of the result path
    res_path = 'results'

    # Gather logs and sort by date
    logs = np.sort([join(res_path, l) for l in listdir_str(res_path) if start <= l <= end])

    
    # Optionally add a specific log at a specific place in the log list
    logs = logs.astype('<U50')
    logs = np.insert(logs, 0, 'results/Log_2022-08-17_16-52-30')
    
    # Give names to the logs (for plot legends)
    logs_names = ['KPConv-mod-G1',
                  'KPConv-mod-G4',
                  'KPConv-mod-G8']

    # safe check log names
    if len(logs) > len(logs_names):
        logs = logs[:len(logs_names)]
    logs_names = np.array(logs_names[:len(logs)])

    return logs, logs_names


def exp_transformer():
    """
    Tested and did not work:
    - trans-no_smax
    - trans-ReLU
    """

    # Using the dates of the logs, you can easily gather consecutive ones. All logs should be of the same dataset.
    start = 'Log_2022-08-20_20-11-45'
    end = 'Log_2022-08-29_23-43-08'

    # Name of the result path
    res_path = 'results'

    # Gather logs and sort by date
    logs = np.sort([join(res_path, l) for l in listdir_str(res_path) if start <= l <= end])

    
    # Optionally add a specific log at a specific place in the log list
    logs = logs.astype('<U50')
    logs = np.insert(logs, 0, 'results/Log_2022-08-17_16-52-30')
    
    # Give names to the logs (for plot legends)
    logs_names = ['baseline',
                  'trans-G1-1-sconv(fast)',
                  'trans-G8-1-sconv(fast)',
                  'trans-G1-1-sconv-no_k',
                  'inv_3-G1-1-sconv-smax',]

    # safe check log names
    if len(logs) > len(logs_names):
        logs = logs[:len(logs_names)]
    logs_names = np.array(logs_names[:len(logs)])

    return logs, logs_names


def exp_involution():
    """
    Tested and did not work:
    - trans-no_smax
    - trans-ReLU
    - inv_3-none
    - inv_3-ReLU
    """

    # Using the dates of the logs, you can easily gather consecutive ones. All logs should be of the same dataset.
    start = 'Log_2022-08-23_18-48-07'
    end = 'Log_2022-08-29_23-43-08'

    # Name of the result path
    res_path = 'results'

    # Gather logs and sort by date
    logs = np.sort([join(res_path, l) for l in listdir_str(res_path) if start <= l <= end])

    
    # Give names to the logs (for plot legends)
    logs_names = ['inv_3-G1-1-sconv-smax',
                  'inv_3-G1-1-sconv-sigm',
                  'inv_2-G1-1-sconv(smax)',
                  'inv_1-G1-1-sconv(smax)',]


    # TODO:
    # "--kp_mode kpconv-geom"
    # "--kp_mode kpconv-geom --neighbor_limits 10 10"
    # "--kp_mode kpconv-mod"
    # "--kp_mode kpconv-mod --neighbor_limits 10 10"
    # "--kp_mode kpconv --neighbor_limits 10 10"
    # Separate in different exp and comment on the last ones

    # safe check log names
    if len(logs) > len(logs_names):
        logs = logs[:len(logs_names)]
    logs_names = np.array(logs_names[:len(logs)])

    return logs, logs_names


def exp_old_params():
    """
    Adding geom nearly doubles the processing time, but improves the performances. Although we do not reach the same score as before
    """

    # Using the dates of the logs, you can easily gather consecutive ones. All logs should be of the same dataset.
    start = 'Log_2022-08-25_16-33-45'
    end = 'Log_2022-08-29_23-43-08'

    # Name of the result path
    res_path = 'results'

    # Gather logs and sort by date
    logs = np.sort([join(res_path, l) for l in listdir_str(res_path) if start <= l <= end])

    # # Optionally add a specific log at a specific place in the log list
    # logs = logs.astype('<U50')
    # logs = np.insert(logs, 0, 'results/Log_2022-08-23_08-43-18')

    # Give names to the logs (for plot legends)
    logs_names = ['kpconv-1.5(16)-bignet (no_rot)',
                  'kpconv-2.5-bignet (no_rot)',
                  'kpconv-1.5(16)-bignet-rot',
                  'kpconv-1.5(16)-bignet-chroma',
                  'kpconv-1.5(16)-bignet-cylinder',
                  'kpconv-1.5(16)-bignet-cube']

    # safe check log names
    if len(logs) > len(logs_names):
        logs = logs[:len(logs_names)]
    logs_names = np.array(logs_names[:len(logs)])

    return logs, logs_names


def exp_kpconv_geom():
    """
    Adding geom nearly doubles the processing time, but improves the performances. Although we do not reach the same score as before
    """

    # Using the dates of the logs, you can easily gather consecutive ones. All logs should be of the same dataset.
    start = 'Log_2022-08-27_07-06-54'
    end = 'Log_2022-09-29_23-43-08'

    # Name of the result path
    res_path = 'results'

    # Gather logs and sort by date
    logs = np.sort([join(res_path, l) for l in listdir_str(res_path) if start <= l <= end])

    # # Optionally add a specific log at a specific place in the log list
    # logs = logs.astype('<U50')
    # logs = np.insert(logs, 0, 'results/Log_2022-08-23_08-43-18')

    # Give names to the logs (for plot legends)
    logs_names = ['kpconv-1.5(16)-bignet-bloc',
                  'kpconv-1.5(16)-bignet-blocsmaller',
                  'small-no-height_norm',
                  'small-kpconv-geom',
                  'small-kpconv-mod-geom',
                  'kpconv-2.5-big-bloc',
                  'kpconv-2.5-big-spheres',
                  'transformer-med-spheres',
                  'kpconv-2.5-med-spheres',
                  'kpconv-2.5-med-blocs']


    # safe check log names
    if len(logs) > len(logs_names):
        logs = logs[:len(logs_names)]
    logs_names = np.array(logs_names[:len(logs)])

    return logs, logs_names


def exp_drop_before_norm():
    """
    Adding geom nearly doubles the processing time, but improves the performances. Although we do not reach the same score as before
    """

    # Using the dates of the logs, you can easily gather consecutive ones. All logs should be of the same dataset.
    start = 'Log_2022-08-31_07-06-54'
    end = 'Log_2022-09-29_23-43-08'

    # Name of the result path
    res_path = 'results'

    # Gather logs and sort by date
    logs = np.sort([join(res_path, l) for l in listdir_str(res_path) if start <= l <= end])

    # Optionally add a specific log at a specific place in the log list
    logs = logs.astype('<U50')
    logs = np.insert(logs, 0, 'results/Log_2022-08-30_18-57-14')

    # Give names to the logs (for plot legends)
    logs_names = ['transformer-med-spheres',
                  'transformer-cyl-norm-drop',
                  'transformer-cyl-drop-norm',
                  'trans-cyl-fps_0.04',
                  'trans-cyl-fps_-4',
                  'kpnext-L_2.5-cyl',
                  'kpnext-L_3.5-cyl',
                  '...',]

    # safe check log names
    if len(logs) > len(logs_names):
        logs = logs[:len(logs_names)]
    logs_names = np.array(logs_names[:len(logs)])

    return logs, logs_names


def exp_shell_optim():
    """
    Test shell conv
    """

    # Using the dates of the logs, you can easily gather consecutive ones. All logs should be of the same dataset.
    start = 'Log_2022-09-02_17-49-22'
    end = 'Log_2022-09-29_23-43-08'

    # Name of the result path
    res_path = 'results'

    # Gather logs and sort by date
    logs = np.sort([join(res_path, l) for l in listdir_str(res_path) if start <= l <= end])

    # # Optionally add a specific log at a specific place in the log list
    # logs = logs.astype('<U50')
    # logs = np.insert(logs, 0, 'results/Log_2022-08-30_18-57-14')

    # Give names to the logs (for plot legends)
    logs_names = ['r=2.5 [1 14]',
                  'r=2.5 [1 14 43]',
                  'r=2.9 [1 14 43]',
                  'abl: kp=nearest',
                  'abl: data=random',
                  'abl: b=layer',
                  'r=2.9 [1 14 43] sig=0.7',
                  'abl: C0=80',
                  'abl: kp=nearest',
                  'abl: kp=nearest 1 14',
                  'abl: kp=nearest 1 14 28',
                  'abl: kp=nearest 1 14 28 r=2.5 f64',
                  '...',
                  '...',
                  '...',]

    # safe check log names
    if len(logs) > len(logs_names):
        logs = logs[:len(logs_names)]
    logs_names = np.array(logs_names[:len(logs)])

    return logs, logs_names


def small_test():
    """
    Test shell conv
    """

    # Using the dates of the logs, you can easily gather consecutive ones. All logs should be of the same dataset.
    start = 'Log_2022-09-08_18-12-24'
    end = 'Log_2022-09-29_23-43-08'

    # Name of the result path
    res_path = 'results'

    # Gather logs and sort by date
    logs = np.sort([join(res_path, l) for l in listdir_str(res_path) if start <= l <= end])

    # # Optionally add a specific log at a specific place in the log list
    # logs = logs.astype('<U50')
    # logs = np.insert(logs, 0, 'results/Log_2022-08-30_18-57-14')

    # Give names to the logs (for plot legends)
    logs_names = ['kp=nearest 1 14 28 r=2.5 f64',
                  'same',
                  'same random']


    # safe check log names
    if len(logs) > len(logs_names):
        logs = logs[:len(logs_names)]
    logs_names = np.array(logs_names[:len(logs)])

    return logs, logs_names


def exp_kpmini():
    """
    Test shell conv
    """

    # Using the dates of the logs, you can easily gather consecutive ones. All logs should be of the same dataset.
    start = 'Log_2022-09-09_18-42-25'
    end = 'Log_2022-09-29_23-43-08'

    # Name of the result path
    res_path = 'results'

    # Gather logs and sort by date
    logs = np.sort([join(res_path, l) for l in listdir_str(res_path) if start <= l <= end])

    # # Optionally add a specific log at a specific place in the log list
    # logs = logs.astype('<U50')
    # logs = np.insert(logs, 0, 'results/Log_2022-08-30_18-57-14')

    # Give names to the logs (for plot legends)
    logs_names = ['mini add sum',
                  'mini add max',
                  'mini mul max',
                  'mini mul sum (we keep this)',
                  'mini 1 14',
                  'mini r=3.0 / 1 14 30 60',
                  'mini mlp',
                  'mini Cmid=8',
                  '...',]

    # safe check log names
    if len(logs) > len(logs_names):
        logs = logs[:len(logs_names)]
    logs_names = np.array(logs_names[:len(logs)])

    return logs, logs_names


def exp_kpinv():
    """
    Test shell conv
    """

    # Using the dates of the logs, you can easily gather consecutive ones. All logs should be of the same dataset.
    start = 'Log_2022-09-12_15-06-06'
    end = 'Log_2022-09-29_23-43-08'

    # Name of the result path
    res_path = 'results'

    # Gather logs and sort by date
    logs = np.sort([join(res_path, l) for l in listdir_str(res_path) if start <= l <= end])

    # # Optionally add a specific log at a specific place in the log list
    logs = logs.astype('<U50')
    logs = np.insert(logs, 0, 'results/Log_2022-09-11_20-53-44')

    # Give names to the logs (for plot legends)
    logs_names = ['kpmini',
                  'kpminiX',
                  'kpinvX E=8 none',
                  'kpinv  G=1 none',
                  'kpminiX_bis',
                  'kpinv  G=1 sigm',
                  'kpinv  G=1 sigm layernorm',
                  'kpmini layernorm',
                  'kpinv  G=1 tanh layernorm',
                  'kpinv  G=1 tanh groupnorm',
                  'kpinv  G=1 tanh nonorm',
                  'TODO',
                  'kpinvX E=8',
                  'kpinv  G=1',
                  'kpinv  G=8',
                  'kpinv  CpG=8',
                  'kpinv  CpG=1',
                  'best-sigm'
                  'best-tanh'
                  'best-smax'
                  'best-sigm+max_aggr' # TODO: see pdf for sparse attention
                  '...',]

    # TODO PLOTS: Show model size and GPU consumption

    # TODO: train regular is buggy
    # TODO: Handle kpnextarchitecture like ConvNext, operate DropPath


    # safe check log names
    if len(logs) > len(logs_names):
        logs = logs[:len(logs_names)]
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
                    # show_params=['model.in_sub_size',
                    #              'train.in_radius'],
                    hide_params=['test.batch_limit',
                                 'train.batch_limit',
                                 'test.batch_size',
                                 'augment_test.height_norm',
                                 'augment_test.chromatic_norm',
                                 'augment_test.chromatic_all',
                                 'augment_test.chromatic_contrast',
                                 'model.neighbor_limits',
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

