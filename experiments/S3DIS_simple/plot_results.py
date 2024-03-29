
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
                  '...',]


    # safe check log names
    if len(logs) > len(logs_names):
        logs = logs[:len(logs_names)]
    logs_names = np.array(logs_names[:len(logs)])

    return logs, logs_names


def exp_kpinv_bis():
    """
    Here we just realise that we hade a bad momentum for batch norm
    """

    # Using the dates of the logs, you can easily gather consecutive ones. All logs should be of the same dataset.
    start = 'Log_2022-09-16_17-04-24'
    end = 'Log_2022-09-29_23-43-08'

    # Name of the result path
    res_path = 'results'

    # Gather logs and sort by date
    logs = np.sort([join(res_path, l) for l in listdir_str(res_path) if start <= l <= end])

    # # Optionally add a specific log at a specific place in the log list
    logs = logs.astype('<U50')
    logs = np.insert(logs, 0, 'results/Log_2022-09-11_20-53-44')

    # Give names to the logs (for plot legends)
    logs_names = ['kpmini-old',
                  'kpmini batchnorm 0.98',
                  'kpinv  batchnorm 0.98',
                  'kpmini batchnorm 0.1',
                  'kpinv  batchnorm 0.1',]


    # safe check log names
    if len(logs) > len(logs_names):
        logs = logs[:len(logs_names)]
    logs_names = np.array(logs_names[:len(logs)])

    return logs, logs_names


def exp_kpinv_tris():
    """
    Here we eventually test KPmini and KPInv with good parameters and they perform quite well!
    """

    # Using the dates of the logs, you can easily gather consecutive ones. All logs should be of the same dataset.
    start = 'Log_2022-09-17_02-15-39'
    end = 'Log_2022-09-29_23-43-08'

    # Name of the result path
    res_path = 'results'

    # Gather logs and sort by date
    logs = np.sort([join(res_path, l) for l in listdir_str(res_path) if start <= l <= end])

    # Give names to the logs (for plot legends)
    logs_names = ['kpmini-old',
                  'kpinv-old',
                  'kpinv grpnorm (G=1)',
                  'kpinv no grpnorm ',
                  'kpinvx',
                  'kpminix',
                  'kpmini',
                  'kpinv  G=8',
                  'kpinv  CpG=8',
                  'kpinv  CpG=1',]


    # safe check log names
    if len(logs) > len(logs_names):
        logs = logs[:len(logs_names)]
    logs_names = np.array(logs_names[:len(logs)])

    return logs, logs_names


def retry_kpmini():
    """
    Here we eventually test KPmini and KPInv with good parameters and they perform quite well!
    """

    # Using the dates of the logs, you can easily gather consecutive ones. All logs should be of the same dataset.
    start = 'Log_2022-09-19_10-09-44'
    end = 'Log_2022-09-29_23-43-08'

    # Name of the result path
    res_path = 'results'

    # Gather logs and sort by date
    logs = np.sort([join(res_path, l) for l in listdir_str(res_path) if start <= l <= end])

    # Optionally add a specific log at a specific place in the log list
    logs = logs.astype('<U50')
    logs = np.insert(logs, 0, 'results/Log_2022-09-18_21-45-51')

    # Give names to the logs (for plot legends)
    logs_names = ['kpmini mulsum',
                  'kpmini addmax',
                  'kpmini mulmax',
                  'kpmini mlp1',
                  'kpmini mlp2']


    # TODO: train regular is buggy
    # TODO: Handle kpnextarchitecture like ConvNext, operate DropPath


    # safe check log names
    if len(logs) > len(logs_names):
        logs = logs[:len(logs_names)]
    logs_names = np.array(logs_names[:len(logs)])

    return logs, logs_names


def test_kptransformer():
    """
    test of kp transformer
    """

    # Using the dates of the logs, you can easily gather consecutive ones. All logs should be of the same dataset.
    start = 'Log_2022-09-19_16-38-38'
    end = 'Log_2022-09-29_23-43-08'

    # Name of the result path
    res_path = 'results'

    # Gather logs and sort by date
    logs = np.sort([join(res_path, l) for l in listdir_str(res_path) if start <= l <= end])

    # Optionally add a specific log at a specific place in the log list
    logs = logs.astype('<U50')
    logs = np.insert(logs, 0, 'results/Log_2022-09-18_21-45-51')

    # Give names to the logs (for plot legends)
    logs_names = ['kpmini(autoH)',
                  'kptran(16) nolinear',
                  'kptran(16) qk_linear',
                  'kptran(16) all_linear',
                  'kptran(16) all_linear G8',
                  'kptran(16) onlymini',  # = kpmini(16),
                  'kptran(16) onlytran',  # = trans no geom,
                  '...',]


    # TODO: train regular is buggy
    # TODO: Handle kpnextarchitecture like ConvNext, operate DropPath


    # safe check log names
    if len(logs) > len(logs_names):
        logs = logs[:len(logs_names)]
    logs_names = np.array(logs_names[:len(logs)])

    return logs, logs_names


def test_kpminimod():
    """
    test of kp transformer
    """

    # Using the dates of the logs, you can easily gather consecutive ones. All logs should be of the same dataset.
    start = 'Log_2022-09-20_12-16-59'
    end = 'Log_2022-09-29_23-43-08'

    # Name of the result path
    res_path = 'results'

    # Gather logs and sort by date
    logs = np.sort([join(res_path, l) for l in listdir_str(res_path) if start <= l <= end])

    # Optionally add a specific log at a specific place in the log list
    logs = logs.astype('<U50')
    logs = np.insert(logs, 0, 'results/Log_2022-09-18_21-45-51')

    # Give names to the logs (for plot legends)
    logs_names = ['kpmini',
                  'transformer (G1)',
                  'kpminimod (16)',
                  'kpminimod (-1)',
                  'transformer (G8)',
                  'kpminimod (1) sigm',
                  'kpminimod (1) none/grpnorm',
                  'kpminimod2 (8) sigm',  # alpha layer = 2
                  'kpminimod2 (8) smax',
                  'kpminimod2 (8) tanh',
                  'kpminimod2 (8) gpnorm-smax',
                  'kpminimod2 (8) gpnorm-sigm',
                  'kpminimod2 (8) gpnorm-tanh',
                  '...',]

    # list of exp during Cuba
    #           > number of groups in minimod => 8 seems good
    #           > attention activation => sigmoid seems to have best results
    #           > with grpnorm => Yes with sigmoid
    #           > 2 layer alpha => seems better


    # safe check log names
    if len(logs) > len(logs_names):
        logs = logs[:len(logs_names)]
    logs_names = np.array(logs_names[:len(logs)])

    # logs = logs[[0, 7, 9, 11, 12]]
    # logs_names = logs_names[[0, 7, 9, 11, 12]]

    return logs, logs_names


def test_kpminimod_2():
    """
    test of kp transformer
    """

    # Using the dates of the logs, you can easily gather consecutive ones. All logs should be of the same dataset.
    start = 'Log_2022-09-27_15-24-13'
    end = 'Log_2022-09-29_23-43-08'

    # Name of the result path
    res_path = 'results'

    # Gather logs and sort by date
    logs = np.sort([join(res_path, l) for l in listdir_str(res_path) if start <= l <= end])

    # Optionally add a specific log at a specific place in the log list
    logs = logs.astype('<U50')
    logs = np.insert(logs, 0, 'results/Log_2022-09-27_10-06-23')

    # Give names to the logs (for plot legends)
    logs_names = ['kpminimod2 (8) gpnorm-sigm',
                  'kpminimod2 (4) gpnorm-sigm',
                  'kpminimod2 (16) gpnorm-sigm',
                  'kpminimod2 (-1) gpnorm-sigm',
                  'kpminimod2 (-1) BN-sigm',
                  'kpminimod2 (8) BN-sigm',
                  'kpminimod2 (8) BN-sigm + alphaBN',
                  'again kpminimod2 (8) gpnorm-sigm']

    # list of exp during Cuba
    #           > number of groups in minimod => 8 seems good
    #           > attention activation => sigmoid seems to have best results
    #           > with grpnorm => Yes with sigmoid
    #           > 2 layer alpha => seems better

    # safe check log names
    if len(logs) > len(logs_names):
        logs = logs[:len(logs_names)]
    logs_names = np.array(logs_names[:len(logs)])

    # logs = logs[[0, 7, 9, 11, 12]]
    # logs_names = logs_names[[0, 7, 9, 11, 12]]

    return logs, logs_names


def test_in_full_rot():
    """
    test of kp transformer
    """

    # Using the dates of the logs, you can easily gather consecutive ones. All logs should be of the same dataset.
    start = 'Log_2022-09-28_07-21-15'
    end = 'Log_2022-12-29_23-43-08'

    # Name of the result path
    res_path = 'results'

    # Gather logs and sort by date
    logs = np.sort([join(res_path, l) for l in listdir_str(res_path) if start <= l <= end])

    # # Optionally add a specific log at a specific place in the log list
    # logs = logs.astype('<U50')
    # logs = np.insert(logs, 0, 'results/Log_2022-09-27_10-06-23')

    # Give names to the logs (for plot legends)
    logs_names = ['kpminimod2 G8  f64 fast',
                  'kpminimod2 G8  f64',
                  'kpminimod2 G16 f64',
                  'kpminimod2 G32 f64',
                  'kpminimod2 G4  f64',
                  'kpminimod2 G-16  f64',
                  'kpminimod2 G-8  f64',
                  'kpminimod2 G-1  f64',
                  'kpminimod2 G1  f64',
                  'kpmini',
                  'kpminimod2 G16 f64',
                  'kpminimod2 G8  f64',
                  'kpminimod2 G8  f64 tanh',
                  'kpminimod2 G8  f96',
                  'kpminimod2 G8  f64 bottleneck/2',
                  'kpminimod1 G8  f64 bottleneck/2',]

    # TODO: list of exp in full rot
    #           > number of groups in minimod
    #           > attention activation
    #           > with grpnorm
    #           > 2 layer alpha
    #           > different radiuses with [1, 14]
    #           > different radiuses with [1, 14, 28]

    # TODO: train regular is buggy
    # TODO: Handle kpnextarchitecture like ConvNext, operate DropPath


    # safe check log names
    if len(logs) > len(logs_names):
        logs = logs[:len(logs_names)]
    logs_names = np.array(logs_names[:len(logs)])

    # logs = logs[[0, 7, 9, 11, 12]]
    # logs_names = logs_names[[0, 7, 9, 11, 12]]

    return logs, logs_names


def test_conv_r():
    """
    test of kp transformer
    """

    # Using the dates of the logs, you can easily gather consecutive ones. All logs should be of the same dataset.
    start = 'Log_2022-10-02_04-31-37'
    end = 'Log_2022-12-29_23-43-08'

    # Name of the result path
    res_path = 'results'

    # Gather logs and sort by date
    logs = np.sort([join(res_path, l) for l in listdir_str(res_path) if start <= l <= end])

    # # Optionally add a specific log at a specific place in the log list
    # logs = logs.astype('<U50')
    # logs = np.insert(logs, 0, 'results/Log_2022-09-27_10-06-23')

    # Give names to the logs (for plot legends)
    logs_names = ['kpminimod2 r=1.0',
                  'kpminimod2 r=1.2',
                  'kpminimod2 r=1.4',
                  'kpminimod2 r=1.6',
                  'kpminimod2 r=1.8',
                  'kpminimod2 r=2.0']


    # safe check log names
    if len(logs) > len(logs_names):
        logs = logs[:len(logs_names)]
    logs_names = np.array(logs_names[:len(logs)])

    # logs = logs[[0, 7, 9, 11, 12]]
    # logs_names = logs_names[[0, 7, 9, 11, 12]]

    return logs, logs_names


def test_input_fixed():
    """
    FROM HERE WE CORRECTED AN ISSUE WITH Z COORDINATE FEATURE.
    """

    # Using the dates of the logs, you can easily gather consecutive ones. All logs should be of the same dataset.
    start = 'Log_2022-10-03_17-16-10'
    end = 'Log_2022-12-29_23-43-08'

    # Name of the result path
    res_path = 'results'

    # Gather logs and sort by date
    logs = np.sort([join(res_path, l) for l in listdir_str(res_path) if start <= l <= end])

    # # Optionally add a specific log at a specific place in the log list
    # logs = logs.astype('<U50')
    # logs = np.insert(logs, 0, 'results/Log_2022-09-27_10-06-23')

    # Give names to the logs (for plot legends)
    logs_names = ['fixed n=15000',
                  'fixed r=1.7',
                  'fixed n=12000',
                  'fixed n=15000+fps-4',
                  'r=1.7 old-residual64',
                  'r=1.7 naive-inverted16',
                  'r=1.7 old-residual96',
                  'r=1.7 naive-inverted24',
                  'r=1.7 test kpconnx',
                  'r=1.7 test kpminimod',
                  'KPNext32 new here',
                  'KPNext48 strided-conv',
                  'KPNext48 strided-x',
                  'KPNext48 kpconvd',
                  '...']

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

    # logs = logs[[0, 7, 9, 11, 12]]
    # logs_names = logs_names[[0, 7, 9, 11, 12]]

    return logs, logs_names


def test_kpnext():
    """
    FROM HERE WE CORRECTED AN ISSUE WITH Z COORDINATE FEATURE.
    """

    # Using the dates of the logs, you can easily gather consecutive ones. All logs should be of the same dataset.
    start = 'Log_2022-10-05_13-37-05'
    end = 'Log_2022-12-29_23-43-08'

    # Name of the result path
    res_path = 'results'

    # Gather logs and sort by date
    logs = np.sort([join(res_path, l) for l in listdir_str(res_path) if start <= l <= end])

    # # Optionally add a specific log at a specific place in the log list
    # logs = logs.astype('<U50')
    # logs = np.insert(logs, 0, 'results/Log_2022-09-27_10-06-23')

    # Give names to the logs (for plot legends)
    logs_names = ['KPNext32',
                  'KPNext48 strided-conv',
                  'KPNext48 strided-x',
                  'KPNext48 kpconvd',
                  'KPNext48 strided-conv + upcut',
                  'KPNext64 kpconvd + upcut',
                  'KPNext48 strided-conv G16',
                  'KPNext48 strided-conv G4',
                  'KPNext48 fps-4 G1',
                  'KPNext48 fps-4 G16',
                  'KPMegaNext48 C=1.41',
                  'KPMegaNext32 C=1.6',
                  'KPMegaNext32 C=1.2',
                  'KPMegaNext32 C=1.3',
                  'KPMegaNext32 C=1.35',
                  'KPBisNext48 C=1.35',
                  'KPMegaNext48 C=1.41 G16',
                  'KPNext48 G1 first_x=1',
                  'KPNext48 G1 first_x=0',
                  'KPNext48 G1 first_x=1',
                  'KPNext48 G1 first_x=2',
                  'KPNext48 G1 first_x=3',
                  'KPNext48 s3 kp_r=1.2',
                  'KPNext48 s3 kp_r=1.4']

    # safe check log names
    if len(logs) > len(logs_names):
        logs = logs[:len(logs_names)]
    logs_names = np.array(logs_names[:len(logs)])

    # logs = logs[[-1, 3]]
    # logs_names = logs_names[[-1, 3]]
    logs = logs[10:]
    logs_names = logs_names[10:]

    return logs, logs_names


def test_kpnext_2():
    """
    GOGO KPNext experiments
    """

    # Using the dates of the logs, you can easily gather consecutive ones. All logs should be of the same dataset.
    start = 'Log_2022-10-17_17-47-40'
    end = 'Log_2022-12-29_23-43-08'

    # Name of the result path
    res_path = 'results'

    # Gather logs and sort by date
    logs = np.sort([join(res_path, l) for l in listdir_str(res_path) if start <= l <= end])

    # Optionally add a specific log at a specific place in the log list
    logs = logs.astype('<U50')
    logs = np.insert(logs, 0, 'results/Log_2022-10-13_17-07-58')

    # Give names to the logs (for plot legends)
    logs_names = ['KPNext48 1,14,28 kp_r=1.2',
                  'KPNext48 1,21    kp_r=0.6',
                  'KPNext48 1,21    kp_r=0.8',
                  'KPNext48 1,21    kp_r=1.0',
                  'KPNext48 1,21    kp_r=1.2',
                  'KPNext48 1,21    kp_r=1.4',
                  'KPNext48 1,21    kp_r=1.6',
                  'KPNext48 1,14,28 kp_r=1.0',
                  'KPNext48 1,14,28 kp_r=1.1',
                  'KPNext48 1,14,28 kp_r=1.2',
                  'KPNext48 1,14,28 kp_r=1.3']


    # safe check log names
    if len(logs) > len(logs_names):
        logs = logs[:len(logs_names)]
    logs_names = np.array(logs_names[:len(logs)])

    # logs = logs[[-1, 3]]
    # logs_names = logs_names[[-1, 3]]

    return logs, logs_names


def test_kpnext_3():
    """
    GOGO KPNext experiments
    """

    # Using the dates of the logs, you can easily gather consecutive ones. All logs should be of the same dataset.
    start = 'Log_2022-10-19_17-37-25'
    end = 'Log_2022-12-29_23-43-08'

    # Name of the result path
    res_path = 'results'

    # Gather logs and sort by date
    logs = np.sort([join(res_path, l) for l in listdir_str(res_path) if start <= l <= end])

    # Optionally add a specific log at a specific place in the log list
    logs = logs.astype('<U50')
    logs = np.insert(logs, 0, 'results/Log_2022-10-13_17-07-58')

    # Give names to the logs (for plot legends)
    logs_names = ['KPNext48 1,14,28 kp_r=1.20 inv1/1',
                  'KPNext48 1,14,28 kp_r=1.10 inv0/1',
                  'KPNext48 1,14,28 kp_r=1.15 inv0/1',
                  'KPNext48 1,14,28 kp_r=1.20 inv1/16',
                  'KPNext48 1,14,28 kp_r=1.20 kpconvd',
                  'KPNext48 1,14,28 kp_r=1.20 kpconvd',
                  'KPNext48 1,14,28 kp_r=1.20 inv1/1',
                  'KPNextBig56 inv1/8',
                  'KPNextMega64 inv1/8',
                  'KPNextMega64 inv1/8 upcut',
                  'KPNextBig56  inv1/8 upcut']

    # safe check log names
    if len(logs) > len(logs_names):
        logs = logs[:len(logs_names)]
    logs_names = np.array(logs_names[:len(logs)])

    # logs = logs[[-1, -3, -4]]
    # logs_names = logs_names[[-1, -3, -4]]

    return logs, logs_names


def test_new_kpnext():
    """
    GOGO KPNext experiments
    """

    # Using the dates of the logs, you can easily gather consecutive ones. All logs should be of the same dataset.
    start = 'Log_2022-10-27_15-42-25'
    end = 'Log_2022-12-29_23-43-08'

    # Name of the result path
    res_path = 'results'

    # Gather logs and sort by date
    logs = np.sort([join(res_path, l) for l in listdir_str(res_path) if start <= l <= end])

    # Optionally add a specific log at a specific place in the log list
    logs = logs.astype('<U50')
    logs = np.insert(logs, 0, 'results/Log_2022-10-13_17-07-58')
    logs = np.insert(logs, 1, 'results/Log_2022-10-20_11-30-07')
    logs = np.insert(logs, 2, 'results/Log_2022-10-20_12-58-47')
    logs = np.insert(logs, 3, 'results/Log_2022-10-20_19-40-50')

    # Give names to the logs (for plot legends)
    logs_names = ['KPNext48 1,14,28 kp_r=1.20 inv1/1',
                  'KPNext48 1,14,28 kp_r=1.20 inv1/1',
                  'KPNextBig56 inv1/8',
                  'KPNextMega64 inv1/8 upcut',
                  'New KPNext (no drop points)',
                  'New KPNext (no drop points no decoder_layer)',
                  'New KPNext 2 + drop_pts',
                  'Same + upcut',
                  'KPSmall r_scal=2.0',
                  'KPMini no upcut',
                  'KPMini upcut',
                  'KPSmall layernorm',
                  'KPSmall r_scal=2.0',
                  'KPSmall r_scal=2.2',
                  'KPSmall r_scal=2.5',
                  'KPSmall r_scal=3.0',
                  'KPMega dl=0.03',
                  'KPMega dl=0.04',
                  'KPMega dl=0.05',
                  ]


    # safe check log names
    if len(logs) > len(logs_names):
        logs = logs[:len(logs_names)]
    logs_names = np.array(logs_names[:len(logs)])

    # logs = logs[-3:]
    # logs_names = logs_names[-3:]

    return logs, logs_names


def test_rooms():
    """
    GOGO new dataset with S3DIS rooms, to compare with recent papers
    """

    # Using the dates of the logs, you can easily gather consecutive ones. All logs should be of the same dataset.
    start = 'Log_2022-11-02_17-59-05'
    end = 'Log_2022-12-29_23-43-08'

    # Name of the result path
    res_path = 'results'

    # Gather logs and sort by date
    logs = np.sort([join(res_path, l) for l in listdir_str(res_path) if start <= l <= end])

    # Optionally add a specific log at a specific place in the log list
    logs = logs.astype('<U50')

    # Give names to the logs (for plot legends)
    logs_names = ['KPNetX-L no droppath',
                  'KPNetX-L',
                  'ignored KPNetX-L (2.0*0.04)',
                  'ignored KPNetX-L (2.0*0.04) 1/14',
                  'KPNetX-L (1/14)',
                  'KPNetX-S (1/14)',
                  ' - decoder_layer',
                  'KPNetX-S 1/14/28',
                  'KPNetX-L - upcut',
                  'KPNetX-L G=1',
                  'KPNetX-L G=4',
                  'KPNetX-L G=8',
                  'KPNetX-L G=8',
                  'KPNetX-L G=8',
                  'KPNetX-L G=16',
                  'KPNetX-L G=-1',
                  'KPNetX-L G=0',
                  'todo']

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

    logs = np.hstack((logs[:2], logs[4:]))
    logs_names = np.hstack((logs_names[:2], logs_names[4:]))
    # logs = logs[[-1, -3, -4]]
    # logs_names = logs_names[[-1, -3, -4]]

    return logs, logs_names


def exp_KPNetX_L():
    """
    Multiple tries of KPNext-L. We see that more neighbors get better scores. Keep that in mind for best network tries
    """

    # Using the dates of the logs, you can easily gather consecutive ones. All logs should be of the same dataset.
    start = 'Log_2022-11-05_03-34-33'
    end = 'Log_2022-11-06_21-20-45'

    # Name of the result path
    res_path = 'results'

    # Gather logs and sort by date
    logs = np.sort([join(res_path, l) for l in listdir_str(res_path) if start <= l <= end])

    # Optionally add a specific log at a specific place in the log list
    logs = logs.astype('<U50')
    logs = np.insert(logs, 0, 'results/Log_2022-11-02_17-59-39')

    # Give names to the logs (for plot legends)
    logs_names = ['KPNetX-L',
                  'KPNetX-L',
                  'KPNetX-L',
                  'KPNetX-L',
                  'todo']

    # safe check log names
    if len(logs) > len(logs_names):
        logs = logs[:len(logs_names)]
    logs_names = np.array(logs_names[:len(logs)])

    # logs = logs[[-1, -3, -4]]
    # logs_names = logs_names[[-1, -3, -4]]

    return logs, logs_names


def exp_ablation():
    """
    Multiple tries of KPNext-L. We see that more neighbors get better scores. Keep that in mind for best network tries
    """

    # Using the dates of the logs, you can easily gather consecutive ones. All logs should be of the same dataset.
    start = 'Log_2022-11-02_17-59-05'
    end = 'Log_2022-11-02_17-59-49'

    # Name of the result path
    res_path = 'results'

    # Gather logs and sort by date
    logs = np.sort([join(res_path, l) for l in listdir_str(res_path) if start <= l <= end])

    print(logs)

    # Optionally add a specific log at a specific place in the log list
    logs = logs.astype('<U50')
    logs = np.insert(logs, 0, 'results/Log_2022-11-04_10-16-53')
    logs = np.insert(logs, 2, 'results/Log_2022-11-05_19-32-44')

    # Give names to the logs (for plot legends)
    logs_names = ['KPNetX-L - upcut',
                  'KPNetX-L no droppath',
                  'KPNetX-L',
                  'KPNetX-L',
                  'todo']

    # safe check log names
    if len(logs) > len(logs_names):
        logs = logs[:len(logs_names)]
    logs_names = np.array(logs_names[:len(logs)])

    # logs = logs[[-1, -3, -4]]
    # logs_names = logs_names[[-1, -3, -4]]

    return logs, logs_names


def exp_shells():
    """
    Multiple tries of KPNext-L. We see that more neighbors get better scores. Keep that in mind for best network tries
    """

    # Using the dates of the logs, you can easily gather consecutive ones. All logs should be of the same dataset.
    start = 'Log_2022-11-03_16-08-25'
    end = 'Log_2022-11-04_04-43-69'

    # Name of the result path
    res_path = 'results'

    # Gather logs and sort by date
    logs = np.sort([join(res_path, l) for l in listdir_str(res_path) if start <= l <= end])

    # Optionally add a specific log at a specific place in the log list
    logs = logs.astype('<U50')
    logs = np.insert(logs, 0, 'results/Log_2022-11-02_17-59-39')
    logs = np.insert(logs, 1, 'results/Log_2022-11-05_19-32-44')
    logs = np.insert(logs, 3, 'results/Log_2022-11-04_09-33-24')

    # Give names to the logs (for plot legends)
    logs_names = ['KPNetX-L (1/14/28)',
                  'KPNetX-L (1/14/28)',
                  'KPNetX-L (1/14)',
                  'KPNetX-S (1/14/28)',
                  'KPNetX-S (1/14)',
                  ' - decoder layer',
                  'todo']

    # safe check log names
    if len(logs) > len(logs_names):
        logs = logs[:len(logs_names)]
    logs_names = np.array(logs_names[:len(logs)])

    # logs = logs[[0]]
    # logs_names = logs_names[[0]]

    return logs, logs_names


def exp_groups():
    """
    Multiple tries of KPNext-L. We see that more neighbors get better scores. Keep that in mind for best network tries
    """

    # Using the dates of the logs, you can easily gather consecutive ones. All logs should be of the same dataset.
    start = 'Log_2022-11-06_21-20-49'
    end = 'Log_2022-11-07_07-17-45'

    # Name of the result path
    res_path = 'results'

    # Gather logs and sort by date
    logs = np.sort([join(res_path, l) for l in listdir_str(res_path) if start <= l <= end])

    # Optionally add a specific log at a specific place in the log list
    logs = logs.astype('<U50')
    logs = np.insert(logs, 0, 'results/Log_2022-11-04_23-40-28')
    logs = np.insert(logs, 1, 'results/Log_2022-11-05_03-34-32')
    logs = np.insert(logs, 2, 'results/Log_2022-11-02_17-59-39')
    logs = np.insert(logs, 5, 'results/Log_2022-11-07_21-45-32')
    logs = np.insert(logs, 6, 'results/Log_2022-11-08_07-31-26')

    # Give names to the logs (for plot legends)
    logs_names = ['KPNetX-L G1',
                  'KPNetX-L G4',
                  'KPNetX-L G8',
                  'KPNetX-L G16',
                  'KPNetX-L G-1',
                  'KPNetX-L G0',
                  'KPNetX-L G0',
                  'todo']

    # safe check log names
    if len(logs) > len(logs_names):
        logs = logs[:len(logs_names)]
    logs_names = np.array(logs_names[:len(logs)])

    # logs = logs[[5, 2]]
    # logs_names = logs_names[[5, 2]]

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
    logs, logs_names = exp_shells()

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
        if not cfg.data.name.startswith('S3DI'):
            err_mess = '\nTrying to plot S3DIS experiments, but {:s} was trained on {:s} dataset.'
            raise ValueError(err_mess.format(cfg.exp.date, cfg.data.name))
            
    # Print differences in a nice table
    print_cfg_diffs(logs_names,
                    all_cfgs,
                    # show_params=['model.in_sub_size',
                    #              'train.in_radius',
                    #              'train.batch_size',
                    #              'train.accum_batch',
                    #              'train.batch_limit'],
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
    # compare_trainings(all_cfgs, logs, logs_names)


    # Test the network or show validation
    perform_test = True
    if perform_test:

        print()
        underline("Test networks")
        print()

        # Plot the validation
        compare_on_test_set(all_cfgs, logs, logs_names)
        # compare_on_test_set(all_cfgs, logs, logs_names, profile=True)


    else:
        print()
        underline("Ploting validation info")
        print()

        # Plot the validation
        compare_convergences_segment(all_cfgs, logs, logs_names)

