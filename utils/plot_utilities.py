
# ----------------------------------------------------------------------------------------------------------------------
#
#           Script Intro
#       \******************/
#
#
#   This script contains a lot of function to plot results. See the plot_results.py scripts in experiments folder
#   for examples of how to use them
#
#




# ----------------------------------------------------------------------------------------------------------------------
#
#           Imports and global variables
#       \**********************************/
#

# Common libs
import os
import shutil
import torch
import numpy as np
from utils.printing import bcolors
import matplotlib.pyplot as plt
from os.path import isfile, join, exists
from os import listdir, remove, getcwd, makedirs
from sklearn.metrics import confusion_matrix
import time
import pickle
from torch.utils.data import DataLoader
from matplotlib.animation import FuncAnimation
import matplotlib.patches as patches
from matplotlib.widgets import Slider, Button, RadioButtons
import imageio

# My libs
from utils.printing import frame_lines_1, underline, print_color
from utils.metrics import IoU_from_confusions, smooth_metrics, fast_confusion
from utils.ply import read_ply, write_ply
from utils.config import load_cfg

from experiments.S3DIS_simple.S3DIS import S3DISDataset
from experiments.S3DIS_simple.test_S3DIS_simple import test_log




# ----------------------------------------------------------------------------------------------------------------------
#
#           Utility functions
#       \***********************/
#


def listdir_str(path):

    # listdir can return binary string instead of decoded string sometimes.
    # This function ensures a steady behavior

    f_list = []
    for f in listdir(path):
        try:
            f = f.decode()
        except (UnicodeDecodeError, AttributeError):
            pass
        f_list.append(f)

    return f_list


def running_mean(signal, n, axis=0, stride=1):
    signal = np.array(signal)
    torch_conv = torch.nn.Conv1d(1, 1, kernel_size=2 * n + 1,
                                 stride=stride,
                                 padding='same',
                                 padding_mode='replicate',
                                 bias=False)
    torch_conv.weight.requires_grad_(False)
    torch_conv.weight *= 0
    torch_conv.weight += 1 / (2 * n + 1)
    if signal.ndim == 1:
        torch_signal = torch.from_numpy(signal.reshape([1, 1, -1]).astype(np.float32))
        return torch_conv(torch_signal).squeeze().numpy()

    elif signal.ndim == 2:
        print('TODO implement with torch and stride here')
        smoothed = np.empty(signal.shape)
        if axis == 0:
            for i, sig in enumerate(signal):
                sig_sum = np.convolve(sig, np.ones((2*n+1,)), mode='same')
                sig_num = np.convolve(sig*0+1, np.ones((2*n+1,)), mode='same')
                smoothed[i, :] = sig_sum / sig_num
        elif axis == 1:
            for i, sig in enumerate(signal.T):
                sig_sum = np.convolve(sig, np.ones((2*n+1,)), mode='same')
                sig_num = np.convolve(sig*0+1, np.ones((2*n+1,)), mode='same')
                smoothed[:, i] = sig_sum / sig_num
        else:
            print('wrong axis')
        return smoothed

    else:
        print('wrong dimensions')
        return None


def IoU_class_metrics(all_IoUs, smooth_n):

    # Get mean IoU per class for consecutive epochs to directly get a mean without further smoothing
    smoothed_IoUs = []
    for epoch in range(len(all_IoUs)):
        i0 = max(epoch - smooth_n, 0)
        i1 = min(epoch + smooth_n + 1, len(all_IoUs))
        smoothed_IoUs += [np.mean(np.vstack(all_IoUs[i0:i1]), axis=0)]
    smoothed_IoUs = np.vstack(smoothed_IoUs)
    smoothed_mIoUs = np.mean(smoothed_IoUs, axis=1)

    return smoothed_IoUs, smoothed_mIoUs


def load_confusions(filename, n_class):

    with open(filename, 'r') as f:
        lines = f.readlines()

    confs = np.zeros((len(lines), n_class, n_class))
    for i, line in enumerate(lines):
        C = np.array([int(value) for value in line.split()])
        confs[i, :, :] = C.reshape((n_class, n_class))

    return confs


def load_training_results(path):

    filename = join(path, 'training.txt')
    with open(filename, 'r') as f:
        lines = f.readlines()

    epochs = []
    steps = []
    L_out = []
    L_p = []
    t = []
    for line in lines[1:]:
        line_info = line.split()
        if (len(line) > 0):
            epochs += [int(line_info[0])]
            steps += [int(line_info[1])]
            L_out += [float(line_info[2])]
            L_p += [float(line_info[3])]
            t += [float(line_info[4])]
        else:
            break

    return epochs, steps, L_out, L_p, t


def load_single_IoU(filename, n_parts):

    with open(filename, 'r') as f:
        lines = f.readlines()

    # Load all IoUs
    all_IoUs = []
    for i, line in enumerate(lines):
        all_IoUs += [np.reshape([float(IoU) for IoU in line.split()], [-1, n_parts])]
    return all_IoUs


def load_snap_clouds_old(path, cfg, only_last=False):

    cloud_folders = np.array([join(path, f) for f in listdir_str(path) if f.startswith('val_preds')])
    cloud_epochs = np.array([int(f.split('_')[-1]) for f in cloud_folders])
    epoch_order = np.argsort(cloud_epochs)
    cloud_epochs = cloud_epochs[epoch_order]
    cloud_folders = cloud_folders[epoch_order]

    Confs = np.zeros((len(cloud_epochs), cfg.data.num_classes, cfg.data.num_classes), dtype=np.int32)
    for c_i, cloud_folder in enumerate(cloud_folders):
        if only_last and c_i < len(cloud_epochs) - 1:
            continue

        # Load confusion if previously saved
        conf_file = join(cloud_folder, 'conf.txt')
        if isfile(conf_file):
            Confs[c_i] += np.loadtxt(conf_file, dtype=np.int32)

        else:
            for f in listdir_str(cloud_folder):
                if f.endswith('.ply') and not f.endswith('sub.ply'):
                    data = read_ply(join(cloud_folder, f))
                    labels = data['class']
                    preds = data['preds']
                    label_values = np.array(cfg.data.label_values, dtype=np.int32)
                    Confs[c_i] += fast_confusion(labels, preds, label_values).astype(np.int32)

            np.savetxt(conf_file, Confs[c_i], '%12d')

        # Erase ply to save disk memory
        if c_i < len(cloud_folders) - 1:
            for f in listdir_str(cloud_folder):
                if f.endswith('.ply'):
                    remove(join(cloud_folder, f))

    # Remove ignored labels from confusions
    for l_ind, label_value in reversed(list(enumerate(cfg.data.label_values))):
        if label_value in cfg.data.ignored_labels:
            Confs = np.delete(Confs, l_ind, axis=1)
            Confs = np.delete(Confs, l_ind, axis=2)

    return cloud_epochs, IoU_from_confusions(Confs)


def load_snap_clouds(path, cfg, only_last=False):

    # Check if we have old or new validation style
    cloud_folders = [join(path, f) for f in listdir_str(path) if f.startswith('val_preds')]
    if len(cloud_folders) > 0:
        return load_snap_clouds_old(path, cfg, only_last=only_last)

    # Verify that we have a validation folder
    val_path = join(path, 'validation')
    if not exists(val_path):
        return load_snap_clouds_old(path, cfg, only_last=only_last)


    # Read prediction and compute confusions
    # **************************************

    # Get list of vote predictions
    preds_names = np.array([f for f in listdir_str(val_path) if f.startswith('preds_')])
    saved_votes = np.array([int(f.split('_')[1]) for f in preds_names])
    preds_names = preds_names[np.argsort(saved_votes)]
    preds_paths = np.array([join(val_path, f) for f in preds_names])
    preds_epochs = np.array([int(f[:-4].split('_')[-1]) for f in preds_names])
    preds_votes = np.array([int(f.split('_')[1]) for f in preds_names])

    dataset = None
    for v_i, preds_path in enumerate(preds_paths):
        
        # compute confusion if not already saved
        conf_path = join(val_path, 'conf_{:d}_{:d}.txt'.format(preds_votes[v_i], preds_epochs[v_i]))
        if not isfile(conf_path):

            # Get dataset in memory to have reproj indices
            if dataset is None:
                dataset_dict = {'S3DIS': S3DISDataset}
                if cfg.data.name not in dataset_dict:
                    raise ValueError('Add your dataset "{:s}" to the dataset_dict just above'.format(cfg.data.name))
                dataset = dataset_dict[cfg.data.name](cfg, chosen_set='validation', calib=False)

            # Load vote predictions
            with open(preds_path, 'rb') as f:
                val_preds = pickle.load(f)

            # Get points
            files = dataset.scene_files
            scene_confs = np.zeros((cfg.data.num_classes, cfg.data.num_classes), dtype=np.int32)
            for c_i, file_path in enumerate(files):

                # Get groundtruth labels
                labels = dataset.val_labels[c_i].astype(np.int32)

                # Reproject preds on the evaluations points
                preds = (val_preds[c_i][dataset.test_proj[c_i]]).astype(np.int32)

                # Confusion matrix
                label_values = np.array(cfg.data.label_values, dtype=np.int32)
                scene_confs += fast_confusion(labels, preds, label_values).astype(np.int32)

            # Save confusion for future use
            np.savetxt(conf_path, scene_confs, '%12d')
        
        # Erase label files to save disk memory
        if v_i < len(preds_paths) - 1 and exists(preds_path):
            remove(preds_path)


    # Read confusions
    # ***************

    conf_names = np.array([f for f in listdir_str(val_path) if f.startswith('conf_')])
    saved_votes = np.array([int(f.split('_')[1]) for f in conf_names])
    conf_names = conf_names[np.argsort(saved_votes)]
    conf_paths = np.array([join(val_path, f) for f in conf_names])
    conf_epochs = np.array([int(f[:-4].split('_')[-1]) for f in conf_names])
    conf_votes = np.array([int(f.split('_')[1]) for f in conf_names])
    
    Confs = np.zeros((len(conf_epochs), cfg.data.num_classes, cfg.data.num_classes), dtype=np.int32)
    for v_i, conf_path in enumerate(conf_paths):
        Confs[v_i] += np.loadtxt(conf_path, dtype=np.int32)

    # Remove ignored labels from confusions
    for l_ind, label_value in reversed(list(enumerate(cfg.data.label_values))):
        if label_value in cfg.data.ignored_labels:
            Confs = np.delete(Confs, l_ind, axis=1)
            Confs = np.delete(Confs, l_ind, axis=2)

    return conf_epochs, IoU_from_confusions(Confs)


def cfg_differences(list_of_cfg, ignore_params=[]):

    # First list all possible keys in case some parameter do not exist in some configs
    all_keys = []
    for cfg in list_of_cfg:
        for k1, v1 in cfg.items():
            for k2, v2 in v1.items():
                all_keys.append(k1 + "." + k2)
    all_keys = np.unique(all_keys)

    # List all parameters
    diff_params = []
    diff_values = []
    for k_str in all_keys:

        # Get the two keys
        k1, k2 = k_str.split('.')

        # skip some parameters that are always different
        if k1 == 'exp' or k_str in ignore_params:
            continue

        # Get value for each config
        values = []
        for cfg in list_of_cfg:
            if k2 in cfg[k1]:
                values.append(cfg[k1][k2])
            else:
                values.append(None)

        if np.any([v != values[0] for v in values[1:]]):
            diff_params.append((k1, k2))
            diff_values.append(values)

    return diff_params, diff_values
    

def print_cfg_diffs(logs_names, log_cfgs, show_params=[], hide_params=[], max_cols=145):
    """
    Print the differences in parameters between logs. Use show_params to force showing 
    some parameters even if no differences are seen.
    """

    # Get differences between configs
    diff_params, diff_values = cfg_differences(log_cfgs, ignore_params=hide_params)
    
    # Add parameters that are shown anyway
    for k_str in show_params:

        # Get the two keys
        k1, k2 = k_str.split('.')

        # Get value for each config
        values = []
        for cfg in log_cfgs:
            if k2 in cfg[k1]:
                values.append(cfg[k1][k2])
            else:
                values.append(None)

        diff_params.append((k1, k2))
        diff_values.append(values)

    # Create the first column of the table with each log
    first_col = ['      \\  Params ',
                 ' Logs  \\        ',
                 '']
    for log in logs_names:
        first_col.append(log)
    n_fmt0 = np.max([len(col_str) for col_str in first_col]) + 2
    first_col[2] = '-' * n_fmt0
    lines = ['{:^{width}s}|'.format(col_str, width=n_fmt0) for col_str in first_col]

    # Add parameters column
    for d_i, values in enumerate(diff_values):
        k1, k2 = diff_params[d_i]

        # Get all the string we want in this column
        col_strings = [k1, k2, '']
        if col_strings[1] == 'augment_symmetries':
            col_strings[1] = 'augm_sym'

        for v in values:

            if isinstance(v, str):
                col_strings.append(v)
            
            elif isinstance(v, int):
                col_strings.append(str(v))

            elif isinstance(v, float):
                if v < 100:
                    col_strings.append('{:.3f}'.format(v))
                else:
                    col_strings.append('{:.1f}'.format(v))

            elif isinstance(v, list):
                if len(v) < 7:
                    col_strings.append(str(v))
                else:
                    col_strings.append('[{:s}, {:s}, ...]'.format(str(v[0]), str(v[1])))
            else:
                col_strings.append(type(v).__name__)

        # Replace true/false with checkmarks
        for c_i, col_str in enumerate(col_strings):
            col_str = col_str.replace("True", '{:s}'.format(u'\u2713'))
            col_strings[c_i] = col_str.replace("False", '{:s}'.format(u'\u2718'))

        # Get the max length of the column
        n_fmt1 = np.max([len(col_str) for col_str in col_strings]) + 2

        # horizontal line
        col_strings[2] = '-' * n_fmt1

        # fill columns
        for c_i, col_str in enumerate(col_strings):
            n_colors = col_str.count(bcolors.ENDC)
            n_fmt_col = n_fmt1 + (len(bcolors.ENDC) + len(bcolors.FAIL)) * n_colors
            lines[c_i] += '{:^{width}s}|'.format(col_str, width=n_fmt_col)

    # Print according to max_cols
    underline("Parameter differences in your logs")
    print('')

    first_i = lines[0].find('|')

    while len(lines[0]) > max_cols:
        last_i = lines[0][:max_cols].rfind('|')
        for l_i, line_str in enumerate(lines):
            print_color(line_str[:last_i + 1])
            lines[l_i] = line_str[:first_i+1] + line_str[last_i + 1:]
        print('')

    for line_str in lines:
        print_color(line_str)
    print('\n')
    
    return


# ----------------------------------------------------------------------------------------------------------------------
#
#           Plot functions
#       \********************/
#


def compare_trainings(list_of_cfg, list_of_paths, list_of_labels=None):

    # Parameters
    # **********

    plot_lr = True
    smooth_epochs = 1.5

    if list_of_labels is None:
        list_of_labels = [str(i) for i in range(len(list_of_paths))]

    # Read Training Logs
    # ******************

    print('\nCollecting log info')
    t0 = time.time()

    all_epochs = []
    all_loss = []
    all_lr = []
    all_times = []
    all_mean_epoch_n = []
    all_batch_num = []
    all_ins_per_s = []
    all_stp_per_s = []
    all_epoch_dt = []
    all_val_dt = []

    for path, cfg in zip(list_of_paths, list_of_cfg):
        

        if not (('val_IoUs.txt' in [f for f in listdir_str(path)]) or ('val_confs.txt' in [f for f in listdir_str(path)])):
            continue

        # Load results
        epochs, steps, L_out, L_p, t = load_training_results(path)
        epochs = np.array(epochs, dtype=np.int32)
        epochs_d = np.array(epochs, dtype=np.float32)
        steps = np.array(steps, dtype=np.float32)
        t = np.array(t, dtype=np.float32)

        # Compute number of steps per epoch
        max_e = np.max(epochs)
        first_e = np.min(epochs)
        epoch_n = []
        for i in range(first_e, max_e+1):
            bool0 = epochs == i
            e_n = np.sum(bool0)
            epochs_d[bool0] += steps[bool0] / e_n
            if i < max_e:
                epoch_n.append(e_n)
        mean_epoch_n = np.mean(epoch_n)
        smooth_n = int(mean_epoch_n * smooth_epochs)
        smooth_loss = running_mean(L_out, smooth_n)
        all_loss += [smooth_loss]
        all_epochs += [epochs_d]
        all_times += [t]
        all_mean_epoch_n += [mean_epoch_n]
        all_batch_num += [cfg.train.batch_size * cfg.train.accum_batch]

        # Learning rate
        if plot_lr:
            lr_decay_v = np.array([lr_d for ep, lr_d in cfg.train.lr_decays.items()])
            lr_decay_e = np.array([int(ep) for ep, lr_d in cfg.train.lr_decays.items()])
            max_ee = max(np.max(all_epochs[-1]) + 1, np.max(lr_decay_e) + 1)
            lr_decays = np.ones(int(np.ceil(max_ee)), dtype=np.float32)
            lr_decays[0] = float(cfg.train.lr)
            lr_decays[lr_decay_e] = lr_decay_v
            lr = np.cumprod(lr_decays)
            all_lr += [lr[np.floor(all_epochs[-1]).astype(np.int32)]]

        # Timing report
        epoch_t0 = []
        epoch_t1 = []
        ins_per_s = []
        step_per_s = []
        for i in range(first_e, max_e):

            # Get epoch times
            epoch_t = t[epochs == i]
            epoch_t0.append(np.min(epoch_t))
            epoch_t1.append(np.max(epoch_t))

            # Get full speed running time at the middle of the epoch
            margin = int(epoch_n[i] * 0.2)
            dt = epoch_t[margin+1:-margin] - epoch_t[margin:-margin-1]
            step_per_s.append(1/dt)
            ins_per_s.append(cfg.train.batch_size * cfg.train.accum_batch * step_per_s[-1])

        epoch_t0 = np.array(epoch_t0)
        epoch_t1 = np.array(epoch_t1)
        epoch_dt = epoch_t1 - epoch_t0
        val_dt = epoch_t0[1:] - epoch_t1[:-1]
        
        all_stp_per_s.append(np.concatenate(step_per_s, axis=0))
        all_ins_per_s.append(np.concatenate(ins_per_s, axis=0))
        all_epoch_dt.append(epoch_dt)
        all_val_dt.append(val_dt)

    print('Done in {:.3f} s'.format(time.time() - t0))


    # Timing report
    # *************

    # Figure
    print('\nInit matplotlib')
    t0 = time.time()
    _, axs = plt.subplots(1, 2, figsize=(9, 4), gridspec_kw={'width_ratios': [1, 2]})
    print('Done in {:.3f} s'.format(time.time() - t0))
    print('\nPlotting training information')

    # Throughput (ins/sec) as boxplot
    axs[0].yaxis.grid(True, linestyle='-', which='major', color='lightgrey', alpha=0.5)
    axs[0].set_axisbelow(True)
    y_data = []
    x_data = []
    colors = []
    for log_i, box_data in enumerate(all_ins_per_s):
        y_data.append(box_data)
        x_data.append('')
        colors.append('C'+ str(log_i))
    bp = axs[0].boxplot(y_data, 
                        labels=x_data,
                        patch_artist=True,
                        showfliers=False,
                        widths=0.4)
    plt.setp(bp['medians'], color='k')
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    blim, tlim = axs[0].get_ylim()
    tlim += 0.3 * (tlim - blim)
    axs[0].set_ylim(bottom=blim, top=tlim)
    axs[0].set_title('Throughput')
    axs[0].set_ylabel('ins/sec')
    axs[0].legend(np.array(bp['boxes']), tuple(list_of_labels), fontsize='small', loc=1)

    # Epoch times as horizontal bars
    axs[1].xaxis.grid(True, linestyle='-', which='major', color='lightgrey', alpha=0.5)
    axs[1].set_axisbelow(True)
    y_pos = np.arange(len(list_of_labels))
    mean_epoch_dt = [np.mean(edt) for edt in all_epoch_dt]
    mean_val_dt = [np.mean(edt) for edt in all_val_dt]
    width = 0.4
    axs[1].barh(y_pos, mean_epoch_dt, width, label = 'Train')
    axs[1].barh(y_pos, mean_val_dt, width, left=mean_epoch_dt, label = 'Val')
    axs[1].invert_yaxis()  # labels read top-to-bottom
    axs[1].set_xlabel('Time (seconds)')
    axs[1].set_title('Epoch Timings => |Train|Val|')
    axs[1].tick_params(axis='y', which='both', right=False, left=False, labelleft=False) 
    for l_i, label in enumerate(list_of_labels):
        axs[1].text(10, y_pos[l_i], label,
                       ha='left', 
                       va='center',
                       fontsize='medium',
                       weight='roman',
                       color='k')


    # Plots learning rate
    # *******************

    if plot_lr:
        # Figure
        fig = plt.figure('lr')
        for i, label in enumerate(list_of_labels):
            plt.plot(all_epochs[i], all_lr[i], linewidth=1, label=label)

        # Set names for axes
        plt.xlabel('epochs')
        plt.ylabel('lr')
        plt.yscale('log')

        # Display legends and title
        plt.legend(loc=1)

        # Customize the graph
        ax = fig.gca()
        ax.grid(linestyle='-.', which='both')
        # ax.set_yticks(np.arange(0.8, 1.02, 0.02))

    # Plots loss
    # **********

    # Figure
    
    fig, ax0 = plt.subplots(1, 1, figsize=(8, 6))
    plots = []
    for i, label in enumerate(list_of_labels):
        plots.append(plt.plot(all_epochs[i], all_loss[i], linewidth=1, label=label))

    # Set names for axes
    plt.xlabel('training epochs')
    plt.ylabel('loss')
    plt.yscale('log')

    # Display legends and title
    plt.legend(loc=1)

    # Customize the graph
    ax0.grid(linestyle='-.', which='both')

    # X-axis controller
    plt.subplots_adjust(right=0.82)
    axcolor = 'lightgrey'
    rax = plt.axes([0.84, 0.4, 0.14, 0.2], facecolor=axcolor)
    radio = RadioButtons(rax, ('epochs', 'steps', 'examples', 'time'))
    def x_func(label):
        if label == 'epochs':
            ax0.set_xlabel('training epochs')
            for i, label in enumerate(list_of_labels):
                for pl in plots[i]:
                    pl.set_xdata(all_epochs[i])
        if label == 'steps':
            ax0.set_xlabel('steps (input batch seen)')
            for i, label in enumerate(list_of_labels):
                for pl in plots[i]:
                    pl.set_xdata(all_epochs[i] * all_mean_epoch_n[i])
        if label == 'examples':
            ax0.set_xlabel('input spheres seen')
            for i, label in enumerate(list_of_labels):
                for pl in plots[i]:
                    pl.set_xdata(all_epochs[i] * all_mean_epoch_n[i] * all_batch_num[i])
        if label == 'time':
            ax0.set_xlabel('hours')
            for i, label in enumerate(list_of_labels):
                for pl in plots[i]:
                    pl.set_xdata(all_times[i] / 3600)
        
        ax0.relim()
        ax0.autoscale_view()
        plt.draw()

    radio.on_clicked(x_func)

    # Show all
    plt.show()


def compare_convergences_segment(list_of_cfg, list_of_paths, list_of_names=None):

    # Parameters
    # **********

    print('\nCollecting validation logs')
    t0 = time.time()

    smooth_n = 10

    if list_of_names is None:
        list_of_names = [str(i) for i in range(len(list_of_paths))]

    # Read Logs
    # *********

    all_pred_epochs = []
    all_mIoUs = []
    all_class_IoUs = []
    all_snap_epochs = []
    all_snap_IoUs = []

    class_list = [name for label, name in list_of_cfg[0].data.label_and_names
                  if label not in list_of_cfg[0].data.ignored_labels]
    
    num_classes = len(class_list)

    for path, cfg in zip(list_of_paths, list_of_cfg):

        # Get validation IoUs
        file = join(path, 'val_IoUs.txt')
        val_IoUs = load_single_IoU(file, num_classes)

        # Get mean IoU
        class_IoUs, mIoUs = IoU_class_metrics(val_IoUs, smooth_n)

        # Aggregate results
        all_pred_epochs += [np.array([i for i in range(len(val_IoUs))])]
        all_mIoUs += [mIoUs]
        all_class_IoUs += [class_IoUs]

        # Get optional full validation on clouds
        snap_epochs, snap_IoUs = load_snap_clouds(path, cfg)

        # smooth_full_n
        # # Get mean IoU per class for consecutive epochs to directly get a mean without further smoothing
        # smoothed_IoUs = []
        # for epoch in range(len(all_IoUs)):
        #     i0 = max(epoch - smooth_n, 0)
        #     i1 = min(epoch + smooth_n + 1, len(all_IoUs))
        #     smoothed_IoUs += [np.mean(np.vstack(all_IoUs[i0:i1]), axis=0)]
        # smoothed_IoUs = np.vstack(smoothed_IoUs)
        # smoothed_mIoUs = np.mean(smoothed_IoUs, axis=1)

        # return smoothed_IoUs, smoothed_mIoUs
        all_snap_epochs += [snap_epochs]
        all_snap_IoUs += [snap_IoUs]

    print('Done in {:.3f} s'.format(time.time() - t0))
    print('\n')

    # Print spheres validation
    s = '{:^10}|'.format('mean')
    for c in class_list:
        s += '{:^10}'.format(c)
    print(s)
    print(10*'-' + '|' + 10*num_classes*'-')
    for mIoUs in all_mIoUs:
        s = '{:^10.1f}|'.format(100*mIoUs[-1])
        for IoU in class_IoUs[-1]:
            s += '{:^10.1f}'.format(100*IoU)
        print(s)

    # Print clouds validation (average over the last ten validations)
    print(10*'-' + '|' + 10*num_classes*'-')
    for snap_IoUs in all_snap_IoUs:
        if len(snap_IoUs) > 0:
            last_avg_n = 3
            if snap_IoUs.shape[0] > last_avg_n:
                last_snaps = snap_IoUs[-last_avg_n:]
            else:
                last_snaps = snap_IoUs
            mean_IoUs = np.mean(last_snaps, axis=0)
            s = '{:^10.1f}|'.format(100*np.mean(mean_IoUs))
            for IoU in mean_IoUs:
                s += '{:^10.1f}'.format(100*IoU)
        else:
            s = '{:^10s}'.format('-')
            for _ in range(num_classes):
                s += '{:^10s}'.format('-')
        print(s)

    # Plots
    # *****

    # Figure
    fig = plt.figure('mIoUs')
    for i, name in enumerate(list_of_names):
        p = plt.plot(all_pred_epochs[i], all_mIoUs[i], '--', linewidth=1, label=name)
        plt.plot(all_snap_epochs[i], np.mean(all_snap_IoUs[i], axis=1), linewidth=1, color=p[-1].get_color())
    plt.xlabel('epochs')
    plt.ylabel('IoU')

    # Set limits for y axis
    #plt.ylim(0.55, 0.95)

    # Display legends and title
    plt.legend(loc=4)

    # Customize the graph
    ax = fig.gca()
    ax.grid(linestyle='-.', which='both')
    #ax.set_yticks(np.arange(0.8, 1.02, 0.02))

    displayed_classes = [0, 1, 2, 3, 4, 5, 6, 7]
    displayed_classes = []
    for c_i, c_name in enumerate(class_list):
        if c_i in displayed_classes:

            # Figure
            fig = plt.figure(c_name + ' IoU')
            for i, name in enumerate(list_of_names):
                plt.plot(all_pred_epochs[i], all_class_IoUs[i][:, c_i], linewidth=1, label=name)
            plt.xlabel('epochs')
            plt.ylabel('IoU')

            # Set limits for y axis
            #plt.ylim(0.8, 1)

            # Display legends and title
            plt.legend(loc=4)

            # Customize the graph
            ax = fig.gca()
            ax.grid(linestyle='-.', which='both')
            #ax.set_yticks(np.arange(0.8, 1.02, 0.02))

    # Show all
    plt.show()


def compare_convergences_classif(list_of_paths, list_of_labels=None):

    # Parameters
    # **********

    steps_per_epoch = 0
    smooth_n = 12

    if list_of_labels is None:
        list_of_labels = [str(i) for i in range(len(list_of_paths))]

    # Read Logs
    # *********

    all_pred_epochs = []
    all_val_OA = []
    all_train_OA = []
    all_vote_OA = []
    all_vote_confs = []


    for path in list_of_paths:

        # Load parameters
        config = Config()
        config.load(list_of_paths[0])

        # Get the number of classes
        n_class = config.num_classes

        # Load epochs
        epochs, _, _, _, _, _ = load_training_results(path)
        first_e = np.min(epochs)

        # Get validation confusions
        file = join(path, 'val_confs.txt')
        val_C1 = load_confusions(file, n_class)
        val_PRE, val_REC, val_F1, val_IoU, val_ACC = smooth_metrics(val_C1, smooth_n=smooth_n)

        # Get vote confusions
        file = join(path, 'vote_confs.txt')
        if exists(file):
            vote_C2 = load_confusions(file, n_class)
            vote_PRE, vote_REC, vote_F1, vote_IoU, vote_ACC = smooth_metrics(vote_C2, smooth_n=2)
        else:
            vote_C2 = val_C1
            vote_PRE, vote_REC, vote_F1, vote_IoU, vote_ACC = (val_PRE, val_REC, val_F1, val_IoU, val_ACC)

        # Aggregate results
        all_pred_epochs += [np.array([i+first_e for i in range(len(val_ACC))])]
        all_val_OA += [val_ACC]
        all_vote_OA += [vote_ACC]
        all_vote_confs += [vote_C2]

    print()

    # Best scores
    # ***********

    for i, label in enumerate(list_of_labels):

        print('\n' + label + '\n' + '*' * len(label) + '\n')
        print(list_of_paths[i])

        best_epoch = np.argmax(all_vote_OA[i])
        print('Best Accuracy : {:.1f} % (epoch {:d})'.format(100 * all_vote_OA[i][best_epoch], best_epoch))

        confs = all_vote_confs[i]

        """
        s = ''
        for cc in confs[best_epoch]:
            for c in cc:
                s += '{:.0f} '.format(c)
            s += '\n'
        print(s)
        """

        TP_plus_FN = np.sum(confs, axis=-1, keepdims=True)
        class_avg_confs = confs.astype(np.float32) / TP_plus_FN.astype(np.float32)
        diags = np.diagonal(class_avg_confs, axis1=-2, axis2=-1)
        class_avg_ACC = np.sum(diags, axis=-1) / np.sum(class_avg_confs, axis=(-1, -2))

        print('Corresponding mAcc : {:.1f} %'.format(100 * class_avg_ACC[best_epoch]))

    # Plots
    # *****

    for fig_name, OA in zip(['Validation', 'Vote'], [all_val_OA, all_vote_OA]):

        # Figure
        fig = plt.figure(fig_name)
        for i, label in enumerate(list_of_labels):
            plt.plot(all_pred_epochs[i], OA[i], linewidth=1, label=label)
        plt.xlabel('epochs')
        plt.ylabel(fig_name + ' Accuracy')

        # Set limits for y axis
        #plt.ylim(0.55, 0.95)

        # Display legends and title
        plt.legend(loc=4)

        # Customize the graph
        ax = fig.gca()
        ax.grid(linestyle='-.', which='both')
        #ax.set_yticks(np.arange(0.8, 1.02, 0.02))

    #for i, label in enumerate(list_of_labels):
    #    print(label, np.max(all_train_OA[i]), np.max(all_val_OA[i]))

    # Show all
    plt.show()


def compare_convergences_SLAM(dataset, list_of_paths, list_of_names=None):

    # Parameters
    # **********

    smooth_n = 10

    if list_of_names is None:
        list_of_names = [str(i) for i in range(len(list_of_paths))]

    # Read Logs
    # *********

    all_pred_epochs = []
    all_val_mIoUs = []
    all_val_class_IoUs = []
    all_subpart_mIoUs = []
    all_subpart_class_IoUs = []

    # Load parameters
    config = Config()
    config.load(list_of_paths[0])

    class_list = [dataset.label_to_names[label] for label in dataset.label_values
                  if label not in dataset.ignored_labels]

    s = '{:^6}|'.format('mean')
    for c in class_list:
        s += '{:^6}'.format(c[:4])
    print(s)
    print(6*'-' + '|' + 6*config.num_classes*'-')
    for path in list_of_paths:

        # Get validation IoUs
        nc_model = dataset.num_classes - len(dataset.ignored_labels)
        file = join(path, 'val_IoUs.txt')
        val_IoUs = load_single_IoU(file, nc_model)

        # Get Subpart IoUs
        file = join(path, 'subpart_IoUs.txt')
        subpart_IoUs = load_single_IoU(file, nc_model)

        # Get mean IoU
        val_class_IoUs, val_mIoUs = IoU_class_metrics(val_IoUs, smooth_n)
        subpart_class_IoUs, subpart_mIoUs = IoU_class_metrics(subpart_IoUs, smooth_n)

        # Aggregate results
        all_pred_epochs += [np.array([i for i in range(len(val_IoUs))])]
        all_val_mIoUs += [val_mIoUs]
        all_val_class_IoUs += [val_class_IoUs]
        all_subpart_mIoUs += [subpart_mIoUs]
        all_subpart_class_IoUs += [subpart_class_IoUs]

        s = '{:^6.1f}|'.format(100*subpart_mIoUs[-1])
        for IoU in subpart_class_IoUs[-1]:
            s += '{:^6.1f}'.format(100*IoU)
        print(s)

    print(6*'-' + '|' + 6*config.num_classes*'-')
    for snap_IoUs in all_val_class_IoUs:
        if len(snap_IoUs) > 0:
            s = '{:^6.1f}|'.format(100*np.mean(snap_IoUs[-1]))
            for IoU in snap_IoUs[-1]:
                s += '{:^6.1f}'.format(100*IoU)
        else:
            s = '{:^6s}'.format('-')
            for _ in range(config.num_classes):
                s += '{:^6s}'.format('-')
        print(s)

    # Plots
    # *****

    # Figure
    fig = plt.figure('mIoUs')
    for i, name in enumerate(list_of_names):
        p = plt.plot(all_pred_epochs[i], all_subpart_mIoUs[i], '--', linewidth=1, label=name)
        plt.plot(all_pred_epochs[i], all_val_mIoUs[i], linewidth=1, color=p[-1].get_color())
    plt.xlabel('epochs')
    plt.ylabel('IoU')

    # Set limits for y axis
    #plt.ylim(0.55, 0.95)

    # Display legends and title
    plt.legend(loc=4)

    # Customize the graph
    ax = fig.gca()
    ax.grid(linestyle='-.', which='both')
    #ax.set_yticks(np.arange(0.8, 1.02, 0.02))

    displayed_classes = [0, 1, 2, 3, 4, 5, 6, 7]
    #displayed_classes = []
    for c_i, c_name in enumerate(class_list):
        if c_i in displayed_classes:

            # Figure
            fig = plt.figure(c_name + ' IoU')
            for i, name in enumerate(list_of_names):
                plt.plot(all_pred_epochs[i], all_val_class_IoUs[i][:, c_i], linewidth=1, label=name)
            plt.xlabel('epochs')
            plt.ylabel('IoU')

            # Set limits for y axis
            #plt.ylim(0.8, 1)

            # Display legends and title
            plt.legend(loc=4)

            # Customize the graph
            ax = fig.gca()
            ax.grid(linestyle='-.', which='both')
            #ax.set_yticks(np.arange(0.8, 1.02, 0.02))



    # Show all
    plt.show()


def compare_on_test_set(list_of_cfg,
                        list_of_paths,
                        list_of_names=None,
                        redo_test=False):

    ############
    # Parameters
    ############

    if list_of_names is None:
        list_of_names = [str(i) for i in range(len(list_of_paths))]

    all_confs = []


    ##############
    # Loop on logs
    ##############

    for path, cfg in zip(list_of_paths, list_of_cfg):

        # Define parameters
        # *****************

        # Change some parameters
        cfg.test.in_radius = 4.0
        cfg.test.batch_limit = 1
        cfg.test.steps_per_epoch = 9999999
        cfg.test.max_votes = 3
        cfg.test.chkp_idx = -1

        # Augmentations
        cfg.train.augment_anisotropic = True
        cfg.train.augment_min_scale = 0.99
        cfg.train.augment_max_scale = 1.01
        cfg.train.augment_symmetries =  [True, False, False]
        cfg.train.augment_rotation = 'vertical'
        cfg.train.augment_noise = 0.0001
        cfg.train.augment_color = 1.0

        
        # Read test results if available
        # ******************************

        found_test = None
        test_folders = []
        test_path = join(cfg.exp.log_dir, 'test')
        if exists(test_path):
            test_folders = [join(test_path, f) for f in listdir(test_path) if f.startswith('test_')]
            for test_folder in test_folders:
                
                # Load the cfg already tested
                test_cfg = load_cfg(test_folder)

                # Get differences between configs
                diff_params, _ = cfg_differences([cfg, test_cfg], ignore_params=['test.max_votes'])

                if len(diff_params) == 0:
                    found_test = test_folder
                    break


        # Compute test results if not found
        # *********************************

        # Perform test
        if found_test is None:
            test_log(path, cfg)
            
        # Get the new test folder
        test_folders2 = [join(test_path, f) for f in listdir(test_path) if f.startswith('test_')]
        for test_folder in test_folders2:
            if test_folder not in test_folders:
                found_test = test_folder
                break


        # Save test data
        # **************

        # Read confision matrix
        all_conf_pathes = np.sort([join(found_test, f) for f in listdir(found_test) if f.startswith('full_conf_')])
        
        log_confs = []
        for conf_path in all_conf_pathes:
            if isfile(conf_path):
                log_confs.append(np.loadtxt(conf_path, dtype=np.int32))

        all_confs.append(np.stack(log_confs,  axis=0))



    ##############
    # Show results
    ##############

    print('\n')
    print('________________________________________________________________________________________________________')
    underline('Test Results')

    # Get IoUs from the final vote
    all_IoUs = [IoU_from_confusions(full_conf[-1]) for full_conf in all_confs]

    class_list = [name for label, name in list_of_cfg[0].data.label_and_names
                  if label not in list_of_cfg[0].data.ignored_labels]
    
    num_classes = len(class_list)


    # Print spheres validation
    s = '{:^10}|'.format('mean')
    for c in class_list:
        s += '{:^10}'.format(c)
    print(s)
    print(10*'-' + '|' + 10*num_classes*'-')
    for log_IoUs in all_IoUs:
        s = '{:^10.1f}|'.format(100*np.mean(log_IoUs))
        for IoU in log_IoUs:
            s += '{:^10.1f}'.format(100*IoU)    
        print(s)

    print('________________________________________________________________________________________________________')
    print('\n')

    return





