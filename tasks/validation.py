
# ----------------------------------------------------------------------------------------------------------------------
#
#           Imports and global variables
#       \**********************************/
#


# Basic libs
import torch
import numpy as np
from os import makedirs, listdir
from os.path import exists, join
import time
import pickle

# PLY reader
from utils.ply import read_ply, write_ply
from utils.metrics import IoU_from_confusions, fast_confusion

from utils.printing import underline

# ----------------------------------------------------------------------------------------------------------------------
#
#           Validation Choice
#       \***********************/
#


def validation_epoch(epoch, net, val_loader, cfg, val_data, device):

    if cfg.data.task == 'classification':
        object_classification_validation(epoch, net, val_loader, cfg, val_data, device)

    elif cfg.data.task == 'part_segmentation':
        object_segmentation_validation(epoch, net, val_loader, cfg, val_data, device)

    elif cfg.data.task == 'multi_part_segmentation':
        object_segmentation_validation(epoch, net, val_loader, cfg, val_data, device)

    elif cfg.data.task == 'cloud_segmentation':
        cloud_segmentation_validation(epoch, net, val_loader, cfg, val_data, device)

    elif cfg.data.task == 'slam_segmentation':
        slam_segmentation_validation(epoch, net, val_loader, cfg, val_data, device)

    elif cfg.data.task == 'normals_regression':
        regression_validation(epoch, net, val_loader, cfg, val_data, device)
    else:
        raise ValueError('No validation method implemented for this network type')

    return


# ----------------------------------------------------------------------------------------------------------------------
#
#           Validation Functions
#       \**************************/
#


def cloud_segmentation_validation(epoch, net, val_loader, cfg, val_data, device, debug=False):
    """
    Validation method for cloud segmentation models
    """
    
    ############
    # Initialize
    ############
    
    underline('Validation epoch {:d}'.format(epoch))
    message =  '\n                                                          Timings        '
    message += '\n Steps |   Votes   | GPU usage |      Speed      |   In   Batch  Forw  End '
    message += '\n-------|-----------|-----------|-----------------|-------------------------'
    print(message)

    t0 = time.time()

    # Choose validation smoothing parameter (0 for no smothing, 0.99 for big smoothing)
    val_smooth = 0.95
    softmax = torch.nn.Softmax(1)

    # Number of classes including ignored labels
    nc_tot = cfg.data.num_classes

    # Number of classes predicted by the model
    nc_model = net.num_logits

    # Initiate global prediction over validation clouds
    if 'probs' not in val_data:
        val_data.probs = [np.zeros((l.shape[0], nc_model))
                                    for l in val_loader.dataset.input_labels]
        val_data.proportions = np.zeros(nc_model, dtype=np.float32)
        i = 0
        for label_value in val_loader.dataset.label_values:
            if label_value not in val_loader.dataset.ignored_labels:
                val_data.proportions[i] = np.sum([np.sum(val_lbls == label_value)
                                                    for val_lbls in val_loader.dataset.val_labels])
                i += 1

    #####################
    # Network predictions
    #####################

    predictions = []
    targets = []

    t = [time.time()]
    last_display = time.time()
    mean_dt = np.zeros(1)


    t1 = time.time()

    # Start validation loop
    for step, batch in enumerate(val_loader):

        # New time
        t = t[-1:]
        if 'cuda' in device.type:
            torch.cuda.synchronize(device)
        t += [time.time()]

        if 'cuda' in device.type:
            batch.to(device)

        if 'cuda' in device.type:
            torch.cuda.synchronize(device)
        t += [time.time()]

        # Forward pass
        outputs = net(batch)
        
        if 'cuda' in device.type:
            torch.cuda.synchronize(device)
        t += [time.time()]

        # Get probs and labels
        stacked_probs = softmax(outputs).cpu().detach().numpy()
        labels = batch.in_dict.labels.cpu().numpy()
        lengths = batch.in_dict.lengths[0].cpu().numpy()
        lengths0 = batch.in_dict.lengths0.cpu().numpy()
        in_inds = batch.in_dict.input_inds.cpu().numpy()
        in_invs = batch.in_dict.input_invs.cpu().numpy()
        cloud_inds = batch.in_dict.cloud_inds.cpu().numpy()

        # Get predictions and labels per instance
        # ***************************************

        i0 = 0
        j0 = 0
        for b_i, length in enumerate(lengths):

            # Get prediction
            length0 = lengths0[b_i]
            target = labels[i0:i0 + length]
            probs = stacked_probs[i0:i0 + length]
            inds = in_inds[j0:j0 + length0]
            invs = in_invs[j0:j0 + length0]
            c_i = cloud_inds[b_i]

            # Update current probs in whole cloud
            val_data.probs[c_i][inds] *= val_smooth
            val_data.probs[c_i][inds] += (1 - val_smooth) * probs[invs]

            # Stack all prediction for this epoch
            predictions.append(probs[invs])
            targets.append(target[invs])
            i0 += length
            j0 += length0


        # Get CUDA memory stat to see what space is used on GPU
        cuda_stats = torch.cuda.memory_stats(device)
        used_GPU_MB = cuda_stats["allocated_bytes.all.peak"]
        _, tot_GPU_MB = torch.cuda.mem_get_info(device)
        gpu_usage = 100 * used_GPU_MB / tot_GPU_MB
        torch.cuda.reset_peak_memory_stats(device)

        # # Empty GPU cache (helps avoiding OOM errors)
        # # Loses ~10% of speed but allows batch 2 x bigger.
        # torch.cuda.empty_cache()

        if 'cuda' in device.type:
            torch.cuda.synchronize(device)
        t += [time.time()]

        # Average timing
        if step < 5:
            mean_dt = np.array(t[1:]) - np.array(t[:-1])
        else:
            mean_dt = 0.8 * mean_dt + 0.2 * (np.array(t[1:]) - np.array(t[:-1]))

        # Display
        if (t[-1] - last_display) > 1.0:
            last_display = t[-1]
            message = ' {:5d} | {:9.2f} | {:7.1f} % | {:7.1f} stp/min | {:6.1f} {:5.1f} {:5.1f} {:5.1f}'
            print(message.format(step,
                                 val_loader.dataset.get_votes(),
                                 gpu_usage,
                                 60 / np.sum(mean_dt),
                                 1000 * mean_dt[0],
                                 1000 * mean_dt[1],
                                 1000 * mean_dt[2],
                                 1000 * mean_dt[3]))

    t2 = time.time()

    # Confusions for our subparts of validation set
    Confs = np.zeros((len(predictions), nc_tot, nc_tot), dtype=np.int32)
    for i, (probs, truth) in enumerate(zip(predictions, targets)):

        # Insert false columns for ignored labels
        for l_ind, label_value in enumerate(val_loader.dataset.label_values):
            if label_value in val_loader.dataset.ignored_labels:
                probs = np.insert(probs, l_ind, 0, axis=1)

        # Predicted labels
        preds = val_loader.dataset.label_values[np.argmax(probs, axis=1)]

        # Confusions
        Confs[i, :, :] = fast_confusion(truth, preds, val_loader.dataset.label_values).astype(np.int32)


    t3 = time.time()

    # Sum all confusions
    sum_Confs = np.sum(Confs, axis=0).astype(np.float32)

    # Remove ignored labels from confusions
    for l_ind, label_value in reversed(list(enumerate(val_loader.dataset.label_values))):
        if label_value in val_loader.dataset.ignored_labels:
            sum_Confs = np.delete(sum_Confs, l_ind, axis=0)
            sum_Confs = np.delete(sum_Confs, l_ind, axis=1)

    # Balance with real validation proportions
    sum_Confs *= np.expand_dims(val_data.proportions / (np.sum(sum_Confs, axis=1) + 1e-6), 1)

    t4 = time.time()

    # Objects IoU
    IoUs = IoU_from_confusions(sum_Confs)

    t5 = time.time()

    # Saving (optionnal)
    if cfg.exp.saving:

        # Name of saving file
        test_file = join(cfg.exp.log_dir, 'val_IoUs.txt')

        # Line to write:
        line = ''
        for IoU in IoUs:
            line += '{:.3f} '.format(IoU)
        line = line + '\n'

        # Write in file
        if exists(test_file):
            with open(test_file, "a") as text_file:
                text_file.write(line)
        else:
            with open(test_file, "w") as text_file:
                text_file.write(line)

        # # Save potentials
        # pot_path = join(cfg.exp.log_dir, 'potentials')
        # if not exists(pot_path):
        #     makedirs(pot_path)
        # files = val_loader.dataset.scene_files
        # for i, file_path in enumerate(files):
        #     pot_points = np.array(val_loader.dataset.pot_trees[i].data, copy=False)
        #     cloud_name = file_path.split('/')[-1]
        #     pot_name = join(pot_path, cloud_name)
        #     pots = val_loader.dataset.potentials[i].numpy().astype(np.float32)
        #     write_ply(pot_name,
        #                 [pot_points.astype(np.float32), pots],
        #                 ['x', 'y', 'z', 'pots'])

    t6 = time.time()

    # Print instance mean
    mIoU = 100 * np.mean(IoUs)
    print('\n{:s} mean IoU = {:.1f}%'.format(cfg.data.name, mIoU))
    print()

    # Save predicted cloud occasionally
    # *********************************

    # Create validation folder
    val_path = join(cfg.exp.log_dir, 'validation')
    if not exists(val_path):
        makedirs(val_path)
    current_votes = val_loader.dataset.get_votes()
    last_vote = int(np.floor(current_votes))

    # Check if vote has already been saved
    saved_votes = np.sort([int(l.split('_')[1]) for l in listdir(val_path) if  l.startswith('preds_')])
    if last_vote not in saved_votes:

        preds_path = join(val_path, 'preds_{:d}_{:d}.pkl'.format(last_vote, epoch + 1))

        # Save the vote predictions as pickle obj
        with open(preds_path, 'wb') as f:
            pickle.dump(val_data.probs, f)

        # Save the subsampled input clouds with latest predictions
        files = val_loader.dataset.scene_files
        for c_i, file_path in enumerate(files):

            # Get subsampled points from tree structure
            points = np.array(val_loader.dataset.input_trees[c_i].data, copy=False)

            # Get probs on our own ply points
            sub_probs = val_data.probs[c_i]

            # Insert false columns for ignored labels
            for l_ind, label_value in enumerate(val_loader.dataset.label_values):
                if label_value in val_loader.dataset.ignored_labels:
                    sub_probs = np.insert(sub_probs, l_ind, 0, axis=1)

            # Get the predicted labels
            sub_preds = val_loader.dataset.label_values[np.argmax(sub_probs, axis=1).astype(np.int32)]

            # Path of saved validation file
            cloud_name = file_path.split('/')[-1]
            val_name = join(val_path, cloud_name)

            # Save file
            labels = val_loader.dataset.input_labels[c_i]
            write_ply(val_name,
                      [points, sub_preds.astype(np.int32), labels.astype(np.int32)],
                      ['x', 'y', 'z', 'preds', 'class'])



    # Display timings
    t7 = time.time()
    if debug:
        print('\n************************\n')
        print('Validation timings:')
        print('Init ...... {:.1f}s'.format(t1 - t0))
        print('Loop ...... {:.1f}s'.format(t2 - t1))
        print('Confs ..... {:.1f}s'.format(t3 - t2))
        print('Confs bis . {:.1f}s'.format(t4 - t3))
        print('IoU ....... {:.1f}s'.format(t5 - t4))
        print('Save1 ..... {:.1f}s'.format(t6 - t5))
        print('Save2 ..... {:.1f}s'.format(t7 - t6))
        print('\n************************\n')

    return


def object_classification_validation(epoch, net, val_loader, cfg, val_data, device, debug=False):
    return


def object_segmentation_validation(epoch, net, val_loader, cfg, val_data, device, debug=False):
    return


def slam_segmentation_validation(epoch, net, val_loader, cfg, val_data, device, debug=False):
    return


def regression_validation(epoch, net, val_loader, cfg, val_data, device, debug=False):
    return

























