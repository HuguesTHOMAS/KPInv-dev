
# ----------------------------------------------------------------------------------------------------------------------
#
#           Imports and global variables
#       \**********************************/
#


# Basic libs
import torch
import numpy as np
from os import makedirs
from os.path import exists, join
import time

# PLY reader
from utils.ply import read_ply, write_ply
from utils.metrics import IoU_from_confusions, fast_confusion


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

    t0 = time.time()

    # Choose validation smoothing parameter (0 for no smothing, 0.99 for big smoothing)
    val_smooth = 0.95
    softmax = torch.nn.Softmax(1)

    # Do not validate if dataset has no validation cloud
    if val_loader.dataset.validation_split not in val_loader.dataset.all_splits:
        return

    # Number of classes including ignored labels
    nc_tot = val_loader.dataset.num_classes

    # Number of classes predicted by the model
    nc_model = cfg.data.num_classes

    # Initiate global prediction over validation clouds
    if 'probs' not in val_data:
        val_data.probs = [np.zeros((l.shape[0], nc_model))
                                    for l in val_loader.dataset.input_labels]
        val_data.proportions = np.zeros(nc_model, dtype=np.float32)
        i = 0
        for label_value in val_loader.dataset.label_values:
            if label_value not in val_loader.dataset.ignored_labels:
                val_data.proportions[i] = np.sum([np.sum(labels == label_value)
                                                    for labels in val_loader.dataset.validation_labels])
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
    for i, batch in enumerate(val_loader):

        # New time
        t = t[-1:]
        t += [time.time()]

        if 'cuda' in device.type:
            batch.to(device)

        # Forward pass
        outputs = net(batch)

        # Get probs and labels
        stacked_probs = softmax(outputs).cpu().detach().numpy()
        labels = batch.labels.cpu().numpy()
        lengths = batch.lengths.cpu().numpy()
        lengths0 = batch.lengths0.cpu().numpy()
        in_inds = batch.input_inds.cpu().numpy()
        in_invs = batch.input_invs.cpu().numpy()
        cloud_inds = batch.cloud_inds.cpu().numpy()
        torch.cuda.synchronize(device)

        # Get predictions and labels per instance
        # ***************************************

        i0 = 0
        for b_i, length in enumerate(lengths):

            # Get prediction
            length0 = lengths0[b_i]
            target = labels[i0:i0 + length]
            probs = stacked_probs[i0:i0 + length]
            inds = in_inds[i0:i0 + length0]
            invs = in_invs[i0:i0 + length0]
            c_i = cloud_inds[b_i]

            # Update current probs in whole cloud
            val_data.probs[c_i][inds] *= val_smooth
            val_data.probs[c_i][inds] += (1 - val_smooth) * probs[invs]

            # Stack all prediction for this epoch
            predictions.append(probs[invs])
            targets.append(target[invs])
            i0 += length

        # Average timing
        t += [time.time()]
        mean_dt = 0.95 * mean_dt + 0.05 * (np.array(t[1:]) - np.array(t[:-1]))

        # Display
        if (t[-1] - last_display) > 1.0:
            last_display = t[-1]
            message = 'Validation : {:.1f}% (timings : {:4.2f} {:4.2f})'
            print(message.format(100 * i / cfg.test.steps_per_epoch,
                                    1000 * (mean_dt[0]),
                                    1000 * (mean_dt[1])))

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
    C = np.sum(Confs, axis=0).astype(np.float32)

    # Remove ignored labels from confusions
    for l_ind, label_value in reversed(list(enumerate(val_loader.dataset.label_values))):
        if label_value in val_loader.dataset.ignored_labels:
            C = np.delete(C, l_ind, axis=0)
            C = np.delete(C, l_ind, axis=1)

    # Balance with real validation proportions
    C *= np.expand_dims(val_data.proportions / (np.sum(C, axis=1) + 1e-6), 1)


    t4 = time.time()

    # Objects IoU
    IoUs = IoU_from_confusions(C)

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

        # Save potentials
        pot_path = join(cfg.exp.log_dir, 'potentials')
        if not exists(pot_path):
            makedirs(pot_path)
        files = val_loader.dataset.files
        for i, file_path in enumerate(files):
            pot_points = np.array(val_loader.dataset.pot_trees[i].data, copy=False)
            cloud_name = file_path.split('/')[-1]
            pot_name = join(pot_path, cloud_name)
            pots = val_loader.dataset.potentials[i].numpy().astype(np.float32)
            write_ply(pot_name,
                        [pot_points.astype(np.float32), pots],
                        ['x', 'y', 'z', 'pots'])

    t6 = time.time()

    # Print instance mean
    mIoU = 100 * np.mean(IoUs)
    print('{:s} mean IoU = {:.1f}%'.format(cfg.data.name, mIoU))

    # Save predicted cloud occasionally
    if cfg.exp.saving and (epoch + 1) % cfg.train.checkpoint_gap == 0:
        val_path = join(cfg.exp.log_dir, 'val_preds_{:d}'.format(epoch + 1))
        if not exists(val_path):
            makedirs(val_path)
        files = val_loader.dataset.files
        for i, file_path in enumerate(files):

            # Get points
            points = val_loader.dataset.load_evaluation_points(file_path)

            # Get probs on our own ply points
            sub_probs = val_data.probs[i]

            # Insert false columns for ignored labels
            for l_ind, label_value in enumerate(val_loader.dataset.label_values):
                if label_value in val_loader.dataset.ignored_labels:
                    sub_probs = np.insert(sub_probs, l_ind, 0, axis=1)

            # Get the predicted labels
            sub_preds = val_loader.dataset.label_values[np.argmax(sub_probs, axis=1).astype(np.int32)]

            tt_prob = (sub_probs[val_loader.dataset.test_proj[i], 0]).astype(np.float32)

            # Reproject preds on the evaluations points
            preds = (sub_preds[val_loader.dataset.test_proj[i]]).astype(np.int32)

            # Path of saved validation file
            cloud_name = file_path.split('/')[-1]
            val_name = join(val_path, cloud_name)

            # Save file
            labels = val_loader.dataset.validation_labels[i].astype(np.int32)
            write_ply(val_name,
                        [points, preds, labels, tt_prob],
                        ['x', 'y', 'z', 'preds', 'class', 'tt_prob'])

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

























