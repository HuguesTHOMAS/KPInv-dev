
# ----------------------------------------------------------------------------------------------------------------------
#
#           Imports and global variables
#       \**********************************/
#


# Basic libs
import time
import torch
from os import makedirs, remove, listdir
from os.path import exists, join
from easydict import EasyDict
import numpy as np

import matplotlib.pyplot as plt

from utils.printing import underline, frame_lines_1, color_str
from utils.gpu_init import init_gpu
from utils.ply import read_ply, write_ply
from utils.config import save_cfg

from tasks.training import training_epoch, training_epoch_debug
from tasks.validation import validation_epoch
from utils.metrics import fast_confusion, IoU_from_confusions

# ----------------------------------------------------------------------------------------------------------------------
#
#           Training and validation Function
#       \**************************************/
#


def test_model(net, test_loader, cfg, on_gpu=True, save_visu=False):
    """
    Training and validation of a model on a particular dataset.    
    Args:
        net (Model): network object.
        test_loader (DataLoader): the loader for test data.
        cfg (EasyDict): configuration dictionary.
        on_gpu (bool=True): Use GPU or CPU.
    """

    ############
    # Parameters
    ############

    # Choose to train on CPU or GPU
    if on_gpu and torch.cuda.is_available():
        device = init_gpu()
    else:
        device = torch.device("cpu")
        

    ####################
    # Initialize network
    ####################

    # Get the network to the device we chose
    net.to(device)


    ######################################
    # Choose which type of test we perform
    ######################################    

    if cfg.data.task == 'cloud_segmentation':
        test_epoch_func = cloud_segmentation_test

    # elif cfg.data.task == 'classification':
    #     test_epoch_func = object_classification_test

    # elif cfg.data.task == 'part_segmentation':
    #     test_epoch_func = object_segmentation_test

    # elif cfg.data.task == 'multi_part_segmentation':
    #     test_epoch_func = object_segmentation_test

    # elif cfg.data.task == 'slam_segmentation':
    #     test_epoch_func = slam_segmentation_test

    else:
        raise ValueError('No validation method implemented for this network type')


    ####################
    # Create test folder
    ####################

    test_path = join(cfg.exp.log_dir, 'test')
    if not exists(test_path):
        makedirs(test_path)
    
    # Create a new test folder
    test_folders = [f for f in listdir(test_path) if f.startswith('test_')]
    test_inds = [int(f.split('_')[-1]) for f in test_folders]
    new_ind = 1
    while new_ind in test_inds:
        new_ind += 1

    new_test_path = join(test_path, 'test_{:03d}'.format(new_ind))
    if not exists(new_test_path):
        makedirs(new_test_path)
        
    # Save parameters used for test
    save_cfg(cfg, path=new_test_path)


    ##################
    # Start experiment
    ##################

    # Validation data (running mean)
    vote_n = 0
    test_data = EasyDict()

    # Start global loop
    while vote_n < cfg.test.max_votes:

        # Perform one peoch of test
        with torch.no_grad():
            test_epoch_func(vote_n,
                            net,
                            test_loader,
                            cfg,
                            test_data,
                            device,
                            saving_path=new_test_path,
                            save_visu=save_visu)

        # Create new sampling points for next test epoch
        t1 = time.time()
        print('Creating new sampling points for next test epoch')
        test_loader.dataset.reg_sampling_i *= 0
        test_loader.dataset.new_reg_sampling_pts()
        test_loader.dataset.reg_votes += 1
        t2 = time.time()
        print('Done in {:.1f}s\n'.format(t2 - t1))

        vote_n += 1

    return






# ----------------------------------------------------------------------------------------------------------------------
#
#           Validation Functions
#       \**************************/
#


def cloud_segmentation_test(epoch, net, test_loader, cfg, test_data, device, saving_path=None, save_visu=False):
    """
    Test method for cloud segmentation models
    """
    
    ############
    # Initialize
    ############

    saving = saving_path is not None
    
    underline('Test epoch {:d}'.format(epoch))
    message =  '\n                                                          Timings        '
    message += '\n Steps |   Votes   | GPU usage |      Speed      |   In   Batch  Forw  End '
    message += '\n-------|-----------|-----------|-----------------|-------------------------'
    print(message)

    t0 = time.time()

    # Choose validation smoothing parameter (0 for no smothing, 0.99 for big smoothing)
    test_smooth = cfg.test.test_momentum
    softmax = torch.nn.Softmax(1)

    # Number of classes predicted by the model
    nc_model = net.num_logits

    # Initiate global prediction over validation clouds
    if 'probs' not in test_data:
        test_data.probs = [torch.zeros((l.shape[0], nc_model), device=device) for l in test_loader.dataset.input_labels]
        test_data.proportions = np.zeros(nc_model, dtype=np.float32)
        i = 0
        for label_value in test_loader.dataset.label_values:
            if label_value not in test_loader.dataset.ignored_labels:
                test_data.proportions[i] = np.sum([np.sum(test_lbls == label_value)
                                                    for test_lbls in test_loader.dataset.val_labels])
                i += 1

    #####################
    # Network predictions
    #####################

    t = [time.time()]
    last_display = time.time()
    mean_dt = np.zeros(1)
    empty_count = 0

    t1 = time.time()

    # Start validation loop
    for step, batch in enumerate(test_loader):

        # First verify that the batch is not empty
        if len(batch.in_dict.points) < 1:
            empty_count += 1
            if empty_count > cfg.test.num_workers:
                break
            continue
        empty_count = 0

        for blengths in batch.in_dict.lengths:
            if blengths.item() < 20:
                print(' ' * 70, blengths.item())

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
        stacked_probs = softmax(outputs)
        lengths = batch.in_dict.lengths[0]
        lengths0 = batch.in_dict.lengths0
        in_inds = batch.in_dict.input_inds
        in_invs = batch.in_dict.input_invs
        cloud_inds = batch.in_dict.cloud_inds

        # Get predictions and labels per instance
        # ***************************************

        i0 = 0
        j0 = 0
        for b_i, length in enumerate(lengths):

            # Get prediction
            length0 = lengths0[b_i]
            probs = stacked_probs[i0:i0 + length]
            inds = in_inds[j0:j0 + length0]
            invs = in_invs[j0:j0 + length0]
            c_i = cloud_inds[b_i]

            # Update current probs in whole cloud
            test_data.probs[c_i][inds] *= test_smooth
            test_data.probs[c_i][inds] += (1 - test_smooth) * probs[invs]


            ####
            # truths = test_loader.dataset.input_labels[c_i][inds]

            # print('')
            # for pppi, ppp in enumerate((probs[invs] * 100).type(torch.int32)):
            #     s = ''
            #     for ppi, pp in enumerate(ppp):
            #         if ppi == truths[pppi]:
            #             s += color_str('{:3d} '.format(pp.item()), 'OKGREEN')
            #         else:
            #             s += color_str('{:3d} '.format(pp.item()), 'FAIL')
            #     print(s)
            # print('')
            
            # print('-------------------------------------------------------------------------------------')
            
            # pred_values = np.array(cfg.data.pred_values, dtype=np.int32)
            # sub_preds = test_loader.dataset.probs_to_preds(probs[invs].cpu().numpy())
            # ccc = fast_confusion(truths, sub_preds, pred_values).astype(np.float32)

            # ccc = (10000 * ccc / np.sum(ccc)).astype(np.int64)
            # print(ccc)
            # print('')
            ####

            i0 += length
            j0 += length0

        # Get CUDA memory stat to see what space is used on GPU
        if 'cuda' in device.type:
            cuda_stats = torch.cuda.memory_stats(device)
            used_GPU_MB = cuda_stats["allocated_bytes.all.peak"]
            _, tot_GPU_MB = torch.cuda.mem_get_info(device)
            gpu_usage = 100 * used_GPU_MB / tot_GPU_MB
            torch.cuda.reset_peak_memory_stats(device)
        else:
            gpu_usage = 0

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
                                 test_loader.dataset.get_votes(),
                                 gpu_usage,
                                 60 / np.sum(mean_dt),
                                 1000 * mean_dt[0],
                                 1000 * mean_dt[1],
                                 1000 * mean_dt[2],
                                 1000 * mean_dt[3]))


    t2 = time.time()

    print('\nOne vote computed in {:.1f}s\n'.format(t2 - t1))
    
    # # Check test data
    # pred_c = 0
    # tot_c = 0
    # for scene_probs in test_data.probs:
    #     predicted_mask = torch.sum(scene_probs, dim=1) > 0.001
    #     pred_c += torch.sum(predicted_mask.type(torch.long)).item()
    #     tot_c += predicted_mask.shape[0]
    # print('Percentage of the points having a prediction: {:.1f}%'.format(100 * pred_c / tot_c))


    # Get scores
    # **********

    print('Get predictions (and scores if labels are available)')

    # Get all predictions
    all_sub_preds = []
    for c_i, sub_probs in enumerate(test_data.probs):
        
        # Get probs on subsampled points
        sub_probs = test_data.probs[c_i].cpu().numpy()

        # Get the predicted labels
        sub_preds = test_loader.dataset.probs_to_preds(sub_probs)
        all_sub_preds.append(sub_preds.astype(np.int32))
        
    # Check if test labels are available
    available_labels = test_loader.dataset.input_labels[0].shape[0] > 0
    all_sub_labels = []
    sub_confs = []
    if available_labels:
        for c_i, sub_labels in enumerate(test_loader.dataset.input_labels):

            # Targets
            all_sub_labels.append(sub_labels)

            # Confs
            pred_values = np.array(cfg.data.pred_values, dtype=np.int32)
            sub_confs += [fast_confusion(sub_labels, all_sub_preds[c_i], pred_values).astype(np.int64)]

    t3 = time.time()
    print('Done in {:.1f}s\n'.format(t3 - t2))

    # Reproject on full clouds
    # ************************

    print('Reproject predictions on full clouds (and optional save)')
    
    # Get points
    all_preds = []
    for c_i, sub_preds in enumerate(all_sub_preds):

        # Reproject preds on the evaluations points
        preds = (sub_preds[test_loader.dataset.test_proj[c_i]]).astype(np.int32)
        all_preds.append(preds)

    t4 = time.time()
    print('Done in {:.1f}s\n'.format(t4 - t3))

        

    # Get scores on full clouds
    # *************************

    full_confs = []
    if available_labels:

        print('Get scores on full clouds')
        for c_i, preds in enumerate(all_preds):

            # Get groundtruth labels
            labels = test_loader.dataset.val_labels[c_i].astype(np.int32)

            # Confusion matrix
            pred_values = np.array(cfg.data.pred_values, dtype=np.int32)
            full_confs.append(fast_confusion(labels, preds, pred_values).astype(np.int64))

    t5 = time.time()
    if available_labels:
        print('Done in {:.1f}s\n'.format(t5 - t4))


    # Save predicted cloud
    # ********************

    if saving:

        print('Save results')
            
        files = test_loader.dataset.scene_files
        for c_i, sub_preds in enumerate(all_sub_preds):

            # Save subsampled for visu
            # ************************

            if save_visu:

                # Path of saved validation file
                cloud_name = files[c_i].split('/')[-1]
                sub_name = join(saving_path, 'sub_' + cloud_name)

                # Get subsampled points from tree structure
                sub_points = np.array(test_loader.dataset.input_trees[c_i].data, copy=False)
                if test_loader.dataset.cylindric_input:
                    support_points = np.hstack((support_points, test_loader.dataset.input_z[i]))

                # We first save the subsampled version for visu
                if available_labels:
                    write_ply(sub_name,
                            [sub_points, sub_preds, all_sub_labels[c_i].astype(np.int32)],
                            ['x', 'y', 'z', 'preds', 'class'])
                else:
                    write_ply(sub_name,
                            [sub_points, sub_preds],
                            ['x', 'y', 'z', 'preds'])


            # Save full for benchmarks
            # ************************

            if cfg.data.name == 'Semantic3D':
                test_name = join(saving_path, cloud_name[:-4] + '.txt')
                #TODO

        t6 = time.time()
        print('Done in {:.1f}s\n'.format(t6 - t5))


    # Report on scores
    # ****************
    

    if available_labels:

        report_lines = ['']
        report_lines += ['Score report on sub_clouds / full_clouds']
        report_lines += ['*' * len(report_lines[1])]
        report_lines += ['']

        # Merge confusions
        sub_confs = np.stack(sub_confs)
        full_confs = np.stack(full_confs)
        sub_conf = np.sum(np.stack(sub_confs), axis=0)
        full_conf = np.sum(np.stack(full_confs), axis=0)

        # Get IoUs
        sub_IoUs = IoU_from_confusions(sub_conf)
        full_IoUs = IoU_from_confusions(full_conf)

        # Print
        class_list = [name for label, name in cfg.data.label_and_names
                    if label not in cfg.data.ignored_labels]
                    
        s = '{:^10}|'.format('mean')
        for c in class_list:
            s += '{:^10}'.format(c)
        report_lines += [s]
        
        report_lines += [10*'-' + '|' + 10*nc_model*'-']

        s = '{:^10.1f}|'.format(100*np.mean(sub_IoUs))
        for IoU in sub_IoUs:
            s += '{:^10.1f}'.format(100*IoU)
        report_lines += [s]
        
        report_lines += [10*'-' + '|' + 10*nc_model*'-']

        s = '{:^10.1f}|'.format(100*np.mean(full_IoUs))
        for IoU in full_IoUs:
            s += '{:^10.1f}'.format(100*IoU)
        report_lines += [s]
        
        report_lines += ['']

        report_str = frame_lines_1(report_lines)

        print('\n')
        
        # Save report in text file
        if saving:
            report_path = join(saving_path, 'report.txt')
            with open(report_path, "a") as text_file:
                text_file.write('\nVote {:d}: \n\n'.format(epoch))
                text_file.write(report_str)
                text_file.write('\n\n')

            conf_path = join(saving_path, 'full_conf_{:03d}.txt'.format(epoch))
            np.savetxt(conf_path, full_conf, '%15d')


    return

















