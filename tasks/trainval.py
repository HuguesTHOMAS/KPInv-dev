
# ----------------------------------------------------------------------------------------------------------------------
#
#           Imports and global variables
#       \**********************************/
#


# Basic libs
import time
import torch
from os import makedirs, remove
from os.path import exists, join
from easydict import EasyDict
import numpy as np

import matplotlib.pyplot as plt

from utils.gpu_init import init_gpu

from tasks.training import training_epoch, training_epoch_debug
from tasks.validation import validation_epoch


# ----------------------------------------------------------------------------------------------------------------------
#
#           Training and validation Function
#       \**************************************/
#


def train_and_validate(net, training_loader, val_loader, cfg, chkp_path=None, finetune=False, on_gpu=True):
    """
    Training and validation of a model on a particular dataset.    
    Args:
        net (Model): network object.
        training_loader (DataLoader): the numbers of points in the batch (B,).
        val_loader (DataLoader): the voxel size.
        cfg (EasyDict): configuration dictionary.
        chkp_path (str=None): path to the checkpoint that needs to be loaded (None for new training).
        finetune (bool=False): Finetuning, if true, the model state are restored but not the training state.
        on_gpu (bool=True): Use GPU or CPU.
    """

    ############
    # Parameters
    ############

    # Epochs and steps
    epoch = 0

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


    ######################
    # Initialize optimizer
    ######################

    # Define optimizer
    if cfg.train.optimizer == 'SGD':
        optimizer = torch.optim.SGD(net.parameters(),
                                    lr=cfg.train.lr,
                                    momentum=cfg.train.sgd_momentum,
                                    weight_decay=cfg.train.weight_decay)
    elif cfg.train.optimizer == 'Adam':
        optimizer = torch.optim.Adam(net.parameters(),
                                     lr=cfg.train.lr,
                                     betas=cfg.train.adam_b,
                                     eps=cfg.train.adam_eps,
                                     weight_decay=cfg.train.weight_decay)
    else:
        raise ValueError('Optimizer \"{:s}\" unknown. Only \"Adam\" and \"SGD\" are accepted.'.format(cfg.train.optimizer))


    ##########################
    # Load previous checkpoint
    ##########################

    if (chkp_path is not None):

        if finetune:
            # Only load model state if finetuning
            checkpoint = torch.load(chkp_path)
            net.load_state_dict(checkpoint['model_state_dict'])
            net.train()
            print("Model restored and ready for finetuning.")

        else:
            # load everything otherwise
            checkpoint = torch.load(chkp_path)
            net.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            epoch = checkpoint['epoch']
            net.train()
            print("Model and training state restored.")


    ############################
    # Prepare experiment folders
    ############################

    if cfg.exp.saving:

        # Training log file
        with open(join(cfg.exp.log_dir, 'training.txt'), "w") as file:
            file.write('epochs steps out_loss offset_loss train_accuracy time\n')

        # Killing file (simply delete this file when you want to stop the training)
        PID_file = join(cfg.exp.log_dir, 'running_PID.txt')
        if not exists(PID_file):
            with open(PID_file, "w") as file:
                file.write('Launched with PyCharm')

        # Checkpoints directory
        checkpoint_directory = join(cfg.exp.log_dir, 'checkpoints')
        if not exists(checkpoint_directory):
            makedirs(checkpoint_directory)
    else:
        checkpoint_directory = None
        PID_file = None


    ##################
    # Start experiment
    ##################

    # Validation data (running mean)
    val_data = EasyDict()

    # Loop variables
    t0 = time.time()

    # Start global loop
    for epoch in range(cfg.train.max_epoch):

        # Perform one epoch of training
        training_epoch(epoch, t0, net, optimizer, training_loader, cfg, PID_file, device)

        # Check kill signal (running_PID.txt deleted)
        if cfg.exp.saving and not exists(PID_file):
            break

        # Update learning rate
        if str(epoch) in cfg.train.lr_decays:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= cfg.train.lr_decays[str(epoch)]

        # Update epoch
        epoch += 1

        # Saving
        if cfg.exp.saving:
            # Get current state dict
            save_dict = {'epoch': epoch,
                         'model_state_dict': net.state_dict(),
                         'optimizer_state_dict': optimizer.state_dict(),
                         'saving_path': cfg.exp.log_dir}

            # Save current state of the network (for restoring purposes)
            checkpoint_path = join(checkpoint_directory, 'current_chkp.tar')
            torch.save(save_dict, checkpoint_path)

            # Save checkpoints occasionally
            if (epoch + 1) % cfg.train.checkpoint_gap == 0:
                checkpoint_path = join(checkpoint_directory, 'chkp_{:04d}.tar'.format(epoch + 1))
                torch.save(save_dict, checkpoint_path)

        # Validation
        net.eval()
        with torch.no_grad():
            validation_epoch(epoch, net, val_loader, cfg, val_data, device)
        net.train()


    # Remove the temporary file used for kill signal
    if exists(PID_file):
        remove(PID_file)

    print('Finished Training')
    return


# ----------------------------------------------------------------------------------------------------------------------
#
#           Debug CUDA Memory Function
#       \********************************/
#



def train_and_debug_cuda(net, training_loader, cfg):
    """
    We start a training (without validation) and collect CUDA memory stats while increasing the batch size 
    slowly until an OOM error occurs .    
    Args:
        net (Model): network object.
        training_loader (DataLoader): the numbers of points in the batch (B,).
        cfg (EasyDict): configuration dictionary.
    """

    ############
    # Parameters
    ############

    # Epochs and steps
    epoch = 0
    device = init_gpu()
        

    ####################
    # Initialize network
    ####################

    # Get the network to the device we chose
    net.to(device)


    ######################
    # Initialize optimizer
    ######################

    # Define optimizer
    if cfg.train.optimizer == 'SGD':
        optimizer = torch.optim.SGD(net.parameters(),
                                    lr=cfg.train.lr,
                                    momentum=cfg.train.sgd_momentum,
                                    weight_decay=cfg.train.weight_decay)
    elif cfg.train.optimizer == 'Adam':
        optimizer = torch.optim.Adam(net.parameters(),
                                     lr=cfg.train.lr,
                                     betas=cfg.train.adam_b,
                                     eps=cfg.train.adam_eps,
                                     weight_decay=cfg.train.weight_decay)
    else:
        raise ValueError('Optimizer \"{:s}\" unknown. Only \"Adam\" and \"SGD\" are accepted.'.format(cfg.train.optimizer))


    ############################
    # Prepare experiment folders
    ############################

    if cfg.exp.saving:

        # File to signal that the folder is a debug one
        PID_file = join(cfg.exp.log_dir, 'THIS_IS_A_DEBUG_EXP.txt')
        if not exists(PID_file):
            with open(PID_file, "w") as file:
                file.write('THIS_IS_A_DEBUG_EXP')

        # Killing file (simply delete this file when you want to stop the training)
        PID_file = join(cfg.exp.log_dir, 'running_PID.txt')
        if not exists(PID_file):
            with open(PID_file, "w") as file:
                file.write('Launched with PyCharm')
    else:
        PID_file = None


    ##################
    # Start experiment
    ##################

    # Perform one epoch of training, should be enough to get an OOM (increase blim_inc otherwise)
    all_cuda_stats = training_epoch_debug(epoch, net, optimizer, training_loader, cfg, PID_file, device, blim_inc=5000)
    

    #################
    # Show Cuda stats
    #################

    print('\n')
    print('Now showing statistics')
    print('\n')

    steps = np.arange(all_cuda_stats.shape[0])
    n_points = all_cuda_stats[:, 0]
    batch_limit = all_cuda_stats[:, 1]
    allocated_MB = all_cuda_stats[:, 2]
    reserved_MB = all_cuda_stats[:, 3]

    # Figure showing point statistics
    fig1 = plt.figure()
    plt.title('Points statistic')
    plt.xlabel('steps')
    plt.ylabel('Kpts')
    plt.plot(steps, batch_limit / 1000, label='batch_limit')
    plt.plot(steps, n_points / 1000, '.', label='batch_points')
    plt.legend()
    
    # Figure showing memory statistics
    fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=[10.4, 4.8])

    # Memory increasing
    ax1.set_title('Memory statistics')
    ax1.set_xlabel('steps')
    ax1.set_ylabel('MB')
    ax1.plot(steps, allocated_MB, '-', label='allocated')
    ax1.plot(steps, reserved_MB, '--', label='reserved')
    ax1.legend()

    # Relatio nbetween Memory given the number of points
    ax2.set_title('Memory given numer of points')
    ax2.set_xlabel('Kpts')
    ax2.set_ylabel('MB')
    ax2.plot(n_points / 1000, allocated_MB, '.')

    plt.show()

    # Remove the temporary file used for kill signal
    if exists(PID_file):
        remove(PID_file)

    print('Finished Training')
    return
