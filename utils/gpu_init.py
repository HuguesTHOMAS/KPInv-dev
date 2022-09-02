
import os
import torch

global USED_GPU
USED_GPU = ''


# ----------------------------------------------------------------------------------------------------------------------
#
#           GPU Init
#       \**************/
#


def init_gpu(gpu_id='0'):
    """
    Initialize the USED_GPU global variable to a free GPU, or retrieve its actual value.
    Args:
        gpu_id (str): Index of the wanted GPU or 'auto' to choose a free GPU automatically.
    """

    global USED_GPU
    
    if len(USED_GPU) == 0:
        if gpu_id == 'auto':
            
            # Automatic GPU choice, find a free GPU on the machine
            # (need pynvml to be installed)
            print('\nSearching a free GPU:')
            for i in range(torch.cuda.device_count()):
                a = torch.cuda.list_gpu_processes(i)
                print(torch.cuda.list_gpu_processes(i))
                a = a.split()
                if a[1] == 'no':
                    gpu_id = a[0][-1:]

        # Safe check no free GPU
        if gpu_id == 'auto':
            print('\nNo free GPU found!\n')
            a = 1 / 0

        else:
            print('Using GPU:', gpu_id, '\n')

        # Set GPU visible device
        USED_GPU = gpu_id

    return torch.device("cuda:" + USED_GPU)


def tensor_MB(a):
    return round(a.element_size() * a.nelement() / 1024 / 1024, 2)