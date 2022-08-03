
import os
import torch


# ----------------------------------------------------------------------------------------------------------------------
#
#           GPU Init
#       \**************/
#


def init_gpu(gpu_id='auto'):
    """
    Initialize the CUDA_VISIBLE_DEVICES environment variable to a unused GPU, or the demanded GPU.
    Args:
        gpu_id (str): Index of the wanted GPU or 'auto' to choose a free GPU automatically.
    """

    if not 'CUDA_VISIBLE_DEVICES' in os.environ:

        # Set which gpu is going to be used (auto for automatic choice)
        gpu_id = 'auto'

        # Automatic choice (need pynvml to be installed)
        if gpu_id == 'auto':
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
            print('\nUsing GPU:', gpu_id, '\n')

        # Set GPU visible device
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id

    return torch.device("cuda")


def tensor_MB(a):
    return round(a.element_size() * a.nelement() / 1024 / 1024, 2)