
# ----------------------------------------------------------------------------------------------------------------------
#
#           Imports and global variables
#       \**********************************/
#


# Basic libs
import torch
import numpy as np
from os.path import exists, join
import time


# ----------------------------------------------------------------------------------------------------------------------
#
#           Training Function
#       \***********************/
#


def training_epoch(epoch, t0, net, optimizer, training_loader, cfg, PID_file, device, cuda_debug=True):

    step = 0
    last_display = time.time()
    t = [time.time()]

    for batch in training_loader:

        # Check kill signal (running_PID.txt deleted)
        if cfg.exp.saving and not exists(PID_file):
            continue

        ##################
        # Processing batch
        ##################

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


        # CUDA debug. Use this to check if you can use more memory on your GPU
        # torch.cuda.ipc_collect()
        torch.cuda.empty_cache()
        if cuda_debug:
            cuda_stats = torch.cuda.memory_stats(device=device)
            # fmt_str = ' ' * 60 + '{:5.0f} Kpts'
            # fmt_str += ' - Alloc {:6.0f} MB / {:6.0f} MB'
            # fmt_str += ' - Res {:6.0f} MB / {:6.0f} MB'
            # fmt_str += '  (Current/Peak)'
            # print(fmt_str.format(batch.points.shape[0] / 1000,
            #                      cuda_stats["allocated_bytes.all.current"] / 1024 ** 2,
            #                      cuda_stats["allocated_bytes.all.peak"] / 1024 ** 2,
            #                      cuda_stats["reserved_bytes.all.current"] / 1024 ** 2,
            #                      cuda_stats["reserved_bytes.all.peak"] / 1024 ** 2))


            fmt_str = ' ' * 20 + '{:8.0f} {:5.0f} {:6.0f} {:6.0f} {:6.0f} {:6.0f}'
            print(fmt_str.format(training_loader.dataset.b_lim,
                                 batch.points.shape[0] / 1000,
                                 cuda_stats["allocated_bytes.all.current"] / 1024 ** 2,
                                 cuda_stats["allocated_bytes.all.peak"] / 1024 ** 2,
                                 cuda_stats["reserved_bytes.all.current"] / 1024 ** 2,
                                 cuda_stats["reserved_bytes.all.peak"] / 1024 ** 2))




            torch.cuda.reset_peak_memory_stats()

        training_loader.dataset.b_lim += 1000



        # zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = net(batch)

        if 'cuda' in device.type:
            torch.cuda.synchronize(device)
        t += [time.time()]

        # Loss with equivar/invar
        loss = net.loss(outputs, batch.labels)
        #acc = net.accuracy(outputs, batch.labels)

        if 'cuda' in device.type:
            torch.cuda.synchronize(device)
        t += [time.time()]

        # Backward gradients
        loss.backward()

        # Clip gradient
        if cfg.train.grad_clip > 0:
            #torch.nn.utils.clip_grad_norm_(net.parameters(), cfg.train.grad_clip)
            torch.nn.utils.clip_grad_value_(net.parameters(), cfg.train.grad_clip)

        # Optimizer step
        optimizer.step()

        if 'cuda' in device.type:
            torch.cuda.synchronize(device)
        t += [time.time()]

        # Average timing
        if step < 2:
            mean_dt = np.array(t[1:]) - np.array(t[:-1])
        else:
            mean_dt = 0.9 * mean_dt + 0.1 * (np.array(t[1:]) - np.array(t[:-1]))

        # # Console display (only one per second)
        # if (t[-1] - last_display) > 1.0:
        #     last_display = t[-1]
        #     message = 'e{:03d}-i{:04d} => L={:.3f} / t(ms): {:5.1f} {:5.1f} {:5.1f} {:5.1f} {:5.1f})'
        #     print(message.format(epoch, step,
        #                             loss.item(),
        #                             1000 * mean_dt[0],
        #                             1000 * mean_dt[1],
        #                             1000 * mean_dt[2],
        #                             1000 * mean_dt[3],
        #                             1000 * mean_dt[4]))

        # Log file
        if cfg.exp.saving:
            with open(join(cfg.exp.log_dir, 'training.txt'), "a") as file:
                message = '{:d} {:d} {:.3f} {:.3f} {:.3f}\n'
                file.write(message.format(epoch,
                                            step,
                                            net.output_loss,
                                            net.reg_loss,
                                            t[-1] - t0))


        step += 1

    return