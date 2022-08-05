
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


def training_epoch(epoch, t0, net, optimizer, training_loader, cfg, PID_file, device):

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

        # Move batch to GPU
        if 'cuda' in device.type:
            batch.to(device)

        if 'cuda' in device.type:
            torch.cuda.synchronize(device)
        t += [time.time()]

        # zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = net(batch)

        if 'cuda' in device.type:
            torch.cuda.synchronize(device)
        t += [time.time()]

        # Loss with equivar/invar
        loss = net.loss(outputs, batch.in_dict.labels)
        #acc = net.accuracy(outputs, batch.in_dict.labels)

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

        # # Empty GPU cache (helps avoiding OOM errors)
        # # Loses ~10% of speed but allows batch 2 x bigger.
        # torch.cuda.empty_cache()

        if 'cuda' in device.type:
            torch.cuda.synchronize(device)
        t += [time.time()]

        # Average timing
        if step < 2:
            mean_dt = np.array(t[1:]) - np.array(t[:-1])
        else:
            mean_dt = 0.9 * mean_dt + 0.1 * (np.array(t[1:]) - np.array(t[:-1]))

        # Console display (only one per second)
        if (t[-1] - last_display) > 1.0:
            last_display = t[-1]
            message = 'e{:03d}-i{:04d} => L={:.3f} / t(ms): {:5.1f} {:5.1f} {:5.1f} {:5.1f} {:5.1f})'
            print(message.format(epoch, step,
                                    loss.item(),
                                    1000 * mean_dt[0],
                                    1000 * mean_dt[1],
                                    1000 * mean_dt[2],
                                    1000 * mean_dt[3],
                                    1000 * mean_dt[4]))

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


def training_epoch_debug(epoch, net, optimizer, training_loader, cfg, PID_file, device, blim_inc=1000):

    # Variables
    step = 0
    t = [time.time()]
    all_cuda_stats = []

    try:

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

            # Move batch to GPU
            if 'cuda' in device.type:
                batch.to(device)

            if 'cuda' in device.type:
                torch.cuda.synchronize(device)
            t += [time.time()]

            # zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = net(batch)

            if 'cuda' in device.type:
                torch.cuda.synchronize(device)
            t += [time.time()]

            # Loss with equivar/invar
            loss = net.loss(outputs, batch.in_dict.labels)
            #acc = net.accuracy(outputs, batch.in_dict.labels)

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


            # CUDA debug. Use this to check if you can use more memory on your GPU
            cuda_stats = torch.cuda.memory_stats(device=device)
            fmt_str = 'DEBUG:  e{:03d}-i{:04d}'.format(epoch, step)
            fmt_str += '     Batch: {:5.0f} Kpts / {:5.0f} Kpts'
            fmt_str += '     Allocated: {:6.0f} MB'
            fmt_str += '     Reserved: {:6.0f} MB'
            print(fmt_str.format(batch.in_dict.points[0].shape[0] / 1000,
                                    training_loader.dataset.b_lim  / 1000,
                                    cuda_stats["allocated_bytes.all.peak"] / 1024 ** 2,
                                    cuda_stats["reserved_bytes.all.peak"] / 1024 ** 2))

            # Save stats
            all_cuda_stats.append([float(batch.in_dict.points[0].shape[0]),
                                   training_loader.dataset.b_lim,
                                   cuda_stats["allocated_bytes.all.peak"] / 1024 ** 2,
                                   cuda_stats["reserved_bytes.all.peak"] / 1024 ** 2])

            # Increase batch limit
            training_loader.dataset.b_lim += blim_inc

            # Empty cache
            # torch.cuda.empty_cache()

            # Reset peak so that the peak reflect the maximum memory usage during a step
            torch.cuda.reset_peak_memory_stats(device)

            step += 1

    except RuntimeError as err:
        print("Caught a CUDA OOM Error:\n{0}".format(err))

    all_cuda_stats = np.array(all_cuda_stats, dtype=np.float32)
    
    return all_cuda_stats







