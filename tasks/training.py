
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


def training_epoch(epoch, t0, net, optimizer, training_loader, cfg, PID_file, device, cuda_debug=False):

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
        t += [time.time()]

        if 'cuda' in device.type:
            batch.to(device)

        t += [time.time()]

        # zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = net(batch)

        t += [time.time()]

        # Loss with equivar/invar
        loss = net.loss(outputs, batch)
        acc = net.accuracy(outputs, batch)

        t += [time.time()]

        # Backward gradients
        loss.backward()

        # Clip gradient
        if cfg.train.grad_clip > 0:
            #torch.nn.utils.clip_grad_norm_(net.parameters(), cfg.train.grad_clip)
            torch.nn.utils.clip_grad_value_(net.parameters(), cfg.train.grad_clip)

        # Optimizer step
        optimizer.step()
        torch.cuda.synchronize(device)

        t += [time.time()]

        # Average timing
        if step < 2:
            mean_dt = np.array(t[1:]) - np.array(t[:-1])
        else:
            mean_dt = 0.9 * mean_dt + 0.1 * (np.array(t[1:]) - np.array(t[:-1]))

        # CUDA debug
        torch.cuda.ipc_collect()
        if cuda_debug:
            fmt_str = ' ' * 10 + '| {:8.0f}Kpts | {:6.0f}MB | {:6.0f}MB | {:6.0f}MB | {:6.0f}MB |'
            print(fmt_str.format(batch.points[0].shape[0] / 1000,
                                    torch.cuda.memory_allocated() / 1024 ** 2,
                                    torch.cuda.max_memory_allocated() / 1024 ** 2,
                                    torch.cuda.memory_reserved() / 1024 ** 2,
                                    torch.cuda.max_memory_reserved() / 1024 ** 2))
            torch.cuda.reset_max_memory_allocated()

        # Console display (only one per second)
        if (t[-1] - last_display) > 1.0:
            last_display = t[-1]
            message = 'e{:03d}-i{:04d} => L={:.3f} acc={:3.0f}% / t(ms): {:5.1f} {:5.1f} {:5.1f} {:5.1f} {:5.1f})'
            print(message.format(epoch, step,
                                    loss.item(),
                                    100*acc,
                                    1000 * mean_dt[0],
                                    1000 * mean_dt[1],
                                    1000 * mean_dt[2],
                                    1000 * mean_dt[3],
                                    1000 * mean_dt[4]))

        # Log file
        if cfg.exp.saving:
            with open(join(cfg.exp.log_dir, 'training.txt'), "a") as file:
                message = '{:d} {:d} {:.3f} {:.3f} {:.3f} {:.3f}\n'
                file.write(message.format(epoch,
                                            step,
                                            net.output_loss,
                                            net.reg_loss,
                                            acc,
                                            t[-1] - t0))


        step += 1

    return