
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

from utils.printing import underline

import subprocess


def get_temperatures_linux():

    # GPU data:

    cmd = 'nvidia-smi dmon -c 1'
    cmd = cmd.split()
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    o, e = proc.communicate()
    smi_lines = o.decode('ascii').split('\n')

    metrics = smi_lines[0].split()[1:]
    units   = smi_lines[1].split()[1:]
    measures = [smi_line.split() for smi_line in smi_lines[2:]]
    measures = [meas for meas in measures if len(meas) > 1]

    s1 = ''
    for i in range(len(measures)):
        for m in metrics:
            s1 += ' {:^6s}'.format(m)

    s2 = ''
    for i in range(len(measures)):
        for u in units:
            s2 += ' {:^6s}'.format(u)

    s3 = ''
    for meas in measures:
        for m in meas:
            s3 += ' {:^6s}'.format(m)
            
    # CPU data:
    cmd = 'sensors'
    cmd = cmd.split()
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    o, e = proc.communicate()


    sensor_lines = o.decode('utf8', 'ignore').split('\n')


    metrics = ['SYSTIN', 'CPUTIN', 'AUXTIN', 'Core']
    data = [[] for _ in metrics]

    for line in sensor_lines:

        for d, m in zip(data, metrics):

            if '°C' in line and line.startswith(m):
                if m == 'Core':
                    d.append(float(line.split()[2][:-2]))
                else:
                    d.append(float(line.split()[1][:-2]))

    cpu1 = ''
    for m in metrics:
        cpu1 += ' {:^8s}'.format(m)

    cpu2 = ''
    for m in metrics:
        cpu2 += ' {:^8s}'.format('C')

    cpu3 = ''
    for d in data:
        cpu3 += ' {:^8.1f}'.format(np.max(d))


    return cpu1+s1, cpu2+s2, cpu3+s3 



# ----------------------------------------------------------------------------------------------------------------------
#
#           Training Function
#       \***********************/
#


def training_epoch(epoch, t0, net, optimizer, training_loader, cfg, PID_file, device):

    run_batch_size = 0
    mini_step = 0
    accum_loss = 0
    step = 0
    last_display = time.time()
    t = [time.time()]
    finished = True
    optimizer.zero_grad()

    underline('Training epoch {:d}'.format(epoch))
    message =  '\n                                                                 Timings            '
    message += '\nEpoch Step |   Loss   | GPU usage |      Speed      |   In   Batch  Forw  Back  End '
    message += '\n-----------|----------|-----------|-----------------|-------------------------------'

    print(message)

    for batch in training_loader:

        # Check kill signal (running_PID.txt deleted)
        if cfg.exp.saving and not exists(PID_file):
            raise ValueError('A user deleted the running_PID.txt file. Experiment is stopped.')

        ##################
        # Processing batch
        ##################
        

        try:

            # New time at first accumulation step
            if mini_step % cfg.train.accum_batch == 0:
                t = t[-1:]


            if 'cuda' in device.type:
                torch.cuda.synchronize(device)
            t += [time.time()]

            # Move batch to GPU
            if 'cuda' in device.type:
                batch.to(device)

            # Update effective batch size
            mean_f = max(0.02, 1.0 / (step + 1))
            run_batch_size *= 1 - mean_f
            run_batch_size += mean_f * len(batch.in_dict.lengths0)


            if 'cuda' in device.type:
                torch.cuda.synchronize(device)
            t += [time.time()]

            # Forward pass
            outputs = net(batch)
            
            # Loss 
            loss = net.loss(outputs, batch.in_dict.labels)
            #acc = net.accuracy(outputs, batch.in_dict.labels)

            # Normalize loss
            loss = loss / cfg.train.accum_batch
            accum_loss += loss.item()

            if 'cuda' in device.type:
                torch.cuda.synchronize(device)
            t += [time.time()]

            # Backward gradients
            loss.backward()

            if 'cuda' in device.type:
                torch.cuda.synchronize(device)
            t += [time.time()]

            # Only perform an optimization step when we have accumulated enough gradients
            if (mini_step + 1) % cfg.train.accum_batch == 0:

                # Clip gradient
                if cfg.train.grad_clip > 0:
                    #torch.nn.utils.clip_grad_norm_(net.parameters(), cfg.train.grad_clip)
                    torch.nn.utils.clip_grad_value_(net.parameters(), cfg.train.grad_clip)

                # Optimizer step
                optimizer.step()
                
                # zero the parameter gradients
                optimizer.zero_grad()

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

                # Acumulate timings from the accumulation steps
                dt = np.array(t[1:]) - np.array(t[:-1])
                accum_dt = np.reshape(dt[:-1], (cfg.train.accum_batch, -1))
                accum_dt = np.sum(accum_dt, axis=0)
                accum_dt = np.append(accum_dt, dt[-1])

                # Average timing
                if step < 5:
                    mean_dt = accum_dt
                else:
                    mean_dt = 0.8 * mean_dt + 0.2 * accum_dt

                # Console display (only one per second)
                if (t[-1] - last_display) > 1.0:
                    last_display = t[-1]

                    message = '{:5d} {:4d} | {:8.3f} | {:7.1f} % | {:7.1f} ins/sec | {:6.1f} {:5.1f} {:5.1f} {:5.1f} {:5.1f}'
                    print(message.format(epoch, step,
                                            accum_loss,
                                            gpu_usage,
                                            cfg.train.accum_batch * run_batch_size / np.sum(mean_dt),
                                            1000 * mean_dt[0],
                                            1000 * mean_dt[1],
                                            1000 * mean_dt[2],
                                            1000 * mean_dt[3],
                                            1000 * mean_dt[4]))

                # Log file
                if cfg.exp.saving:
                    with open(join(cfg.exp.log_dir, 'training.txt'), "a") as file:
                        message = '{:d} {:d} {:.5f} {:.5f} {:.3f}\n'
                        file.write(message.format(epoch,
                                                    step,
                                                    accum_loss,
                                                    net.deform_loss,
                                                    t[-1] - t0))
                    
                    # s1, s2, s3  = get_temperatures_linux()
                    # temp_f = join(cfg.exp.log_dir, 'temperatures.txt')
                    # if exists(temp_f):
                    #     with open(temp_f, "a") as file:
                    #         file.write(s3 + '\n')
                    # else:
                    #     with open(temp_f, "a") as file:
                    #         file.write(s1 + '\n')
                    #         file.write(s2 + '\n')
                    #         file.write(s3 + '\n')

                accum_loss = 0
                step += 1

            mini_step += 1
                
        except RuntimeError as err:
            if 'CUDA out of memory' in str(err):
                print("Caught a CUDA OOM Error:\n{0}".format(err))
                print("Reduce batch limit by 10% and restart epoch")
                training_loader.dataset.b_lim -= int(training_loader.dataset.b_lim * 0.1)
                for p in net.parameters():
                    if p.grad is not None:
                        del p.grad  # free some memory
                torch.cuda.empty_cache()
                finished = False
                break
            else:
                raise err

    return finished


def training_epoch_debug(epoch, net, optimizer, training_loader, cfg, PID_file, device, blim_inc=1000):

    # Variables
    step = 0
    t = [time.time()]
    all_cuda_stats = []

    try:

        for batch in training_loader:
                
            # Check kill signal (running_PID.txt deleted)
            if cfg.exp.saving and not exists(PID_file):
                raise ValueError('A user deleted the running_PID.txt file. Experiment is stopped.')

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
            torch.cuda.empty_cache()

            # Reset peak so that the peak reflect the maximum memory usage during a step
            torch.cuda.reset_peak_memory_stats(device)

            step += 1

    except RuntimeError as err:
        print("Caught a CUDA OOM Error:\n{0}".format(err))

    all_cuda_stats = np.array(all_cuda_stats, dtype=np.float32)
    
    return all_cuda_stats







