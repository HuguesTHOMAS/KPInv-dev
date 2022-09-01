
# ----------------------------------------------------------------------------------------------------------------------
#
#           Imports and global variables
#       \**********************************/
#


# Common libs
import time

import numpy as np
import torch

from sklearn.neighbors import KDTree

from utils.ply import read_ply, write_ply
from utils.cpp_funcs import grid_subsampling
from utils.batch_conversion import list_to_pack
from utils.gpu_subsampling import grid_subsample, subsample_pack_batch, init_gpu
from utils.gpu_neigbors import radius_search_pack_mode, radius_search_list_mode
from utils.cpp_funcs import batch_radius_neighbors, batch_knn_neighbors
from utils.printing import bcolors

from utils.cuda_funcs import furthest_point_sample, furthest_point_sample_3
from utils.cpp_funcs import furthest_point_sample_cpp

import matplotlib.pyplot as plt
import numpy as np


def load_S3DIS(crop_ratio=-1):

    ##############
    # Prepare Data
    ##############

    print()
    print('Data Preparation')
    print('****************')

    print('\nLoad ply')
    t1 = time.time()

    # Load a S3DIS point cloud
    file_path = '../Data/S3DIS/original_ply/Area_1.ply'
    data = read_ply(file_path)
    points = np.vstack((data['x'], data['y'], data['z'])).T
    colors = np.vstack((data['red'], data['green'], data['blue'])).T
    labels = data['class']

    t2 = time.time()
    print('Done in {:.3f}s'.format(t2 - t1))

    if 0 < crop_ratio < 1.0:

        print('\nCrop points')
        t1 = time.time()

        minp = np.min(points, axis=0)
        maxp = np.max(points, axis=0)
        midp = crop_ratio * maxp + (1 - crop_ratio) * minp

        mask = np.logical_and(points[:, 0] < midp[0], points[:, 1] < midp[1])
        points = points[mask]
        colors = colors[mask]
        labels = labels[mask]

        t2 = time.time()
        print('Done in {:.3f}s'.format(t2 - t1))

    return points, colors, labels


def test_gpu():

    # Init GPU
    init_gpu()

    # Get data
    points, colors, labels = load_S3DIS(crop_ratio=0.3)
    print(points.shape)

    # To torch
    print('\nInit')
    t1 = time.time()
    point_tensor = torch.from_numpy(np.copy(points))

    # To GPU
    device = torch.device("cuda")
    points_gpu = point_tensor.clone()
    points_gpu = points_gpu.to(device)

    print(points_gpu.device)
    t2 = time.time()
    print('Done in {:.3f}s'.format(t2 - t1))

    # Test
    N = 10

    print('\nNumpy CPU')
    t1 = time.time()
    test = points * (points + 1)
    for i in range(N):
        test += points * (points + i)
    t2 = time.time()
    print('Done in {:.3f}s'.format(t2 - t1))
    print('{:.1f}ms per ex'.format(1000 * (t2 - t1) / N))

    print('\nTorch CPU')
    t1 = time.time()
    test = point_tensor * (point_tensor + 1)
    for i in range(N):
        test += point_tensor * (point_tensor + i)
    t2 = time.time()
    print('Done in {:.3f}s'.format(t2 - t1))
    print('{:.1f}ms per ex'.format(1000 * (t2 - t1) / N))

    print('\nTorch GPU')
    t1 = time.time()
    test = points_gpu * (points_gpu + 1)
    for i in range(N):
        test += points_gpu * (points_gpu + i)
        print(test.device)
    t2 = time.time()
    print('Done in {:.3f}s'.format(t2 - t1))
    print('{:.1f}ms per ex'.format(1000 * (t2 - t1) / N))

    return


def test_grid_subsample():

    ############################
    # Initialize the environment
    ############################

    device = init_gpu()

    ##############
    # Prepare Data
    ##############

    # Get data
    points, colors, labels = load_S3DIS(crop_ratio=0.33)

    print('\nConvert points to GPU')
    t1 = time.time()

    # To torch
    point_tensor = torch.from_numpy(np.copy(points))

    print('device:', device)

    # To GPU
    point_gpu = point_tensor.clone()
    point_gpu = point_gpu.to(device)

    t2 = time.time()
    print('Done in {:.3f}s'.format(t2 - t1))

    ##################
    # Test subsampling
    ##################

    if points.shape[0] < 1:
        a = 1/0

    # Subsample cloud
    print('\nCpp Wrapper subsampling')
    t1 = time.time()
    sub_points1, sub_colors, sub_labels = grid_subsampling(points,
                                                           features=colors,
                                                           labels=labels,
                                                           sampleDl=0.04)

    t2 = time.time()
    print('Done in {:.3f}s'.format(t2 - t1))
    print(points.shape, '=>', sub_points1.shape)

    print('\nCPU Pytorch subsampling')
    t1 = time.time()
    sub_points2, _ = subsample_pack_batch(point_tensor, [point_tensor.shape[0]], 0.04)
    t2 = time.time()
    print('Done in {:.3f}s'.format(t2 - t1))
    print(points.shape, '=>', sub_points2.shape)

    print('\nGPU Pytorch subsampling')
    torch.cuda.synchronize()
    t1 = time.time()
    sub_points3, _ = subsample_pack_batch(point_gpu, [point_gpu.shape[0]], 0.04)
    torch.cuda.synchronize()
    t2 = time.time()
    print('Done in {:.3f}s'.format(t2 - t1))
    print(points.shape, '=>', sub_points3.shape)
    
    # # Save as ply
    # print('results/test1.ply')
    # write_ply('results/test1.ply',
    #           (sub_points1.astype(np.float32)),
    #           ['x', 'y', 'z'])

    # print('results/test2.ply')
    # write_ply('results/test2.ply',
    #           (sub_points2.numpy().astype(np.float32)),
    #           ['x', 'y', 'z'])

    # print('results/test3.ply')
    # write_ply('results/test3.ply',
    #           (sub_points3.detach().cpu().numpy().astype(np.float32)),
    #           ['x', 'y', 'z'])
              
    # print('\nGPU FPS')
    # torch.cuda.synchronize()
    # t1 = time.time()
    # sub_inds2 = furthest_point_sample(point_gpu, stride=4)
    # sub_points4 = point_gpu[sub_inds2, :]
    # torch.cuda.synchronize()
    # t2 = time.time()
    # print('Done in {:.3f}s'.format(t2 - t1))
    # print(points.shape, '=>', sub_points4.shape)

    # # Save as ply
    # print('results/test4.ply')
    # write_ply('results/test4.ply',
    #           (sub_points4.detach().cpu().numpy().astype(np.float32)),
    #           ['x', 'y', 'z'])

    # print('\nCPU FPS')
    # torch.cuda.synchronize()
    # t1 = time.time()
    # sub_inds2 = furthest_point_sample_cpp(points, stride=4, min_d=0.04)

    # sub_points5 = points[sub_inds2, :]
    # torch.cuda.synchronize()
    # t2 = time.time()
    # print('Done in {:.3f}s'.format(t2 - t1))
    # print(points.shape, '=>', sub_points5.shape)

    # # Get distance to nearest neighbors
    # neighbors, dists = batch_knn_neighbors(sub_points5,
    #                                        sub_points5,
    #                                        [sub_points5.shape[0]],
    #                                        [sub_points5.shape[0]],
    #                                        0,
    #                                        5,
    #                                        return_dist=True)

    # # Save as ply
    # print('results/test5.ply')
    # write_ply('results/test5.ply',
    #           (sub_points5.astype(np.float32), dists.astype(np.float32)),
    #           ['x', 'y', 'z', 'f1', 'f2', 'f3', 'f4', 'f5'])

    # a = 1/0

    #############################
    # Test subsampling of spheres
    #############################

    # Rescale float color and squeeze label
    sub_colors = sub_colors / 255
    sub_labels = np.squeeze(sub_labels)

    # Get chosen neighborhoods
    search_tree = KDTree(sub_points1, leaf_size=10)

    # Get an input sphere
    print('\nSpheres')

    in_R = 1.8
    newDl = 0.08

    t1 = time.time()
    all_inputs_points = []
    all_gpu_pts = []
    for i in range(0, sub_points1.shape[0] - 1, 10000):

        center_point = sub_points1[i:i + 1, :]
        input_inds = search_tree.query_radius(center_point, r=in_R)[0]
        all_inputs_points.append(sub_points1[input_inds])
        gpu_pts = torch.from_numpy(sub_points1[input_inds])
        all_gpu_pts.append(gpu_pts.to(device))

    t2 = time.time()
    print('Done in {:.3f}s'.format(t2 - t1))

    print(len(all_inputs_points))
    N = len(all_inputs_points)

    print('\nCpp Wrapper subsampling')
    t1 = time.time()
    for i, pts in enumerate(all_inputs_points):
        sub_pts1 = grid_subsampling(pts, sampleDl=newDl)
    t2 = time.time()
    print('Done in {:.3f}s'.format(t2 - t1))
    print('{:.3f}ms per sphere'.format(1000 * (t2 - t1) / N))

    print('\nCPU Pytorch subsampling')
    torch.cuda.synchronize()
    t1 = time.time()
    for i, pts in enumerate(all_inputs_points):
        cpu_pts = torch.from_numpy(pts)
        sub_pts1 = grid_subsample(cpu_pts, newDl)
    torch.cuda.synchronize()
    t2 = time.time()
    print('Done in {:.3f}s'.format(t2 - t1))
    print('{:.3f}ms per sphere'.format(1000 * (t2 - t1) / N))

    print('\nGPU Pytorch subsampling')
    torch.cuda.synchronize()
    t1 = time.time()
    all_N = []
    for i, gpu_pts in enumerate(all_gpu_pts):
        sub_pts1 = grid_subsample(gpu_pts, newDl)
        all_N.append(int(sub_pts1.shape[0]))
    torch.cuda.synchronize()
    t2 = time.time()
    print('Done in {:.3f}s'.format(t2 - t1))
    print('{:.3f}ms per sphere'.format(1000 * (t2 - t1) / N))

    print('\nGPU Pytorch subsampling FPS')
    torch.cuda.synchronize()
    t1 = time.time()
    all_N2 = []
    for i, gpu_pts in enumerate(all_gpu_pts):
        sub_inds2 = furthest_point_sample(gpu_pts, stride=2, min_d=newDl*0.67)
        # sub_inds2 = furthest_point_sample(gpu_pts, stride=2)
        sub_points2 = gpu_pts[sub_inds2, :]
        all_N2.append(int(sub_points2.shape[0]))
    torch.cuda.synchronize()
    t2 = time.time()
    print('Done in {:.3f}s'.format(t2 - t1))
    print('{:.3f}ms per sphere'.format(1000 * (t2 - t1) / N))

    print('\nCPU wrapper subsampling FPS')
    torch.cuda.synchronize()
    all_N3 = []
    t1 = time.time()
    for i, pts in enumerate(all_inputs_points):
        cpu_pts = torch.from_numpy(pts)
        sub_inds3 = furthest_point_sample_cpp(cpu_pts, stride=1, min_d=newDl*0.67)
        sub_points3 = cpu_pts[sub_inds3, :]
        all_N3.append(int(sub_points3.shape[0]))
    torch.cuda.synchronize()
    t2 = time.time()
    print('Done in {:.3f}s'.format(t2 - t1))
    print('{:.3f}ms per sphere'.format(1000 * (t2 - t1) / N))

    # print('\nGPU Pytorch subsampling FPS Naive')
    # torch.cuda.synchronize()
    # t1 = time.time()
    # all_N2 = []
    # for i, gpu_pts in enumerate(all_gpu_pts):
    #     sub_inds2 = furthest_point_sample_3(gpu_pts, stride=4)
    #     sub_points2 = gpu_pts[sub_inds2, :]
    #     all_N2.append(int(sub_points2.shape[0]))
    # torch.cuda.synchronize()
    # t2 = time.time()
    # print('Done in {:.3f}s'.format(t2 - t1))
    # print('{:.3f}ms per sphere'.format(1000 * (t2 - t1) / N))
    
    # print('\nCPU Pytorch subsampling FPS Naive')
    # torch.cuda.synchronize()
    # t1 = time.time()
    # for i, pts in enumerate(all_inputs_points):
    #     cpu_pts = torch.from_numpy(pts)
    #     sub_inds2 = furthest_point_sample_3(cpu_pts, stride=4)
    #     sub_points2 = cpu_pts[sub_inds2, :]
    #     all_N2.append(int(sub_points2.shape[0]))
    # torch.cuda.synchronize()
    # t2 = time.time()
    # print('Done in {:.3f}s'.format(t2 - t1))
    # print('{:.3f}ms per sphere'.format(1000 * (t2 - t1) / N))


    print(input_inds.shape)
    print('------------ OK ------------')
    
    print('grid: ', np.mean([int(gpu_pts.shape[0]) for gpu_pts in all_gpu_pts]), np.mean(all_N))
    print('fps: ', np.mean([int(gpu_pts.shape[0]) for gpu_pts in all_gpu_pts]), np.mean(all_N2))
    print('fps(min_d): ', np.mean([int(gpu_pts.shape[0]) for gpu_pts in all_gpu_pts]), np.mean(all_N3))

    return


def test_neighbors():

    ############################
    # Initialize the environment
    ############################

    init_gpu()

    ##############
    # Prepare Data
    ##############

    # Get data
    points, colors, labels = load_S3DIS(crop_ratio=0.33)

    print('\nConvert points to GPU')
    torch.cuda.synchronize()
    t1 = time.time()

    # To torch
    point_tensor = torch.from_numpy(np.copy(points))

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print('device:', device)

    # To GPU
    point_gpu = point_tensor.clone()
    point_gpu = point_gpu.to(device)

    torch.cuda.synchronize()
    t2 = time.time()
    print('Done in {:.3f}s'.format(t2 - t1))

    ##################
    # Test subsampling
    ##################

    dl0 = 0.03

    # Subsample cloud
    print('\nGPU Pytorch subsampling')
    torch.cuda.synchronize()
    t1 = time.time()
    sub_points0, sub_lengths1 = subsample_pack_batch(point_gpu, [point_gpu.shape[0]], dl0, method='grid')
    torch.cuda.synchronize()
    t2 = time.time()
    print('Done in {:.3f}s'.format(t2 - t1))
    print(points.shape, '=>', sub_points0.shape)


    ###########################
    # Get subsampled of spheres
    ###########################

    # Get chosen neighborhoods
    sub_points1 = sub_points0.cpu().numpy()
    search_tree = KDTree(sub_points1, leaf_size=10)

    # Get an input sphere
    print('\nSpheres in batches')

    in_R = 1.0
    batch_num = 4

    torch.cuda.synchronize()
    t1 = time.time()
    all_inputs_points = []
    all_gpu_pts = []
    all_N = []
    for i in range(0, sub_points1.shape[0] - 1, 2000):

        center_point = sub_points1[i:i + 1, :]
        input_inds = search_tree.query_radius(center_point, r=in_R)[0]
        all_inputs_points.append(sub_points1[input_inds])
        gpu_pts = torch.from_numpy(sub_points1[input_inds])
        all_gpu_pts.append(gpu_pts.to(device))
        all_N.append(int(gpu_pts.shape[0]))

    # Get batch limit for varaible size of batch
    mean_N = np.mean(all_N)
    batch_limit = mean_N * batch_num

    print()
    print('-------------------------')
    print('length mean / std', np.mean(all_N), np.std(all_N))
    print('-------------------------')
    print()

    all_cpu_batches = []
    all_gpu_batches = []
    batch_n = 0
    current_batch = []
    current_batch_cpu = []
    for i, gpu_pts in enumerate(all_gpu_pts):
        if len(current_batch) > 0 and batch_n + int(gpu_pts.shape[0]) > batch_limit:
            all_gpu_batches.append(current_batch)
            all_cpu_batches.append(current_batch_cpu)
            batch_n = 0
            current_batch = []
            current_batch_cpu = []
        current_batch.append(gpu_pts)
        current_batch_cpu.append(gpu_pts.cpu())
        batch_n += int(gpu_pts.shape[0])

    all_batch_n = []
    all_batch_L = []
    for i, gpu_batch in enumerate(all_gpu_batches):
        all_batch_n.append(len(gpu_batch))
        all_batch_L.append(np.sum([int(gpu_pts.shape[0]) for gpu_pts in gpu_batch]))

    print()
    print('-------------------------')
    print('batch_num', np.mean(all_batch_n))
    print('-------------------------')
    print('length mean / std', np.mean(all_batch_L), np.std(all_batch_L))
    print('-------------------------')
    print('batch limit', np.max(all_batch_L), batch_limit)
    print('-------------------------')
    print()


    torch.cuda.synchronize()
    t2 = time.time()
    print('Done in {:.3f}s'.format(t2 - t1))

    print(len(all_inputs_points))
    N = len(all_inputs_points)


    ################
    # Test neighbors
    ################

    conv_r = dl0 * 2.5
    neighbor_limit = 15

    
    print()
    print()

    print('\nCPU cpp_radius')
    torch.cuda.synchronize()
    t1 = time.time()
    all_neighborscpp1 = []
    for i, cpu_batch in enumerate(all_cpu_batches):
        pack_tensor, lengths = list_to_pack(cpu_batch)
        conv_i = batch_radius_neighbors(pack_tensor, pack_tensor, lengths, lengths, conv_r, neighbor_limit)
        all_neighborscpp1.append(conv_i)
    torch.cuda.synchronize()
    t2 = time.time()
    print('Done in {:.3f}s'.format(t2 - t1))
    print('{:.3f}ms per batch'.format(1000 * (t2 - t1) / N))

    print('\nCPU cpp_knn')
    torch.cuda.synchronize()
    t1 = time.time()
    all_neighborscpp2 = []
    all_d_cpp = []
    for i, cpu_batch in enumerate(all_cpu_batches):
        pack_tensor, lengths = list_to_pack(cpu_batch)
        conv_i, dists = batch_knn_neighbors(pack_tensor, pack_tensor, lengths, lengths, conv_r, neighbor_limit, return_dist=True)
        all_neighborscpp2.append(conv_i)
        all_d_cpp.append(dists)
    torch.cuda.synchronize()
    t2 = time.time()
    print('Done in {:.3f}s'.format(t2 - t1))
    print('{:.3f}ms per batch'.format(1000 * (t2 - t1) / N))


    print('\nGPU Pytorch neighbors')
    torch.cuda.synchronize()
    t1 = time.time()
    all_neighbors1 = []
    for i, gpu_batch in enumerate(all_gpu_batches):
        pack_tensor, lengths = list_to_pack(gpu_batch)
        conv_i = radius_search_list_mode(pack_tensor, pack_tensor, lengths, lengths, conv_r, neighbor_limit, shadow=False)
        all_neighbors1.append(conv_i)
    torch.cuda.synchronize()
    t2 = time.time()
    print('Done in {:.3f}s'.format(t2 - t1))
    print('{:.3f}ms per batch'.format(1000 * (t2 - t1) / N))

    print('\nGPU Pytorch neighbors 2')
    torch.cuda.synchronize()
    t1 = time.time()
    all_neighbors2 = []
    all_d_gpu = []
    for i, gpu_batch in enumerate(all_gpu_batches):
        pack_tensor, lengths = list_to_pack(gpu_batch)
        conv_i, dists = radius_search_pack_mode(pack_tensor, pack_tensor, lengths, lengths, conv_r, neighbor_limit, shadow=False, return_dist=True)
        all_neighbors2.append(conv_i)
        all_d_gpu.append(dists)
    torch.cuda.synchronize()
    t2 = time.time()
    print('Done in {:.3f}s'.format(t2 - t1))
    print('{:.3f}ms per batch'.format(1000 * (t2 - t1) / N))
    
    print('\nPytorch neighbors cannot work on CPU as it uses keops')

    print()
    print()

    print('Verify that the two torch neighbors implem returns the same result')

    all_max = []
    all_median = []
    for d_cpp, d_gpu in zip(all_d_cpp, all_d_gpu):
        all_max.append(torch.max(torch.abs(d_cpp - d_gpu.cpu())).item())
        all_median.append(torch.median(torch.abs(d_cpp - d_gpu.cpu())).item())

    print(np.max(all_max))
    print(np.max(all_median))


    print()
    print()

    for i in range(10):
        print('{:6.6f} {:6.6f} {:6.6f}     {:6.6f} {:6.6f} {:6.6f}'.format(d_cpp[i, 0].item(),
                                                                           d_cpp[i, 1].item(),
                                                                           d_cpp[i, 2].item(),
                                                                           d_gpu[i, 0].item(),
                                                                           d_gpu[i, 1].item(),
                                                                           d_gpu[i, 2].item()))

    print()
    print()

    print('Verify distances')

    for i, neighb2 in enumerate(all_neighbors2):
        all_good.append(torch.all(all_neighbors1[i] - neighb2).item())


    print()
    print()

    print('------------ OK ------------')

    print()
    print()

    print('Verify that the cpp knn returns the same as torch knn (ordered neighbors)')

    # Verify that cpp neighbors are ordered as well
    for i, cpu_batch in enumerate(all_cpu_batches):
        s = ''
        for test1 in [all_neighborscpp1[i], all_neighborscpp2[i], all_neighbors1[i], all_neighbors2[i]]:
            for test2 in [all_neighborscpp1[i], all_neighborscpp2[i], all_neighbors1[i], all_neighbors2[i]]:
                test_bool = torch.all(test1.cpu() == test2.cpu()).item()
                if test_bool:
                    s += ' {:}{:s}{:}'.format(bcolors.OKBLUE, u'\u2713', bcolors.ENDC)
                else:
                    s += ' {:}{:s}{:}'.format(bcolors.FAIL, u'\u2718', bcolors.ENDC)
            s += '\n'
        print('-----------------')
        print(s)



        if not (torch.all(all_neighborscpp2[i].cpu() == all_neighbors1[i].cpu()).item()):

            # get lines that are differents
            cpp_neighs = all_neighborscpp2[i].cpu().numpy()
            torch_neighs = all_neighbors1[i].cpu().numpy()
            mask = cpp_neighs == torch_neighs  # (N, K)

            bad_lines = np.where(np.logical_not(np.all(mask, axis=1)))[0]

            for l in bad_lines:

                s = 'L={:6d} : '.format(l)
                for k in range(neighbor_limit):
                    if mask[l, k]:
                        s += ' {:6d}'.format(cpp_neighs[l, k])
                    else:
                        s += ' {:}{:6d}{:}'.format(bcolors.FAIL, cpp_neighs[l, k], bcolors.ENDC)
                s += '\n'
                s += '           '
                
                for k in range(neighbor_limit):
                    if mask[l, k]:
                        s += '       '
                    else:
                        s += ' {:}{:6d}{:}'.format(bcolors.OKBLUE, torch_neighs[l, k], bcolors.ENDC)
                s += '\n'
                
                print(s)

            break


    print('-----------------')

    print()
    print()

    print('There are only some intervertion due to float approximation. SO OK!')

    print()
    print()
    

    all_n_valid = []
    all_trucated = []
    for i, neighb1 in enumerate(all_neighborscpp1):

        shadow_mask = neighb1 == int(neighb1.shape[0])
        n_shadows = torch.sum(shadow_mask.type(torch.long), dim=1)
        n_valid = neighbor_limit - n_shadows
        all_n_valid.append(n_valid)
        all_trucated.append(n_shadows == 0)


    all_trucated = torch.cat(all_trucated, dim=0).cpu().numpy()

    n_trunc = np.sum(all_trucated.astype(np.int32))
    print('Trunc: {:d}/{:d}  = {:.1f}%'.format(n_trunc, all_trucated.shape[0], 100 * n_trunc / all_trucated.shape[0]))

    all_n_valid = torch.cat(all_n_valid, dim=0).cpu().numpy()

    # plt.hist(x, density=False, bins=30)
    plt.hist(all_n_valid, density=False, bins=np.arange(neighbor_limit + 1, dtype=np.float32) - 0.1)
    plt.ylabel('Counts')
    plt.xlabel('neighbors')

    plt.show()
    
    print('------------ OK ------------')

    return


# ----------------------------------------------------------------------------------------------------------------------
#
#           Main Call
#       \***************/
#


if __name__ == '__main__':

    test_grid_subsample()

    # test_neighbors()

    # Conclusion:
    # Use grid subsampling list for fastest results
    # Use neighbors pack mode for fastest results

    # We can create a calibration script that give the recommended values for parameters
    # But we can let the user decide the parameters itself



    a = 0
