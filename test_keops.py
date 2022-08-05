
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

    init_gpu()

    ##############
    # Prepare Data
    ##############

    # Get data
    points, colors, labels = load_S3DIS(crop_ratio=0.33)

    print('\nConvert points to GPU')
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

    t2 = time.time()
    print('Done in {:.3f}s'.format(t2 - t1))

    ##################
    # Test subsampling
    ##################

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

    sub_points2, sub_lengths2 = subsample_pack_batch(point_tensor, [point_tensor.shape[0]], 0.04)

    t2 = time.time()
    print('Done in {:.3f}s'.format(t2 - t1))
    print(points.shape, '=>', sub_points2.shape)

    print('\nGPU Pytorch subsampling')
    t1 = time.time()

    sub_points3, sub_lengths3 = subsample_pack_batch(point_gpu, [point_gpu.shape[0]], 0.04)

    t2 = time.time()
    print('Done in {:.3f}s'.format(t2 - t1))
    print(points.shape, '=>', sub_points3.shape)


    # Save as ply
    write_ply('results/test1.ply',
              (sub_points1.astype(np.float32)),
              ['x', 'y', 'z'])
    write_ply('results/test2.ply',
              (sub_points2.numpy().astype(np.float32)),
              ['x', 'y', 'z'])
    write_ply('results/test3.ply',
              (sub_points3.cpu().numpy().astype(np.float32)),
              ['x', 'y', 'z'])

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

    in_R = 5.0
    newDl = 0.08

    t1 = time.time()
    all_inputs_points = []
    all_gpu_pts = []
    for i in range(0, sub_points1.shape[0] - 1, 1000):

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
    t1 = time.time()
    for i, pts in enumerate(all_inputs_points):
        cpu_pts = torch.from_numpy(pts)
        sub_pts1 = grid_subsample(cpu_pts, newDl)
    t2 = time.time()
    print('Done in {:.3f}s'.format(t2 - t1))
    print('{:.3f}s per sphere'.format(1000 * (t2 - t1) / N))

    print('\nGPU Pytorch subsampling')
    t1 = time.time()
    for i, gpu_pts in enumerate(all_gpu_pts):
        sub_pts1 = grid_subsample(gpu_pts, newDl)
    t2 = time.time()
    print('Done in {:.3f}s'.format(t2 - t1))
    print('{:.3f}ms per sphere'.format(1000 * (t2 - t1) / N))

    print(input_inds.shape)
    print('------------ OK ------------')

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

    in_R = 2.0
    batch_num = 5

    torch.cuda.synchronize()
    t1 = time.time()
    all_inputs_points = []
    all_gpu_pts = []
    all_N = []
    for i in range(0, sub_points1.shape[0] - 1, 1000):

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
    neighbor_limit = 30

    
    print()
    print()


    print('\nGPU Pytorch neighbors')
    torch.cuda.synchronize()
    t1 = time.time()
    all_neighbors1 = []
    for i, gpu_batch in enumerate(all_gpu_batches):
        pack_tensor, lengths = list_to_pack(gpu_batch)
        conv_i = radius_search_list_mode(pack_tensor, pack_tensor, lengths, lengths, conv_r, neighbor_limit, shadow=True)
        all_neighbors1.append(conv_i)
    torch.cuda.synchronize()
    t2 = time.time()
    print('Done in {:.3f}s'.format(t2 - t1))
    print('{:.3f}ms per batch'.format(1000 * (t2 - t1) / N))

    print('\nGPU Pytorch neighbors 2')
    torch.cuda.synchronize()
    t1 = time.time()
    all_neighbors2 = []
    for i, gpu_batch in enumerate(all_gpu_batches):
        pack_tensor, lengths = list_to_pack(gpu_batch)
        conv_i = radius_search_pack_mode(pack_tensor, pack_tensor, lengths, lengths, conv_r, neighbor_limit)
        all_neighbors2.append(conv_i)
    torch.cuda.synchronize()
    t2 = time.time()
    print('Done in {:.3f}s'.format(t2 - t1))
    print('{:.3f}ms per batch'.format(1000 * (t2 - t1) / N))
    
    print('\nCPU Pytorch neighbors 2')
    torch.cuda.synchronize()
    t1 = time.time()
    all_neighbors3 = []
    for i, gpu_batch in enumerate(all_cpu_batches):
        pack_tensor, lengths = list_to_pack(gpu_batch)
        conv_i = radius_search_pack_mode(pack_tensor, pack_tensor, lengths, lengths, conv_r, neighbor_limit)
        all_neighbors3.append(conv_i)
    torch.cuda.synchronize()
    t2 = time.time()
    print('Done in {:.3f}s'.format(t2 - t1))
    print('{:.3f}ms per batch'.format(1000 * (t2 - t1) / N))

    print()
    print()

    all_good = []
    for i, neighb2 in enumerate(all_neighbors2):
        all_good.append(torch.all(all_neighbors1[i] == neighb2).item())
    print(np.all(all_good))

    print()
    print()

    print('------------ OK ------------')

    print()
    print()

    all_n_valid = []
    all_trucated = []
    for i, neighb1 in enumerate(all_neighbors1):

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

    # test_grid_subsample()

    test_neighbors()

    # Conclusion:
    # Use grid subsampling list for fastest results
    # Use neighbors pack mode for fastest results

    # We can create a calibration script that give the recommended values for parameters
    # But we can let the user decide the parameters itself



    a = 0
