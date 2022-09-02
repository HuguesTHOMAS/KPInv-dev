#
#
#      0=================================0
#      |    Kernel Point Convolutions    |
#      0=================================0
#
#
# ----------------------------------------------------------------------------------------------------------------------
#
#      Functions handling the disposition of kernel points.
#
# ----------------------------------------------------------------------------------------------------------------------
#
#      Hugues THOMAS - 11/06/2018
#


# ------------------------------------------------------------------------------------------
#
#          Imports and global variables
#      \**********************************/
#


# Import numpy package and name it "np"
import time
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib import cm
from os import makedirs
from os.path import join, exists

from utils.ply import read_ply, write_ply
from utils.printing import bcolors
from utils.rotation import create_3D_rotations
from utils.gpu_init import init_gpu


# ------------------------------------------------------------------------------------------
#
#           Functions
#       \***************/
#
#


def get_identity_lrfs(shape=None, dtype=np.float32):
    """
    Creates identity rotations in the shape asked
    :param shape: a list/tuple of the wanted shape: (d1, ..., dn). If None, returns a single rotation
    :param dtype: Type of the returned array
    :return: AS many rotations as asked, shaped like asked: (d1, ..., dn, 3, 3)
    """

    if shape is None:
        return np.eye(3, dtype=dtype)
    else:
        shape_tuple = tuple(shape) + (3, 3)
        n_rot = np.prod(shape)
        R = np.expand_dims(np.eye(3, dtype=dtype), 0)
        R = np.tile(R, (n_rot, 1, 1))
        return R.reshape(shape_tuple)

@torch.no_grad()
def spherical_Lloyd_gpu(radius, num_cells, num_kernels=1, dimension=3, fixed='center', approximation='monte-carlo',
                        approx_n=50000, max_iter=500, momentum=0.9, verbose=0):
    """
    Creation of kernel point via Lloyd algorithm. We use an approximation of the algorithm, and compute the Voronoi
    cell centers with discretization  of space. The exact formula is not trivial with part of the sphere as sides.
    :param radius: Radius of the kernels
    :param num_cells: Number of cell (kernel points) in the Voronoi diagram.
    :param dimension: dimension of the space
    :param fixed: fix position of certain kernel points ('none', 'center' or 'verticals')
    :param approximation: Approximation method for Lloyd's algorithm ('discretization', 'monte-carlo')
    :param approx_n: Number of point used for approximation.
    :param max_iter: Maximum nu;ber of iteration for the algorithm.
    :param momentum: Momentum of the low pass filter smoothing kernel point positions
    :param verbose: display option
    :return: points [num_kernels, num_points, dimension]
    """


    #######################
    # Parameters definition
    #######################

    # Radius used for optimization (points are rescaled afterwards)
    radius0 = 1.0
    device = init_gpu()

    #######################
    # Kernel initialization
    #######################

    # Random kernel points (Uniform distribution in a sphere)
    gen_radius = radius0
    k_pts = torch.zeros((0, dimension), device=device)
    tot_k = num_kernels * num_cells
    while k_pts.shape[0] < tot_k:
        new_points = torch.rand(tot_k , dimension, device=device) * 2 * gen_radius - gen_radius
        k_pts = torch.cat((k_pts, new_points))
        d2 = torch.sum(torch.pow(k_pts, 2), axis=1)
        k_pts = k_pts[torch.logical_and(d2 < gen_radius ** 2, (0.1 * gen_radius) ** 2 < d2), :]
    k_pts = k_pts[:tot_k, :].reshape((num_kernels, num_cells, -1))

    # Optional fixing
    if fixed == 'center':
        k_pts[:, 0, :] *= 0

    # Initialize figure
    if verbose > 1:
        fig = plt.figure()

    # Verify we use monte carlo
    if approximation != 'monte-carlo':
        raise ValueError('Wrong approximation method chosen: "{:s}"'.format(approximation))

    #####################
    # Kernel optimization
    #####################

    # Warning if at least one kernel point has no cell
    warning = False

    # moving vectors of kernel points saved to detect convergence
    avg_volumes = k_pts.new_zeros((num_kernels, num_cells,))
    all_volumes = []
    max_moves = []

    for iter in range(max_iter):

        # In the case of monte-carlo, renew the sampled points
        X = torch.rand(approx_n, dimension, device=device) * 2 * radius0 - radius0
        d2 = torch.sum(torch.pow(X, 2), dim=1)
        X = X[d2 < radius0 * radius0, :]

        # Get the distances matrix [n_approx, nk, K, dim] -> [n_approx, nk, K]
        differences = X.unsqueeze(1).unsqueeze(2) - k_pts
        sq_distances = torch.sum(torch.square(differences), dim=-1)
        
        # Compute cell centers [n_approx, nk]
        cell_inds = torch.argmin(sq_distances, dim=-1)

        # Get unique values and indices (M,) (N, nk) (M,), (M < K)
        uniq_v, inv_i, uniq_c = torch.unique(cell_inds,
                                            return_inverse=True,
                                            return_counts=True)

        # Get num point per cell
        ones_tmp = k_pts.new_ones((approx_n, num_kernels))  # (N, nk)
        c_n = k_pts.new_zeros((uniq_v.shape[0], num_kernels))  # (M, nk)
        c_n.scatter_add_(0, inv_i, ones_tmp)  # (M, nk)

        # Get avg point per cell
        inv_i = inv_i.unsqueeze(2).expand(-1, -1, dimension)  # (N, nk, D)
        X_tmp = X.unsqueeze(1).expand(-1, num_kernels, -1)  # (N, nk, D)
        c_pts = k_pts.new_zeros((uniq_v.shape[0], num_kernels, dimension))  # (M, nk, D)
        c_pts.scatter_add_(0, inv_i, X_tmp)  # (M, nk, D)
        c_pts /= c_n.unsqueeze(2).float()  # (M, nk, D)

        # Handle case where cell is empty
        c_pts = c_pts.transpose(0, 1)
        if c_pts.shape[1] < num_cells:
            centers = k_pts.clone()
            volumes = k_pts.new_zeros((num_kernels, num_cells,))
            centers[:, uniq_v, :] = c_pts
            volumes[:, uniq_v] = c_n.transpose(0, 1)
        else:
            centers = c_pts
            volumes = c_n.transpose(0, 1)

        # Update kernel points with low pass filter to smooth mote carlo
        moves = (1 - momentum) * (centers - k_pts)
        k_pts += moves

        # Compute volumes
        volumes = volumes.type(torch.float32)
        avg_volumes = momentum * avg_volumes + (1 - momentum) * volumes
        all_volumes.append(avg_volumes)

        # Check moves for convergence
        max_moves.append(torch.max(torch.linalg.norm(moves, dim=-1)).item())

        # Optional fixing
        if fixed == 'center':
            k_pts[:, 0, :] *= 0

        if verbose:
            print('iter {:5d} / max move = {:f}'.format(iter, max_moves[-1]))
            if warning:
                print('{:}WARNING: at least one point has no cell{:}'.format(bcolors.WARNING, bcolors.ENDC))
        

    ###################
    # User verification
    ###################

    # Convert to numpy
    all_volumes = torch.stack(all_volumes, 0)  # (iter, nk, K)
    all_volumes = all_volumes.cpu().numpy()
    max_moves = np.array(max_moves)   # (iter,)
    k_pts = k_pts.cpu().numpy()  # (nk, K, 3)

    # Get volumes
    avg_volume = np.mean(all_volumes[-20:], axis=0)  # (nk, K)
    avg_volume = avg_volume / np.sum(avg_volume, axis=1, keepdims=True)

    all_shell_std = []
    all_last_ring = []
    for kp, v in zip(k_pts, avg_volume):
        d = np.linalg.norm(kp, axis=-1)
        order = np.argsort(d)
        ordered_d = d[order]
        ordered_v = v[order]
        jumps = ordered_d[1:] - ordered_d[:-1]
        ring_inds = np.where(jumps > 0.05)[0] + 1
        ring_inds = np.insert(ring_inds, 0, 0)
        ring_inds = np.append(ring_inds, num_cells)
        shell_volumes = [ordered_v[i:j]  for i, j in zip(ring_inds[:-1], ring_inds[1:])]
        stds_per_shell = [np.std(s_v)/np.mean(s_v) for s_v in shell_volumes]
        ring_sizes = ring_inds[1:] - ring_inds[:-1]
        all_shell_std.append(np.max(stds_per_shell))
        all_last_ring.append(ring_sizes[-1])
        # stds = np.std(v)/np.mean(v)
        # print('n_ring {:d}  |  std1 {:.6f}  |  std2 {:.6f}  | '.format(len(ring_sizes), np.max(stds_per_shell), stds), ring_sizes)

    # Ignore candidates with too many points in the last ring
    all_shell_std = np.array(all_shell_std)
    all_last_ring = np.array(all_last_ring)
    mask = all_last_ring < np.max(all_last_ring)
    all_shell_std[mask] = 1.0

    # Choose the best candidate
    best_i = np.argmin(all_shell_std)
    k_pts = k_pts[best_i]
    avg_volume = avg_volume[best_i]
    
    # Show the convergence to ask user if this kernel is correct
    if verbose:
        all_volumes = all_volumes[:, best_i, :]
        plt.figure()
        plt.plot(max_moves)
        plt.figure()
        plt.plot(torch.arange(all_volumes.shape[0]), all_volumes)
        plt.title('Check if kernel is correct.')
        plt.show()

    # Rescale kernels with real radius
    return k_pts * radius, avg_volume, all_shell_std[best_i]


def kernel_point_optimization_debug(radius, num_points, num_kernels=1, dimension=3,
                                    fixed='center', ratio=0.66, verbose=0):
    """
    Creation of kernel point via optimization of potentials.
    :param radius: Radius of the kernels
    :param num_points: points composing kernels
    :param num_kernels: number of wanted kernels
    :param dimension: dimension of the space
    :param fixed: fix position of certain kernel points ('none', 'center' or 'verticals')
    :param ratio: ratio of the radius where you want the kernels points to be placed
    :param verbose: display option
    :return: points [num_kernels, num_points, dimension]
    """

    #######################
    # Parameters definition
    #######################

    # Radius used for optimization (points are rescaled afterwards)
    radius0 = 1
    diameter0 = 2

    # Factor multiplicating gradients for moving points (~learning rate)
    moving_factor = 1e-2
    continuous_moving_decay = 0.9995

    # Gradient threshold to stop optimization
    thresh = 1e-5

    # Gradient clipping value
    clip = 0.05 * radius0

    #######################
    # Kernel initialization
    #######################

    # Random kernel points
    kernel_points = np.random.rand(num_kernels * num_points - 1, dimension) * diameter0 - radius0
    while (kernel_points.shape[0] < num_kernels * num_points):
        new_points = np.random.rand(num_kernels * num_points - 1, dimension) * diameter0 - radius0
        kernel_points = np.vstack((kernel_points, new_points))
        d2 = np.sum(np.power(kernel_points, 2), axis=1)
        kernel_points = kernel_points[d2 < 0.5 * radius0 * radius0, :]
    kernel_points = kernel_points[:num_kernels * num_points, :].reshape((num_kernels, num_points, -1))

    # Optionnal fixing
    if fixed == 'center':
        kernel_points[:, 0, :] *= 0
    if fixed == 'verticals':
        kernel_points[:, :3, :] *= 0
        kernel_points[:, 1, -1] += 2 * radius0 / 3
        kernel_points[:, 2, -1] -= 2 * radius0 / 3

    #####################
    # Kernel optimization
    #####################

    # Initialize figure
    if verbose>1:
        fig = plt.figure()

    saved_gradient_norms = np.zeros((10000, num_kernels))
    old_gradient_norms = np.zeros((num_kernels, num_points))
    for iter in range(10000):

        # Compute gradients
        # *****************

        # Derivative of the sum of potentials of all points
        A = np.expand_dims(kernel_points, axis=2)
        B = np.expand_dims(kernel_points, axis=1)
        interd2 = np.sum(np.power(A - B, 2), axis=-1)
        inter_grads = (A - B) / (np.power(np.expand_dims(interd2, -1), 3/2) + 1e-6)
        inter_grads = np.sum(inter_grads, axis=1)

        # Derivative of the radius potential
        circle_grads = 10*kernel_points

        # All gradients
        gradients = inter_grads + circle_grads

        if fixed == 'verticals':
            gradients[:, 1:3, :-1] = 0

        # Stop condition
        # **************

        # Compute norm of gradients
        gradients_norms = np.sqrt(np.sum(np.power(gradients, 2), axis=-1))
        saved_gradient_norms[iter, :] = np.max(gradients_norms, axis=1)

        # Stop if all moving points are gradients fixed (low gradients diff)

        if fixed == 'center' and np.max(np.abs(old_gradient_norms[:, 1:] - gradients_norms[:, 1:])) < thresh:
            break
        elif fixed == 'verticals' and np.max(np.abs(old_gradient_norms[:, 3:] - gradients_norms[:, 3:])) < thresh:
            break
        elif np.max(np.abs(old_gradient_norms - gradients_norms)) < thresh:
            break
        old_gradient_norms = gradients_norms

        # Move points
        # ***********

        # Clip gradient to get moving dists
        moving_dists = np.minimum(moving_factor * gradients_norms, clip)

        # Fix central point
        if fixed == 'center':
            moving_dists[:, 0] = 0
        if fixed == 'verticals':
            moving_dists[:, 0] = 0

        # Move points
        kernel_points -= np.expand_dims(moving_dists, -1) * gradients / np.expand_dims(gradients_norms + 1e-6, -1)

        if verbose:
            print('iter {:5d} / max grad = {:f}'.format(iter, np.max(gradients_norms[:, 3:])))
        if verbose > 1:
            plt.clf()
            plt.plot(kernel_points[0, :, 0], kernel_points[0, :, 1], '.')
            circle = plt.Circle((0, 0), radius, color='r', fill=False)
            fig.axes[0].add_artist(circle)
            fig.axes[0].set_xlim((-radius*1.1, radius*1.1))
            fig.axes[0].set_ylim((-radius*1.1, radius*1.1))
            fig.axes[0].set_aspect('equal')
            plt.draw()
            plt.pause(0.001)
            plt.show(block=False)
            print(moving_factor)

        # moving factor decay
        moving_factor *= continuous_moving_decay

    # Rescale radius to fit the wanted ratio of radius
    r = np.sqrt(np.sum(np.power(kernel_points, 2), axis=-1))
    kernel_points *= ratio / np.mean(r[:, 1:])

    # Rescale kernels with real radius
    return kernel_points * radius, saved_gradient_norms


@torch.no_grad()
def shell_kernel_generator(radius, shell_n_pts, num_kernels=1, dimension=3, verbose=0):
    """
    Creation of kernel point via optimization of potentials.
    :param radius: Radius of the kernels
    :param shell_n_pts: list of the number of points per shell
    :param num_kernels: number of wanted kernels
    :param dimension: dimension of the space
    :param verbose: display option
    :return: points [num_kernels, num_points, dimension]
    """

    #######################
    # Parameters definition
    #######################

    n_shell = len(shell_n_pts)
    assert n_shell > 1
    assert shell_n_pts[0] == 1

    device = init_gpu()

    # Radius used for optimization (points are rescaled afterwards)
    radius0 = 1
    diameter0 = 2
    shell_l = diameter0 / (2 * n_shell - 1)
    shell_radiuses = [s * shell_l for s in range(n_shell)]
    total_points = int(np.sum(shell_n_pts))

    # Factor multiplicating gradients for moving points (~learning rate)
    moving_factor = 1e-2
    continuous_moving_decay = 0.9995

    # Gradient threshold to stop optimization
    thresh = 1e-5

    # Gradient clipping value
    clip = 0.05 * radius0

    #######################
    # Kernel initialization
    #######################

    # Add center points
    kernel_points = torch.zeros((num_kernels, 1, dimension))

    # Add random shell points
    for num_points, shell_r in zip(shell_n_pts[1:], shell_radiuses[1:]):

        if dimension == 2:
            theta = torch.rand(num_kernels, num_points) * 2 * np.pi
            u = torch.stack([torch.cos(theta), torch.sin(theta)], dim=2)

        elif dimension == 3:
            theta = torch.rand(num_kernels, num_points) * 2 * np.pi
            phi = (torch.rand(num_kernels, num_points) - 0.5) * np.pi
            u = torch.stack([torch.cos(theta) * torch.cos(phi), torch.sin(theta) * torch.cos(phi), torch.sin(phi)], axis=2)

        else:
            raise ValueError('Unsupported dimension for shelled kernel generation')
        kernel_points = torch.cat((kernel_points, u*shell_r), dim=1)

    kernel_points = kernel_points.to(device)


    #####################
    # Kernel optimization
    #####################

    # Initialize figure
    if verbose>1:
        fig = plt.figure()

    n_iter = 10000
    saved_gradient_norms = []
    old_gradient_norms = kernel_points.new_zeros((num_kernels, total_points))
    for iter in range(n_iter):

        # Compute gradients
        # *****************

        # Derivative of the sum of potentials of all points
        A = kernel_points.unsqueeze(2)
        B = kernel_points.unsqueeze(1)
        interd2 = torch.sum(torch.pow(A - B, 2), dim=-1)
        inter_grads = (A - B) / (torch.pow(interd2.unsqueeze(-1), 3/2) + 1e-6)
        gradients = torch.sum(inter_grads, dim=1)

        # Reduce gradients to tangential components (nk, K, 3)
        normals = kernel_points / (torch.linalg.norm(kernel_points, dim=-1, keepdims=True) + 1e-6)
        gradients -= torch.sum(gradients * normals, dim=-1, keepdims=True) * normals
            

        # Stop condition
        # **************

        # Compute norm of gradients
        gradients_norms = torch.sqrt(torch.sum(torch.pow(gradients, 2), dim=-1))
        saved_gradient_norms.append(torch.max(gradients_norms, dim=1)[0])

        # Stop if all moving points are gradients fixed (low gradients diff)

        if torch.max(torch.abs(old_gradient_norms[:, 1:] - gradients_norms[:, 1:])).item() < thresh:
            break
        old_gradient_norms = gradients_norms

        # Move points
        # ***********

        # Clip gradient to get moving dists
        moving_dists = torch.clamp(moving_factor * gradients_norms, max=clip)

        # Move points
        kernel_points -= moving_dists.unsqueeze(-1) * gradients / (gradients_norms.unsqueeze(-1) + 1e-6)

        # Readjust radiuses to remain on the shell
        i0 = 0
        for n_p, shell_r in zip(shell_n_pts, shell_radiuses):
            kernel_points[:, i0:i0+n_p] *= shell_r / (torch.linalg.norm(kernel_points[:, i0:i0+n_p], dim=-1, keepdims=True) + 1e-6)
            i0 += n_p

        if verbose:
            print('iter {:5d} / max grad = {:f}'.format(iter, torch.max(gradients_norms[:, 3:]).item()))
        if verbose > 1:
            plt.clf()
            p = kernel_points[0].cpu().numpy()
            plt.plot(p[:, 0], p[:, 1], '.')
            circle = plt.Circle((0, 0), radius, color='r', fill=False)
            fig.axes[0].add_artist(circle)
            for s_r in shell_radiuses[1:]:
                circle = plt.Circle((0, 0), s_r, color='g', ls='--', lw=0.1, fill=False)
                fig.axes[0].add_artist(circle)
            fig.axes[0].set_xlim((-radius*1.1, radius*1.1))
            fig.axes[0].set_ylim((-radius*1.1, radius*1.1))
            fig.axes[0].set_aspect('equal')
            plt.draw()
            plt.pause(0.001)
            plt.show(block=False)

        # moving factor decay
        moving_factor *= continuous_moving_decay

    # Rescale kernels with real radius
    kernel_points = (kernel_points * radius).cpu().numpy()
    saved_gradient_norms = torch.stack(saved_gradient_norms).cpu().numpy()

    return kernel_points, saved_gradient_norms


def load_kernels(radius, shell_sizes, dimension, fixed, lloyd=False):

    # Kernel directory
    kernel_dir = 'kernels/dispositions'
    if not exists(kernel_dir):
        makedirs(kernel_dir)

    # If we only give one number (total K), use Lloyd optimization
    lloyd = len(shell_sizes) < 2

    # Get number of kernel points
    num_kpoints = np.sum(shell_sizes)

    # Kernel_file
    kernel_file = join(kernel_dir, 'k_{:03d}_{:s}_{:d}D_{:d}.ply'.format(num_kpoints, fixed, dimension, int(lloyd)))

    # Check if already done
    if not exists(kernel_file):

        if lloyd:
            
            # Try multiple times and take the minimum std of volumes
            min_std = 1.0
            for _ in range(10):
                k_candidates, volumes, std = spherical_Lloyd_gpu(1.0,
                                                                    num_kpoints,
                                                                    num_kernels=10,
                                                                    dimension=dimension,
                                                                    fixed=fixed,
                                                                    verbose=0)
                if std < min_std:
                    min_std = std
                    kernel_points = k_candidates
            
        else:
            # Create kernels
            kernel_points, grad_norms = shell_kernel_generator(1.0,
                                                                shell_sizes,
                                                                num_kernels=100,
                                                                dimension=dimension,
                                                                verbose=0)

            # Find best candidate
            best_k = np.argmin(grad_norms[-1, :])

            # Save points
            kernel_points0 = kernel_points[best_k, :, :]
            volumes = np.zeros(kernel_points0.shape[0])

            # plt.figure()
            # plt.plot(torch.arange(grad_norms.shape[0]), grad_norms, '--')
            # plt.plot(torch.arange(grad_norms.shape[0]), grad_norms[:, best_k])
            # plt.title('Check if kernel is correct.')
            # plt.show()

        write_ply(kernel_file,
                    (kernel_points, volumes),
                    ['x', 'y', 'z', 'v'])

    else:
        data = read_ply(kernel_file)
        kernel_points = np.vstack((data['x'], data['y'], data['z'])).T
        volumes = data['v']


    # Random roations for the kernel
    # N.B. 4D random rotations not supported yet
    R = np.eye(dimension)
    theta = np.random.rand() * 2 * np.pi
    if dimension == 2:
        if fixed != 'vertical':
            c, s = np.cos(theta), np.sin(theta)
            R = np.array([[c, -s], [s, c]], dtype=np.float32)

    elif dimension == 3:
        if fixed != 'vertical':
            c, s = np.cos(theta), np.sin(theta)
            R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float32)

        else:
            phi = (np.random.rand() - 0.5) * np.pi

            # Create the first vector in carthesian coordinates
            u = np.array([np.cos(theta) * np.cos(phi), np.sin(theta) * np.cos(phi), np.sin(phi)])

            # Choose a random rotation angle
            alpha = np.random.rand() * 2 * np.pi

            # Create the rotation matrix with this vector and angle
            R = create_3D_rotations(np.reshape(u, (1, -1)), np.reshape(alpha, (1, -1)))[0]

            R = R.astype(np.float32)

    # Add a small noise
    kernel_points = kernel_points + np.random.normal(scale=0.001, size=kernel_points.shape)

    # Scale kernels
    kernel_points = radius * kernel_points

    # Rotate kernels
    kernel_points = np.matmul(kernel_points, R)

    return kernel_points.astype(np.float32)

