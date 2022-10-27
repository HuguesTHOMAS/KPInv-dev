
import numpy as np
import torch
from utils.rotation import create_3D_rotations

# Subsampling extension
import cpp_wrappers.cpp_subsampling.cpp_subsampling as cpp_subsampling
import cpp_wrappers.cpp_neighbors.cpp_neighbors as cpp_neighbors


# ----------------------------------------------------------------------------------------------------------------------
#
#           Utility functions
#       \***********************/
#


def furthest_point_sample_cpp(points, new_n=None, stride=4, min_d=0.):
    """
    CPP wrapper for a furthest point subsampling
    """

    # Create new_lengths if not provided
    if new_n is None:
        new_n = int(np.floor(points.shape[0] / stride))

    return cpp_subsampling.furthest_point_sample(points,
                                                 new_n=new_n,
                                                 min_d=min_d)


def grid_subsampling(points, features=None, labels=None, sampleDl=0.1, verbose=0):
    """
    CPP wrapper for a grid subsampling (method = barycenter for points and features)
    :param points: (N, 3) matrix of input points
    :param features: optional (N, d) matrix of features (floating number)
    :param labels: optional (N,) matrix of integer labels
    :param sampleDl: parameter defining the size of grid voxels
    :param verbose: 1 to display
    :return: subsampled points, with features and/or labels depending of the input
    """

    if (features is None) and (labels is None):
        return cpp_subsampling.grid_subsample(points,
                                         sampleDl=sampleDl,
                                         verbose=verbose)
    elif (labels is None):
        return cpp_subsampling.grid_subsample(points,
                                         features=features,
                                         sampleDl=sampleDl,
                                         verbose=verbose)
    elif (features is None):
        return cpp_subsampling.grid_subsample(points,
                                         classes=labels,
                                         sampleDl=sampleDl,
                                         verbose=verbose)
    else:
        return cpp_subsampling.grid_subsample(points,
                                         features=features,
                                         classes=labels,
                                         sampleDl=sampleDl,
                                         verbose=verbose)


def batch_grid_partition(points, batches_len, sampleDl=0.1):
    """
    CPP wrapper for a grid subsampling (method = barycenter for points and features).
    Also returns pooling and upsampling inds
    :param points: (N, 3) matrix of input points
    :param sampleDl: parameter defining the size of grid voxels
    :return: subsampled points, with features and/or labels depending of the input
    """


    s_points, s_len, pools, ups = cpp_subsampling.batch_grid_partitionning(points,
                                                                           batches_len,
                                                                           sampleDl=sampleDl)

    if torch.is_tensor(points):
        s_points = torch.from_numpy(s_points).to(points.device)
        s_len = torch.from_numpy(s_len).to(points.device)
        pools = torch.from_numpy(pools).to(points.device)
        ups = torch.from_numpy(ups).to(points.device)

    return s_points, s_len, pools, ups



def batch_grid_subsampling(points, batches_len, features=None, labels=None,
                           sampleDl=0.1, max_p=0, verbose=0, random_grid_orient=True):
    """
    CPP wrapper for a grid subsampling (method = barycenter for points and features)
    :param points: (N, 3) matrix of input points
    :param features: optional (N, d) matrix of features (floating number)
    :param labels: optional (N,) matrix of integer labels
    :param sampleDl: parameter defining the size of grid voxels
    :param verbose: 1 to display
    :return: subsampled points, with features and/or labels depending of the input
    """

    R = None
    B = len(batches_len)
    if random_grid_orient:

        ########################################################
        # Create a random rotation matrix for each batch element
        ########################################################

        # Choose two random angles for the first vector in polar coordinates
        theta = np.random.rand(B) * 2 * np.pi
        phi = (np.random.rand(B) - 0.5) * np.pi

        # Create the first vector in carthesian coordinates
        u = np.vstack([np.cos(theta) * np.cos(phi), np.sin(theta) * np.cos(phi), np.sin(phi)])

        # Choose a random rotation angle
        alpha = np.random.rand(B) * 2 * np.pi

        # Create the rotation matrix with this vector and angle
        R = create_3D_rotations(u.T, alpha).astype(np.float32)

        #################
        # Apply rotations
        #################

        i0 = 0
        points = points.copy()
        for bi, length in enumerate(batches_len):
            # Apply the rotation
            points[i0:i0 + length, :] = np.sum(np.expand_dims(points[i0:i0 + length, :], 2) * R[bi], axis=1)
            i0 += length

    #######################
    # Sunsample and realign
    #######################

    if (features is None) and (labels is None):
        s_points, s_len = cpp_subsampling.grid_subsample_batch(points,
                                                          batches_len,
                                                          sampleDl=sampleDl,
                                                          max_p=max_p,
                                                          verbose=verbose)
        if random_grid_orient:
            i0 = 0
            for bi, length in enumerate(s_len):
                s_points[i0:i0 + length, :] = np.sum(np.expand_dims(s_points[i0:i0 + length, :], 2) * R[bi].T, axis=1)
                i0 += length
        return s_points, s_len

    elif (labels is None):
        s_points, s_len, s_features = cpp_subsampling.grid_subsample_batch(points,
                                                                      batches_len,
                                                                      features=features,
                                                                      sampleDl=sampleDl,
                                                                      max_p=max_p,
                                                                      verbose=verbose)
        if random_grid_orient:
            i0 = 0
            for bi, length in enumerate(s_len):
                # Apply the rotation
                s_points[i0:i0 + length, :] = np.sum(np.expand_dims(s_points[i0:i0 + length, :], 2) * R[bi].T, axis=1)
                i0 += length
        return s_points, s_len, s_features

    elif (features is None):
        s_points, s_len, s_labels = cpp_subsampling.grid_subsample_batch(points,
                                                                    batches_len,
                                                                    classes=labels,
                                                                    sampleDl=sampleDl,
                                                                    max_p=max_p,
                                                                    verbose=verbose)
        if random_grid_orient:
            i0 = 0
            for bi, length in enumerate(s_len):
                # Apply the rotation
                s_points[i0:i0 + length, :] = np.sum(np.expand_dims(s_points[i0:i0 + length, :], 2) * R[bi].T, axis=1)
                i0 += length
        return s_points, s_len, s_labels

    else:
        s_points, s_len, s_features, s_labels = cpp_subsampling.grid_subsample_batch(points,
                                                                              batches_len,
                                                                              features=features,
                                                                              classes=labels,
                                                                              sampleDl=sampleDl,
                                                                              max_p=max_p,
                                                                              verbose=verbose)
        if random_grid_orient:
            i0 = 0
            for bi, length in enumerate(s_len):
                # Apply the rotation
                s_points[i0:i0 + length, :] = np.sum(np.expand_dims(s_points[i0:i0 + length, :], 2) * R[bi].T, axis=1)
                i0 += length
        return s_points, s_len, s_features, s_labels


def batch_radius_neighbors(queries, supports, q_batches, s_batches, radius, neighbor_limit):
    """
    Computes neighbors for a batch of queries and supports
    :param queries: (N1, 3) the query points
    :param supports: (N2, 3) the support points
    :param q_batches: (B) the list of lengths of batch elements in queries
    :param s_batches: (B)the list of lengths of batch elements in supports
    :param radius: float32
    :return: neighbors indices
    """

    from_torch = False
    if isinstance(queries, torch.Tensor):
        queries = queries.numpy()
        from_torch = True

    if isinstance(supports, torch.Tensor):
        supports = supports.numpy()

    if isinstance(q_batches, torch.Tensor):
        q_batches = q_batches.numpy().astype(np.int32)

    if isinstance(s_batches, torch.Tensor):
        s_batches = s_batches.numpy().astype(np.int32)

    # Get radius neighbors
    neighbors = cpp_neighbors.batch_radius_neighbors(queries, supports, q_batches, s_batches, radius=radius)

    # Apply limit
    neighbors = neighbors[:, :neighbor_limit]
    
    # Reconvert to tensor if needed
    if from_torch:
        neighbors = torch.from_numpy(neighbors).to(torch.long)

    return neighbors


def batch_knn_neighbors(queries, supports, q_batches, s_batches, radius, neighbor_limit, return_dist=False):
    """
    Computes neighbors for a batch of queries and supports
    :param queries: (N1, 3) the query points
    :param supports: (N2, 3) the support points
    :param q_batches: (B) the list of lengths of batch elements in queries
    :param s_batches: (B)the list of lengths of batch elements in supports
    :param radius: float32
    :return: neighbors indices
    """

    from_torch = False
    if isinstance(queries, torch.Tensor):
        queries = queries.numpy()
        supports = supports.numpy()
        from_torch = True

    if isinstance(q_batches, torch.Tensor):
        q_batches = q_batches.numpy().astype(np.int32)
        s_batches = s_batches.numpy().astype(np.int32)

    # Get radius neighbors
    neighbors, sq_dists = cpp_neighbors.batch_knn_neighbors(queries, supports, q_batches, s_batches, n_neighbors=neighbor_limit)
    
    # Reconvert to tensor if needed
    if from_torch:
        neighbors = torch.from_numpy(neighbors).to(torch.long)

    if return_dist:
        dists = np.sqrt(sq_dists)
        if from_torch:
            dists = torch.from_numpy(dists)
        return neighbors, dists

    return neighbors



