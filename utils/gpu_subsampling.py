
import imp
import numpy as np
import torch
import itertools
from pykeops.torch import Vi, Vj

from utils.gpu_init import init_gpu









# ----------------------------------------------------------------------------------------------------------------------
#
#           Grid Subsampling
#       \**********************/
#


@torch.no_grad()
def ravel_hash_func(voxels, max_voxels):
    dimension = voxels.shape[1]
    hash_values = voxels[:, 0].clone()
    for i in range(1, dimension):
        hash_values *= max_voxels[i]
        hash_values += voxels[:, i]
    return hash_values


@torch.no_grad()
def grid_subsample(points, voxel_size, features=None, labels=None, return_inverse=False):
    """Grid subsample of a simple point cloud on GPU.
    Args:
        points (Tensor): the original points (N, D).
        voxel_size (float): the voxel size.
    Returns:
        sampled_points (Tensor): the subsampled points (M, D).
    """

    # Parameters
    inv_voxel_size = 1.0 / voxel_size
    pts_dim = int(points.shape[1])

    # Get voxel indices for each points
    voxels = torch.floor(points * inv_voxel_size).long()

    # Pad to avoid negative pixel values
    voxels -= voxels.amin(0, keepdim=True)  # (L, D)

    # Vectorize multiple dimensions
    max_voxels = voxels.amax(0) + 1
    hash_values = ravel_hash_func(voxels, max_voxels)

    # Get unique values and subsampled indices
    _, inv_indices0, unique_counts = torch.unique(hash_values,
                                                 return_inverse=True,
                                                 return_counts=True)  # (M) (N) (M)

    # Get average points per voxel
    inv_indices = inv_indices0.unsqueeze(1).expand(-1, pts_dim)  # (N, D)
    s_points = torch.zeros(size=(unique_counts.shape[0], pts_dim)).to(points.device)  # (M, D)
    s_points.scatter_add_(0, inv_indices, points)  # (M, D)
    s_points /= unique_counts.unsqueeze(1).float()  # (M, D)

    if features is not None:
        # Get average features per voxel
        f_dim = int(features.shape[1])
        inv_indices = inv_indices0.unsqueeze(1).expand(-1, f_dim)  # (N, D)
        s_features = torch.zeros(size=(unique_counts.shape[0], f_dim)).to(points.device)  # (M, D)
        s_features.scatter_add_(0, inv_indices, features)  # (M, D)
        s_features /= unique_counts.unsqueeze(1).float()  # (M, D)

    if labels is not None:
        # Get most represented label per voxel
        one_hot_labels = torch.nn.functional.one_hot(labels) # (N, C)
        l_dim = int(one_hot_labels.shape[1])
        inv_indices = inv_indices0.unsqueeze(1).expand(-1, l_dim)  # (N, C)
        s_labels = torch.zeros(size=(unique_counts.shape[0], l_dim), dtype=torch.long).to(points.device)  # (M, C)
        s_labels.scatter_add_(0, inv_indices, one_hot_labels)  # (M, C)
        s_labels = torch.argmax(s_labels, dim=1) # (M, C)
    
    return_list = [s_points]
    if (features is not None):
        return_list.append(s_features)
    if (labels is not None):
        return_list.append(s_labels)
    if return_inverse:
        return_list.append(inv_indices0)
    if len(return_list) > 1:
        return return_list
    else:
        return return_list[0]


# ----------------------------------------------------------------------------------------------------------------------
#
#           Permutohedral Subsampling
#       \*******************************/
#


def ph_proj_mat(dim, debug=False):


    if debug:
        # vectors = [[1,  1,  1, -3],
        #            [1,  1, -2,  0],
        #            [1, -1,  0,  0],
        #            [1,  1,  1,  1]]

        # vectors = [[1,  1, -2],
        #            [1, -1,  0],
        #            [1,  1,  1]]

            
        dim = 4
        vectors = []
        for i in range(1, dim):
            v = np.zeros((dim,))
            for j in range(i):
                v[j] = 1
            v[i] = -i
            vectors.append(v)
        vectors.append(np.ones((dim,)))

        A = np.vstack(vectors).T
        print()
        print(A)
        print()

        A = A / np.linalg.norm(A, axis=0, keepdims=True)

        print()
        print(A)
        print()
        print(np.linalg.inv(A))
        print()
        print(np.linalg.det(A))
        print()

        # Verify that all vectors are othogonal
        print((np.matmul(A, A.T)*100).astype(np.int32))
        print()

        # Verify that any vector belonging to the plane z = 0 ,belongs to the plane x + y + z = 0 after projection
        points = (np.random.rand(7, dim) * 10).astype(np.float32)
        points[:, -1] *= 0

        pp = np.matmul(points, A.T)

        print()
        print(points)
        print()
        print(pp)
        print()
        print(np.sum(pp, axis=1, keepdims=True))
        print()

        a = 1/0

    else:
         
        # Create a set of vectors all othogonal to each other, including a 1-vector
        vectors = []
        for i in range(1, dim):
            v = np.zeros((dim,))
            for j in range(i):
                v[j] = 1
            v[i] = -i
            vectors.append(v)
        vectors.append(np.ones((dim,)))
        A = np.vstack(vectors).T

        # Normalize all vectors
        A = A / np.linalg.norm(A, axis=0, keepdims=True)

    return A.astype(np.float32)


@torch.no_grad()
def permutohedral_subsample(points, sub_size0, features=None, labels=None, return_inverse=False):
    """Permutohedral subsample of a simple point cloud on GPU.
    Args:
        points (Tensor): the original points (N, D).
        sub_size (float): the latice size (distance between pairs of subsampled points).
    Returns:
        sampled_points (Tensor): the subsampled points (M, D).
    """

    # Subampling size in elevated dimension should be higher sqrt(3/2)
    sub_size = sub_size0 * np.sqrt(3/2)

    # Parameters
    inv_sub_size = 1.0 / sub_size
    pts_dim = int(points.shape[1])

    # Project points to a higher dimension in the 1-plane Rotate to avoid deformation
    points_elevated = torch.cat((points, torch.zeros_like(points[:, :1])), dim=1)
    proj_mat = torch.from_numpy(ph_proj_mat(pts_dim + 1)).to(points.device)
    points_elevated = torch.matmul(points_elevated, proj_mat.T)

    # Trick: offset the hyper plane by 1/2 of the 1-vector (normal of the 1-plane)
    points_elevated += sub_size * 0.5

    # Get voxel indices for each points
    voxels = torch.floor(points_elevated * inv_sub_size).long()

    # Pad to avoid negative pixel values
    voxels -= voxels.amin(0, keepdim=True)  # (L, D + 1)

    # Vectorize multiple dimensions
    max_voxels = voxels.amax(0) + 1
    hash_values = ravel_hash_func(voxels, max_voxels)

    # # Debug:
    # f = hash_values.cpu().numpy()
    # p = points.cpu().numpy()
    # # Save ply
    # from utils.ply import write_ply
    # write_ply('results/test_ph.ply',
    #             [p, f.astype(np.int32)],
    #             ['x', 'y', 'z', 'f'])

    # Get unique values and subsampled indices
    aaa, inv_indices0, unique_counts = torch.unique(hash_values,
                                                    return_inverse=True,
                                                    return_counts=True)  # (M) (N) (M)

    # Get average points per voxel
    inv_indices = inv_indices0.unsqueeze(1).expand(-1, pts_dim)  # (N, D)
    s_points = torch.zeros(size=(unique_counts.shape[0], pts_dim)).to(points.device)  # (M, D)
    s_points.scatter_add_(0, inv_indices, points)  # (M, D)
    s_points /= unique_counts.unsqueeze(1).float()  # (M, D)

    if features is not None:
        # Get average features per voxel
        f_dim = int(features.shape[1])
        inv_indices = inv_indices0.unsqueeze(1).expand(-1, f_dim)  # (N, D)
        s_features = torch.zeros(size=(unique_counts.shape[0], f_dim)).to(points.device)  # (M, D)
        s_features.scatter_add_(0, inv_indices, features)  # (M, D)
        s_features /= unique_counts.unsqueeze(1).float()  # (M, D)

    if labels is not None:
        # Get most represented label per voxel
        one_hot_labels = torch.nn.functional.one_hot(labels) # (N, C)
        l_dim = int(one_hot_labels.shape[1])
        inv_indices = inv_indices0.unsqueeze(1).expand(-1, l_dim)  # (N, C)
        s_labels = torch.zeros(size=(unique_counts.shape[0], l_dim), dtype=torch.long).to(points.device)  # (M, C)
        s_labels.scatter_add_(0, inv_indices, one_hot_labels)  # (M, C)
        s_labels = torch.argmax(s_labels, dim=1) # (M, C)

    return_list = [s_points]
    if (features is not None):
        return_list.append(s_features)
    if (labels is not None):
        return_list.append(s_labels)
    if return_inverse:
        return_list.append(inv_indices0)
    if len(return_list) > 1:
        return return_list
    else:
        return return_list[0]


@torch.no_grad()
def hexagonal_subsample(points, sub_size0, features=None, labels=None, return_inverse=False):
    """
    Similar to permutohedral subsample but with overlap between subampling regions. Slower but better quality.
    Cannot return inverse indices though.
    Args:
         points (Tensor): the original points (N, D).
        sub_size (float): the latice size (distance between pairs of subsampled points).
    Returns:
        sampled_points (Tensor): the subsampled points (M, D).
    """

    # Subampling size in elevated dimension should be higher sqrt(3/2)
    sub_size = sub_size0 * np.sqrt(3/2)

    # Parameters
    inv_sub_size = 1.0 / sub_size
    pts_dim = int(points.shape[1])

    # Project points to a higher dimension in the 1-plane Rotate to avoid deformation
    points_elevated = torch.cat((points, torch.zeros_like(points[:, :1])), dim=1)
    proj_mat = torch.from_numpy(ph_proj_mat(pts_dim + 1)).to(points.device)
    points_elevated = torch.matmul(points_elevated, proj_mat.T)

    # Get voxel indices for each points
    voxels = torch.floor(points_elevated * inv_sub_size).long()

    # Pad to avoid negative pixel values
    voxels -= voxels.amin(0, keepdim=True)  # (L, D + 1)

    # We padd in all positive direction excpet the 1 vector
    offsets = list(itertools.product([0, 1], repeat=pts_dim+1))
    offsets = np.array(offsets[1:-1])

    print(offsets)
    print(offsets.shape)

    a = 1/0



    # Vectorize multiple dimensions
    max_voxels = voxels.amax(0) + 1
    hash_values = ravel_hash_func(voxels, max_voxels)

    # # Debug:
    # f = hash_values.cpu().numpy()
    # p = points.cpu().numpy()
    # # Save ply
    # from utils.ply import write_ply
    # write_ply('results/test_ph.ply',
    #             [p, f.astype(np.int32)],
    #             ['x', 'y', 'z', 'f'])

    # Get unique values and subsampled indices
    aaa, inv_indices0, unique_counts = torch.unique(hash_values,
                                                    return_inverse=True,
                                                    return_counts=True)  # (M) (N) (M)

    # Get average points per voxel
    inv_indices = inv_indices0.unsqueeze(1).expand(-1, pts_dim)  # (N, D)
    s_points = torch.zeros(size=(unique_counts.shape[0], pts_dim)).to(points.device)  # (M, D)
    s_points.scatter_add_(0, inv_indices, points)  # (M, D)
    s_points /= unique_counts.unsqueeze(1).float()  # (M, D)

    if features is not None:
        # Get average features per voxel
        f_dim = int(features.shape[1])
        inv_indices = inv_indices0.unsqueeze(1).expand(-1, f_dim)  # (N, D)
        s_features = torch.zeros(size=(unique_counts.shape[0], f_dim)).to(points.device)  # (M, D)
        s_features.scatter_add_(0, inv_indices, features)  # (M, D)
        s_features /= unique_counts.unsqueeze(1).float()  # (M, D)

    if labels is not None:
        # Get most represented label per voxel
        one_hot_labels = torch.nn.functional.one_hot(labels) # (N, C)
        l_dim = int(one_hot_labels.shape[1])
        inv_indices = inv_indices0.unsqueeze(1).expand(-1, l_dim)  # (N, C)
        s_labels = torch.zeros(size=(unique_counts.shape[0], l_dim), dtype=torch.long).to(points.device)  # (M, C)
        s_labels.scatter_add_(0, inv_indices, one_hot_labels)  # (M, C)
        s_labels = torch.argmax(s_labels, dim=1) # (M, C)

    return_list = [s_points]
    if (features is not None):
        return_list.append(s_features)
    if (labels is not None):
        return_list.append(s_labels)
    if return_inverse:
        return_list.append(inv_indices0)
    if len(return_list) > 1:
        return return_list
    else:
        return return_list[0]

# ----------------------------------------------------------------------------------------------------------------------
#
#           High level functions
#       \**************************/
#


def subsample_numpy(points, sub_size, features=None, labels=None, method='grid', on_gpu=True, return_inverse=False):
    """
    Subsample numpy point clouds on GPU
    """

    # Get the right subsampling func
    if method == 'grid':
        subsampling_func = grid_subsample
    elif method == 'ph':
        subsampling_func = permutohedral_subsample
    elif method == 'hex':
        subsampling_func = hexagonal_subsample
    else:
        raise ValueError('Wrong Subsampling method: {:s}'.format(method))

    # Convert to GPU
    if on_gpu:
        device = init_gpu()
    else:
        device = torch.device("cpu")
    point_gpu = torch.from_numpy(np.copy(points)).to(device)
    if features is None:
        features_gpu = None
    else:
        features_gpu = torch.from_numpy(np.copy(features)).to(device)
    if labels is None:
        labels_gpu = None
    else:
        labels_gpu = torch.from_numpy(np.copy(labels)).type(torch.long).to(device)

    # Subsample cloud
    return_list = subsampling_func(point_gpu,
                                   sub_size,
                                   features=features_gpu,
                                   labels=labels_gpu,
                                   return_inverse=return_inverse)

    # Convert back to numpy
    if isinstance(return_list, list):
        return_list = [gpu_tensor.cpu().numpy() for gpu_tensor in return_list]
    else:
        return_list = return_list.cpu().numpy()

    return return_list


def subsample_list_mode(points, lengths, sub_size, method='grid'):
    """
    Subsample torch batch of point clouds with different method,
    separating the batch in a list of point clouds.
    Args:
        points (Tensor): the original points (N, D).
        lengths (LongTensor): the numbers of points in the batch (B,).
        sub_size (float): the voxel size.
        method (str): the subsampling method ('grid', 'ph', 'fps').
    Returns:
        sampled_points (Tensor): the subsampled points (M, D).
        sampled_lengths (Tensor): the numbers of subsampled points in the batch (B,).
    """

    # Get the right subsampling func
    if method == 'grid':
        subsampling_func = grid_subsample
    elif method == 'ph':
        subsampling_func = permutohedral_subsample
    elif method == 'hex':
        subsampling_func = hexagonal_subsample
    else:
        raise ValueError('Wrong Subsampling method: {:s}'.format(method))

    # Convert length to tensor
    if type(lengths) == list:
        lengths = torch.LongTensor(lengths).to(points.device)

    # Parameters
    batch_size = lengths.shape[0]
    start_index = 0
    sampled_points_list = []

    # Looping on each batch point cloud
    for i in range(batch_size):
        length = lengths[i].item()
        end_index = start_index + length
        points0 = points[start_index:end_index]
        sampled_points_list.append(subsampling_func(points0, sub_size))
        start_index = end_index

    # Packing the list of points
    sampled_points = torch.cat(sampled_points_list, dim=0)
    sampled_lengths = [x.shape[0] for x in sampled_points_list]
    return sampled_points, sampled_lengths
