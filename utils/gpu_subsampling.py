
import numpy as np
import torch
import itertools

from utils.gpu_init import init_gpu, tensor_MB

from utils.cuda_funcs import furthest_point_sample
from utils.cpp_funcs import furthest_point_sample_cpp


# ----------------------------------------------------------------------------------------------------------------------
#
#           Grid Subsampling
#       \**********************/
#

@torch.no_grad()
def reverse_ravel_hash(hash_values, max_voxels):
    dimension = int(max_voxels.shape[-1])
    voxels = torch.zeros((int(hash_values.shape[0]), dimension),
                         dtype=hash_values.dtype).to(hash_values.device)
    for i in range(0, dimension):
        divisor = torch.prod(max_voxels[i+1:])
        divided = torch.div(hash_values, divisor, rounding_mode="floor")
        voxels[..., i] = divided
        hash_values -= divided * divisor
    return voxels

@torch.no_grad()
def ravel_hash_func(voxels, max_voxels):
    dimension = voxels.shape[-1]
    hash_values = voxels[..., 0].clone()
    for i in range(1, dimension):
        hash_values *= max_voxels[i]
        hash_values += voxels[..., i]
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

    # # Trick: offset the hyper plane by 1/2 of the 1-vector (normal of the 1-plane)
    # points_elevated += sub_size * 0.5

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
    # if p.shape[1] == 2:
    #     p = np.concatenate((p, p[:, :1]*0), axis=1)
    #     write_ply('results/test_ph2.ply',
    #                 [p, voxels.cpu().numpy().astype(np.int32), f.astype(np.int32)],
    #                 ['x', 'y', 'z', 'i', 'j', 'k', 'f'])
    # else:   
    #     write_ply('results/test_ph2.ply',
    #                 [p, voxels.cpu().numpy().astype(np.int32), f.astype(np.int32)],
    #                 ['x', 'y', 'z', 'i', 'j', 'k', 'l', 'f'])

    # write_ply('results/test_ph.ply',
    #             [p, f.astype(np.int32)],
    #             ['x', 'y', 'z', 'f'])

    # a = 1/0

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


# # Test Permutohedral
# x = torch.arange(-1, 5, 0.05)
# y = torch.arange(-1, 5, 0.05)
# z = torch.arange(-1, 5, 0.05)
# grid_x, grid_y, grid_z = torch.meshgrid(x, y, z, indexing='xy')
# pts = torch.concat((torch.reshape(grid_x, (-1, 1)), torch.reshape(grid_y, (-1, 1)), torch.reshape(grid_z, (-1, 1))), dim=1)

# s = permutohedral_subsample(pts, 0.3)

# a = 1/0


@torch.no_grad()
def hexagonal_subsample(points, sub_size0, features=None, labels=None, return_inverse=False, eps=1e-6):
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
    
    # Prepare labels if needed
    if labels is not None:
        one_hot_labels = torch.nn.functional.one_hot(labels).type(torch.int16)  # (N, C)
    else:
        one_hot_labels = None

    # Project points to a higher dimension in the 1-plane Rotate to avoid deformation
    points_elevated = torch.cat((points + eps, torch.zeros_like(points[:, :1])), dim=1)
    proj_mat = torch.from_numpy(ph_proj_mat(pts_dim + 1)).to(points.device)
    points_elevated = torch.matmul(points_elevated, proj_mat.T)

    # Get voxel indices for each points
    voxels = torch.floor(points_elevated * inv_sub_size).long()

    # We offset in diferrent directions to get a neighborhood of multiple permutohedral cells
    offsets = np.array(list(itertools.product([0, 1], repeat=pts_dim+1)), dtype=np.int64)
    offsets[:, -1] *= -1
    offsets = torch.from_numpy(offsets).to(points.device) # (O, D)

    # Pad to avoid negative pixel values  (-1 because of offsets)
    vox_origin = voxels.amin(0, keepdim=True) - 1
    voxels -= vox_origin

    # Get maximum values (+1 because of offsets)
    max_voxels = voxels.amax(0) + 2
    
    # Get the type of voxel with the sum of coordinates
    type_values = torch.unique(torch.sum(voxels, dim=1))
    if pts_dim == 2:
        wanted_type = type_values[0].item()
    elif pts_dim == 3:
        wanted_type = type_values[1].item()
    else:
        wanted_type = type_values[0].item()

    # get hash values for each offset
    hash_values = []
    hash_inds = []
    for offset in offsets:

        # We automatically know an offset will not be valid
        ###################################################

        # We already the sum of coordinates after offset:
        new_types = type_values - torch.sum(offset)

        # if the sum of it is not contained in the valid types
        if wanted_type not in new_types:
            continue

        # Otherwise offset and only keep the wanted cells
        #################################################

        # offset voxels
        voxels_off = voxels - offset

        # Get voxel type 
        vox_types = torch.sum(voxels_off, dim=1)  # (N, )

        # Only consider valid voxels
        valid_mask = vox_types == wanted_type
        voxels_off = voxels_off[valid_mask]  # (Ni, )

        # Vectorize multiple dimensions
        hash_values.append(ravel_hash_func(voxels_off, max_voxels)) # (Ni,)
        hash_inds.append(torch.nonzero(valid_mask, as_tuple=True)[0])

    # Stack all offset points and hash
    hash_values = torch.concat(hash_values, dim=0)  # (Ntot)
    hash_inds = torch.concat(hash_inds, dim=0)  # (Ntot)

    # Get unique values and subsampled indices
    unique_hashs, inv_indices0, unique_counts = torch.unique(hash_values,
                                                  return_inverse=True,
                                                  return_counts=True)  # (M,) (Ntot,) (M,)

    # Get average points per voxel
    inv_indices = inv_indices0.unsqueeze(1).expand(-1, pts_dim)  # (Ntot, D)
    s_points = torch.zeros(size=(unique_counts.shape[0], pts_dim)).to(points.device)  # (M, D)
    s_points.scatter_add_(0, inv_indices, points[hash_inds])  # (M, D)
    s_points /= unique_counts.unsqueeze(1).float()  # (M, D)


    # #### DEBUG: central point instead of barycenter ####
    # #
    # # Get central point for each occupied cell
    # unique_voxels = reverse_ravel_hash(unique_hashs, max_voxels)
    # unique_voxels += vox_origin
    # # unique_voxels += torch.tensor([1, 1, 1, -1], dtype=torch.long, device=unique_voxels.device)
    # central_pts_e = unique_voxels * sub_size
    # central_pts = torch.matmul(central_pts_e, proj_mat)
    # s_points = central_pts[:, :pts_dim]
    # #### DEBUG: central point instead of barycenter ####

    # #
    # #   Next step erase barycenters that are to far from cell center
    # #


    if features is not None:
        # Get average features per voxel
        f_dim = int(features.shape[1])
        inv_indices = inv_indices0.unsqueeze(1).expand(-1, f_dim)  # (N, D)
        s_features = torch.zeros(size=(unique_counts.shape[0], f_dim)).to(points.device)  # (M, D)
        s_features.scatter_add_(0, inv_indices, features[hash_inds])  # (M, D)
        s_features /= unique_counts.unsqueeze(1).float()  # (M, D)

    if labels is not None:
        # Get most represented label per voxel
        l_dim = int(one_hot_labels.shape[1])
        inv_indices = inv_indices0.unsqueeze(1).expand(-1, l_dim)  # (N, C)
        s_labels = torch.zeros(size=(unique_counts.shape[0], l_dim), dtype=torch.int16).to(points.device)  # (M, C)
        s_labels.scatter_add_(0, inv_indices, one_hot_labels[hash_inds])  # (M, C)
        s_labels = torch.argmax(s_labels, dim=1) # (M,)

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


# # Test Hexagonal
# x = torch.arange(-1, 5, 0.05)
# y = torch.arange(-1, 5, 0.05)
# z = torch.arange(-1, 5, 0.05)
# grid_x, grid_y, grid_z = torch.meshgrid(x, y, z, indexing='xy')
# pts = torch.concat((torch.reshape(grid_x, (-1, 1)), torch.reshape(grid_y, (-1, 1)), torch.reshape(grid_z, (-1, 1))), dim=1)

# s = hexagonal_subsample(pts, 0.3)

# a = 1/0


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


def fp_subsample(points, sub_size,
                 features=None,
                 labels=None,
                 return_inverse=False):

    # Choose which function we use
    if 'cuda' in points.device.type:
        subsampling_func = furthest_point_sample
    else:
        subsampling_func = furthest_point_sample_cpp

    # Choose if we use min distance or stride for fps
    if sub_size < 0:
        stride = -sub_size
        min_d = 0
    else:
        stride = 1
        min_d = sub_size

    # Get subsample indices
    sub_inds = subsampling_func(points, stride=stride, min_d=min_d)

    # Get subsampled data
    sub_points = points[sub_inds, :]
    if features is not None:
        sub_features = features[sub_inds, :]
    if labels is not None:
        sub_labels = labels[sub_inds]

    # FPS does not provide inverse indices
    inv_indices0 = None

    return_list = [sub_points]
    if (features is not None):
        return_list.append(sub_features)
    if (labels is not None):
        return_list.append(sub_labels)
    if return_inverse:
        return_list.append(inv_indices0)
    if len(return_list) > 1:
        return return_list
    else:
        return return_list[0]


def subsample_cloud(points, sub_size, features=None, labels=None, method='grid', return_inverse=False):
    """
    Subsample torch point clouds with different method, handling features and labels as well.
    Args:
        points (Tensor): the original points (N, D).
        sub_size (float): the voxel size.
        features (Tensor): the original features (N, F).
        labels (LongTensor): the original labels (N,).
        method (str): the subsampling method ('grid', 'ph', 'fps').
        return_inverse (bool): Do we return inverse indices.
    Returns:
        sampled_points (Tensor): the subsampled points (M, D).
    """

    # Get the right subsampling func
    if method == 'grid':
        subsampling_func = grid_subsample
    elif method == 'ph':
        subsampling_func = permutohedral_subsample
    elif method == 'hex':
        subsampling_func = hexagonal_subsample
    elif method == 'fps':
        subsampling_func = fp_subsample
    else:
        raise ValueError('Wrong Subsampling method: {:s}'.format(method))

    # Subsample cloud
    return_list = subsampling_func(points,
                                   sub_size,
                                   features=features,
                                   labels=labels,
                                   return_inverse=return_inverse)

    return return_list

    
def subsample_pack_batch(points, lengths, sub_size, method='grid'):
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
    elif method == 'fps':
        subsampling_func = fp_subsample
    else:
        raise ValueError('Wrong Subsampling method: {:s}'.format(method))
    
    if method != 'fps' and sub_size < 0:
        raise ValueError('Negative subsampling size (stride) is only supported with fps subsampling.')

    # Convert length to tensor
    length_as_list = type(lengths) == list
    if length_as_list:
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
    if not length_as_list:
        sampled_lengths = torch.LongTensor(sampled_lengths).to(points.device)

    return sampled_points, sampled_lengths
