from typing import Tuple

import torch
from torch import Tensor
import pykeops
from pykeops.torch import LazyTensor
pykeops.set_verbose(False)

from utils.batch_conversion import batch_to_pack, pack_to_batch, pack_to_list, list_to_pack
from utils.gpu_subsampling import subsample_list_mode


# ----------------------------------------------------------------------------------------------------------------------
#
#           Implementation of k-nn search in Keops
#       \********************************************/
#

@torch.no_grad()
def keops_radius_count(q_points: Tensor, s_points: Tensor, radius: float) -> Tensor:
    """
    Count neigbors inside radius with PyKeOps.
    Args:
        q_points (Tensor): (*, N, C)
        s_points (Tensor): (*, M, C)
        radius (float)
    Returns:
        radius_counts (Tensor): (*, N)
    """
    num_batch_dims = q_points.dim() - 2
    xi = LazyTensor(q_points.unsqueeze(-2))  # (*, N, 1, C)
    xj = LazyTensor(s_points.unsqueeze(-3))  # (*, 1, M, C)
    dij = (xi - xj).norm2()  # (*, N, M)
    vij = (radius - dij).relu().sign()  # (*, N, M)
    radius_counts = vij.sum(dim=num_batch_dims + 1)  # (*, N)
    return radius_counts

@torch.no_grad()
def keops_knn(q_points: Tensor, s_points: Tensor, k: int) -> Tuple[Tensor, Tensor]:
    """
    kNN with PyKeOps.
    Args:
        q_points (Tensor): (*, N, C)
        s_points (Tensor): (*, M, C)
        k (int)
    Returns:
        knn_distance (Tensor): (*, N, k)
        knn_indices (LongTensor): (*, N, k)
    """
    xi = LazyTensor(q_points.unsqueeze(-2))  # (*, N, 1, C)
    xj = LazyTensor(s_points.unsqueeze(-3))  # (*, 1, M, C)
    dij = (xi - xj).norm2()  # (*, N, M)
    knn_distances, knn_indices = dij.Kmin_argKmin(k, dim=q_points.dim() - 1)  # (*, N, K)
    return knn_distances, knn_indices

@torch.no_grad()
def knn(q_points: Tensor,
        s_points: Tensor,
        k: int,
        dilation: int = 1,
        distance_limit: float = None,
        return_distance: bool = False,
        remove_nearest: bool = False,
        transposed: bool = False,
        padding_mode: str = "nearest",
        inf: float = 1e10):
    """
    Compute the kNNs of the points in `q_points` from the points in `s_points`.
    Use KeOps to accelerate computation.
    Args:
        s_points (Tensor): coordinates of the support points, (*, C, N) or (*, N, C).
        q_points (Tensor): coordinates of the query points, (*, C, M) or (*, M, C).
        k (int): number of nearest neighbors to compute.
        dilation (int): dilation for dilated knn.
        distance_limit (float=None): if further than this radius, the neighbors are replaced according to `padding_mode`.
        return_distance (bool=False): whether return distances.
        remove_nearest (bool=True) whether remove the nearest neighbor (itself).
        transposed (bool=False): if True, the points shape is (*, C, N).
        padding_mode (str='nearest'): padding mode for neighbors further than distance radius. ('nearest', 'empty').
        inf (float=1e10): infinity value for padding.
    Returns:
        knn_distances (Tensor): The distances of the kNNs, (*, M, k).
        knn_indices (LongTensor): The indices of the kNNs, (*, M, k).
    """
    if transposed:
        q_points = q_points.transpose(-1, -2)  # (*, C, N) -> (*, N, C)
        s_points = s_points.transpose(-1, -2)  # (*, C, M) -> (*, M, C)

    num_s_points = s_points.shape[-2]

    dilated_k = (k - 1) * dilation + 1
    if remove_nearest:
        dilated_k += 1
    final_k = min(dilated_k, num_s_points)

    knn_distances, knn_indices = keops_knn(q_points, s_points, final_k)  # (*, N, k)

    if remove_nearest:
        knn_distances = knn_distances[..., 1:]
        knn_indices = knn_indices[..., 1:]

    if dilation > 1:
        knn_distances = knn_distances[..., ::dilation]
        knn_indices = knn_indices[..., ::dilation]

    knn_distances = knn_distances.contiguous()
    knn_indices = knn_indices.contiguous()

    if distance_limit is not None:
        assert padding_mode in ["nearest", "empty"]
        knn_masks = torch.ge(knn_distances, distance_limit)
        if padding_mode == "nearest":
            knn_distances[knn_masks] = knn_distances[..., 0]
            knn_indices[knn_masks] = knn_indices[..., 0]
        else:
            knn_distances[knn_masks] = inf
            knn_indices[knn_masks] = num_s_points

    if return_distance:
        return knn_distances, knn_indices

    return knn_indices

@torch.no_grad()
def radius_search_pack_mode(q_points, s_points, q_lengths, s_lengths, radius, neighbor_limit, shadow=False, inf=1e10):
    """Radius search in pack mode (fast version).
    Args:
        q_points (Tensor): query points (M, 3).
        s_points (Tensor): support points (N, 3).
        q_lengths (LongTensor): the numbers of query points in the batch (B,).
        s_lengths (LongTensor): the numbers of support points in the batch (B,).
        radius (float): radius radius.
        neighbor_limit (int): neighbor radius.
        inf (float=1e10): infinity value.
    Returns:
        neighbor_indices (LongTensor): the indices of the neighbors. Equal to N if not exist.
    """

    # pack to batch
    batch_q_points, batch_q_masks = pack_to_batch(q_points, q_lengths, fill_value=inf)  # (B, M', 3)
    batch_s_points, batch_s_masks = pack_to_batch(s_points, s_lengths, fill_value=inf)  # (B, N', 3)

    # knn
    batch_knn_distances, batch_knn_indices = keops_knn(batch_q_points, batch_s_points, neighbor_limit)  # (B, M', K)

    # accumulate index
    batch_start_index = torch.cumsum(s_lengths, dim=0) - s_lengths
    batch_knn_indices += batch_start_index.view(-1, 1, 1)
    if shadow:
        batch_knn_masks = torch.gt(batch_knn_distances, radius)
        batch_knn_indices.masked_fill_(batch_knn_masks, s_points.shape[0])  # (B, M', K)
    
    # batch to pack
    knn_indices, _ = batch_to_pack(batch_knn_indices, batch_q_masks)  # (M, K)
    return knn_indices

@torch.no_grad()
def radius_search_list_mode(q_points, s_points, q_lengths, s_lengths, radius, neighbor_limit, shadow=False):
    """
    Radius search in pack mode (fast version). This function is actually a knn search 
    but with option to shadow furthest neighbors (d > radius).
    Args:
        q_points (Tensor): query points (M, 3).
        s_points (Tensor): support points (N, 3).
        q_lengths (LongTensor): the numbers of query points in the batch (B,).
        s_lengths (LongTensor): the numbers of support points in the batch (B,).
        radius (float): search radius, only used for shadowing furthest neighbors.
        neighbor_limit (int): max number of neighbors, actual knn limit used for computing neighbors.
        inf (float=1e10): infinity value.
    Returns:
        neighbor_indices (LongTensor): the indices of the neighbors. Equal to N if not exist.
    """

    # pack to batch
    batch_q_list = pack_to_list(q_points, q_lengths)  # (B)(?, 3)
    batch_s_list = pack_to_list(s_points, s_lengths)  # (B)(?, 3)

    # knn on each element of the list (B)[(?, K), (?, K)]
    knn_dists_inds = [keops_knn(b_q_pts, b_s_pts, neighbor_limit)
                          for b_q_pts, b_s_pts in zip(batch_q_list, batch_s_list)]

    # Accumualte indices
    b_start_ind = torch.cumsum(s_lengths, dim=0) - s_lengths
    knn_inds_list = [b_knn_inds + b_start_ind[i] for i, (_, b_knn_inds) in enumerate(knn_dists_inds)]

    # Convert list to pack (B)[(?, K) -> (M, K)
    knn_indices, _ = list_to_pack(knn_inds_list)

    # Apply shadow inds (optional because knn to far away from convolution kernel will be ignored anyway)
    if shadow:
        knn_dists_list = [b_knn_dists for b_knn_dists, _ in knn_dists_inds]
        knn_dists, _ = list_to_pack(knn_dists_list)
        knn_masks = torch.gt(knn_dists, radius)
        knn_indices.masked_fill_(knn_masks, s_points.shape[0])

    return knn_indices


@torch.no_grad()
def pyramid_neighbor_stats(points: Tensor,
                           num_layers: int,
                           sub_size: float,
                           search_radius: float,
                           sub_mode: str = 'grid'):
    """
    Function used for neighbors calibration. Return the average number of neigbors at each layer.
    Args:
        points (Tensor): initial layer points (M, 3).
        num_layers (int): number of layers.
        sub_size (float): initial subsampling size
        radius (float): search radius.
        sub_mode (str): the subsampling method ('grid', 'ph', 'fps').
    Returns:
        counts_list (List[Tensor]): All neigbors counts at each layers
    """

    counts_list = []
    lengths = [points.shape[0]]
    for i in range(num_layers):
        if i > 0:
            points, lengths = subsample_list_mode(points, lengths, sub_size, method=sub_mode)
        counts = keops_radius_count(points, points, search_radius)
        # neighbors = radius_search_pack_mode(points, points, lengths, lengths, search_radius, neighbor_limits[i])
        counts_list.append(counts)
        sub_size *= 2
        search_radius *= 2
    return counts_list