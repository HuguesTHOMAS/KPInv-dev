
import torch
from torch import Tensor
from typing import Tuple, List
from easydict import EasyDict

from utils.batch_conversion import batch_to_pack, pack_to_batch, pack_to_list, list_to_pack
from utils.gpu_subsampling import subsample_pack_batch

from utils.gpu_neigbors import radius_search_pack_mode, keops_radius_count
from utils.cpp_funcs import batch_radius_neighbors


# ----------------------------------------------------------------------------------------------------------------------
#
#           Implementation of k-nn search in Keops
#       \********************************************/
#




@torch.no_grad()
def build_base_pyramid(points: Tensor,
                        lengths: Tensor):
    """
    Only build the base of the graph pyramid, consisting of:
        > The subampled points for the first layer, in pack mode.
        > The lengths of the pack at first layer.
    """

    # Results lists
    pyramid = EasyDict()
    pyramid.points = []
    pyramid.lengths = []
    pyramid.neighbors = []
    pyramid.pools = []
    pyramid.upsamples = []

    pyramid.points.append(points)
    pyramid.lengths.append(lengths)

    return pyramid


def fill_pyramid(pyramid: EasyDict,
                 num_layers: int,
                 sub_size: float,
                 search_radius: float,
                 neighbor_limits: List[int],
                 sub_mode: str = 'grid'):
    """
    Fill the graph pyramid, with:
        > The subampled points for each layer, in pack mode.
        > The lengths of the pack for each layer.
        > The neigbors indices for convolutions.
        > The pooling indices (neighbors from one layer to another).
        > The upsampling indices (opposite of pooling indices).
    """

    
    # Check if pyramid is already full
    if len(pyramid.neighbors) > 0:
        raise ValueError('Trying to fill a pyramid that already have neighbors')
    if len(pyramid.pools) > 0:
        raise ValueError('Trying to fill a pyramid that already have pools')
    if len(pyramid.upsamples) > 0:
        raise ValueError('Trying to fill a pyramid that already have upsamples')
    if len(pyramid.points) < 1:
        raise ValueError('Trying to fill a pyramid that does not have first points')
    if len(pyramid.lengths) < 1:
        raise ValueError('Trying to fill a pyramid that does not have first lengths')
    if len(pyramid.points) > 1:
        raise ValueError('Trying to fill a pyramid that already have more than one points')
    if len(pyramid.lengths) > 1:
        raise ValueError('Trying to fill a pyramid that already have more than one lengths')

    # Choose neighbor function depending on device
    if 'cuda' in pyramid.points[0].device.type:
        neighb_func = radius_search_pack_mode
    else:
        neighb_func = batch_radius_neighbors

    # Subsample all point clouds on GPU
    for i in range(num_layers):
        if i > 0:
            sub_points, sub_lengths = subsample_pack_batch(pyramid.points[0], pyramid.lengths[0], sub_size, method=sub_mode)
            pyramid.points.append(sub_points)
            pyramid.lengths.append(sub_lengths)
        sub_size *= 2.0

    # Find all neighbors
    for i in range(num_layers):

        # Get current points
        cur_points = pyramid.points[i]
        cur_lengths = pyramid.lengths[i]

        # Get convolution indices
        neighbors = neighb_func(cur_points, cur_points, cur_lengths, cur_lengths, search_radius, neighbor_limits[i])
        pyramid.neighbors.append(neighbors)

        # Relation with next layer 
        if i < num_layers - 1:
            sub_points = pyramid.points[i + 1]
            sub_lengths = pyramid.lengths[i + 1]

            # Get pooling indices
            subsampling_inds = neighb_func(sub_points, cur_points, sub_lengths, cur_lengths, search_radius, neighbor_limits[i])
            pyramid.pools.append(subsampling_inds)

            upsampling_inds = neighb_func(cur_points, sub_points, cur_lengths, sub_lengths, search_radius * 2, 1)
            pyramid.upsamples.append(upsampling_inds)

        # Increase radius for next layer
        search_radius *= 2

    # mean_dt = 1000 * (np.array(t[1:]) - np.array(t[:-1]))
    # message = ' ' * 2
    # for dt in mean_dt:
    #     message += ' {:5.1f}'.format(dt)
    # print(message)

    return


@torch.no_grad()
def build_full_pyramid(points: Tensor,
                        lengths: Tensor,
                        num_layers: int,
                        sub_size: float,
                        search_radius: float,
                        neighbor_limits: List[int],
                        sub_mode: str = 'grid'):
    """
    Build the graph pyramid, consisting of:
        > The subampled points for each layer, in pack mode.
        > The lengths of the pack.
        > The neigbors indices for convolutions.
        > The pooling indices (neighbors from one layer to another).
        > The upsampling indices (opposite of pooling indices).
    """

    pyramid = build_base_pyramid(points, lengths)

    fill_pyramid(pyramid,
                 num_layers,
                 sub_size,
                 search_radius,
                 neighbor_limits,
                 sub_mode)

    return pyramid


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
            points, lengths = subsample_pack_batch(points, lengths, sub_size, method=sub_mode)
        counts = keops_radius_count(points, points, search_radius)
        # neighbors = radius_search_pack_mode(points, points, lengths, lengths, search_radius, neighbor_limits[i])
        counts_list.append(counts)
        sub_size *= 2
        search_radius *= 2
    return counts_list