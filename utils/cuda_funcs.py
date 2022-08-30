

# Cuda extensions
import cpp_wrappers.pointops.pointops_cuda as pointops_cuda
import cpp_wrappers.pointnet2_batch.pointnet2_batch_cuda as pointnet2_cuda

import torch


@torch.no_grad()
def furthest_point_sample_1(points, lengths, new_lengths=[], stride=4):
    """
    Fursthest sampling controlled by the new number of points wanted. Points in pack mode.
    Args:
        points          (Tensor): the original points of the batch in a pack mode (N, D).
        lengths     (LongTensor): the numbers of points in each cloud of the batch (B,).
        new_lengths (LongTensor): the new numbers of points wanted (B,).
        stride             (int): If new_lengths is not provided, N is divided by stride
    Returns:
        sample_inds (LongTensor): the indices of the subsampled points (M,).
    """
    
    # Verify contiguous
    assert points.is_contiguous()

    # Convert length to tensor
    l_as_list = type(lengths) == list
    if l_as_list:
        lengths = torch.IntTensor(lengths).to(points.device)
    nl_as_list = type(new_lengths) == list
    if nl_as_list:
        new_lengths = torch.IntTensor(new_lengths).to(points.device)

    # Get dimensions
    N, B, n_max = points.shape[0], lengths.shape[0], torch.max(lengths).item()

    # Create new_lengths if not provided
    if new_lengths.shape[0] < B:
        new_lengths = lengths // stride

    # Convert lengths to offsets (lib is coded like that)
    offsets = torch.cumsum(lengths, 0).int()
    new_offsets = torch.cumsum(new_lengths, 0).int()

    # Perform subsampling
    idx = torch.cuda.IntTensor(new_offsets[B-1].item()).zero_()
    tmp = torch.cuda.FloatTensor(N).fill_(1e10)
    pointops_cuda.furthestsampling_cuda(B, n_max, points, offsets, new_offsets, tmp, idx)
    del tmp

    return idx.long(), new_lengths.long()


@torch.no_grad()
def furthest_point_sample_2(points, new_n=None, stride=4):
    """
    Fursthest sampling controlled by the new number of points wanted. Single point cloud.
    Args:
        points (Tensor): the original points (N, D).
        new_n     (int): the new number of points wanted .
        stride    (int): If new_n is not provided, N is divided by stride
    Returns:
        sample_inds (LongTensor): the indices of the subsampled points (M,).
    """
    
    # Verify contiguous
    assert points.is_contiguous()

    # Get dimensions
    N = points.shape[0]
    B = 1
    points = points.view(B, N, -1)

    # Create new_lengths if not provided
    if new_n is None:
        new_n = N // stride

    # # Perform subsampling
    # idx = torch.cuda.IntTensor(new_offsets[B-1].item()).zero_()
    # tmp = torch.cuda.FloatTensor(N).fill_(1e10)
    # pointops_cuda.furthestsampling_cuda(B, n_max, points, offsets, new_offsets, tmp, idx)
    # del tmp

    idx = torch.cuda.IntTensor(B, new_n)
    temp = torch.cuda.FloatTensor(B, N).fill_(1e10)
    pointnet2_cuda.furthest_point_sampling_wrapper(B, N, new_n, points, temp, idx)
    del temp

    return idx.view(-1).long()


