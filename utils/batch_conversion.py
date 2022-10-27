from typing import List, Optional, Tuple

import torch
from torch import Tensor

@torch.no_grad()
def _get_indices_from_lengths(lengths: Tensor, max_length: int) -> Tensor:
    """Compute the indices in flattened batch tensor from the lengths in pack mode."""
    length_list = lengths.detach().cpu().numpy().tolist()
    chunks = [(i * max_length, i * max_length + length) for i, length in enumerate(length_list)]
    indices = torch.cat([torch.arange(i1, i2) for i1, i2 in chunks], dim=0).to(lengths.device)
    return indices

@torch.no_grad()
def _get_slices_from_lengths(lengths: Tensor) -> Tensor:
    """Compute the slices indices from lengths."""
    batch_end_index = torch.cumsum(lengths, dim=0)
    batch_start_index = batch_end_index - lengths
    batch_slices_index = torch.stack((batch_start_index, batch_end_index), dim=1)
    return batch_slices_index


def batch_to_pack(batch_tensor: Tensor, masks: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
    """Convert Tensor from batch mode to stack mode with masks.
    Args:
        batch_tensor (Tensor): the input tensor in batch mode (B, N, C) or (B, N).
        masks (BoolTensor): the masks of items of each sample in the batch (B, N).
    Returns:
        A Tensor in pack mode in the shape of (M, C) or (M).
        A LongTensor of the length of each sample in the batch in the shape of (B).
    """
    if masks is not None:
        pack_tensor = batch_tensor[masks]
        lengths = masks.sum(dim=1)
    else:
        lengths = torch.full(size=(batch_tensor.shape[0],), fill_value=batch_tensor.shape[1], dtype=torch.long).to(batch_tensor.device)
        pack_tensor = batch_tensor
    return pack_tensor, lengths


def pack_to_batch(pack_tensor: Tensor, lengths: Tensor, max_length=None, fill_value=0.0) -> Tuple[Tensor, Tensor]:
    """Convert Tensor from pack mode to batch mode.
    Args:
        pack_tensor (Tensor): The input tensors in pack mode (M, C).
        lengths (LongTensor): The number of items of each sample in the batch (B)
        max_length (int, optional): The maximal length of each sample in the batch.
        fill_value (float or int or bool): The default value in the empty regions. Default: 0.
    Returns:
        A Tensor in stack mode in the shape of (B, N, C), where N is max(lengths).
        A BoolTensor of the masks of each sample in the batch in the shape of (B, N).
    """

    # Get batch size and maximum length of elements
    batch_size = lengths.shape[0]
    if max_length is None:
        max_length = lengths.max().item()

    # Get the batch index for each packed point 
    tgt_indices = _get_indices_from_lengths(lengths, max_length)

    # Create batch tensor (B*N, C)
    num_channels = pack_tensor.shape[1]
    batch_tensor = pack_tensor.new_full(size=(batch_size * max_length, num_channels), fill_value=fill_value)
    batch_tensor[tgt_indices] = pack_tensor

    # Reshape batch tensor (B, N, C)
    batch_tensor = batch_tensor.view(batch_size, max_length, num_channels)

    # Get valid points masks
    masks = torch.zeros(size=(batch_size * max_length,), dtype=torch.bool).to(pack_tensor.device)
    masks[tgt_indices] = True
    masks = masks.view(batch_size, max_length)

    return batch_tensor, masks


def pack_to_list(pack_tensor: Tensor, lengths: Tensor) -> List[Tensor]:
    """Convert Tensor from pack mode to list mode.
    Args:
        pack_tensor (Tensor): The input tensors in pack mode (M, C).
        lengths (LongTensor): The number of items of each sample in the batch (B)
    Returns:
        A List of Tensors of length B, where each element has a shape (?, C).
    """

    # Get slices indices
    b_slices = _get_slices_from_lengths(lengths)
    
    # Slice packed tensor in a list of tensors
    batch_tensor_list = [pack_tensor[b_slice[0]:b_slice[1]] for b_slice in b_slices]

    return batch_tensor_list


def list_to_pack(tensor_list: List[Tensor]) -> Tuple[Tensor, Tensor]:
    """Convert Tensor from list mode to stack mode.
    Args:
        tensor_list (Tensor): the input tensors in a list.
    Returns:
        A Tensor in pack mode in the shape of (M, C) or (M).
        A LongTensor of the length of each sample in the batch in the shape of (B).
    """
    pack_tensor = torch.cat(tensor_list, dim=0)
    lengths = torch.LongTensor([tens.shape[0] for tens in tensor_list]).to(pack_tensor.device)
    return pack_tensor, lengths