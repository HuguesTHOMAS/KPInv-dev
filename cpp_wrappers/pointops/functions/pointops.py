from typing import Tuple

import torch
from torch.autograd import Function
import torch.nn as nn

import pointops_cuda


class FurthestSampling(Function):
    @staticmethod
    def forward(ctx, xyz, offset, new_offset):
        """
        input: xyz: (n, 3), offset: (b), new_offset: (b)
        output: idx: (m)
        """
        assert xyz.is_contiguous()
        n, b, n_max = xyz.shape[0], offset.shape[0], offset[0]
        for i in range(1, b):
            n_max = max(offset[i] - offset[i-1], n_max)
        idx = torch.cuda.IntTensor(new_offset[b-1].item()).zero_()
        tmp = torch.cuda.FloatTensor(n).fill_(1e10)
        pointops_cuda.furthestsampling_cuda(b, n_max, xyz, offset, new_offset, tmp, idx)
        del tmp
        return idx

furthestsampling = FurthestSampling.apply
