/*
batch version of point sampling and gathering, modified from the original implementation of official PointNet++ codes.
Written by Shaoshuai Shi
All Rights Reserved 2018.
*/


#include <torch/serialize/tensor.h>
#include <ATen/cuda/CUDAContext.h>
#include <vector>
#include "sampling_gpu.h"


int furthest_point_sampling_wrapper(int b, int n, int m, float min_d,
    at::Tensor points_tensor, at::Tensor temp_tensor, at::Tensor idx_tensor) {

    const float *points = points_tensor.data<float>();
    float *temp = temp_tensor.data<float>();
    int *idx = idx_tensor.data<int>();

    // Fill idx with -1
    for (int j = 0; j < m; j++)
        idx[j] = -1;

    furthest_point_sampling_kernel_launcher(b, n, m, min_d * min_d, points, temp, idx);
    return 1;
}
