#include <torch/serialize/tensor.h>
#include <torch/extension.h>

#include "sampling/sampling_cuda_kernel.h"


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("furthestsampling_cuda", &furthestsampling_cuda, "furthestsampling_cuda");
}
