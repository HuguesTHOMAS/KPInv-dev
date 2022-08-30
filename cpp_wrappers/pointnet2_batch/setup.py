from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='pointnet2_cuda',
    ext_modules=[
        CUDAExtension('pointnet2_batch_cuda', [
        'src/pointnet2_api.cpp',
        'src/sampling.cpp',
        'src/sampling_gpu.cu',
        ],
        extra_compile_args={'cxx': ['-g'],
                            'nvcc': ['-O2']})
    ],
    cmdclass={'build_ext': BuildExtension}
)

