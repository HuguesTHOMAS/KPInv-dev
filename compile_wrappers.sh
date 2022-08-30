#!/bin/bash

cd cpp_wrappers

# Compile cpp subsampling
cd cpp_subsampling
python3 setup.py build_ext --inplace
cd ..

# Compile cpp neighbors
cd cpp_neighbors
python3 setup.py build_ext --inplace
cd ..

export TORCH_CUDA_ARCH_LIST="6.1;6.2;7.0;7.5;8.0;8.6"   # a100: 8.0; v100: 7.0; 2080ti: 7.5; titan xp: 6.1


# Compile cuda ops from PointNext/Openpoints https://github.com/guochengqian/openpoints
echo " "
echo " "
echo "[PtNxt INFO] Installing PointNext/Openpoints cuda operations..."
cd pointnet2_batch
python3 setup.py build_ext --inplace
cd ..
echo " "
echo "[PtNxt INFO] Done !" 
echo " "
echo " "