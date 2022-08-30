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


# Compile cuda ops from PointTransformers https://github.com/POSTECH-CVLab/point-transformer
echo "[PtTr INFO] Installing Point Transformers cuda operations..."
cd pointops
python3 setup.py build_ext --inplace
# python3 setup.py install
cd ..
echo "[PtTr INFO] Done !"

# Compile cuda ops from PointNext/Openpoints https://github.com/guochengqian/openpoints
echo "[PtNxt INFO] Installing PointNext/Openpoints cuda operations..."
cd pointnet2_batch
# python3 setup.py install
python3 setup.py build_ext --inplace
cd ..
echo "[PtNxt INFO] Done !"