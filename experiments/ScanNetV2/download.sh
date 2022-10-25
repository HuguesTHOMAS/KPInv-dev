#!/bin/bash

# Some useful info:
# Scan data is named by scene[spaceid]_[scanid], or scene%04d_%02d, where each space corresponds to a unique location (0-indexed).
# Script usage:
# - To download the entire ScanNet release (1.3TB): download-scannet.py -o [directory in which to download] 
# - To download a specific scan (e.g., scene0000_00): download-scannet.py -o [directory in which to download] --id scene0000_00
# - To download a specific file type (e.g., *.sens, valid file suffixes listed here): download-scannet.py -o [directory in which to download] --type .sens
# - To download the ScanNet v1 task data (inc. trained models): download-scannet.py -o [directory in which to download] --task_data


# - Train/test splits are given in the main ScanNet project repository: https://github.com/ScanNet/ScanNet/tree/master/Tasks/Benchmark

# - ScanNet200 preprocessing information: https://github.com/ScanNet/ScanNet/tree/master/BenchmarkScripts/ScanNet200; 
#   to download the label map file: download-scannet.py -o [directory in which to download] --label_map

########
# Init #
########

# Number of docker allowed to work in parrallel
# *********************************************

python3 experiments/ScanNetV2/download_scannetv2.py -o ../Data/ScanNetV2 --type _vh_clean_2.ply

