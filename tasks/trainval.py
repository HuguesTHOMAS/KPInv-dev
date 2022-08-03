
# ----------------------------------------------------------------------------------------------------------------------
#
#           Imports and global variables
#       \**********************************/
#


# Basic libs
import torch
import numpy as np
import pickle
from os import makedirs, remove
from os.path import exists, join
import time

# PLY reader
from utils.ply import read_ply, write_ply

# Metrics
from utils.metrics import IoU_from_confusions, fast_confusion
from utils.config import Config
from sklearn.neighbors import KDTree

from models.blocks import KPConv
from models.architectures import rot_loss


# ----------------------------------------------------------------------------------------------------------------------
#
#           Trainer Class
#       \*******************/
#
