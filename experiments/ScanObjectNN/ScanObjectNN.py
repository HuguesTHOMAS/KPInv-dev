
# ----------------------------------------------------------------------------------------------------------------------
#
#           Script Intro
#       \******************/
#
#
#   Use this script to define the dataset specific configuration. YOu should be able to adapt this file for other dataset 
#   that share the same file structure as ScanObjectNN.
#
#

# ----------------------------------------------------------------------------------------------------------------------
#
#           Imports and global variables
#       \**********************************/
#

import time
import numpy as np
from os.path import join, isfile
import h5py
import pickle

from utils.config import init_cfg
from datasets.scene_seg import SceneSegDataset
from datasets.object_classification import ObjClassifDataset

from utils.ply import write_ply



# ----------------------------------------------------------------------------------------------------------------------
#
#           Config Class
#       \******************/
#


def ScanObjectNN_cfg(cfg):

    # cfg = init_cfg()
        
    # Dataset path
    cfg.data.name = 'ScanObjectNN'
    cfg.data.path = '../Data/ScanObjectNN/h5_files/main_split'
    cfg.data.task = 'classification'

    # Dataset dimension
    cfg.data.dim = 3

    # Dict from labels to names
    cfg.data.label_and_names = [(0, 'bag'),
                                (1, 'bin'),
                                (2, 'box'),
                                (3, 'cabinet'),
                                (4, 'chair'),
                                (5, 'desk'),
                                (6, 'display'),
                                (7, 'door'),
                                (8, 'shelf'),
                                (9, 'table'),
                                (10, 'bed'),
                                (11, 'pillow'),
                                (12, 'sink'),
                                (13, 'sofa'),
                                (14, 'toilet')]
      
      
      

    # Initialize all label parameters given the label_and_names list
    cfg.data.num_classes = len(cfg.data.label_and_names)
    cfg.data.label_values = [k for k, v in cfg.data.label_and_names]
    cfg.data.label_names = [v for k, v in cfg.data.label_and_names]
    cfg.data.name_to_label = {v: k for k, v in cfg.data.label_and_names}
    cfg.data.name_to_idx = {v: i for i, v in enumerate(cfg.data.label_names)}

    # Ignored labels
    cfg.data.ignored_labels = []
    cfg.data.pred_values = [k for k in cfg.data.label_values if k not in cfg.data.ignored_labels]

    return cfg


# ----------------------------------------------------------------------------------------------------------------------
#
#           Dataset class definition
#       \******************************/
#


class ScanObjectNNDataset(ObjClassifDataset):

    def __init__(self, cfg, chosen_set='training', precompute_pyramid=False, load_data=True, uniform_sample=True):
        """
        Class to handle ScanObjectNN dataset.
        Simple implementation.
            > Input only consist of the first cloud with features
            > Neigborhood and subsamplings are computed on the fly in the network
            > Sampling is done simply with random picking (X spheres per class)
        """
        ObjClassifDataset.__init__(self,
                                 cfg,
                                 chosen_set=chosen_set,
                                 precompute_pyramid=precompute_pyramid)

        ###################
        # ScanObjectNN data
        ###################

        if chosen_set == 'training':
            h5_file = join(cfg.data.path, 'training_objectdataset_augmentedrot_scale75.h5')

        elif chosen_set in ['validation', 'test']:
            sampled_file = join(cfg.data.path, 'test_objectdataset_augmentedrot_scale75_1024_fps.pkl')
            h5_file = join(cfg.data.path, 'test_objectdataset_augmentedrot_scale75.h5')
        else:
            raise ValueError('chosen_set set not recognised')

        # Check file
        if not isfile(h5_file):
            raise FileExistsError(f'{h5_file} does not exist, please download dataset at first')

        # Load file (points and label)
        with h5py.File(h5_file, 'r') as f:
            self.input_points = np.array(f['data']).astype(np.float32)
            self.input_labels = np.array(f['label']).astype(int)

        # Reload sampled point for test
        if chosen_set in ['validation', 'test'] and uniform_sample:
            if not isfile(sampled_file):
                raise FileExistsError(f'{sampled_file} does not exist, please download dataset at first') 
            with open(sampled_file, 'rb') as f:
                self.input_points = pickle.load(f)

        # Height is in y coordinates so permute
        self.input_points = np.ascontiguousarray(self.input_points[:, :, [2, 0, 1]])
        
        # This dataset does not have features use point coordinates
        self.input_features = np.copy(self.input_points)

        # # Test random objects
        # idx = [0, 100, 1000, 2000, 3000, 4000, 5000, 6000, 10000]
        # for i in idx:
        #     write_ply(join(cfg.data.path, 'test{:d}.ply'.format(i)),
        #             (self.input_points[i]),
        #             ['x', 'y', 'z'])
        #     print(self.labels[i])

        # Handle number of objects and number of steps per epoch
        self.objects_per_epoch(cfg)

        return


    def select_features(self, in_features):

        # Input features
        selected_features = np.ones_like(in_features[:, :1], dtype=np.float32)
        if self.cfg.model.input_channels == 1:
            pass
        elif self.cfg.model.input_channels == 2:
            selected_features = np.hstack((selected_features, in_features[:, 2:3]))
        elif self.cfg.model.input_channels == 4:
            selected_features = np.hstack((selected_features, in_features[:, :3]))
        else:
            raise ValueError('Only accepted input dimensions are 1, 4 and 5')

        return selected_features

