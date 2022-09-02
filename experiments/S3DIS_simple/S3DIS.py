
# ----------------------------------------------------------------------------------------------------------------------
#
#           Script Intro
#       \******************/
#
#
#   Use this script to define the dataset specific configuration. YOu should be able to adapt this file for other dataset 
#   that share the same file structure as S3DIS.
#
#

# ----------------------------------------------------------------------------------------------------------------------
#
#           Imports and global variables
#       \**********************************/
#

import time
import numpy as np
from os.path import join

from utils.config import init_cfg
from datasets.scene_seg import SceneSegDataset



# ----------------------------------------------------------------------------------------------------------------------
#
#           Config Class
#       \******************/
#


def S3DIS_cfg(cfg):

    # cfg = init_cfg()
        
    # Dataset path
    cfg.data.name = 'S3DIS'
    cfg.data.path = '../Data/S3DIS'
    cfg.data.task = 'cloud_segmentation'

    # Dataset dimension
    cfg.data.dim = 3

    # Dict from labels to names
    cfg.data.label_and_names = [(0, 'ceiling'),
                                (1, 'floor'),
                                (2, 'wall'),
                                (3, 'beam'),
                                (4, 'column'),
                                (5, 'window'),
                                (6, 'door'),
                                (7, 'chair'),
                                (8, 'table'),
                                (9, 'bookcase'),
                                (10, 'sofa'),
                                (11, 'board'),
                                (12, 'clutter')]

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


class S3DISDataset(SceneSegDataset):

    def __init__(self, cfg, chosen_set='training', precompute_pyramid=False, load_data=True):
        """
        Class to handle S3DIS dataset.
        Simple implementation.
            > Input only consist of the first cloud with features
            > Neigborhood and subsamplings are computed on the fly in the network
            > Sampling is done simply with random picking (X spheres per class)
        """
        SceneSegDataset.__init__(self,
                                 cfg,
                                 chosen_set=chosen_set,
                                 precompute_pyramid=precompute_pyramid)

        ############
        # S3DIS data
        ############

        # Here provide the list of .ply files depending on the set (training/validation/test)
        self.scene_names, self.scene_files = self.S3DIS_files()

        # Stop data is not needed
        if not load_data:
            return
        
        # Start loading
        self.load_scenes_in_memory(label_property='class',
                                   f_properties=['red', 'green', 'blue'],
                                   f_scales=[1/255, 1/255, 1/255])

        ###########################
        # Sampling data preparation
        ###########################

        if self.data_sampler == 'regular':
            # In case regular sampling, generate the first sampling points
            self.new_reg_sampling_pts()

        else:
            # To pick points randomly per class, we need every point index from each class
            self.prepare_label_inds()

        return


    def S3DIS_files(self):
        """
        Function returning a list of file path. One for each scene in the dataset.
        """
    
        # Path where files can be found
        ply_path = join(self.path, 'original_ply')

        # Scene names
        scene_names = ['Area_1', 'Area_2', 'Area_3', 'Area_4', 'Area_5', 'Area_6']

        # Scene files
        scene_files = [join(ply_path, f + '.ply') for f in scene_names]

        # Only get a specific split
        all_splits = [0, 1, 2, 3, 4, 5]
        val_split = 4

        if self.set == 'training':
            scene_names = [f for i, f in enumerate(scene_names) if all_splits[i] != val_split]
            scene_files = [f for i, f in enumerate(scene_files) if all_splits[i] != val_split]

        elif self.set in ['validation', 'test']:
            scene_names = [f for i, f in enumerate(scene_names) if all_splits[i] == val_split]
            scene_files = [f for i, f in enumerate(scene_files) if all_splits[i] == val_split]

        return scene_names, scene_files


    def select_features(self, in_features):

        # Input features
        selected_features = np.ones_like(in_features[:, :1], dtype=np.float32)
        if self.cfg.model.input_channels == 1:
            pass
        elif self.cfg.model.input_channels == 4:
            selected_features = np.hstack((selected_features, in_features[:, :3]))
        elif self.cfg.model.input_channels == 5:
            selected_features = np.hstack((selected_features, in_features))
        else:
            raise ValueError('Only accepted input dimensions are 1, 4 and 5')

        return selected_features

