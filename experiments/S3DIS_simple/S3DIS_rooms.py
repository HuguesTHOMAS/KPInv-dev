
# ----------------------------------------------------------------------------------------------------------------------
#
#           Script Intro
#       \******************/
#
#
#   Use this script to define the dataset specific configuration. YOu should be able to adapt this file for other dataset 
#   that share the same file structure as S3DIR.
#
#

# ----------------------------------------------------------------------------------------------------------------------
#
#           Imports and global variables
#       \**********************************/
#

import time
import numpy as np
from os import listdir, makedirs
from os.path import join, exists

from utils.config import init_cfg
from datasets.scene_seg import SceneSegDataset
from utils.ply import read_ply, write_ply



# ----------------------------------------------------------------------------------------------------------------------
#
#           Config Class
#       \******************/
#


def S3DIR_cfg(cfg):

    # cfg = init_cfg()
        
    # Dataset path
    cfg.data.name = 'S3DIR'
    cfg.data.path = '../Data/S3DIS_rooms'
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


class S3DIRDataset(SceneSegDataset):

    def __init__(self, cfg, chosen_set='training', precompute_pyramid=False, load_data=True):
        """
        Class to handle S3DIR dataset.
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
        # S3DIR data
        ############

        # Here provide the list of .ply files depending on the set (training/validation/test)
        self.scene_names, self.scene_files = self.S3DIR_files()

        # Stop data is not needed
        if not load_data:
            return
        
        # Properties of input files
        self.label_property = 'class'
        self.f_properties = ['red', 'green', 'blue']

        # Start loading (merge when testing)
        self.load_scenes_in_memory(label_property=self.label_property,
                                   f_properties=self.f_properties,
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


    def S3DIR_files(self):
        """
        Function returning a list of file path. One for each scene in the dataset.
        """
    
        # Path where files can be found
        npy_path = join(self.path, 's3disfull', 'raw')

        # Get room names
        scene_names = np.sort([sc[:-4] for sc in listdir(npy_path)])
        scene_files = [join(npy_path, f + '.npy') for f in scene_names]

        # Get scene indices
        merge_inds = np.array([int(f.split('_')[1]) - 1 for f in scene_names])

        # Only get a specific split
        if self.set == 'training':
            split_inds = [0, 1, 2, 3, 5]
        elif self.set in ['validation', 'test']:
            split_inds = [4]

        scene_names = np.array([f for i, f in enumerate(scene_names) if merge_inds[i] in split_inds])
        scene_files = np.array([f for i, f in enumerate(scene_files) if merge_inds[i] in split_inds])
        merge_inds = np.array([f for f in merge_inds if f in split_inds])

        # When do we merge?
        merge = self.set in ['validation', 'test'] and False

        # In case of merge, change the files
        self.room_lists = None

        if merge:

            # Define names of merged scenes 
            all_merge_names = np.array(['Area_1', 'Area_2', 'Area_3', 'Area_4', 'Area_5', 'Area_6'])
            merge_names = [f for i, f in enumerate(all_merge_names) if i in split_inds]

            # The .merge extension triggers a specific loading function for S3DIS full scenes
            merge_files = [join(npy_path, f + '.merge') for f in merge_names]

            # Get the room list for each merged file
            merge_masks = [all_merge_names[merge_inds] == merge_n for merge_n in merge_names]
            self.room_lists = {merge_f: scene_files[merge_masks[mi]] for mi, merge_f in enumerate(merge_files)}

            # Update list of files and names
            scene_names = merge_names
            scene_files = merge_files

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

    def load_scene_file(self, file_path):

        if file_path.endswith('.ply'):
            
            data = read_ply(file_path)
            points = np.vstack((data['x'], data['y'], data['z'])).T
            if self.label_property in [p for p, _ in data.dtype.fields.items()]:
                labels = data[self.label_property].astype(np.int32)
            else:
                labels = None
            features = np.vstack([data[f_prop].astype(np.float32) for f_prop in self.f_properties]).T

        elif file_path.endswith('.npy'):

            cdata = np.load(file_path)
            
            points = cdata[:,0:3].astype(np.float32)
            features = cdata[:, 3:6].astype(np.float32)
            labels = cdata[:, 6:7].astype(np.int32)

        elif file_path.endswith('.merge'): # loads all the files that share a same root

            # Merge data
            all_points = []
            all_features = []
            all_labels = []
            for room_file in self.room_lists[file_path]:
                points, features, labels = self.load_scene_file(room_file)
                all_points.append(points)
                all_features.append(features)
                all_labels.append(labels)
            points = np.concatenate(all_points, axis=0)
            features = np.concatenate(all_features, axis=0)
            labels = np.concatenate(all_labels, axis=0)

        return points, features, np.squeeze(labels)
