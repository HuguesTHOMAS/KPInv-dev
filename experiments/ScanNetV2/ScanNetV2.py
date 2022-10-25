
# ----------------------------------------------------------------------------------------------------------------------
#
#           Script Intro
#       \******************/
#
#
#   Use this script to define the dataset specific configuration. YOu should be able to adapt this file for other dataset 
#   that share the same file structure as ScanNetV2.
#
#

# ----------------------------------------------------------------------------------------------------------------------
#
#           Imports and global variables
#       \**********************************/
#

import time
import numpy as np
from os.path import join, exists
from os import listdir, makedirs

from utils.config import init_cfg
from datasets.scene_seg import SceneSegDataset
from utils.ply import read_ply, write_ply



# ----------------------------------------------------------------------------------------------------------------------
#
#           Config Class
#       \******************/
#


def ScanNetV2_cfg(cfg):

    # cfg = init_cfg()
        
    # Dataset path
    cfg.data.name = 'ScanNetV2'
    cfg.data.path = '../Data/ScanNetV2'
    cfg.data.task = 'cloud_segmentation'

    # Dataset dimension
    cfg.data.dim = 3

    # Dict from labels to names
    cfg.data.label_and_names = [(0, 'unclassified'),
                                (1, 'wall'),
                                (2, 'floor'),
                                (3, 'cabinet'),
                                (4, 'bed'),
                                (5, 'chair'),
                                (6, 'sofa'),
                                (7, 'table'),
                                (8, 'door'),
                                (9, 'window'),
                                (10, 'bookshelf'),
                                (11, 'picture'),
                                (12, 'counter'),
                                (14, 'desk'),
                                (16, 'curtain'),
                                (24, 'refrigerator'),
                                (28, 'shower curtain'),
                                (33, 'toilet'),
                                (34, 'sink'),
                                (36, 'bathtub'),
                                (39, 'otherfurniture')]


    # Initialize all label parameters given the label_and_names list
    cfg.data.num_classes = len(cfg.data.label_and_names)
    cfg.data.label_values = [k for k, v in cfg.data.label_and_names]
    cfg.data.label_names = [v for k, v in cfg.data.label_and_names]
    cfg.data.name_to_label = {v: k for k, v in cfg.data.label_and_names}
    cfg.data.name_to_idx = {v: i for i, v in enumerate(cfg.data.label_names)}

    # Ignored labels
    cfg.data.ignored_labels = [0]
    cfg.data.pred_values = [k for k in cfg.data.label_values if k not in cfg.data.ignored_labels]

    return cfg


# ----------------------------------------------------------------------------------------------------------------------
#
#           Dataset class definition
#       \******************************/
#


class ScanNetV2Dataset(SceneSegDataset):

    def __init__(self, cfg, chosen_set='training', precompute_pyramid=False, load_data=True):
        """
        Class to handle ScanNetV2 dataset.
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
        # ScanNetV2 data
        ############

        # Here provide the list of .ply files depending on the set (training/validation/test)
        self.scene_names, self.scene_files = self.ScanNetV2_files()

        # Stop data is not needed
        if not load_data:
            return
        
        # Start loading
        self.load_scenes_in_memory(label_property='label',
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


    def ScanNetV2_files(self):
        """
        Function returning a list of file path. One for each scene in the dataset.
        """

        #####################################
        # First preprocessign of the raw data
        #####################################

        # Path for raw data abd new light data
        raw_path = join(self.path, 'scans')
        new_path = join(self.path, 'light_ply')

        # Skip if already done
        perform_preprocess = False
        train_scenes = np.sort(np.loadtxt(join(self.path, 'scannetv2_train.txt'), dtype=str))
        val_scenes = np.sort(np.loadtxt(join(self.path, 'scannetv2_val.txt'), dtype=str))
        test_scenes = np.sort(np.loadtxt(join(self.path, 'scannetv2_test.txt'), dtype=str))
        all_scenes = np.hstack((train_scenes, val_scenes, test_scenes))
        for sc in all_scenes:
            new_ply_name = sc + '_light.ply'
            if not exists(join(new_path, new_ply_name)):
                perform_preprocess = True

        if perform_preprocess:

            # Verification of download issues:
            scenes = np.sort([f for f in listdir(raw_path)])
            print('Problems:')
            for sc in scenes:
                files = np.sort([f for f in listdir(join(raw_path, sc))])
                if  len(files) > 2:
                    print(sc, len(files))
            print('OK')

            # Save as single lighter ply format
            if not exists(new_path):
                makedirs(new_path)

            for sc in scenes:
                
                new_ply_name = sc + '_light.ply'
                if exists(join(new_path, new_ply_name)):
                    continue

                files = np.sort([f for f in listdir(join(raw_path, sc))])
                plyfiles = [f for f in files if f.endswith('.ply')]

                if sc in test_scenes and len(plyfiles) > 0:
                    data1, _ = read_ply(join(raw_path, sc, plyfiles[0]), triangular_mesh=True)
                    points = np.vstack((data1['x'], data1['y'], data1['z'])).T  
                    colors = np.vstack((data1['red'], data1['green'], data1['blue'])).T
                    write_ply(join(new_path, new_ply_name),
                            [points, colors],
                            ['x', 'y', 'z', 'red', 'green', 'blue'])

                elif len(plyfiles) > 1:
                    data1, _ = read_ply(join(raw_path, sc, plyfiles[0]), triangular_mesh=True)
                    points = np.vstack((data1['x'], data1['y'], data1['z'])).T  
                    labels = data1['label']
                    data2, _ = read_ply(join(raw_path, sc, plyfiles[1]), triangular_mesh=True)
                    colors = np.vstack((data2['red'], data2['green'], data2['blue'])).T
                    write_ply(join(new_path, new_ply_name),
                            [points, colors, labels],
                            ['x', 'y', 'z', 'red', 'green', 'blue', 'label'])


        # ######
        # # TEMP
        # ######
        
        # print()
        # print('Ply')
        # x = []
        # y = []
        # for sc in scenes:
        #     files = np.sort([f for f in listdir(join(raw_path, sc))])
        #     plyfiles = [f for f in files if f.endswith('.ply')]

        #     if len(plyfiles) > 1:
                
        #         data1, _ = read_ply(join(raw_path, sc, plyfiles[0]), triangular_mesh=True)
        #         points = np.vstack((data1['x'], data1['y'], data1['z'])).T  
        #         labels = data1['label']

        #         data2, _ = read_ply(join(raw_path, sc, plyfiles[1]), triangular_mesh=True)
        #         colors = np.vstack((data2['red'], data2['green'], data2['blue'])).T

        #         print(points.shape[0])
        #         x.append(points.shape[0])

        #         import torch
        #         from utils.gpu_subsampling import subsample_cloud
        #         torch_points = torch.from_numpy(points)
        #         sub_points = subsample_cloud(torch_points,
        #                                      0.04,
        #                                      method='grid',
        #                                      return_inverse=False)
        #         y.append(sub_points.shape[0])

        # import matplotlib.pyplot as plt
        # plt.figure()    
        # plt.hist(x, bins=30)
        # plt.ylabel('Count')
        # plt.xlabel('Data')
        # plt.figure()    
        # plt.hist(y, bins=30)
        # plt.ylabel('Count')
        # plt.xlabel('Data')
        # plt.show()

        # a = 1/0


        ####################
        # Now define dataset 
        ####################

        # Get train/val/test split
        if self.set == 'training':
            scene_names = train_scenes
        elif self.set == 'val':
            scene_names = val_scenes
        elif self.set == 'test':
            scene_names = test_scenes

        scene_files = [join(new_path, sc + '_light.ply') for sc in scene_names]

        # Scenes
        print()
        print('Scenes')
        for n, f in zip(scene_names, scene_files):
            print(n, f, exists(f))
        
        a = 1/0

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

