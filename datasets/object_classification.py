
# ----------------------------------------------------------------------------------------------------------------------
#
#           Script Intro
#       \******************/
#
#
#   This file contains the defintion of a point cloud segmentation dataset that can be specialize to any dataset like
#   MOdelNet40/ScanObjectNN by creating a subclass.
#
#   This dataset is used as an input pipeline for a KPNet. 
# 
#   You can choose a simple pipeline version (precompute_pyramid=False) which does not precompute neighbors in advance. 
#   Therefore the network will compute them on the fly on GPU. Overall this pipeline is simpler to use but slower than 
#   the normal pipeline.
#
#   You can choose a complex pipeline version (precompute_pyramid=False) which does precomputes the neighbors/etc. for 
#   all layers in advance. This is the fastest pipeline for training.
#
#

# ----------------------------------------------------------------------------------------------------------------------
#
#           Imports and global variables
#       \**********************************/
#

# Common libs
import time
import numpy as np
import pickle
from os import makedirs
from os.path import join, exists
import torch
from torch.utils.data import Dataset, Sampler
from sklearn.neighbors import KDTree
from easydict import EasyDict
import h5py

from torch.multiprocessing import Lock
# from torch.multiprocessing import set_start_method
# try:
#      set_start_method('spawn')
# except RuntimeError:
#     pass

import pyvista as pv


from utils.printing import frame_lines_1
from utils.ply import read_ply, write_ply
from utils.gpu_init import init_gpu
from utils.gpu_subsampling import subsample_numpy, subsample_pack_batch, subsample_cloud
from utils.gpu_neigbors import tiled_knn
from utils.torch_pyramid import build_full_pyramid, pyramid_neighbor_stats, build_base_pyramid
from utils.cpp_funcs import furthest_point_sample_cpp

from utils.transform import ComposeAugment, RandomRotate, RandomScaleFlip, RandomJitter, FloorCentering, \
    ChromaticAutoContrast, ChromaticTranslation, ChromaticJitter, HueSaturationTranslation, RandomDropColor, \
    ChromaticNormalize, HeightNormalize, RandomFullColor, UnitScaleCentering


# ----------------------------------------------------------------------------------------------------------------------
#
#           Class definition
#       \**********************/
#

class ObjClassifDataset(Dataset):

    def __init__(self, cfg, chosen_set='training', precompute_pyramid=False):
        """
        Initialize parameters of the dataset here.
        """

        # Dataset path
        self.name = cfg.data.name
        self.path = cfg.data.path
        self.task = cfg.data.task
        self.cfg = cfg

        # Training or test set
        self.set = chosen_set
        self.precompute_pyramid = precompute_pyramid

        # Parameters depending on training or test
        if self.set == 'training':
            b_cfg = cfg.train
        else:
            b_cfg = cfg.test
        self.b_n = b_cfg.batch_size
        self.b_lim = b_cfg.batch_limit
        self.data_sampler = b_cfg.data_sampler

        # Dataset dimension
        self.dim = cfg.data.dim

        # Additional label variables
        self.num_classes = cfg.data.num_classes
        self.label_values = np.array(cfg.data.label_values, dtype=np.int32)
        self.label_names = cfg.data.label_names
        self.name_to_label = cfg.data.name_to_label
        self.name_to_idx = cfg.data.name_to_idx
        self.label_to_names = {k: v for k, v in cfg.data.label_and_names}
        self.label_to_idx = {v: i for i, v in enumerate(cfg.data.label_values)}
        self.ignored_labels = np.array(cfg.data.ignored_labels, dtype=np.int32)
        self.pred_values = np.array(cfg.data.pred_values, dtype=np.int32)

        
        # Variables that will be automatically populated
        self.n_objects = None
        self.input_points = []
        self.input_features = []
        self.input_labels = []
        self.n_votes = torch.from_numpy(np.zeros((1,), dtype=np.float32))
        self.n_votes.share_memory_()
        

        # Get augmentation transform
        if self.set == 'training':
            a_cfg = cfg.augment_train
        else:
            a_cfg = cfg.augment_test

        self.base_augments = []
        self.base_augments.append(RandomScaleFlip(scale=a_cfg.scale,
                                                  anisotropic=a_cfg.anisotropic,
                                                  flip_p=a_cfg.flips))

        self.base_augments.append(UnitScaleCentering())
        self.base_augments.append(RandomRotate(mode=a_cfg.rotations))

        self.full_augments = [a for a in self.base_augments]

        # # The color augment are applied to coordz feature
        # self.full_augments = [a for a in self.base_augments]
        # if a_cfg.chromatic_contrast:
        #     self.full_augments += [ChromaticAutoContrast()]
        # if a_cfg.chromatic_all:
        #     self.full_augments += [ChromaticTranslation(),
        #                      ChromaticJitter(),
        #                      HueSaturationTranslation()]
        # # self.full_augments.append(RandomDropColor(p=a_cfg.color_drop))
        # if a_cfg.chromatic_norm:
        #     self.full_augments += [ChromaticNormalize()]
        # self.full_augments.append(RandomFullColor(p=a_cfg.color_drop))
            

        # TRAIN AUGMENT
        # Transformer: no drop color and chromatic contrast/transl/jitter/HSV
        #   PointNext: floor centering, chromatic contrast/drop/norm
        #        test: drop before or after norm? In theory it should be after norm

        # TEST AUGMENT
        # Transformer: no augment at all
        #   PointNext: floor centering and chromatic_norm (drop color if vote)

        self.augmentation_transform = ComposeAugment(self.full_augments)

        return

    def objects_per_epoch(self, cfg):
        """
        Compute number of objects per epoch and update config accordingly
        """

        # Total number of objects
        self.n_objects = len(self.input_points)

        # Number of step per epoch
        if self.set == 'training':
            epoch_step = cfg.train.steps_per_epoch
            batch_size = cfg.train.batch_size * cfg.train.accum_batch
        else:
            epoch_step = cfg.test.max_steps_per_epoch
            batch_size = cfg.test.batch_size

        # Number of objects per epoch
        if epoch_step:
            self.epoch_n = min(batch_size * epoch_step, self.n_objects)
        else:
            self.epoch_n = self.n_objects

        return

    def probs_to_preds(self, probs):
        return self.pred_values[np.argmax(probs, axis=1).astype(np.int32)]
    
    def select_features(self, in_features):
        print('ERROR: This function select_features needs to be redifined in the child dataset class. It depends on the dataset')
        return 

    def get_votes(self):
        return float(self.n_votes.item())

    def __len__(self):
        """
        Return the length of data here
        """
        return len(self.n_objects)

    def __getitem__(self, idx_list):
        """
        The main thread gives a list of indices to load a batch. Each worker is going to work in parallel to load a
        different list of indices.
        """

        ###################
        # Gather batch data
        ###################

        tp_list = []
        tf_list = []
        tl_list = []
        ti_list = []
        s_list = []
        R_list = []

        for p_i in idx_list:

            # Get points, features and labels
            in_points = self.input_points[p_i]
            in_features = self.input_features[p_i]
            label = self.label_to_idx[self.input_labels[p_i]]

            # Data augmentation
            in_points, in_features, _ = self.augmentation_transform(in_points, in_features, None)
            
            # Select features for the network
            in_features = self.select_features(in_features)

            # View the arrays as torch tensors
            torch_points = torch.from_numpy(in_points)
            torch_features = torch.from_numpy(in_features)

            # Input subsampling if asked
            if self.cfg.data.init_sub_mode == 'grid':
                torch_points, torch_features = subsample_cloud(torch_points,
                                                               self.cfg.data.init_sub_size,
                                                               features=torch_features,
                                                               labels=None,
                                                               method=self.cfg.data.init_sub_mode,
                                                               return_inverse=False)

            elif self.cfg.data.init_sub_mode == 'fps' and self.cfg.train.in_radius < 0:
                # initial fps subsampling like pointnext
                npoints = -self.cfg.train.in_radius
                if torch_points.shape[0] > npoints:  # point resampling strategy
                    point_all = npoints
                    if self.set == 'training':
                        if npoints in [1024, 1200]:
                            point_all = 1200
                        elif npoints == 4096:
                            point_all = 4800
                        elif npoints == 8192:
                            point_all = 8192
                        else:
                            raise NotImplementedError()
                            
                    sub_inds = furthest_point_sample_cpp(torch_points, new_n=point_all)
                    torch_points = torch_points[sub_inds]
                    torch_features = torch_features[sub_inds]

            # Random drop if asked      
            if self.cfg.train.in_radius < 0:
                npoints = -self.cfg.train.in_radius
                if torch_points.shape[0] > npoints:
                    selection = torch.randperm(torch_points.shape[0])[:npoints]
                    torch_points = torch_points[selection]
                    torch_features = torch_features[selection]


            # write_ply('results/test_' +str(p_i)+'.ply',
            #           [in_points],
            #           ['x', 'y', 'z'])

            # Stack batch
            tp_list += [torch_points]
            tf_list += [torch_features]
            tl_list += [label]
            ti_list += [p_i]


        ###################
        # Concatenate batch
        ###################
        
        stacked_points = torch.cat(tp_list, dim=0)
        stacked_features = torch.cat(tf_list, dim=0)
        labels = torch.LongTensor(tl_list)
        obj_inds = torch.LongTensor(ti_list)
        stack_lengths = torch.LongTensor([int(pp.shape[0]) for pp in tp_list])


        #######################
        # Create network inputs
        #######################
        #
        #   Points, features, etc.
        #

        # Get the whole input list without upsample indices that we do not need.
        if self.precompute_pyramid:
            radius0 = self.cfg.model.in_sub_size * self.cfg.model.kp_radius
            if radius0 < 0:
                radius0 = self.cfg.data.init_sub_size * self.cfg.model.kp_radius
            input_dict = build_full_pyramid(stacked_points,
                                            stack_lengths,
                                            len(self.cfg.model.layer_blocks),
                                            self.cfg.model.in_sub_size,
                                            radius0,
                                            self.cfg.model.radius_scaling,
                                            self.cfg.model.neighbor_limits,
                                            0,
                                            sub_mode=self.cfg.model.in_sub_mode,
                                            grid_pool_mode=self.cfg.model.grid_pool)
            
            # for i in range(len(input_dict.lengths[0])):
            #     print([int(lll[i]) for lll in input_dict.lengths])
            # print('----------------------------------------------------------------')

        else:
            input_dict = build_base_pyramid(stacked_points,
                                            stack_lengths)

        # Add other input to the pyramid dictionary
        input_dict.features = stacked_features
        input_dict.labels = labels
        input_dict.obj_inds = obj_inds

        return input_dict

    def calib_batch_size(self, samples=20, verbose=True):

        #############################
        # If all clouds are the same
        #############################

        # No need for calibration of all the clouds are the same length
        cloud_lengths = [in_p.shape[0] for in_p in self.input_points]
        maxp = np.max(cloud_lengths)
        minp = np.min(cloud_lengths)
        if maxp - minp < 1:
            cloud_lengths = int(maxp)
            self.b_n = cloud_lengths / self.b_lim
            return

        
        ###############################
        # Otherwise perform quick calib
        ###############################

        a = 1/0 # Need to modify this

        t0 = time.time()

        # Get gpu for faster calibration
        device = init_gpu()

        # Get augmentation transform
        calib_augment = ComposeAugment(self.base_augments)
        
        all_batch_n = []
        all_batch_n_pts = []
        for i in range(samples):

            batch_n = 0
            batch_n_pts = 0
            while True:
                cloud_ind, center_p = self.sample_input_center()
                _, in_points, _, _ = self.get_input_area(cloud_ind, center_p)
                in_points, _, _ = calib_augment(in_points, np.copy(in_points), None)
                if in_points.shape[0] > 0:
                    in_dl = self.cfg.model.in_sub_size
                    if in_dl > 0 and in_dl > self.cfg.data.init_sub_size * 1.01:
                        gpu_points = torch.from_numpy(in_points).to(device)
                        sub_points, _ = subsample_pack_batch(gpu_points,
                                                            [gpu_points.shape[0]],
                                                            self.cfg.model.in_sub_size,
                                                            method=self.cfg.model.in_sub_mode)
                        batch_n_pts += sub_points.shape[0]
                    else:
                        batch_n_pts += in_points.shape[0]
                    batch_n += 1

                    # In case batch is full, stop
                    if batch_n_pts > self.b_lim:
                        break

            all_batch_n.append(batch_n)
            all_batch_n_pts.append(batch_n_pts)
        t1 = time.time()

        if verbose:

            report_lines = ['Batch Size Calibration Report:']
            report_lines += ['******************************']
            report_lines += ['']
            report_lines += ['{:d} batches tested in {:.1f}s'.format(samples, t1 - t0)]
            report_lines += ['']
            report_lines += ['Batch limit stats:']
            report_lines += ['     batch limit = {:.3f}'.format(self.b_lim)]
            report_lines += ['avg batch points = {:.3f}'.format(np.mean(all_batch_n_pts))]
            report_lines += ['std batch points = {:.3f}'.format(np.std(all_batch_n_pts))]
            report_lines += ['']
            report_lines += ['New batch size obtained from calibration:']
            report_lines += ['  avg batch size = {:.1f}'.format(np.mean(all_batch_n))]
            report_lines += ['  std batch size = {:.2f}'.format(np.std(all_batch_n))]

            frame_lines_1(report_lines)

        self.b_n = np.mean(all_batch_n)

        return

    def calib_batch_limit(self, batch_size, samples=100, verbose=True):
        """
        Find the batch_limit given the target batch_size. 
        The batch size varies randomly so we prefer a quick calibration to find 
        an approximate batch limit.
        """

        #############################
        # If all clouds are the same
        #############################

        # No need for calibration of all the clouds are the same length
        cloud_lengths = [in_p.shape[0] for in_p in self.input_points]
        maxp = np.max(cloud_lengths)
        minp = np.min(cloud_lengths)
        if maxp - minp < 1:
            cloud_lengths = int(minp)
            new_b_lim = cloud_lengths * batch_size - 1
            return new_b_lim

        
        ###############################
        # Otherwise perform quick calib
        ###############################

        a = 1/0 # Need to modify this

        t0 = time.time()

        # Get gpu for faster calibration
        device = init_gpu()

        # Get augmentation transform
        calib_augment = ComposeAugment(self.base_augments)

        # Advanced display
        pi = 0
        pN = samples
        progress_n = 30
        fmt_str = '[{:<' + str(progress_n) + '}] {:5.1f}%'
        print('\nSearching batch_limit given the target batch_size.')

        # First get a avg of the pts per point cloud
        all_cloud_n = []
        while len(all_cloud_n) < samples:
            cloud_ind, center_p = self.sample_input_center()
            if cloud_ind is None:
                break
            _, in_points, feat, label = self.get_input_area(cloud_ind, center_p)

            if in_points.shape[0] > 0:
                in_points, feat, label = calib_augment(in_points, feat, label)
                # pl = pv.Plotter(window_size=[1600, 900])
                # pl.add_points(in_points,
                #               render_points_as_spheres=False,
                #               scalars=label,
                #               point_size=8.0)

                # pl.set_background('white')
                # pl.enable_eye_dome_lighting()
                # pl.show()
                in_dl = self.cfg.model.in_sub_size
                if in_dl > 0 and in_dl > self.cfg.data.init_sub_size * 1.01:
                    gpu_points = torch.from_numpy(in_points).to(device)
                    sub_points, _ = subsample_pack_batch(gpu_points,
                                                        [gpu_points.shape[0]],
                                                        self.cfg.model.in_sub_size,
                                                        method=self.cfg.model.in_sub_mode)
                    all_cloud_n.append(sub_points.shape[0])
                else:
                    all_cloud_n.append(in_points.shape[0])
                
            pi += 1
            print('', end='\r')
            print(fmt_str.format('#' * ((pi * progress_n) // pN), 100 * pi / pN), end='', flush=True)

        print('', end='\r')
        print(fmt_str.format('#' * progress_n, 100), end='', flush=True)
        print('\n')

        # Initial batch limit thanks to average points per batch
        mean_cloud_n = np.mean(all_cloud_n)
        new_b_lim = mean_cloud_n * batch_size - 1

        # Verify the batch size 
        all_batch_n = []
        all_batch_n_pts = []
        for i in range(samples):
            batch_n = 0
            batch_n_pts = 0
            while True:
                rand_i = np.random.choice(samples)
                batch_n_pts += all_cloud_n[rand_i]
                batch_n += 1
                if batch_n_pts > new_b_lim:
                    break
            all_batch_n.append(batch_n)
            all_batch_n_pts.append(batch_n_pts)

        t1 = time.time()

        if verbose:

            report_lines = ['Batch Limit Calibration Report:']
            report_lines += ['*******************************']
            report_lines += ['']
            report_lines += ['{:d} batches tested in {:.1f}s'.format(samples, t1 - t0)]
            report_lines += ['']
            report_lines += ['Batch limit stats:']
            report_lines += ['     batch limit = {:.3f}'.format(new_b_lim)]
            report_lines += ['avg batch points = {:.3f}'.format(np.mean(all_batch_n_pts))]
            report_lines += ['std batch points = {:.3f}'.format(np.std(all_batch_n_pts))]
            report_lines += ['']
            report_lines += ['New batch size obtained from calibration:']
            report_lines += ['  avg batch size = {:.1f}'.format(np.mean(all_batch_n))]
            report_lines += ['  std batch size = {:.2f}'.format(np.std(all_batch_n))]

            frame_lines_1(report_lines)


        return new_b_lim
   
    def calib_batch(self, cfg, update_test=True):

        ###################
        # Quick calibration
        ###################

        if self.b_lim > 0:
            # If the batch limit is already set, update the corresponding batch size
            print('\nWARNING: batch_limit is set by user and batch_size is ignored.\n')
            self.calib_batch_size()
        else:
            # If the batch limit is not set, use batch size to find it
            self.b_lim = self.calib_batch_limit(self.b_n)

        # Update configuration
        if self.set == 'training':
            cfg.train.batch_size = self.b_n
            cfg.train.batch_limit = self.b_lim
            if update_test:
                cfg.test.batch_size = self.b_n
                cfg.test.batch_limit = self.b_lim

        else:
            cfg.test.batch_size = self.b_n
            cfg.test.batch_limit = self.b_lim

        # # After calibration reset counters for regular sampling
        # self.reg_sampling_i *= 0

        print('\n')

        return

    def calib_neighbors(self, cfg, samples=100, verbose=True):

        t0 = time.time()

        # Get gpu for faster calibration
        device = init_gpu()

        # Get augmentation transform
        calib_augment = ComposeAugment(self.base_augments)
        
        # Advanced display
        pi = 0
        pN = samples
        progress_n = 30
        fmt_str = '[{:<' + str(progress_n) + '}] {:5.1f}%'
        print('\nNeighbors calibration')

        # Verify if we already have neighbor values
        overwrite = False
        num_layers = len(cfg.model.layer_blocks)
        if len(cfg.model.neighbor_limits) != num_layers:
            overwrite = True
            cfg.model.neighbor_limits = [30 for _ in range(num_layers)]

        # First get a avg of the pts per point cloud
        all_neighbor_counts = [[] for _ in range(num_layers)]
        truncated_n = [0 for _ in range(num_layers)]
        all_n = [0 for _ in range(num_layers)]
        while len(all_neighbor_counts[0]) < samples:

            # Get points, features and labels
            p_i = np.random.choice(self.n_objects)
            in_points = self.input_points[p_i]
            
            if in_points.shape[0] > 0:
                in_points, _, _ = calib_augment(in_points, np.copy(in_points), None)
                gpu_points = torch.from_numpy(in_points).to(device)
                in_dl = self.cfg.model.in_sub_size
                if in_dl > 0 and in_dl > self.cfg.data.init_sub_size * 1.01:
                    radius0 = self.cfg.model.in_sub_size * self.cfg.model.kp_radius
                    sub_points, _ = subsample_pack_batch(gpu_points,
                                                        [gpu_points.shape[0]],
                                                        cfg.model.in_sub_size,
                                                        method=cfg.model.in_sub_mode)
                else:
                    sub_points = gpu_points
                    radius0 = self.cfg.data.init_sub_size * self.cfg.model.kp_radius

                neighb_counts = pyramid_neighbor_stats(sub_points,
                                                       num_layers,
                                                       cfg.model.in_sub_size,
                                                       radius0,
                                                       cfg.model.radius_scaling,
                                                       sub_mode=cfg.model.in_sub_mode)

                # Update number of trucated_neighbors
                for j, neighb_c in enumerate(neighb_counts):
                    trucated_mask = neighb_c > cfg.model.neighbor_limits[j]
                    truncated_n[j] += int(torch.sum(trucated_mask.type(torch.long)))
                    all_n[j] += int(trucated_mask.shape[0])
                    all_neighbor_counts[j].append(neighb_c)
              
            pi += 1  
            print('', end='\r')
            print(fmt_str.format('#' * ((pi * progress_n) // pN), 100 * pi / pN), end='', flush=True)
        
        print('', end='\r')
        print(fmt_str.format('#' * progress_n, 100), end='', flush=True)
        print()

        t1 = time.time()

        # Collect results
        trunc_percents = [100.0 * n / m for n, m in zip(truncated_n, all_n)]
        all_neighbor_counts = [torch.concat(neighb_c_list, dim=0) for neighb_c_list in all_neighbor_counts]

        # Collect results
        advised_neighbor_limits = [int(torch.quantile(neighb_c, 0.95)) for neighb_c in all_neighbor_counts]

        if verbose:

            report_lines = ['Neighbors Calibration Report:']
            report_lines += ['*****************************']
            report_lines += ['']
            report_lines += ['{:d} clouds tested in {:.1f}s'.format(samples, t1 - t0)]
            report_lines += ['']

            if overwrite:
                report_lines += ['Calibrating for 5.0% of bigger neighborhoods:']
                str_format = num_layers * '{:6d} '
                limit_str = str_format.format(*advised_neighbor_limits)
                report_lines += ['   Neighbor limits = {:s}'.format(limit_str)]

            else:
                str_format = num_layers * '{:6d} '
                limit_str = str_format.format(*cfg.model.neighbor_limits)
                report_lines += ['    Current limits = {:s}'.format(limit_str)]
                str_format = num_layers * '{:5.1f}% '
                trunc_str = str_format.format(*trunc_percents)
                report_lines += ['total above limits = {:s}'.format(trunc_str)]
                
                report_lines += ['']
                report_lines += ['Advised values for 5.0%:']
                str_format = num_layers * '{:6d} '
                limit_str = str_format.format(*advised_neighbor_limits)
                report_lines += ['    Advised limits = {:s}'.format(limit_str)]

            frame_lines_1(report_lines)

        if overwrite:
            cfg.model.neighbor_limits = advised_neighbor_limits

        # # After calibration reset counters for regular sampling
        # self.reg_sampling_i *= 0

        return




# ----------------------------------------------------------------------------------------------------------------------
#
#           Utility classes definition
#       \********************************/


class ObjClassifSampler(Sampler):
    """Sampler for ObjClassifDataset"""

    def __init__(self, dataset: ObjClassifDataset):
        Sampler.__init__(self, dataset)

        # Does the sampler use potential for regular sampling
        self.use_potential = 'regular' in dataset.data_sampler

        # Should be balance the classes when sampling
        self.balance_labels = 'c-' in dataset.data_sampler

        # Dataset used by the sampler (no copy is made in memory)
        self.dataset = dataset

        # Create potentials
        if self.use_potential:
            self.potentials = np.random.rand(len(dataset.input_labels)) * 0.1 + 0.1
        else:
            self.potentials = None

        return

    def __iter__(self):
        """
        Yield next batch indices here
        """

        ##########################################
        # Initialize the list of generated indices
        ##########################################

        if self.use_potential:
            if self.balance_labels:

                gen_indices = []
                pick_n = self.dataset.epoch_n // self.dataset.num_classes + 1
                for i, l in enumerate(self.dataset.label_values):

                    # Get the potentials of the objects of this class
                    label_inds = np.where(np.equal(self.dataset.input_labels, l))[0]
                    class_potentials = self.potentials[label_inds]

                    # Get the indices to generate thanks to potentials
                    if pick_n < class_potentials.shape[0]:
                        pick_indices = np.argpartition(class_potentials, pick_n)[:pick_n]
                    else:
                        pick_indices = np.random.permutation(class_potentials.shape[0])
                    class_indices = label_inds[pick_indices]
                    gen_indices.append(class_indices)

                # Stack the chosen indices of all classes
                gen_indices = np.random.permutation(np.hstack(gen_indices))

            else:

                # Get indices with the minimum potential
                if self.dataset.epoch_n < self.potentials.shape[0]:
                    gen_indices = np.argpartition(self.potentials, self.dataset.epoch_n)[:self.dataset.epoch_n]
                else:
                    gen_indices = np.random.permutation(self.potentials.shape[0])
                gen_indices = np.random.permutation(gen_indices)

            # Update potentials (Change the order for the next epoch)
            self.potentials[gen_indices] = np.ceil(self.potentials[gen_indices])
            self.potentials[gen_indices] += np.random.rand(gen_indices.shape[0]) * 0.1 + 0.1

            votes = np.floor(self.potentials)
            self.dataset.n_votes *= 0
            self.dataset.n_votes += float(np.mean(votes))

        else:
            if self.balance_labels:
                pick_n = self.dataset.epoch_n // self.dataset.num_classes + 1
                gen_indices = []
                for l in self.dataset.label_values:
                    label_inds = np.where(np.equal(self.dataset.input_labels, l))[0]
                    rand_inds = np.random.choice(label_inds, size=pick_n, replace=True)
                    gen_indices += [rand_inds]
                gen_indices = np.random.permutation(np.hstack(gen_indices))
            else:
                gen_indices = np.random.permutation(self.dataset.num_models)[:self.dataset.epoch_n]

        ################
        # Generator loop
        ################

        # Initialize concatenation lists
        ti_list = []
        batch_n = 0

        # Generator loop
        for p_i in gen_indices:

            # Size of picked cloud
            n = self.dataset.input_points[p_i].shape[0]

            # In case batch is full, yield it and reset it
            if batch_n + n > self.dataset.b_lim and batch_n > 0:
                yield np.array(ti_list, dtype=np.int32)
                ti_list = []
                batch_n = 0

            # Add data to current batch
            ti_list += [p_i]

            # Update batch size
            batch_n += n

        yield np.array(ti_list, dtype=np.int32)

        return 0

    def __len__(self):
        """
        The number of yielded samples is variable
        """
        return None


class ObjClassifBatch:
    """Custom batch definition with memory pinning for ObjClassifDataset"""

    def __init__(self, input_dict):
        """
        Initialize a batch from the list of data returned by the dataset __get_item__ function.
        Here the data does not contain every subsampling/neighborhoods, and all arrays are in pack mode.
        """

        # Get rid of batch dimension
        self.in_dict = input_dict[0]

        return

    def pin_memory(self):
        """
        Manual pinning of the memory
        """

        for var_name, var_value in self.in_dict.items():
            if isinstance(var_value, list):
                self.in_dict[var_name] = [list_item.pin_memory() for list_item in var_value]
            else:
                self.in_dict[var_name] = var_value.pin_memory()

        return self

    def to(self, device):
        """
        Manual convertion to a different device.
        """

        for var_name, var_value in self.in_dict.items():
            if isinstance(var_value, list):
                self.in_dict[var_name] = [list_item.to(device) for list_item in var_value]
            else:
                self.in_dict[var_name] = var_value.to(device)

        return self

    def device(self):
        return self.in_dict.points[0].device


def ObjClassifCollate(batch_data):
    return ObjClassifBatch(batch_data)


