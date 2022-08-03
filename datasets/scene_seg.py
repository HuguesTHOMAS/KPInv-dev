
# ----------------------------------------------------------------------------------------------------------------------
#
#           Imports and global variables
#       \**********************************/
#

# Common libs
import copy
import time
import numpy as np
import pickle
from os import makedirs
from os.path import join, exists
from multiprocessing import Lock

import torch
from torch.utils.data import DataLoader, Dataset, Sampler
from sklearn.neighbors import KDTree

from kernels.kernel_points import create_3D_rotations, get_random_rotations
from utils.ply import read_ply, write_ply
from utils.cpp_funcs import grid_subsampling
from utils.gpu_subsampling import grid_subsample as gpu_grid_subsample
from utils.gpu_subsampling import init_gpu, subsample_numpy, subsample_list_mode
from utils.gpu_neigbors import pyramid_neighbor_stats
from utils.printing import frame_lines_1


# ----------------------------------------------------------------------------------------------------------------------
#
#           Class definition
#       \**********************/
#

class GpuSceneSegDataset(Dataset):
    """Parent class for Scene Segmentation Datasets."""

    def __init__(self, cfg, chosen_set='training', regular_sampling=False):
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
        self.regular_sampling = regular_sampling

        # Parameters depending on training or test
        if self.set == 'training':
            b_cfg = cfg.train
        else:
            b_cfg = cfg.test
        self.in_radius = b_cfg.in_radius
        self.b_n = b_cfg.batch_size
        self.b_lim = b_cfg.batch_limit
        self.max_p = b_cfg.max_points

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

        # Variables you need to populate for your own dataset
        self.scene_files = []
        
        # Variables taht will be automatically populated
        self.input_trees = []
        self.input_features = []
        self.input_labels = []
        self.val_proj = []
        self.val_labels = []
        self.label_indices = []
        
        # Regular sampling varaibles
        self.worker_lock = Lock()
        self.reg_sampling_i = torch.from_numpy(np.zeros((1,), dtype=np.int64))
        self.reg_sampling_i.share_memory_()
        self.reg_sample_pts = None
        self.reg_sample_clouds = None

        return

    def load_scenes_in_memory(self, label_property='label', f_properties=[], f_scales=[]):

        # Parameter
        dl = self.cfg.data.sub_size

        # Create path for files
        if dl > 0:
            tree_path = join(self.path, 'input_{:s}_{:.3f}'.format(self.cfg.model.sub_mode, dl))
        else:
            tree_path = join(self.path, 'input_no_sub')

        if not exists(tree_path):
            makedirs(tree_path)

        ##############
        # Load KDTrees
        ##############

        for i, file_path in enumerate(self.scene_files):

            # Restart timer
            t0 = time.time()

            # Get cloud name
            cloud_name = self.scene_names[i]

            # Name of the input files
            KDTree_file = join(tree_path, '{:s}_KDTree.pkl'.format(cloud_name))
            sub_ply_file = join(tree_path, '{:s}.ply'.format(cloud_name))

            # Check if inputs have already been computed
            if False and exists(KDTree_file):
                print('\nFound KDTree for cloud {:s}, subsampled at {:.3f}'.format(cloud_name, dl))

                # read ply with data
                if dl > 0:
                    data = read_ply(sub_ply_file)
                else:
                    data = read_ply(file_path)
                sub_features = np.vstack([data[f_prop].astype(np.float32) for f_prop in f_properties]).T
                sub_labels = data[label_property]

                # Read pkl with search tree
                with open(KDTree_file, 'rb') as f:
                    search_tree = pickle.load(f)

            else:
                print('\nPreparing KDTree for cloud {:s}, subsampled at {:.3f}'.format(cloud_name, dl))

                # Read ply file
                data = read_ply(file_path)
                points = np.vstack((data['x'], data['y'], data['z'])).T
                labels = data[label_property].astype(np.int32)
                features = np.vstack([data[f_prop].astype(np.float32) for f_prop in f_properties]).T

                # Subsample cloud (optional)
                if dl > 0:
                    sub_points, sub_features, sub_labels = subsample_numpy(points,
                                                                           dl,
                                                                           features=features,
                                                                           labels=labels,
                                                                           method=self.cfg.model.sub_mode)
                    write_ply(sub_ply_file,
                            [sub_points, sub_features, sub_labels.astype(np.int32)],
                            ['x', 'y', 'z'] + f_properties + [label_property])
                else:
                    sub_points, sub_features, sub_labels = (points, features, labels)
                
                # Get chosen neighborhoods
                search_tree = KDTree(sub_points, leaf_size=10)

                # Save KDTree
                with open(KDTree_file, 'wb') as f:
                    pickle.dump(search_tree, f)

            # Check data types and scale features
            sub_labels = sub_labels.astype(np.int32)
            sub_features = sub_features.astype(np.float32)
            sub_features *= np.array(f_scales, dtype=np.float32)

            # Fill data containers
            self.input_trees += [search_tree]
            self.input_features += [sub_features]
            self.input_labels += [sub_labels]

            size = sub_features.shape[0] * 4 * 7
            print('{:.1f} MB loaded in {:.1f}s'.format(size * 1e-6, time.time() - t0))


        ######################
        # Reprojection indices
        ######################

        # Only necessary for validation and test sets
        if dl > 0 and self.set in ['validation', 'test']:

            print('\nPreparing reprojection indices for testing')

            # Get validation/test reprojection indices
            for i, file_path in enumerate(self.scene_files):

                # Restart timer
                t0 = time.time()

                # File name for saving
                proj_file = join(tree_path, '{:s}_proj.pkl'.format(self.scene_names[i]))

                # Try to load previous indices
                if exists(proj_file):
                    with open(proj_file, 'rb') as f:
                        proj_inds, labels = pickle.load(f)
                else:
                    data = read_ply(file_path)
                    points = np.vstack((data['x'], data['y'], data['z'])).T
                    if self.set == 'test':
                        labels = np.zeros((data.shape[0],), dtype=np.int32)
                    else:
                        labels = data[label_property].astype(np.int32)

                    # Compute projection inds
                    idxs = self.input_trees[i].query(points, return_distance=False)
                    proj_inds = np.squeeze(idxs).astype(np.int32)

                    # Save
                    with open(proj_file, 'wb') as f:
                        pickle.dump([proj_inds, labels], f)

                self.val_proj += [proj_inds]
                self.val_labels += [labels]
                print('{:s} done in {:.1f}s'.format(self.scene_names[i], time.time() - t0))

        print()


        return

    def prepare_label_inds(self):

        # Choose random points of each class for each cloud
        for label in self.label_values:

            # Gather indices of the points with this label in all the input clouds [2, N1], [2, N2], ...]
            l_inds = []
            for cloud_ind, cloud_labels in enumerate(self.input_labels):
                label_indices = np.where(np.equal(cloud_labels, label))[0]
                l_inds.append(np.vstack((np.full(label_indices.shape, cloud_ind, dtype=np.int64), label_indices)))

            # Stack them: [2, N1+N2+...]
            l_inds = np.hstack(l_inds)
            self.label_indices.append(l_inds)

        return

    def new_reg_sampling_pts(self, subsample_ratio=1.0):
        """
        Function selecting points from the dataset to be used as sphere center.
        Points are chosen with a regular subsampling so that the whole dataset get tested.
        """

        # Get gpu for faster calibration
        device = init_gpu()

        # Subsampling size (subsample_ratio should be < sqrt(3)/2)
        dl = self.in_radius * subsample_ratio

        all_reg_pts = []
        all_reg_clouds = []
        for cloud_ind, tree in enumerate(self.input_trees):

            # Subsample scene clouds
            points = np.array(tree.data, copy=False).astype(np.float32)
            gpu_points = torch.from_numpy(points).to(device)
            sub_points, _ = subsample_list_mode(gpu_points,
                                                [gpu_points.shape[0]],
                                                dl,
                                                method='hex')
            
            # Stack points and cloud indices
            all_reg_pts.append(sub_points.cpu())
            all_reg_clouds.append(torch.full((sub_points.shape[0],), cloud_ind, dtype=torch.long))

        # Shuffle
        self.reg_sample_pts = torch.concat(all_reg_pts, dim=0)
        self.reg_sample_clouds = torch.concat(all_reg_clouds, dim=0)
        rand_shuffle = torch.randperm(self.reg_sample_clouds.shape[0])
        self.reg_sample_pts = self.reg_sample_pts[rand_shuffle]
        self.reg_sample_clouds = self.reg_sample_clouds[rand_shuffle]

        # Share memory
        self.reg_sample_pts.share_memory_()
        self.reg_sample_clouds.share_memory_()

        return
    
    def sample_random_sphere(self,  center_noise=0.1):

        if self.regular_sampling:
            
            with self.worker_lock:

                # If first time, compute new regular sampling points
                if self.reg_sample_pts is None:
                    self.new_reg_sampling_pts()
                    
                # If we reach the end of the regular sampling points, Recompute new ones
                reg_sampling_N = int(self.reg_sample_pts.shape[0])
                if self.reg_sampling_i >= reg_sampling_N:
                    self.reg_sampling_i -= reg_sampling_N
                    self.new_reg_sampling_pts()

                # Get next regular sampling element
                cloud_ind = int(self.reg_sample_clouds[self.reg_sampling_i])
                center_point = self.reg_sample_pts[self.reg_sampling_i].numpy()

                # Update sampling index
                self.reg_sampling_i += 1

        else:

            # Choose a random label
            rand_l = np.random.choice(self.num_classes)
            while rand_l in self.ignored_labels:
                rand_l = np.random.choice(self.num_classes)

            # Choose a random point from this class
            rand_ind = np.random.choice(self.label_indices[rand_l].shape[1])
            cloud_ind, point_ind = self.label_indices[rand_l][:, rand_ind]
            
            # Get points from tree structure
            points = np.array(self.input_trees[cloud_ind].data, copy=False)

            # Center point of input region
            center_point = points[point_ind, :].reshape(1, -1)

            # Add a small noise to center point
            center_point += np.random.normal(scale=center_noise * self.in_radius, size=center_point.shape)

        return cloud_ind, center_point

    def get_sphere(self, cloud_ind, center_point, only_inds=False):

        # Get points from tree structure
        points = np.array(self.input_trees[cloud_ind].data, copy=False)

        # Indices of points in input region
        input_inds = self.input_trees[cloud_ind].query_radius(center_point, r=self.in_radius)[0]
        
        if only_inds:
            return input_inds

        # Collect labels and colors
        input_points = (points[input_inds] - center_point).astype(np.float32)
        input_features = self.input_features[cloud_ind][input_inds]
        if self.set in ['test', 'ERF']:
            input_labels = np.zeros(input_points.shape[0])
        else:
            input_labels = self.input_labels[cloud_ind][input_inds]
            # input_labels = np.array([self.label_to_idx[l] for l in input_labels])

        return input_inds, input_points, input_features, input_labels

    def select_features(self, in_features):
        print('ERROR: This function select_features needs to be redifined in the child dataset class. It depends on the dataset')
        return 

    def __len__(self):
        """
        Return the length of data here
        """
        return len(self.scene_names)

    def __getitem__(self, batch_i):
        """
        Getting item from random sampling, and returning simple input (without subsamplings and neighbors).
        """

        # Initiate concatanation lists
        p_list = []
        f_list = []
        l_list = []
        i_list = []
        pi_list = []
        pinv_list = []
        ci_list = []
        s_list = []
        R_list = []
        batch_n_pts = 0

        while True:

            # Pick a sphere center randomly
            cloud_ind, c_point = self.sample_random_sphere()

            # Get the input sphere
            in_inds, in_points, in_features, in_labels = self.get_sphere(cloud_ind, c_point)
            
            # Color augmentation
            if self.set == 'training' and np.random.rand() > self.cfg.train.augment_color:
                in_features *= 0
            
            # Add original height as additional feature
            in_features = np.hstack((in_features, in_points[:, 2:] + c_point[:, 2:])).astype(np.float32)

            # Select features for the network
            in_features = self.select_features(in_features)

            # Data augmentation
            in_points, scale, R = self.augmentation_transform(in_points)

            # Input subampling (on CPU to be parrallelizable)
            in_points, in_features, in_labels, inv_inds = subsample_numpy(in_points,
                                                                          self.cfg.model.init_sub_size,
                                                                          features=in_features,
                                                                          labels=in_labels,
                                                                          method=self.cfg.model.sub_mode,
                                                                          on_gpu=False,
                                                                          return_inverse=True)


            # Stack batch
            p_list += [in_points]
            f_list += [in_features]
            l_list += [in_labels]
            pi_list += [in_inds]
            pinv_list += [inv_inds]
            i_list += [c_point]
            ci_list += [cloud_ind]
            s_list += [scale]
            R_list += [R]

            # Update batch size
            batch_n_pts += in_inds.shape[0]

            # In case batch is full, stop
            if batch_n_pts > int(self.b_lim):
                break


        ###################
        # Concatenate batch
        ###################




        stacked_points = np.concatenate(p_list, axis=0)
        stacked_features = np.concatenate(f_list, axis=0)
        stacked_labels = np.concatenate(l_list, axis=0)
        center_points = np.concatenate(i_list, axis=0)
        cloud_inds = np.array(ci_list, dtype=np.int32)
        input_inds = np.concatenate(pi_list, axis=0)
        input_invs = np.concatenate(pinv_list, axis=0)


        print(stacked_points.shape)
        print(stacked_features.shape)
        print(stacked_labels.shape)
        print(center_points.shape)
        print(cloud_inds.shape)
        print(input_inds.shape)
        print(input_invs.shape)

        a = 1/0

        stack_lengths = np.array([pp.shape[0] for pp in p_list], dtype=np.int32)
        scales = np.array(s_list, dtype=np.float32)
        rots = np.stack(R_list, axis=0)


        #######################
        # Create network inputs
        #######################
        #
        #   Points, features, etc.
        #

        # # Get the whole input list
        # input_list = self.segmentation_inputs(stacked_points,
        #                                       stack_lengths)

        input_list = [stacked_points, stacked_features, stacked_labels, stack_lengths]

        # Add scale and rotation for testing
        input_list += [scales, rots, cloud_inds, center_points, input_inds, input_invs]

        return input_list

    def calib_batch_size(self, samples=20, verbose=True):

        t0 = time.time()

        # Get gpu for faster calibration
        device = init_gpu()
        
        all_batch_n = []
        all_batch_n_pts = []
        for i in range(samples):

            batch_n = 0
            batch_n_pts = 0
            while True:
                cloud_ind, center_p = self.sample_random_sphere()
                _, in_points, _, _ = self.get_sphere(cloud_ind, center_p)
                in_points, _, _ = self.augmentation_transform(in_points)
                gpu_points = torch.from_numpy(in_points).to(device)
                sub_points, _ = subsample_list_mode(gpu_points,
                                                    [gpu_points.shape[0]],
                                                    self.cfg.model.init_sub_size,
                                                    method=self.cfg.model.sub_mode)
                batch_n_pts += sub_points.shape[0]
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

    def calib_batch_limit(self, samples=100, verbose=True):
        """
        Find the batch_limit given the target batch_size. 
        The batch size varies randomly so we prefer a quick calibration to find 
        an approximate batch limit.
        """


        t0 = time.time()

        # Get gpu for faster calibration
        device = init_gpu()

        # Advanced display
        pi = 0
        pN = samples
        progress_n = 30
        fmt_str = '[{:<' + str(progress_n) + '}] {:5.1f}%'
        print('\nSearching batch_limit given the target batch_size.')

        # First get a avg of the pts per point cloud
        all_cloud_n = []
        dt = np.zeros((1,))
        for i in range(samples):
            cloud_ind, center_p = self.sample_random_sphere()
            _, in_points, _, _ = self.get_sphere(cloud_ind, center_p)
            in_points, _, _ = self.augmentation_transform(in_points)
            gpu_points = torch.from_numpy(in_points).to(device)
            sub_points, _ = subsample_list_mode(gpu_points,
                                                [gpu_points.shape[0]],
                                                self.cfg.model.init_sub_size,
                                                method=self.cfg.model.sub_mode)
            all_cloud_n.append(sub_points.shape[0])
                
            pi += 1
            print('', end='\r')
            print(fmt_str.format('#' * ((pi * progress_n) // pN), 100 * pi / pN), end='', flush=True)

        print('', end='\r')
        print(fmt_str.format('#' * progress_n, 100), end='', flush=True)
        print('\n')

        # Initial batch limit thanks to average points per batch
        mean_cloud_n = np.mean(all_cloud_n)
        self.b_lim = mean_cloud_n *self.b_n

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
                if batch_n_pts > self.b_lim:
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
            report_lines += ['     batch limit = {:.3f}'.format(self.b_lim)]
            report_lines += ['avg batch points = {:.3f}'.format(np.mean(all_batch_n_pts))]
            report_lines += ['std batch points = {:.3f}'.format(np.std(all_batch_n_pts))]
            report_lines += ['']
            report_lines += ['New batch size obtained from calibration:']
            report_lines += ['  avg batch size = {:.1f}'.format(np.mean(all_batch_n))]
            report_lines += ['  std batch size = {:.2f}'.format(np.std(all_batch_n))]

            frame_lines_1(report_lines)


        return

    def calib_neighbors(self, cfg, samples=100, verbose=True):

        t0 = time.time()

        # Get gpu for faster calibration
        device = init_gpu()
        
        # Advanced display
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
        for pi in range(samples):
            cloud_ind, center_p = self.sample_random_sphere()
            _, in_points, _, _ = self.get_sphere(cloud_ind, center_p)
            in_points, _, _ = self.augmentation_transform(in_points)
            gpu_points = torch.from_numpy(in_points).to(device)
            sub_points, _ = subsample_list_mode(gpu_points,
                                                [gpu_points.shape[0]],
                                                cfg.model.init_sub_size,
                                                method=cfg.model.sub_mode)
            neighb_counts = pyramid_neighbor_stats(sub_points,
                                                   num_layers,
                                                   cfg.model.init_sub_size,
                                                   cfg.model.init_sub_size * cfg.model.kp_radius,
                                                   sub_mode=cfg.model.sub_mode)

            # Update number of trucated_neighbors
            for j, neighb_c in enumerate(neighb_counts):
                trucated_mask = neighb_c > cfg.model.neighbor_limits[j]
                truncated_n[j] += int(torch.sum(trucated_mask.type(torch.long)))
                all_n[j] += int(trucated_mask.shape[0])
                all_neighbor_counts[j].append(neighb_c)
                
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
        advised_neighbor_limits = [int(torch.quantile(neighb_c, 0.98)) for neighb_c in all_neighbor_counts]

        if verbose:

            report_lines = ['Neighbors Calibration Report:']
            report_lines += ['*****************************']
            report_lines += ['']
            report_lines += ['{:d} clouds tested in {:.1f}s'.format(samples, t1 - t0)]
            report_lines += ['']

            if overwrite:
                report_lines += ['Calibrating for 2.0% of bigger neighborhoods:']
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
                report_lines += ['Advised values for 2.0%:']
                str_format = num_layers * '{:6d} '
                limit_str = str_format.format(*advised_neighbor_limits)
                report_lines += ['    Advised limits = {:s}'.format(limit_str)]

            frame_lines_1(report_lines)

        if overwrite:
            cfg.model.neighbor_limits = advised_neighbor_limits

        return

    def augmentation_transform(self, points, normals=None, verbose=False):
        """Implementation of an augmentation transform for point clouds."""

        ##########
        # Rotation
        ##########

        # Initialize rotation matrix
        R = np.eye(points.shape[1])

        if points.shape[1] == 3:
            if self.cfg.train.augment_rotation == 'vertical':

                # Create random rotations
                theta = np.random.rand() * 2 * np.pi
                c, s = np.cos(theta), np.sin(theta)
                R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float32)

            elif self.cfg.train.augment_rotation == 'all':

                    R = get_random_rotations(shape=None)

        R = R.astype(np.float32)

        #######
        # Scale
        #######

        # Choose random scales for each example
        min_s = self.cfg.train.augment_min_scale
        max_s = self.cfg.train.augment_max_scale
        if self.cfg.train.augment_anisotropic:
            scale = np.random.rand(points.shape[1]) * (max_s - min_s) + min_s
        else:
            scale = np.random.rand() * (max_s - min_s) + min_s

        # Add random symmetries to the scale factor
        symmetries = np.array(self.cfg.train.augment_symmetries).astype(np.int32)
        symmetries *= np.random.randint(2, size=points.shape[1])
        scale = (scale * (1 - symmetries * 2)).astype(np.float32)

        #######
        # Noise
        #######

        noise = (np.random.randn(points.shape[0], points.shape[1]) * self.cfg.train.augment_noise).astype(np.float32)

        ##################
        # Apply transforms
        ##################

        # Do not use np.dot because it is multi-threaded
        #augmented_points = np.dot(points, R) * scale + noise
        augmented_points = np.sum(np.expand_dims(points, 2) * R, axis=1) * scale + noise


        if normals is None:
            return augmented_points, scale, R
        else:
            # Anisotropic scale of the normals thanks to cross product formula
            normal_scale = scale[[1, 2, 0]] * scale[[2, 0, 1]]
            #augmented_normals = np.dot(normals, R) * normal_scale
            augmented_normals = np.sum(np.expand_dims(normals, 2) * R, axis=1) * normal_scale
            # Renormalise
            augmented_normals *= 1 / (np.linalg.norm(augmented_normals, axis=1, keepdims=True) + 1e-6)

            if verbose:
                test_p = [np.vstack([points, augmented_points])]
                test_n = [np.vstack([normals, augmented_normals])]
                test_l = [np.hstack([points[:, 2]*0, augmented_points[:, 2]*0+1])]
                show_ModelNet_examples(test_p, test_n, test_l)

            return augmented_points, augmented_normals, scale, R


    # Old functions


    def rotate_batch(self, batch, rotation_mode='all'):

        # create a copy of the batch
        rotated_batch = copy.deepcopy(batch)
        B = len(rotated_batch.lengths[0])
        dim = rotated_batch.points[0].shape[1]

        # Create random rotations
        rots = []
        for b_i in range(B):

            R = np.eye(dim)

            if dim == 3:
                if rotation_mode == 'vertical':

                    # Create random rotations
                    theta = np.random.rand() * 2 * np.pi
                    c, s = np.cos(theta), np.sin(theta)
                    R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float32)

                elif rotation_mode == 'all':

                    R = get_random_rotations(shape=None)

            rots.append(torch.from_numpy(R.astype(np.float32)).to(batch.points[0].device))

        # Deal with stacked points in batch
        for l_i, lengths in enumerate(rotated_batch.lengths):

            i0 = 0
            for b_i, length in enumerate(lengths):

                # Get points and rotate them
                points = rotated_batch.points[l_i][i0:i0 + length]
                rotated_batch.points[l_i][i0:i0 + length] = torch.matmul(points, rots[b_i])

                # Same for normals if necessary
                if l_i == 0 and hasattr(rotated_batch, 'normals'):
                    normals = rotated_batch.normals[i0:i0 + length]
                    rotated_batch.normals[l_i][i0:i0 + length] = torch.matmul(normals, rots[b_i])

                # Same for lrf if necessary
                if l_i == 0 and hasattr(rotated_batch, 'lrf'):
                    lrf = (rotated_batch.lrf[i0:i0 + length]).transpose(-1, -2)
                    rotated_batch.lrf[i0:i0 + length] = torch.matmul(lrf, rots[b_i]).transpose(-1, -2)

                i0 += length

        return rotated_batch, rots

    def big_neighborhood_filter(self, neighbors, layer):
        """
        Filter neighborhoods with max number of neighbors. Limit is set to keep XX% of the neighborhoods untouched.
        Limit is computed at initialization
        """

        # crop neighbors matrix
        if len(self.neighborhood_limits) > 0:
            return neighbors[:, :self.neighborhood_limits[layer]]
        else:
            return neighbors

    def classification_inputs(self,
                              stacked_points,
                              stack_lengths):

        # Starting radius of convolutions
        r_normal = self.config.first_subsampling_dl * self.config.conv_radius

        # Starting layer
        layer_blocks = []

        # Lists of inputs
        input_points = []
        input_neighbors = []
        input_pools = []
        input_stack_lengths = []
        deform_layers = []

        ######################
        # Loop over the blocks
        ######################

        arch = self.config.architecture

        for block_i, block in enumerate(arch):

            # Get all blocks of the layer
            if not ('pool' in block or 'strided' in block or 'global' in block or 'upsample' in block):
                layer_blocks += [block]
                continue

            # Convolution neighbors indices
            # *****************************

            deform_layer = False
            if layer_blocks:
                # Convolutions are done in this layer, compute the neighbors with the good radius
                if np.any(['deformable' in blck for blck in layer_blocks]):
                    r = r_normal * self.config.deform_radius / self.config.conv_radius
                    deform_layer = True
                else:
                    r = r_normal
                conv_i = batch_neighbors(stacked_points, stacked_points, stack_lengths, stack_lengths, r)

            else:
                # This layer only perform pooling, no neighbors required
                conv_i = np.zeros((0, 1), dtype=np.int32)

            # Pooling neighbors indices
            # *************************

            # If end of layer is a pooling operation
            if 'pool' in block or 'strided' in block:

                # New subsampling length
                dl = 2 * r_normal / self.config.conv_radius

                # Subsampled points
                pool_p, pool_b = batch_grid_subsampling(stacked_points, stack_lengths, sampleDl=dl)

                # Radius of pooled neighbors
                if 'deformable' in block:
                    r = r_normal * self.config.deform_radius / self.config.conv_radius
                    deform_layer = True
                else:
                    r = r_normal

                # Subsample indices
                pool_i = batch_neighbors(pool_p, stacked_points, pool_b, stack_lengths, r)

            else:
                # No pooling in the end of this layer, no pooling indices required
                pool_i = np.zeros((0, 1), dtype=np.int32)
                pool_p = np.zeros((0, 1), dtype=np.float32)
                pool_b = np.zeros((0,), dtype=np.int32)

            # Reduce size of neighbors matrices by eliminating furthest point
            conv_i = self.big_neighborhood_filter(conv_i, len(input_points))
            pool_i = self.big_neighborhood_filter(pool_i, len(input_points))

            # Updating input lists
            input_points += [stacked_points]
            input_neighbors += [conv_i.astype(np.int64)]
            input_pools += [pool_i.astype(np.int64)]
            input_stack_lengths += [stack_lengths]
            deform_layers += [deform_layer]

            # New points for next layer
            stacked_points = pool_p
            stack_lengths = pool_b

            # Update radius and reset blocks
            r_normal *= 2
            layer_blocks = []

            # Stop when meeting a global pooling or upsampling
            if 'global' in block or 'upsample' in block:
                break

        ###############
        # Return inputs
        ###############

        # Save deform layers

        # list of network inputs
        li = input_points + input_neighbors + input_pools + input_stack_lengths

        return li

    def segmentation_inputs(self,
                            stacked_points,
                            stack_lengths):

        # Starting radius of convolutions
        r_normal = self.config.first_subsampling_dl * self.config.conv_radius

        # Starting layer
        layer_blocks = []

        # Lists of inputs
        input_points = []
        input_neighbors = []
        input_pools = []
        input_upsamples = []
        input_stack_lengths = []
        deform_layers = []

        ######################
        # Loop over the blocks
        ######################

        arch = self.config.architecture

        for block_i, block in enumerate(arch):

            # Get all blocks of the layer
            if not ('pool' in block or 'strided' in block or 'global' in block or 'upsample' in block):
                layer_blocks += [block]
                continue

            # Convolution neighbors indices
            # *****************************

            deform_layer = False
            if layer_blocks:
                # Convolutions are done in this layer, compute the neighbors with the good radius
                if np.any(['deformable' in blck for blck in layer_blocks]):
                    r = r_normal * self.config.deform_radius / self.config.conv_radius
                    deform_layer = True
                else:
                    r = r_normal
                conv_i = batch_neighbors(stacked_points, stacked_points, stack_lengths, stack_lengths, r)

            else:
                # This layer only perform pooling, no neighbors required
                conv_i = np.zeros((0, 1), dtype=np.int32)

            # Pooling neighbors indices
            # *************************

            # If end of layer is a pooling operation
            if 'pool' in block or 'strided' in block:

                # New subsampling length
                dl = 2 * r_normal / self.config.conv_radius

                # Subsampled points
                pool_p, pool_b = batch_grid_subsampling(stacked_points, stack_lengths, sampleDl=dl)

                # Radius of pooled neighbors
                if 'deformable' in block:
                    r = r_normal * self.config.deform_radius / self.config.conv_radius
                    deform_layer = True
                else:
                    r = r_normal

                # Subsample indices
                pool_i = batch_neighbors(pool_p, stacked_points, pool_b, stack_lengths, r)

                # Upsample indices (with the radius of the next layer to keep wanted density)
                up_i = batch_neighbors(stacked_points, pool_p, stack_lengths, pool_b, 2 * r)

            else:
                # No pooling in the end of this layer, no pooling indices required
                pool_i = np.zeros((0, 1), dtype=np.int32)
                pool_p = np.zeros((0, 3), dtype=np.float32)
                pool_b = np.zeros((0,), dtype=np.int32)
                up_i = np.zeros((0, 1), dtype=np.int32)

            # Reduce size of neighbors matrices by eliminating furthest point
            conv_i = self.big_neighborhood_filter(conv_i, len(input_points))
            pool_i = self.big_neighborhood_filter(pool_i, len(input_points))
            if up_i.shape[0] > 0:
                up_i = self.big_neighborhood_filter(up_i, len(input_points)+1)

            # Updating input lists
            input_points += [stacked_points]
            input_neighbors += [conv_i.astype(np.int64)]
            input_pools += [pool_i.astype(np.int64)]
            input_upsamples += [up_i.astype(np.int64)]
            input_stack_lengths += [stack_lengths]
            deform_layers += [deform_layer]

            # New points for next layer
            stacked_points = pool_p
            stack_lengths = pool_b

            # Update radius and reset blocks
            r_normal *= 2
            layer_blocks = []

            # Stop when meeting a global pooling or upsampling
            if 'global' in block or 'upsample' in block:
                break

        ###############
        # Return inputs
        ###############

        # list of network inputs
        li = input_points + input_neighbors + input_pools + input_upsamples + input_stack_lengths

        return li



# ----------------------------------------------------------------------------------------------------------------------
#
#           Utility classes definition
#       \********************************/


class GpuSceneSegSampler(Sampler):
    """Sampler for GpuSceneSegDataset"""

    def __init__(self, dataset: GpuSceneSegDataset):
        Sampler.__init__(self, dataset)

        # Dataset used by the sampler (no copy is made in memory)
        self.dataset = dataset

        # Number of step per epoch
        if dataset.set == 'training':
            self.N = dataset.cfg.train.steps_per_epoch
        else:
            self.N = dataset.cfg.test.steps_per_epoch
        return

    def __iter__(self):
        """
        Yield next batch indices here. In this dataset, this is a dummy sampler that yield the index of batch element
        (input sphere) in epoch instead of the list of point indices
        """

        # Generator loop
        for i in range(self.N):
            yield i

    def __len__(self):
        """
        The number of yielded samples is variable
        """
        return self.N


class GpuSceneSegBatch:
    """Custom batch definition with memory pinning for GpuSceneSegDataset"""

    def __init__(self, input_list):
        """
        Initialize a batch from the list of data returned by the dataset __get_item__ function.
        Here the data does not contain every subsampling/neighborhoods, and all arrays are in pack mode.
        """

        # Get rid of batch dimension
        input_list = input_list[0]

        # List of attributes we set in this function
        attribute_list = ['points',
                          'labels',
                          'lengths',
                          'scales',
                          'rots',
                          'cloud_inds',
                          'center_points',
                          'input_inds',
                          'input_invs']

        # All attributes are np arrays. MOdify this if you have types other values
        for i, input_array in input_list:
            setattr(self, attribute_list[i], torch.from_numpy(input_array))

        return

    def pin_memory(self):
        """
        Manual pinning of the memory
        """
        for var_name, var_value in vars(self).items():
            setattr(self, var_name, var_value.pin_memory())
        return self

    def to(self, device):
        """
        Manual convertion to a different device.
        """
        for var_name, var_value in vars(self).items():
            setattr(self, var_name, var_value.to(device))
        return self


def GpuSceneSegCollate(batch_data):
    return GpuSceneSegBatch(batch_data)


