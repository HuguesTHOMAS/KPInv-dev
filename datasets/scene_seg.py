
# ----------------------------------------------------------------------------------------------------------------------
#
#           Script Intro
#       \******************/
#
#
#   This file contains the defintion of a point cloud segmentation dataset that can be specialize to any dataset like
#   S3DIS/NPM3D/Semantic3D by creating a subclass.
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
from utils.gpu_neigbors import keops_knn, tiled_knn
from utils.torch_pyramid import build_full_pyramid, pyramid_neighbor_stats, build_base_pyramid

from utils.transform import ComposeAugment, RandomRotate, RandomScaleFlip, RandomJitter, FloorCentering, \
    ChromaticAutoContrast, ChromaticTranslation, ChromaticJitter, HueSaturationTranslation, RandomDropColor, \
    ChromaticNormalize, HeightNormalize


# ----------------------------------------------------------------------------------------------------------------------
#
#           Class definition
#       \**********************/
#

class SceneSegDataset(Dataset):

    def __init__(self, cfg, chosen_set='training', regular_sampling=False, precompute_pyramid=False):
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
        self.precompute_pyramid = precompute_pyramid

        # Parameters depending on training or test
        if self.set == 'training':
            b_cfg = cfg.train
        else:
            b_cfg = cfg.test
        self.in_radius = b_cfg.in_radius
        self.b_n = b_cfg.batch_size
        self.b_lim = b_cfg.batch_limit
        self.max_p = b_cfg.max_points

        # Cube sampling or sphere sampling
        self.use_cubes = cfg.data.use_cubes
        self.cylindric_input = cfg.data.cylindric_input

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

        # Variables you need to populate for your own dataset
        self.scene_files = []
        
        # Variables taht will be automatically populated
        self.input_trees = []
        self.input_features = []
        self.input_labels = []
        self.input_z = []
        self.test_proj = []
        self.val_labels = []
        self.label_indices = []
        
        # Regular sampling varaibles
        self.worker_lock = Lock()
        self.reg_sampling_i = torch.from_numpy(np.zeros((1,), dtype=np.int64))
        self.reg_sampling_i.share_memory_()
        self.reg_votes = torch.from_numpy(np.zeros((1,), dtype=np.int64))
        self.reg_votes.share_memory_()

        self.reg_sample_pts = None
        self.reg_sample_clouds = None

        # Get augmentation transform
        if self.set == 'training':
            a_cfg = cfg.augment_train
        else:
            a_cfg = cfg.augment_test
        augment_list = []
        augment_list.append(RandomScaleFlip(scale=a_cfg.scale,
                                            anisotropic=a_cfg.anisotropic,
                                            flip_p=a_cfg.flips))
        augment_list.append(RandomJitter(sigma=a_cfg.jitter,
                                         clip=a_cfg.jitter * 5))
        augment_list.append(FloorCentering())
        augment_list.append(RandomRotate(mode=a_cfg.rotations))
        if a_cfg.chromatic_contrast:
            augment_list += [ChromaticAutoContrast()]
        if a_cfg.chromatic_all:
            augment_list += [ChromaticTranslation(),
                             ChromaticJitter(),
                             HueSaturationTranslation()]
        if a_cfg.chromatic_norm:
            augment_list += [ChromaticNormalize(),
                             HeightNormalize()]
        augment_list.append(RandomDropColor(p=a_cfg.color_drop))

        # TRAIN AUGMENT
        # Transformer: no drop color and chromatic contrast/transl/jitter/HSV
        #   PointNext: floor centering, chromatic contrast/drop/norm
        #        test: drop before or after norm? In theory it should be after norm

        # TEST AUGMENT
        # Transformer: no augment at all
        #   PointNext: floor centering and chromatic_norm (drop color if vote)

        self.augmentation_transform = ComposeAugment(augment_list)

        return

    def load_evaluation_points(self, file_path):

        # Get original points
        data = read_ply(file_path)
        return np.vstack((data['x'], data['y'], data['z'])).T

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
            inf_str = ''
            if self.cylindric_input:
                inf_str = '_2D'
            KDTree_file = join(tree_path, '{:s}_KDTree{:s}.pkl'.format(cloud_name, inf_str))
            sub_ply_file = join(tree_path, '{:s}.ply'.format(cloud_name))

            # Check if inputs have already been computed
            if exists(KDTree_file):
                print('\nFound KDTree for cloud {:s}, subsampled at {:.3f}'.format(cloud_name, dl))

                # read ply with data
                if dl > 0:
                    data = read_ply(sub_ply_file)
                else:
                    data = read_ply(file_path)
                sub_features = np.vstack([data[f_prop].astype(np.float32) for f_prop in f_properties]).T

                if label_property in [p for p, _ in data.dtype.fields.items()]:
                    sub_labels = data[label_property]
                else:
                    sub_labels = None

                # Read pkl with search tree
                with open(KDTree_file, 'rb') as f:
                    if self.cylindric_input:
                        search_tree, sub_z = pickle.load(f)
                    else:
                        search_tree = pickle.load(f)
                        sub_z = None

            else:
                print('\nPreparing KDTree for cloud {:s}, subsampled at {:.3f}'.format(cloud_name, dl))

                # Read ply file
                data = read_ply(file_path)
                points = np.vstack((data['x'], data['y'], data['z'])).T
                if label_property in [p for p, _ in data.dtype.fields.items()]:
                    labels = data[label_property].astype(np.int32)
                else:
                    labels = None
                features = np.vstack([data[f_prop].astype(np.float32) for f_prop in f_properties]).T

                # Subsample cloud (optional)
                if dl > 0:
                    sub_data = subsample_numpy(points,
                                               dl,
                                               features=features,
                                               labels=labels,
                                               method=self.cfg.model.sub_mode)
                                               
                    if labels is None:
                        sub_points, sub_features = sub_data
                        sub_labels = None
                        sub_features *= np.array(f_scales, dtype=np.float32)
                        write_ply(sub_ply_file,
                                [sub_points, sub_features],
                                ['x', 'y', 'z'] + f_properties)

                    else:
                        sub_points, sub_features, sub_labels = sub_data
                        sub_features *= np.array(f_scales, dtype=np.float32)
                        write_ply(sub_ply_file,
                                [sub_points, sub_features, sub_labels.astype(np.int32)],
                                ['x', 'y', 'z'] + f_properties + [label_property])

                else:
                    sub_points, sub_features, sub_labels = (points, features, labels)
                
                # Project data in 2D if we want infinite height
                if self.cylindric_input:
                    sub_z = sub_points[:, 2:].astype(np.float32)
                    sub_points = sub_points[:, :2]
                else:
                    sub_z = None

                # Compute KD Tree
                search_tree = KDTree(sub_points, leaf_size=10)

                # Save KDTree
                with open(KDTree_file, 'wb') as f:
                    if self.cylindric_input:
                        pickle.dump((search_tree, sub_z), f)
                    else:
                        pickle.dump(search_tree, f)

            # Check data types and scale features
            sub_features = sub_features.astype(np.float32)
            if sub_labels is not None:
                sub_labels = sub_labels.astype(np.int32)
            else:
                sub_labels = np.zeros((0,), dtype=np.int32)

            # Fill data containers
            self.input_trees += [search_tree]
            self.input_features += [sub_features]
            self.input_labels += [sub_labels]
            self.input_z += [sub_z]

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
                    if label_property in [p for p, _ in data.dtype.fields.items()]:
                        labels = data[label_property].astype(np.int32)
                    else:
                        labels = np.zeros((0,), dtype=np.int32)

                    # Get data on GPU
                    device = init_gpu()
                    support_points = np.array(self.input_trees[i].data, copy=False)
                    if self.cylindric_input:
                        support_points = np.hstack((support_points, self.input_z[i]))
                    s_pts = torch.from_numpy(support_points).to(device).type(torch.float32)
                    q_pts = torch.from_numpy(points).to(device).type(torch.float32)

                    # Compute nearest neighbors per tiles
                    _, idxs = tiled_knn(q_pts, s_pts, k=1, tile_size=3.5, margin=2 * dl)
                    proj_inds = np.squeeze(idxs.cpu().numpy()).astype(np.int32)

                    # Save
                    with open(proj_file, 'wb') as f:
                        pickle.dump([proj_inds, labels], f)

                self.test_proj += [proj_inds]
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

    def probs_to_preds(self, probs):
        return self.pred_values[np.argmax(probs, axis=1).astype(np.int32)]

    def new_reg_sampling_pts(self, overlap_ratio=1.1):

        # Subsampling size so that the overlap is reduced to the minimum
        reg_dl = self.in_radius * 2
        if not self.use_cubes:
            if self.cylindric_input:
                reg_dl *= 1 / np.sqrt(self.dim - 1)
            else:
                reg_dl *= 1 / np.sqrt(self.dim)

        # Subsampling size with overlap (overlap_ratio should be > 1.0)
        reg_dl *= 1 / max(overlap_ratio, 1.01)

        # Get data        
        all_reg_pts = []
        all_reg_clouds = []
        for cloud_ind, tree in enumerate(self.input_trees):

            # Random offset to vary the border effects
            if self.cylindric_input:
                offset = torch.rand(1, self.dim - 1) * reg_dl
            else:
                offset = torch.rand(1, self.dim) * reg_dl

            # Subsample scene clouds
            points = np.array(tree.data, copy=False).astype(np.float32)
            cpu_points = torch.from_numpy(points)
            sub_points, _ = subsample_pack_batch(cpu_points + offset,
                                                [cpu_points.shape[0]],
                                                reg_dl,
                                                method='grid')

            # Re-adjust sampling points
            reg_points = sub_points - offset
            
            # Add z coordinate in case of cylindric input
            if self.cylindric_input:
                reg_points = torch.cat((reg_points, torch.zeros_like(reg_points[:, :1])), dim=1)

            # Stack points and cloud indices
            all_reg_pts.append(reg_points)
            all_reg_clouds.append(torch.full((reg_points.shape[0],), cloud_ind, dtype=torch.long))

        # Shuffle
        all_reg_pts = torch.concat(all_reg_pts, dim=0)
        all_reg_clouds = torch.concat(all_reg_clouds, dim=0)
        rand_shuffle = torch.randperm(all_reg_clouds.shape[0])

        # Put in queue. Memory is shared automatically
        self.reg_sample_pts = all_reg_pts[rand_shuffle]
        self.reg_sample_clouds = all_reg_clouds[rand_shuffle]

        # Share memory
        self.reg_sample_pts.share_memory_()
        self.reg_sample_clouds.share_memory_()

        return

    def get_votes(self):
        v = 0
        if self.regular_sampling:
            with self.worker_lock:
                reg_sampling_N = float(self.reg_sample_pts.shape[0])
                v = float(self.reg_votes.item())
                v += float(self.reg_sampling_i.item()) / reg_sampling_N
        return v
    
    def sample_random_sphere(self, center_noise=0.1):

        if self.regular_sampling:
            
            with self.worker_lock:
                    
                # Case if we reach the end of the regular sampling points
                reg_sampling_N = int(self.reg_sample_pts.shape[0])
                if self.reg_sampling_i >= reg_sampling_N:
                    if self.set == 'validation':
                        # Recompute new ones if we are in validation
                        self.reg_sampling_i *= 0
                        self.new_reg_sampling_pts()
                        self.reg_votes += 1
                    else:
                        # Stop generating if we are in test 
                        return None, None

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
            if self.cylindric_input:
                center_point = np.hstack((center_point, self.input_z[cloud_ind][point_ind].reshape(1, 1)))

            # Add a small noise to center point
            center_point += np.random.normal(scale=center_noise * self.in_radius, size=center_point.shape)

        return cloud_ind, center_point

    def get_input_area(self, cloud_ind, center_point, only_inds=False):
        """
        This function gets the input area. Depending on parameters, the area is:
            > a sphere of radius R          IF  use_cube = False  and  cylindric_input = False
            > a cylinder of radius R        IF  use_cube = False  and  cylindric_input = True
            > a cube of size 2R             IF  use_cube = True   and  cylindric_input = False
            > a cubic cylinder of size 2R   IF  use_cube = True   and  cylindric_input = True
        """

        # Indices of points in input region
        q_point = center_point
        if self.cylindric_input:
            q_point = q_point[:, :2]

        # Radius of query (larger in case we want a cube)
        r = self.in_radius
        if self.use_cubes:
            if self.cylindric_input:
                r *= np.sqrt(self.dim - 1)
            else:
                r *= np.sqrt(self.dim)

        # Query points
        input_inds = self.input_trees[cloud_ind].query_radius(q_point, r=r)[0]

        # Get points from tree structure
        points = np.array(self.input_trees[cloud_ind].data, copy=False)
        input_points = points[input_inds].astype(np.float32)
        
        # Crop the cube if wanted
        if self.use_cubes:
            cube_mask = np.logical_and(np.all(input_points > q_point - self.in_radius, axis=-1),
                                       np.all(input_points < q_point + self.in_radius, axis=-1))
            input_points = input_points[cube_mask]
            input_inds = input_inds[cube_mask]

        if only_inds:
            return input_inds

        # Get neighbors
        if self.cylindric_input:
            input_points = np.hstack((input_points, self.input_z[cloud_ind][input_inds]))

        # # Center neighbors actually useless here
        # input_points = (query_points - center_point).astype(np.float32)

        # Collect labels and colors
        input_features = self.input_features[cloud_ind][input_inds]

        if self.set in ['test', 'ERF']:
            input_labels = np.zeros(input_points.shape[0], dtype=np.int64)
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
        batch_n_pts = 0

        while True:

            # Pick a sphere center randomly
            cloud_ind, c_point = self.sample_random_sphere()

            # In case we reach the end of the test epoch
            if cloud_ind is None:
                break

            # Get the input sphere
            in_inds, in_points, in_features, in_labels = self.get_input_area(cloud_ind, c_point)
            
            if in_points.shape[0] < 1:
                continue
            
            # Add original height as additional feature
            in_features = np.hstack((in_features, in_points[:, 2:] + c_point[:, 2:])).astype(np.float32)

            # Data augmentation
            in_points, in_features, in_labels = self.augmentation_transform(in_points, in_features, in_labels)

            # Select features for the network
            in_features = self.select_features(in_features)

            # View the arrays as torch tensors
            torch_points = torch.from_numpy(in_points)
            torch_features = torch.from_numpy(in_features)
            torch_labels = torch.from_numpy(in_labels).type(torch.long)


            # Input subsampling (on CPU to be parrallelizable)
            in_points, in_features, in_labels, inv_inds = subsample_cloud(torch_points,
                                                                          self.cfg.model.init_sub_size,
                                                                          features=torch_features,
                                                                          labels=torch_labels,
                                                                          method=self.cfg.model.sub_mode,
                                                                          return_inverse=True)

            # Stack batch
            p_list += [in_points]
            f_list += [in_features]
            l_list += [in_labels]
            pi_list += [torch.from_numpy(in_inds)]
            pinv_list += [inv_inds]
            i_list += [torch.from_numpy(c_point)]
            ci_list += [cloud_ind]

            # Update batch size
            batch_n_pts += int(in_points.shape[0])

            # In case batch is full, stop
            if batch_n_pts > int(self.b_lim):
                break

        

        #####################
        # Handle epmpty batch
        #####################

        # Return empty input list
        if len(p_list) < 1:
            input_dict = EasyDict()
            input_dict.points = []
            # Wait to let time for other worker to return their last batch
            time.sleep(0.1)
            return input_dict
            

        ###################
        # Concatenate batch
        ###################
        
        stacked_points = torch.cat(p_list, dim=0)
        stacked_features = torch.cat(f_list, dim=0)
        stacked_labels = torch.cat(l_list, dim=0)
        center_points = torch.cat(i_list, dim=0)
        cloud_inds = torch.LongTensor(ci_list)
        input_inds = torch.cat(pi_list, dim=0)
        input_invs = torch.cat(pinv_list, dim=0)
        stack_lengths = torch.LongTensor([int(pp.shape[0]) for pp in p_list])
        stack_lengths0 = torch.LongTensor([int(pp.shape[0]) for pp in pi_list])


        #######################
        # Create network inputs
        #######################
        #
        #   Points, features, etc.
        #

        # Get the whole input list
        if self.precompute_pyramid:

            input_dict = build_full_pyramid(stacked_points,
                                            stack_lengths,
                                            len(self.cfg.model.layer_blocks),
                                            self.cfg.model.init_sub_size,
                                            self.cfg.model.init_sub_size * self.cfg.model.kp_radius,
                                            self.cfg.model.neighbor_limits,
                                            sub_mode=self.cfg.model.sub_mode)

        else:

            input_dict = build_base_pyramid(stacked_points,
                                            stack_lengths)

        # Add other input to the pyramid dictionary
        input_dict.features = stacked_features
        input_dict.labels = stacked_labels
        input_dict.lengths0 = stack_lengths0
        input_dict.cloud_inds = cloud_inds
        input_dict.center_points = center_points
        input_dict.input_inds = input_inds
        input_dict.input_invs = input_invs

        return input_dict

    def calib_batch_size(self, samples=20, verbose=True):

        t0 = time.time()

        # Get gpu for faster calibration
        device = init_gpu()

        # Get augmentation transform
        calib_augment = ComposeAugment(self.augmentation_transform.transforms[:4])
        
        all_batch_n = []
        all_batch_n_pts = []
        for i in range(samples):

            batch_n = 0
            batch_n_pts = 0
            while True:
                cloud_ind, center_p = self.sample_random_sphere()
                _, in_points, _, _ = self.get_input_area(cloud_ind, center_p)
                in_points, _, _ = calib_augment(in_points, None, None)
                if in_points.shape[0] > 0:
                    gpu_points = torch.from_numpy(in_points).to(device)
                    sub_points, _ = subsample_pack_batch(gpu_points,
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

    def calib_batch_limit(self, batch_size, samples=100, verbose=True):
        """
        Find the batch_limit given the target batch_size. 
        The batch size varies randomly so we prefer a quick calibration to find 
        an approximate batch limit.
        """

        t0 = time.time()

        # Get gpu for faster calibration
        device = init_gpu()

        # Get augmentation transform
        calib_augment = ComposeAugment(self.augmentation_transform.transforms[:4])

        # Advanced display
        pi = 0
        pN = samples
        progress_n = 30
        fmt_str = '[{:<' + str(progress_n) + '}] {:5.1f}%'
        print('\nSearching batch_limit given the target batch_size.')

        # First get a avg of the pts per point cloud
        all_cloud_n = []
        while len(all_cloud_n) < samples:
            cloud_ind, center_p = self.sample_random_sphere()
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
                gpu_points = torch.from_numpy(in_points).to(device)
                sub_points, _ = subsample_pack_batch(gpu_points,
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
        new_b_lim = mean_cloud_n * batch_size

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

        # After calibration reset counters for regular sampling
        self.reg_sampling_i *= 0

        print('\n')

        return

    def calib_neighbors(self, cfg, samples=100, verbose=True):

        t0 = time.time()

        # Get gpu for faster calibration
        device = init_gpu()

        # Get augmentation transform
        calib_augment = ComposeAugment(self.augmentation_transform.transforms[:4])
        
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
            cloud_ind, center_p = self.sample_random_sphere()
            _, in_points, _, _ = self.get_input_area(cloud_ind, center_p)
            
            if in_points.shape[0] > 0:
                in_points, _, _ = calib_augment(in_points, None, None)
                gpu_points = torch.from_numpy(in_points).to(device)
                sub_points, _ = subsample_pack_batch(gpu_points,
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

        # After calibration reset counters for regular sampling
        self.reg_sampling_i *= 0

        return




# ----------------------------------------------------------------------------------------------------------------------
#
#           Utility classes definition
#       \********************************/


class SceneSegSampler(Sampler):
    """Sampler for SceneSegDataset"""

    def __init__(self, dataset: SceneSegDataset):
        Sampler.__init__(self, dataset)

        # Dataset used by the sampler (no copy is made in memory)
        self.dataset = dataset

        # Number of step per epoch
        if dataset.set == 'training':
            self.N = dataset.cfg.train.steps_per_epoch * dataset.cfg.train.accum_batch
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


class SceneSegBatch:
    """Custom batch definition with memory pinning for SceneSegDataset"""

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



def SceneSegCollate(batch_data):
    return SceneSegBatch(batch_data)


