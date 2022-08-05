#
#
#      0=================================0
#      |    Kernel Point Convolutions    |
#      0=================================0
#
#
# ----------------------------------------------------------------------------------------------------------------------
#
#      Configuration class
#
# ----------------------------------------------------------------------------------------------------------------------
#
#      Hugues THOMAS - 11/06/2018
#

import os
from os import makedirs, getcwd
from os.path import join, basename, dirname, realpath, exists
import numpy as np
import json
import time

from easydict import EasyDict


# Colors for printing
class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def init_cfg():

    # Use easydict for parameters
    cfg = EasyDict()

    # Experiment specification
    # ------------------------

    cfg.exp = EasyDict()
    cfg.exp.working_dir = ''            #   Str, current working directory
    cfg.exp.date = ''                   #   Str, date of the strat of the experiment
    cfg.exp.results_dir = ''            #   Str, directory for all results
    cfg.exp.log_dir = ''                #   Str, directory where this experiemnt is saved
    cfg.exp.seed = 42                   #   Int, seed for random stuff
    cfg.exp.saving = True               #  Bool, is the experiment saved or not


    # Data parameters
    # ---------------

    cfg.data = EasyDict()
    cfg.data.name = ''                  #   Str, name of the dataset
    cfg.data.path = ''                  #   Str, path of the dataset
    cfg.data.task = ''                  #   Str, name of the task performed

    cfg.data.dim = 3                    #   Int, dimension of data points
    cfg.data.num_classes = 2            #   Int, Number of classes

    cfg.data.label_and_names = []       #  List, value and name of each label
    cfg.data.label_names = []           #  List, name of each label
    cfg.data.label_values = []          #  List, value of each label

    cfg.data.name_to_label = {}         #  Dict, get label values from names
    cfg.data.name_to_idx = {}           #  Dict, get label index from values

    cfg.data.ignored_labels = []        #  List, value of ignored label values

    cfg.data.sub_size = -1.0            # Float, data subampling size, negative value means we use original data


    # Network parameters
    # ------------------

    cfg.model = EasyDict()
    cfg.model.kp_mode = 'kpconv'        #   Str, choice of basic operator ('kpconv', 'kpinv', 'kpdef')
    cfg.model.kernel_size = 15          #   Int, number of kernel points
    cfg.model.kp_radius = 2.5           # Float, radius of the basic operator
    cfg.model.kp_sigma = 1.0            # Float, radius of the kernel point influence
    cfg.model.kp_influence = 'linear'   #   Str, Influence function when d < KP_extent. ('constant', 'linear', 'gaussian')
    cfg.model.kp_aggregation = 'sum'    #   Str, Aggregation mode ('nearest', 'sum')
    cfg.model.kp_fixed = 'center'       #   Str, Fixed points in the kernel ('none', 'center', 'verticals')
    cfg.model.conv_groups = 1           #   Int, number of groups for group conv

    cfg.model.kp_deform_w = 1.0         # Float, multiplier for deformation the fitting/repulsive loss
    cfg.model.kp_deform_grad = 0.1      # Float, multiplier for deformation gradient
    cfg.model.kp_repulse_dist = 1.0     # Float, distance of repulsion for deformed kernel points
    
    cfg.model.init_sub_size = 0.04      # Float, initial subampling size, (voxel size, lattice size of fps minimum distance)
    cfg.model.sub_mode = 'ph'           #   Str, subsampling mode ('grid', 'ph', 'fps')

    cfg.model.input_channels = 1        #   Int, dimension of input feaures
    cfg.model.init_channels = 64        #   Int, dimension of first network features
    cfg.model.norm = 'batch'            #   Str, type of normalization in the network ('group', 'batch', 'none')
    cfg.model.bn_momentum = 0.98        # Float, Momentum for batch normalization.

    cfg.model.layer_blocks = (2, 1, 1)  # Tuple, number of blocks in each layers (in addition to the strided ones in between).
    cfg.model.neighbor_limits = []      #  List, maximum number of neigbors per layers

    cfg.model.process_ratio = 1.0       # Float, ratio between the radius of processed volume and the radius of the input volume
    cfg.model.n_frames = 1              #   Int, number of frames used (Specific to SLAM)


    # Training parameters
    # -------------------

    # train
    cfg.train = EasyDict()
    cfg.train.num_workers = 16          #   Int, number of parallel workers for input dataset
    cfg.train.checkpoint_gap = 50       #   Int, gap between each saved checkpoint

    cfg.train.max_epoch = 300           #   Int, number of training epochs
    cfg.train.steps_per_epoch = 1000    #   Int, number of steps per epoch

    cfg.train.in_radius = 1.0           # Float, radius of the input sphere
    cfg.train.batch_size = 8            #   Int, number of input point cloud per batch
    cfg.train.batch_limit = -1          #   Int, maximum number of points in total in a batch
    cfg.train.max_points = -1           #   Int, maximum number of points per element (randomly drop the excedent)
    
    cfg.train.optimizer = 'SGD'         #   Str, optimizer ('SGD' or 'Adam')
    cfg.train.lr = 1e-2                 # Float, initial learning rate
    cfg.train.sgd_momentum = 0.95       # Float, learning rate momentum for sgd optimizer
    cfg.train.adam_b = (0.9, 0.999)     # Float, betas for Adam optimizer
    cfg.train.adam_eps = 1e-08          # Float, eps for Adam optimizer
    cfg.train.weight_decay = 1e-4       # Float, weight decay
    cfg.train.lr_decays = {'10': 0.1}   #  Dict, decay values with their epoch {epoch: decay}
    cfg.train.warmup = True             #  Bool, should the first epoch be a warmup
    cfg.train.grad_clip = 100.0         #   Int, gradient clipping value (negative means no clipping)
    cfg.train.class_w = []              #  List, weight for each class in the segmentation loss

    cfg.train.augment_anisotropic = True    #  Bool, Should he scale augmentatio nbe anisotropic
    cfg.train.augment_min_scale = 0.9       # Float, min scaling value
    cfg.train.augment_max_scale = 1.1       # Float, max scaling value
    cfg.train.augment_symmetries = []       #  List, symmetries (boolean for each dimension)
    cfg.train.augment_rotation = 'none'     #   Str, type of rotation augmentation ('none', 'vertical', 'all')
    cfg.train.augment_noise = 0.005         # Float, normal offset noise sigma value
    cfg.train.augment_color = 0.7           # Float, probability to drop input features

    cfg.train.segloss_balance = 'none'      #   Str, Respectively each point, class, or cloud in the batch has
                                            #        the same loss contribution ('none', 'class', 'batch'). 
    
    
    # Test parameters
    # ---------------

    # test
    cfg.test = EasyDict()
    cfg.test.num_workers = 16           #   Int, number of parallel workers for input dataset.

    cfg.test.max_votes = 10             #   Int, number of training epochs
    cfg.test.steps_per_epoch = 100      #   Int, number of steps per epoch

    cfg.test.in_radius = 1.0            # Float, radius of the input sphere
    cfg.test.batch_size = 8             #   Int, number of input point cloud per batch
    cfg.test.batch_limit = -1           #   Int, maximum number of points in total in a batch
    cfg.test.max_points = -1            #   Int, maximum number of points per element (randomly drop the excedent)


    return cfg


def get_directories(cfg, date=None, seed=None):
    
    # Get date unless it is given
    if date is None:
        cfg.exp.date = time.strftime('Log_%Y-%m-%d_%H-%M-%S')
    else:
        cfg.exp.date = date
    cfg.exp.working_dir = getcwd() # dirname(realpath(__file__))
    cfg.exp.results_dir = join(cfg.exp.working_dir, "results")
    cfg.exp.log_dir = join(cfg.exp.results_dir, cfg.exp.date)
    if seed is not None:
        cfg.exp.seed = seed

    if not exists(cfg.exp.log_dir):
        makedirs(cfg.exp.log_dir)

    return

def save_cfg(cfg):
    # Serialize data into file:
    with open(join(cfg.exp.log_dir, 'parameters.json'), "w") as jsonfile:
        json.dump(cfg, jsonfile, indent=4, sort_keys=False)
    return

def load_cfg(log_path):

    # Create an empty cfg
    cfg = init_cfg()

    # Read data from file:
    with open(join(log_path, 'parameters.json'), "r") as jsonfile:
        cfg.update(json.load(jsonfile))

    return cfg



    # def load0(self, log_path):



    #     # List all parameters
    #     config_names = [['exp', 'Experiment parameters'],
    #                     ['data', 'Data parameters'],
    #                     ['model', 'Model parameters'],
    #                     ['train', 'Training parameters'],
    #                     ['test', 'Test parameters']]

    #     filename = join(log_path, 'parameters.txt')
    #     with open(filename, 'r') as f:
    #         lines = f.readlines()

        
    #     title_lines = []
    #     start_lines = []
    #     for config_name, config_title in config_names:
    #         for i, lines in enumerate(lines):
    #             if config_title in line:
    #                 title_lines.append(i)
    #                 start_lines.append(i)
    #                 break

            
    #         while not lines[start_lines[-1]].startswith('{'):
    #             start_lines[-1] += 1



    #     for start_i in start_lines:

    #         config_name, config_title = config_names[start_i]

    #         i = start_i
    #         while i < len():

    #             i += 1



    #     str_dict = ''.join(lines)
        
    #     easy_dict = EasyDict(eval(str_dict))

    #     setattr(self, config_name, easy_dict)






    #     # Class variable dictionary
    #     for line in lines:
    #         line_info = line.split()
    #         if len(line_info) > 2 and line_info[0] != '#':

    #             if line_info[2] == 'None':
    #                 setattr(self, line_info[0], None)

    #             elif line_info[0] == 'lr_decay_epochs':
    #                 self.lr_decays = {int(b.split(':')[0]): float(b.split(':')[1]) for b in line_info[2:]}

    #             elif line_info[0] == 'architecture':
    #                 self.architecture = [b for b in line_info[2:]]

    #             elif line_info[0] == 'augment_symmetries':
    #                 self.augment_symmetries = [bool(int(b)) for b in line_info[2:]]

    #             elif line_info[0] == 'num_classes':
    #                 if len(line_info) > 3:
    #                     self.num_classes = [int(c) for c in line_info[2:]]
    #                 else:
    #                     self.num_classes = int(line_info[2])

    #             elif line_info[0] == 'class_w':
    #                 self.class_w = [float(w) for w in line_info[2:]]

    #             elif hasattr(self, line_info[0]):
    #                 attr_type = type(getattr(self, line_info[0]))
    #                 if attr_type == bool:
    #                     setattr(self, line_info[0], attr_type(int(line_info[2])))
    #                 else:
    #                     setattr(self, line_info[0], attr_type(line_info[2]))

    # def save0(self):

        

    #     # List all parameters
    #     config_names = [['exp', 'Experiment parameters'],
    #                     ['data', 'Data parameters'],
    #                     ['model', 'Model parameters'],
    #                     ['train', 'Training parameters'],
    #                     ['test', 'Test parameters']]

    #     with open(join(self.saving_path, 'parameters.txt'), "w") as text_file:

    #         for config_name, config_title in config_names:

    #             # Title of the config category
    #             str = '# ' + config_title + '\n'
    #             text_file.write('# ' + config_title + '\n')
    #             text_file.write('# ' + '*' * len(config_title) + '\n\n')
                
    #             # Verify element
    #             if hasattr(self, config_name):
    #                 conf_dict = getattr(self, config_name)
    #             else:
    #                 raise ValueError('Wrong config definition. Parameter category {:s} is not defined'.format(config_name))

    #             if not isinstance(conf_dict, EasyDict):
    #                 raise ValueError('Wrong config definition. Parameter category {:s} is not an EasyDict'.format(config_name))

    #             text_dict = json.dumps(conf_dict, sort_keys=False, indent=4)

    #             text_file.write(text_dict)
    #             text_file.write('\n\n')

