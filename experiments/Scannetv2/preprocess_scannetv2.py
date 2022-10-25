

# TODO: See these file: 
#       https://github.com/chrischoy/SpatioTemporalSegmentation-ScanNet/blob/master/lib/datasets/scannet.py
#       https://github.com/chrischoy/SpatioTemporalSegmentation-ScanNet/blob/master/lib/sparse_voxelization/voxelizer_wrapper.py
#
#       and my old one:
#       https://github.com/HuguesTHOMAS/KPConv/blob/16bfbb96ac9af48a3c829d1f8123152d35b63862/datasets/Scannet.py#L164


from itertools import product
from pathlib import Path
from random import shuffle

import numpy as np

from lib.pc_utils import read_plyfile, save_point_cloud


SCANNET_RAW_PATH = Path('/data/chrischoy/datasets/scannet_raw')
SCANNET_OUT_PATH = Path('/data/chrischoy/datasets/scannet_no_crop')
TRAIN_DEST = 'train'
TEST_DEST = 'test'
SUBSETS = {TRAIN_DEST: 'scans', TEST_DEST: 'scans_test'}
POINTCLOUD_FILE = '_vh_clean_2.ply'
CROP_SIZE = -1
BUGS = {
    'train/scene0270_00_*.ply': 50,
    'train/scene0270_02_*.ply': 50,
    'train/scene0384_00_*.ply': 149,
}


# Preprocess data.
for out_path, in_path in SUBSETS.items():
  phase_out_path = SCANNET_OUT_PATH / out_path
  phase_out_path.mkdir(parents=True, exist_ok=True)
  for f in (SCANNET_RAW_PATH / in_path).glob('*/*' + POINTCLOUD_FILE):
    # Load pointcloud file.
    pointcloud = read_plyfile(f)
    # Make sure alpha value is meaningless.
    assert np.unique(pointcloud[:, -1]).size == 1
    # Load label file.
    label_f = f.parent / (f.stem + '.labels' + f.suffix)
    if label_f.is_file():
      label = read_plyfile(label_f)
      # Sanity check that the pointcloud and its label has same vertices.
      assert pointcloud.shape[0] == label.shape[0]
      assert np.allclose(pointcloud[:, :3], label[:, :3])
    else:  # Label may not exist in test case.
      label = np.zeros_like(pointcloud)
    xyz = pointcloud[:, :3]
    if CROP_SIZE > 0:
      xyz_range = xyz.max(0) - xyz.min(0)
      crop_half = CROP_SIZE / 2
      crop_bound = np.maximum(np.ceil(xyz_range / crop_half), 2).astype(int)
      crop_min = xyz.min(0) - (crop_bound * crop_half - xyz_range) / 2

    all_points = np.empty((0, 3))
    if CROP_SIZE > 0:
      for i, crop_idx in enumerate(product(*map(range, crop_bound - 1))):
        curr_crop_min = crop_min + np.array(crop_idx) * crop_half
        mask = np.logical_and(np.all(xyz > curr_crop_min, 1),
                              np.all(xyz <= curr_crop_min + crop_half * 2, 1))
        if not np.any(mask):
          continue
        processed = np.hstack((pointcloud[mask, :6], np.array([label[mask, -1]]).T))
        # Check if the crop size is correctly applied.
        assert np.all((processed.max(0) - processed.min(0))[:3] < CROP_SIZE)
        out_f = phase_out_path / (f.name[:-len(POINTCLOUD_FILE)] + f'_{i:02}' + f.suffix)
        save_point_cloud(processed, out_f, with_label=True, verbose=False)
        all_points = np.vstack((all_points, xyz[mask]))
    else:
      processed = np.hstack((pointcloud[:, :6], np.array([label[:, -1]]).T))

    # Check that all points are included in the crops.
    assert set(tuple(l) for l in all_points.tolist()) == set(tuple(l) for l in xyz.tolist())

# Split trainval data to train/val according to scene.
trainval_files = [f.name for f in (SCANNET_OUT_PATH / TRAIN_DEST).glob('*.ply')]
trainval_scenes = list(set(f.split('_')[0] for f in trainval_files))
shuffle(trainval_scenes)
num_train = int(len(trainval_scenes))
train_scenes = trainval_scenes[:num_train]
val_scenes = trainval_scenes[num_train:]

# Collect file list for all phase.
train_files = [f'{TRAIN_DEST}/{f}' for f in trainval_files if any(s in f for s in train_scenes)]
val_files = [f'{TRAIN_DEST}/{f}' for f in trainval_files if any(s in f for s in val_scenes)]
test_files = [f'{TEST_DEST}/{f.name}' for f in (SCANNET_OUT_PATH / TEST_DEST).glob('*.ply')]

# Data sanity check.
assert not set(train_files).intersection(val_files)
assert all((SCANNET_OUT_PATH / f).is_file() for f in train_files)
assert all((SCANNET_OUT_PATH / f).is_file() for f in val_files)
assert all((SCANNET_OUT_PATH / f).is_file() for f in test_files)

# Write file list for all phase.
with open(SCANNET_OUT_PATH / 'train.txt', 'w') as f:
  f.writelines([f + '\n' for f in train_files])
with open(SCANNET_OUT_PATH / 'val.txt', 'w') as f:
  f.writelines([f + '\n' for f in val_files])
with open(SCANNET_OUT_PATH / 'test.txt', 'w') as f:
  f.writelines([f + '\n' for f in test_files])

# Fix bug in the data.
# for files, bug_index in BUGS.items():
#   for f in SCANNET_OUT_PATH.glob(files):
#     pointcloud = read_plyfile(f)
#     bug_mask = pointcloud[:, -1] == bug_index
#     print(f'Fixing {f} bugged label {bug_index} x {bug_mask.sum()}')
#     pointcloud[bug_mask, -1] = 0
#     save_point_cloud(pointcloud, f, with_label=True, verbose=False)