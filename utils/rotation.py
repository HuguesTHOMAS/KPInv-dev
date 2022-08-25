import numpy as np

import torch

def create_3D_rotations(axis, angle):
    """
    Create rotation matrices from a list of axes and angles. Code from wikipedia on quaternions
    :param axis: float32[N, 3]
    :param angle: float32[N,]
    :return: float32[N, 3, 3]
    """

    t1 = np.cos(angle)
    t2 = 1 - t1
    t3 = axis[:, 0] * axis[:, 0]
    t6 = t2 * axis[:, 0]
    t7 = t6 * axis[:, 1]
    t8 = np.sin(angle)
    t9 = t8 * axis[:, 2]
    t11 = t6 * axis[:, 2]
    t12 = t8 * axis[:, 1]
    t15 = axis[:, 1] * axis[:, 1]
    t19 = t2 * axis[:, 1] * axis[:, 2]
    t20 = t8 * axis[:, 0]
    t24 = axis[:, 2] * axis[:, 2]
    R = np.stack([t1 + t2 * t3,
                  t7 - t9,
                  t11 + t12,
                  t7 + t9,
                  t1 + t2 * t15,
                  t19 - t20,
                  t11 - t12,
                  t19 + t20,
                  t1 + t2 * t24], axis=1)

    return np.reshape(R, (-1, 3, 3))

def get_random_rotations(shape=None):
    """
    Creates random rotations in the shape asked
    :param shape: a list/tuple of the wanted shape: (d1, ..., dn). If None, returns a single rotation
    :return: AS many rotations as asked, shaped like asked: (d1, ..., dn, 3, 3)
    """

    if shape is None:
        shape_tuple = (3, 3)
        n_rot = 1
    else:
        shape_tuple = tuple(shape) + (3, 3)
        n_rot = np.prod(shape)

    # Choose two random angles for the first vector in polar coordinates
    theta = np.random.rand(n_rot) * 2 * np.pi
    phi = (np.random.rand(n_rot) - 0.5) * np.pi

    # Create the first vector in carthesian coordinates
    u = np.stack([np.cos(theta) * np.cos(phi), np.sin(theta) * np.cos(phi), np.sin(phi)], axis=1)

    # Choose a random rotation angle
    alpha = np.random.rand(n_rot) * 2 * np.pi

    # Create the rotation matrix with this vector and angle
    R = create_3D_rotations(u, alpha)

    return R.reshape(shape_tuple)
    
def get_random_vertical_rotations(shape=None):
    """
    Creates random rotations in the shape asked
    :param shape: a list/tuple of the wanted shape: (d1, ..., dn). If None, returns a single rotation
    :return: AS many rotations as asked, shaped like asked: (d1, ..., dn, 3, 3)
    """

    if shape is None:
        shape_tuple = (3, 3)
        n_rot = 1
    else:
        shape_tuple = tuple(shape) + (3, 3)
        n_rot = np.prod(shape)

    # Create the first vector in carthesian coordinates
    u = np.zeros((n_rot, 3), dtype=np.float32)
    u[:, 2] = 1

    # Choose a random rotation angle
    theta = np.random.rand(n_rot) * 2 * np.pi

    # Create the rotation matrix with this vector and angle
    R = create_3D_rotations(u, theta)

    return R.reshape(shape_tuple)
