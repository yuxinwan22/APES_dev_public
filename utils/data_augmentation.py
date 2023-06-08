import numpy as np


def jitter(pcd, std=0.01, clip=0.05):
    num_points, num_features = pcd.shape  # pcd.shape == (N, 3)
    jittered_point = np.clip(std * np.random.randn(num_points, num_features), -clip, clip)
    jittered_point += pcd
    return jittered_point


def rotate(pcd, which_axis, angle_range):
    angle = np.random.uniform(angle_range[0], angle_range[1])
    angle = np.pi * angle / 180
    cos_theta = np.cos(angle)
    sin_theta = np.sin(angle)
    if which_axis == 'x':
        rotation_matrix = np.array([[1, 0, 0], [0, cos_theta, sin_theta], [0, -sin_theta, cos_theta]])
    elif which_axis == 'y':
        rotation_matrix = np.array([[cos_theta, 0,  -sin_theta], [0, 1, 0], [sin_theta, 0, cos_theta]])
    elif which_axis == 'z':
        rotation_matrix = np.array([[cos_theta, sin_theta, 0], [-sin_theta, cos_theta, 0], [0, 0, 1]])
    else:
        raise ValueError(f'which_axis should be one of x, y and z, but got {which_axis}!')
    rotated_points = pcd @ rotation_matrix
    return rotated_points


def translate(pcd, x_range, y_range, z_range):
    num_points = pcd.shape[0]
    x_translation = np.random.uniform(x_range[0], x_range[1])
    y_translation = np.random.uniform(y_range[0], y_range[1])
    z_translation = np.random.uniform(z_range[0], z_range[1])
    x = np.full(num_points, x_translation)
    y = np.full(num_points, y_translation)
    z = np.full(num_points, z_translation)
    translation = np.stack([x, y, z], axis=-1)
    return pcd + translation


def anisotropic_scale(pcd, x_range, y_range, z_range):
    x_factor = np.random.uniform(x_range[0], x_range[1])
    y_factor = np.random.uniform(y_range[0], y_range[1])
    z_factor = np.random.uniform(z_range[0], z_range[1])
    scale_matrix = np.array([[x_factor, 0, 0], [0, y_factor, 0], [0, 0, z_factor]])
    scaled_points = pcd @ scale_matrix
    return scaled_points
