import warnings
import logging
import pickle
from functools import reduce, partial
from functional import seq


import numpy as np

from som_vae.settings import config, skeleton

def _load_positional_data_(path):
    with open(path, 'rb') as f:
        pose_data_raw = pickle.load(f)
        return pose_data_raw['points2d']


def _check_shape_(joint_positions):
    """ should be (7, <nb frames>, 38, 2)
    7 for the images, some should be 0 because it didn't record the images for these points
    1000 for the nb of frames
    38 for the features (some for the legs, antennae ...) check skeleton.py in semigh's code
    2 for the pose dimensions
    """
    s = joint_positions.shape

    if s[0] != config.NB_CAMERAS or s[2] != len(skeleton.tracked_points) or s[3] != config.NB_RECORDED_DIMESIONS:
        raise ValueError(f"shape of pose data is wrong, it's {joint_positions.shape}")

    return joint_positions


def _crude_value_check_(joint_positions):
    if np.sum(joint_positions == 0) == np.product(joint_positions.shape):
        raise ValueError('not every value should be zero')

    return joint_positions


def _simple_checks_(data):
    return reduce(lambda acc, el: el(acc), [_check_shape_, _crude_value_check_], data)


def _get_camera_of_interest_(joint_positions, camera_idx=config.CAMERA_OF_INTEREST):
    return joint_positions[camera_idx]


def _get_visible_legs_(joint_positions, camera_idx=config.CAMERA_OF_INTEREST):
    idx_visible_joints = [skeleton.camera_see_joint(camera_idx, j) for j in range(len(skeleton.tracked_points))]
    return joint_positions[:, idx_visible_joints, :]


def get_positional_data(path):
    fns = [_load_positional_data_, _simple_checks_, _get_camera_of_interest_, _get_visible_legs_]
    return reduce(lambda acc, el: el(acc), fns, path)


def add_third_dimension(joint_positions):
    # just add a z-axis
    # look up np.pad...
    # assumes that the positional (channels) data is in the last axis
    paddings = [[0, 0] for i in joint_positions.shape]
    paddings[-1][1] = 1

    return np.pad(joint_positions, paddings, mode='constant', constant_values=0)

def get_only_first_legs(joint_positions):
    logging.warn('this works only for the first legs!')
    return joint_positions[:, list(range(len(config.LEGS) * config.NB_TRACKED_POINTS)), :]

def normalize(joint_positions, using_median=True, to_probability_distr=False):
    # alternatives could be to use only the median of the first joint -> data is then fixed to top (is that differnt to now?)
    # TODO clean up signature.
    #warnings.warn('here in normalize signature is deprecated')
    applied = np.median(joint_positions.reshape(-1, joint_positions.shape[-1]), axis=0)
    return joint_positions - applied, applied

def normalize_ts(time_series, ax=0):
    # for shape (frame,feat)
    eps = 0.0001
    print("shapes:", np.shape(np.transpose(time_series)), np.shape(np.mean(np.transpose(time_series), axis=ax)))
#     n_time_series = (np.transpose(time_series) - np.mean(np.transpose(time_series), axis=ax))/(np.std(np.transpose(time_series), axis=ax) + eps)
    norm = np.sum(np.transpose(time_series), axis=ax); norm = np.transpose(norm) #shape = 1,frames
    n_time_series = np.transpose(time_series) / np.sum(np.transpose(time_series), axis=ax)
    n_time_series = np.transpose(n_time_series)
#     n_time_series = np.zeros(shape=np.shape(time_series))
#     for i in range(np.shape(time_series)[1]):
#         n_time_series[:,i] = (time_series[:,i] - np.mean(time_series[:,i])) / (np.std(time_series[:,i]) + eps)
    return n_time_series, norm


def normalize_pose(points3d, median3d=False):
    # normalize experiment
    if median3d:
        points3d -= np.median(points3d.reshape(-1, 3), axis=0)
    else:
        for i in range(np.shape(points3d)[1]): #frames
            for j in range(np.shape(points3d)[2]): #xyz
                points3d[:,i,j] = normalize_ts(points3d[:,i,j])
    return points3d


def get_data_and_normalization(data, per_experiment=False):
    ret = seq(data).map(partial(config.positional_data))\
                    .filter(lambda x: x is not None)\
                    .map(_simple_checks_)\
                    .map(_get_camera_of_interest_)\
                    .map(_get_visible_legs_)\
                    .map(add_third_dimension)\
                    .map(get_only_first_legs)\
                    .to_list()

    if per_experiment:
        return ret
    else:
        return normalize(np.vstack(ret))


def get_frames_with_idx_and_labels(data):
    frames_idx_with_labels = seq(data)\
        .flat_map(lambda x: [(i, x.label) for i in range(*x.sequence)]).to_pandas()
    frames_idx_with_labels.columns = ['frame_id_in_experiment', 'label']

    return frames_idx_with_labels
