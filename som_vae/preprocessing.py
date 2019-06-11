import warnings
import logging
import pickle
from functools import reduce, partial
from functional import seq


import numpy as np

from som_vae.settings import config, skeleton
from som_vae.settings.config import SetupConfig

def _load_positional_data_(path):
    raise DeprecationWarning
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

    if s[0] != SetupConfig.value('n_cameras') \
            or s[2] != len(skeleton.tracked_points) \
            or s[3] != SetupConfig.value('n_recorded_dimesions'):
        raise ValueError(f"shape of pose data is wrong, it's {joint_positions.shape}")

    return joint_positions


def _crude_value_check_(joint_positions):
    if np.sum(joint_positions == 0) == np.product(joint_positions.shape):
        raise ValueError('not every value should be zero')

    return joint_positions


def _simple_checks_(data):
    return reduce(lambda acc, el: el(acc), [_check_shape_, _crude_value_check_], data)


def _get_camera_of_interest_(joint_positions, camera_idx=SetupConfig.value('camera_of_interest')):
    return joint_positions[camera_idx]


def _get_visible_legs_(joint_positions, camera_idx=SetupConfig.value('camera_of_interest')):
    idx_visible_joints = [skeleton.camera_see_joint(camera_idx, j) for j in range(len(skeleton.tracked_points))]
    return joint_positions[:, idx_visible_joints, :]


def get_positional_data(path):
    raise DeprecationWarning('use `get_data_and_normalization`')
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
    return joint_positions[:, list(range(len(SetupConfig.value('legs')) *
                                         SetupConfig.value('n_tracked_points'))), :]

def normalize(joint_positions):
    # alternatives could be to use only the median of the first joint -> data is then fixed to top (is that differnt to now?)
    # TODO clean up signature.
    #warnings.warn('here in normalize signature is deprecated')
    applied = np.median(joint_positions.reshape(-1, joint_positions.shape[-1]), axis=0)
    return joint_positions - applied, applied



def get_frames_with_idx_and_labels(data):
    frames_idx_with_labels = seq(data)\
        .flat_map(lambda x: [(i, x.label) for i in range(*x.sequence)]).to_pandas()
    frames_idx_with_labels.columns = ['frame_id_in_experiment', 'label']

    return frames_idx_with_labels


def _angle_three_points_(a, b, c):
    """
    Given a set of any 3 points, (a,b,c), returns the angle ba^bc.
    """
    ba = a - b
    bc = c - b
    denom = (np.linalg.norm(ba) * np.linalg.norm(bc))

    if np.abs(denom) <= 1e-8:
        raise ValueError('denom too small')

    cosine_angle = np.dot(ba, bc) / denom
    return np.arccos(cosine_angle)


def _convert_3d_to_angle_(data):
    data_angle = np.zeros((data.shape[0], data.shape[1]), dtype=np.float32)
    joint_blacklist = [skeleton.is_body_coxa,
                       skeleton.is_tarsus_tip,
                       skeleton.is_stripe,
                       skeleton.is_antenna]

    for img_id in range(data.shape[0]):
        for j_id in range(1, data.shape[1]-1):
            if any([fn(j_id) for fn in joint_blacklist]):
                continue
            data_angle[img_id, j_id] = _angle_three_points_(
                data[img_id, j_id - 1, :],
                data[img_id, j_id, :],
                data[img_id, j_id + 1, :])

    data_angle[np.isnan(data_angle) | np.isinf(data_angle)] = 0
    return data_angle

def preprocess_3d_angle_data(frame_data, frame_labels, low_variance_cutoff=0., blacklist_behavior=None, normalize_features=True):
    """ Full preprocessing pipeline for 3d angle data
    """
    frame_data = _convert_3d_to_angle_(frame_data)

    selected_cols = np.where(np.var(frame_data, axis=0) > low_variance_cutoff)[0]
    frame_data = frame_data[:, selected_cols]

    if len(blacklist_behavior) > 0:
        black_idx = np.array(seq(frame_labels).map(lambda x: x[1].label.value in blacklist_behavior).to_list())

        frame_data = frame_data[~black_idx]
        frame_labels = np.array(frame_labels)[~black_idx]

    if normalize_features:
        frame_data, normalisation_factors = normalize(frame_data)
    else:
        normalisation_factors = None

    return frame_data, frame_labels, selected_cols, normalisation_factors


