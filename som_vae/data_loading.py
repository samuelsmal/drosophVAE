"""Here you'll find the overall definitions

"""
import json
import pickle
import pathlib
import numpy as np
from functools import partial
from functional import seq

from som_vae.settings import skeleton
from som_vae.settings.data import EXPERIMENTS, LABELLED_SEQUENCES, experiment_key, Behavior, LabelledSequence
from som_vae.settings.config import DataType


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
            data_angle[img_id, j_id] = angle_three_points(
                data[img_id, j_id - 1, :],
                data[img_id, j_id, :],
                data[img_id, j_id + 1, :])

    data_angle[np.isnan(data_angle) | np.isinf(data_angle)] = 0
    return data_angle

def _load_and_fix_(sequence, data):
    # I have no clue why I need to do this. Semihg knows of this data problem.
    e = min(sequence.sequence[1], data.shape[0])
    s = LabelledSequence(**{**sequence._asdict(),
                                   'sequence': (sequence.sequence[0], e)})

    d = data[sequence.sequence[0]:e]

    return s, d


def load_labelled_data(run_config, setup_config):
    """
    Parameters
    ----------
        data_type

    Returns
    -------
        frame_data, one single numpy array
        frame_labels, lookup table (frame_data_pos, Label)
    """
    data_type = run_config['data_type']

    dim = '3d' if data_type == DataType.ANGLE_3D else '2d'

    # such a pain...
    get_pos_data = partial(positional_data,
                           dimensions=dim,
                           base_path=setup_config['experiment_root_path'],
                           experiment_path_template=setup_config['experiment_path_template'],
                           positional_data_path_template=setup_config['experiment_limb_pos_data_dir'])

    # dict of experiments and their data
    data_raw = seq(EXPERIMENTS).map(lambda x: (experiment_key(obj=x), get_pos_data(x)))\
                               .filter(lambda x: x[1] is not None)\
                               .to_dict()

    sequence_labels, sequence_data = zip(*seq(LABELLED_SEQUENCES).filter(lambda x:
                                                                         experiment_key(obj=x) in data_raw)\
                                                             .map(lambda x: _load_and_fix_(x, data_raw[experiment_key(obj=x)])))
    frame_data = np.vstack(sequence_data)
    frame_labels = seq(sequence_labels).flat_map(lambda x: [(i, x) for i in range(*x.sequence)]).to_list()

    #return sequence_data, frame_labels, sequence_labels
    return frame_data, frame_labels



def positional_data(experiment, dimensions='2d', pattern='pose_result', base_path=None,
                    experiment_path_template=None, positional_data_path_template=None,  return_experiment_id=False):
    """
    Returns the positional data for the given experiment.

    Parameters:
    -----------
        experiment: Can be either data.Experiment or data.__LabelledData__
        dimensions: String that indicates if `2d` or `3d`.
        pattern:    String of the corresponding file name.

    Returns:
    --------
        numpy array of found data

    """
    base = experiment_path_template.format(base_path=base_path,
                                           study_id=experiment.study_id,
                                           fly_id=experiment.fly_id,
                                           experiment_id=experiment.experiment_id)

    pos_data_path = positional_data_path_template.format(base_experiment_path=base)

    try:
        pose = [p for p in pathlib.Path(pos_data_path).iterdir() if pattern in p.name]

        if len(pose) == 0:
            raise FileNotFoundError('Could not find the pose data file')
        else:
            pose = pose[0]
    except FileNotFoundError as e:
        print(f"huh?? something odd with {experiment}: {pathlib.Path(pos_data_path)}: {e}")
        return None

    with open(pose, 'rb') as f:
        data = pickle.load(f)[f'points{dimensions}']
        if return_experiment_id:
            return experiment.key, data
        else:
            return data


#def images_paths(experiment, camera_id):
#    base = PATH_EXPERIMENT.format(base_path=__EXPERIMENT_ROOT__,
#                                  study_id=experiment.study_id,
#                                  fly_id=experiment.fly_id,
#                                  experiment_id=experiment.experiment_id)
#
#    image_dir = PATH_EXPERIMENT_POSITIONAL_DATA.format(base_experiment_path=base)
#
#    images = sorted([(int(p.stem[-6:]), str(p)) for p in pathlib.Path(image_dir).iterdir()
#                     if f'camera_{camera_id}' in p.name], key=lambda x: x[0])
#
#    return images
#
#
#def get_path_for_image(d, frame_id, base_path=__EXPERIMENT_ROOT__, camera_id=CAMERA_OF_INTEREST):
#    base = PATH_EXPERIMENT.format(base_path=base_path,
#                                  study_id=d.study_id,
#                                  fly_id=d.fly_id,
#                                  experiment_id=d.experiment_id)
#
#    path_image = PATH_EXPERIMENT_IMAGE.format(base_experiment_path=base,
#                                              camera_id=camera_id)
#
#    return path_image.format(image_id=frame_id)
#
#
#def n_frames_per_experiment(labelled_data):
#    return seq(labelled_data)\
#        .flat_map(lambda x: ((data.experiment_key(labelled_sequence=x), frame_id)
#                             for frame_id in range(*x.sequence)))\
#        .count_by_key()\
#        .to_list()
#
#def filter_by_study_and_fly(study_id, fly_id, labelled_data):
#    return seq(labelled_data)\
#        .filter(lambda x: x.study_id == study_id and x.fly_id == fly_id)\
#        .to_list()


def get_3d_columns_names(selected_columns):
    return np.array([f"limb: {skeleton.limb_id[i]}: {p.name}" for i, p in enumerate(skeleton.tracked_points)])[selected_columns]


def experiments_from_root(root):
    # amazing clusterfuck of loading the data, sorry
    return seq(Path(root).iterdir()).flat_map(lambda p: (c for c in p.iterdir() if c.is_dir()))\
                                    .flat_map(lambda p: (c for c in p.iterdir() if c.is_dir()))\
                                    .map(lambda p: reduce(lambda acc, el: ([*acc[0], acc[1].stem], acc[1].parent), range(3), ([], p)))\
                                    .map(lambda pair: list(reversed(pair[0])))\
                                    .map(lambda ls: Experiment._make(*ls))\
                                    .to_list()
