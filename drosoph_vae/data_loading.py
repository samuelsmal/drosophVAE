"""Here you'll find the overall definitions

"""
import json
import pickle
import pathlib
import numpy as np
from pathlib import Path
from functools import partial, reduce
from functional import seq

from drosoph_vae.settings import skeleton
from drosoph_vae.settings.data import EXPERIMENTS, LABELLED_SEQUENCES, experiment_key, Behavior, LabelledSequence, Experiment
from drosoph_vae.settings.config import DataType, SetupConfig
from drosoph_vae import preprocessing



def _load_and_fix_(sequence, data):
    # I have no clue why I need to do this. Semihg knows of this data problem.
    e = min(sequence.sequence[1], data[0].shape[0])
    s = LabelledSequence(**{**sequence._asdict(),
                                   'sequence': (sequence.sequence[0], e)})

    d = data[0][sequence.sequence[0]:e]

    return s, d, data[1]

def _standarize_(data):
    m = np.mean(data, axis=0)
    data = data - m
    s = np.std(data, axis=0)
    data /= s

    return data, (m, s)


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

    # experiment_key, experiment_data (for each fly, the lowest experiment-unit)
    data_raw = seq(EXPERIMENTS).map(lambda x: (experiment_key(obj=x), get_pos_data(x)))\
                               .filter(lambda x: x[1] is not None)

    def _map_values_(fn):
        return lambda x: (x[0], fn(x[1]))

    if run_config['data_type'] == DataType.POS_2D:
        #data_raw = data_raw\ #.map(_map_values_(preprocessing._simple_checks_))\
        data_raw = data_raw.map(_map_values_(preprocessing._get_camera_of_interest_))\
                .map(_map_values_(preprocessing._get_visible_legs_))\
                .map(_map_values_(preprocessing.get_only_first_legs))

    data_raw = data_raw.to_dict()

    if run_config['data_type'] == DataType.POS_2D and \
    run_config.preprocessing_parameters()['normalize_for_each_experiment']:
        data_raw = {exp_id: _standarize_(data) for exp_id, data in data_raw.items()}
    else:
        data_raw = {exp_id: (data, (None, None)) for exp_id, data in data_raw.items()}


    # if the data is somehow missing for some experiments...
    labelled_sequences = seq(LABELLED_SEQUENCES).filter(lambda x: experiment_key(obj=x) in data_raw)

    if run_config['use_single_fly']:
        # I called it (him? her?, x?) Hubert.
        # filter but use all experiments done on Hubert
        hubert = SetupConfig.value('hubert')
        labelled_sequences = labelled_sequences.filter(lambda x: x.study_id == hubert['study_id'] and x.fly_id == hubert['fly_id'])
        #labelled_sequences = labelled_sequences.filter(lambda x: experiment_key(obj=x) ==
        #                             experiment_key(**SetupConfig.value('hubert')))

    blacklist_behavior = run_config.value('preprocessing', 'common', 'blacklist_behavior')
    if len(blacklist_behavior) > 0:
        labelled_sequences = labelled_sequences.filter(lambda x: x.label not in blacklist_behavior)

    labelled_sequences = labelled_sequences.map(lambda x: _load_and_fix_(x, data_raw[experiment_key(obj=x)]))

    sequence_labels, sequence_data, normalisation_factors = zip(*labelled_sequences)
    normalisation_factors = dict(zip((experiment_key(obj=l) for l in sequence_labels),
                                 normalisation_factors))

    # up to here you can play around with each experiment by itself
    frame_data = np.vstack(sequence_data)
    # flat out the label for each frame in the sequence
    frame_labels = np.array(seq(sequence_labels)\
                            .flat_map(lambda x: [(i, x) for i in range(*x.sequence)]).to_list())

    return frame_data.astype(np.float32), frame_labels, normalisation_factors

def get_data_and_normalization(experiments, normalize_data=False, dimensions='2d', return_with_experiment_id=False):
    if normalize_data and return_with_experiment_id:
        raise ValueError('choose one')

    # TODO wtf?
    ret = seq(experiments).map(partial(positional_data,
                                       base_path=SetupConfig.value('experiment_root_path'),
                                       experiment_path_template=SetupConfig.value('experiment_path_template'),
                                       positional_data_path_template=SetupConfig.value('experiment_limb_pos_data_dir'),
                                       dimensions=dimensions,
                                return_experiment_id=return_with_experiment_id))\
                   .filter(lambda x: x is not None)\

    if return_with_experiment_id:
        exp_ids, ret = zip(*ret.to_list())
        ret = seq(ret)

    if dimensions == '2d':
        ret = ret.map(_simple_checks_)\
                 .map(_get_camera_of_interest_)\
                 .map(_get_visible_legs_)\
                 .map(add_third_dimension)\
                 .map(get_only_first_legs)

    if normalize_data:
        ret = preprocessing.normalize(np.vstack(ret.to_list()))
    else:
        ret = ret.to_list()

    if return_with_experiment_id:
        return list(zip(exp_ids, ret))
    else:
        return ret


def load_all_experiments(data_type,
                         with_experiment_ids=False,
                         normalize_data=True,
                         experiment_black_list=SetupConfig.value('experiment_black_list'),
                         fly_black_list=SetupConfig.value('fly_black_list')):
    all_experiments = [e for e in experiments_from_root() if e.study_id not in experiment_black_list
                       or experiment_key(obj=e) in fly_black_list]
    return get_data_and_normalization(all_experiments, normalize_data=normalize_data, return_with_experiment_id=with_experiment_ids,
                                                 dimensions='3d' if data_type == DataType.ANGLE_3D else '2d')


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
            return experiment_key(obj=experiment), data
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


def experiments_from_root(root=SetupConfig.value('experiment_root_path')):
    # amazing clusterfuck of loading the data, sorry
    return seq(Path(root).iterdir()).flat_map(lambda p: (c for c in p.iterdir() if c.is_dir()))\
                                    .flat_map(lambda p: (c for c in p.iterdir() if c.is_dir()))\
                                    .map(lambda p: reduce(lambda acc, el: ([*acc[0], acc[1].stem], acc[1].parent), range(3), ([], p)))\
                                    .map(lambda pair: list(reversed(pair[0])))\
                                    .map(lambda ls: Experiment(*ls))\
                                    .to_list()
