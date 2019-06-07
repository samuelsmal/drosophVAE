import json
import pickle
from datetime import datetime
import pathlib

from som_vae.helpers.misc import get_hostname
from som_vae.settings import data

NB_DIMS = 3

LEGS = [0, 1, 2, 5, 6, 7]
LEGS = [0, 1, 2] #, 5, 6, 7] # since we do not care about the other side
CAMERA_OF_INTEREST = 1
NB_OF_AXIS = 2
NB_TRACKED_POINTS = 5 # per leg, igoring the rest for now
NB_CAMERAS = 7
NB_RECORDED_DIMESIONS = 2

FRAMES_PER_SECOND = 100
NYQUIST_FREQUENCY_OF_MEASUREMENTS = FRAMES_PER_SECOND / 2

# TODO make this loadable
STUDY_ID = "180920_aDN_CsCh"
EXPERIMENT_ID = "001_SG1"
FLY_ID = "Fly2"



if get_hostname() == 'upramdyapc6':
    __DATA_ROOT__ = "/home/samuel/neural_clustering_data"
    __EXPERIMENT_ROOT__ =  "/ramdya-nas/SVB/experiments"
elif get_hostname() == 'contosam':
    __DATA_ROOT__ = "/home/sam/proj/epfl/neural_clustering_data"
    __EXPERIMENT_ROOT__ =  "/home/sam/Dropbox"
else:
    __DATA_ROOT__ = "/home/sam/proj/epfl/neural_clustering_data"
    __EXPERIMENT_ROOT__ = f"{__DATA_ROOT__}/experiments"

__VIDEO_ROOT__ = f"{__DATA_ROOT__}/videos"


# Check out the functions below, use them and not these fragments
PATH_EXPERIMENT = "{base_path}/{study_id}/{fly_id}/{experiment_id}"
PATH_EXPERIMENT_POSITIONAL_DATA = "{base_experiment_path}/behData/images/"
PATH_EXPERIMENT_IMAGE = "{base_experiment_path}/behData/images/camera_{camera_id}_img_{{image_id:0>6}}.jpg"

PATH_TO_FIGURES = f"{__DATA_ROOT__}/figures"

#POSE_DATA_PATH = POSE_DATA_BASE_PATH.format(experiment_id=EXPERIMENT_ID)

#POSE_DATA_PATH = "/ramdya-nas/SVB/experiments/{fly_id}/001_coronal/behData/images_renamed/pose_result__mnt_NAS_SVB_181220_Rpr_R57C10_GC6s_tdTom_001_coronal_behData_images_renamed.pkl".format(fly_id=FLY)
#POSE_FRAME_PATH = "/ramdya-nas/SVB/experiments/{fly_id}/001_coronal/behData/images_renamed/camera_{{camera_id}}_img_{{frame_id:06d}}.jpg".format(fly_id=FLY)

SEQUENCE_GIF_PATH = "/home/samuel/Videos/{fly_id}/sequence_{{begin_frame}}-{{end_frame}}.mp4".format(fly_id="todo_dummy_change_this")
EXPERIMENT_VIDEO_PATH = __VIDEO_ROOT__ + "/{experiment_id}_sequence_{vid_id}.mp4"


def full_experiment_id(study_id=STUDY_ID, experiment_id=EXPERIMENT_ID, fly_id=FLY_ID):
    return f"{study_id}-{experiment_id}-{fly_id}"


def get_experiment_id(experiment):
    return full_experiment_id(study_id=experiment.study_id,
                              experiment_id=experiment.experiment_id,
                              fly_id=experiment.fly_id)


def positional_data(experiment, dimensions='2d', pattern='pose_result', base_path=__EXPERIMENT_ROOT__, return_experiment_id=False):
    """
    Returns the positional data for the given experiment.

    Input
    -----

    experiment: Can be either data.Experiment or data.__LabelledData__
    dimensions: String that indicates if `2d` or `3d`.
    pattern:    String of the corresponding file name.

    Returns
    -------

    numpy array of found data

    """
    base = PATH_EXPERIMENT.format(base_path=base_path,
                                  study_id=experiment.study_id,
                                  fly_id=experiment.fly_id,
                                  experiment_id=experiment.experiment_id)

    pos_data_path = PATH_EXPERIMENT_POSITIONAL_DATA.format(base_experiment_path=base)

    try:
        pose = [p for p in pathlib.Path(pos_data_path).iterdir() if pattern in p.name]

        if len(pose) == 0:
            raise FileNotFoundError('Could not find the pose data file')
        else:
            pose = pose[0]
    except FileNotFoundError as e:
        print(f"huh?? something odd with {experiment.key}: {pathlib.Path(pos_data_path)}: {e}")
        return None

    with open(pose, 'rb') as f:
        data = pickle.load(f)[f'points{dimensions}']
        if return_experiment_id:
            return experiment.key, data
        else:
            return data


def images_paths(experiment, camera_id=CAMERA_OF_INTEREST):
    base = PATH_EXPERIMENT.format(base_path=__EXPERIMENT_ROOT__,
                                  study_id=experiment.study_id,
                                  fly_id=experiment.fly_id,
                                  experiment_id=experiment.experiment_id)

    image_dir = PATH_EXPERIMENT_POSITIONAL_DATA.format(base_experiment_path=base)

    images = sorted([(int(p.stem[-6:]), str(p)) for p in pathlib.Path(image_dir).iterdir()
                     if f'camera_{camera_id}' in p.name], key=lambda x: x[0])

    return images


def get_path_for_image(d, frame_id, base_path=__EXPERIMENT_ROOT__, camera_id=CAMERA_OF_INTEREST):
    base = PATH_EXPERIMENT.format(base_path=base_path,
                                  study_id=d.study_id,
                                  fly_id=d.fly_id,
                                  experiment_id=d.experiment_id)

    path_image = PATH_EXPERIMENT_IMAGE.format(base_experiment_path=base,
                                              camera_id=camera_id)

    return path_image.format(image_id=frame_id)


def config_description(config, short=False):
    """to be used with a `run_config` from reparam_vae
    """
    def _bool_(v):
        return 'T' if config[v] else 'F'

    valus_of_interest = [
        ('data', '', config['data_type']),
        ('time', 't', config['time_series_length'] if config['use_time_series'] else 'F'),
        ('kernel', 'k', config['conv_layer_kernel_size']),
        ('n_clayers', 'ncl', config['n_conv_layers']),
        ('latent_dim', 'ld', config['latent_dim']),
        ('multiple_flys', 'mf', _bool_('use_all_experiments')),
        ('optimizer', 'opt', config.get('optimizer')),
        ('loss_weight_recon', 'lwr', config.get('loss_weight_reconstruction')),
        ('loss_weight_kl', 'lwkl', config.get('loss_weight_kl')),
        ('dropout_rate', 'dr', config.get('dropout_rate')),
        ('model_impl', 'mi', config.get('model_impl')),
        ('with_batch_norm', 'bn', _bool_('with_batch_norm'))
    ]

    descr_idx = 1 if short else 0
    descr_str = '-'.join((f"{v[descr_idx]}-{v[2]}" for v in valus_of_interest[1:]))

    descr_str = valus_of_interest[0][2] + '-' + descr_str

    if config['debug']:
        descr_str += '_' + ''.join([k for k, v in config.items() if k.startswith('d_') and v])
    else:
        descr_str += '_' + ('all_data' if config['use_all_experiments'] else 'small')

    return descr_str


def model_config_description(config, short=False):
    valus_of_interest = [
        ("epochs", "e", config['epochs']),
        ("loss_weight_recon", "lwr", config['loss_weight_reconstruction']),
        ("loss_weight_kl", "lwkl", config['loss_weight_kl']),
    ]

    descr_idx = 1 if short else 0

    return '-'.join((f"{v[descr_idx]}-{v[2]}" for v in valus_of_interest))


def exp_desc(run_config, model_config, short=False):
    return config_description(run_config, short=short) + '-' + model_config_description(model_config, short=short)

def get_config_hash(config, digest_length=5):
    return str(hash(json.dumps({**config, '_executed_at_': str(datetime.now())}, sort_keys=True)))[:digest_length]
