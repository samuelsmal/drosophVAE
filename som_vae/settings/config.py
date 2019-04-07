import pickle
import pathlib

from som_vae.settings import data

NB_DIMS = 2

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

__DATA_ROOT__ = "/home/sam/proj/epfl/neural_clustering_data"

## if in lab
# __DATA_ROOT__ = "/home/samuel/"
# __EXPERIMENT_ROOT__ =  "/ramdya-nas/SVB/experiments"

__VIDEO_ROOT__ = f"{__DATA_ROOT__}/videos"
__EXPERIMENT_ROOT__ = f"{__DATA_ROOT__}/experiments"


# Check out the functions below, use them and not these fragments
PATH_EXPERIMENT = "{base_path}/{study_id}/{fly_id}/{experiment_id}"
PATH_EXPERIMENT_POSITIONAL_DATA = "{base_experiment_path}/behData/images/"
PATH_EXPERIMENT_IMAGE = "{base_experiment_path}/behData/images/camera_{camera_id}_img_{{image_id:0>6}}.jpg"

#POSE_DATA_PATH = POSE_DATA_BASE_PATH.format(experiment_id=EXPERIMENT_ID)

#POSE_DATA_PATH = "/ramdya-nas/SVB/experiments/{fly_id}/001_coronal/behData/images_renamed/pose_result__mnt_NAS_SVB_181220_Rpr_R57C10_GC6s_tdTom_001_coronal_behData_images_renamed.pkl".format(fly_id=FLY)
#POSE_FRAME_PATH = "/ramdya-nas/SVB/experiments/{fly_id}/001_coronal/behData/images_renamed/camera_{{camera_id}}_img_{{frame_id:06d}}.jpg".format(fly_id=FLY)

SEQUENCE_GIF_PATH = "/home/samuel/Videos/{fly_id}/sequence_{{begin_frame}}-{{end_frame}}.mp4".format(fly_id="todo_dummy_change_this")
EXPERIMENT_VIDEO_PATH = __VIDEO_ROOT__ + "/{experiment_id}_sequence_{begin}-{end}.mp4"


def full_experiment_id(study_id=STUDY_ID, experiment_id=EXPERIMENT_ID, fly_id=FLY_ID):
    return f"{study_id}-{experiment_id}-{fly_id}"


def positional_data(experiment, dimensions='2d', pattern='pose_result'):
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
    base = PATH_EXPERIMENT.format(base_path=__EXPERIMENT_ROOT__,
                                  study_id=experiment.study_id,
                                  fly_id=experiment.fly_id,
                                  experiment_id=experiment.experiment_id)

    pos_data_path = PATH_EXPERIMENT_POSITIONAL_DATA.format(base_experiment_path=base)

    pose = [p for p in pathlib.Path(pos_data_path).iterdir()
            if pattern in p.name][0]

    with open(pose, 'rb') as f:
        return pickle.load(f)[f'points{dimensions}']


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

