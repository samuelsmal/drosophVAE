import pathlib

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

STUDY_ID = "180920_aDN_CsCh"
EXPERIMENT_ID = "001_SG1"
FLY_ID = "Fly2"

__PATH_TO_DATA__ =  "/ramdya-nas/SVB/experiments"
PATH_EXPERIMENT = "{base_path}/{study_id}/{fly_id}/{experiment_id}"
PATH_EXPERIMENT_IMAGE= "{base_experiment_path}/behData/images/camera_{camera_id}_img_{image_id:0>6}.jpg"

#def images(study=STUDY_ID, experiment=EXPERIMENT_ID, fly=FLY_ID, camera=CAMERA_OF_INTEREST):
#    pathlib.Path(PATH_EXPERIMENT.
#

#POSE_DATA_PATH = POSE_DATA_BASE_PATH.format(experiment_id=EXPERIMENT_ID)

#POSE_DATA_PATH = "/ramdya-nas/SVB/experiments/{fly_id}/001_coronal/behData/images_renamed/pose_result__mnt_NAS_SVB_181220_Rpr_R57C10_GC6s_tdTom_001_coronal_behData_images_renamed.pkl".format(fly_id=FLY)
#POSE_FRAME_PATH = "/ramdya-nas/SVB/experiments/{fly_id}/001_coronal/behData/images_renamed/camera_{{camera_id}}_img_{{frame_id:06d}}.jpg".format(fly_id=FLY)

SEQUENCE_GIF_PATH = "/home/samuel/Videos/{fly_id}/sequence_{{begin_frame}}-{{end_frame}}.mp4".format(fly_id="todo_dummy_change_this")
EXPERIMENT_VIDEO_PATH = "/home/samuel/Videos/{experiment_id}/sequence_{begin}-{end}.mp4"

def full_experiment_id(study_id=STUDY_ID, experiment_id=EXPERIMENT_ID, fly_id=FLY_ID):
    return

