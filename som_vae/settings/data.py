from collections import namedtuple
from enum import Enum

from functional import seq


class _BehaviorLabel_(Enum):
    WALK_FORW = 0
    WALK_BACKW = 1
    PUSH_BALL = 2
    REST = 3
    GROOM_FLEG = 4
    GROOM_ANT = 5
    NONE = 6

Experiment = namedtuple('Experiment', 'study_id, fly_id, experiment_id, key')

__LabelledData__ = namedtuple('LabelledData', 'sequence, label, study_id, fly_id, experiment_id')


def _key_from_list_(ls):
    return '-'.join(ls)

def _key_(x:__LabelledData__):
    return f"{x.study_id}-{x.fly_id}-{x.experiment_id}"


def _experiment_(x:__LabelledData__):
    return Experiment(x.study_id, x.fly_id, x.experiment_id, _key_(x))


def n_frames_per_experiment(labelled_data):
    return seq(labelled_data)\
        .flat_map(lambda x: ((_key_(x), frame_id) for frame_id in range(*x.sequence)))\
        .count_by_key()\
        .to_list()


def filter_by_study_and_fly(study_id, fly_id, labelled_data):
    return seq(labelled_data)\
        .filter(lambda x: x.study_id == study_id and x.fly_id == fly_id)\
        .to_list()


__LABELLED_DATA_RAW__ = [
    #((  0, 140), _BehaviorLabel_.REST,       '/180919_MDN_CsCh/Fly6/001_SG1'),
    #((140, 460), _BehaviorLabel_.WALK_BACKW, '/180919_MDN_CsCh/Fly6/001_SG1'),
    #((600, 750), _BehaviorLabel_.WALK_FORW,  '/180919_MDN_CsCh/Fly6/001_SG1'),
    #((750, 900), _BehaviorLabel_.REST,       '/180919_MDN_CsCh/Fly6/001_SG1'),

    #((  0, 140), _BehaviorLabel_.REST,       '/180919_MDN_CsCh/Fly6/002_SG1'),
    #((140, 500), _BehaviorLabel_.WALK_BACKW, '/180919_MDN_CsCh/Fly6/002_SG1'),
    #((630, 800), _BehaviorLabel_.WALK_FORW,  '/180919_MDN_CsCh/Fly6/002_SG1'),
    #((790, 900), _BehaviorLabel_.REST,       '/180919_MDN_CsCh/Fly6/002_SG1'),

    #((  0, 140), _BehaviorLabel_.REST,       '/180919_MDN_CsCh/Fly6/003_SG1'),
    #((140, 500), _BehaviorLabel_.WALK_BACKW, '/180919_MDN_CsCh/Fly6/003_SG1'),
    #((570, 750), _BehaviorLabel_.WALK_FORW,  '/180919_MDN_CsCh/Fly6/003_SG1'),

    #((  0, 140), _BehaviorLabel_.REST,       '/180919_MDN_CsCh/Fly6/004_SG1'),
    #((140, 500), _BehaviorLabel_.WALK_BACKW, '/180919_MDN_CsCh/Fly6/004_SG1'),
    #((600, 750), _BehaviorLabel_.WALK_FORW,  '/180919_MDN_CsCh/Fly6/004_SG1'),

    #((  0, 140), _BehaviorLabel_.REST,       '/180919_MDN_CsCh/Fly6/005_SG1'),
    #((140, 500), _BehaviorLabel_.WALK_BACKW, '/180919_MDN_CsCh/Fly6/005_SG1'),
    #((600, 750), _BehaviorLabel_.WALK_FORW,  '/180919_MDN_CsCh/Fly6/005_SG1'),

    #((  0, 150), _BehaviorLabel_.GROOM_FLEG, '/180921_aDN_CsCh/Fly6/003_SG1'),
    #((170, 350), _BehaviorLabel_.GROOM_ANT,  '/180921_aDN_CsCh/Fly6/003_SG1'),
    #((450, 600), _BehaviorLabel_.REST,       '/180921_aDN_CsCh/Fly6/003_SG1'),

    #((  0, 150), _BehaviorLabel_.REST,       '/180921_aDN_CsCh/Fly6/001_SG1'),
    #((180, 350), _BehaviorLabel_.GROOM_ANT,  '/180921_aDN_CsCh/Fly6/001_SG1'),
    #((400, 580), _BehaviorLabel_.REST,       '/180921_aDN_CsCh/Fly6/001_SG1'),

    #((250, 600), _BehaviorLabel_.WALK_BACKW, '/180918_MDN_CsCh/Fly2/004_SG1'),

    #((190, 300), _BehaviorLabel_.GROOM_ANT,  '/180921_aDN_CsCh/Fly4/003_SG1'),

    #((400, 900), _BehaviorLabel_.WALK_FORW,  '/180918_MDN_PR/Fly1/003_SG1'),

    #((  0, 500), _BehaviorLabel_.REST,       '/180918_MDN_PR/Fly1/004_SG1'),
    #((650, 900), _BehaviorLabel_.WALK_FORW,  '/180918_MDN_PR/Fly1/004_SG1'),

    #((  0, 500), _BehaviorLabel_.REST,       '/180918_MDN_PR/Fly1/005_SG1'),
    #((500, 900), _BehaviorLabel_.WALK_FORW,  '/180918_MDN_PR/Fly1/005_SG1'),

    #((  0, 100), _BehaviorLabel_.PUSH_BALL,  '/180918_MDN_PR/Fly2/001_SG1'),
    #((350, 500), _BehaviorLabel_.GROOM_FLEG, '/180918_MDN_PR/Fly2/002_SG1'),
    #((400, 530), _BehaviorLabel_.GROOM_FLEG, '/180918_MDN_PR/Fly2/003_SG1'),

    #((150, 230), _BehaviorLabel_.GROOM_ANT,  '/180921_aDN_CsCh/Fly3/001_SG1'),

    ##((170, 210), _BehaviorLabel_.WALK_BACKW, '/180919_MDN_CsCh/Fly4/005_SG1'),
    ##((210, 600), _BehaviorLabel_.WALK_FORW,  '/180919_MDN_CsCh/Fly4/005_SG1'),
    ##((600, 700), _BehaviorLabel_.PUSH_BALL,  '/180919_MDN_CsCh/Fly4/005_SG1'),

    ##((600, 700), _BehaviorLabel_.PUSH_BALL,  '/180919_MDN_CsCh/Fly4/005_SG1'),

    ((  0, 145), _BehaviorLabel_.WALK_FORW,  '180920_aDN_CsCh', 'Fly2', '001_SG1'),
    ((145, 225), _BehaviorLabel_.GROOM_ANT,  '180920_aDN_CsCh', 'Fly2', '001_SG1'),
    ((225, 671), _BehaviorLabel_.REST,       '180920_aDN_CsCh', 'Fly2', '001_SG1'),
    ((671, 683), _BehaviorLabel_.GROOM_ANT,  '180920_aDN_CsCh', 'Fly2', '001_SG1'),
    ((683, 761), _BehaviorLabel_.WALK_FORW,  '180920_aDN_CsCh', 'Fly2', '001_SG1'),
    ((761, 778), _BehaviorLabel_.REST,       '180920_aDN_CsCh', 'Fly2', '001_SG1'),
    ((778, 809), _BehaviorLabel_.WALK_FORW,  '180920_aDN_CsCh', 'Fly2', '001_SG1'),
    ((809, 813), _BehaviorLabel_.REST,       '180920_aDN_CsCh', 'Fly2', '001_SG1'),
    ((813, 820), _BehaviorLabel_.WALK_BACKW, '180920_aDN_CsCh', 'Fly2', '001_SG1'),
    ((820, 861), _BehaviorLabel_.WALK_FORW,  '180920_aDN_CsCh', 'Fly2', '001_SG1'),
    ((861, 868), _BehaviorLabel_.REST,       '180920_aDN_CsCh', 'Fly2', '001_SG1'),
    ((868, 879), _BehaviorLabel_.GROOM_ANT,  '180920_aDN_CsCh', 'Fly2', '001_SG1'),
    ((879, 900), _BehaviorLabel_.WALK_BACKW, '180920_aDN_CsCh', 'Fly2', '001_SG1'),

    ((  0, 143), _BehaviorLabel_.WALK_BACKW, '180920_aDN_CsCh', 'Fly2', '002_SG1'),
    ((143, 254), _BehaviorLabel_.GROOM_ANT,  '180920_aDN_CsCh', 'Fly2', '002_SG1'),
    ((254, 822), _BehaviorLabel_.REST,       '180920_aDN_CsCh', 'Fly2', '002_SG1'),
    ((822, 900), _BehaviorLabel_.GROOM_ANT,  '180920_aDN_CsCh', 'Fly2', '002_SG1'),

    ((  0, 145), _BehaviorLabel_.REST,       '180920_aDN_CsCh', 'Fly2', '003_SG1'),
    ((145, 247), _BehaviorLabel_.GROOM_ANT,  '180920_aDN_CsCh', 'Fly2', '003_SG1'),
    ((247, 653), _BehaviorLabel_.REST,       '180920_aDN_CsCh', 'Fly2', '003_SG1'),
    ((653, 785), _BehaviorLabel_.WALK_FORW,  '180920_aDN_CsCh', 'Fly2', '003_SG1'),
    ((785, 803), _BehaviorLabel_.REST,       '180920_aDN_CsCh', 'Fly2', '003_SG1'),
    ((803, 820), _BehaviorLabel_.NONE,       '180920_aDN_CsCh', 'Fly2', '003_SG1'),
    ((820, 859), _BehaviorLabel_.WALK_FORW,  '180920_aDN_CsCh', 'Fly2', '003_SG1'),
    ((859, 900), _BehaviorLabel_.REST,       '180920_aDN_CsCh', 'Fly2', '003_SG1'),

    ((  0, 147), _BehaviorLabel_.REST,       '180920_aDN_CsCh', 'Fly2', '004_SG1'),
    ((147, 235), _BehaviorLabel_.GROOM_ANT,  '180920_aDN_CsCh', 'Fly2', '004_SG1'),
    ((235, 657), _BehaviorLabel_.REST,       '180920_aDN_CsCh', 'Fly2', '004_SG1'),
    ((657, 816), _BehaviorLabel_.WALK_FORW,  '180920_aDN_CsCh', 'Fly2', '004_SG1'),
    ((816, 820), _BehaviorLabel_.GROOM_ANT,  '180920_aDN_CsCh', 'Fly2', '004_SG1'),
    ((820, 900), _BehaviorLabel_.REST,       '180920_aDN_CsCh', 'Fly2', '004_SG1'),

    ((  0, 144), _BehaviorLabel_.REST,       '180920_aDN_CsCh', 'Fly2', '005_SG1'),
    ((144, 226), _BehaviorLabel_.GROOM_ANT,  '180920_aDN_CsCh', 'Fly2', '005_SG1'),
    ((226, 239), _BehaviorLabel_.REST,       '180920_aDN_CsCh', 'Fly2', '005_SG1'),
    ((239, 253), _BehaviorLabel_.GROOM_ANT,  '180920_aDN_CsCh', 'Fly2', '005_SG1'),
    ((253, 267), _BehaviorLabel_.REST,       '180920_aDN_CsCh', 'Fly2', '005_SG1'),
    ((267, 278), _BehaviorLabel_.GROOM_ANT,  '180920_aDN_CsCh', 'Fly2', '005_SG1'),
    ((278, 656), _BehaviorLabel_.REST,       '180920_aDN_CsCh', 'Fly2', '005_SG1'),
    ((656, 659), _BehaviorLabel_.WALK_FORW,  '180920_aDN_CsCh', 'Fly2', '005_SG1'),
    ((659, 665), _BehaviorLabel_.GROOM_ANT,  '180920_aDN_CsCh', 'Fly2', '005_SG1'),
    ((665, 757), _BehaviorLabel_.WALK_FORW,  '180920_aDN_CsCh', 'Fly2', '005_SG1'),
    ((757, 768), _BehaviorLabel_.REST,       '180920_aDN_CsCh', 'Fly2', '005_SG1'),
    ((768, 799), _BehaviorLabel_.WALK_BACKW, '180920_aDN_CsCh', 'Fly2', '005_SG1'),
    ((799, 900), _BehaviorLabel_.REST,       '180920_aDN_CsCh', 'Fly2', '005_SG1'),
 ]

LABELLED_DATA = [__LabelledData__._make(i) for i in __LABELLED_DATA_RAW__]
EXPERIMENTS = seq(LABELLED_DATA).map(_experiment_).group_by(lambda x: x.key).map(lambda k: k[1][0])
