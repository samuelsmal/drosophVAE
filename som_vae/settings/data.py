from enum import Enum
from collections import namedtuple
import json
import pickle
from datetime import datetime
import pathlib

class Behavior(Enum):
    WALK_FORW = 0
    WALK_BACKW = 1
    PUSH_BALL = 2
    REST = 3
    GROOM_FLEG = 4
    GROOM_ANT = 5
    NONE = 6

Experiment = namedtuple('Experiment', 'study_id, fly_id, experiment_id')
LabelledSequence = namedtuple('LabelledSequence', ('sequence', 'label') + Experiment._fields)


def experiment_key(study_id=None, experiment_id=None, fly_id=None, obj=None):
    """Exhibit A why duck typing is just shit sometimes"""

    if obj:
        return f"{obj.study_id}-{obj.experiment_id}-{obj.fly_id}"
    elif labelled_sequence is not None:
        return experiment_key(experiment=labelled_sequence.experiment)
    else:
        return f"{study_id}-{experiment_id}-{fly_id}"


# They the ranges are half-open: [0, 14) in "mathy" writing
_LABELLED_DATA_RAW_ = [
    ((  0, 140), Behavior.REST,       '180919_MDN_CsCh', 'Fly6', '001_SG1'),
    ((140, 460), Behavior.WALK_BACKW, '180919_MDN_CsCh', 'Fly6', '001_SG1'),
    ((600, 750), Behavior.WALK_FORW,  '180919_MDN_CsCh', 'Fly6', '001_SG1'),
    ((750, 900), Behavior.REST,       '180919_MDN_CsCh', 'Fly6', '001_SG1'),

    ((  0, 140), Behavior.REST,       '180919_MDN_CsCh', 'Fly6', '002_SG1'),
    ((140, 500), Behavior.WALK_BACKW, '180919_MDN_CsCh', 'Fly6', '002_SG1'),
    ((630, 800), Behavior.WALK_FORW,  '180919_MDN_CsCh', 'Fly6', '002_SG1'),
    ((790, 900), Behavior.REST,       '180919_MDN_CsCh', 'Fly6', '002_SG1'),

    ((  0, 140), Behavior.REST,       '180919_MDN_CsCh', 'Fly6', '003_SG1'),
    ((140, 500), Behavior.WALK_BACKW, '180919_MDN_CsCh', 'Fly6', '003_SG1'),
    ((570, 750), Behavior.WALK_FORW,  '180919_MDN_CsCh', 'Fly6', '003_SG1'),

    ((  0, 140), Behavior.REST,       '180919_MDN_CsCh', 'Fly6', '004_SG1'),
    ((140, 500), Behavior.WALK_BACKW, '180919_MDN_CsCh', 'Fly6', '004_SG1'),
    ((600, 750), Behavior.WALK_FORW,  '180919_MDN_CsCh', 'Fly6', '004_SG1'),

    ((  0, 140), Behavior.REST,       '180919_MDN_CsCh', 'Fly6', '005_SG1'),
    ((140, 500), Behavior.WALK_BACKW, '180919_MDN_CsCh', 'Fly6', '005_SG1'),
    ((600, 750), Behavior.WALK_FORW,  '180919_MDN_CsCh', 'Fly6', '005_SG1'),

    ((  0, 150), Behavior.GROOM_FLEG, '180921_aDN_CsCh', 'Fly6', '003_SG1'),
    ((170, 350), Behavior.GROOM_ANT,  '180921_aDN_CsCh', 'Fly6', '003_SG1'),
    ((450, 600), Behavior.REST,       '180921_aDN_CsCh', 'Fly6', '003_SG1'),

    ((  0, 150), Behavior.REST,       '180921_aDN_CsCh', 'Fly6', '001_SG1'),
    ((180, 350), Behavior.GROOM_ANT,  '180921_aDN_CsCh', 'Fly6', '001_SG1'),
    ((400, 580), Behavior.REST,       '180921_aDN_CsCh', 'Fly6', '001_SG1'),

    ((250, 600), Behavior.WALK_BACKW, '180918_MDN_CsCh', 'Fly2', '004_SG1'),

    ((190, 300), Behavior.GROOM_ANT,  '180921_aDN_CsCh', 'Fly4', '003_SG1'),

    ((400, 900), Behavior.WALK_FORW,  '180918_MDN_PR', 'Fly1', '003_SG1'),

    ((  0, 500), Behavior.REST,       '180918_MDN_PR', 'Fly1', '004_SG1'),
    ((650, 900), Behavior.WALK_FORW,  '180918_MDN_PR', 'Fly1', '004_SG1'),

    ((  0, 500), Behavior.REST,       '180918_MDN_PR', 'Fly1', '005_SG1'),
    ((500, 900), Behavior.WALK_FORW,  '180918_MDN_PR', 'Fly1', '005_SG1'),

    ((  0, 100), Behavior.PUSH_BALL,  '180918_MDN_PR', 'Fly2', '001_SG1'),
    ((350, 500), Behavior.GROOM_FLEG, '180918_MDN_PR', 'Fly2', '002_SG1'),
    ((400, 530), Behavior.GROOM_FLEG, '180918_MDN_PR', 'Fly2', '003_SG1'),

    ((150, 230), Behavior.GROOM_ANT,  '180921_aDN_CsCh', 'Fly3', '001_SG1'),

    #((170, 210), Behavior.WALK_BACKW,  '180919_MDN_CsCh', 'Fly4', '005_SG1'),
    #((210, 600), Behavior.WALK_FORW,   '180919_MDN_CsCh', 'Fly4', '005_SG1'),
    #((600, 700), Behavior.PUSH_BALL,   '180919_MDN_CsCh', 'Fly4', '005_SG1'),

    #((600, 700), Behavior.PUSH_BALL,   '180919_MDN_CsCh', 'Fly4', '005_SG1'),

    ((  0, 145), Behavior.WALK_FORW,  '180920_aDN_CsCh', 'Fly2', '001_SG1'),
    ((145, 225), Behavior.GROOM_ANT,  '180920_aDN_CsCh', 'Fly2', '001_SG1'),
    ((225, 671), Behavior.REST,       '180920_aDN_CsCh', 'Fly2', '001_SG1'),
    ((671, 683), Behavior.GROOM_ANT,  '180920_aDN_CsCh', 'Fly2', '001_SG1'),
    ((683, 761), Behavior.WALK_FORW,  '180920_aDN_CsCh', 'Fly2', '001_SG1'),
    ((761, 778), Behavior.REST,       '180920_aDN_CsCh', 'Fly2', '001_SG1'),
    ((778, 809), Behavior.WALK_FORW,  '180920_aDN_CsCh', 'Fly2', '001_SG1'),
    ((809, 813), Behavior.REST,       '180920_aDN_CsCh', 'Fly2', '001_SG1'),
    ((813, 820), Behavior.WALK_BACKW, '180920_aDN_CsCh', 'Fly2', '001_SG1'),
    ((820, 861), Behavior.WALK_FORW,  '180920_aDN_CsCh', 'Fly2', '001_SG1'),
    ((861, 868), Behavior.REST,       '180920_aDN_CsCh', 'Fly2', '001_SG1'),
    ((868, 879), Behavior.GROOM_ANT,  '180920_aDN_CsCh', 'Fly2', '001_SG1'),
    ((879, 900), Behavior.WALK_BACKW, '180920_aDN_CsCh', 'Fly2', '001_SG1'),

    ((  0, 143), Behavior.WALK_BACKW, '180920_aDN_CsCh', 'Fly2', '002_SG1'),
    ((143, 254), Behavior.GROOM_ANT,  '180920_aDN_CsCh', 'Fly2', '002_SG1'),
    ((254, 822), Behavior.REST,       '180920_aDN_CsCh', 'Fly2', '002_SG1'),
    ((822, 900), Behavior.GROOM_ANT,  '180920_aDN_CsCh', 'Fly2', '002_SG1'),

    ((  0, 145), Behavior.REST,       '180920_aDN_CsCh', 'Fly2', '003_SG1'),
    ((145, 247), Behavior.GROOM_ANT,  '180920_aDN_CsCh', 'Fly2', '003_SG1'),
    ((247, 653), Behavior.REST,       '180920_aDN_CsCh', 'Fly2', '003_SG1'),
    ((653, 785), Behavior.WALK_FORW,  '180920_aDN_CsCh', 'Fly2', '003_SG1'),
    ((785, 803), Behavior.REST,       '180920_aDN_CsCh', 'Fly2', '003_SG1'),
    ((803, 820), Behavior.NONE,       '180920_aDN_CsCh', 'Fly2', '003_SG1'),
    ((820, 859), Behavior.WALK_FORW,  '180920_aDN_CsCh', 'Fly2', '003_SG1'),
    ((859, 900), Behavior.REST,       '180920_aDN_CsCh', 'Fly2', '003_SG1'),

    ((  0, 147), Behavior.REST,       '180920_aDN_CsCh', 'Fly2', '004_SG1'),
    ((147, 235), Behavior.GROOM_ANT,  '180920_aDN_CsCh', 'Fly2', '004_SG1'),
    ((235, 657), Behavior.REST,       '180920_aDN_CsCh', 'Fly2', '004_SG1'),
    ((657, 816), Behavior.WALK_FORW,  '180920_aDN_CsCh', 'Fly2', '004_SG1'),
    ((816, 820), Behavior.GROOM_ANT,  '180920_aDN_CsCh', 'Fly2', '004_SG1'),
    ((820, 900), Behavior.REST,       '180920_aDN_CsCh', 'Fly2', '004_SG1'),

    ((  0, 144), Behavior.REST,       '180920_aDN_CsCh', 'Fly2', '005_SG1'),
    ((144, 226), Behavior.GROOM_ANT,  '180920_aDN_CsCh', 'Fly2', '005_SG1'),
    ((226, 239), Behavior.REST,       '180920_aDN_CsCh', 'Fly2', '005_SG1'),
    ((239, 253), Behavior.GROOM_ANT,  '180920_aDN_CsCh', 'Fly2', '005_SG1'),
    ((253, 267), Behavior.REST,       '180920_aDN_CsCh', 'Fly2', '005_SG1'),
    ((267, 278), Behavior.GROOM_ANT,  '180920_aDN_CsCh', 'Fly2', '005_SG1'),
    ((278, 656), Behavior.REST,       '180920_aDN_CsCh', 'Fly2', '005_SG1'),
    ((656, 659), Behavior.WALK_FORW,  '180920_aDN_CsCh', 'Fly2', '005_SG1'),
    ((659, 665), Behavior.GROOM_ANT,  '180920_aDN_CsCh', 'Fly2', '005_SG1'),
    ((665, 757), Behavior.WALK_FORW,  '180920_aDN_CsCh', 'Fly2', '005_SG1'),
    ((757, 768), Behavior.REST,       '180920_aDN_CsCh', 'Fly2', '005_SG1'),
    ((768, 799), Behavior.WALK_BACKW, '180920_aDN_CsCh', 'Fly2', '005_SG1'),
    ((799, 900), Behavior.REST,       '180920_aDN_CsCh', 'Fly2', '005_SG1'),
]

LABELLED_SEQUENCES = [LabelledSequence._make(i) for i in _LABELLED_DATA_RAW_]
EXPERIMENTS = list(set(Experiment(study_id=l.study_id, fly_id=l.fly_id,
                                    experiment_id=l.experiment_id) \
                         for l in LABELLED_SEQUENCES))
