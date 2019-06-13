from functools import reduce
from datetime import datetime

import json
import numpy as np

from som_vae.helpers.misc import EEnum, get_hostname
from som_vae.settings.data import Behavior

class DataType(EEnum):
    # Can't start a member with a number...
    ANGLE_3D = 0
    POS_2D = 1

class ModelType(EEnum):
    TEMP_CONV = 0
    PADD_CONV = 1
    SKIP_PADD_CONV = 2

class BaseConfig(dict):
    def __init__(self, **kwargs):
        dict.__init__(self)
        self.update(kwargs)

    def __getitem__(self, key):
        return dict.__getitem__(self, key)

    def __setitem__(self, key, val):
        dict.__setitem__(self, key, val)

    def hash(self, digest_length=5):
        return str(hash(json.dumps({**self, '_executed_at_': str(datetime.now())}, sort_keys=True)))[:digest_length]

    @classmethod
    def value(cls, *keys):
        """Helper to return one single value (may be what ever that value is).

        E.g. value('hubert', 'fly_id')
        """
        try:
            val = reduce(lambda d, k: d[k], keys, cls())

            if isinstance(val, str) and 'UNKOWN' in val:
                raise ValueError('{0} value is not set! Reading {1}'.format(keys, val))
        except KeyError:
            raise ValueError("Could not find a value for the given key: {}".format(keys))

        return val

####################################################
#                                                  #
# Setup config: Paths, experiment definitions, ... #
#                                                  #
####################################################

class SetupConfig(BaseConfig):
    DEFAULT_VALUES = {
        'frames_per_second': 100,
        'legs': [0, 1, 2], # no longer in use
        'camera_of_interest': 1,
        'n_axis': 2,
        'n_tracked_points': 5, # per leg, igoring the rest for now
        'n_cameras': 7,
        'experiment_black_list': ['181220_Rpr_R57C10_GC6s_tdTom'], # all other experiments are used
        'fly_black_list': ['180920_aDN_PR-Fly2-005_SG1',
                           '180921_aDN_CsCh-Fly6-003_SG1'], # for those flys the angle conversion give odd results,
                                                           # and their distributions seem way off compared to the others)
        'data_root_path': None,
        'experiment_root_path': None,
        'hubert': { # there can only be one. hubert is the fly with which I started, use him to debug stuff
            'study_id': '180920_aDN_CsCh',
            'experiment_id': '001_SG1',
            'fly_id': 'Fly2',
        },
    }

    def __init__(self, **kwargs):
        super(BaseConfig, self).__init__({**SetupConfig.DEFAULT_VALUES, **kwargs})

        # host specific location for data
        if get_hostname() == 'upramdyapc6':
            data_root_path = '/home/samuel/neural_clustering_data'
            experiment_root_path =  '/ramdya-nas/SVB/experiments'
        elif get_hostname() == 'contosam':
            data_root_path = '/home/sam/proj/epfl/neural_clustering_data'
            experiment_root_path =  '/home/sam/Dropbox'
        else:
            data_root_path = '/home/sam/proj/epfl/neural_clustering_data'
            experiment_root_path = f"{data_root_path}/experiments"

        # This is so much pain, but since the data is all somewhere in some folder...
        self['data_root_path'] = data_root_path
        self['experiment_root_path'] = experiment_root_path
        self['video_root_path'] = f"{data_root_path}/videos"
        self['figures_root_path'] = f"{data_root_path}/figures"
        self['grid_search_root_path'] = f"{data_root_path}/grid_search"

        self['experiment_path_template'] = '{base_path}/{study_id}/{fly_id}/{experiment_id}'
        # to be filled with `experiment_path_template` as `base_experiment_path`
        self['experiment_limb_pos_data_dir'] = '{base_experiment_path}/behData/images/'
        self['fly_image_template'] = f"{{base_experiment_path}}/behData/images/camera_{self['camera_of_interest']}_img_{{image_id:0>6}}.jpg"

    @classmethod
    def runs_on_lab_server(cls):
        return get_hostname() == 'upramdyapc6'


###############################################[MaF##
#                                               #
# Run config, model definition, hyperparameters #F
#                                               #[MaF
#################################################

# Note that not all variables will be used by all models


class RunConfig(BaseConfig):
    DEFAULT_VALUES = {
        'debug': False,                 # general flag for debug mode, triggers all `d_.*`-options.
        'd_zero_data': False,          # overwrite the data with zeroed out data, the overall shape is kept.
        'd_sinoid_data': False,
        'd_sinoid_cluster_data': True,
        'd_no_compression': False,     # if true, the latent_space will be the same dimension as the input.
                                       # allowing the model to learn the identity function.
        'use_single_fly': True,
        'data_type': DataType.POS_2D,
        'use_time_series': True,       # triggers time series application, without this the model is only dense layers
        'time_series_length': 16,      # note that this is equal to the minimal wanted receptive field length
        'conv_layer_kernel_size': 2,   # you can set either this or `n_conv_layers` to None,
                                       # it will be automatically computed.
        'n_conv_layers': None,         # you can set either this or `conv_layer_kernel_size` to None,
                                       # it will be automatically computed.
        'latent_dim': None,               # should be adapted given the input dim
        'batch_size': 128,             # if you plan on using the supervised parts as well keep this
                                       # small. it will create a batch_size^3 dim tensor to compute
                                       # the loss
        'loss_weight_reconstruction': 1.0,
        'loss_weight_kl': 0.0,             # if zero it will not even be computed
        'dropout_rate': 0.,
        'with_batch_norm': True,
        'model_impl': ModelType.SKIP_PADD_CONV,
        'optimizer': 'Adam',
        'train_test_ratio': 0.7,
        'angle_3d_params': {
            'preprocessing': {
                'low_variance_cutoff': 0.,
                'blacklist_behavior': [Behavior.REST, Behavior.NONE],
                'normalize_features': False
            }
        },
        'pos_2d_params': {
            'preprocessing': {
                'normalize_for_each_experiment': True
            },
        },
        'model_created_at': None,
        'supervised_learning_rate': 0.00005
    }

    def __init__(self, **kwargs):
        super(BaseConfig, self).__init__({**RunConfig.DEFAULT_VALUES, **kwargs})

        if self['use_single_fly']:
            self['batch_size'] = 128
            self['train_test_ratio'] = 0.9

        if not(self['data_type'] in DataType):
            raise NotImplementedError(f"This data type is not supported. Must be one of either"
                                      f"{DataType.list()}")

        if self['n_conv_layers'] is None:
            self['n_conv_layers'] = np.int(np.ceil(np.log2((self['time_series_length'] - 1) / (2 * (self['conv_layer_kernel_size'] - 1)) + 1)))

        if self['conv_layer_kernel_size'] is None:
            raise NotImplementedError('ups')

        if self['data_type'] == DataType.POS_2D:
            # goes from 15 * 2 = 30 -> 8
            self['latent_dim'] = 8
        elif self['data_type'] == DataType.ANGLE_3D:
            # goes from 18 -> 4
            self['latent_dim'] = 4
        else:
            raise ValueError(f"this data_type is not supported: {self['data_type']}")

    def description(self, short=True, verbosity=4):
        def _bool_(v):
            return 'T' if self[v] else 'F'

        def _combine_(a, b):
            if a == '':
                return b
            else:
                return f"{a}-{b}"

        valus_of_interest = [
            ('model_impl', '', self.get('model_impl').name),
            ('data', '', self['data_type'].name),
            ('time', 't', self['time_series_length'] if self['use_time_series'] else 'F'),
            ('latent_dim', 'ld', self['latent_dim']),
            ('use_single_fly', 'mf', _bool_('use_single_fly')),
            ('loss_weight_recon', 'lwr', self.get('loss_weight_reconstruction')),
            ('loss_weight_kl', 'lwkl', self.get('loss_weight_kl')),
            ('dropout_rate', 'dr', self.get('dropout_rate')),
            ('kernel', 'k', self['conv_layer_kernel_size']),
            ('optimizer', 'opt', self.get('optimizer')),
            ('with_batch_norm', 'bn', _bool_('with_batch_norm')),
            ('n_clayers', 'ncl', self['n_conv_layers']),
        ]

        valus_of_interest = valus_of_interest[:verbosity]

        descr_idx = 1 if short else 0
        descr_str = '-'.join((_combine_(v[descr_idx], v[2]) for v in valus_of_interest))


        if self['debug']:
            descr_str += '_' + ''.join([k for k, v in self.items() if k.startswith('d_') and v])

        if self['model_created_at']:
            descr_str += '_' + self['model_created_at']

        return descr_str

    def value(self, *keys):
        """Helper to return one single value (may be what ever that value is).

        E.g. value('hubert', 'fly_id')

        NOTE THAT THIS IS NOT THE SAME METHOD FROM THE BASE CLASS.
        This one here is "alive" the other is one is "dead".
        """
        try:
            val = reduce(lambda d, k: d[k], keys, self)

            if isinstance(val, str) and 'UNKOWN' in val:
                raise ValueError('{0} value is not set! Reading {1}'.format(keys, val))
        except KeyError:
            raise ValueError("Could not find a value for the given key: {}".format(keys))

        return val

    @classmethod
    def POS_2D(cls):
        return cls(data_type=DataType.POS_2D)

    @classmethod
    def ANGLE_3D(cls):
        return cls(data_type=DataType.ANGLE_3D)
