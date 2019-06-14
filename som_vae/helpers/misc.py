import pathlib
from enum import Enum
import numpy as np
import socket
from functools import reduce
import inspect

def is_file(path):
    return pathlib.Path(path).is_file()

def flatten(listOfLists):
    return reduce(list.__add__, listOfLists, [])

def extract_args(config, function):
    """filters config for keys that part of function's arguments

    useful if you have functions that take too many arguments.
    """
    return {k:config[k] for k in inspect.getfullargspec(function).args if k in config}

def chunks(l, n):
    """Yield successive n-sized chunks from l. Use it to create batches."""
    for i in range(0, len(l), n):
        yield l[i:i + n]

def foldl(x, *functions):
    return reduce(lambda acc, el: el(acc), functions, x)

def get_hostname():
    return socket.gethostname()

def to_time_series(data, sequence_length):
    for i in range(len(data)):
        if i + sequence_length <= len(data):
            yield data[i:i+sequence_length]

def if_last(ls):
    for i, x in enumerate(ls):
        yield i, i + 1 == len(ls), x

def n_layers_for_dilated_conv(n_time_steps, kernel_size, dilation_rate=2):
    if dilation_rate != 2:
        raise NotImplementedError('left as an exercise for the reader')
    return np.int(np.ceil(np.log2((n_time_steps -1) / (2 * (kernel_size - 1)) + 1)))

def to_time_series_np(x, sequence_length):
    return np.array(list(to_time_series(x, sequence_length=sequence_length)))


def prep_2d_pos_data(x):
    return x[:,:,:2].reshape(x.shape[0], -1).astype(np.float32)


class EEnum(Enum):
    @classmethod
    def list(cls):
        return list(map(lambda c: c.name, cls))

    @classmethod
    def values(cls):
        return list(cls)



def create_parents(path):
    pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)


def interpolate_arrays(arr1, arr2, num_steps=100, interpolation_length=0.3):
    """Interpolates linearly between two arrays over a given number of steps.
    The actual interpolation happens only across a fraction of those steps.

    Args:
        arr1 (np.array): The starting array for the interpolation.
        arr2 (np.array): The end array for the interpolation.
        num_steps (int): The length of the interpolation array along the newly created axis (default: 100).
        interpolation_length (float): The fraction of the steps across which the actual interpolation happens (default: 0.3).

    Returns:
        np.array: The final interpolated array of shape ([num_steps] + arr1.shape).
    """
    assert arr1.shape == arr2.shape, "The two arrays have to be of the same shape"
    start_steps = int(num_steps*interpolation_length)
    inter_steps = int(num_steps*((1-interpolation_length)/2))
    end_steps = num_steps - start_steps - inter_steps
    interpolation = np.zeros([inter_steps]+list(arr1.shape))
    arr_diff = arr2 - arr1
    for i in range(inter_steps):
        interpolation[i] = arr1 + (i/(inter_steps-1))*arr_diff
    start_arrays = np.concatenate([np.expand_dims(arr1, 0)] * start_steps)
    end_arrays = np.concatenate([np.expand_dims(arr2, 0)] * end_steps)
    final_array = np.concatenate((start_arrays, interpolation, end_arrays))
    return final_array
