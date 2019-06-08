import numpy as np
import socket
from functools import reduce
import inspect

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


