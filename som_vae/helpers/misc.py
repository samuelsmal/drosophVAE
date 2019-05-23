import socket
from functools import reduce
import inspect


def flatten(listOfLists):
    return reduce(list.__add__, listOfLists, [])

def extract_args(config, function):
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

