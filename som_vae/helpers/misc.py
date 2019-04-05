from functools import reduce
import inspect

def extract_args(config, function):
    return {k:config[k] for k in inspect.getfullargspec(function).args if k in config}

def chunks(l, n):
    """Yield successive n-sized chunks from l. Use it to create batches."""
    for i in range(0, len(l), n):
        yield l[i:i + n]

def foldl(x, *functions):
    return reduce(lambda acc, el: el(acc), functions, x)
