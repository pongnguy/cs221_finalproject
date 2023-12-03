#import diskcache as dc
import pandas.core.util.hashing
import hashlib
import inspect
import os
from functools import wraps
import time

# importing the hashlib module
import hashlib

def hash_file(filename):
    """"This function returns the SHA-1 hash
    of the file passed into it"""

    # make a hash object
    h = hashlib.sha1()

    # open file for reading in binary mode
    with open(filename,'rb') as file:

        # loop till the end of the file
        chunk = 0
        while chunk != b'':
            # read only 1024 bytes at a time
            chunk = file.read(1024)
            h.update(chunk)

    # return the hex representation of digest
    return h.hexdigest()

def timeit(func):
    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        print(f'Function {func.__name__} Took {total_time:.4f} seconds')
        #print(f'Function {func.__name__}{args} {kwargs} Took {total_time:.4f} seconds')
        return result
    return timeit_wrapper


def handle_value(value):
    # Alfred pure class reference hash changes
    if inspect.isclass(value):
        key = str(type(value))
    else:
        try:
            # Alfred all instantiated classes should implement the __hash__ function
            key = hash(value)
        except:
            match str(type(value)):
                case "<class 'list'>":
                    key = value
                case "<class 'pandas.core.frame.DataFrame'>":
                    pandas_hash = int(hashlib.sha256(pandas.core.util.hashing.hash_pandas_object(value, index=True).values).hexdigest(), 16)
                    key = pandas_hash
                case _:
                    key = str(type(value))
    return key

def keycompute(*args, **kwargs):
    key = []
    #b = locals()
    for i in range(0, len(args)):
        value = args[i]
        key.append(handle_value(value))

    for k in kwargs:
        value = kwargs[k]
        key.append(handle_value(value))

    return key

def memoize(keyspace='unnamed'):
    """
    Wrapper function, which requires bucket parameter to be passed in
    :param keyspace: the keyspace for the values
    :return: wrapped function
    """
    path = f'diskcache/{keyspace}'
    if not os.path.exists(path):
        os.makedirs(path)

    cache = dc.Cache(path, EVICTION_POLICY='none')

    def inner(func):
        def memoized_func(*args, **kwargs):
            key_computed = keycompute(*args, **kwargs)
            #old_key_computed1 = cache['alfred1']
            #old_key_computed2 = cache['alfred2']
            if key_computed in cache:
                return cache[key_computed]
            result = func(*args, **kwargs)
            cache[key_computed] = result
            return result

        return memoized_func

    return inner


# TODO Alfred can I inherit diskcache and overload function used to computed keys from arguments??
#class AlfredCache(dc):
#    def __cache_key__(self, *args, **kwargs):
#        a=2