import diskcache as dc
import pandas.core.util.hashing
import hashlib
import inspect

def memoize(isMethod):
    """
    Wrapper function, which requires the isMethod to be passed in, otherwise returns an error
    :param isMethod:
    :return:
    """
    cache = dc.Cache('diskcache')

    def handle_value(value):
        # Alfred pure class reference should be skipped
        if inspect.isclass(value):
            key = None
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

    def key_compute(*args, **kwargs):
        key = []
        #b = locals()
        for i in range(0, len(args)):
            value = args[i]
            key.append(handle_value(value))

        for k in kwargs:
            value = kwargs[k]
            key.append(handle_value(value))

        return key

    def inner(func):
        def memoized_func(*args, **kwargs):
            key_computed = key_compute(*args, **kwargs)
            old_key_computed1 = cache['alfred1']
            #old_key_computed2 = cache['alfred2']
            if key_computed in cache:
                return cache[key_computed]
            result = func(*args, **kwargs)
            cache[key_computed] = result
            return result

        return memoized_func

    return inner
