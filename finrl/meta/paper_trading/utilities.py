import diskcache as dc
import pandas.core.util.hashing
import hashlib


def memoize(isMethod):
    """
    Wrapper function, which requires the isMethod to be passed in, otherwise returns an error
    :param isMethod:
    :return:
    """
    cache = dc.Cache('diskcache')

    def key_compute(*args, **kwargs):
        # TODO Alfred apparently does not match when dataframe in the arguments since Python cannot assess truth value
        # TODO Alfred loop through arguments and do some special processing for dataframes
        # TODO Alfred this is intended when the first argument is self.  For regular functions this should look at the value of the first variable
        key = []
        b = locals()
        for i in range(0, len(args)):
            value = args[i]
            a = str(type(value))
            #key.append(hash(i))
            # Alfred hack to account for self parameter when instantiating a class method
            if i == 0 and isMethod:
                key.append(str(type(value)))
            else:
                if hasattr(i, '__class__'):
                    match str(type(value)):
                        case "<class 'list'>":
                            key.append(value)
                        case "<class 'pandas.core.frame.DataFrame'>":
                            pandas_hash = int(hashlib.sha256(pandas.core.util.hashing.hash_pandas_object(value, index=True).values).hexdigest(), 16)
                            key.append(pandas_hash)
                        case _:
                            key.append(hash(i))
                else:
                    key.append(hash(i))
            #    class_name = str(i.__class__)
            #    if class_name == 'pandas.core.frame.DataFrame':
            #        key.append(hash(i))
            #    else:
            #        key.append(class_name)
            #else:
            #    key.append(i)
        return key
        #return (args[0].__class__, args[1:], kwargs)

    def inner(func):
        def memoized_func(*args, **kwargs):
            # Alfred matches only based on class name, not the particular instantiation which includes the memory address
            key_computed = key_compute(*args, **kwargs)
            if key_computed in cache:
                return cache[key_computed]
            result = func(*args, **kwargs)
            cache[key_computed] = result
            return result

        return memoized_func

    return inner
