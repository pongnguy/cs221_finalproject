import diskcache as dc

def memoize(func):
    cache = dc.Cache('diskcache')

    def key_compute(*args, **kwargs):
        # TODO Alfred apparently does not match when dataframe in the arguments since Python cannot assess truth value
        # TODO Alfred loop through arguments and do some special processing for dataframes
        # TODO Alfred this is intended when the first argument is self.  For regular functions this should look at the value of the first variable
        return (args[0].__class__, args[1:], kwargs)

    def memoized_func(*args, **kwargs):
        # Alfred matches only based on class name, not the particular instantiation which includes the memory address
        key_computed = key_compute(*args, **kwargs)
        if key_computed in cache:
            return cache[key_computed]
        result = func(*args, **kwargs)
        cache[key_computed] = result
        return result

    return memoized_func