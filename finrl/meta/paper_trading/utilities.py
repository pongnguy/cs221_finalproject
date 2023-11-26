import diskcache as dc

def memoize(func):
    cache = dc.Cache('diskcache')

    def memoized_func(*args, **kwargs):
        if (args, kwargs) in cache:
            return cache[(args, kwargs)]
        result = func(*args, **kwargs)
        cache[(args, kwargs)] = result
        return result

    return memoized_func