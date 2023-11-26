import diskcache as dc

def memoize(func):
    cache = dc.Cache('diskcache')

    def memoized_func(**kwargs):
        if kwargs in cache:
            return cache[kwargs]
        result = func(**kwargs)
        cache[kwargs] = result
        return result

    return memoized_func