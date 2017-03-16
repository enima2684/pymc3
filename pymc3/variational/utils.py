from itertools import chain


def get_transformed(v):
    if hasattr(v, 'transformed'):
        return v.transformed
    return v


def flatten(l):
    return list(chain.from_iterable(l))
