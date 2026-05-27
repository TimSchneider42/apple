from __future__ import annotations

import inspect
from typing import Callable


def cached(
    fn: Callable, key: Callable = lambda **kwargs: tuple(sorted(kwargs.items()))
):
    cache = {}

    def inner(*args, **kwargs):
        kwargs_full = inspect.getcallargs(fn, *args, **kwargs)
        key_val = key(**kwargs_full)
        if key_val not in cache:
            cache[key_val] = fn(*args, **kwargs)
        return cache[key_val]

    return inner
