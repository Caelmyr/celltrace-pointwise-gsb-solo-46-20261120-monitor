from collections import OrderedDict
from threading import RLock
from typing import Callable

import numpy as np


def array_bytes(value):
    if isinstance(value, np.ndarray):
        return value.nbytes
    if isinstance(value, (bytes, bytearray)):
        return len(value)
    if isinstance(value, dict):
        return 256 + sum(array_bytes(item) for item in value.values())
    if isinstance(value, (tuple, list)):
        return 128 + sum(array_bytes(item) for item in value)
    if hasattr(value, "__dict__"):
        return array_bytes(vars(value))
    return 96


class MemoryCache:
    def __init__(self, maximum_bytes: int):
        self.limit = maximum_bytes
        self.total = 0
        self.entries = OrderedDict()
        self.lock = RLock()

    def get(self, key, build: Callable):
        with self.lock:
            if key in self.entries:
                value, size = self.entries.pop(key)
                self.entries[key] = value, size
                return value
        value = build()
        size = array_bytes(value)
        if size > self.limit:
            return value
        with self.lock:
            existing = self.entries.pop(key, None)
            if existing is not None:
                existing_value, existing_size = existing
                self.entries[key] = existing_value, existing_size
                return existing_value
            while self.entries and self.total + size > self.limit:
                _, (_, removed) = self.entries.popitem(last=False)
                self.total -= removed
            self.entries[key] = value, size
            self.total += size
        return value
