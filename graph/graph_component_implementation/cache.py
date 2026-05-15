from collections import OrderedDict, deque
from typing import Callable, Dict, Generic, Optional, Tuple, TypeVar
import sys

K = TypeVar("K")
V = TypeVar("V")


class BaseCache(Generic[K, V]):
    def get(self, key: K) -> Optional[V]:
        raise NotImplementedError

    def put(self, key: K, value: V) -> None:
        raise NotImplementedError

    def clear(self) -> None:
        raise NotImplementedError


class NoCache(BaseCache[K, V]):
    def get(self, key: K) -> Optional[V]:
        return None

    def put(self, key: K, value: V) -> None:
        return None

    def clear(self) -> None:
        return None


class LruCache(BaseCache[K, V]):
    def __init__(self, ram_limit_mb: int, size_func: Optional[Callable[[V], int]] = None) -> None:
        self.ram_limit = ram_limit_mb * 1024 * 1024
        self.size_func = size_func or (lambda v: sys.getsizeof(v))
        self.items: OrderedDict[K, Tuple[V, int]] = OrderedDict()
        self.current_size = 0

    def get(self, key: K) -> Optional[V]:
        item = self.items.get(key)
        if item is None:
            return None
        value, size = item
        self.items.move_to_end(key)
        return value

    def put(self, key: K, value: V) -> None:
        size = self.size_func(value)
        if key in self.items:
            _, old_size = self.items.pop(key)
            self.current_size -= old_size
        self.items[key] = (value, size)
        self.current_size += size
        self.items.move_to_end(key)
        while self.current_size > self.ram_limit and self.items:
            _, (_, evicted_size) = self.items.popitem(last=False)
            self.current_size -= evicted_size

    def clear(self) -> None:
        self.items.clear()
        self.current_size = 0


class ArcCache(BaseCache[K, V]):
    def __init__(self, ram_limit_mb: int, size_func: Optional[Callable[[V], int]] = None) -> None:
        self.ram_limit = ram_limit_mb * 1024 * 1024
        self.size_func = size_func or (lambda v: sys.getsizeof(v))
        self.t1: OrderedDict[K, Tuple[V, int]] = OrderedDict()
        self.t2: OrderedDict[K, Tuple[V, int]] = OrderedDict()
        self.b1: deque[K] = deque()
        self.b2: deque[K] = deque()
        self.p = 0
        self.current_size = 0

    def get(self, key: K) -> Optional[V]:
        if key in self.t1:
            value, size = self.t1.pop(key)
            self.t2[key] = (value, size)
            return value
        if key in self.t2:
            value, _ = self.t2[key]
            self.t2.move_to_end(key)
            return value
        return None

    def put(self, key: K, value: V) -> None:
        size = self.size_func(value)
        if key in self.t1 or key in self.t2:
            if key in self.t1:
                _, old_size = self.t1.pop(key)
            else:
                _, old_size = self.t2.pop(key)
            self.current_size -= old_size
        if key in self.b1:
            self.p = min(self.p + 1, self.ram_limit)
            self.b1.remove(key)
        elif key in self.b2:
            self.p = max(self.p - 1, 0)
            self.b2.remove(key)
        self._replace(size)
        self.t1[key] = (value, size)
        self.current_size += size

    def _replace(self, incoming_size: int) -> None:
        while self.current_size + incoming_size > self.ram_limit and (self.t1 or self.t2):
            if len(self.t1) > 0 and (len(self.t1) > self.p or not self.t2):
                key, (_, size) = self.t1.popitem(last=False)
                self.b1.append(key)
                self.current_size -= size
            else:
                key, (_, size) = self.t2.popitem(last=False)
                self.b2.append(key)
                self.current_size -= size

    def clear(self) -> None:
        self.t1.clear()
        self.t2.clear()
        self.b1.clear()
        self.b2.clear()
        self.current_size = 0
        self.p = 0


def build_cache(cache_type: str, ram_limit_mb: int, size_func: Optional[Callable[[V], int]] = None) -> BaseCache[K, V]:
    if cache_type == "none":
        return NoCache()
    if cache_type == "lru":
        return LruCache(ram_limit_mb, size_func=size_func)
    if cache_type == "arc":
        return ArcCache(ram_limit_mb, size_func=size_func)
    raise ValueError(f"Unknown cache type: {cache_type}")
