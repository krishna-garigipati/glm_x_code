from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Deque, Dict, Iterable, List, Optional
import threading


class MarkovPrefetcher:
    def __init__(self, order: int, threads: int, fetch_fn: Callable[[int], None], top_k: int = 3) -> None:
        self.order = order
        self.fetch_fn = fetch_fn
        self.top_k = top_k
        self.history: Deque[int] = deque(maxlen=order)
        self.transitions: Dict[tuple[int, ...], Dict[int, int]] = defaultdict(dict)
        self.lock = threading.Lock()
        self.executor = ThreadPoolExecutor(max_workers=threads)

    def record(self, node_id: int) -> None:
        with self.lock:
            if len(self.history) == self.order:
                key = tuple(self.history)
                next_counts = self.transitions.get(key)
                if next_counts is None:
                    next_counts = {}
                    self.transitions[key] = next_counts
                next_counts[node_id] = next_counts.get(node_id, 0) + 1
            self.history.append(node_id)
            predictions = self._predict_next()
        for candidate in predictions:
            self.executor.submit(self.fetch_fn, candidate)

    def _predict_next(self) -> List[int]:
        if len(self.history) < self.order:
            return []
        key = tuple(self.history)
        next_counts = self.transitions.get(key, {})
        if not next_counts:
            return []
        sorted_items = sorted(next_counts.items(), key=lambda kv: kv[1], reverse=True)
        return [item[0] for item in sorted_items[:self.top_k]]

    def shutdown(self) -> None:
        self.executor.shutdown(wait=False)
