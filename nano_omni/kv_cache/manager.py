from __future__ import annotations

from transformers import DynamicCache


class KVCacheManager:
    """
    Per-request KV Cache manager.
    Uses HF DynamicCache as underlying storage, tracks capacity limit.
    """

    def __init__(self, max_requests: int = 32):
        self.max_requests = max_requests
        self._caches: dict[str, DynamicCache] = {}

    def get_or_create(self, request_id: str) -> DynamicCache:
        """Return existing or new DynamicCache for request. Raises RuntimeError if at capacity."""
        if request_id in self._caches:
            return self._caches[request_id]
        if len(self._caches) >= self.max_requests:
            raise RuntimeError(
                f"KV cache capacity exceeded: max_requests={self.max_requests}"
            )
        cache = DynamicCache()
        self._caches[request_id] = cache
        return cache

    def free(self, request_id: str) -> None:
        """Release cache for request. No-op if not found."""
        self._caches.pop(request_id, None)

    def has_capacity(self) -> bool:
        return len(self._caches) < self.max_requests

    @property
    def num_active(self) -> int:
        return len(self._caches)
