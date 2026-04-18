import pytest
from transformers import DynamicCache
from nano_omni.kv_cache.manager import KVCacheManager


def test_manager_initial_state():
    mgr = KVCacheManager(max_requests=4)
    assert mgr.num_active == 0
    assert mgr.has_capacity() is True


def test_get_or_create_returns_cache():
    mgr = KVCacheManager(max_requests=4)
    cache = mgr.get_or_create("req_1")
    assert isinstance(cache, DynamicCache)


def test_get_or_create_same_object():
    mgr = KVCacheManager(max_requests=4)
    c1 = mgr.get_or_create("req_1")
    c2 = mgr.get_or_create("req_1")
    assert c1 is c2


def test_capacity_tracking():
    mgr = KVCacheManager(max_requests=2)
    mgr.get_or_create("req_1")
    assert mgr.num_active == 1
    assert mgr.has_capacity() is True
    mgr.get_or_create("req_2")
    assert mgr.num_active == 2
    assert mgr.has_capacity() is False


def test_free_releases_slot():
    mgr = KVCacheManager(max_requests=2)
    mgr.get_or_create("req_1")
    mgr.get_or_create("req_2")
    assert mgr.has_capacity() is False
    mgr.free("req_1")
    assert mgr.num_active == 1
    assert mgr.has_capacity() is True


def test_free_nonexistent_is_noop():
    mgr = KVCacheManager(max_requests=4)
    mgr.free("does_not_exist")   # should not raise


def test_get_cache_raises_when_full():
    mgr = KVCacheManager(max_requests=1)
    mgr.get_or_create("req_1")
    with pytest.raises(RuntimeError, match="KV cache capacity"):
        mgr.get_or_create("req_2")
