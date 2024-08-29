import os
import shutil
import sys
from collections import OrderedDict

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kylin.retriever.cache import SQLLRUCache


def test_cache():
    # test init
    if os.path.exists("/tmp/cache_test/cache.db"):
        os.remove("/tmp/cache_test/cache.db")
    cache = SQLLRUCache("/tmp/cache_test/cache.db", maxsize=3)
    assert cache.maxsize == 3

    # test setitem
    cache["a"] = "a"
    cache["b"] = "b"
    cache["c"] = "c"
    assert list(cache.keys()) == ["a", "b", "c"]
    assert list(cache.values()) == ["a", "b", "c"]
    cache["a"] = "x"
    assert len(cache) == 3
    assert list(cache.keys()) == ["b", "c", "a"]
    assert list(cache.values()) == ["b", "c", "x"]
    cache["d"] = "d"
    assert list(cache.keys()) == ["c", "a", "d"]
    assert list(cache.values()) == ["c", "x", "d"]

    # test popitem
    x = cache.popitem()
    assert x == ("c", "c")
    assert list(cache.keys()) == ["a", "d"]
    assert list(cache.values()) == ["x", "d"]

    # test pop
    y = cache.pop("a")
    assert y == "x"
    assert list(cache.keys()) == ["d"]
    assert list(cache.values()) == ["d"]

    # test to_dict
    assert cache.to_dict() == OrderedDict({"d": "d"})
    assert len(cache) == 1

    # test loading from disk
    del cache
    cache = SQLLRUCache("/tmp/cache_test/cache.db", maxsize=3)
    assert cache.to_dict() == OrderedDict({"d": "d"})
    assert len(cache) == 1

    # test clean
    cache.clean()
    assert len(cache) == 0

    # cleanup
    shutil.rmtree("/tmp/cache_test")
    return
