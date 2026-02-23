from __future__ import annotations

import numpy as np

from pwrag.retriever.caches import ProximityCache


def make_cache(
    policy="fifo",
    tolerance=0.8,
    capacity=3,
    lsh_bucket_capacity=5,
    lsh_num_hashes=64,
    lsh_dim=8,        # keep tiny for unit tests
    lsh_seed=42,
):
    return ProximityCache(
        policy=policy,
        tolerance=tolerance,
        capacity=capacity,
        lsh_bucket_capacity=lsh_bucket_capacity,
        lsh_num_hashes=lsh_num_hashes,
        lsh_dim=lsh_dim,
        lsh_seed=lsh_seed,
    )


def test_exact_match():
    c = make_cache(policy="fifo", tolerance=0.999, capacity=3)
    k = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=np.float32)
    c.insert(k, "hello")
    assert c.find(k) == "hello"


def test_approx_match():
    # pick tolerance that should allow a small perturbation
    c = make_cache(policy="fifo", tolerance=0.95, capacity=3)
    k1 = np.array([1, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32)
    k2 = np.array([0.98, 0.02, 0, 0, 0, 0, 0, 0], dtype=np.float32)  # very close
    c.insert(k1, "near")
    got = c.find(k2)
    assert got == "near", f"Expected approx match, got={got!r}"


def test_cache_miss():
    c = make_cache(policy="fifo", tolerance=0.9, capacity=3)
    k1 = np.zeros(8, dtype=np.float32)
    k2 = np.ones(8, dtype=np.float32)
    c.insert(k1, "zero")
    assert c.find(k2) is None


def test_fifo_eviction():
    c = make_cache(policy="fifo", tolerance=0.999, capacity=2)
    k1 = np.array([1, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32)
    k2 = np.array([0, 1, 0, 0, 0, 0, 0, 0], dtype=np.float32)
    k3 = np.array([0, 0, 1, 0, 0, 0, 0, 0], dtype=np.float32)

    c.insert(k1, "a")
    c.insert(k2, "b")
    c.insert(k3, "c")  # should evict k1 under FIFO

    assert c.find(k1) is None, "FIFO should evict oldest (k1)"
    assert c.find(k2) == "b"
    assert c.find(k3) == "c"
    assert len(c) == 2


def test_capacity_never_exceeded():
    c = make_cache(policy="fifo", tolerance=0.999, capacity=3)
    for i in range(20):
        k = np.eye(8, dtype=np.float32)[i % 8]
        c.insert(k, i)
        assert len(c) <= 3, f"Cache exceeded capacity: len={len(c)}"


def test_batch_find():
    c = make_cache(policy="fifo", tolerance=0.999, capacity=10)
    keys = [np.eye(8, dtype=np.float32)[i] for i in range(5)]
    for i, k in enumerate(keys):
        c.insert(k, f"v{i}")

    got = c.batch_find(keys + [np.ones(8, dtype=np.float32)])
    assert got[:5] == [f"v{i}" for i in range(5)]
    assert got[5] is None


def test_lsh_cache_smoke():
    # This is a smoke test: just make sure it constructs, inserts, finds something.
    # LSH is approximate + probabilistic, so avoid tight assertions across random data.
    c = make_cache(policy="lsh_fifo", tolerance=0.9, capacity=3, lsh_dim=8, lsh_bucket_capacity=10, lsh_num_hashes=8)

    k = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=np.float32)
    c.insert(k, "x")
    assert c.find(k) == "x"


if __name__ == "__main__":
    test_exact_match()
    test_approx_match()
    test_cache_miss()
    test_fifo_eviction()
    test_capacity_never_exceeded()
    test_batch_find()
    test_lsh_cache_smoke()
    print("All cache tests passed!")