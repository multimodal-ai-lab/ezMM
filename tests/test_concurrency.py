"""Tests the item registry under parallel access from many threads and processes."""
import multiprocessing
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from ezmm import File, Image
from ezmm.common import item_registry
from ezmm.common.vector_index import truncate

N_THREADS = 16


def _make_files(root: Path, n: int, prefix: str = "f") -> list[Path]:
    root.mkdir(parents=True, exist_ok=True)
    paths = []
    for i in range(n):
        path = root / f"{prefix}{i}.txt"
        path.write_text(f"{prefix} file number {i}", encoding="utf-8")
        paths.append(path)
    return paths


def test_parallel_registration(tmp_path):
    files = _make_files(tmp_path, 200)
    with ThreadPoolExecutor(N_THREADS) as pool:
        items = list(pool.map(lambda p: File(p, source_url=f"https://example.com/{p.name}"), files))
    assert sorted(item.id for item in items) == list(range(1, 201))  # Unique IDs, no gaps
    assert item_registry.count_items() == 200
    for path, item in zip(files, items):
        assert item_registry.get_by_source_url(f"https://example.com/{path.name}") is item


def test_parallel_duplicates(tmp_path):
    """Threads registering the same content concurrently all end up with the same item,
    and the temporary files of the duplicates get deleted."""
    data = Path("in/sample.pdf").read_bytes()
    with ThreadPoolExecutor(N_THREADS) as pool:
        items = list(pool.map(lambda i: File(binary_data=data, suffix=".pdf", source_url=f"https://mirror{i}.com/a.pdf"),
                              range(50)))
    assert len({item.id for item in items}) == 1
    assert item_registry.count_items() == 1
    assert len(item_registry.get_source_urls("file", items[0].id)) == 50
    assert len(list((item_registry.path / "items").glob("*.pdf"))) == 1  # Only the adopted file is left


def test_parallel_loading_returns_one_instance(tmp_path):
    files = _make_files(tmp_path, 20)
    references = [File(path).reference for path in files]
    item_registry.clear_cache()
    with ThreadPoolExecutor(N_THREADS) as pool:
        loaded = list(pool.map(item_registry.get, references * 10))
    for reference in references:
        assert len({id(item) for item in loaded if item.reference == reference}) == 1


def test_lookups_are_free_of_writes(tmp_path):
    File(_make_files(tmp_path, 1)[0], source_url="https://example.com/a.txt")
    last_accessed = item_registry.get_sources("file", 1)[0]["last_accessed"]
    item_registry.conn.execute("PRAGMA query_only = ON;")  # Any write would fail now
    try:
        assert item_registry.get_by_source_url("https://example.com/a.txt") is not None
        File(tmp_path / "f0.txt", source_url="https://example.com/a.txt")
    finally:
        item_registry.conn.execute("PRAGMA query_only = OFF;")
    assert item_registry.get_sources("file", 1)[0]["last_accessed"] == last_accessed


def test_threads_use_own_connections(tmp_path):
    connections = set()
    main_connection = id(item_registry.conn)

    def connect(_):
        item_registry.count_items()
        return id(item_registry.conn)

    with ThreadPoolExecutor(4) as pool:
        connections.update(pool.map(connect, range(40)))
    assert main_connection not in connections
    assert 1 <= len(connections) <= 4

    # Closing the registry closes the connections of all threads
    item_registry.close()
    assert len(item_registry._connections) == 0
    assert item_registry.count_items() == 0  # Reconnects on demand


def test_index_loading_does_not_block(tmp_path, monkeypatch):
    """While the search index loads, the registry stays usable, concurrent searches wait for
    the index, and embeddings added during loading are not lost."""
    items = [Image(p) for p in ["in/roses.jpg", "in/garden.jpg", "in/tulips.jpg"]]
    vectors = truncate(np.random.default_rng(0).standard_normal((4, 768)), 768)
    item_registry.set_embeddings("test-model", [(i.kind, i.id, v) for i, v in zip(items, vectors)])

    loading, proceed = threading.Event(), threading.Event()
    original_load = item_registry._load_index

    def slow_load(index, model):
        loading.set()
        proceed.wait(timeout=10)
        original_load(index, model)

    monkeypatch.setattr(item_registry, "_load_index", slow_load)
    with ThreadPoolExecutor(4) as pool:
        searches = [pool.submit(item_registry.search, vectors[0], "test-model", device="cpu") for _ in range(3)]
        assert loading.wait(timeout=10)
        # The registry is usable while the index loads, also for writes
        assert item_registry.count_items() == 3
        new_item = File(_make_files(tmp_path, 1)[0])
        item_registry.set_embedding(new_item.kind, new_item.id, "test-model", vectors[3])
        proceed.set()
        results = [search.result(timeout=30) for search in searches]
    assert all(r[0]["reference"] == items[0].reference for r in results)
    assert item_registry.search(vectors[3], "test-model", device="cpu")[0]["reference"] == new_item.reference


def _register_in_process(args):
    registry_path, files, worker = args
    from ezmm import File, set_ezmm_path
    set_ezmm_path(registry_path)
    return [File(path, source_url=f"https://example.com/{worker}/{path.name}").id for path in files]


def test_parallel_processes(tmp_path):
    item_registry.connect()
    chunks = [_make_files(tmp_path / f"w{w}", 25, f"w{w}_") for w in range(4)]
    with multiprocessing.get_context("spawn").Pool(4) as pool:
        ids = pool.map(_register_in_process, [(item_registry.path, chunk, w) for w, chunk in enumerate(chunks)])
    assert sorted(i for chunk in ids for i in chunk) == list(range(1, 101))
    assert item_registry.count_items() == 100
