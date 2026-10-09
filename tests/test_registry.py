from pathlib import Path
from shutil import copyfile

import pytest

from ezmm import Image
from ezmm.common import item_registry


def test_registry():
    img = Image("in/roses.jpg")  # Load the image to automatically register it in the registry
    assert item_registry.get(img.reference) is img


def test_cache_miss():
    img1 = Image("in/roses.jpg")

    # Reset cache (as if the registry was just restarted with an existing DB)
    item_registry.cache = dict()

    img2 = Image("in/roses.jpg")

    assert img1 is not img2  # Due to cache miss, but...
    assert img1.id == img2.id


def test_reference_loads_from_db():
    img1 = Image("in/roses.jpg")
    item_registry.clear_cache()
    img2 = Image(reference=img1.reference)
    assert img2 == img1
    assert Image(reference=img1.reference) is img2  # Now cached


def test_paths_relative_inside_registry():
    img = Image("in/roses.jpg")
    outside_path = item_registry.get_row("image", img.id)["path"]
    stored = item_registry.conn.execute("SELECT path FROM items WHERE id = ?;", (img.id,)).fetchone()[0]
    assert stored == Path("in/roses.jpg").absolute().as_posix()  # Outside: absolute
    assert outside_path == Path("in/roses.jpg").absolute()

    img.relocate()
    stored = item_registry.conn.execute("SELECT path FROM items WHERE id = ?;", (img.id,)).fetchone()[0]
    assert stored == f"image/{img.id}.jpg"  # Inside: relative to the registry root


def test_timestamps():
    img = Image("in/roses.jpg")
    row = item_registry.get_row("image", img.id)
    assert row["created_at"] and row["updated_at"]
    item_registry.conn.execute("UPDATE items SET updated_at = '2000-01-01' WHERE id = ?;", (img.id,))
    img.relocate()
    assert item_registry.get_row("image", img.id)["updated_at"] > "2000-01-01"


def test_get_by_source_url():
    img = Image("in/roses.jpg", source_url="https://example.com/roses.jpg")
    assert item_registry.get_by_source_url("https://example.com/roses.jpg") is img
    assert item_registry.get_by_source_url("https://example.com/roses.jpg", kind="image") is img
    assert item_registry.get_by_source_url("https://example.com/roses.jpg", kind="video") is None
    assert item_registry.get_by_source_url("https://example.com/unknown.jpg") is None


def test_source_last_accessed():
    url = "https://example.com/roses.jpg"
    Image("in/roses.jpg", source_url=url)
    item_registry.conn.execute("UPDATE sources SET last_accessed = '2000-01-01' WHERE url = ?;", (url,))
    item_registry.get_by_source_url(url)
    assert item_registry.get_sources("image", 1)[0]["last_accessed"] > "2000-01-01"


def test_urls_stored_once():
    url = "https://example.com/roses.jpg"
    for _ in range(3):
        Image("in/roses.jpg", source_url=url)
        item_registry.clear_cache()
    assert item_registry.conn.execute("SELECT COUNT(*) FROM sources WHERE url = ?;", (url,)).fetchone()[0] == 1


def test_list_and_stats():
    img = Image("in/roses.jpg", source_url="https://flowers.com/roses.jpg")
    Image("in/garden.jpg")
    assert item_registry.count_items() == 2
    assert item_registry.count_items(query="flowers.com") == 1
    assert item_registry.list_items(query="flowers.com")[0]["id"] == img.id
    assert item_registry.list_items(query=img.sha256)[0]["id"] == img.id
    assert item_registry.stats()["image"]["count"] == 2


def _make_missing_image(tmp_path) -> Image:
    gone = tmp_path / "gone.jpg"
    copyfile("in/garden.jpg", gone)
    img = Image(gone)
    gone.unlink()
    return img


def test_exclude_missing(tmp_path):
    missing_img = _make_missing_image(tmp_path)
    img = Image("in/roses.jpg")
    # The flag is not updated until the file is found missing
    assert item_registry.count_items(include_missing=False) == 2

    result = item_registry.check_files()
    assert result == dict(checked=2, missing=1, changed=1)
    assert item_registry.get_row("image", missing_img.id)["missing"]
    assert item_registry.count_items() == 2
    assert item_registry.count_items(include_missing=False) == 1
    assert [e["id"] for e in item_registry.list_items(include_missing=False)] == [img.id]
    assert {e["id"] for e in item_registry.list_items()} == {img.id, missing_img.id}


def test_reinstantiation_marks_missing(tmp_path):
    img = _make_missing_image(tmp_path)
    item_registry.clear_cache()
    with pytest.raises(FileNotFoundError):
        Image.from_id(img.id)
    assert item_registry.get_row("image", img.id)["missing"]


def test_reinstantiation_ignores_missing_flag():
    img = Image("in/roses.jpg")
    item_registry.set_missing("image", img.id, True)  # Wrong flag
    item_registry.clear_cache()
    loaded = Image.from_id(img.id)  # Must not rely on the flag
    assert loaded == img
    assert not item_registry.get_row("image", img.id)["missing"]  # Flag got corrected


def test_relocation_clears_missing_flag():
    img = Image("in/roses.jpg")
    item_registry.set_missing("image", img.id, True)
    img.relocate()
    assert not item_registry.get_row("image", img.id)["missing"]


def test_check_files_clears_flag():
    img = Image("in/roses.jpg")
    item_registry.set_missing("image", img.id, True)
    assert item_registry.check_files() == dict(checked=1, missing=0, changed=1)
    assert not item_registry.get_row("image", img.id)["missing"]


def test_compute_sha256():
    import hashlib

    from ezmm.common.registry import compute_sha256
    data = Path("in/mountains.mp4").read_bytes()
    assert compute_sha256(Path("in/mountains.mp4")) == hashlib.sha256(data).hexdigest()


def test_parallel_map_preserves_order():
    from ezmm.common.registry import _parallel_map
    assert _parallel_map(lambda x: x * 2, range(5000)) == [x * 2 for x in range(5000)]
