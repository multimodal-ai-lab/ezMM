from pathlib import Path

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
