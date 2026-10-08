import sqlite3
from pathlib import Path
from shutil import copyfile

from ezmm import Item
from ezmm.common import item_registry
from ezmm.common.registry import SCHEMA_VERSION


def _create_legacy_registry(root: Path):
    """Creates a registry with the legacy (v1) schema: one table per item kind and absolute paths."""
    (root / "image").mkdir(parents=True)
    copyfile("in/roses.jpg", root / "image" / "1.jpg")
    copyfile("in/tulips.jpg", root / "image" / "3.jpg")
    copyfile("in/roses.jpg", root / "image" / "4.jpg")  # A duplicate of image 1
    (root / "items").mkdir()
    copyfile("in/roses_smaller.jpg", root / "items" / "2025-01-01_00-00-00-000000.jpg")
    conn = sqlite3.connect(root / "item_registry.db")
    for kind in ["image", "video", "audio"]:
        conn.execute(f"CREATE TABLE {kind} (id INTEGER PRIMARY KEY, path TEXT NOT NULL UNIQUE, "
                     f"source_url TEXT NOT NULL);")
        conn.execute(f"CREATE UNIQUE INDEX {kind}_path_idx ON {kind}(path);")
    conn.executemany("INSERT INTO image(id, path, source_url) VALUES (?, ?, ?);", [
        (1, (root / "image" / "1.jpg").as_posix(), "https://example.com/roses.jpg"),
        (2, Path("in/garden.jpg").absolute().as_posix(), Path("in/garden.jpg").absolute().as_uri()),
        (3, "X:/moved/registry/image/3.jpg", "https://example.com/tulips.jpg"),  # Stale absolute path
        (4, (root / "image" / "4.jpg").as_posix(), "https://mirror.com/roses.jpg"),
        (5, "X:/moved/registry/items/2025-01-01_00-00-00-000000.jpg", "https://example.com/small.jpg"),
    ])
    conn.executemany("INSERT INTO video(id, path, source_url) VALUES (?, ?, ?);", [
        (7, Path("in/mountains.mp4").absolute().as_posix(), "https://example.com/mountains.mp4"),
        (8, "X:/gone/forever.mp4", "https://example.com/gone.mp4"),  # Unrecoverable
    ])
    conn.commit()
    conn.close()


def test_migration():
    root = item_registry.path
    _create_legacy_registry(root)
    item_registry.connect()  # Triggers the automatic migration

    conn = item_registry.conn
    assert conn.execute("PRAGMA user_version;").fetchone()[0] == SCHEMA_VERSION
    tables = {name for (name,) in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table';")}
    assert {"items", "sources"} <= tables
    assert not {"image", "video", "audio"} & tables
    assert (root / "item_registry.v1.bak.db").exists()

    # IDs are preserved, paths are relative inside the registry and absolute outside of it
    rows = {(kind, i): path for kind, i, path in conn.execute("SELECT kind, id, path FROM items;")}
    assert rows[("image", 1)] == "image/1.jpg"
    assert rows[("image", 2)] == Path("in/garden.jpg").absolute().as_posix()
    assert rows[("image", 3)] == "image/3.jpg"  # Healed via default location
    assert rows[("image", 5)] == "items/2025-01-01_00-00-00-000000.jpg"  # Healed via original folder
    assert rows[("video", 7)] == Path("in/mountains.mp4").absolute().as_posix()

    # Hashes and sizes are filled in, missing files are flagged
    assert conn.execute("SELECT kind, id FROM items WHERE sha256 IS NULL OR size IS NULL;").fetchall() == [("video", 8)]
    assert conn.execute("SELECT kind, id FROM items WHERE missing = 1;").fetchall() == [("video", 8)]

    # References still resolve
    assert Item.from_reference("<image:3>").source_url == "https://example.com/tulips.jpg"
    assert Item.from_reference("<video:7>").source_urls == ["https://example.com/mountains.mp4"]


def test_migration_is_idempotent():
    _create_legacy_registry(item_registry.path)
    item_registry.connect()
    n_items = item_registry.count_items()
    item_registry.migrate()
    item_registry.reset()
    assert item_registry.count_items() == n_items == 7


def test_migration_then_deduplication():
    _create_legacy_registry(item_registry.path)
    item_registry.connect()
    report = item_registry.deduplicate()
    assert report["removed"] == [("image", 4, 1)]
    assert Item.from_reference("<image:4>") is Item.from_reference("<image:1>")
    assert Item.from_reference("<image:1>").source_urls == ["https://example.com/roses.jpg",
                                                            "https://mirror.com/roses.jpg"]


def test_new_registry_has_current_schema():
    item_registry.connect()
    assert item_registry.conn.execute("PRAGMA user_version;").fetchone()[0] == SCHEMA_VERSION
    assert not (item_registry.path / "item_registry.v1.bak.db").exists()


# The schema of ezMM 0.6.0 (v2), i.e., without the `missing` column
SCHEMA_V2 = """
    CREATE TABLE items (
        row_id INTEGER PRIMARY KEY, kind TEXT NOT NULL, id INTEGER NOT NULL, path TEXT, sha256 TEXT,
        size INTEGER, canonical_id INTEGER, created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
        UNIQUE (kind, id));
    CREATE TABLE sources (
        id INTEGER PRIMARY KEY, url TEXT NOT NULL UNIQUE, item_row_id INTEGER NOT NULL REFERENCES items(row_id),
        created_at TEXT NOT NULL, last_accessed TEXT NOT NULL);
    INSERT INTO items(kind, id, path, sha256, size, canonical_id, created_at, updated_at) VALUES
        ('image', 1, 'image/1.jpg', 'abc', 1, NULL, '2026-01-01', '2026-01-01'),
        ('image', 2, 'image/gone.jpg', 'def', 1, NULL, '2026-01-01', '2026-01-01'),
        ('image', 3, NULL, NULL, NULL, 1, '2026-01-01', '2026-01-01');
    INSERT INTO sources(url, item_row_id, created_at, last_accessed) VALUES
        ('https://example.com/1.jpg', 1, '2026-01-01', '2026-01-01');
    PRAGMA user_version = 2;
"""


def test_migration_from_v2():
    root = item_registry.path
    (root / "image").mkdir(parents=True)
    copyfile("in/roses.jpg", root / "image" / "1.jpg")
    conn = sqlite3.connect(root / "item_registry.db")
    conn.executescript(SCHEMA_V2)
    conn.close()

    item_registry.connect()
    assert item_registry.conn.execute("PRAGMA user_version;").fetchone()[0] == SCHEMA_VERSION
    assert (root / "item_registry.v2.bak.db").exists()
    assert not item_registry.get_row("image", 1)["missing"]
    assert item_registry.get_row("image", 2)["missing"]
    assert item_registry.count_items(include_missing=False) == 1
    assert item_registry.get_source_urls("image", 1) == ["https://example.com/1.jpg"]
    assert Item.from_reference("<image:3>").id == 1  # Aliases still resolve
