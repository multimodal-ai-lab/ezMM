import os
import time
from datetime import UTC, datetime
from pathlib import Path

from ezmm import Image
from ezmm.common import item_registry

HOUR = 3600


def _age(path: Path, age: float = 2 * HOUR):
    """Sets the file's modification time `age` seconds into the past."""
    past = time.time() - age
    os.utime(path, (past, past))


def _write(path: Path, data: bytes = b"orphan", age: float = 2 * HOUR) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    _age(path, age)
    return path


def test_orphans_get_deleted():
    img = Image("in/roses.jpg")
    img.relocate()
    _age(img.file_path)
    root = item_registry.path
    orphans = [_write(root / "image" / "99.jpg"), _write(root / "video" / "5.mp4"),
               _write(root / "items" / "2020-01-01_00-00-00-000000_abcdef12.png"),
               _write(root / "file" / "sub" / "old.pdf")]

    report = item_registry.remove_orphaned_files()
    assert sorted(report["orphans"]) == sorted(path.as_posix() for path in orphans)
    assert report["deleted"] == 4
    assert report["freed_bytes"] == 4 * len(b"orphan")
    assert report["scanned"] == 5
    assert not any(path.exists() for path in orphans)
    assert img.file_path.exists()  # Referenced


def test_dry_run_deletes_nothing():
    orphan = _write(item_registry.path / "image" / "1.jpg")
    report = item_registry.remove_orphaned_files(dry_run=True)
    assert report["orphans"] == [orphan.as_posix()]
    assert report["deleted"] == 0
    assert orphan.exists()


def test_referenced_files_are_kept(tmp_path):
    outside = Image("in/roses.jpg")  # File outside of the registry
    missing = Image("in/garden.jpg")
    missing.relocate()
    _age(missing.file_path)
    item_registry.set_missing("image", missing.id, True)  # Flagged as missing, but the file exists
    temp = Image(binary_data=Path("in/tulips.jpg").read_bytes())  # Temporary file in items/
    assert item_registry.is_temp_path(temp.file_path)
    _age(temp.file_path)

    report = item_registry.remove_orphaned_files()
    assert report["orphans"] == []
    assert report["scanned"] == 2
    assert outside.file_path.exists() and missing.file_path.exists() and temp.file_path.exists()


def test_paths_are_compared_normalized():
    img = Image("in/roses.jpg")
    img.relocate()
    _age(img.file_path)
    # A non-canonical absolute path pointing into the registry (e.g., written by another tool)
    stored = (item_registry.path / "image" / ".." / "image" / img.file_path.name).as_posix()
    if os.name == "nt":
        stored = stored.upper()  # Windows paths are case-insensitive
    with item_registry._transaction() as conn:
        conn.execute("UPDATE items SET path = ? WHERE kind = 'image' AND id = ?;", (stored, img.id))

    report = item_registry.remove_orphaned_files()
    assert report["orphans"] == []
    assert img.file_path.exists()


def test_recent_files_are_skipped():
    recent = _write(item_registry.path / "items" / "in_flight.jpg", age=0)
    old = _write(item_registry.path / "items" / "old.jpg", age=2 * HOUR)
    report = item_registry.remove_orphaned_files()
    assert report["skipped_recent"] == 1
    assert report["orphans"] == [old.as_posix()]
    assert recent.exists() and not old.exists()

    report = item_registry.remove_orphaned_files(min_age=0)  # No grace period
    assert report["orphans"] == [recent.as_posix()]
    assert not recent.exists()


def test_other_registry_files_are_untouched():
    item_registry.connect()
    root = item_registry.path
    others = [_write(root / "item_registry.v2.bak.db"), _write(root / "sequences" / "12345678.html"),
              _write(root / "notes.txt"), _write(root / "other" / "1.jpg")]
    report = item_registry.remove_orphaned_files(min_age=0)
    assert report["orphans"] == []
    assert all(path.exists() for path in others)
    assert (root / "item_registry.db").exists()


def test_default_locations_are_referenced(tmp_path):
    """An item with a stale stored path heals to its default location, so that file is not orphaned."""
    item_registry.connect()
    default = _write(item_registry.path / "image" / "7.jpg")
    _insert_items([(7, (tmp_path / "moved" / "7.jpg").as_posix(), None)])
    report = item_registry.remove_orphaned_files(min_age=0)
    assert report["orphans"] == []
    assert default.exists()


# -------------------------------------------------------------------------------------------------
# Dead entries

OLD = "2026-01-01T00:00:00+00:00"


def _insert_items(rows: list[tuple], timestamp: str = OLD, missing: int = 0):
    """Inserts registry rows (id, path, canonical_id) of images directly."""
    with item_registry._transaction() as conn:
        conn.executemany("""
            INSERT INTO items(kind, id, path, sha256, size, canonical_id, missing, created_at, updated_at)
            VALUES ('image', ?, ?, NULL, 6, ?, ?, ?, ?);
        """, [(*row, missing, timestamp, timestamp) for row in rows])


def _row_ids(*identifiers: int) -> list[int]:
    return [item_registry.conn.execute("SELECT row_id FROM items WHERE kind = 'image' AND id = ?;",
                                       (i,)).fetchone()[0] for i in identifiers]


def _count(table: str) -> int:
    return item_registry.conn.execute(f"SELECT COUNT(*) FROM {table};").fetchone()[0]


def test_dead_entries_get_purged():
    item_registry.connect()
    kept = _write(item_registry.path / "image" / "4.jpg")
    # Dead item 1 with an alias chain 3 -> 2 -> 1; item 4 is flagged missing, but its file exists
    _insert_items([(1, "image/1.jpg", None), (2, None, 1), (3, None, 2)])
    _insert_items([(4, "image/4.jpg", None)], missing=1)
    dead_row, alias_row, kept_row = _row_ids(1, 2, 4)
    with item_registry._transaction() as conn:
        conn.executemany("INSERT INTO sources(url, item_row_id, created_at, last_accessed) VALUES (?, ?, ?, ?);",
                         [("https://a.org/1.jpg", dead_row, OLD, OLD), ("https://a.org/2.jpg", alias_row, OLD, OLD),
                          ("https://a.org/4.jpg", kept_row, OLD, OLD)])
        conn.executemany("INSERT INTO embeddings(item_row_id, model, vector, created_at) VALUES (?, 'm', x'00', ?);",
                         [(dead_row, OLD), (alias_row, OLD), (kept_row, OLD)])

    report = item_registry.remove_dead_entries()
    assert report["checked"] == 2
    assert report["removed"] == [("image", 1)]
    assert report["removed_aliases"] == 2
    assert report["healed"] == report["skipped_unreachable"] == report["skipped_recent"] == 0
    for identifier in (1, 2, 3):
        assert item_registry.get_row("image", identifier) is None
        assert item_registry.get(kind="image", identifier=identifier) is None
    assert _count("items") == _count("sources") == _count("embeddings") == 1
    assert item_registry.get_source_urls("image", 4) == ["https://a.org/4.jpg"]
    assert kept.exists()


def test_existing_files_are_kept():
    img = Image("in/roses.jpg")  # File outside of the registry
    item_registry.set_missing("image", img.id, True)
    with item_registry._transaction() as conn:
        conn.execute("UPDATE items SET created_at = ?, updated_at = ?;", (OLD, OLD))
    report = item_registry.remove_dead_entries()
    assert report["checked"] == 1 and report["removed"] == []
    assert item_registry.get_row("image", img.id) is not None


def test_dead_entries_heal_with_default_location(tmp_path):
    item_registry.connect()
    default = _write(item_registry.path / "image" / "7.jpg")
    _insert_items([(7, (tmp_path / "moved" / "7.jpg").as_posix(), None)], missing=1)  # Stale path

    report = item_registry.remove_dead_entries()
    assert report["healed"] == 1 and report["removed"] == []
    row = item_registry.get_row("image", 7)
    assert row["path"] == default and not row["missing"]
    # The healed item's file survives the subsequent orphan cleanup
    assert item_registry.remove_orphaned_files(min_age=0)["orphans"] == []
    assert default.exists()
    assert item_registry.get(kind="image", identifier=7).file_path == default


def test_unreachable_locations_are_skipped(tmp_path):
    item_registry.connect()
    (tmp_path / "mounted").mkdir()
    _insert_items([(1, (tmp_path / "unmounted" / "1.jpg").as_posix(), None),  # Folder is missing, too
                   (2, (tmp_path / "mounted" / "2.jpg").as_posix(), None),  # Folder exists, file is gone
                   (3, "image/3.jpg", None)])  # Inside the registry, folder is missing: files are gone
    report = item_registry.remove_dead_entries()
    assert report["skipped_unreachable"] == 1
    assert report["removed"] == [("image", 2), ("image", 3)]
    assert item_registry.get_row("image", 1) is not None


def test_recent_entries_are_skipped():
    item_registry.connect()
    _insert_items([(1, "image/1.jpg", None)], timestamp=datetime.now(UTC).isoformat(timespec="seconds"))
    report = item_registry.remove_dead_entries()
    assert report["skipped_recent"] == 1
    assert report["checked"] == 0 and report["removed"] == []
    assert item_registry.get_row("image", 1) is not None

    report = item_registry.remove_dead_entries(min_age=0)  # No grace period
    assert report["removed"] == [("image", 1)]
    assert item_registry.get_row("image", 1) is None


def test_entries_changed_during_check_are_kept(monkeypatch):
    """An item updated by another process after its file check (e.g., healed) does not get purged."""
    item_registry.connect()
    _insert_items([(1, "image/1.jpg", None), (2, "image/2.jpg", None)])
    check = item_registry._file_status

    def check_and_update(row):
        if row[1] == 1:
            with item_registry._transaction() as conn:
                conn.execute("UPDATE items SET updated_at = '2026-02-01T00:00:00+00:00' WHERE id = 1;")
        return check(row)

    monkeypatch.setattr(item_registry, "_file_status", check_and_update)
    report = item_registry.remove_dead_entries()
    assert report["removed"] == [("image", 2)]
    assert item_registry.get_row("image", 1) is not None


def test_dead_entries_dry_run_changes_nothing(tmp_path):
    item_registry.connect()
    _write(item_registry.path / "image" / "7.jpg")
    _insert_items([(1, "image/1.jpg", None), (2, None, 1), (7, (tmp_path / "moved" / "7.jpg").as_posix(), None)])
    rows = item_registry.conn.execute("SELECT * FROM items ORDER BY row_id;").fetchall()

    report = item_registry.remove_dead_entries(dry_run=True)
    assert report["removed"] == [("image", 1)]
    assert report["removed_aliases"] == 1 and report["healed"] == 1
    assert item_registry.conn.execute("SELECT * FROM items ORDER BY row_id;").fetchall() == rows
