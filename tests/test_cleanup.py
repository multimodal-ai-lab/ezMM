import os
import time
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
