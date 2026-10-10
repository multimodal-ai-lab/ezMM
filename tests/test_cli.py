import pytest

from ezmm import Image, embedding
from ezmm.__main__ import main
from ezmm.common import item_registry


def test_check(capsys):
    Image("in/roses.jpg")
    main(["--path", str(item_registry.path), "check"])
    out = capsys.readouterr().out
    assert "Checked 1 items: 0 files missing" in out
    assert "0 file sizes updated" in out and "Total size:" in out


def test_dedup_dry_run(capsys):
    main(["--path", str(item_registry.path), "dedup", "--dry-run"])
    assert "[DRY RUN] Would remove 0 duplicates" in capsys.readouterr().out


def test_cleanup(capsys):
    from tests.test_cleanup import _write
    orphans = [_write(item_registry.path / "items" / f"{i}.jpg") for i in range(25)]
    _write(item_registry.path / "items" / "recent.jpg", age=0)

    main(["cleanup", "--path", str(item_registry.path), "--dry-run"])
    out = capsys.readouterr().out
    assert out.count("/items/") == 20
    assert "... and 5 more (use --verbose to list all)" in out
    assert "[DRY RUN] Would delete 25 orphaned files (150 B) of 26 scanned files. Skipped 1 files" in out
    assert all(path.exists() for path in orphans)

    main(["--path", str(item_registry.path), "cleanup", "--min-age", "0.5", "--verbose"])
    out = capsys.readouterr().out
    assert out.count("/items/") == 25
    assert "Deleted 25 orphaned files" in out
    assert "within the last 0.5 hours" in out
    assert not any(path.exists() for path in orphans)


def test_cleanup_purges_dead_entries(capsys):
    from tests.test_cleanup import _count, _insert_items, _write
    item_registry.connect()
    _insert_items([(i, f"image/{i}.jpg", None) for i in range(1, 26)] + [(26, None, 1)])
    orphan = _write(item_registry.path / "items" / "orphan.jpg")

    main(["--path", str(item_registry.path), "cleanup", "--dry-run"])
    out = capsys.readouterr().out
    assert "<image:20>" in out and "<image:21>" not in out
    assert "... and 5 more (use --verbose to list all)" in out
    assert ("[DRY RUN] Would remove 25 dead entries (with 1 aliases) of 25 checked items. Would heal 0 paths. "
            "Skipped 0 items with unreachable locations and 0 items changed within the last 1 hours.") in out
    assert "[DRY RUN] Would delete 1 orphaned files" in out
    assert _count("items") == 26 and orphan.exists()

    main(["--path", str(item_registry.path), "cleanup", "--verbose"])
    out = capsys.readouterr().out
    assert out.count("<image:") == 25
    assert "Removed 25 dead entries (with 1 aliases) of 25 checked items." in out
    assert "Deleted 1 orphaned files" in out
    assert _count("items") == 0 and not orphan.exists()


def test_embed_without_embed_extra(capsys, monkeypatch):
    monkeypatch.setattr(embedding, "is_available", lambda: False)
    with pytest.raises(SystemExit) as exit_info:
        main(["--path", str(item_registry.path), "embed"])
    assert exit_info.value.code == 1
    assert capsys.readouterr().err == embedding.INSTALL_HINT + "\n"


def test_dedup_lists_first_duplicates(capsys):
    from tests.test_dedup import _insert_items
    item_registry.connect()
    _insert_items([(i, f"image/{i}.jpg", "same", None) for i in range(1, 31)])
    main(["--path", str(item_registry.path), "dedup"])
    out = capsys.readouterr().out
    assert out.count(" -> <image:1>") == 20
    assert "... and 9 more (use --verbose to list all)" in out
    assert "Removed 29 duplicates in 1 groups" in out


def test_dedup_shows_progress_bars(capsys):
    from tests.test_dedup import _insert_items
    item_registry.connect()
    _insert_items([(i, f"image/{i}.jpg", None if i <= 3 else "same", None) for i in range(1, 7)])
    main(["--path", str(item_registry.path), "dedup"])
    err = capsys.readouterr().err
    for phase in ("Hashing", "Checking files", "Deduplicating"):
        assert phase in err
    assert "3/3" in err  # tqdm's counter of the hashing phase
