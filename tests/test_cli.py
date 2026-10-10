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
