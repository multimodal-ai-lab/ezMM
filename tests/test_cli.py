import pytest

from ezmm import Image, embedding
from ezmm.__main__ import main
from ezmm.common import item_registry


def test_check(capsys):
    Image("in/roses.jpg")
    main(["--path", str(item_registry.path), "check"])
    assert "Checked 1 items: 0 files missing" in capsys.readouterr().out


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


def test_progress_bar_log_output(capsys):
    from ezmm.__main__ import ProgressBar
    with ProgressBar() as progress:
        for i in range(1, 101):
            progress("Hashing", i, 100)
        progress("Deduplicating", 1, 1)
    lines = capsys.readouterr().err.strip().split("\n")  # Not a terminal: a line at the start and per 10%
    expected = ["Hashing: 1%"] + [f"Hashing: {p}%" for p in range(10, 101, 10)] + ["Deduplicating: 100%"]
    assert [line.split(" (")[0] for line in lines] == expected
