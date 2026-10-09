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
