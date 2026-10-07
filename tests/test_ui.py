from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from ezmm import Image, Video, Audio, File, MultimodalSequence
from ezmm.common import item_registry
from ezmm.ui.common import get_seq_path
from ezmm.ui.main import app


@pytest.fixture
def client():
    return TestClient(app)


def test_browse(client):
    img = Image("in/roses.jpg", source_url="https://flowers.com/roses.jpg")
    vid = Video("in/mountains.mp4")
    audio = Audio("in/tone.wav")
    file = File("in/sample.pdf")
    response = client.get("/")
    assert response.status_code == 200
    for item in [img, vid, audio, file]:
        assert f"/item/{item.kind}/{item.id}" in response.text
    assert "flowers.com" in response.text


def test_browse_filters(client):
    Image("in/roses.jpg", source_url="https://flowers.com/roses.jpg")
    Video("in/mountains.mp4")
    assert "/item/video/" not in client.get("/?kind=image").text
    assert "/item/image/" in client.get("/?q=flowers").text
    assert "No matches" in client.get("/?q=nothing-matches-this").text


def test_browse_empty(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "Nothing here yet" in response.text


@pytest.mark.parametrize("path", ["in/roses.jpg", "in/mountains.mp4", "in/tone.wav", "in/sample.pdf"])
def test_item_page(client, path):
    cls = {".jpg": Image, ".mp4": Video, ".wav": Audio, ".pdf": File}[Path(path).suffix]
    item = cls(path, source_url="https://example.com/source")
    response = client.get(f"/item/{item.kind}/{item.id}")
    assert response.status_code == 200
    assert item.sha256 in response.text
    assert "https://example.com/source" in response.text


def test_item_file(client):
    img = Image("in/roses.jpg")
    response = client.get(img.file_url)
    assert response.status_code == 200
    assert response.content == Path("in/roses.jpg").read_bytes()


def test_unknown_item(client):
    assert client.get("/item/image/999").status_code == 404
    assert client.get("/item/image/999/file").status_code == 404
    assert client.get("/item/unknown/1").status_code == 404


def test_alias_redirects(client):
    img = Image("in/roses.jpg")
    with item_registry._transaction():
        item_registry.conn.execute("""
            INSERT INTO items(kind, id, path, sha256, size, canonical_id, created_at, updated_at)
            VALUES ('image', 99, NULL, NULL, NULL, ?, '2026-01-01', '2026-01-01');
        """, (img.id,))
    response = client.get("/item/image/99", follow_redirects=False)
    assert response.status_code in (302, 307)
    assert response.headers["location"] == f"/item/image/{img.id}"


def test_sequences(client):
    img = Image("in/roses.jpg")
    seq = MultimodalSequence("The image", img, "shows roses.")
    seq_path = get_seq_path()
    seq_path.mkdir(parents=True, exist_ok=True)
    (seq_path / "12345678.md").write_text(str(seq), encoding="utf-8")

    response = client.get("/sequences")
    assert response.status_code == 200
    assert "/sequence/12345678" in response.text

    response = client.get("/sequence/12345678")
    assert response.status_code == 200
    assert img.file_url in response.text


def test_missing_files_hidden_by_default(client, tmp_path):
    from shutil import copyfile
    gone = tmp_path / "gone.jpg"
    copyfile("in/garden.jpg", gone)
    missing_img = Image(gone)
    img = Image("in/roses.jpg")
    gone.unlink()

    response = client.get("/")
    assert f'href="/item/image/{img.id}"' in response.text
    assert f'href="/item/image/{missing_img.id}"' not in response.text
    assert "1 with missing file hidden" in response.text

    response = client.get("/?missing=1")
    assert f'href="/item/image/{missing_img.id}"' in response.text
    assert "FILE MISSING" in response.text


def test_only_missing_files(client, tmp_path):
    from shutil import copyfile
    gone = tmp_path / "gone.jpg"
    copyfile("in/garden.jpg", gone)
    Image(gone)
    gone.unlink()
    response = client.get("/")
    assert "Only items with missing files" in response.text
    assert 'href="/?missing=1"' in response.text
