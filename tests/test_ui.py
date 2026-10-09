from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from ezmm import Image, Video, Audio, File, MultimodalSequence
from ezmm.common import item_registry
from ezmm.ui.common import get_seq_path
from ezmm.ui.main import app
from ezmm import embedding

requires_embed = pytest.mark.skipif(not embedding.is_available(), reason="Requires ezmm[embed]")


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


@requires_embed
def test_search_page_empty(client):
    Image("in/roses.jpg")
    response = client.get("/search")
    assert response.status_code == 200
    assert "1 item is not indexed yet" in response.text


@requires_embed
def test_search_by_text(client):
    roses = Image("in/roses.jpg")
    snow = Video("in/snow.mp4")
    roses.embedding, snow.embedding
    response = client.get("/search?q=red+roses")
    assert response.status_code == 200
    assert response.text.index(f'href="/item/image/{roses.id}"') < response.text.index(f'href="/item/video/{snow.id}"')
    assert "chip score" in response.text

    response = client.get("/search?q=red+roses&kind=video")
    assert f'href="/item/image/{roses.id}"' not in response.text
    assert f'href="/item/video/{snow.id}"' in response.text


@requires_embed
def test_search_like_item(client):
    roses = Image("in/roses.jpg")
    tulips = Image("in/tulips.jpg")
    roses.embedding, tulips.embedding
    assert f"/search?like=image%3A{roses.id}" in client.get(f"/item/image/{roses.id}").text
    response = client.get(f"/search?like=image:{roses.id}")
    assert response.status_code == 200
    assert f'href="/item/image/{tulips.id}"' in response.text
    assert f'href="/item/image/{roses.id}"' not in response.text  # The query item itself is excluded
    assert client.get("/search?like=image:999").status_code == 404


@requires_embed
@pytest.mark.parametrize("path", ["in/roses_smaller.jpg", "in/tone.wav", "in/table.csv"])
def test_search_by_file(client, path):
    roses = Image("in/roses.jpg")
    tone = Audio("in/tone.wav")
    roses.embedding, tone.embedding
    with open(path, "rb") as f:
        response = client.post("/search", files={"file": (Path(path).name, f)})
    assert response.status_code == 200
    assert Path(path).name in response.text
    assert f'href="/item/image/{roses.id}"' in response.text
    # The query file does not get added to the registry
    assert item_registry.count_items() == 2


@requires_embed
def test_search_index(client):
    roses = Image("in/roses.jpg")
    response = client.post("/search/index", data={"q": "roses"}, follow_redirects=False)
    assert response.status_code == 303
    assert response.headers["location"] == "/search?q=roses"
    from ezmm.ui.main import indexer
    indexer.thread.join(timeout=300)
    assert indexer.done == indexer.total == 1
    assert f'href="/item/image/{roses.id}"' in client.get("/search?q=roses").text


def test_search_without_embed_extra(client, monkeypatch):
    img = Image("in/roses.jpg")
    monkeypatch.setattr(embedding, "is_available", lambda: False)
    response = client.get("/search?q=roses")
    assert response.status_code == 200
    assert "pip install ezmm[embed]" in response.text
    assert 'id="dropzone"' not in response.text
    assert "Find similar items" not in client.get(f"/item/image/{img.id}").text


def test_audio_only_video_page(client, audio_only_video):
    vid = Video(audio_only_video)
    response = client.get(f"/item/video/{vid.id}")
    assert response.status_code == 200
    assert "Audio only (no video stream)" in response.text
    assert "0 × 0" not in response.text and "Frame rate" not in response.text
    assert "<audio" in response.text and "<video" not in response.text


def test_duration_format():
    from ezmm.ui.main import _duration
    assert _duration(734) == "12:14"
    assert _duration(3725) == "1:02:05"
    assert _duration(-1) == "0:00"
