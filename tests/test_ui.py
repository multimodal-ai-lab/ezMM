import re
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from ezmm import Audio, File, Image, MultimodalSequence, Video, embedding
from ezmm.common import item_registry
from ezmm.ui import main as ui_main
from ezmm.ui.common import get_seq_path
from ezmm.ui.main import app

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
    _ = roses.embedding, snow.embedding
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
    _ = roses.embedding, tulips.embedding
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
    _ = roses.embedding, tone.embedding
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


def _card_refs(html: str) -> list[str]:
    """Returns the item links of all cards in the order they appear."""
    return re.findall(r'<a class="card kind-\w+" href="/item/(\w+/\d+)"', html)


def test_browse_partial_pages(client, monkeypatch):
    monkeypatch.setattr(ui_main, "PAGE_SIZE", 2)
    for path in ["in/roses.jpg", "in/garden.jpg", "in/tulips.jpg"]:
        Image(path)
    Video("in/snow.mp4")
    image_refs = ["image/3", "image/2", "image/1"]  # Newest first

    response = client.get("/")
    page1 = _card_refs(response.text)
    assert len(page1) == 2
    assert 'data-next="/?page=2"' in response.text
    assert "Page 1 of 2" in response.text  # Pagination links stay as fallback without JS
    assert "4 items" in response.text  # The total, not the number of shown items

    response = client.get("/?page=2&partial=1")
    assert response.status_code == 200
    assert "<html" not in response.text and "toolbar" not in response.text
    page2 = _card_refs(response.text)
    assert sorted(page1 + page2) == ["image/1", "image/2", "image/3", "video/1"]  # No overlaps, no gaps
    assert 'data-next=""' in response.text  # No more pages
    assert _card_refs(client.get("/?page=3&partial=1").text) == []  # Beyond the end: empty, no repetition

    # Filters are respected and carried over to the next page
    response = client.get("/?kind=image&partial=1")
    assert _card_refs(response.text) == image_refs[:2]
    assert 'data-next="/?kind=image&amp;page=2"' in response.text
    assert _card_refs(client.get("/?kind=image&page=2&partial=1").text) == image_refs[2:]
    response = client.get("/?q=snow&partial=1")
    assert _card_refs(response.text) == ["video/1"] and 'data-next=""' in response.text


def test_browse_partial_missing(client, monkeypatch, tmp_path):
    from shutil import copyfile
    monkeypatch.setattr(ui_main, "PAGE_SIZE", 1)
    gone = tmp_path / "gone.jpg"
    copyfile("in/garden.jpg", gone)
    img = Image("in/roses.jpg")
    missing_img = Image(gone)
    gone.unlink()
    item_registry.set_missing("image", missing_img.id, True)
    response = client.get("/?partial=1")
    assert _card_refs(response.text) == [f"image/{img.id}"] and 'data-next=""' in response.text
    response = client.get("/?missing=1&partial=1")
    assert _card_refs(response.text) == [f"image/{missing_img.id}"]
    assert 'data-next="/?missing=1&amp;page=2"' in response.text
    assert _card_refs(client.get("/?missing=1&page=2&partial=1").text) == [f"image/{img.id}"]


def _random_embeddings(items, seed: int = 0):
    rng = np.random.default_rng(seed)
    vectors = embedding.truncate(rng.standard_normal((len(items), embedding.MODEL_DIM)), embedding.MODEL_DIM)
    item_registry.set_embeddings(embedding.MODEL_NAME, [(item.kind, item.id, v) for item, v in zip(items, vectors)])
    return vectors


def test_registry_search_offset(tmp_path):
    files = []
    for i in range(20):
        path = tmp_path / f"file{i}.txt"
        path.write_text(f"File number {i}")
        files.append(File(path))
    _random_embeddings(files)
    query = embedding.truncate(np.random.default_rng(1).standard_normal(embedding.MODEL_DIM), 256)

    def refs(**kwargs):
        return [row["reference"] for row in item_registry.search(query, embedding.MODEL_NAME, device="cpu", **kwargs)]

    full = refs(limit=100)
    assert len(full) == 20
    pages = [refs(offset=offset, limit=6) for offset in range(0, 24, 6)]
    assert [len(page) for page in pages] == [6, 6, 6, 2]
    assert [ref for page in pages for ref in page] == full  # No overlaps, no gaps
    assert refs(offset=20, limit=6) == []
    # Excluded and missing items are skipped consistently across pages
    excluded = (files[0].kind, files[0].id)
    item_registry.set_missing(files[1].kind, files[1].id, True)
    filtered = refs(limit=100, exclude=excluded)
    assert len(filtered) == 18
    pages = [refs(offset=offset, limit=5, exclude=excluded) for offset in range(0, 20, 5)]
    assert [ref for page in pages for ref in page] == filtered
    assert refs(limit=3, kind="image") == []


@pytest.fixture
def fake_embed(monkeypatch):
    """Lets the search UI run without the embedding model: queries embed to a fixed vector."""
    vector = embedding.truncate(np.random.default_rng(1).standard_normal(embedding.MODEL_DIM), 256)
    calls = []

    def embed_query(query, dim=None):
        calls.append(query)
        return vector

    monkeypatch.setattr(embedding, "is_available", lambda: True)
    monkeypatch.setattr(embedding, "get_index_device", lambda: "cpu")
    monkeypatch.setattr(embedding, "embed_query", embed_query)
    monkeypatch.setattr(embedding, "embed_file", lambda path, kind=None, dim=None: vector)
    monkeypatch.setattr(ui_main, "PAGE_SIZE", 2)
    ui_main.query_cache.clear()
    yield calls
    ui_main.query_cache.clear()


def _expected_refs(query=None, exclude=None):
    """Returns the item links of all search results (default: for the fake query vector), best first."""
    if query is None:
        query = embedding.truncate(np.random.default_rng(1).standard_normal(embedding.MODEL_DIM), 256)
    rows = item_registry.search(query, embedding.MODEL_NAME, limit=100, exclude=exclude, device="cpu")
    return [f"{row['kind']}/{row['id']}" for row in rows]


def test_search_partial_pages(client, fake_embed):
    items = [Image(p) for p in ["in/roses.jpg", "in/garden.jpg", "in/tulips.jpg"]] + [Video("in/snow.mp4")]
    _random_embeddings(items)
    expected = _expected_refs()

    response = client.get("/search?q=fake+query")
    assert response.status_code == 200
    assert _card_refs(response.text) == expected[:2]
    assert 'data-next="/search?q=fake+query&amp;page=2"' in response.text
    assert "More results" in response.text  # Fallback link without JS

    response = client.get("/search?q=fake+query&page=2&partial=1")
    assert "<html" not in response.text
    assert _card_refs(response.text) == expected[2:]
    assert 'data-next=""' in response.text
    assert fake_embed == ["fake query"]  # The query vector got cached instead of re-embedded

    image_refs = [ref for ref in expected if ref.startswith("image/")]
    response = client.get("/search?q=fake+query&kind=image&partial=1")
    assert _card_refs(response.text) == image_refs[:2]
    assert 'data-next="/search?q=fake+query&amp;kind=image&amp;page=2"' in response.text

    like = items[0]
    expected_like = _expected_refs(like.embedding, exclude=(like.kind, like.id))
    assert len(expected_like) == 3
    pages = [_card_refs(client.get(f"/search?like=image:{like.id}&page={page}&partial=1").text) for page in (1, 2)]
    assert [ref for page in pages for ref in page] == expected_like


def test_search_by_file_pages(client, fake_embed):
    items = [Image(p) for p in ["in/roses.jpg", "in/garden.jpg", "in/tulips.jpg"]]
    _random_embeddings(items)
    expected = _expected_refs()
    with open("in/roses_smaller.jpg", "rb") as f:
        response = client.post("/search", files={"file": ("roses_smaller.jpg", f)})
    assert response.status_code == 200
    assert "roses_smaller.jpg" in response.text
    assert _card_refs(response.text) == expected[:2]
    next_url = re.search(r'data-next="(/search\?vec=[\w-]+&amp;page=2)"', response.text).group(1).replace("&amp;", "&")
    assert item_registry.count_items() == 3  # The query file does not get added to the registry

    response = client.get(next_url + "&partial=1")
    assert _card_refs(response.text) == expected[2:] and 'data-next=""' in response.text
    response = client.get(next_url)  # Full page without JS
    assert "roses_smaller.jpg" in response.text and _card_refs(response.text) == expected[2:]

    ui_main.query_cache.clear()  # E.g., after a server restart
    response = client.get(next_url)
    assert response.status_code == 200
    assert "This search has expired" in response.text
    assert _card_refs(client.get(next_url + "&partial=1").text) == []


def test_query_cache_is_bounded():
    cache = ui_main._QueryCache(max_size=2)
    cache.put("a", 1)
    cache.put("b", 2)
    assert cache.get("a") == 1  # Marks "a" as recently used
    cache.put("c", 3)
    assert cache.get("b") is None and cache.get("a") == 1 and cache.get("c") == 3


def test_duration_format():
    from ezmm.ui.main import _duration
    assert _duration(734) == "12:14"
    assert _duration(3725) == "1:02:05"
    assert _duration(-1) == "0:00"
