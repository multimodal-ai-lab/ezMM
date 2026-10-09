import sqlite3

import numpy as np
import pytest

from ezmm import Image, Video, Audio, File, MultimodalSequence
from ezmm import embedding
from ezmm.common import item_registry
from ezmm.common.registry import SCHEMA, SCHEMA_VERSION
from ezmm.embedding import MODEL_NAME, MODEL_DIM, DEFAULT_DIM, embed_query, embed_text, embed_file, truncate

requires_embed = pytest.mark.skipif(not embedding.is_available(), reason="Requires ezmm[embed]")


@requires_embed
@pytest.mark.parametrize("cls, path", [(Image, "in/roses.jpg"), (Video, "in/mountains.mp4"),
                                       (Audio, "in/tone.wav"), (File, "in/sample.pdf"), (File, "in/table.csv")])
def test_item_embedding(cls, path):
    item = cls(path)
    embedding = item.embedding
    assert embedding.shape == (DEFAULT_DIM,)
    assert embedding.dtype == np.float32
    assert np.isclose(np.linalg.norm(embedding), 1, atol=1e-3)


@requires_embed
def test_embedding_persisted():
    img = Image("in/roses.jpg")
    embedding = img.embedding
    stored = item_registry.get_embedding(img.kind, img.id, MODEL_NAME)
    assert stored.shape == (MODEL_DIM,)  # The registry stores the full embedding
    assert np.allclose(truncate(stored, DEFAULT_DIM), embedding, atol=1e-3)

    # A fresh instance loads the embedding from the registry instead of recomputing it
    item_registry.clear_cache()
    img2 = Image(reference=img.reference)
    assert img2 is not img
    assert img2._embedding is None
    assert np.allclose(img2.embedding, embedding, atol=1e-3)  # Stored as float16


@requires_embed
def test_similar_images():
    roses = Image("in/roses.jpg")
    roses_smaller = Image("in/roses_smaller.jpg")
    roses_cropped = Image("in/roses_cropped.JPG")
    tulips = Image("in/tulips.jpg")
    # Near-duplicates are much more similar than other images of flowers
    assert roses.cos_sim(roses_smaller) > 0.95
    assert roses.cos_sim(roses_cropped) > 0.95
    assert roses.cos_sim(tulips) < 0.85


@requires_embed
def test_cross_modal():
    """Text queries should match the items they describe."""
    roses = Image("in/roses.jpg")
    garden = Image("in/garden.jpg")
    snow = Video("in/snow.mp4")
    query = embed_query("red roses")
    assert roses.cos_sim(query) > garden.cos_sim(query)
    assert roses.cos_sim(query) > snow.cos_sim(query)
    query = embed_query("snowfall in winter")
    assert snow.cos_sim(query) > roses.cos_sim(query)


@requires_embed
def test_embed_text():
    a = embed_text("A red rose in bloom.")
    b = embed_text("A blossoming red rose.")
    c = embed_text("Quarterly tax report for 2025.")
    assert a.shape == (DEFAULT_DIM,)
    assert np.dot(a, b) > np.dot(a, c)


@requires_embed
def test_sequence_embedding():
    roses = Image("in/roses.jpg")
    snow = Video("in/snow.mp4")
    seq = MultimodalSequence("The image", roses, "shows roses while the video", snow, "shows snowfall.")
    embedding = seq.embedding
    assert embedding.shape == (DEFAULT_DIM,)
    assert np.isclose(np.linalg.norm(embedding), 1, atol=1e-3)

    # The sequence embedding is the (normalized) average of its text and item embeddings
    expected = np.mean([roses.embedding, snow.embedding, embed_text(seq.text)], axis=0)
    assert np.allclose(embedding, expected / np.linalg.norm(expected), atol=1e-4)
    assert seq.cos_sim(roses) > 0.5


@requires_embed
def test_sequence_embedding_edge_cases():
    roses = Image("in/roses.jpg")
    assert np.allclose(MultimodalSequence(roses).embedding, roses.embedding, atol=1e-5)
    assert MultimodalSequence(roses, roses).cos_sim(MultimodalSequence(roses)) > 0.999
    assert MultimodalSequence("Just text.").embedding.shape == (DEFAULT_DIM,)
    with pytest.raises(ValueError):
        MultimodalSequence().embedding


@requires_embed
def test_registry_search():
    roses = Image("in/roses.jpg")
    tulips = Image("in/tulips.jpg")
    snow = Video("in/snow.mp4")
    tone = Audio("in/tone.wav")
    for item in (roses, tulips, snow, tone):
        item.embedding
    assert item_registry.list_unembedded(MODEL_NAME) == []
    assert item_registry.count_embedded(MODEL_NAME) == 4

    results = item_registry.search(embed_query("snow"), MODEL_NAME)
    assert results[0]["reference"] == snow.reference
    assert len(results) == 4
    assert results[0]["score"] >= results[-1]["score"]

    results = item_registry.search(roses.embedding, MODEL_NAME, kind="image", exclude=("image", roses.id))
    assert [r["reference"] for r in results] == [tulips.reference]


@requires_embed
def test_list_unembedded():
    roses = Image("in/roses.jpg")
    snow = Video("in/snow.mp4")
    assert set(item_registry.list_unembedded(MODEL_NAME)) == {("image", roses.id), ("video", snow.id)}
    assert item_registry.list_unembedded(MODEL_NAME, kind="video") == [("video", snow.id)]
    assert item_registry.count_unembedded(MODEL_NAME) == 2
    roses.embedding
    assert item_registry.list_unembedded(MODEL_NAME) == [("video", snow.id)]
    assert item_registry.count_unembedded(MODEL_NAME) == 1


@requires_embed
def test_embed_file_without_registry(tmp_path):
    """Files can be embedded directly, e.g., to search with an uploaded file."""
    roses = Image("in/roses.jpg")
    embedding = embed_file("in/roses.jpg", "image")
    assert np.allclose(embedding, roses.embedding, atol=1e-4)


@requires_embed
def test_binary_file_embedding(tmp_path):
    path = tmp_path / "random.bin"
    path.write_bytes(np.random.default_rng(0).bytes(1024))
    assert File(path).embedding.shape == (DEFAULT_DIM,)


@requires_embed
def test_embeddings_table_added_to_existing_registry():
    """Registries of ezMM v0.6 get the embeddings table without a schema version
    change, so they stay readable by older ezMM versions."""
    item_registry.path.mkdir(parents=True)
    conn = sqlite3.connect(item_registry.path / "item_registry.db")
    conn.executescript(SCHEMA)
    conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION};")
    conn.close()

    roses = Image("in/roses.jpg")
    assert item_registry.get_embedding(roses.kind, roses.id, MODEL_NAME) is None
    roses.embedding
    assert item_registry.get_embedding(roses.kind, roses.id, MODEL_NAME) is not None
    assert item_registry._execute("PRAGMA user_version;")[0][0] == SCHEMA_VERSION


def test_without_embed_extra(monkeypatch):
    """Without the optional dependencies, embedding fails with a helpful error, but
    embeddings stored in the registry remain accessible."""
    roses = Image("in/roses.jpg")
    item_registry.set_embedding(roses.kind, roses.id, MODEL_NAME, np.ones(MODEL_DIM, dtype=np.float32))
    tulips = Image("in/tulips.jpg")
    monkeypatch.setattr(embedding, "is_available", lambda: False)
    monkeypatch.setattr(embedding, "_model", None)
    with pytest.raises(ImportError, match=r"ezmm\[embed\]"):
        tulips.embedding
    with pytest.raises(ImportError, match=r"ezmm\[embed\]"):
        embed_query("roses")
    assert np.allclose(roses.embedding, 1 / np.sqrt(DEFAULT_DIM))


def test_device_override(monkeypatch):
    pytest.importorskip("torch")
    monkeypatch.setenv("EZMM_DEVICE", "cpu")
    assert embedding.get_device() == "cpu"


@requires_embed
def test_embed_files_bulk():
    """Bulk embedding (threaded decoding, batched encoding) matches one-by-one embedding,
    keeps the order, and reports errors per file."""
    files = [("in/roses.jpg", "image"), ("in/tone.wav", "audio"), ("in/does_not_exist.jpg", "image"),
             ("in/garden.jpg", "image"), ("in/table.csv", "file"), ("in/snow.mp4", "video")]
    results = list(embedding.embed_files(files, chunk_size=4))
    assert len(results) == len(files)
    assert isinstance(results[2], FileNotFoundError)
    for (path, kind), result in zip(files, results):
        if path != "in/does_not_exist.jpg":
            assert np.dot(result, embed_file(path, kind)) > 0.999


@requires_embed
def test_embed_registry(tmp_path):
    from shutil import copyfile
    roses = Image("in/roses.jpg")
    tone = Audio("in/tone.wav")
    pdf = File("in/sample.pdf")
    gone = tmp_path / "gone.jpg"
    copyfile("in/garden.jpg", gone)
    missing = Image(gone)
    gone.unlink()

    progress = []
    result = embedding.embed_registry(chunk_size=2, on_progress=lambda *args: progress.append(args))
    assert result == dict(embedded=3, failed=1)
    assert progress[-1] == (4, 4, 1)
    assert item_registry.list_unembedded(MODEL_NAME) == []  # The missing file got flagged
    assert item_registry.get_row("image", missing.id)["missing"]
    for item in (roses, tone, pdf):
        assert item_registry.get_embedding(item.kind, item.id, MODEL_NAME) is not None


@requires_embed
def test_embedding_dim(monkeypatch):
    monkeypatch.setattr(embedding, "_embedding_dim", DEFAULT_DIM)  # Restored after the test
    roses = Image("in/roses.jpg")
    assert roses.embedding.shape == (256,)
    embedding.set_embedding_dim(512)
    assert roses.embedding.shape == (512,)  # No re-embedding needed
    assert embed_query("roses").shape == (512,)
    assert MultimodalSequence("Roses", roses).embedding.shape == (512,)
    assert embed_query("roses", dim=128).shape == (128,)
    with pytest.raises(ValueError):
        embedding.set_embedding_dim(300)

    # Embeddings of different dimensions are compared in the smaller dimension
    small, large = embed_query("roses", dim=128), embed_query("roses", dim=768)
    assert np.isclose(embedding.cos_sim(small, large), 1, atol=1e-5)


def test_float16_storage():
    roses = Image("in/roses.jpg")
    vector = truncate(np.random.default_rng(0).standard_normal(MODEL_DIM), MODEL_DIM)
    item_registry.set_embedding(roses.kind, roses.id, MODEL_NAME, vector)
    blob, dtype = item_registry._execute("SELECT vector, dtype FROM embeddings;")[0]
    assert dtype == "float16" and len(blob) == MODEL_DIM * 2
    assert np.allclose(item_registry.get_embedding(roses.kind, roses.id, MODEL_NAME), vector, atol=1e-3)


def test_float32_rows_remain_readable():
    """Embeddings stored as float32 (during the development of v0.7.0) remain readable."""
    roses = Image("in/roses.jpg")
    vector = truncate(np.arange(MODEL_DIM), MODEL_DIM)
    with item_registry._transaction():
        item_registry.conn.execute("INSERT INTO embeddings(item_row_id, model, vector, created_at) "
                                   "VALUES (?, ?, ?, '2026-01-01');",
                                   (item_registry._get_row_id("image", roses.id), MODEL_NAME, vector.tobytes()))
    assert np.allclose(item_registry.get_embedding(roses.kind, roses.id, MODEL_NAME), vector)
    assert item_registry.search(vector, MODEL_NAME)[0]["reference"] == roses.reference


def _random_embeddings(items):
    rng = np.random.default_rng(0)
    vectors = truncate(rng.standard_normal((len(items), MODEL_DIM)), MODEL_DIM)
    item_registry.set_embeddings(MODEL_NAME, [(item.kind, item.id, v) for item, v in zip(items, vectors)])
    return vectors


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(
    not embedding.is_available() or not __import__("torch").cuda.is_available(), reason="Requires a GPU"))])
def test_search_index_devices(device):
    items = [Image(p) for p in ["in/roses.jpg", "in/garden.jpg", "in/tulips.jpg"]]
    vectors = _random_embeddings(items)
    for dim in (128, 256, 768):
        results = item_registry.search(vectors[1][:dim], MODEL_NAME, device=device)
        assert results[0]["reference"] == items[1].reference
        assert np.isclose(results[0]["score"], 1, atol=1e-2)
        index = item_registry.get_embedding_index(MODEL_NAME, dim, device)
        assert index.device == device and index.dim == dim and len(index) == 3
        assert index.dtype == (np.float32 if device == "cpu" else np.float16)

    # New embeddings get added to the loaded index without rebuilding it
    index = item_registry.get_embedding_index(MODEL_NAME, 256, device)
    video = Video("in/snow.mp4")
    new_vector = truncate(np.ones(MODEL_DIM), MODEL_DIM)
    item_registry.set_embedding(video.kind, video.id, MODEL_NAME, new_vector)
    assert item_registry.search(new_vector[:256], MODEL_NAME, device=device)[0]["reference"] == video.reference
    assert item_registry.get_embedding_index(MODEL_NAME, 256, device) is index
    assert len(index) == 4


@requires_embed
def test_search_with_default_dim():
    roses = Image("in/roses.jpg")
    snow = Video("in/snow.mp4")
    roses.embedding, snow.embedding
    query = embed_query("snow")
    assert query.shape == (DEFAULT_DIM,)
    assert item_registry.search(query, MODEL_NAME)[0]["reference"] == snow.reference


def test_index_device_detection(monkeypatch, caplog):
    """Unless set by the user, the index is kept in GPU memory if a GPU is available, else in RAM."""
    monkeypatch.setattr(embedding, "_index_device", None)
    gpu = embedding.is_available() and __import__("torch").cuda.is_available()
    with caplog.at_level("INFO", logger="ezMM"):
        assert embedding.get_index_device() == ("cuda" if gpu else "cpu")
        assert embedding.get_index_device() == ("cuda" if gpu else "cpu")
    assert len([r for r in caplog.records if "search index in GPU memory" in r.message]) == (1 if gpu else 0)

    monkeypatch.setattr(embedding, "_index_device", None)
    monkeypatch.setattr(embedding, "is_available", lambda: False)
    assert embedding.get_index_device() == "cpu"

    embedding.set_index_device("cpu")  # The user's choice wins
    assert embedding.get_index_device() == "cpu"
    with pytest.raises(ValueError):
        embedding.set_index_device("vram")
