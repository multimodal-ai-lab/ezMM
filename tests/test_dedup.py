from pathlib import Path
from shutil import copyfile

from ezmm import File, Image, Item, Video
from ezmm.common import item_registry


def test_copy_collapses_to_same_item(tmp_path):
    copy_path = tmp_path / "roses_copy.jpg"
    copyfile("in/roses.jpg", copy_path)

    img1 = Image("in/roses.jpg")
    img2 = Image(copy_path)
    assert img1 == img2
    assert img1.id == img2.id
    assert img1.reference == img2.reference
    assert img2.file_path == img1.file_path  # Adopts the existing file
    assert item_registry.count_items("image") == 1


def test_source_urls_accumulate(tmp_path):
    copy_path = tmp_path / "roses_copy.jpg"
    copyfile("in/roses.jpg", copy_path)

    img1 = Image("in/roses.jpg", source_url="https://a.com/roses.jpg")
    Image(copy_path, source_url="https://b.com/roses.jpg")
    Image("in/roses.jpg", source_url="https://a.com/roses.jpg")  # Known URL, must not be duplicated
    assert img1.source_urls == ["https://a.com/roses.jpg", "https://b.com/roses.jpg"]


def test_binary_duplicate_removes_temp_file():
    data = Path("in/mountains.mp4").read_bytes()
    vid1 = Video("in/mountains.mp4")
    vid2 = Video(binary_data=data, source_url="https://example.com/video.mp4")
    assert vid1 == vid2
    assert vid2.id == vid1.id
    assert vid2.file_path == vid1.file_path
    temp_dir = item_registry.path / "items"
    assert not any(temp_dir.iterdir())  # The temporary file of the duplicate got deleted
    assert "https://example.com/video.mp4" in vid1.source_urls


def test_binary_duplicates_among_each_other():
    data = Path("in/mountains.mp4").read_bytes()
    vid1 = Video(binary_data=data)
    vid2 = Video(binary_data=data)
    assert vid1.id == vid2.id
    assert len(list((item_registry.path / "items").iterdir())) == 1


def test_different_kinds_are_not_merged():
    img = Image("in/roses.jpg")
    file = File("in/roses.jpg")
    assert img.kind != file.kind
    assert item_registry.count_items() == 2


def _make_duplicates_in_registry() -> tuple[Image, Image, Path]:
    """Simulates a registry that contains duplicates (e.g., from before deduplication existed)."""
    img1 = Image("in/roses.jpg", source_url="https://a.com/1.jpg")
    img1.relocate()
    dup_path = item_registry.path / "image" / "dup.jpg"
    copyfile("in/roses.jpg", dup_path)
    with item_registry._transaction():
        item_registry.conn.execute("""
            INSERT INTO items(kind, id, path, sha256, size, canonical_id, created_at, updated_at)
            VALUES ('image', 2, 'image/dup.jpg', ?, ?, NULL, '2026-01-01', '2026-01-01');
        """, (img1.sha256, img1.size))
        row_id = item_registry._get_row_id("image", 2)
        item_registry._link_source(row_id, "https://b.com/2.jpg")
    item_registry.clear_cache()
    return img1, Image.from_id(2), dup_path


def test_deduplicate_dry_run():
    _, _, dup_path = _make_duplicates_in_registry()
    report = item_registry.deduplicate(dry_run=True)
    assert report["removed"] == [("image", 2, 1)]
    assert dup_path.exists()
    assert item_registry.get_row("image", 2)["canonical_id"] is None


def test_deduplicate():
    _, _, dup_path = _make_duplicates_in_registry()
    report = item_registry.deduplicate()
    assert report["groups"] == 1
    assert report["removed"] == [("image", 2, 1)]
    assert not dup_path.exists()

    # Old references remain valid and resolve to the kept item
    kept = Item.from_reference("<image:1>")
    assert Item.from_reference("<image:2>") is kept
    assert kept.source_urls == ["https://a.com/1.jpg", "https://b.com/2.jpg"]
    assert item_registry.get_aliases("image", 1) == [2]
    assert item_registry.count_items("image") == 1

    # Running it again changes nothing
    assert item_registry.deduplicate()["removed"] == []


def test_deduplicate_never_deletes_files_outside_registry(tmp_path):
    outside = tmp_path / "outside.jpg"
    copyfile("in/roses.jpg", outside)
    img = Image(outside)
    with item_registry._transaction():
        item_registry.conn.execute("""
            INSERT INTO items(kind, id, path, sha256, size, canonical_id, created_at, updated_at)
            VALUES ('image', 2, ?, ?, ?, NULL, '2026-01-01', '2026-01-01');
        """, (Path("in/roses.jpg").absolute().as_posix(), img.sha256, img.size))
    item_registry.deduplicate()
    assert outside.exists()
    assert Path("in/roses.jpg").exists()
