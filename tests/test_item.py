from pathlib import Path
from shutil import copyfile

import pytest

from ezmm import Image, Item, download_item


def test_item():
    img = Image("in/roses.jpg")
    print(img)


def test_equality():
    img1 = Image("in/roses.jpg")
    img2 = Image("in/roses.jpg")
    assert img1 == img2


def test_identity():
    img1 = Image("in/roses.jpg")
    img2 = Image("in/roses.jpg")
    assert img1 is img2


def test_inequality():
    img1 = Image("in/roses.jpg")
    img2 = Image("in/garden.jpg")
    assert img1 != img2


def test_reference():
    img1 = Image("in/roses.jpg")
    img2 = Image(reference=img1.reference)
    assert img1 == img2
    assert img1 is img2


def test_from_reference():
    img1 = Image("in/roses.jpg")
    img2 = Item.from_reference(img1.reference)
    assert img1 == img2
    assert img1 is img2


def test_source_default():
    img_path = Path("in/roses.jpg")
    img = Image(img_path)
    assert img_path.absolute().as_uri() == img.source_url


def test_source_custom():
    img_path = Path("in/roses.jpg")
    img = Image(img_path, source_url="https://example.com/image.jpg")
    assert "https://example.com/image.jpg" == img.source_url


def test_relocate_copy():
    img1 = Image("in/roses.jpg")
    img1.relocate()
    new_filepath = img1.file_path.as_posix()
    assert "in/roses.jpg" not in new_filepath
    assert new_filepath.endswith(f"image/{img1.id}.jpg")

    # Loading the original image file should result in an equal but
    # different Image object
    img2 = Image("in/roses.jpg")
    assert img1 == img2
    assert img1 is not img2


def test_relocate_move():
    # Create temp image file from existing
    source_path = Path("in/roses_copy.jpg")
    copyfile("in/roses.jpg", source_path)

    img = Image(source_path)
    img.relocate(move_not_copy=True)
    new_filepath = img.file_path.as_posix()
    assert "in/roses.jpg" not in new_filepath
    assert new_filepath.endswith(f"image/{img.id}.jpg")
    assert not source_path.exists()


@pytest.mark.asyncio
@pytest.mark.parametrize("url", [
        "https://img.zeit.de/wissen/2025-06/hitzewelle-barcelona-temperaturrekord/wide__1000x562__desktop__scale_2",
        "https://upload.wikimedia.org/wikipedia/commons/transcoded/a/a7/How_to_make_video.webm/How_to_make_video.webm.1080p.vp9.webm"
])
async def test_download_items(url):
    item = await download_item(url)
    assert isinstance(item, Item)
    print(item)
