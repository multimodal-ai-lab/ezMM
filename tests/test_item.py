from ezmm import Image, Item


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


def test_relocate():
    img1 = Image("in/roses.jpg")
    img1.relocate()
    new_filepath = img1.file_path.as_posix()
    assert "in/roses.jpg" not in new_filepath
    assert new_filepath.endswith(f"image/{img1.id}.jpg")

    # Original image file should still exist, but loading it
    # should return the one saved in temp
    img2 = Image("in/roses.jpg")
    assert img1 is img2
