from ezmm import Image


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
