from ezmm import Image
from ezmm.common.registry import item_registry


def test_registry():
    img = Image("in/roses.jpg")  # Load the image to automatically register it in the registry
    assert item_registry.get(img.reference) is img


def test_cache_miss():
    img1 = Image("in/roses.jpg")

    # Reset cache (as if the registry was just restarted with an existing DB)
    item_registry.cache = dict()

    img2 = Image("in/roses.jpg")

    assert img1 is not img2  # Due to cache miss, but...
    assert img1.id == img2.id
