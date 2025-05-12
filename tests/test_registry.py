from ezmm import Image
from ezmm.common.registry import item_registry


def test_registry():
    img = Image("in/roses.jpg")  # Load the image to automatically register it in the registry
    assert item_registry.get(img.reference) is img
