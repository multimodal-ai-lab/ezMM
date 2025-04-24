from ezmm import Image
from ezmm.common.registry import item_registry


def test_registry():
    img = Image("in/roses.jpg")  # Loading the image automatically registers it in the registry
    assert item_registry.get(img.reference) == img
    assert item_registry.contains(img.kind, img.file_path)
