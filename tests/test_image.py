from ezmm import Image, MultimodalSequence


def test_image_equality():
    img1 = Image("in/roses.jpg")
    img2 = Image("in/roses_copy.jpg")
    assert img1 == img2


def test_images_in_sequence():
    img1 = Image("in/roses.jpg")
    img2 = Image("in/garden.jpg")
    seq = MultimodalSequence("The images", img1, img2, "show two beautiful roses and a garden.")
    images = seq.images
    print(images)
