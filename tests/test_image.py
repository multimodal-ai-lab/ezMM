from shutil import copyfile

from PIL import Image as PillowImage

from ezmm import Image, MultimodalSequence


def test_image_equality():
    # Duplicate image file
    copyfile("in/roses.jpg", "in/roses_copy.jpg")

    img1 = Image("in/roses.jpg")
    img2 = Image("in/roses_copy.jpg")
    assert img1 == img2
    assert img1 is not img2


def test_images_in_sequence():
    img1 = Image("in/roses.jpg")
    img2 = Image("in/garden.jpg")
    seq = MultimodalSequence("The images", img1, img2, "show two beautiful roses and a garden.")
    images = seq.images
    assert len(images) == 2
    assert img1 in images
    assert img2 in images
    assert img1 in seq
    assert img2 in seq


def test_binary():
    # Load tulips image with Pillow
    pillow_img = PillowImage.open("in/tulips.jpg")
    img = Image(pillow_image=pillow_img)
    print(img.file_path)


def test_similar():
    img1 = Image("in/roses.jpg")
    img2 = Image("in/roses_smaller.jpg")
    img3 = Image("in/roses_cropped.jpg")
    img4 = Image("in/tulips.jpg")
    assert 0.99 < img1.cos_sim(img2) <= 1
    print(img1.cos_sim(img2))
    assert 0.95 < img1.cos_sim(img3) < 0.99
    print(img1.cos_sim(img3))
    assert 0.95 < img2.cos_sim(img3) < 0.99
    print(img2.cos_sim(img3))
    assert 0.5 < img1.cos_sim(img4) < 0.8
    print(img1.cos_sim(img4))
