import asyncio
from shutil import copyfile

import aiohttp
from PIL import Image as PillowImage

from ezmm import Image, MultimodalSequence, download_image


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


async def download_img(url):
    async with aiohttp.ClientSession() as session:
        return await download_image(url, session)


def test_download_image():
    urls = [
        "https://img.zeit.de/wissen/2025-06/hitzewelle-barcelona-temperaturrekord/wide__1000x562__desktop__scale_2"
    ]
    for url in urls:
        img = asyncio.run(download_img(url))
        print(img)
