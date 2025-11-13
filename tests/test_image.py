from shutil import copyfile

import aiohttp
import pytest
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


@pytest.mark.asyncio
@pytest.mark.parametrize("url", [
    "https://media.cnn.com/api/v1/images/stellar/prod/02-overview-of-kursk-training-area-15april2025-wv2.jpg?q=w_1110,c_fill",
    "https://factly.in/wp-content/uploads/2025/02/Train-fire-in-Prayagraj-Claim.jpg",
    "https://www.washingtonpost.com/wp-apps/imrs.php?src=https://arc-anglerfish-washpost-prod-washpost.s3.amazonaws.com/public/MBWA4LJ5XLVC6CJLZG2OQFMGWE.JPG&w=1440&impolicy=high_res",
    "https://factuel.afp.com/sites/default/files/styles/header_article/public/medias/factchecking/g2/2025-07/c1452a5562cfe3e178b0d5c6681c940e-fr.jpeg?itok=a9Wc3hEY"
])
async def test_download_image(url):
    img = await download_img(url)
    print(img)
    assert isinstance(img, Image)
