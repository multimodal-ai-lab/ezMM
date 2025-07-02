import base64
from io import BytesIO
from pathlib import Path
from typing import Optional
import logging

import aiohttp
from PIL.Image import Image as PillowImage, open as pillow_open, new as pillow_new
import pillow_avif  # Keep this import for AVIF support

from ezmm.common.items.item import Item
from ezmm.request import request_static

logger = logging.getLogger("ezMM")
logger.debug(f"`pillow_avif` v{pillow_avif.__version__} loaded for AVIF image support.")

class Image(Item):
    kind = "image"
    _image: Optional[PillowImage] = None

    def __init__(self, file_path: str | Path = None,
                 pillow_image: PillowImage = None,
                 binary_data: bytes = None,
                 source_url: str = None,
                 reference: str = None):
        assert file_path or pillow_image or binary_data or reference

        if hasattr(self, "id"):
            # The image is already initialized (existing instance returned via __new__())
            return

        if binary_data is not None:
            pillow_image = pillow_open(BytesIO(binary_data))

        if pillow_image is not None:
            pillow_image = _ensure_rgb_mode(pillow_image)
            # Save the image in a temporary folder
            file_path = self._temp_file_path(suffix=".jpg")
            file_path.parent.mkdir(parents=True, exist_ok=True)
            pillow_image.save(file_path)
            self._image = pillow_image

        super().__init__(file_path,
                         source_url=source_url,
                         reference=reference)

    @property
    def image(self) -> PillowImage:
        """Lazy-loads the PIL image of this Image item."""
        if not self._image:
            image = pillow_open(self.file_path)
            self._image = _ensure_rgb_mode(image)
        return self._image

    def get_base64_encoded(self) -> str:
        buffered = BytesIO()
        self.image.save(buffered, format="JPEG")
        return base64.b64encode(buffered.getvalue()).decode('utf-8')

    @property
    def width(self) -> int:
        return self.image.width

    @property
    def height(self) -> int:
        return self.image.height

    def _same(self, other):
        return (
                self.image.mode == other.image.mode and
                self.image.size == other.image.size and
                self.image.tobytes() == other.image.tobytes()
        )

    def as_html(self) -> str:
        img = f'<img src="/{self.file_path.as_posix()}" alt="{self.reference}">'
        if self.source_url:
            return f'<a href="{self.source_url}">{img}</a>'
        else:
            return img

    def close(self):
        if self._image:
            self._image.close()
            self._image = None


def _ensure_rgb_mode(pillow_image: PillowImage) -> PillowImage:
    """Turns any kind of image (incl. PNGs) into RGB mode to make it JPEG-saveable."""
    if pillow_image.mode in ["RGBA", "P"]:
        pillow_image = pillow_image.convert('RGBA')
        converted = pillow_new("RGB", pillow_image.size, (255, 255, 255))
        converted.paste(pillow_image, mask=pillow_image.split()[3])  # 3 is the alpha channel
        return converted
    if pillow_image.mode != "RGB":
        return pillow_image.convert('RGB')
    else:
        return pillow_image


async def download_image(
        image_url: str,
        session: aiohttp.ClientSession,
        ignore_small_images: bool = True,
        max_size: tuple[int, int] = (2048, 2048)
) -> Optional[Image]:
    """Download an image from a URL and return it as an Image object."""
    # TODO: Handle very large images like: https://eoimages.gsfc.nasa.gov/images/imagerecords/144000/144225/campfire_oli_2018312_lrg.jpg
    content = await request_static(image_url, session, get_text=False)
    if content:
        pillow_img = pillow_open(BytesIO(content))

        if pillow_img:
            if pillow_img.width > max_size[0] or pillow_img.height > max_size[1]:
                pillow_img.thumbnail(max_size, PillowImage.LANCZOS)  # Preserves aspect ratio

            if not ignore_small_images or (pillow_img.width > 256 and pillow_img.height > 256):
                # TODO: Check for duplicates, i.e., reuse an existing image if it already exists in the registry
                image = Image(pillow_image=pillow_img, source_url=image_url)
                image.relocate(move_not_copy=True)  # Ensure the image is in the temp dir + follows simple naming
                return image
