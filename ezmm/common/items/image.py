import logging
from io import BytesIO
from pathlib import Path
from typing import Optional

import pillow_avif  # Keep this import for AVIF support
from PIL.Image import Image as PillowImage, open as pillow_open, new as pillow_new

from ezmm.common.items.item import Item
from ezmm.util import to_base64

logger = logging.getLogger("ezMM")
logger.debug(f"`pillow_avif` v{pillow_avif.__version__} loaded for AVIF image support.")


class Image(Item):
    kind = "image"
    _image: Optional[PillowImage] = None

    def __init__(self, file_path: str | Path | None = None,
                 pillow_image: PillowImage | None = None,
                 binary_data: bytes | None = None,
                 source_url: str | None = None,
                 reference: str | None = None,
                 id: int = None):
        assert file_path or pillow_image or binary_data or reference or id is not None

        if binary_data is not None:
            pillow_image = pillow_open(BytesIO(binary_data))

        if pillow_image is not None:
            pillow_image = _ensure_rgb_mode(pillow_image)

            # Save the image in a temporary folder
            file_path = self._temp_file_path(suffix=".jpg")
            file_path.parent.mkdir(parents=True, exist_ok=True)
            pillow_image.save(file_path)
            self._image = pillow_image

            # Free memory and file handle
            self.close()

        super().__init__(file_path,
                         source_url=source_url,
                         reference=reference,
                         id=id)

    @property
    def image(self) -> PillowImage:
        """Lazy-loads the PIL image of this Image item."""
        if not self._image:
            with pillow_open(self.file_path) as img:
                self._image = _ensure_rgb_mode(img).copy()
        return self._image

    def get_base64_encoded(self) -> str:
        return to_base64(self.image)

    @property
    def width(self) -> int:
        return self.image.width

    @property
    def height(self) -> int:
        return self.image.height

    def _same(self, other):
        # TODO: Check for duplicates, i.e., reuse an existing image if it already exists in the registry
        return (
                self.image.mode == other.image.mode and
                self.image.size == other.image.size and
                self.image.tobytes() == other.image.tobytes()
        )

    def as_html(self) -> str:
        img = f'<img src="/items/{self.file_path_relative.as_posix()}" alt="{self.reference}">'
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
