import base64
from io import BytesIO
from pathlib import Path
from typing import Optional

from PIL.Image import Image as PillowImage, open as pillow_open

from ezmm.common.items.item import Item


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

    def close(self):
        if self._image:
            self._image.close()
            self._image = None


def _ensure_rgb_mode(pillow_image: PillowImage) -> PillowImage:
    """Turns any kind of image (incl. PNGs) into RGB mode to make it JPEG-saveable."""
    if pillow_image.mode != "RGB":
        return pillow_image.convert('RGB')
    else:
        return pillow_image
