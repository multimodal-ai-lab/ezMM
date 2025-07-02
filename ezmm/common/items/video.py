import base64

import aiohttp
import cv2
from pathlib import Path
from typing import Optional
from ezmm.common.items import Item


class Video(Item):
    kind = "video"
    _video: Optional[cv2.VideoCapture] = None

    def __init__(self, file_path: str | Path = None,
                 binary_data: bytes = None,
                 source_url: str = None,
                 reference: str = None):
        assert file_path or binary_data or reference

        if hasattr(self, "id"):
            return

        if binary_data is not None:
            # Save binary data to temporary file
            file_path = self._temp_file_path(suffix=".mp4")
            file_path.parent.mkdir(parents=True, exist_ok=True)
            with open(file_path, 'wb') as f:
                f.write(binary_data)

        super().__init__(file_path,
                         source_url=source_url,
                         reference=reference)

    @property
    def video(self) -> cv2.VideoCapture:
        """Lazy-loads the video capture of this Video item."""
        if not self._video:
            self._video = cv2.VideoCapture(str(self.file_path))
        return self._video

    @property
    def width(self) -> int:
        return int(self.video.get(cv2.CAP_PROP_FRAME_WIDTH))

    @property
    def height(self) -> int:
        return int(self.video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    @property
    def frame_count(self) -> int:
        return int(self.video.get(cv2.CAP_PROP_FRAME_COUNT))

    @property
    def fps(self) -> float:
        return self.video.get(cv2.CAP_PROP_FPS)

    @property
    def duration(self) -> float:
        """Returns the duration of the video in seconds."""
        return self.frame_count / self.fps

    @property
    def bytes(self) -> bytes:
        """Returns the video as bytes."""
        return self.file_path.read_bytes()

    def get_base64_encoded(self) -> str:
        """Returns the base64-encoded video as a string."""
        return base64.b64encode(self.bytes).decode('utf-8')

    def _same(self, other):
        return (
                self.width == other.width and
                self.height == other.height and
                self.frame_count == other.frame_count and
                self.file_path.read_bytes() == other.file_path.read_bytes()
        )

    def as_html(self) -> str:
        return f'<video controls src="/{self.file_path.as_posix()}"></video>'

    def close(self):
        if self._video:
            self._video.release()
            self._video = None


async def download_video(
        video_url: str,
        session: aiohttp.ClientSession
) -> Optional[Video]:
    """Download a video from a URL and return it as a Video object."""
    try:
        async with session.get(video_url) as response:
            if response.status == 200:
                content = await response.read()
                video = Video(binary_data=content, source_url=video_url)
                video.relocate(move_not_copy=True)
                return video
    except Exception:
        pass
