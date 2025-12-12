import base64
import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from ezmm.common.items.item import Item

logger = logging.getLogger("ezMM")


class Video(Item):
    kind = "video"
    _video: Optional[cv2.VideoCapture] = None

    def __init__(self, file_path: str | Path = None,
                 binary_data: bytes = None,
                 source_url: str = None,
                 reference: str = None,
                 id: int = None):
        assert file_path or binary_data or reference or id is not None

        if binary_data:
            # Save binary data to temporary file
            file_path = self._temp_file_path(suffix=".mp4")
            file_path.parent.mkdir(parents=True, exist_ok=True)
            with open(file_path, 'wb') as f:
                f.write(binary_data)

        super().__init__(file_path,
                         source_url=source_url,
                         reference=reference,
                         id=id)

    @property
    def video(self) -> cv2.VideoCapture:
        """Lazy-loads the video capture of this Video item."""
        # Re-open capture if it was never opened or has been released
        if not self._video or not self._video.isOpened():
            self._open_video()
        return self._video

    def _open_video(self):
        """Opens the video file for reading."""
        self._video = cv2.VideoCapture(str(self.file_path))

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

    def sample_frames(self, n_frames: int = 5) -> list[np.ndarray]:
        """Returns the specified number of frames sampled evenly spaced from the video.
        Always includes the first frame. Includes the last frame if n_frames > 1."""
        assert n_frames > 0, "Number of frames must be greater than 0."
        # Efficient sampling: seek directly to the required frame indices instead of
        # decoding the entire video into memory.
        # Open video if necessary
        if not self.video.isOpened():
            self._open_video()

        total_frames = self.frame_count
        if total_frames <= 0:
            return []

        n_frames = min(n_frames, total_frames)
        frame_ids = np.linspace(0, total_frames - 1, n_frames, dtype=int)

        sampled: list[np.ndarray] = []
        try:
            for fid in frame_ids:
                # Seek to the desired frame index
                self.video.set(cv2.CAP_PROP_POS_FRAMES, int(fid))
                success, frame = self.video.read()
                if not success or frame is None:
                    break
                _, enc = cv2.imencode(".jpeg", frame)
                sampled.append(enc)
        finally:
            # Match previous behavior: release capture after sampling
            # Ensure internal handle is reset so future property reads re-open
            self.close()

        return sampled

    def get_base64_encoded(self, n_frames: int = 5) -> list[str]:
        """Returns base64-encoded frames, evenly sampled from the video."""
        frames = self.sample_frames(n_frames)
        frames_encoded = [base64.b64encode(frame).decode("utf-8") for frame in frames]
        return frames_encoded

    def _same(self, other):
        return (
                self.width == other.width and
                self.height == other.height and
                self.frame_count == other.frame_count and
                self.file_path.read_bytes() == other.file_path.read_bytes()
        )

    def as_html(self) -> str:
        return f'<video controls src="/items/{self.file_path_relative.as_posix()}"></video>'

    def close(self):
        if self._video:
            self._video.release()
            self._video = None
