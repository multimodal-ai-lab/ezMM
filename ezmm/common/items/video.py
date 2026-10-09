import base64
import logging
from pathlib import Path

import cv2
import mutagen
import numpy as np

from ezmm.common.items.item import Item

logger = logging.getLogger("ezMM")


class Video(Item):
    kind = "video"

    def __init__(self, file_path: str | Path | None = None,
                 binary_data: bytes | None = None,
                 source_url: str | None = None,
                 reference: str | None = None,
                 id: int | None = None):
        assert file_path or binary_data or reference or id is not None

        if binary_data:
            # Save binary data to temporary file
            file_path = self._write_temp_file(binary_data, suffix=".mp4")

        super().__init__(file_path,
                         source_url=source_url,
                         reference=reference,
                         id=id)

    def _open_cap(self) -> cv2.VideoCapture:
        cap = cv2.VideoCapture(str(self.file_path))
        if not cap.isOpened():
            cap.release()  # Ensure resources are cleaned even on failure
            raise OSError(f"Failed to open video: {self.file_path}")
        return cap

    def _read_prop(self, prop_id: int) -> float:
        cap = self._open_cap()
        try:
            return float(cap.get(prop_id))
        finally:
            cap.release()

    @property
    def width(self) -> int:
        return int(self._read_prop(cv2.CAP_PROP_FRAME_WIDTH))

    @property
    def height(self) -> int:
        return int(self._read_prop(cv2.CAP_PROP_FRAME_HEIGHT))

    @property
    def has_video_stream(self) -> bool:
        """Returns False if the file contains no video stream (e.g., an audio-only MP4)."""
        return self.width > 0 and self.height > 0

    @property
    def frame_count(self) -> int:
        """Returns the number of frames (0 if unknown or if there is no video stream)."""
        if not self.has_video_stream:
            return 0
        return max(int(self._read_prop(cv2.CAP_PROP_FRAME_COUNT)), 0)

    @property
    def fps(self) -> float:
        """Returns the frame rate (0 if unknown or if there is no video stream)."""
        if not self.has_video_stream:
            return 0.0
        return max(float(self._read_prop(cv2.CAP_PROP_FPS)), 0.0)

    @property
    def duration(self) -> float:
        """Returns the duration of the video in seconds (also for videos without video stream)."""
        fps, frame_count = self.fps, self.frame_count
        if fps > 0 and frame_count > 0:
            return frame_count / fps
        try:  # Read the duration from the container instead
            info = getattr(mutagen.File(self.file_path), "info", None)
            return float(getattr(info, "length", 0.0) or 0.0)
        except Exception:
            return 0.0

    def sample_frames(self, n_frames: int = 5, *, format: str = "rgb") -> list[np.ndarray] | list[bytes]:
        """Returns ``n_frames`` frames sampled evenly from the video.

        - ``format='rgb'`` (default): returns a list of RGB numpy arrays with shape ``(H, W, 3)``.
        - ``format='jpeg'``: returns a list of JPEG-encoded bytes for each sampled frame.

        Always includes the first frame. Includes the last frame if ``n_frames > 1``.
        """
        assert n_frames > 0, "Number of frames must be greater than 0."

        cap = self._open_cap()
        try:
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total_frames <= 0:
                return []

            n_frames = int(min(n_frames, total_frames))
            frame_ids = np.linspace(0, total_frames - 1, n_frames, dtype=int)

            if format not in {"rgb", "jpeg"}:
                raise ValueError("format must be either 'rgb' or 'jpeg'")

            sampled: list[np.ndarray | bytes] = []
            for fid in frame_ids:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(fid))
                success, frame = cap.read()
                if not success or frame is None:
                    break
                if format == "rgb":
                    # Convert BGR (OpenCV) to RGB (expected for PIL/fromarray and general usage)
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    sampled.append(rgb)
                else:
                    # Directly JPEG-encode the BGR frame to avoid extra conversions
                    ok, enc = cv2.imencode(".jpeg", frame)
                    if not ok:
                        break
                    sampled.append(enc.tobytes())
            return sampled
        finally:
            cap.release()

    def get_base64_encoded(self, n_frames: int = 5) -> list[str]:
        """Returns base64-encoded JPEG frames, evenly sampled from the video."""
        frames = self.sample_frames(n_frames, format="jpeg")
        return [base64.b64encode(frame).decode("utf-8") for frame in frames]

    def as_html(self) -> str:
        return f'<video controls preload="metadata" src="{self.file_url}"></video>'

