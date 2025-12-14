import base64
import logging
from pathlib import Path
from PIL.Image import fromarray

import cv2
import numpy as np

from ezmm.common.items.item import Item

logger = logging.getLogger("ezMM")


class Video(Item):
    kind = "video"

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

    def _open_cap(self) -> cv2.VideoCapture:
        cap = cv2.VideoCapture(str(self.file_path))
        if not cap.isOpened():
            # Ensure resources are cleaned even on failure
            try:
                cap.release()
            except Exception:
                pass
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
    def frame_count(self) -> int:
        return int(self._read_prop(cv2.CAP_PROP_FRAME_COUNT))

    @property
    def fps(self) -> float:
        return float(self._read_prop(cv2.CAP_PROP_FPS))

    @property
    def duration(self) -> float:
        """Returns the duration of the video in seconds."""
        fps = self.fps
        if fps <= 0:
            return 0.0
        return self.frame_count / fps

    @property
    def bytes(self) -> bytes:
        """Returns the video as bytes."""
        return self.file_path.read_bytes()

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
        return f'<video controls src="/items/{self.file_path_relative.as_posix()}"></video>'

    def _compute_embedding(self) -> np.ndarray:
        """Returns the average embedding of 5 equal-distance video frames."""
        from ezmm.embedding import embed
        frames = self.sample_frames(5, format='rgb')
        pillow_images = [fromarray(frame) for frame in frames]
        embeddings = embed(pillow_images)
        return np.mean(embeddings, axis=0)
