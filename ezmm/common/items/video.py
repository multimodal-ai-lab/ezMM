import base64
import logging
from pathlib import Path
from typing import Optional
from urllib.parse import urljoin

import aiohttp
import cv2
import m3u8
import numpy as np

from ezmm.common.items.item import Item
from ezmm.request import fetch_headers, HEADERS
from ezmm.util import ts_to_mp4, to_base64

logger = logging.getLogger("ezMM")


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

        if binary_data:
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

    def _get_frames(self) -> list[bytes]:
        """Returns the list of all frames from the video."""
        frames = []
        while self.video.isOpened():
            success, frame = self.video.read()
            if not success:
                break
            _, frame = cv2.imencode(".jpeg", frame)
            frames.append(frame)
        self.video.release()
        return frames

    def sample_frames(self, n_frames: int = 5) -> list[bytes]:
        """Returns the specified number of frames sampled evenly spaced from the video.
        Always includes the first frame. Includes the last frame if n_frames > 1."""
        assert n_frames > 0, "Number of frames must be greater than 0."
        frames = self._get_frames()
        frame_ids = np.linspace(0, len(frames) - 1, n_frames, dtype=int)
        sampled = [frames[i] for i in frame_ids]
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
        return f'<video controls src="/{self.file_path.as_posix()}"></video>'

    def close(self):
        if self._video:
            self._video.release()
            self._video = None


async def download_video(
        video_url: str,
        session: aiohttp.ClientSession
) -> Optional[Video]:
    """Downloads the linked video (stream) and returns it as a Video object."""

    try:
        headers = await fetch_headers(video_url, session, timeout=3)
        content_type = headers.get('Content-Type') or headers.get('content-type')
        if content_type.startswith("video/"):
            return await download_video_file(video_url, session)
        elif content_type == "application/vnd.apple.mpegurl":
            return await download_hls_video(video_url, session)
        else:
            logger.warning(f"Cannot download video! Unable to handle content type: {content_type}.")

    except Exception as e:
        logger.debug(f"Error downloading video from {video_url}"
                     f"\n{type(e).__name__}: {e}")


async def download_video_file(
        video_url: str,
        session: aiohttp.ClientSession
) -> Optional[Video]:
    """Download a single video file from a URL and return it as a Video object."""
    try:
        async with session.get(video_url, headers=HEADERS) as response:
            if response.status == 200:
                content = await response.read()
                video = Video(binary_data=content, source_url=video_url)
                video.relocate(move_not_copy=True)
                return video
    except Exception as e:
        logger.debug(f"Error downloading video file from {video_url}"
                     f"\n{type(e).__name__}: {e}")


async def download_hls_video(
        playlist_url: str,
        session: aiohttp.ClientSession
) -> Optional[Video]:
    """Download an HTTP Live Streaming (HLS) video from a playlist URL and return it as a Video object."""
    try:
        # Download the m3u8 playlist file
        async with session.get(playlist_url, headers=HEADERS) as response:
            if response.status != 200:
                logger.debug(f"Failed to download playlist: {response.status}")
                return None
            playlist_content = await response.text()

        playlist = m3u8.loads(playlist_content)
        base_url = playlist_url.rsplit('/', 1)[0] + '/'

        # Check if this is a master playlist (contains variant playlists)
        if playlist.is_variant:
            # Choose the highest quality variant
            best_playlist = playlist.playlists[-1]  # Usually the last one is of highest quality

            # Manually construct the absolute URL for the variant playlist
            variant_url = urljoin(base_url, best_playlist.uri)

            # Download the variant playlist
            async with session.get(variant_url, headers=HEADERS) as var_response:
                if var_response.status != 200:
                    logger.error(f"Failed to download variant playlist: {var_response.status}")
                    return None
                variant_content = await var_response.text()

            # Parse the variant playlist
            variant_playlist = m3u8.loads(variant_content)
            playlist = variant_playlist  # Use this for segment downloads

            # Update base_url for segment downloads
            base_url = variant_url.rsplit('/', 1)[0] + '/'

        # Download all segments
        video_segments = []

        for i, segment in enumerate(playlist.segments):
            # Construct full URL for the segment
            if segment.uri.startswith('http'):
                segment_url = segment.uri
            else:
                segment_url = urljoin(base_url, segment.uri)

            # Download the segment with SSL disabled
            try:
                async with session.get(segment_url, headers=HEADERS) as seg_response:
                    if seg_response.status == 200:
                        segment_data = await seg_response.read()
                        video_segments.append(segment_data)
            except Exception as e:
                logger.debug(f"Failed to download segment {i} from {segment_url}: {e}")

        # Combine all segments
        if video_segments:
            ts_bytes = b''.join(video_segments)
            mp4_bytes = ts_to_mp4(ts_bytes)

            # Create Video object with MP4 content
            video = Video(binary_data=mp4_bytes, source_url=playlist_url)
            video.relocate(move_not_copy=True)
            return video

    except Exception as e:
        logger.debug(f"Error downloading HLS video from {playlist_url}"
                     f"\n{type(e).__name__}: {e}")

    return None


async def download_vid(url):
    async with aiohttp.ClientSession() as session:
        return await download_video(url, session)


if __name__ == "__main__":
    import asyncio

    video = asyncio.run(
        download_vid("https://upload.wikimedia.org/wikipedia/commons/transcoded/a/a7/How_to_make_video.webm/How_to_make_video.webm.1080p.vp9.webm"))
    print(video)
