import base64
import logging
import mimetypes
from io import BytesIO
from pathlib import Path

import mutagen

from ezmm.common.items.item import Item

logger = logging.getLogger("ezMM")

# Maps mutagen's file type classes to file extensions
_TYPE2SUFFIX = {
    "MP3": ".mp3", "EasyMP3": ".mp3", "WAVE": ".wav", "FLAC": ".flac", "OggVorbis": ".ogg",
    "OggOpus": ".opus", "OggFLAC": ".oga", "OggSpeex": ".spx", "MP4": ".m4a", "EasyMP4": ".m4a",
    "AAC": ".aac", "AIFF": ".aiff", "ASF": ".wma", "AC3": ".ac3", "MonkeysAudio": ".ape",
    "WavPack": ".wv", "TrueAudio": ".tta", "DSF": ".dsf",
}


class Audio(Item):
    """An audio-only media file (e.g., MP3, WAV, FLAC, OGG, M4A)."""
    kind = "audio"
    _info = None  # Cached mutagen stream info

    def __init__(self, file_path: str | Path = None,
                 binary_data: bytes = None,
                 source_url: str = None,
                 reference: str = None,
                 id: int = None,
                 mime_type: str = None):
        assert file_path or binary_data or reference or id is not None

        if binary_data and not hasattr(self, "id"):
            # Save binary data to temporary file with a suitable extension
            suffix = (mimetypes.guess_extension(mime_type) if mime_type else None) or _sniff_suffix(binary_data)
            file_path = self._write_temp_file(binary_data, suffix=suffix)

        super().__init__(file_path,
                         source_url=source_url,
                         reference=reference,
                         id=id)

    @property
    def info(self):
        """Lazy-loads the stream info (duration, sample rate, ...) of the audio file."""
        if self._info is None:
            audio = mutagen.File(self.file_path)
            if audio is None:
                raise OSError(f"Unsupported or invalid audio file: {self.file_path}")
            self._info = audio.info
        return self._info

    @property
    def duration(self) -> float:
        """Returns the duration of the audio in seconds."""
        return float(getattr(self.info, "length", 0.0) or 0.0)

    @property
    def sample_rate(self) -> int:
        return int(getattr(self.info, "sample_rate", 0) or 0)

    @property
    def channels(self) -> int:
        return int(getattr(self.info, "channels", 0) or 0)

    @property
    def bitrate(self) -> int:
        """Returns the bitrate in bits per second."""
        return int(getattr(self.info, "bitrate", 0) or 0)

    def get_base64_encoded(self) -> str:
        """Returns the audio file's raw bytes as a base64-encoded string."""
        return base64.b64encode(self.bytes).decode("utf-8")

    def as_html(self) -> str:
        return f'<audio controls preload="metadata" src="{self.file_url}"></audio>'


def _sniff_suffix(binary_data: bytes) -> str:
    """Determines the file extension from the audio's content."""
    try:
        audio = mutagen.File(BytesIO(binary_data))
    except Exception:
        audio = None
    if audio is None:
        logger.warning("Could not determine the audio format. Saving it with extension '.bin'.")
        return ".bin"
    return _TYPE2SUFFIX.get(type(audio).__name__, ".bin")
