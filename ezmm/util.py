import re
import subprocess
import tempfile
from pathlib import Path
from typing import Optional
import imageio_ffmpeg as ffmpeg
from io import BytesIO
import base64
from PIL.Image import Image as PillowImage


def get_item_refs(text: str) -> list[str]:
    """Extracts all item references from the text."""
    from ezmm.common.items import ITEM_REF_REGEX
    pattern = re.compile(ITEM_REF_REGEX, re.DOTALL)
    matches = pattern.findall(text)
    return matches


def parse_ref(ref: str) -> (str, int):
    result = parse_item_ref(ref)
    if result is None:
        raise ValueError(f"Invalid item reference: {ref}")
    return result


def parse_item_ref(reference: str) -> Optional[tuple[str, int]]:
    """Returns the first matching kind and identifier from the reference."""
    from ezmm.common.items import ITEM_KIND_ID_REGEX
    pattern = re.compile(ITEM_KIND_ID_REGEX, re.DOTALL)
    result = pattern.findall(reference)
    if len(result) > 0:
        match = result[0]
        return match[0], int(match[1])
    else:
        return None


def is_item_ref(string: str) -> bool:
    """Returns True iff the string represents an item reference."""
    from ezmm.common.items import ITEM_REF_REGEX
    pattern = re.compile(ITEM_REF_REGEX, re.DOTALL)
    return pattern.fullmatch(string) is not None


def validate_references(text: str) -> bool:
    """Verifies that each item reference can be resolved to a registered item."""
    from ezmm.common.registry import item_registry
    refs = get_item_refs(text)
    for ref in refs:
        if item_registry.get(ref) is None:
            return False
    return True


def to_base64(image: PillowImage) -> str:
    """Converts the given Pillow Image to a base64-encoded string."""
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')


def ts_to_mp4(ts_bytes: bytes) -> bytes:
    """Converts a TS video (bytes) to MP4 (bytes) using FFmpeg with temporary files."""
    with tempfile.NamedTemporaryFile(suffix='.ts', delete=False) as temp_ts_file:
        temp_ts_file.write(ts_bytes)
        temp_ts_path = Path(temp_ts_file.name)

    temp_mp4_path = temp_ts_path.with_suffix('.mp4')

    try:
        # Run FFmpeg to convert TS to MP4
        cmd = [
            ffmpeg.get_ffmpeg_exe(),  # ensure ffmpeg is available in the environment
            "-i", str(temp_ts_path),
            "-c:v", "copy",
            "-c:a", "copy",
            "-f", "mp4",
            str(temp_mp4_path)
        ]
        result = subprocess.run(cmd, stderr=subprocess.PIPE, stdout=subprocess.PIPE)

        if result.returncode != 0:
            raise RuntimeError(f"FFmpeg error:\n{result.stderr.decode(errors='ignore')}")

        # Read and return the MP4 bytes
        with open(temp_mp4_path, "rb") as mp4_file:
            mp4_bytes = mp4_file.read()

    finally:
        # Clean up temporary files
        temp_ts_path.unlink(missing_ok=True)
        temp_mp4_path.unlink(missing_ok=True)

    return mp4_bytes
