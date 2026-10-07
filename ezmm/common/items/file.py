import base64
import logging
import mimetypes
import os
from html import escape
from pathlib import Path
from typing import Optional

from ezmm.common.items.item import Item

logger = logging.getLogger("ezMM")

DEFAULT_MAX_FILE_SIZE = 100 * 1024 ** 2  # 100 MB


class FileTooLargeError(ValueError):
    """Raised when a file exceeds the maximum file size allowed for File items."""


def _read_max_file_size_env() -> Optional[int]:
    value = os.getenv("EZMM_MAX_FILE_SIZE")
    if value is None:
        return DEFAULT_MAX_FILE_SIZE
    return None if value.strip().lower() in ("", "none", "0") else int(value)


_max_file_size: Optional[int] = _read_max_file_size_env()


def set_max_file_size(n_bytes: Optional[int]):
    """Sets the maximum size (in bytes) of files that can be loaded as File items.
    Use None to disable the limit. Can also be set via the EZMM_MAX_FILE_SIZE
    environment variable. Default: 100 MB."""
    global _max_file_size
    if n_bytes is not None and n_bytes <= 0:
        raise ValueError("The maximum file size must be positive (or None for no limit).")
    _max_file_size = n_bytes


def get_max_file_size() -> Optional[int]:
    return _max_file_size


def _check_size(size: int):
    if _max_file_size is not None and size > _max_file_size:
        raise FileTooLargeError(f"File size of {size} bytes exceeds the maximum "
                                f"file size of {_max_file_size} bytes.")


class File(Item):
    """Any other kind of file (e.g., PDFs, Excel sheets, Word documents, archives).
    Files larger than the maximum file size (see `set_max_file_size()`) are rejected
    with a `FileTooLargeError`."""
    kind = "file"

    def __init__(self, file_path: str | Path = None,
                 binary_data: bytes = None,
                 source_url: str = None,
                 reference: str = None,
                 id: int = None,
                 mime_type: str = None,
                 suffix: str = None):
        assert file_path or binary_data or reference or id is not None

        if binary_data and not hasattr(self, "id"):
            _check_size(len(binary_data))  # Check before writing anything
            suffix = suffix or (mimetypes.guess_extension(mime_type) if mime_type else None) or ".bin"
            file_path = self._write_temp_file(binary_data, suffix=suffix)

        super().__init__(file_path,
                         source_url=source_url,
                         reference=reference,
                         id=id)

    def _validate_new_file(self):
        _check_size(self.file_path.stat().st_size)

    def get_base64_encoded(self) -> str:
        """Returns the file's raw bytes as a base64-encoded string."""
        return base64.b64encode(self.bytes).decode("utf-8")

    def as_html(self) -> str:
        """Returns a download link to the file."""
        suffix = self.file_path.suffix
        label = f"{self.reference} ({suffix.lstrip('.').upper() or 'FILE'}, {format_size(self.size)})"
        return (f'<a class="file-link" href="{self.file_url}" download="{self.kind}_{self.id}{suffix}">'
                f'{escape(label)}</a>')


def format_size(n_bytes: int) -> str:
    """Returns the size in a human-readable format, e.g., '1.2 MB'."""
    size = float(n_bytes or 0)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024:
            break
        size /= 1024
    else:
        unit = "TB"
    return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
