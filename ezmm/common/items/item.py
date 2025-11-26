import asyncio
import logging
import re
from abc import ABC
from datetime import datetime
from pathlib import Path
from shutil import copyfile, move
from typing import Sequence, Optional

import aiohttp

from ezmm.request import is_maybe_image_url, is_maybe_video_url
from ezmm.util import is_item_ref, normalize_path

logger = logging.getLogger("ezMM")

REF = "<{kind}:{id}>"  # General reference template, defining the reference syntax


class Item(ABC):
    """An element of MultimodalSequences. The data of each item is saved in an individual file."""
    kind: str  # Specifies the type of the item (image, video, ...)
    id: int  # Unique identifier of this item within its kind
    file_path: Path  # The (absolute) path to the file where the data of this item is stored
    source_url: str  # The (web or file) URL pointing at the Item data's origin

    def __new__(cls, file_path: Path | str = None, source_url: str = None, reference: str = None, **kwargs):
        """Checks if there already exists an instance of the item with the given reference.
        If yes, returns the existing reference. Otherwise, instantiates a new one."""

        if file_path or reference:
            # Look up an existing instance instead of creating a new one
            from ezmm.common.registry import item_registry
            item = item_registry.get_cached(reference=reference, kind=cls.kind, file_path=file_path)
            if item:
                item.source_url = source_url or item.source_url
                return item
            elif reference:
                raise ValueError(f"No item with reference '{reference}'.")

        return super().__new__(cls)

    def __init__(self, file_path: Path | str, source_url: str = None, reference: str = None):
        if hasattr(self, "id"):
            # The item is already initialized (existing instance returned via __new__())
            if not self.file_path.exists():
                logger.warning(f"Item {self.reference} does not exist at {self.file_path}.")
            return
        self.file_path = normalize_path(file_path)
        self.source_url = source_url or self.file_path.absolute().as_uri()
        from ezmm.common.registry import item_registry
        item_registry.add_and_assign_id(self)  # Ensure the item is registered and get an ID assigned

    @property
    def reference(self) -> str:
        return REF.format(kind=self.kind, id=self.id)

    def _same(self, other) -> bool:
        """Compares the content data with the other item for equality."""
        raise NotImplementedError

    @staticmethod
    def from_reference(reference: str) -> Optional["Item"]:
        from ezmm.common.registry import item_registry
        return item_registry.get(reference)

    def close(self):
        """Closes any resources held by this item."""
        pass

    def as_html(self, path_relative_to: str | Path = None) -> str:
        """Returns the item as HTML code."""
        return f"<p>Item {self.reference} does not support HTML yet.</p>"

    def relocate(self, move_not_copy=False):
        """Copies the item's file to the temp_dir if not
        located there already. Moves it instead if move=True."""
        new_path = self._default_file_path(suffix=self.file_path.suffix)
        if self.file_path != new_path:
            # Ensure the target directory exists
            new_path.parent.mkdir(parents=True, exist_ok=True)

            # Move/copy file to target directory
            self.close()
            move(self.file_path, new_path) if move_not_copy else copyfile(self.file_path, new_path)

            # Update the file path to the new location
            self.file_path = new_path
            from ezmm.common.registry import item_registry
            item_registry.update_file_path(self)

    def get_size(self) -> int:
        """Returns the size of the item in bytes."""
        return self.file_path.stat().st_size

    def _temp_file_path(self, suffix: str = "") -> Path:
        """Used when the item's ID is not set yet."""
        from ezmm.common.registry import item_registry
        filename = datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f") + suffix
        return normalize_path(item_registry.path / "items" / filename)

    def _default_file_path(self, suffix: str = "") -> Path:
        """Only usable after item initialization."""
        from ezmm.common.registry import item_registry
        default_filename = str(self.id) + suffix
        return normalize_path(item_registry.path / self.kind / default_filename)

    def __eq__(self, other):
        return (self is other or
                isinstance(other, Item) and (
                        self.kind == other.kind and self.id == other.id or  # Should never trigger
                        self._same(other)
                ))

    def __hash__(self):
        # TODO: Make hash content-dependent => identify known items by hash
        return hash((self.kind, self.id))


def resolve_references_from_sequence(seq: Sequence[str | Item]) -> list[str | Item]:
    """Identifies all item references within the sequence and replaces them with
    an instance of the referenced item. Returns the (interleaved) list of
    strings and items."""
    processed = []
    for item in seq:
        if isinstance(item, str):
            if item.strip(" "):  # Drop excess whitespaces
                resolved = resolve_references_from_string(item)
                processed.extend(resolved)
        elif item:  # Drop Nones
            processed.append(item)
    return processed


def resolve_references_from_string(string: str) -> list[str | Item]:
    """Identifies all item references within the string and replaces them with
    an instance of the referenced item. Returns the (interleaved) list of
    strings and items."""
    from ezmm.common.registry import item_registry
    from ezmm.common.items import ITEM_REF_REGEX
    ref_regex = rf"\s?{ITEM_REF_REGEX}\s?"  # Extend to optional whitespaces before and after the ref
    split = re.split(ref_regex, string)
    # Replace each reference with its actual item object
    for i in range(len(split)):
        substr = split[i]
        if is_item_ref(substr):
            item = item_registry.get(substr)
            if item is not None:
                split[i] = item
    return split


async def download_item(url: str,
                        session: Optional[aiohttp.ClientSession] = None,
                        ignore_small_images: bool = True) -> Optional[Item]:
    """Downloads the item from the given URL and returns an instance of the
    corresponding item class. Reuses a session if provided."""

    own_session = session is None
    if own_session:
        session = aiohttp.ClientSession()

    try:
        if await is_maybe_image_url(url, session):
            from ezmm.common.items.image import download_image
            return await download_image(url, ignore_small_images=ignore_small_images, session=session)
        if await is_maybe_video_url(url, session):
            from ezmm.common.items.video import download_video
            return await download_video(url, session)
        # TODO: Handle audios
    finally:
        if own_session:
            await session.close()


if __name__ == "__main__":
    result = asyncio.run(download_item(
        "https://media.cnn.com/api/v1/images/stellar/prod/02-overview-of-kursk-training-area-15april2025-wv2.jpg?q=w_1110,c_fill"))
    print(result)
