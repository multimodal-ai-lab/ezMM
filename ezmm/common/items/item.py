import logging
import re
from abc import ABC
from datetime import datetime
from pathlib import Path
from shutil import copyfile, move
from typing import Sequence, Optional

import numpy as np

from ezmm.util import is_item_ref

logger = logging.getLogger("ezMM")

REF = "<{kind}:{id}>"  # General reference template, defining the reference syntax


class Item(ABC):
    """An element of MultimodalSequences. The data of each item is saved in an individual file."""
    kind: str  # Specifies the type of the item (image, video, ...)
    id: int  # Unique identifier of this item within its kind
    file_path: Path  # The absolute path to the file
    source_url: str  # The (web or file) URL pointing at the Item data's origin

    _sha256: str = None  # Cached SHA-256 hash of the item's file
    _embedding: np.ndarray = None  # Cached CLIP embedding of the item file

    def __new__(cls, file_path: Path | str = None, source_url: str = None, reference: str = None,
                id: int = None, **kwargs):
        """Checks if there already exists an instance of the item with the given reference.
        If yes, returns the existing reference. Otherwise, instantiates a new one."""
        if id is not None:
            # This is a re-instantiation of an existing item
            from ezmm.common.registry import item_registry
            item = item_registry.get_cached(reference=reference, kind=cls.kind, file_path=file_path)
            if item:
                item.source_url = source_url or item.source_url
                return item

        elif file_path or reference:
            # Look up an existing instance instead of creating a new one
            from ezmm.common.registry import item_registry
            item = item_registry.get_cached(reference=reference, kind=cls.kind, file_path=file_path)
            if item:
                item.source_url = source_url or item.source_url
                return item
            elif reference:
                raise ValueError(f"No item with reference '{reference}'.")

        return super().__new__(cls)

    def __init__(self, file_path: Path | str, source_url: str = None, reference: str = None, id: int = None):
        if hasattr(self, "id"):
            # This item is already instantiated, no init needed
            return

        self.file_path = Path(file_path).absolute()
        self.source_url = source_url or self.file_path.as_uri()
        if id is not None:
            self.id = id
        else:
            # This item is new, so save it to the registry and get an ID assigned
            from ezmm.common.registry import item_registry
            self.id = item_registry.add_item(self)
        self.validate_file_path()  # Make sure the file still exists

    @property
    def reference(self) -> str:
        return REF.format(kind=self.kind, id=self.id)

    @property
    def file_path_relative(self) -> Path:
        """Returns the path to this file relative from the registry's root.
        Relocates the file into the registry if it is not located there already."""
        from ezmm.common.registry import item_registry
        try:
            return self.file_path.relative_to(item_registry.path)
        except ValueError:
            self.relocate()
            return self.file_path.relative_to(item_registry.path)

    def validate_file_path(self):
        """Checks if the file of this item still exists at the specified path."""
        if not self.file_path.exists():
            logger.warning(f"File '{self.file_path}' referenced by item '{self.reference}' does not exist anymore. "
                           f"Healing it by using the default file path.")
            self.file_path = self._default_file_path()
            if self.file_path.exists():
                from ezmm.common.registry import item_registry
                item_registry.update_file_path(self)
                logger.info(f"File path successfully healed.")
            else:
                raise FileNotFoundError(f"File of item '{self.reference}' does not exist anymore.")

    @staticmethod
    def from_reference(reference: str) -> Optional["Item"]:
        from ezmm.common.registry import item_registry
        return item_registry.get(reference)

    @classmethod
    def from_id(cls, id: int) -> Optional["Item"]:
        reference = REF.format(kind=cls.kind, id=id)
        return cls.from_reference(reference)

    def as_html(self) -> str:
        """Returns the item as HTML code. File paths will be relative to the registry's root."""
        return f"<p>Item {self.reference} does not support HTML yet.</p>"

    def relocate(self, move_not_copy=False):
        """Copies the item's file into the media registry (if not
        located there already). Moves it instead if move_not_copy=True."""
        from ezmm.common.registry import item_registry
        new_path = self._default_file_path()

        if self.file_path != new_path:
            # Ensure the target directory exists
            new_path.parent.mkdir(parents=True, exist_ok=True)

            # Move/copy file to target directory
            move(self.file_path, new_path) \
                if move_not_copy \
                else copyfile(self.file_path, new_path)

            # Update the file path to the new location
            self.file_path = new_path
            item_registry.update_file_path(self)

    @property
    def size(self) -> int:
        """Returns the size of the item in bytes."""
        return self.file_path.stat().st_size

    @property
    def sha256(self) -> str:
        """Returns the SHA-256 hash of the item's file."""
        # TODO: Save hash in registry and keep only unique hashes to reuse existing images
        if self._sha256:
            return self._sha256
        import hashlib
        h = hashlib.sha256()
        # Read the file in chunks of 8192 bytes to reduce peak memory usage
        with self.file_path.open("rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        self._sha256 = h.hexdigest()
        return self._sha256

    @property
    def embedding(self) -> np.ndarray:
        """Returns the CLIP embedding of the item.
        TODO: Make this persistent in the item registry."""
        if self._embedding is not None:
            return self._embedding
        self._embedding = self._compute_embedding()
        return self._embedding

    def _compute_embedding(self) -> np.ndarray:
        """Computes the CLIP embedding of the item."""
        raise NotImplementedError

    def cos_sim(self, other: "Item") -> float:
        """Computes the cosine similarity between this item and another."""
        return (np.dot(self.embedding, other.embedding) /
                (np.linalg.norm(self.embedding) * np.linalg.norm(other.embedding)))

    def _temp_file_path(self, suffix: str = "") -> Path:
        """Returns a path that can be used for temporary storage.
        Use it when the item's ID is not set yet."""
        from ezmm.common.registry import item_registry
        filename = datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f") + suffix
        return (item_registry.path / "items" / filename).absolute()

    def _default_file_path(self) -> Path:
        """Returns the absolute, canonical file path for this item located in the registry,
        based on its kind and ID."""
        from ezmm.common.registry import item_registry
        return (item_registry.path / self.kind / f"{self.id}{self.file_path.suffix}").absolute()

    def __eq__(self, other):
        return (self is other or
                isinstance(other, Item) and (
                        self.kind == other.kind and self.id == other.id or  # Should never trigger
                        self.sha256 == other.sha256
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
            if item is None:
                raise ValueError(f"Item with reference {substr} does not exist.")
            split[i] = item
    return split
