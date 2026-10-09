import logging
import mimetypes
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
    _embedding: np.ndarray = None  # Cached (full) embedding of the item file

    def __new__(cls, file_path: Path | str = None, source_url: str = None, reference: str = None,
                id: int = None, **kwargs):
        """Checks if there already exists an instance of the item with the given reference.
        If yes, returns the existing reference. Otherwise, instantiates a new one."""
        from ezmm.common.registry import item_registry
        item = None
        if id is not None:
            # This is a re-instantiation of an existing item
            item = item_registry.get_cached(kind=cls.kind, identifier=id)

        elif reference:
            item = item_registry.get(reference)
            if item is None:
                raise ValueError(f"No item with reference '{reference}'.")

        elif file_path:
            # Look up an existing instance instead of creating a new one
            item = item_registry.get_cached(kind=cls.kind, file_path=file_path)

        if item:
            if source_url and id is None:
                item.source_url = source_url
                item_registry.add_source_url(item.kind, item.id, source_url)
            return item

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
            # This item is new, so save it to the registry and get an ID assigned. If an
            # identical file is registered already, the existing ID and file get adopted.
            self._validate_new_file()
            from ezmm.common.registry import item_registry
            self.id = item_registry.add_item(self)
        self.validate_file_path()  # Make sure the file still exists

    def _validate_new_file(self):
        """Hook to check a new file before it gets registered. Raises an error if invalid."""
        pass

    @property
    def reference(self) -> str:
        return REF.format(kind=self.kind, id=self.id)

    @property
    def source_urls(self) -> list[str]:
        """Returns all known source URLs of this item, ordered by the time they were added."""
        from ezmm.common.registry import item_registry
        return item_registry.get_source_urls(self.kind, self.id)

    @property
    def file_url(self) -> str:
        """The URL under which the ezMM web UI serves this item's file."""
        return f"/item/{self.kind}/{self.id}/file"

    @property
    def mime_type(self) -> str:
        """Returns the MIME type of the item's file, guessed from its file extension."""
        return mimetypes.guess_type(self.file_path.name)[0] or "application/octet-stream"

    @property
    def bytes(self) -> bytes:
        """Returns the raw content of the item's file."""
        return self.file_path.read_bytes()

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
        """Returns the SHA-256 hash of the item's file (stored in the registry)."""
        if self._sha256:
            return self._sha256
        from ezmm.common.registry import compute_sha256
        self._sha256 = compute_sha256(self.file_path)
        return self._sha256

    @property
    def embedding(self) -> np.ndarray:
        """Returns the (normalized) embedding of the item, computed with EmbeddingGemma 2, in the
        configured dimension (see `ezmm.embedding.set_embedding_dim()`). The full embeddings are
        stored in the item registry, so each item is embedded only once."""
        from ezmm.embedding import MODEL_NAME, get_embedding_dim, truncate
        if self._embedding is None:
            from ezmm.common.registry import item_registry
            embedding = item_registry.get_embedding(self.kind, self.id, MODEL_NAME)
            if embedding is None:
                embedding = self._compute_embedding()
                item_registry.set_embedding(self.kind, self.id, MODEL_NAME, embedding)
            self._embedding = embedding
        return truncate(self._embedding, get_embedding_dim())

    def _compute_embedding(self) -> np.ndarray:
        """Computes the full embedding of the item's file."""
        from ezmm.embedding import embed_file, MODEL_DIM
        return embed_file(self.file_path, self.kind, dim=MODEL_DIM)

    def cos_sim(self, other: "Item | np.ndarray") -> float:
        """Computes the cosine similarity between this item and another item,
        MultimodalSequence, or embedding vector."""
        from ezmm.embedding import cos_sim
        return cos_sim(self.embedding, getattr(other, "embedding", other))

    def _write_temp_file(self, data: bytes, suffix: str) -> Path:
        """Saves the binary data into a temporary file inside the registry and returns its path."""
        file_path = self._temp_file_path(suffix=suffix)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_bytes(data)
        return file_path

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
                        self.kind == other.kind and self.id == other.id or
                        self.sha256 == other.sha256
                ))

    def __hash__(self):
        return hash((self.kind, self.id))


def resolve_references_from_sequence(seq: Sequence[str | Item]) -> list[str | Item]:
    """Identifies all item references within the sequence and replaces them with
    an instance of the referenced item. Returns the (interleaved) list of
    strings and items."""
    processed = []
    for item in seq:
        if isinstance(item, str):
            # Resolve the string
            if item.strip(" "):  # Drop excess whitespaces
                resolved = resolve_references_from_string(item)
                processed.extend(resolved)
        else:
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
    return [el for el in split if el != ""]  # Drop empty strings, e.g., after a trailing reference
