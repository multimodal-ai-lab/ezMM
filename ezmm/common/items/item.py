import re
from abc import ABC
from pathlib import Path
from typing import Sequence

from ezmm.util import is_item_ref


REF = "<{kind}:{id}>"  # General reference template, defining the reference syntax


class Item(ABC):
    """An element of MultimodalSequences. The data of each item is saved in an individual file."""
    kind: str  # Specifies the type of the item (image, video, ...)
    id: int  # Unique identifier of this item within its kind
    file_path: Path  # The path to the file where the data of this item is stored

    def __new__(cls, file_path: Path | str = None, reference: str = None, **kwargs):
        """Checks if there already exists an instance of the item with the given reference.
        If yes, returns the existing reference. Otherwise, instantiates a new one."""

        if file_path or reference:
            # Look up an existing instance instead of creating a new one
            from ezmm.common.registry import item_registry
            item = item_registry.get_cached(reference=reference, kind=cls.kind, file_path=file_path)
            if item:
                return item
            elif reference:
                raise ValueError(f"No item with reference '{reference}'.")

        return super().__new__(cls)

    def __init__(self, file_path: Path | str, reference: str = None):
        if hasattr(self, "id"):
            # The item is already initialized (existing instance returned via __new__())
            return
        self.file_path = Path(file_path)
        from ezmm.common.registry import item_registry
        item_registry.add_and_assign_id(self)  # Ensure the item is registered and get an ID assigned

    @property
    def reference(self):
        return REF.format(kind=self.kind, id=self.id)

    def _same(self, other):
        """Compares the content data with the other item for equality."""
        raise NotImplementedError

    def __eq__(self, other):
        return (self is other or
                isinstance(other, Item) and (
                        self.kind == other.kind and self.id == other.id or  # Should never trigger
                        self._same(other)
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
            if item is not None:
                split[i] = item
    return split
