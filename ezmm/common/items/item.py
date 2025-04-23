import re
from abc import ABC
from pathlib import Path
from typing import Sequence

from ezmm.util import is_item_ref


class Item(ABC):
    """An element of MultimodalSequences. The data of each item is saved in an individual file."""
    kind: str  # Specifies the type of the item (image, video, ...)
    id: int  # Unique identifier of this item within its kind
    file_path: Path  # The path to the file where the data of this item is stored

    def __new__(cls, *args, reference: str = None, **kwargs):
        """Checks if there already exists an instance of the item with the given reference.
        If yes, returns the existing reference. Otherwise, instantiates a new one."""
        if reference:
            # Catch cases where the reference consists of only the ID
            if str(reference).isdigit():
                reference = f"<{cls.kind}:{reference}>"

            # Look up an existing instance instead of creating a new one
            from ezmm.common.registry import item_registry
            item = item_registry.get(reference)
            if not item:
                raise ValueError(f"No item with reference {reference}.")
            return item
        else:
            return super().__new__(cls)

    def __init__(self, file_path: Path | str, reference: str = None):
        if reference:
            # The item is already initialized (existing instance returned via __new__())
            return
        self.file_path = Path(file_path)
        from ezmm.common.registry import item_registry
        item_registry.add_and_assign_id(self)  # Ensure the item is registered and get an ID assigned

    @property
    def reference(self):
        return f"<{self.kind}:{self.id}>"

    def __eq__(self, other):
        return isinstance(other, Item) and self.kind == other.kind and self.id == other.id

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
    split = re.split(ITEM_REF_REGEX, string)
    # Replace each reference with its actual item object
    for i in range(len(split)):
        substr = split[i]
        if is_item_ref(substr):
            item = item_registry.get(substr)
            if item is not None:
                split[i] = item
    return split
