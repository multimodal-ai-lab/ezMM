from __future__ import annotations

import os
from typing import Sequence, Iterable
from collections.abc import Sequence as SequenceABC

from markdown import markdown

from ezmm.common.items import Image, Audio, Video, File
from ezmm.common.items.item import Item, resolve_references_from_sequence


class MultimodalSequence:
    """A sequence of data of any kind. Can be serialized to a string where each
    non-verbalizable element is referenced in-place, e.g., `<image:0>` for image
    with ID 0. Nested sequences are always flattened, both at creation time and
    when manipulating the sequence (e.g., via `append()` or `insert()`).

    Note that the hash of a sequence changes when the sequence is manipulated."""
    data: list[str | Item]

    def __init__(self, *args: str | Item | MultimodalSequence |
                              Sequence[str | Item | MultimodalSequence | None] | None):
        self.data = _normalize(args)

    @property
    def images(self) -> list[Image]:
        return [item for item in self.data if isinstance(item, Image)]

    @property
    def videos(self) -> list[Video]:
        return [item for item in self.data if isinstance(item, Video)]

    @property
    def audios(self) -> list[Audio]:
        return [item for item in self.data if isinstance(item, Audio)]

    @property
    def files(self) -> list[File]:
        return [item for item in self.data if isinstance(item, File)]

    @property
    def text(self) -> str:
        """Returns all text parts of the sequence (i.e., without any items) as a single string."""
        return " ".join(el for el in self.data if isinstance(el, str))

    def has_images(self) -> bool:
        return len(self.images) > 0

    def has_videos(self) -> bool:
        return len(self.videos) > 0

    def has_audios(self) -> bool:
        return len(self.audios) > 0

    def has_files(self) -> bool:
        return len(self.files) > 0

    # Manipulation (all inputs get flattened and their references resolved)

    def append(self, *elements: str | Item | MultimodalSequence | Sequence | None) -> None:
        """Appends the element(s) to the end of the sequence."""
        self.data.extend(_normalize(elements))

    def extend(self, elements: Iterable[str | Item | MultimodalSequence | Sequence | None]) -> None:
        """Appends all elements of the iterable to the end of the sequence."""
        self.data.extend(_normalize(list(elements)))

    def insert(self, index: int, element: str | Item | MultimodalSequence | Sequence | None) -> None:
        """Inserts the element(s) before the given index."""
        if index < 0:
            index = max(len(self.data) + index, 0)
        self.data[index:index] = _normalize([element])

    def remove(self, element: str | Item) -> None:
        """Removes the first occurrence of the element. Raises ValueError if not present."""
        self.data.remove(element)

    def pop(self, index: int = -1) -> str | Item:
        """Removes and returns the element at the given index (default: last)."""
        return self.data.pop(index)

    def clear(self) -> None:
        self.data.clear()

    def index(self, element: str | Item) -> int:
        return self.data.index(element)

    def count(self, element: str | Item) -> int:
        return self.data.count(element)

    def copy(self) -> MultimodalSequence:
        return MultimodalSequence(self)

    def __setitem__(self, index: int | slice, value):
        """Replaces the element(s) at the index. The value may be any (nested)
        element(s), which get flattened into the sequence."""
        if isinstance(index, slice):
            self.data[index] = _normalize([value])
        else:
            index = range(len(self.data))[index]  # Resolves negative indices, raises IndexError
            self.data[index:index + 1] = _normalize([value])

    def __delitem__(self, index: int | slice):
        del self.data[index]

    def __add__(self, other) -> MultimodalSequence:
        return MultimodalSequence(self, other)

    def __radd__(self, other) -> MultimodalSequence:
        return MultimodalSequence(other, self)

    def __iadd__(self, other) -> MultimodalSequence:
        self.append(other)
        return self

    def to_list(self):
        return self.data

    def as_html(self) -> str:
        """Returns the sequence as HTML code. Any files of contained items are
        referenced by paths relative to the registry's root."""
        htmls = []
        for item in self:
            if isinstance(item, Item):
                html = f'<div class="media-container">{item.as_html()}</div>'
                htmls.append(html)
            else:
                htmls.append(markdown(item))
        return " ".join(htmls)

    def unique_items(self) -> set[Item]:
        """Returns the set of all items (not strings) occurring in the sequence."""
        return set([item for item in self if isinstance(item, Item)])

    def __str__(self):
        """Turns itself into a single string where each item is replaced by its reference."""
        substrings = []
        for item in self:
            if isinstance(item, Item):
                substrings.append(item.reference)
            else:
                substrings.append(item)
        return " ".join(substrings)

    def __repr__(self):
        return f"MultimodalSequence(str_len={len(self.__str__())}, n_items={len(self.data)})"

    def __iter__(self):
        return iter(self.data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

    def __eq__(self, other):
        return isinstance(other, MultimodalSequence) and self.data == other.data

    def __hash__(self):
        return hash(str(self))

    def __bool__(self):
        return len(self) > 0

    def render(self):
        """Saves the given MultimodalSequence to a .md file in SEQ_PATH to make it
        viewable through a link in the browser. Starts a server (if not started yet)
        that serves the UI. If `blocking` is False, the server is started in a separate
        process."""
        from ezmm.ui.common import get_seq_path
        SEQ_PATH = get_seq_path()

        # Create sequences directory if it doesn't exist
        SEQ_PATH.mkdir(parents=True, exist_ok=True)

        # Generate 8-char ID string for the sequence
        seq_id = str(abs(hash(self)))[:8]

        # Save to file with unique name
        file_path = SEQ_PATH / f"{seq_id}.md"
        file_path.write_text(str(self), encoding="utf-8")

        from ezmm.common import item_registry
        os.environ["EZMM"] = item_registry.path.as_posix()

        # Start the server
        from ezmm.ui.main import run_server
        print(f"You can view the sequence at http://localhost:7878/sequence/{seq_id}")
        run_server()


def _normalize(data: Sequence[None | str | Item | MultimodalSequence | Sequence]) -> list[str | Item]:
    """Turns the (potentially nested) data into a flat list of strings and items
    where all references are resolved."""
    flattened = _flatten(data)
    return resolve_references_from_sequence(flattened) if flattened else []


def _flatten(
        data: Sequence[None | str | Item | MultimodalSequence | Sequence]
) -> list[str | Item]:
    """Recursively turns a potentially nested sequence into a flat list of strings and items."""
    flattened = []
    for el in data:
        match el:
            case None:
                continue
            case str():
                if el:  # Skip empty strings
                    flattened.append(el)
            case Item():
                flattened.append(el)
            case MultimodalSequence():
                flattened.extend(el.data)
            case SequenceABC():  # Must come after str() case
                flattened.extend(_flatten(el))
            case _:
                raise TypeError(f"Unsupported type: {type(el)}")
    return flattened


if __name__ == "__main__":
    seq = MultimodalSequence(f"Hello world! Here is an image", Image('in/garden.jpg'),
                             "and another image", Image('in/roses.jpg'),
                             "and a video", Video('in/snow.mp4'))
    print(seq)
    seq.render()
