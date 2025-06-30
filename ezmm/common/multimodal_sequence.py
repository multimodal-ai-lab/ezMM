from __future__ import annotations

from typing import Sequence

from ezmm.common.items import Image, Audio, Video
from ezmm.common.items.item import Item, resolve_references_from_sequence
from ezmm.ui.common import SEQ_PATH
from multiprocessing import Process


class MultimodalSequence:
    """A sequence of data of any kind. Can be serialized to a string where each
    non-verbalizable element is referenced in-place, e.g., `<image:0>` for image
    with ID 0."""
    data: list[str | Item]

    def __init__(self, *args: str | Item | MultimodalSequence | Sequence[str | Item | None] | None):
        data = args[0] if len(args) == 1 else list(args)
        if isinstance(data, (str, Item)):
            data = [data]
        elif isinstance(data, MultimodalSequence):
            data = data.data
        self.data = resolve_references_from_sequence(data) if data else []

    @property
    def images(self) -> list[Image]:
        return [item for item in self.data if isinstance(item, Image)]

    @property
    def videos(self) -> list[Video]:
        return [item for item in self.data if isinstance(item, Video)]

    @property
    def audios(self) -> list[Audio]:
        return [item for item in self.data if isinstance(item, Audio)]

    def has_images(self) -> bool:
        return len(self.images) > 0

    def has_videos(self) -> bool:
        return len(self.videos) > 0

    def has_audios(self) -> bool:
        return len(self.audios) > 0

    def to_list(self):
        return self.data

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
        viewable through a link in the browser."""
        # Create sequences directory if it doesn't exist
        SEQ_PATH.mkdir(parents=True, exist_ok=True)

        # Generate 8-char ID string for the sequence
        seq_id = str(abs(hash(self)))[:8]

        # Save to file with unique name
        file_path = SEQ_PATH / f"{seq_id}.md"
        file_path.write_text(str(self))

        # Start the server
        from ezmm.ui.main import run_server
        Process(target=run_server).start()

        print(f"You can view the sequence at http://localhost:8000/sequence/{seq_id}")
