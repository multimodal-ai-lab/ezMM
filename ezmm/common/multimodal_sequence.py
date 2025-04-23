from typing import Sequence

from ezmm.common.items import Image, Audio, Video
from ezmm.common.items.item import Item, resolve_references_from_sequence


class MultimodalSequence:
    """A sequence of data of any kind. Can be serialized to a string where each
    non-verbalizable element is referenced in-place, e.g., `<image:0>` for image
    with ID 0."""
    data: list[str | Item]
    string: str

    def __init__(self, *args: str | Item | Sequence[str | Item]):
        data = args[0] if len(args) == 1 else list(args)
        if isinstance(data, (str, Item)):
            data = [data]
        self.data = resolve_references_from_sequence(data)

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
        return (f"MultimodalSequence(len={len(self.__str__())}, n_images={len(self.images)}, "
                f"n_videos={len(self.videos)}, n_audios={len(self.audios)})")

    def __iter__(self):
        return iter(self.data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]
