from ezmm.common.items.item import Item, REF
from ezmm.common.items.image import Image
from ezmm.common.items.video import Video
from ezmm.common.items.audio import Audio

ITEM_CLASSES = [Image, Video, Audio]
KIND2ITEM = {item.kind: item for item in ITEM_CLASSES}
KINDS = [item.kind for item in ITEM_CLASSES]

# Regex patterns
KINDS_ALTERNATIVES = "|".join(KINDS)
ITEM_REF_REGEX = "(" + REF.format(kind=f"(?:{KINDS_ALTERNATIVES})", id="[0-9]+") + ")"  # Captures full ref
ITEM_ID_REGEX = REF.format(kind=f"(?:{KINDS_ALTERNATIVES})", id="([0-9]+)")  # Captures only ID
ITEM_KIND_REGEX = REF.format(kind=f"({KINDS_ALTERNATIVES})", id="[0-9]+")  # Captures only kind
ITEM_KIND_ID_REGEX = REF.format(kind=f"({KINDS_ALTERNATIVES})", id="([0-9]+)")  # Captures kind and ID
