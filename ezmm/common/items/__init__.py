from ezmm.common.items.item import Item
from ezmm.common.items.image import Image
from ezmm.common.items.video import Video
from ezmm.common.items.audio import Audio

ITEM_CLASSES = [Image, Video, Audio]
KIND2ITEM = {item.kind: item for item in ITEM_CLASSES}
KINDS = [item.kind for item in ITEM_CLASSES]

# Regex patterns
KINDS_ALTERNATIVES = "|".join(KINDS)
ITEM_REF_REGEX = rf"(<(?:{KINDS_ALTERNATIVES}):[0-9]+>)"
ITEM_ID_REGEX = rf"(?:<(?:{KINDS_ALTERNATIVES}):([0-9]+)>)"
ITEM_KIND_REGEX = rf"(?:<({KINDS_ALTERNATIVES}):([0-9]+)>)"
