import sys

from ezmm.common.items import *
from ezmm.common.multimodal_sequence import MultimodalSequence
from ezmm.common.registry import ItemRegistry
import logging

APP_NAME = "ezMM"

# Set up logger
logger = logging.getLogger(APP_NAME)
logger.setLevel(logging.DEBUG)

# Only add handler if none exists (avoid duplicate logs on rerun)
if not logger.hasHandlers():
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter('[%(levelname)s]: %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
