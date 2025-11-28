import logging
import os
import sys
from pathlib import Path

from ezmm.common.items import *
from ezmm.common.multimodal_sequence import MultimodalSequence
from ezmm.common.registry import ItemRegistry, item_registry

APP_NAME = "ezMM"
PROJECT_ROOT = Path(__file__).parent.parent.parent

# Set up logger
logger = logging.getLogger(APP_NAME)
logger.setLevel(logging.DEBUG)

# Only add handler if none exists (avoid duplicate logs on rerun)
if not logger.hasHandlers():
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter('[%(levelname)s]: %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)


def set_ezmm_path(path: Path | str):
    logger.info(f"Setting ezMM path to {path}")
    os.environ["EZMM"] = Path(path).as_posix()
    item_registry.set_path(path)


def reset_ezmm():
    item_registry.reset()
