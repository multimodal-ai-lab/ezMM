from pathlib import Path

SEQ_PATH = Path("temp/sequences")  # Deprecated: sequences are now stored inside the registry, see get_seq_path()


def get_seq_path() -> Path:
    """Returns the directory where rendered MultimodalSequences are stored."""
    from ezmm.common.registry import item_registry
    return item_registry.path / "sequences"
