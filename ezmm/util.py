import re
from pathlib import Path
from typing import Optional


def normalize_path(path: Path | str) -> Path:
    """Ensures the path is a Path object and relative to the current working directory."""
    path = Path(path).absolute()
    cwd = Path.cwd()
    return path.relative_to(cwd)


def get_item_refs(text: str) -> list[str]:
    """Extracts all item references from the text."""
    from ezmm.common.items import ITEM_REF_REGEX
    pattern = re.compile(ITEM_REF_REGEX, re.DOTALL)
    matches = pattern.findall(text)
    return matches


def parse_ref(ref: str) -> (str, int):
    result = parse_item_ref(ref)
    if result is None:
        raise ValueError(f"Invalid item reference: {ref}")
    return result


def parse_item_ref(reference: str) -> Optional[tuple[str, int]]:
    """Returns the first matching kind and identifier from the reference."""
    from ezmm.common.items import ITEM_KIND_ID_REGEX
    pattern = re.compile(ITEM_KIND_ID_REGEX, re.DOTALL)
    result = pattern.findall(reference)
    if len(result) > 0:
        match = result[0]
        return match[0], int(match[1])
    else:
        return None


def is_item_ref(string: str) -> bool:
    """Returns True iff the string represents an item reference."""
    from ezmm.common.items import ITEM_REF_REGEX
    pattern = re.compile(ITEM_REF_REGEX, re.DOTALL)
    return pattern.fullmatch(string) is not None


def validate_references(text: str) -> bool:
    """Verifies that each item reference can be resolved to a registered item."""
    from ezmm.common.registry import item_registry
    refs = get_item_refs(text)
    for ref in refs:
        if item_registry.get(ref) is None:
            return False
    return True
