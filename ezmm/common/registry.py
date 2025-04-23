import sqlite3
from pathlib import Path
from typing import Optional

from ezmm.config import temp_dir
from ezmm.util import parse_ref, normalize_path
from ezmm.common.items import Item, ITEM_CLASSES, KIND2ITEM


class ItemRegistry:
    """Keeps track of all the occurring items efficiently.
    Also holds a cache of already loaded items for efficiency."""
    db_path = Path(temp_dir) / "item_registry.db"

    def __init__(self):
        # Initialize folder, DB, and cache
        if not self.db_path.parent.exists():
            self.db_path.parent.mkdir(exist_ok=True, parents=True)
        is_new = not self.db_path.exists()
        self.conn = sqlite3.connect(self.db_path, timeout=10, check_same_thread=False)
        self.conn.execute("PRAGMA journal_mode=WAL;")
        self.cur = self.conn.cursor()
        if is_new:
            self._init_db()
        self.cache: dict[tuple[str, int], Item] = dict()

    def _init_db(self):
        """Initializes a clean, new DB."""
        for item_cls in ITEM_CLASSES:
            kind = item_cls.kind
            stmt = f"""
                CREATE TABLE {kind}(id INTEGER PRIMARY KEY, path TEXT);
            """
            self.cur.execute(stmt)
            stmt = f"""
                CREATE UNIQUE INDEX {kind}_path_idx ON {kind}(path);
            """
            self.cur.execute(stmt)
        self.conn.commit()

    def get(self, reference: str) -> Optional[Item]:
        """Gets the referenced item object by loading it from the cache or,
        if not in the cache, from the disk."""
        kind, identifier = parse_ref(reference)

        # Read from cache
        item = self._get_cached(kind, identifier)

        if item is None:
            # Initialize new item object
            item = self._initialize_item(kind, identifier)

        return item

    def add_and_assign_id(self, item: Item):
        """Adds a (partially initialized) item to the registry, if not yet registered.
        Assigns an ID to the item in that case."""
        if not self.contains(item.kind, item.file_path):
            item.id = self._insert_into_registry(item.file_path, item.kind)
            self._add_to_cache(item)
        else:
            item.id = self._get_id_by_path(item.kind, item.file_path)

    def _initialize_item(self, kind: str, identifier: int) -> Optional[Item]:
        """Initializes the specified item if it is known to the registry."""
        item_path = self._get_path_by_id(kind, identifier)
        if item_path is not None and item_path.exists():
            item_cls = KIND2ITEM[kind]
            return item_cls(item_path=item_path)

    def _get_id_by_path(self, kind: str, item_path: Path) -> Optional[int]:
        stmt = f"""
            SELECT id
            FROM {kind}
            WHERE path = ?;
        """
        response = self.cur.execute(stmt, (normalize_path(item_path),))
        result = response.fetchone()
        if result is not None:
            return result[0]
        else:
            return None

    def _get_path_by_id(self, kind: str, identifier: int) -> Optional[Path]:
        stmt = f"""
            SELECT path
            FROM {kind}
            WHERE id = ?;
        """
        response = self.cur.execute(stmt, (identifier,))
        result = response.fetchone()
        if result is not None:
            return Path(result[0])
        else:
            return None

    def _insert_into_registry(self, item_path: Path, kind: str) -> int:
        """Adds the new item directly to the database and returns its assigned ID."""
        stmt = f"""
            INSERT INTO {kind}(path)
            VALUES (?);
        """
        self.cur.execute(stmt, (normalize_path(item_path),))
        self.conn.commit()
        return self._get_id_by_path(kind, item_path)

    def contains(self, kind: str, item_path: Path | str) -> bool:
        return self._get_id_by_path(kind, item_path) is not None

    def _get_cached(self, kind: str, identifier: int) -> Optional[Item]:
        """Tries to retrieve the specified item from the cache. Returns
        None if it is not in the cache."""
        return self.cache.get((kind, identifier))

    def _add_to_cache(self, item: Item) -> None:
        """Adds the given item to the cache."""
        self.cache[(item.kind, item.id)] = item


item_registry = ItemRegistry()
