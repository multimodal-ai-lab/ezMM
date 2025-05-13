import sqlite3
from pathlib import Path
from typing import Optional

from ezmm.common.items import Item, ITEM_CLASSES, KIND2ITEM
from ezmm.config import temp_dir
from ezmm.util import parse_ref, normalize_path


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

    def get(self, reference: str = None, kind: str = None, identifier: int = None) -> Optional[Item]:
        """Gets the referenced item object by loading it from the cache or,
        if not in the cache, from the disk."""
        if kind is None or identifier is None:
            assert reference
            kind, identifier = parse_ref(reference)

        # Read from cache
        item = self._get_cached(kind, identifier)

        if item is None:
            # Initialize new item object
            item = self._initialize_item(kind, identifier)

        return item

    def get_by_path(self, kind: str, path: Path | str) -> Optional[Item]:
        """Returns the item object located at the path ONLY IF it is
        already registered in the registry."""
        identifier = self._get_id_by_path(kind, path)
        if identifier is not None:
            return self.get(kind=kind, identifier=identifier)

    def add_and_assign_id(self, item: Item):
        """Adds a (partially initialized) item to the registry, if not yet registered.
        Assigns a proper ID to the item and adds it to the cache."""
        if not self.contains(item.kind, item.file_path):
            item.id = self._insert_into_registry(item.file_path, item.kind)
        else:
            item.id = self._get_id_by_path(item.kind, item.file_path)
        self._add_to_cache(item)

    def get_cached(self, reference: str = None, kind: str = None, file_path: Path | str= None) -> Optional[Item]:
        if reference:
            kind, identifier = parse_ref(reference)
        else:
            assert kind is not None
            identifier = self._get_id_by_path(kind, file_path)
        return self._get_cached(kind, identifier)

    def _initialize_item(self, kind: str, identifier: int) -> Optional[Item]:
        """Initializes the specified item if it is known to the registry."""
        item_path = self._get_path_by_id(kind, identifier)
        if item_path is not None and item_path.exists():
            item_cls = KIND2ITEM[kind]
            return item_cls(file_path=item_path)

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

    def update_file_path(self, item: Item):
        """Updates the path for the corresponding item in the registry."""
        stmt = f"""
            UPDATE {item.kind}
            SET path = ?
            WHERE id = ?;
        """
        self.cur.execute(stmt, (normalize_path(item.file_path), item.id))
        self.conn.commit()

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
