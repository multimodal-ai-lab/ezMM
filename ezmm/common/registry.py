import logging
import sqlite3
from pathlib import Path
from typing import Optional

from ezmm.common.items import Item, ITEM_CLASSES, KIND2ITEM
from ezmm.util import parse_ref, normalize_path

logger = logging.getLogger("ezMM")


class ItemRegistry:
    """Keeps track of all the occurring items efficiently.
    Also holds a cache of already loaded items for efficiency."""
    path: Path  # Directory where the registry is stored
    _db_path: Path  # Path to the SQLite DB file

    conn: Optional[sqlite3.Connection] = None
    cur: Optional[sqlite3.Cursor] = None
    cache: dict[tuple[str, int], Item] = dict()

    def __init__(self, path: Path | str = "temp/"):
        self.set_path(path)

    def set_path(self, path: Path | str):
        if self.conn:
            raise RuntimeError("Cannot change path for an established ezMM Item Registry.")
        self.path = Path(path)
        self._db_path = self.path / "item_registry.db"

    def _ensure_connected(self):
        if self.conn is None:
            self.connect()

    def connect(self):
        # Initialize folder, DB, and cache
        logger.info(f"Connecting to item registry at {self.path.as_posix()}...")
        if not self.path.exists():
            self.path.mkdir(exist_ok=True, parents=True)
        self.conn = sqlite3.connect(self._db_path, timeout=10, check_same_thread=False)
        self.conn.execute("PRAGMA journal_mode=WAL;")
        self.cur = self.conn.cursor()
        self._init_db()
        logger.debug(f"Successfully connected to item registry.")

    def _init_db(self):
        """Initializes a clean, new DB."""
        for item_cls in ITEM_CLASSES:
            kind = item_cls.kind
            stmt = f"""
                CREATE TABLE IF NOT EXISTS  {kind} (
                    id INTEGER PRIMARY KEY,
                    path TEXT NOT NULL UNIQUE,
                    source_url TEXT NOT NULL
                );
            """
            self.cur.execute(stmt)
            stmt = f"""
                CREATE UNIQUE INDEX IF NOT EXISTS {kind}_path_idx ON {kind}(path);
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
            # Initialize new item object from DB
            item = self._get_item_by_id(kind, identifier)

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
            item.id = self._insert_into_registry(item, item.kind)
        else:
            item.id = self._get_id_by_path(item.kind, item.file_path)
        self._add_to_cache(item)

    def get_cached(self, reference: str = None, kind: str = None, file_path: Path | str = None) -> Optional[Item]:
        if reference:
            kind, identifier = parse_ref(reference)
        else:
            assert kind is not None
            identifier = self._get_id_by_path(kind, file_path)
        return self._get_cached(kind, identifier)

    def _get_id_by_path(self, kind: str, item_path: Path) -> Optional[int]:
        self._ensure_connected()
        item_path = normalize_path(item_path)
        stmt = f"""
            SELECT id
            FROM {kind}
            WHERE path = ?;
        """
        response = self.cur.execute(stmt, (item_path.as_posix(),))
        result = response.fetchone()
        if result is not None:
            return result[0]
        else:
            return None

    def _get_item_by_id(self, kind: str, identifier: int) -> Optional[Item]:
        self._ensure_connected()
        stmt = f"""
            SELECT path, source_url
            FROM {kind}
            WHERE id = ?;
        """
        response = self.cur.execute(stmt, (identifier,))
        if result := response.fetchone():
            item_cls = KIND2ITEM[kind]
            return item_cls(file_path=Path(result[0]),
                            source_url=result[1])

    def _insert_into_registry(self, item: Item, kind: str) -> int:
        """Adds the new item directly to the database and returns its assigned ID."""
        self._ensure_connected()
        stmt = f"""
            INSERT INTO {kind}(path, source_url)
            VALUES (?, ?);
        """
        self.cur.execute(stmt, (item.file_path.as_posix(), item.source_url))
        self.conn.commit()

        stmt = """SELECT last_insert_rowid();"""
        response = self.cur.execute(stmt)
        return response.fetchone()[0]

    def update_file_path(self, item: Item):
        """Updates the path for the corresponding item in the registry."""
        self._ensure_connected()
        stmt = f"""
            UPDATE {item.kind}
            SET path = ?
            WHERE id = ?;
        """
        self.cur.execute(stmt, (item.file_path.as_posix(), item.id))
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

    def close(self):
        self.conn.close()
        self.conn = None

    def reset(self):
        if self.conn:
            self.close()
        self.cache = dict()
        self.connect()


item_registry = ItemRegistry()
