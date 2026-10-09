import hashlib
import logging
import os
import sqlite3
import threading
import weakref
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional, Callable, Iterable, TypeVar

import numpy as np

from ezmm.common.items import Item, KIND2ITEM
from ezmm.common.vector_index import VectorIndex
from ezmm.util import parse_ref

logger = logging.getLogger("ezMM")

SCHEMA_VERSION = 3  # Stored in the DB via PRAGMA user_version (legacy per-kind schema = 0)

# Number of threads used to read/hash files in bulk operations (migration, deduplication, file checks)
N_WORKERS = min(32, (os.cpu_count() or 1) + 4)

BUSY_TIMEOUT = 60  # Seconds a write waits for other writers (threads or processes) to finish
ACCESS_UPDATE_INTERVAL = timedelta(hours=1)  # Sources' last access times are updated at most this often

SCHEMA = """
    CREATE TABLE IF NOT EXISTS items (
        row_id INTEGER PRIMARY KEY,
        kind TEXT NOT NULL,
        id INTEGER NOT NULL,
        path TEXT,                -- Relative to the registry root if inside it, else absolute (NULL for aliases)
        sha256 TEXT,              -- Hash of the raw file bytes, used for deduplication
        size INTEGER,             -- File size in bytes
        canonical_id INTEGER,     -- If set, this row is an alias (removed duplicate) of item (kind, canonical_id)
        missing INTEGER NOT NULL DEFAULT 0,  -- 1 if the file was found missing (a shortcut for browsing only)
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        UNIQUE (kind, id)
    );
    CREATE INDEX IF NOT EXISTS items_sha256_idx ON items(kind, sha256);
    CREATE INDEX IF NOT EXISTS items_path_idx ON items(kind, path);
    CREATE INDEX IF NOT EXISTS items_created_idx ON items(created_at);

    -- Each source URL is stored exactly once and points to the item it delivered
    CREATE TABLE IF NOT EXISTS sources (
        id INTEGER PRIMARY KEY,
        url TEXT NOT NULL UNIQUE,
        item_row_id INTEGER NOT NULL REFERENCES items(row_id),
        created_at TEXT NOT NULL,
        last_accessed TEXT NOT NULL  -- Last time an item was loaded from or looked up via this URL
    );
    CREATE INDEX IF NOT EXISTS sources_item_idx ON sources(item_row_id);
"""

# Embedding vectors (normalized float32) of items, one per item and embedding model. An optional
# addition to schema v3 (created on connect), so registries remain readable by older ezMM versions.
EMBEDDINGS_SCHEMA = """
    CREATE TABLE IF NOT EXISTS embeddings (
        item_row_id INTEGER NOT NULL REFERENCES items(row_id),
        model TEXT NOT NULL,
        vector BLOB NOT NULL,
        dtype TEXT NOT NULL DEFAULT 'float32',  -- Data type of the vector's values
        created_at TEXT NOT NULL,
        PRIMARY KEY (item_row_id, model)
    );
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def compute_sha256(path: Path) -> str:
    """Returns the SHA-256 hash of the file's raw bytes."""
    with open(path, "rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


T = TypeVar("T")
R = TypeVar("R")


def _parallel_map(fn: Callable[[T], R], inputs: Iterable[T], label: str = None) -> list[R]:
    """Applies the (I/O-bound) function to all inputs using a thread pool, preserving the order.
    Hashing releases the GIL, so file reading and hashing run truly in parallel."""
    inputs = list(inputs)
    if len(inputs) <= 1:
        return [fn(x) for x in inputs]
    results = []
    with ThreadPoolExecutor(max_workers=N_WORKERS) as pool:
        for i, result in enumerate(pool.map(fn, inputs), start=1):
            results.append(result)
            if label and i % 1000 == 0:
                logger.info(f"{label}: {i}/{len(inputs)} files processed...")
    return results


def _hash_file(path: Optional[Path]) -> Optional[tuple[str, int]]:
    """Returns the SHA-256 hash and size of the file, or None if it cannot be read."""
    try:
        return compute_sha256(path), path.stat().st_size
    except (OSError, TypeError):
        return None


class _Connection(sqlite3.Connection):
    """SQLite connection that supports weak references (for tracking the threads' connections)."""


def _access_threshold() -> str:
    """Returns the time before which a source's last access time gets updated on access."""
    return (datetime.now(timezone.utc) - ACCESS_UPDATE_INTERVAL).isoformat(timespec="seconds")


def _decode_vector(blob: bytes, dtype: str) -> np.ndarray:
    return np.frombuffer(blob, dtype=dtype).astype(np.float32)


def _exists(path: Optional[Path]) -> bool:
    return path is not None and path.exists()


class ItemRegistry:
    """Keeps track of all the occurring items efficiently.
    Also holds a cache of already loaded items for efficiency."""
    path: Path  # Absolute path to the root directory of the registry
    _db_path: Path  # Path to the SQLite DB file

    cache: dict[tuple[str, int], Item] = dict()

    def __init__(self, path: Path | str = None):
        # Each thread uses its own DB connection, so reads run in parallel and SQLite
        # serializes the (short) write transactions, also across processes
        self._local = threading.local()
        self._connections: weakref.WeakSet[_Connection] = weakref.WeakSet()  # Open connections of all threads
        self._generation = 0  # Incremented on close() to invalidate the threads' connections
        self._initialized = False
        self._conn_lock = threading.RLock()  # Guards connecting and closing only
        self._embedding_index: dict[tuple[str, int, str], VectorIndex] = dict()  # (model, dim, device) -> index
        self._index_lock = threading.Lock()  # Guards the creation of indices only
        if path is None:
            path = os.getenv("EZMM")
            if path:
                logger.info(f"Found OS environment variable EZMM={path}. "
                            f"Using it as the root of the ezMM item registry.")
            else:
                path = "temp/"
        self.set_path(path)

    def set_path(self, path: Path | str):
        path = Path(path)
        if not hasattr(self, "path") or path.absolute() != self.path:
            if self._initialized:
                raise RuntimeError("Cannot change path for an established ezMM Item Registry.")
            self.path = path.absolute()
            self._db_path = self.path / "item_registry.db"
            self._embedding_index.clear()

    @property
    def conn(self) -> sqlite3.Connection:
        """The current thread's connection to the DB (opened on first use)."""
        local = self._local
        if getattr(local, "generation", None) != self._generation:
            if not self._initialized:
                self.connect()
            if getattr(local, "generation", None) != self._generation:
                self._open_connection()
        return local.conn

    def _open_connection(self) -> sqlite3.Connection:
        """Opens a connection for the current thread. It gets closed when the thread ends."""
        with self._conn_lock:
            # Autocommit mode: transactions are managed explicitly (see _transaction)
            conn = sqlite3.connect(self._db_path, timeout=BUSY_TIMEOUT, check_same_thread=False,
                                   isolation_level=None, factory=_Connection)
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("PRAGMA synchronous=NORMAL;")  # No fsync per commit; safe from corruption in WAL mode
            self._connections.add(conn)
            self._local.conn, self._local.generation = conn, self._generation
            return conn

    def _ensure_connected(self):
        if not self._initialized:
            self.connect()

    def connect(self):
        """Initializes the registry (folder and DB, incl. migration if needed)."""
        with self._conn_lock:
            if self._initialized:
                return
            logger.info(f"Connecting to item registry at {self.path.as_posix()}")
            self.path.mkdir(exist_ok=True, parents=True)
            self._open_connection()
            self._init_db()
            self._initialized = True
            logger.debug(f"Successfully connected to item registry.")

    @contextmanager
    def _transaction(self, mode: str = "IMMEDIATE"):
        """Runs the enclosed statements in one (write) transaction. Keep transactions short
        and free of slow work (like file I/O), as they block all other writers."""
        conn = self.conn
        conn.execute(f"BEGIN {mode};")
        try:
            yield conn
            conn.execute("COMMIT;")
        except BaseException:
            conn.execute("ROLLBACK;")
            raise

    def _execute(self, stmt: str, params: tuple = ()) -> list[tuple]:
        return self.conn.execute(stmt, params).fetchall()

    # ---------------------------------------------------------------------------------------------
    # Schema and migration

    def _init_db(self):
        """Creates the schema of a new DB or migrates a legacy DB to the current schema."""
        with self._conn_lock:
            version = self.conn.execute("PRAGMA user_version;").fetchone()[0]
            if version > SCHEMA_VERSION:
                raise RuntimeError(f"The ezMM registry at {self.path.as_posix()} uses schema version {version}, "
                                   f"but this ezMM version supports only up to {SCHEMA_VERSION}. Please upgrade ezMM.")
            if version < SCHEMA_VERSION:
                self.migrate()
            self.conn.execute(EMBEDDINGS_SCHEMA)
            columns = [row[1] for row in self.conn.execute("PRAGMA table_info(embeddings);").fetchall()]
            if "dtype" not in columns:  # Added in ezMM 0.7.0 during development
                self.conn.execute("ALTER TABLE embeddings ADD COLUMN dtype TEXT NOT NULL DEFAULT 'float32';")

    def _create_schema(self):
        """Creates all tables and indices (if not existing) and sets the schema version."""
        for stmt in SCHEMA.split(";"):
            if stmt.strip():
                self.conn.execute(stmt)
        self.conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION};")

    def _legacy_tables(self) -> list[str]:
        """Returns the names of the per-kind tables of the legacy (v1) schema."""
        rows = self.conn.execute("SELECT name FROM sqlite_master WHERE type = 'table';").fetchall()
        return [name for (name,) in rows
                if name not in ("items", "sources", "embeddings") and not name.startswith("sqlite_")]

    def migrate(self):
        """Migrates the DB to the current schema (creates the schema for a new DB). Item IDs
        are preserved so that existing references remain valid. A backup of the old DB is
        written next to it. Does nothing if the DB is already up to date."""
        with self._conn_lock:
            version = self.conn.execute("PRAGMA user_version;").fetchone()[0]
            if version >= SCHEMA_VERSION:
                return
            if version == 0 and not self._legacy_tables():
                with self._transaction("EXCLUSIVE"):
                    self._create_schema()  # New DB
            elif version == 0:
                self._migrate_from_v1()
            elif version == 2:
                self._migrate_from_v2()
            else:
                raise RuntimeError(f"Cannot migrate the ezMM registry from unknown schema version {version}.")

    def _backup(self, version: int):
        backup_path = self.path / f"item_registry.v{version}.bak.db"
        if backup_path.exists():
            backup_path = self.path / f"item_registry.v{version}.bak.{datetime.now():%Y%m%d-%H%M%S}.db"
        backup = sqlite3.connect(backup_path)
        self.conn.backup(backup)
        backup.close()
        logger.info(f"Backup of the registry written to {backup_path.as_posix()}")

    def _migrate_from_v1(self):
        """Migrates a legacy DB (one table per item kind) to the unified schema."""
        legacy_tables = self._legacy_tables()
        logger.info(f"Migrating legacy ezMM registry at {self.path.as_posix()} to schema v{SCHEMA_VERSION}...")
        self._backup(1)

        # Read, heal, and hash everything first (slow part, outside the write transaction)
        rows = []
        for kind in legacy_tables:
            for identifier, path, source_url in self.conn.execute(
                    f"SELECT id, path, source_url FROM {kind} ORDER BY id;").fetchall():
                rows.append((kind, identifier, Path(path), source_url))

        def heal(row) -> Path:
            kind, identifier, path, _ = row
            if path.exists():
                return path
            # Heal the path, e.g., if the registry was moved: try the default location
            # inside the registry and the file's original folder inside the registry
            candidates = [self.path / kind / f"{identifier}{path.suffix}",
                          self.path / path.parent.name / path.name]
            healed = next((c for c in candidates if c.exists()), None)
            if healed is None:
                logger.warning(f"File of <{kind}:{identifier}> not found at '{path.as_posix()}'.")
            return healed or path

        paths = _parallel_map(heal, rows)
        hashes = _parallel_map(_hash_file, paths, label="Migration")

        now = _now()
        with self._transaction("EXCLUSIVE"):
            if self.conn.execute("PRAGMA user_version;").fetchone()[0] >= SCHEMA_VERSION:
                return  # Another process migrated the DB in the meantime
            self._create_schema()
            for (kind, identifier, _, source_url), path, hashed in zip(rows, paths, hashes):
                sha256, size = hashed or (None, None)
                row_id = self.conn.execute("""
                    INSERT INTO items(kind, id, path, sha256, size, canonical_id, missing, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, NULL, ?, ?, ?);
                """, (kind, identifier, self._to_db_path(path), sha256, size, int(hashed is None), now, now)).lastrowid
                self._link_source(row_id, source_url)
            for kind in legacy_tables:
                self.conn.execute(f"DROP TABLE {kind};")
        logger.info(f"Migration of {len(rows)} items completed.")

    def _migrate_from_v2(self):
        """Adds the `missing` column (schema v3) and determines its values."""
        logger.info(f"Migrating ezMM registry at {self.path.as_posix()} to schema v{SCHEMA_VERSION}...")
        self._backup(2)
        rows = self.conn.execute("SELECT row_id, path FROM items WHERE canonical_id IS NULL;").fetchall()
        exists = _parallel_map(_exists, [self._from_db_path(path) for _, path in rows])
        with self._transaction("EXCLUSIVE"):
            if self.conn.execute("PRAGMA user_version;").fetchone()[0] >= SCHEMA_VERSION:
                return  # Another process migrated the DB in the meantime
            self.conn.execute("ALTER TABLE items ADD COLUMN missing INTEGER NOT NULL DEFAULT 0;")
            self.conn.executemany("UPDATE items SET missing = 1 WHERE row_id = ?;",
                                  [(row_id,) for (row_id, _), e in zip(rows, exists) if not e])
            self._create_schema()
        logger.info(f"Migration completed. {exists.count(False)} of {len(rows)} files are missing.")

    # ---------------------------------------------------------------------------------------------
    # Path handling

    def _to_db_path(self, path: Path | str) -> str:
        """Returns the path as stored in the DB: relative to the registry root if
        the file is located inside the registry, absolute otherwise."""
        path = Path(path).absolute()
        try:
            return path.relative_to(self.path).as_posix()
        except ValueError:
            return path.as_posix()

    def _from_db_path(self, path: Optional[str]) -> Optional[Path]:
        if path is None:
            return None
        path = Path(path)
        return path if path.is_absolute() else self.path / path

    def is_inside(self, path: Path) -> bool:
        """Returns True iff the path is located inside the registry's root directory."""
        return Path(path).absolute().is_relative_to(self.path)

    def is_temp_path(self, path: Path) -> bool:
        """Returns True iff the file was created by ezMM as temporary storage
        (e.g., for items initialized from binary data)."""
        return Path(path).absolute().parent == self.path / "items"

    # ---------------------------------------------------------------------------------------------
    # Item retrieval

    def get(self, reference: str = None, kind: str = None, identifier: int = None) -> Optional[Item]:
        """Gets the referenced item object by loading it from the cache or,
        if not in the cache, from the disk. References of removed duplicates
        resolve to the remaining (canonical) item."""
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

    def get_by_source_url(self, url: str, kind: str = None) -> Optional[Item]:
        """Returns the item that originates from the given URL, or None if no such
        item exists. Optionally restricted to a kind. Updates the source's last access
        time (at most once per hour, to keep lookups free of writes)."""
        stmt = """
            SELECT i.kind, i.id, s.last_accessed FROM sources s
            JOIN items i ON i.row_id = s.item_row_id
            WHERE s.url = ?"""
        params = (url,)
        if kind is not None:
            stmt += " AND i.kind = ?"
            params += (kind,)
        rows = self._execute(stmt + ";", params)
        if rows:
            if rows[0][2] < _access_threshold():
                with self._transaction():
                    self.conn.execute("UPDATE sources SET last_accessed = ? WHERE url = ?;", (_now(), url))
            return self.get(kind=rows[0][0], identifier=rows[0][1])

    def get_by_sha256(self, kind: str, sha256: str) -> Optional[Item]:
        """Returns the item with the given content hash, if any."""
        identifier = self._get_id_by_sha256(kind, sha256)
        if identifier is not None:
            return self.get(kind=kind, identifier=identifier)

    def get_source_urls(self, kind: str, identifier: int) -> list[str]:
        """Returns all source URLs of the item, ordered by the time they were added."""
        return [source["url"] for source in self.get_sources(kind, identifier)]

    def get_sources(self, kind: str, identifier: int) -> list[dict]:
        """Returns all sources of the item (URL, creation and last access time),
        ordered by the time they were added."""
        rows = self._execute("""
            SELECT s.url, s.created_at, s.last_accessed FROM items i
            JOIN sources s ON s.item_row_id = i.row_id
            WHERE i.kind = ? AND i.id = ? ORDER BY s.id;
        """, (kind, identifier))
        return [dict(url=url, created_at=created_at, last_accessed=last_accessed)
                for url, created_at, last_accessed in rows]

    def get_aliases(self, kind: str, identifier: int) -> list[int]:
        """Returns the IDs of all removed duplicates that now resolve to the given item."""
        rows = self._execute("SELECT id FROM items WHERE kind = ? AND canonical_id = ? ORDER BY id;",
                             (kind, identifier))
        return [i for (i,) in rows]

    def get_row(self, kind: str, identifier: int) -> Optional[dict]:
        """Returns the raw registry entry of the item as a dict (without loading the item)."""
        rows = self._execute(f"""
            SELECT {self._COLUMNS} FROM items WHERE kind = ? AND id = ?;
        """, (kind, identifier))
        return self._row_to_dict(rows[0]) if rows else None

    def get_cached(self, reference: str = None,
                   kind: str = None,
                   file_path: Path | str = None,
                   identifier: int = None) -> Optional[Item]:
        if reference:
            kind, identifier = parse_ref(reference)
        elif kind is not None and file_path is not None:
            identifier = self._get_id_by_path(kind, file_path)
        else:
            assert identifier is not None
        return self._get_cached(kind, identifier)

    def _get_id_by_path(self, kind: str, item_path: Path | str) -> Optional[int]:
        rows = self._execute("SELECT id FROM items WHERE kind = ? AND path = ? AND canonical_id IS NULL LIMIT 1;",
                             (kind, self._to_db_path(item_path)))
        return rows[0][0] if rows else None

    def _get_id_by_sha256(self, kind: str, sha256: str) -> Optional[int]:
        rows = self._execute("""
            SELECT id FROM items WHERE kind = ? AND sha256 = ? AND canonical_id IS NULL ORDER BY id LIMIT 1;
        """, (kind, sha256))
        return rows[0][0] if rows else None

    def _get_item_by_id(self, kind: str, identifier: int) -> Optional[Item]:
        # No lock needed: if threads load the same item concurrently, all get the cached instance
        row = self.get_row(kind, identifier)
        if row is None or kind not in KIND2ITEM:
            return None
        if row["canonical_id"] is not None:
            return self.get(kind=kind, identifier=row["canonical_id"])
        source_urls = self.get_source_urls(kind, identifier)
        item_cls = KIND2ITEM[kind]
        # Never rely on the `missing` flag here: the item validates its file itself
        try:
            item = item_cls(id=identifier,
                            file_path=row["path"],
                            source_url=source_urls[0] if source_urls else None)
        except FileNotFoundError:
            self.set_missing(kind, identifier, True)
            raise
        if row["missing"]:
            self.set_missing(kind, identifier, False)
        if row["sha256"] and item._sha256 is None:
            item._sha256 = row["sha256"]
        return self._add_to_cache(item, identifier)

    # ---------------------------------------------------------------------------------------------
    # Item insertion and updates

    def add_item(self, item: Item) -> Optional[int]:
        """Adds an item (without an ID) to the registry, if not yet registered.
        If an identical file (same kind and content) is registered already, the
        item collapses to the existing registry entry: it adopts the existing ID
        and file path (temporary files created by ezMM get deleted) and the
        item's source URL gets added to the entry. Returns the assigned item ID."""
        if hasattr(item, "id"):
            logger.warning(f"Item {item.reference} already has an ID assigned. Not adding to the DB...")
            return

        identifier = self._get_id_by_path(item.kind, item.file_path)
        if identifier is not None:
            self.add_source_url(item.kind, identifier, item.source_url)
            self._add_to_cache(item, identifier)
            return identifier

        # Do the slow work (hashing, file checks) before the transaction, which blocks all writers
        sha256 = item.sha256
        size = item.file_path.stat().st_size
        duplicate = self._find_duplicate(item.kind, sha256)
        obsolete_file = None
        now = _now()
        with self._transaction():
            identifier = self._get_id_by_sha256(item.kind, sha256)
            if identifier is None:
                # Unknown file: create a new registry entry
                identifier = self.conn.execute("SELECT COALESCE(MAX(id), 0) + 1 FROM items WHERE kind = ?;",
                                               (item.kind,)).fetchone()[0]
                row_id = self.conn.execute("""
                    INSERT INTO items(kind, id, path, sha256, size, canonical_id, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, NULL, ?, ?);
                """, (item.kind, identifier, self._to_db_path(item.file_path), sha256, size, now, now)).lastrowid
            else:
                if duplicate is None or duplicate[0] != identifier:  # Registered concurrently in the meantime
                    duplicate = identifier, self._from_db_path(self.get_row(item.kind, identifier)["path"])
                obsolete_file = self._adopt_duplicate(item, *duplicate)
                row_id = self._get_row_id(item.kind, identifier)
            if self._link_source(row_id, item.source_url):
                self.conn.execute("UPDATE items SET updated_at = ? WHERE row_id = ?;", (now, row_id))
        if obsolete_file is not None:
            obsolete_file.unlink(missing_ok=True)

        self._add_to_cache(item, identifier)
        return identifier

    def _find_duplicate(self, kind: str, sha256: str) -> Optional[tuple[int, Optional[Path]]]:
        """Returns the ID and file path of the registered item with the given hash, if any."""
        identifier = self._get_id_by_sha256(kind, sha256)
        if identifier is not None:
            return identifier, self._from_db_path(self.get_row(kind, identifier)["path"])

    def _adopt_duplicate(self, item: Item, identifier: int, existing_path: Optional[Path]) -> Optional[Path]:
        """Lets the (new) item point to the file of the existing, identical item. Must be called
        within a transaction. Returns the item's (temporary) file if it is obsolete now."""
        if existing_path is not None and existing_path.exists():
            logger.debug(f"File '{item.file_path.as_posix()}' is a duplicate of <{item.kind}:{identifier}>.")
            obsolete_file = None
            if self.is_temp_path(item.file_path) and item.file_path != existing_path:
                obsolete_file = item.file_path  # Gets deleted after the transaction
            item.file_path = existing_path
            return obsolete_file
        # The existing entry's file is gone, so heal it with the new file
        self.conn.execute("UPDATE items SET path = ?, missing = 0, updated_at = ? WHERE kind = ? AND id = ?;",
                          (self._to_db_path(item.file_path), _now(), item.kind, identifier))

    def add_source_url(self, kind: str, identifier: int, url: Optional[str]):
        """Records the URL as a source of the item (if not recorded yet) and updates
        the source's last access time (at most once per hour)."""
        if not url:
            return
        # Skip the write if the source is known for this item and was accessed recently
        known = self._execute("""
            SELECT 1 FROM sources s JOIN items i ON i.row_id = s.item_row_id
            WHERE s.url = ? AND i.kind = ? AND i.id = ? AND s.last_accessed >= ?;
        """, (url, kind, identifier, _access_threshold()))
        if known:
            return
        with self._transaction():
            row_id = self._get_row_id(kind, identifier)
            if row_id is not None and self._link_source(row_id, url):
                self.conn.execute("UPDATE items SET updated_at = ? WHERE row_id = ?;", (_now(), row_id))

    def _link_source(self, item_row_id: int, url: Optional[str]) -> bool:
        """Inserts the source URL (if new), marks it as accessed, and lets it point to the
        item. If the URL pointed to another item before (i.e., the content behind the URL
        changed), it now points to the given item. Returns True iff the source is new for
        this item. Must be called within a transaction."""
        if not url:
            return False
        now = _now()
        previous = self.conn.execute("SELECT item_row_id FROM sources WHERE url = ?;", (url,)).fetchone()
        self.conn.execute("""
            INSERT INTO sources(url, item_row_id, created_at, last_accessed) VALUES (?, ?, ?, ?)
            ON CONFLICT(url) DO UPDATE SET item_row_id = excluded.item_row_id,
                                           last_accessed = excluded.last_accessed;
        """, (url, item_row_id, now, now))
        return previous is None or previous[0] != item_row_id

    def _get_row_id(self, kind: str, identifier: int) -> Optional[int]:
        row = self.conn.execute("SELECT row_id FROM items WHERE kind = ? AND id = ?;", (kind, identifier)).fetchone()
        return row[0] if row else None

    def update_file_path(self, item: Item):
        """Updates the path for the corresponding item in the registry."""
        with self._transaction():
            self.conn.execute("UPDATE items SET path = ?, missing = 0, updated_at = ? WHERE kind = ? AND id = ?;",
                              (self._to_db_path(item.file_path), _now(), item.kind, item.id))

    def set_missing(self, kind: str, identifier: int, missing: bool):
        """Records whether the item's file is missing. The flag is only a shortcut for
        browsing (e.g., in the web UI); loading an item always checks its file directly."""
        with self._transaction():
            self.conn.execute("UPDATE items SET missing = ? WHERE kind = ? AND id = ? AND missing != ?;",
                              (int(missing), kind, identifier, int(missing)))

    def check_files(self) -> dict:
        """Checks for all items whether their file exists and updates the `missing` flags
        accordingly. Returns the number of checked and missing files."""
        rows = self._execute("SELECT row_id, path, missing FROM items WHERE canonical_id IS NULL;")
        exists = _parallel_map(_exists, [self._from_db_path(path) for _, path, _ in rows], label="File check")
        changes = [(int(not e), row_id) for (row_id, _, missing), e in zip(rows, exists) if bool(missing) == e]
        with self._transaction():
            self.conn.executemany("UPDATE items SET missing = ? WHERE row_id = ?;", changes)
        return dict(checked=len(rows), missing=exists.count(False), changed=len(changes))

    def contains(self, kind: str, item_path: Path | str) -> bool:
        return self._get_id_by_path(kind, item_path) is not None

    # ---------------------------------------------------------------------------------------------
    # Deduplication

    def deduplicate(self, dry_run: bool = False) -> dict:
        """Goes over the entire registry, identifies identical files (same kind and same
        SHA-256 hash of the raw file bytes) and removes the duplicates: per group, the item
        with the lowest ID is kept, receives all source URLs, and the other entries become
        aliases of it so that their references remain resolvable. Duplicate files are
        deleted only if they are located inside the registry. Returns a report."""
        report = dict(hashed=0, groups=0, removed=[], freed_bytes=0, deleted_files=[])

        # Backfill missing hashes (in parallel)
        unhashed = self._execute("SELECT row_id, path FROM items WHERE canonical_id IS NULL AND sha256 IS NULL;")
        hashes = _parallel_map(_hash_file, [self._from_db_path(path) for _, path in unhashed], label="Hashing")
        updates = [(sha256, size, _now(), row_id) for (row_id, _), (sha256, size) in
                   ((row, hashed) for row, hashed in zip(unhashed, hashes) if hashed)]
        report["hashed"] = len(updates)
        if updates and not dry_run:
            with self._transaction():
                self.conn.executemany("UPDATE items SET sha256 = ?, size = ?, missing = 0, updated_at = ? "
                                      "WHERE row_id = ?;", updates)

        groups = self._execute("""
            SELECT kind, sha256 FROM items
            WHERE canonical_id IS NULL AND sha256 IS NOT NULL
            GROUP BY kind, sha256 HAVING COUNT(*) > 1;
        """)
        report["groups"] = len(groups)
        to_delete: list[Path] = []
        for kind, sha256 in groups:
            with self._transaction():
                rows = self.conn.execute("""
                    SELECT id, path, size FROM items
                    WHERE kind = ? AND sha256 = ? AND canonical_id IS NULL ORDER BY id;
                """, (kind, sha256)).fetchall()
                (keeper_id, keeper_path, _), duplicates = rows[0], rows[1:]
                keeper_path = self._from_db_path(keeper_path)

                # Prefer keeping an existing file located inside the registry, then any existing file
                candidates = [keeper_path] + [self._from_db_path(path) for _, path, _ in duplicates]
                existing = [c for c in candidates if _exists(c)]
                keeper_path = next((c for c in existing if self.is_inside(c)), existing[0] if existing else keeper_path)

                now = _now()
                for dup_id, dup_path, dup_size in duplicates:
                    report["removed"].append((kind, dup_id, keeper_id))
                    dup_path = self._from_db_path(dup_path)
                    if dup_path and dup_path != keeper_path and dup_path.exists() and self.is_inside(dup_path):
                        to_delete.append(dup_path)
                        report["deleted_files"].append(dup_path.as_posix())
                        report["freed_bytes"] += dup_size or 0
                    if dry_run:
                        continue
                    keeper_row_id, dup_row_id = self._get_row_id(kind, keeper_id), self._get_row_id(kind, dup_id)
                    self.conn.execute("UPDATE sources SET item_row_id = ? WHERE item_row_id = ?;",
                                      (keeper_row_id, dup_row_id))
                    self.conn.execute("UPDATE items SET canonical_id = ?, updated_at = ? "
                                      "WHERE kind = ? AND canonical_id = ?;", (keeper_id, now, kind, dup_id))
                    self.conn.execute("UPDATE items SET canonical_id = ?, path = NULL, updated_at = ? "
                                      "WHERE kind = ? AND id = ?;", (keeper_id, now, kind, dup_id))
                if not dry_run:
                    self.conn.execute("UPDATE items SET path = ?, missing = ?, updated_at = ? "
                                      "WHERE kind = ? AND id = ?;",
                                      (self._to_db_path(keeper_path), int(not _exists(keeper_path)), now, kind,
                                       keeper_id))

        if not dry_run:
            for path in to_delete:
                path.unlink(missing_ok=True)
            self.clear_cache()  # Cached objects of removed duplicates are outdated now
        return report

    # ---------------------------------------------------------------------------------------------
    # Embeddings

    def get_embedding(self, kind: str, identifier: int, model: str) -> Optional[np.ndarray]:
        """Returns the stored (full) embedding of the item computed by the given model, if any."""
        rows = self._execute("""
            SELECT e.vector, e.dtype FROM embeddings e JOIN items i ON i.row_id = e.item_row_id
            WHERE i.kind = ? AND i.id = ? AND e.model = ?;
        """, (kind, identifier, model))
        return _decode_vector(*rows[0]) if rows else None

    def set_embedding(self, kind: str, identifier: int, model: str, vector: np.ndarray):
        """Stores the embedding of the item computed by the given model."""
        self.set_embeddings(model, [(kind, identifier, vector)])

    def set_embeddings(self, model: str, embeddings: Iterable[tuple[str, int, np.ndarray]]):
        """Stores the embeddings (kind, id, vector) of multiple items, computed by the given
        model, in one transaction. Vectors are stored as float16."""
        embeddings = list(embeddings)
        now = _now()
        with self._transaction():
            rows = [(row_id, model, np.asarray(vector, dtype=np.float16).tobytes(), "float16", now)
                    for kind, identifier, vector in embeddings
                    if (row_id := self._get_row_id(kind, identifier)) is not None]
            self.conn.executemany("""
                INSERT INTO embeddings(item_row_id, model, vector, dtype, created_at) VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(item_row_id, model) DO UPDATE SET vector = excluded.vector, dtype = excluded.dtype,
                                                              created_at = excluded.created_at;
            """, rows)
        # Add the new embeddings to the already loaded indices
        for (index_model, _, _), index in self._embedding_index.items():
            if index_model == model and embeddings:
                index.add([(kind, identifier) for kind, identifier, _ in embeddings],
                          np.stack([vector for _, _, vector in embeddings]))

    def list_unembedded(self, model: str, kind: str = None, limit: int = None) -> list[tuple[str, int]]:
        """Returns (kind, id) of all items (without aliases and missing files) that have
        no embedding computed by the given model yet."""
        where, params = self._filter(kind, None, include_missing=False)
        rows = self._execute(f"""
            SELECT kind, id FROM items WHERE {where} AND NOT EXISTS (
                SELECT 1 FROM embeddings e WHERE e.item_row_id = items.row_id AND e.model = ?)
            ORDER BY created_at DESC, id DESC LIMIT ?;
        """, params + (model, -1 if limit is None else limit))
        return [(k, i) for k, i in rows]

    def count_unembedded(self, model: str) -> int:
        where, params = self._filter(None, None, include_missing=False)
        return self._execute(f"""
            SELECT COUNT(*) FROM items WHERE {where} AND NOT EXISTS (
                SELECT 1 FROM embeddings e WHERE e.item_row_id = items.row_id AND e.model = ?);
        """, params + (model,))[0][0]

    def count_embedded(self, model: str) -> int:
        return self._execute("""
            SELECT COUNT(*) FROM embeddings e JOIN items i ON i.row_id = e.item_row_id
            WHERE e.model = ? AND i.canonical_id IS NULL;
        """, (model,))[0][0]

    def get_embedding_index(self, model: str, dim: int, device: str = None) -> VectorIndex:
        """Returns the in-memory index of all stored embeddings of the given model (without
        aliases), truncated to `dim` dimensions and located on the given device ('cpu' for
        RAM, 'cuda' for GPU memory; default: see `ezmm.embedding.get_index_device()`).
        Built on first use, then kept up to date as embeddings get added."""
        if device is None:
            from ezmm.embedding import get_index_device
            device = get_index_device()
        key = (model, dim, device)
        with self._index_lock:  # Only guards the creation, the loading happens without lock
            index = self._embedding_index.get(key)
            load = index is None
            if load:
                # Register the (empty) index first, so embeddings added while loading get added to it
                index = self._embedding_index[key] = VectorIndex(dim, device)
        if load:
            try:
                self._load_index(index, model)
            except BaseException as e:
                self._embedding_index.pop(key, None)
                index.fail(e)
                raise
        return index.wait_until_loaded()

    def _load_index(self, index: VectorIndex, model: str):
        logger.info(f"Loading embeddings into the search index ({index.device}, {index.dim} dimensions)...")
        with self._transaction("DEFERRED"):  # One consistent snapshot of the DB
            n = self.count_embedded(model)
            cursor = self.conn.execute("""
                SELECT i.kind, i.id, e.vector, e.dtype FROM embeddings e JOIN items i ON i.row_id = e.item_row_id
                WHERE e.model = ? AND i.canonical_id IS NULL ORDER BY i.row_id;
            """, (model,))

            def batches():
                while rows := cursor.fetchmany(65536):
                    yield ([(kind, identifier) for kind, identifier, _, _ in rows],
                           np.stack([_decode_vector(vector, dtype)[:index.dim] for _, _, vector, dtype in rows]))

            index.load(n, batches())

    def search(self, vector: np.ndarray, model: str, kind: str = None, limit: int = 48,
               include_missing: bool = False, exclude: tuple[str, int] = None, device: str = None) -> list[dict]:
        """Returns the registry entries (as dicts with an additional `score`) whose embeddings
        are most similar (cosine similarity) to the given vector, most similar first. Only
        items with an embedding by the given model are considered. Embeddings are compared
        in the dimension of the given vector (Matryoshka truncation)."""
        index = self.get_embedding_index(model, dim=len(vector), device=device)
        if len(index) == 0:
            return []
        scores = index.scores(vector)
        results = []
        for i in self._ranking(scores, n_candidates=limit * 4 if kind is None else limit * 40):
            item_kind, identifier = index.keys[i]
            if kind and item_kind != kind or (item_kind, identifier) == exclude:
                continue
            row = self.get_row(item_kind, identifier)
            if row is None or row["canonical_id"] is not None or row["missing"] and not include_missing:
                continue
            row["score"] = float(scores[i])
            results.append(row)
            if len(results) >= limit:
                break
        return results

    @staticmethod
    def _ranking(scores: np.ndarray, n_candidates: int) -> Iterable[int]:
        """Yields the indices of the scores in descending order. Sorts only the top candidates
        first (fast for large registries) and falls back to sorting the rest if more are needed."""
        if len(scores) <= n_candidates:
            yield from np.argsort(-scores)
            return
        top = np.argpartition(-scores, n_candidates)[:n_candidates]
        top = top[np.argsort(-scores[top])]
        yield from top
        rest = np.setdiff1d(np.arange(len(scores)), top, assume_unique=True)
        yield from rest[np.argsort(-scores[rest])]

    # ---------------------------------------------------------------------------------------------
    # Browsing

    def list_items(self, kind: str = None, query: str = None,
                   offset: int = 0, limit: int = 50, include_missing: bool = True) -> list[dict]:
        """Returns the registry entries (newest first, without aliases) as dicts,
        optionally filtered by kind and a search query (matching source URLs and paths).
        With include_missing=False, entries flagged as missing are skipped (see `check_files()`)."""
        where, params = self._filter(kind, query, include_missing)
        rows = self._execute(f"""
            SELECT {self._COLUMNS} FROM items WHERE {where}
            ORDER BY created_at DESC, id DESC LIMIT ? OFFSET ?;
        """, params + (limit, offset))
        return [self._row_to_dict(row) for row in rows]

    def count_items(self, kind: str = None, query: str = None, include_missing: bool = True) -> int:
        where, params = self._filter(kind, query, include_missing)
        return self._execute(f"SELECT COUNT(*) FROM items WHERE {where};", params)[0][0]

    def stats(self) -> dict[str, dict]:
        """Returns the number of items and the total file size per kind."""
        rows = self._execute("""
            SELECT kind, COUNT(*), COALESCE(SUM(size), 0) FROM items
            WHERE canonical_id IS NULL GROUP BY kind;
        """)
        return {kind: dict(count=count, size=size) for kind, count, size in rows}

    @staticmethod
    def _filter(kind: Optional[str], query: Optional[str], include_missing: bool = True) -> tuple[str, tuple]:
        where, params = ["canonical_id IS NULL"], []
        if not include_missing:
            where.append("missing = 0")
        if kind:
            where.append("kind = ?")
            params.append(kind)
        if query:
            pattern = f"%{query}%"
            where.append("(path LIKE ? OR sha256 = ? OR EXISTS (SELECT 1 FROM sources s "
                         "WHERE s.item_row_id = items.row_id AND s.url LIKE ?))")
            params += [pattern, query, pattern]
        return " AND ".join(where), tuple(params)

    _COLUMNS = "kind, id, path, sha256, size, canonical_id, missing, created_at, updated_at"

    def _row_to_dict(self, row: tuple) -> dict:
        kind, identifier, path, sha256, size, canonical_id, missing, created_at, updated_at = row
        return dict(kind=kind, id=identifier, path=self._from_db_path(path), sha256=sha256, size=size,
                    canonical_id=canonical_id, missing=bool(missing), created_at=created_at,
                    updated_at=updated_at, reference=f"<{kind}:{identifier}>")

    # ---------------------------------------------------------------------------------------------
    # Cache and connection

    def _get_cached(self, kind: str, identifier: int) -> Optional[Item]:
        """Tries to retrieve the specified item from the cache. Returns
        None if it is not in the cache."""
        return self.cache.get((kind, identifier))

    def _add_to_cache(self, item: Item, identifier: int) -> Item:
        """Adds the given item to the cache. Keeps and returns an already cached instance
        (e.g., if another thread loaded the same item concurrently)."""
        return self.cache.setdefault((item.kind, identifier), item)
        # TODO: Specify a maximum cache size and evict old items

    def close(self):
        """Closes the connections of all threads."""
        with self._conn_lock:
            for conn in list(self._connections):
                conn.close()
            self._connections.clear()
            self._generation += 1
            self._initialized = False

    def reset(self):
        """Reopens the connection to the DB and clears the cache. No persistent
        data will be deleted."""
        self.close()
        self.clear_cache()
        self.connect()

    def clear_cache(self):
        """Resets the cache to free resources. Call this function if you experience
        out-of-memory issues. This will not affect the persistent data (media files and DB)."""
        self.cache.clear()
        self._embedding_index.clear()


item_registry = ItemRegistry()
