import hashlib
import logging
import os
import sqlite3
import threading
import time
import weakref
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from itertools import groupby
from pathlib import Path
from typing import TypeVar

import numpy as np

from ezmm.common.items import KIND2ITEM, Item
from ezmm.common.vector_index import VectorIndex
from ezmm.util import parse_ref, progress_bar

logger = logging.getLogger("ezMM")

SCHEMA_VERSION = 3  # Stored in the DB via PRAGMA user_version (legacy per-kind schema = 0)

# Number of threads used to read/hash files in bulk operations (migration, deduplication, file checks)
N_WORKERS = min(32, (os.cpu_count() or 1) + 4)

DEDUP_BATCH_SIZE = 1000  # Groups of duplicates removed per transaction
HASH_BATCH_SIZE = 1000  # Hashes saved per transaction during deduplication (makes it resumable)
BUSY_TIMEOUT = 60  # Seconds a write waits for other writers (threads or processes) to finish
ACCESS_UPDATE_INTERVAL = timedelta(hours=1)  # Sources' last access times are updated at most this often
ORPHAN_MIN_AGE = 3600  # Seconds an unreferenced file (or a dead entry) must be unchanged before it gets removed
PURGE_BATCH_SIZE = 500  # Dead entries purged per transaction (also bounds the number of SQL parameters)

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

# Index for finding the aliases of an item (an optional addition to schema v3, created on connect)
ALIASES_INDEX = "CREATE INDEX IF NOT EXISTS items_canonical_idx ON items(kind, canonical_id);"

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
    return datetime.now(UTC).isoformat(timespec="seconds")


def compute_sha256(path: Path) -> str:
    """Returns the SHA-256 hash of the file's raw bytes."""
    with open(path, "rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


T = TypeVar("T")
R = TypeVar("R")


def _parallel_map(fn: Callable[[T], R], inputs: Iterable[T],
                  on_progress: Callable[[int], None] | None = None) -> list[R]:
    """Applies the (I/O-bound) function to all inputs using a thread pool, preserving the order.
    Hashing releases the GIL, so file reading and hashing run truly in parallel. Calls
    `on_progress(done)` after each input."""
    inputs = list(inputs)
    results = []
    with ThreadPoolExecutor(max_workers=N_WORKERS) as pool:
        for i, result in enumerate(pool.map(fn, inputs) if len(inputs) > 1 else map(fn, inputs), start=1):
            results.append(result)
            if on_progress:
                on_progress(i)
    return results


def _hash_file(path: Path | None) -> tuple[str, int] | None:
    """Returns the SHA-256 hash and size of the file, or None if it cannot be read."""
    try:
        return compute_sha256(path), path.stat().st_size
    except (OSError, TypeError):
        return None


class _Connection(sqlite3.Connection):
    """SQLite connection that supports weak references (for tracking the threads' connections)."""


def _access_threshold() -> str:
    """Returns the time before which a source's last access time gets updated on access."""
    return (datetime.now(UTC) - ACCESS_UPDATE_INTERVAL).isoformat(timespec="seconds")


def _decode_vector(blob: bytes, dtype: str) -> np.ndarray:
    return np.frombuffer(blob, dtype=dtype).astype(np.float32)


def _exists(path: Path | None) -> bool:
    return path is not None and path.exists()


def _file_size(path: Path | None) -> int | None:
    """Returns the size of the file in bytes, or None if it does not exist or cannot be read."""
    try:
        return path.stat().st_size
    except (OSError, AttributeError):
        return None


def _normalize_path(path: Path | str) -> str:
    """Returns a normalized form of the (absolute) path for comparisons (collapses '..',
    unifies separators and, on case-insensitive Windows, the case)."""
    return os.path.normcase(os.path.normpath(path))


def _scan_files(folder: Path) -> Iterable[os.DirEntry]:
    """Yields all files inside the folder and its subfolders (without following symlinks)."""
    try:
        with os.scandir(folder) as entries:
            for entry in entries:
                if entry.is_dir(follow_symlinks=False):
                    yield from _scan_files(Path(entry.path))
                else:
                    yield entry
    except FileNotFoundError:
        return


class ItemRegistry:
    """Keeps track of all the occurring items efficiently.
    Also holds a cache of already loaded items for efficiency."""
    path: Path  # Absolute path to the root directory of the registry
    _db_path: Path  # Path to the SQLite DB file

    def __init__(self, path: Path | str | None = None):
        self.cache: dict[tuple[str, int], Item] = {}  # Loaded items by (kind, ID)
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
            logger.debug("Successfully connected to item registry.")

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
            self.conn.execute(ALIASES_INDEX)
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
            backup_path = self.path / f"item_registry.v{version}.bak.{datetime.now().astimezone():%Y%m%d-%H%M%S}.db"
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

        with progress_bar("Migration: checking files", len(rows)) as update:
            paths = _parallel_map(heal, rows, on_progress=update)
        with progress_bar("Migration: hashing", len(paths)) as update:
            hashes = _parallel_map(_hash_file, paths, on_progress=update)

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
        with progress_bar("Migration: checking files", len(rows)) as update:
            exists = _parallel_map(_exists, [self._from_db_path(path) for _, path in rows], on_progress=update)
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

    def _from_db_path(self, path: str | None) -> Path | None:
        if path is None:
            return None
        path = Path(path)
        return path if path.is_absolute() else self.path / path

    def _default_path(self, kind: str, identifier: int, path: str) -> Path:
        """Returns the item's default location inside the registry (see `Item._default_file_path`),
        given its stored path (which determines the suffix)."""
        return self.path / kind / f"{identifier}{Path(path).suffix}"

    def is_inside(self, path: Path) -> bool:
        """Returns True iff the path is located inside the registry's root directory."""
        return Path(path).absolute().is_relative_to(self.path)

    def is_temp_path(self, path: Path) -> bool:
        """Returns True iff the file was created by ezMM as temporary storage
        (e.g., for items initialized from binary data)."""
        return Path(path).absolute().parent == self.path / "items"

    # ---------------------------------------------------------------------------------------------
    # Item retrieval

    def get(self, reference: str | None = None, kind: str | None = None, identifier: int | None = None) -> Item | None:
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

    def get_by_path(self, kind: str, path: Path | str) -> Item | None:
        """Returns the item object located at the path ONLY IF it is
        already registered in the registry."""
        identifier = self._get_id_by_path(kind, path)
        if identifier is not None:
            return self.get(kind=kind, identifier=identifier)

    def get_by_source_url(self, url: str, kind: str | None = None) -> Item | None:
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

    def get_by_sha256(self, kind: str, sha256: str) -> Item | None:
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

    def get_row(self, kind: str, identifier: int) -> dict | None:
        """Returns the raw registry entry of the item as a dict (without loading the item)."""
        rows = self._execute(f"""
            SELECT {self._COLUMNS} FROM items WHERE kind = ? AND id = ?;
        """, (kind, identifier))
        return self._row_to_dict(rows[0]) if rows else None

    def get_cached(self, reference: str | None = None,
                   kind: str | None = None,
                   file_path: Path | str | None = None,
                   identifier: int | None = None) -> Item | None:
        if reference:
            kind, identifier = parse_ref(reference)
        elif kind is not None and file_path is not None:
            identifier = self._get_id_by_path(kind, file_path)
        else:
            assert identifier is not None
        return self._get_cached(kind, identifier)

    def _get_id_by_path(self, kind: str, item_path: Path | str) -> int | None:
        rows = self._execute("SELECT id FROM items WHERE kind = ? AND path = ? AND canonical_id IS NULL LIMIT 1;",
                             (kind, self._to_db_path(item_path)))
        return rows[0][0] if rows else None

    def _get_id_by_sha256(self, kind: str, sha256: str) -> int | None:
        rows = self._execute("""
            SELECT id FROM items WHERE kind = ? AND sha256 = ? AND canonical_id IS NULL ORDER BY id LIMIT 1;
        """, (kind, sha256))
        return rows[0][0] if rows else None

    def _get_item_by_id(self, kind: str, identifier: int) -> Item | None:
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

    def add_item(self, item: Item) -> int | None:
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

    def _find_duplicate(self, kind: str, sha256: str) -> tuple[int, Path | None] | None:
        """Returns the ID and file path of the registered item with the given hash, if any."""
        identifier = self._get_id_by_sha256(kind, sha256)
        if identifier is not None:
            return identifier, self._from_db_path(self.get_row(kind, identifier)["path"])

    def _adopt_duplicate(self, item: Item, identifier: int, existing_path: Path | None) -> Path | None:
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

    def add_source_url(self, kind: str, identifier: int, url: str | None):
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

    def _link_source(self, item_row_id: int, url: str | None) -> bool:
        """Inserts the source URL (if new), marks it as accessed, and lets it point to the
        item. If the URL pointed to another item before (i.e., the content behind the URL
        changed), it now points to the given item. Returns True iff the source is new for
        this item. Must be called within a transaction."""
        # TODO: Keep the history of sources (schema v4). Today, each URL points to exactly one item
        #  (`sources.url` is UNIQUE), so when the content behind a URL changes (e.g., a web page updates
        #  its media), the previous item loses the URL and possibly all of its provenance. The v1 -> v3
        #  migration lost URLs the same way when several legacy rows shared one URL (the original URLs
        #  are still in the `item_registry.v1.bak.db` backups). Plan:
        #  - Let a URL point to several items: UNIQUE(url, item_row_id) instead of UNIQUE(url), and keep
        #    `created_at` (first seen) and `last_accessed` (last seen) per link.
        #  - `get_by_source_url()` returns the most recently delivered item; `get_sources()` lists all
        #    URLs that ever delivered an item.
        #  - Migration to v4 restores lost URLs from the v1 backups where available.
        #  - Older ezMM versions cannot read v4 registries (needs a schema version bump and a backup).
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

    def _get_row_id(self, kind: str, identifier: int) -> int | None:
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
        accordingly. Also backfills unknown file sizes and refreshes outdated ones (a changed
        size also invalidates the stored hash). Missing files keep their last known size.
        Returns the number of checked and missing files, of updated flags (`changed`), and
        of updated sizes (`sizes_updated`)."""
        rows = self._execute("SELECT row_id, path, missing, size FROM items WHERE canonical_id IS NULL;")
        with progress_bar("Checking files", len(rows)) as update:
            sizes = _parallel_map(_file_size, [self._from_db_path(row[1]) for row in rows], on_progress=update)
        changes = [(int(size is None), row_id) for (row_id, _, missing, _), size in zip(rows, sizes, strict=True)
                   if bool(missing) != (size is None)]
        now = _now()
        size_updates = [(size, now, row_id) for (row_id, _, _, old_size), size in zip(rows, sizes, strict=True)
                        if size is not None and size != old_size]
        with self._transaction():
            self.conn.executemany("UPDATE items SET missing = ? WHERE row_id = ?;", changes)
            # SQLite evaluates all expressions with the old values, so the hash is only cleared if the size was known
            self.conn.executemany("UPDATE items SET sha256 = CASE WHEN size IS NULL THEN sha256 END, "
                                  "size = ?, updated_at = ? WHERE row_id = ?;", size_updates)
        return dict(checked=len(rows), missing=sizes.count(None), changed=len(changes),
                    sizes_updated=len(size_updates))

    def contains(self, kind: str, item_path: Path | str) -> bool:
        return self._get_id_by_path(kind, item_path) is not None

    # ---------------------------------------------------------------------------------------------
    # Deduplication

    def deduplicate(self, dry_run: bool = False,
                    on_progress: Callable[[str, int, int], None] | None = None) -> dict:
        """Goes over the entire registry, identifies identical files (same kind and same
        SHA-256 hash of the raw file bytes) and removes the duplicates: per group, the item
        with the lowest ID is kept, receives all source URLs, and the other entries become
        aliases of it so that their references remain resolvable. Duplicate files are
        deleted only if they are located inside the registry. Returns a report.

        With `dry_run=True`, items and files remain unchanged; only the reported duplicates
        are determined. Missing file hashes get computed and saved in both modes, as they
        merely describe the files' contents.

        Shows progress bars of the phases 'Hashing', 'Checking files', and 'Deduplicating' and
        reports their progress to the optional `on_progress(phase, done, total)` callback.
        Files are read and checked in parallel; the DB work takes O(N + D log N) for N items
        and D duplicates. Hashes are saved in batches, so an interrupted run resumes hashing
        where it stopped."""
        report = dict(hashed=0, groups=0, removed=[], freed_bytes=0, deleted_files=[])

        # Backfill missing hashes (in parallel)
        unhashed = self._execute("SELECT row_id, path FROM items "
                                 "WHERE canonical_id IS NULL AND sha256 IS NULL ORDER BY row_id;")
        with progress_bar("Hashing", len(unhashed), on_progress=on_progress) as update:
            for start in range(0, len(unhashed), HASH_BATCH_SIZE):
                batch = unhashed[start:start + HASH_BATCH_SIZE]
                hashes = _parallel_map(_hash_file, [self._from_db_path(path) for _, path in batch],
                                       on_progress=lambda done, start=start: update(start + done))
                now = _now()
                updates = [(hashed[0], hashed[1], now, row_id)
                           for (row_id, _), hashed in zip(batch, hashes, strict=True) if hashed]
                report["hashed"] += len(updates)
                if updates:  # Saved also in dry runs: hashes describe the files and spare later runs the work
                    with self._transaction():
                        self.conn.executemany("UPDATE items SET sha256 = ?, size = ?, missing = 0, updated_at = ? "
                                              "WHERE row_id = ?;", updates)

        # All items having duplicates, grouped by content, the item with the lowest ID first
        rows = self._execute("""
            SELECT row_id, kind, id, path, size, sha256 FROM items
            WHERE canonical_id IS NULL AND (kind, sha256) IN (
                SELECT kind, sha256 FROM items WHERE canonical_id IS NULL AND sha256 IS NOT NULL
                GROUP BY kind, sha256 HAVING COUNT(*) > 1)
            ORDER BY kind, sha256, id;
        """)
        groups = [list(group) for _, group in groupby(rows, key=lambda row: (row[1], row[5]))]
        report["groups"] = len(groups)

        # Check which files exist (in parallel and outside of any transaction, as it may be slow)
        paths = {row_id: self._from_db_path(path) for row_id, _, _, path, _, _ in rows}
        with progress_bar("Checking files", len(paths), on_progress=on_progress) as update:
            exists = dict(zip(paths, _parallel_map(_exists, list(paths.values()), on_progress=update), strict=True))

        # Decide which file to keep per group: prefer an existing file inside the registry, then any existing file
        plan = []  # (kind, keeper row ID, keeper ID, keeper path, keeper file exists, duplicates)
        to_delete: list[Path] = []
        for (keeper_row_id, kind, keeper_id, _, _, _), *duplicates in groups:
            existing = [row[0] for row in [(keeper_row_id,), *duplicates] if exists[row[0]]]
            kept_row_id = next((r for r in existing if self.is_inside(paths[r])), existing[0] if existing else None)
            keeper_path = paths[kept_row_id] if kept_row_id is not None else paths[keeper_row_id]
            for dup_row_id, _, dup_id, _, dup_size, _ in duplicates:
                report["removed"].append((kind, dup_id, keeper_id))
                dup_path = paths[dup_row_id]
                if dup_path and dup_path != keeper_path and exists[dup_row_id] and self.is_inside(dup_path):
                    to_delete.append(dup_path)
                    report["deleted_files"].append(dup_path.as_posix())
                    report["freed_bytes"] += dup_size or 0
            plan.append((kind, keeper_row_id, keeper_id, keeper_path, kept_row_id is not None, duplicates))
        if dry_run:
            return report

        # Turn the duplicates into aliases, in batches of groups per (short) transaction
        with progress_bar("Deduplicating", len(plan), unit="group", on_progress=on_progress) as update:
            self._remove_duplicates(plan, update)

        for path in to_delete:
            path.unlink(missing_ok=True)
        self.clear_cache()  # Cached objects of removed duplicates are outdated now
        return report

    def _remove_duplicates(self, plan: list[tuple], update: Callable[[int], None]):
        """Turns the duplicates of the plan (see `deduplicate()`) into aliases of the kept items."""
        for start in range(0, len(plan), DEDUP_BATCH_SIZE):
            batch, now = plan[start:start + DEDUP_BATCH_SIZE], _now()
            duplicates = [(kind, keeper_row_id, keeper_id, dup_row_id, dup_id)
                          for kind, keeper_row_id, keeper_id, _, _, dups in batch
                          for dup_row_id, _, dup_id, _, _, _ in dups]
            with self._transaction():
                self.conn.executemany(
                    "UPDATE sources SET item_row_id = ? WHERE item_row_id = ?;",
                    [(keeper_row_id, dup_row_id) for _, keeper_row_id, _, dup_row_id, _ in duplicates])
                # Aliases of a duplicate now point to the keeper (uses the index on canonical_id)
                self.conn.executemany(
                    "UPDATE items SET canonical_id = ?, updated_at = ? WHERE kind = ? AND canonical_id = ?;",
                    [(keeper_id, now, kind, dup_id) for kind, _, keeper_id, _, dup_id in duplicates])
                self.conn.executemany("UPDATE items SET canonical_id = ?, path = NULL, updated_at = ? "
                                      "WHERE row_id = ? AND canonical_id IS NULL;",
                                      [(keeper_id, now, dup_row_id) for _, _, keeper_id, dup_row_id, _ in duplicates])
                self.conn.executemany("UPDATE items SET path = ?, missing = ?, updated_at = ? WHERE row_id = ?;",
                                      [(self._to_db_path(keeper_path), int(not keeper_exists), now, keeper_row_id)
                                       for _, keeper_row_id, _, keeper_path, keeper_exists, _ in batch])
            update(start + len(batch))

    # ---------------------------------------------------------------------------------------------
    # Dead entries

    def remove_dead_entries(self, dry_run: bool = False, min_age: float = ORPHAN_MIN_AGE) -> dict:
        """Purges dead registry entries: items whose file does not exist anymore, together with
        their aliases, sources, and embeddings. This is irreversible: references to purged items
        (e.g., `<image:5>` in stored texts) do not resolve anymore. Returns a report.

        To be conservative, each item's file is checked freshly (the `missing` flag is ignored):
        - If the file exists at the item's default location (`<registry>/<kind>/<id><suffix>`),
          the item is not dead: its stored path gets healed (counted as `healed`).
        - If the file is located outside the registry and its folder does not exist either
          (e.g., an unmounted drive or network share), or cannot be accessed, the item is
          kept (counted as `skipped_unreachable`). Inside the registry, a missing folder means
          that the files are really gone, as the registry root itself is reachable.
        - Items created or updated within the last `min_age` seconds are skipped, as they may
          be in the middle of a registration or move by another thread or process.
        With `dry_run=True`, nothing gets changed. Files are checked in parallel; the DB work
        takes a few queries plus batched writes, so it scales to millions of items."""
        report = dict(checked=0, removed=[], removed_aliases=0, healed=0,
                      skipped_unreachable=0, skipped_recent=0)
        threshold = (datetime.now(UTC) - timedelta(seconds=min_age)).isoformat(timespec="seconds")
        rows = []  # (row_id, kind, id, path, updated_at) of all canonical items old enough
        for row_id, kind, identifier, path, created_at, updated_at in self._execute(
                "SELECT row_id, kind, id, path, created_at, updated_at FROM items WHERE canonical_id IS NULL;"):
            if max(created_at, updated_at) > threshold:
                report["skipped_recent"] += 1
            else:
                rows.append((row_id, kind, identifier, path, updated_at))
        report["checked"] = len(rows)

        # Check the files (in parallel and outside of any transaction, as it may be slow)
        with progress_bar("Checking files", len(rows)) as update:
            statuses = _parallel_map(self._file_status, [row[1:4] for row in rows], on_progress=update)
        healed = [row for row, status in zip(rows, statuses, strict=True) if status == "healed"]
        dead = [row for row, status in zip(rows, statuses, strict=True) if status == "dead"]
        report["healed"] = len(healed)
        report["skipped_unreachable"] = statuses.count("unreachable")

        # Aliases resolving to dead items (repeated to be robust to alias chains)
        aliases_of: dict[tuple[str, int], list[int]] = {(kind, identifier): [] for _, kind, identifier, _, _ in dead}
        remaining = self._execute("SELECT row_id, kind, id, canonical_id FROM items WHERE canonical_id IS NOT NULL;")
        dead_ids = {key: key for key in aliases_of}  # (kind, ID) -> (kind, ID) of the dead canonical item
        while remaining:
            unresolved = []
            for alias in remaining:
                row_id, kind, identifier, canonical_id = alias
                target = dead_ids.get((kind, canonical_id))
                if target is None:
                    unresolved.append(alias)
                else:
                    aliases_of[target].append(row_id)
                    dead_ids[(kind, identifier)] = target
            if len(unresolved) == len(remaining):
                break
            remaining = unresolved
        report["removed"] = [(kind, identifier) for _, kind, identifier, _, _ in dead]
        report["removed_aliases"] = sum(len(aliases) for aliases in aliases_of.values())
        if dry_run:
            return report

        if healed:  # Unless changed in the meantime
            now = _now()
            with self._transaction():
                self.conn.executemany("UPDATE items SET path = ?, missing = 0, updated_at = ? "
                                      "WHERE row_id = ? AND updated_at = ?;",
                                      [(self._to_db_path(self._default_path(kind, identifier, path)), now,
                                        row_id, updated_at) for row_id, kind, identifier, path, updated_at in healed])
        with progress_bar("Purging entries", len(dead), unit="item") as update:
            self._purge(dead, aliases_of, report, update)
        self.clear_cache()  # Cached objects of purged (and healed) items are outdated now
        return report

    def _file_status(self, row: tuple[str, int, str | None]) -> str:
        """Returns whether the item's (kind, ID, stored path) file 'exists', can be 'healed' with
        the default location, is 'unreachable' (see `remove_dead_entries()`), or is 'dead'."""
        kind, identifier, path = row
        if path is None:
            return "dead"  # Cannot point to any file
        stored = self._from_db_path(path)
        try:
            if stored.exists():
                return "exists"
            if self._default_path(kind, identifier, path).exists():
                return "healed"
            if not self.is_inside(stored) and not stored.parent.exists():
                return "unreachable"
        except OSError:  # E.g., no permission to access the location
            return "unreachable"
        return "dead"

    def _purge(self, dead: list[tuple], aliases_of: dict[tuple[str, int], list[int]],
               report: dict, update: Callable[[int], None]):
        """Deletes the dead items (see `remove_dead_entries()`) with their aliases, sources,
        and embeddings, in batches per (short) transaction. Items that changed since they were
        checked (e.g., got healed by another process) are kept and dropped from the report."""
        kept = set()
        for start in range(0, len(dead), PURGE_BATCH_SIZE):
            batch = dead[start:start + PURGE_BATCH_SIZE]
            with self._transaction():
                # Re-check within the transaction that the items are unchanged
                current = {row_id: (path, updated_at) for row_id, path, updated_at in self.conn.execute(
                    f"SELECT row_id, path, updated_at FROM items WHERE canonical_id IS NULL "
                    f"AND row_id IN ({', '.join('?' * len(batch))});", [row[0] for row in batch])}
                unchanged = []
                for row in batch:
                    row_id, kind, identifier, path, updated_at = row
                    if current.get(row_id) == (path, updated_at):
                        unchanged.append(row)
                    else:
                        kept.add((kind, identifier))
                row_ids = [(row_id,) for row_id, kind, identifier, _, _ in unchanged
                           for row_id in (row_id, *aliases_of[(kind, identifier)])]
                self.conn.executemany("DELETE FROM sources WHERE item_row_id = ?;", row_ids)
                self.conn.executemany("DELETE FROM embeddings WHERE item_row_id = ?;", row_ids)
                self.conn.executemany("DELETE FROM items WHERE row_id = ?;", row_ids)
            update(start + len(batch))
        if kept:
            report["removed"] = [key for key in report["removed"] if key not in kept]
            report["removed_aliases"] = sum(len(aliases_of[key]) for key in report["removed"])

    # ---------------------------------------------------------------------------------------------
    # Orphaned files

    def remove_orphaned_files(self, dry_run: bool = False, min_age: float = ORPHAN_MIN_AGE) -> dict:
        """Deletes orphaned files: files inside the registry's item folders (one per kind, plus
        `items/` holding the temporary files of items created from binary data) that are not
        referenced by any item, e.g., left over from interrupted registrations, removed items,
        or older ezMM versions. Nothing else in the registry root (the DB, its backups,
        rendered sequences, ...) is touched. Returns a report.

        Files changed within the last `min_age` seconds are skipped, because a new item's file
        gets written before its registration completes (possibly in another thread or process).
        With `dry_run=True`, the orphans are only reported. Takes one directory scan and one DB
        query in total, so it scales to millions of files."""
        report = dict(scanned=0, orphans=[], freed_bytes=0, skipped_recent=0, deleted=0)

        # Scan the files before loading the referenced paths, so that all files registered at scan
        # time are known as referenced. Files registered later are recent and thus skipped.
        threshold = time.time() - min_age
        candidates = []  # (path, size) of all files old enough
        folders = [self.path / kind for kind in KIND2ITEM] + [self.path / "items"]
        for folder in folders:
            for entry in _scan_files(folder):
                report["scanned"] += 1
                try:
                    stat = entry.stat(follow_symlinks=False)
                except FileNotFoundError:
                    continue  # Deleted in the meantime
                if stat.st_mtime > threshold:
                    report["skipped_recent"] += 1
                else:
                    candidates.append((entry.path, stat.st_size))

        # All rows, incl. those flagged as missing (their files may have come back). An item also
        # uses its default location, as it heals a stale path with it (see Item.validate_file_path).
        referenced = set()
        for kind, identifier, path, canonical_id in self._execute(
                "SELECT kind, id, path, canonical_id FROM items WHERE path IS NOT NULL;"):
            referenced.add(_normalize_path(self._from_db_path(path)))
            if canonical_id is None:
                referenced.add(_normalize_path(self._default_path(kind, identifier, path)))
        orphans = [(path, size) for path, size in candidates if _normalize_path(path) not in referenced]
        report["orphans"] = [Path(path).as_posix() for path, _ in orphans]
        report["freed_bytes"] = sum(size for _, size in orphans)
        if dry_run:
            return report

        with progress_bar("Deleting orphans", len(orphans)) as update:
            for done, (path, size) in enumerate(orphans, start=1):
                try:
                    os.remove(path)
                    report["deleted"] += 1
                except OSError as e:  # E.g., deleted in the meantime or in use
                    report["freed_bytes"] -= size
                    if not isinstance(e, FileNotFoundError):
                        logger.warning(f"Could not delete orphaned file '{path}': {e}")
                update(done)
        return report

    # ---------------------------------------------------------------------------------------------
    # Embeddings

    def get_embedding(self, kind: str, identifier: int, model: str) -> np.ndarray | None:
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

    def list_unembedded(self, model: str, kind: str | None = None, limit: int | None = None) -> list[tuple[str, int]]:
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

    def get_embedding_index(self, model: str, dim: int, device: str | None = None) -> VectorIndex:
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

    def search(self, vector: np.ndarray, model: str, kind: str | None = None, limit: int = 48,
               include_missing: bool = False, exclude: tuple[str, int] | None = None, device: str | None = None) -> list[dict]:
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

    def list_items(self, kind: str | None = None, query: str | None = None,
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

    def count_items(self, kind: str | None = None, query: str | None = None, include_missing: bool = True) -> int:
        where, params = self._filter(kind, query, include_missing)
        return self._execute(f"SELECT COUNT(*) FROM items WHERE {where};", params)[0][0]

    def stats(self) -> dict[str, dict]:
        """Returns per kind the number of items, their total (stored) file size in bytes, and
        the number of items with unknown size (see `check_files()` to backfill them)."""
        rows = self._execute("""
            SELECT kind, COUNT(*), COALESCE(SUM(size), 0), COUNT(*) - COUNT(size) FROM items
            WHERE canonical_id IS NULL GROUP BY kind;
        """)
        return {kind: dict(count=count, size=size, unknown_size=unknown)
                for kind, count, size, unknown in rows}

    def total_size(self, kind: str | None = None) -> int:
        """Returns the total stored file size in bytes of all items (optionally of the given kind),
        excluding aliases. Fast, as it reads only the DB; items with unknown size count as 0."""
        where, params = self._filter(kind, None)
        return self._execute(f"SELECT COALESCE(SUM(size), 0) FROM items WHERE {where};", params)[0][0]

    @staticmethod
    def _filter(kind: str | None, query: str | None, include_missing: bool = True) -> tuple[str, tuple]:
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

    def _get_cached(self, kind: str, identifier: int) -> Item | None:
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
