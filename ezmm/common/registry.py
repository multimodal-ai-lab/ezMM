import hashlib
import logging
import os
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from ezmm.common.items import Item, KIND2ITEM
from ezmm.util import parse_ref

logger = logging.getLogger("ezMM")

SCHEMA_VERSION = 2  # Stored in the DB via PRAGMA user_version (legacy per-kind schema = 0)

SCHEMA = """
    CREATE TABLE IF NOT EXISTS items (
        row_id INTEGER PRIMARY KEY,
        kind TEXT NOT NULL,
        id INTEGER NOT NULL,
        path TEXT,                -- Relative to the registry root if inside it, else absolute (NULL for aliases)
        sha256 TEXT,              -- Hash of the raw file bytes, used for deduplication
        size INTEGER,             -- File size in bytes
        canonical_id INTEGER,     -- If set, this row is an alias (removed duplicate) of item (kind, canonical_id)
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


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def compute_sha256(path: Path) -> str:
    """Returns the SHA-256 hash of the file's raw bytes."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class ItemRegistry:
    """Keeps track of all the occurring items efficiently.
    Also holds a cache of already loaded items for efficiency."""
    path: Path  # Absolute path to the root directory of the registry
    _db_path: Path  # Path to the SQLite DB file

    conn: Optional[sqlite3.Connection] = None
    cur: Optional[sqlite3.Cursor] = None
    cache: dict[tuple[str, int], Item] = dict()

    def __init__(self, path: Path | str = None):
        self._lock = threading.RLock()
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
            if self.conn:
                raise RuntimeError("Cannot change path for an established ezMM Item Registry.")
            self.path = path.absolute()
            self._db_path = self.path / "item_registry.db"

    def _ensure_connected(self):
        if self.conn is None:
            self.connect()

    def connect(self):
        # Initialize folder, DB, and cache
        logger.info(f"Connecting to item registry at {self.path.as_posix()}")
        if not self.path.exists():
            self.path.mkdir(exist_ok=True, parents=True)
        # Autocommit mode: transactions are managed explicitly (see _transaction)
        self.conn = sqlite3.connect(self._db_path, timeout=10, check_same_thread=False, isolation_level=None)
        self.conn.execute("PRAGMA journal_mode=WAL;")
        self.cur = self.conn.cursor()
        self._init_db()
        logger.debug(f"Successfully connected to item registry.")

    @contextmanager
    def _transaction(self, mode: str = "IMMEDIATE"):
        """Runs the enclosed statements in one (write) transaction."""
        with self._lock:
            self._ensure_connected()
            self.conn.execute(f"BEGIN {mode};")
            try:
                yield self.conn
                self.conn.execute("COMMIT;")
            except BaseException:
                self.conn.execute("ROLLBACK;")
                raise

    def _execute(self, stmt: str, params: tuple = ()) -> list[tuple]:
        with self._lock:
            self._ensure_connected()
            return self.conn.execute(stmt, params).fetchall()

    # ---------------------------------------------------------------------------------------------
    # Schema and migration

    def _init_db(self):
        """Creates the schema of a new DB or migrates a legacy DB to the current schema."""
        with self._lock:
            version = self.conn.execute("PRAGMA user_version;").fetchone()[0]
            if version > SCHEMA_VERSION:
                raise RuntimeError(f"The ezMM registry at {self.path.as_posix()} uses schema version {version}, "
                                   f"but this ezMM version supports only up to {SCHEMA_VERSION}. Please upgrade ezMM.")
            if version < SCHEMA_VERSION:
                if self._legacy_tables():
                    self.migrate()
                else:
                    with self._transaction("EXCLUSIVE"):
                        self._create_schema()

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
                if name not in ("items", "sources") and not name.startswith("sqlite_")]

    def migrate(self):
        """Migrates a legacy DB (one table per item kind) to the unified schema. Item IDs
        are preserved so that existing references remain valid. A backup of the legacy DB
        is written next to it. Does nothing if the DB is already up to date."""
        with self._lock:
            self._ensure_connected()
            legacy_tables = self._legacy_tables()
            if self.conn.execute("PRAGMA user_version;").fetchone()[0] >= SCHEMA_VERSION or not legacy_tables:
                return

            logger.info(f"Migrating legacy ezMM registry at {self.path.as_posix()} to schema v{SCHEMA_VERSION}...")
            backup_path = self.path / "item_registry.v1.bak.db"
            if backup_path.exists():
                backup_path = self.path / f"item_registry.v1.bak.{datetime.now():%Y%m%d-%H%M%S}.db"
            with sqlite3.connect(backup_path) as backup:
                self.conn.backup(backup)
            backup.close()
            logger.info(f"Backup of the legacy registry written to {backup_path.as_posix()}")

            # Read and hash everything first (slow part, outside the write transaction)
            rows = []
            for kind in legacy_tables:
                for identifier, path, source_url in self.conn.execute(
                        f"SELECT id, path, source_url FROM {kind} ORDER BY id;").fetchall():
                    rows.append((kind, identifier, path, source_url))
            now = _now()
            records = []
            for i, (kind, identifier, path, source_url) in enumerate(rows, start=1):
                path = Path(path)
                if not path.exists():
                    # Heal the path, e.g., if the registry was moved: try the default location
                    # inside the registry and the file's original folder inside the registry
                    candidates = [self.path / kind / f"{identifier}{path.suffix}",
                                  self.path / path.parent.name / path.name]
                    healed = next((c for c in candidates if c.exists()), None)
                    if healed:
                        path = healed
                    else:
                        logger.warning(f"File of <{kind}:{identifier}> not found at '{path.as_posix()}'.")
                sha256 = size = None
                if path.exists():
                    sha256 = compute_sha256(path)
                    size = path.stat().st_size
                records.append((kind, identifier, self._to_db_path(path), sha256, size, source_url))
                if i % 100 == 0:
                    logger.info(f"Migrated {i}/{len(rows)} items...")

            with self._transaction("EXCLUSIVE"):
                if self.conn.execute("PRAGMA user_version;").fetchone()[0] >= SCHEMA_VERSION:
                    return  # Another process migrated the DB in the meantime
                self._create_schema()
                for kind, identifier, path, sha256, size, source_url in records:
                    row_id = self.conn.execute("""
                        INSERT INTO items(kind, id, path, sha256, size, canonical_id, created_at, updated_at)
                        VALUES (?, ?, ?, ?, ?, NULL, ?, ?);
                    """, (kind, identifier, path, sha256, size, now, now)).lastrowid
                    self._link_source(row_id, source_url)
                for kind in legacy_tables:
                    self.conn.execute(f"DROP TABLE {kind};")
            logger.info(f"Migration of {len(records)} items completed.")

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
        item exists. Optionally restricted to a kind. Updates the source's last access time."""
        stmt = """
            SELECT i.kind, i.id FROM sources s
            JOIN items i ON i.row_id = s.item_row_id
            WHERE s.url = ?"""
        params = (url,)
        if kind is not None:
            stmt += " AND i.kind = ?"
            params += (kind,)
        rows = self._execute(stmt + ";", params)
        if rows:
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
        rows = self._execute("""
            SELECT kind, id, path, sha256, size, canonical_id, created_at, updated_at
            FROM items WHERE kind = ? AND id = ?;
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
        with self._lock:
            row = self.get_row(kind, identifier)
            if row is None or kind not in KIND2ITEM:
                return None
            if row["canonical_id"] is not None:
                return self.get(kind=kind, identifier=row["canonical_id"])
            source_urls = self.get_source_urls(kind, identifier)
            item_cls = KIND2ITEM[kind]
            item = item_cls(id=identifier,
                            file_path=row["path"],
                            source_url=source_urls[0] if source_urls else None)
            if row["sha256"] and item._sha256 is None:
                item._sha256 = row["sha256"]
            self._add_to_cache(item, identifier)
            return item

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
        if identifier is None:
            sha256 = item.sha256  # Hash outside the transaction as it may take a while
            size = item.file_path.stat().st_size
            now = _now()
            with self._transaction():
                identifier = self._get_id_by_sha256(item.kind, sha256)
                if identifier is None:
                    # Unknown file: create a new registry entry
                    identifier = self.conn.execute("SELECT COALESCE(MAX(id), 0) + 1 FROM items WHERE kind = ?;",
                                                   (item.kind,)).fetchone()[0]
                    self.conn.execute("""
                        INSERT INTO items(kind, id, path, sha256, size, canonical_id, created_at, updated_at)
                        VALUES (?, ?, ?, ?, ?, NULL, ?, ?);
                    """, (item.kind, identifier, self._to_db_path(item.file_path), sha256, size, now, now))
                else:
                    self._adopt_duplicate(item, identifier)

        self.add_source_url(item.kind, identifier, item.source_url)
        self._add_to_cache(item, identifier)
        return identifier

    def _adopt_duplicate(self, item: Item, identifier: int):
        """Lets the (new) item point to the file of the existing, identical item."""
        existing_path = self._from_db_path(self.get_row(item.kind, identifier)["path"])
        if existing_path is not None and existing_path.exists():
            logger.debug(f"File '{item.file_path.as_posix()}' is a duplicate of <{item.kind}:{identifier}>.")
            if self.is_temp_path(item.file_path) and item.file_path != existing_path:
                item.file_path.unlink(missing_ok=True)
            item.file_path = existing_path
        else:
            # The existing entry's file is gone, so heal it with the new file
            self.conn.execute("UPDATE items SET path = ?, updated_at = ? WHERE kind = ? AND id = ?;",
                              (self._to_db_path(item.file_path), _now(), item.kind, identifier))

    def add_source_url(self, kind: str, identifier: int, url: Optional[str]):
        """Records the URL as a source of the item (if not recorded yet) and
        updates the source's last access time."""
        if not url:
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
            self.conn.execute("UPDATE items SET path = ?, updated_at = ? WHERE kind = ? AND id = ?;",
                              (self._to_db_path(item.file_path), _now(), item.kind, item.id))

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

        # Backfill missing hashes
        missing = self._execute("SELECT kind, id, path FROM items WHERE canonical_id IS NULL AND sha256 IS NULL;")
        for kind, identifier, path in missing:
            path = self._from_db_path(path)
            if path is not None and path.exists():
                report["hashed"] += 1
                if not dry_run:
                    with self._transaction():
                        self.conn.execute("UPDATE items SET sha256 = ?, size = ?, updated_at = ? "
                                          "WHERE kind = ? AND id = ?;",
                                          (compute_sha256(path), path.stat().st_size, _now(), kind, identifier))

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

                # Prefer keeping a file that is located inside the registry
                if not (keeper_path and keeper_path.exists() and self.is_inside(keeper_path)):
                    for _, dup_path, _ in duplicates:
                        dup_path = self._from_db_path(dup_path)
                        if dup_path and dup_path.exists() and self.is_inside(dup_path):
                            keeper_path = dup_path
                            break

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
                    self.conn.execute("UPDATE items SET path = ?, updated_at = ? WHERE kind = ? AND id = ?;",
                                      (self._to_db_path(keeper_path), now, kind, keeper_id))

        if not dry_run:
            for path in to_delete:
                path.unlink(missing_ok=True)
            self.clear_cache()  # Cached objects of removed duplicates are outdated now
        return report

    # ---------------------------------------------------------------------------------------------
    # Browsing

    def list_items(self, kind: str = None, query: str = None,
                   offset: int = 0, limit: int = 50) -> list[dict]:
        """Returns the registry entries (newest first, without aliases) as dicts,
        optionally filtered by kind and a search query (matching source URLs and paths)."""
        where, params = self._filter(kind, query)
        rows = self._execute(f"""
            SELECT kind, id, path, sha256, size, canonical_id, created_at, updated_at
            FROM items WHERE {where}
            ORDER BY created_at DESC, id DESC LIMIT ? OFFSET ?;
        """, params + (limit, offset))
        return [self._row_to_dict(row) for row in rows]

    def count_items(self, kind: str = None, query: str = None) -> int:
        where, params = self._filter(kind, query)
        return self._execute(f"SELECT COUNT(*) FROM items WHERE {where};", params)[0][0]

    def stats(self) -> dict[str, dict]:
        """Returns the number of items and the total file size per kind."""
        rows = self._execute("""
            SELECT kind, COUNT(*), COALESCE(SUM(size), 0) FROM items
            WHERE canonical_id IS NULL GROUP BY kind;
        """)
        return {kind: dict(count=count, size=size) for kind, count, size in rows}

    @staticmethod
    def _filter(kind: Optional[str], query: Optional[str]) -> tuple[str, tuple]:
        where, params = ["canonical_id IS NULL"], []
        if kind:
            where.append("kind = ?")
            params.append(kind)
        if query:
            pattern = f"%{query}%"
            where.append("(path LIKE ? OR sha256 = ? OR EXISTS (SELECT 1 FROM sources s "
                         "WHERE s.item_row_id = items.row_id AND s.url LIKE ?))")
            params += [pattern, query, pattern]
        return " AND ".join(where), tuple(params)

    def _row_to_dict(self, row: tuple) -> dict:
        kind, identifier, path, sha256, size, canonical_id, created_at, updated_at = row
        return dict(kind=kind, id=identifier, path=self._from_db_path(path), sha256=sha256, size=size,
                    canonical_id=canonical_id, created_at=created_at, updated_at=updated_at,
                    reference=f"<{kind}:{identifier}>")

    # ---------------------------------------------------------------------------------------------
    # Cache and connection

    def _get_cached(self, kind: str, identifier: int) -> Optional[Item]:
        """Tries to retrieve the specified item from the cache. Returns
        None if it is not in the cache."""
        return self.cache.get((kind, identifier))

    def _add_to_cache(self, item: Item, identifier: int) -> None:
        """Adds the given item to the cache (keeps an already cached instance)."""
        self.cache.setdefault((item.kind, identifier), item)
        # TODO: Specify a maximum cache size and evict old items

    def close(self):
        if self.conn:
            self.conn.close()
        self.conn = None
        self.cur = None

    def reset(self):
        """Reopens the connection to the DB and clears the cache. No persistent
        data will be deleted."""
        if self.conn:
            self.close()
        self.clear_cache()
        self.connect()

    def clear_cache(self):
        """Resets the cache to free resources. Call this function if you experience
        out-of-memory issues. This will not affect the persistent data (media files and DB)."""
        self.cache.clear()


item_registry = ItemRegistry()
