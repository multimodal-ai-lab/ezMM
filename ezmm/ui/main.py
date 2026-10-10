"""Local web UI for browsing the ezMM item registry and rendered MultimodalSequences.
Start it with `python -m ezmm ui` (requires `pip install ezmm[ui]`)."""
import logging
import math
import secrets
import socket
import tempfile
import threading
from collections import OrderedDict
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlencode, urlparse

import uvicorn
from fastapi import FastAPI, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.requests import Request

from ezmm import MultimodalSequence
from ezmm.common import item_registry
from ezmm.common.items import KIND2ITEM, KINDS
from ezmm.common.items.file import format_size
from ezmm.ui.common import get_seq_path
from ezmm.util import parse_ref

UI_DIR = Path(__file__).parent
PAGE_SIZE = 48
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 7878

logger = logging.getLogger("ezMM")

app = FastAPI(title="ezMM")
app.mount("/static", StaticFiles(directory=UI_DIR / "static"), name="static")

templates = Jinja2Templates(directory=UI_DIR / "templates")


def _time_ago(timestamp: str | None) -> str:
    if not timestamp:
        return ""
    then = datetime.fromisoformat(timestamp)
    if then.tzinfo is None:
        then = then.replace(tzinfo=UTC)
    seconds = (datetime.now(UTC) - then).total_seconds()
    for unit, length in (("y", 31536000), ("mo", 2592000), ("d", 86400), ("h", 3600), ("min", 60)):
        if seconds >= length:
            return f"{int(seconds // length)} {unit} ago"
    return "just now"


def _host(url: str) -> str:
    parsed = urlparse(url)
    return parsed.netloc or ("local file" if parsed.scheme == "file" else url)


def _duration(seconds: float) -> str:
    """Formats the duration as m:ss, or h:mm:ss for durations of an hour or more."""
    seconds = max(round(seconds or 0), 0)
    hours, minutes = seconds // 3600, seconds // 60 % 60
    return f"{hours}:{minutes:02d}:{seconds % 60:02d}" if hours else f"{minutes}:{seconds % 60:02d}"


def _browse_url(kind: str = "", q: str = "", missing: bool = False, page: int = 1) -> str:
    """Returns the URL of the browse page with the given filters (omitting defaults)."""
    params = {"kind": kind, "q": q, "missing": 1 if missing else None, "page": page if page > 1 else None}
    query = urlencode({key: value for key, value in params.items() if value})
    return "/?" + query if query else "/"


def _search_url(q: str = "", kind: str = "", like: str = "", vec: str = "", page: int = 1) -> str:
    """Returns the URL of the search page with the given query and filter (omitting defaults)."""
    params = dict(q=q, kind=kind, like=like, vec=vec, page=page if page > 1 else None)
    query = urlencode({key: value for key, value in params.items() if value})
    return "/search?" + query if query else "/search"


class _QueryCache:
    """Small, thread-safe LRU cache of query vectors (with their labels). Lets further pages
    of a search reuse the query vector instead of re-embedding (or re-uploading) the query."""

    def __init__(self, max_size: int = 100):
        self.max_size = max_size
        self._entries = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key: str):
        with self._lock:
            value = self._entries.get(key)
            if value is not None:
                self._entries.move_to_end(key)
            return value

    def put(self, key: str, value):
        with self._lock:
            self._entries[key] = value
            self._entries.move_to_end(key)
            while len(self._entries) > self.max_size:
                self._entries.popitem(last=False)

    def clear(self):
        with self._lock:
            self._entries.clear()


query_cache = _QueryCache()


templates.env.filters.update(size=format_size, time_ago=_time_ago, host=_host, duration=_duration)
def _embeddings_available() -> bool:
    from ezmm.embedding import is_available
    return is_available()


templates.env.globals.update(kinds=KINDS, browse_url=_browse_url, search_url=_search_url,
                             embeddings_available=_embeddings_available)


def _get_entry(kind: str, identifier: int) -> dict:
    if kind not in KIND2ITEM:
        raise HTTPException(404, f"Unknown item kind '{kind}'.")
    row = item_registry.get_row(kind, identifier)
    if row is None:
        raise HTTPException(404, f"Item <{kind}:{identifier}> does not exist.")
    return row


def _add_sources(entries: list[dict]):
    for entry in entries:
        source_urls = item_registry.get_source_urls(entry["kind"], entry["id"])
        entry["source_url"] = source_urls[0] if source_urls else None
        entry["suffix"] = entry["path"].suffix.lstrip(".").upper() if entry["path"] else ""


def _render_cards(request: Request, entries: list[dict], next_url: str | None):
    """Renders only the grid of cards (for endless scrolling), linking to the next page if any."""
    return templates.TemplateResponse(request, "_cards.html", {"entries": entries, "next_url": next_url})


@app.get("/", response_class=HTMLResponse)
async def browse(request: Request, kind: str = "", q: str = "", page: int = 1, missing: bool = False,
                 partial: bool = False):
    """Shows a grid of all items in the registry, filterable by kind and search query.
    Items whose file is missing are hidden unless `missing` is set. Filtering relies on the
    registry's `missing` flags, which get corrected for all items shown on the page.
    With `partial`, returns only the cards of the page (used for endless scrolling)."""
    kind = kind if kind in KINDS else ""
    for _ in range(2):  # Repeat once if the flags of shown items turned out to be outdated
        total_all = item_registry.count_items(kind or None, q or None)
        total = total_all if missing else item_registry.count_items(kind or None, q or None, include_missing=False)
        n_pages = max(math.ceil(total / PAGE_SIZE), 1)
        page = max(page, 1) if partial else min(max(page, 1), n_pages)
        entries = item_registry.list_items(kind or None, q or None, offset=(page - 1) * PAGE_SIZE,
                                           limit=PAGE_SIZE, include_missing=missing)
        outdated = False
        for entry in entries:
            exists = bool(entry["path"] and entry["path"].exists())
            if entry["missing"] == exists:
                item_registry.set_missing(entry["kind"], entry["id"], not exists)
                entry["missing"] = not exists
                outdated = True
        if missing or not outdated:
            break

    _add_sources(entries)
    next_url = _browse_url(kind, q, missing, page + 1) if page < n_pages else None
    if partial:
        return _render_cards(request, entries, next_url)

    stats = item_registry.stats()
    return templates.TemplateResponse(request, "browse.html", {
        "entries": entries,
        "next_url": next_url,
        "kind": kind,
        "q": q,
        "page": page,
        "n_pages": n_pages,
        "total": total,
        "missing": missing,
        "n_hidden": total_all - total,
        "stats": stats,
        "total_count": sum(s["count"] for s in stats.values()),
        "total_size": sum(s["size"] for s in stats.values()),
    })


@app.get("/item/{kind}/{identifier}", response_class=HTMLResponse)
async def show_item(request: Request, kind: str, identifier: int):
    """Shows a single item along with all its metadata."""
    entry = _get_entry(kind, identifier)
    if entry["canonical_id"] is not None:
        return RedirectResponse(f"/item/{kind}/{entry['canonical_id']}")

    details, error, audio_only = {}, None, False
    try:
        item = item_registry.get(kind=kind, identifier=identifier)
        details["MIME type"] = item.mime_type
        if kind == "image":
            details["Dimensions"] = f"{item.width} × {item.height} px"
        if kind == "video":
            audio_only = not item.has_video_stream
            if audio_only:
                details["Content"] = "Audio only (no video stream)"
            else:
                details["Dimensions"] = f"{item.width} × {item.height} px"
            details["Duration"] = _duration(item.duration)
            if not audio_only and item.fps > 0:
                details["Frame rate"] = f"{item.fps:.2f} fps"
        if kind == "audio":
            details["Duration"] = _duration(item.duration)
            details["Sample rate"] = f"{item.sample_rate / 1000:g} kHz"
            details["Channels"] = item.channels
            details["Bitrate"] = f"{item.bitrate // 1000} kbps"
    except Exception as e:
        error = str(e)
        logger.warning(f"Could not load item <{kind}:{identifier}>: {e}")

    entry["suffix"] = entry["path"].suffix.lstrip(".").upper() if entry["path"] else ""
    file_exists = bool(entry["path"] and entry["path"].exists())
    if entry["missing"] == file_exists:
        item_registry.set_missing(kind, identifier, not file_exists)
    return templates.TemplateResponse(request, "item.html", {
        "entry": entry,
        "details": details,
        "error": error,
        "file_exists": file_exists,
        "audio_only": audio_only,
        "sources": item_registry.get_sources(kind, identifier),
        "aliases": item_registry.get_aliases(kind, identifier),
    })


@app.get("/item/{kind}/{identifier}/file")
async def item_file(kind: str, identifier: int):
    """Serves the file of the item. Only files of registered items can be served."""
    entry = _get_entry(kind, identifier)
    if entry["canonical_id"] is not None:
        return RedirectResponse(f"/item/{kind}/{entry['canonical_id']}/file")
    path = entry["path"]
    if path is None or not path.exists():
        item_registry.set_missing(kind, identifier, True)
        raise HTTPException(404, f"File of <{kind}:{identifier}> not found.")
    return FileResponse(path)


class _Indexer:
    """Embeds all not yet embedded items of the registry in a background thread."""

    def __init__(self):
        self.thread: threading.Thread | None = None
        self.done = self.total = self.failed = 0

    @property
    def running(self) -> bool:
        return self.thread is not None and self.thread.is_alive()

    def start(self):
        if not self.running:
            self.thread = threading.Thread(target=self._run, daemon=True)
            self.thread.start()

    def _run(self):
        from ezmm.embedding import MODEL_NAME, embed_registry
        self.done, self.total, self.failed = 0, item_registry.count_unembedded(MODEL_NAME), 0
        try:
            embed_registry(on_progress=self._on_progress)
        except Exception as e:
            logger.error(f"Indexing failed: {e}")

    def _on_progress(self, done: int, total: int, failed: int):
        self.done, self.total, self.failed = done, total, failed


indexer = _Indexer()


def _render_search(request: Request, kind: str = "", q: str = "", like: str = "", vec: str = "",
                   page: int = 1, partial: bool = False, error: str | None = None):
    """Renders the search page (or, with `partial`, only the result cards) for a text query (`q`),
    an existing item (`like`), or a query vector cached under the token `vec` (uploaded files)."""
    from ezmm.embedding import INSTALL_HINT, MODEL_NAME, embed_query, get_embedding_dim, is_available
    kind = kind if kind in KINDS else ""
    page = max(page, 1)
    results, exclude, query_vector, query_label, next_url = [], None, None, "", None
    if not is_available():
        q, like, vec, error = "", "", "", INSTALL_HINT
    try:
        if like:
            exclude = parse_ref(f"<{like}>")
            item = item_registry.get(kind=exclude[0], identifier=exclude[1])
            if item is None:
                raise HTTPException(404, f"Item <{like}> does not exist.")
            query_vector, query_label = item.embedding, item.reference
        elif q:
            key = f"q:{get_embedding_dim()}:{q}"
            if (cached := query_cache.get(key)) is None:
                cached = embed_query(q), f"“{q}”"
                query_cache.put(key, cached)
            query_vector, query_label = cached
        elif vec:
            if (cached := query_cache.get(f"vec:{vec}")) is None:
                error = "This search has expired. Please search with the file again."
            else:
                query_vector, query_label = cached
        if query_vector is not None:
            results = item_registry.search(query_vector, MODEL_NAME, kind=kind or None, limit=PAGE_SIZE + 1,
                                           offset=(page - 1) * PAGE_SIZE, exclude=exclude)
            if len(results) > PAGE_SIZE:
                results = results[:PAGE_SIZE]
                next_url = _search_url(q, kind, like, vec, page + 1)
    except HTTPException:
        raise
    except Exception as e:
        error = str(e)
        logger.warning(f"Search failed: {e}")

    _add_sources(results)
    if partial:
        return _render_cards(request, results, next_url)
    return templates.TemplateResponse(request, "search.html", {
        "entries": results,
        "next_url": next_url,
        "page": page,
        "kind": kind,
        "q": q,
        "like": like,
        "vec": vec,
        "query_label": query_label,
        "searched": query_vector is not None,
        "error": error,
        "n_unindexed": item_registry.count_unembedded(MODEL_NAME),
        "n_indexed": item_registry.count_embedded(MODEL_NAME),
        "indexer": indexer,
    })


@app.get("/search", response_class=HTMLResponse)
async def search(request: Request, q: str = "", kind: str = "", like: str = "", vec: str = "",
                 page: int = 1, partial: bool = False):
    """Semantic search over all items with a text query (`q`) or by an existing item (`like`,
    e.g., `image:3`). Only items that are embedded already are searched. Further pages of
    a search with an uploaded file reference its cached query vector by the token `vec`.
    With `partial`, returns only the result cards of the page (used for endless scrolling)."""
    return await run_in_threadpool(_render_search, request, kind=kind, q=q.strip(), like=like, vec=vec,
                                   page=page, partial=partial)


@app.post("/search", response_class=HTMLResponse)
async def search_by_file(request: Request, file: UploadFile, kind: str = Form("")):
    """Semantic search over all items with an uploaded file of any kind as the query.
    The file is embedded on the fly and not added to the registry. Its embedding is
    cached briefly (in memory) so that further pages of the results can be loaded."""
    from ezmm.embedding import embed_file, is_available, kind_of_file
    data = await file.read()
    if not is_available():
        return _render_search(request, kind=kind)
    if not data:
        return _render_search(request, kind=kind, error="The uploaded file is empty.")
    suffix = Path(file.filename or "").suffix
    with tempfile.TemporaryDirectory() as tmp_dir:
        path = Path(tmp_dir) / f"query{suffix}"
        path.write_bytes(data)
        try:
            vector = await run_in_threadpool(embed_file, path, kind_of_file(path, file.content_type))
        except Exception as e:
            return _render_search(request, kind=kind, error=f"Could not embed the uploaded file: {e}")
    token = secrets.token_urlsafe(9)
    query_cache.put(f"vec:{token}", (vector, file.filename or "uploaded file"))
    return await run_in_threadpool(_render_search, request, kind=kind, vec=token)


@app.post("/search/index")
async def index_items(kind: str = Form(""), q: str = Form(""), like: str = Form(""), vec: str = Form("")):
    """Starts embedding all not yet embedded items in the background."""
    from ezmm.embedding import is_available
    if is_available():
        indexer.start()
    return RedirectResponse(_search_url(q, kind, like, vec), status_code=303)


@app.get("/sequences", response_class=HTMLResponse)
async def sequences_overview(request: Request):
    """Shows a list of all rendered MultimodalSequences."""
    sequences = []
    seq_files = sorted(get_seq_path().glob("*.md"), key=lambda f: f.stat().st_mtime, reverse=True)
    for seq_file in seq_files:
        try:
            seq = MultimodalSequence(seq_file.read_text(encoding="utf-8"))
        except Exception as e:
            logger.warning(f"Could not load sequence {seq_file.stem}: {e}")
            continue
        sequences.append(dict(id=seq_file.stem, seq=seq, preview=seq.text[:180]))
    return templates.TemplateResponse(request, "sequences.html", {"sequences": sequences})


@app.get("/sequence/{seq_id}", response_class=HTMLResponse)
async def show_sequence(request: Request, seq_id: int):
    """Reads the specified rendered MultimodalSequence and displays it."""
    file_path = get_seq_path() / f"{seq_id}.md"
    if not file_path.exists():
        raise HTTPException(404, f"Sequence {seq_id} does not exist.")
    sequence = MultimodalSequence(file_path.read_text(encoding="utf-8"))
    return templates.TemplateResponse(request, "sequence.html", {"sequence": sequence, "seq_id": seq_id})


def is_port_in_use(port: int, host: str = DEFAULT_HOST) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind((host, port))
            return False
        except OSError:
            return True


def run_server(host: str = DEFAULT_HOST, port: int = DEFAULT_PORT):
    """Starts the web UI server (blocking) unless the port is already in use."""
    if is_port_in_use(port, host):
        logger.info(f"Port {port} is in use already, assuming the ezMM UI is running.")
        return
    print(f"ezMM UI running at http://{'localhost' if host in ('127.0.0.1', '0.0.0.0') else host}:{port}")
    uvicorn.run(app, host=host, port=port, log_level="warning")


if __name__ == "__main__":
    run_server()
