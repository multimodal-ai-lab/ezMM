"""Local web UI for browsing the ezMM item registry and rendered MultimodalSequences.
Start it with `python -m ezmm ui` (requires `pip install ezmm[ui]`)."""
import logging
import math
import socket
import tempfile
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse, urlencode

import uvicorn
from fastapi import FastAPI, HTTPException, UploadFile, Form
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import HTMLResponse, FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.requests import Request

from ezmm import MultimodalSequence
from ezmm.common import item_registry
from ezmm.common.items import KINDS, KIND2ITEM
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


def _time_ago(timestamp: Optional[str]) -> str:
    if not timestamp:
        return ""
    then = datetime.fromisoformat(timestamp)
    if then.tzinfo is None:
        then = then.replace(tzinfo=timezone.utc)
    seconds = (datetime.now(timezone.utc) - then).total_seconds()
    for unit, length in (("y", 31536000), ("mo", 2592000), ("d", 86400), ("h", 3600), ("min", 60)):
        if seconds >= length:
            return f"{int(seconds // length)} {unit} ago"
    return "just now"


def _host(url: str) -> str:
    parsed = urlparse(url)
    return parsed.netloc or ("local file" if parsed.scheme == "file" else url)


def _duration(seconds: float) -> str:
    seconds = int(round(seconds or 0))
    return f"{seconds // 60}:{seconds % 60:02d}"


def _browse_url(kind: str = "", q: str = "", missing: bool = False, page: int = 1) -> str:
    """Returns the URL of the browse page with the given filters (omitting defaults)."""
    params = {"kind": kind, "q": q, "missing": 1 if missing else None, "page": page if page > 1 else None}
    query = urlencode({key: value for key, value in params.items() if value})
    return "/?" + query if query else "/"


def _search_url(q: str = "", kind: str = "", like: str = "") -> str:
    """Returns the URL of the search page with the given query and filter (omitting defaults)."""
    query = urlencode({key: value for key, value in dict(q=q, kind=kind, like=like).items() if value})
    return "/search?" + query if query else "/search"


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


@app.get("/", response_class=HTMLResponse)
async def browse(request: Request, kind: str = "", q: str = "", page: int = 1, missing: bool = False):
    """Shows a grid of all items in the registry, filterable by kind and search query.
    Items whose file is missing are hidden unless `missing` is set. Filtering relies on the
    registry's `missing` flags, which get corrected for all items shown on the page."""
    kind = kind if kind in KINDS else ""
    for _ in range(2):  # Repeat once if the flags of shown items turned out to be outdated
        total_all = item_registry.count_items(kind or None, q or None)
        total = total_all if missing else item_registry.count_items(kind or None, q or None, include_missing=False)
        n_pages = max(math.ceil(total / PAGE_SIZE), 1)
        page = min(max(page, 1), n_pages)
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

    for entry in entries:
        source_urls = item_registry.get_source_urls(entry["kind"], entry["id"])
        entry["source_url"] = source_urls[0] if source_urls else None
        entry["suffix"] = entry["path"].suffix.lstrip(".").upper() if entry["path"] else ""

    stats = item_registry.stats()
    return templates.TemplateResponse(request, "browse.html", {
        "entries": entries,
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

    details, error = {}, None
    try:
        item = item_registry.get(kind=kind, identifier=identifier)
        details["MIME type"] = item.mime_type
        if kind in ("image", "video"):
            details["Dimensions"] = f"{item.width} × {item.height} px"
        if kind == "video":
            details["Duration"] = _duration(item.duration)
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
        self.thread: Optional[threading.Thread] = None
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


def _render_search(request: Request, kind: str = "", q: str = "", like: str = "",
                   query_vector=None, query_label: str = "", error: str = None):
    from ezmm.embedding import MODEL_NAME, embed_query, is_available, INSTALL_HINT
    kind = kind if kind in KINDS else ""
    results, exclude = [], None
    if not is_available():
        q, like, query_vector, error = "", "", None, INSTALL_HINT
    try:
        if like:
            exclude = parse_ref(f"<{like}>")
            item = item_registry.get(kind=exclude[0], identifier=exclude[1])
            if item is None:
                raise HTTPException(404, f"Item <{like}> does not exist.")
            query_vector, query_label = item.embedding, item.reference
        elif q:
            query_vector, query_label = embed_query(q), f"“{q}”"
        if query_vector is not None:
            results = item_registry.search(query_vector, MODEL_NAME, kind=kind or None,
                                           limit=PAGE_SIZE, exclude=exclude)
    except HTTPException:
        raise
    except Exception as e:
        error = str(e)
        logger.warning(f"Search failed: {e}")

    for entry in results:
        source_urls = item_registry.get_source_urls(entry["kind"], entry["id"])
        entry["source_url"] = source_urls[0] if source_urls else None
        entry["suffix"] = entry["path"].suffix.lstrip(".").upper() if entry["path"] else ""
    return templates.TemplateResponse(request, "search.html", {
        "results": results,
        "kind": kind,
        "q": q,
        "like": like,
        "query_label": query_label,
        "searched": query_vector is not None,
        "error": error,
        "n_unindexed": item_registry.count_unembedded(MODEL_NAME),
        "n_indexed": item_registry.count_embedded(MODEL_NAME),
        "indexer": indexer,
    })


@app.get("/search", response_class=HTMLResponse)
async def search(request: Request, q: str = "", kind: str = "", like: str = ""):
    """Semantic search over all items with a text query (`q`) or by an existing item (`like`,
    e.g., `image:3`). Only items that are embedded already are searched."""
    return await run_in_threadpool(_render_search, request, kind=kind, q=q.strip(), like=like)


@app.post("/search", response_class=HTMLResponse)
async def search_by_file(request: Request, file: UploadFile, kind: str = Form("")):
    """Semantic search over all items with an uploaded file of any kind as the query.
    The file is embedded on the fly and not added to the registry."""
    from ezmm.embedding import embed_file, kind_of_file, is_available
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
    return await run_in_threadpool(_render_search, request, kind=kind, query_vector=vector,
                                   query_label=file.filename or "uploaded file")


@app.post("/search/index")
async def index_items(kind: str = Form(""), q: str = Form(""), like: str = Form("")):
    """Starts embedding all not yet embedded items in the background."""
    from ezmm.embedding import is_available
    if is_available():
        indexer.start()
    return RedirectResponse(_search_url(q, kind, like), status_code=303)


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
        except socket.error:
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
