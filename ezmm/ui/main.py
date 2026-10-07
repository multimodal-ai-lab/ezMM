"""Local web UI for browsing the ezMM item registry and rendered MultimodalSequences.
Start it with `python -m ezmm ui` (requires `pip install ezmm[ui]`)."""
import logging
import math
import socket
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.requests import Request

from ezmm import MultimodalSequence
from ezmm.common import item_registry
from ezmm.common.items import KINDS, KIND2ITEM
from ezmm.common.items.file import format_size
from ezmm.ui.common import get_seq_path

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


templates.env.filters.update(size=format_size, time_ago=_time_ago, host=_host, duration=_duration)
templates.env.globals.update(kinds=KINDS)


def _get_entry(kind: str, identifier: int) -> dict:
    if kind not in KIND2ITEM:
        raise HTTPException(404, f"Unknown item kind '{kind}'.")
    row = item_registry.get_row(kind, identifier)
    if row is None:
        raise HTTPException(404, f"Item <{kind}:{identifier}> does not exist.")
    return row


@app.get("/", response_class=HTMLResponse)
async def browse(request: Request, kind: str = "", q: str = "", page: int = 1):
    """Shows a grid of all items in the registry, filterable by kind and search query."""
    kind = kind if kind in KINDS else ""
    total = item_registry.count_items(kind or None, q or None)
    n_pages = max(math.ceil(total / PAGE_SIZE), 1)
    page = min(max(page, 1), n_pages)
    entries = item_registry.list_items(kind or None, q or None, offset=(page - 1) * PAGE_SIZE, limit=PAGE_SIZE)
    for entry in entries:
        source_urls = item_registry.get_source_urls(entry["kind"], entry["id"])
        entry["source_url"] = source_urls[0] if source_urls else None
        entry["suffix"] = entry["path"].suffix.lstrip(".").upper() if entry["path"] else ""
        entry["missing"] = not (entry["path"] and entry["path"].exists())

    stats = item_registry.stats()
    return templates.TemplateResponse(request, "browse.html", {
        "entries": entries,
        "kind": kind,
        "q": q,
        "page": page,
        "n_pages": n_pages,
        "total": total,
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
    return templates.TemplateResponse(request, "item.html", {
        "entry": entry,
        "details": details,
        "error": error,
        "file_exists": bool(entry["path"] and entry["path"].exists()),
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
        raise HTTPException(404, f"File of <{kind}:{identifier}> not found.")
    return FileResponse(path)


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
