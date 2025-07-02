import socket
import uvicorn
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.requests import Request

from ezmm import MultimodalSequence
from ezmm.ui.common import SEQ_PATH, TEMP_PATH, PROJECT_ROOT

app = FastAPI()

# Mount templates files (CSS, JS, etc. if needed)
app.mount("/templates", StaticFiles(directory=PROJECT_ROOT / "ezmm/ui/templates"), name="templates")
app.mount("/static", StaticFiles(directory=PROJECT_ROOT / "ezmm/ui/static"), name="static")
app.mount("/temp", StaticFiles(directory=TEMP_PATH), name="temp")
app.mount("/in", StaticFiles(directory=PROJECT_ROOT / "in"), name="in")

# Jinja2 templates directory
templates = Jinja2Templates(directory=PROJECT_ROOT / "ezmm/ui/templates")


@app.get("/sequence/{seq_id}", response_class=HTMLResponse)
async def show_claim(request: Request, seq_id: int):
    """Reads the specified MultimodalSequence saved in static/sequences
    and displays it."""
    file_path = SEQ_PATH / f"{seq_id}.md"
    seq_str = file_path.read_text(encoding="utf-8")
    sequence = MultimodalSequence(seq_str)

    return templates.TemplateResponse("sequence.html", {
        "request": request,
        "sequence": sequence,
        "seq_id": seq_id,
    })


@app.get("/", response_class=HTMLResponse)
async def overview(request: Request):
    """Shows a list of all saved MultimodalSequences."""
    sequences = {}
    for seq_file in SEQ_PATH.glob("*.md"):
        seq_id = seq_file.stem
        seq_str = seq_file.read_text(encoding="utf-8")
        seq = MultimodalSequence(seq_str)
        sequences[seq_id] = seq

    return templates.TemplateResponse("overview.html", {
        "request": request,
        "sequences": sequences
    })


def is_port_in_use(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(('0.0.0.0', port))
            return False
        except socket.error:
            return True


def run_server():
    if not is_port_in_use(7878):
        uvicorn.run("ezmm.ui.main:app", host="0.0.0.0", port=7878, reload=True)


if __name__ == "__main__":
    run_server()
