import uvicorn
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.requests import Request

from ezmm import MultimodalSequence
from ezmm.ui.common import SEQ_PATH

app = FastAPI()

# Mount templates files (CSS, JS, etc. if needed)
app.mount("/templates", StaticFiles(directory="ezmm/ui/templates"), name="templates")
app.mount("/static", StaticFiles(directory="ezmm/ui/static"), name="static")
app.mount("/temp", StaticFiles(directory="temp"), name="temp")
app.mount("/in", StaticFiles(directory="in"), name="in")

# Jinja2 templates directory
templates = Jinja2Templates(directory="ezmm/ui/templates")


@app.get("/sequence/{seq_id}", response_class=HTMLResponse)
async def show_claim(request: Request, seq_id: int):
    """Reads the specified MultimodalSequence saved in static/sequences
    and displays it."""
    file_path = SEQ_PATH / f"{seq_id}.md"
    seq_str = file_path.read_text()
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
        seq_str = seq_file.read_text()
        seq = MultimodalSequence(seq_str)
        sequences[seq_id] = seq

    return templates.TemplateResponse("overview.html", {
        "request": request,
        "sequences": sequences
    })


def run_server():
    uvicorn.run("ezmm.ui.main:app", host="0.0.0.0", port=8000, reload=True)


if __name__ == "__main__":
    run_server()
