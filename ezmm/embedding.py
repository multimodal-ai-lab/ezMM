"""Embeds text and items of any kind (images, videos, audios, and files) into one shared
vector space using Google DeepMind's EmbeddingGemma 2. The model is loaded lazily on first use."""
import importlib.util
import logging
import mimetypes
import os
import subprocess
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from itertools import islice
from pathlib import Path
from typing import Iterable, Iterator, Callable

import cv2
import imageio_ffmpeg
import numpy as np
from PIL import Image as PillowImageModule
from PIL.Image import Image as PillowImage

from ezmm.common.vector_index import truncate

logger = logging.getLogger("ezMM")

MODEL_NAME = "google/embeddinggemma-2"
MODEL_DIM = 768  # Native dimension of the model's embeddings (stored in full in the registry)
SUPPORTED_DIMS = (128, 256, 512, 768)  # Matryoshka dimensions the model was trained for
DEFAULT_DIM = 256  # Default dimension of embeddings (close to lossless, 3x smaller than native)

AUDIO_SAMPLE_RATE = 16000  # The model expects mono audio at 16 kHz
MAX_AUDIO_SECONDS = 300  # The context fits ~327 seconds of audio
VIDEO_FPS = 1  # Frames per second sampled from videos (the model's default)
MAX_VIDEO_FRAMES = 32  # Longer videos get sampled evenly
MAX_PDF_PAGES = 8  # PDFs are embedded as images of their first pages
MAX_TEXT_BYTES = 64 * 1024  # Text files are truncated to this size before embedding
MAX_IMAGE_PIXELS = 1024 * 1024  # Larger images get downscaled (the model uses at most ~650k pixels)
MAX_FRAME_PIXELS = 640 * 640  # Larger video frames get downscaled (the model uses at most ~320k pixels)

# Bulk embedding: number of threads decoding files, files per chunk, and inputs per model call
N_WORKERS = min(16, (os.cpu_count() or 1) + 4)
CHUNK_SIZE = 64
BATCH_SIZES = {"text": 64, "image": 16, "audio": 8, "video": 2}

# Text prompts (task instructions) the model was trained with
QUERY_PROMPT = "SearchQuery"
DOCUMENT_PROMPT = "Document"

INSTALL_HINT = "Embeddings require ezMM's optional dependencies. Install them with `pip install ezmm[embed]`."

_embedding_dim = int(os.getenv("EZMM_EMBEDDING_DIM") or DEFAULT_DIM)
_index_device = os.getenv("EZMM_INDEX_DEVICE")  # Determined on first use if not set
_model = None
_model_lock = threading.Lock()
_pdfium_lock = threading.Lock()


def is_available() -> bool:
    """Returns True iff the optional dependencies for embeddings (`ezmm[embed]`) are installed."""
    return all(importlib.util.find_spec(module) is not None
               for module in ("torch", "torchvision", "transformers", "sentence_transformers", "pypdfium2"))


def get_device() -> str:
    """Returns the device to run the model on: the one set via the EZMM_DEVICE environment
    variable (e.g., 'cpu' or 'cuda:1') or, by default, the GPU if available. AMD GPUs
    (with a ROCm build of PyTorch) are used just like NVIDIA GPUs."""
    import torch
    return os.getenv("EZMM_DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")


def get_embedding_dim() -> int:
    """Returns the dimension of the embeddings returned by ezMM (e.g., `item.embedding`)."""
    return _embedding_dim


def set_embedding_dim(dim: int):
    """Sets the dimension of the embeddings returned by ezMM and used for search: one of 128,
    256 (default), 512, or 768. Smaller embeddings are faster to compare and need less memory,
    at some cost of quality. Can also be set via the EZMM_EMBEDDING_DIM environment variable.
    The registry always stores the full embeddings, so changing the dimension requires no
    re-embedding."""
    global _embedding_dim
    if dim not in SUPPORTED_DIMS:
        raise ValueError(f"Unsupported embedding dimension {dim}. Choose one of {SUPPORTED_DIMS}.")
    _embedding_dim = dim


def get_index_device() -> str:
    """Returns the device holding the search index: 'cpu' (RAM) or a GPU, e.g., 'cuda' (VRAM).
    Unless set by the user, it is determined on first use: the GPU if available, else RAM."""
    global _index_device
    if _index_device is None:
        _index_device = "cpu"
        if is_available():
            import torch
            if torch.cuda.is_available():
                _index_device = "cuda"
                logger.info(f"GPU found ({torch.cuda.get_device_name()}): keeping the search index in GPU "
                            f"memory. To keep it in RAM instead, call ezmm.embedding.set_index_device('cpu') "
                            f"or set the environment variable EZMM_INDEX_DEVICE=cpu.")
    return _index_device


def set_index_device(device: str):
    """Sets where the search index (all embeddings of the registry) is kept: 'cpu' for RAM
    (as float32) or 'cuda' (or 'cuda:1', etc.) for GPU memory (as float16, faster search).
    Can also be set via the EZMM_INDEX_DEVICE environment variable. Default: the GPU if
    available, else RAM."""
    global _index_device
    if device != "cpu" and not device.startswith("cuda"):
        raise ValueError(f"Unsupported index device '{device}'. Use 'cpu' (RAM) or 'cuda' (GPU memory).")
    _index_device = device


def get_model():
    """Returns the (lazily loaded) EmbeddingGemma 2 model."""
    global _model
    with _model_lock:
        if _model is None:
            if not is_available():
                raise ImportError(INSTALL_HINT)
            import torch
            from sentence_transformers import SentenceTransformer
            device = get_device()
            # Use bfloat16 on GPUs with support for it, float32 elsewhere (float16 yields NaNs)
            gpu = device.startswith("cuda")
            dtype = torch.bfloat16 if gpu and torch.cuda.is_bf16_supported() else torch.float32
            gpu_name = f" ({torch.cuda.get_device_name(device)})" if gpu else ""
            logger.info(f"Loading embedding model {MODEL_NAME} on {device}{gpu_name}...")
            _model = SentenceTransformer(MODEL_NAME, device=device, model_kwargs={"dtype": dtype})
        return _model


def _encode(inputs: list, prompt_name: str = None, **kwargs) -> np.ndarray:
    """Embeds the inputs (in the model's input format) into normalized float32 vectors."""
    embeddings = get_model().encode(inputs, prompt_name=prompt_name, normalize_embeddings=True,
                                    convert_to_numpy=True, show_progress_bar=False, **kwargs)
    # Normalize again in float32, as models running in bfloat16 return only roughly normalized vectors
    embeddings = np.asarray(embeddings, dtype=np.float32)
    return embeddings / np.linalg.norm(embeddings, axis=-1, keepdims=True)


def normalize(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float32)
    norm = np.linalg.norm(vector)
    return vector / norm if norm > 0 else vector


def cos_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Computes the cosine similarity between two embedding vectors. Vectors of different
    dimensions are compared in the smaller dimension (Matryoshka truncation)."""
    dim = min(len(a), len(b))
    return float(np.dot(truncate(a, dim), truncate(b, dim)))


# -------------------------------------------------------------------------------------------------
# Text and images

def embed_text(text: str | list[str], prompt_name: str = DOCUMENT_PROMPT, dim: int = None) -> np.ndarray:
    """Embeds the text(s), by default as a document (corpus entry). Use `embed_query()`
    to embed search queries. All embedding functions return embeddings of the configured
    dimension (see `set_embedding_dim()`) unless `dim` is given."""
    if isinstance(text, str):
        return truncate(_encode([text], prompt_name)[0], dim or _embedding_dim)
    return truncate(_encode(list(text), prompt_name), dim or _embedding_dim)


def embed_query(query: str, dim: int = None) -> np.ndarray:
    """Embeds a text search query, to be compared with item and document embeddings."""
    return embed_text(query, prompt_name=QUERY_PROMPT, dim=dim)


def embed(pillow_images: PillowImage | Iterable[PillowImage], dim: int = None) -> np.ndarray | list[np.ndarray]:
    """Embeds one or multiple Pillow images."""
    if isinstance(pillow_images, PillowImage):
        return truncate(_encode([{"image": pillow_images}])[0], dim or _embedding_dim)
    return list(truncate(_encode([{"image": image} for image in pillow_images]), dim or _embedding_dim))


# -------------------------------------------------------------------------------------------------
# Files

def kind_of_file(path: Path | str, mime_type: str = None) -> str:
    """Guesses the ezMM item kind (image, video, audio, or file) of the file from its
    extension or, if unknown, from the given MIME type."""
    mime_type = mimetypes.guess_type(Path(path).name)[0] or mime_type or ""
    kind = mime_type.split("/")[0]
    return kind if kind in ("image", "video", "audio") else "file"


def embed_file(path: Path | str, kind: str = None, dim: int = None) -> np.ndarray:
    """Embeds the file at the given path as the given item kind (guessed if not specified)."""
    result = next(embed_files([(path, kind)], dim=dim))
    if isinstance(result, Exception):
        raise result
    return result


def prepare_input(path: Path | str, kind: str = None) -> str | dict:
    """Decodes the file into the model's input format (CPU-bound, thread-safe):
    - images: the image,
    - videos: frames sampled at 1 fps together with the audio track (if any),
    - audios: the (mono, 16 kHz) audio,
    - other files: PDFs as images of their first pages, text files by their text,
      and any other (binary) file by its file name."""
    if path is None or not Path(path).exists():
        raise FileNotFoundError(f"File '{path}' does not exist.")
    path = Path(path)
    kind = kind or kind_of_file(path)

    if kind == "image":
        return {"image": load_image(path)}

    if kind == "video":
        frames = sample_video_frames(path)
        if not frames:
            raise ValueError(f"Cannot compute embedding without video frames: {path}")
        audio = load_audio(path)
        if audio is None or len(audio) == 0:
            return {"video": np.stack(frames)}
        return {"text": "<|video|><|audio|>", "video": np.stack(frames),
                "audio": {"array": audio, "sampling_rate": AUDIO_SAMPLE_RATE}}

    if kind == "audio":
        audio = load_audio(path)
        if audio is None or len(audio) == 0:
            raise ValueError(f"Cannot compute embedding of an audio without samples: {path}")
        return {"audio": {"array": audio, "sampling_rate": AUDIO_SAMPLE_RATE}}

    if path.suffix.lower() == ".pdf":
        pages = render_pdf_pages(path)
        if pages:
            return {"text": "<|image|>" * len(pages), "image": pages}
    text = read_text(path)
    return f"title: {path.name} | text: {text if text and text.strip() else 'none'}"


def embed_files(files: Iterable[tuple[Path | str, str | None]], n_workers: int = N_WORKERS,
                chunk_size: int = CHUNK_SIZE, dim: int = None) -> Iterator[np.ndarray | Exception]:
    """Embeds the files, given as (path, kind) pairs, efficiently: files are decoded by a pool of
    threads while the model embeds the previously decoded chunk of files, in batches of inputs
    of the same modality. Yields one embedding per file, in order, or the exception that
    occurred for that file."""
    if not is_available():
        raise ImportError(INSTALL_HINT)
    files = iter(files)
    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        def submit_next_chunk():
            return [pool.submit(_prepare_or_error, *file) for file in islice(files, chunk_size)]

        pending = submit_next_chunk()
        while pending:
            inputs = [future.result() for future in pending]
            pending = submit_next_chunk()  # Decode the next chunk while this one gets embedded
            for result in _encode_batched(inputs):
                yield result if isinstance(result, Exception) else truncate(result, dim or _embedding_dim)


def _prepare_or_error(path: Path | str, kind: str = None) -> str | dict | Exception:
    try:
        return prepare_input(path, kind)
    except Exception as e:
        return e


def _encode_batched(inputs: list[str | dict | Exception]) -> list[np.ndarray | Exception]:
    """Embeds the (prepared) inputs, grouped by modality. Exceptions are passed through."""
    results = list(inputs)
    groups = defaultdict(list)
    for i, model_input in enumerate(inputs):
        if not isinstance(model_input, Exception):
            groups[("text",) if isinstance(model_input, str) else tuple(sorted(model_input))].append(i)
    for modalities, indices in groups.items():
        # Video frames are sampled already, so the model's processor must not sample them again
        kwargs = {"processing_kwargs": {"video": {"do_sample_frames": False}}} if "video" in modalities else {}
        batch_size = min(BATCH_SIZES[m] for m in modalities)
        try:
            vectors = _encode([inputs[i] for i in indices], batch_size=batch_size, **kwargs)
            for i, vector in zip(indices, vectors):
                results[i] = vector
        except Exception:
            # Embed one by one to isolate the failing input(s)
            for i in indices:
                try:
                    results[i] = _encode([inputs[i]], **kwargs)[0]
                except Exception as e:
                    results[i] = e
    return results


def embed_registry(kind: str = None, n_workers: int = N_WORKERS, chunk_size: int = CHUNK_SIZE,
                   on_progress: Callable[[int, int, int], None] = None) -> dict:
    """Embeds all items of the registry (optionally only of the given kind) that are not
    embedded yet, see `embed_files()`. Calls `on_progress(done, total, failed)` after each
    chunk. Returns the number of embedded and failed items."""
    from ezmm.common.registry import item_registry
    todo = sorted(item_registry.list_unembedded(MODEL_NAME, kind=kind))  # Group kinds for larger batches
    files = ((item_registry.get_row(k, identifier)["path"], k) for k, identifier in todo)
    done, failed, buffer = 0, 0, []
    for (k, identifier), result in zip(todo, embed_files(files, n_workers, chunk_size, dim=MODEL_DIM)):
        done += 1
        if isinstance(result, Exception):
            failed += 1
            logger.warning(f"Could not embed <{k}:{identifier}>: {result}")
            if isinstance(result, FileNotFoundError):
                item_registry.set_missing(k, identifier, True)
        else:
            buffer.append((k, identifier, result))
        if done % chunk_size == 0 or done == len(todo):
            item_registry.set_embeddings(MODEL_NAME, buffer)
            buffer = []
            if on_progress:
                on_progress(done, len(todo), failed)
    return dict(embedded=done - failed, failed=failed)


# -------------------------------------------------------------------------------------------------
# Decoding

def load_image(path: Path | str, max_pixels: int = MAX_IMAGE_PIXELS) -> PillowImage:
    """Loads the image in RGB mode, downscaled to at most `max_pixels` pixels (the model's
    processor downscales larger images anyway, so this saves memory and time)."""
    with PillowImageModule.open(path) as image:
        image.draft("RGB", _fit(image.size, max_pixels))  # Fast downscaled decoding of JPEGs
        image = image.convert("RGB")
    if image.width * image.height > max_pixels:
        image = image.resize(_fit(image.size, max_pixels), PillowImageModule.Resampling.BICUBIC)
    return image


def _fit(size: tuple[int, int], max_pixels: int) -> tuple[int, int]:
    """Returns the (aspect-preserving) size with at most `max_pixels` pixels."""
    width, height = size
    scale = min(1.0, (max_pixels / (width * height)) ** 0.5)
    return max(1, int(width * scale)), max(1, int(height * scale))


def sample_video_frames(path: Path | str, fps: float = VIDEO_FPS,
                        max_frames: int = MAX_VIDEO_FRAMES) -> list[np.ndarray]:
    """Returns RGB frames sampled from the video at the given rate (evenly sampled
    if the video has more than `max_frames` frames at that rate)."""
    cap = cv2.VideoCapture(str(path))
    try:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_fps = cap.get(cv2.CAP_PROP_FPS) or 0
        if total_frames <= 0:
            return []
        duration = total_frames / video_fps if video_fps > 0 else 1
        n_frames = int(min(max(round(duration * fps), 1), max_frames, total_frames))
        frames = []
        for frame_id in np.linspace(0, total_frames - 1, n_frames, dtype=int):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_id))
            success, frame = cap.read()
            if not success or frame is None:
                break
            height, width = frame.shape[:2]
            if width * height > MAX_FRAME_PIXELS:
                frame = cv2.resize(frame, _fit((width, height), MAX_FRAME_PIXELS), interpolation=cv2.INTER_AREA)
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        return frames
    finally:
        cap.release()


def load_audio(path: Path | str, max_seconds: float = MAX_AUDIO_SECONDS) -> np.ndarray | None:
    """Decodes the (first `max_seconds` of the) audio of the file to mono float32 samples
    at 16 kHz using FFmpeg. Returns None if the file has no audio stream."""
    cmd = [imageio_ffmpeg.get_ffmpeg_exe(), "-nostdin", "-v", "error", "-i", str(path), "-vn",
           "-t", str(max_seconds), "-ac", "1", "-ar", str(AUDIO_SAMPLE_RATE), "-f", "f32le", "-"]
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        if b"does not contain any stream" in result.stderr or b"matches no streams" in result.stderr:
            return None
        raise OSError(f"FFmpeg could not decode the audio of {path}: {result.stderr.decode(errors='ignore')}")
    return np.frombuffer(result.stdout, dtype=np.float32)


def render_pdf_pages(path: Path | str, max_pages: int = MAX_PDF_PAGES) -> list[PillowImage]:
    """Renders the first pages of the PDF as images. Returns an empty list if the PDF cannot be read."""
    import pypdfium2
    with _pdfium_lock:  # PDFium is not thread-safe
        try:
            pdf = pypdfium2.PdfDocument(str(path))
        except pypdfium2.PdfiumError as e:
            logger.warning(f"Could not read PDF {path}: {e}")
            return []
        try:
            return [pdf[i].render(scale=1.5).to_pil().convert("RGB") for i in range(min(len(pdf), max_pages))]
        finally:
            pdf.close()


def read_text(path: Path | str, max_bytes: int = MAX_TEXT_BYTES) -> str | None:
    """Returns the (beginning of the) file's content if it is a text file, else None."""
    with open(path, "rb") as f:
        data = f.read(max_bytes)
    if b"\x00" in data:
        return None  # Binary file
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError as e:
        if e.start >= len(data) - 3:
            return data[:e.start].decode("utf-8")  # Truncated in the middle of a character
        return None
