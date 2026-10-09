# ezMM: Mini-Suite for Easy Multimodal Data Processing
This lightweight Python package aims to streamline and simplify the processing of multimodal data. The core philosophy of ezMM is to treat any data (whether strings, images, audios, tables, etc.) as a **multimodal sequence**.

## Usage
Core is the `MultimodalSequence` class. Here is an example:
```python
from ezmm import MultimodalSequence, Image

img1 = Image("in/roses.jpg")
img2 = Image("in/garden.jpg")

seq = MultimodalSequence("The image", img1, "shows two beautiful roses while",
                         img2, "shows a nice garden with many flowers.")
```

`seq` comprehensively aggregates the different modalities into one handy object. It also offers some useful features:

### `MultimodalSequence` is stringifyable
```python
print(seq)
```
will return
```
The image <image:1> shows two beautiful roses while <image:2> shows a nice garden with many flowers.
```
That is, non-string items in the `MultimodalSequence` get replaced by their unique reference when turned into strings. 

### `MultimodalSequence` understands references
Conversely, you can do
```python
seq2 = MultimodalSequence("The image <image:1> shows two beautiful roses while <image:2> shows a nice garden with many flowers.")
```
which obeys `seq == seq2`. That is, `MultimodalSequence` resolves references within the input string and loads the corresponding items under the hood.

### Access `MultimodalSequence` like a list
You can apply list comprehension to `seq`. For example,
`seq[1] == img`.

### Easy modality checks
You can check for specific modalities like images quickly, e.g., with `seq.has_images()`.

### Get the text and manipulate the sequence
`seq.text` returns all text parts (without items) as a single string. You can manipulate a `MultimodalSequence` like a list via `append()`, `extend()`, `insert()`, `remove()`, `pop()`, `seq[i] = ...`, `del seq[i]`, `+` and `+=`. Nested sequences are always flattened automatically, so a `MultimodalSequence` never contains another one.

## Item Types
| Class   | Reference    | For                                               |
|---------|--------------|---------------------------------------------------|
| `Image` | `<image:1>`  | Images (JPEG, PNG, AVIF, ...)                     |
| `Video` | `<video:1>`  | Videos (MP4, ...)                                 |
| `Audio` | `<audio:1>`  | Audio-only files (MP3, WAV, FLAC, OGG, M4A, ...)  |
| `File`  | `<file:1>`   | Any other file (PDF, Excel sheets, archives, ...) |

All items can be created from a file path or from binary data, e.g., `Audio(binary_data=data, mime_type="audio/mpeg")`. `File`s larger than 100 MB are rejected with a `FileTooLargeError`; change the limit with `set_max_file_size(n_bytes)` (or `None` for no limit) or the `EZMM_MAX_FILE_SIZE` environment variable.

## Item Registry
ezMM keeps track of all items in a registry (an SQLite DB plus media files) located at `temp/` or at the path specified by the `EZMM` environment variable (or `set_ezmm_path()`).
- **Deduplication:** Identical files (same kind and byte-identical content, incl. embedded metadata) collapse to the same item and reference. Each item keeps all its source URLs (`item.source_urls`). To find an item by URL, use `item_registry.get_by_source_url(url)`.
- **Cleanup:** `python -m ezmm dedup [--dry-run]` removes duplicates from existing registries. References to removed duplicates keep resolving to the remaining item.
- **Missing files:** The registry remembers which items' files are missing (used by the web UI to hide them). Run `python -m ezmm check` to re-check all files, e.g., after moving or deleting files outside of ezMM.
- **Migration:** Registries created with older ezMM versions are migrated automatically on first use (a backup `item_registry.v<version>.bak.db` is kept). Older ezMM versions cannot read migrated registries.

## Embeddings
Every item and every `MultimodalSequence` can be embedded into one shared vector space with Google DeepMind's [EmbeddingGemma 2](https://huggingface.co/google/embeddinggemma-2) (normalized vectors, 256 dimensions by default). Embeddings are optional, install them with
```
pip install ezmm[embed]
```
```python
from ezmm import Image, Video, MultimodalSequence
from ezmm.embedding import embed_query

img = Image("in/roses.jpg")
vid = Video("in/snow.mp4")
img.embedding                            # np.ndarray of shape (256,)
img.cos_sim(vid)                         # Cosine similarity between any items, sequences, or vectors
img.cos_sim(embed_query("red flowers"))  # Compare with a text search query

seq = MultimodalSequence("The image", img, "shows roses.")
seq.embedding                            # Average of the embeddings of the full text and of all items
```
| Kind    | Embedded as                                                                                  |
|---------|----------------------------------------------------------------------------------------------|
| `Image` | the image                                                                                    |
| `Video` | frames sampled at 1 fps (max. 32) together with the audio track                              |
| `Audio` | the first 5 minutes of the audio (mono, 16 kHz)                                              |
| `File`  | PDFs: images of the first 8 pages; text files: their text; other files: their file name      |

The model (~1.5 GB) is downloaded and loaded on first use. Item embeddings are stored in the item registry, so each item gets embedded only once. Run `python -m ezmm embed` to embed all items of the registry at once, e.g., to make them searchable in the web UI.

For many items, use the bulk functions (also used by `python -m ezmm embed`): `embed_registry()` embeds all items of the registry that are not embedded yet, and `embed_files([(path, kind), ...])` embeds any files. They decode files in parallel threads while the model embeds the previously decoded files in batches of the same modality, which is several times faster than embedding items one by one.

### Embedding size and search index
| Setting                | Function                                | Environment variable  | Default                     |
|------------------------|-----------------------------------------|-----------------------|-----------------------------|
| Embedding dimension    | `ezmm.embedding.set_embedding_dim(dim)` | `EZMM_EMBEDDING_DIM`  | `256` (or `128`, `512`, `768`) |
| Location of the index  | `ezmm.embedding.set_index_device(dev)`  | `EZMM_INDEX_DEVICE`   | `cuda` (VRAM) if a GPU is available, else `cpu` (RAM) |

The model produces 768-dimensional [Matryoshka](https://arxiv.org/abs/2205.13147) embeddings, which can be truncated to fewer dimensions at little cost of quality (close to lossless down to 256). Truncation keeps the leading dimensions and re-normalizes, exactly like Sentence Transformers' `truncate_dim`. The registry always stores the full embeddings (as float16), so the dimension can be changed at any time without re-embedding. For search, all embeddings are kept in an in-memory index in the configured dimension: in VRAM as float16 (about 1.5 GB for 3 million items at 256 dimensions) or in RAM as float32 (about 3 GB). Unless set otherwise, ezMM keeps the index in VRAM if a GPU is available.

### Running on a GPU
The model runs on the GPU automatically if PyTorch detects one (in bfloat16), otherwise on the CPU (in float32). Set the environment variable `EZMM_DEVICE` (e.g., `cpu` or `cuda:1`) to choose the device explicitly. A GPU needs a matching PyTorch build:
- **NVIDIA:** Install PyTorch with CUDA support, see [pytorch.org](https://pytorch.org/get-started/locally/).
- **AMD Radeon (Windows and Linux):** Install AMD's ROCm build of PyTorch for your GPU architecture, see [AMD's install guide](https://rocm.docs.amd.com/projects/ai-ecosystem/en/latest/frameworks/pytorch/install.html). For example, for the RX 9000 series (`gfx1201`):
  ```
  pip install --index-url https://stable.repo.amd.com/rocm/whl-next/ "torch[device-gfx1201]==2.14.0+rocm10.1.0" "torchvision[device-gfx1201]==0.29.0a0+rocm10.1.0"
  ```
  ROCm GPUs are addressed like CUDA devices (`cuda`) in PyTorch, so no further configuration is needed.

## Web UI
Install with `pip install ezmm[ui]` and run
```
python -m ezmm ui [--path REGISTRY_PATH] [--port 7878]
```
to browse all items of the registry (and sequences rendered via `seq.render()`) at http://localhost:7878. The **Search** page (requires `ezmm[embed]`) finds items by meaning: search with a text query, with any file (image, video, audio, PDF, ...), or by an existing item ("Find similar items"). Items that are not embedded yet can be indexed directly from the search page.

## Feature Overview
- ✅ Image support
- ✅ Video support
- ✅ Audio support
- ✅ Support for any other file type
- ✅ Saving and organizing media in a database along with their origin URLs
- ✅ Rendering `MultimodalSequence` and browsing the registry in a web UI
- ✅ Duplication management: Identify and re-use duplicates
- ✅ Multimodal embeddings of all items and sequences with EmbeddingGemma 2
- ✅ Semantic search over the registry in the web UI (by text, file, or item)
