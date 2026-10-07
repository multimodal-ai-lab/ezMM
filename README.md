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
- **Migration:** Registries created with ezMM < 0.6 are migrated automatically on first use (a backup `item_registry.v1.bak.db` is kept). Older ezMM versions cannot read migrated registries.

## Web UI
Install with `pip install ezmm[ui]` and run
```
python -m ezmm ui [--path REGISTRY_PATH] [--port 7878]
```
to browse all items of the registry (and sequences rendered via `seq.render()`) at http://localhost:7878.

## Feature Overview
- ✅ Image support
- ✅ Video support
- ✅ Audio support
- ✅ Support for any other file type
- ✅ Saving and organizing media in a database along with their origin URLs
- ✅ Rendering `MultimodalSequence` and browsing the registry in a web UI
- ✅ Duplication management: Identify and re-use duplicates
