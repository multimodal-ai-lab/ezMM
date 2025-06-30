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

## Feature Overview
- ✅ Image support
- ✅ Video support
- ✅ Saving and organizing media in a database along with their origin URL
- ✅ Rendering `MultimodalSequence` in a web UI
- ⏳ Duplication management: Identify and re-use duplicates
