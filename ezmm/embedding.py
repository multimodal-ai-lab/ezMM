from typing import Iterable

import numpy as np
from PIL.Image import Image as PillowImage
from fastembed import ImageEmbedding

model = ImageEmbedding(model_name="Qdrant/clip-ViT-B-32-vision")


def embed(pillow_images: PillowImage | Iterable[PillowImage]) -> np.ndarray | list[np.ndarray]:
    embedded = list(model.embed(pillow_images))
    if isinstance(pillow_images, PillowImage):
        return embedded[0]
    return embedded
