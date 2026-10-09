"""In-memory index of (normalized) embedding vectors for fast similarity search, located
either in RAM (device 'cpu', float32) or in GPU memory (device 'cuda', float16, requires PyTorch)."""
from typing import Hashable, Iterable

import numpy as np


def truncate(vectors: np.ndarray, dim: int) -> np.ndarray:
    """Returns the first `dim` dimensions of the (Matryoshka) vector(s), normalized, as float32."""
    vectors = np.asarray(vectors, dtype=np.float32)[..., :dim]
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    return vectors / np.where(norms > 0, norms, 1)


class VectorIndex:
    """Holds one vector of dimension `dim` per key: float32 in RAM (CPUs compute float16 slowly)
    and float16 in GPU memory (half the memory, fast on GPUs). Added vectors are truncated to
    `dim` and normalized; they get merged into the index lazily on the next query."""

    def __init__(self, dim: int, device: str = "cpu"):
        self.dim = dim
        self.device = device
        self.dtype = np.float32 if device == "cpu" else np.float16
        self.keys: list[Hashable] = []
        self._positions: dict[Hashable, int] = {}
        self._matrix = self._to_device(np.empty((0, dim), dtype=self.dtype))
        self._pending_keys: list[Hashable] = []
        self._pending_vectors: list[np.ndarray] = []

    @classmethod
    def build(cls, dim: int, device: str, n: int,
              batches: Iterable[tuple[list[Hashable], np.ndarray]]) -> "VectorIndex":
        """Builds the index from batches of keys and vectors with `n` vectors in total,
        without holding more than one copy of the matrix in memory."""
        index = cls(dim, device)
        matrix = np.empty((n, dim), dtype=index.dtype)
        for keys, vectors in batches:
            start = len(index.keys)
            matrix[start:start + len(keys)] = truncate(vectors, dim)
            for key in keys:
                index._positions[key] = len(index.keys)
                index.keys.append(key)
        index._matrix = index._to_device(matrix[:len(index.keys)])
        return index

    def add(self, keys: list[Hashable], vectors: np.ndarray):
        """Adds (or replaces) the vectors of the given keys."""
        if keys:
            self._pending_keys.extend(keys)
            self._pending_vectors.append(truncate(vectors, self.dim).astype(self.dtype))

    def __len__(self) -> int:
        self._merge()
        return len(self.keys)

    def scores(self, query: np.ndarray) -> np.ndarray:
        """Returns the cosine similarity of the query to each indexed vector (in the order of `keys`)."""
        self._merge()
        query = truncate(query, self.dim)
        if self.device == "cpu":
            return self._matrix @ query
        import torch
        query = torch.from_numpy(query).to(self.device, dtype=torch.float16)
        return (self._matrix @ query).float().cpu().numpy()

    def _merge(self):
        if not self._pending_keys:
            return
        keys, vectors = self._pending_keys, np.concatenate(self._pending_vectors)
        self._pending_keys, self._pending_vectors = [], []
        new_rows = {}
        for key, vector in zip(keys, vectors):
            position = self._positions.get(key)
            if position is None:
                new_rows[key] = vector  # Later duplicates of a key overwrite earlier ones
            else:
                self._matrix[position] = self._to_device(vector)
        if new_rows:
            for key in new_rows:
                self._positions[key] = len(self.keys)
                self.keys.append(key)
            new_matrix = self._to_device(np.stack(list(new_rows.values())))
            if self.device == "cpu":
                self._matrix = np.concatenate([self._matrix, new_matrix])
            else:
                import torch
                self._matrix = torch.cat([self._matrix, new_matrix])

    def _to_device(self, array: np.ndarray):
        if self.device == "cpu":
            return array
        import torch
        return torch.from_numpy(np.ascontiguousarray(array)).to(self.device)
