"""Loading, shape-wrangling and standardization for paired V-JEPA / LTX-2 latents.

This is the subset of ``src/standard_adaptors/latents.py`` that the diffusion decoder actually
needs, copied so this directory stands on its own. The grid-*recovery* helpers
(``suggest_grids``, ``locality_score``) deliberately did not come along: those
belong to the one-off inspection step that establishes the patch geometry, and by
the time you are training a decoder that question is already settled and passed
in as ``--jepa-grid``.

The duplication is intentional and bounded. The alternative -- a ``sys.path``
insert into a sibling directory -- broke once already when the sibling was
renamed, and a decoder that stops importing because an unrelated
experiment moved is not worth the saved lines.

Note that ``train.py`` and ``sample.py`` reach this module as a plain ``from data
import ...``, which works because running a script puts its own directory first
on ``sys.path``. That is also why these files are run as
``python src/diffusion_adaptor/train.py`` rather than imported from elsewhere.

Storage
-------
One file per video -- ``embedding_{video_idx}.pt`` -- in a flat directory.
:class:`EmbeddingStore` hides that behind a ``key -> tensor`` mapping keyed by
filename stem, reading one file at a time so a directory never has to be
materialized whole. ``.npy``/``.npz``/``.pth`` are accepted too, since the early
dumps were saved that way.

**Pairing is by key.** ``embedding_42`` in the V-JEPA store is assumed to be the
same source video as ``embedding_42`` in the LTX-2 store. Nothing downstream can
detect a violation -- a consistent off-by-one would train on systematically
mismatched pairs and simply look like a negative result. Confirm it once by hand
before trusting any number this pipeline produces.

Shapes
------
* **V-JEPA** stores ``(N, D)`` per clip -- a flat bag of ``N`` spatiotemporal
  patches. The grid it was flattened from is not recorded, so it is supplied.
* **LTX-2 VAE** stores a 4-D latent ``(C, F, H, W)``, which is what the decoder
  wants back.

Both are canonicalized to ``(channels, F, H, W)``.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

FILE_SUFFIXES = (".pt", ".pth", ".npy", ".npz")


def _natural_key(key: str):
    """Sort ``embedding_9`` before ``embedding_10`` rather than after it."""
    match = re.search(r"(\d+)\s*$", key)
    return (0, int(match.group(1)), "") if match else (1, 0, key)


def load_any(path: Path) -> np.ndarray:
    """Read a whole-file embedding (the one-file-per-video layout)."""
    path = Path(path)
    if path.suffix == ".npy":
        return np.load(path, mmap_mode="r")
    if path.suffix == ".npz":
        with np.load(path) as z:
            keys = list(z.keys())
            if not keys:
                raise ValueError(f"{path} is an empty .npz")
            return z[keys[0]]
    if path.suffix in (".pt", ".pth"):
        import torch

        # weights_only=False: the PyTorch >=2.6 default flipped and these dumps
        # predate it. Kept in step with standard_adaptors/latents.py, per the
        # vendoring note at the top of this file.
        try:
            obj = torch.load(path, map_location="cpu", weights_only=False)
        except Exception as exc:
            # Not a corrupt tensor: this is JSON wearing a .pt extension, which
            # at least one file in this dataset is.
            if "invalid load key" in str(exc):
                raise ValueError(
                    f"{path} is not a pickle. The leading byte suggests JSON "
                    f"saved under a .pt extension; re-run the encoding pass for "
                    f"this clip or convert it. Underlying error: {exc}"
                ) from exc
            raise
        if isinstance(obj, dict):
            tensors = [v for v in obj.values() if hasattr(v, "shape")]
            if not tensors:
                raise ValueError(f"{path} holds a dict with no tensors")
            obj = tensors[0]
        if isinstance(obj, np.ndarray):  # some dumps hold numpy, not torch
            return obj.astype(np.float32)
        return obj.float().detach().cpu().numpy()
    raise ValueError(f"unsupported embedding format: {path.suffix!r}")


class EmbeddingStore:
    """A directory of embeddings, addressed by filename stem."""

    def __init__(self, directory: Path, key_prefix: Optional[str] = None) -> None:
        self.directory = Path(directory)
        if not self.directory.is_dir():
            raise NotADirectoryError(f"not a directory: {self.directory}")

        self._index: Dict[str, Path] = {}
        self.sources = sorted(
            p for p in self.directory.iterdir() if p.suffix.lower() in FILE_SUFFIXES
        )
        for path in self.sources:
            if key_prefix and not path.stem.startswith(key_prefix):
                continue
            self._index[path.stem] = path

        if not self._index:
            raise ValueError(
                f"no embeddings found in {self.directory} "
                f"(looked for {'/'.join('*' + s for s in FILE_SUFFIXES)})"
            )

    def keys(self) -> List[str]:
        return sorted(self._index, key=_natural_key)

    def source_of(self, key: str) -> Path:
        return self._index[key]

    def get(self, key: str) -> np.ndarray:
        """Read one embedding as a float32 numpy array."""
        return np.asarray(load_any(self._index[key]))

    def __len__(self) -> int:
        return len(self._index)

    def __repr__(self) -> str:
        return f"EmbeddingStore({self.directory.name}, {len(self)} keys)"


def pair_stores(a: EmbeddingStore, b: EmbeddingStore) -> Tuple[List[str], List[str], List[str]]:
    """Match two stores by key. Returns ``(paired, a_only, b_only)``."""
    ka, kb = set(a.keys()), set(b.keys())
    return (
        sorted(ka & kb, key=_natural_key),
        sorted(ka - kb, key=_natural_key),
        sorted(kb - ka, key=_natural_key),
    )


def as_vjepa_grid(
    raw: np.ndarray, grid: Tuple[int, int, int], clip_index: int = 0, name: str = "?"
) -> np.ndarray:
    """Fold a stored V-JEPA embedding back onto its patch grid -> ``(D, F, H, W)``."""
    a = np.asarray(raw)
    if a.ndim == 5:  # (B, F, H, W, D)
        a = a[clip_index]
    if a.ndim == 3:  # (B, N, D)
        a = a[clip_index]
    a = np.asarray(a, dtype=np.float32)

    if a.ndim == 2:  # (N, D)
        n, d = a.shape
        gf, gh, gw = grid
        if gf * gh * gw != n:
            raise ValueError(
                f"{name}: grid {grid} implies {gf * gh * gw} patches but the "
                f"entry has {n}. Check --jepa-grid."
            )
        a = a.reshape(gf, gh, gw, d)
    elif a.ndim != 4:
        raise ValueError(f"{name}: unexpected V-JEPA shape {a.shape}")

    return np.ascontiguousarray(a.transpose(3, 0, 1, 2))  # (D, F, H, W)


def as_ltx_grid(
    raw: np.ndarray, layout: str = "cfhw", clip_index: int = 0, name: str = "?"
) -> np.ndarray:
    """Canonicalize a stored LTX-2 VAE latent -> ``(C, F, H, W)``."""
    a = np.asarray(raw)
    if a.ndim == 5:  # (B, C, F, H, W)
        a = a[clip_index]
    a = np.asarray(a, dtype=np.float32)

    if a.ndim == 1:
        raise ValueError(
            f"{name} is a 1-D vector of length {a.shape[0]}. These look like the "
            "pooled-and-flattened latents saved for the UMAP plots. The decoder "
            "needs the full 4-D latent, so the LTX-2 encoding pass has to be "
            "re-run saving the un-pooled tensor."
        )
    if a.ndim != 4:
        raise ValueError(f"{name}: expected a 4-D LTX latent, got {a.shape}")

    if layout == "fchw":
        a = a.transpose(1, 0, 2, 3)
    elif layout != "cfhw":
        raise ValueError(f"unknown ltx layout: {layout!r}")
    return np.ascontiguousarray(a)


def vjepa_loader(store: EmbeddingStore, grid, clip_index: int = 0) -> Callable[[str], np.ndarray]:
    """Bind a store and a grid into a plain ``key -> (D, F, H, W)`` function."""
    return lambda key: as_vjepa_grid(store.get(key), grid, clip_index, name=key)


def ltx_loader(
    store: EmbeddingStore, layout: str = "cfhw", clip_index: int = 0
) -> Callable[[str], np.ndarray]:
    """Bind a store and a layout into a plain ``key -> (C, F, H, W)`` function."""
    return lambda key: as_ltx_grid(store.get(key), layout, clip_index, name=key)


class ChannelStats:
    """Per-channel mean and std, used to put both sides on a comparable scale.

    Fitted on the *training* split only, then applied to validation as well --
    fitting on everything would leak the validation distribution into the
    normalization and flatter the numbers.

    Standardizing the target matters twice over for diffusion: the flow objective
    assumes the data distribution is roughly unit-scale, and an unnormalized
    latent would let whichever channels happen to have the largest variance
    dominate the loss.
    """

    def __init__(self, mean: np.ndarray, std: np.ndarray) -> None:
        self.mean = mean.astype(np.float32)
        self.std = std.astype(np.float32)

    @classmethod
    def fit(cls, keys: Sequence[str], loader: Callable[[str], np.ndarray]) -> "ChannelStats":
        """Stream over entries accumulating sums, so nothing large stays resident."""
        total = None
        total_sq = None
        count = 0
        for key in keys:
            a = loader(key)
            flat = a.reshape(a.shape[0], -1).astype(np.float64)
            if total is None:
                total = flat.sum(axis=1)
                total_sq = (flat**2).sum(axis=1)
            else:
                total += flat.sum(axis=1)
                total_sq += (flat**2).sum(axis=1)
            count += flat.shape[1]
        if total is None:
            raise ValueError("no entries to fit statistics on")
        mean = total / count
        var = np.maximum(total_sq / count - mean**2, 1e-12)
        return cls(mean, np.sqrt(var))

    def transform(self, a: np.ndarray) -> np.ndarray:
        return (a - self.mean[:, None, None, None]) / self.std[:, None, None, None]

    def inverse(self, a: np.ndarray) -> np.ndarray:
        return a * self.std[:, None, None, None] + self.mean[:, None, None, None]

    def save(self, path: Path) -> None:
        np.savez(path, mean=self.mean, std=self.std)

    @classmethod
    def load(cls, path: Path) -> "ChannelStats":
        with np.load(path) as z:
            return cls(z["mean"], z["std"])
