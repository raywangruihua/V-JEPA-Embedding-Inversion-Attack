"""Loading and standardization for V-JEPA embeddings paired with **captions**.

This is the subset of ``standard_adaptors/latents.py`` that the captioner needs,
copied so this directory stands on its own -- same arrangement, and same reason,
as ``diffusion_adaptor/data.py``. The LTX side is gone entirely: the target here
is text, so ``as_ltx_grid`` and ``ltx_loader`` have no counterpart. What is new
is everything below :class:`Caption`.

Flat imports throughout (``from data import ...``), which works because running a
script puts its own directory first on ``sys.path``. No package, no install, no
``-m``.

Storage
-------
Embeddings: one file per video in a flat directory, keyed by filename stem, as
before. V-JEPA stores ``(N, D)`` per clip and the grid it was flattened from is
supplied as ``--jepa-grid``.

Captions: one JSON file, in whichever of these shapes you have --

* **SSv2 label file** -- a list of objects carrying ``id``, ``label``,
  ``template`` and ``placeholders``. This is what ``something-something-v2
  -train.json`` already is, and it is the richest form: ``template`` and
  ``placeholders`` are what make the per-attribute leakage profile possible
  without any extra annotation work.
* **flat mapping** -- ``{key: caption}``, which is what a VLM re-captioning pass
  will most naturally produce.
* **JSONL** -- one object per line, same fields as the SSv2 form.

**Pairing is by key**, exactly as everywhere else in this project: ``16290`` in
the embedding directory is assumed to be the video captioned as ``16290`` here.
Nothing downstream can detect a violation. A systematic mismatch would look like
a clean negative result, which is the failure mode this whole project is most
exposed to -- confirm a few by hand before trusting a number.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

FILE_SUFFIXES = (".pt", ".pth", ".npy", ".npz")

# SSv2 placeholders are frequently the literal word "something", which carries no
# object information at all. Scoring against them would inflate object recall
# with tokens the caption could not possibly have got wrong.
UNINFORMATIVE_PLACEHOLDERS = {"something", "some thing", "somethings", "it", "things"}


def _natural_key(key: str):
    """Sort ``9`` before ``10`` rather than after it."""
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
        # predate it. A file that fails here with "invalid load key '{'" is JSON
        # wearing a .pt extension, not a corrupt tensor -- see the state notes.
        obj = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(obj, dict):
            tensors = [v for v in obj.values() if hasattr(v, "shape")]
            if not tensors:
                raise ValueError(f"{path} holds a dict with no tensors")
            obj = tensors[0]
        if isinstance(obj, np.ndarray):
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
            patterns = "/".join("*" + s for s in FILE_SUFFIXES)
            raise ValueError(
                f"no embeddings found in {self.directory} (looked for {patterns})"
            )

    def keys(self) -> List[str]:
        return sorted(self._index, key=_natural_key)

    def source_of(self, key: str) -> Path:
        return self._index[key]

    def get(self, key: str) -> np.ndarray:
        return np.asarray(load_any(self._index[key]))

    def __len__(self) -> int:
        return len(self._index)

    def __repr__(self) -> str:
        return f"EmbeddingStore({self.directory.name}, {len(self)} keys)"


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


class VJepaLoader:
    """Picklable ``key -> (D, F, H, W)`` over an :class:`EmbeddingStore`.

    A class rather than the obvious closure, because ``DataLoader`` workers have
    to receive the dataset -- and therefore this -- across a process boundary. A
    lambda or a local function cannot be pickled, which fails on Windows (spawn)
    and, from **Python 3.14**, on Linux too: the default POSIX start method
    changed from ``fork`` to ``forkserver``, so code that quietly relied on
    inheriting closures through a fork breaks on an interpreter upgrade alone.
    """

    def __init__(self, store: EmbeddingStore, grid, clip_index: int = 0) -> None:
        self.store = store
        self.grid = tuple(int(g) for g in grid)
        self.clip_index = clip_index

    def __call__(self, key: str) -> np.ndarray:
        return as_vjepa_grid(self.store.get(key), self.grid, self.clip_index, name=key)


def vjepa_loader(store: EmbeddingStore, grid, clip_index: int = 0) -> Callable[[str], np.ndarray]:
    """Bind a store and a grid into a plain ``key -> (D, F, H, W)`` callable."""
    return VJepaLoader(store, grid, clip_index)


class PackedClips:
    """Every clip in one memmapped ``clips.npy``, addressed by the same keys.

    Built by ``pack.py``; see that module for why. Exposes the same
    ``keys()`` / ``key -> (D, F, H, W) float32`` pair as
    :class:`EmbeddingStore` + :func:`vjepa_loader`, so it drops into the loaders
    without any other change.

    The memmap is opened **lazily**, on first access in whichever process asks.
    That matters for ``DataLoader`` workers: a memmap handle inherited across a
    fork is shared state, and one pickled to a spawned worker would not survive
    at all. ``__getstate__`` drops it so both start methods behave.
    """

    def __init__(self, directory: Path) -> None:
        self.directory = Path(directory)
        meta_path = self.directory / "meta.json"
        if not meta_path.exists():
            raise SystemExit(
                f"{self.directory} is not a packed directory (no meta.json). "
                "Build one with: run.py pack --vjepa-dir ... --out ..."
            )
        self.meta = json.loads(meta_path.read_text())
        self._keys: List[str] = json.loads((self.directory / "keys.json").read_text())
        self.grid = tuple(self.meta["grid"])
        self.shape = tuple(self.meta["shape"])
        self.path = self.directory / "clips.npy"
        self._array = None

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_array"] = None
        return state

    @property
    def array(self) -> np.ndarray:
        if self._array is None:
            self._array = np.load(self.path, mmap_mode="r")
        return self._array

    def keys(self) -> List[str]:
        return list(self._keys)

    def loader(self) -> Callable[[str], np.ndarray]:
        """``key -> (D, F, H, W)`` float32, read straight out of the memmap."""
        return PackedLoader(self)

    def close(self) -> None:
        """Release the memmap handle.

        A memmap keeps the file open until it is closed or garbage collected.
        POSIX tolerates unlinking a file that is still mapped; Windows does not,
        so anything that maps a clips.npy inside a temporary directory has to
        release it before that directory can be removed. Training never noticed
        because process exit closes it anyway.
        """
        array, self._array = self._array, None
        if array is not None and getattr(array, "_mmap", None) is not None:
            array._mmap.close()

    def __enter__(self) -> "PackedClips":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def __len__(self) -> int:
        return len(self._keys)

    def __repr__(self) -> str:
        gb = len(self) * int(np.prod(self.shape)) * np.dtype(self.meta["dtype"]).itemsize
        return (f"PackedClips({self.directory.name}, {len(self)} clips, "
                f"{self.meta['dtype']}, {gb/1e9:.1f} GB)")


class PackedLoader:
    """Picklable ``key -> (D, F, H, W)`` over a :class:`PackedClips` memmap.

    Same reason as :class:`VJepaLoader` for being a class. ``PackedClips`` drops
    its memmap handle in ``__getstate__``, so each worker reopens its own -- which
    is what you want anyway: a handle shared across processes is a footgun, and
    reopening a memmap is cheap because it maps rather than reads.
    """

    def __init__(self, clips: "PackedClips") -> None:
        self.clips = clips
        self.index = {key: i for i, key in enumerate(clips.keys())}

    def __call__(self, key: str) -> np.ndarray:
        return np.asarray(self.clips.array[self.index[key]], dtype=np.float32)


def resolve_source(
    vjepa_dir: Optional[Path],
    packed: Optional[Path],
    grid: Tuple[int, int, int],
    key_prefix: Optional[str] = None,
    clip_index: int = 0,
):
    """Pick the embedding source. Returns ``(keys, load_v, grid, description)``.

    ``--packed`` wins when both are given, and it also supplies the grid, since
    ``meta.json`` recorded the one the pack was built with -- which removes the
    chance of packing at one ``--jepa-grid`` and training at another.
    """
    if packed:
        clips = PackedClips(packed)
        return clips.keys(), clips.loader(), clips.grid, repr(clips)
    if not vjepa_dir:
        raise SystemExit("pass either --vjepa-dir or --packed")
    store = EmbeddingStore(vjepa_dir, key_prefix=key_prefix)
    return store.keys(), vjepa_loader(store, grid, clip_index), grid, repr(store)


class ChannelStats:
    """Per-channel mean and std over the V-JEPA side.

    Fitted on the *training* split only, then applied to validation as well --
    fitting on everything leaks the validation distribution into the
    normalization.

    Standardizing matters less here than it did for the latent regressors (there
    is no target to collapse onto), but it still buys a well-conditioned input to
    the projector, and it makes the ``random`` control in ``evaluate.py``
    meaningful: a draw from N(0, 1) in standardized space is a plausible-looking
    embedding, which is a far stricter control than feeding zeros.
    """

    def __init__(self, mean: np.ndarray, std: np.ndarray) -> None:
        self.mean = mean.astype(np.float32)
        self.std = std.astype(np.float32)

    @classmethod
    def fit(
        cls,
        keys: Sequence[str],
        loader: Callable[[str], np.ndarray],
        limit: Optional[int] = None,
    ) -> "ChannelStats":
        """Stream over entries accumulating sums, so nothing large stays resident."""
        keys = list(keys)[:limit] if limit else list(keys)
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

    def save(self, path: Path) -> None:
        np.savez(path, mean=self.mean, std=self.std)

    @classmethod
    def load(cls, path: Path) -> "ChannelStats":
        with np.load(path) as z:
            return cls(z["mean"], z["std"])


# ---------------------------------------------------------------------------
# Captions
# ---------------------------------------------------------------------------


@dataclass
class Caption:
    """One video's text, plus the structure that makes attribute scoring possible.

    ``template`` and ``placeholders`` are optional because a VLM re-captioning
    pass will not produce them. When they are absent the action/object split in
    ``evaluate.py`` degrades to whole-caption word overlap, which is weaker but
    still runs.
    """

    text: str
    template: Optional[str] = None
    placeholders: List[str] = field(default_factory=list)

    @property
    def informative_placeholders(self) -> List[str]:
        return [
            p for p in self.placeholders
            if p and p.strip().lower() not in UNINFORMATIVE_PLACEHOLDERS
        ]


def normalize_template(template: str) -> str:
    """Strip the bracket markers from an SSv2 template and collapse whitespace.

    Two spellings of the same template then compare equal. SSv2 has 174 of these
    and they are the action label -- the thing V-JEPA is actually expected to
    have retained, and therefore the headline number of the attribute profile.
    """
    t = template.replace("[", " ").replace("]", " ")
    return re.sub(r"\s+", " ", t).strip().lower()


def _coerce_entry(obj: dict) -> Tuple[Optional[str], Optional["Caption"]]:
    key = obj.get("id") or obj.get("key") or obj.get("video_id") or obj.get("stem")
    text = obj.get("label") or obj.get("caption") or obj.get("text") or obj.get("sentence")
    if key is None or text is None:
        return None, None
    placeholders = obj.get("placeholders") or []
    if isinstance(placeholders, str):
        placeholders = [placeholders]
    return str(key), Caption(
        text=str(text).strip(),
        template=obj.get("template"),
        placeholders=[str(p) for p in placeholders],
    )


def load_captions(path: Path) -> Dict[str, Caption]:
    """Read captions from any of the three supported shapes. Keys are strings.

    Keys are matched against embedding filename *stems*, so ``16290.pt`` pairs
    with an entry whose id is ``16290``. If your stems carry a prefix
    (``embedding_16290``), the stem still has to match the caption key -- pass
    ``--caption-key-prefix`` to have it stripped before matching.
    """
    path = Path(path)
    raw = path.read_text(encoding="utf-8")

    out: Dict[str, Caption] = {}
    if path.suffix == ".jsonl":
        for line in raw.splitlines():
            line = line.strip()
            if not line:
                continue
            key, cap = _coerce_entry(json.loads(line))
            if key is not None:
                out[key] = cap
        return out

    obj = json.loads(raw)
    if isinstance(obj, list):
        for entry in obj:
            key, cap = _coerce_entry(entry)
            if key is not None:
                out[key] = cap
    elif isinstance(obj, dict):
        for key, value in obj.items():
            if isinstance(value, str):
                out[str(key)] = Caption(text=value.strip())
            elif isinstance(value, dict):
                _, cap = _coerce_entry({"id": key, **value})
                if cap is not None:
                    out[str(key)] = cap
    else:
        raise ValueError(f"{path}: expected a list or an object at the top level")

    if not out:
        raise ValueError(
            f"{path}: parsed 0 captions. Expected a list of objects carrying 'id' "
            "and 'label', a flat key-to-caption mapping, or JSONL of the former."
        )
    return out


def pair_with_captions(
    keys: Sequence[str], captions: Dict[str, "Caption"], key_prefix: str = ""
) -> Tuple[List[str], List[str], List[str]]:
    """Match embedding keys against a caption table.

    Returns ``(paired, embeddings_without_caption, captions_without_embedding)``.
    ``key_prefix`` is stripped from the embedding key before it is looked up in
    the caption table, for the case where the dumps are named ``embedding_16290``
    but the labels are keyed ``16290``.

    Takes a key *list* rather than a store so a :class:`PackedClips` source works
    through the same path.
    """
    lookup = {k: (k[len(key_prefix):] if key_prefix and k.startswith(key_prefix) else k)
              for k in keys}
    paired = sorted((k for k, c in lookup.items() if c in captions), key=_natural_key)
    missing = sorted((k for k, c in lookup.items() if c not in captions), key=_natural_key)
    used = {lookup[k] for k in paired}
    unused = sorted((c for c in captions if c not in used), key=_natural_key)
    return paired, missing, unused


def caption_for(
    key: str, captions: Dict[str, "Caption"], key_prefix: str = ""
) -> "Caption":
    """Look up a caption by store key, honouring ``key_prefix`` stripping."""
    if key_prefix and key.startswith(key_prefix):
        key = key[len(key_prefix):]
    return captions[key]


def template_vocabulary(captions: Dict[str, "Caption"]) -> List[str]:
    """The distinct normalized templates present, sorted. Empty if none carry one.

    On SSv2 this comes back with 174 entries and is the label set the action
    metric ranks against; chance is one over its length.
    """
    seen = {normalize_template(c.template) for c in captions.values() if c.template}
    return sorted(seen)
