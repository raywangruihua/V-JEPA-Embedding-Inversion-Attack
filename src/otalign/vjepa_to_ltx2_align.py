""" Solve GW alignment between VJEPA and LTX2 VAE latent space.

Both sides are stored one embedding per file in a flat directory
(``embedding_{video_idx}.pt``, keyed by filename stem), and the aligned
latents are written back out the same way so the rest of the pipeline can
read them with the loaders it already has.

Run it as a script -- ``python src/otalign/vjepa_to_ltx2_align.py`` -- which
is what puts this directory on ``sys.path`` and makes ``import otalign``
resolve to the sibling module.
"""

import re
from pathlib import Path

import numpy as np
import torch

import otalign

PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
VJEPA_DIR: Path = PROJECT_ROOT / "outputs" / "embeddings" / "jepa"
LTX2_DIR: Path = PROJECT_ROOT / "outputs" / "embeddings" / "ltx-2"
ALIGNED_DIR: Path = PROJECT_ROOT / "outputs" / "embeddings" / "aligned"
NUM_VIDEOS: int = 1000

FILE_SUFFIXES = (".pt", ".pth")


def main() -> None:
    vjepa_keys, vjepa_embeddings_np, _ = get_2d_embeddings(VJEPA_DIR)
    ltx2_keys, ltx2_embeddings_np, ltx2_shape = get_2d_embeddings(LTX2_DIR)
    print(f"Loaded {len(vjepa_keys)} VJEPA and {len(ltx2_keys)} LTX2 embeddings.")

    xs = otalign.normalize_vectors(vjepa_embeddings_np, "both")
    xt = otalign.normalize_vectors(ltx2_embeddings_np, "both")

    aligner = otalign.GromovWassersteinAligner(
        metric="cosine",
        normalize_dists="mean",
        entreg=5e-4,
        maxiter=1000,
        verbose=True
    )

    # Unsupervised alignment. T has shape (len(vjepa_keys), len(ltx2_keys)):
    # row i indexes vjepa_keys, column j indexes ltx2_keys.
    T = aligner.solve(xs, xt)

    ALIGNED_DIR.mkdir(parents=True, exist_ok=True)
    for i, key in enumerate(vjepa_keys):
        # Soft mapping of a VJEPA embedding into the LTX2 VAE embedding space.
        # The row is copied before rescaling so the transport plan is left intact.
        target_weights = np.array(T[i], dtype=np.float64)
        total = target_weights.sum()
        if total <= 0:
            raise ValueError(
                f"transport plan row for {key} sums to {total}; the solve did not "
                "produce a usable coupling (try a smaller entreg)"
            )
        target_weights /= total

        # Mapped back through the *raw* latents, not the normalized ones, so the
        # result lives in the space the LTX2 VAE decoder expects.
        mapped_embedding = target_weights @ ltx2_embeddings_np
        mapped_embedding = mapped_embedding.reshape(ltx2_shape)

        torch.save(
            torch.from_numpy(mapped_embedding.astype(np.float32)),
            ALIGNED_DIR / f"{key}.pt",
        )

    print(f"Wrote {len(vjepa_keys)} aligned embeddings to {ALIGNED_DIR}.")

    # Find top1 vjepa embedding to ltx2 vae embedding
    # top1 = np.argmax(T[0])
    # mapped_embedding = xt[top1]


def get_2d_embeddings(
    embeddings_dir: Path
) -> tuple[list[str], np.ndarray, tuple[int, ...]]:
    """ Load a directory of one-embedding-per-file dumps as a 2D matrix.

    Returns the filename stems (the keys the stores are paired by), the
    embeddings flattened to ``(num_videos, -1)``, and the shape a single
    embedding was stored with, so the mapped latents can be reshaped back.
    """
    if not embeddings_dir.is_dir():
        raise NotADirectoryError(f"not a directory: {embeddings_dir}")

    paths = sorted(
        (p for p in embeddings_dir.iterdir() if p.suffix.lower() in FILE_SUFFIXES),
        key=lambda p: _natural_key(p.stem),
    )[:NUM_VIDEOS]
    if not paths:
        raise ValueError(
            f"no embeddings found in {embeddings_dir} "
            f"(looked for {'/'.join('*' + s for s in FILE_SUFFIXES)})"
        )

    keys: list[str] = []
    embeddings: list[torch.Tensor] = []
    for path in paths:
        keys.append(path.stem)
        embeddings.append(_load_tensor(path))

    shape = tuple(embeddings[0].shape)
    for key, embedding in zip(keys, embeddings):
        if tuple(embedding.shape) != shape:
            raise ValueError(
                f"{key} has shape {tuple(embedding.shape)}, expected {shape}; "
                "all embeddings in a directory must share a shape to be stacked"
            )

    # reshape to a 2D matrix, one flattened embedding per row
    embeddings_np = torch.stack(embeddings).float().cpu().numpy()
    embeddings_np = embeddings_np.reshape(embeddings_np.shape[0], -1)
    return keys, embeddings_np, shape


def _load_tensor(path: Path) -> torch.Tensor:
    """ Read one dump, unwrapping the single-tensor dicts some passes saved. """
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(obj, dict):
        tensors = [v for v in obj.values() if isinstance(v, torch.Tensor)]
        if not tensors:
            raise ValueError(f"{path} holds a dict with no tensors")
        obj = tensors[0]
    return obj.detach().cpu()


def _natural_key(key: str):
    """ Sort ``embedding_9`` before ``embedding_10`` rather than after it. """
    match = re.search(r"(\d+)\s*$", key)
    return (0, int(match.group(1)), "") if match else (1, 0, key)


if __name__ == "__main__":
    main()
