""" Calculates Gromov-Wasserstein distance between 2 video embedding datasets.

The distance is a Monte Carlo estimate: GW is quadratic in the number of points,
so rather than solving once over every clip it solves ``NUM_ITERATIONS`` times on
random subsets of ``BATCH_SIZE`` and reports the mean and spread.

Both datasets are read from a flat directory holding one embedding per file,
keyed by filename stem -- the layout the rest of the project uses. The two sides
need not share a dimensionality: GW compares the *intra*-domain distance
matrices, never a vector from one space against a vector from the other.

Run it as a script -- ``python src/otalign/comparison.py`` -- which is what puts
this directory on ``sys.path`` and makes ``import otalign`` resolve to the
sibling module.
"""

import re
from pathlib import Path

import numpy as np
import torch

import otalign

PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
A_DIR: Path = PROJECT_ROOT / "outputs" / "embeddings" / "jepa"
B_DIR: Path = PROJECT_ROOT / "outputs" / "embeddings" / "ltx-2"

SEED: int = 47
BATCH_SIZE: int = 512
NUM_ITERATIONS: int = 50

# Entropic regularization. The GW gradient scales like 1/n, so the usable value
# tracks BATCH_SIZE: smaller is sharper but needs more Sinkhorn iterations, and
# too small for the batch underflows the kernel and returns a degenerate plan.
ENTREG: float = 5e-3
MAXITER: int = 200

FILE_SUFFIXES = (".pt", ".pth")


def main() -> None:
    a_keys, a_embeddings = get_pooled_embeddings(A_DIR)
    b_keys, b_embeddings = get_pooled_embeddings(B_DIR)
    print(f"A: {len(a_keys)} embeddings of {a_embeddings.shape[1]} features from {A_DIR}")
    print(f"B: {len(b_keys)} embeddings of {b_embeddings.shape[1]} features from {B_DIR}")

    # Calculate monte carlo estimate of GW distance

    rng = np.random.default_rng(SEED)
    gw_distances: list[float] = []

    num_a, num_b = len(a_embeddings), len(b_embeddings)
    batch_size = min(BATCH_SIZE, num_a, num_b)
    print(f"{NUM_ITERATIONS} iterations of {batch_size} sampled clips per side "
          f"(entreg {ENTREG:g}, maxiter {MAXITER})")

    for i in range(NUM_ITERATIONS):
        # Sample videos uniformly and use their pooled embeddings
        vid_idx_a = rng.choice(num_a, batch_size, replace=False)
        vid_idx_b = rng.choice(num_b, batch_size, replace=False)

        d = gw_distance(a_embeddings[vid_idx_a], b_embeddings[vid_idx_b])
        gw_distances.append(d)
        print(f"  [{i + 1:>3}/{NUM_ITERATIONS}] {d:.6f}")

    final_score = float(np.mean(gw_distances))
    std_dev = float(np.std(gw_distances))
    print(f"GW score: {final_score:.6f} +/- {std_dev:.6f}")


def gw_distance(xs: np.ndarray, xt: np.ndarray) -> float:
    """ Entropic Gromov-Wasserstein distance between two point clouds.

    Uses the project's own solver rather than POT -- see the dependency note in
    otalign.py, which exists precisely to avoid that dependency.
    """
    aligner = otalign.GromovWassersteinAligner(
        metric="cosine",
        # "none" keeps the raw cosine distances, so the number stays on the
        # scale of the metric rather than one rescaled per batch.
        normalize_dists="none",
        entreg=ENTREG,
        maxiter=MAXITER,
        verbose=False,
        # Supplying any callback makes the solver record every iterate, so
        # history[-1] is the final one even when it stops at maxiter rather
        # than converging.
        callback=lambda record, plan: None,
    )
    plan = aligner.solve(xs, xt)
    if not np.isfinite(plan).all() or plan.sum() <= 0:
        raise ValueError(
            f"the solve returned a degenerate coupling; entreg={ENTREG:g} is too "
            f"small for a batch of {xs.shape[0]} (the safe direction is larger)"
        )
    return aligner.history[-1].gw_distance


def get_pooled_embeddings(embeddings_dir: Path) -> tuple[list[str], np.ndarray]:
    """ Load a directory of one-embedding-per-file dumps, pooled and normalized.

    Returns the filename stems and an ``(n_videos, n_features)`` matrix whose
    rows are unit length, which is what makes the cosine distances below the
    same quantity on both sides.
    """
    if not embeddings_dir.is_dir():
        raise NotADirectoryError(f"not a directory: {embeddings_dir}")

    paths = sorted(
        (p for p in embeddings_dir.iterdir() if p.suffix.lower() in FILE_SUFFIXES),
        key=lambda p: _natural_key(p.stem),
    )
    if not paths:
        raise ValueError(
            f"no embeddings found in {embeddings_dir} "
            f"(looked for {'/'.join('*' + s for s in FILE_SUFFIXES)})"
        )

    keys: list[str] = []
    pooled: list[np.ndarray] = []
    for path in paths:
        keys.append(path.stem)
        pooled.append(pool_embedding(_load_tensor(path), path.stem))

    width = pooled[0].shape[0]
    for key, vector in zip(keys, pooled):
        if vector.shape[0] != width:
            raise ValueError(
                f"{key} pools to {vector.shape[0]} features, expected {width}; "
                "all embeddings in a directory must share a shape"
            )

    return keys, otalign.normalize_vectors(np.stack(pooled), "unit")


def pool_embedding(embedding: torch.Tensor, name: str) -> np.ndarray:
    """ Average a stored embedding down to one feature vector per video.

    Handles both storage layouts in this project, since the two directories do
    not agree: V-JEPA keeps ``(N, D)`` -- a flat bag of N spatiotemporal patches
    -- while the LTX-2 VAE keeps a 4-D ``(C, F, H, W)`` latent. Either way the
    feature axis survives and everything else is averaged away, which is what
    ``normalise_vjepa`` and ``normalise_ltx2_vae`` in src/utils.py do.
    """
    a = embedding.detach().cpu().float().numpy()
    while a.ndim > 1 and a.shape[0] == 1:  # a leading batch axis of one clip
        a = a[0]

    if a.ndim == 2:      # (N, D) V-JEPA patches -> mean over patches
        return a.mean(axis=0)
    if a.ndim == 4:      # (C, F, H, W) LTX-2 latent -> mean over the grid
        return a.mean(axis=(1, 2, 3))
    if a.ndim == 1:      # already pooled
        return a
    raise ValueError(
        f"{name}: cannot pool an embedding of shape {tuple(embedding.shape)}; "
        "expected (N, D), (C, F, H, W), or an already-pooled vector"
    )


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
