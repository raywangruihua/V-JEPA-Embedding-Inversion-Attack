"""Sample LTX-2 latents from V-JEPA embeddings and write them to disk.

The counterpart to ``standard_adaptors/export_latents.py``, and it stops in the same place
for the same reason: the LTX-2 VAE lives in its own repo with its own
environment, so this script emits latents in the VAE's units and leaves the
decode to you.

One difference matters. The adaptor produced exactly one latent per clip, because
a regressor has only one answer. This produces ``--num-samples`` of them, because
the whole point of the change is that ``p(latent | embedding)`` is a distribution
rather than a point -- and for an inversion attack that plurality is an asset,
not an inconvenience. See "Best-of-N" below.

Usage
-----
    python src/diffusion_adaptor/sample.py --ckpt outputs/runs/diffusion/best.pt \\
        --vjepa-dir VDIR --out-dir outputs/runs/diffusion/samples \\
        --num-samples 8 --cfg 1.5

Omit ``--keys`` and it samples the held-out validation clips recorded in
``split.json`` beside the checkpoint -- the only ones whose reconstructions mean
anything, since a training clip can simply have been memorized.

Layout
------
One subdirectory per source video, named after the video index parsed off the
end of the key -- ``embedding_42`` lands in ``42/``. Inside it the latents keep
the key as their filename, so a clip's candidates travel together and can be
handed to the VAE a directory at a time.

Note that this makes the output directory *not* an ``EmbeddingStore``: that class
reads one flat directory and does not recurse, so anything consuming these
latents should be pointed at ``out_dir/<index>`` rather than at ``out_dir``.

Best-of-N
---------
Candidates are written as ``{index}/{key}_s{i}.pt`` precisely so they can be
ranked (with a single sample the ``_s{i}`` is dropped and the key stands alone).
The
rerank that closes the loop needs both models and therefore cannot live here, but
it is short, and it is the step that turns this from a visualization into an
attack::

    for each candidate:
        video = ltx_vae.decode(candidate)          # LTX-2 environment
        z     = vjepa_encoder(video)               # V-JEPA environment
        score = cosine(z, intercepted_embedding)
    keep argmax

Ranking against the *intercepted* embedding -- the thing you actually have --
rather than against ground truth is what makes this legitimate at attack time.
It also doubles as an honest per-clip confidence signal, and as the objective to
sweep ``--cfg`` against: the visually sharpest setting is rarely the most
faithful one.

Scale
-----
The model works in standardized space, so what it emits is not a latent the
decoder will accept until it is pushed back through the inverse of the LTX
channel statistics. That happens by default; ``--keep-standardized`` skips it,
which is only useful for comparing against training-time numbers.

Unlike the adaptor, there is no ``--match-variance`` here, and its absence is the
result. A regressor needed that lie because predicting the conditional mean
shrinks the output toward zero; a flow model integrates from noise to the data
distribution and lands at the right scale on its own. The ``spread`` column in
the manifest should sit near 1.0 without any help. If it does not, fix the
sampler rather than rescaling the symptom.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import torch

# Flat imports -- see the note in train.py.
from data import ChannelStats, EmbeddingStore, vjepa_loader
from flow import sample
from model import LatentDiT


def load_model(ckpt_path: Path, device: torch.device, weights: str = "ema"):
    """Rebuild the DiT described by a checkpoint's embedded config."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if "model" not in ckpt or "config" not in ckpt:
        raise SystemExit(f"{ckpt_path} is not a diffusion checkpoint from train.py")
    cfg = ckpt["config"]

    model = LatentDiT(
        in_ch=cfg["in_ch"], latent_grid=cfg["latent_grid"],
        cond_dim=cfg["cond_dim"], cond_grid=cfg["cond_grid"],
        width=cfg["width"], depth=cfg["depth"], heads=cfg["heads"],
        cond_pool=cfg.get("cond_pool"),
    )
    # 'model' holds the EMA weights, which is what you want for sampling; 'raw'
    # is kept for resuming and for checking whether EMA is actually helping.
    state = ckpt["model"] if weights == "ema" else ckpt["raw"]
    missing, unexpected = model.load_state_dict(state, strict=False)
    if unexpected:
        raise SystemExit(f"unexpected keys in checkpoint: {sorted(unexpected)[:5]}")
    if missing:
        # EMA only tracks floating-point entries, so integer buffers can be
        # absent legitimately. Anything with a gradient going missing is not.
        print(f"note: {len(missing)} key(s) not in the EMA state, kept at init")
    return model.eval().to(device), cfg


def resolve_keys(args, store: EmbeddingStore) -> list[str]:
    """Explicit ``--keys``, ``--all``, or the held-out validation split."""
    available = set(store.keys())

    if args.all:
        keys = store.keys()
    elif args.keys:
        missing = [k for k in args.keys if k not in available]
        if missing:
            raise SystemExit(
                f"not in {store.directory}: {', '.join(missing)}\n"
                f"(store holds {len(store)} keys, e.g. {', '.join(store.keys()[:3])})"
            )
        keys = list(args.keys)
    else:
        split_path = args.split or (args.ckpt.parent / "split.json")
        if not split_path.exists():
            raise SystemExit(
                f"no --keys given and no split file at {split_path}. "
                "Pass --keys explicitly, or --all to sample the whole store."
            )
        val_keys = json.loads(split_path.read_text())["val"]
        keys = [k for k in val_keys if k in available]
        if not keys:
            raise SystemExit(f"none of the {len(val_keys)} validation keys are in {store.directory}")
        print(f"no --keys given: using {len(keys)} validation clips from {split_path.name}")

    if args.limit is not None:
        keys = keys[: args.limit]
    if not keys:
        raise SystemExit("nothing to sample")
    return keys


def video_dir(key: str) -> str:
    """Name of the per-clip subdirectory: the video index parsed off the key.

    ``embedding_42`` -> ``42``, matching the numbering the stores are keyed by.
    A key with no trailing number keeps its own name, which is less tidy but
    never collides -- silently dropping such clips into a shared bucket would
    overwrite one candidate with another.
    """
    match = re.search(r"(\d+)\s*$", key)
    return match.group(1) if match else key


def clear_stale(out_dir: Path) -> int:
    """Delete latents from a previous run.

    Necessary rather than tidy: a shorter re-run left beside a longer one would
    present the old run's leftovers as part of the new one, and per-clip
    subdirectories only narrow that to one clip rather than fixing it. Both the
    current nested layout and the flat one that preceded it are swept, and the
    directories emptied in the process go with them.
    """
    stale = sorted(out_dir.glob("*.pt")) + sorted(out_dir.glob("*/*.pt"))
    for path in stale:
        path.unlink()
    for entry in sorted(p for p in out_dir.iterdir() if p.is_dir()):
        if not any(entry.iterdir()):
            entry.rmdir()
    return len(stale)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--ckpt", type=Path, required=True, help="outputs/runs/<name>/best.pt")
    ap.add_argument("--vjepa-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--keys", nargs="*", default=None, help="clips to sample; default is the val split")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--split", type=Path, default=None)

    ap.add_argument("--num-samples", type=int, default=1,
                    help="candidates per clip; >1 enables best-of-N reranking")
    ap.add_argument("--steps", type=int, default=50, help="Euler integration steps")
    ap.add_argument("--cfg", type=float, default=1.5,
                    help="classifier-free guidance scale; 1.0 disables it")
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--weights", default="ema", choices=["ema", "raw"])
    ap.add_argument("--vjepa-stats", type=Path, default=None)
    ap.add_argument("--ltx-stats", type=Path, default=None)
    ap.add_argument("--jepa-grid", default=None, help="override the grid in the checkpoint")
    ap.add_argument("--clip-index", type=int, default=None)
    ap.add_argument("--key-prefix", default=None)
    ap.add_argument("--keep-standardized", action="store_true")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model, cfg = load_model(args.ckpt, device, args.weights)
    latent_shape = (cfg["in_ch"], *cfg["latent_grid"])
    print(f"DiT [{args.weights}] on {device}: {cfg['cond_dim']}ch condition -> "
          f"{cfg['in_ch']} x {tuple(cfg['latent_grid'])}, {cfg['n_params']/1e6:.1f}M params")

    # ---- statistics ---------------------------------------------------------
    # These are half the model: the weights are meaningless without the exact
    # statistics they were trained under, so refuse to guess.
    v_stats_path = args.vjepa_stats or (args.ckpt.parent / "vjepa_stats.npz")
    l_stats_path = args.ltx_stats or (args.ckpt.parent / "ltx_stats.npz")
    for path in (v_stats_path, l_stats_path):
        if not path.exists():
            raise SystemExit(f"missing channel statistics: {path} (written by train.py)")
    v_stats = ChannelStats.load(v_stats_path)
    l_stats = ChannelStats.load(l_stats_path)

    # ---- data ---------------------------------------------------------------
    grid_spec = args.jepa_grid or ",".join(str(v) for v in cfg["grid"])
    grid = tuple(int(x) for x in str(grid_spec).split(","))
    clip_index = args.clip_index if args.clip_index is not None else int(cfg.get("clip_index", 0))
    prefix = args.key_prefix if args.key_prefix is not None else cfg.get("key_prefix")

    store = EmbeddingStore(args.vjepa_dir, key_prefix=prefix)
    load_v = vjepa_loader(store, grid, clip_index)
    keys = resolve_keys(args, store)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    removed = clear_stale(args.out_dir)
    if removed:
        print(f"removed {removed} file(s) from a previous run")

    space = "standardized" if args.keep_standardized else "LTX units"
    print(f"sampling {len(keys)} clips x {args.num_samples} candidates "
          f"({args.steps} steps, cfg {args.cfg}) -> {args.out_dir} ({space})")

    # ---- sample -------------------------------------------------------------
    header = f"{'file':<34} {'shape':>18} {'mean':>9} {'std':>9} {'spread':>8}"
    print("-" * len(header)); print(header); print("-" * len(header))

    generator = torch.Generator(device=device).manual_seed(args.seed)
    manifest = []
    spreads = []

    for key in keys:
        cond = torch.from_numpy(v_stats.transform(load_v(key))).unsqueeze(0).to(device)
        # One clip at a time, but all its candidates at once: the condition is
        # identical across them, so this is a free batch dimension.
        batch = cond.expand(args.num_samples, *cond.shape[1:])
        z = sample(model, batch, latent_shape, steps=args.steps, cfg=args.cfg,
                   generator=generator).float().cpu().numpy()

        clip_dir = args.out_dir / video_dir(key)
        clip_dir.mkdir(parents=True, exist_ok=True)

        for i in range(args.num_samples):
            zi = z[i]
            spread = float(zi.reshape(zi.shape[0], -1).std(axis=1).mean())
            spreads.append(spread)

            latent = zi if args.keep_standardized else l_stats.inverse(zi)
            latent = np.ascontiguousarray(latent, dtype=np.float32)

            name = f"{key}_s{i}" if args.num_samples > 1 else key
            path = clip_dir / f"{name}.pt"
            torch.save(torch.from_numpy(latent), path)

            # Recorded relative to --out-dir so the manifest survives the whole
            # directory being moved, which is the usual way these reach the VAE.
            rel = path.relative_to(args.out_dir).as_posix()
            manifest.append({
                "key": key, "sample": i, "file": rel, "dir": clip_dir.name,
                "shape": list(latent.shape),
                "mean": float(latent.mean()), "std": float(latent.std()),
                "absmax": float(np.abs(latent).max()),
                "standardized_spread": spread,
            })
            print(f"{rel:<34} {str(tuple(latent.shape)):>18} "
                  f"{latent.mean():+9.4f} {latent.std():9.4f} {spread:8.3f}")

    (args.out_dir / "manifest.json").write_text(json.dumps({
        "ckpt": str(args.ckpt), "vjepa_dir": str(args.vjepa_dir),
        "grid": list(grid), "clip_index": clip_index,
        "num_samples": args.num_samples, "steps": args.steps, "cfg": args.cfg,
        "seed": args.seed, "weights": args.weights,
        "standardized": bool(args.keep_standardized),
        "ltx_layout": cfg.get("ltx_layout", "cfhw"),
        "latents": manifest,
    }, indent=2))

    # ---- sanity -------------------------------------------------------------
    mean_spread = float(np.mean(spreads))
    print()
    print(f"standardized spread {mean_spread:.3f} (1.0 = training-target scale)")
    if mean_spread < 0.8:
        print("WARNING: samples are under-dispersed. Unlike the adaptor this is not")
        print("  expected behaviour and not a case for rescaling -- a flow model")
        print("  should land at the data scale on its own. Suspect the sampler.")
    if args.num_samples > 1:
        per_key = {}
        for row in manifest:
            per_key.setdefault(row["key"], []).append(row["standardized_spread"])
        var = float(np.mean([np.std(v) for v in per_key.values()]))
        print(f"within-clip spread variation {var:.4f}"
              + ("  -- candidates look identical; check that --seed is advancing"
                 if var < 1e-5 else ""))

    n_dirs = len({row["dir"] for row in manifest})
    print(f"wrote {len(manifest)} .pt files across {n_dirs} clip directories "
          f"+ manifest.json under {args.out_dir}")
    print("decode with the LTX-2 VAE, e.g.:")
    print(f"  latent = torch.load('{args.out_dir.as_posix()}/{manifest[0]['file']}')")
    print("  video  = vae.decode(latent.unsqueeze(0).to(device, dtype)).sample")


if __name__ == "__main__":
    main()
