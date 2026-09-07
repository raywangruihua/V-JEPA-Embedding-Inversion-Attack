"""Train a supervised adaptor from V-JEPA embeddings to LTX-2 VAE latents.

This is the "deliberately cheating" experiment: paired supervision with a perfect
answer key. It is not the attack -- the unsupervised Gromov-Wasserstein route is
-- but it sets the **upper bound** that route gets measured against. If a head
trained on ground-truth pairs cannot recover shape, no alignment method that
assumes strictly less is going to.

Usage
-----
Run the ladder, in this order::

    python src/standard_adaptors/train_adaptor.py --vjepa-dir VDIR --ltx-dir LDIR \\
        --jepa-grid 8,14,14 --kind linear --resample pool --out outputs/runs/linear

    ... --kind mlp  --resample pool --out outputs/runs/mlp
    ... --kind conv --resample pool --out outputs/runs/conv

``linear -> mlp`` isolates nonlinearity, ``mlp -> conv`` isolates receptive
field. Keep ``--seed`` and ``--val-frac`` identical across the three or the
comparison is meaningless; the split is derived from them and written to
``split.json`` so ``diffusion_adaptor/train.py --split`` can reuse it.

Reading the output
------------------
``r2`` is the fraction of the target's variance explained, in standardized
space. Read it in bands with ``freq_r2.py`` -- a respectable full-band number
is mostly colour and coarse layout, and says almost nothing about whether shape
transferred.

``spread`` is per-channel standard deviation of the predictions, also
standardized. A collapsing regressor drives it toward 0. It is a *magnitude*,
not an accuracy: correctly-scaled noise scores 1.0 too, so never read it alone.

``top1`` is retrieval on the validation split, and it is the number to trust for
information content -- but not for visual quality. See ``retrieval_accuracy``.
"""

from __future__ import annotations

import argparse
import json
import time
from functools import partial
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

# Flat imports, resolved because running a script puts its own directory first on
# sys.path. Same arrangement as diffusion_adaptor/ -- these stay plain scripts, no
# package, no installation, no -m. Package-qualified imports do not work here:
# nothing puts the repo root on sys.path, so ``import src...`` raises.
from adaptor import (
    adaptor_loss,
    build_adaptor,
    count_parameters,
    describe,
    r2_standardized,
    retrieval_accuracy,
    standardized_spread,
)
from latents import ChannelStats, EmbeddingStore, ltx_loader, pair_stores, vjepa_loader


class PairedLatents(Dataset):
    """Paired (V-JEPA, LTX-2) clips, read lazily and standardized on the fly.

    Lazy because a V-JEPA clip is ~6 MB in float32 and a few thousand will not
    sit in RAM beside a training run.
    """

    def __init__(self, keys, load_v, load_l, v_stats, l_stats):
        self.keys = list(keys)
        self.load_v, self.load_l = load_v, load_l
        self.v_stats, self.l_stats = v_stats, l_stats

    def __len__(self) -> int:
        return len(self.keys)

    def __getitem__(self, i: int):
        key = self.keys[i]
        x = self.v_stats.transform(self.load_v(key))
        y = self.l_stats.transform(self.load_l(key))
        return torch.from_numpy(x), torch.from_numpy(y)


def resolve_split(args, paired, rng):
    """Reuse a recorded split if given, else derive one from ``--seed``.

    The derivation matches ``diffusion_adaptor/train.py`` exactly -- same
    ``np.random.default_rng(seed)``, same permutation, same ``n_val`` formula,
    same naturally-sorted key list. So matching ``--seed``, ``--val-frac``,
    dirs, ``--limit`` and ``--key-prefix`` is sufficient for comparability and
    no split file needs to change hands.
    """
    if args.split:
        recorded = json.loads(Path(args.split).read_text())
        available = set(paired)
        train_keys = [k for k in recorded["train"] if k in available]
        val_keys = [k for k in recorded["val"] if k in available]
        if not val_keys:
            raise SystemExit(f"none of the validation keys in {args.split} are present here")
        print(f"reusing split from {args.split}")
        return train_keys, val_keys

    if len(paired) < 2:
        raise SystemExit(f"need at least 2 paired clips to split, got {len(paired)}")
    order = rng.permutation(len(paired))
    # The floor of 2 keeps retrieval meaningful on tiny sets; the ceiling keeps
    # it from swallowing the training split, which would otherwise surface as an
    # opaque "no entries to fit statistics on" from ChannelStats.
    n_val = min(max(2, int(len(paired) * args.val_frac)), len(paired) - 1)
    return [paired[i] for i in order[n_val:]], [paired[i] for i in order[:n_val]]


@torch.no_grad()
def evaluate(model, loader, device) -> dict:
    """Full-set metrics on one split. Predictions are gathered so retrieval sees
    every validation clip at once rather than one batch at a time -- retrieval
    against 16 candidates and against 150 are very different questions."""
    model.eval()
    preds, targets = [], []
    for x, y in loader:
        preds.append(model(x.to(device)).cpu())
        targets.append(y)
    pred = torch.cat(preds)
    target = torch.cat(targets)

    out = {
        "mse": float(((pred - target) ** 2).mean()),
        "r2": r2_standardized(pred, target),
        "spread": standardized_spread(pred),
    }
    out.update(retrieval_accuracy(pred, target))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--vjepa-dir", type=Path, required=True)
    ap.add_argument("--ltx-dir", type=Path, required=True)
    ap.add_argument("--jepa-grid", required=True, help="F,H,W, e.g. 8,14,14")
    ap.add_argument("--ltx-layout", default="cfhw", choices=["cfhw", "fchw"])
    ap.add_argument("--clip-index", type=int, default=0)
    ap.add_argument("--key-prefix", default=None)

    ap.add_argument("--kind", default="mlp", choices=["linear", "mlp", "conv"])
    ap.add_argument("--hidden", type=int, default=0, help="0 = per-kind default")
    ap.add_argument("--depth", type=int, default=0, help="0 = per-kind default")
    ap.add_argument("--resample", default="interp", choices=["interp", "pool"],
                    help="interp is the default only for reproducibility of old "
                         "runs; pool is correct and should be used for new ones")

    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--cos-weight", type=float, default=0.0,
                    help="weight on a cosine term alongside MSE; 3.0-5.0 is the "
                         "useful range. See adaptor_loss.")
    ap.add_argument("--grad-clip", type=float, default=1.0)

    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--split", type=Path, default=None, help="reuse an existing split.json")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--stats-limit", type=int, default=None,
                    help="fit ChannelStats on at most N clips; the estimates "
                         "converge long before the full set")
    ap.add_argument("--eval-every", type=int, default=5)
    ap.add_argument("--workers", type=int, default=0, help="keep 0 on Windows")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path("outputs/runs/adaptor"))
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- data --------------------------------------------------------------
    try:
        grid = tuple(int(x) for x in args.jepa_grid.split(","))
    except ValueError:
        raise SystemExit(f"--jepa-grid takes integers, got {args.jepa_grid!r}")
    if len(grid) != 3:
        raise SystemExit(f"--jepa-grid takes F,H,W (three integers), got {args.jepa_grid!r}")

    # After validation, so a rejected argument does not leave an empty run
    # directory behind for the next `ls outputs/runs/` to puzzle over.
    args.out.mkdir(parents=True, exist_ok=True)

    v_store = EmbeddingStore(args.vjepa_dir, key_prefix=args.key_prefix)
    l_store = EmbeddingStore(args.ltx_dir, key_prefix=args.key_prefix)
    paired, v_only, l_only = pair_stores(v_store, l_store)
    if not paired:
        raise SystemExit("no paired keys -- check the two directories share filename stems")
    if v_only or l_only:
        print(f"note: {len(v_only)} V-JEPA-only and {len(l_only)} LTX-only keys ignored")
    if args.limit:
        paired = paired[: args.limit]

    load_v = vjepa_loader(v_store, grid, args.clip_index)
    load_l = ltx_loader(l_store, args.ltx_layout, args.clip_index)

    train_keys, val_keys = resolve_split(args, paired, rng)
    print(f"{len(train_keys)} train / {len(val_keys)} val pairs on {device}")

    print("fitting channel statistics on the training split...")
    v_stats = ChannelStats.fit(train_keys, load_v, limit=args.stats_limit)
    l_stats = ChannelStats.fit(train_keys, load_l, limit=args.stats_limit)
    v_stats.save(args.out / "vjepa_stats.npz")
    l_stats.save(args.out / "ltx_stats.npz")

    make = partial(PairedLatents, load_v=load_v, load_l=load_l, v_stats=v_stats, l_stats=l_stats)
    train_ds, val_ds = make(train_keys), make(val_keys)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, drop_last=len(train_ds) > args.batch_size,
    )
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, num_workers=args.workers)
    # Retrieval on the training split too: the train/val gap is the whole
    # argument about whether more data would help.
    train_eval_loader = DataLoader(
        train_ds, batch_size=args.batch_size, num_workers=args.workers
    )

    # ---- model -------------------------------------------------------------
    x0, y0 = train_ds[0]
    in_ch, in_grid = x0.shape[0], tuple(x0.shape[1:])
    out_ch, out_grid = y0.shape[0], tuple(y0.shape[1:])
    if in_grid != grid:
        raise SystemExit(f"--jepa-grid {grid} but loader produced {in_grid}")

    model = build_adaptor(
        args.kind, in_ch, out_ch, out_grid,
        hidden=args.hidden, depth=args.depth, resample=args.resample,
    ).to(device)
    print(f"V-JEPA {in_ch}ch x {in_grid}  ->  LTX {out_ch}ch x {out_grid}")
    print(describe(model))
    if args.resample == "interp":
        print("WARNING: --resample interp samples rather than averages, so some "
              "input frames contribute nothing. Use --resample pool unless you "
              "are reproducing an old run.")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    config = {
        "vjepa_dir": str(args.vjepa_dir), "ltx_dir": str(args.ltx_dir),
        "grid": list(grid), "ltx_layout": args.ltx_layout,
        "clip_index": args.clip_index, "key_prefix": args.key_prefix,
        "kind": args.kind, "hidden": args.hidden, "depth": args.depth,
        "resample": args.resample, "cos_weight": args.cos_weight,
        "in_ch": in_ch, "out_ch": out_ch, "out_grid": list(out_grid),
        "n_params": count_parameters(model), "seed": args.seed,
        "lr": args.lr, "epochs": args.epochs, "batch_size": args.batch_size,
    }
    (args.out / "config.json").write_text(json.dumps(config, indent=2))
    (args.out / "split.json").write_text(
        json.dumps({"train": train_keys, "val": val_keys}, indent=2)
    )

    # ---- train -------------------------------------------------------------
    history = []
    best_top1 = -1.0
    chance = 1.0 / max(1, len(val_keys))
    print(f"\nretrieval chance top1 on {len(val_keys)} val clips = {chance:.4f}")
    print(f"{'epoch':>5} {'train':>9} {'val_mse':>9} {'r2':>7} {'spread':>7} "
          f"{'top1':>6} {'top5':>6} {'medrank':>8} {'s':>6}")

    for epoch in range(args.epochs):
        model.train()
        t0 = time.time()
        running, seen = 0.0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            loss = adaptor_loss(model(x), y, args.cos_weight)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if args.grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()
            running += float(loss.detach()) * x.shape[0]
            seen += x.shape[0]
        sched.step()

        record = {"epoch": epoch, "train_loss": running / max(1, seen)}

        # Epoch 0 always evaluates, so the table is never blank while you wait
        # to find out whether the run is alive.
        if epoch == 0 or (epoch + 1) % args.eval_every == 0 or epoch == args.epochs - 1:
            record.update({f"val_{k}": v for k, v in evaluate(model, val_loader, device).items()})
            record["secs"] = time.time() - t0
            history.append(record)
            print(f"{epoch:5d} {record['train_loss']:9.4f} {record['val_mse']:9.4f} "
                  f"{record['val_r2']:7.3f} {record['val_spread']:7.3f} "
                  f"{record['val_top1']:6.3f} {record['val_top5']:6.3f} "
                  f"{record['val_median_rank']:8.1f} {record['secs']:6.1f}")

            if record["val_top1"] > best_top1:
                best_top1 = record["val_top1"]
                torch.save({"model": model.state_dict(), "config": config}, args.out / "best.pt")
        else:
            history.append(record)

        (args.out / "history.json").write_text(json.dumps(history, indent=2))

    torch.save({"model": model.state_dict(), "config": config}, args.out / "last.pt")

    # ---- verdict -----------------------------------------------------------
    final_val = evaluate(model, val_loader, device)
    final_train = evaluate(model, train_eval_loader, device)
    (args.out / "final.json").write_text(
        json.dumps({"val": final_val, "train": final_train}, indent=2)
    )

    print(f"\nfinal val   r2 {final_val['r2']:.4f}  spread {final_val['spread']:.4f}  "
          f"top1 {final_val['top1']:.3f} (chance {chance:.4f})")
    print(f"final train r2 {final_train['r2']:.4f}  spread {final_train['spread']:.4f}")
    print(f"\nwrote {args.out}/")

    gap = final_train["r2"] - final_val["r2"]
    if gap > 0.05:
        print(f"\ntrain-val r2 gap is {gap:.3f}. More data would move val up -- but "
              f"train r2 {final_train['r2']:.3f} is the ceiling that buys, so check "
              f"whether decodes are acceptable *at the train number* before "
              f"concluding data is the binding constraint.")
    if final_val["spread"] < 0.6:
        print(f"\nspread {final_val['spread']:.3f} means the predictions carry only "
              f"{final_val['spread'] ** 2:.0%} of the target's per-channel variance. "
              f"That is the collapse toward the conditional mean, measured directly.")
    print("\nNext: run freq_r2.py on this checkpoint. A full-band r2 hides which "
          "band it came from, and the answer is nearly always 'the coarse one'.")


if __name__ == "__main__":
    main()
