"""Train the conditional diffusion decoder on paired V-JEPA / LTX-2 latents.

This is the successor to ``src/standard_adaptors/train_adaptor.py`` and answers a sharper
version of the same question. The adaptor asked whether a V-JEPA embedding can
be *deterministically* converted into a decodable LTX latent, and the answer came
back as "roughly, in the sense that colours transfer and shapes do not" -- the
signature of a regressor collapsing onto the conditional mean.

This script asks instead:

    Does ``p(LTX latent | V-JEPA embedding)`` concentrate enough to identify the
    source clip?

which is the question the attack actually needs answered, and the one a
generative model can answer without being punished for the ambiguity that is
genuinely there.

Usage
-----
    python src/diffusion_adaptor/train.py --vjepa-dir VDIR --ltx-dir LDIR \
        --jepa-grid 8,14,14 --out outputs/runs/diffusion

Reuse the adaptor's split so the two are directly comparable::

    python src/diffusion_adaptor/train.py ... --split outputs/runs/conv/split.json

Reading the output
------------------
``val`` is the flow-matching loss. It falls fast and then crawls, and its
absolute value means very little -- most of it is the irreducible noise in the
regression target. Do not tune against it beyond checking it is not diverging.

``top1`` is what matters, and it is measured by actually sampling the model on
the validation set every ``--eval-every`` epochs and asking whether each sample
retrieves its own target. Chance is printed beside it. This is the number that
distinguishes "generates a plausible SSv2 clip" from "generates *this* clip", and
it is the one to put in the report.

``spread`` is the per-channel standard deviation of the samples in standardized
space. A regressor drove this toward zero, which was the whole grey-mush problem;
a working diffusion model should sit near 1.0. Well below 0.8 means something is
wrong with the sampler or the schedule, not with the data.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import time
from functools import partial
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

# Flat imports, resolved because running a script puts its own directory first on
# sys.path. Same arrangement as src/standard_adaptors/, and it means these files
# stay plain scripts -- no package, no installation, no -m. Package-qualified
# imports do not work here: nothing puts the repo root on sys.path.
from data import ChannelStats, EmbeddingStore, ltx_loader, pair_stores, vjepa_loader
from flow import EMA, flow_loss, retrieval_accuracy, sample
from model import LatentDiT


class PairedLatents(Dataset):
    """Paired (V-JEPA, LTX-2) clips, read lazily and standardized on the fly.

    Identical in spirit to the adaptor's dataset: a V-JEPA clip is ~6 MB in
    float32 and a few thousand will not sit in RAM beside a training run.
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


@contextlib.contextmanager
def fixed_rng(seed: int):
    """Run a block at a fixed seed without disturbing the training RNG stream.

    Both evaluations below want noise that is identical from epoch to epoch, so
    that what moves in the numbers is the model rather than the draw. The naive
    way to get that -- calling ``torch.manual_seed`` inline -- also resets the
    *global* generator, so training would resume from the same state after every
    eval and every epoch would then see the same noise. That silently guts the
    stochasticity a diffusion objective depends on, while still looking like it
    is training. Hence save and restore around the block.
    """
    cpu_state = torch.get_rng_state()
    cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    torch.manual_seed(seed)
    try:
        yield
    finally:
        torch.set_rng_state(cpu_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state_all(cuda_state)


@torch.no_grad()
def val_loss(model, loader, device, cfg_dropout: float, seed: int = 0) -> float:
    """Flow loss over the validation set at fixed noise, so epochs are comparable."""
    model.eval()
    total, seen = 0.0, 0
    with fixed_rng(seed):
        for cond, x1 in loader:
            cond, x1 = cond.to(device), x1.to(device)
            total += float(flow_loss(model, x1, cond, cfg_dropout)) * x1.shape[0]
            seen += x1.shape[0]
    return total / max(seen, 1)


@torch.no_grad()
def sampled_metrics(model, loader, device, latent_shape, steps: int, cfg: float,
                    seed: int = 0) -> dict:
    """Draw one sample per validation clip and score identifiability."""
    model.eval()
    preds, targets = [], []
    with fixed_rng(seed):
        for cond, x1 in loader:
            pred = sample(model, cond.to(device), latent_shape, steps=steps, cfg=cfg)
            preds.append(pred.float().cpu())
            targets.append(x1.float())
    pred = torch.cat(preds)
    target = torch.cat(targets)

    metrics = retrieval_accuracy(pred, target)
    metrics["cosine"] = float(
        torch.nn.functional.cosine_similarity(pred.flatten(1), target.flatten(1), dim=1).mean()
    )
    metrics["spread"] = float(pred.reshape(pred.shape[0], pred.shape[1], -1).std(dim=2).mean())
    return metrics


def resolve_split(args, paired, rng):
    """Either reuse a split file (comparability) or make a fresh one."""
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
    # Ceiling as well as floor: without it a small set leaves no training clips,
    # which surfaces far away as "no entries to fit statistics on".
    n_val = min(max(2, int(len(paired) * args.val_frac)), len(paired) - 1)
    return [paired[i] for i in order[n_val:]], [paired[i] for i in order[:n_val]]


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

    ap.add_argument("--width", type=int, default=768)
    ap.add_argument("--depth", type=int, default=12)
    ap.add_argument("--heads", type=int, default=None,
                    help="attention heads; default is width//64, the usual head "
                         "dimension. Pinning this to a constant makes --width a "
                         "trap, since most widths do not divide by it.")
    ap.add_argument("--cond-pool", default=None,
                    help="pool the V-JEPA grid before cross-attention, e.g. 8,7,7")

    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--warmup", type=int, default=500, help="linear warmup steps")
    ap.add_argument("--cfg-dropout", type=float, default=0.1)
    ap.add_argument("--ema-decay", type=float, default=0.999)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--amp", action="store_true", help="bf16 autocast")

    ap.add_argument("--eval-every", type=int, default=20, help="epochs between sampling evals")
    ap.add_argument("--eval-steps", type=int, default=50, help="sampler steps during eval")
    ap.add_argument("--eval-cfg", type=float, default=1.5)

    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--split", type=Path, default=None, help="reuse an existing split.json")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=0, help="keep 0 on Windows")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path("outputs/runs/diffusion"))
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Derive heads from width at a head dimension of 64 unless told otherwise, so
    # that changing --width alone stays valid.
    if args.heads is None:
        args.heads = max(1, args.width // 64)
    if args.width % args.heads:
        raise SystemExit(
            f"--width {args.width} is not divisible by --heads {args.heads}. "
            f"Nearest workable head counts: "
            f"{sorted(h for h in (4, 6, 8, 12, 16) if args.width % h == 0)}"
        )

    # ---- data --------------------------------------------------------------
    try:
        grid = tuple(int(x) for x in args.jepa_grid.split(","))
        cond_pool = tuple(int(x) for x in args.cond_pool.split(",")) if args.cond_pool else None
    except ValueError:
        raise SystemExit(f"--jepa-grid/--cond-pool take integers, got {args.jepa_grid!r}"
                         f" and {args.cond_pool!r}")
    if len(grid) != 3:
        raise SystemExit(f"--jepa-grid takes F,H,W (three integers), got {args.jepa_grid!r}")
    if cond_pool is not None and len(cond_pool) != 3:
        raise SystemExit(f"--cond-pool takes F,H,W (three integers), got {args.cond_pool!r}")

    # After validation, so a rejected argument does not leave an empty run
    # directory behind for the next `ls outputs/runs/` to puzzle over.
    args.out.mkdir(parents=True, exist_ok=True)

    v_store = EmbeddingStore(args.vjepa_dir, key_prefix=args.key_prefix)
    l_store = EmbeddingStore(args.ltx_dir, key_prefix=args.key_prefix)
    paired, _, _ = pair_stores(v_store, l_store)
    if not paired:
        raise SystemExit("no paired keys -- check the two directories share filename stems")
    if args.limit:
        paired = paired[: args.limit]

    load_v = vjepa_loader(v_store, grid, args.clip_index)
    load_l = ltx_loader(l_store, args.ltx_layout, args.clip_index)

    train_keys, val_keys = resolve_split(args, paired, rng)
    print(f"{len(train_keys)} train / {len(val_keys)} val pairs on {device}")
    if len(train_keys) < 2000:
        print(f"NOTE: {len(train_keys)} training clips is small for a diffusion model.")
        print("  Expect memorization. Encoding more of SSv2, and taking several")
        print("  spatial/temporal crops per video, is the highest-leverage fix.")

    print("fitting channel statistics on the training split...")
    v_stats = ChannelStats.fit(train_keys, load_v)
    l_stats = ChannelStats.fit(train_keys, load_l)
    v_stats.save(args.out / "vjepa_stats.npz")
    l_stats.save(args.out / "ltx_stats.npz")

    make = partial(PairedLatents, load_v=load_v, load_l=load_l, v_stats=v_stats, l_stats=l_stats)
    train_ds, val_ds = make(train_keys), make(val_keys)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, drop_last=len(train_ds) > args.batch_size,
    )
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, num_workers=args.workers)

    # ---- model -------------------------------------------------------------
    cond0, x0 = train_ds[0]
    cond_dim, cond_grid = cond0.shape[0], tuple(cond0.shape[1:])
    in_ch, latent_grid = x0.shape[0], tuple(x0.shape[1:])
    latent_shape = (in_ch, *latent_grid)

    if cond_grid != grid:
        raise SystemExit(f"--jepa-grid {grid} but loader produced {cond_grid}")

    n_cond_tokens = int(np.prod(cond_pool or cond_grid))
    print(f"condition: {cond_dim}ch x {cond_grid} -> {n_cond_tokens} tokens"
          + (f" (pooled from {int(np.prod(cond_grid))})" if cond_pool else ""))
    print(f"target:    {in_ch}ch x {latent_grid} -> {int(np.prod(latent_grid))} tokens")

    model = LatentDiT(
        in_ch=in_ch, latent_grid=latent_grid,
        cond_dim=cond_dim, cond_grid=cond_grid,
        width=args.width, depth=args.depth, heads=args.heads, cond_pool=cond_pool,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"DiT: width {args.width}, depth {args.depth}, heads {args.heads} "
          f"(head dim {args.width // args.heads}) -> {n_params/1e6:.1f}M parameters")

    ema = EMA(model, decay=args.ema_decay)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
                            betas=(0.9, 0.95))

    config = {
        "vjepa_dir": str(args.vjepa_dir), "ltx_dir": str(args.ltx_dir),
        "grid": list(grid), "cond_pool": list(cond_pool) if cond_pool else None,
        "ltx_layout": args.ltx_layout, "clip_index": args.clip_index,
        "key_prefix": args.key_prefix,
        "in_ch": in_ch, "latent_grid": list(latent_grid),
        "cond_dim": cond_dim, "cond_grid": list(cond_grid),
        "width": args.width, "depth": args.depth, "heads": args.heads,
        "cfg_dropout": args.cfg_dropout, "n_params": n_params, "seed": args.seed,
    }
    (args.out / "config.json").write_text(json.dumps(config, indent=2))
    (args.out / "split.json").write_text(
        json.dumps({"train": train_keys, "val": val_keys}, indent=2)
    )

    # ---- train -------------------------------------------------------------
    print(f"sampling eval at epoch 0, then every {args.eval_every} epochs "
          f"({args.eval_steps} steps, cfg {args.eval_cfg}); the top1..spread "
          f"columns are blank in between")

    header = (f"{'ep':>5} {'train':>9} {'val':>9} {'top1':>7} {'top5':>7} "
              f"{'medrank':>8} {'cos':>7} {'spread':>7} {'s':>6}")
    print("-" * len(header)); print(header); print("-" * len(header))

    amp_dtype = torch.bfloat16 if args.amp else None
    best_top1 = -1.0
    history = []
    started = time.time()
    step = 0

    for epoch in range(args.epochs):
        model.train()
        running, seen = 0.0, 0
        for cond, x1 in train_loader:
            cond, x1 = cond.to(device), x1.to(device)

            lr = args.lr * min(1.0, (step + 1) / max(args.warmup, 1))
            for group in opt.param_groups:
                group["lr"] = lr

            if amp_dtype is not None:
                with torch.autocast(device_type=device.type, dtype=amp_dtype):
                    loss = flow_loss(model, x1, cond, args.cfg_dropout)
            else:
                loss = flow_loss(model, x1, cond, args.cfg_dropout)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            if args.grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()
            ema.update(model)

            running += float(loss.detach()) * x1.shape[0]
            seen += x1.shape[0]
            step += 1

        record = {"epoch": epoch, "train_loss": running / max(seen, 1), "lr": lr}
        record["val_loss"] = val_loss(model, val_loader, device, args.cfg_dropout, args.seed)

        # Sampling is the expensive part -- `eval_steps` forward passes per batch,
        # doubled by guidance -- so it runs on a schedule rather than every epoch.
        # Epoch 0 is always included: it costs one eval and it establishes both
        # that the sampling path works at all and what chance looks like on this
        # split, rather than leaving the columns blank until epoch --eval-every.
        due = (
            epoch == 0
            or (epoch + 1) % args.eval_every == 0
            or epoch == args.epochs - 1
        )
        if due:
            backup = ema.copy_to(model)
            record.update(
                sampled_metrics(model, val_loader, device, latent_shape,
                                args.eval_steps, args.eval_cfg, args.seed)
            )
            ema.restore(model, backup)

        history.append(record)
        line = (f"{epoch:5d} {record['train_loss']:9.4f} {record['val_loss']:9.4f} ")
        if due:
            line += (f"{record['top1']:7.3f} {record['top5']:7.3f} {record['median_rank']:8.1f} "
                     f"{record['cosine']:7.3f} {record['spread']:7.3f} ")
        else:
            line += f"{'':>7} {'':>7} {'':>8} {'':>7} {'':>7} "
        print(line + f"{time.time()-started:6.0f}")

        if due and record["top1"] > best_top1:
            best_top1 = record["top1"]
            torch.save({"model": ema.state_dict(), "raw": model.state_dict(),
                        "config": config, "epoch": epoch}, args.out / "best.pt")
        torch.save({"model": ema.state_dict(), "raw": model.state_dict(),
                    "config": config, "epoch": epoch}, args.out / "last.pt")
        (args.out / "history.json").write_text(json.dumps(history, indent=2))

    # ---- verdict -----------------------------------------------------------
    final = history[-1]
    chance = final.get("chance_top1", 1.0 / max(len(val_keys), 1))
    print()
    print("=" * 76)
    print(f"best top1 {best_top1:.3f} (chance {chance:.4f}) | "
          f"final spread {final.get('spread', float('nan')):.3f} | "
          f"median rank {final.get('median_rank', float('nan')):.1f} of {len(val_keys)}")
    if final.get("spread", 1.0) < 0.8:
        print("WARNING: samples carry less variance than real latents. That is a")
        print("  sampler or schedule bug, not a data problem -- a flow model has no")
        print("  reason to shrink. Check the integration direction in flow.sample.")
    if len(val_keys) < 20:
        print(f"NOTE: retrieval over {len(val_keys)} validation clips is too coarse to")
        print("  mean much -- chance alone is high and one clip moves top1 a lot.")

    # Capped, because '5x chance' is unreachable on a small validation set: with
    # 3 clips chance is 0.33 and the bar would sit above 1.0, so every run would
    # report a negative verdict no matter how well it did.
    threshold = min(5 * chance, 0.5)
    if best_top1 < threshold:
        print("VERDICT: samples are not clip-specific. The model has learned the")
        print("  marginal distribution of SSv2 latents and is ignoring the condition.")
        print("  Check the pairing first (data.py warns about this: a")
        print("  consistent off-by-one looks exactly like a negative result), then")
        print("  train longer -- cross-attention takes a while to come online.")
    else:
        print("VERDICT: samples are clip-specific. Export and decode them:")
        print(f"  python src/diffusion_adaptor/sample.py --ckpt {args.out.as_posix()}/best.pt \\")
        print(f"      --vjepa-dir {args.vjepa_dir} --out-dir {args.out.as_posix()}/samples \\")
        print("      --num-samples 8 --cfg 1.5")
    print("=" * 76)


if __name__ == "__main__":
    main()
