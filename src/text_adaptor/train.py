"""Train the projector to caption V-JEPA embeddings through a frozen LLM.

The successor to ``diffusion_adaptor/train.py``, asking a question the earlier
ones could not answer. The adaptor asked whether an embedding converts into a
decodable LTX latent (colours yes, shapes no). The diffusion decoder asked
whether ``p(latent | embedding)`` concentrates enough to identify the clip. Both
were asking for *appearance*, which V-JEPA discards on purpose.

This asks instead:

    How much of what the victim was doing can be read out of the embedding **in
    language**?

which is the question the measured negatives point at, and the one the encoder is
built to answer well.

Usage
-----
::

    python src/text_adaptor/train.py --vjepa-dir VDIR --captions labels.json \\
        --jepa-grid 8,14,14 --proj perceiver --out outputs/runs/caption_perceiver

Run the ladder, not one rung::

    --proj mean        # gist only: no structure at all
    --proj pool        # structure, no selection (LLaVA stage-1)
    --proj perceiver   # structure and learned selection

Reuse an existing split so numbers stay comparable across rungs::

    --split outputs/runs/caption_mean/split.json

Reading the output
------------------
``val`` is the next-token loss and it is nearly uninformative on its own -- most
of it is the irreducible ambiguity of describing a video, and a model that
ignored its input entirely would still drive it down by learning SSv2's idiom.
Do not tune against it beyond checking it is falling.

``top1`` is the number that matters: caption-to-clip retrieval on the validation
split, with chance printed beside it. It is the one metric fluency cannot game.
``action`` and ``object`` are the per-attribute profile, and the gap between them
is the result this project has been trying to reach.

``shufAct`` is action accuracy with every clip handed a *different* clip's
embedding. It is the control, it is computed every time the other numbers are,
and ``action`` minus ``shufAct`` is the only part of ``action`` that came from
the victim. A run where those two move together has learned a prior and nothing
else, however good the captions look.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

# Flat imports, resolved because running a script puts its own directory first on
# sys.path. Same arrangement as the sibling experiment directories.
from data import (
    ChannelStats,
    caption_for,
    load_captions,
    pair_with_captions,
    resolve_source,
    template_vocabulary,
)
from evaluate import (
    REPORT_HEADER, caption_retrieval, format_row, printable, summarize, verdict,
)
from model import CONTROLS, CaptionModel, apply_control, load_llm
from projector import KINDS, build_projector, token_budget


class CaptionedEmbeddings(Dataset):
    """V-JEPA clips paired with tokenized captions, read lazily.

    One clip is ~6 MB in float32 and a few thousand will not sit in RAM beside a
    language model, so files are opened per item exactly as in the sibling
    directories. Captions are tokenized up front -- they are tiny, and doing it
    here keeps the tokenizer off the worker processes.
    """

    def __init__(self, keys, load_v, stats, caption_ids: Dict[str, torch.Tensor]):
        self.keys = list(keys)
        self.load_v = load_v
        self.stats = stats
        self.caption_ids = caption_ids

    def __len__(self) -> int:
        return len(self.keys)

    def __getitem__(self, i: int):
        key = self.keys[i]
        x = self.stats.transform(self.load_v(key))
        return torch.from_numpy(x), self.caption_ids[key], key


def collate(batch):
    """Stack the grids, keep the captions ragged.

    Padding happens inside ``model.assemble_batch``, which is where the label
    masking that makes it safe also lives. Doing it here would separate the two.
    """
    xs, ids, keys = zip(*batch)
    return torch.stack(xs), list(ids), list(keys)


def resolve_split(args, keys: Sequence[str], rng) -> tuple:
    """Honour ``--split`` if given, otherwise draw one and record it.

    Reusing a split across the ladder is what makes ``mean`` and ``perceiver``
    directly comparable, so this mirrors the sibling trainers exactly.
    """
    if args.split:
        payload = json.loads(Path(args.split).read_text())
        known = set(keys)
        train = [k for k in payload["train"] if k in known]
        val = [k for k in payload["val"] if k in known]
        if not train or not val:
            raise SystemExit(f"--split {args.split} shares no keys with this data")
        return train, val

    order = list(keys)
    rng.shuffle(order)
    if len(order) < 2:
        raise SystemExit(f"need at least 2 paired clips to split, got {len(order)}")
    # Ceiling as well as floor: without it a small set leaves no training clips.
    n_val = min(max(1, int(len(order) * args.val_frac)), len(order) - 1)
    return order[n_val:], order[:n_val]


@torch.no_grad()
def val_loss(model, loader, device) -> float:
    """Next-token loss over the validation split."""
    model.projector.eval()
    total, seen = 0.0, 0
    for x, ids, _ in loader:
        x = x.to(device)
        total += float(model(x, ids)) * x.shape[0]
        seen += x.shape[0]
    model.projector.train()
    return total / max(seen, 1)


@torch.no_grad()
def evaluate_split(
    model,
    dataset: CaptionedEmbeddings,
    device,
    captions,
    templates,
    caption_key_prefix: str,
    n_clips: int,
    max_new_tokens: int,
    batch_size: int,
    seed: int,
    repetition_penalty: float = 1.0,
    retrieval_chunk: int = 16,
) -> Dict[str, object]:
    """Generation metrics under two controls, plus retrieval. The whole report.

    ``shuffle`` rather than ``random`` is the control carried into training,
    because it is the strict one -- see ``evaluate.py``. ``random`` is available
    from ``run.py evaluate`` for the full table.
    """
    model.projector.eval()
    keys = dataset.keys[:n_clips]
    refs = {k: caption_for(k, captions, caption_key_prefix) for k in keys}

    grids, ids = [], []
    for key in keys:
        x, cap_ids, _ = dataset[dataset.keys.index(key)]
        grids.append(x)
        ids.append(cap_ids)
    # Stays on the CPU. Moving all `n_clips` clips to the GPU at once costs
    # 6.4 MB each at the real shapes, which is memory that sits idle beside the
    # language model for the whole evaluation.
    embeddings = torch.stack(grids)

    generator = torch.Generator(device="cpu").manual_seed(seed)
    report: Dict[str, object] = {}
    per_control = {}
    gen_started = time.time()

    for control in ("none", "shuffle"):
        predictions: Dict[str, str] = {}
        for start in range(0, len(keys), batch_size):
            chunk = embeddings[start:start + batch_size]
            if control == "shuffle" and chunk.shape[0] < 2:
                continue  # a 1-row tail cannot be shuffled against anything
            conditioned = apply_control(chunk, control, generator).to(device)
            texts = model.generate(conditioned, max_new_tokens=max_new_tokens,
                                   repetition_penalty=repetition_penalty)
            del conditioned
            for key, text in zip(keys[start:start + batch_size], texts):
                predictions[key] = text
        per_control[control] = summarize(predictions, refs, templates)
        if control == "none":
            report["samples"] = [
                {"key": k, "predicted": predictions.get(k, ""), "reference": refs[k].text}
                for k in keys[:8]
            ]

    report["controls"] = per_control
    report["generate_s"] = time.time() - gen_started

    retrieval_started = time.time()
    report.update(caption_retrieval(model, embeddings, ids,
                                    chunk=retrieval_chunk, device=device))
    report["retrieval_s"] = time.time() - retrieval_started

    model.projector.train()
    return report


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])

    # data
    ap.add_argument("--vjepa-dir", type=Path, default=None,
                    help="directory of per-clip files; or use --packed")
    ap.add_argument("--packed", type=Path, default=None,
                    help="a directory built by 'run.py pack' -- one fp16 memmap "
                         "instead of N files. Halves the bytes read per epoch and "
                         "supplies the grid from its meta.json")
    ap.add_argument("--captions", type=Path, required=True,
                    help="SSv2 label json, a flat key->caption mapping, or jsonl")
    ap.add_argument("--jepa-grid", default="8,14,14",
                    help="(F,H,W) the stored (N,D) dump was flattened from")
    ap.add_argument("--key-prefix", default=None,
                    help="only load embedding files whose stem starts with this")
    ap.add_argument("--caption-key-prefix", default="",
                    help="strip this from an embedding stem before caption lookup")
    ap.add_argument("--clip-index", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0, help="use only the first N pairs")
    ap.add_argument("--split", type=Path, default=None, help="reuse a split.json")
    ap.add_argument("--val-frac", type=float, default=0.1)

    # model
    ap.add_argument("--llm", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--dtype", default="bfloat16", choices=("bfloat16", "float16", "float32"))
    ap.add_argument("--load-4bit", action="store_true",
                    help="quantize the frozen LLM; needs bitsandbytes")
    ap.add_argument("--checkpointing", action="store_true",
                    help="trade a third of throughput for activation memory. Off by "
                         "default: it costs a recompute forward pass, and the run "
                         "falls back to it automatically if it hits OOM")
    ap.add_argument("--proj", default="perceiver", choices=KINDS)
    ap.add_argument("--n-tokens", type=int, default=64, help="soft token budget (not used by pool)")
    ap.add_argument("--pool-grid", default="2,4,4", help="pool projector output grid")
    ap.add_argument("--proj-width", type=int, default=1024)
    ap.add_argument("--proj-depth", type=int, default=4)
    ap.add_argument("--proj-heads", type=int, default=8)
    ap.add_argument("--proj-hidden", type=int, default=2048, help="mlp width for mean/pool")
    ap.add_argument("--prompt", default="Describe what is happening in this video.")
    ap.add_argument("--max-caption-len", type=int, default=48)

    # optimization
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=4,
                    help="lower this before anything else on OOM; pair with --accum")
    ap.add_argument("--eval-batch-size", type=int, default=2,
                    help="generation and retrieval batch; peaks higher than training")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--accum", type=int, default=1, help="gradient accumulation steps")

    # evaluation
    ap.add_argument("--eval-every", type=int, default=2)
    ap.add_argument("--eval-clips", type=int, default=32,
                    help="validation clips for generation and retrieval. Retrieval is "
                         "O(N^2), so this is the expensive knob: 64 costs ~4x what 32 "
                         "does. The publishable number comes from 'run.py evaluate', "
                         "which has its own larger --retrieval-clips")
    ap.add_argument("--max-new-tokens", type=int, default=32)
    ap.add_argument("--retrieval-chunk", type=int, default=16,
                    help="batch for the N^2 retrieval scoring. Inference-only and "
                         "no_grad, so it can run far wider than the training batch")
    ap.add_argument("--repetition-penalty", type=float, default=1.0,
                    help="1.0 = off. Checkpoints ship their own (Qwen2.5: 1.05); "
                         "this overrides it so decoding is recorded, not inherited")

    ap.add_argument("--workers", type=int, default=(0 if os.name == "nt" else 4),
                    help="dataloader processes. 0 on Windows (spawn cost), 4 elsewhere: "
                         "each item is a ~6 MB file read, so 0 leaves the GPU waiting on disk")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path("outputs/runs/caption"))
    args = ap.parse_args(argv)

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.out.mkdir(parents=True, exist_ok=True)

    # ---- data --------------------------------------------------------------
    grid = tuple(int(x) for x in args.jepa_grid.split(","))
    pool_grid = tuple(int(x) for x in args.pool_grid.split(","))

    all_keys, load_v, grid, source = resolve_source(
        args.vjepa_dir, args.packed, grid, args.key_prefix, args.clip_index)
    print(f"source: {source}")
    captions = load_captions(args.captions)
    paired, no_caption, _ = pair_with_captions(all_keys, captions, args.caption_key_prefix)
    if not paired:
        raise SystemExit(
            f"no key matched between the embeddings and {args.captions}. "
            f"Embedding stems look like {all_keys[:3]}; caption keys look like "
            f"{list(captions)[:3]}. --caption-key-prefix strips a prefix from the former."
        )
    if no_caption:
        print(f"note: {len(no_caption)} embeddings have no caption and were dropped")
    if args.limit:
        paired = paired[: args.limit]

    templates = template_vocabulary(captions)
    train_keys, val_keys = resolve_split(args, paired, rng)
    print(f"{len(train_keys)} train / {len(val_keys)} val pairs on {device}")
    print(f"{len(templates)} distinct templates"
          + (f" (chance action accuracy {1/len(templates):.4f})" if templates else
             " -- action accuracy will be unavailable, retrieval is the metric"))

    print("fitting channel statistics on the training split...")
    stats = ChannelStats.fit(train_keys, load_v, limit=512)
    stats.save(args.out / "vjepa_stats.npz")

    # ---- model -------------------------------------------------------------
    print(f"loading {args.llm} (frozen)...")
    llm, tokenizer, d_llm = load_llm(
        args.llm, dtype=args.dtype, load_4bit=args.load_4bit, device=device,
        gradient_checkpointing=args.checkpointing,
    )

    probe = load_v(train_keys[0])
    in_dim, probe_grid = probe.shape[0], tuple(probe.shape[1:])
    if probe_grid != grid:
        raise SystemExit(f"--jepa-grid {grid} but the loader produced {probe_grid}")

    projector = build_projector(
        args.proj, in_dim=in_dim, out_dim=d_llm, grid=grid,
        n_tokens=args.n_tokens, pool_grid=pool_grid, width=args.proj_width,
        depth=args.proj_depth, heads=args.proj_heads, hidden=args.proj_hidden,
    ).to(device)

    model = CaptionModel(projector, llm, tokenizer, prompt=args.prompt)
    n_params = sum(p.numel() for p in projector.parameters())
    k = token_budget(args.proj, args.n_tokens, pool_grid)
    print(f"projector '{args.proj}': {in_dim}ch x {grid} "
          f"({int(np.prod(grid))} patches) -> {k} soft tokens x {d_llm} "
          f"-> {n_params:,} trainable parameters ({n_params/1e6:.1f}M)")

    caption_ids = {
        key: model.encode_caption(caption_for(key, captions, args.caption_key_prefix).text,
                                  args.max_caption_len)
        for key in paired
    }

    train_ds = CaptionedEmbeddings(train_keys, load_v, stats, caption_ids)
    val_ds = CaptionedEmbeddings(val_keys, load_v, stats, caption_ids)
    # persistent_workers matters at 20 epochs: without it the worker pool is torn
    # down and respawned every epoch, and each spawn re-imports torch.
    loader_extras = (
        {"persistent_workers": True, "prefetch_factor": 4} if args.workers else {}
    )
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate,
        num_workers=args.workers, drop_last=len(train_ds) > args.batch_size,
        pin_memory=(device.type == "cuda"), **loader_extras,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, collate_fn=collate, num_workers=args.workers,
        pin_memory=(device.type == "cuda"), **loader_extras,
    )

    opt = torch.optim.AdamW(model.trainable_parameters(), lr=args.lr,
                            weight_decay=args.weight_decay, betas=(0.9, 0.95))

    config = {
        "vjepa_dir": str(args.vjepa_dir),
        "packed": str(args.packed) if args.packed else None,
        "captions": str(args.captions),
        "grid": list(grid), "clip_index": args.clip_index,
        "key_prefix": args.key_prefix, "caption_key_prefix": args.caption_key_prefix,
        "llm": args.llm, "dtype": args.dtype, "load_4bit": args.load_4bit,
        "checkpointing": args.checkpointing,
        "d_llm": d_llm, "in_dim": in_dim,
        "proj": args.proj, "n_tokens": args.n_tokens, "pool_grid": list(pool_grid),
        "proj_width": args.proj_width, "proj_depth": args.proj_depth,
        "proj_heads": args.proj_heads, "proj_hidden": args.proj_hidden,
        "prompt": args.prompt, "max_caption_len": args.max_caption_len,
        "generation": {"max_new_tokens": args.max_new_tokens, "do_sample": False,
                       "repetition_penalty": args.repetition_penalty},
        "soft_tokens": k, "n_params": n_params, "seed": args.seed,
    }
    (args.out / "config.json").write_text(json.dumps(config, indent=2))
    (args.out / "split.json").write_text(
        json.dumps({"train": train_keys, "val": val_keys}, indent=2)
    )

    # ---- train -------------------------------------------------------------
    print(f"\ngradient checkpointing {'ON' if args.checkpointing else 'OFF'}"
          + ("" if args.checkpointing else
             " (a third faster; falls back automatically on OOM)"))
    print(f"generating and retrieving on {min(args.eval_clips, len(val_keys))} val clips "
          f"every {args.eval_every} epochs; those columns are blank in between")
    header = (f"{'ep':>4} {'train':>8} {'val':>8} {'top1':>7} {'medR':>6} "
              f"{'action':>7} {'shufAct':>8} {'object':>7} {'s':>6}")
    print("-" * len(header)); print(header); print("-" * len(header))

    history: List[dict] = []
    checkpointing_on = args.checkpointing
    best_top1 = -1.0
    started = time.time()
    step = 0

    for epoch in range(args.epochs):
        model.projector.train()
        # Accumulated on-device. float(loss) every step forces a host sync, which
        # serializes the very overlap the prefetching workers exist to create.
        running = torch.zeros((), device=device)
        seen = 0
        opt.zero_grad(set_to_none=True)
        train_started = time.time()

        for i, (x_cpu, ids, _) in enumerate(train_loader):
            x = x_cpu.to(device, non_blocking=True)

            lr = args.lr * min(1.0, (step + 1) / max(args.warmup, 1))
            for group in opt.param_groups:
                group["lr"] = lr

            try:
                loss = model(x, ids)
                (loss / args.accum).backward()
            except torch.cuda.OutOfMemoryError:
                # Checkpointing is off by default because it costs a third of the
                # throughput, and on a 12 GB card with a 1.5B model it is not
                # needed. But that depends on the LLM, the batch and the vocab
                # size, so rather than make the user discover the flag from a
                # traceback, turn it on and retry the batch once.
                if checkpointing_on:
                    raise SystemExit(
                        "OOM even with gradient checkpointing enabled. In order:\n"
                        f"  1. --batch-size {max(1, args.batch_size // 2)} "
                        f"--accum {args.accum * 2}   (same effective batch)\n"
                        "  2. --eval-batch-size 1 --retrieval-chunk 4\n"
                        "  3. --max-caption-len 32 --n-tokens 32\n"
                        "  4. PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True\n"
                        "  5. --load-4bit  (needs bitsandbytes; helps weights, not logits)"
                    )
                print()
                print("  OOM -- enabling gradient checkpointing and retrying this batch.")
                print("  Expect roughly a third less throughput from here. Pass")
                print("  --checkpointing to start this way and skip the stumble.")
                opt.zero_grad(set_to_none=True)
                del x
                torch.cuda.empty_cache()
                model.llm.config.use_cache = False
                model.llm.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs={"use_reentrant": False})
                checkpointing_on = True
                config["checkpointing"] = "enabled after OOM"
                (args.out / "config.json").write_text(json.dumps(config, indent=2))

                x = x_cpu.to(device, non_blocking=True)
                loss = model(x, ids)
                (loss / args.accum).backward()

            if (i + 1) % args.accum == 0:
                if args.grad_clip:
                    torch.nn.utils.clip_grad_norm_(model.trainable_parameters(), args.grad_clip)
                opt.step()
                opt.zero_grad(set_to_none=True)

            running += loss.detach() * x.shape[0]
            seen += x.shape[0]
            step += 1

        train_s = time.time() - train_started
        record = {"epoch": epoch, "train_loss": float(running) / max(seen, 1),
                  "lr": lr, "train_s": train_s}
        val_started = time.time()
        record["val_loss"] = val_loss(model, val_loader, device)
        record["val_s"] = time.time() - val_started

        due = epoch == 0 or (epoch + 1) % args.eval_every == 0 or epoch == args.epochs - 1
        if due:
            # Training peak and evaluation peak do not overlap, but the allocator
            # does not know that and will happily OOM on fragmentation alone.
            if device.type == "cuda":
                torch.cuda.empty_cache()
            record.update(evaluate_split(
                model, val_ds, device, captions, templates, args.caption_key_prefix,
                args.eval_clips, args.max_new_tokens, args.eval_batch_size, args.seed,
                args.repetition_penalty, args.retrieval_chunk,
            ))

        history.append(record)

        line = f"{epoch:4d} {record['train_loss']:8.4f} {record['val_loss']:8.4f} "
        if due:
            real = record["controls"]["none"]
            shuf = record["controls"]["shuffle"]

            def cell(v, width=7):
                return f"{'':>{width}}" if v is None else f"{v:{width}.3f}"

            line += (f"{record['top1']:7.3f} {record['median_rank']:6.1f} "
                     f"{cell(real['action_accuracy'])} {cell(shuf['action_accuracy'], 8)} "
                     f"{cell(real['object_recall'])} ")
        else:
            line += f"{'':>7} {'':>6} {'':>7} {'':>8} {'':>7} "
        print(line + f"{time.time()-started:6.0f}")
        if due:
            print(f"      time: train {record['train_s']:.0f}s | val {record['val_s']:.0f}s"
                  f" | generate {record['generate_s']:.0f}s"
                  f" | retrieval {record['retrieval_s']:.0f}s"
                  f"   ({len(train_ds)} clips, {args.workers} workers)")

        payload = {"projector": model.projector_state(), "config": config, "epoch": epoch}
        if due and record["top1"] > best_top1:
            best_top1 = record["top1"]
            torch.save(payload, args.out / "best.pt")
        torch.save(payload, args.out / "last.pt")
        (args.out / "history.json").write_text(json.dumps(history, indent=2, default=float))

    # ---- verdict -----------------------------------------------------------
    final = history[-1]
    print()
    print("=" * 72)
    for entry in final.get("samples", [])[:4]:
        print(f"  pred: {printable(entry['predicted'])}")
        print(f"  true: {printable(entry['reference'])}")
        print()
    print(REPORT_HEADER)
    for name, metrics in final["controls"].items():
        print(format_row(name, metrics))
    print()
    print(f"retrieval top1 {final['top1']:.3f} (chance {final['chance_top1']:.4f}), "
          f"median rank {final['median_rank']:.1f} of {min(args.eval_clips, len(val_keys))}")
    print()
    chance = 1.0 / len(templates) if templates else None
    for line in verdict(final["controls"]["none"], final["controls"]["shuffle"], chance):
        print(line)
    print()
    print("Next: the full control table, including the random-embedding row --")
    print(f"  python src/text_adaptor/run.py evaluate --ckpt {args.out.as_posix()}/best.pt \\")
    print(f"      --vjepa-dir {args.vjepa_dir} --captions {args.captions}")
    print("=" * 72)


if __name__ == "__main__":
    main()
