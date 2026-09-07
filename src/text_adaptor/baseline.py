"""Nearest-neighbour caption transfer. No training, no LLM, no GPU.

For a held-out embedding, find its nearest captioned neighbour in the training
split by cosine similarity and copy that caption verbatim.

Run this **before** training anything. It costs one pass over the data and it
answers the question the expensive route assumes the answer to: is
caption-relevant information present in these embeddings at all? The UMAP result
from earlier in the project -- semantically similar clips cluster -- says it
should be, and this turns that impression into a number.

It is also the floor. A learned projector that does not beat nearest-neighbour
transfer has not learned to read the embedding; it has learned to write SSv2
sentences, and the honest reading of that run is negative no matter how fluent
the output. Nothing else in this directory is worth running until this number
exists.

Similarity is computed on the **mean-pooled** embedding, which throws away all
spatial and temporal structure on purpose. That makes it the exact counterpart of
the ``mean`` rung of the projector ladder: same information budget, no learning.

Usage
-----
::

    python src/text_adaptor/baseline.py --vjepa-dir VDIR --captions labels.json \\
        --jepa-grid 8,14,14 --split outputs/runs/caption_perceiver/split.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from data import (
    ChannelStats,
    caption_for,
    load_captions,
    pair_with_captions,
    resolve_source,
    template_vocabulary,
)
from evaluate import REPORT_HEADER, format_row, printable, summarize


def pooled_matrix(keys, load_v, stats) -> np.ndarray:
    """``(len(keys), D)`` of L2-normalized mean-pooled embeddings.

    Normalizing here means the dot product below *is* cosine similarity, so the
    neighbour search is one matrix multiply.
    """
    rows = []
    for key in keys:
        x = stats.transform(load_v(key))          # (D, F, H, W)
        v = x.reshape(x.shape[0], -1).mean(axis=1)
        rows.append(v / (np.linalg.norm(v) + 1e-8))
    return np.stack(rows).astype(np.float32)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--vjepa-dir", type=Path, default=None)
    ap.add_argument("--packed", type=Path, default=None,
                    help="a directory built by 'run.py pack'")
    ap.add_argument("--captions", type=Path, required=True)
    ap.add_argument("--jepa-grid", default="8,14,14")
    ap.add_argument("--key-prefix", default=None)
    ap.add_argument("--caption-key-prefix", default="")
    ap.add_argument("--clip-index", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--split", type=Path, default=None,
                    help="reuse a training split.json so the number is comparable")
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path("outputs/runs/nn_baseline"))
    args = ap.parse_args(argv)

    rng = np.random.default_rng(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)

    grid = tuple(int(x) for x in args.jepa_grid.split(","))
    all_keys, load_v, grid, source = resolve_source(
        args.vjepa_dir, args.packed, grid, args.key_prefix, args.clip_index)
    print(f"source: {source}")
    captions = load_captions(args.captions)
    paired, no_caption, _ = pair_with_captions(all_keys, captions, args.caption_key_prefix)
    if not paired:
        raise SystemExit(
            f"no key matched between the embeddings and {args.captions}. "
            f"Embedding stems look like {all_keys[:3]}; caption keys look like "
            f"{list(captions)[:3]}."
        )
    if no_caption:
        print(f"note: {len(no_caption)} embeddings have no caption and were dropped")
    if args.limit:
        paired = paired[: args.limit]

    if args.split:
        payload = json.loads(Path(args.split).read_text())
        known = set(paired)
        train_keys = [k for k in payload["train"] if k in known]
        val_keys = [k for k in payload["val"] if k in known]
        if not train_keys or not val_keys:
            raise SystemExit(f"--split {args.split} shares no keys with this data")
    else:
        order = list(paired)
        rng.shuffle(order)
        if len(order) < 2:
            raise SystemExit(f"need at least 2 paired clips to split, got {len(order)}")
        # Ceiling as well as floor: without it a small set leaves no training clips.
        n_val = min(max(1, int(len(order) * args.val_frac)), len(order) - 1)
        train_keys, val_keys = order[n_val:], order[:n_val]

    templates = template_vocabulary(captions)
    print(f"{len(train_keys)} train / {len(val_keys)} val, "
          f"{len(templates)} templates"
          + (f" (chance {1/len(templates):.4f})" if templates else ""))

    stats = ChannelStats.fit(train_keys, load_v, limit=512)
    print("pooling training embeddings...")
    train_matrix = pooled_matrix(train_keys, load_v, stats)
    print("pooling validation embeddings...")
    val_matrix = pooled_matrix(val_keys, load_v, stats)

    similarity = val_matrix @ train_matrix.T           # (n_val, n_train), cosine
    nearest = similarity.argmax(axis=1)

    refs = {k: caption_for(k, captions, args.caption_key_prefix) for k in val_keys}
    predictions = {
        key: caption_for(train_keys[j], captions, args.caption_key_prefix).text
        for key, j in zip(val_keys, nearest)
    }

    metrics = summarize(predictions, refs, templates)
    top1_self = float(np.mean([
        predictions[k] == refs[k].text for k in val_keys
    ]))

    print()
    print("=" * 72)
    for key in val_keys[:6]:
        print(f"  copied: {printable(predictions[key])}")
        print(f"  true:   {printable(refs[key].text)}")
        print()
    print(REPORT_HEADER)
    print(format_row("nn", metrics))
    print()
    print(f"exact caption match {top1_self:.3f}, "
          f"mean cosine to neighbour {float(similarity.max(axis=1).mean()):.4f}")
    print()
    if templates and metrics["action_accuracy"] is not None:
        chance = 1.0 / len(templates)
        ratio = metrics["action_accuracy"] / chance
        print(f"action accuracy is {ratio:.1f}x chance.")
        if ratio < 2.0:
            print("VERDICT: barely above chance. Caption-relevant information is not")
            print("  linearly available in the pooled embedding. Before blaming V-JEPA,")
            print("  check the pairing -- data.py warns that a systematic key mismatch")
            print("  looks exactly like this, and it would also explain the earlier")
            print("  latent-decoding negatives.")
        else:
            print("VERDICT: semantics are present and retrievable without any training.")
            print("  This is the floor the learned projector has to clear. If it does not,")
            print("  the projector is writing SSv2 sentences rather than reading the input.")
    print("=" * 72)

    payload = {
        "metrics": metrics,
        "exact_caption_match": top1_self,
        "n_train": len(train_keys),
        "n_val": len(val_keys),
        "chance_action": 1.0 / len(templates) if templates else None,
    }
    (args.out / "nn_baseline.json").write_text(json.dumps(payload, indent=2, default=float))
    (args.out / "split.json").write_text(
        json.dumps({"train": train_keys, "val": val_keys}, indent=2)
    )
    with (args.out / "captions.jsonl").open("w", encoding="utf-8") as fh:
        for key in val_keys:
            fh.write(json.dumps({
                "key": key, "reference": refs[key].text, "copied": predictions[key],
            }) + "\n")
    print(f"wrote {args.out}/nn_baseline.json")


if __name__ == "__main__":
    main()
