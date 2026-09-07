"""Metrics, and the controls without which none of them mean anything.

A frozen LLM is a fluency machine. Point one at noise and it will write a
confident, well-formed, plausibly-SSv2 caption, because that is what the
projector's unconditional prior plus a competent decoder produce. So the only
honest question is never "are the captions good" but **"are they better than the
same model given nothing"**, and every number below is reported per control:

* ``none``    -- the real embedding.
* ``random``  -- N(0, 1) in standardized space, i.e. a plausible-looking fake.
* ``shuffle`` -- another clip's real embedding. The strict one: it is exactly
  in-distribution, so it cannot be beaten by a model that has merely learned what
  V-JEPA embeddings look like in general.
* ``zero``    -- the obvious null. Kept because seeing it fail *differently* from
  the other two is informative.

``none`` minus ``shuffle`` is the leakage. If that gap is small, the captions are
the prior talking and the correct conclusion is that nothing was recovered --
regardless of how good they read.

What is measured
----------------
**Action accuracy.** The generated caption is matched to the nearest of SSv2's
174 templates by token F1, and scored against the clip's true template. Chance is
1/174 = 0.6%. This is the attribute V-JEPA is expected to have kept.

**Object recall.** The fraction of the clip's informative placeholder words that
appear in the generated caption -- "book" in "Pushing a book from left to right".
This is the attribute the latent-decoding work says should be *gone*, and the
two numbers side by side are the per-attribute leakage profile.

**Caption-to-clip retrieval.** For each candidate clip j, score
``log p(caption_i | embedding_j)`` and rank. Top-1 is the fraction where the true
clip wins. This is the metric that cannot be gamed by fluency at all: a caption
generated from the prior scores the same against every clip, so it lands at
chance by construction. When the generative numbers and this one disagree, this
one is right.

BLEU and CIDEr are deliberately absent. They reward exactly the fluency that is
the failure mode here, and a model that memorized the SSv2 idiom and ignored its
input would score well on both.
"""

from __future__ import annotations

import re
import sys
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from data import Caption, normalize_template

# Function words carry no information about the clip but dominate token overlap
# on captions this short, so template matching would otherwise be decided by how
# many times "the" appears.
STOPWORDS = {
    "a", "an", "the", "of", "to", "from", "in", "on", "at", "it", "its", "and",
    "or", "with", "into", "onto", "out", "up", "down", "is", "are", "be", "being",
    "so", "that", "this", "then", "but", "not", "for", "by", "as", "something",
}


def words(text: str) -> List[str]:
    """Lowercase alphanumeric tokens, stopwords removed."""
    return [w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in STOPWORDS]


def printable(text: str) -> str:
    """Drop characters the active stdout encoding cannot represent.

    Generated captions are arbitrary model output, and the Windows console
    defaults to cp1252, where one stray glyph raises UnicodeEncodeError. Without
    this, a finished run dies while printing its own report -- which is exactly
    when it is most annoying. Losing a character from a printed sample is
    strictly better than losing the summary; the JSON on disk keeps the original.
    """
    encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
    return text.encode(encoding, errors="replace").decode(encoding, errors="replace")


def token_f1(a: Sequence[str], b: Sequence[str]) -> float:
    """Symmetric bag-of-words F1. Multiplicity ignored -- these captions are short."""
    sa, sb = set(a), set(b)
    if not sa or not sb:
        return 0.0
    overlap = len(sa & sb)
    if not overlap:
        return 0.0
    precision, recall = overlap / len(sa), overlap / len(sb)
    return 2 * precision * recall / (precision + recall)


def match_template(caption: str, templates: Sequence[str]) -> Optional[str]:
    """Snap a free-text caption onto the nearest template. ``None`` if no overlap.

    The model writes prose; the label set is 174 fixed strings. Something has to
    bridge them, and token F1 against each template is the cheapest bridge that
    does not need another model in the loop. Ties break toward the first
    alphabetically, which is arbitrary but at least deterministic.
    """
    if not templates:
        return None
    cw = words(caption)
    if not cw:
        return None
    best, best_score = None, 0.0
    for template in templates:
        score = token_f1(cw, words(template))
        if score > best_score:
            best, best_score = template, score
    return best


def action_accuracy(
    predictions: Dict[str, str], references: Dict[str, Caption], templates: Sequence[str]
) -> Optional[float]:
    """Fraction of clips whose predicted caption snaps to the true template.

    ``None`` when the caption source carries no templates -- a VLM re-captioning
    pass will not, and reporting 0.0 there would read as a negative result rather
    than an absent measurement.
    """
    if not templates:
        return None
    scored = [
        float(match_template(pred, templates) == normalize_template(references[key].template))
        for key, pred in predictions.items()
        if key in references and references[key].template
    ]
    return float(np.mean(scored)) if scored else None


def object_recall(
    predictions: Dict[str, str], references: Dict[str, Caption]
) -> Optional[float]:
    """Mean fraction of a clip's informative placeholder words present in its caption.

    Clips whose placeholders are all uninformative ("something") are skipped
    rather than scored 0 -- there was nothing to recover, and counting them would
    drag the number down in proportion to how generic the dataset is.
    """
    scored = []
    for key, pred in predictions.items():
        ref = references.get(key)
        if ref is None:
            continue
        targets = ref.informative_placeholders
        if not targets:
            continue
        found = sum(
            any(w in words(pred) for w in words(target))
            for target in targets
        )
        scored.append(found / len(targets))
    return float(np.mean(scored)) if scored else None


def caption_word_f1(
    predictions: Dict[str, str], references: Dict[str, Caption]
) -> float:
    """Whole-caption token F1. The fallback when no template structure exists."""
    scored = [
        token_f1(words(pred), words(references[key].text))
        for key, pred in predictions.items()
        if key in references
    ]
    return float(np.mean(scored)) if scored else 0.0


@torch.no_grad()
def caption_retrieval(
    model,
    embeddings: torch.Tensor,
    caption_ids: Sequence[torch.Tensor],
    chunk: int = 4,
    device=None,
) -> Dict[str, float]:
    """Rank clips by ``log p(caption_i | embedding_j)``.

    ``embeddings`` is ``(N, D, F, H, W)`` and is expected to live on the **CPU**:
    at the real shapes one clip is 1024 x 8 x 14 x 14 x 4 B = 6.4 MB, so 128 of
    them pinned to the GPU is 0.8 GB sitting idle beside a language model for the
    whole evaluation. Chunks are moved across as they are needed instead.

    ``caption_ids`` are the matching N ground-truth captions. Costs N^2 forward
    passes, so N is kept small by the caller -- the number is stable well before
    it needs to be large.

    Ground-truth captions rather than generated ones, on purpose: it removes
    sampling noise, it costs no generation, and it asks the cleaner question of
    whether the embedding *identifies* the caption at all.
    """
    n = len(caption_ids)
    if n < 2:
        return {"top1": float("nan"), "top5": float("nan"),
                "median_rank": float("nan"), "chance_top1": float("nan")}

    if device is None:
        device = next(model.projector.parameters()).device

    scores = torch.zeros(n, n)
    for i in range(n):
        for start in range(0, n, chunk):
            stop = min(start + chunk, n)
            batch = embeddings[start:stop].to(device, non_blocking=True)
            repeated = [caption_ids[i]] * (stop - start)
            scores[i, start:stop] = model.sequence_logprob(batch, repeated).float().cpu()
            del batch

    order = scores.argsort(dim=1, descending=True)
    truth = torch.arange(n)[:, None]
    ranks = (order == truth).float().argmax(dim=1)  # 0-based rank of the true clip

    return {
        "top1": float((ranks == 0).float().mean()),
        "top5": float((ranks < 5).float().mean()),
        "median_rank": float(ranks.median()) + 1.0,
        "chance_top1": 1.0 / n,
    }


def summarize(
    predictions: Dict[str, str],
    references: Dict[str, Caption],
    templates: Sequence[str],
) -> Dict[str, Optional[float]]:
    """The generative half of the report, for one control."""
    return {
        "action_accuracy": action_accuracy(predictions, references, templates),
        "object_recall": object_recall(predictions, references),
        "word_f1": caption_word_f1(predictions, references),
    }


def format_row(name: str, metrics: Dict[str, Optional[float]]) -> str:
    def cell(value) -> str:
        return "     --" if value is None else f"{value:7.3f}"

    return (
        f"{name:>9} {cell(metrics.get('action_accuracy'))} "
        f"{cell(metrics.get('object_recall'))} {cell(metrics.get('word_f1'))}"
    )


REPORT_HEADER = f"{'control':>9} {'action':>7} {'object':>7} {'wordF1':>7}"


def verdict(real: Dict[str, Optional[float]], shuffled: Dict[str, Optional[float]],
            chance: Optional[float]) -> List[str]:
    """The sentence the report exists to produce, as printable lines."""
    lines = []
    a_real = real.get("action_accuracy")
    a_fake = shuffled.get("action_accuracy")
    if a_real is None or a_fake is None:
        lines.append("VERDICT: no templates in the caption source, so action accuracy could")
        lines.append("  not be computed. Retrieval top1 is the number to read instead.")
        return lines

    gap = a_real - a_fake
    lines.append(f"action accuracy {a_real:.3f} real vs {a_fake:.3f} on shuffled embeddings"
                 + (f" (chance {chance:.4f})" if chance else ""))
    if gap < 0.02:
        lines.append("VERDICT: the captions are the projector's prior, not the embedding.")
        lines.append("  A shuffled embedding scores the same, so nothing clip-specific is")
        lines.append("  getting through. Check the pairing before concluding anything about")
        lines.append("  V-JEPA -- data.py warns that a systematic mismatch looks exactly")
        lines.append("  like this.")
    else:
        lines.append("VERDICT: the embedding carries clip-specific action information.")
        lines.append("  Compare object recall against action accuracy: a large gap is the")
        lines.append("  per-attribute leakage profile the paper is built on.")
    return lines


# ---------------------------------------------------------------------------
# CLI -- the full control table, run against a trained checkpoint
# ---------------------------------------------------------------------------


def main(argv=None) -> None:
    """Score a checkpoint under every control and write ``eval.json``.

    Separate from the in-training evaluation, which carries only ``shuffle`` to
    keep epochs cheap. This is the one that goes in the writeup.
    """
    import argparse
    import json
    from pathlib import Path

    from data import (
        ChannelStats, caption_for, load_captions, pair_with_captions,
        resolve_source, template_vocabulary,
    )
    from model import CONTROLS, apply_control, load_checkpoint

    ap = argparse.ArgumentParser(description="score a trained projector under every control")
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--vjepa-dir", type=Path, default=None)
    ap.add_argument("--packed", type=Path, default=None,
                    help="a directory built by 'run.py pack'")
    ap.add_argument("--captions", type=Path, required=True)
    ap.add_argument("--split", type=Path, default=None,
                    help="defaults to split.json beside the checkpoint")
    ap.add_argument("--stats", type=Path, default=None,
                    help="defaults to vjepa_stats.npz beside the checkpoint")
    ap.add_argument("--n-clips", type=int, default=128)
    ap.add_argument("--retrieval-clips", type=int, default=64,
                    help="retrieval costs N^2 forwards, so it gets its own smaller N")
    ap.add_argument("--batch-size", type=int, default=2,
                    help="generation and retrieval batch; lower this on OOM")
    ap.add_argument("--max-new-tokens", type=int, default=32)
    ap.add_argument("--sample", action="store_true", help="sample instead of greedy decoding")
    ap.add_argument("--repetition-penalty", type=float, default=1.0,
                    help="1.0 = off; overrides whatever the checkpoint ships")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=None,
                    help="defaults to eval.json beside the checkpoint")
    args = ap.parse_args(argv)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    run_dir = args.ckpt.parent

    model, config = load_checkpoint(args.ckpt, device)
    stats = ChannelStats.load(args.stats or run_dir / "vjepa_stats.npz")

    all_keys, load_v, _, source = resolve_source(
        args.vjepa_dir, args.packed, tuple(config["grid"]),
        config.get("key_prefix"), config.get("clip_index", 0))
    print(f"source: {source}")
    captions = load_captions(args.captions)
    prefix = config.get("caption_key_prefix", "")
    paired, _, _ = pair_with_captions(all_keys, captions, prefix)
    templates = template_vocabulary(captions)

    split_path = args.split or run_dir / "split.json"
    if split_path.exists():
        val_keys = [k for k in json.loads(split_path.read_text())["val"] if k in set(paired)]
    else:
        print(f"warning: no split at {split_path} -- scoring on ALL keys, which")
        print("  includes anything the projector was trained on. Not a held-out number.")
        val_keys = paired

    keys = val_keys[: args.n_clips]
    if not keys:
        raise SystemExit("no validation keys to score")
    refs = {k: caption_for(k, captions, prefix) for k in keys}

    print(f"scoring {len(keys)} held-out clips on {device}")
    # Held on the CPU; chunks move to the GPU as they are used. See
    # caption_retrieval for why.
    embeddings = torch.stack(
        [torch.from_numpy(stats.transform(load_v(k))) for k in keys]
    )

    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    results = {
        "n_clips": len(keys), "controls": {}, "checkpoint": str(args.ckpt),
        "generation": {"max_new_tokens": args.max_new_tokens, "do_sample": args.sample,
                       "repetition_penalty": args.repetition_penalty},
    }
    predictions_by_control = {}

    for control in CONTROLS:
        predictions: Dict[str, str] = {}
        for start in range(0, len(keys), args.batch_size):
            chunk = embeddings[start:start + args.batch_size]
            if control == "shuffle" and chunk.shape[0] < 2:
                continue
            texts = model.generate(
                apply_control(chunk, control, generator).to(device),
                max_new_tokens=args.max_new_tokens, do_sample=args.sample,
                repetition_penalty=args.repetition_penalty,
            )
            for key, text in zip(keys[start:start + args.batch_size], texts):
                predictions[key] = text
        predictions_by_control[control] = predictions
        results["controls"][control] = summarize(predictions, refs, templates)

    n_ret = min(args.retrieval_clips, len(keys))
    caption_ids = [model.encode_caption(refs[k].text, config["max_caption_len"])
                   for k in keys[:n_ret]]
    print(f"caption-to-clip retrieval over {n_ret} clips ({n_ret ** 2} forward passes)...")
    results["retrieval"] = caption_retrieval(
        model, embeddings[:n_ret], caption_ids,
        chunk=max(1, args.batch_size // 2), device=device,
    )

    # ---- report ------------------------------------------------------------
    print()
    print("=" * 72)
    for key in keys[:6]:
        print(f"  pred: {printable(predictions_by_control['none'][key])}")
        print(f"  true: {printable(refs[key].text)}")
        print()
    print(REPORT_HEADER)
    for control in CONTROLS:
        print(format_row(control, results["controls"][control]))
    print()
    ret = results["retrieval"]
    print(f"retrieval  top1 {ret['top1']:.3f}  top5 {ret['top5']:.3f}  "
          f"median rank {ret['median_rank']:.1f} of {n_ret}  "
          f"(chance top1 {ret['chance_top1']:.4f})")
    print()
    chance = 1.0 / len(templates) if templates else None
    for line in verdict(results["controls"]["none"], results["controls"]["shuffle"], chance):
        print(line)
    print("=" * 72)

    out = args.out or run_dir / "eval.json"
    out.write_text(json.dumps(results, indent=2, default=float))
    captions_out = run_dir / "captions.jsonl"
    with captions_out.open("w", encoding="utf-8") as fh:
        for key in keys:
            fh.write(json.dumps({
                "key": key,
                "reference": refs[key].text,
                **{f"pred_{c}": predictions_by_control[c].get(key, "")
                   for c in CONTROLS},
            }) + "\n")
    print(f"wrote {out} and {captions_out}")


if __name__ == "__main__":
    main()
