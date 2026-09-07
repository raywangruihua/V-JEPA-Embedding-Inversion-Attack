# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "torch>=2.4",
#     "transformers>=4.45",
#     "numpy>=1.24",
#     "accelerate>=0.30",
#     "sentencepiece",
#     "protobuf",
# ]
# ///
"""Single entry point for the text adaptor. ``uv run`` installs everything.

The dependency block above is PEP 723 inline script metadata, so the whole
directory is self-contained: copy it to the Linux box and run

    uv run src/text_adaptor/run.py selftest

with no venv, no requirements file and no install step. uv reads the header,
builds an isolated environment the first time and caches it after that. This is
the same ``uv run --no-project`` arrangement already used for verification on the
Windows box, moved into the file so the invocation stops needing to carry it.

Subcommands, in the order they should actually be run::

    selftest   shapes, masking and the position-embedding check. CPU, seconds,
               downloads nothing. Run it after copying the directory anywhere.
    pack       fold N per-clip files into one fp16 memmap. Optional, but at
               5000 clips it halves 32 GB/epoch of reads to 16 GB.
    baseline   nearest-neighbour caption transfer. No training, no LLM.
               This is the floor; get the number before training anything.
    train      the projector -> frozen LLM run.
    evaluate   the full control table against a trained checkpoint.

Every subcommand takes ``--help``.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Running a script already puts its directory on sys.path, but uv's launcher and
# any future `python -m` invocation do not both guarantee it, and a silent
# ImportError here would be a poor welcome to a freshly copied directory.
sys.path.insert(0, str(Path(__file__).resolve().parent))


def cmd_selftest(argv) -> int:
    """Check the parts that fail *quietly* if they are wrong.

    Every assertion here corresponds to a bug that produces plausible captions
    rather than an exception:

    * a projector whose output width does not match the LLM would raise, so it
      is not interesting -- but a projector that ignores frame order would not,
      and on a motion dataset that is fatal;
    * label masking that is off by one trains the model to predict its own
      prompt, which lowers the loss and teaches it nothing;
    * a caption file parsed into the wrong shape pairs every clip with the wrong
      text, which looks exactly like "V-JEPA leaks nothing".
    """
    import json
    import tempfile

    import torch
    import torch.nn as nn

    from data import load_captions, normalize_template, template_vocabulary
    from evaluate import match_template, token_f1
    from model import IGNORE_INDEX, apply_control, assemble_batch
    from projector import KINDS, build_projector, token_budget

    grid = (4, 3, 3)
    d_in, d_out, batch = 16, 32, 2
    x = torch.randn(batch, d_in, *grid)
    failures = []

    def check(name: str, condition: bool, detail: str = "") -> None:
        print(f"  {'PASS' if condition else 'FAIL'}  {name}" + (f" -- {detail}" if detail and not condition else ""))
        if not condition:
            failures.append(name)

    print("projector shapes")
    for kind in KINDS:
        proj = build_projector(kind, d_in, d_out, grid, n_tokens=8,
                               pool_grid=(2, 2, 2), width=32, depth=2, heads=4, hidden=64)
        out = proj(x)
        expected = token_budget(kind, 8, (2, 2, 2))
        check(f"{kind}: (B,{d_in},{grid}) -> (B,{expected},{d_out})",
              tuple(out.shape) == (batch, expected, d_out), f"got {tuple(out.shape)}")
        check(f"{kind}: n_tokens attribute agrees", proj.n_tokens == expected)

    print("\ngeometry")
    # The check that matters: reversing frame order must change the output, or
    # the head is a bag of patches on a motion dataset.
    torch.manual_seed(0)
    perceiver = build_projector("perceiver", d_in, d_out, grid, n_tokens=8,
                                width=32, depth=2, heads=4)
    perceiver.eval()
    with torch.no_grad():
        straight, reversed_ = perceiver(x), perceiver(x.flip(2))
    check("perceiver distinguishes frame order",
          not torch.allclose(straight, reversed_, atol=1e-4),
          "position embedding is not reaching the cross-attention memory")

    mean_proj = build_projector("mean", d_in, d_out, grid, n_tokens=8, hidden=64)
    mean_proj.eval()
    with torch.no_grad():
        check("mean is order-blind, as designed",
              torch.allclose(mean_proj(x), mean_proj(x.flip(2)), atol=1e-4))

    print("\nsequence assembly")
    vocab, width = 64, 8
    embed = nn.Embedding(vocab, width)
    soft = torch.randn(3, 4, width)
    pre = torch.tensor([1], dtype=torch.long)
    mid = torch.tensor([2, 3], dtype=torch.long)
    caps = [torch.tensor([10, 11, 12]), torch.tensor([20, 21, 22, 23]), torch.tensor([30])]
    embeds, mask, labels = assemble_batch(soft, pre, mid, caps, embed, pad_id=0)

    prefix_len = 1 + 4 + 2
    total = prefix_len + 4
    check("embeds shape", tuple(embeds.shape) == (3, total, width), f"got {tuple(embeds.shape)}")
    check("prefix is fully masked out of the loss",
          bool((labels[:, :prefix_len] == IGNORE_INDEX).all()))
    check("caption ids land immediately after the prefix",
          labels[0, prefix_len:prefix_len + 3].tolist() == [10, 11, 12],
          f"got {labels[0, prefix_len:prefix_len + 3].tolist()}")
    check("ragged tail is masked, not learned",
          int(labels[2, prefix_len + 1]) == IGNORE_INDEX)
    check("attention ignores padding",
          mask[2].tolist() == [1] * (prefix_len + 1) + [0, 0, 0],
          f"got {mask[2].tolist()}")
    check("soft tokens survive into the sequence unchanged",
          torch.allclose(embeds[:, 1:5], soft))

    print("\ncontrols")
    check("random control resamples", not torch.allclose(apply_control(x, "random"), x))
    check("shuffle control moves rows", not torch.allclose(apply_control(x, "shuffle"), x))
    check("shuffle preserves the multiset of embeddings",
          torch.allclose(apply_control(x, "shuffle").sort(dim=0).values, x.sort(dim=0).values))
    check("none is a passthrough", torch.allclose(apply_control(x, "none"), x))

    print("\ncaption parsing")
    ssv2 = [{"id": "16290", "label": "Pushing a book from left to right",
             "template": "Pushing [something] from left to right", "placeholders": ["book"]}]
    flat = {"16290": "Pushing a book from left to right"}
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        (tmp / "ssv2.json").write_text(json.dumps(ssv2))
        (tmp / "flat.json").write_text(json.dumps(flat))
        (tmp / "lines.jsonl").write_text("\n".join(json.dumps(e) for e in ssv2))

        parsed = load_captions(tmp / "ssv2.json")
        check("ssv2 list form", parsed["16290"].informative_placeholders == ["book"])
        check("ssv2 template survives", parsed["16290"].template is not None)
        check("flat mapping form", load_captions(tmp / "flat.json")["16290"].text.startswith("Pushing"))
        check("jsonl form", load_captions(tmp / "lines.jsonl")["16290"].placeholders == ["book"])
        check("template vocabulary", template_vocabulary(parsed) ==
              ["pushing something from left to right"])

    print("\ndataloader worker safety")
    # DataLoader workers receive the dataset across a process boundary. On
    # Windows that is spawn; from Python 3.14 the POSIX default is forkserver
    # too, so anything unpicklable here turns --workers N into a crash on an
    # interpreter upgrade alone. Both loaders were closures once and both broke.
    import pickle

    from data import PackedClips, VJepaLoader, as_vjepa_grid  # noqa: F401

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        raw = tmp / "raw"
        raw.mkdir()
        import numpy as _np

        for i in range(3):
            _np.save(raw / f"{i}.npy", _np.zeros((36, d_in), dtype=_np.float32))

        from data import EmbeddingStore, vjepa_loader

        loader = vjepa_loader(EmbeddingStore(raw), (4, 3, 3))
        try:
            revived = pickle.loads(pickle.dumps(loader))
            ok = revived("0").shape == (d_in, 4, 3, 3)
        except Exception as exc:  # pragma: no cover - that is the failure we test
            ok = False
            print(f"        {type(exc).__name__}: {exc}")
        check("vjepa_loader survives pickling (spawn/forkserver workers)", ok)

        packed = tmp / "packed"
        import pack as _pack

        _pack.main(["--vjepa-dir", str(raw), "--jepa-grid", "4,3,3",
                    "--out", str(packed)])
        clips = PackedClips(packed)
        try:
            revived = pickle.loads(pickle.dumps(clips.loader()))
            ok = revived("0").shape == (d_in, 4, 3, 3)
            revived.clips.close()
        except Exception as exc:  # pragma: no cover
            ok = False
            print(f"        {type(exc).__name__}: {exc}")
        finally:
            # Windows will not remove a directory holding a mapped file.
            clips.close()
        check("packed loader survives pickling", ok)

    print("\nmetrics")
    templates = ["pushing something from left to right",
                 "dropping something into something",
                 "tearing something into two pieces"]
    check("template matching picks the right action",
          match_template("Pushing a mug from left to right", templates) == templates[0],
          f"got {match_template('Pushing a mug from left to right', templates)}")
    check("token_f1 is 1.0 on identical bags", abs(token_f1(["a", "b"], ["b", "a"]) - 1.0) < 1e-9)
    check("token_f1 is 0.0 on disjoint bags", token_f1(["a"], ["b"]) == 0.0)
    check("normalize_template strips brackets",
          normalize_template("Pushing [something] from left to right")
          == "pushing something from left to right")

    llm_name = None
    if argv:
        if argv[0] in ("--llm", "-m") and len(argv) > 1:
            llm_name = argv[1]
        elif argv[0] in ("-h", "--help"):
            print("usage: run.py selftest [--llm NAME]")
            print("  --llm NAME  additionally check that gradients reach the projector")
            print("              through a frozen, checkpointed LLM. Downloads NAME.")
            print("              'hf-internal-testing/tiny-random-LlamaForCausalLM' is")
            print("              a few MB and exercises the same code path as Qwen.")
            return 0

    if llm_name:
        print(f"\ngradient flow through a frozen {llm_name}")
        import torch.nn as _nn  # noqa: F401  (kept local; this branch is opt-in)

        from model import CaptionModel, load_llm

        for checkpointing in (True, False):
            llm, tok, d_llm = load_llm(llm_name, dtype="float32",
                                       device=torch.device("cpu"),
                                       gradient_checkpointing=checkpointing)
            proj = build_projector("perceiver", d_in, d_llm, grid, n_tokens=4,
                                   width=32, depth=2, heads=4)
            captioner = CaptionModel(proj, llm, tok, prompt="Describe the video.")
            caps = [captioner.encode_caption("Pushing a book from left to right"),
                    captioner.encode_caption("Moving a pen up")]
            captioner(x, caps).backward()

            trainable = list(captioner.trainable_parameters())
            got = sum(1 for p in trainable
                      if p.grad is not None and float(p.grad.abs().sum()) > 0)
            leaked = any(p.grad is not None and float(p.grad.abs().sum()) > 0
                         for p in llm.parameters())

            # The one that fails silently: reentrant checkpointing over a stack
            # with no trainable parameters builds no graph, the projector gets
            # zero gradient, and the loss curve still looks like training.
            check(f"checkpointing={checkpointing}: gradient reaches the projector",
                  got == len(trainable), f"only {got}/{len(trainable)} tensors")
            check(f"checkpointing={checkpointing}: frozen LLM stays frozen", not leaked)

    print()
    if failures:
        print(f"{len(failures)} check(s) FAILED: {', '.join(failures)}")
        return 1
    print("all checks passed -- shapes, masking, geometry and parsing are sound.")
    if not llm_name:
        print("Nothing here touched a GPU or downloaded a model. To also verify that")
        print("gradients reach the projector through a frozen LLM (the failure that")
        print("looks like training but is not), re-run with:")
        print("  run.py selftest --llm hf-internal-testing/tiny-random-LlamaForCausalLM")
    return 0


COMMANDS = {
    "selftest": cmd_selftest,
    "pack": lambda argv: __import__("pack").main(argv),
    "baseline": lambda argv: __import__("baseline").main(argv),
    "train": lambda argv: __import__("train").main(argv),
    "evaluate": lambda argv: __import__("evaluate").main(argv),
}


def main() -> int:
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help", "help"):
        print(__doc__)
        print("commands: " + ", ".join(COMMANDS))
        return 0
    command = sys.argv[1]
    if command not in COMMANDS:
        print(f"unknown command: {command!r}")
        print("commands: " + ", ".join(COMMANDS))
        return 2
    return COMMANDS[command](sys.argv[2:]) or 0


if __name__ == "__main__":
    raise SystemExit(main())
