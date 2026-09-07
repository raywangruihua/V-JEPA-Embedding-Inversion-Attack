"""Pack a directory of per-clip embeddings into one fp16 memmap.

At the real shapes a clip is ``1024 x 8 x 14 x 14`` float32 = **6.42 MB**, so a
5000-clip dataset is **32 GB read per epoch** — spread over 5000 separate
``torch.load`` calls, each of which opens a zipfile and runs the pickle
machinery. On a network-backed volume that alone can be the whole epoch time.

This converts the lot into a single ``clips.npy`` in float16 plus a key index:

* **half the bytes.** 32 GB becomes 16 GB, which is the difference between a
  dataset the OS page cache can hold and one it cannot. On a box with 32 GB of
  RAM the second epoch onward reads from memory rather than disk.
* **one file handle, contiguous rows.** No pickle, no zipfile, no 5000 opens per
  epoch. Row ``i`` is one contiguous slab, so random access across the shuffled
  epoch is still a single seek per clip.
* **memmap, not a load.** Nothing is resident until touched, so this works
  unchanged if the dataset grows past RAM.

float16 is safe here and the reason is worth stating: these values feed a
projector whose input is standardized anyway, so only ~3 significant digits ever
survive to matter, and fp16 covers V-JEPA's range with room to spare (max 65504
against activations of order 1). ``--dtype float32`` is there if you want to rule
it out — it removes the size win but keeps the handle-count win. The packed dtype
is recorded in ``meta.json``, and clips are always handed back as float32.

Usage
-----
::

    python src/text_adaptor/run.py pack --vjepa-dir /path/to/vjepa \\
        --jepa-grid 8,14,14 --out /path/to/packed

then pass ``--packed /path/to/packed`` to ``train`` or ``baseline`` in place of
``--vjepa-dir``.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from data import EmbeddingStore, vjepa_loader


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--vjepa-dir", type=Path, required=True)
    ap.add_argument("--jepa-grid", default="8,14,14")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--dtype", default="float16", choices=("float16", "float32"))
    ap.add_argument("--key-prefix", default=None)
    ap.add_argument("--clip-index", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args(argv)

    grid = tuple(int(x) for x in args.jepa_grid.split(","))
    store = EmbeddingStore(args.vjepa_dir, key_prefix=args.key_prefix)
    keys = store.keys()
    if args.limit:
        keys = keys[: args.limit]
    load_v = vjepa_loader(store, grid, args.clip_index)

    probe = load_v(keys[0])
    shape = tuple(probe.shape)
    dtype = np.dtype(args.dtype)
    total = len(keys) * int(np.prod(shape)) * dtype.itemsize

    args.out.mkdir(parents=True, exist_ok=True)
    print(f"packing {len(keys)} clips of {shape} as {dtype.name}")
    print(f"  -> {args.out / 'clips.npy'} ({total/1e9:.1f} GB)")

    # open_memmap writes a real .npy header, so the result loads with a plain
    # np.load(..., mmap_mode="r") and needs no custom reader.
    array = np.lib.format.open_memmap(
        args.out / "clips.npy", mode="w+", dtype=dtype, shape=(len(keys), *shape)
    )

    started = time.time()
    finite_warnings = 0
    for i, key in enumerate(keys):
        clip = load_v(key)
        if clip.shape != shape:
            raise SystemExit(
                f"{key}: shape {clip.shape} but the first clip was {shape}. "
                "A packed file has to be rectangular -- check --jepa-grid and "
                "whether this directory mixes encodings."
            )
        if dtype == np.float16 and finite_warnings < 5:
            # fp16 overflows to inf above 65504. Silent inf would poison the
            # projector's very first forward pass, so it is worth one check.
            peak = float(np.abs(clip).max())
            if peak > 60000:
                print(f"  WARNING {key}: peak |value| {peak:.0f} is near the fp16 "
                      f"ceiling; consider --dtype float32")
                finite_warnings += 1
        array[i] = clip.astype(dtype, copy=False)

        if (i + 1) % 200 == 0 or i + 1 == len(keys):
            done = i + 1
            rate = done / (time.time() - started)
            eta = (len(keys) - done) / max(rate, 1e-9)
            print(f"  {done}/{len(keys)}  {rate:.0f} clips/s  eta {eta:.0f}s")

    array.flush()
    del array

    (args.out / "keys.json").write_text(json.dumps(keys, indent=2))
    (args.out / "meta.json").write_text(json.dumps({
        "source": str(args.vjepa_dir),
        "grid": list(grid),
        "shape": list(shape),
        "dtype": dtype.name,
        "n_clips": len(keys),
        "clip_index": args.clip_index,
        "key_prefix": args.key_prefix,
    }, indent=2))

    elapsed = time.time() - started
    print(f"\ndone in {elapsed:.0f}s ({total/1e9/max(elapsed,1e-9):.2f} GB/s write)")
    print(f"train against it with:  --packed {args.out.as_posix()}")
    print("(--jepa-grid is read from meta.json from here on)")


if __name__ == "__main__":
    main()
