"""The projector heads: ``mean``, ``pool``, ``perceiver``. And the ladder is the point.

Each head maps a standardized V-JEPA grid ``(D, Fv, Hv, Wv)`` onto a fixed budget
of ``K`` **soft tokens** of width ``d_llm`` -- vectors that get spliced into the
language model's input sequence where token embeddings would normally go. The LLM
never sees the video and its weights never move; everything the caption can
possibly know about the clip has to arrive through these K vectors.

Run all three, for the same reason ``standard_adaptors/adaptor.py`` runs
``linear -> mlp -> conv``:

* ``mean`` mean-pools every patch into one vector, then expands it to K tokens.
  It destroys all spatial and temporal structure by construction. Whatever it
  scores is what a *bag of features* is worth, and it is the number that says
  how much of any later result was structure rather than gist.
* ``pool`` average-pools onto a small grid and applies a positionwise MLP.
  Structure survives, but nothing chooses *what* to keep -- every region gets
  equal budget. This is the LLaVA stage-1 recipe with the token count brought
  down to something a 12 GB card can carry.
* ``perceiver`` lets K learned query vectors cross-attend into all N patches, so
  the model decides which regions and frames to spend its budget on.

``mean -> pool`` isolates the value of spatial and temporal structure;
``pool -> perceiver`` isolates the value of *learned selection*. Skipping ``pool``
makes a ``perceiver`` win uninterpretable, because two things changed at once.
That is the same trap the earlier ladder was built to avoid.

Position embeddings are not optional
------------------------------------
Cross-attention is permutation-invariant over its memory. Feed raw V-JEPA patches
to ``perceiver`` with no position information and the head genuinely cannot tell
frame 0 from frame 7, or top-left from bottom-right -- it is a bag-of-patches
model on a task where *motion is the entire signal*. It will still emit fluent
captions, so the failure is invisible except through the retrieval control.

Hence :class:`Factorised3DPosition`, added to the memory before attention. It is
factorised (separate F, H and W tables summed) rather than a full N-entry table
so it stays meaningful when the grid changes, and it is the same construction the
DiT in ``diffusion_adaptor/model.py`` uses for the same tensor.

``pool`` needs none of this: its output tokens are *ordered* by grid position, so
the geometry is carried by which slot a token sits in.
"""

from __future__ import annotations

import math
from typing import Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

KINDS = ("mean", "pool", "perceiver")


class Factorised3DPosition(nn.Module):
    """Learned position embedding for a ``(F, H, W)`` patch grid.

    Three small tables summed, rather than one table of ``F*H*W`` rows. Costs
    ``(F + H + W) * d`` parameters instead of ``F*H*W*d``, and -- more usefully --
    a model trained at one grid degrades gracefully at another.
    """

    def __init__(self, dim: int, grid: Sequence[int]) -> None:
        super().__init__()
        gf, gh, gw = (int(g) for g in grid)
        self.grid = (gf, gh, gw)
        self.frame = nn.Parameter(torch.randn(gf, dim) * 0.02)
        self.row = nn.Parameter(torch.randn(gh, dim) * 0.02)
        self.col = nn.Parameter(torch.randn(gw, dim) * 0.02)

    def forward(self) -> torch.Tensor:
        """-> ``(N, dim)`` in the same patch order as ``flatten_grid``."""
        gf, gh, gw = self.grid
        pos = (
            self.frame[:, None, None, :]
            + self.row[None, :, None, :]
            + self.col[None, None, :, :]
        )
        return pos.reshape(gf * gh * gw, -1)


def flatten_grid(x: torch.Tensor) -> torch.Tensor:
    """``(B, D, F, H, W)`` -> ``(B, N, D)`` with N ordered frame-major.

    This is the inverse of the reshape in ``data.as_vjepa_grid``, so a patch's
    index here is its index in the original stored ``(N, D)`` dump.
    """
    b, d = x.shape[0], x.shape[1]
    return x.reshape(b, d, -1).transpose(1, 2).contiguous()


class FeedForward(nn.Module):
    """The usual two-layer MLP block, GELU in the middle."""

    def __init__(self, dim: int, mult: int = 4) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * mult),
            nn.GELU(),
            nn.Linear(dim * mult, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MeanProjector(nn.Module):
    """Mean-pool everything, then expand to K tokens through an MLP.

    The deliberate floor. If this matches ``perceiver``, structure bought nothing
    and the honest headline is that V-JEPA leaks a *gist vector*, no more.
    """

    def __init__(self, in_dim: int, out_dim: int, n_tokens: int, hidden: int = 2048,
                 depth: int = 2) -> None:
        super().__init__()
        self.n_tokens = n_tokens
        self.out_dim = out_dim
        layers = [nn.LayerNorm(in_dim), nn.Linear(in_dim, hidden), nn.GELU()]
        for _ in range(max(depth - 1, 0)):
            layers += [nn.Linear(hidden, hidden), nn.GELU()]
        layers += [nn.Linear(hidden, n_tokens * out_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pooled = x.mean(dim=(2, 3, 4))                      # (B, D)
        out = self.net(pooled)                              # (B, K*d)
        return out.reshape(x.shape[0], self.n_tokens, self.out_dim)


class PoolProjector(nn.Module):
    """Average-pool onto a small grid, then a positionwise MLP. LLaVA stage-1.

    ``adaptive_avg_pool3d`` rather than ``interpolate``: the adaptor session
    established the hard way that trilinear interpolation *samples* rather than
    averages, so on ``(8,14,14) -> (2,4,4)`` whole frames would contribute
    nothing at all. That bug cost real time once already.
    """

    def __init__(self, in_dim: int, out_dim: int, pool_grid: Tuple[int, int, int],
                 hidden: int = 2048, depth: int = 2) -> None:
        super().__init__()
        self.pool_grid = tuple(int(g) for g in pool_grid)
        self.n_tokens = int(math.prod(self.pool_grid))
        layers = [nn.LayerNorm(in_dim), nn.Linear(in_dim, hidden), nn.GELU()]
        for _ in range(max(depth - 1, 0)):
            layers += [nn.Linear(hidden, hidden), nn.GELU()]
        layers += [nn.Linear(hidden, out_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pooled = F.adaptive_avg_pool3d(x, self.pool_grid)   # (B, D, f, h, w)
        tokens = flatten_grid(pooled)                       # (B, K, D)
        return self.net(tokens)


class PerceiverResampler(nn.Module):
    """K learned queries cross-attend into all N patches, a few times over.

    The queries are ``nn.Parameter`` -- they do not come from the input at all,
    which is what fixes the output length at K regardless of N. Cost per layer is
    ``O(K*N)`` rather than the ``O(N^2)`` of self-attention over the patches,
    about 24x cheaper at K=64, N=1568.

    Following Flamingo, the keys and values are ``[patches ; latents]`` rather
    than the patches alone, so the K queries can also see each other and divide
    the labour instead of all converging on the same salient region.
    """

    def __init__(self, in_dim: int, out_dim: int, n_tokens: int = 64, width: int = 1024,
                 depth: int = 4, heads: int = 8, grid: Sequence[int] = (8, 14, 14)) -> None:
        super().__init__()
        if width % heads:
            raise ValueError(f"--proj-width {width} not divisible by --proj-heads {heads}")
        self.n_tokens = n_tokens
        self.latents = nn.Parameter(torch.randn(n_tokens, width) * 0.02)
        self.input_proj = nn.Linear(in_dim, width)
        self.input_norm = nn.LayerNorm(width)
        self.pos = Factorised3DPosition(width, grid)

        self.layers = nn.ModuleList()
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                nn.LayerNorm(width),
                nn.LayerNorm(width),
                nn.MultiheadAttention(width, heads, batch_first=True),
                FeedForward(width),
            ]))
        self.out_norm = nn.LayerNorm(width)
        self.out_proj = nn.Linear(width, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mem = self.input_norm(self.input_proj(flatten_grid(x)))     # (B, N, width)
        mem = mem + self.pos()[None]                                # geometry, see docstring
        z = self.latents[None].expand(mem.shape[0], -1, -1)         # (B, K, width)

        for norm_z, norm_m, attn, ff in self.layers:
            zq = norm_z(z)
            kv = torch.cat([norm_m(mem), zq], dim=1)                # Flamingo-style
            z = z + attn(zq, kv, kv, need_weights=False)[0]
            z = z + ff(z)

        return self.out_proj(self.out_norm(z))


def build_projector(
    kind: str,
    in_dim: int,
    out_dim: int,
    grid: Sequence[int],
    n_tokens: int = 64,
    pool_grid: Sequence[int] = (2, 4, 4),
    width: int = 1024,
    depth: int = 4,
    heads: int = 8,
    hidden: int = 2048,
) -> nn.Module:
    """Build one rung of the ladder.

    ``n_tokens`` is ignored by ``pool``, whose token count is fixed by
    ``pool_grid`` -- the two are kept separate so that ``--proj pool --pool-grid
    2,4,4`` and ``--proj perceiver --n-tokens 32`` can be compared at a matched
    budget without one flag silently overriding the other.
    """
    if kind == "mean":
        return MeanProjector(in_dim, out_dim, n_tokens, hidden=hidden, depth=2)
    if kind == "pool":
        return PoolProjector(in_dim, out_dim, tuple(pool_grid), hidden=hidden, depth=2)
    if kind == "perceiver":
        return PerceiverResampler(
            in_dim, out_dim, n_tokens=n_tokens, width=width,
            depth=depth, heads=heads, grid=grid,
        )
    raise ValueError(f"unknown projector kind: {kind!r} (expected one of {KINDS})")


def token_budget(kind: str, n_tokens: int, pool_grid: Sequence[int]) -> int:
    """How many soft tokens a configuration will actually produce."""
    if kind == "pool":
        return int(math.prod(int(g) for g in pool_grid))
    return int(n_tokens)
