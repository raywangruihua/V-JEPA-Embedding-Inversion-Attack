"""Rectified flow: the training objective and the sampler.

Flow matching rather than DDPM, for three reasons that all matter on a project
this size:

* the objective is one line -- interpolate between noise and data, regress the
  straight-line velocity -- with no beta schedule, no variance parameterisation
  and no ``v``/``epsilon``/``x0`` prediction choice to get subtly wrong;
* the learned paths are close to straight, so 30-50 Euler steps produce what a
  DDPM needs several hundred for. That directly buys you the best-of-N sampling
  budget the attack wants;
* it is what LTX-2 itself is trained with, so the target latents are already
  living in the regime this objective assumes.

The convention throughout: ``t = 0`` is pure noise, ``t = 1`` is data, and

    x_t = (1 - t) * x0 + t * x1        with x0 ~ N(0, I), x1 = data

whose time derivative is a constant ``x1 - x0``. That constant is what the model
regresses, and sampling is just integrating it forward from noise.

Note the sign convention is not universal -- several papers run time the other
way. Everything here is internally consistent; the thing to preserve if you edit
it is that :func:`sample` integrates in the same direction the loss trains.
"""

from __future__ import annotations

from typing import Optional, Sequence

import torch
import torch.nn.functional as F


def sample_timesteps(n: int, device: torch.device, mu: float = 0.0, sigma: float = 1.0) -> torch.Tensor:
    """Logit-normal timesteps on ``(0, 1)``.

    Uniform ``t`` wastes capacity: near ``t=0`` the target is almost pure noise
    and near ``t=1`` it is almost the data, and both are easy. The interesting,
    high-loss region is the middle, and a logit-normal puts most of the samples
    there. This is the SD3 sampling choice and it is worth more than it looks --
    on small datasets it noticeably speeds up the early part of training.
    """
    return torch.sigmoid(torch.randn(n, device=device) * sigma + mu)


def flow_loss(
    model,
    x1: torch.Tensor,
    cond: torch.Tensor,
    cfg_dropout: float = 0.0,
    mu: float = 0.0,
    sigma: float = 1.0,
) -> torch.Tensor:
    """Rectified-flow regression loss for one batch.

    ``x1`` is the standardized LTX latent, ``cond`` the standardized V-JEPA grid.
    """
    b = x1.shape[0]
    device = x1.device

    t = sample_timesteps(b, device, mu, sigma)
    t_b = t.view(-1, *([1] * (x1.dim() - 1)))

    x0 = torch.randn_like(x1)
    x_t = (1.0 - t_b) * x0 + t_b * x1
    target = x1 - x0

    drop = None
    if cfg_dropout > 0:
        drop = torch.rand(b, device=device) < cfg_dropout

    pred = model(x_t, t, cond, drop)
    return F.mse_loss(pred.float(), target.float())


@torch.no_grad()
def sample(
    model,
    cond: torch.Tensor,
    latent_shape: Sequence[int],
    steps: int = 50,
    cfg: float = 1.0,
    generator: Optional[torch.Generator] = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Integrate noise -> latent. Returns ``(B, C, F, H, W)`` in standardized space.

    ``cfg`` is the classifier-free guidance scale: 1.0 disables it (and halves
    the cost, since no unconditional pass is needed). Above 1.0 the update is
    extrapolated away from the unconditional prediction, which sharpens output
    but pushes toward the *mode* of ``p(latent | embedding)``.

    That distinction matters more here than in ordinary generation. For an
    inversion attack the mode is not the goal -- a confidently-rendered
    prototypical SSv2 clip scores well on realism and tells you nothing about the
    intercepted video. Sweep this against the re-encoding metric rather than by
    eye; the visually nicest setting is usually not the most faithful one.
    """
    model.eval()
    device = cond.device
    b = cond.shape[0]

    x = torch.randn(b, *latent_shape, device=device, dtype=dtype, generator=generator)
    ts = torch.linspace(0.0, 1.0, steps + 1, device=device, dtype=dtype)

    guided = cfg != 1.0
    if guided:
        # One doubled batch instead of two forward passes: the condition encoder
        # runs twice either way, but this keeps the GPU busy with one launch.
        cond_pair = torch.cat([cond, cond], dim=0)
        drop_pair = torch.cat(
            [torch.zeros(b, dtype=torch.bool, device=device),
             torch.ones(b, dtype=torch.bool, device=device)]
        )

    for i in range(steps):
        t_now, dt = ts[i], ts[i + 1] - ts[i]
        if guided:
            v_both = model(
                torch.cat([x, x], dim=0),
                t_now.expand(2 * b),
                cond_pair,
                drop_pair,
            )
            v_cond, v_uncond = v_both[:b], v_both[b:]
            v = v_uncond + cfg * (v_cond - v_uncond)
        else:
            v = model(x, t_now.expand(b), cond, None)
        x = x + dt * v.to(x.dtype)

    return x


@torch.no_grad()
def retrieval_accuracy(pred: torch.Tensor, target: torch.Tensor, ks=(1, 5)) -> dict:
    """Can each sampled latent find its own true latent among the others?

    Mirrors the function of the same name in ``standard_adaptors/adaptor.py`` -- kept here
    rather than imported so this directory has no import-order dependency on a
    sibling module that shares its name with a package directory.

    This remains the metric to trust, with one caveat specific to diffusion:
    a sample is *not* trying to equal the target, so its cosine similarity will
    read lower than a regressor's even when it is far more informative. Retrieval
    is robust to that, because it only asks whether the sample is closer to its
    own target than to anyone else's.
    """
    p = F.normalize(pred.flatten(1).float(), dim=1)
    t = F.normalize(target.flatten(1).float(), dim=1)
    sim = p @ t.T
    n = sim.shape[0]
    truth = torch.arange(n, device=sim.device)
    ranks = (sim > sim[truth, truth].unsqueeze(1)).sum(dim=1)
    out = {f"top{k}": float((ranks < k).float().mean()) for k in ks}
    out["median_rank"] = float(ranks.float().median()) + 1.0
    out["chance_top1"] = 1.0 / n
    return out


class EMA:
    """Exponential moving average of model weights.

    Not optional for diffusion. The raw weights bounce around enough that samples
    drawn from them look meaningfully worse than samples from the average, and
    the gap is largest exactly when the dataset is small. Costs one extra copy of
    the parameters in memory.
    """

    def __init__(self, model, decay: float = 0.999) -> None:
        self.decay = decay
        self.shadow = {
            k: v.detach().clone().float() for k, v in model.state_dict().items()
            if v.dtype.is_floating_point
        }

    @torch.no_grad()
    def update(self, model) -> None:
        for k, v in model.state_dict().items():
            if k in self.shadow:
                self.shadow[k].mul_(self.decay).add_(v.detach().float(), alpha=1.0 - self.decay)

    def state_dict(self) -> dict:
        return self.shadow

    def copy_to(self, model) -> dict:
        """Swap EMA weights into ``model``, returning the originals for restoring."""
        backup = {k: v.detach().clone() for k, v in model.state_dict().items() if k in self.shadow}
        model.load_state_dict(
            {k: v.to(dtype=model.state_dict()[k].dtype) for k, v in self.shadow.items()},
            strict=False,
        )
        return backup

    def restore(self, model, backup: dict) -> None:
        model.load_state_dict(backup, strict=False)
