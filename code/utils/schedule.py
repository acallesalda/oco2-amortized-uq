"""schedule.py
One learning-rate schedule, shared by the five trainers, and the two knobs that go with it.

Why this exists
---------------
The six published learning curves show two separate problems, both of
them in the *schedule* rather than in the searched hyperparameters:

* **The three deterministic regressors had not converged at epoch 40.** Two of them take
  their checkpoint at epoch 38 and 39 of 40 -- the budget ran out before the validation
  curve did -- and all three are visibly noisy from one epoch to the next (mean absolute
  epoch-to-epoch change in validation RMSE: 0.032 standardized for `det_xco2` over the
  first third, 0.017 for `det_profiles` over the last). They use cosine annealing with
  `eta_min = lr/10`, so the floor of the schedule is a tenth of the peak and the model is
  still taking large steps at the end.
* **Two of the three flows had no schedule at all.** `train_nf.py` and
  `train_nf_profiles.py` run AdamW at a constant learning rate for forty epochs. The scalar
  flow's validation NLL wanders by 0.24 nats from epoch to epoch over the last third and
  ends at 1.162 -- *above* its epoch-1 value of 1.033 -- while its best, 0.698, is at epoch
  32. That is not over-fitting, it is a schedule that never anneals.

The fix is the same in both places and is deliberately not a re-tuning: the peak learning
rate stays whatever the search chose, because that is the number the search ranked
configurations by. What changes is the shape around it -- a short warm-up so the first
epochs do not spike, and a floor low enough that the last epochs actually settle.

    lr(t) = lr_peak * t / W                                      t < W   (linear warm-up)
    lr(t) = eta_min + (lr_peak - eta_min) * (1 + cos(pi * u)) / 2  else  (cosine)
        with u = (t - W) / (T - W)  and  eta_min = eta_min_frac * lr_peak

Defaults reproduce the published behaviour exactly: `warmup_epochs=0` and
`eta_min_frac=0.1` give `CosineAnnealingLR(T_max=T, eta_min=lr/10)` to within float error,
and `grad_clip=None` is no clipping. A trainer that does not pass anything trains the model
it trained before -- which is what keeps the published checkpoints reproducible while the
new runs get a schedule that suits the longer budget.

Stepping is PER EPOCH, matching what `train.py` and `train_nf_profiles_4d.py` already do,
so `T` is the epoch count and `warmup_epochs` is a number of epochs.
"""

from __future__ import annotations

import math
from typing import Optional

import torch


def cosine_with_warmup(optimizer: "torch.optim.Optimizer", epochs: int,
                       warmup_epochs: int = 0, eta_min_frac: float = 0.1
                       ) -> "torch.optim.lr_scheduler.LambdaLR":
    """Cosine decay to `eta_min_frac * lr`, after `warmup_epochs` of linear warm-up.

    Built on LambdaLR, which multiplies each param group's own `initial_lr`, so a trainer
    with several groups at different learning rates keeps their ratio.
    """
    epochs = max(int(epochs), 1)
    warmup = max(int(warmup_epochs), 0)
    floor = float(eta_min_frac)
    if not 0.0 <= floor < 1.0:
        raise ValueError(f"eta_min_frac must be in [0, 1), got {floor}")
    if warmup >= epochs:
        raise ValueError(f"warmup_epochs={warmup} must be less than epochs={epochs}")

    def factor(epoch: int) -> float:
        if warmup and epoch < warmup:
            # epoch 0 would be a learning rate of exactly zero and a wasted epoch, so the
            # ramp starts one step in: 1/W, 2/W, ... , W/W.
            return (epoch + 1) / float(warmup)
        u = (epoch - warmup) / float(max(epochs - warmup, 1))
        return floor + (1.0 - floor) * 0.5 * (1.0 + math.cos(math.pi * min(u, 1.0)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def clip_gradients(parameters, grad_clip: Optional[float]) -> Optional[float]:
    """`clip_grad_norm_` when `grad_clip` is set, and nothing at all when it is not.

    Returns the pre-clipping total norm, which is worth logging: a run whose gradient norm
    sits an order of magnitude above the clip for most of training is being clipped rather
    than stabilized, and that is a different situation from the occasional spike this is
    meant for.
    """
    if grad_clip is None:
        return None
    return float(torch.nn.utils.clip_grad_norm_(parameters, float(grad_clip)))


def describe(epochs: int, lr: float, warmup_epochs: int = 0,
             eta_min_frac: float = 0.1, grad_clip: Optional[float] = None) -> str:
    """One line for the log, so the schedule is recoverable from a .out file."""
    return (f"schedule: cosine over {epochs} epochs | peak lr {lr:.3e}"
            f" -> floor {eta_min_frac * lr:.3e}"
            + (f" | warmup {warmup_epochs} ep" if warmup_epochs else " | no warmup")
            + (f" | grad clip {grad_clip}" if grad_clip is not None else " | no grad clip"))
