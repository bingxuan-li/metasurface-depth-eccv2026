"""Paper-style per-image metrics and the original supervised training loss."""

import torch
from torch.nn import functional as F


def gradient_loss(prediction, target):
    px = (prediction[..., :-1] - prediction[..., 1:]).abs()
    py = (prediction[..., :-1, :] - prediction[..., 1:, :]).abs()
    tx = (target[..., :-1] - target[..., 1:]).abs()
    ty = (target[..., :-1, :] - target[..., 1:, :]).abs()
    return (px - tx).abs().mean() + (py - ty).abs().mean()


def training_loss(prediction, target):
    # Training is intentionally NOT clipped; clipping is evaluation-only.
    if not torch.isfinite(target).all() or (target <= 0).any():
        raise ValueError("Training requires finite, positive dense metric depth")
    return F.l1_loss(prediction, target) + 0.5 * gradient_loss(prediction, target)


def metrics(prediction, target, foreground=False):
    valid = torch.isfinite(target) & (target > 1e-5)
    if foreground and valid.any():
        t = target[valid]
        valid &= target < 0.99 * t.max() + 0.01 * t.min()
    if not valid.any():
        raise ValueError("No valid evaluation pixels")
    p, t = prediction[valid], target[valid]
    if not torch.isfinite(p).all() or (p <= 0).any():
        raise ValueError("Evaluation prediction must be finite and positive")
    ratio = torch.maximum(p / t, t / p)
    return dict(
        mae=(p - t).abs().mean().item(),
        rmse=((p - t) ** 2).mean().sqrt().item(),
        abs_rel=((p - t).abs() / t).mean().item(),
        delta_0_5=(ratio < 1.25**0.5).float().mean().item(),
        delta_1=(ratio < 1.25).float().mean().item(),
        log10=(p.log10() - t.log10()).abs().mean().item(),
    )
