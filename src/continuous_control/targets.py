"""Frozen target networks and Polyak parameter updates for DDPG."""
from copy import deepcopy
from math import isfinite

import torch
from torch import nn


def make_target(online: nn.Module) -> nn.Module:
    """Start from identical values without sharing parameters or gradients."""
    target = deepcopy(online)
    target.requires_grad_(False)
    target.eval()
    return target


@torch.no_grad()
def soft_update(target: nn.Module, online: nn.Module, tau: float) -> None:
    """target <- (1 - tau) * target + tau * online, for each parameter."""
    if not isfinite(tau) or not 0 < tau <= 1:
        raise ValueError('tau must be finite and in (0,1]')
    if target is online:
        raise ValueError('target and online networks must be distinct')
    if any(parameter.requires_grad for parameter in target.parameters()):
        raise ValueError('target network parameters must be frozen')
    target_params = dict(target.named_parameters())
    online_params = dict(online.named_parameters())
    if target_params.keys() != online_params.keys() or any(
            target_params[name].shape != parameter.shape
            for name, parameter in online_params.items()):
        raise ValueError('target and online parameter structures differ')
    # Actor and Critic currently contain only parameters. Reject buffers so a
    # future BatchNorm addition cannot silently leave target state stale.
    if list(target.named_buffers()) or list(online.named_buffers()):
        raise ValueError('soft_update requires buffer-free networks')
    for name, parameter in online_params.items():
        target_params[name].lerp_(parameter, tau)
