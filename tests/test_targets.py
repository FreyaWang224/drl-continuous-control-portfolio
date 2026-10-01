import pytest
import torch
from torch import nn

from continuous_control.networks import Actor, Critic
from continuous_control.targets import make_target, soft_update


@pytest.mark.parametrize('network', [Actor, Critic])
def test_target_starts_equal_but_independent_and_frozen(network):
    torch.manual_seed(9)
    online = network()
    target = make_target(online)
    assert not target.training
    for online_param, target_param in zip(online.parameters(), target.parameters()):
        assert online_param is not target_param
        assert torch.equal(online_param, target_param)
        assert not target_param.requires_grad
    with torch.no_grad():
        next(online.parameters()).add_(1)
    assert not torch.equal(next(online.parameters()), next(target.parameters()))


def test_soft_update_numerical_direction_and_no_optimizer_step():
    online = nn.Linear(1, 1)
    with torch.no_grad():
        online.weight.fill_(2)
        online.bias.fill_(2)
    target = make_target(online)
    with torch.no_grad():
        online.weight.fill_(10)
        online.bias.fill_(10)
    soft_update(target, online, 0.1)
    assert target.weight.item() == pytest.approx(2.8)
    assert target.bias.item() == pytest.approx(2.8)
    assert online.weight.item() == 10
    assert target.weight.grad is None


def test_invalid_update_rejected_without_mutation():
    online = Actor()
    target = make_target(online)
    before = [p.clone() for p in target.parameters()]
    for tau in (0, -0.1, 1.1, float('nan')):
        with pytest.raises(ValueError):
            soft_update(target, online, tau)
    with pytest.raises(ValueError):
        soft_update(target, Critic(), 0.1)
    assert all(torch.equal(p, old) for p, old in zip(target.parameters(), before))
