import pytest
import torch
from continuous_control.networks import Actor, Critic


@pytest.mark.parametrize('batch', [1, 20, 64, 128])
def test_interfaces_and_bounds(batch):
    torch.manual_seed(7)
    states = torch.randn(batch, 33)
    actor, critic = Actor(), Critic()
    actions = actor(states)
    assert actions.shape == (batch, 4)
    assert torch.isfinite(actions).all() and (actions.abs() <= 1).all()
    assert critic(states, actions).shape == (batch, 1)
    assert torch.equal(actions, actor(states))
    assert not ({id(p) for p in actor.parameters()} &
                {id(p) for p in critic.parameters()})


def test_critic_can_express_values_outside_action_range():
    critic = Critic()
    with torch.no_grad():
        critic.output.weight.zero_()
        critic.output.bias.fill_(12)
    assert torch.equal(critic(torch.zeros(2, 33), torch.zeros(2, 4)),
                       torch.full((2, 1), 12.))


def test_frozen_critic_preserves_actor_gradient_path():
    torch.manual_seed(7)
    actor, critic = Actor(), Critic()
    critic.requires_grad_(False)
    states = torch.randn(8, 33)
    actions = actor(states)
    actions.retain_grad()
    before = [p.detach().clone() for p in actor.parameters()]
    (-critic(states, actions).mean()).backward()
    assert actions.grad.abs().sum() > 0
    assert sum(p.grad.abs().sum() for p in actor.parameters()) > 0
    assert all(p.grad is None for p in critic.parameters())
    assert all(torch.equal(a, p) for a, p in zip(before, actor.parameters()))


def test_rejects_misaligned_samples():
    with pytest.raises(ValueError):
        Critic()(torch.zeros(8, 33), torch.zeros(1, 4))
    with pytest.raises(ValueError):
        Actor()(torch.zeros(33))


def test_seed_reproduces_initial_parameters():
    torch.manual_seed(11)
    first = Actor()
    torch.manual_seed(11)
    second = Actor()
    assert all(torch.equal(a, b) for a, b in zip(first.parameters(), second.parameters()))
