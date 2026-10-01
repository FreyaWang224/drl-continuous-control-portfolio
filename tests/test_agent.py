import numpy as np
import pytest
import torch
from torch import nn

from continuous_control.agent import DDPGAgent, DDPGConfig, bellman_targets
from continuous_control.replay import ReplayBatch


class FixedActor(nn.Module):
    def forward(self, states):
        return torch.zeros((len(states), 4))


class FixedCritic(nn.Module):
    def forward(self, states, actions):
        return torch.full((len(states), 1), 10.)


def batch(n=8):
    rng = np.random.default_rng(5)
    return ReplayBatch(rng.normal(size=(n, 33)).astype('float32'),
                       rng.uniform(-1, 1, size=(n, 4)).astype('float32'),
                       rng.uniform(size=(n, 1)).astype('float32'),
                       rng.normal(size=(n, 33)).astype('float32'),
                       np.zeros((n, 1), dtype=bool))


def test_bellman_target_uses_true_termination_mask_and_keeps_shape():
    rewards = torch.full((2, 1), 0.1, requires_grad=True)
    terminal = torch.tensor([[0.], [1.]])
    values = bellman_targets(rewards, torch.zeros(2, 33), terminal, 0.99,
                              FixedActor(), FixedCritic())
    assert values.shape == (2, 1)
    torch.testing.assert_close(values, torch.tensor([[10.], [0.1]]))
    assert not values.requires_grad


def test_learning_updates_online_networks_and_target_tracks_them():
    agent = DDPGAgent(DDPGConfig(seed=1, tau=0.2))
    examples = batch()
    actor_before = [p.clone() for p in agent.actor.parameters()]
    critic_before = [p.clone() for p in agent.critic.parameters()]
    target_before = [p.clone() for p in agent.target_actor.parameters()]
    stats = agent.learn(examples, np.zeros((8, 1), dtype=np.float32))
    assert agent.updates == 1
    assert np.isfinite(list(stats.values())).all()
    assert any(not torch.equal(a, b) for a, b in zip(actor_before, agent.actor.parameters()))
    assert any(not torch.equal(a, b) for a, b in zip(critic_before, agent.critic.parameters()))
    for old, online, target in zip(target_before, agent.actor.parameters(), agent.target_actor.parameters()):
        torch.testing.assert_close(target, 0.8 * old + 0.2 * online)
        assert not target.requires_grad
    assert all(p.requires_grad for p in agent.critic.parameters())


def test_action_modes_and_seed_reproducibility():
    a, b = DDPGAgent(DDPGConfig(seed=19)), DDPGAgent(DDPGConfig(seed=19))
    states = np.zeros((20, 33), dtype=np.float32)
    np.testing.assert_array_equal(a.act(states, explore=False), a.act(states, explore=False))
    np.testing.assert_array_equal(a.act(states, explore=True), b.act(states, explore=True))
    assert (np.abs(a.act(states)) <= 1).all()


def test_checkpoint_round_trip_for_inference():
    agent = DDPGAgent(DDPGConfig(seed=2))
    agent.learn(batch(), np.zeros((8, 1), dtype=np.float32))
    states = np.ones((3, 33), dtype=np.float32)
    expected = agent.act(states, explore=False)
    restored = DDPGAgent(DDPGConfig(seed=77))
    restored.restore(agent.checkpoint())
    np.testing.assert_array_equal(expected, restored.act(states, explore=False))
    assert restored.updates == 1


def test_invalid_terminal_mask_fails_before_update():
    agent = DDPGAgent(DDPGConfig())
    with pytest.raises(ValueError):
        agent.learn(batch(), np.zeros((8,), dtype=np.float32))
    assert agent.updates == 0
