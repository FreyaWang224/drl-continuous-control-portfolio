import numpy as np
import pytest

from continuous_control.agent import DDPGAgent, DDPGConfig
from continuous_control.cli import episode, load_checkpoint, save_checkpoint
from continuous_control.environment import Step


class FakeEnvironment:
    expected_agents = 20

    def __init__(self, done_at=None):
        self.done_at = done_at
        self.steps = 0

    def reset(self, training=True):
        self.steps = 0
        return Step(np.zeros((20, 33), dtype=np.float32), np.zeros(20, dtype=np.float32),
                    np.zeros(20, dtype=bool), tuple(range(20)))

    def step(self, actions):
        assert actions.shape == (20, 4)
        self.steps += 1
        rewards = np.arange(20, dtype=np.float32) / 10
        done = np.full(20, self.steps == self.done_at)
        return Step(np.zeros((20, 33), dtype=np.float32), rewards, done, tuple(range(20)))


def test_partial_run_is_not_marked_complete_or_solved():
    result = episode(FakeEnvironment(), DDPGAgent(DDPGConfig()),
                     training=False, max_steps=3)
    assert not result['complete_episode']
    assert result['steps'] == 3
    assert result['mean_score'] == pytest.approx(2.85)
    assert len(result['arm_scores']) == 20
    assert result['updates'] == 0


def test_early_done_is_not_silently_treated_as_time_limit():
    with pytest.raises(RuntimeError, match='Unexpected'):
        episode(FakeEnvironment(done_at=2), DDPGAgent(DDPGConfig()),
                training=False, max_steps=3)


def test_checkpoint_file_loads_identical_deterministic_policy(tmp_path):
    agent = DDPGAgent(DDPGConfig(seed=8))
    path = tmp_path / 'agent.pt'
    save_checkpoint(agent, path)
    restored = load_checkpoint(path)
    states = np.ones((20, 33), dtype=np.float32)
    np.testing.assert_array_equal(agent.act(states, explore=False),
                                  restored.act(states, explore=False))
    assert not path.with_name(path.name + '.tmp').exists()
