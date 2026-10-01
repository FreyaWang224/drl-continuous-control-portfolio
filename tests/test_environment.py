from types import SimpleNamespace as NS
import numpy as np
import pytest
from continuous_control.environment import ReacherEnvironment


class Fake:
    def __init__(self, **kwargs):
        self.brain_names = ['ReacherBrain']
        self.brains = {'ReacherBrain': NS(vector_observation_space_size=33,
            vector_action_space_size=4, vector_action_space_type='continuous')}
        self.closed = 0
        self.done = False
        self.ids = list(range(20))
        self.sent = None
    def data(self):
        return {'ReacherBrain': NS(agents=self.ids, vector_observations=np.arange(660).reshape(20,33),
                                  rewards=np.arange(20), local_done=[self.done]*20)}
    def reset(self, **kwargs):
        self.done = False
        return self.data()
    def step(self, actions):
        self.sent = actions
        return self.data()
    def close(self):
        self.closed += 1


@pytest.fixture
def pair():
    fake = Fake()
    env = ReacherEnvironment('unused', factory=lambda **kw: fake)
    yield env, fake
    env.close()


def test_preserves_all_agents_and_actions(pair):
    env, fake = pair
    first = env.reset()
    assert first.observations.shape == (20,33)
    assert first.observations[19,32] == 659
    actions = np.linspace(-1,1,80).reshape(20,4).astype(np.float32)
    result = env.step(actions)
    np.testing.assert_array_equal(fake.sent, actions)
    np.testing.assert_array_equal(result.rewards, np.arange(20))
    assert result.legacy_done.shape == (20,)


@pytest.mark.parametrize('actions', [np.zeros(4), np.zeros((1,4)), np.full((20,4),1.01),
                                     np.full((20,4),np.nan), np.full((20,4),np.inf)])
def test_rejects_invalid_actions_before_send(pair, actions):
    env, fake = pair
    env.reset()
    with pytest.raises(ValueError):
        env.step(actions)
    assert fake.sent is None


def test_done_requires_reset(pair):
    env, fake = pair
    with pytest.raises(RuntimeError): env.step(np.zeros((20,4)))
    env.reset()
    fake.done = True
    assert env.step(np.zeros((20,4))).legacy_done.all()
    with pytest.raises(RuntimeError): env.step(np.zeros((20,4)))
    env.reset()
    env.step(np.zeros((20,4)))


def test_agent_reordering_is_rejected(pair):
    env, fake = pair
    env.reset()
    fake.ids = fake.ids[::-1]
    with pytest.raises(ValueError, match='order'):
        env.step(np.zeros((20,4)))


def test_close_idempotent_and_exception_cleanup(pair):
    env, fake = pair
    with pytest.raises(RuntimeError):
        with env:
            raise RuntimeError('test')
    env.close()
    assert fake.closed == 1
    with pytest.raises(RuntimeError): env.reset()


def test_wrong_brain_contract_closes():
    fake = Fake()
    fake.brains['ReacherBrain'].vector_action_space_type = 'discrete'
    with pytest.raises(ValueError):
        ReacherEnvironment('unused', factory=lambda **kw: fake)
    assert fake.closed == 1
