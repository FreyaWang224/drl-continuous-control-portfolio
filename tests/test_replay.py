import numpy as np
import pytest

from continuous_control.replay import ReplayBuffer


def transitions(n):
    ids = np.arange(n, dtype=np.float32)
    states = np.repeat(ids[:, None], 33, axis=1)
    actions = np.repeat((ids / 100)[:, None], 4, axis=1)
    rewards = ids.copy()
    next_states = states + 1
    done = (ids.astype(int) % 2 == 0)
    return states, actions, rewards, next_states, done


def test_stores_all_arms_with_alignment_and_shapes():
    buffer = ReplayBuffer(capacity=200, seed=3)
    for offset in range(0, 100, 20):
        batch = transitions(100)
        buffer.add(*(x[offset:offset + 20] for x in batch))
    sample = buffer.sample(64)
    assert len(buffer) == 100
    assert (sample.states.shape, sample.actions.shape, sample.rewards.shape,
            sample.next_states.shape, sample.legacy_done.shape) == (
                (64, 33), (64, 4), (64, 1), (64, 33), (64, 1))
    ids = sample.states[:, 0]
    assert len(np.unique(ids)) == 64
    np.testing.assert_array_equal(sample.rewards[:, 0], ids)
    np.testing.assert_array_equal(sample.next_states[:, 0], ids + 1)
    np.testing.assert_array_equal(sample.legacy_done[:, 0], ids.astype(int) % 2 == 0)


def test_circular_overwrite_retains_latest_entries():
    buffer = ReplayBuffer(capacity=3)
    buffer.add(*transitions(8))
    assert len(buffer) == 3
    np.testing.assert_array_equal(np.sort(buffer.sample(3).states[:, 0]), [5, 6, 7])


def test_add_and_sample_do_not_share_mutable_storage():
    inputs = list(transitions(2))
    buffer = ReplayBuffer(capacity=2)
    buffer.add(*inputs)
    inputs[0][:] = -999
    sampled = buffer.sample(2)
    sampled.states[:] = -888
    np.testing.assert_array_equal(np.sort(buffer.sample(2).states[:, 0]), [0, 1])


def test_seeded_sampling_is_reproducible():
    a, b = ReplayBuffer(100, seed=12), ReplayBuffer(100, seed=12)
    for buffer in (a, b):
        buffer.add(*transitions(100))
    np.testing.assert_array_equal(a.sample(16).states, b.sample(16).states)


def test_preserves_executed_clipped_action():
    batch = list(transitions(1))
    batch[1][0, 0] = 1.0
    buffer = ReplayBuffer(1)
    buffer.add(*batch)
    assert buffer.sample(1).actions[0, 0] == 1.0


def test_rejects_unavailable_or_invalid_samples():
    buffer = ReplayBuffer(2)
    with pytest.raises(ValueError):
        buffer.sample(1)
    bad = list(transitions(1))
    bad[1][0, 0] = 1.2
    with pytest.raises(ValueError):
        buffer.add(*bad)
    assert len(buffer) == 0
