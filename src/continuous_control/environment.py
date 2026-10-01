"""Multi-agent bridge to the original Udacity unityagents API.

Legacy local_done does not distinguish termination from time-limit truncation.
Keep that signal raw rather than inventing a Bellman bootstrap mask.
"""
from dataclasses import dataclass
from pathlib import Path
import numpy as np


@dataclass(frozen=True)
class Step:
    observations: np.ndarray
    rewards: np.ndarray
    legacy_done: np.ndarray
    agent_ids: tuple


class ReacherEnvironment:
    def __init__(self, path, *, seed=0, worker_id=0, expected_agents=20,
                 no_graphics=True, factory=None):
        self.expected_agents = expected_agents
        self._closed = False
        self._ids = None
        self._needs_reset = True
        if factory is None:
            if not Path(path).exists():
                raise FileNotFoundError(path)
            # ML-Agents 0.4 uses the alias removed by NumPy 2.
            if not hasattr(np, 'float_'):
                np.float_ = np.float64
            from unityagents import UnityEnvironment
            factory = UnityEnvironment
        self._env = factory(file_name=str(path), seed=seed, worker_id=worker_id,
                            no_graphics=no_graphics)
        try:
            if len(self._env.brain_names) != 1:
                raise ValueError('Expected exactly one Unity brain')
            self.brain_name = self._env.brain_names[0]
            brain = self._env.brains[self.brain_name]
            if (int(brain.vector_observation_space_size) != 33 or
                    int(brain.vector_action_space_size) != 4 or
                    brain.vector_action_space_type != 'continuous'):
                raise ValueError('Expected 33 observations and 4 continuous actions')
        except Exception:
            self.close()
            raise

    def _check_open(self):
        if self._closed:
            raise RuntimeError('Environment is closed')

    def _decode(self, info, *, reset=False):
        ids = tuple(info.agents)
        n = len(ids)
        if n != self.expected_agents or len(set(ids)) != n:
            raise ValueError('Unexpected agent count or duplicate IDs')
        if not reset and ids != self._ids:
            raise ValueError('Agent order changed; transition alignment is unsafe')
        states = np.asarray(info.vector_observations, dtype=np.float32)
        rewards = np.asarray(info.rewards, dtype=np.float32)
        done = np.asarray(info.local_done, dtype=bool)
        if states.shape != (n, 33) or rewards.shape != (n,) or done.shape != (n,):
            raise ValueError('Invalid observation/reward/done shape')
        if not np.isfinite(states).all() or not np.isfinite(rewards).all():
            raise ValueError('Non-finite environment output')
        self._ids = ids
        self._needs_reset = bool(done.any())
        return Step(states.copy(), rewards.copy(), done.copy(), ids)

    def reset(self, *, training=True):
        self._check_open()
        return self._decode(self._env.reset(train_mode=training)[self.brain_name], reset=True)

    def step(self, actions):
        self._check_open()
        if self._needs_reset:
            raise RuntimeError('Reset required before step')
        actions = np.asarray(actions, dtype=np.float32)
        if actions.shape != (self.expected_agents, 4):
            raise ValueError('Actions must have shape [N,4]')
        if not np.isfinite(actions).all() or (np.abs(actions) > 1).any():
            raise ValueError('Actions must be finite and within [-1,1]')
        # Reject invalid actions instead of silently changing Replay semantics.
        return self._decode(self._env.step(actions.copy())[self.brain_name])

    def close(self):
        if not self._closed:
            self._closed = True
            self._env.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
