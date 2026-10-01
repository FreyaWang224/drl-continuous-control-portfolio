"""Seeded, fixed-capacity replay of actual environment transitions."""
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ReplayBatch:
    states: np.ndarray       # [B,33], float32
    actions: np.ndarray      # [B,4], float32; action actually sent to Unity
    rewards: np.ndarray      # [B,1], float32
    next_states: np.ndarray  # [B,33], float32
    legacy_done: np.ndarray  # [B,1], bool; semantics not inferred here


class ReplayBuffer:
    """Store one aligned transition per arm and sample without replacement."""

    def __init__(self, capacity=1_000_000, seed=0):
        if not isinstance(capacity, int) or capacity < 1:
            raise ValueError('capacity must be a positive integer')
        self.capacity = capacity
        self._rng = np.random.default_rng(seed)
        self._size = 0
        self._next = 0
        # Allocate on first insertion so construction alone is inexpensive.
        self._states = None

    def __len__(self):
        return self._size

    def _allocate(self):
        self._states = np.empty((self.capacity, 33), dtype=np.float32)
        self._actions = np.empty((self.capacity, 4), dtype=np.float32)
        self._rewards = np.empty((self.capacity, 1), dtype=np.float32)
        self._next_states = np.empty((self.capacity, 33), dtype=np.float32)
        self._legacy_done = np.empty((self.capacity, 1), dtype=bool)

    def add(self, states, actions, rewards, next_states, legacy_done):
        """Append N matched transitions; the caller supplies executed actions."""
        states = np.asarray(states, dtype=np.float32)
        actions = np.asarray(actions, dtype=np.float32)
        rewards = np.asarray(rewards, dtype=np.float32)
        next_states = np.asarray(next_states, dtype=np.float32)
        legacy_done = np.asarray(legacy_done)
        if states.ndim != 2 or states.shape[1] != 33 or states.shape[0] == 0:
            raise ValueError('states must have shape [N,33], N >= 1')
        n = states.shape[0]
        if (actions.shape != (n, 4) or rewards.shape not in ((n,), (n, 1)) or
                next_states.shape != (n, 33) or
                legacy_done.shape not in ((n,), (n, 1)) or
                legacy_done.dtype != np.dtype('bool')):
            raise ValueError('unaligned or invalid transition shapes/types')
        if (not np.isfinite(states).all() or not np.isfinite(actions).all() or
                not np.isfinite(rewards).all() or
                not np.isfinite(next_states).all() or
                (np.abs(actions) > 1).any()):
            raise ValueError('transitions must be finite; actions within [-1,1]')
        if self._states is None:
            self._allocate()
        rewards = rewards.reshape(n, 1)
        legacy_done = legacy_done.reshape(n, 1)
        # Sequential writes preserve the latest capacity transitions even if N
        # exceeds capacity or crosses the circular array boundary.
        for i in range(n):
            j = self._next
            self._states[j] = states[i]
            self._actions[j] = actions[i]
            self._rewards[j] = rewards[i]
            self._next_states[j] = next_states[i]
            self._legacy_done[j] = legacy_done[i]
            self._next = (j + 1) % self.capacity
            self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size):
        if not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError('batch_size must be a positive integer')
        if batch_size > self._size:
            raise ValueError('not enough transitions to sample without replacement')
        indices = self._rng.choice(self._size, size=batch_size, replace=False)
        return ReplayBatch(self._states[indices].copy(),
                           self._actions[indices].copy(),
                           self._rewards[indices].copy(),
                           self._next_states[indices].copy(),
                           self._legacy_done[indices].copy())
