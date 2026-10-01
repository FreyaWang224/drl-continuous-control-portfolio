"""DDPG learner. The collector supplies true termination flags explicitly."""
from dataclasses import asdict, dataclass

import numpy as np
import torch
from torch import nn

from .networks import Actor, Critic
from .replay import ReplayBatch
from .targets import make_target, soft_update


@dataclass(frozen=True)
class DDPGConfig:
    seed: int = 0
    gamma: float = 0.99
    tau: float = 0.001
    actor_lr: float = 1e-4
    critic_lr: float = 1e-3
    noise_std: float = 0.2
    device: str = 'cpu'

    def __post_init__(self):
        if not 0 <= self.gamma <= 1 or not 0 < self.tau <= 1:
            raise ValueError('gamma must be in [0,1], tau in (0,1]')
        if self.actor_lr <= 0 or self.critic_lr <= 0 or self.noise_std < 0:
            raise ValueError('learning rates must be positive and noise nonnegative')


def bellman_targets(rewards, next_states, terminated, gamma,
                    target_actor, target_critic):
    """No gradient into target networks; all tensors use [B,1] values."""
    if rewards.ndim != 2 or rewards.shape[1] != 1 or terminated.shape != rewards.shape:
        raise ValueError('rewards and terminated must both have shape [B,1]')
    if next_states.shape != (rewards.shape[0], 33):
        raise ValueError('next_states must have shape [B,33]')
    with torch.no_grad():
        next_actions = target_actor(next_states)
        next_q = target_critic(next_states, next_actions)
        return rewards + gamma * (1 - terminated) * next_q


class DDPGAgent:
    def __init__(self, config: DDPGConfig):
        self.config = config
        torch.manual_seed(config.seed)
        self.device = torch.device(config.device)
        self.actor = Actor().to(self.device)
        self.critic = Critic().to(self.device)
        self.target_actor = make_target(self.actor)
        self.target_critic = make_target(self.critic)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=config.actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=config.critic_lr)
        self.noise_rng = np.random.default_rng(config.seed + 1)
        self.updates = 0

    def act(self, states: np.ndarray, *, explore: bool = True) -> np.ndarray:
        states = np.asarray(states, dtype=np.float32)
        if states.ndim != 2 or states.shape[1] != 33:
            raise ValueError('states must have shape [N,33]')
        with torch.no_grad():
            actions = self.actor(torch.as_tensor(states, device=self.device)).cpu().numpy()
        if explore:
            actions = actions + self.noise_rng.normal(0, self.config.noise_std,
                                                       actions.shape).astype(np.float32)
        return np.clip(actions, -1, 1).astype(np.float32)

    def learn(self, batch: ReplayBatch, terminated: np.ndarray) -> dict:
        """Update Critic, Actor, then targets; terminated is not legacy_done."""
        def tensor(values):
            return torch.as_tensor(values, dtype=torch.float32, device=self.device)

        states, actions = tensor(batch.states), tensor(batch.actions)
        rewards, next_states = tensor(batch.rewards), tensor(batch.next_states)
        terminated = tensor(terminated)
        if terminated.shape != rewards.shape or not bool(((terminated == 0) | (terminated == 1)).all()):
            raise ValueError('terminated must be a [B,1] binary mask')
        targets = bellman_targets(rewards, next_states, terminated,
                                  self.config.gamma, self.target_actor, self.target_critic)
        predicted = self.critic(states, actions)
        critic_loss = nn.functional.mse_loss(predicted, targets)
        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_optimizer.step()

        self.critic.requires_grad_(False)
        try:
            actor_loss = -self.critic(states, self.actor(states)).mean()
            self.actor_optimizer.zero_grad(set_to_none=True)
            actor_loss.backward()
            self.actor_optimizer.step()
        finally:
            self.critic.requires_grad_(True)
        soft_update(self.target_actor, self.actor, self.config.tau)
        soft_update(self.target_critic, self.critic, self.config.tau)
        self.updates += 1
        return {'critic_loss': float(critic_loss.detach()),
                'actor_loss': float(actor_loss.detach()),
                'mean_target_q': float(targets.mean()),
                'mean_predicted_q': float(predicted.detach().mean())}

    def checkpoint(self):
        return {'config': asdict(self.config), 'updates': self.updates,
                'actor': self.actor.state_dict(), 'critic': self.critic.state_dict(),
                'target_actor': self.target_actor.state_dict(),
                'target_critic': self.target_critic.state_dict(),
                'actor_optimizer': self.actor_optimizer.state_dict(),
                'critic_optimizer': self.critic_optimizer.state_dict(),
                'noise_rng': self.noise_rng.bit_generator.state,
                'torch_rng': torch.get_rng_state()}

    def restore(self, checkpoint, *, for_training=False):
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.target_actor.load_state_dict(checkpoint['target_actor'])
        self.target_critic.load_state_dict(checkpoint['target_critic'])
        self.updates = checkpoint['updates']
        if for_training:
            self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
            self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
            self.noise_rng.bit_generator.state = checkpoint['noise_rng']
            torch.set_rng_state(checkpoint['torch_rng'])
