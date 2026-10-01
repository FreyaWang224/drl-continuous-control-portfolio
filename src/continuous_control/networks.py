"""Independent deterministic Actor and action-value Critic for Reacher.

Seed torch before construction. Exploration and optimizer updates belong to the
agent, not these forward passes. Hidden layers use PyTorch's default init.
"""
import torch
from torch import nn


class Actor(nn.Module):
    """Map states [B,33] to bounded actions [B,4]."""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(33, 256)
        self.fc2 = nn.Linear(256, 256)
        self.output = nn.Linear(256, 4)
        # Small initial actions; exploration is added separately by the agent.
        nn.init.uniform_(self.output.weight, -3e-3, 3e-3)
        nn.init.zeros_(self.output.bias)

    def forward(self, states):
        if states.ndim != 2 or states.shape[1] != 33:
            raise ValueError('Actor states must have shape [B,33]')
        hidden = torch.relu(self.fc1(states))
        hidden = torch.relu(self.fc2(hidden))
        return torch.tanh(self.output(hidden))


class Critic(nn.Module):
    """Map aligned states [B,33] and actions [B,4] to Q values [B,1]."""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(37, 256)
        self.fc2 = nn.Linear(256, 256)
        self.output = nn.Linear(256, 1)
        nn.init.uniform_(self.output.weight, -3e-3, 3e-3)
        nn.init.zeros_(self.output.bias)

    def forward(self, states, actions):
        if states.ndim != 2 or states.shape[1] != 33:
            raise ValueError('Critic states must have shape [B,33]')
        if actions.ndim != 2 or actions.shape != (states.shape[0], 4):
            raise ValueError('Critic actions must have matching shape [B,4]')
        inputs = torch.cat([states, actions], dim=-1)
        hidden = torch.relu(self.fc1(inputs))
        hidden = torch.relu(self.fc2(hidden))
        # No output activation: Q is not restricted to the action range.
        return self.output(hidden)
