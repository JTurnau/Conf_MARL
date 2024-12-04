import torch
import torch.nn as nn

class DQN(nn.Module):
    def __init__(self, obs_dim, num_actions):
        """
        Args:
            obs_dim (int): Observation size for a single agent.
            num_actions (int): Number of actions available to the agent.
        """
        super(DQN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_actions)
        )

    def forward(self, obs):
        """
        Forward pass for the DQN.

        Args:
            obs (torch.Tensor): Observations (batch_size, obs_dim).

        Returns:
            torch.Tensor: Q-values for each action (batch_size, num_actions).
        """
        return self.network(obs)
