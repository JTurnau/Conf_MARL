import torch
import torch.nn as nn
import torch.nn.functional as F


class ConfidenceNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super(ConfidenceNetwork, self).__init__()
        # Define the layers
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)  # Input layer
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)             # Hidden layer
        self.fc3 = nn.Linear(hidden_dim, 1)                     # Output layer

    def forward(self, state, action):
        # Concatenate state and action along the last dimension
        x = torch.cat([state, action], dim=-1)
        x = F.relu(self.fc1(x))  # Apply ReLU to the first layer
        x = F.relu(self.fc2(x))  # Apply ReLU to the second layer
        confidence_score = torch.sigmoid(self.fc3(x))  # Output confidence score (0 to 1)
        return confidence_score
