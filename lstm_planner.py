import torch
import torch.nn as nn
import torch.nn.functional as F


# Define the LSTM model
class LSTMPlanner(nn.Module):
    def __init__(self, state_size, action_size, hidden_size, plan_length):
        super(LSTMPlanner, self).__init__()
        self.hidden_size = hidden_size
        self.plan_length = plan_length

        # Input layer combines state and action
        self.input_size = state_size + 1    # +1 because action size is 1 (int)

        self.lstm = nn.LSTM(self.input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, action_size)

    def forward(self, state, action, num_plans):
        """
        Generate plans for `num_plans` agents.
        """
        batch_size = state.size(0)

        # Concatenate state and action for each agent
        inputs = torch.cat([state, action], dim=-1)  # (batch_size, num_plans, input_size)

        # Initialize hidden state with batch_size
        hidden = (torch.zeros(1, batch_size, self.hidden_size).to(state.device),
                  torch.zeros(1, batch_size, self.hidden_size).to(state.device))

        # Generate plans for all agents in parallel
        all_plans = []
        for _ in range(self.plan_length):
            output, hidden = self.lstm(inputs, hidden)
            action_next = self.fc(output)  # Output shape: (batch_size, num_plans, action_size)

            # Discretize action
            discrete_action_next = torch.argmax(action_next, dim=-1)  # (batch_size, num_plans)

            # Update the inputs with the predicted discrete action for the next step
            inputs = torch.cat([state, discrete_action_next.unsqueeze(-1)], dim=-1)

            all_plans.append(discrete_action_next.unsqueeze(-1))  # Append the action for each time step

        return torch.stack(all_plans, dim=1), action_next  # [batch_size, plan_length, num_plans, action_size]

