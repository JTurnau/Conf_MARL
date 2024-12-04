from collections import deque
import random
import torch


class ReplayBuffer:
    def __init__(self, capacity):
        """
        Initialize the replay buffer.

        Args:
            capacity (int): Maximum number of transitions to store.
        """
        self.buffer = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        """
        Add a transition to the replay buffer.

        Args:
            state (torch.Tensor): The current state.
            action (int): The action taken.
            reward (float): The reward received.
            next_state (torch.Tensor): The next state.
            done (bool): Whether the episode ended after this transition.
        """
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        """
        Sample a batch of transitions from the replay buffer.

        Args:
            batch_size (int): Number of transitions to sample.

        Returns:
            tuple: Batch of states, actions, rewards, next states, and dones.
        """
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.stack(states),
            torch.tensor(actions, dtype=torch.long),
            torch.tensor(rewards, dtype=torch.float32),
            torch.stack(next_states),
            torch.tensor(dones, dtype=torch.float32),
        )

    def sample_lstm(self, batch_size):
        """
        Sample a batch of transitions from the replay buffer.

        Args:
            batch_size (int): Number of transitions to sample.

        Returns:
            tuple: Batch of states, actions, rewards, next states, and dones.
        """
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.stack(states),
            torch.stack(actions),
            torch.tensor(rewards, dtype=torch.float32),
            torch.stack(next_states),
            torch.tensor(dones, dtype=torch.float32),
        )

    def get_all(self):
        states = torch.stack([entry[0] for entry in self.buffer])
        actions = torch.tensor([entry[1] for entry in self.buffer])
        rewards = torch.tensor([entry[2] for entry in self.buffer])
        next_states = torch.stack([entry[3] for entry in self.buffer])
        dones = torch.tensor([entry[4] for entry in self.buffer])
        return states, actions, rewards, next_states, dones

    def __len__(self):
        """
        Get the current size of the replay buffer.

        Returns:
            int: Number of transitions stored in the buffer.
        """
        return len(self.buffer)

    def clear(self):
        """
        Clear all transitions from the replay buffer.
        """
        self.buffer.clear()
