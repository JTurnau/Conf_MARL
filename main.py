import gymnasium as gym
import multigrid.envs
import torch
import os
import numpy as np
from dqn import DQN
from replay import ReplayBuffer
import torch.nn.functional as F
import random


# Set # of agents for the environment
num_agents = 2

# Create the MultiGrid environment
env = gym.make('MultiGrid-Empty-8x8-v0', agents=num_agents, joint_reward=True, max_steps=500,
               success_termination_mode="any", agent_view_size=3, allow_agent_overlap=False, render_mode='rgb_array')

# Access the unwrapped environment
base_env = env.unwrapped

# Reset the environment and retrieve initial observations
observations, infos = env.reset()


view_size = 3
world_dim = 3
obs_dim = (view_size ** 2) * world_dim + 4  # Flattened image + one-hot direction
num_actions = 3  # Number of possible actions for each agent

# Replay buffer parameters
replay_buffer_capacity = 10000
batch_size = 32

# Initialize separate replay buffers for each agent
replay_buffers = [ReplayBuffer(replay_buffer_capacity) for _ in range(num_agents)]

# Set device (use CUDA if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize DQNs and target networks for each agent
dqns = [DQN(obs_dim, num_actions) for _ in range(num_agents)]
target_dqns = [DQN(obs_dim, num_actions) for _ in range(num_agents)]

for dqn in dqns:
    dqn.to(device)

# Initialize optimizers
optimizers = [torch.optim.Adam(dqns[i].parameters(), lr=1e-3) for i in range(num_agents)]

# print(dir(base_env))  # Lists all attributes and methods of the base environment
# print(dir(base_env.agents[0]))



def test_dqn(env, dqns, num_agents, num_episodes=10, device="cpu"):
    """
    Test the performance of trained DQNs in the environment.

    Args:
        env (gym.Env): The environment to test in.
        dqns (list): List of trained DQNs for each agent.
        num_agents (int): Number of agents in the environment.
        num_episodes (int): Number of episodes to test.
        device (str or torch.device): Device to run the computations on.

    Returns:
        list: Average reward per agent over all test episodes.
    """
    total_rewards = [0] * num_agents  # Track total rewards for each agent

    for episode in range(num_episodes):
        # Reset the environment
        observations, _ = env.reset()
        done = False
        episode_rewards = [0] * num_agents  # Rewards for this episode

        while not done:
            actions = {}
            for agent_idx in range(num_agents):
                # Get the observation for the current agent
                obs = preprocess_observation(observations[agent_idx]).unsqueeze(0).to(device)

                # Forward pass through the DQN to get Q-values
                with torch.no_grad():
                    q_values = dqns[agent_idx](obs)

                # Select the action with the highest Q-value (greedy policy)
                action = torch.argmax(q_values, dim=1).item()
                actions[agent_idx] = action

            # Step the environment with actions
            observations, rewards, terminations, truncations, infos = env.step(actions)

            # Update rewards for each agent
            for agent_idx in range(num_agents):
                episode_rewards[agent_idx] += rewards[agent_idx]

            # Check if the environment has finished
            done = all(terminations.values()) or all(truncations.values())

        # Accumulate rewards for each agent
        for agent_idx in range(num_agents):
            total_rewards[agent_idx] += episode_rewards[agent_idx]

        print(f"Test Episode {episode + 1}, Rewards: {episode_rewards}")

    # Calculate average rewards for each agent
    avg_rewards = [total / num_episodes for total in total_rewards]
    print(f"Average Rewards per Agent: {avg_rewards}")

    return avg_rewards


def preprocess_observation(obs):
    """
    Preprocess the observation dictionary into a neural network input tensor.

    Args:
        obs (dict): Observation dictionary with 'image' and 'direction'.

    Returns:
        torch.Tensor: Processed observation tensor.
    """
    # Flatten the 'image' array
    image = obs['image'].reshape(-1)  # (view_size * view_size * world_dim)

    # Normalize image if needed (optional; here, we assume no normalization is needed)
    # image = image / 255.0

    # One-hot encode the 'direction'
    direction = np.zeros(4, dtype=np.float32)
    direction[obs['direction']] = 1.0

    # Combine image and direction into a single tensor
    combined = np.concatenate((image, direction))

    # Convert to torch tensor
    return torch.tensor(combined, dtype=torch.float32)


def save_models(dqns, target_dqns, optimizers, save_dir="models2", file_prefix="dqn_agent"):
    """
    Save DQN models2, target DQN models2, and their optimizers to disk.

    Args:
        dqns (list): List of DQN models2 to save.
        target_dqns (list): List of target DQN models2 to save.
        optimizers (list): List of optimizers corresponding to the DQN models2.
        save_dir (str): Directory where models2 will be saved.
        file_prefix (str): Prefix for the saved model filenames.
    """
    os.makedirs(save_dir, exist_ok=True)  # Create the directory if it doesn't exist

    for i, (dqn, target_dqn, optimizer) in enumerate(zip(dqns, target_dqns, optimizers)):
        torch.save({
            'model_state_dict': dqn.state_dict(),
            'target_model_state_dict': target_dqn.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        }, os.path.join(save_dir, f"{file_prefix}_{i}.pth"))
    print(f"Models and target models2 saved to '{save_dir}'")


def load_models(dqns, target_dqns, optimizers, load_dir="models2", file_prefix="dqn_agent", device="cpu"):
    """
    Load DQN models2, target DQN models2, and their optimizers from disk.

    Args:
        dqns (list): List of DQN models2 to load states into.
        target_dqns (list): List of target DQN models2 to load states into.
        optimizers (list): List of optimizers to load states into.
        load_dir (str): Directory where models2 are saved.
        file_prefix (str): Prefix for the saved model filenames.
        device (str or torch.device): Device to map the loaded tensors to.

    Returns:
        list: Loaded models2.
        list: Loaded target models2.
        list: Loaded optimizers.
    """
    for i, (dqn, target_dqn, optimizer) in enumerate(zip(dqns, target_dqns, optimizers)):
        checkpoint = torch.load(os.path.join(load_dir, f"{file_prefix}_{i}.pth"), map_location=device, weights_only=True)
        dqn.load_state_dict(checkpoint['model_state_dict'])
        target_dqn.load_state_dict(checkpoint['target_model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        dqn.to(device)
        target_dqn.to(device)

    print(f"Models and target models2 loaded from '{load_dir}'")


# Load models2 if available
load_models(dqns, target_dqns, optimizers, device=device)

# Hyperparameters for training
gamma = 0.99  # Discount factor
tau = 0.005  # Target network soft update factor
update_frequency = 10  # Frequency of target network update
epsilon = 0.2  # Exploration probability

# Training loop
num_episodes = 1  # Define the number of training episodes

for episode in range(num_episodes):
    done = False
    total_reward = [0] * num_agents  # Track rewards per agent

    # Reset the environment every episode
    env.reset()

    while not done:
        actions = {}
        for agent_idx in range(num_agents):
            # Get the observation for the current agent
            obs = preprocess_observation(observations[agent_idx]).unsqueeze(0).to(device)

            # Forward pass through DQN to get Q-values
            q_values = dqns[agent_idx](obs)

            # Decide whether to explore or exploit
            if random.random() < epsilon:  # Explore: choose a random action
                action = random.randint(0, num_actions - 1)
            else:  # Exploit: choose the action with the highest Q-value
                q_values = dqns[agent_idx](obs)
                action = torch.argmax(q_values, dim=1).item()

            actions[agent_idx] = action

        # Store past observations
        past_observations = observations

        # Step the environment with actions
        observations, rewards, terminations, truncations, infos = env.step(actions)

        # Add transitions to the respective agent's replay buffer
        for agent_idx in range(num_agents):
            state = preprocess_observation(past_observations[agent_idx])
            next_state = preprocess_observation(observations[agent_idx])
            replay_buffers[agent_idx].add(
                state,  # Current state
                actions[agent_idx],  # Action taken
                rewards[agent_idx],  # Reward received
                next_state,  # Next state
                terminations[agent_idx] or truncations[agent_idx]  # Done flag
            )
            total_reward[agent_idx] += rewards[agent_idx]

        # Check if the environment has finished
        done = all(terminations.values()) or all(truncations.values())

        if done:
            # Train the DQN for each agent using all samples in the replay buffer
            for agent_idx in range(num_agents):
                print(f"Training DQN for Agent {agent_idx}...")

                # Retrieve all samples from the replay buffer
                states, actions, rewards, next_states, dones = replay_buffers[agent_idx].get_all()

                states = states.to(device)
                next_states = next_states.to(device)
                actions = actions.to(device)
                rewards = rewards.to(device)
                dones = dones.to(device)

                # Process the entire buffer in mini-batches (if necessary for memory efficiency)
                num_samples = states.size(0)
                for start_idx in range(0, num_samples, batch_size):
                    end_idx = min(start_idx + batch_size, num_samples)

                    # Mini-batch
                    batch_states = states[start_idx:end_idx]
                    batch_next_states = next_states[start_idx:end_idx]
                    batch_actions = actions[start_idx:end_idx]
                    batch_rewards = rewards[start_idx:end_idx]
                    batch_dones = dones[start_idx:end_idx]

                    # Compute Q-values for the current state and next state
                    q_values = dqns[agent_idx](batch_states)
                    next_q_values = target_dqns[agent_idx](batch_next_states)

                    # Get the Q-value of the selected action
                    q_value = q_values.gather(1, batch_actions.unsqueeze(1))

                    # Compute the target Q-value using the Bellman equation
                    target_q_value = batch_rewards.unsqueeze(1) + (gamma * next_q_values.max(1)[0].unsqueeze(1)) * (1 - batch_dones.unsqueeze(1).float())

                    # Compute the loss (Mean Squared Error)
                    loss = F.mse_loss(q_value, target_q_value)

                    # Optimize the DQN
                    optimizers[agent_idx].zero_grad()
                    loss.backward()
                    optimizers[agent_idx].step()

        # Periodically update the target networks with the online DQN weights
        if episode % update_frequency == 0:
            for agent_idx in range(num_agents):
                # Soft update the target network
                for target_param, param in zip(target_dqns[agent_idx].parameters(), dqns[agent_idx].parameters()):
                    target_param.data.copy_((1 - tau) * target_param.data + tau * param.data)

    # Print total reward for this episode
    print(f"Episode {episode + 1}, Rewards: {total_reward}")

    # Optionally save models2 at regular intervals
    if episode % num_episodes*.1 == 0:
        save_models(dqns, target_dqns, optimizers)

# After training, save final models2
save_models(dqns, target_dqns, optimizers)
env.close()


# Test the trained DQNs
avg_rewards = test_dqn(env, dqns, num_agents, num_episodes=1, device=device)
print(f"Average rewards across 10 test episodes: {avg_rewards}")


