import gymnasium as gym
import multigrid.envs
import torch
import os
import numpy as np
from dqn import DQN
from replay import ReplayBuffer
import torch.nn.functional as F
import random
import time


class MultiAgentDQN:
    def __init__(self, num_agents=2, env_name='MultiGrid-Empty-8x8-v0', obs_dim=31, num_actions=3, replay_buffer_capacity=100000, batch_size=32, device=None):
        # Initialize environment and parameters
        self.num_agents = num_agents
        self.env = gym.make(env_name, agents=num_agents, joint_reward=False, max_steps=500,
                            success_termination_mode="all", agent_view_size=3, allow_agent_overlap=True, render_mode='rgb_array')
        self.base_env = self.env.unwrapped
        self.obs_dim = obs_dim
        self.num_actions = num_actions
        self.replay_buffer_capacity = replay_buffer_capacity
        self.batch_size = batch_size
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.best_reward = 0
        self.best_eval_reward = 0

        # Initialize replay buffers for each agent
        self.replay_buffers = [ReplayBuffer(self.replay_buffer_capacity) for _ in range(self.num_agents)]

        # Initialize DQNs and target networks for each agent
        self.dqns = [DQN(self.obs_dim, self.num_actions) for _ in range(self.num_agents)]
        self.target_dqns = [DQN(self.obs_dim, self.num_actions) for _ in range(self.num_agents)]

        # Move DQNs and Target DQNs to GPU
        for dqn in self.dqns:
            dqn.to(self.device)

        for tdqn in self.target_dqns:
            tdqn.to(self.device)

        # Initialize optimizers
        self.optimizers = [torch.optim.Adam(self.dqns[i].parameters(), lr=1e-3) for i in range(self.num_agents)]

    def preprocess_observation(self, obs):
        image = obs['image'].reshape(-1)
        direction = np.zeros(4, dtype=np.float32)
        direction[obs['direction']] = 1.0
        combined = np.concatenate((image, direction))
        return torch.tensor(combined, dtype=torch.float32)

    def place_agents_at_start(self):
        fixed_position1 = (np.int64(2), np.int64(1))
        fixed_position2 = (np.int64(1), np.int64(2))

        # Place each agent in an empty position at the start of the episode
        for agent_idx in range(self.num_agents):

            agent = self.env.unwrapped.agents[agent_idx]  # Get the agent object
            # You can adjust the top-left position and size as per your environment's grid setup
            if agent_idx == 0:
                self.env.unwrapped.place_agent(agent, fixed_position1, (1, 1), rand_dir=False)
            else:
                self.env.unwrapped.place_agent(agent, fixed_position2, (1, 1), rand_dir=False)

    def test_dqn(self, num_episodes=10):
        total_rewards = [0] * self.num_agents

        for episode in range(num_episodes):
            observations, _ = self.env.reset()
            self.place_agents_at_start()

            done = False
            episode_rewards = [0] * self.num_agents

            while not done:
                actions = {}
                for agent_idx in range(self.num_agents):
                    obs = self.preprocess_observation(observations[agent_idx]).unsqueeze(0).to(self.device)
                    with torch.no_grad():
                        q_values = self.dqns[agent_idx](obs)
                    action = torch.argmax(q_values, dim=1).item()
                    actions[agent_idx] = action

                observations, rewards, terminations, truncations, infos = self.env.step(actions)

                for agent_idx in range(self.num_agents):
                    episode_rewards[agent_idx] += rewards[agent_idx]

                done = all(terminations.values()) or all(truncations.values())

            for agent_idx in range(self.num_agents):
                total_rewards[agent_idx] += episode_rewards[agent_idx]

            print(f"Test Episode {episode + 1}, Rewards: {episode_rewards}")

        avg_rewards = [total / num_episodes for total in total_rewards]
        print(f"\nAverage Rewards per Agent: {avg_rewards}\n")
        return avg_rewards

    def save_models(self, save_dir="models", file_prefix="dqn_agent"):
        os.makedirs(save_dir, exist_ok=True)

        for i, (dqn, target_dqn, optimizer) in enumerate(zip(self.dqns, self.target_dqns, self.optimizers)):
            torch.save({
                'model_state_dict': dqn.state_dict(),
                'target_model_state_dict': target_dqn.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
            }, os.path.join(save_dir, f"{file_prefix}_{i}.pth"))
        print(f"Models and target models saved to '{save_dir}'")

    def load_models(self, load_dir="models", file_prefix="dqn_agent"):
        for i, (dqn, target_dqn, optimizer) in enumerate(zip(self.dqns, self.target_dqns, self.optimizers)):
            checkpoint = torch.load(os.path.join(load_dir, f"{file_prefix}_{i}.pth"), map_location=self.device, weights_only=True)
            dqn.load_state_dict(checkpoint['model_state_dict'])
            target_dqn.load_state_dict(checkpoint['target_model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            dqn.to(self.device)
            target_dqn.to(self.device)
        print(f"Models and target models loaded from '{load_dir}'")

    def train(self, num_episodes=100, gamma=0.99, tau=0.005, update_frequency=1, epsilon=0.2, epsilon_decay=0.99):
        total_sum_reward = 0
        for episode in range(num_episodes):
            done = False
            episode_reward = [0] * self.num_agents

            done_flags = [False for _ in range(self.num_agents)]

            observations, _ = self.env.reset()

            self.place_agents_at_start()  # Place agents at empty positions at the start

            while not done:
                actions = {}
                for agent_idx in range(self.num_agents):
                    obs = self.preprocess_observation(observations[agent_idx]).unsqueeze(0).to(self.device)
                    q_values = self.dqns[agent_idx](obs)

                    if random.random() < epsilon:
                        action = random.randint(0, self.num_actions - 1)
                    else:
                        action = torch.argmax(q_values, dim=1).item()

                    actions[agent_idx] = action

                past_observations = observations
                observations, rewards, terminations, truncations, infos = self.env.step(actions)

                for agent_idx in range(self.num_agents):

                    if not done_flags[agent_idx]:
                        state = self.preprocess_observation(past_observations[agent_idx])
                        next_state = self.preprocess_observation(observations[agent_idx])
                        self.replay_buffers[agent_idx].add(
                            state, actions[agent_idx], rewards[agent_idx], next_state,
                            terminations[agent_idx] or truncations[agent_idx]
                        )

                    if terminations[agent_idx]:
                        done_flags[agent_idx] = True

                    episode_reward[agent_idx] += rewards[agent_idx]

                done = all(terminations.values()) or all(truncations.values())

            print(f"Episode {episode + 1}, Rewards: {episode_reward}")

            if (episode+1) % update_frequency == 0:
                for agent_idx in range(self.num_agents):
                    states, actions, rewards, next_states, dones = self.replay_buffers[agent_idx].get_all()
                    states = states.to(self.device)
                    next_states = next_states.to(self.device)
                    actions = actions.to(self.device)
                    rewards = rewards.to(self.device)
                    dones = dones.to(self.device)

                    num_samples = states.size(0)

                    for start_idx in range(0, num_samples, self.batch_size):
                        end_idx = min(start_idx + self.batch_size, num_samples)

                        batch_states = states[start_idx:end_idx]
                        batch_next_states = next_states[start_idx:end_idx]
                        batch_actions = actions[start_idx:end_idx]
                        batch_rewards = rewards[start_idx:end_idx]
                        batch_dones = dones[start_idx:end_idx]

                        q_values = self.dqns[agent_idx](batch_states)
                        self.target_dqns[agent_idx].to(self.device)
                        next_q_values = self.target_dqns[agent_idx](batch_next_states.to(self.device))

                        q_value = q_values.gather(1, batch_actions.unsqueeze(1))
                        target_q_value = batch_rewards.unsqueeze(1) + (gamma * next_q_values.max(1)[0].unsqueeze(1)) * (1 - batch_dones.unsqueeze(1).float())

                        loss = F.mse_loss(q_value, target_q_value)

                        self.optimizers[agent_idx].zero_grad()
                        loss.backward()

                        # Debug gradients for the current agent's DQN
                        # for name, param in self.dqns[agent_idx].named_parameters():
                        #     if param.grad is None:
                        #         print(f"Agent {agent_idx}: No gradient for {name}")
                        #     else:
                        #         print(
                        #             f"Agent {agent_idx}: Gradient for {name} has mean {param.grad.mean().item()}")

                        self.optimizers[agent_idx].step()

                    print(f"Updated Main DQN")

                    for agent_idx in range(self.num_agents):
                        for target_param, param in zip(self.target_dqns[agent_idx].parameters(), self.dqns[agent_idx].parameters()):
                            target_param.data.copy_((1 - tau) * target_param.data + tau * param.data)
                    print(f"Updated Target DQN")

            # Decay epsilon value
            #epsilon *= epsilon_decay

            # Check and save the best models
            total_sum_reward += sum(episode_reward)

            if episode > 1 and episode % (num_episodes) == 0:
                self.save_models()

        return total_sum_reward


if __name__ == "__main__":
    train_from_scratch = True

    agent_dqn = MultiAgentDQN()

    if not train_from_scratch:
        agent_dqn.load_models("best_models")

    #Run training and validation epochs
    # for e in range(250):
    #
    #     print(f"\n-- Starting Epoch {e+1}/250 --")
    #
    #     if not train_from_scratch:
    #         #Check if the "best_models" directory exists
    #         if os.path.exists("best_models"):
    #             agent_dqn.load_models("best_models")
    #         else:
    #             agent_dqn.load_models()
    #
    #     # Train for x number of episodes
    #     train_reward = agent_dqn.train(num_episodes=1)
    #
    #     # Evaluate x number of episodes
    #     eval_rewards = agent_dqn.test_dqn(num_episodes=1)
    #
    #     # Save new best model if it evaluates the best
    #     if np.sum(eval_rewards) > agent_dqn.best_eval_reward:
    #         agent_dqn.best_eval_reward = np.sum(eval_rewards)
    #
    #         agent_dqn.save_models(save_dir="IDQN_5x5_final")
    #         print(f"New best reward {agent_dqn.best_eval_reward}. Models saved.")

    agent_dqn.env = gym.make('MultiGrid-Empty-5x5-v0', agents=agent_dqn.num_agents, joint_reward=False, max_steps=50,
                        success_termination_mode="all", agent_view_size=3, allow_agent_overlap=True,
                        render_mode='human')

    print(f"\nStarting evaluation demonstration...")
    print(f"Best Model evaluation")
    agent_dqn.load_models("IDQN_5x5_final")
    agent_dqn.test_dqn(num_episodes=10)
