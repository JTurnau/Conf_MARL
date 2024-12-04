import gymnasium as gym
import multigrid.envs
import torch
import os
import json
import numpy as np
from dqn import DQN
from replay import ReplayBuffer
from lstm_planner import LSTMPlanner
from confidence_net import ConfidenceNetwork
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

        self.confidence_threshold = 1.2

        # Store best fantasy rewards for each agent
        self.best_rewards = [0 for _ in range(self.num_agents)]
        self.best_traj = [0 for _ in range(self.num_agents)]

        # Initialize replay buffers for each agent
        self.replay_buffers = [ReplayBuffer(self.replay_buffer_capacity) for _ in range(self.num_agents)]

        self.lstm_replay_buffers = [ReplayBuffer(self.replay_buffer_capacity) for _ in range(self.num_agents)]

        self.confidence_replay_buffers = [ReplayBuffer(self.replay_buffer_capacity) for _ in range(self.num_agents)]

        # Initialize DQNs and target networks for each agent
        self.dqns = [DQN(self.obs_dim, self.num_actions) for _ in range(self.num_agents)]
        self.target_dqns = [DQN(self.obs_dim, self.num_actions) for _ in range(self.num_agents)]

        # LSTM Planner Hyperparameters
        self.lstm_hidden_size = 32  # Number of LSTM units
        self.lstm_plan_length = 1  # Number of actions to generate

        # Initialize LSTM planners for each agent
        self.lstm_planners = [
            LSTMPlanner(state_size=self.obs_dim*self.num_agents, action_size=self.num_actions,
                        hidden_size=self.lstm_hidden_size, plan_length=self.lstm_plan_length)
            for _ in range(self.num_agents)
        ]

        # Initialize Confidence Networks for each agent
        self.confidence_networks = [
            ConfidenceNetwork(state_dim=self.obs_dim, action_dim=self.num_actions).to(self.device)
            for _ in range(self.num_agents)
        ]

        # Move DQNs to GPU
        for dqn in self.dqns:
            dqn.to(self.device)

        # Move Target DQNs to GPU
        for tdqn in self.target_dqns:
            tdqn.to(self.device)

        # Move LSTM Planners to GPU
        for lstm_p in self.lstm_planners:
            lstm_p.to(self.device)

        # Move Confidence Networks to GPU
        for conf_nn in self.confidence_networks:
            conf_nn.to(self.device)

        # Initialize optimizers
        self.optimizers = [torch.optim.Adam(self.dqns[i].parameters(), lr=1e-3) for i in range(self.num_agents)]
        self.lstm_optimizers = [torch.optim.Adam(lstm_p.parameters(), lr=1e-3) for lstm_p in self.lstm_planners]
        self.confidence_optimizers = [torch.optim.Adam(self.confidence_networks[i].parameters(), lr=1e-3) for i in range(self.num_agents)]

    def replay(self, traj):
        observations, _ = self.env.reset()
        self.place_agents_at_start()

        for actions in traj:
            with torch.no_grad():

                observations, rewards, terminations, truncations, infos = self.env.step(actions)


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

        # Save DQN models
        for i, (dqn, target_dqn, optimizer) in enumerate(zip(self.dqns, self.target_dqns, self.optimizers)):
            torch.save({
                'model_state_dict': dqn.state_dict(),
                'target_model_state_dict': target_dqn.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
            }, os.path.join(save_dir, f"{file_prefix}_dqn_{i}.pth"))

        # Save confidence networks
        for i, (conf_net, conf_optimizer) in enumerate(zip(self.confidence_networks, self.confidence_optimizers)):
            torch.save({
                'model_state_dict': conf_net.state_dict(),
                'optimizer_state_dict': conf_optimizer.state_dict()
            }, os.path.join(save_dir, f"{file_prefix}_conf_net_{i}.pth"))

        # Save LSTM planners
        for i, (lstm, lstm_optimizer) in enumerate(zip(self.lstm_planners, self.lstm_optimizers)):
            torch.save({
                'model_state_dict': lstm.state_dict(),
                'optimizer_state_dict': lstm_optimizer.state_dict()
            }, os.path.join(save_dir, f"{file_prefix}_lstm_{i}.pth"))

        print(f"All models saved to '{save_dir}'")

    def load_models(self, load_dir="models", file_prefix="dqn_agent"):
        # Load DQN models
        for i, (dqn, target_dqn, optimizer) in enumerate(zip(self.dqns, self.target_dqns, self.optimizers)):
            checkpoint = torch.load(os.path.join(load_dir, f"{file_prefix}_dqn_{i}.pth"), map_location=self.device, weights_only=True)
            dqn.load_state_dict(checkpoint['model_state_dict'])
            target_dqn.load_state_dict(checkpoint['target_model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            dqn.to(self.device)
            target_dqn.to(self.device)

        # Load confidence networks
        for i, (conf_net, conf_optimizer) in enumerate(zip(self.confidence_networks, self.confidence_optimizers)):
            checkpoint = torch.load(os.path.join(load_dir, f"{file_prefix}_conf_net_{i}.pth"), map_location=self.device, weights_only=True)
            conf_net.load_state_dict(checkpoint['model_state_dict'])
            conf_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            conf_net.to(self.device)

        # Load LSTM planners
        for i, (lstm, lstm_optimizer) in enumerate(zip(self.lstm_planners, self.lstm_optimizers)):
            checkpoint = torch.load(os.path.join(load_dir, f"{file_prefix}_lstm_{i}.pth"), map_location=self.device, weights_only=True)
            lstm.load_state_dict(checkpoint['model_state_dict'])
            lstm_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            lstm.to(self.device)

        print(f"All models loaded from '{load_dir}'")

    def train_confidence_network(self, agent_idx, gamma=0.99):
        print(f"\n--- Training Confidence Network for Agent {agent_idx} ---")

        confidence_network = self.confidence_networks[agent_idx]
        optimizer = self.confidence_optimizers[agent_idx]
        replay_buffer = self.confidence_replay_buffers[agent_idx]

        # Get all values from the replay buffer
        states, actions, rewards, next_states, dones = replay_buffer.get_all()

        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)

        # Compute returns (discounted rewards)
        returns = []
        discounted_sum = 0.0

        # Step 1: Compute the returns
        for reward, done in zip(reversed(rewards), reversed(dones)):
            if done:
                discounted_sum = 0.0
            discounted_sum = reward + gamma * discounted_sum
            returns.insert(0, discounted_sum)

        returns = torch.tensor(returns, dtype=torch.float32).to(self.device)

        actions_one_hot = F.one_hot(actions, num_classes=self.num_actions).float().to(self.device)

        # Forward pass through confidence network
        predicted_confidences = confidence_network(states, actions_one_hot)

        # Compute MSE loss between confidence and returns
        mse_loss = F.mse_loss(predicted_confidences, returns.reshape(-1, 1))

        # Compute the confidence-weighted loss (modulate based on reward scaling or confidence)
        confidence_weighted_loss = (mse_loss * predicted_confidences).mean()

        # Optionally scale reward with confidence (alternative loss form)
        #reward_scaled_loss = (confidence_weighted_loss * rewards).mean()

        # Backpropagation
        optimizer.zero_grad()
        confidence_weighted_loss.backward()  # Or confidence_weighted_loss depending on what you want
        optimizer.step()

        return confidence_weighted_loss.item()

    def train_lstm(self, agent_idx, gamma=0.99):
        """Train the LSTM planner using vanilla policy gradient."""
        print(f"\n--- Training LSTM for Agent {agent_idx} ---")

        lstm_planner = self.lstm_planners[agent_idx]
        optimizer = self.lstm_optimizers[agent_idx]
        replay_buffer = self.lstm_replay_buffers[agent_idx]

        # Sample a batch from the replay buffer
        states, actions, rewards, next_states, dones = replay_buffer.get_all()

        # Convert data to tensors
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)

        # Compute returns (discounted rewards)
        returns = []
        discounted_sum = 0.0

        # Step 1: Compute the returns
        for reward, done in zip(reversed(rewards), reversed(dones)):
            if done:
                discounted_sum = 0.0
            discounted_sum = reward + gamma * discounted_sum
            returns.insert(0, discounted_sum)

        # Step 2: Check if all values are 0, and if so, set all to -1
        if all(value == 0 for value in returns):
            returns = [-1]  # Set value to -1 if all are 0

        returns = torch.tensor(returns, dtype=torch.float32).to(self.device)

        # Normalize returns for stability
        # returns_mean = returns.mean().item()
        # returns_std = returns.std().item()
        # returns = (returns - returns_mean) / (returns_std + 1e-8)

        # Forward pass through LSTM planner
        optimizer.zero_grad()

        lstm_outputs, raw_output = lstm_planner(states, actions.unsqueeze(-1).unsqueeze(-1), self.num_agents - 1)

        # Compute log probabilities of taken actions
        action_probs = F.softmax(raw_output, dim=-1)

        max_action_prob, _ = torch.max(action_probs, dim=-1)

        log_probs = torch.log(max_action_prob + 1e-8)  # Avoid log(0)

        returns = returns.unsqueeze(1).unsqueeze(2)

        # Compute policy gradient loss
        loss = (log_probs * returns).mean()

        # Backpropagation
        loss.backward()
        optimizer.step()
        print(f"Updated LSTM for Agent {agent_idx}. Loss: {loss.item()}")

        return loss.item()

    def fantasy_rollout(self, active_agent_idx, num_epochs=1, gamma=0.99):
        total_rewards = [0] * self.num_agents
        total_lstm_loss = 0
        total_conf_loss = 0

        for epoch in range(num_epochs):
            # Clear the lstm and confidence net replay buffers to store single trajectory at a time (online learning)
            for agent_idx in range(self.num_agents):
                self.lstm_replay_buffers[agent_idx].clear()
                self.confidence_replay_buffers[agent_idx].clear()

            print(f"\n--- Epoch {epoch + 1}, Fantasy Rollout for Agent {active_agent_idx} ---")  # Debug: Epoch Start
            done_flags = [False] * self.num_agents  # To track if an agent is done
            episode_rewards = [0] * self.num_agents
            observations, _ = self.env.reset()
            self.place_agents_at_start()  # Reset agent positions

            done = False
            step_counter = 0  # Debug: Step counter
            current_traj = []
            while not done:

                actions = {}
                plan_obs = {}

                # Initialize the actions list
                for agent_idx in range(self.num_agents):
                    actions[agent_idx] = 0

                # Active agent's DQN action
                for agent_idx in range(self.num_agents):
                    if agent_idx == active_agent_idx:
                        obs = self.preprocess_observation(observations[agent_idx]).unsqueeze(0).to(self.device)
                        q_values = self.dqns[agent_idx](obs)
                        action = torch.argmax(q_values, dim=1).item()
                        actions[agent_idx] = action
                    else:
                        plan_obs[agent_idx] = self.preprocess_observation(observations[agent_idx]).unsqueeze(0).to(
                            self.device)

                # LSTM planner actions for other agents
                action_tensor = (torch.tensor(actions[active_agent_idx], dtype=torch.float32)
                                 .unsqueeze(0).reshape(1,1).to(self.device))
                concatenated_plan_obs = torch.cat([obs] + list(plan_obs.values()), dim=-1)
                lstm_plans, raw_output = self.lstm_planners[active_agent_idx](concatenated_plan_obs.unsqueeze(0), action_tensor.unsqueeze(0),
                                                                              self.num_agents - 1)

                # Set other agent's actions based on plan generated by active agent
                for agent_idx in range(self.num_agents):
                    if agent_idx != active_agent_idx:

                        # Compute log probabilities of taken actions
                        action_probs = F.softmax(raw_output, dim=-1)

                        action = torch.multinomial(action_probs.squeeze(0), num_samples=1).squeeze(-1).item()

                        actions[agent_idx] = action

                # Store in current trajectory
                current_traj.append(actions)

                # Take a step in the environment
                past_observations = observations
                observations, rewards, terminations, truncations, infos = self.env.step(actions)

                # Update LSTM replay buffer
                for agent_idx in range(self.num_agents):

                    if not done_flags[agent_idx]:

                        state = torch.cat([self.preprocess_observation(past_observations[i]).unsqueeze(0) for i in
                                               range(self.num_agents)], dim=-1)
                        next_state = torch.cat(
                            [self.preprocess_observation(observations[i]).unsqueeze(0) for i in range(self.num_agents)],
                            dim=-1)

                        done_flag = terminations[agent_idx] or truncations[agent_idx]

                        reward = rewards[agent_idx]

                        # Add to the replay buffers (aka store single trajectory) for the non-active agents
                        if agent_idx != active_agent_idx:

                            # Add to LSTM replay buffer (clearing every loop so using this to store online trajectory)
                            self.lstm_replay_buffers[active_agent_idx].add(
                                state,
                                action_tensor.item(),
                                reward,
                                next_state,
                                done_flag
                            )

                            # Add to confidence replay buffer (clearing every loop so using this to store online trajectory)
                            conf_state = self.preprocess_observation(past_observations[agent_idx])
                            conf_next_state = self.preprocess_observation(observations[agent_idx])
                            self.confidence_replay_buffers[active_agent_idx].add(
                                conf_state, actions[agent_idx], reward, conf_next_state,
                                terminations[agent_idx] or truncations[agent_idx]
                            )

                        if done_flag:
                            done_flags[agent_idx] = True

                        episode_rewards[agent_idx] += rewards[agent_idx]

                done = all(done_flags)
                step_counter += 1

            states, actions, rewards, next_states, dones = self.lstm_replay_buffers[active_agent_idx].get_all()

            # Convert rewards to tensors
            rewards = rewards.to(self.device)

            if rewards.max().item() > self.best_rewards[active_agent_idx]:
                self.best_rewards[active_agent_idx] = rewards.max().item()
                self.best_traj[active_agent_idx] = current_traj

            lstm_loss = self.train_lstm(active_agent_idx, gamma=gamma)
            conf_loss = self.train_confidence_network(active_agent_idx, gamma=gamma)

            total_lstm_loss += lstm_loss
            total_conf_loss += conf_loss

            # Print rewards for the epoch
            print(f"\nEpoch {epoch + 1} completed. Rewards: {episode_rewards}")
            for agent_idx in range(self.num_agents):
                total_rewards[agent_idx] += episode_rewards[agent_idx]

        avg_rewards = [total / num_epochs for total in total_rewards]
        print(f"Average Rewards per Agent: {avg_rewards}")
        return avg_rewards, total_lstm_loss/num_epochs, total_conf_loss/num_epochs

    def train(self, num_episodes=1, gamma=0.99, tau=0.005, update_frequency=1, epsilon=0.2, epsilon_decay=0.99):
        total_sum_reward = 0

        for episode in range(num_episodes):

            print(f"\n--- Episode {episode+1}, Training DQNs using current LSTM and Confidence nets ---")

            done = False
            episode_reward = [0] * self.num_agents

            done_flags = [False for _ in range(self.num_agents)]

            observations, _ = self.env.reset()

            self.place_agents_at_start()  # Place agents at empty positions at the start

            while not done:
                actions = {}
                plan_obs = {}
                for agent_idx in range(self.num_agents):
                    obs = self.preprocess_observation(observations[agent_idx]).unsqueeze(0).to(self.device)

                    q_values = self.dqns[agent_idx](obs)

                    if random.random() < epsilon:
                        action = random.randint(0, self.num_actions - 1)
                    else:
                        action = torch.argmax(q_values, dim=1).item()

                    plan_obs[agent_idx] = obs

                    actions[agent_idx] = action

                # Store the confidence scores of agents at each timestep
                confidence_scores = []

                # Store the plans for other agents at each timestep
                plans_for_others = []

                # Generate plans based on observations of teammates
                for agent_idx in range(self.num_agents):

                    # Convert action to tensor and ensure it's the correct shape
                    action_tensor = torch.tensor(actions[agent_idx], dtype=torch.long).unsqueeze(0).unsqueeze(0).to(self.device)

                    concatenated_plan_obs = torch.cat(list(plan_obs.values()), dim=-1)

                    plans, _ = self.lstm_planners[agent_idx](concatenated_plan_obs.unsqueeze(0), action_tensor.unsqueeze(0), self.num_agents - 1)

                    plans_for_others.append(plans)

                    action_one_hot = F.one_hot(action_tensor, num_classes=self.num_actions).float().to(self.device)

                    # Forward pass through confidence network
                    predicted_confidence = self.confidence_networks[agent_idx](plan_obs[agent_idx], action_one_hot.squeeze(0))

                    confidence_scores.append(predicted_confidence.item())

                # Rethink phase based on confidence scores
                for agent_idx in range(self.num_agents):

                    # If external agent confidence is greater than the active agent's internal confidence
                    # by a set threshold, active agent changes actions
                    for agent_idx2 in range(self.num_agents):

                        if confidence_scores[agent_idx2] != confidence_scores[agent_idx]:
                            if confidence_scores[agent_idx2] > self.confidence_threshold:
                                # print(f"Agent {agent_idx}'s action {actions[agent_idx]} is being overwritten "
                                #       f"to {plans_for_others[agent_idx2].item()} "
                                #       f"confidence_scores[agent_idx2]: {confidence_scores[agent_idx2]}"
                                #       f" > self.confidence_threshold: {self.confidence_threshold}")

                                # Change active agent's action to defer to the planner of the other agent
                                actions[agent_idx] = plans_for_others[agent_idx2].item()

                        # # Checks if external - internal confidence is greater than a set threshold
                        # if confidence_scores[agent_idx2] - confidence_scores[agent_idx] > self.confidence_threshold:
                        #
                        #     # print(f"Agent {agent_idx2}'s external confidence of {confidence_scores[agent_idx2]}"
                        #     #       f" is more than {self.confidence_threshold} greater than agent {agent_idx}'s"
                        #     #       f" internal confidence: {confidence_scores[agent_idx]}")
                        #
                        #     # Change active agent's action to defer to the planner of the other agent
                        #     actions[agent_idx] = plans_for_others[agent_idx2].item()

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

            if episode > 1 and episode % num_episodes == 0:
                self.save_models()

        return total_sum_reward


if __name__ == "__main__":
    train_from_scratch = True
    metrics_file = "8x8_training_metrics_IDQN.json"
    agent_0_fantasy_replay = "8x8_IDQN_agent_0_fantasy_replay.json"
    agent_1_fantasy_replay = "8x8_IDQN_agent_1_fantasy_replay.json"

    # Initialize metric storage
    if os.path.exists(metrics_file):
        with open(metrics_file, "r") as f:
            metrics = json.load(f)
    else:
        metrics = []

    agent_dqn = MultiAgentDQN()

    if not train_from_scratch:
        # Load pretrained models
        agent_dqn.load_models("models")

    epochs = 250

    # Run training and validation epochs
    for epoch in range(epochs):
        print(f"Starting Epoch {epoch + 1}/{epochs}")

        # Train for a specified number of episodes
        train_reward = agent_dqn.train(num_episodes=1, update_frequency=1)

        # Perform fantasy rollouts for selected agents
        #fantasy_rewards_0, fantasy_lstm_loss_0, fantasy_conf_loss_0 = agent_dqn.fantasy_rollout(active_agent_idx=0, num_epochs=1)
        #fantasy_rewards_1, fantasy_lstm_loss_1, fantasy_conf_loss_1 = agent_dqn.fantasy_rollout(active_agent_idx=1, num_epochs=1)

        # Save metrics for this epoch
        # epoch_metrics = {
        #     "epoch": epoch + 1,
        #     "train_reward": train_reward,
        #     "fantasy_rewards": {
        #         "agent_0": fantasy_rewards_0,
        #         "agent_1": fantasy_rewards_1
        #     },
        #     "fantasy_lstm_loss": {
        #         "agent_0": fantasy_lstm_loss_0,
        #         "agent_1": fantasy_lstm_loss_1
        #     },
        #     "fantasy_conf_loss": {
        #         "agent_0": fantasy_conf_loss_0,
        #         "agent_1": fantasy_conf_loss_1
        #     }
        # }

        epoch_metrics = {
            "epoch": epoch + 1,
            "train_reward": train_reward
        }
        metrics.append(epoch_metrics)

        # Write metrics to JSON file
        with open(metrics_file, "w") as f:
            json.dump(metrics, f, indent=4)
        print(f"Metrics for Epoch {epoch + 1} saved.")

    # Save models after training
    agent_dqn.save_models(save_dir="models", file_prefix="dqn_agent")
    print("Training completed. Models saved.")

    # agent_dqn.env = gym.make('MultiGrid-Empty-5x5-v0', agents=agent_dqn.num_agents, joint_reward=False, max_steps=100,
    #                          success_termination_mode="all", agent_view_size=3, allow_agent_overlap=True,
    #                          render_mode='human')

    # try:
    #     with open(agent_0_fantasy_replay, 'w') as json_file:
    #         json.dump(agent_dqn.best_traj[0], json_file, indent=4)
    #     print(f"Trajectories saved to {agent_0_fantasy_replay}")
    # except Exception as e:
    #     print(f"Failed to save trajectories: {e}")
    #
    # try:
    #     with open(agent_1_fantasy_replay, 'w') as json_file:
    #         json.dump(agent_dqn.best_traj[1], json_file, indent=4)
    #     print(f"Trajectories saved to {agent_1_fantasy_replay}")
    # except Exception as e:
    #     print(f"Failed to save trajectories: {e}")

    # agent_dqn.replay(agent_dqn.best_traj[0])
    # agent_dqn.replay(agent_dqn.best_traj[1])


