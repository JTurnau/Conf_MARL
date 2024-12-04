import json
import matplotlib.pyplot as plt

# Load the JSON file
file_path = "16x16_training_metrics_0.25_cf.json"  # Replace with the actual file path
with open(file_path, "r") as file:
    data = json.load(file)

# Extract data
epochs = [entry["epoch"] for entry in data]

# Rewards
train_rewards = [entry["train_reward"] for entry in data]
agent_0_rewards = [sum(entry["fantasy_rewards"]["agent_0"]) for entry in data]
agent_1_rewards = [sum(entry["fantasy_rewards"]["agent_1"]) for entry in data]

# Losses
agent_0_lstm_losses = [entry["fantasy_lstm_loss"]["agent_0"] for entry in data]
agent_1_lstm_losses = [entry["fantasy_lstm_loss"]["agent_1"] for entry in data]
agent_0_conf_losses = [entry["fantasy_conf_loss"]["agent_0"] for entry in data]
agent_1_conf_losses = [entry["fantasy_conf_loss"]["agent_1"] for entry in data]

# Plot Train Reward
plt.figure(figsize=(8, 5))
plt.plot(epochs, train_rewards, marker='o', color='blue')
plt.title("Train Reward over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Train Reward")
plt.grid(True)
plt.show()

# Plot Agent 0 Fantasy Reward
plt.figure(figsize=(8, 5))
plt.plot(epochs, agent_0_rewards, marker='s', color='green')
plt.title("Agent 0 Fantasy Reward over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Agent 0 Fantasy Reward")
plt.grid(True)
plt.show()

# Plot Agent 1 Fantasy Reward
plt.figure(figsize=(8, 5))
plt.plot(epochs, agent_1_rewards, marker='^', color='orange')
plt.title("Agent 1 Fantasy Reward over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Agent 1 Fantasy Reward")
plt.grid(True)
plt.show()

# Plot Agent 0 LSTM Loss
plt.figure(figsize=(8, 5))
plt.plot(epochs, agent_0_lstm_losses, marker='o', color='purple')
plt.title("Agent 0 LSTM Loss over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Agent 0 LSTM Loss")
plt.grid(True)
plt.show()

# Plot Agent 1 LSTM Loss
plt.figure(figsize=(8, 5))
plt.plot(epochs, agent_1_lstm_losses, marker='s', color='cyan')
plt.title("Agent 1 LSTM Loss over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Agent 1 LSTM Loss")
plt.grid(True)
plt.show()

# Plot Agent 0 Confidence Loss
plt.figure(figsize=(8, 5))
plt.plot(epochs, agent_0_conf_losses, marker='o', color='red')
plt.title("Agent 0 Confidence Loss over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Agent 0 Confidence Loss")
plt.grid(True)
plt.show()

# Plot Agent 1 Confidence Loss
plt.figure(figsize=(8, 5))
plt.plot(epochs, agent_1_conf_losses, marker='^', color='brown')
plt.title("Agent 1 Confidence Loss over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Agent 1 Confidence Loss")
plt.grid(True)
plt.show()
