import json
import matplotlib.pyplot as plt
import numpy as np

# List of file paths to process
file_paths_8x8 = [
    "evaluation_results/8x8_training_metrics_IDQN.json",
    "evaluation_results/8x8_training_metrics_IDQN_cf_0.25.json",
    "evaluation_results/8x8_training_metrics_IDQN_cf_0.5.json",
    "evaluation_results/8x8_training_metrics_IDQN_cf_0.75.json"
]

file_paths_5x5 = [
    "evaluation_results/5x5_training_metrics_IDQN.json",
    "evaluation_results/5x5_training_metrics_IDQN_cf_0.25.json",
    "evaluation_results/5x5_training_metrics_IDQN_cf_0.5.json",
    "evaluation_results/5x5_training_metrics_IDQN_cf_0.75.json"
]

# Custom labels for the files
custom_labels_8x8 = [
    "IDQN",
    "IDQN with CF 0.25",
    "IDQN with CF 0.5",
    "IDQN with CF 0.75"
]

custom_labels_5x5 = [
    "IDQN",
    "IDQN with CF 0.25",
    "IDQN with CF 0.5",
    "IDQN with CF 0.75"
]

# Function to apply simple moving average for smoothing
def smooth(data, window_size=5):
    return [sum(data[max(0, i - window_size + 1):i + 1]) / min(i + 1, window_size) for i in range(len(data))]

# Initialize lists to store average loss data for both 8x8 and 5x5
fantasy_lstm_losses_all_8x8 = []
fantasy_conf_losses_all_8x8 = []
fantasy_lstm_losses_all_5x5 = []
fantasy_conf_losses_all_5x5 = []
epochs_all_8x8 = None
epochs_all_5x5 = None

# Collect data for Fantasy Loss and Confidence Loss, excluding the first file
min_epochs_length_8x8 = float('inf')

for i, file_path in enumerate(file_paths_8x8[1:]):  # Skip the first file
    with open(file_path, "r") as file:
        data = json.load(file)

    epochs = [entry["epoch"] for entry in data]
    min_epochs_length_8x8 = min(min_epochs_length_8x8, len(epochs))  # Track the minimum length

min_epochs_length_5x5 = float('inf')

for i, file_path in enumerate(file_paths_5x5[1:]):  # Skip the first file
    with open(file_path, "r") as file:
        data = json.load(file)

    epochs = [entry["epoch"] for entry in data]
    min_epochs_length_5x5 = min(min_epochs_length_5x5, len(epochs))  # Track the minimum length

# Collect data for Fantasy Loss and Confidence Loss for 8x8 and 5x5
for i, file_path in enumerate(file_paths_8x8[1:]):  # Skip the first file in 8x8
    with open(file_path, "r") as file:
        data = json.load(file)

    epochs = [entry["epoch"] for entry in data]
    fantasy_lstm_losses = []
    fantasy_conf_losses = []

    # Collect fantasy losses for files that have them
    if "fantasy_lstm_loss" in data[0]:
        fantasy_lstm_losses = [sum(entry["fantasy_lstm_loss"].values()) for entry in data]
    if "fantasy_conf_loss" in data[0]:
        fantasy_conf_losses = [sum(entry["fantasy_conf_loss"].values()) for entry in data]

    # Trim the data to the minimum length across all files
    fantasy_lstm_losses_all_8x8.append(fantasy_lstm_losses[:min_epochs_length_8x8])
    fantasy_conf_losses_all_8x8.append(fantasy_conf_losses[:min_epochs_length_8x8])

    if epochs_all_8x8 is None:
        epochs_all_8x8 = epochs[:min_epochs_length_8x8]

for i, file_path in enumerate(file_paths_5x5[1:]):  # Skip the first file in 5x5
    with open(file_path, "r") as file:
        data = json.load(file)

    epochs = [entry["epoch"] for entry in data]
    fantasy_lstm_losses = []
    fantasy_conf_losses = []

    # Collect fantasy losses for files that have them
    if "fantasy_lstm_loss" in data[0]:
        fantasy_lstm_losses = [sum(entry["fantasy_lstm_loss"].values()) for entry in data]
    if "fantasy_conf_loss" in data[0]:
        fantasy_conf_losses = [sum(entry["fantasy_conf_loss"].values()) for entry in data]

    # Trim the data to the minimum length across all files
    fantasy_lstm_losses_all_5x5.append(fantasy_lstm_losses[:min_epochs_length_5x5])
    fantasy_conf_losses_all_5x5.append(fantasy_conf_losses[:min_epochs_length_5x5])

    if epochs_all_5x5 is None:
        epochs_all_5x5 = epochs[:min_epochs_length_5x5]

# Average the losses across all files (excluding the first one) for both 8x8 and 5x5
avg_fantasy_lstm_losses_8x8 = np.mean(fantasy_lstm_losses_all_8x8, axis=0)
avg_fantasy_conf_losses_8x8 = np.mean(fantasy_conf_losses_all_8x8, axis=0)

avg_fantasy_lstm_losses_5x5 = np.mean(fantasy_lstm_losses_all_5x5, axis=0)
avg_fantasy_conf_losses_5x5 = np.mean(fantasy_conf_losses_all_5x5, axis=0)

# Smooth the averaged losses for both 8x8 and 5x5
smoothed_avg_fantasy_lstm_losses_8x8 = smooth(avg_fantasy_lstm_losses_8x8)
smoothed_avg_fantasy_conf_losses_8x8 = smooth(avg_fantasy_conf_losses_8x8)

smoothed_avg_fantasy_lstm_losses_5x5 = smooth(avg_fantasy_lstm_losses_5x5)
smoothed_avg_fantasy_conf_losses_5x5 = smooth(avg_fantasy_conf_losses_5x5)

# Plot the average Fantasy Loss for both 8x8 and 5x5 in one figure
plt.figure(figsize=(10, 6))
# Plot 8x8 fantasy loss
plt.plot(epochs_all_8x8, smoothed_avg_fantasy_lstm_losses_8x8, label="Avg Fantasy LSTM Loss 8x8", color='b')
# Plot 5x5 fantasy loss
plt.plot(epochs_all_5x5, smoothed_avg_fantasy_lstm_losses_5x5, label="Avg Fantasy LSTM Loss 5x5", color='g')

# Customize the plot
plt.title("Average Fantasy Loss over Epochs (8x8 and 5x5)")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# Plot the average Confidence Loss for both 8x8 and 5x5 in another figure
plt.figure(figsize=(10, 6))
# Plot 8x8 confidence loss
plt.plot(epochs_all_8x8, smoothed_avg_fantasy_conf_losses_8x8, label="Avg Confidence Loss 8x8", color='r')
# Plot 5x5 confidence loss
plt.plot(epochs_all_5x5, smoothed_avg_fantasy_conf_losses_5x5, label="Avg Confidence Loss 5x5", color='y')

# Customize the plot
plt.title("Average Confidence Loss over Epochs (8x8 and 5x5)")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
