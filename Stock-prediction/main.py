import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

# Load the dataset
df = pd.read_csv("NVIDIA_STOCK.csv")  # Replace with the actual filename
df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values(by="Date")
data = df["Adj Close"].values  # Use the 'Adj Close' column for prediction

# Use Open, High, Low, Close columns for prediction
data = df[["Open", "High", "Low", "Close"]].values

# Normalize the data
scaler = MinMaxScaler(feature_range=(0, 1))
data = scaler.fit_transform(data)

# Create sequences for multi-output prediction
def create_sequences(data, seq_length):
    sequences = []
    labels = []
    for i in range(len(data) - seq_length):
        seq = data[i: i + seq_length]
        label = data[i + seq_length]
        sequences.append(seq)
        labels.append(label)
    return np.array(sequences), np.array(labels)

sequence_length = 50  # Number of previous time steps to use
X, y = create_sequences(data, sequence_length)

# Convert data to PyTorch tensors
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32)

# Create a PyTorch dataset and dataloader
class MultiOutputDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = sequences
        self.labels = labels

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]

dataset = MultiOutputDataset(X, y)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Define the GRU-based model for multi-output regression
class MultiOutputGRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(MultiOutputGRUModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.gru(x, h0)
        out = self.fc(out[:, -1, :])  # Use the last hidden state for prediction
        return out

# Model hyperparameters
input_size = 4  # Open, High, Low, Close
hidden_size = 64
num_layers = 2
output_size = 4  # Predict Open, High, Low, Close
learning_rate = 0.001
num_epochs = 50

# Initialize the model, loss function, and optimizer
model = MultiOutputGRUModel(input_size, hidden_size, num_layers, output_size)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Training the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    for sequences, labels in dataloader:
        sequences, labels = sequences.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(sequences)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    train_loss /= len(dataloader)
    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {train_loss:.4f}")

# Make predictions for the entire dataset
model.eval()
with torch.no_grad():
    X = X.to(device)
    predictions = model(X).cpu().numpy()

# Rescale the predictions back to the original scale
predictions = scaler.inverse_transform(predictions)
actual = scaler.inverse_transform(y.numpy())

# Add predictions to the original DataFrame
df = df.iloc[sequence_length:].reset_index(drop=True)
df["Predicted_Open"] = predictions[:, 0]
df["Predicted_High"] = predictions[:, 1]
df["Predicted_Low"] = predictions[:, 2]
df["Predicted_Close"] = predictions[:, 3]

# Plot candlestick chart with actual and predicted values
def plot_candlestick(df, title):
    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot actual candlestick chart
    for i in range(len(df)):
        color = "green" if df["Close"].iloc[i] >= df["Open"].iloc[i] else "red"
        ax.plot(
            [i, i],
            [df["Low"].iloc[i], df["High"].iloc[i]],
            color="black",
            linewidth=0.5,
        )
        ax.plot(
            [i, i],
            [df["Open"].iloc[i], df["Close"].iloc[i]],
            color=color,
            linewidth=3,
        )

    # Overlay predicted candlestick chart
    for i in range(len(df)):
        color = "blue" if df["Predicted_Close"].iloc[i] >= df["Predicted_Open"].iloc[i] else "orange"
        ax.plot(
            [i + 0.2, i + 0.2],
            [df["Predicted_Low"].iloc[i], df["Predicted_High"].iloc[i]],
            color="black",
            linewidth=0.5,
        )
        ax.plot(
            [i + 0.2, i + 0.2],
            [df["Predicted_Open"].iloc[i], df["Predicted_Close"].iloc[i]],
            color=color,
            linewidth=3,
        )

    ax.set_title(title)
    ax.set_xlabel("Time")
    ax.set_ylabel("Price")
    plt.show()

plot_candlestick(df, "Actual vs Predicted Candlestick Chart")