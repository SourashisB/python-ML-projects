import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import joblib

# ------------ CONFIGURATION ------------
SEQ_LENGTH = 30
BATCH_SIZE = 64
EPOCHS = 30
LEARNING_RATE = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TEST_SPLIT = 0.1  # 10% for test

# ------------ DATASET CLASS ------------
class StockDataset(Dataset):
    def __init__(self, data, seq_length):
        self.X = []
        self.y = []
        for i in range(len(data) - seq_length):
            self.X.append(data[i:i+seq_length])
            self.y.append(data[i+seq_length][0])  # Predict Adj Close
        self.X = np.array(self.X)
        self.y = np.array(self.y)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return torch.tensor(self.X[idx], dtype=torch.float32), torch.tensor(self.y[idx], dtype=torch.float32)

# ------------ MODEL DEFINITION ------------
class LSTMRegressor(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out.squeeze()

# ------------ DATA LOADING & PREPROCESSING ------------
df = pd.read_csv('nvidia_stock.csv')
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date')

feature_cols = ['Adj Close', 'Close', 'High', 'Low', 'Open', 'Volume']
data = df[feature_cols].values

scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data)
joblib.dump(scaler, 'scaler.save')

dataset = StockDataset(data_scaled, SEQ_LENGTH)

# Split into train and test
test_size = int(len(dataset) * TEST_SPLIT)
train_size = len(dataset) - test_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size], generator=torch.Generator().manual_seed(42))

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# ------------ MODEL TRAINING ------------
model = LSTMRegressor(input_size=len(feature_cols)).to(DEVICE)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

loss_history = []

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
        optimizer.zero_grad()
        y_pred = model(X_batch)
        loss = criterion(y_pred, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * X_batch.size(0)
    avg_loss = total_loss / len(train_dataset)
    loss_history.append(avg_loss)
    print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {avg_loss:.6f}")

# ------------ SAVE MODEL ------------
torch.save(model.state_dict(), 'nvidia_lstm.pth')
print("Model and scaler saved.")
