import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# 1. Load and preprocess the data
class RevenueDataset(Dataset):
    def __init__(self, features, targets):
        self.features = features
        self.targets = targets

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]


def preprocess_data(csv_file):
    # Load the dataset
    data = pd.read_csv(csv_file)
    
    # Check for missing values and drop rows with NaN values
    data = data.dropna()
    
    # Separate features and target variable
    X = data.drop(columns=['Daily_Revenue']).values
    y = data['Daily_Revenue'].values
    
    # Standardize the features using StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y, scaler


# 2. Define the PyTorch neural network
class RevenuePredictor(nn.Module):
    def __init__(self, input_dim):
        super(RevenuePredictor, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.network(x)


# 3. Training function
def train_model(model, train_loader, criterion, optimizer, num_epochs):
    train_losses = []
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        
        for features, targets in train_loader:
            features, targets = features.float(), targets.float().view(-1, 1)
            
            # Forward pass
            outputs = model(features)
            loss = criterion(outputs, targets)
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        train_losses.append(epoch_loss / len(train_loader))
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")
    
    return train_losses


# 4. Evaluation function
def evaluate_model(model, test_loader, criterion):
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for features, targets in test_loader:
            features, targets = features.float(), targets.float().view(-1, 1)
            outputs = model(features)
            loss = criterion(outputs, targets)
            test_loss += loss.item()
    
    return test_loss / len(test_loader)


# 5. Main function
def main():
    # File path to your CSV file
    csv_file = 'coffee_shop_revenue.csv'
    
    # Preprocess the data
    X, y, scaler = preprocess_data(csv_file)
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Convert the data into PyTorch datasets
    train_dataset = RevenueDataset(torch.tensor(X_train), torch.tensor(y_train))
    test_dataset = RevenueDataset(torch.tensor(X_test), torch.tensor(y_test))
    
    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Initialize the neural network, loss function, and optimizer
    input_dim = X_train.shape[1]
    model = RevenuePredictor(input_dim)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Train the model
    num_epochs = 50
    print("Training the model...")
    train_losses = train_model(model, train_loader, criterion, optimizer, num_epochs)
    
    # Plot training loss
    plt.plot(range(num_epochs), train_losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.show()
    
    # Evaluate the model
    print("Evaluating the model...")
    test_loss = evaluate_model(model, test_loader, criterion)
    print(f"Test Loss: {test_loss:.4f}")
    
    # Save the model
    torch.save(model.state_dict(), 'revenue_predictor_model.pth')
    print("Model saved as 'revenue_predictor_model.pth'")
    
    # Load the model (for demonstration)
    loaded_model = RevenuePredictor(input_dim)
    loaded_model.load_state_dict(torch.load('revenue_predictor_model.pth'))
    print("Model loaded successfully!")


if __name__ == '__main__':
    main()