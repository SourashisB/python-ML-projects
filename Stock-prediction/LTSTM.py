import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from keras.api.models import Sequential
from keras.api.layers import Dense, LSTM, Dropout
import math

# Load the stock data
def load_data(file_path):
    df = pd.read_csv(file_path)
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    return df

# Prepare the data for LSTM
def prepare_data(data, feature='Adj Close', look_back=60):
    # Use only the selected feature for prediction
    dataset = data[feature].values.reshape(-1, 1)
    
    # Normalize the data to the range (0, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(dataset)
    
    # Create sequences of data for training
    X, y = [], []
    for i in range(len(scaled_data) - look_back):
        X.append(scaled_data[i:i+look_back, 0])
        y.append(scaled_data[i+look_back, 0])
    
    X, y = np.array(X), np.array(y)
    
    # Reshape input to be [samples, time steps, features]
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    
    return X, y, scaler, dataset

# Split data into training and testing sets
def split_data(X, y, train_size=0.9):
    split_idx = int(len(X) * train_size)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    return X_train, y_train, X_test, y_test, split_idx

# Build and train the LSTM model
def build_model(X_train, y_train, X_test, y_test):
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(units=25))
    model.add(Dense(units=1))
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    # Train the model
    history = model.fit(
        X_train, y_train,
        epochs=50,
        batch_size=32,
        validation_data=(X_test, y_test),
        verbose=1
    )
    
    return model, history

# Make predictions and evaluate the model
def make_predictions(model, X_test, y_test, scaler):
    # Predict on test data
    predictions = model.predict(X_test)
    
    # Invert predictions to original scale
    predictions = scaler.inverse_transform(predictions)
    y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))
    
    # Calculate error metrics
    rmse = math.sqrt(mean_squared_error(y_test_actual, predictions))
    mae = mean_absolute_error(y_test_actual, predictions)
    
    print(f"Test RMSE: {rmse:.2f}")
    print(f"Test MAE: {mae:.2f}")
    
    return predictions, y_test_actual

# Visualize the results
def visualize_results(data, test_predictions, test_actual, feature='Adj Close', look_back=60, split_idx=0):
    # Create a DataFrame for the actual stock prices
    actual_data = data[feature]
    
    # Create a plot
    plt.figure(figsize=(16, 8))
    
    # Plot the entire historical data
    plt.plot(data.index, actual_data, label='Historical Stock Price', color='gray', alpha=0.6)
    
    # Calculate test data starting point
    # We need to add look_back to split_idx to get the actual start of test predictions
    # since our first prediction corresponds to the (split_idx + look_back) point
    test_start_idx = split_idx + look_back
    
    # Get the dates for test predictions
    test_dates = data.index[test_start_idx:test_start_idx+len(test_predictions)]
    
    # Plot actual test values
    plt.plot(test_dates, test_actual, label='Actual Stock Price', color='blue', linewidth=2)
    
    # Plot predicted test values
    plt.plot(test_dates, test_predictions, label='Predicted Stock Price', color='red', linewidth=2, linestyle='--')
    
    # Add a vertical line to separate training and testing data
    train_end_date = data.index[split_idx]
    plt.axvline(x=train_end_date, color='green', linestyle='--', label='Train-Test Split')
    
    plt.title(f'Stock Price Prediction using LSTM - {feature}')
    plt.xlabel('Date')
    plt.ylabel(f'Stock Price ({feature})')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Visualize the training history
def plot_training_history(history):
    plt.figure(figsize=(12, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()

def main(file_path, feature='Adj Close', look_back=60, train_size=0.9):
    # Load and prepare the data
    data = load_data(file_path)
    print(f"Data loaded: {len(data)} entries")
    
    # Prepare sequences for LSTM
    X, y, scaler, dataset = prepare_data(data, feature, look_back)
    
    # Split the data
    X_train, y_train, X_test, y_test, split_idx = split_data(X, y, train_size)
    print(f"Training data shape: {X_train.shape}")
    print(f"Testing data shape: {X_test.shape}")
    
    # Build and train the model
    model, history = build_model(X_train, y_train, X_test, y_test)
    
    # Make predictions
    predictions, actual = make_predictions(model, X_test, y_test, scaler)
    
    # Visualize the results
    visualize_results(data, predictions, actual, feature, look_back, split_idx)
    plot_training_history(history)
    
# Example usage
if __name__ == "__main__":
    # Replace with your actual file path
    file_path = "Ali_Baba_Stock_Data.csv"  
    main(file_path, feature='Adj Close', look_back=60, train_size=0.9)