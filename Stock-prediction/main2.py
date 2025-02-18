# Import required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from keras import Sequential
from keras.api.optimizers import Adam
from keras.api.callbacks import ReduceLROnPlateau
from keras.api.layers import Dense, LSTM, Dropout, Bidirectional
from sklearn.metrics import mean_absolute_error, mean_squared_error
import math

# Load the dataset
file_path = "Ali_Baba_Stock_Data.csv"  # Change this to your actual file path
df = pd.read_csv(file_path, parse_dates=['Date'], index_col='Date')

# Use relevant features (Multivariate)
features = ['Adj Close', 'Open', 'High', 'Low', 'Volume']
df = df[features]

# Normalize the dataset
scaler = MinMaxScaler(feature_range=(0,1))
df_scaled = scaler.fit_transform(df)

# Splitting dataset (90% train, 10% test)
train_size = int(len(df_scaled) * 0.90)
train_data, test_data = df_scaled[:train_size], df_scaled[train_size:]

# Function to create sequences
def create_sequences(data, seq_length=100):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length, 0])  # Predicting 'Adj Close' only
    return np.array(X), np.array(y)

sequence_length = 100
X_train, y_train = create_sequences(train_data, sequence_length)
X_test, y_test = create_sequences(test_data, sequence_length)

# Reshape for LSTM input
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], X_train.shape[2]))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], X_test.shape[2]))

# Build an optimized LSTM model
model = Sequential([
    Bidirectional(LSTM(units=100, return_sequences=True, input_shape=(sequence_length, len(features)))),
    Dropout(0.2),
    
    Bidirectional(LSTM(units=100, return_sequences=True)),
    Dropout(0.2),
    
    Bidirectional(LSTM(units=50, return_sequences=False)),
    Dropout(0.2),
    
    Dense(units=25),
    Dense(units=1)  # Predicting 'Adj Close'
])

# Compile model with Adam optimizer and learning rate scheduling
optimizer = Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='mean_squared_error')

# Learning rate scheduler
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', patience=5, factor=0.5, verbose=1)

# Train the model
model.fit(X_train, y_train, batch_size=32, epochs=50, validation_data=(X_test, y_test), callbacks=[lr_scheduler])

# Predictions
predictions = model.predict(X_test)
predictions = scaler.inverse_transform(np.hstack((predictions, np.zeros((predictions.shape[0], len(features)-1)))))[:,0]

# Convert actual values back to original scale
y_test_actual = scaler.inverse_transform(np.hstack((y_test.reshape(-1,1), np.zeros((y_test.shape[0], len(features)-1)))))[:,0]

# Calculate RMSE & MAE
rmse = math.sqrt(mean_squared_error(y_test_actual, predictions))
mae = mean_absolute_error(y_test_actual, predictions)
print(f'Optimized RMSE: {rmse}, Optimized MAE: {mae}')

# Plot actual vs predicted stock prices
plt.figure(figsize=(12,6))
plt.plot(df.index[train_size+sequence_length:], y_test_actual, label="Actual Price", color='blue')
plt.plot(df.index[train_size+sequence_length:], predictions, label="Predicted Price", color='red')
plt.title('Alibaba Stock Price Prediction (Optimized)')
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.legend()
plt.show()