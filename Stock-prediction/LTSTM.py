import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from keras.api.models import Sequential
from keras.api.layers import Dense, LSTM, Dropout
import math
from keras.api.callbacks import EarlyStopping

# Load the stock data
def load_data(file_path):
    df = pd.read_csv(file_path)
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    return df

# Custom implementation of technical indicators
def add_technical_indicators(data):
    df = data.copy()
    
    # Simple Moving Averages
    df['MA5'] = df['Close'].rolling(window=5).mean()
    df['MA10'] = df['Close'].rolling(window=10).mean()
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['MA50'] = df['Close'].rolling(window=50).mean()
    
    # Exponential Moving Averages
    df['EMA12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['EMA26'] = df['Close'].ewm(span=26, adjust=False).mean()
    
    # MACD (Moving Average Convergence Divergence)
    df['MACD'] = df['EMA12'] - df['EMA26']
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
    
    # RSI (Relative Strength Index)
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    
    # For smoother calculation after the first 14 periods
    for i in range(14, len(gain)):
        avg_gain.iloc[i] = (avg_gain.iloc[i-1] * 13 + gain.iloc[i]) / 14
        avg_loss.iloc[i] = (avg_loss.iloc[i-1] * 13 + loss.iloc[i]) / 14
    
    rs = avg_gain / avg_loss
    df['RSI14'] = 100 - (100 / (1 + rs))
    
    # Bollinger Bands
    df['BB_Middle'] = df['Close'].rolling(window=20).mean()
    df['BB_Std'] = df['Close'].rolling(window=20).std()
    df['BB_Upper'] = df['BB_Middle'] + (df['BB_Std'] * 2)
    df['BB_Lower'] = df['BB_Middle'] - (df['BB_Std'] * 2)
    df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / df['BB_Middle']
    
    # Stochastic Oscillator
    low_min = df['Low'].rolling(window=14).min()
    high_max = df['High'].rolling(window=14).max()
    df['STOCH_K'] = 100 * ((df['Close'] - low_min) / (high_max - low_min))
    df['STOCH_D'] = df['STOCH_K'].rolling(window=3).mean()
    
    # Average True Range (ATR)
    high_low = df['High'] - df['Low']
    high_close = abs(df['High'] - df['Close'].shift())
    low_close = abs(df['Low'] - df['Close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    df['ATR'] = true_range.rolling(window=14).mean()
    
    # On-Balance Volume (OBV)
    df['OBV'] = 0
    df.loc[df['Close'] > df['Close'].shift(1), 'OBV'] = df['Volume']
    df.loc[df['Close'] < df['Close'].shift(1), 'OBV'] = -df['Volume']
    df['OBV'] = df['OBV'].cumsum()
    
    # Commodity Channel Index (CCI)
    tp = (df['High'] + df['Low'] + df['Close']) / 3
    ma_tp = tp.rolling(window=20).mean()
    md_tp = tp.rolling(window=20).apply(lambda x: np.mean(np.abs(x - np.mean(x))))
    df['CCI'] = (tp - ma_tp) / (0.015 * md_tp)
    
    # Price-Volume Trend (PVT)
    df['PVT'] = ((df['Close'] - df['Close'].shift(1)) / df['Close'].shift(1)) * df['Volume']
    df['PVT'] = df['PVT'].fillna(0).cumsum()
    
    # Rate of Change
    df['ROC'] = ((df['Close'] - df['Close'].shift(10)) / df['Close'].shift(10)) * 100
    
    # Price relative to moving averages
    df['Price_to_MA50'] = df['Close'] / df['MA50'] - 1
    df['Price_to_MA20'] = df['Close'] / df['MA20'] - 1
    
    # Volatility (standard deviation of returns)
    df['Volatility'] = df['Close'].pct_change().rolling(window=20).std()
    
    # Momentum indicators
    df['Momentum'] = df['Close'] - df['Close'].shift(10)
    
    # Williams %R
    df['Williams_%R'] = -100 * (high_max - df['Close']) / (high_max - low_min)
    
    # Average Directional Index (ADX)
    # First, calculate the +DI and -DI
    plus_dm = df['High'].diff()
    minus_dm = df['Low'].diff()
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm > 0] = 0
    minus_dm = abs(minus_dm)
    
    # When current high - previous high > previous low - current low
    condition = (df['High'] - df['High'].shift()) > (df['Low'].shift() - df['Low'])
    plus_dm[~condition] = 0
    
    # When previous low - current low > current high - previous high
    condition = (df['Low'].shift() - df['Low']) > (df['High'] - df['High'].shift())
    minus_dm[~condition] = 0
    
    # Calculate the smoothed moving averages of TR, +DM, -DM
    tr_smooth = true_range.rolling(window=14).mean()
    plus_dm_smooth = plus_dm.rolling(window=14).mean()
    minus_dm_smooth = minus_dm.rolling(window=14).mean()
    
    # Calculate the +DI and -DI
    plus_di = 100 * (plus_dm_smooth / tr_smooth)
    minus_di = 100 * (minus_dm_smooth / tr_smooth)
    
    # Calculate the DX
    dx = 100 * (abs(plus_di - minus_di) / (plus_di + minus_di))
    
    # Calculate the ADX
    df['ADX'] = dx.rolling(window=14).mean()
    df['Plus_DI'] = plus_di
    df['Minus_DI'] = minus_di
    
    # Keltner Channels
    df['KC_Middle'] = df['Close'].rolling(window=20).mean()
    df['KC_Width'] = df['ATR'] * 2
    df['KC_Upper'] = df['KC_Middle'] + df['KC_Width']
    df['KC_Lower'] = df['KC_Middle'] - df['KC_Width']
    
    # Percentage Price Oscillator (PPO)
    df['PPO'] = ((df['EMA12'] - df['EMA26']) / df['EMA26']) * 100
    df['PPO_Signal'] = df['PPO'].ewm(span=9, adjust=False).mean()
    df['PPO_Hist'] = df['PPO'] - df['PPO_Signal']
    
    # Parabolic SAR (simplified version)
    df['SAR'] = df['Close'].shift(1)
    acceleration_factor = 0.02
    max_acceleration = 0.2
    
    # Money Flow Index (MFI)
    typical_price = (df['High'] + df['Low'] + df['Close']) / 3
    money_flow = typical_price * df['Volume']
    
    positive_flow = money_flow.where(typical_price > typical_price.shift(1), 0)
    negative_flow = money_flow.where(typical_price < typical_price.shift(1), 0)
    
    positive_mf = positive_flow.rolling(window=14).sum()
    negative_mf = negative_flow.rolling(window=14).sum()
    
    money_ratio = positive_mf / negative_mf
    df['MFI'] = 100 - (100 / (1 + money_ratio))
    
    # Chaikin Money Flow (CMF)
    mf_multiplier = ((df['Close'] - df['Low']) - (df['High'] - df['Close'])) / (df['High'] - df['Low'])
    mf_volume = mf_multiplier * df['Volume']
    df['CMF'] = mf_volume.rolling(window=20).sum() / df['Volume'].rolling(window=20).sum()
    
    # Fibonacci Retracement levels (based on the last 30 days)
    df['Fib_38.2%'] = df['Close'].rolling(window=30).max() - (0.382 * (df['Close'].rolling(window=30).max() - df['Close'].rolling(window=30).min()))
    df['Fib_50%'] = df['Close'].rolling(window=30).max() - (0.5 * (df['Close'].rolling(window=30).max() - df['Close'].rolling(window=30).min()))
    df['Fib_61.8%'] = df['Close'].rolling(window=30).max() - (0.618 * (df['Close'].rolling(window=30).max() - df['Close'].rolling(window=30).min()))
    
    # Detrended Price Oscillator (DPO)
    df['DPO'] = df['Close'] - df['Close'].rolling(window=20).mean().shift(10)
    
    # Ease of Movement (EOM)
    distance_moved = ((df['High'] + df['Low']) / 2) - ((df['High'].shift(1) + df['Low'].shift(1)) / 2)
    box_ratio = (df['Volume'] / 100000000) / (df['High'] - df['Low'])
    df['EOM'] = distance_moved / box_ratio
    df['EOM_SMA'] = df['EOM'].rolling(window=14).mean()
    
    # Mass Index
    range_ema1 = (df['High'] - df['Low']).ewm(span=9, adjust=False).mean()
    range_ema2 = range_ema1.ewm(span=9, adjust=False).mean()
    ratio = range_ema1 / range_ema2
    df['Mass_Index'] = ratio.rolling(window=25).sum()
    
    # Drop rows with NaN values
    df = df.dropna()
    
    return df

# Prepare the data for LSTM with technical indicators
def prepare_data_with_indicators(data, target_feature='Adj Close', look_back=60):
    # Select all features for the model input
    # We exclude the Date column which is now the index
    features = data.columns.tolist()
    
    # Make a copy of the data
    df = data.copy()
    
    # The target variable is what we want to predict (usually Adj Close)
    target = df[target_feature].values.reshape(-1, 1)
    
    # Scale all features
    scaler_features = MinMaxScaler(feature_range=(0, 1))
    scaled_features = scaler_features.fit_transform(df)
    
    # Separate scaler for the target variable for easy inverse transform later
    scaler_target = MinMaxScaler(feature_range=(0, 1))
    scaled_target = scaler_target.fit_transform(target)
    
    # Create sequences for training
    X, y = [], []
    for i in range(len(scaled_features) - look_back):
        X.append(scaled_features[i:i+look_back, :])  # All features
        y.append(scaled_target[i+look_back, 0])      # Target variable
    
    X, y = np.array(X), np.array(y)
    
    # No need to reshape X as it already has three dimensions: [samples, time steps, features]
    
    return X, y, scaler_target, target, features

# Split data into training and testing sets
def split_data(X, y, train_size=0.9):
    split_idx = int(len(X) * train_size)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    return X_train, y_train, X_test, y_test, split_idx

# Build and train the LSTM model with technical indicators
def build_model_with_indicators(X_train, y_train, X_test, y_test):
    # Number of features is the last dimension of X_train
    n_features = X_train.shape[2]
    
    model = Sequential()
    model.add(LSTM(units=100, return_sequences=True, input_shape=(X_train.shape[1], n_features)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=100, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(units=25))
    model.add(Dense(units=1))
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    # Early stopping callback
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    
    # Train the model
    history = model.fit(
        X_train, y_train,
        epochs=100,
        batch_size=32,
        validation_data=(X_test, y_test),
        callbacks=[early_stop],
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
    
    plt.title(f'Stock Price Prediction using LSTM with Technical Indicators - {feature}')
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

# Feature importance analysis
def analyze_feature_importance(model, features):
    # Get the weights of the first LSTM layer
    lstm_weights = model.layers[0].get_weights()[0]
    
    # Calculate feature importance scores (simplified)
    # Sum the absolute values of weights for each feature
    feature_importance = np.sum(np.abs(lstm_weights), axis=(0, 1))
    
    # Create a dictionary mapping features to their importance scores
    feature_importance_dict = dict(zip(features, feature_importance))
    
    # Sort the features by importance
    sorted_features = sorted(feature_importance_dict.items(), key=lambda x: x[1], reverse=True)
    
    # Visualize the feature importance
    plt.figure(figsize=(12, 8))
    features_name = [f[0] for f in sorted_features]
    importance = [f[1] for f in sorted_features]
    
    plt.barh(features_name, importance)
    plt.xlabel('Importance Score')
    plt.title('Feature Importance')
    plt.tight_layout()
    plt.show()
    
    return sorted_features

# Plot selected technical indicators
def plot_technical_indicators(data, n=5):
    plt.figure(figsize=(15, 15))
    
    # Plot 1: Closing Price, MA5, MA10, MA20, MA50
    plt.subplot(5, 1, 1)
    plt.plot(data['Close'], label='Close Price')
    plt.plot(data['MA5'], label='5-day MA')
    plt.plot(data['MA10'], label='10-day MA')
    plt.plot(data['MA20'], label='20-day MA')
    plt.plot(data['MA50'], label='50-day MA')
    plt.title('Moving Averages')
    plt.legend()
    
    # Plot 2: MACD
    plt.subplot(5, 1, 2)
    plt.plot(data['MACD'], label='MACD')
    plt.plot(data['MACD_Signal'], label='Signal Line')
    plt.bar(data.index, data['MACD_Hist'], label='Histogram')
    plt.title('MACD')
    plt.legend()
    
    # Plot 3: RSI
    plt.subplot(5, 1, 3)
    plt.plot(data['RSI14'], label='RSI 14')
    plt.axhline(y=70, color='r', linestyle='-', alpha=0.3)
    plt.axhline(y=30, color='g', linestyle='-', alpha=0.3)
    plt.title('RSI')
    plt.legend()
    
    # Plot 4: Bollinger Bands
    plt.subplot(5, 1, 4)
    plt.plot(data['Close'], label='Close Price')
    plt.plot(data['BB_Upper'], label='Upper BB')
    plt.plot(data['BB_Middle'], label='Middle BB')
    plt.plot(data['BB_Lower'], label='Lower BB')
    plt.fill_between(data.index, data['BB_Upper'], data['BB_Lower'], alpha=0.1)
    plt.title('Bollinger Bands')
    plt.legend()
    
    # Plot 5: Stochastic Oscillator
    plt.subplot(5, 1, 5)
    plt.plot(data['STOCH_K'], label='%K')
    plt.plot(data['STOCH_D'], label='%D')
    plt.axhline(y=80, color='r', linestyle='-', alpha=0.3)
    plt.axhline(y=20, color='g', linestyle='-', alpha=0.3)
    plt.title('Stochastic Oscillator')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

# Calculate correlation between features and target
def analyze_feature_correlation(data, target_feature='Adj Close', top_n=15):
    # Calculate correlation with target feature
    correlation = data.corrwith(data[target_feature]).sort_values(ascending=False)
    
    # Visualize top correlations
    plt.figure(figsize=(12, 8))
    top_corr = correlation.drop(target_feature).abs().nlargest(top_n)
    plt.barh(top_corr.index, top_corr.values)
    plt.xlabel('Absolute Correlation')
    plt.title(f'Feature Correlation with {target_feature}')
    plt.tight_layout()
    plt.show()
    
    return correlation

def main(file_path, target_feature='Adj Close', look_back=60, train_size=0.9):
    # Load data
    data = load_data(file_path)
    print(f"Data loaded: {len(data)} entries")
    
    # Add technical indicators
    data_with_indicators = add_technical_indicators(data)
    print(f"Data with technical indicators: {len(data_with_indicators)} entries")
    print(f"Features: {data_with_indicators.columns.tolist()}")
    
    # Plot some technical indicators
    plot_technical_indicators(data_with_indicators)
    
    # Analyze feature correlation
    correlation = analyze_feature_correlation(data_with_indicators, target_feature)
    
    # Prepare sequences for LSTM with all features
    X, y, scaler_target, target, features = prepare_data_with_indicators(data_with_indicators, target_feature, look_back)
    
    # Split the data
    X_train, y_train, X_test, y_test, split_idx = split_data(X, y, train_size)
    print(f"Training data shape: {X_train.shape}")
    print(f"Testing data shape: {X_test.shape}")
    
    # Build and train the model
    model, history = build_model_with_indicators(X_train, y_train, X_test, y_test)
    
    # Make predictions
    predictions, actual = make_predictions(model, X_test, y_test, scaler_target)
    
    # Visualize the results
    visualize_results(data, predictions, actual, target_feature, look_back, split_idx)
    plot_training_history(history)
    
    # Analyze feature importance
    feature_importance = analyze_feature_importance(model, features)
    print("Top 10 most important features:")
    for feature, importance in feature_importance[:10]:
        print(f"{feature}: {importance:.4f}")
    
# Example usage
if __name__ == "__main__":
    # Replace with your actual file path
    file_path = "Ali_Baba_Stock_Data.csv"  
    main(file_path, target_feature='Adj Close', look_back=60, train_size=0.9)