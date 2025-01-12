import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import time
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def calculate_technical_indicators(df):
    """Calculate technical indicators manually"""
    
    # SMA
    def calculate_sma(data, window):
        return data.rolling(window=window).mean()
    
    # EMA
    def calculate_ema(data, window):
        return data.ewm(span=window, adjust=False).mean()
    
    # MACD
    def calculate_macd(data, fast=12, slow=26, signal=9):
        fast_ema = calculate_ema(data, fast)
        slow_ema = calculate_ema(data, slow)
        macd_line = fast_ema - slow_ema
        signal_line = calculate_ema(macd_line, signal)
        macd_hist = macd_line - signal_line
        return macd_line, signal_line, macd_hist
    
    # RSI
    def calculate_rsi(data, window=14):
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    # Bollinger Bands
    def calculate_bollinger_bands(data, window=20, num_std=2):
        sma = calculate_sma(data, window)
        std = data.rolling(window=window).std()
        upper_band = sma + (std * num_std)
        lower_band = sma - (std * num_std)
        return upper_band, lower_band, sma
    
    # VWAP
    def calculate_vwap(df):
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        return (typical_price * df['volume']).cumsum() / df['volume'].cumsum()
    
    return calculate_sma, calculate_ema, calculate_macd, calculate_rsi, calculate_bollinger_bands, calculate_vwap

def get_bitcoin_data():
    base_url = "https://api.binance.com/api/v3/klines"
    
    end_time = int(time.time() * 1000)
    start_time = end_time - (30 * 24 * 60 * 60 * 1000)  # 30 days
    
    params = {
        'symbol': 'BTCUSDT',
        'interval': '1h',
        'startTime': start_time,
        'endTime': end_time,
        'limit': 1000
    }
    
    try:
        response = requests.get(base_url, params=params)
        data = response.json()
        
        df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 
                                       'volume', 'close_time', 'quote_asset_volume',
                                       'number_of_trades', 'taker_buy_base_asset_volume',
                                       'taker_buy_quote_asset_volume', 'ignore'])
        
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = df[col].astype(float)
        
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        
        return df
    
    except Exception as e:
        print(f"Error fetching data: {e}")
        return None

def create_features(df):
    df = df.copy()
    
    # Get technical indicator functions
    calculate_sma, calculate_ema, calculate_macd, calculate_rsi, calculate_bollinger_bands, calculate_vwap = calculate_technical_indicators(df)
    
    # Price and returns based features
    df['returns'] = df['close'].pct_change()
    df['log_returns'] = np.log1p(df['returns'])
    
    # Rolling statistics
    windows = [6, 12, 24, 48]
    for window in windows:
        df[f'rolling_mean_{window}h'] = df['close'].rolling(window=window).mean()
        df[f'rolling_std_{window}h'] = df['close'].rolling(window=window).std()
        df[f'rolling_vol_{window}h'] = df['volume'].rolling(window=window).mean()
        df[f'momentum_{window}h'] = df['close'] - df['close'].shift(window)
        df[f'volume_momentum_{window}h'] = df['volume'] - df['volume'].shift(window)
        df[f'return_volatility_{window}h'] = df['returns'].rolling(window=window).std()
    
    # Technical Indicators
    for window in [7, 14, 21]:
        df[f'sma_{window}'] = calculate_sma(df['close'], window)
        df[f'ema_{window}'] = calculate_ema(df['close'], window)
    
    # MACD
    macd_line, signal_line, macd_hist = calculate_macd(df['close'])
    df['macd'] = macd_line
    df['macd_signal'] = signal_line
    df['macd_hist'] = macd_hist
    
    # RSI
    df['rsi'] = calculate_rsi(df['close'])
    
    # Bollinger Bands
    bb_upper, bb_lower, bb_mid = calculate_bollinger_bands(df['close'])
    df['bb_upper'] = bb_upper
    df['bb_lower'] = bb_lower
    df['bb_mid'] = bb_mid
    df['bb_width'] = (bb_upper - bb_lower) / bb_mid
    
    # VWAP
    df['vwap'] = calculate_vwap(df)
    
    # Price channels
    df['high_low_ratio'] = df['high'] / df['low']
    df['close_position'] = (df['close'] - df['low']) / (df['high'] - df['low'])
    
    # Additional features
    df['price_volatility'] = df['high'] / df['low'] - 1
    df['volume_intensity'] = df['volume'] / df['volume'].rolling(window=24).mean()
    
    # Drop NaN values
    df.dropna(inplace=True)
    
    return df

def prepare_data(df, target_hours=1):
    """Prepare data for training with proper alignment of features and target"""
    
    # Create target (future price change)
    df['target'] = df['close'].shift(-target_hours) / df['close'] - 1
    
    # Remove rows with NaN values first
    df.dropna(inplace=True)
    
    # Separate features and target
    features = df.drop(['target', 'close_time', 'quote_asset_volume', 
                       'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 
                       'ignore'], axis=1)
    target = df['target']
    
    # Scale features
    scaler = RobustScaler()
    scaled_features = scaler.fit_transform(features)
    scaled_features = pd.DataFrame(scaled_features, columns=features.columns, index=features.index)
    
    # Ensure features and target have same index
    assert len(scaled_features) == len(target), "Features and target must have same length"
    
    # Split the data
    train_size = int(len(scaled_features) * 0.8)
    X_train = scaled_features[:train_size]
    X_test = scaled_features[train_size:]
    y_train = target[:train_size]
    y_test = target[train_size:]
    
    # Verify shapes
    print(f"Training shapes: X={X_train.shape}, y={y_train.shape}")
    print(f"Testing shapes: X={X_test.shape}, y={y_test.shape}")
    
    return X_train, X_test, y_train, y_test, scaler

def train_model(X_train, y_train, X_test, y_test):
    """Train XGBoost model with verified data"""
    
    # Verify data alignment
    assert len(X_train) == len(y_train), "Training features and labels must have same length"
    assert len(X_test) == len(y_test), "Test features and labels must have same length"
    
    params = {
        'objective': 'reg:squarederror',
        'learning_rate': 0.005,
        'max_depth': 6,
        'min_child_weight': 3,
        'subsample': 0.7,
        'colsample_bytree': 0.7,
        'gamma': 0.1,
        'alpha': 0.1,
        'lambda': 1,
        'random_state': 42
    }
    
    # Create DMatrix objects with explicit feature names
    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=list(X_train.columns))
    dval = xgb.DMatrix(X_test, label=y_test, feature_names=list(X_train.columns))
    
    print(f"DTrain shape: {dtrain.num_row()} rows, {dtrain.num_col()} columns")
    print(f"DVal shape: {dval.num_row()} rows, {dval.num_col()} columns")
    
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=2000,
        evals=[(dtrain, 'train'), (dval, 'val')],
        early_stopping_rounds=100,
        verbose_eval=100
    )
    
    return model

def main():
    print("Fetching Bitcoin data from Binance...")
    df = get_bitcoin_data()
    
    if df is None:
        print("Failed to fetch data. Exiting...")
        return
    
    print("Creating features...")
    df = create_features(df)
    
    print("Preparing data...")
    try:
        X_train, X_test, y_train, y_test, scaler = prepare_data(df)
    except Exception as e:
        print(f"Error in data preparation: {e}")
        return
    
    print("Training model...")
    try:
        model = train_model(X_train, y_train, X_test, y_test)
    except Exception as e:
        print(f"Error in model training: {e}")
        return
    
    # Make predictions
    dtest = xgb.DMatrix(X_test, feature_names=list(X_test.columns))
    test_predictions = model.predict(dtest)
    
    # Convert predictions back to prices
    actual_prices = df['close'][X_test.index]
    predicted_prices = actual_prices * (1 + test_predictions)
    
    # Calculate metrics
    mse = mean_squared_error(actual_prices, predicted_prices)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(actual_prices, predicted_prices)
    r2 = r2_score(actual_prices, predicted_prices)
    
    print("\nModel Evaluation:")
    print(f"Root Mean Square Error: ${rmse:.2f}")
    print(f"Mean Absolute Error: ${mae:.2f}")
    print(f"R² Score: {r2:.4f}")
    
    # Feature importance
    importance_scores = model.get_score(importance_type='gain')
    feature_importance = pd.DataFrame({
        'feature': list(importance_scores.keys()),
        'importance': list(importance_scores.values())
    }).sort_values('importance', ascending=False)
    
    print("\nTop 10 Most Important Features:")
    print(feature_importance.head(10))
    
    # Make future predictions
    try:
        last_data = X_test.iloc[-24:].copy()
        future_predictions = []
        last_price = df['close'].iloc[-1]
        
        # Predict next 24 hours
        for i in range(24):
            dpred = xgb.DMatrix(last_data.iloc[[-1]], feature_names=list(last_data.columns))
            pred_return = model.predict(dpred)[0]
            pred_price = last_price * (1 + pred_return)
            future_predictions.append(pred_price)
            last_price = pred_price
        
        # Create timestamps for predictions
        future_timestamps = [df.index[-1] + timedelta(hours=x) for x in range(1, 25)]
        
        # Visualization
        fig = make_subplots(rows=1, cols=1)
        
        # Historical data
        fig.add_trace(
            go.Scatter(x=df.index[-168:], y=df['close'][-168:], 
                      name='Historical Price', line=dict(color='blue'))
        )
        
        # Predictions
        fig.add_trace(
            go.Scatter(x=future_timestamps, y=future_predictions, 
                      name='Predictions', line=dict(color='red', dash='dash'))
        )
        
        fig.update_layout(
            title='Bitcoin Price Prediction (XGBoost)',
            xaxis_title='Date',
            yaxis_title='Price (USD)',
            hovermode='x unified',
            showlegend=True
        )
        
        fig.write_html('bitcoin_prediction_xgboost.html')
        print("\nPrediction visualization saved as 'bitcoin_prediction_xgboost.html'")
        
        print("\nPredicted prices for the next 24 hours:")
        for timestamp, price in zip(future_timestamps, future_predictions):
            print(f"{timestamp}: ${price:.2f}")
            
    except Exception as e:
        print(f"Error in prediction generation: {e}")

if __name__ == "__main__":
    main()