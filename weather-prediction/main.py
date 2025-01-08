import tensorflow as tf
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import requests
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.api.layers import *
from keras.api.optimizers import *


# Fetch data from OpenMeteo API with more parameters
def get_weather_data():
    end_date = datetime.now()
    start_date = end_date - timedelta(days=90)  # Extended historical data
    
    url = (f"https://api.open-meteo.com/v1/forecast?"
           f"latitude=22.2783&longitude=114.1747&"
           f"hourly=temperature_2m,relative_humidity_2m,dew_point_2m,"
           f"apparent_temperature,precipitation,surface_pressure&"
           f"start_date={start_date.strftime('%Y-%m-%d')}&"
           f"end_date={end_date.strftime('%Y-%m-%d')}")
    
    response = requests.get(url)
    data = response.json()
    
    df = pd.DataFrame({
        'datetime': pd.to_datetime(data['hourly']['time']),
        'temperature': data['hourly']['temperature_2m'],
        'humidity': data['hourly']['relative_humidity_2m'],
        'dew_point': data['hourly']['dew_point_2m'],
        'apparent_temp': data['hourly']['apparent_temperature'],
        'precipitation': data['hourly']['precipitation'],
        'pressure': data['hourly']['surface_pressure']
    })
    
    # Add time-based features
    df.set_index('datetime', inplace=True)
    df['hour'] = df.index.hour
    df['day_of_week'] = df.index.dayofweek
    df['month'] = df.index.month
    df['day_of_year'] = df.index.dayofyear
    
    return df

# Attention layer
class AttentionLayer(Layer):
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)
        
    def build(self, input_shape):
        self.W = self.add_weight(name='attention_weight',
                                shape=(input_shape[-1], 1),
                                initializer='random_normal',
                                trainable=True)
        self.b = self.add_weight(name='attention_bias',
                                shape=(input_shape[1], 1),
                                initializer='zeros',
                                trainable=True)
        super(AttentionLayer, self).build(input_shape)
    
    def call(self, x):
        e = tf.keras.backend.tanh(tf.keras.backend.dot(x, self.W) + self.b)
        a = tf.keras.backend.softmax(e, axis=1)
        output = x * a
        return tf.keras.backend.sum(output, axis=1)

# Prepare data for model
def prepare_data(data, n_past=168, n_future=336):  # 7 days past, 14 days future
    X, y = [], []
    for i in range(len(data) - n_past - n_future):
        X.append(data[i:(i + n_past)])
        y.append(data[i + n_past:i + n_past + n_future, 0])  # Only temperature for output
    return np.array(X), np.array(y)

# Create improved model
def create_model(input_shape, output_length):
    inputs = Input(shape=input_shape)
    
    # First Bidirectional LSTM layer with larger units
    x = Bidirectional(LSTM(128, return_sequences=True))(inputs)
    x = Dropout(0.2)(x)
    
    # Second Bidirectional LSTM layer
    x = Bidirectional(LSTM(64, return_sequences=True))(x)
    x = Dropout(0.2)(x)
    
    # Attention mechanism
    x = AttentionLayer()(x)
    
    # Dense layers
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.2)(x)
    x = Dense(64, activation='relu')(x)
    
    # Output layer
    outputs = Dense(output_length, activation='linear')(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model

def main():
    # Get data
    df = get_weather_data()
    
    # Scale data
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df)
    scaled_df = pd.DataFrame(scaled_data, columns=df.columns, index=df.index)
    
    # Prepare training data
    X, y = prepare_data(scaled_data)
    
    # Split data
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    # Create and train model
    model = create_model((X.shape[1], X.shape[2]), y.shape[1])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                 loss='huber')  # Huber loss is more robust to outliers
    
    # Early stopping and learning rate reduction callbacks
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )
    
    lr_reducer = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5
    )
    
    # Train model
    history = model.fit(X_train, y_train,
                       epochs=100,
                       batch_size=32,
                       validation_split=0.2,
                       callbacks=[early_stopping, lr_reducer],
                       verbose=1)
    
    # Make forecast
    last_sequence = scaled_data[-168:]  # Last 7 days
    forecast_scaled = model.predict(last_sequence.reshape(1, 168, scaled_data.shape[1]))
    
    # Inverse transform the forecast (only temperature)
    forecast_reshaped = np.zeros((len(forecast_scaled[0]), scaled_data.shape[1]))
    forecast_reshaped[:, 0] = forecast_scaled[0]  # Temperature column
    forecast = scaler.inverse_transform(forecast_reshaped)[:, 0]
    
    # Create forecast dates
    last_date = df.index[-1]
    forecast_dates = pd.date_range(start=last_date + timedelta(hours=1),
                                 periods=len(forecast),
                                 freq='H')
    
    # Plot results
    plt.figure(figsize=(15, 6))
    plt.plot(df.index[-7*24:], df['temperature'][-7*24:], label='Historical')
    plt.plot(forecast_dates, forecast, label='Forecast')
    plt.title('Hong Kong Temperature Forecast (Improved Model)')
    plt.xlabel('Date')
    plt.ylabel('Temperature (°C)')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # Print forecast summary (daily averages)
    forecast_df = pd.DataFrame({'temperature': forecast}, index=forecast_dates)
    daily_forecast = forecast_df.resample('D').agg({'temperature': ['mean', 'min', 'max']})
    print("\nDaily Temperature Forecast (°C):")
    print(daily_forecast)

if __name__ == "__main__":
    main()