import pandas as pd
import numpy as np
import tensorflow as tf
from keras.api.models import Sequential
from keras.api.layers import LSTM, Dense, Dropout
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import math
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import datetime

# Load and prepare the data
def load_data(file_path):
    df = pd.read_csv(file_path)
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    return df

# Function to create sequences for time series forecasting
def create_sequences(data, seq_length):
    xs, ys = [], []
    for i in range(len(data) - seq_length - 1):
        x = data[i:(i + seq_length)]
        y = data[i + seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

# TensorFlow LSTM Model
class TFMultivariateLSTM:
    def __init__(self, seq_length, n_features):
        self.seq_length = seq_length
        self.n_features = n_features
        self.model = self._build_model()
        
    def _build_model(self):
        model = Sequential()
        model.add(LSTM(50, return_sequences=True, input_shape=(self.seq_length, self.n_features)))
        model.add(Dropout(0.2))
        model.add(LSTM(50, return_sequences=False))
        model.add(Dropout(0.2))
        model.add(Dense(self.n_features))
        model.compile(optimizer='adam', loss='mse')
        return model
    
    def fit(self, X_train, y_train, epochs=50, batch_size=32, validation_split=0.2):
        history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            verbose=1
        )
        return history
    
    def predict(self, X_test):
        return self.model.predict(X_test)

# PyTorch LSTM Model
class PyTorchMultivariateLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(PyTorchMultivariateLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0))
        out = self.dropout(out[:, -1, :])
        out = self.fc(out)
        return out

# Main function to run the forecasting
def run_forecasting(file_path, seq_length=10, test_size=30):
    # Load data
    df = load_data(file_path)
    print(f"Loaded data with shape: {df.shape}")
    
    # Scale the data
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(df.values)
    
    # Create sequences
    X, y = create_sequences(data_scaled, seq_length)
    
    # Split into train and test sets
    train_size = len(X) - test_size
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    print(f"Training data shape: X={X_train.shape}, y={y_train.shape}")
    print(f"Testing data shape: X={X_test.shape}, y={y_test.shape}")
    
    # TensorFlow Model
    print("Training TensorFlow LSTM model...")
    tf_model = TFMultivariateLSTM(seq_length, df.shape[1])
    tf_history = tf_model.fit(X_train, y_train, epochs=20)
    
    # Make predictions with TensorFlow model
    tf_predictions = tf_model.predict(X_test)
    
    # Inverse transform the predictions
    tf_predictions_rescaled = scaler.inverse_transform(tf_predictions)
    y_test_rescaled = scaler.inverse_transform(y_test)
    
    # Calculate RMSE for TensorFlow model
    tf_rmse = math.sqrt(mean_squared_error(y_test_rescaled, tf_predictions_rescaled))
    print(f"TensorFlow LSTM RMSE: {tf_rmse}")
    
    # PyTorch Model
    print("Training PyTorch LSTM model...")
    # Convert data to PyTorch tensors
    X_train_torch = torch.FloatTensor(X_train)
    y_train_torch = torch.FloatTensor(y_train)
    X_test_torch = torch.FloatTensor(X_test)
    
    # Initialize model
    input_size = df.shape[1]  # Number of features
    hidden_size = 50
    num_layers = 2
    output_size = df.shape[1]
    
    torch_model = PyTorchMultivariateLSTM(input_size, hidden_size, num_layers, output_size)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(torch_model.parameters(), lr=0.001)
    
    # Train PyTorch model
    num_epochs = 20
    for epoch in range(num_epochs):
        torch_model.train()
        optimizer.zero_grad()
        outputs = torch_model(X_train_torch)
        loss = criterion(outputs, y_train_torch)
        loss.backward()
        optimizer.step()
        
        if (epoch+1) % 5 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
    
    # Make predictions with PyTorch model
    torch_model.eval()
    with torch.no_grad():
        torch_predictions = torch_model(X_test_torch).numpy()
    
    # Inverse transform the predictions
    torch_predictions_rescaled = scaler.inverse_transform(torch_predictions)
    
    # Calculate RMSE for PyTorch model
    torch_rmse = math.sqrt(mean_squared_error(y_test_rescaled, torch_predictions_rescaled))
    print(f"PyTorch LSTM RMSE: {torch_rmse}")
    
    # Create visualizations using Plotly
    return create_plotly_visualizations(df, y_test_rescaled, tf_predictions_rescaled, torch_predictions_rescaled, test_size, seq_length)

def create_plotly_visualizations(df, actual, tf_pred, torch_pred, test_size, seq_length):
    feature_names = df.columns
    
    # Create dates for the test predictions - convert to string to avoid Plotly timestamp issues
    test_dates = df.index[-test_size-1:-1]
    test_dates_str = [date.strftime('%Y-%m-%d') for date in test_dates]
    
    # Individual feature predictions vs actual
    feature_plots_data = []
    
    for i, feature in enumerate(feature_names):
        feature_data = {
            "title": f"{feature} - Actual vs Predicted Values",
            "data": [
                {
                    "x": test_dates_str,
                    "y": actual[:, i].tolist(),
                    "type": "scatter",
                    "mode": "lines",
                    "name": f"Actual {feature}",
                    "line": {"color": "blue", "width": 2}
                },
                {
                    "x": test_dates_str,
                    "y": tf_pred[:, i].tolist(),
                    "type": "scatter",
                    "mode": "lines",
                    "name": "TensorFlow Prediction",
                    "line": {"color": "red", "width": 2, "dash": "dash"}
                },
                {
                    "x": test_dates_str,
                    "y": torch_pred[:, i].tolist(),
                    "type": "scatter",
                    "mode": "lines",
                    "name": "PyTorch Prediction",
                    "line": {"color": "green", "width": 2, "dash": "dot"}
                }
            ],
            "layout": {
                "title": f"{feature} - Actual vs Predicted Values",
                "xaxis": {"title": "Date"},
                "yaxis": {"title": "Value"},
                "legend": {"title": "Legend"},
                "hovermode": "x unified",
                "template": "plotly_white"
            }
        }
        feature_plots_data.append(feature_data)
    
    # Error metrics comparison
    tf_rmse_by_feature = [math.sqrt(mean_squared_error(actual[:, i], tf_pred[:, i])) for i in range(len(feature_names))]
    torch_rmse_by_feature = [math.sqrt(mean_squared_error(actual[:, i], torch_pred[:, i])) for i in range(len(feature_names))]
    
    metrics_plot_data = {
        "data": [
            {
                "x": feature_names.tolist(),
                "y": tf_rmse_by_feature,
                "type": "bar",
                "name": "TensorFlow RMSE",
                "marker": {"color": "red"}
            },
            {
                "x": feature_names.tolist(),
                "y": torch_rmse_by_feature,
                "type": "bar",
                "name": "PyTorch RMSE",
                "marker": {"color": "green"}
            }
        ],
        "layout": {
            "title": "Model Performance Comparison (RMSE)",
            "xaxis": {"title": "Features"},
            "yaxis": {"title": "RMSE"},
            "barmode": "group",
            "template": "plotly_white"
        }
    }
    
    # Create a heatmap for the correlation between actual and predicted values
    correlation_data = []
    
    for i, feature in enumerate(feature_names):
        correlation_data.append([
            np.corrcoef(actual[:, i], tf_pred[:, i])[0, 1],
            np.corrcoef(actual[:, i], torch_pred[:, i])[0, 1]
        ])
    
    correlation_plot_data = {
        "data": [
            {
                "z": correlation_data,
                "x": ["TensorFlow", "PyTorch"],
                "y": feature_names.tolist(),
                "type": "heatmap",
                "colorscale": "Viridis",
                "text": [[f"{val:.3f}" for val in row] for row in correlation_data],
                "texttemplate": "%{text}",
                "colorbar": {"title": "Correlation"}
            }
        ],
        "layout": {
            "title": "Correlation between Actual and Predicted Values",
            "template": "plotly_white"
        }
    }
    
    # Combined prediction plot for a key feature (e.g., 'Close')
    if 'Close' in feature_names:
        key_feature_idx = list(feature_names).index('Close')
    else:
        key_feature_idx = 0  # Default to first feature
    
    # Add a section of training data for context
    train_display_size = min(50, len(df) - test_size - seq_length)
    train_display_dates = df.index[-(test_size + train_display_size + 1):-(test_size + 1)]
    train_display_dates_str = [date.strftime('%Y-%m-%d') for date in train_display_dates]
    train_display_values = df.iloc[-(test_size + train_display_size + 1):-(test_size + 1), key_feature_idx].values
    
    # Get the date string for train/test split for the vertical line
    train_test_split_x = df.index[-(test_size + 1)].strftime('%Y-%m-%d')
    
    combined_plot_data = {
        "data": [
            {
                "x": train_display_dates_str,
                "y": train_display_values.tolist(),
                "type": "scatter",
                "mode": "lines",
                "name": f"Training Data ({feature_names[key_feature_idx]})",
                "line": {"color": "gray", "width": 1.5}
            },
            {
                "x": test_dates_str,
                "y": actual[:, key_feature_idx].tolist(),
                "type": "scatter",
                "mode": "lines",
                "name": f"Actual Test Data ({feature_names[key_feature_idx]})",
                "line": {"color": "blue", "width": 2}
            },
            {
                "x": test_dates_str,
                "y": tf_pred[:, key_feature_idx].tolist(),
                "type": "scatter",
                "mode": "lines",
                "name": "TensorFlow Prediction",
                "line": {"color": "red", "width": 2, "dash": "dash"}
            },
            {
                "x": test_dates_str,
                "y": torch_pred[:, key_feature_idx].tolist(),
                "type": "scatter", 
                "mode": "lines",
                "name": "PyTorch Prediction",
                "line": {"color": "green", "width": 2, "dash": "dot"}
            }
        ],
        "layout": {
            "title": f"Stock Price Forecasting: {feature_names[key_feature_idx]} (Training + Testing)",
            "xaxis": {"title": "Date"},
            "yaxis": {"title": "Value"},
            "legend": {"title": "Legend"},
            "hovermode": "x unified",
            "template": "plotly_white",
            "shapes": [
                {
                    "type": "line",
                    "xref": "x",
                    "yref": "paper",
                    "x0": train_test_split_x,
                    "y0": 0,
                    "x1": train_test_split_x,
                    "y1": 1,
                    "line": {
                        "color": "black",
                        "width": 2,
                        "dash": "dash"
                    }
                }
            ],
            "annotations": [
                {
                    "x": train_test_split_x,
                    "y": 1.05,
                    "xref": "x",
                    "yref": "paper",
                    "text": "Train/Test Split",
                    "showarrow": False,
                    "font": {"color": "black"}
                }
            ]
        }
    }
    
    return feature_plots_data, metrics_plot_data, correlation_plot_data, combined_plot_data

# Example usage
if __name__ == "__main__":
    # Generate sample data
    dates = pd.date_range(start='2020-01-01', periods=500)
    np.random.seed(42)
    
    # Create synthetic stock data
    close = 100 + np.cumsum(np.random.normal(0, 1, 500))
    adj_close = close * np.random.uniform(0.99, 1.01, 500)
    high = close + np.random.uniform(0, 2, 500)
    low = close - np.random.uniform(0, 2, 500)
    open_price = close - np.random.normal(0, 1, 500)
    volume = np.random.normal(1000000, 200000, 500).astype(int)
    
    # Create DataFrame
    sample_df = pd.DataFrame({
        'Close': close,
        'Adj Close': adj_close,
        'High': high,
        'Low': low,
        'Open': open_price,
        'Volume': volume
    }, index=dates)
    
    # Save to CSV
    csv_path = 'NVIDIA_STOCK.csv'
    sample_df.reset_index().rename(columns={'index': 'Date'}).to_csv(csv_path, index=False)
    
    # Run forecasting
    feature_plots_data, metrics_plot_data, correlation_plot_data, combined_plot_data = run_forecasting(csv_path, seq_length=10, test_size=30)
    
    # Create HTML with the interactive Plotly visualizations
    html_output = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Interactive Stock Price Forecasting</title>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        <style>
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                margin: 0;
                padding: 0;
                background-color: #f5f5f5;
            }
            .container {
                max-width: 1200px;
                margin: 0 auto;
                padding: 20px;
            }
            .dashboard {
                display: grid;
                grid-template-columns: 1fr;
                gap: 20px;
                margin-top: 20px;
            }
            .card {
                background-color: white;
                border-radius: 8px;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                padding: 20px;
                margin-bottom: 20px;
            }
            h1, h2, h3 {
                color: #333;
            }
            .plot {
                width: 100%;
                height: 500px;
                margin: 15px 0;
            }
            .metrics-container {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(500px, 1fr));
                gap: 20px;
            }
            .feature-tabs {
                display: flex;
                flex-wrap: wrap;
                margin-bottom: 10px;
                border-bottom: 1px solid #ddd;
            }
            .feature-tab {
                padding: 10px 15px;
                cursor: pointer;
                background-color: #f1f1f1;
                border: none;
                border-radius: 5px 5px 0 0;
                margin-right: 5px;
                transition: background-color 0.3s;
            }
            .feature-tab.active {
                background-color: #4CAF50;
                color: white;
            }
            @media (max-width: 768px) {
                .metrics-container {
                    grid-template-columns: 1fr;
                }
                .feature-tabs {
                    overflow-x: auto;
                }
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Interactive Stock Price Forecasting Dashboard</h1>
            <p>Comparison of TensorFlow and PyTorch models for predicting stock market data</p>
            
            <div class="dashboard">
                <div class="card">
                    <h2>Overview: Combined Forecast for Primary Feature</h2>
                    <div id="combined-plot" class="plot"></div>
                </div>
                
                <div class="card">
                    <h2>Detailed Feature Analysis</h2>
                    <div class="feature-tabs" id="feature-tabs"></div>
                    <div id="feature-plot" class="plot"></div>
                </div>
                
                <div class="metrics-container">
                    <div class="card">
                        <h3>Model Performance Comparison</h3>
                        <div id="metrics-plot" class="plot"></div>
                    </div>
                    <div class="card">
                        <h3>Prediction-Actual Correlation</h3>
                        <div id="correlation-plot" class="plot"></div>
                    </div>
                </div>
            </div>
        </div>
        
        <script>
            // Parse the JSON data
            const featurePlots = FEATURE_PLOTS_PLACEHOLDER;
            const metricsPlot = METRICS_PLOT_PLACEHOLDER;
            const correlationPlot = CORRELATION_PLOT_PLACEHOLDER;
            const combinedPlot = COMBINED_PLOT_PLACEHOLDER;
            
            // Create the tabs for feature selection
            const featureTabsContainer = document.getElementById('feature-tabs');
            let currentFeatureIndex = 0;
            
            featurePlots.forEach((plot, index) => {
                const featureName = plot.title.split(' - ')[0];
                
                const tab = document.createElement('button');
                tab.className = 'feature-tab ' + (index === 0 ? 'active' : '');
                tab.textContent = featureName;
                tab.onclick = function() {
                    document.querySelectorAll('.feature-tab').forEach(t => t.classList.remove('active'));
                    this.classList.add('active');
                    currentFeatureIndex = index;
                    Plotly.react('feature-plot', featurePlots[index].data, featurePlots[index].layout);
                };
                
                featureTabsContainer.appendChild(tab);
            });
            
            // Initial plots
            Plotly.newPlot('feature-plot', featurePlots[0].data, featurePlots[0].layout);
            Plotly.newPlot('metrics-plot', metricsPlot.data, metricsPlot.layout);
            Plotly.newPlot('correlation-plot', correlationPlot.data, correlationPlot.layout);
            Plotly.newPlot('combined-plot', combinedPlot.data, combinedPlot.layout);
            
            // Make plots responsive
            window.onresize = function() {
                Plotly.relayout('feature-plot', {
                    'width': document.getElementById('feature-plot').clientWidth
                });
                Plotly.relayout('metrics-plot', {
                    'width': document.getElementById('metrics-plot').clientWidth
                });
                Plotly.relayout('correlation-plot', {
                    'width': document.getElementById('correlation-plot').clientWidth
                });
                Plotly.relayout('combined-plot', {
                    'width': document.getElementById('combined-plot').clientWidth
                });
            };
        </script>
    </body>
    </html>
    """
    
    # Replace the placeholders with JSON strings
    html_output = html_output.replace('FEATURE_PLOTS_PLACEHOLDER', json.dumps(feature_plots_data))
    html_output = html_output.replace('METRICS_PLOT_PLACEHOLDER', json.dumps(metrics_plot_data))
    html_output = html_output.replace('CORRELATION_PLOT_PLACEHOLDER', json.dumps(correlation_plot_data))
    html_output = html_output.replace('COMBINED_PLOT_PLACEHOLDER', json.dumps(combined_plot_data))
    
    # Save the HTML to a file
    with open('stock_forecasting_dashboard.html', 'w') as f:
        f.write(html_output)
    
    print("Interactive dashboard has been created and saved as 'stock_forecasting_dashboard.html'")