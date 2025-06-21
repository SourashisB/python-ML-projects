from flask import Flask, request, jsonify
import torch
import torch.nn as nn
import numpy as np
import joblib
import pandas as pd
import io

# ------------ CONFIGURATION ------------
SEQ_LENGTH = 30
FEATURES = ['Adj Close', 'Close', 'High', 'Low', 'Open', 'Volume']

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

# ------------ LOAD MODEL & SCALER ------------
scaler = joblib.load('scaler.save')
model = LSTMRegressor(input_size=len(FEATURES))
model.load_state_dict(torch.load('stock_ltstm.pth', map_location='cpu'))
model.eval()

# ------------ FLASK APP ------------
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    """
    Accepts a CSV file with columns:
    Date,Adj Close,Close,High,Low,Open,Volume
    Returns predicted next Adj Close.
    """
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded. Please upload a CSV file with the key 'file'."}), 400

    file = request.files['file']
    try:
        # Read CSV into DataFrame
        df = pd.read_csv(file)
    except Exception as e:
        return jsonify({"error": f"Failed to parse CSV: {str(e)}"}), 400

    # Check for required columns
    if not all(col in df.columns for col in FEATURES):
        return jsonify({"error": f"CSV file must contain columns: {FEATURES}"}), 400

    # Sort by Date if possible
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df = df.sort_values('Date')

    # Use only the latest SEQ_LENGTH rows for prediction
    if len(df) < SEQ_LENGTH:
        return jsonify({"error": f"CSV must have at least {SEQ_LENGTH} rows."}), 400

    recent_data = df[FEATURES].tail(SEQ_LENGTH).values

    # Scale input
    try:
        recent_data_scaled = scaler.transform(recent_data)
    except Exception as e:
        return jsonify({"error": f"Scaling error: {str(e)}"}), 400

    # Model expects shape (batch, seq_len, features)
    X = torch.tensor(recent_data_scaled, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        pred_scaled = model(X).item()
    # Inverse-transform the prediction to original scale for 'Adj Close'
    dummy = np.zeros((1, len(FEATURES)))
    dummy[0, 0] = pred_scaled  # 'Adj Close' is the first feature
    adj_close_pred = scaler.inverse_transform(dummy)[0, 0]
    return jsonify({'adj_close_pred': float(adj_close_pred)})

if __name__ == '__main__':
    app.run(debug=True)