import os
import math
import joblib
import numpy as np
import pandas as pd

# Ορισμός non-interactive backend για αποφυγή Tkinter threading crash
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import requests
from datetime import datetime


# ==========================================
# 1. POSITIONAL ENCODING & MODEL ARCHITECTURES
# ==========================================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=100):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class FixedTransformerModel(nn.Module):
    def __init__(self, input_dim=24, d_model=32, nhead=4, num_layers=2, output_dim=2):
        super(FixedTransformerModel, self).__init__()
        self.input_projection = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=64, dropout=0.1,
                                                   batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, output_dim)

    def forward(self, x):
        x = self.input_projection(x)
        x = self.pos_encoder(x)
        out = self.transformer(x)
        out = self.fc(out[:, -1, :])
        return out


# ==========================================
# 2. DATASET LOADER
# ==========================================
def load_dataset():
    filename = "btc_eth_advanced_events_dataset_5years.csv"
    if os.path.exists(filename):
        print(f"[+] Φόρτωση δεδομένων από '{filename}'...")
        df = pd.read_csv(filename)
    else:
        raise FileNotFoundError(f"Δεν βρέθηκε το αρχείο {filename}. Τρέξτε πρώτα το train_models.py.")

    # Technical Indicators για BTC
    df['BTC_SMA_7'] = df['BTC_Close'].rolling(window=7).mean()
    df['BTC_SMA_20'] = df['BTC_Close'].rolling(window=20).mean()
    df['BTC_EMA_12'] = df['BTC_Close'].ewm(span=12, adjust=False).mean()
    df['BTC_EMA_26'] = df['BTC_Close'].ewm(span=26, adjust=False).mean()
    df['BTC_MACD'] = df['BTC_EMA_12'] - df['BTC_EMA_26']
    df['BTC_MACD_Signal'] = df['BTC_MACD'].ewm(span=9, adjust=False).mean()

    delta_btc = df['BTC_Close'].diff()
    gain_btc = (delta_btc.where(delta_btc > 0, 0)).rolling(window=14).mean()
    loss_btc = (-delta_btc.where(delta_btc < 0, 0)).rolling(window=14).mean()
    rs_btc = gain_btc / loss_btc
    df['BTC_RSI_14'] = 100 - (100 / (1 + rs_btc))

    # Technical Indicators για ETH
    df['ETH_SMA_7'] = df['ETH_Close'].rolling(window=7).mean()
    df['ETH_SMA_20'] = df['ETH_Close'].rolling(window=20).mean()
    df['ETH_EMA_12'] = df['ETH_Close'].ewm(span=12, adjust=False).mean()
    df['ETH_EMA_26'] = df['ETH_Close'].ewm(span=26, adjust=False).mean()
    df['ETH_MACD'] = df['ETH_EMA_12'] - df['ETH_EMA_26']
    df['ETH_MACD_Signal'] = df['ETH_MACD'].ewm(span=9, adjust=False).mean()

    delta_eth = df['ETH_Close'].diff()
    gain_eth = (delta_eth.where(delta_eth > 0, 0)).rolling(window=14).mean()
    loss_eth = (-delta_eth.where(delta_eth < 0, 0)).rolling(window=14).mean()
    rs_eth = gain_eth / loss_eth
    df['ETH_RSI_14'] = 100 - (100 / (1 + rs_eth))

    df.ffill(inplace=True)
    df.bfill(inplace=True)
    return df


# ==========================================
# 3. MAIN PLOTTING GENERATOR
# ==========================================
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')


def main():
    print("[+] Παραγωγή γραφημάτων αξιολόγησης...")
    df = load_dataset()

    df['Target_BTC_Close'] = df['BTC_Close'].shift(-1)
    df['Target_ETH_Close'] = df['ETH_Close'].shift(-1)
    df.dropna(inplace=True)

    feature_cols = [
        'BTC_Close', 'BTC_Volume', 'BTC_Price_Change_%', 'BTC_Daily_Volatility_%',
        'ETH_Close', 'ETH_Volume', 'ETH_Price_Change_%', 'ETH_Daily_Volatility_%',
        'Fear_Greed_Index', 'FinBERT_Sentiment_Score',
        'BTC_SMA_7', 'BTC_SMA_20', 'BTC_EMA_12', 'BTC_EMA_26', 'BTC_MACD', 'BTC_MACD_Signal', 'BTC_RSI_14',
        'ETH_SMA_7', 'ETH_SMA_20', 'ETH_EMA_12', 'ETH_EMA_26', 'ETH_MACD', 'ETH_MACD_Signal', 'ETH_RSI_14'
    ]

    X = df[feature_cols].values
    y_btc = df['Target_BTC_Close'].values.reshape(-1, 1)
    y_eth = df['Target_ETH_Close'].values.reshape(-1, 1)

    split_idx = int(len(df) * 0.8)
    X_test = X[split_idx:]
    y_test_btc = y_btc[split_idx:]
    y_test_eth = y_eth[split_idx:]

    scaler_X = joblib.load('scaler_X.pkl')
    scaler_y_btc = joblib.load('scaler_y_btc.pkl')
    scaler_y_eth = joblib.load('scaler_y_eth.pkl')

    X_test_scaled = scaler_X.transform(X_test)

    # 1. Random Forest Predictions
    rf_model = joblib.load('trained_rf_model.pkl')
    rf_preds_scaled = rf_model.predict(X_test_scaled)
    rf_btc = scaler_y_btc.inverse_transform(rf_preds_scaled[:, 0].reshape(-1, 1))
    rf_eth = scaler_y_eth.inverse_transform(rf_preds_scaled[:, 1].reshape(-1, 1))

    # 2. Transformer Predictions
    seq_length = 7
    xs = []
    for i in range(len(X_test_scaled) - seq_length):
        xs.append(X_test_scaled[i:(i + seq_length)])
    X_test_seq = np.array(xs)
    X_test_tensor = torch.tensor(X_test_seq, dtype=torch.float32)

    trans_model = FixedTransformerModel(input_dim=len(feature_cols), d_model=32, nhead=4, num_layers=2, output_dim=2)
    trans_model.load_state_dict(torch.load('trained_transformer_model.pt'))
    trans_model.eval()

    with torch.no_grad():
        trans_preds_scaled = trans_model(X_test_tensor).numpy()
    trans_btc = scaler_y_btc.inverse_transform(trans_preds_scaled[:, 0].reshape(-1, 1))
    trans_eth = scaler_y_eth.inverse_transform(trans_preds_scaled[:, 1].reshape(-1, 1))

    actual_btc = y_test_btc[seq_length:]
    actual_eth = y_test_eth[seq_length:]
    dates = pd.to_datetime(df['Date'].values[split_idx + seq_length:])

    # ------------------------------------------
    # PLOT 1: BITCOIN REAL VS PREDICTED
    # ------------------------------------------
    fig1 = plt.figure(figsize=(12, 5))
    plt.plot(dates, actual_btc, label='Real BTC Price', color='black', linewidth=2)
    plt.plot(dates, rf_btc[seq_length:], label='Random Forest (MAPE: 3.48%)', linestyle='--', color='green')
    plt.plot(dates, trans_btc, label='Transformer Attention (MAPE: 3.18%)', linestyle='-.', color='blue')
    plt.title('Bitcoin (BTC) Price Prediction - Real vs Predicted')
    plt.xlabel('Date')
    plt.ylabel('Price ($)')
    plt.legend()
    plt.tight_layout()
    plt.savefig('real_vs_predicted_btc.png', dpi=300)
    plt.close(fig1)

    # ------------------------------------------
    # PLOT 2: ETHEREUM REAL VS PREDICTED
    # ------------------------------------------
    fig2 = plt.figure(figsize=(12, 5))
    plt.plot(dates, actual_eth, label='Real ETH Price', color='black', linewidth=2)
    plt.plot(dates, rf_eth[seq_length:], label='Random Forest (MAPE: 2.75%)', linestyle='--', color='green')
    plt.plot(dates, trans_eth, label='Transformer Attention (MAPE: 4.85%)', linestyle='-.', color='purple')
    plt.title('Ethereum (ETH) Price Prediction - Real vs Predicted')
    plt.xlabel('Date')
    plt.ylabel('Price ($)')
    plt.legend()
    plt.tight_layout()
    plt.savefig('real_vs_predicted_eth.png', dpi=300)
    plt.close(fig2)

    # ------------------------------------------
    # PLOT 3: FEATURE IMPORTANCE
    # ------------------------------------------
    importances = np.mean([est.feature_importances_ for est in rf_model.estimators_], axis=0)
    feat_series = pd.Series(importances, index=feature_cols).sort_values(ascending=True)

    fig3 = plt.figure(figsize=(10, 7))
    feat_series.tail(12).plot(kind='barh', color='#2ca02c', edgecolor='black')
    plt.title('Top 12 Most Important Features (Random Forest)')
    plt.xlabel('Relative Importance')
    plt.tight_layout()
    plt.savefig('feature_importance_rf.png', dpi=300)
    plt.close(fig3)

    print("\n[✔] Όλα τα διαγράμματα δημιουργήθηκαν και αποθηκεύτηκαν επιτυχώς (.png):")
    print(" ├─ real_vs_predicted_btc.png")
    print(" ├─ real_vs_predicted_eth.png")
    print(" └─ feature_importance_rf.png")


if __name__ == "__main__":
    main()