import os
import math
import joblib
import numpy as np
import pandas as pd

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from datetime import datetime, timedelta


# ==========================================
# 1. POSITIONAL ENCODING & TRANSFORMER
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
# 2. DETAILED DUAL FORECAST GENERATOR (BTC & ETH)
# ==========================================
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')


def generate_detailed_forecast(days_to_predict=14):
    print(f"[+] Υπολογισμός αναλυτικής μελλοντικής πρόβλεψης (BTC & ETH) για {days_to_predict} ημέρες...")

    df = pd.read_csv("btc_eth_advanced_events_dataset_5years.csv")

    feature_cols = [
        'BTC_Close', 'BTC_Volume', 'BTC_Price_Change_%', 'BTC_Daily_Volatility_%',
        'ETH_Close', 'ETH_Volume', 'ETH_Price_Change_%', 'ETH_Daily_Volatility_%',
        'Fear_Greed_Index', 'FinBERT_Sentiment_Score',
        'BTC_SMA_7', 'BTC_SMA_20', 'BTC_EMA_12', 'BTC_EMA_26', 'BTC_MACD', 'BTC_MACD_Signal', 'BTC_RSI_14',
        'ETH_SMA_7', 'ETH_SMA_20', 'ETH_EMA_12', 'ETH_EMA_26', 'ETH_MACD', 'ETH_MACD_Signal', 'ETH_RSI_14'
    ]

    scaler_X = joblib.load('scaler_X.pkl')
    scaler_y_btc = joblib.load('scaler_y_btc.pkl')
    scaler_y_eth = joblib.load('scaler_y_eth.pkl')

    transformer_model = FixedTransformerModel(input_dim=len(feature_cols), d_model=32, nhead=4, num_layers=2,
                                              output_dim=2)
    transformer_model.load_state_dict(torch.load('trained_transformer_model.pt'))
    transformer_model.eval()

    # Παίρνουμε τις τελευταίες 45 ημέρες ιστορικού
    last_historical_df = df.tail(45).copy()
    last_date = pd.to_datetime(df['Date'].iloc[-1])

    history_dates = pd.to_datetime(last_historical_df['Date'].values)
    history_btc = last_historical_df['BTC_Close'].values
    history_eth = last_historical_df['ETH_Close'].values

    # Sequential Recursive Forecast
    current_features = scaler_X.transform(last_historical_df[feature_cols].values)
    curr_seq = current_features[-7:].copy()

    future_dates = [last_date + timedelta(days=i) for i in range(1, days_to_predict + 1)]
    future_btc_preds = []
    future_eth_preds = []

    for _ in range(days_to_predict):
        input_tensor = torch.tensor(curr_seq.reshape(1, 7, -1), dtype=torch.float32)
        with torch.no_grad():
            pred_scaled = transformer_model(input_tensor).numpy()

        btc_pred = float(scaler_y_btc.inverse_transform(pred_scaled[:, 0].reshape(-1, 1))[0][0])
        eth_pred = float(scaler_y_eth.inverse_transform(pred_scaled[:, 1].reshape(-1, 1))[0][0])

        future_btc_preds.append(btc_pred)
        future_eth_preds.append(eth_pred)

        # Update sequence
        new_feature_row = curr_seq[-1].copy()
        new_feature_row[0] = pred_scaled[0, 0]
        new_feature_row[4] = pred_scaled[0, 1]
        curr_seq = np.vstack([curr_seq[1:], new_feature_row])

    # ------------------------------------------
    # PLOTTING DETAILED DUAL SUBPLOTS
    # ------------------------------------------
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

    forecast_dates = [history_dates[-1]] + future_dates
    forecast_btc = [history_btc[-1]] + future_btc_preds
    forecast_eth = [history_eth[-1]] + future_eth_preds

    # --- SUBPLOT 1: BITCOIN (BTC) ---
    ax1.plot(history_dates, history_btc, label='Historical Price (Real BTC)', color='#1f77b4', linewidth=2)
    ax1.plot(forecast_dates, forecast_btc, label=f'Transformer Prediction (Next {days_to_predict} Days)',
             color='#d62728', linestyle='--', linewidth=2.5, marker='o')

    # Confidence Interval Bounds (±2.5%)
    btc_upper = [p * 1.025 for p in forecast_btc]
    btc_lower = [p * 0.975 for p in forecast_btc]
    ax1.fill_between(forecast_dates, btc_lower, btc_upper, color='#d62728', alpha=0.15,
                     label='Confidence Interval (±2.5%)')

    ax1.axvline(x=history_dates[-1], color='gray', linestyle=':', linewidth=1.5, label='Forecast Horizon Start')

    # Callout Annotation for BTC Final Price
    btc_change_pct = ((forecast_btc[-1] - forecast_btc[0]) / forecast_btc[0]) * 100
    ax1.annotate(f'Target: ${forecast_btc[-1]:,.2f}\n({btc_change_pct:+.2f}%)',
                 xy=(forecast_dates[-1], forecast_btc[-1]),
                 xytext=(forecast_dates[-1] - timedelta(days=4), forecast_btc[-1] + (max(forecast_btc) * 0.03)),
                 arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=6),
                 fontsize=10, fontweight='bold', bbox=dict(boxstyle="round,pad=0.3", fc="yellow", ec="black", lw=1))

    ax1.set_title('Bitcoin (BTC/USD) Future Price Forecast & Technical Trend', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Price ($)', fontsize=11)
    ax1.legend(loc='upper left', frameon=True)

    # --- SUBPLOT 2: ETHEREUM (ETH) ---
    ax2.plot(history_dates, history_eth, label='Historical Price (Real ETH)', color='#2ca02c', linewidth=2)
    ax2.plot(forecast_dates, forecast_eth, label=f'Transformer Prediction (Next {days_to_predict} Days)',
             color='#9467bd', linestyle='--', linewidth=2.5, marker='s')

    # Confidence Interval Bounds (±3.0%)
    eth_upper = [p * 1.030 for p in forecast_eth]
    eth_lower = [p * 0.970 for p in forecast_eth]
    ax2.fill_between(forecast_dates, eth_lower, eth_upper, color='#9467bd', alpha=0.15,
                     label='Confidence Interval (±3.0%)')

    ax2.axvline(x=history_dates[-1], color='gray', linestyle=':', linewidth=1.5)

    # Callout Annotation for ETH Final Price
    eth_change_pct = ((forecast_eth[-1] - forecast_eth[0]) / forecast_eth[0]) * 100
    ax2.annotate(f'Target: ${forecast_eth[-1]:,.2f}\n({eth_change_pct:+.2f}%)',
                 xy=(forecast_dates[-1], forecast_eth[-1]),
                 xytext=(forecast_dates[-1] - timedelta(days=4), forecast_eth[-1] + (max(forecast_eth) * 0.04)),
                 arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=6),
                 fontsize=10, fontweight='bold', bbox=dict(boxstyle="round,pad=0.3", fc="cyan", ec="black", lw=1))

    ax2.set_title('Ethereum (ETH/USD) Future Price Forecast & Technical Trend', fontsize=13, fontweight='bold')
    ax2.set_xlabel('Date', fontsize=11)
    ax2.set_ylabel('Price ($)', fontsize=11)
    ax2.legend(loc='upper left', frameon=True)

    plt.tight_layout()
    plt.savefig('future_forecast_btc_eth_detailed.png', dpi=300)
    plt.close(fig)
    print(" [✔] Το αναλυτικό διπλό γράφημα δημιουργήθηκε: 'future_forecast_btc_eth_detailed.png'")


if __name__ == "__main__":
    generate_detailed_forecast(14)