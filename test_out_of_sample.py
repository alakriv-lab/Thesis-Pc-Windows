import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import yfinance as yf
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
import joblib
import math
import os

# ==========================================
# 1. ΟΡΙΣΜΟΣ ΝΕΥΡΩΝΙΚΩΝ ΔΙΚΤΥΩΝ
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

class AdvancedLSTMModel(nn.Module):
    def __init__(self, input_dim=24, hidden_dim=128, num_layers=2, output_dim=2, dropout=0.1):
        super(AdvancedLSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.fc1 = nn.Linear(hidden_dim, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, output_dim)

    def forward(self, x):
        out, _ = self.lstm(x)
        if x.dim() == 2:
            out = out.unsqueeze(1)
        out = self.fc1(out[:, -1, :])
        out = self.relu(out)
        out = self.fc2(out)
        return out

class GRUModel(nn.Module):
    def __init__(self, input_dim=24, hidden_dim=128, num_layers=2, output_dim=2, dropout=0.1):
        super(GRUModel, self).__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.fc1 = nn.Linear(hidden_dim, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, output_dim)

    def forward(self, x):
        out, _ = self.gru(x)
        if x.dim() == 2:
            out = out.unsqueeze(1)
        out = self.fc1(out[:, -1, :])
        out = self.relu(out)
        out = self.fc2(out)
        return out

class FixedTransformerModel(nn.Module):
    def __init__(self, input_dim=24, d_model=32, nhead=4, num_layers=2, output_dim=2):
        super(FixedTransformerModel, self).__init__()
        self.input_projection = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=64, dropout=0.1, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, output_dim)

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        x = self.input_projection(x)
        x = self.pos_encoder(x)
        out = self.transformer(x)
        out = self.fc(out[:, -1, :])
        return out

# ==========================================
# 2. ΛΗΨΗ & ΠΡΟΕΤΟΙΜΑΣΙΑ ΔΕΔΟΜΕΝΩΝ (2018-2020)
# ==========================================
print("[+] Κατέβασμα δεδομένων (2018-2020) από yfinance...")
start_date = "2018-01-01"
end_date = "2020-12-31"

btc = yf.download("BTC-USD", start=start_date, end=end_date)
eth = yf.download("ETH-USD", start=start_date, end=end_date)

df_test = pd.DataFrame(index=btc.index)
df_test['BTC_Close'] = btc['Close']
df_test['BTC_Volume'] = btc['Volume']
df_test['ETH_Close'] = eth['Close']
df_test['ETH_Volume'] = eth['Volume']

# Βασικά χαρακτηριστικά
df_test['BTC_Price_Change_%'] = df_test['BTC_Close'].pct_change() * 100
df_test['BTC_Daily_Volatility_%'] = ((btc['High'] - btc['Low']) / btc['Open']) * 100
df_test['ETH_Price_Change_%'] = df_test['ETH_Close'].pct_change() * 100
df_test['ETH_Daily_Volatility_%'] = ((eth['High'] - eth['Low']) / eth['Open']) * 100

df_test['Fear_Greed_Index'] = 50.0
df_test['FinBERT_Sentiment_Score'] = 0.0

# Τεχνικοί Δείκτες BTC
df_test['BTC_SMA_7'] = df_test['BTC_Close'].rolling(window=7).mean()
df_test['BTC_SMA_20'] = df_test['BTC_Close'].rolling(window=20).mean()
df_test['BTC_EMA_12'] = df_test['BTC_Close'].ewm(span=12, adjust=False).mean()
df_test['BTC_EMA_26'] = df_test['BTC_Close'].ewm(span=26, adjust=False).mean()
df_test['BTC_MACD'] = df_test['BTC_EMA_12'] - df_test['BTC_EMA_26']
df_test['BTC_MACD_Signal'] = df_test['BTC_MACD'].ewm(span=9, adjust=False).mean()
delta_btc = df_test['BTC_Close'].diff()
gain_btc = (delta_btc.where(delta_btc > 0, 0)).rolling(window=14).mean()
loss_btc = (-delta_btc.where(delta_btc < 0, 0)).rolling(window=14).mean()
df_test['BTC_RSI_14'] = 100 - (100 / (1 + (gain_btc / loss_btc)))

# Τεχνικοί Δείκτες ETH
df_test['ETH_SMA_7'] = df_test['ETH_Close'].rolling(window=7).mean()
df_test['ETH_SMA_20'] = df_test['ETH_Close'].rolling(window=20).mean()
df_test['ETH_EMA_12'] = df_test['ETH_Close'].ewm(span=12, adjust=False).mean()
df_test['ETH_EMA_26'] = df_test['ETH_Close'].ewm(span=26, adjust=False).mean()
df_test['ETH_MACD'] = df_test['ETH_EMA_12'] - df_test['ETH_EMA_26']
df_test['ETH_MACD_Signal'] = df_test['ETH_MACD'].ewm(span=9, adjust=False).mean()
delta_eth = df_test['ETH_Close'].diff()
gain_eth = (delta_eth.where(delta_eth > 0, 0)).rolling(window=14).mean()
loss_eth = (-delta_eth.where(delta_eth < 0, 0)).rolling(window=14).mean()
df_test['ETH_RSI_14'] = 100 - (100 / (1 + (gain_eth / loss_eth)))

# 1. Υπολογισμός Log Returns (TARGETS)
df_test['Target_BTC_Log_Return'] = np.log1p(df_test['BTC_Close'].pct_change().shift(-1))
df_test['Target_ETH_Log_Return'] = np.log1p(df_test['ETH_Close'].pct_change().shift(-1))

# 2. Αποθήκευση πραγματικής τιμής επόμενης ημέρας
df_test['Target_BTC_Actual_Price'] = df_test['BTC_Close'].shift(-1)
df_test['Target_ETH_Actual_Price'] = df_test['ETH_Close'].shift(-1)

# 3. Αφαίρεση N/A (συγχρονίζει όλες τις γραμμές)
df_test.dropna(inplace=True)

feature_cols = [
    'BTC_Close', 'BTC_Volume', 'BTC_Price_Change_%', 'BTC_Daily_Volatility_%',
    'ETH_Close', 'ETH_Volume', 'ETH_Price_Change_%', 'ETH_Daily_Volatility_%',
    'Fear_Greed_Index', 'FinBERT_Sentiment_Score',
    'BTC_SMA_7', 'BTC_SMA_20', 'BTC_EMA_12', 'BTC_EMA_26', 'BTC_MACD', 'BTC_MACD_Signal', 'BTC_RSI_14',
    'ETH_SMA_7', 'ETH_SMA_20', 'ETH_EMA_12', 'ETH_EMA_26', 'ETH_MACD', 'ETH_MACD_Signal', 'ETH_RSI_14'
]

X_test = df_test[feature_cols].values
actual_btc_test = df_test['BTC_Close'].values
actual_eth_test = df_test['ETH_Close'].values
y_test_btc_actual = df_test['Target_BTC_Actual_Price'].values
y_test_eth_actual = df_test['Target_ETH_Actual_Price'].values

# ==========================================
# 3. ΦΟΡΤΩΣΗ SCALERS & ΜΟΝΤΕΛΩΝ
# ==========================================
print("[+] Φόρτωση Scalers και εκπαιδευμένων μοντέλων...")
scaler_X = joblib.load('scaler_X.pkl')
scaler_y_btc = joblib.load('scaler_y_btc.pkl')
scaler_y_eth = joblib.load('scaler_y_eth.pkl')

rf_model = joblib.load('trained_rf_model.pkl')

lstm_model = AdvancedLSTMModel()
if os.path.exists('trained_lstm_model.pt'):
    lstm_model.load_state_dict(torch.load('trained_lstm_model.pt'))
    lstm_model.eval()

gru_model = GRUModel()
if os.path.exists('trained_gru_model.pt'):
    gru_model.load_state_dict(torch.load('trained_gru_model.pt'))
    gru_model.eval()

transformer_model = FixedTransformerModel()
if os.path.exists('trained_transformer_model.pt'):
    transformer_model.load_state_dict(torch.load('trained_transformer_model.pt'))
    transformer_model.eval()

# ==========================================
# 4. ΑΞΙΟΛΟΓΗΣΗ (TESTING & ΑΝΑΣΤΡΟΦΗ LOG RETURNS)
# ==========================================
X_test_scaled = scaler_X.transform(X_test)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)

def print_metrics(name, pred_btc_scaled, pred_eth_scaled):
    pred_btc_log = scaler_y_btc.inverse_transform(pred_btc_scaled.reshape(-1, 1)).flatten()
    pred_eth_log = scaler_y_eth.inverse_transform(pred_eth_scaled.reshape(-1, 1)).flatten()

    pred_btc_dollars = actual_btc_test * np.exp(pred_btc_log)
    pred_eth_dollars = actual_eth_test * np.exp(pred_eth_log)

    rmse_btc = np.sqrt(mean_squared_error(y_test_btc_actual, pred_btc_dollars))
    mae_btc = mean_absolute_error(y_test_btc_actual, pred_btc_dollars)
    mape_btc = mean_absolute_percentage_error(y_test_btc_actual, pred_btc_dollars) * 100

    rmse_eth = np.sqrt(mean_squared_error(y_test_eth_actual, pred_eth_dollars))
    mae_eth = mean_absolute_error(y_test_eth_actual, pred_eth_dollars)
    mape_eth = mean_absolute_percentage_error(y_test_eth_actual, pred_eth_dollars) * 100

    print(f"\n--- Αποτελέσματα {name} (2018-2020) ---")
    print(f"BTC -> RMSE: ${rmse_btc:.2f} | MAE: ${mae_btc:.2f} | MAPE: {mape_btc:.2f}%")
    print(f"ETH -> RMSE: ${rmse_eth:.2f} | MAE: ${mae_eth:.2f} | MAPE: {mape_eth:.2f}%")

# 1. Random Forest
rf_preds = rf_model.predict(X_test_scaled)
print_metrics("Random Forest", rf_preds[:, 0], rf_preds[:, 1])

# 2. Deep Learning Models
with torch.no_grad():
    if os.path.exists('trained_lstm_model.pt'):
        lstm_preds = lstm_model(X_test_tensor).numpy()
        print_metrics("LSTM", lstm_preds[:, 0], lstm_preds[:, 1])

    if os.path.exists('trained_gru_model.pt'):
        gru_preds = gru_model(X_test_tensor).numpy()
        print_metrics("GRU", gru_preds[:, 0], gru_preds[:, 1])

    if os.path.exists('trained_transformer_model.pt'):
        trans_preds = transformer_model(X_test_tensor).numpy()
        print_metrics("Transformer", trans_preds[:, 0], trans_preds[:, 1])