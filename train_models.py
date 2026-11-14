import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
import joblib
import math
import os

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
# 2. ΦΟΡΤΩΣΗ ΚΑΙ ΠΡΟΕΤΟΙΜΑΣΙΑ ΔΕΔΟΜΕΝΩΝ
# ==========================================
print("[+] Φόρτωση δεδομένων εκπαίδευσης...")
df = pd.read_csv("btc_eth_advanced_events_dataset_5years.csv", parse_dates=['Date'])
df.sort_values('Date', inplace=True)
df.set_index('Date', inplace=True)

# Υπολογισμός Log Returns (TARGETS)
df['Target_BTC_Log_Return'] = np.log1p(df['BTC_Close'].pct_change().shift(-1))
df['Target_ETH_Log_Return'] = np.log1p(df['ETH_Close'].pct_change().shift(-1))
df['Target_BTC_Actual_Price'] = df['BTC_Close'].shift(-1)
df['Target_ETH_Actual_Price'] = df['ETH_Close'].shift(-1)
df.dropna(inplace=True)

feature_cols = [
    'BTC_Close', 'BTC_Volume', 'BTC_Price_Change_%', 'BTC_Daily_Volatility_%',
    'ETH_Close', 'ETH_Volume', 'ETH_Price_Change_%', 'ETH_Daily_Volatility_%',
    'Fear_Greed_Index', 'FinBERT_Sentiment_Score',
    'BTC_SMA_7', 'BTC_SMA_20', 'BTC_EMA_12', 'BTC_EMA_26', 'BTC_MACD', 'BTC_MACD_Signal', 'BTC_RSI_14',
    'ETH_SMA_7', 'ETH_SMA_20', 'ETH_EMA_12', 'ETH_EMA_26', 'ETH_MACD', 'ETH_MACD_Signal', 'ETH_RSI_14'
]

X = df[feature_cols].values
y_btc_log = df['Target_BTC_Log_Return'].values
y_eth_log = df['Target_ETH_Log_Return'].values
actual_btc_prices = df['BTC_Close'].values
actual_eth_prices = df['ETH_Close'].values

# Διαχωρισμός Train/Test (80-20)
train_size = int(len(df) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train_btc, y_test_btc = y_btc_log[:train_size], y_btc_log[train_size:]
y_train_eth, y_test_eth = y_eth_log[:train_size], y_eth_log[train_size:]

# Scaling
scaler_X = MinMaxScaler(feature_range=(-1, 1))
scaler_y_btc = MinMaxScaler(feature_range=(-1, 1))
scaler_y_eth = MinMaxScaler(feature_range=(-1, 1))

X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)

y_train_btc_scaled = scaler_y_btc.fit_transform(y_train_btc.reshape(-1, 1))
y_train_eth_scaled = scaler_y_eth.fit_transform(y_train_eth.reshape(-1, 1))

y_train_combined = np.hstack((y_train_btc_scaled, y_train_eth_scaled))

# Αποθήκευση Scalers
joblib.dump(scaler_X, 'scaler_X.pkl')
joblib.dump(scaler_y_btc, 'scaler_y_btc.pkl')
joblib.dump(scaler_y_eth, 'scaler_y_eth.pkl')

X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train_combined, dtype=torch.float32)

# ==========================================
# 3. ΕΚΠΑΙΔΕΥΣΗ ΜΟΝΤΕΛΩΝ (REGULARIZED FINE-TUNED)
# ==========================================
print("[+] Εκπαίδευση Regularized Random Forest (200 estimators, max_depth=8)...")
rf_model = RandomForestRegressor(
    n_estimators=200,
    max_depth=8,              # Περιορισμός βάθους για αποφυγή overfitting
    min_samples_split=10,     # Ελάχιστα δείγματα για διακλάδωση
    min_samples_leaf=5,       # Ελάχιστα δείγματα σε κάθε φύλλο (εξομάλυνση)
    max_features='sqrt',      # Τυχαιότητα χαρακτηριστικών για γενίκευση
    random_state=42,
    n_jobs=-1
)
rf_model.fit(X_train_scaled, y_train_combined)
joblib.dump(rf_model, 'trained_rf_model.pkl')
print("[✔] Random Forest εκπαιδεύτηκε και αποθηκεύτηκε.")

def train_pytorch_model(model, name, epochs=200, lr=0.0005):
    print(f"[+] Εκπαίδευση {name} ({epochs} Epochs)...")
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

    torch.save(model.state_dict(), f'trained_{name.lower()}_model.pt')
    print(f"[✔] {name} ολοκληρώθηκε.")

lstm_model = AdvancedLSTMModel()
train_pytorch_model(lstm_model, "LSTM")

gru_model = GRUModel()
train_pytorch_model(gru_model, "GRU")

transformer_model = FixedTransformerModel()
train_pytorch_model(transformer_model, "Transformer")

print("[✔] Επανεκπαίδευση ολοκληρώθηκε επιτυχώς!")

# ==========================================
# 4. ΑΞΙΟΛΟΓΗΣΗ ΜΟΝΤΕΛΩΝ ΣΤΟ TEST SET (20%)
# ==========================================
print("\n[+] Υπολογισμός μετρικών (RMSE, MAE, MAPE) στο Test Set...")

actual_btc_test = actual_btc_prices[train_size:]
actual_eth_test = actual_eth_prices[train_size:]

y_test_btc_actual = df['Target_BTC_Actual_Price'].values[train_size:]
y_test_eth_actual = df['Target_ETH_Actual_Price'].values[train_size:]

def evaluate_in_sample(name, pred_btc_scaled, pred_eth_scaled):
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

    print(f"\n--- {name} (Test Set 2021-2026) ---")
    print(f"BTC -> RMSE: ${rmse_btc:.2f} | MAE: ${mae_btc:.2f} | MAPE: {mape_btc:.2f}%")
    print(f"ETH -> RMSE: ${rmse_eth:.2f} | MAE: ${mae_eth:.2f} | MAPE: {mape_eth:.2f}%")

# 1. Random Forest Evaluation
rf_preds = rf_model.predict(X_test_scaled)
evaluate_in_sample("Random Forest", rf_preds[:, 0], rf_preds[:, 1])

# 2. Deep Learning Evaluation
X_test_tensor_dl = torch.tensor(X_test_scaled, dtype=torch.float32)
with torch.no_grad():
    lstm_model.eval()
    lstm_preds = lstm_model(X_test_tensor_dl).numpy()
    evaluate_in_sample("LSTM", lstm_preds[:, 0], lstm_preds[:, 1])

    gru_model.eval()
    gru_preds = gru_model(X_test_tensor_dl).numpy()
    evaluate_in_sample("GRU", gru_preds[:, 0], gru_preds[:, 1])

    transformer_model.eval()
    trans_preds = transformer_model(X_test_tensor_dl).numpy()
    evaluate_in_sample("Transformer", trans_preds[:, 0], trans_preds[:, 1])