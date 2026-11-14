import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go

st.set_page_config(page_title="Crypto AI Live Dashboard", layout="wide", page_icon="🪙")

st.title("🪙 Crypto AI Live Predictor & Sentiment Dashboard")
st.markdown("Πλατφόρμα Ζωντανής Πρόβλεψης Τιμών Κρυπτονομισμάτων (BTC/ETH) & Ανάλυσης Συναισθήματος")

API_URL = "http://localhost:8001"

# Sidebar
st.sidebar.header("⚙️ Ρυθμίσεις & Controls")
if st.sidebar.button("🔄 Ανανέωση Δεδομένων"):
    st.rerun()

# 1. Fetch Metrics & Predictions
try:
    live_res = requests.get(f"{API_URL}/api/live-metrics", timeout=5).json()
    pred_res = requests.get(f"{API_URL}/api/predict-live", timeout=5).json()
except Exception as e:
    st.error("⚠️ Αδυναμία σύνδεσης με το FastAPI Backend (http://localhost:8001).")
    st.warning("👉 Βεβαιωθείτε ότι έχετε εκκινήσει τον server τρέχοντας 'python app.py' σε ένα ξεχωριστό Terminal.")
    st.stop()

# Top Metrics Row
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Bitcoin (BTC)", f"${live_res['bitcoin']['close_price']:,.2f}", f"{live_res['bitcoin']['change_pct']:+.2f}%")
with col2:
    st.metric("Ethereum (ETH)", f"${live_res['ethereum']['close_price']:,.2f}", f"{live_res['ethereum']['change_pct']:+.2f}%")
with col3:
    st.metric("Fear & Greed Index", f"{live_res['market_sentiment']['fear_greed_index']}/100", live_res['market_sentiment']['fear_greed_class'])
with col4:
    sentiment = live_res['market_sentiment']['finbert_sentiment_score']
    st.metric("FinBERT Sentiment", f"{sentiment:+.2f}", "Positive 🟢" if sentiment > 0 else ("Negative 🔴" if sentiment < 0 else "Neutral 🟡"))

st.divider()

# 2. Prediction Cards Row
st.subheader("🤖 AI Live Predictions (Next-Day Target Price)")

col_rf, col_trans = st.columns(2)

preds = pred_res['predictions']

with col_rf:
    st.info("🌲 **Random Forest Regressor**")
    c1, c2 = st.columns(2)
    c1.metric("BTC Target", f"${preds['random_forest']['BTC_target']:,.2f}", preds['random_forest']['BTC_signal'])
    c2.metric("ETH Target", f"${preds['random_forest']['ETH_target']:,.2f}", preds['random_forest']['ETH_signal'])

with col_trans:
    st.success("🤖 **Transformer Attention Model**")
    c1, c2 = st.columns(2)
    c1.metric("BTC Target", f"${preds['transformer_attention']['BTC_target']:,.2f}", preds['transformer_attention']['BTC_signal'])
    c2.metric("ETH Target", f"${preds['transformer_attention']['ETH_target']:,.2f}", preds['transformer_attention']['ETH_signal'])

st.divider()

# 3. Interactive Chart
st.subheader("📈 Τρέχουσες Τιμές vs AI Προβλέψεις")

fig = go.Figure()

curr_btc = live_res['bitcoin']['close_price']
rf_btc = preds['random_forest']['BTC_target']
trans_btc = preds['transformer_attention']['BTC_target']

fig.add_trace(go.Bar(
    x=['Current Price', 'Random Forest Target', 'Transformer Target'],
    y=[curr_btc, rf_btc, trans_btc],
    name='Bitcoin (BTC)',
    marker_color=['#f7931a', '#2ca02c', '#1f77b4']
))

fig.update_layout(title="Bitcoin Live Price vs AI Target Predictions ($)", yaxis_title="USD ($)", template="plotly_white")
st.plotly_chart(fig, use_container_width=True)

# 4. Extracted Entities Stream
st.subheader("📰 NLP Event & Named Entity Recognition (NER)")
st.write(f"**Εντοπισμένες Οντότητες (SpaCy NER):** `{live_res['market_sentiment']['extracted_entities']}`")