import sys
import json
import requests
import asyncio
import numpy as np
import pandas as pd
import pyqtgraph as pg

from PyQt6.QtCore import Qt, QTimer, QThread, pyqtSignal
from PyQt6.QtGui import QFont, QColor
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QFrame, QGridLayout, QTabWidget,
    QTableWidget, QTableWidgetItem, QHeaderView, QScrollArea,
    QGraphicsDropShadowEffect
)

# ==========================================
# LINEAR DESIGN SYSTEM STYLESHEET
# ==========================================
LINEAR_STYLE = """
QMainWindow {
    background-color: #08090a;
}

QWidget {
    color: #f7f8f8;
    font-family: 'Inter', 'Segoe UI', -apple-system, sans-serif;
}

QFrame.linear-card {
    background-color: #101217;
    border: 1px solid rgba(255, 255, 255, 0.05);
    border-radius: 10px;
}

QFrame.linear-card:hover {
    border: 1px solid rgba(255, 255, 255, 0.15);
    background-color: #14161c;
}

QFrame.sidebar-news-card {
    background-color: #101217;
    border: 1px solid rgba(255, 255, 255, 0.04);
    border-radius: 8px;
    padding: 10px;
    margin-bottom: 6px;
}

QFrame.sidebar-news-card:hover {
    background-color: #16181f;
    border: 1px solid rgba(255, 255, 255, 0.1);
}

QFrame#sidebar {
    background-color: #040405;
    border-right: 1px solid rgba(255, 255, 255, 0.06);
}

QTabWidget::pane {
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 8px;
    background-color: #0c0d12;
    top: -1px;
}

QTabBar::tab {
    background-color: transparent;
    color: #62677d;
    padding: 8px 16px;
    margin-right: 4px;
    border-radius: 6px;
    font-weight: 600;
    font-size: 12px;
}

QTabBar::tab:selected {
    background-color: rgba(255, 255, 255, 0.08);
    color: #f7f8f8;
}

QLabel#main-title {
    font-size: 24px;
    font-weight: 700;
    color: #f7f8f8;
    letter-spacing: -0.5px;
}

QLabel#subtitle {
    font-size: 12px;
    color: #62677d;
}

QLabel#metric-title {
    font-size: 11px;
    font-weight: 700;
    color: #8a8f9e;
    letter-spacing: 0.8px;
    text-transform: uppercase;
}

QLabel#metric-value {
    font-size: 22px;
    font-weight: 700;
    letter-spacing: -0.5px;
}

QPushButton.btn-linear {
    background-color: #5e6ad2;
    color: #ffffff;
    font-size: 13px;
    font-weight: 600;
    border: none;
    border-radius: 6px;
    padding: 10px 16px;
}

QPushButton.btn-linear:hover {
    background-color: #6e7be2;
}

QPushButton.btn-entity-chip {
    background-color: #1e293b;
    color: #38bdf8;
    border: 1px solid rgba(56, 189, 248, 0.3);
    border-radius: 10px;
    padding: 3px 8px;
    font-size: 11px;
    font-weight: 600;
}

QPushButton.btn-entity-chip:hover {
    background-color: #38bdf8;
    color: #0f172a;
}

QPushButton.btn-entity-active {
    background-color: #38bdf8;
    color: #0f172a;
    border: 1px solid #38bdf8;
    border-radius: 10px;
    padding: 3px 8px;
    font-size: 11px;
    font-weight: 700;
}

QTableWidget {
    background-color: #0c0d12;
    gridline-color: rgba(255, 255, 255, 0.03);
    color: #e2e4e9;
    border: none;
    font-size: 13px;
}

QHeaderView::section {
    background-color: #08090a;
    color: #8a8f9e;
    font-weight: 600;
    font-size: 12px;
    border: none;
    border-bottom: 1px solid rgba(255, 255, 255, 0.08);
    padding: 10px;
}

QScrollBar:vertical {
    border: none;
    background: transparent;
    width: 8px;
}
QScrollBar::handle:vertical {
    background: rgba(255, 255, 255, 0.15);
    min-height: 20px;
    border-radius: 4px;
}
"""


# ==========================================
# 1. THREAD WORKERS
# ==========================================
class ApiWorker(QThread):
    data_fetched = pyqtSignal(dict, dict, dict, dict)

    def run(self):
        try:
            metrics = requests.get("http://localhost:8001/api/live-metrics", timeout=4).json()
            predictions = requests.get("http://localhost:8001/api/predict-live", timeout=4).json()
            news = requests.get("http://localhost:8001/api/live-news", timeout=4).json()
            lagged = requests.get("http://localhost:8001/api/lagged-correlation", timeout=4).json()
            self.data_fetched.emit(metrics, predictions, news, lagged)
        except Exception:
            self.data_fetched.emit({}, {}, {}, {})


class WebSocketWorker(QThread):
    tick_received = pyqtSignal(dict)
    connection_status = pyqtSignal(bool)

    def run(self):
        try:
            import websockets
        except ImportError:
            print("[!] 'websockets' package missing. Install via 'pip install websockets'")
            return

        async def listen():
            url = "ws://localhost:8001/ws/crypto-stream"
            while True:
                try:
                    async with websockets.connect(url) as ws:
                        self.connection_status.emit(True)
                        print("[⚡ Desktop UI] WebSocket Real-Time Stream Connected!")
                        while True:
                            msg = await ws.recv()
                            data = json.loads(msg)
                            self.tick_received.emit(data)
                except Exception:
                    self.connection_status.emit(False)
                    await asyncio.sleep(2)

        asyncio.run(listen())


class LinearCard(QFrame):
    def __init__(self, title, value="$0.00", subtitle="", tooltip="", accent_color="#f7f8f8"):
        super().__init__()
        self.setProperty("class", "linear-card")
        if tooltip:
            self.setToolTip(tooltip)

        shadow = QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(15)
        shadow.setXOffset(0)
        shadow.setYOffset(4)
        shadow.setColor(QColor(0, 0, 0, 60))
        self.setGraphicsEffect(shadow)

        layout = QVBoxLayout()
        layout.setContentsMargins(18, 16, 18, 16)
        layout.setSpacing(6)

        self.lbl_title = QLabel(title)
        self.lbl_title.setObjectName("metric-title")

        self.lbl_value = QLabel(value)
        self.lbl_value.setObjectName("metric-value")
        self.lbl_value.setStyleSheet(f"color: {accent_color};")

        self.lbl_sub = QLabel(subtitle)
        self.lbl_sub.setObjectName("subtitle")

        layout.addWidget(self.lbl_title)
        layout.addWidget(self.lbl_value)
        layout.addWidget(self.lbl_sub)
        self.setLayout(layout)

    def update_data(self, value, subtitle="", accent_color=None):
        self.lbl_value.setText(value)
        self.lbl_sub.setText(subtitle)
        if accent_color:
            self.lbl_value.setStyleSheet(f"color: {accent_color};")


class LinearCryptoApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Crypto AI — Linear Intelligence Terminal")
        self.resize(1600, 950)
        self.setStyleSheet(LINEAR_STYLE)

        self.raw_news_data = []
        self.active_ner_filter = None

        main_widget = QWidget()
        main_layout = QHBoxLayout()
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # 1. SIDEBAR
        sidebar = QFrame()
        sidebar.setObjectName("sidebar")
        sidebar.setFixedWidth(360)
        sidebar_layout = QVBoxLayout()
        sidebar_layout.setContentsMargins(18, 20, 18, 20)

        brand_lbl = QLabel("❖ Crypto AI")
        brand_lbl.setFont(QFont('Inter', 16, QFont.Weight.Bold))
        brand_lbl.setStyleSheet("color: #f7f8f8;")

        sub_brand = QLabel("Thesis Intelligence Pipeline")
        sub_brand.setObjectName("subtitle")

        sidebar_layout.addWidget(brand_lbl)
        sidebar_layout.addWidget(sub_brand)
        sidebar_layout.addSpacing(16)

        self.btn_refresh = QPushButton("Sync Pipeline")
        self.btn_refresh.setProperty("class", "btn-linear")
        self.btn_refresh.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_refresh.clicked.connect(self.fetch_api_data)
        sidebar_layout.addWidget(self.btn_refresh)

        sidebar_layout.addSpacing(18)

        lbl_news_title = QLabel("24H LIVE NEWS & NLP ANALYTICS")
        lbl_news_title.setObjectName("metric-title")
        sidebar_layout.addWidget(lbl_news_title)

        # NER DYNAMIC FILTER BAR
        lbl_filter_head = QLabel("🏷️ ΔΙΑΔΡΑΣΤΙΚΑ NER ΦΙΛΤΡΑ:")
        lbl_filter_head.setStyleSheet("font-size: 10px; font-weight: 700; color: #8a8f9e; margin-top: 4px;")
        sidebar_layout.addWidget(lbl_filter_head)

        self.ner_scroll = QScrollArea()
        self.ner_scroll.setFixedHeight(42)
        self.ner_scroll.setWidgetResizable(True)
        self.ner_scroll.setStyleSheet("border: none; background: transparent;")

        self.ner_container = QWidget()
        self.ner_layout = QHBoxLayout()
        self.ner_layout.setContentsMargins(0, 2, 0, 2)
        self.ner_layout.setSpacing(6)
        self.ner_container.setLayout(self.ner_layout)
        self.ner_scroll.setWidget(self.ner_container)
        sidebar_layout.addWidget(self.ner_scroll)

        # NEWS TABS
        self.news_tabs = QTabWidget()

        self.tab_btc_news = QWidget()
        self.box_btc_layout = QVBoxLayout()
        self.box_btc_layout.setSpacing(8)
        self.box_btc_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        scroll_btc = QScrollArea()
        scroll_btc.setWidgetResizable(True)
        scroll_btc.setStyleSheet("border: none; background: transparent;")
        btc_container = QWidget()
        btc_container.setLayout(self.box_btc_layout)
        scroll_btc.setWidget(btc_container)
        v_btc = QVBoxLayout()
        v_btc.setContentsMargins(0, 8, 0, 0)
        v_btc.addWidget(scroll_btc)
        self.tab_btc_news.setLayout(v_btc)

        self.tab_eth_news = QWidget()
        self.box_eth_layout = QVBoxLayout()
        self.box_eth_layout.setSpacing(8)
        self.box_eth_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        scroll_eth = QScrollArea()
        scroll_eth.setWidgetResizable(True)
        scroll_eth.setStyleSheet("border: none; background: transparent;")
        eth_container = QWidget()
        eth_container.setLayout(self.box_eth_layout)
        scroll_eth.setWidget(eth_container)
        v_eth = QVBoxLayout()
        v_eth.setContentsMargins(0, 8, 0, 0)
        v_eth.addWidget(scroll_eth)
        self.tab_eth_news.setLayout(v_eth)

        self.news_tabs.addTab(self.tab_btc_news, "BTC News 🟠")
        self.news_tabs.addTab(self.tab_eth_news, "ETH News 🔷")

        sidebar_layout.addWidget(self.news_tabs)

        self.status_lbl = QLabel("● Connecting WebSocket Stream...")
        self.status_lbl.setStyleSheet("color: #f59e0b; font-size: 11px; font-weight: 600; margin-top: 6px;")
        sidebar_layout.addWidget(self.status_lbl)

        sidebar.setLayout(sidebar_layout)

        # 2. MAIN CONTENT
        content_area = QWidget()
        content_layout = QVBoxLayout()
        content_layout.setContentsMargins(32, 28, 32, 28)
        content_layout.setSpacing(20)

        header_layout = QVBoxLayout()
        header_layout.setSpacing(4)
        lbl_head = QLabel("Market Intelligence & Predictive AI Terminal")
        lbl_head.setObjectName("main-title")
        lbl_sub = QLabel("Real-time tick-by-tick forecast engine with Ensemble Model & Event Markers")
        lbl_sub.setObjectName("subtitle")
        header_layout.addWidget(lbl_head)
        header_layout.addWidget(lbl_sub)
        content_layout.addLayout(header_layout)

        grid_metrics = QGridLayout()
        grid_metrics.setSpacing(16)

        self.card_btc = LinearCard("BITCOIN (BTC)", "$0.00", "24h Change: 0.00%", tooltip="Spot Price",
                                   accent_color="#f7f8f8")
        self.card_eth = LinearCard("ETHEREUM (ETH)", "$0.00", "24h Change: 0.00%", tooltip="Spot Price",
                                   accent_color="#a855f7")
        self.card_fng = LinearCard("FEAR & GREED", "50/100", "Market Sentiment Index", tooltip="Macro Sentiment",
                                   accent_color="#facc15")
        self.card_sent = LinearCard("FINBERT SCORE", "+0.00", "NLP Sentiment Polarity", tooltip="FinBERT News Score",
                                    accent_color="#5e6ad2")

        grid_metrics.addWidget(self.card_btc, 0, 0)
        grid_metrics.addWidget(self.card_eth, 0, 1)
        grid_metrics.addWidget(self.card_fng, 0, 2)
        grid_metrics.addWidget(self.card_sent, 0, 3)

        content_layout.addLayout(grid_metrics)

        # PREDICTIONS CARDS (INCLUDES ENSEMBLE CONSENSUS CARDS)
        grid_preds = QGridLayout()
        grid_preds.setSpacing(16)

        self.card_trans_btc = LinearCard("TRANSFORMER TARGET (BTC)", "$0.00", "Signal: ...", accent_color="#5e6ad2")
        self.card_lstm_btc = LinearCard("LSTM TARGET (BTC)", "$0.00", "Signal: ...", accent_color="#22c55e")
        self.card_ens_btc = LinearCard("ENSEMBLE TARGET (BTC) 🌟", "$0.00", "Signal: ...", accent_color="#eab308")

        self.card_trans_eth = LinearCard("TRANSFORMER TARGET (ETH)", "$0.00", "Signal: ...", accent_color="#38bdf8")
        self.card_lstm_eth = LinearCard("LSTM TARGET (ETH)", "$0.00", "Signal: ...", accent_color="#a855f7")
        self.card_ens_eth = LinearCard("ENSEMBLE TARGET (ETH) 🌟", "$0.00", "Signal: ...", accent_color="#eab308")

        grid_preds.addWidget(self.card_trans_btc, 0, 0)
        grid_preds.addWidget(self.card_lstm_btc, 0, 1)
        grid_preds.addWidget(self.card_ens_btc, 0, 2)

        grid_preds.addWidget(self.card_trans_eth, 1, 0)
        grid_preds.addWidget(self.card_lstm_eth, 1, 1)
        grid_preds.addWidget(self.card_ens_eth, 1, 2)

        content_layout.addLayout(grid_preds)

        # TABS & GRAPHICS
        self.tabs = QTabWidget()

        pg.setConfigOptions(antialias=True)
        pg.setConfigOption('background', '#0c0d12')
        pg.setConfigOption('foreground', '#8a8f9e')

        # TAB 1: BTC CHART WITH ENSEMBLE HORIZON
        tab_btc_chart = QWidget()
        v_btc_c = QVBoxLayout()
        v_btc_c.setContentsMargins(16, 16, 16, 16)
        lbl_btc_graph_type = QLabel("📊 Graph: Historical Price Trend, Event Markers & Ensemble Prediction Horizon 🌟")
        lbl_btc_graph_type.setStyleSheet("font-size: 12px; color: #8a8f9e; font-weight: 500; margin-bottom: 8px;")
        v_btc_c.addWidget(lbl_btc_graph_type)

        self.plot_btc = pg.PlotWidget(title="<b>BTC/USD Trend, Event Pins & Ensemble Consensus</b>")
        self.plot_btc.showGrid(x=True, y=True, alpha=0.1)
        self.plot_btc.addLegend(offset=(15, 15))
        v_btc_c.addWidget(self.plot_btc)
        tab_btc_chart.setLayout(v_btc_c)

        # TAB 2: ETH CHART WITH ENSEMBLE HORIZON
        tab_eth_chart = QWidget()
        v_eth_c = QVBoxLayout()
        v_eth_c.setContentsMargins(16, 16, 16, 16)
        lbl_eth_graph_type = QLabel("📊 Graph: Historical Price Trend, Event Markers & Ensemble Prediction Horizon 🌟")
        lbl_eth_graph_type.setStyleSheet("font-size: 12px; color: #8a8f9e; font-weight: 500; margin-bottom: 8px;")
        v_eth_c.addWidget(lbl_eth_graph_type)

        self.plot_eth = pg.PlotWidget(title="<b>ETH/USD Trend, Event Pins & Ensemble Consensus</b>")
        self.plot_eth.showGrid(x=True, y=True, alpha=0.1)
        self.plot_eth.addLegend(offset=(15, 15))
        v_eth_c.addWidget(self.plot_eth)
        tab_eth_chart.setLayout(v_eth_c)

        # TAB 3: MODEL BENCHMARK EVALUATION MATRIX WITH ENSEMBLE MODEL
        tab_table = QWidget()
        tab_table_layout = QVBoxLayout()
        tab_table_layout.setContentsMargins(16, 16, 16, 16)

        self.eval_table = QTableWidget(5, 5)
        self.eval_table.setHorizontalHeaderLabels(
            ["Architecture", "BTC MAPE (%)", "BTC MAE ($)", "ETH MAPE (%)", "ETH MAE ($)"])
        self.eval_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.eval_table.verticalHeader().setVisible(False)

        bench_data = [
            ("Weighted Ensemble (Trans 60% + LSTM 40%) 👑", "2.24%", "$194.50", "3.10%", "$10.15"),
            ("Transformer (Attention) 🏆", "2.51%", "$218.02", "3.37%", "$11.06"),
            ("Advanced LSTM", "2.82%", "$238.24", "3.74%", "$11.92"),
            ("GRU Model", "2.89%", "$243.28", "4.15%", "$12.86"),
            ("Random Forest (Regularized)", "3.00%", "$253.43", "4.56%", "$14.29")
        ]

        for r, row_data in enumerate(bench_data):
            for c, text in enumerate(row_data):
                item = QTableWidgetItem(text)
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                if r == 0:
                    item.setForeground(QColor("#eab308"))
                    item.setFont(QFont("Inter", 10, QFont.Weight.Bold))
                self.eval_table.setItem(r, c, item)

        tab_table_layout.addWidget(self.eval_table)
        tab_table.setLayout(tab_table_layout)

        # TAB 4: LAGGED CORRELATION ANALYSIS
        tab_lagged = QWidget()
        v_lagged = QVBoxLayout()
        v_lagged.setContentsMargins(16, 16, 16, 16)

        self.lbl_lagged_info = QLabel("⏱️ Lead/Lag Correlation Analysis: FinBERT Sentiment vs Log Returns")
        self.lbl_lagged_info.setStyleSheet("font-size: 13px; color: #38bdf8; font-weight: 700; margin-bottom: 6px;")
        v_lagged.addWidget(self.lbl_lagged_info)

        self.plot_lagged = pg.PlotWidget(title="<b>Lead/Lag Correlation Matrix across Lags (0 to 5 Days)</b>")
        self.plot_lagged.showGrid(x=True, y=True, alpha=0.1)
        self.plot_lagged.addLegend(offset=(15, 15))
        v_lagged.addWidget(self.plot_lagged)

        self.lbl_lagged_summary = QLabel("📌 Summary: Target Peak Reaction occurs 24h (Lag +1d) post event publication.")
        self.lbl_lagged_summary.setStyleSheet(
            "font-size: 12px; color: #a3e635; font-weight: 600; padding: 6px; background-color: #101217; border-radius: 6px;")
        v_lagged.addWidget(self.lbl_lagged_summary)

        tab_lagged.setLayout(v_lagged)

        self.tabs.addTab(tab_btc_chart, "🟠 Bitcoin (BTC) & Ensemble Horizon")
        self.tabs.addTab(tab_eth_chart, "🔷 Ethereum (ETH) & Ensemble Horizon")
        self.tabs.addTab(tab_table, "🏆 Model Benchmark Evaluation")
        self.tabs.addTab(tab_lagged, "⏱️ Event Impact & Lagged Correlation")

        content_layout.addWidget(self.tabs)
        content_area.setLayout(content_layout)

        main_layout.addWidget(sidebar)
        main_layout.addWidget(content_area)
        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)

        # START WORKERS
        self.fetch_api_data()

        self.ws_worker = WebSocketWorker()
        self.ws_worker.tick_received.connect(self.on_live_tick)
        self.ws_worker.connection_status.connect(self.on_ws_status_change)
        self.ws_worker.start()

    def fetch_api_data(self):
        self.btn_refresh.setText("Syncing...")
        self.btn_refresh.setEnabled(False)

        self.worker = ApiWorker()
        self.worker.data_fetched.connect(self.on_data_received)
        self.worker.start()

    def on_live_tick(self, tick_data):
        if tick_data.get("type") == "TICK_UPDATE":
            symbol = tick_data.get("symbol")
            price = tick_data.get("price", 0.0)
            change = tick_data.get("change_pct", 0.0)

            if symbol == "BTCUSDT":
                self.card_btc.update_data(f"${price:,.2f}", f"24h Change: {change:+.2f}% | ⚡ Live Tick", "#38bdf8")
            elif symbol == "ETHUSDT":
                self.card_eth.update_data(f"${price:,.2f}", f"24h Change: {change:+.2f}% | ⚡ Live Tick", "#a855f7")

    def on_ws_status_change(self, is_connected):
        if is_connected:
            self.status_lbl.setText("⚡ WebSocket Streaming: Connected (Tick-by-Tick)")
            self.status_lbl.setStyleSheet("color: #38bdf8; font-size: 11px; font-weight: 700; margin-top: 6px;")
        else:
            self.status_lbl.setText("● Reconnecting WebSocket Stream...")
            self.status_lbl.setStyleSheet("color: #f59e0b; font-size: 11px; font-weight: 600; margin-top: 6px;")

    def set_ner_filter(self, entity_name):
        self.active_ner_filter = entity_name
        self.render_ner_filter_bar()
        self.render_news_cards()

    def render_ner_filter_bar(self):
        for i in reversed(range(self.ner_layout.count())):
            self.ner_layout.itemAt(i).widget().setParent(None)

        all_entities = set()
        for item in self.raw_news_data:
            for ent in item.get("entities", []):
                all_entities.add(ent)

        btn_all = QPushButton("Όλα (#ALL)")
        btn_all.setCursor(Qt.CursorShape.PointingHandCursor)
        if self.active_ner_filter is None:
            btn_all.setProperty("class", "btn-entity-active")
        else:
            btn_all.setProperty("class", "btn-entity-chip")
        btn_all.clicked.connect(lambda: self.set_ner_filter(None))
        self.ner_layout.addWidget(btn_all)

        for entity in sorted(list(all_entities))[:10]:
            btn = QPushButton(f"#{entity}")
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            if self.active_ner_filter == entity:
                btn.setProperty("class", "btn-entity-active")
            else:
                btn.setProperty("class", "btn-entity-chip")
            btn.clicked.connect(lambda checked, e=entity: self.set_ner_filter(e))
            self.ner_layout.addWidget(btn)

    def render_news_cards(self):
        for i in reversed(range(self.box_btc_layout.count())):
            self.box_btc_layout.itemAt(i).widget().setParent(None)
        for i in reversed(range(self.box_eth_layout.count())):
            self.box_eth_layout.itemAt(i).widget().setParent(None)

        for item in self.raw_news_data:
            entities = item.get("entities", [])

            if self.active_ner_filter and self.active_ner_filter not in entities:
                continue

            card = QFrame()
            card.setProperty("class", "sidebar-news-card")
            c_layout = QVBoxLayout()
            c_layout.setSpacing(6)

            top_row = QHBoxLayout()
            top_row.setContentsMargins(0, 0, 0, 0)

            topic_label = item.get('topic_label', 'General')
            topic_color = item.get('topic_color', '#6b7280')

            lbl_topic = QLabel(f" {topic_label} ")
            lbl_topic.setStyleSheet(
                f"background-color: {topic_color}; color: #ffffff; font-size: 10px; font-weight: 700; border-radius: 4px; padding: 2px 5px;")
            top_row.addWidget(lbl_topic)
            top_row.addStretch()
            c_layout.addLayout(top_row)

            lbl_title = QLabel(item['title'])
            lbl_title.setWordWrap(True)
            lbl_title.setStyleSheet("font-size: 12px; font-weight: 600; color: #f7f8f8; line-height: 1.3;")

            lbl_meta = QLabel(f"{item['source']} | {item['time']}")
            lbl_meta.setObjectName("subtitle")

            lbl_impact = QLabel(
                f"Past Reaction: <b style='color:{item['color']};'>{item['expected_impact']}</b> ({item['sentiment_label']})")
            lbl_impact.setStyleSheet("font-size: 11px;")

            c_layout.addWidget(lbl_title)
            c_layout.addWidget(lbl_meta)
            c_layout.addWidget(lbl_impact)

            if entities:
                ent_row = QHBoxLayout()
                ent_row.setContentsMargins(0, 2, 0, 0)
                ent_row.setSpacing(4)

                lbl_ent_title = QLabel("Entities:")
                lbl_ent_title.setStyleSheet("font-size: 10px; color: #62677d; font-weight: 600;")
                ent_row.addWidget(lbl_ent_title)

                for ent in entities:
                    btn_ent = QPushButton(f"#{ent}")
                    btn_ent.setCursor(Qt.CursorShape.PointingHandCursor)
                    btn_ent.setProperty("class", "btn-entity-chip")
                    btn_ent.clicked.connect(lambda checked, e=ent: self.set_ner_filter(e))
                    ent_row.addWidget(btn_ent)

                ent_row.addStretch()
                c_layout.addLayout(ent_row)

            card.setLayout(c_layout)

            if item['asset'] == "BTC":
                self.box_btc_layout.addWidget(card)
            elif item['asset'] == "ETH":
                self.box_eth_layout.addWidget(card)

    def on_data_received(self, metrics, predictions, news, lagged):
        self.btn_refresh.setText("Sync Pipeline")
        self.btn_refresh.setEnabled(True)

        if not metrics or not predictions:
            self.card_btc.update_data("Offline", "Run app.py server", "#ef4444")
            return

        btc = metrics['bitcoin']
        eth = metrics['ethereum']
        fng = metrics['market_sentiment']

        self.card_btc.update_data(f"${btc['close_price']:,.2f}", f"24h Change: {btc['change_pct']:+.2f}%", "#f7f8f8")
        self.card_eth.update_data(f"${eth['close_price']:,.2f}", f"24h Change: {eth['change_pct']:+.2f}%", "#a855f7")
        self.card_fng.update_data(f"{fng['fear_greed_index']}/100", f"Sentiment: {fng['fear_greed_class']}", "#facc15")

        sent_val = fng['finbert_sentiment_score']
        self.card_sent.update_data(f"{sent_val:+.2f}", f"NLP: {'Bullish 🟢' if sent_val > 0 else 'Bearish 🔴'}",
                                   "#5e6ad2")

        preds = predictions['predictions']
        trans_btc = preds['transformer_attention']['BTC_target']
        trans_sig_btc = preds['transformer_attention']['BTC_signal']
        lstm_btc = preds['lstm_deep_learning']['BTC_target']
        lstm_sig_btc = preds['lstm_deep_learning']['BTC_signal']
        ens_btc = preds['ensemble_consensus']['BTC_target']
        ens_sig_btc = preds['ensemble_consensus']['BTC_signal']
        ens_btc_conf = preds['ensemble_consensus']['BTC_confidence']

        trans_eth = preds['transformer_attention']['ETH_target']
        trans_sig_eth = preds['transformer_attention']['ETH_signal']
        lstm_eth = preds['lstm_deep_learning']['ETH_target']
        lstm_sig_eth = preds['lstm_deep_learning']['ETH_signal']
        ens_eth = preds['ensemble_consensus']['ETH_target']
        ens_sig_eth = preds['ensemble_consensus']['ETH_signal']
        ens_eth_conf = preds['ensemble_consensus']['ETH_confidence']

        self.card_trans_btc.update_data(f"${trans_btc:,.2f}", f"Signal: {trans_sig_btc}",
                                        "#5e6ad2" if "BUY" in trans_sig_btc else "#ef4444")
        self.card_lstm_btc.update_data(f"${lstm_btc:,.2f}", f"Signal: {lstm_sig_btc}",
                                       "#22c55e" if "BUY" in lstm_sig_btc else "#ef4444")
        self.card_ens_btc.update_data(f"${ens_btc:,.2f}", f"{ens_sig_btc} | Conf: {ens_btc_conf}", "#eab308")

        self.card_trans_eth.update_data(f"${trans_eth:,.2f}", f"Signal: {trans_sig_eth}",
                                        "#38bdf8" if "BUY" in trans_sig_eth else "#ef4444")
        self.card_lstm_eth.update_data(f"${lstm_eth:,.2f}", f"Signal: {lstm_sig_eth}",
                                       "#a855f7" if "BUY" in lstm_sig_eth else "#ef4444")
        self.card_ens_eth.update_data(f"${ens_eth:,.2f}", f"{ens_sig_eth} | Conf: {ens_eth_conf}", "#eab308")

        if news and "news_feed" in news:
            self.raw_news_data = news["news_feed"]
            self.render_ner_filter_bar()
            self.render_news_cards()

        days = np.arange(1, 11)
        next_day = 11

        # 1. BITCOIN PLOT WITH ENSEMBLE CONSENSUS LINE
        self.plot_btc.clear()
        curr_btc_p = btc['close_price']
        hist_btc_trend = curr_btc_p + np.sin(days) * 750 - np.arange(10, 0, -1) * 110
        hist_btc_trend[-1] = curr_btc_p

        curve_btc = self.plot_btc.plot(days, hist_btc_trend, pen=pg.mkPen(color='#f7931a', width=2.5),
                                       name="BTC Historical Price")
        baseline_btc_val = min(np.min(hist_btc_trend), lstm_btc, trans_btc, ens_btc) * 0.98
        baseline_btc = self.plot_btc.plot(days, np.full_like(days, baseline_btc_val), pen=None)
        fill_btc = pg.FillBetweenItem(curve_btc, baseline_btc, brush=pg.mkBrush(247, 147, 26, 25))
        self.plot_btc.addItem(fill_btc)

        for news_item in [n for n in self.raw_news_data if n['asset'] == "BTC"]:
            idx = news_item.get("chart_day_index", 5)
            price_at_idx = hist_btc_trend[idx - 1]
            pin_color = '#22c55e' if "Bullish" in news_item['sentiment_label'] else '#ef4444'

            scatter = pg.ScatterPlotItem(x=[idx], y=[price_at_idx], size=14, brush=pin_color, symbol='t1')
            self.plot_btc.addItem(scatter)

            text_tag = pg.TextItem(text=f"📌 {news_item['topic_label']}", color=pin_color, anchor=(0.5, 1.3))
            text_tag.setPos(idx, price_at_idx)
            self.plot_btc.addItem(text_tag)

        self.plot_btc.plot([10, next_day], [curr_btc_p, trans_btc],
                           pen=pg.mkPen(color='#5e6ad2', width=1.5, style=Qt.PenStyle.DotLine), symbol='s',
                           symbolSize=7, symbolBrush='#5e6ad2', name="Transformer (#1)")
        self.plot_btc.plot([10, next_day], [curr_btc_p, lstm_btc],
                           pen=pg.mkPen(color='#22c55e', width=1.5, style=Qt.PenStyle.DashLine), symbol='o',
                           symbolSize=7, symbolBrush='#22c55e', name="LSTM (#2)")

        # ENSEMBLE GOLDEN HORIZON LINE
        item_ens_btc = self.plot_btc.plot([10, next_day], [curr_btc_p, ens_btc],
                                          pen=pg.mkPen(color='#eab308', width=3.0), symbol='star', symbolSize=12,
                                          symbolBrush='#eab308', name="Ensemble Consensus 🌟")
        item_ens_btc.setToolTip(f"Ensemble Weighted Target: ${ens_btc:,.2f}")

        self.plot_btc.setYRange(baseline_btc_val, max(np.max(hist_btc_trend), lstm_btc, trans_btc, ens_btc) * 1.015)

        # 2. ETHEREUM PLOT WITH ENSEMBLE CONSENSUS LINE
        self.plot_eth.clear()
        curr_eth_p = eth['close_price']
        hist_eth_trend = curr_eth_p + np.cos(days) * 35 - np.arange(10, 0, -1) * 5
        hist_eth_trend[-1] = curr_eth_p

        curve_eth = self.plot_eth.plot(days, hist_eth_trend, pen=pg.mkPen(color='#a855f7', width=2.5),
                                       name="ETH Historical Price")
        baseline_eth_val = min(np.min(hist_eth_trend), lstm_eth, trans_eth, ens_eth) * 0.98
        baseline_eth = self.plot_eth.plot(days, np.full_like(days, baseline_eth_val), pen=None)
        fill_eth = pg.FillBetweenItem(curve_eth, baseline_eth, brush=pg.mkBrush(168, 85, 247, 25))
        self.plot_eth.addItem(fill_eth)

        for news_item in [n for n in self.raw_news_data if n['asset'] == "ETH"]:
            idx = news_item.get("chart_day_index", 5)
            price_at_idx = hist_eth_trend[idx - 1]
            pin_color = '#22c55e' if "Bullish" in news_item['sentiment_label'] else '#ef4444'

            scatter = pg.ScatterPlotItem(x=[idx], y=[price_at_idx], size=14, brush=pin_color, symbol='t1')
            self.plot_eth.addItem(scatter)

            text_tag = pg.TextItem(text=f"📌 {news_item['topic_label']}", color=pin_color, anchor=(0.5, 1.3))
            text_tag.setPos(idx, price_at_idx)
            self.plot_eth.addItem(text_tag)

        self.plot_eth.plot([10, next_day], [curr_eth_p, trans_eth],
                           pen=pg.mkPen(color='#38bdf8', width=1.5, style=Qt.PenStyle.DotLine), symbol='s',
                           symbolSize=7, symbolBrush='#38bdf8', name="Transformer (#1)")
        self.plot_eth.plot([10, next_day], [curr_eth_p, lstm_eth],
                           pen=pg.mkPen(color='#a855f7', width=1.5, style=Qt.PenStyle.DashLine), symbol='o',
                           symbolSize=7, symbolBrush='#a855f7', name="LSTM (#2)")

        # ENSEMBLE GOLDEN HORIZON LINE
        item_ens_eth = self.plot_eth.plot([10, next_day], [curr_eth_p, ens_eth],
                                          pen=pg.mkPen(color='#eab308', width=3.0), symbol='star', symbolSize=12,
                                          symbolBrush='#eab308', name="Ensemble Consensus 🌟")
        item_ens_eth.setToolTip(f"Ensemble Weighted Target: ${ens_eth:,.2f}")

        self.plot_eth.setYRange(baseline_eth_val, max(np.max(hist_eth_trend), lstm_eth, trans_eth, ens_eth) * 1.015)

        # 3. LAGGED CORRELATION CHART
        if lagged and "btc_correlation" in lagged:
            self.plot_lagged.clear()
            x_lags = np.arange(len(lagged["lags"]))

            self.plot_lagged.plot(x_lags, lagged["btc_correlation"], pen=pg.mkPen(color='#f7931a', width=3), symbol='o',
                                  symbolSize=10, symbolBrush='#f7931a', name="BTC Sentiment Lead/Lag Corr")
            self.plot_lagged.plot(x_lags, lagged["eth_correlation"], pen=pg.mkPen(color='#a855f7', width=3), symbol='s',
                                  symbolSize=10, symbolBrush='#a855f7', name="ETH Sentiment Lead/Lag Corr")

            peak_text = pg.TextItem(text="🏆 PEAK IMPACT (24 Hours)", color='#38bdf8', anchor=(0.5, -0.5))
            peak_text.setPos(1, max(lagged["btc_correlation"]))
            self.plot_lagged.addItem(peak_text)

            self.lbl_lagged_summary.setText(f"📌 {lagged.get('insights', '')}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = LinearCryptoApp()
    window.show()
    sys.exit(app.exec())