import os
import sys
import json
import logging
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time

# Konfiguriere Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Trading AI Dashboard (Simple)",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialisiere Session-State-Variablen
if 'initialized' not in st.session_state:
    st.session_state.initialized = False
if 'selected_stocks' not in st.session_state:
    st.session_state.selected_stocks = []
if 'historical_data' not in st.session_state:
    st.session_state.historical_data = {}
if 'last_update' not in st.session_state:
    st.session_state.last_update = datetime.now()


# Lade Konfiguration
def load_config():
    config_path = os.path.join('config', 'settings.json')
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            return json.load(f)
    else:
        # Verwende Umgebungsvariablen als Fallback
        return {
            'ALPACA_API_KEY': os.environ.get('ALPACA_API_KEY', ''),
            'ALPACA_API_SECRET': os.environ.get('ALPACA_API_SECRET', ''),
            'ALPACA_API_BASE_URL': os.environ.get('ALPACA_API_BASE_URL', 'https://paper-api.alpaca.markets'),
            'TRADING_SYMBOLS': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
        }


# Generiere simulierte Daten für Demo-Zwecke
def generate_demo_data(symbol, days=30):
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)

    # Erzeuge Datumsreihe
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')

    # Basierend auf dem Symbol verschiedene Startwerte verwenden
    if symbol == 'AAPL':
        price = 150.0
    elif symbol == 'MSFT':
        price = 300.0
    elif symbol == 'GOOGL':
        price = 120.0
    elif symbol == 'AMZN':
        price = 100.0
    elif symbol == 'TSLA':
        price = 250.0
    else:
        price = 100.0

    data = []

    for _ in date_range:
        # Generiere zufällige Preisänderung
        change_percent = np.random.normal(0, 0.015)  # Normalverteilte Änderung mit Mittelwert 0 und StdAbw 1.5%
        price = price * (1 + change_percent)

        # Generiere OHLC-Daten
        high = price * (1 + abs(np.random.normal(0, 0.005)))
        low = price * (1 - abs(np.random.normal(0, 0.005)))
        open_price = low + np.random.random() * (high - low)
        close = low + np.random.random() * (high - low)
        volume = int(np.random.normal(1000000, 300000))

        data.append({
            'open': open_price,
            'high': high,
            'low': low,
            'close': close,
            'volume': volume
        })

    # Erstelle DataFrame
    df = pd.DataFrame(data, index=date_range)
    return df


# Erstelle Candlestick-Chart mit Plotly
def create_candlestick_chart(data, title):
    fig = go.Figure(data=[go.Candlestick(
        x=data.index,
        open=data['open'],
        high=data['high'],
        low=data['low'],
        close=data['close'],
        name='Price'
    )])

    # Füge Volumen als Balkendiagramm hinzu
    if 'volume' in data.columns:
        fig.add_trace(go.Bar(
            x=data.index,
            y=data['volume'],
            name='Volume',
            marker_color='rgba(0, 0, 255, 0.3)',
            opacity=0.3,
            yaxis='y2'
        ))

    # Update layout
    fig.update_layout(
        title=title,
        xaxis_title='Date',
        yaxis_title='Price ($)',
        yaxis2=dict(
            title='Volume',
            overlaying='y',
            side='right',
            showgrid=False
        ),
        height=600,
        margin=dict(l=50, r=50, t=50, b=50)
    )

    return fig


# Hauptanwendungslogik
def main():
    # Sidebar
    st.sidebar.title("Trading AI Dashboard (Simple)")

    # Lade Konfiguration
    config = load_config()

    # Status-Indikatoren
    st.sidebar.header("Status")

    # In dieser vereinfachten Version verwenden wir simulierte Daten
    st.sidebar.info("Hinweis: Dies ist eine vereinfachte Version mit simulierten Daten.")

    # Aktienauswahl
    st.sidebar.header("Aktienauswahl")
    default_stocks = config.get('TRADING_SYMBOLS', ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA'])
    selected_stocks = st.sidebar.multiselect(
        "Wähle Aktien zum Verfolgen",
        options=default_stocks + ['NVDA', 'META', 'NFLX', 'PYPL', 'DIS', 'INTC', 'AMD', 'CSCO', 'ADBE', 'ORCL'],
        default=default_stocks[:3]  # Standard: erste 3 Aktien
    )

    # Speichere ausgewählte Aktien im Session-State
    if selected_stocks != st.session_state.selected_stocks:
        st.session_state.selected_stocks = selected_stocks

        # Generiere simulierte Daten für ausgewählte Aktien
        st.session_state.historical_data = {}
        for symbol in selected_stocks:
            st.session_state.historical_data[symbol] = generate_demo_data(symbol)

    # Hauptinhalt
    st.title("Aktien-Dashboard (Demo-Modus)")

    # Zeige ausgewählte Aktien an
    if not st.session_state.selected_stocks:
        st.warning("Bitte wähle mindestens eine Aktie zum Verfolgen aus.")
        return

    # Erstelle Tabs für jede Aktie
    tabs = st.tabs(st.session_state.selected_stocks)

    # Zeige Daten für jede Aktie in ihrem Tab an
    for i, symbol in enumerate(st.session_state.selected_stocks):
        with tabs[i]:
            st.header(f"{symbol} Dashboard")

            # Erstelle Spalten für die Organisation des Inhalts
            col1, col2 = st.columns([3, 1])

            with col1:
                # Zeige historischen Candlestick-Chart an
                if symbol in st.session_state.historical_data:
                    st.subheader("Historische Preisdaten")
                    fig = create_candlestick_chart(st.session_state.historical_data[symbol],
                                                   f"{symbol} - Historischer Preis")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning(f"Keine historischen Daten verfügbar für {symbol}")

            with col2:
                # Zeige neuestes Angebot an (simuliert)
                st.subheader("Neuestes Angebot")
                latest_data = st.session_state.historical_data[symbol].iloc[-1]

                st.metric("Letzte Schlusskurs", f"${latest_data['close']:.2f}")
                st.metric("Volumen", f"{latest_data['volume']:,}")

                # Berechne Änderung im Vergleich zum Vortag
                if len(st.session_state.historical_data[symbol]) > 1:
                    previous_close = st.session_state.historical_data[symbol].iloc[-2]['close']
                    change = latest_data['close'] - previous_close
                    percent_change = (change / previous_close) * 100
                    st.metric("Änderung", f"${change:.2f} ({percent_change:.2f}%)")

    # Zeige Zeit der letzten Aktualisierung an
    st.sidebar.text(f"Zuletzt aktualisiert: {st.session_state.last_update.strftime('%H:%M:%S')}")

    # Aktualisieren-Button
    if st.sidebar.button("Aktualisieren"):
        # Simuliere Aktualisierung durch Neugenerieren der Daten
        for symbol in st.session_state.selected_stocks:
            st.session_state.historical_data[symbol] = generate_demo_data(symbol)

        st.session_state.last_update = datetime.now()
        st.experimental_rerun()


if __name__ == "__main__":
    main()