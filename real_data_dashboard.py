import os
import sys
import json
import logging
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
from datetime import datetime, timedelta

# Füge den Root-Path zum PYTHONPATH hinzu, um die Backend-Module zu importieren
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
sys.path.append(root_dir)

# Importiere Alpaca API
from backend.alpaca_api import AlpacaAPI

# Konfiguriere Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Trading AI Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialisiere Session-State-Variablen
if 'api_connected' not in st.session_state:
    st.session_state.api_connected = False
if 'alpaca_api' not in st.session_state:
    st.session_state.alpaca_api = None
if 'selected_stocks' not in st.session_state:
    st.session_state.selected_stocks = []
if 'historical_data' not in st.session_state:
    st.session_state.historical_data = {}
if 'account_info' not in st.session_state:
    st.session_state.account_info = None
if 'positions' not in st.session_state:
    st.session_state.positions = []
if 'last_update' not in st.session_state:
    st.session_state.last_update = datetime.now()
if 'ai_predictions' not in st.session_state:
    st.session_state.ai_predictions = {}
if 'chart_timeframe' not in st.session_state:
    st.session_state.chart_timeframe = '1Day'
if 'chart_period' not in st.session_state:
    st.session_state.chart_period = 30
if 'show_indicators' not in st.session_state:
    st.session_state.show_indicators = True


# Lade Konfiguration
def load_config():
    # Versuche, die Konfigurationsdatei an verschiedenen möglichen Orten zu finden
    possible_paths = [
        os.path.join(root_dir, 'config', 'settings.json'),  # Trading/config/settings.json
        os.path.join('config', 'settings.json'),  # config/settings.json
        os.path.join(os.getcwd(), 'config', 'settings.json'),  # aktuelles Verzeichnis/config/settings.json
        'settings.json'  # settings.json im aktuellen Verzeichnis
    ]

    for config_path in possible_paths:
        if os.path.exists(config_path):
            logger.info(f"Konfigurationsdatei gefunden in: {config_path}")
            with open(config_path, 'r') as f:
                return json.load(f)

    # Wenn keine Konfigurationsdatei gefunden wurde, zeige Warnung und verwende Umgebungsvariablen
    logger.warning("Keine settings.json gefunden. Verwende Umgebungsvariablen.")

    # Zeige verfügbare Umgebungsvariablen (ohne sensible Daten)
    env_vars = {k: '***' if 'KEY' in k or 'SECRET' in k else v
                for k, v in os.environ.items() if k.startswith('ALPACA')}
    logger.info(f"Verfügbare Alpaca-Umgebungsvariablen: {env_vars}")

    # Verwende Umgebungsvariablen als Fallback
    return {
        'ALPACA_API_KEY': os.environ.get('ALPACA_API_KEY', ''),
        'ALPACA_API_SECRET': os.environ.get('ALPACA_API_SECRET', ''),
        'ALPACA_API_BASE_URL': os.environ.get('ALPACA_API_BASE_URL', 'https://paper-api.alpaca.markets'),
        'TRADING_SYMBOLS': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
    }


# Initialisiere Alpaca API
def initialize_alpaca_api(config):
    try:
        # Extrahiere die API-Schlüssel aus der Konfiguration
        api_key = config.get('ALPACA_API_KEY', '')
        api_secret = config.get('ALPACA_API_SECRET', '')
        base_url = config.get('ALPACA_API_BASE_URL', 'https://paper-api.alpaca.markets')

        if not api_key or not api_secret:
            # Zeige Fehlermeldung in der UI
            st.sidebar.error("API-Schlüssel fehlen. Bitte überprüfe deine settings.json oder Umgebungsvariablen.")
            logger.error("API-Schlüssel fehlen in der Konfiguration und den Umgebungsvariablen")
            return False

        logger.info(f"Versuche Verbindung mit Alpaca API aufzubauen...")

        # Setze die Umgebungsvariablen, die von der API intern verwendet werden
        os.environ['ALPACA_API_KEY'] = api_key
        os.environ['ALPACA_API_SECRET'] = api_secret
        os.environ['ALPACA_API_BASE_URL'] = base_url

        # Erstelle Alpaca API-Client
        # Wir erstellen ein temporäres Config-Objekt für die API-Klasse
        temp_config = {
            'ALPACA_API_KEY': api_key,
            'ALPACA_API_SECRET': api_secret,
            'ALPACA_API_BASE_URL': base_url
        }

        # Schreibe temp_config in eine temporäre Datei
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as temp:
            json.dump(temp_config, temp)
            temp_config_path = temp.name

        try:
            # Erstelle die API mit dem Pfad zur temporären Konfigurationsdatei
            alpaca_api = AlpacaAPI(config_path=temp_config_path)

            # Lösche die temporäre Datei
            os.unlink(temp_config_path)

            # Teste die Verbindung durch Abrufen von Kontoinformationen
            account_info = alpaca_api.get_account_info()

            # Speichern im Session State
            st.session_state.api_connected = True
            st.session_state.alpaca_api = alpaca_api
            st.session_state.account_info = account_info

            # Versuche, aktuelle Positionen zu bekommen
            try:
                positions = alpaca_api.get_positions()
                st.session_state.positions = positions
            except Exception as e:
                logger.warning(f"Konnte Positionen nicht abrufen: {e}")
                st.session_state.positions = []

            return True
        except Exception as inner_e:
            logger.error(f"Fehler beim Erstellen der API mit temporärer Konfiguration: {inner_e}")

            # Versuche es noch einmal mit direktem Ansatz
            try:
                from alpaca_trade_api import REST
                direct_api = REST(api_key, api_secret, base_url)

                # Wenn wir hierher kommen, hat die direkte Verbindung funktioniert
                st.sidebar.success("Direkte Verbindung zur Alpaca API hergestellt!")
                logger.info("Erfolgreiche direkte Verbindung zur Alpaca API")

                # Erstelle eine Wrapper-Klasse, die unsere AlpacaAPI-Schnittstelle nachahmt
                class DirectAlpacaAPI:
                    def __init__(self, api):
                        self.api = api

                    def get_account_info(self):
                        return self.api.get_account()

                    def get_positions(self):
                        return self.api.list_positions()

                    def get_historical_bars(self, symbols, timeframe='1Day', start=None, end=None, limit=1000):
                        if not start:
                            start = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
                        if not end:
                            end = datetime.now().strftime('%Y-%m-%d')

                        result = {}
                        for symbol in symbols:
                            try:
                                bars = self.api.get_bars(
                                    symbol,
                                    timeframe,
                                    start=start,
                                    end=end,
                                    limit=limit,
                                    adjustment='raw',
                                    feed='iex'
                                ).df

                                if not bars.empty:
                                    if isinstance(bars.index, pd.MultiIndex):
                                        bars = bars.reset_index()
                                        if 'symbol' in bars.columns:
                                            bars = bars.drop('symbol', axis=1)
                                        if 'timestamp' in bars.columns:
                                            bars = bars.set_index('timestamp')

                                    result[symbol] = bars
                            except Exception as e:
                                logger.warning(f"Error fetching data for {symbol}: {e}")

                        return result

                # Erstelle unsere Wrapper-Instanz
                alpaca_api = DirectAlpacaAPI(direct_api)

                # Teste die Verbindung
                account_info = alpaca_api.get_account_info()

                # Speichern im Session State
                st.session_state.api_connected = True
                st.session_state.alpaca_api = alpaca_api
                st.session_state.account_info = account_info

                # Versuche, aktuelle Positionen zu bekommen
                try:
                    positions = alpaca_api.get_positions()
                    st.session_state.positions = positions
                except Exception as e:
                    logger.warning(f"Konnte Positionen nicht abrufen: {e}")
                    st.session_state.positions = []

                return True
            except Exception as direct_e:
                logger.error(f"Direkter Verbindungsversuch fehlgeschlagen: {direct_e}")
                return False
    except Exception as e:
        logger.error(f"Fehler beim Initialisieren der Alpaca API: {e}")
        # Zeige detailliertere Fehlermeldung
        error_msg = str(e)
        if "credentials" in error_msg.lower():
            st.sidebar.error("API-Anmeldedaten nicht gefunden oder ungültig. Bitte überprüfe deine API-Schlüssel.")
        elif "connection" in error_msg.lower():
            st.sidebar.error("Verbindungsproblem mit Alpaca. Bitte überprüfe deine Internetverbindung.")
        else:
            st.sidebar.error(f"Fehler: {error_msg}")

        st.session_state.api_connected = False
        return False


# Funktion zum Abrufen historischer Daten
def fetch_historical_data(alpaca_api, symbols, timeframe='1Day', days=30):
    start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
    end_date = datetime.now().strftime('%Y-%m-%d')

    try:
        # Holen Sie Balken-Daten mit 'iex' Feed für kostenlose Konten
        return alpaca_api.get_historical_bars(
            symbols,
            timeframe=timeframe,
            start=start_date,
            end=end_date
        )
    except Exception as e:
        logger.error(f"Fehler beim Abrufen historischer Daten: {e}")
        return {}


# Funktion zum Berechnen technischer Indikatoren
def calculate_indicators(data):
    """Berechne technische Indikatoren für die gegebenen Preisdaten."""
    if data.empty:
        return data

    # Erstelle eine Kopie, um die Originaldaten nicht zu verändern
    df = data.copy()

    # Simple Moving Averages (SMA)
    df['sma20'] = df['close'].rolling(window=20).mean()
    df['sma50'] = df['close'].rolling(window=50).mean()
    df['sma200'] = df['close'].rolling(window=200).mean()

    # Relative Strength Index (RSI)
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    df['rsi'] = 100 - (100 / (1 + rs))

    # MACD (Moving Average Convergence Divergence)
    df['ema12'] = df['close'].ewm(span=12, adjust=False).mean()
    df['ema26'] = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = df['ema12'] - df['ema26']
    df['signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['histogram'] = df['macd'] - df['signal']

    # Bollinger Bands
    df['middle_band'] = df['sma20']
    df['std'] = df['close'].rolling(window=20).std()
    df['upper_band'] = df['middle_band'] + (df['std'] * 2)
    df['lower_band'] = df['middle_band'] - (df['std'] * 2)

    return df


# Funktion zum Erstellen eines Candlestick-Charts mit technischen Indikatoren
def create_advanced_chart(data, symbol, timeframe, show_volume=True, show_indicators=True):
    # Berechne Indikatoren
    data_with_indicators = calculate_indicators(data)

    # Erstelle Subplot mit 2 Zeilen, 1 Spalte (Preis oben, Volumen unten)
    fig = make_subplots(
        rows=2 if show_volume else 1,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.8, 0.2] if show_volume else [1]
    )

    # Füge Candlestick-Chart hinzu
    fig.add_trace(
        go.Candlestick(
            x=data.index,
            open=data['open'],
            high=data['high'],
            low=data['low'],
            close=data['close'],
            name=symbol
        ),
        row=1, col=1
    )

    # Füge technische Indikatoren hinzu, wenn aktiviert
    if show_indicators:
        # Moving Averages
        fig.add_trace(
            go.Scatter(
                x=data_with_indicators.index,
                y=data_with_indicators['sma20'],
                name='SMA 20',
                line=dict(color='blue', width=1)
            ),
            row=1, col=1
        )

        fig.add_trace(
            go.Scatter(
                x=data_with_indicators.index,
                y=data_with_indicators['sma50'],
                name='SMA 50',
                line=dict(color='orange', width=1)
            ),
            row=1, col=1
        )

        # Bollinger Bands
        fig.add_trace(
            go.Scatter(
                x=data_with_indicators.index,
                y=data_with_indicators['upper_band'],
                name='Upper Band',
                line=dict(color='rgba(0, 128, 0, 0.3)', width=1),
                mode='lines'
            ),
            row=1, col=1
        )

        fig.add_trace(
            go.Scatter(
                x=data_with_indicators.index,
                y=data_with_indicators['lower_band'],
                name='Lower Band',
                line=dict(color='rgba(0, 128, 0, 0.3)', width=1),
                fill='tonexty',
                fillcolor='rgba(0, 128, 0, 0.05)',
                mode='lines'
            ),
            row=1, col=1
        )

    # Füge Volumen-Diagramm hinzu, wenn aktiviert
    if show_volume and 'volume' in data.columns:
        # Färbe die Volumenbalken basierend auf der Preisbewegung
        colors = ['red' if data.iloc[i]['close'] < data.iloc[i]['open']
                  else 'green' for i in range(len(data))]

        fig.add_trace(
            go.Bar(
                x=data.index,
                y=data['volume'],
                name='Volume',
                marker_color=colors,
                opacity=0.5
            ),
            row=2, col=1
        )

    # Update Layout
    timeframe_name = {
        '1Min': '1 Minute',
        '5Min': '5 Minutes',
        '15Min': '15 Minutes',
        '1Hour': '1 Hour',
        '1Day': 'Daily'
    }.get(timeframe, timeframe)

    fig.update_layout(
        title=f"{symbol} - {timeframe_name} Chart",
        xaxis_title="Date",
        yaxis_title="Price ($)",
        yaxis2_title="Volume" if show_volume else None,
        height=700,
        xaxis_rangeslider_visible=False,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )

    return fig, data_with_indicators


# Funktion zum Simulieren von KI-Vorhersagen (Platzhalter für zukünftige Implementierung)
def simulate_ai_predictions(symbol, data):
    """Simuliere KI-Vorhersagen für das gegebene Symbol und die Daten."""
    if data.empty:
        return {}

    # Verwende die letzten 20 Tage Daten für die Vorhersage
    recent_data = data.iloc[-20:]

    # Einfache Simulation: Vorhersage "UP", wenn der 5-Tage-SMA über dem 20-Tage-SMA liegt
    sma5 = recent_data['close'].rolling(window=5).mean()
    sma20 = recent_data['close'].rolling(window=20).mean()

    # Hole die letzten Werte
    try:
        last_sma5 = sma5.iloc[-1]
        last_sma20 = sma20.iloc[-1]
        last_close = recent_data['close'].iloc[-1]

        if last_sma5 > last_sma20:
            prediction = 'KAUFEN'
            confidence = min(0.5 + (last_sma5 - last_sma20) / last_close, 0.95)
        elif last_sma5 < last_sma20:
            prediction = 'VERKAUFEN'
            confidence = min(0.5 + (last_sma20 - last_sma5) / last_close, 0.95)
        else:
            prediction = 'HALTEN'
            confidence = 0.5

        return {
            'prediction': prediction,
            'confidence': confidence,
            'timestamp': datetime.now(),
            'price_target': last_close * (
                1.05 if prediction == 'KAUFEN' else 0.95 if prediction == 'VERKAUFEN' else 1.0),
            'stop_loss': last_close * (0.97 if prediction == 'KAUFEN' else 1.03 if prediction == 'VERKAUFEN' else 1.0)
        }
    except:
        return {
            'prediction': 'UNBEKANNT',
            'confidence': 0.0,
            'timestamp': datetime.now(),
            'price_target': None,
            'stop_loss': None
        }


# Hauptanwendungslogik
def main():
    # Sidebar
    st.sidebar.title("Trading AI Dashboard")

    # Lade Konfiguration
    config = load_config()

    # Initialisiere Alpaca API, falls noch nicht geschehen
    if not st.session_state.api_connected:
        with st.sidebar.status("Verbinde zur Alpaca API..."):
            if initialize_alpaca_api(config):
                st.sidebar.success("Verbindung zur Alpaca API hergestellt!")
            else:
                st.sidebar.error("Fehler bei der Verbindung zur Alpaca API. Bitte überprüfe deine Anmeldedaten.")
                return

    # Zeige Kontoinformationen an
    if st.session_state.api_connected and st.session_state.account_info:
        account = st.session_state.account_info

        # Erstelle Karte für Kontoinformationen
        with st.sidebar.expander("Konto-Informationen", expanded=True):
            st.write(f"**ID:** {account.id}")
            st.write(f"**Bargeld:** ${float(account.cash):.2f}")
            st.write(f"**Portfoliowert:** ${float(account.equity):.2f}")
            st.write(f"**Kaufkraft:** ${float(account.buying_power):.2f}")
            st.write(f"**Status:** {account.status}")
            st.write(f"**Handelssperren:** {account.trading_blocked}")

            # Sicheres Abrufen des Erstellungsdatums
            try:
                # Falls created_at ein String ist, versuche es zu teilen
                if hasattr(account, 'created_at') and account.created_at:
                    if isinstance(account.created_at, str) and 'T' in account.created_at:
                        creation_date = account.created_at.split('T')[0]
                    else:
                        # Falls es kein String ist oder kein 'T' enthält
                        creation_date = str(account.created_at)
                    st.write(f"**Kontoerstellung:** {creation_date}")
            except Exception as e:
                logger.warning(f"Konnte Erstellungsdatum nicht formatieren: {e}")
                # Zeige das Rohdatum oder nichts an
                if hasattr(account, 'created_at'):
                    st.write(f"**Kontoerstellung:** {account.created_at}")

    # Aktienauswahl
    st.sidebar.header("Aktienauswahl")
    default_stocks = config.get('TRADING_SYMBOLS', ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA'])
    selected_stocks = st.sidebar.multiselect(
        "Wähle Aktien zum Verfolgen",
        options=default_stocks + ['NVDA', 'META', 'NFLX', 'PYPL', 'DIS', 'INTC', 'AMD', 'CSCO', 'ADBE', 'ORCL'],
        default=default_stocks[:3]  # Standard: erste 3 Aktien
    )

    # Chart-Einstellungen
    st.sidebar.header("Chart-Einstellungen")
    timeframe = st.sidebar.selectbox(
        "Zeitraum",
        options=['1Day', '1Hour', '15Min', '5Min', '1Min'],
        index=0  # Standard: 1Day
    )

    days = st.sidebar.slider(
        "Datenzeitraum (Tage)",
        min_value=1,
        max_value=90,
        value=30
    )

    # Indikatoren ein-/ausschalten
    show_indicators = st.sidebar.checkbox("Technische Indikatoren anzeigen", value=True)

    # Speichere Einstellungen im Session-State
    if timeframe != st.session_state.chart_timeframe or days != st.session_state.chart_period or show_indicators != st.session_state.show_indicators:
        st.session_state.chart_timeframe = timeframe
        st.session_state.chart_period = days
        st.session_state.show_indicators = show_indicators
        st.session_state.historical_data = {}  # Zurücksetzen der historischen Daten bei Änderung der Einstellungen

    # Speichere ausgewählte Aktien im Session-State
    if selected_stocks != st.session_state.selected_stocks:
        st.session_state.selected_stocks = selected_stocks
        st.session_state.historical_data = {}  # Zurücksetzen der historischen Daten bei Änderung der Aktienauswahl

    # Hauptinhalt
    st.title("Trading AI Dashboard")

    # Zeige ausgewählte Aktien an
    if not st.session_state.selected_stocks:
        st.warning("Bitte wähle mindestens eine Aktie zum Verfolgen aus.")
        return

    # Hole historische Daten, wenn noch nicht vorhanden
    if not st.session_state.historical_data and st.session_state.api_connected:
        with st.status("Hole historische Daten..."):
            st.session_state.historical_data = fetch_historical_data(
                st.session_state.alpaca_api,
                st.session_state.selected_stocks,
                timeframe=timeframe,
                days=days
            )

    # Erstelle Tabs für verschiedene Ansichten
    tab1, tab2, tab3 = st.tabs(["Aktien-Charts", "Portfolio", "KI-Vorhersagen"])

    with tab1:
        # Aktien-Charts-Ansicht
        if not st.session_state.historical_data:
            st.warning("Keine historischen Daten verfügbar. Bitte aktualisiere die Daten.")
        else:
            # Erstelle Aktienauswahl
            selected_stock = st.selectbox(
                "Wähle eine Aktie zum Anzeigen",
                options=st.session_state.selected_stocks
            )

            # Zeige Diagramm für die ausgewählte Aktie
            if selected_stock in st.session_state.historical_data:
                data = st.session_state.historical_data[selected_stock]

                if not data.empty:
                    # Erstelle erweitertes Diagramm
                    fig, data_with_indicators = create_advanced_chart(
                        data,
                        selected_stock,
                        timeframe,
                        show_volume=True,
                        show_indicators=show_indicators
                    )

                    # Zeige Diagramm
                    st.plotly_chart(fig, use_container_width=True)

                    # Zeige technische Indikatoren in einem Expander
                    with st.expander("Technische Indikatoren", expanded=False):
                        st.subheader(f"Technische Analyse für {selected_stock}")

                        # Erstelle Spalten für verschiedene Indikatoren
                        col1, col2 = st.columns(2)

                        with col1:
                            st.subheader("Gleitende Durchschnitte")
                            latest = data_with_indicators.iloc[-1]

                            st.metric("SMA 20", f"${latest.get('sma20', 'N/A'):.2f}" if not pd.isna(
                                latest.get('sma20', pd.NA)) else "N/A")
                            st.metric("SMA 50", f"${latest.get('sma50', 'N/A'):.2f}" if not pd.isna(
                                latest.get('sma50', pd.NA)) else "N/A")
                            st.metric("SMA 200", f"${latest.get('sma200', 'N/A'):.2f}" if not pd.isna(
                                latest.get('sma200', pd.NA)) else "N/A")

                        with col2:
                            st.subheader("Oszillatoren")
                            st.metric("RSI (14)", f"{latest.get('rsi', 'N/A'):.2f}" if not pd.isna(
                                latest.get('rsi', pd.NA)) else "N/A")
                            st.metric("MACD", f"{latest.get('macd', 'N/A'):.2f}" if not pd.isna(
                                latest.get('macd', pd.NA)) else "N/A")
                            st.metric("Signal", f"{latest.get('signal', 'N/A'):.2f}" if not pd.isna(
                                latest.get('signal', pd.NA)) else "N/A")

                    # Simuliere KI-Vorhersagen
                    if selected_stock not in st.session_state.ai_predictions:
                        st.session_state.ai_predictions[selected_stock] = simulate_ai_predictions(selected_stock, data)

                    # Zeige aktuelle KI-Vorhersage
                    pred = st.session_state.ai_predictions[selected_stock]
                    if pred and pred.get('prediction') != 'UNBEKANNT':
                        st.info(
                            f"**KI-Vorhersage:** {pred['prediction']} mit {pred['confidence'] * 100:.1f}% Konfidenz")
                else:
                    st.warning(f"Keine Daten verfügbar für {selected_stock}")
            else:
                st.warning(f"Keine historischen Daten verfügbar für {selected_stock}")

    with tab2:
        # Portfolio-Ansicht
        st.header("Portfolio-Übersicht")

        if st.session_state.api_connected:
            # Hole aktuelle Positionen
            try:
                positions = st.session_state.alpaca_api.get_positions()
                st.session_state.positions = positions

                if positions:
                    # Erstelle DataFrame für bessere Anzeige
                    positions_data = []

                    for position in positions:
                        positions_data.append({
                            'Symbol': position.symbol,
                            'Menge': float(position.qty),
                            'Einstiegspreis': f"${float(position.avg_entry_price):.2f}",
                            'Aktueller Preis': f"${float(position.current_price):.2f}",
                            'Marktwert': f"${float(position.market_value):.2f}",
                            'G/V': f"${float(position.unrealized_pl):.2f}",
                            'G/V %': f"{float(position.unrealized_plpc) * 100:.2f}%",
                            'Heute Änderung': f"{float(position.change_today) * 100:.2f}%"
                        })

                    # Erstelle DataFrame
                    positions_df = pd.DataFrame(positions_data)

                    # Zeige Positionen als Tabelle
                    st.dataframe(positions_df, use_container_width=True)

                    # Erstelle Tortendiagramm für Portfolio-Allokation
                    if positions_data:
                        symbols = [p['Symbol'] for p in positions_data]
                        market_values = [float(p['Marktwert'].replace('$', '')) for p in positions_data]

                        fig = go.Figure(data=[go.Pie(
                            labels=symbols,
                            values=market_values,
                            textinfo='label+percent',
                            insidetextorientation='radial'
                        )])

                        fig.update_layout(
                            title="Portfolio-Allokation",
                            height=400
                        )

                        st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Keine offenen Positionen im Portfolio.")
            except Exception as e:
                st.error(f"Fehler beim Abrufen der Positionen: {e}")
        else:
            st.warning("Nicht mit Alpaca API verbunden. Kann keine Portfolio-Informationen anzeigen.")

    with tab3:
        # KI-Vorhersagen-Ansicht
        st.header("KI-Trading-Vorhersagen (Simuliert)")

        # Hole historische Daten, falls noch nicht geschehen
        if not st.session_state.historical_data and st.session_state.api_connected:
            with st.status("Hole historische Daten..."):
                st.session_state.historical_data = fetch_historical_data(
                    st.session_state.alpaca_api,
                    st.session_state.selected_stocks,
                    timeframe=timeframe,
                    days=days
                )

        # Generiere Vorhersagen für alle ausgewählten Aktien
        for symbol in st.session_state.selected_stocks:
            if symbol not in st.session_state.ai_predictions and symbol in st.session_state.historical_data:
                data = st.session_state.historical_data[symbol]
                st.session_state.ai_predictions[symbol] = simulate_ai_predictions(symbol, data)

        # Zeige Vorhersagen in einer Tabelle
        predictions_data = []

        for symbol in st.session_state.selected_stocks:
            if symbol in st.session_state.ai_predictions:
                pred = st.session_state.ai_predictions[symbol]

                if pred and pred.get('prediction') != 'UNBEKANNT':
                    predictions_data.append({
                        'Symbol': symbol,
                        'Vorhersage': pred['prediction'],
                        'Konfidenz': f"{pred['confidence'] * 100:.1f}%",
                        'Kursziel': f"${pred['price_target']:.2f}" if pred.get('price_target') else "N/A",
                        'Stop-Loss': f"${pred['stop_loss']:.2f}" if pred.get('stop_loss') else "N/A",
                        'Generiert': pred['timestamp'].strftime('%H:%M:%S')
                    })

        if predictions_data:
            # Erstelle DataFrame
            predictions_df = pd.DataFrame(predictions_data)

            # Zeige Vorhersagen als Tabelle
            st.dataframe(predictions_df, use_container_width=True)

            # Erstelle Balkendiagramm der Konfidenzwerte
            symbols = [p['Symbol'] for p in predictions_data]
            confidences = [float(p['Konfidenz'].replace('%', '')) for p in predictions_data]
            colors = ['green' if p['Vorhersage'] == 'KAUFEN' else 'red' if p['Vorhersage'] == 'VERKAUFEN' else 'gray'
                      for p in predictions_data]

            fig = go.Figure(data=[go.Bar(
                x=symbols,
                y=confidences,
                marker_color=colors,
                text=confidences,
                texttemplate='%{text:.1f}%',
                textposition='auto',
            )])

            fig.update_layout(
                title="KI-Vorhersage Konfidenz",
                xaxis_title="Symbol",
                yaxis_title="Konfidenz (%)",
                height=400
            )

            st.plotly_chart(fig, use_container_width=True)

            # Hinweis zur Simulation
            st.info(
                "Hinweis: Diese Vorhersagen sind Simulationen und nicht auf einem echten KI-Modell basierend. Diese Funktion wird in Zukunft implementiert.")
        else:
            st.info("Keine Vorhersagen verfügbar.")

    # Zeige Zeit der letzten Aktualisierung an
    st.sidebar.text(f"Zuletzt aktualisiert: {st.session_state.last_update.strftime('%H:%M:%S')}")

    # Aktualisieren-Button
    if st.sidebar.button("Aktualisieren"):
        # Aktualisiere historische Daten
        with st.status("Aktualisiere Daten..."):
            st.session_state.historical_data = fetch_historical_data(
                st.session_state.alpaca_api,
                st.session_state.selected_stocks,
                timeframe=timeframe,
                days=days
            )

            # Aktualisiere Kontoinformationen
            if st.session_state.api_connected:
                try:
                    st.session_state.account_info = st.session_state.alpaca_api.get_account_info()
                    st.session_state.positions = st.session_state.alpaca_api.get_positions()
                except Exception as e:
                    logger.error(f"Fehler beim Aktualisieren der Kontoinformationen: {e}")

            # Aktualisiere KI-Vorhersagen
            st.session_state.ai_predictions = {}
            for symbol in st.session_state.selected_stocks:
                if symbol in st.session_state.historical_data:
                    data = st.session_state.historical_data[symbol]
                    st.session_state.ai_predictions[symbol] = simulate_ai_predictions(symbol, data)

        st.session_state.last_update = datetime.now()
        st.success("Daten aktualisiert!")


if __name__ == "__main__":
    # Stelle sicher, dass das logs-Verzeichnis existiert
    logs_dir = os.path.join(root_dir, 'logs')
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)

    main()