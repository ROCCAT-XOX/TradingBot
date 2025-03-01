import os
import sys
import json
import logging
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from datetime import datetime, timedelta
import subprocess
import time
import threading
import queue

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import our modules
from backend.alpaca_api import AlpacaAPI
from backend.database import Database
from backend.trade_bot import TradingBot


def check_training_completion():
    """
    Überprüft, ob ein Training abgeschlossen wurde, indem nach der Benachrichtigungsdatei gesucht wird.

    Returns:
        dict or None: Benachrichtigungsdaten, falls das Training abgeschlossen ist, sonst None
    """
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    notification_file = os.path.join(parent_dir, 'logs', 'training_complete.json')

    if os.path.exists(notification_file):
        try:
            with open(notification_file, 'r') as f:
                notification = json.load(f)
            return notification
        except Exception as e:
            logging.error(f"Fehler beim Lesen der Trainings-Abschlussbenachrichtigung: {e}")

    return None


def clear_training_notification():
    """Löscht die Trainingsabschluss-Benachrichtigungsdatei."""
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    notification_file = os.path.join(parent_dir, 'logs', 'training_complete.json')

    if os.path.exists(notification_file):
        try:
            os.remove(notification_file)
            logging.info(f"Trainingsabschluss-Benachrichtigung gelöscht: {notification_file}")
            return True
        except Exception as e:
            logging.error(f"Fehler beim Löschen der Trainingsabschluss-Benachrichtigung: {e}")

    return False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'logs',
                                         'training_dashboard.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Trading AI - Trainingsmodus",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state variables
if 'initialized' not in st.session_state:
    st.session_state.initialized = False
if 'api_connected' not in st.session_state:
    st.session_state.api_connected = False
if 'db_connected' not in st.session_state:
    st.session_state.db_connected = False
if 'training_running' not in st.session_state:
    st.session_state.training_running = False
if 'bot_running' not in st.session_state:
    st.session_state.bot_running = False
if 'training_logs' not in st.session_state:
    st.session_state.training_logs = []
if 'bot_logs' not in st.session_state:
    st.session_state.bot_logs = []
if 'training_status' not in st.session_state:
    st.session_state.training_status = {'status': 'idle', 'progress': 0, 'message': 'Bereit'}
if 'model_metrics' not in st.session_state:
    st.session_state.model_metrics = {}
if 'last_update' not in st.session_state:
    st.session_state.last_update = datetime.now()


# Load configuration
@st.cache_resource
def load_config():
    config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'config', 'settings.json')
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            return json.load(f)
    else:
        # Use environment variables as fallback
        return {
            'ALPACA_API_KEY': os.environ.get('ALPACA_API_KEY', ''),
            'ALPACA_API_SECRET': os.environ.get('ALPACA_API_SECRET', ''),
            'ALPACA_API_BASE_URL': os.environ.get('ALPACA_API_BASE_URL', 'https://paper-api.alpaca.markets'),
            'TRADING_SYMBOLS': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA'],
            'DEFAULT_TIMEFRAME': '1Day',
        }


# Initialize API clients
@st.cache_resource
def initialize_alpaca_api(config):
    try:
        # Try to load from config file first
        config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'config',
                                   'settings.json')

        # Use cached config if available
        alpaca_api = AlpacaAPI(config_path=config_path)

        # Test the connection
        alpaca_api.get_account_info()

        st.session_state.api_connected = True
        return alpaca_api
    except Exception as e:
        logger.error(f"Error initializing Alpaca API: {e}")
        st.session_state.api_connected = False
        return None


@st.cache_resource
def initialize_database(config):
    try:
        db = Database()
        st.session_state.db_connected = True
        return db
    except Exception as e:
        logger.error(f"Error initializing database: {e}")
        st.session_state.db_connected = False
        return None


# Function to start training
def start_training(symbols=None, epochs=50, learning_rate=0.001, timeframe='1Day', days=365, unlimited=False,
                   patience=20):
    """Start the ML model training process."""
    if st.session_state.training_running:
        st.warning("Training is already running!")
        return

    # Create command for training
    cmd = ["python", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "backend", "train.py")]

    # Add parameters if needed
    if symbols:
        symbols_str = ",".join(symbols)
        cmd.extend(["--symbols", symbols_str])

    cmd.extend(["--epochs", str(epochs)])
    cmd.extend(["--learning_rate", str(learning_rate)])
    cmd.extend(["--timeframe", timeframe])
    cmd.extend(["--days", str(days)])

    # Unbegrenztes Training mit Early Stopping
    if unlimited:
        cmd.append("--unlimited")
        cmd.extend(["--patience", str(patience)])

    # Create a queue for logs
    log_queue = queue.Queue()

    # Update session state
    st.session_state.training_running = True
    st.session_state.training_logs = []
    st.session_state.training_status = {'status': 'running', 'progress': 0.5, 'message': 'Training läuft...'}

    # Function to read output and update logs
    def read_output(process, log_queue):
        for line in iter(process.stdout.readline, ""):
            if line:
                log_queue.put(("INFO", line.strip()))

        for line in iter(process.stderr.readline, ""):
            if line:
                log_queue.put(("ERROR", line.strip()))

    # Function to update session state with logs
    def update_logs(log_queue):
        while st.session_state.training_running:
            try:
                log_type, message = log_queue.get(timeout=0.1)
                timestamp = datetime.now().strftime("%H:%M:%S")
                st.session_state.training_logs.append(f"[{timestamp}] {log_type}: {message}")

                # Keep only the last 100 logs
                if len(st.session_state.training_logs) > 100:
                    st.session_state.training_logs = st.session_state.training_logs[-100:]
            except queue.Empty:
                time.sleep(0.1)
            except Exception as e:
                logger.error(f"Error updating logs: {e}")

    # Start the process
    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            universal_newlines=True
        )

        # Start threads for reading output and updating logs
        output_thread = threading.Thread(target=read_output, args=(process, log_queue), daemon=True)
        logs_thread = threading.Thread(target=update_logs, args=(log_queue,), daemon=True)

        output_thread.start()
        logs_thread.start()

        st.success("Training gestartet!")
        return process

    except Exception as e:
        logger.error(f"Error starting training: {e}")
        st.error(f"Fehler beim Starten des Trainings: {e}")
        st.session_state.training_running = False
        st.session_state.training_status = {'status': 'error', 'progress': 0, 'message': f'Fehler: {e}'}
        return None


# Function to start trading bot
def start_trading_bot(mode='paper', test_mode=True):
    """Start the trading bot."""
    if st.session_state.bot_running:
        st.warning("Trading bot is already running!")
        return

    # Create command for trading bot
    cmd = ["python",
           os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "backend", "trade_bot.py")]

    # Add parameters
    cmd.extend(["--mode", mode])

    if test_mode:
        cmd.append("--test")

    # Create a queue for logs
    log_queue = queue.Queue()

    # Update session state
    st.session_state.bot_running = True
    st.session_state.bot_logs = []

    # Function to read output and update logs
    def read_output(process, log_queue):
        for line in iter(process.stdout.readline, ""):
            if line:
                log_queue.put(("INFO", line.strip()))

        for line in iter(process.stderr.readline, ""):
            if line:
                log_queue.put(("ERROR", line.strip()))

    # Function to update session state with logs
    def update_logs(log_queue):
        while st.session_state.bot_running:
            try:
                log_type, message = log_queue.get(timeout=0.1)
                timestamp = datetime.now().strftime("%H:%M:%S")
                st.session_state.bot_logs.append(f"[{timestamp}] {log_type}: {message}")

                # Keep only the last 100 logs
                if len(st.session_state.bot_logs) > 100:
                    st.session_state.bot_logs = st.session_state.bot_logs[-100:]
            except queue.Empty:
                time.sleep(0.1)
            except Exception as e:
                logger.error(f"Error updating logs: {e}")

    # Start the process
    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            universal_newlines=True
        )

        # Start threads for reading output and updating logs
        output_thread = threading.Thread(target=read_output, args=(process, log_queue), daemon=True)
        logs_thread = threading.Thread(target=update_logs, args=(log_queue,), daemon=True)

        output_thread.start()
        logs_thread.start()

        st.success("Trading bot started!")
        return process
    except Exception as e:
        logger.error(f"Error starting trading bot: {e}")
        st.error(f"Error starting trading bot: {e}")
        st.session_state.bot_running = False
        return None


# Function to stop a process
def stop_process(process, is_training=True):
    """Stop a running process."""
    if process:
        try:
            process.terminate()
            process.wait(timeout=5)

            if is_training:
                st.session_state.training_running = False
                st.session_state.training_status = {'status': 'stopped', 'progress': 0, 'message': 'Training gestoppt'}
            else:
                st.session_state.bot_running = False

            return True
        except Exception as e:
            logger.error(f"Error stopping process: {e}")
            return False
    return False


# Function to get model metrics from the database
def get_model_metrics(db, symbol=None):
    """Get the performance metrics of all models or a specific symbol."""
    try:
        if symbol:
            # Get the model for this symbol
            model = db.get_model_by_symbol(symbol)
            if not model:
                return {}

            # Get performance metrics
            metrics = db.get_model_performance(model_id=model.id)

            return {
                'symbol': symbol,
                'model_name': model.name,
                'model_type': model.model_type,
                'created_at': model.created_at,
                'metrics': metrics
            }
        else:
            # Get all active models
            active_model = db.get_active_model()
            if not active_model:
                return {}

            metrics = db.get_model_performance(model_id=active_model.id)

            return {
                'model_name': active_model.name,
                'model_type': active_model.model_type,
                'created_at': active_model.created_at,
                'metrics': metrics
            }
    except Exception as e:
        logger.error(f"Error getting model metrics: {e}")
        return {}


# Function to open other dashboards
def open_dashboard(dashboard_type):
    """Open a specific dashboard in a new browser tab."""
    ports = {
        "main": 8501,
        "simple": 8502,
        "ai": 8503
    }

    if dashboard_type in ports:
        import webbrowser
        webbrowser.open(f"http://localhost:{ports[dashboard_type]}")


# Main function
def main():
    # Set up page header
    st.title("Trading AI - Trainingsmodus Cockpit")

    # Sidebar
    st.sidebar.title("Trainingssteuerung")

    # Load configuration
    config = load_config()

    # Initialize components if not already done
    if not st.session_state.initialized:
        alpaca_api = initialize_alpaca_api(config)
        db = initialize_database(config)

        if alpaca_api and db:
            st.session_state.initialized = True
            st.sidebar.success("Komponenten initialisiert!")
        else:
            if not alpaca_api:
                st.sidebar.error("Fehler bei der Verbindung zur Alpaca API. Überprüfen Sie Ihre Zugangsdaten.")
            if not db:
                st.sidebar.error(
                    "Fehler bei der Verbindung zur Datenbank. Überprüfen Sie Ihre Verbindungseinstellungen.")
            return
    else:
        alpaca_api = initialize_alpaca_api(config)
        db = initialize_database(config)

    # Connection status indicators
    st.sidebar.header("Verbindungsstatus")
    api_status = "✅ Verbunden" if st.session_state.api_connected else "❌ Nicht verbunden"
    db_status = "✅ Verbunden" if st.session_state.db_connected else "❌ Nicht verbunden"

    st.sidebar.text(f"Alpaca API: {api_status}")
    st.sidebar.text(f"Datenbank: {db_status}")

    # Dashboard navigation
    st.sidebar.header("Dashboards öffnen")

    col1, col2, col3 = st.sidebar.columns(3)

    with col1:
        if st.button("Hauptdashboard"):
            open_dashboard("main")

    with col2:
        if st.button("Einfach"):
            open_dashboard("simple")

    with col3:
        if st.button("KI-Dashboard"):
            open_dashboard("ai")

    # Training controls
    st.sidebar.header("ML-Training")

    # Dropdown for symbol selection
    symbols = config.get('TRADING_SYMBOLS', ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA'])
    selected_symbols = st.sidebar.multiselect(
        "Aktien für Training",
        options=symbols,
        default=symbols[:2]  # Default to first 2 symbols
    )

    # Training parameters
    # In training_dashboard.py
    epochs = st.sidebar.slider(
        "Trainings-Epochen",
        min_value=10,
        max_value=10000,
        value=1000,
        step=100
    )
    learning_rate = st.sidebar.select_slider(
        "Lernrate",
        options=[0.0001, 0.0005, 0.001, 0.005, 0.01],
        value=0.001
    )
    timeframe = st.sidebar.selectbox(
        "Zeitrahmen",
        options=['1Day', '1Hour', '15Min', '5Min', '1Min'],
        index=0
    )
    days = st.sidebar.slider("Datenzeitraum (Tage)", min_value=30, max_value=730, value=365, step=30)

    # Start/Stop training buttons
    col1, col2 = st.sidebar.columns(2)

    with col1:
        if st.button("Training starten", disabled=st.session_state.training_running):
            training_process = start_training(
                symbols=selected_symbols,
                epochs=epochs,
                learning_rate=learning_rate,
                timeframe=timeframe,
                days=days
            )
            st.session_state.training_process = training_process

    with col2:
        if st.button("Training stoppen", disabled=not st.session_state.training_running):
            if hasattr(st.session_state, 'training_process'):
                success = stop_process(st.session_state.training_process, is_training=True)
                if success:
                    st.success("Training gestoppt!")
                else:
                    st.error("Fehler beim Stoppen des Trainings!")

    # Trading bot controls
    st.sidebar.header("Trading Bot")

    # Bot parameters
    bot_mode = st.sidebar.selectbox(
        "Bot-Modus",
        options=['paper', 'test', 'live'],
        index=0
    )

    test_mode = st.sidebar.checkbox("Testmodus (keine echten Trades)", value=True)

    # Start/Stop bot buttons
    col1, col2 = st.sidebar.columns(2)

    with col1:
        if st.button("Bot starten", disabled=st.session_state.bot_running):
            bot_process = start_trading_bot(mode=bot_mode, test_mode=test_mode)
            st.session_state.bot_process = bot_process

    with col2:
        if st.button("Bot stoppen", disabled=not st.session_state.bot_running):
            if hasattr(st.session_state, 'bot_process'):
                success = stop_process(st.session_state.bot_process, is_training=False)
                if success:
                    st.success("Trading Bot gestoppt!")
                else:
                    st.error("Fehler beim Stoppen des Trading Bots!")

    # Main content area

    # Training status section
    st.header("Training Status")

    # Progress bar for training
    if st.session_state.training_running:
        status = st.session_state.training_status
        st.progress(status['progress'])
        st.write(status['message'])
    else:
        status = st.session_state.training_status
        if status['status'] == 'completed':
            st.success(status['message'])
        elif status['status'] == 'stopped':
            st.warning(status['message'])
        else:
            st.info("Kein Training aktiv. Starten Sie das Training über die Seitenleiste.")

    # Create tabs for different views
    tab1, tab2, tab3 = st.tabs(["Training Logs", "Bot Logs", "Modellmetriken"])

    with tab1:
        # Display training logs
        st.subheader("Training Logs")

        log_container = st.container()

        with log_container:
            if st.session_state.training_logs:
                for log in reversed(st.session_state.training_logs):
                    if "ERROR" in log:
                        st.error(log)
                    else:
                        st.text(log)
            else:
                st.info("Keine Logs verfügbar. Starten Sie das Training, um Logs zu sehen.")

    with tab2:
        # Display bot logs
        st.subheader("Trading Bot Logs")

        log_container = st.container()

        with log_container:
            if st.session_state.bot_logs:
                for log in reversed(st.session_state.bot_logs):
                    if "ERROR" in log:
                        st.error(log)
                    else:
                        st.text(log)
            else:
                st.info("Keine Logs verfügbar. Starten Sie den Trading Bot, um Logs zu sehen.")

    with tab3:
        # Display model metrics
        st.subheader("Modellmetriken")

        # Get model metrics from database
        if st.button("Metriken aktualisieren"):
            if selected_symbols:
                metrics = {}
                for symbol in selected_symbols:
                    metrics[symbol] = get_model_metrics(db, symbol)
                st.session_state.model_metrics = metrics

        # Display metrics
        if st.session_state.model_metrics:
            for symbol, model_data in st.session_state.model_metrics.items():
                if model_data:
                    with st.expander(f"Modell für {symbol}", expanded=True):
                        st.write(f"**Modellname:** {model_data.get('model_name', 'N/A')}")
                        st.write(f"**Modelltyp:** {model_data.get('model_type', 'N/A')}")
                        st.write(f"**Erstellt am:** {model_data.get('created_at', 'N/A')}")

                        # Display metrics if available
                        metrics = model_data.get('metrics', {})
                        if metrics:
                            col1, col2, col3 = st.columns(3)

                            with col1:
                                st.metric("Genauigkeit", f"{metrics.get('accuracy', 0) * 100:.1f}%")

                            with col2:
                                st.metric("RMSE", f"{metrics.get('rmse', 0):.4f}")

                            with col3:
                                st.metric("Vorhersagen", f"{metrics.get('total_predictions', 0)}")
                        else:
                            st.info("Keine Metriken verfügbar für dieses Modell.")
        else:
            st.info("Keine Modellmetriken verfügbar. Klicken Sie auf 'Metriken aktualisieren'.")

    # Refresh button (for status updates without full page reload)
    if st.button("Status aktualisieren"):
        st.session_state.last_update = datetime.now()
        st.experimental_rerun()

    # Display last update time
    st.sidebar.text(f"Zuletzt aktualisiert: {st.session_state.last_update.strftime('%Y-%m-%d %H:%M:%S')}")

    # Auto-refresh mechanism for logs (every 10 seconds)
    auto_refresh = st.sidebar.checkbox("Auto-Aktualisierung (10s)", value=False)
    if auto_refresh:
        time_since_update = (datetime.now() - st.session_state.last_update).total_seconds()
        if time_since_update > 10:
            st.session_state.last_update = datetime.now()
            st.experimental_rerun()


if __name__ == "__main__":
    # Make sure logs directory exists
    logs_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'logs')
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)

    main()