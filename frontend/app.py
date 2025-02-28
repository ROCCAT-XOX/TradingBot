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

# Add the parent directory to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import our modules
from backend.alpaca_api import AlpacaAPI
from backend.websocket_listener import WebSocketListener
from backend.database import Database

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(
            os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'logs', 'app.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Trading AI Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state variables
if 'initialized' not in st.session_state:
    st.session_state.initialized = False
if 'ws_listener' not in st.session_state:
    st.session_state.ws_listener = None
if 'selected_stocks' not in st.session_state:
    st.session_state.selected_stocks = []
if 'real_time_data' not in st.session_state:
    st.session_state.real_time_data = {}
if 'historical_data' not in st.session_state:
    st.session_state.historical_data = {}
if 'last_update' not in st.session_state:
    st.session_state.last_update = datetime.now()
if 'api_connected' not in st.session_state:
    st.session_state.api_connected = False
if 'db_connected' not in st.session_state:
    st.session_state.db_connected = False


# Load configuration
@st.cache_resource
def load_config(force_reload=False):
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
            'DB_HOST': os.environ.get('DB_HOST', 'localhost'),
            'DB_PORT': os.environ.get('DB_PORT', '5432'),
            'DB_NAME': os.environ.get('DB_NAME', 'trading_bot'),
            'DB_USER': os.environ.get('DB_USER', 'postgres'),
            'DB_PASSWORD': os.environ.get('DB_PASSWORD', '')
        }


# Initialize API clients
@st.cache_resource
def initialize_alpaca_api(config):
    try:
        api = AlpacaAPI(None)  # Pass None to use environment variables
        api.get_account_info()  # Test the connection
        st.session_state.api_connected = True
        return api
    except Exception as e:
        logger.error(f"Error initializing Alpaca API: {e}")
        st.session_state.api_connected = False
        return None


@st.cache_resource
def initialize_database(config):
    try:
        db = Database(None)  # Pass None to use environment variables
        st.session_state.db_connected = True
        return db
    except Exception as e:
        logger.error(f"Error initializing database: {e}")
        st.session_state.db_connected = False
        return None


# Function to start the WebSocket listener
def start_websocket_listener(alpaca_api, symbols):
    try:
        if st.session_state.ws_listener:
            st.session_state.ws_listener.stop()

        st.session_state.ws_listener = WebSocketListener(alpaca_api, symbols)
        st.session_state.ws_listener.start()
        return True
    except Exception as e:
        logger.error(f"Error starting WebSocket listener: {e}")
        return False


# Function to fetch historical data
@st.cache_data(ttl=3600)  # Cache for 1 hour
def fetch_historical_data(alpaca_api, symbols, timeframe='1Day', days=30):
    start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
    end_date = datetime.now().strftime('%Y-%m-%d')

    try:
        return alpaca_api.get_historical_bars(symbols, timeframe=timeframe, start=start_date, end=end_date)
    except Exception as e:
        logger.error(f"Error fetching historical data: {e}")
        return {}


# Function to update real-time data from WebSocket
def update_real_time_data():
    if not st.session_state.ws_listener:
        return

    # Get the latest data
    latest_trades = st.session_state.ws_listener.get_latest_trades()
    latest_quotes = st.session_state.ws_listener.get_latest_quotes()
    latest_bars = st.session_state.ws_listener.get_latest_bars()

    # Update session state
    for symbol in st.session_state.selected_stocks:
        if symbol not in st.session_state.real_time_data:
            st.session_state.real_time_data[symbol] = {
                'trades': [],
                'quotes': [],
                'bars': []
            }

        if symbol in latest_trades:
            st.session_state.real_time_data[symbol]['trades'].append(latest_trades[symbol])
            # Keep only the last 100 trades
            st.session_state.real_time_data[symbol]['trades'] = st.session_state.real_time_data[symbol]['trades'][-100:]

        if symbol in latest_quotes:
            st.session_state.real_time_data[symbol]['quotes'].append(latest_quotes[symbol])
            # Keep only the last 100 quotes
            st.session_state.real_time_data[symbol]['quotes'] = st.session_state.real_time_data[symbol]['quotes'][-100:]

        if symbol in latest_bars:
            st.session_state.real_time_data[symbol]['bars'].append(latest_bars[symbol])
            # Keep only the last 100 bars
            st.session_state.real_time_data[symbol]['bars'] = st.session_state.real_time_data[symbol]['bars'][-100:]

    st.session_state.last_update = datetime.now()


# Function to create a candlestick chart with Plotly
def create_candlestick_chart(data, title):
    fig = go.Figure(data=[go.Candlestick(
        x=data.index,
        open=data['open'],
        high=data['high'],
        low=data['low'],
        close=data['close'],
        name='Price'
    )])

    # Add volume as bar chart
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


# Main application logic
def main():
    # Sidebar
    st.sidebar.title("Trading AI Dashboard")

    # Load configuration
    config = load_config()

    # Connection status indicators
    st.sidebar.header("Connection Status")
    api_status = "✅ Connected" if st.session_state.api_connected else "❌ Not Connected"
    db_status = "✅ Connected" if st.session_state.db_connected else "❌ Not Connected"
    ws_status = "✅ Connected" if st.session_state.ws_listener and st.session_state.ws_listener.running else "❌ Not Connected"

    st.sidebar.text(f"Alpaca API: {api_status}")
    st.sidebar.text(f"Database: {db_status}")
    st.sidebar.text(f"WebSocket: {ws_status}")

    # Initialize connections if not already done
    if not st.session_state.initialized:
        alpaca_api = initialize_alpaca_api(config)
        db = initialize_database(config)

        if alpaca_api and db:
            st.session_state.initialized = True
            st.sidebar.success("Connections initialized successfully!")
        else:
            if not alpaca_api:
                st.sidebar.error("Failed to connect to Alpaca API. Check your credentials.")
            if not db:
                st.sidebar.error("Failed to connect to database. Check your connection settings.")

    # Stock selection
    st.sidebar.header("Stock Selection")
    default_stocks = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
    selected_stocks = st.sidebar.multiselect(
        "Select stocks to track",
        options=default_stocks + ['NVDA', 'META', 'NFLX', 'PYPL', 'DIS', 'INTC', 'AMD', 'CSCO', 'ADBE', 'ORCL'],
        default=default_stocks[:3]  # Default to first 3 stocks
    )

    # Store selected stocks in session state
    if selected_stocks != st.session_state.selected_stocks:
        st.session_state.selected_stocks = selected_stocks
        # Start WebSocket listener with the selected stocks
        if st.session_state.initialized and selected_stocks:
            start_websocket_listener(initialize_alpaca_api(config), selected_stocks)
            # Fetch historical data for the selected stocks
            st.session_state.historical_data = fetch_historical_data(initialize_alpaca_api(config), selected_stocks)

    # Main content
    st.title("Stock Dashboard")

    # Display selected stocks
    if not st.session_state.selected_stocks:
        st.warning("Please select at least one stock to track.")
        return

    # Create tabs for each stock
    tabs = st.tabs(st.session_state.selected_stocks)

    # Update real-time data
    update_real_time_data()

    # Display data for each stock in its tab
    for i, symbol in enumerate(st.session_state.selected_stocks):
        with tabs[i]:
            st.header(f"{symbol} Dashboard")

            # Create columns for organizing content
            col1, col2 = st.columns([3, 1])

            with col1:
                # Display historical candlestick chart
                if symbol in st.session_state.historical_data:
                    st.subheader("Historical Price Data")
                    fig = create_candlestick_chart(st.session_state.historical_data[symbol],
                                                   f"{symbol} - Historical Price")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning(f"No historical data available for {symbol}")

            with col2:
                # Display latest quote
                st.subheader("Latest Quote")
                if symbol in st.session_state.real_time_data and st.session_state.real_time_data[symbol]['quotes']:
                    latest_quote = st.session_state.real_time_data[symbol]['quotes'][-1]
                    st.metric("Bid", f"${latest_quote['bid_price']:.2f}")
                    st.metric("Ask", f"${latest_quote['ask_price']:.2f}")
                    st.metric("Spread", f"${latest_quote['ask_price'] - latest_quote['bid_price']:.2f}")
                    st.text(f"Bid Size: {latest_quote['bid_size']}")
                    st.text(f"Ask Size: {latest_quote['ask_size']}")
                    st.text(f"Updated: {datetime.fromisoformat(latest_quote['received_at']).strftime('%H:%M:%S')}")
                else:
                    st.info("Waiting for real-time quotes...")

            # Display recent trades
            st.subheader("Recent Trades")
            if symbol in st.session_state.real_time_data and st.session_state.real_time_data[symbol]['trades']:
                trades_df = pd.DataFrame(st.session_state.real_time_data[symbol]['trades'][-10:])
                if not trades_df.empty:
                    # Convert timestamp to readable format
                    trades_df['timestamp'] = pd.to_datetime(trades_df['timestamp']).dt.strftime('%H:%M:%S')
                    st.dataframe(trades_df[['timestamp', 'price', 'size']], use_container_width=True)
                else:
                    st.info("No trades received yet")
            else:
                st.info("Waiting for real-time trades...")

    # Display last update time
    st.sidebar.text(f"Last updated: {st.session_state.last_update.strftime('%H:%M:%S')}")

    # Refresh every 5 seconds
    time.sleep(5)
    st.experimental_rerun()


if __name__ == "__main__":
    # Make sure logs directory exists
    logs_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'logs')
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)

    main()