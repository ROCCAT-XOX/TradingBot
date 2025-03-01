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
            os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'logs', 'dashboard.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Advanced Trading Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state variables if not already done
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
if 'portfolio' not in st.session_state:
    st.session_state.portfolio = {}
if 'trades' not in st.session_state:
    st.session_state.trades = []
if 'last_update' not in st.session_state:
    st.session_state.last_update = datetime.now()
if 'api_connected' not in st.session_state:
    st.session_state.api_connected = False
if 'db_connected' not in st.session_state:
    st.session_state.db_connected = False
if 'ai_predictions' not in st.session_state:
    st.session_state.ai_predictions = {}
if 'chart_timeframe' not in st.session_state:
    st.session_state.chart_timeframe = '1Day'
if 'chart_period' not in st.session_state:
    st.session_state.chart_period = 30


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
            'DB_HOST': os.environ.get('DB_HOST', 'localhost'),
            'DB_PORT': os.environ.get('DB_PORT', '5432'),
            'DB_NAME': os.environ.get('DB_NAME', 'trading_bot'),
            'DB_USER': os.environ.get('DB_USER', 'postgres'),
            'DB_PASSWORD': os.environ.get('DB_PASSWORD', '')
        }


# Initialize API clients
# This is a partial update for the dashboard.py file
# Replace the initialize_alpaca_api function with this improved version

@st.cache_resource
def initialize_alpaca_api(_config):
    try:
        # Try to load from config file first
        config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'config',
                                   'settings.json')

        # Check if config file exists
        if os.path.exists(config_path):
            logger.info(f"Loading Alpaca API configuration from: {config_path}")
            with open(config_path, 'r') as f:
                config = json.load(f)

            # Set environment variables for Alpaca API
            os.environ['ALPACA_API_KEY'] = config.get('ALPACA_API_KEY', '')
            os.environ['ALPACA_API_SECRET'] = config.get('ALPACA_API_SECRET', '')
            os.environ['ALPACA_API_BASE_URL'] = config.get('ALPACA_API_BASE_URL', 'https://paper-api.alpaca.markets')

            api = AlpacaAPI(config_path=config_path)
        else:
            # Fall back to environment variables
            logger.info("Using environment variables for Alpaca API credentials")
            api = AlpacaAPI()

        # Test the connection
        api.get_account_info()

        st.session_state.api_connected = True
        return api
    except Exception as e:
        logger.error(f"Error initializing Alpaca API: {e}")
        st.session_state.api_connected = False
        return None

@st.cache_resource
def initialize_database(_config):
    try:
        db = Database()  # Use environment variables
        st.session_state.db_connected = True
        return db
    except Exception as e:
        logger.error(f"Error initializing database: {e}")
        st.session_state.db_connected = False
        return None


# Function to start the WebSocket listener
def start_websocket_listener(alpaca_api, symbols):
    """Start the WebSocket listener with error handling."""
    try:
        # Stop existing listener if it exists
        if st.session_state.ws_listener:
            try:
                st.session_state.ws_listener.stop()
            except Exception as stop_error:
                logger.warning(f"Error stopping existing WebSocket listener: {stop_error}")

        # Create new WebSocket listener
        new_listener = WebSocketListener(alpaca_api, symbols)
        new_listener.start()

        # Update session state
        st.session_state.ws_listener = new_listener
        return True
    except Exception as e:
        logger.error(f"Error starting WebSocket listener: {e}")
        st.sidebar.error(f"Could not start WebSocket listener: {e}")
        return False

# Function to fetch historical data
# Update this function in your dashboard.py

# Change the function signature to add underscore to alpaca_api parameter
@st.cache_data(ttl=3600)  # Cache for 1 hour
def fetch_historical_data(_alpaca_api, symbols, timeframe='1Day', days=30):
    """
    Fetch historical price data.

    Args:
        _alpaca_api (AlpacaAPI): The Alpaca API instance (not hashed)
        symbols (list): List of stock symbols
        timeframe (str): Timeframe for the data
        days (int): Number of days of historical data to fetch

    Returns:
        dict: Dictionary of DataFrames with historical price data
    """
    start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
    end_date = datetime.now().strftime('%Y-%m-%d')

    try:
        return _alpaca_api.get_historical_bars(symbols, timeframe=timeframe, start=start_date, end=end_date)
    except Exception as e:
        logger.error(f"Error fetching historical data: {e}")
        return {}


# Function to get account information
def get_account_info(alpaca_api):
    try:
        return alpaca_api.get_account_info()
    except Exception as e:
        logger.error(f"Error getting account info: {e}")
        return None


# Function to get current positions
def get_positions(alpaca_api):
    try:
        positions = alpaca_api.get_positions()
        result = {}

        for position in positions:
            result[position.symbol] = {
                'qty': float(position.qty),
                'avg_entry_price': float(position.avg_entry_price),
                'market_value': float(position.market_value),
                'current_price': float(position.current_price),
                'unrealized_pl': float(position.unrealized_pl),
                'unrealized_plpc': float(position.unrealized_plpc) * 100,  # Convert to percentage
                'change_today': float(position.change_today) * 100  # Convert to percentage
            }

        return result
    except Exception as e:
        logger.error(f"Error getting positions: {e}")
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


# Function to create an advanced candlestick chart with Plotly
def create_advanced_chart(data, symbol, timeframe, indicators=None, trades=None):
    # Create subplots with 2 rows and 1 column
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.7, 0.3]
    )

    # Add candlestick trace
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

    # Add volume trace
    if 'volume' in data.columns:
        # Color volume bars based on price movement
        colors = ['red' if data.iloc[i]['close'] < data.iloc[i]['open'] else 'green' for i in range(len(data))]

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

    # Add indicators if provided
    if indicators:
        if 'sma20' in indicators:
            fig.add_trace(
                go.Scatter(
                    x=indicators['sma20'].index,
                    y=indicators['sma20'],
                    name='SMA 20',
                    line=dict(color='blue', width=1)
                ),
                row=1, col=1
            )

        if 'sma50' in indicators:
            fig.add_trace(
                go.Scatter(
                    x=indicators['sma50'].index,
                    y=indicators['sma50'],
                    name='SMA 50',
                    line=dict(color='orange', width=1)
                ),
                row=1, col=1
            )

        if 'sma200' in indicators:
            fig.add_trace(
                go.Scatter(
                    x=indicators['sma200'].index,
                    y=indicators['sma200'],
                    name='SMA 200',
                    line=dict(color='purple', width=1)
                ),
                row=1, col=1
            )

    # Add trade markers if provided
    if trades:
        buy_times = [t['timestamp'] for t in trades if t['trade_type'] == 'buy']
        buy_prices = [t['price'] for t in trades if t['trade_type'] == 'buy']

        sell_times = [t['timestamp'] for t in trades if t['trade_type'] == 'sell']
        sell_prices = [t['price'] for t in trades if t['trade_type'] == 'sell']

        if buy_times:
            fig.add_trace(
                go.Scatter(
                    x=buy_times,
                    y=buy_prices,
                    mode='markers',
                    name='Buy',
                    marker=dict(
                        symbol='triangle-up',
                        size=10,
                        color='green',
                        line=dict(width=1, color='green')
                    )
                ),
                row=1, col=1
            )

        if sell_times:
            fig.add_trace(
                go.Scatter(
                    x=sell_times,
                    y=sell_prices,
                    mode='markers',
                    name='Sell',
                    marker=dict(
                        symbol='triangle-down',
                        size=10,
                        color='red',
                        line=dict(width=1, color='red')
                    )
                ),
                row=1, col=1
            )

    # Update layout
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
        yaxis2_title="Volume",
        height=800,
        xaxis_rangeslider_visible=False,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )

    # Update y-axis settings
    fig.update_yaxes(title_text="Price ($)", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)

    return fig


# Function to calculate technical indicators
def calculate_indicators(data):
    """Calculate technical indicators for the given price data."""
    if data.empty:
        return {}

    # Make a copy to avoid modifying the original data
    df = data.copy()

    # Simple Moving Averages
    df['sma20'] = df['close'].rolling(window=20).mean()
    df['sma50'] = df['close'].rolling(window=50).mean()
    df['sma200'] = df['close'].rolling(window=200).mean()

    # Relative Strength Index (RSI)
    # First, calculate price changes
    delta = df['close'].diff()

    # Separate gains and losses
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    # Calculate average gain and loss over 14 periods
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()

    # Calculate Relative Strength
    rs = avg_gain / avg_loss

    # Calculate RSI
    df['rsi'] = 100 - (100 / (1 + rs))

    # MACD
    df['ema12'] = df['close'].ewm(span=12, adjust=False).mean()
    df['ema26'] = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = df['ema12'] - df['ema26']
    df['signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['histogram'] = df['macd'] - df['signal']

    # Bollinger Bands
    df['middle_band'] = df['close'].rolling(window=20).mean()
    df['std'] = df['close'].rolling(window=20).std()
    df['upper_band'] = df['middle_band'] + (df['std'] * 2)
    df['lower_band'] = df['middle_band'] - (df['std'] * 2)

    return df


# Function to simulate AI predictions (placeholder for future implementation)
def simulate_ai_predictions(symbol, data):
    """Simulate AI predictions for the given symbol and data."""
    if data.empty:
        return {}

    # Use the last 20 days of data for prediction
    recent_data = data.iloc[-20:]

    # Simple simulation: predict up if the 5-day SMA is above the 20-day SMA
    sma5 = recent_data['close'].rolling(window=5).mean()
    sma20 = recent_data['close'].rolling(window=20).mean()

    # Get the last values
    try:
        last_sma5 = sma5.iloc[-1]
        last_sma20 = sma20.iloc[-1]
        last_close = recent_data['close'].iloc[-1]

        if last_sma5 > last_sma20:
            prediction = 'BUY'
            confidence = min(0.5 + (last_sma5 - last_sma20) / last_close, 0.95)
        elif last_sma5 < last_sma20:
            prediction = 'SELL'
            confidence = min(0.5 + (last_sma20 - last_sma5) / last_close, 0.95)
        else:
            prediction = 'HOLD'
            confidence = 0.5

        return {
            'prediction': prediction,
            'confidence': confidence,
            'timestamp': datetime.now(),
            'price_target': last_close * (1.05 if prediction == 'BUY' else 0.95 if prediction == 'SELL' else 1.0),
            'stop_loss': last_close * (0.97 if prediction == 'BUY' else 1.03 if prediction == 'SELL' else 1.0)
        }
    except:
        return {
            'prediction': 'UNKNOWN',
            'confidence': 0.0,
            'timestamp': datetime.now(),
            'price_target': None,
            'stop_loss': None
        }


# Function to create a portfolio performance chart
def create_portfolio_chart(portfolio_history):
    """Create a chart showing portfolio performance over time."""
    if not portfolio_history:
        return None

    df = pd.DataFrame(portfolio_history)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.set_index('timestamp')

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['equity'],
        mode='lines',
        name='Equity',
        line=dict(color='blue', width=2)
    ))

    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['cash'],
        mode='lines',
        name='Cash',
        line=dict(color='green', width=2)
    ))

    fig.update_layout(
        title="Portfolio Performance",
        xaxis_title="Date",
        yaxis_title="Value ($)",
        height=400,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )

    return fig


# Function to create a pie chart of current portfolio allocation
def create_portfolio_allocation_chart(positions):
    """Create a pie chart showing the current portfolio allocation."""
    if not positions:
        return None

    # Extract market values
    symbols = list(positions.keys())
    market_values = [float(positions[symbol]['market_value']) for symbol in symbols]

    # Create pie chart
    fig = go.Figure(data=[go.Pie(
        labels=symbols,
        values=market_values,
        textinfo='label+percent',
        insidetextorientation='radial'
    )])

    fig.update_layout(
        title="Portfolio Allocation",
        height=400
    )

    return fig


# Main dashboard function
def main():
    # Sidebar
    st.sidebar.title("Advanced Trading Dashboard")

    # Load configuration
    config = load_config()

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
            st.stop()
    else:
        alpaca_api = initialize_alpaca_api(config)
        db = initialize_database(config)

    # Connection status indicators
    st.sidebar.header("Connection Status")
    api_status = "✅ Connected" if st.session_state.api_connected else "❌ Not Connected"
    db_status = "✅ Connected" if st.session_state.db_connected else "❌ Not Connected"
    ws_status = "✅ Connected" if st.session_state.ws_listener and st.session_state.ws_listener.running else "❌ Not Connected"

    st.sidebar.text(f"Alpaca API: {api_status}")
    st.sidebar.text(f"Database: {db_status}")
    st.sidebar.text(f"WebSocket: {ws_status}")

    # Chart settings
    st.sidebar.header("Chart Settings")

    timeframe = st.sidebar.selectbox(
        "Timeframe",
        options=['1Min', '5Min', '15Min', '1Hour', '1Day'],
        index=4,  # Default to 1Day
        key="timeframe_selector"
    )

    days = st.sidebar.slider(
        "Data Period (days)",
        min_value=1,
        max_value=90,
        value=30,
        key="days_selector"
    )

    # Update session state if settings changed
    if timeframe != st.session_state.chart_timeframe or days != st.session_state.chart_period:
        st.session_state.chart_timeframe = timeframe
        st.session_state.chart_period = days
        # Clear historical data to force refresh
        st.session_state.historical_data = {}

    # Stock selection
    st.sidebar.header("Stock Selection")
    default_stocks = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
    selected_stocks = st.sidebar.multiselect(
        "Select stocks to track",
        options=default_stocks + ['NVDA', 'META', 'NFLX', 'PYPL', 'DIS', 'INTC', 'AMD', 'CSCO', 'ADBE', 'ORCL'],
        default=default_stocks[:3]  # Default to first 3 stocks
    )

    # Store selected stocks in session state and update connections if needed
    if selected_stocks != st.session_state.selected_stocks:
        st.session_state.selected_stocks = selected_stocks
        # Start WebSocket listener with the selected stocks
        if st.session_state.initialized and selected_stocks:
            start_websocket_listener(alpaca_api, selected_stocks)
            # Clear historical data to force refresh
            st.session_state.historical_data = {}

    # Fetch account information
    if st.session_state.api_connected:
        account = get_account_info(alpaca_api)
        positions = get_positions(alpaca_api)

        # Store in session state
        st.session_state.portfolio = {
            'equity': float(account.equity) if account else 0.0,
            'cash': float(account.cash) if account else 0.0,
            'buying_power': float(account.buying_power) if account else 0.0,
            'positions': positions
        }

    # Main content
    st.title("Advanced Trading Dashboard")

    # Portfolio summary at the top
    if st.session_state.api_connected and 'portfolio' in st.session_state:
        portfolio = st.session_state.portfolio

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Equity", f"${portfolio.get('equity', 0):.2f}")

        with col2:
            st.metric("Cash", f"${portfolio.get('cash', 0):.2f}")

        with col3:
            st.metric("Buying Power", f"${portfolio.get('buying_power', 0):.2f}")

        with col4:
            positions_count = len(portfolio.get('positions', {}))
            st.metric("Positions", f"{positions_count}")

    # Display tabs for different views
    tab1, tab2, tab3 = st.tabs(["Stock Charts", "Portfolio", "AI Predictions"])

    with tab1:
        # Stock charts view
        if not st.session_state.selected_stocks:
            st.warning("Please select at least one stock to track.")
        else:
            # Fetch historical data if needed
            if not st.session_state.historical_data:
                with st.spinner("Fetching historical data..."):
                    st.session_state.historical_data = fetch_historical_data(
                        alpaca_api,
                        st.session_state.selected_stocks,
                        timeframe=st.session_state.chart_timeframe,
                        days=st.session_state.chart_period
                    )

            # Update real-time data
            update_real_time_data()

            # Create stock selector
            selected_stock = st.selectbox(
                "Select a stock to view",
                options=st.session_state.selected_stocks
            )

            # Display advanced chart for the selected stock
            if selected_stock in st.session_state.historical_data:
                # Calculate indicators
                data = st.session_state.historical_data[selected_stock]
                data_with_indicators = calculate_indicators(data)

                # Extract indicators for chart
                indicators = {
                    'sma20': data_with_indicators['sma20'],
                    'sma50': data_with_indicators['sma50'],
                    'sma200': data_with_indicators['sma200']
                }

                # Simulate AI predictions (placeholder for future implementation)
                if selected_stock not in st.session_state.ai_predictions:
                    st.session_state.ai_predictions[selected_stock] = simulate_ai_predictions(selected_stock, data)

                # Create advanced chart
                fig = create_advanced_chart(
                    data,
                    selected_stock,
                    st.session_state.chart_timeframe,
                    indicators=indicators
                    # We'll add trades later when we have them
                )

                st.plotly_chart(fig, use_container_width=True)

                # Display technical indicators in an expander
                with st.expander("Technical Indicators"):
                    st.subheader(f"Technical Analysis for {selected_stock}")

                    # Create columns for different indicators
                    col1, col2 = st.columns(2)

                    with col1:
                        st.subheader("Moving Averages")
                        latest = data_with_indicators.iloc[-1]

                        st.metric("SMA 20", f"${latest['sma20']:.2f}")
                        st.metric("SMA 50", f"${latest['sma50']:.2f}")
                        st.metric("SMA 200", f"${latest['sma200']:.2f}")

                    with col2:
                        st.subheader("Oscillators")
                        st.metric("RSI (14)", f"{latest['rsi']:.2f}")
                        st.metric("MACD", f"{latest['macd']:.2f}")
                        st.metric("Signal", f"{latest['signal']:.2f}")
            else:
                st.warning(f"No historical data available for {selected_stock}")

    with tab2:
        # Portfolio view
        if st.session_state.api_connected:
            st.header("Portfolio Overview")

            # Display positions in a table
            if st.session_state.portfolio.get('positions'):
                positions = st.session_state.portfolio['positions']

                # Create DataFrame for better display
                positions_df = pd.DataFrame([
                    {
                        'Symbol': symbol,
                        'Quantity': pos['qty'],
                        'Avg. Entry': f"${pos['avg_entry_price']:.2f}",
                        'Current Price': f"${pos['current_price']:.2f}",
                        'Market Value': f"${pos['market_value']:.2f}",
                        'P&L': f"${pos['unrealized_pl']:.2f}",
                        'P&L %': f"{pos['unrealized_plpc']:.2f}%",
                        'Change Today': f"{pos['change_today']:.2f}%"
                    }
                    for symbol, pos in positions.items()
                ])

                st.dataframe(positions_df, use_container_width=True)

                # Portfolio allocation pie chart
                allocation_chart = create_portfolio_allocation_chart(positions)
                if allocation_chart:
                    st.plotly_chart(allocation_chart, use_container_width=True)
            else:
                st.info("No positions in portfolio.")
        else:
            st.warning("Not connected to Alpaca API. Cannot display portfolio information.")

    with tab3:
        # AI Predictions view
        st.header("AI Trading Predictions (Simulated)")

        if st.session_state.selected_stocks:
            # Create a table of predictions
            predictions_data = []

            for symbol in st.session_state.selected_stocks:
                if symbol in st.session_state.ai_predictions:
                    pred = st.session_state.ai_predictions[symbol]
                    predictions_data.append({
                        'Symbol': symbol,
                        'Prediction': pred['prediction'],
                        'Confidence': f"{pred['confidence'] * 100:.1f}%",
                        'Price Target': f"${pred['price_target']:.2f}" if pred['price_target'] else "N/A",
                        'Stop Loss': f"${pred['stop_loss']:.2f}" if pred['stop_loss'] else "N/A",
                        'Generated': pred['timestamp'].strftime('%H:%M:%S')
                    })

            if predictions_data:
                predictions_df = pd.DataFrame(predictions_data)
                st.dataframe(predictions_df, use_container_width=True)

                # Display a notice that these are simulated
                st.info(
                    "Note: These predictions are simulations and not based on an actual AI model yet. This feature will be implemented in the future.")
            else:
                st.info("No predictions available yet.")
        else:
            st.warning("Please select stocks to generate predictions.")

    # Display last update time
    st.sidebar.text(f"Last updated: {st.session_state.last_update.strftime('%H:%M:%S')}")

    # Refresh button
    if st.sidebar.button("Refresh Data"):
        st.session_state.historical_data = {}
        # Anstatt den gesamten App-Code neu zu starten, kannst du hier z.B. eine einfache Nachricht anzeigen
        st.write("Daten werden aktualisiert... Bitte manuell neu laden.")


if __name__ == "__main__":
    # Make sure logs directory exists
    logs_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'logs')
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)

    main()