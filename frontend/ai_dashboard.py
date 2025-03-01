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
import uuid  # Für eindeutige Schlüssel

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import our modules
from backend.alpaca_api import AlpacaAPI
from backend.database import Database
from backend.trade_bot import TradingBot, MLTradingStrategy
from backend.train import LSTMModel, create_features

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(
            os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'logs', 'ai_dashboard.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Trading AI Dashboard - ML Models",
    page_icon="🧠",
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
if 'selected_stocks' not in st.session_state:
    st.session_state.selected_stocks = []
if 'historical_data' not in st.session_state:
    st.session_state.historical_data = {}
if 'ml_predictions' not in st.session_state:
    st.session_state.ml_predictions = {}
if 'trading_signals' not in st.session_state:
    st.session_state.trading_signals = {}
if 'model_info' not in st.session_state:
    st.session_state.model_info = {}
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


# Initialize Trading Bot (without running it)
@st.cache_resource
def initialize_trading_bot(config):
    try:
        # Create trading bot instance but don't start it
        bot = TradingBot(config_path=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'config',
                                                  'settings.json'), mode='paper')
        return bot
    except Exception as e:
        logger.error(f"Error initializing trading bot: {e}")
        return None


# Fetch historical data for a symbol
def fetch_historical_data(alpaca_api, symbols, timeframe='1Day', days=30):
    start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
    end_date = datetime.now().strftime('%Y-%m-%d')

    try:
        logger.info(f"Fetching historical data for {symbols}")
        return alpaca_api.get_historical_bars(symbols, timeframe=timeframe, start=start_date, end=end_date)
    except Exception as e:
        logger.error(f"Error fetching historical data: {e}")
        return {}


# Get ML model predictions for a symbol
def get_ml_predictions(trading_bot, symbol, data, num_predictions=5):
    # Get the ML strategy for the symbol
    if symbol not in trading_bot.strategies:
        logger.warning(f"No strategy found for {symbol}")
        return None

    strategy = trading_bot.strategies.get(symbol)
    if not isinstance(strategy, MLTradingStrategy):
        logger.warning(f"Strategy for {symbol} is not an ML strategy")
        return None

    # Generate prediction
    current_signal = strategy.generate_signal(data)

    # If we have a valid prediction, forecast future values
    if current_signal and 'predicted_price' in current_signal:
        # Create a list of predictions (current + future)
        predictions = [
            {
                'timestamp': datetime.now() + timedelta(days=i),
                'predicted_price': current_signal['predicted_price'] * (1 + i * 0.01),  # Simple future projection
                'confidence': max(0.1, current_signal['confidence'] - i * 0.1)
                # Lower confidence for further predictions
            }
            for i in range(num_predictions)
        ]

        # Add the current price
        current_price = data.iloc[-1]['close']

        return {
            'current_price': current_price,
            'current_signal': current_signal,
            'predictions': predictions
        }

    return None


# Get model information from the database
def get_model_info(db, symbol):
    try:
        # Get active model from database
        active_model = db.get_active_model()
        if not active_model:
            logger.warning("No active model found in database")
            return None

        # Check if the model is for this symbol
        if symbol not in active_model.name:
            logger.warning(f"Active model is not for {symbol}")
            return None

        # Extract performance metrics
        if active_model.performance_metrics:
            try:
                metrics = json.loads(active_model.performance_metrics)
            except:
                metrics = {}
        else:
            metrics = {}

        # Get recent trades for this model
        trades = db.get_stock_trades(symbol=symbol, limit=10)

        return {
            'name': active_model.name,
            'version': active_model.version,
            'model_type': active_model.model_type,
            'created_at': active_model.created_at,
            'metrics': metrics,
            'trades': trades
        }
    except Exception as e:
        logger.error(f"Error getting model info: {e}")
        return None


# Create a prediction chart
def create_prediction_chart(symbol, data, predictions):
    fig = go.Figure()

    # Add historical prices
    fig.add_trace(go.Scatter(
        x=data.index,
        y=data['close'],
        mode='lines',
        name='Historical Price',
        line=dict(color='blue', width=2)
    ))

    # Add predictions
    if predictions and 'predictions' in predictions:
        pred_dates = [p['timestamp'] for p in predictions['predictions']]
        pred_prices = [p['predicted_price'] for p in predictions['predictions']]

        fig.add_trace(go.Scatter(
            x=pred_dates,
            y=pred_prices,
            mode='lines+markers',
            name='Predicted Price',
            line=dict(color='green', width=2, dash='dash')
        ))

        # Add confidence intervals (simple implementation)
        upper_bound = [p['predicted_price'] * (1 + p['confidence'] * 0.1) for p in predictions['predictions']]
        lower_bound = [p['predicted_price'] * (1 - p['confidence'] * 0.1) for p in predictions['predictions']]

        fig.add_trace(go.Scatter(
            x=pred_dates + pred_dates[::-1],
            y=upper_bound + lower_bound[::-1],
            fill='toself',
            fillcolor='rgba(0, 255, 0, 0.1)',
            line=dict(color='rgba(0, 255, 0, 0.2)'),
            name='Confidence Interval'
        ))

    # Update layout
    fig.update_layout(
        title=f"{symbol} - Price Prediction",
        xaxis_title="Date",
        yaxis_title="Price ($)",
        height=500,
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )

    return fig


# Create a model performance chart
def create_model_performance_chart(model_info):
    if not model_info or 'metrics' not in model_info:
        return None

    metrics = model_info['metrics']

    # Create figure
    fig = go.Figure()

    # Add metrics as a bar chart
    metric_names = []
    metric_values = []

    for name, value in metrics.items():
        metric_names.append(name.upper())
        metric_values.append(value)

    fig.add_trace(go.Bar(
        x=metric_names,
        y=metric_values,
        marker_color='rgb(55, 83, 109)'
    ))

    # Update layout
    fig.update_layout(
        title="Model Performance Metrics",
        xaxis_title="Metric",
        yaxis_title="Value",
        height=400
    )

    return fig


# Create a trades performance chart
def create_trades_performance_chart(trades):
    if not trades:
        return None

    # Convert trades to DataFrame
    trades_data = []
    for trade in trades:
        trades_data.append({
            'timestamp': trade.timestamp,
            'symbol': trade.stock.symbol if hasattr(trade, 'stock') else 'Unknown',
            'type': trade.trade_type,
            'quantity': trade.quantity,
            'price': trade.price,
            'profit_loss': trade.profit_loss if trade.profit_loss else 0.0
        })

    if not trades_data:
        return None

    df = pd.DataFrame(trades_data)

    # Create figure
    fig = go.Figure()

    # Add profit/loss as a bar chart
    fig.add_trace(go.Bar(
        x=df['timestamp'],
        y=df['profit_loss'],
        marker_color=['green' if pl >= 0 else 'red' for pl in df['profit_loss']]
    ))

    # Update layout
    fig.update_layout(
        title="Trading Performance",
        xaxis_title="Date",
        yaxis_title="Profit/Loss ($)",
        height=400
    )

    return fig


# Main function
def main():
    # Set up page header
    st.title("Trading AI Dashboard - ML Models")

    # Sidebar
    st.sidebar.title("ML Model Settings")

    # Load configuration
    config = load_config()

    # Initialize components if not already done
    if not st.session_state.initialized:
        alpaca_api = initialize_alpaca_api(config)
        db = initialize_database(config)
        trading_bot = initialize_trading_bot(config)

        if alpaca_api and db and trading_bot:
            st.session_state.initialized = True
            st.sidebar.success("Components initialized successfully!")
        else:
            if not alpaca_api:
                st.sidebar.error("Failed to connect to Alpaca API. Check your credentials.")
            if not db:
                st.sidebar.error("Failed to connect to database. Check your connection settings.")
            if not trading_bot:
                st.sidebar.error("Failed to initialize Trading Bot.")
            return
    else:
        alpaca_api = initialize_alpaca_api(config)
        db = initialize_database(config)
        trading_bot = initialize_trading_bot(config)

    # Connection status indicators
    st.sidebar.header("Connection Status")
    api_status = "✅ Connected" if st.session_state.api_connected else "❌ Not Connected"
    db_status = "✅ Connected" if st.session_state.db_connected else "❌ Not Connected"

    st.sidebar.text(f"Alpaca API: {api_status}")
    st.sidebar.text(f"Database: {db_status}")

    # Chart settings
    st.sidebar.header("Chart Settings")

    timeframe = st.sidebar.selectbox(
        "Timeframe",
        options=['1Day', '1Hour', '15Min', '5Min', '1Min'],
        index=0,  # Default to 1Day
        key="timeframe_selector"
    )

    days = st.sidebar.slider(
        "Data Period (days)",
        min_value=1,
        max_value=90,
        value=30,
        key="days_selector"
    )

    # Stock selection
    st.sidebar.header("Stock Selection")
    default_stocks = config.get('TRADING_SYMBOLS', ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA'])
    selected_stocks = st.sidebar.multiselect(
        "Select stocks for analysis",
        options=default_stocks + ['NVDA', 'META', 'NFLX', 'PYPL', 'DIS', 'INTC', 'AMD', 'CSCO', 'ADBE', 'ORCL'],
        default=default_stocks[:3]  # Default to first 3 stocks
    )

    # Store selected stocks in session state
    if selected_stocks != st.session_state.selected_stocks:
        st.session_state.selected_stocks = selected_stocks
        st.session_state.historical_data = {}  # Clear historical data

    # Main content
    if not st.session_state.selected_stocks:
        st.warning("Please select at least one stock for analysis.")
        return

    # Fetch historical data if needed
    if not st.session_state.historical_data:
        with st.spinner("Fetching historical data..."):
            st.session_state.historical_data = fetch_historical_data(
                alpaca_api,
                st.session_state.selected_stocks,
                timeframe=timeframe,
                days=days
            )

    # Create tabs for each selected stock
    tabs = st.tabs(st.session_state.selected_stocks)

    # Process each selected stock
    for i, symbol in enumerate(st.session_state.selected_stocks):
        with tabs[i]:
            st.header(f"{symbol} - ML Model Analysis")

            # Check if we have data for this symbol
            if symbol not in st.session_state.historical_data or st.session_state.historical_data[symbol].empty:
                st.warning(f"No historical data available for {symbol}")
                continue

            # Get data for this symbol
            data = st.session_state.historical_data[symbol]

            # Create columns for layout
            col1, col2 = st.columns([2, 1])

            with col1:
                # Get ML predictions
                predictions = get_ml_predictions(trading_bot, symbol, data)

                if predictions:
                    # Create prediction chart
                    fig = create_prediction_chart(symbol, data, predictions)
                    # Eindeutiger Schlüssel für jedes Chart
                    st.plotly_chart(fig, use_container_width=True, key=f"prediction_chart_{symbol}")

                    # Show prediction details
                    current_signal = predictions.get('current_signal', {})
                    action = current_signal.get('action', 'UNKNOWN')
                    confidence = current_signal.get('confidence', 0.0)

                    # Display signal with appropriate color
                    if action == 'BUY':
                        st.success(f"**Signal:** {action} with {confidence * 100:.1f}% confidence")
                    elif action == 'SELL':
                        st.error(f"**Signal:** {action} with {confidence * 100:.1f}% confidence")
                    else:
                        st.info(f"**Signal:** {action} with {confidence * 100:.1f}% confidence")

                    # Show current and predicted prices
                    current_price = predictions.get('current_price', 0.0)
                    predicted_price = current_signal.get('predicted_price', 0.0)
                    price_change = (predicted_price - current_price) / current_price * 100

                    col1a, col1b, col1c = st.columns(3)
                    col1a.metric("Current Price", f"${current_price:.2f}")
                    col1b.metric("Predicted Next Price", f"${predicted_price:.2f}", f"{price_change:.2f}%")
                    col1c.metric("Confidence", f"{confidence * 100:.1f}%")

                else:
                    st.warning(f"No ML predictions available for {symbol}. The model may not be trained yet.")

            with col2:
                # Get model information
                model_info = get_model_info(db, symbol)

                if model_info:
                    st.subheader("Model Information")
                    st.write(f"**Name:** {model_info['name']}")
                    st.write(f"**Type:** {model_info['model_type']}")
                    st.write(f"**Version:** {model_info['version']}")
                    st.write(f"**Created:** {model_info['created_at']}")

                    # Create metrics chart
                    metrics_fig = create_model_performance_chart(model_info)
                    if metrics_fig:
                        # Eindeutiger Schlüssel für Metriken-Chart
                        st.plotly_chart(metrics_fig, use_container_width=True, key=f"metrics_chart_{symbol}")

                    # Show trades if available
                    if 'trades' in model_info and model_info['trades']:
                        st.subheader("Recent Trades")
                        trades_fig = create_trades_performance_chart(model_info['trades'])
                        if trades_fig:
                            # Eindeutiger Schlüssel für Trades-Chart
                            st.plotly_chart(trades_fig, use_container_width=True, key=f"trades_chart_{symbol}")
                else:
                    st.warning(f"No model information available for {symbol}")

            # Additional detailed analysis
            st.subheader("ML Model Detailed Analysis")

            # Show feature importance (simulated)
            st.write("#### Feature Importance")
            feature_data = {
                'Feature': ['price_momentum', 'volume', 'rsi', 'sma_cross', 'macd'],
                'Importance': [0.35, 0.25, 0.20, 0.15, 0.05]
            }
            feature_df = pd.DataFrame(feature_data)

            # Create horizontal bar chart
            fig = px.bar(
                feature_df,
                y='Feature',
                x='Importance',
                orientation='h',
                title="Feature Importance (Simulated)",
                color='Importance',
                color_continuous_scale='Viridis'
            )

            fig.update_layout(height=300)
            # Eindeutiger Schlüssel für Feature-Importance-Chart
            st.plotly_chart(fig, use_container_width=True, key=f"feature_importance_{symbol}")

            st.write("#### Model Performance Over Time")
            st.info(
                "This is a placeholder for model performance tracking over time. This will be implemented as you collect more prediction data.")

    # Refresh button
    if st.sidebar.button("Refresh Data"):
        st.session_state.historical_data = {}  # Clear historical data to force refresh
        st.session_state.last_update = datetime.now()
        st.rerun()

    # Display last update time
    st.sidebar.text(f"Last updated: {st.session_state.last_update.strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    # Make sure logs directory exists
    logs_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'logs')
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)

    main()