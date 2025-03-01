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

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import our modules
from backend.alpaca_api import AlpacaAPI
from backend.crypto_api import CryptoAPI
from backend.database import Database
from backend.trading_bot_extension import ExtendedTradingBot

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'logs',
                                         'prediction_dashboard.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Trading AI - Bot Predictions",
    page_icon="🤖",
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
if 'bot_initialized' not in st.session_state:
    st.session_state.bot_initialized = False
if 'predictions' not in st.session_state:
    st.session_state.predictions = {}
if 'market_summary' not in st.session_state:
    st.session_state.market_summary = {}
if 'recommendations' not in st.session_state:
    st.session_state.recommendations = []
if 'selected_assets' not in st.session_state:
    st.session_state.selected_assets = []
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
            'CRYPTO_SYMBOLS': ['BTC/USD', 'ETH/USD', 'SOL/USD', 'DOGE/USD', 'AVAX/USD'],
            'DEFAULT_TIMEFRAME': '1Day',
        }


# Initialize API clients
@st.cache_resource
def initialize_alpaca_api(config):
    try:
        alpaca_api = AlpacaAPI(
            config_path=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'config',
                                     'settings.json'))

        # Test the connection
        alpaca_api.get_account_info()

        st.session_state.api_connected = True
        return alpaca_api
    except Exception as e:
        logger.error(f"Error initializing Alpaca API: {e}")
        st.session_state.api_connected = False
        return None


@st.cache_resource
def initialize_crypto_api(config):
    try:
        crypto_api = CryptoAPI(
            config_path=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'config',
                                     'settings.json'))

        # Test the connection
        crypto_api.get_account_info()

        return crypto_api
    except Exception as e:
        logger.error(f"Error initializing Crypto API: {e}")
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


def initialize_trading_bot():
    """Initialize the Extended Trading Bot."""
    try:
        bot = ExtendedTradingBot(
            config_path=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'config',
                                     'settings.json'),
            mode='paper'
        )
        st.session_state.bot_initialized = True
        return bot
    except Exception as e:
        logger.error(f"Error initializing Trading Bot: {e}")
        st.session_state.bot_initialized = False
        return None


def get_bot_predictions(bot, assets=None):
    """Get trading predictions from the bot."""
    try:
        predictions = bot.get_prediction_signals(symbols=assets)
        st.session_state.predictions = predictions
        st.session_state.last_update = datetime.now()
        return predictions
    except Exception as e:
        logger.error(f"Error getting bot predictions: {e}")
        return {}


def get_market_summary(bot):
    """Get market summary from the bot."""
    try:
        summary = bot.get_market_summary()
        st.session_state.market_summary = summary
        st.session_state.last_update = datetime.now()
        return summary
    except Exception as e:
        logger.error(f"Error getting market summary: {e}")
        return {}


def get_asset_recommendations(bot, top_n=5):
    """Get top asset recommendations from the bot."""
    try:
        recommendations = bot.get_asset_recommendation(top_n=top_n)
        st.session_state.recommendations = recommendations
        st.session_state.last_update = datetime.now()
        return recommendations
    except Exception as e:
        logger.error(f"Error getting asset recommendations: {e}")
        return []


def create_prediction_table(predictions):
    """Create a DataFrame for predictions display."""
    if not predictions:
        return pd.DataFrame()

    pred_data = []

    for symbol, signal in predictions.items():
        action = signal.get('action', 'UNKNOWN')
        confidence = signal.get('confidence', 0.0)
        current_price = signal.get('current_price', 0.0)

        # Additional details for ML-based predictions
        predicted_price = signal.get('predicted_price', None)
        predicted_change = signal.get('predicted_change_pct', None)

        pred_row = {
            'Symbol': symbol,
            'Action': action,
            'Confidence': f"{confidence * 100:.1f}%",
            'Current Price': f"${current_price:.2f}",
        }

        if predicted_price:
            pred_row['Predicted Price'] = f"${predicted_price:.2f}"
            pred_row['Change %'] = f"{predicted_change:.2f}%" if predicted_change else "N/A"

        pred_data.append(pred_row)

    return pd.DataFrame(pred_data)


def create_confidence_chart(predictions):
    """Create a bar chart of prediction confidences."""
    if not predictions:
        return None

    symbols = []
    confidences = []
    colors = []

    for symbol, signal in predictions.items():
        symbols.append(symbol)
        confidences.append(signal.get('confidence', 0.0) * 100)

        action = signal.get('action', 'HOLD')
        if action == 'BUY':
            colors.append('green')
        elif action == 'SELL':
            colors.append('red')
        else:
            colors.append('gray')

    fig = go.Figure(data=[
        go.Bar(
            x=symbols,
            y=confidences,
            marker_color=colors,
            text=[f"{c:.1f}%" for c in confidences],
            textposition='auto',
        )
    ])

    fig.update_layout(
        title="Trading Bot Prediction Confidence",
        xaxis_title="Symbol",
        yaxis_title="Confidence (%)",
        height=400
    )

    return fig


def create_market_summary_card(summary):
    """Create a card with market summary information."""
    if not summary:
        return st.info("No market summary available")

    # Create columns for summary data
    col1, col2, col3 = st.columns(3)

    # Market regime
    with col1:
        regime = summary.get('market_regime', 'unknown')
        if regime == 'bullish':
            st.success("Market Regime: Bullish 📈")
        elif regime == 'slightly_bullish':
            st.info("Market Regime: Slightly Bullish 📈")
        elif regime == 'bearish':
            st.error("Market Regime: Bearish 📉")
        elif regime == 'slightly_bearish':
            st.warning("Market Regime: Slightly Bearish 📉")
        else:
            st.info("Market Regime: Mixed/Neutral ↔️")

    # Signal counts
    with col2:
        signals = summary.get('signal_counts', {})

        st.metric("Buy Signals", signals.get('BUY', 0))
        st.metric("Sell Signals", signals.get('SELL', 0))
        st.metric("Hold Signals", signals.get('HOLD', 0))

    # Last updated
    with col3:
        if 'timestamp' in summary:
            update_time = summary['timestamp']
            st.write("Last Updated:")
            st.write(update_time.strftime("%Y-%m-%d %H:%M:%S"))

        total_assets = len(summary.get('assets', {}))
        st.metric("Total Assets Tracked", total_assets)


def create_recommendations_section(recommendations):
    """Create a section with top asset recommendations."""
    if not recommendations:
        st.info("No recommendations available")
        return

    st.subheader("Top Asset Recommendations")

    # Create a card for each recommendation
    for i, rec in enumerate(recommendations):
        with st.expander(f"#{i + 1}: {rec['symbol']} - {rec['confidence'] * 100:.1f}% Confidence", expanded=i == 0):
            col1, col2 = st.columns([1, 2])

            with col1:
                st.metric("Current Price", f"${rec['current_price']:.2f}")
                if rec.get('predicted_price'):
                    change = (rec['predicted_price'] - rec['current_price']) / rec['current_price'] * 100
                    st.metric("Target Price", f"${rec['predicted_price']:.2f}", f"{change:.2f}%")

            with col2:
                st.write("**Reasoning:**")
                st.markdown(rec.get('reasoning', 'No detailed reasoning available'))


def main():
    # Set up page header
    st.title("Trading AI - Bot Predictions")

    # Sidebar
    st.sidebar.title("Trading Bot Controls")

    # Load configuration
    config = load_config()

    # Initialize components if not already done
    if not st.session_state.initialized:
        alpaca_api = initialize_alpaca_api(config)
        crypto_api = initialize_crypto_api(config)
        db = initialize_database(config)

        if alpaca_api and db:
            st.session_state.initialized = True
            st.sidebar.success("API connections established")

            # Initialize trading bot
            bot = initialize_trading_bot()
            if bot:
                st.session_state.trading_bot = bot
                st.sidebar.success("Trading Bot initialized")
            else:
                st.sidebar.error("Failed to initialize Trading Bot")
        else:
            if not alpaca_api:
                st.sidebar.error("Failed to connect to Alpaca API. Check your credentials.")
            if not db:
                st.sidebar.error("Failed to connect to database. Check your connection settings.")
            st.stop()
    else:
        alpaca_api = initialize_alpaca_api(config)
        crypto_api = initialize_crypto_api(config)
        db = initialize_database(config)

        if not hasattr(st.session_state, 'trading_bot') or not st.session_state.trading_bot:
            bot = initialize_trading_bot()
            if bot:
                st.session_state.trading_bot = bot

    # Connection status indicators
    st.sidebar.header("Connection Status")
    api_status = "✅ Connected" if st.session_state.api_connected else "❌ Not Connected"
    db_status = "✅ Connected" if st.session_state.db_connected else "❌ Not Connected"
    bot_status = "✅ Initialized" if st.session_state.bot_initialized else "❌ Not Initialized"

    st.sidebar.text(f"API: {api_status}")
    st.sidebar.text(f"Database: {db_status}")
    st.sidebar.text(f"Trading Bot: {bot_status}")

    # Asset selection
    st.sidebar.header("Asset Selection")

    # Get available assets from config
    stock_symbols = config.get('TRADING_SYMBOLS', [])
    crypto_symbols = config.get('CRYPTO_SYMBOLS', [])
    all_symbols = stock_symbols + crypto_symbols

    # Create asset type selector
    asset_type = st.sidebar.selectbox(
        "Asset Type",
        options=["All", "Stocks", "Crypto"],
        index=0
    )

    # Filter assets based on selection
    if asset_type == "Stocks":
        available_assets = stock_symbols
    elif asset_type == "Crypto":
        available_assets = crypto_symbols
    else:
        available_assets = all_symbols

    # Asset multi-select
    selected_assets = st.sidebar.multiselect(
        "Select Assets",
        options=available_assets,
        default=available_assets[:5] if len(available_assets) >= 5 else available_assets
    )

    # Store selected assets in session state
    st.session_state.selected_assets = selected_assets

    # Prediction controls
    st.sidebar.header("Prediction Controls")

    if st.sidebar.button("Generate Predictions", disabled=not st.session_state.bot_initialized):
        with st.spinner("Generating predictions..."):
            # Get predictions
            predictions = get_bot_predictions(st.session_state.trading_bot, selected_assets)

            # Get market summary
            summary = get_market_summary(st.session_state.trading_bot)

            # Get recommendations
            recommendations = get_asset_recommendations(st.session_state.trading_bot, top_n=5)

            st.success(f"Generated predictions for {len(predictions)} assets")

    # Create tabs for different views
    tab1, tab2, tab3 = st.tabs(["Predictions Overview", "Market Summary", "Detailed Analysis"])

    with tab1:
        st.header("Trading Bot Predictions")

        if st.session_state.predictions:
            # Create prediction table
            pred_df = create_prediction_table(st.session_state.predictions)

            # Display table
            st.dataframe(pred_df, use_container_width=True)

            # Create confidence chart
            conf_chart = create_confidence_chart(st.session_state.predictions)

            # Display chart
            if conf_chart:
                st.plotly_chart(conf_chart, use_container_width=True)

            # Add note about prediction generation
            st.info("Trading Bot predictions are based on technical analysis, price patterns, and market conditions.")
        else:
            st.info("No predictions available. Please select assets and click 'Generate Predictions'.")

    with tab2:
        st.header("Market Summary")

        if st.session_state.market_summary:
            # Create market summary card
            create_market_summary_card(st.session_state.market_summary)

            # Display recommendations
            if st.session_state.recommendations:
                create_recommendations_section(st.session_state.recommendations)
            else:
                st.info("No asset recommendations available.")
        else:
            st.info("No market summary available. Please generate predictions first.")

    with tab3:
        st.header("Detailed Analysis")

        if st.session_state.predictions:
            # Create asset selector
            detailed_asset = st.selectbox(
                "Select Asset for Detailed Analysis",
                options=list(st.session_state.predictions.keys())
            )

            # Display detailed prediction and reasoning
            if detailed_asset in st.session_state.predictions:
                signal = st.session_state.predictions[detailed_asset]

                # Basic information
                col1, col2, col3 = st.columns(3)

                with col1:
                    action = signal.get('action', 'UNKNOWN')
                    if action == 'BUY':
                        st.success(f"Recommendation: {action}")
                    elif action == 'SELL':
                        st.error(f"Recommendation: {action}")
                    else:
                        st.info(f"Recommendation: {action}")

                with col2:
                    confidence = signal.get('confidence', 0.0)
                    st.metric("Confidence", f"{confidence * 100:.1f}%")

                with col3:
                    current_price = signal.get('current_price', 0.0)
                    st.metric("Current Price", f"${current_price:.2f}")

                # Prediction reasoning
                if 'reasoning' in signal and signal['reasoning']:
                    st.subheader("Prediction Reasoning")
                    st.markdown(signal['reasoning'])
                else:
                    st.info("No detailed reasoning available for this prediction.")

                # Display analysis details if available
                if 'analysis' in signal and signal['analysis']:
                    st.subheader("Analysis Details")
                    analysis = signal['analysis']

                    # Price trend
                    if 'price_trend' in analysis:
                        with st.expander("Price Trend Analysis", expanded=True):
                            trend = analysis['price_trend']

                            col1, col2, col3 = st.columns(3)

                            with col1:
                                trend_type = trend.get('trend', 'neutral').replace('_', ' ').title()
                                st.write(f"**Trend:** {trend_type}")
                                st.write(f"**Strength:** {trend.get('trend_strength', 0) * 100:.1f}%")

                            with col2:
                                st.write("**Moving Averages:**")
                                st.write(f"SMA5: ${trend.get('sma5', 0):.2f}")
                                st.write(f"SMA20: ${trend.get('sma20', 0):.2f}")

                            with col3:
                                momentum = trend.get('momentum', 0) * 100
                                if momentum > 0:
                                    st.write(f"**Momentum:** +{momentum:.2f}%")
                                else:
                                    st.write(f"**Momentum:** {momentum:.2f}%")

                    # Support/Resistance
                    if 'support_resistance' in analysis:
                        with st.expander("Support & Resistance"):
                            sr = analysis['support_resistance']

                            col1, col2 = st.columns(2)

                            with col1:
                                st.write("**Nearest Support:**")
                                if sr.get('nearest_support'):
                                    st.write(f"${sr.get('nearest_support'):.2f}")
                                    st.write(f"Distance: {sr.get('dist_to_support', 0):.2f}%")
                                else:
                                    st.write("None detected")

                            with col2:
                                st.write("**Nearest Resistance:**")
                                if sr.get('nearest_resistance'):
                                    st.write(f"${sr.get('nearest_resistance'):.2f}")
                                    st.write(f"Distance: {sr.get('dist_to_resistance', 0):.2f}%")
                                else:
                                    st.write("None detected")

                    # Other analysis sections
                    for section_name in ['volume_analysis', 'market_regime', 'volatility']:
                        if section_name in analysis:
                            with st.expander(section_name.replace('_', ' ').title()):
                                for key, value in analysis[section_name].items():
                                    if isinstance(value, float):
                                        st.write(f"**{key.replace('_', ' ').title()}:** {value:.4f}")
                                    else:
                                        st.write(f"**{key.replace('_', ' ').title()}:** {value}")
            else:
                st.warning(f"No prediction data available for {detailed_asset}")
        else:
            st.info("No predictions available. Please generate predictions first.")

    # Display last update time
    st.sidebar.text(f"Last updated: {st.session_state.last_update.strftime('%H:%M:%S')}")


if __name__ == "__main__":
    # Make sure logs directory exists
    logs_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'logs')
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)

    main()