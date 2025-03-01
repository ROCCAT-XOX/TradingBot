import os
import sys
import json
import logging
import numpy as np
import pandas as pd
import torch
from datetime import datetime, timedelta
import time
import pickle
from pathlib import Path
import threading
import queue

# Add the parent directory to the path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

# Import our modules
from backend.alpaca_api import AlpacaAPI
from backend.database import Database, AIModel
from backend.train import LSTMModel, create_features

# For environment variable access
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(parent_dir, 'logs', 'trade_bot.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class TradingStrategy:
    """Base class for trading strategies."""

    def __init__(self, symbol, timeframe='1Day'):
        """
        Initialize the trading strategy.

        Args:
            symbol (str): Stock symbol
            timeframe (str): Timeframe for the data
        """
        self.symbol = symbol
        self.timeframe = timeframe

    def generate_signal(self, data):
        """
        Generate trading signal.

        Args:
            data (pd.DataFrame): DataFrame with price data

        Returns:
            dict: Trading signal with action, confidence, etc.
        """
        raise NotImplementedError("Subclasses must implement generate_signal method")


class SimpleTechnicalStrategy(TradingStrategy):
    """Simple trading strategy based on technical indicators."""

    def __init__(self, symbol, timeframe='1Day', sma_short=20, sma_long=50, rsi_period=14):
        """
        Initialize the simple technical strategy.

        Args:
            symbol (str): Stock symbol
            timeframe (str): Timeframe for the data
            sma_short (int): Short-term SMA period
            sma_long (int): Long-term SMA period
            rsi_period (int): RSI period
        """
        super().__init__(symbol, timeframe)
        self.sma_short = sma_short
        self.sma_long = sma_long
        self.rsi_period = rsi_period

    def generate_signal(self, data):
        """
        Generate trading signal using technical indicators.

        Args:
            data (pd.DataFrame): DataFrame with OHLCV data

        Returns:
            dict: Trading signal with action, confidence, etc.
        """
        try:
            # Calculate SMA
            data['sma_short'] = data['close'].rolling(window=self.sma_short).mean()
            data['sma_long'] = data['close'].rolling(window=self.sma_long).mean()

            # Calculate RSI
            delta = data['close'].diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            avg_gain = gain.rolling(window=self.rsi_period).mean()
            avg_loss = loss.rolling(window=self.rsi_period).mean()
            rs = avg_gain / avg_loss
            data['rsi'] = 100 - (100 / (1 + rs))

            # Get the most recent values
            last_close = data.iloc[-1]['close']
            last_sma_short = data.iloc[-1]['sma_short']
            last_sma_long = data.iloc[-1]['sma_long']
            last_rsi = data.iloc[-1]['rsi']

            # Generate signals
            sma_signal = 1 if last_sma_short > last_sma_long else (-1 if last_sma_short < last_sma_long else 0)
            rsi_signal = 1 if last_rsi < 30 else (-1 if last_rsi > 70 else 0)

            # Combine signals
            signal_strength = (sma_signal + rsi_signal) / 2

            # Determine action
            if signal_strength > 0.5:
                action = 'BUY'
                confidence = min(0.5 + abs(signal_strength) * 0.5, 0.95)
            elif signal_strength < -0.5:
                action = 'SELL'
                confidence = min(0.5 + abs(signal_strength) * 0.5, 0.95)
            else:
                action = 'HOLD'
                confidence = 0.5

            # Create signal dictionary
            signal = {
                'symbol': self.symbol,
                'timestamp': datetime.now(),
                'action': action,
                'confidence': confidence,
                'current_price': last_close,
                'sma_short': last_sma_short,
                'sma_long': last_sma_long,
                'rsi': last_rsi,
                'timeframe': self.timeframe
            }

            logger.info(f"Generated technical signal for {self.symbol}: {action} with {confidence:.2f} confidence")

            return signal

        except Exception as e:
            logger.error(f"Error generating technical signal for {self.symbol}: {e}")
            return {'action': 'HOLD', 'confidence': 0.0}


class TradingBot:
    """AI-powered trading bot using ML models or technical strategies."""

    def __init__(self, config_path=None, mode='paper'):
        """
        Initialize the trading bot.

        Args:
            config_path (str, optional): Path to the configuration file
            mode (str): Trading mode ('paper' or 'live')
        """
        self.config_path = config_path
        self.mode = mode
        self.running = False
        self.strategies = {}
        self.signals_queue = queue.Queue()

        # Load configuration
        self.load_config()

        # Initialize components
        self.initialize_components()

    def load_config(self):
        """Load configuration from file."""
        if self.config_path is None:
            self.config_path = os.path.join(parent_dir, 'config', 'settings.json')

        if os.path.exists(self.config_path):
            with open(self.config_path, 'r') as f:
                self.config = json.load(f)
                logger.info(f"Loaded configuration from {self.config_path}")
        else:
            logger.warning(f"Configuration file not found: {self.config_path}")
            # Default configuration
            self.config = {
                'TRADING_SYMBOLS': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA'],
                'DEFAULT_TIMEFRAME': '1Day',
                'PAPER_TRADING': True,
                'TRADE_AMOUNT': 1000.0,
                'CONFIDENCE_THRESHOLD': 0.6,
                'MAX_POSITION_SIZE': 0.1,  # Max 10% of portfolio in a single position
                'TRADING_STRATEGY': 'ML',  # 'ML' or 'TECHNICAL'
                'USE_ML_MODEL': True,
                'USE_TECHNICAL_INDICATORS': True,
                'DATA_REFRESH_INTERVAL': 300  # 5 minutes
            }

    def initialize_components(self):
        """Initialize components (API, database, strategies)."""
        try:
            # Initialize AlpacaAPI
            self.alpaca_api = AlpacaAPI(config_path=self.config_path)

            # Initialize database
            self.db = Database()

            # Initialize trading strategies
            self.initialize_strategies()

            logger.info("All components initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing components: {e}")
            raise

    def initialize_strategies(self):
        """Initialize trading strategies for each symbol."""
        symbols = self.config.get('TRADING_SYMBOLS', [])
        strategy_type = self.config.get('TRADING_STRATEGY', 'ML')

        model_path = os.path.join(parent_dir, 'backend', 'models')
        timeframe = self.config.get('DEFAULT_TIMEFRAME', '1Day')

        for symbol in symbols:
            try:
                if strategy_type == 'ML' and self.config.get('USE_ML_MODEL', True):
                    # Try to load ML strategy first
                    ml_strategy = MLTradingStrategy(symbol, model_path, timeframe)
                    if ml_strategy.model is not None:
                        self.strategies[symbol] = ml_strategy
                        logger.info(f"Using ML strategy for {symbol}")
                        continue

                # Fall back to technical strategy if ML strategy fails or is not selected
                if self.config.get('USE_TECHNICAL_INDICATORS', True):
                    tech_strategy = SimpleTechnicalStrategy(symbol, timeframe)
                    self.strategies[symbol] = tech_strategy
                    logger.info(f"Using technical strategy for {symbol}")
            except Exception as e:
                logger.error(f"Error initializing strategy for {symbol}: {e}")

    def fetch_latest_data(self, symbol, timeframe='1Day', bars=100):
        """
        Fetch the latest data for a symbol.

        Args:
            symbol (str): Stock symbol
            timeframe (str): Timeframe for the data
            bars (int): Number of bars to fetch

        Returns:
            pd.DataFrame: DataFrame with the latest data
        """
        try:
            # Calculate start date (bars * timeframe) days ago
            days = bars
            if timeframe == '1Min':
                days = max(1, bars // 1440)  # ~1440 minutes in a day
            elif timeframe == '5Min':
                days = max(1, bars // 288)  # ~288 5-minute bars in a day
            elif timeframe == '15Min':
                days = max(1, bars // 96)  # ~96 15-minute bars in a day
            elif timeframe == '1Hour':
                days = max(1, bars // 24)  # 24 hours in a day

            start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
            end_date = datetime.now().strftime('%Y-%m-%d')

            # Fetch data
            data = self.alpaca_api.get_historical_bars(
                [symbol],
                timeframe=timeframe,
                start=start_date,
                end=end_date,
                limit=bars
            )

            if symbol in data and not data[symbol].empty:
                return data[symbol]
            else:
                logger.warning(f"No data fetched for {symbol}")
                return pd.DataFrame()

        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {e}")
            return pd.DataFrame()

    def generate_signals(self):
        """Generate trading signals for all symbols."""
        signals = []

        for symbol, strategy in self.strategies.items():
            try:
                # Fetch latest data
                data = self.fetch_latest_data(
                    symbol,
                    timeframe=strategy.timeframe,
                    bars=100  # Fetch enough bars for feature calculation
                )

                if data.empty:
                    logger.warning(f"No data available for {symbol}, skipping signal generation")
                    continue

                # Generate signal
                signal = strategy.generate_signal(data)

                # Add to signals list
                if signal:
                    signals.append(signal)
                    # Add to queue for processing
                    self.signals_queue.put(signal)

            except Exception as e:
                logger.error(f"Error generating signal for {symbol}: {e}")

        return signals

    def process_signals(self):
        """Process trading signals and execute trades."""
        while self.running:
            try:
                # Get signal from queue (timeout to prevent blocking forever)
                signal = self.signals_queue.get(timeout=1)

                # Process signal
                self.process_signal(signal)

                # Mark task as done
                self.signals_queue.task_done()

            except queue.Empty:
                # Queue is empty, continue the loop
                pass
            except Exception as e:
                logger.error(f"Error processing signals: {e}")

            # Sleep to prevent tight looping
            time.sleep(0.1)

    def process_signal(self, signal):
        """
        Process a trading signal and execute trade if appropriate.

        Args:
            signal (dict): Trading signal
        """
        try:
            # Extract signal data
            symbol = signal.get('symbol')
            action = signal.get('action', 'HOLD')
            confidence = signal.get('confidence', 0.0)
            current_price = signal.get('current_price')

            # Check confidence threshold
            confidence_threshold = self.config.get('CONFIDENCE_THRESHOLD', 0.6)

            if confidence < confidence_threshold:
                logger.info(
                    f"Signal for {symbol} rejected: confidence {confidence:.2f} below threshold {confidence_threshold}")
                return

            # Get account info
            account = self.alpaca_api.get_account_info()
            buying_power = float(account.buying_power)
            portfolio_value = float(account.equity)

            # Get current positions
            positions = self.alpaca_api.get_positions()
            current_position = None

            for position in positions:
                if position.symbol == symbol:
                    current_position = position
                    break

            # Determine trade type and quantity
            if action == 'BUY':
                # Check if we already have a position
                if current_position is not None:
                    logger.info(f"Already have position in {symbol}, skipping buy")
                    return

                # Calculate trade amount
                trade_amount = min(
                    self.config.get('TRADE_AMOUNT', 1000.0),
                    portfolio_value * self.config.get('MAX_POSITION_SIZE', 0.1)
                )

                # Check buying power
                if trade_amount > buying_power:
                    logger.warning(
                        f"Not enough buying power for {symbol}: need ${trade_amount:.2f}, have ${buying_power:.2f}")
                    return

                # Calculate quantity
                quantity = int(trade_amount / current_price)

                if quantity <= 0:
                    logger.warning(f"Calculated quantity for {symbol} is {quantity}, skipping trade")
                    return

                # Execute buy order
                self.execute_trade(symbol, 'buy', quantity, reason=f"ML signal with {confidence:.2f} confidence")

            elif action == 'SELL':
                # Check if we have a position to sell
                if current_position is None:
                    logger.info(f"No position in {symbol} to sell, skipping")
                    return

                # Get position quantity
                quantity = abs(int(float(current_position.qty)))

                if quantity <= 0:
                    logger.warning(f"Position quantity for {symbol} is {quantity}, skipping trade")
                    return

                # Execute sell order
                self.execute_trade(symbol, 'sell', quantity, reason=f"ML signal with {confidence:.2f} confidence")

        except Exception as e:
            logger.error(f"Error processing signal for {symbol}: {e}")

    def execute_trade(self, symbol, side, quantity, order_type='market', time_in_force='day', reason=None):
        """
        Execute a trade.

        Args:
            symbol (str): Stock symbol
            side (str): 'buy' or 'sell'
            quantity (int): Number of shares
            order_type (str): Order type ('market', 'limit', etc.)
            time_in_force (str): Time in force ('day', 'gtc', etc.)
            reason (str, optional): Reason for the trade

        Returns:
            dict: Order information
        """
        if not self.running:
            logger.warning(f"Trading bot is not running, skipping {side} order for {symbol}")
            return None

        try:
            # Log the trade
            logger.info(f"Executing {side} order for {quantity} shares of {symbol} ({reason})")

            # Skip execution if in test mode
            if hasattr(self, 'test_mode') and self.test_mode:
                logger.info(f"TEST MODE: Skipping order execution")
                return {
                    'symbol': symbol,
                    'side': side,
                    'qty': quantity,
                    'type': order_type,
                    'time_in_force': time_in_force
                }

            # Execute the order
            order = self.alpaca_api.place_order(
                symbol=symbol,
                qty=quantity,
                side=side,
                order_type=order_type,
                time_in_force=time_in_force
            )

            # Store the trade in the database
            if order and hasattr(order, 'id'):
                # Get the strategy that generated the signal
                strategy = self.strategies.get(symbol)
                strategy_name = strategy.__class__.__name__ if strategy else "Unknown"

                # Get the active model from the database
                model = self.db.get_active_model()
                model_id = model.id if model else None

                # Add trade to database
                self.db.add_trade(
                    stock_symbol=symbol,
                    trade_type=side,
                    quantity=quantity,
                    price=float(order.filled_avg_price) if hasattr(order,
                                                                   'filled_avg_price') and order.filled_avg_price else 0.0,
                    timestamp=datetime.now(),
                    order_id=order.id,
                    strategy=strategy_name,
                    model_id=model_id,
                    notes=reason
                )

            return order

        except Exception as e:
            logger.error(f"Error executing {side} order for {symbol}: {e}")
            return None

    def run(self, test_mode=False):
        """Run the trading bot in the background."""
        self.test_mode = test_mode
        self.running = True

        # Start signal processing thread
        self.processing_thread = threading.Thread(target=self.process_signals, daemon=True)
        self.processing_thread.start()

        logger.info(f"Trading bot started in {'TEST' if test_mode else 'LIVE'} mode")

        # Main loop
        try:
            while self.running:
                # Generate signals for all symbols
                self.generate_signals()

                # Wait for the next iteration
                interval = self.config.get('DATA_REFRESH_INTERVAL', 300)  # Default 5 minutes
                logger.info(f"Waiting {interval} seconds until next signal generation...")

                # Sleep in small increments to allow for clean shutdown
                for _ in range(interval):
                    if not self.running:
                        break
                    time.sleep(1)

        except KeyboardInterrupt:
            logger.info("Trading bot stopped by user")
            self.stop()
        except Exception as e:
            logger.error(f"Error in trading bot main loop: {e}")
            self.stop()

    def stop(self):
        """Stop the trading bot."""
        logger.info("Stopping trading bot...")
        self.running = False

        # Wait for processing thread to finish
        if hasattr(self, 'processing_thread') and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=5)

        logger.info("Trading bot stopped")

    def run_backtest(self, start_date, end_date, initial_capital=10000):
        """
        Run a backtest of the strategy.

        Args:
            start_date (str): Start date in format 'YYYY-MM-DD'
            end_date (str): End date in format 'YYYY-MM-DD'
            initial_capital (float): Initial capital

        Returns:
            dict: Backtest results
        """
        # TODO: Implement backtesting functionality
        logger.info("Backtesting not implemented yet")
        return {}


if __name__ == "__main__":
    # Parse command line arguments
    import argparse

    parser = argparse.ArgumentParser(description='Trading AI Bot')
    parser.add_argument('--config', type=str, help='Path to configuration file')
    parser.add_argument('--mode', type=str, choices=['paper', 'live', 'test'], default='paper', help='Trading mode')
    parser.add_argument('--backtest', action='store_true', help='Run in backtest mode')
    parser.add_argument('--start', type=str, help='Start date for backtest (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, help='End date for backtest (YYYY-MM-DD)')

    args = parser.parse_args()

    # Initialize trading bot
    bot = TradingBot(config_path=args.config, mode=args.mode)

    # Run trading bot
    if args.backtest:
        if not args.start or not args.end:
            print("Start and end dates required for backtesting")
            sys.exit(1)

        results = bot.run_backtest(args.start, args.end)
        print(f"Backtest results: {results}")
    else:
        bot.run(test_mode=(args.mode == 'test'))


class MLTradingStrategy(TradingStrategy):
    """Trading strategy based on machine learning model."""

    def __init__(self, symbol, model_path, timeframe='1Day', sequence_length=20):
        """
        Initialize the ML trading strategy.

        Args:
            symbol (str): Stock symbol
            model_path (str): Path to the model directory
            timeframe (str): Timeframe for the data
            sequence_length (int): Sequence length for LSTM input
        """
        super().__init__(symbol, timeframe)
        self.model_path = model_path
        self.sequence_length = sequence_length
        self.model = None
        self.scaler = None
        self.metadata = None

        # Load model and metadata
        self.load_model()

    def load_model(self):
        """Load the trained model and metadata."""
        try:
            # Find the latest model for the symbol
            model_dir = Path(self.model_path)
            model_files = list(model_dir.glob(f"{self.symbol}_lstm_*.pth"))

            if not model_files:
                logger.error(f"No model found for {self.symbol}")
                return False

            # Get the most recent model file
            latest_model = sorted(model_files)[-1]
            model_name = latest_model.stem

            # Load metadata
            metadata_file = model_dir / f"{model_name}_metadata.json"
            if metadata_file.exists():
                with open(metadata_file, 'r') as f:
                    self.metadata = json.load(f)

                # Update sequence length from metadata if available
                if 'model_params' in self.metadata and 'sequence_length' in self.metadata['model_params']:
                    self.sequence_length = self.metadata['model_params']['sequence_length']

            # Load scaler
            scaler_file = model_dir / f"{model_name}_scaler.pkl"
            if scaler_file.exists():
                with open(scaler_file, 'rb') as f:
                    self.scaler = pickle.load(f)

            # Initialize model
            input_size = len(self.metadata.get('feature_columns', [])) if self.metadata else 64
            hidden_size = self.metadata.get('model_params', {}).get('hidden_size', 128)
            num_layers = self.metadata.get('model_params', {}).get('num_layers', 2)
            dropout = self.metadata.get('model_params', {}).get('dropout', 0.2)

            # Create model with the same architecture as during training
            self.model = LSTMModel(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                output_size=1,
                dropout=dropout
            )

            # Load model state
            self.model.load_state_dict(torch.load(latest_model, map_location=torch.device('cpu')))
            self.model.eval()

            logger.info(f"Model loaded successfully for {self.symbol}")
            return True

        except Exception as e:
            logger.error(f"Error loading model for {self.symbol}: {e}")
            return False

    def preprocess_data(self, data):
        """
        Preprocess data for model input.

        Args:
            data (pd.DataFrame): DataFrame with OHLCV data

        Returns:
            torch.Tensor: Preprocessed data ready for model input
        """
        # Create features
        data_with_features = create_features(data)

        # Drop rows with NaN values
        data_with_features = data_with_features.dropna()

        if len(data_with_features) < self.sequence_length:
            logger.warning(
                f"Not enough data points for {self.symbol} (need {self.sequence_length}, got {len(data_with_features)})")
            return None

        # Get feature columns
        if self.metadata and 'feature_columns' in self.metadata:
            feature_cols = self.metadata['feature_columns']
        else:
            # If metadata not available, use all columns except 'close'
            feature_cols = [col for col in data_with_features.columns if col != 'close']

        # Scale features
        if self.scaler:
            data_scaled = data_with_features.copy()
            data_scaled[feature_cols] = self.scaler.transform(data_with_features[feature_cols])
        else:
            # If scaler not available, use min-max scaling between -1 and 1
            data_scaled = data_with_features.copy()
            for col in feature_cols:
                data_scaled[col] = 2 * (data_with_features[col] - data_with_features[col].min()) / \
                                   (data_with_features[col].max() - data_with_features[col].min()) - 1

        # Get the most recent sequence
        sequence = data_scaled.iloc[-self.sequence_length:].values

        # Convert to tensor
        sequence_tensor = torch.tensor(sequence, dtype=torch.float32).unsqueeze(0)  # Add batch dimension

        return sequence_tensor

    def generate_signal(self, data):
        """
        Generate trading signal using the trained model.

        Args:
            data (pd.DataFrame): DataFrame with OHLCV data

        Returns:
            dict: Trading signal with action, confidence, etc.
        """
        if self.model is None:
            logger.error(f"Model not loaded for {self.symbol}")
            return {'action': 'HOLD', 'confidence': 0.0}

        try:
            # Preprocess data
            sequence_tensor = self.preprocess_data(data)

            if sequence_tensor is None:
                return {'action': 'HOLD', 'confidence': 0.0}

            # Make prediction
            with torch.no_grad():
                predicted_value = self.model(sequence_tensor).item()

            # Get the last close price
            last_close = data.iloc[-1]['close']

            # Calculate predicted percent change
            predicted_change = (predicted_value - last_close) / last_close

            # Define thresholds for buy/sell signals
            buy_threshold = 0.01  # 1% increase
            sell_threshold = -0.01  # 1% decrease

            # Generate signal
            if predicted_change > buy_threshold:
                action = 'BUY'
                confidence = min(predicted_change * 5, 0.95)  # Scale confidence
            elif predicted_change < sell_threshold:
                action = 'SELL'
                confidence = min(abs(predicted_change) * 5, 0.95)  # Scale confidence
            else:
                action = 'HOLD'
                confidence = 0.5

            # Create signal dictionary
            signal = {
                'symbol': self.symbol,
                'timestamp': datetime.now(),
                'action': action,
                'confidence': confidence,
                'predicted_price': predicted_value,
                'current_price': last_close,
                'predicted_change': predicted_change,
                'timeframe': self.timeframe
            }

            logger.info(f"Generated signal for {self.symbol}: {action} with {confidence:.2f} confidence")

            return signal

        except Exception as e:
            logger.error(f"Error generating signal for {self.symbol}: {e}")
            return {'action': 'HOLD', 'confidence': 0.0}