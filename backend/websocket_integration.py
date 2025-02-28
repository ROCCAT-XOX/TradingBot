import os
import sys
import json
import logging
import time
from datetime import datetime
import pandas as pd
import signal

# Add the parent directory to the path so we can import our modules
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

# Import our modules
from backend.alpaca_api import AlpacaAPI
from backend.websocket_listener import WebSocketListener
from backend.database import Database

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(parent_dir, 'logs', 'websocket.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class WebSocketIntegration:
    """Main class to handle WebSocket integration with the trading system."""

    def __init__(self, config_path=None):
        """Initialize the WebSocket integration.

        Args:
            config_path (str, optional): Path to the configuration file.
        """
        self.running = False
        self.initialized = False
        self.config = self.load_config(config_path)
        self.symbols = self.config.get('TRADING_SYMBOLS', ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA'])

        # Set environment variables for Alpaca API
        os.environ['ALPACA_API_KEY'] = self.config.get('ALPACA_API_KEY', '')
        os.environ['ALPACA_API_SECRET'] = self.config.get('ALPACA_API_SECRET', '')
        os.environ['ALPACA_API_BASE_URL'] = self.config.get('ALPACA_API_BASE_URL', 'https://paper-api.alpaca.markets')

        # Initialize components
        try:
            # Create a temporary config file for AlpacaAPI
            import tempfile
            with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as temp:
                temp_config = {
                    'ALPACA_API_KEY': self.config.get('ALPACA_API_KEY', ''),
                    'ALPACA_API_SECRET': self.config.get('ALPACA_API_SECRET', ''),
                    'ALPACA_API_BASE_URL': self.config.get('ALPACA_API_BASE_URL', 'https://paper-api.alpaca.markets')
                }
                json.dump(temp_config, temp)
                temp_config_path = temp.name

            self.alpaca_api = AlpacaAPI(config_path=temp_config_path)

            # Delete the temporary file
            os.unlink(temp_config_path)

            self.database = Database()
            self.ws_listener = None
            self.initialized = True
            logger.info("WebSocket integration initialized successfully.")
        except Exception as e:
            logger.error(f"Error initializing WebSocket integration: {e}")
            raise

    def load_config(self, config_path=None):
        """Load configuration from file.

        Args:
            config_path (str, optional): Path to the configuration file.

        Returns:
            dict: Configuration dictionary.
        """
        if not config_path:
            config_path = os.path.join(parent_dir, 'config', 'settings.json')

        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
                logger.info(f"Loaded configuration from {config_path}")
                # Log the API key (first few characters for verification)
                api_key = config.get('ALPACA_API_KEY', '')
                if api_key:
                    logger.info(f"Found API key in config: {api_key[:4]}...{api_key[-4:]}")
                else:
                    logger.warning("No API key found in config file")
                return config
        else:
            # Use environment variables as fallback
            logger.warning(f"Configuration file not found: {config_path}. Using environment variables.")
            api_key = os.environ.get('ALPACA_API_KEY', '')
            api_secret = os.environ.get('ALPACA_API_SECRET', '')

            if api_key and api_secret:
                logger.info(f"Using API credentials from environment variables: {api_key[:4]}...{api_key[-4:]}")
            else:
                logger.error("No API credentials found in environment variables")

            return {
                'ALPACA_API_KEY': api_key,
                'ALPACA_API_SECRET': api_secret,
                'ALPACA_API_BASE_URL': os.environ.get('ALPACA_API_BASE_URL', 'https://paper-api.alpaca.markets'),
                'TRADING_SYMBOLS': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
            }

    def start(self):
        """Start the WebSocket listener and begin processing data."""
        if not self.initialized:
            logger.error("WebSocket integration not initialized. Cannot start.")
            return

        self.running = True

        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self.handle_shutdown)
        signal.signal(signal.SIGTERM, self.handle_shutdown)

        try:
            # Create and start WebSocket listener
            self.ws_listener = WebSocketListener(self.alpaca_api, self.symbols)
            self.ws_listener.start()
            logger.info(f"WebSocket listener started for symbols: {', '.join(self.symbols)}")

            # Main processing loop
            self.process_data_loop()
        except Exception as e:
            logger.error(f"Error in WebSocket integration: {e}")
            self.stop()

    def process_data_loop(self):
        """Main loop to process incoming WebSocket data."""
        logger.info("Starting data processing loop...")

        while self.running:
            try:
                # Process trades
                self.process_trades()

                # Process quotes
                self.process_quotes()

                # Process bars
                self.process_bars()

                # Sleep to avoid excessive CPU usage
                time.sleep(1)
            except Exception as e:
                logger.error(f"Error processing data: {e}")

    def process_trades(self):
        """Process incoming trade data."""
        if not self.ws_listener or self.ws_listener.trade_queue.empty():
            return

        # Process all available trades
        while not self.ws_listener.trade_queue.empty():
            trade = self.ws_listener.trade_queue.get()

            try:
                # Log the trade
                logger.debug(f"Processing trade: {trade['symbol']} @ ${trade['price']:.2f} x {trade['size']}")

                # Save to database if it's a symbol we're tracking
                # This is just a placeholder - implement your actual logic here
                if trade['symbol'] in self.symbols:
                    # You might want to save this to the database or process it further
                    # For example, you could update a price cache or trigger trading logic
                    pass
            except Exception as e:
                logger.error(f"Error processing trade {trade}: {e}")

    def process_quotes(self):
        """Process incoming quote data."""
        if not self.ws_listener or self.ws_listener.quote_queue.empty():
            return

        # Process all available quotes
        while not self.ws_listener.quote_queue.empty():
            quote = self.ws_listener.quote_queue.get()

            try:
                # Log the quote
                logger.debug(
                    f"Processing quote: {quote['symbol']} bid: ${quote['bid_price']:.2f}, ask: ${quote['ask_price']:.2f}")

                # Process the quote as needed
                # This is just a placeholder - implement your actual logic here
                if quote['symbol'] in self.symbols:
                    # You might want to save this to the database or process it further
                    pass
            except Exception as e:
                logger.error(f"Error processing quote {quote}: {e}")

    def process_bars(self):
        """Process incoming bar data."""
        if not self.ws_listener or self.ws_listener.bar_queue.empty():
            return

        # Process all available bars
        while not self.ws_listener.bar_queue.empty():
            bar = self.ws_listener.bar_queue.get()

            try:
                # Log the bar
                logger.debug(
                    f"Processing bar: {bar['symbol']} OHLC: ${bar['open']:.2f}/${bar['high']:.2f}/${bar['low']:.2f}/${bar['close']:.2f}")

                # Save to database
                # This is just a placeholder - implement your actual logic here
                if bar['symbol'] in self.symbols:
                    try:
                        # Convert the bar to a pandas DataFrame for storage
                        bar_df = pd.DataFrame([{
                            'open': bar['open'],
                            'high': bar['high'],
                            'low': bar['low'],
                            'close': bar['close'],
                            'volume': bar['volume']
                        }], index=[pd.to_datetime(bar['timestamp'])])

                        # Add to database
                        self.database.add_price_data(bar['symbol'], bar_df, timeframe='1Min')
                        logger.debug(f"Saved bar data for {bar['symbol']} to database")
                    except Exception as db_error:
                        logger.error(f"Error saving bar data to database: {db_error}")
            except Exception as e:
                logger.error(f"Error processing bar {bar}: {e}")

    def stop(self):
        """Stop the WebSocket listener and clean up."""
        self.running = False

        if self.ws_listener:
            logger.info("Stopping WebSocket listener...")
            self.ws_listener.stop()

        logger.info("WebSocket integration stopped.")

    def handle_shutdown(self, signum, frame):
        """Handle shutdown signals gracefully."""
        logger.info(f"Received signal {signum}. Shutting down...")
        self.stop()
        sys.exit(0)


if __name__ == "__main__":
    # Make sure logs directory exists
    logs_dir = os.path.join(parent_dir, 'logs')
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)

    logger.info("Starting WebSocket integration...")

    try:
        # Try to load from config file first
        config_path = os.path.join(parent_dir, 'config', 'settings.json')

        integration = WebSocketIntegration(config_path if os.path.exists(config_path) else None)
        integration.start()
    except KeyboardInterrupt:
        logger.info("Interrupted by user. Shutting down...")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Error running WebSocket integration: {e}")
        sys.exit(1)