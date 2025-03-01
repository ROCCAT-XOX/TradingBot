import os
import sys
import json
import logging
import asyncio
import signal
from datetime import datetime
import pandas as pd

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

            # Initialize Alpaca API
            self.alpaca_api = AlpacaAPI(config_path=temp_config_path)

            # Delete the temporary file
            os.unlink(temp_config_path)

            # Initialize database
            self.database = Database()

            # WebSocket listener will be initialized in start method
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
            return {
                'ALPACA_API_KEY': os.environ.get('ALPACA_API_KEY', ''),
                'ALPACA_API_SECRET': os.environ.get('ALPACA_API_SECRET', ''),
                'ALPACA_API_BASE_URL': os.environ.get('ALPACA_API_BASE_URL', 'https://paper-api.alpaca.markets'),
                'TRADING_SYMBOLS': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
            }

    async def process_trades(self):
        """Asynchronously process incoming trade data."""
        while self.running and self.ws_listener:
            try:
                # Check if there are trades in the queue
                while not self.ws_listener.trade_queue.empty():
                    trade = await self.ws_listener.trade_queue.get()

                    try:
                        # Log the trade
                        logger.debug(f"Processing trade: {trade['symbol']} @ ${trade['price']:.2f} x {trade['size']}")

                        # Save to database if it's a symbol we're tracking
                        if trade['symbol'] in self.symbols:
                            # Placeholder for trade processing logic
                            pass
                    except Exception as trade_error:
                        logger.error(f"Error processing trade {trade}: {trade_error}")

                # Prevent tight looping
                await asyncio.sleep(0.1)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in trade processing loop: {e}")
                await asyncio.sleep(1)

    async def process_quotes(self):
        """Asynchronously process incoming quote data."""
        while self.running and self.ws_listener:
            try:
                # Check if there are quotes in the queue
                while not self.ws_listener.quote_queue.empty():
                    quote = await self.ws_listener.quote_queue.get()

                    try:
                        # Log the quote
                        logger.debug(
                            f"Processing quote: {quote['symbol']} bid: ${quote['bid_price']:.2f}, ask: ${quote['ask_price']:.2f}")

                        # Placeholder for quote processing logic
                        if quote['symbol'] in self.symbols:
                            pass
                    except Exception as quote_error:
                        logger.error(f"Error processing quote {quote}: {quote_error}")

                # Prevent tight looping
                await asyncio.sleep(0.1)
            except asyncio.CancelledError:
                break