import os
import json
import logging
from datetime import datetime, timedelta
import pandas as pd
import alpaca_trade_api as tradeapi
from alpaca_trade_api.stream import Stream

# Configure logging
logger = logging.getLogger(__name__)


class AlpacaAPI:
    """Interface to the Alpaca Markets API for trading and market data."""

    def __init__(self, config_path=None):
        """Initialize the Alpaca API client.

        Args:
            config_path (str, optional): Path to the settings.json file.
                If None, will use environment variables.
        """
        self.api_key = None
        self.api_secret = None
        self.base_url = None
        self.data_url = None

        if config_path:
            logger.info(f"Loading Alpaca API configuration from: {config_path}")
            with open(config_path, 'r') as f:
                config = json.load(f)
                self.api_key = config.get('ALPACA_API_KEY')
                self.api_secret = config.get('ALPACA_API_SECRET')
                self.base_url = config.get('ALPACA_API_BASE_URL', 'https://paper-api.alpaca.markets')
                self.data_url = config.get('ALPACA_DATA_URL', 'https://data.alpaca.markets')

                # Log partial API key for verification
                if self.api_key:
                    logger.info(f"API key from config: {self.api_key[:4]}...{self.api_key[-4:]}")
                else:
                    logger.warning("No API key found in config file")
        else:
            # Use environment variables if no config file provided
            logger.info("Using environment variables for Alpaca API credentials")
            self.api_key = os.environ.get('ALPACA_API_KEY')
            self.api_secret = os.environ.get('ALPACA_API_SECRET')
            self.base_url = os.environ.get('ALPACA_API_BASE_URL', 'https://paper-api.alpaca.markets')
            self.data_url = os.environ.get('ALPACA_DATA_URL', 'https://data.alpaca.markets')

            # Log partial API key for verification
            if self.api_key:
                logger.info(f"API key from env vars: {self.api_key[:4]}...{self.api_key[-4:]}")
            else:
                logger.warning("No API key found in environment variables")

        if not self.api_key or not self.api_secret:
            error_msg = "Alpaca API credentials not found. Please provide them in config file or as environment variables."
            logger.error(error_msg)
            raise ValueError(error_msg)

        # Initialize REST API client
        try:
            self.api = tradeapi.REST(
                self.api_key,
                self.api_secret,
                self.base_url,
                api_version='v2'
            )

            # Test connection with a simple call
            self.api.get_account()
            logger.info("Alpaca API client initialized and verified successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Alpaca API client: {e}")
            raise

        # Store stream configuration for later use
        self.stream_config = {
            'key_id': self.api_key,
            'secret_key': self.api_secret,
            'base_url': self.base_url,
            'data_feed': 'iex'  # Using 'iex' for free tier, 'sip' requires paid subscription
        }

    def get_account_info(self):
        """Get account information."""
        try:
            return self.api.get_account()
        except Exception as e:
            logger.error(f"Error getting account info: {e}")
            raise

    def get_historical_bars(self, symbols, timeframe='1Day', start=None, end=None, limit=1000):
        """Get historical bar data for specified symbols.

        Args:
            symbols (list): List of stock symbols.
            timeframe (str): Bar timeframe (e.g., '1Min', '1Hour', '1Day').
            start (datetime, optional): Start date for data.
            end (datetime, optional): End date for data.
            limit (int, optional): Maximum number of bars to return.

        Returns:
            dict: Dictionary of pandas DataFrames with historical data for each symbol.
        """
        if not start:
            start = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
        if not end:
            end = datetime.now().strftime('%Y-%m-%d')

        try:
            # Get bars data - specify 'adjustment' as 'raw' for the free tier
            result = {}

            # For free tier, we'll fetch each symbol separately
            for symbol in symbols:
                try:
                    bars = self.api.get_bars(
                        symbol,
                        timeframe,
                        start=start,
                        end=end,
                        limit=limit,
                        adjustment='raw',  # Use 'raw' for free tier
                        feed='iex'  # Use 'iex' for free tier
                    ).df

                    # If data is not empty
                    if not bars.empty:
                        # Reset index to get timestamp as column and remove the symbol level
                        if isinstance(bars.index, pd.MultiIndex):
                            # Get the level containing the timestamp
                            bars = bars.reset_index()
                            # If 'symbol' is a column, we don't need it since we're organizing by symbol
                            if 'symbol' in bars.columns:
                                bars = bars.drop('symbol', axis=1)
                            # Set timestamp back as index
                            if 'timestamp' in bars.columns:
                                bars = bars.set_index('timestamp')

                        result[symbol] = bars
                    else:
                        logger.warning(f"No data returned for {symbol}")
                except Exception as e:
                    logger.warning(f"Error fetching data for {symbol}: {e}")

            return result
        except Exception as e:
            logger.error(f"Error fetching historical bars: {e}")
            raise

    def create_websocket_connection(self, symbols=None):
        """Create a websocket connection for streaming data.

        Args:
            symbols (list, optional): List of stock symbols to subscribe to.

        Returns:
            Stream: Alpaca stream object ready to be configured with callbacks.
        """
        stream = Stream(**self.stream_config)

        if symbols:
            # Set up listeners for trades and quotes
            stream.subscribe_trades(self._trade_callback, *symbols)
            stream.subscribe_quotes(self._quote_callback, *symbols)
            stream.subscribe_bars(self._bar_callback, *symbols)

        return stream

    def _trade_callback(self, trade):
        """Default callback for trade updates."""
        logger.info(f"Trade: {trade.symbol} at {trade.price}")
        # This will be overridden by the actual application

    def _quote_callback(self, quote):
        """Default callback for quote updates."""
        logger.info(f"Quote: {quote.symbol} bid: {quote.bid_price}, ask: {quote.ask_price}")
        # This will be overridden by the actual application

    def _bar_callback(self, bar):
        """Default callback for bar updates."""
        logger.info(f"Bar: {bar.symbol} open: {bar.open}, close: {bar.close}")
        # This will be overridden by the actual application

    def place_order(self, symbol, qty, side, order_type='market', time_in_force='gtc', limit_price=None,
                    stop_price=None):
        """Place an order.

        Args:
            symbol (str): Stock symbol.
            qty (int): Order quantity.
            side (str): 'buy' or 'sell'.
            order_type (str, optional): 'market', 'limit', 'stop', 'stop_limit'.
            time_in_force (str, optional): 'day', 'gtc' (good till cancelled), 'opg' (market on open), 'cls' (market on close).
            limit_price (float, optional): Limit price for 'limit' or 'stop_limit' orders.
            stop_price (float, optional): Stop price for 'stop' or 'stop_limit' orders.

        Returns:
            Order object from Alpaca.
        """
        try:
            return self.api.submit_order(
                symbol=symbol,
                qty=qty,
                side=side,
                type=order_type,
                time_in_force=time_in_force,
                limit_price=limit_price,
                stop_price=stop_price
            )
        except Exception as e:
            logger.error(f"Error placing order: {e}")
            raise

    def get_positions(self):
        """Get current positions."""
        try:
            return self.api.list_positions()
        except Exception as e:
            logger.error(f"Error getting positions: {e}")
            raise

    def get_orders(self, status='open'):
        """Get orders with the specified status."""
        try:
            return self.api.list_orders(status=status)
        except Exception as e:
            logger.error(f"Error getting orders: {e}")
            raise


if __name__ == "__main__":
    # Simple test to make sure the API connection works
    import os
    import sys

    # Add parent directory to path to import config
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    # Set up logging for the test
    logging.basicConfig(level=logging.INFO)

    try:
        # Try to load from config file first
        config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'config',
                                   'settings.json')
        if os.path.exists(config_path):
            alpaca = AlpacaAPI(config_path=config_path)
        else:
            # Fall back to environment variables
            alpaca = AlpacaAPI()

        # Test account info
        account = alpaca.get_account_info()
        print(f"Account: {account.id}")
        print(f"Cash: ${account.cash}")
        print(f"Equity: ${account.equity}")

        # Test getting historical data
        symbols = ['AAPL', 'MSFT', 'GOOGL']
        bars = alpaca.get_historical_bars(symbols, timeframe='1Day', limit=5)

        for symbol, data in bars.items():
            print(f"\n{symbol} - Last 5 days:")
            print(data.tail())

        print("\nAPI connection test successful!")

    except Exception as e:
        print(f"Error testing Alpaca API: {e}")
        sys.exit(1)