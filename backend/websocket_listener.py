import os
import sys
import asyncio
import logging
import threading
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add parent directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from alpaca_trade_api.stream import Stream


class WebSocketListener:
    """Handles WebSocket connections to Alpaca for live market data."""

    def __init__(self, alpaca_api, symbols=None):
        """Initialize the WebSocket listener.

        Args:
            alpaca_api (AlpacaAPI): Initialized Alpaca API object.
            symbols (list, optional): List of stock symbols to track.
        """
        self.alpaca_api = alpaca_api
        self.symbols = symbols or []
        self.stream = None
        self.running = False
        self._stream_thread = None

        # Queues for different data types
        self.trade_queue = asyncio.Queue()
        self.quote_queue = asyncio.Queue()
        self.bar_queue = asyncio.Queue()

        # Latest data storage
        self.latest_trades = {}
        self.latest_quotes = {}
        self.latest_bars = {}

        logger.info(f"WebSocket listener initialized for symbols: {', '.join(self.symbols)}")

    async def _trade_callback(self, trade):
        """Async callback for trade updates."""
        try:
            trade_data = {
                'symbol': trade.symbol,
                'price': trade.price,
                'size': trade.size,
                'timestamp': trade.timestamp,
                'received_at': datetime.now().isoformat()
            }
            self.latest_trades[trade.symbol] = trade_data
            await self.trade_queue.put(trade_data)
            logger.debug(f"Trade received: {trade.symbol} @ {trade.price}")
        except Exception as e:
            logger.error(f"Error in trade callback: {e}")

    async def _quote_callback(self, quote):
        """Async callback for quote updates."""
        try:
            quote_data = {
                'symbol': quote.symbol,
                'bid_price': quote.bid_price,
                'bid_size': quote.bid_size,
                'ask_price': quote.ask_price,
                'ask_size': quote.ask_size,
                'timestamp': quote.timestamp,
                'received_at': datetime.now().isoformat()
            }
            self.latest_quotes[quote.symbol] = quote_data
            await self.quote_queue.put(quote_data)
            logger.debug(f"Quote received: {quote.symbol} bid: {quote.bid_price}, ask: {quote.ask_price}")
        except Exception as e:
            logger.error(f"Error in quote callback: {e}")

    async def _bar_callback(self, bar):
        """Async callback for bar updates."""
        try:
            bar_data = {
                'symbol': bar.symbol,
                'open': bar.open,
                'high': bar.high,
                'low': bar.low,
                'close': bar.close,
                'volume': bar.volume,
                'timestamp': bar.timestamp,
                'received_at': datetime.now().isoformat()
            }
            self.latest_bars[bar.symbol] = bar_data
            await self.bar_queue.put(bar_data)
            logger.debug(f"Bar received: {bar.symbol} OHLC: {bar.open}/{bar.high}/{bar.low}/{bar.close}")
        except Exception as e:
            logger.error(f"Error in bar callback: {e}")

    def _stream_runner(self):
        """Run the stream in a separate thread."""
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            # Create the stream
            self.stream = Stream(
                key_id=self.alpaca_api.api_key,
                secret_key=self.alpaca_api.api_secret,
                base_url=self.alpaca_api.base_url,
                data_feed='iex'
            )

            # Subscribe to trades, quotes, and bars for all symbols
            if self.symbols:
                logger.info(f"Subscribing to {len(self.symbols)} symbols")
                for symbol in self.symbols:
                    self.stream.subscribe_trades(self._trade_callback, symbol)
                    self.stream.subscribe_quotes(self._quote_callback, symbol)
                    self.stream.subscribe_bars(self._bar_callback, symbol)

            # Set running flag
            self.running = True

            # Run the stream
            loop.run_until_complete(self.stream.run())
        except Exception as e:
            logger.error(f"WebSocket stream error: {e}")
            self.running = False
        finally:
            # Ensure the stream is stopped if an error occurs
            if self.stream:
                try:
                    loop.run_until_complete(self.stream.stop())
                except Exception:
                    pass

    def start(self):
        """Start the WebSocket listener in a separate thread."""
        # Ensure we're not already running
        if self.running:
            logger.warning("WebSocket stream is already running")
            return

        # Create and start the stream thread
        self._stream_thread = threading.Thread(target=self._stream_runner, daemon=True)
        self._stream_thread.start()

    def stop(self):
        """Stop the WebSocket listener."""
        try:
            # Stop the stream if it exists
            if self.stream and self.running:
                # Use a new thread to run the async stop method
                def stop_stream():
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    try:
                        loop.run_until_complete(self.stream.stop())
                    except Exception as e:
                        logger.error(f"Error stopping stream: {e}")
                    finally:
                        loop.close()

                stop_thread = threading.Thread(target=stop_stream)
                stop_thread.start()
                stop_thread.join(timeout=5)  # Wait up to 5 seconds

            # Reset flags
            self.running = False
            self.stream = None

            # Join the stream thread if it exists
            if self._stream_thread and self._stream_thread.is_alive():
                self._stream_thread.join(timeout=5)

            logger.info("WebSocket stream stopped")
        except Exception as e:
            logger.error(f"Error stopping WebSocket stream: {e}")

    def get_latest_trades(self):
        """Get the latest trades for all tracked symbols."""
        return self.latest_trades.copy()

    def get_latest_quotes(self):
        """Get the latest quotes for all tracked symbols."""
        return self.latest_quotes.copy()

    def get_latest_bars(self):
        """Get the latest bars for all tracked symbols."""
        return self.latest_bars.copy()


if __name__ == "__main__":
    # Simple test of the WebSocket listener
    import sys
    from backend.alpaca_api import AlpacaAPI

    # Configure more verbose logging for testing
    logging.getLogger().setLevel(logging.DEBUG)

    try:
        # Try to load from config file first
        config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'config',
                                   'settings.json')

        # Initialize Alpaca API
        if os.path.exists(config_path):
            alpaca = AlpacaAPI(config_path=config_path)
        else:
            alpaca = AlpacaAPI()

        # Test WebSocket listener with some common symbols
        symbols = ['AAPL', 'MSFT', 'GOOGL']
        listener = WebSocketListener(alpaca, symbols)

        print(f"Listening for updates on {', '.join(symbols)} for 60 seconds...")

        # Start the listener
        listener.start()

        # Keep the main thread running
        import time

        time.sleep(60)

    except Exception as e:
        print(f"Error testing WebSocket listener: {e}")
        sys.exit(1)