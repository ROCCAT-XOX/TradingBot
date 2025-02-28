import logging
import threading
import queue
from datetime import datetime

logger = logging.getLogger(__name__)


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
        self.thread = None

        # Queues for different data types
        self.trade_queue = queue.Queue()
        self.quote_queue = queue.Queue()
        self.bar_queue = queue.Queue()

        # Latest data storage
        self.latest_trades = {}
        self.latest_quotes = {}
        self.latest_bars = {}

        logger.info(f"WebSocket listener initialized for symbols: {', '.join(self.symbols)}")

    def _trade_callback(self, trade):
        """Callback for trade updates."""
        trade_data = {
            'symbol': trade.symbol,
            'price': trade.price,
            'size': trade.size,
            'timestamp': trade.timestamp,
            'received_at': datetime.now().isoformat()
        }
        self.latest_trades[trade.symbol] = trade_data
        self.trade_queue.put(trade_data)
        logger.debug(f"Trade received: {trade.symbol} @ {trade.price}")

    def _quote_callback(self, quote):
        """Callback for quote updates."""
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
        self.quote_queue.put(quote_data)
        logger.debug(f"Quote received: {quote.symbol} bid: {quote.bid_price}, ask: {quote.ask_price}")

    def _bar_callback(self, bar):
        """Callback for bar updates."""
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
        self.bar_queue.put(bar_data)
        logger.debug(f"Bar received: {bar.symbol} OHLC: {bar.open}/{bar.high}/{bar.low}/{bar.close}")

    def start(self):
        """Start the WebSocket connection in a separate thread."""
        if self.running:
            logger.warning("WebSocket listener is already running")
            return

        if not self.symbols:
            logger.error("No symbols specified for WebSocket listener")
            return

        def run_ws():
            """Run the WebSocket connection."""
            try:
                self.stream = self.alpaca_api.create_websocket_connection()

                # Override the default callbacks
                self.stream.subscribe_trades(self._trade_callback, *self.symbols)
                self.stream.subscribe_quotes(self._quote_callback, *self.symbols)
                self.stream.subscribe_bars(self._bar_callback, *self.symbols)

                logger.info(f"Starting WebSocket connection for {len(self.symbols)} symbols")
                self.stream.run()
            except Exception as e:
                logger.error(f"WebSocket error: {e}")
                self.running = False

        self.running = True
        self.thread = threading.Thread(target=run_ws, daemon=True)
        self.thread.start()
        logger.info("WebSocket listener started")

    def stop(self):
        """Stop the WebSocket connection."""
        if not self.running:
            logger.warning("WebSocket listener is not running")
            return

        self.running = False
        if self.stream:
            self.stream.stop()

        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=5)

        logger.info("WebSocket listener stopped")

    def add_symbols(self, symbols):
        """Add symbols to track.

        Args:
            symbols (list): List of stock symbols to add.
        """
        new_symbols = [s for s in symbols if s not in self.symbols]
        if not new_symbols:
            return

        self.symbols.extend(new_symbols)

        # If already running, update the stream subscriptions
        if self.running and self.stream:
            self.stream.subscribe_trades(self._trade_callback, *new_symbols)
            self.stream.subscribe_quotes(self._quote_callback, *new_symbols)
            self.stream.subscribe_bars(self._bar_callback, *new_symbols)

        logger.info(f"Added symbols to WebSocket listener: {', '.join(new_symbols)}")

    def remove_symbols(self, symbols):
        """Remove symbols from tracking.

        Args:
            symbols (list): List of stock symbols to remove.
        """
        # Currently, Alpaca's API doesn't support unsubscribing from individual symbols
        # We'd need to stop and restart the connection with the updated symbol list
        symbols_to_remove = [s for s in symbols if s in self.symbols]
        if not symbols_to_remove:
            return

        for s in symbols_to_remove:
            self.symbols.remove(s)

            # Clean up stored data
            if s in self.latest_trades:
                del self.latest_trades[s]
            if s in self.latest_quotes:
                del self.latest_quotes[s]
            if s in self.latest_bars:
                del self.latest_bars[s]

        # Restart the connection if running
        if self.running:
            self.stop()
            if self.symbols:  # Only restart if we still have symbols to track
                self.start()

        logger.info(f"Removed symbols from WebSocket listener: {', '.join(symbols_to_remove)}")

    def get_latest_trades(self):
        """Get the latest trades for all tracked symbols.

        Returns:
            dict: Dictionary of the latest trade data for each symbol.
        """
        return self.latest_trades.copy()

    def get_latest_quotes(self):
        """Get the latest quotes for all tracked symbols.

        Returns:
            dict: Dictionary of the latest quote data for each symbol.
        """
        return self.latest_quotes.copy()

    def get_latest_bars(self):
        """Get the latest bars for all tracked symbols.

        Returns:
            dict: Dictionary of the latest bar data for each symbol.
        """
        return self.latest_bars.copy()


if __name__ == "__main__":
    # Simple test of the WebSocket listener
    import os
    import sys
    import time
    from alpaca_api import AlpacaAPI

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

        # Test WebSocket listener with some common symbols
        symbols = ['AAPL', 'MSFT', 'GOOGL']
        listener = WebSocketListener(alpaca, symbols)

        # Start the listener
        listener.start()

        # Run for 60 seconds and print updates
        print(f"Listening for updates on {', '.join(symbols)} for 60 seconds...")
        start_time = time.time()

        try:
            while time.time() - start_time < 60:
                # Check if we have any trade data
                if not listener.trade_queue.empty():
                    trade = listener.trade_queue.get()
                    print(f"Trade: {trade['symbol']} @ ${trade['price']:.2f} x {trade['size']}")

                # Check if we have any quote data
                if not listener.quote_queue.empty():
                    quote = listener.quote_queue.get()
                    print(
                        f"Quote: {quote['symbol']} bid: ${quote['bid_price']:.2f} x {quote['bid_size']}, ask: ${quote['ask_price']:.2f} x {quote['ask_size']}")

                # Check if we have any bar data
                if not listener.bar_queue.empty():
                    bar = listener.bar_queue.get()
                    print(
                        f"Bar: {bar['symbol']} OHLC: ${bar['open']:.2f}/${bar['high']:.2f}/${bar['low']:.2f}/${bar['close']:.2f} Volume: {bar['volume']}")

                # Sleep briefly to avoid hammering the CPU
                time.sleep(0.1)

        except KeyboardInterrupt:
            print("Interrupted by user")

        # Stop the listener when done
        listener.stop()
        print("WebSocket test completed")

    except Exception as e:
        print(f"Error testing WebSocket listener: {e}")
        sys.exit(1)