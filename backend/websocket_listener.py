import logging
import threading
import queue
import asyncio
import random
from datetime import datetime

logger = logging.getLogger(__name__)


class WebSocketListener:
    """Handles WebSocket connections to Alpaca for live market data."""

    def __init__(self, alpaca_api, symbols=None, max_retries=5, max_symbols=5):
        """Initialize the WebSocket listener."""
        self.alpaca_api = alpaca_api
        self.symbols = symbols[:max_symbols] if symbols else []
        self.max_retries = max_retries

        # Threading and async control
        self.running = False
        self.stop_event = threading.Event()

        # Data management
        self.trade_queue = queue.Queue()
        self.quote_queue = queue.Queue()
        self.bar_queue = queue.Queue()

        self.latest_trades = {}
        self.latest_quotes = {}
        self.latest_bars = {}

        # WebSocket stream and event loop
        self._stream = None
        self.event_loop = None

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
            self.trade_queue.put(trade_data)
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
            self.quote_queue.put(quote_data)
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
            self.bar_queue.put(bar_data)
        except Exception as e:
            logger.error(f"Error in bar callback: {e}")

    def _run_async_loop(self):
        """Run the async event loop in a separate thread."""
        try:
            # Create a new event loop and set it for this thread
            self.event_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.event_loop)
            try:
                self.event_loop.run_until_complete(self._websocket_connection())
            except Exception as e:
                logger.error(f"WebSocket connection error: {e}")
            finally:
                if not self.event_loop.is_closed():
                    self.event_loop.close()
        except Exception as e:
            logger.error(f"Async loop error: {e}")
        finally:
            self.running = False
            self.stop_event.set()

    async def _websocket_connection(self):
        """Manage WebSocket connection with retry mechanism."""
        while not self.stop_event.is_set():
            try:
                logger.info("Establishing WebSocket connection")
                connection_symbols = self.symbols[:3]
                logger.info(f"Creating WebSocket stream for {len(connection_symbols)} symbols")
                stream = self.alpaca_api.create_websocket_connection()
                self._stream = stream

                # Subscribe using async callbacks
                stream.subscribe_trades(self._trade_callback, *connection_symbols)
                stream.subscribe_quotes(self._quote_callback, *connection_symbols)
                stream.subscribe_bars(self._bar_callback, *connection_symbols)

                logger.info(f"Starting WebSocket stream for {len(connection_symbols)} symbols")
                try:
                    if asyncio.iscoroutinefunction(stream.run):
                        await stream.run()
                    else:
                        await asyncio.to_thread(stream.run)
                except Exception as stream_error:
                    error_msg = str(stream_error)
                    logger.warning(f"Stream connection error: {error_msg}")

                    if "connection limit exceeded" in error_msg:
                        # Längeres Backoff, wenn das Verbindungs-Limit erreicht ist
                        wait_time = 300 * (1 + random.random())
                        logger.info(f"Connection limit exceeded. Waiting {wait_time:.2f} seconds before retry...")
                    else:
                        wait_time = min(5 * (2 ** (self.max_retries // 2)), 300) * (1 + random.random())
                        logger.info(f"Waiting {wait_time:.2f} seconds before retry...")
                    await asyncio.sleep(wait_time)

            except Exception as e:
                error_msg = str(e)
                logger.warning(f"Connection attempt failed: {error_msg}")
                if self.stop_event.is_set():
                    break
                if "connection limit exceeded" in error_msg:
                    wait_time = 300 * (1 + random.random())
                    logger.info(f"Connection limit exceeded. Waiting {wait_time:.2f} seconds before retry...")
                else:
                    wait_time = min(5 * (2 ** (self.max_retries // 2)), 300) * (1 + random.random())
                    logger.info(f"Waiting {wait_time:.2f} seconds before retry...")
                await asyncio.sleep(wait_time)

    def start(self):
        """Start the WebSocket connection."""
        if self.running:
            logger.warning("WebSocket listener is already running")
            return

        self.stop_event.clear()
        self.running = True

        self.thread = threading.Thread(target=self._run_async_loop, daemon=True)
        self.thread.start()

        logger.info("WebSocket listener started")

    def stop(self):
        """Stop the WebSocket connection."""
        if not self.running:
            logger.warning("WebSocket listener is not running")
            return

        try:
            self.stop_event.set()
            stream = self._stream
            if stream is not None:
                try:
                    stream.stop()
                except Exception as stop_error:
                    logger.error(f"Error stopping stream: {stop_error}")
                finally:
                    self._stream = None

            if hasattr(self, 'thread') and self.thread.is_alive():
                self.thread.join(timeout=5)
        except Exception as e:
            logger.error(f"Error during WebSocket stop: {e}")
        finally:
            self.running = False
            logger.info("WebSocket listener stopped")

    def add_symbols(self, symbols):
        """Add symbols to track."""
        max_symbols = 5
        new_symbols = [s for s in symbols if s not in self.symbols]
        new_symbols = new_symbols[:max_symbols - len(self.symbols)]

        if not new_symbols:
            return

        self.symbols.extend(new_symbols)

        if self.running:
            self.stop()
            self.start()

        logger.info(f"Added symbols to WebSocket listener: {', '.join(new_symbols)}")

    def remove_symbols(self, symbols):
        """Remove symbols from tracking."""
        symbols_to_remove = [s for s in symbols if s in self.symbols]
        if not symbols_to_remove:
            return

        for s in symbols_to_remove:
            self.symbols.remove(s)
            self.latest_trades.pop(s, None)
            self.latest_quotes.pop(s, None)
            self.latest_bars.pop(s, None)

        if self.running:
            self.stop()
            if self.symbols:
                self.start()

        logger.info(f"Removed symbols from WebSocket listener: {', '.join(symbols_to_remove)}")

    def get_latest_trades(self):
        """Get the latest trades for all tracked symbols."""
        return self.latest_trades.copy()

    def get_latest_quotes(self):
        """Get the latest quotes for all tracked symbols."""
        return self.latest_quotes.copy()

    def get_latest_bars(self):
        """Get the latest bars for all tracked symbols."""
        return self.latest_bars.copy()
