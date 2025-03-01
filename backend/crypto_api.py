import os
import json
import logging
from datetime import datetime, timedelta
import pandas as pd
from backend.alpaca_api import AlpacaAPI

# Configure logging
logger = logging.getLogger(__name__)


class CryptoAPI(AlpacaAPI):
    """Extension of AlpacaAPI to support cryptocurrency trading and data."""

    def get_crypto_data(self, symbols, timeframe='1Day', start=None, end=None, limit=1000):
        """Get historical data for cryptocurrencies.

        Args:
            symbols (list): List of cryptocurrency symbols (e.g., 'BTC/USD', 'SOL/USD').
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
            # Get bars data for cryptocurrencies
            result = {}

            # Alpaca requires crypto symbols in the format BTC/USD
            for symbol in symbols:
                try:
                    # Make sure symbols are properly formatted
                    if '/' not in symbol:
                        formatted_symbol = f"{symbol}/USD"
                    else:
                        formatted_symbol = symbol

                    logger.info(f"Fetching crypto data for {formatted_symbol}")

                    bars = self.api.get_crypto_bars(
                        formatted_symbol,
                        timeframe,
                        start=start,
                        end=end,
                        limit=limit
                    ).df

                    # If data is not empty
                    if not bars.empty:
                        # Reset index to get timestamp as column and remove the symbol level
                        if isinstance(bars.index, pd.MultiIndex):
                            # Get the level containing the timestamp
                            bars = bars.reset_index()
                            # If 'symbol' is a column, drop it
                            if 'symbol' in bars.columns:
                                bars = bars.drop('symbol', axis=1)
                            # Set timestamp back as index
                            if 'timestamp' in bars.columns:
                                bars = bars.set_index('timestamp')

                        # Store using the original symbol name
                        result[symbol] = bars
                    else:
                        logger.warning(f"No data returned for {formatted_symbol}")
                except Exception as e:
                    logger.warning(f"Error fetching crypto data for {symbol}: {e}")

            return result
        except Exception as e:
            logger.error(f"Error fetching historical crypto bars: {e}")
            raise

    def get_crypto_account(self):
        """Get crypto account information."""
        try:
            return self.api.get_account()
        except Exception as e:
            logger.error(f"Error getting crypto account info: {e}")
            raise

    def place_crypto_order(self, symbol, qty, side, order_type='market', time_in_force='gtc', limit_price=None):
        """Place a cryptocurrency order.

        Args:
            symbol (str): Crypto symbol (e.g., 'BTC/USD').
            qty (float): Order quantity.
            side (str): 'buy' or 'sell'.
            order_type (str, optional): 'market' or 'limit'.
            time_in_force (str, optional): 'gtc' (good till cancelled) or 'ioc' (immediate or cancel).
            limit_price (float, optional): Limit price for limit orders.

        Returns:
            Order object from Alpaca.
        """
        try:
            # Make sure symbol is properly formatted
            if '/' not in symbol:
                formatted_symbol = f"{symbol}/USD"
            else:
                formatted_symbol = symbol

            logger.info(f"Placing {side} {order_type} order for {qty} of {formatted_symbol}")

            return self.api.submit_order(
                symbol=formatted_symbol,
                qty=qty,
                side=side,
                type=order_type,
                time_in_force=time_in_force,
                limit_price=limit_price
            )
        except Exception as e:
            logger.error(f"Error placing crypto order: {e}")
            raise