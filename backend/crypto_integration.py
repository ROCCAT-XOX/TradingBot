import os
import sys
import json
import logging
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from backend.crypto_api import CryptoAPI
from backend.train import create_features, train_trading_model
from backend.trade_bot import MLTradingStrategy

logger = logging.getLogger(__name__)


class CryptoIntegration:
    """Enhanced integration for cryptocurrency trading and training."""

    def __init__(self, config_path=None):
        """Initialize crypto integration with configuration."""
        # Initialize crypto API
        self.crypto_api = CryptoAPI(config_path=config_path)

        # Default crypto symbols
        self.default_symbols = ['BTC/USD', 'ETH/USD', 'SOL/USD', 'AVAX/USD', 'DOGE/USD', 'XRP/USD']

        # Load configuration
        if config_path:
            with open(config_path, 'r') as f:
                self.config = json.load(f)
                self.symbols = self.config.get('CRYPTO_SYMBOLS', self.default_symbols)
        else:
            self.config = {}
            self.symbols = self.default_symbols

    def get_available_symbols(self):
        """Get list of available crypto symbols for trading."""
        return self.symbols

    def fetch_historical_data(self, symbols=None, timeframe='1Day', days=30):
        """
        Fetch historical data for cryptocurrency symbols.

        Args:
            symbols (list): List of crypto symbols (e.g., 'BTC/USD')
            timeframe (str): Timeframe (e.g., '1Day', '1Hour')
            days (int): Number of days of data to fetch

        Returns:
            dict: Dictionary with historical data for each symbol
        """
        if symbols is None:
            symbols = self.symbols

        try:
            return self.crypto_api.get_crypto_data(
                symbols=symbols,
                timeframe=timeframe,
                start=(datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d'),
                end=datetime.now().strftime('%Y-%m-%d')
            )
        except Exception as e:
            logger.error(f"Error fetching crypto data: {e}")
            return {}

    def train_crypto_model(self, symbol, epochs=5000, learning_rate=0.001, timeframe='1Day', days=365,
                           batch_size=64, device=None, early_stopping=True, patience=20):
        """
        Train ML model for cryptocurrency prediction.

        Args:
            symbol (str): Crypto symbol (e.g., 'BTC/USD')
            epochs (int): Training epochs
            learning_rate (float): Learning rate
            timeframe (str): Timeframe (e.g., '1Day', '1Hour')
            days (int): Days of historical data to use
            batch_size (int): Batch size for training
            device (str): Computing device ('cpu', 'cuda', 'mps')
            early_stopping (bool): Use early stopping
            patience (int): Early stopping patience

        Returns:
            dict: Training results
        """
        try:
            # Prepare training configuration
            training_config = {
                'TRADING_SYMBOLS': [symbol],
                'DEFAULT_TIMEFRAME': timeframe,
                'TRAINING_PERIOD_DAYS': days,
                'AI_MODEL_SETTINGS': {
                    'num_epochs': epochs,
                    'learning_rate': learning_rate,
                    'batch_size': batch_size,
                    'early_stopping': early_stopping,
                    'patience': patience
                }
            }

            # Add device configuration if specified
            if device:
                training_config['DEVICE'] = device

            # Train the model
            results = train_trading_model(config=training_config)
            return results

        except Exception as e:
            logger.error(f"Error training crypto model for {symbol}: {e}")
            return {'error': str(e), 'symbol': symbol}

    def place_crypto_order(self, symbol, quantity, side='buy', order_type='market', limit_price=None):
        """
        Place a cryptocurrency trade order.

        Args:
            symbol (str): Crypto symbol (e.g., 'BTC/USD')
            quantity (float): Order quantity
            side (str): 'buy' or 'sell'
            order_type (str): 'market' or 'limit'
            limit_price (float): Limit price (for limit orders)

        Returns:
            dict: Order information
        """
        try:
            order = self.crypto_api.place_crypto_order(
                symbol=symbol,
                qty=quantity,
                side=side,
                order_type=order_type,
                limit_price=limit_price
            )

            return order
        except Exception as e:
            logger.error(f"Error placing crypto order for {symbol}: {e}")
            return None

    def get_crypto_account(self):
        """Get crypto account information."""
        try:
            return self.crypto_api.get_crypto_account()
        except Exception as e:
            logger.error(f"Error getting crypto account: {e}")
            return None


# Example usage
if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO)

    # Initialize crypto integration
    config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'config', 'settings.json')
    crypto = CryptoIntegration(config_path=config_path)

    # Fetch historical data
    symbols = ['BTC/USD', 'ETH/USD']
    data = crypto.fetch_historical_data(symbols=symbols, days=30)

    # Print sample data
    for symbol, df in data.items():
        print(f"\n{symbol} - Sample data:")
        print(df.head())

    print("\nCrypto integration test successful!")