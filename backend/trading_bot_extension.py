import logging
import pandas as pd
from datetime import datetime
from backend.trade_bot import TradingBot

# Configure logging
logger = logging.getLogger(__name__)


class ExtendedTradingBot(TradingBot):
    """Extended Trading Bot with prediction capabilities."""

    def get_prediction_signals(self, symbols=None):
        """Generate trading predictions for specified symbols without executing trades.

        Args:
            symbols (list, optional): List of symbols to generate predictions for.
                If None, uses all configured symbols.

        Returns:
            dict: Dictionary of prediction signals for each symbol.
        """
        if symbols is None:
            symbols = list(self.strategies.keys())

        predictions = {}

        for symbol in symbols:
            try:
                if symbol not in self.strategies:
                    logger.warning(f"No strategy configured for {symbol}, skipping")
                    continue

                # Fetch latest data
                data = self.fetch_latest_data(
                    symbol,
                    timeframe=self.strategies[symbol].timeframe,
                    bars=100  # Fetch enough bars for feature calculation
                )

                if data.empty:
                    logger.warning(f"No data available for {symbol}, skipping prediction")
                    continue

                # Generate signal
                signal = self.strategies[symbol].generate_signal(data)

                if signal:
                    # Add symbol and timestamp if not present
                    if 'symbol' not in signal:
                        signal['symbol'] = symbol
                    if 'timestamp' not in signal:
                        signal['timestamp'] = datetime.now()

                    # Calculate additional metrics for visualization
                    signal['current_price'] = data.iloc[-1]['close']

                    # For ML strategies, add prediction details
                    if hasattr(self.strategies[symbol], 'model') and 'predicted_price' in signal:
                        price_diff = signal['predicted_price'] - signal['current_price']
                        signal['predicted_change_pct'] = (price_diff / signal['current_price']) * 100

                    # For reasoning strategies, extract the reasoning
                    if 'reasoning' in signal:
                        signal['prediction_details'] = signal['reasoning']

                    predictions[symbol] = signal

                    # Log the prediction
                    logger.info(
                        f"Generated prediction for {symbol}: {signal['action']} with {signal['confidence']:.2f} confidence")
            except Exception as e:
                logger.error(f"Error generating prediction for {symbol}: {e}")

        return predictions

    def get_market_summary(self):
        """Generate a summary of the current market conditions for all tracked symbols.

        Returns:
            dict: Dictionary with market summary information
        """
        summary = {
            'timestamp': datetime.now(),
            'assets': {},
            'market_regime': 'mixed',  # Default value
        }

        # Count different signals
        signals_count = {'BUY': 0, 'SELL': 0, 'HOLD': 0}
        total_assets = 0

        # Get predictions for all symbols
        predictions = self.get_prediction_signals()

        for symbol, signal in predictions.items():
            summary['assets'][symbol] = {
                'action': signal.get('action', 'HOLD'),
                'confidence': signal.get('confidence', 0.0),
                'current_price': signal.get('current_price', 0.0)
            }

            # Count signal types
            signals_count[signal.get('action', 'HOLD')] += 1
            total_assets += 1

        # Determine overall market regime
        if total_assets > 0:
            buy_ratio = signals_count['BUY'] / total_assets
            sell_ratio = signals_count['SELL'] / total_assets

            if buy_ratio > 0.6:
                summary['market_regime'] = 'bullish'
            elif sell_ratio > 0.6:
                summary['market_regime'] = 'bearish'
            elif buy_ratio > 0.4 and sell_ratio < 0.3:
                summary['market_regime'] = 'slightly_bullish'
            elif sell_ratio > 0.4 and buy_ratio < 0.3:
                summary['market_regime'] = 'slightly_bearish'
            else:
                summary['market_regime'] = 'mixed'

                # Add signal counts to summary
            summary['signal_counts'] = signals_count

            return summary

        def get_asset_recommendation(self, top_n=5):
            """Get top asset recommendations based on signal confidence.

            Args:
                top_n (int): Number of top recommendations to return

            Returns:
                list: List of top asset recommendations
            """
            # Get predictions for all symbols
            predictions = self.get_prediction_signals()

            # Filter for buy signals only
            buy_signals = {symbol: pred for symbol, pred in predictions.items()
                           if pred.get('action', '') == 'BUY'}

            # Sort by confidence
            sorted_signals = sorted(buy_signals.items(),
                                    key=lambda x: x[1].get('confidence', 0),
                                    reverse=True)

            # Format recommendations
            recommendations = []
            for symbol, signal in sorted_signals[:top_n]:
                recommendations.append({
                    'symbol': symbol,
                    'confidence': signal.get('confidence', 0),
                    'current_price': signal.get('current_price', 0),
                    'predicted_price': signal.get('predicted_price', None),
                    'reasoning': signal.get('reasoning', 'No detailed reasoning available')
                })

            return recommendations