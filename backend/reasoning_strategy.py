import os
import sys
import logging
import numpy as np
import pandas as pd
from datetime import datetime

# Configure logging
logger = logging.getLogger(__name__)

# Add parent directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from backend.trade_bot import TradingStrategy


class ReasoningTradingStrategy(TradingStrategy):
    """Advanced trading strategy with detailed reasoning capabilities."""

    def __init__(self, symbol, timeframe='1Day', model_path=None):
        """
        Initialize the reasoning trading strategy.

        Args:
            symbol (str): Stock or crypto symbol
            timeframe (str): Timeframe for data analysis
            model_path (str, optional): Path to saved model
        """
        super().__init__(symbol, timeframe)
        self.model_path = model_path
        self.reasoning_factors = [
            'price_trend',  # Short-term price trend
            'volume_analysis',  # Volume patterns
            'support_resistance',  # Support/resistance levels
            'market_regime',  # Bull/bear market conditions
            'volatility',  # Current volatility
            'macro_indicators',  # Overall market conditions
            'sentiment'  # Market sentiment indicators (if available)
        ]

    def analyze_price_trend(self, data):
        """Analyze recent price trends."""
        # Use last 20 bars for trend analysis
        recent_data = data.iloc[-20:]

        # Calculate various moving averages
        sma5 = recent_data['close'].rolling(window=5).mean()
        sma10 = recent_data['close'].rolling(window=10).mean()
        sma20 = recent_data['close'].rolling(window=20).mean()

        # Get latest values
        latest_close = recent_data['close'].iloc[-1]
        latest_sma5 = sma5.iloc[-1]
        latest_sma10 = sma10.iloc[-1]
        latest_sma20 = sma20.iloc[-1]

        # Determine current trend based on MA alignment
        # Strong uptrend: price > sma5 > sma10 > sma20
        # Strong downtrend: price < sma5 < sma10 < sma20

        if latest_close > latest_sma5 > latest_sma10 > latest_sma20:
            trend = 'strong_uptrend'
            trend_strength = 0.9
        elif latest_close > latest_sma5 > latest_sma10:
            trend = 'moderate_uptrend'
            trend_strength = 0.7
        elif latest_close > latest_sma5:
            trend = 'weak_uptrend'
            trend_strength = 0.6
        elif latest_close < latest_sma5 < latest_sma10 < latest_sma20:
            trend = 'strong_downtrend'
            trend_strength = 0.9
        elif latest_close < latest_sma5 < latest_sma10:
            trend = 'moderate_downtrend'
            trend_strength = 0.7
        elif latest_close < latest_sma5:
            trend = 'weak_downtrend'
            trend_strength = 0.6
        else:
            trend = 'neutral'
            trend_strength = 0.5

        # Calculate price momentum
        momentum = latest_close / recent_data['close'].iloc[0] - 1

        return {
            'trend': trend,
            'trend_strength': trend_strength,
            'momentum': momentum,
            'latest_close': latest_close,
            'sma5': latest_sma5,
            'sma10': latest_sma10,
            'sma20': latest_sma20
        }

    def analyze_volume(self, data):
        """Analyze volume patterns."""
        # Use last 20 bars for volume analysis
        recent_data = data.iloc[-20:]

        # Calculate average volume
        avg_volume = recent_data['volume'].mean()
        latest_volume = recent_data['volume'].iloc[-1]

        # Volume change vs average
        vol_change = latest_volume / avg_volume - 1

        # Volume trend (last 5 days)
        vol_trend = recent_data['volume'].iloc[-5:].mean() / recent_data['volume'].iloc[-10:-5].mean() - 1

        # Check for volume confirmation of price moves
        price_up = recent_data['close'].iloc[-1] > recent_data['close'].iloc[-2]
        volume_up = latest_volume > recent_data['volume'].iloc[-2]

        # Volume confirmation (price up + volume up or price down + volume down)
        if (price_up and volume_up) or (not price_up and not volume_up):
            vol_confirmation = True
        else:
            vol_confirmation = False

        return {
            'latest_volume': latest_volume,
            'avg_volume': avg_volume,
            'vol_change': vol_change,
            'vol_trend': vol_trend,
            'vol_confirmation': vol_confirmation
        }

    def analyze_support_resistance(self, data):
        """Identify key support and resistance levels."""
        # Use more data for support/resistance
        hist_data = data.iloc[-50:]

        # Simplified approach: find recent highs and lows
        highs = hist_data['high'].rolling(window=5, center=True).max()
        lows = hist_data['low'].rolling(window=5, center=True).min()

        # Current price
        latest_close = hist_data['close'].iloc[-1]

        # Find recent significant highs (potential resistance)
        resistance_levels = []
        for i in range(len(highs) - 10):
            if highs.iloc[i] == max(highs.iloc[max(0, i - 5):min(len(highs), i + 6)]):
                resistance_levels.append(highs.iloc[i])

        # Find recent significant lows (potential support)
        support_levels = []
        for i in range(len(lows) - 10):
            if lows.iloc[i] == min(lows.iloc[max(0, i - 5):min(len(lows), i + 6)]):
                support_levels.append(lows.iloc[i])

        # Keep only the most recent levels (last 3)
        resistance_levels = sorted([r for r in resistance_levels if r > latest_close])[-3:] if resistance_levels else []
        support_levels = sorted([s for s in support_levels if s < latest_close])[-3:] if support_levels else []

        # Calculate distance to nearest support/resistance
        nearest_resistance = min(resistance_levels) if resistance_levels else float('inf')
        nearest_support = max(support_levels) if support_levels else 0

        # Calculate relative distance in percentage
        dist_to_resistance = (nearest_resistance / latest_close - 1) * 100 if nearest_resistance < float(
            'inf') else None
        dist_to_support = (1 - nearest_support / latest_close) * 100 if nearest_support > 0 else None

        return {
            'resistance_levels': resistance_levels,
            'support_levels': support_levels,
            'nearest_resistance': nearest_resistance if nearest_resistance < float('inf') else None,
            'nearest_support': nearest_support if nearest_support > 0 else None,
            'dist_to_resistance': dist_to_resistance,
            'dist_to_support': dist_to_support
        }

    def analyze_market_regime(self, data):
        """Determine current market regime (bull/bear/sideways)."""
        # Use longer timeframe for market regime
        hist_data = data.iloc[-100:]

        # Calculate long-term moving averages
        sma50 = hist_data['close'].rolling(window=50).mean()
        sma100 = hist_data['close'].rolling(window=100).mean()

        # Latest values
        latest_close = hist_data['close'].iloc[-1]
        latest_sma50 = sma50.iloc[-1] if not pd.isna(sma50.iloc[-1]) else latest_close
        latest_sma100 = sma100.iloc[-1] if not pd.isna(sma100.iloc[-1]) else latest_close

        # Determine market regime
        if latest_close > latest_sma50 > latest_sma100:
            regime = 'bull_market'
            regime_strength = 0.8
        elif latest_close < latest_sma50 < latest_sma100:
            regime = 'bear_market'
            regime_strength = 0.8
        elif abs(latest_sma50 / latest_sma100 - 1) < 0.02:
            regime = 'sideways'
            regime_strength = 0.6
        else:
            regime = 'transitioning'
            regime_strength = 0.5

        return {
            'regime': regime,
            'regime_strength': regime_strength,
            'latest_close': latest_close,
            'sma50': latest_sma50,
            'sma100': latest_sma100
        }

    def analyze_volatility(self, data):
        """Analyze current market volatility."""
        # Calculate daily returns
        returns = data['close'].pct_change().dropna()

        # Recent volatility (20 days)
        recent_volatility = returns.iloc[-20:].std() * np.sqrt(252)  # Annualized

        # Historical volatility (100 days)
        hist_volatility = returns.iloc[-100:].std() * np.sqrt(252)  # Annualized

        # Relative volatility (recent vs historical)
        rel_volatility = recent_volatility / hist_volatility if hist_volatility > 0 else 1.0

        # Volatility regime
        if rel_volatility > 1.5:
            vol_regime = 'high_volatility'
        elif rel_volatility < 0.7:
            vol_regime = 'low_volatility'
        else:
            vol_regime = 'normal_volatility'

        return {
            'recent_volatility': recent_volatility,
            'hist_volatility': hist_volatility,
            'rel_volatility': rel_volatility,
            'vol_regime': vol_regime
        }

    def generate_reasoning(self, analysis_results):
        """Generate human-readable reasoning based on analysis."""
        reasoning = []

        # Price trend reasoning
        trend_info = analysis_results['price_trend']
        if 'uptrend' in trend_info['trend']:
            strength = trend_info['trend'].split('_')[0]
            reasoning.append(f"Price shows a {strength} uptrend (price: ${trend_info['latest_close']:.2f}, "
                             f"above SMA5: ${trend_info['sma5']:.2f}, SMA20: ${trend_info['sma20']:.2f}).")
            if trend_info['momentum'] > 0.05:
                reasoning.append(
                    f"Strong positive momentum of {trend_info['momentum'] * 100:.1f}% over the analyzed period.")
        elif 'downtrend' in trend_info['trend']:
            strength = trend_info['trend'].split('_')[0]
            reasoning.append(f"Price shows a {strength} downtrend (price: ${trend_info['latest_close']:.2f}, "
                             f"below SMA5: ${trend_info['sma5']:.2f}, SMA20: ${trend_info['sma20']:.2f}).")
            if trend_info['momentum'] < -0.05:
                reasoning.append(
                    f"Strong negative momentum of {trend_info['momentum'] * 100:.1f}% over the analyzed period.")
        else:
            reasoning.append(f"Price trend is neutral at ${trend_info['latest_close']:.2f}.")

        # Volume analysis reasoning
        vol_info = analysis_results['volume_analysis']
        if vol_info['vol_change'] > 0.2:
            reasoning.append(
                f"Volume is {vol_info['vol_change'] * 100:.1f}% higher than average, showing increased interest.")
            if vol_info['vol_confirmation']:
                reasoning.append("Volume confirms the price movement, adding strength to the signal.")
            else:
                reasoning.append("Volume doesn't confirm price direction, suggesting potential weakness.")
        elif vol_info['vol_change'] < -0.2:
            reasoning.append(
                f"Volume is {-vol_info['vol_change'] * 100:.1f}% lower than average, showing decreased interest.")

        # Support/resistance reasoning
        sr_info = analysis_results['support_resistance']
        if sr_info['dist_to_resistance'] is not None and sr_info['dist_to_resistance'] < 3.0:
            reasoning.append(f"Price is approaching resistance at ${sr_info['nearest_resistance']:.2f} "
                             f"({sr_info['dist_to_resistance']:.1f}% away).")
        if sr_info['dist_to_support'] is not None and sr_info['dist_to_support'] < 3.0:
            reasoning.append(f"Price is near support at ${sr_info['nearest_support']:.2f} "
                             f"({sr_info['dist_to_support']:.1f}% away).")

        # Market regime reasoning
        regime_info = analysis_results['market_regime']
        if regime_info['regime'] == 'bull_market':
            reasoning.append("Overall market conditions show a bullish trend.")
        elif regime_info['regime'] == 'bear_market':
            reasoning.append("Overall market conditions show a bearish trend.")
        else:
            reasoning.append("Market conditions appear to be sideways or transitioning.")

        # Volatility reasoning
        vol_info = analysis_results['volatility']
        if vol_info['vol_regime'] == 'high_volatility':
            reasoning.append(f"Market shows high volatility ({vol_info['recent_volatility'] * 100:.1f}% annualized), "
                             f"{vol_info['rel_volatility']:.1f}x the historical average.")
        elif vol_info['vol_regime'] == 'low_volatility':
            reasoning.append(f"Market shows low volatility ({vol_info['recent_volatility'] * 100:.1f}% annualized), "
                             f"only {vol_info['rel_volatility']:.1f}x the historical average.")

        # Final decision reasoning
        if analysis_results['final_action'] == 'BUY':
            reasoning.append(
                f"RECOMMENDATION: BUY with {analysis_results['confidence'] * 100:.1f}% confidence based on the above factors.")
        elif analysis_results['final_action'] == 'SELL':
            reasoning.append(
                f"RECOMMENDATION: SELL with {analysis_results['confidence'] * 100:.1f}% confidence based on the above factors.")
        else:
            reasoning.append(
                f"RECOMMENDATION: HOLD with {analysis_results['confidence'] * 100:.1f}% confidence based on the above factors.")

        return "\n".join(reasoning)

    def generate_signal(self, data):
        """
        Generate trading signal with detailed reasoning.

        Args:
            data (pd.DataFrame): DataFrame with price data

        Returns:
            dict: Trading signal with action, confidence, reasoning, etc.
        """
        try:
            # Perform multiple analyses
            analysis_results = {
                'price_trend': self.analyze_price_trend(data),
                'volume_analysis': self.analyze_volume(data),
                'support_resistance': self.analyze_support_resistance(data),
                'market_regime': self.analyze_market_regime(data),
                'volatility': self.analyze_volatility(data)
            }

            # Determine final action based on combined factors
            trend_score = 0
            if 'strong_uptrend' in analysis_results['price_trend']['trend']:
                trend_score = 1.0
            elif 'moderate_uptrend' in analysis_results['price_trend']['trend']:
                trend_score = 0.7
            elif 'weak_uptrend' in analysis_results['price_trend']['trend']:
                trend_score = 0.3
            elif 'strong_downtrend' in analysis_results['price_trend']['trend']:
                trend_score = -1.0
            elif 'moderate_downtrend' in analysis_results['price_trend']['trend']:
                trend_score = -0.7
            elif 'weak_downtrend' in analysis_results['price_trend']['trend']:
                trend_score = -0.3

            # Adjust based on volume confirmation
            if analysis_results['volume_analysis']['vol_confirmation']:
                trend_score *= 1.2
            else:
                trend_score *= 0.8

            # Adjust based on support/resistance proximity
            sr_info = analysis_results['support_resistance']
            if sr_info['dist_to_resistance'] is not None and sr_info['dist_to_resistance'] < 2.0:
                trend_score -= 0.3  # Approaching resistance is bearish
            if sr_info['dist_to_support'] is not None and sr_info['dist_to_support'] < 2.0:
                trend_score += 0.3  # Approaching support is bullish

            # Adjust based on market regime
            if analysis_results['market_regime']['regime'] == 'bull_market':
                trend_score *= 1.2
            elif analysis_results['market_regime']['regime'] == 'bear_market':
                trend_score *= 0.8

            # Adjust based on volatility
            vol_info = analysis_results['volatility']
            if vol_info['vol_regime'] == 'high_volatility':
                # Reduce confidence in high volatility
                confidence_modifier = 0.8
            elif vol_info['vol_regime'] == 'low_volatility':
                # Increase confidence in low volatility
                confidence_modifier = 1.2
            else:
                confidence_modifier = 1.0

            # Determine final action and confidence
            if trend_score > 0.5:
                final_action = 'BUY'
                confidence = min(0.5 + abs(trend_score) * 0.4, 0.95) * confidence_modifier
            elif trend_score < -0.5:
                final_action = 'SELL'
                confidence = min(0.5 + abs(trend_score) * 0.4, 0.95) * confidence_modifier
            else:
                final_action = 'HOLD'
                confidence = 0.5 + (0.5 - abs(trend_score)) * 0.5

            # Add final action to analysis results
            analysis_results['final_action'] = final_action
            analysis_results['confidence'] = confidence

            # Generate human-readable reasoning
            reasoning = self.generate_reasoning(analysis_results)

            # Create the signal dictionary
            signal = {
                'symbol': self.symbol,
                'timestamp': datetime.now(),
                'action': final_action,
                'confidence': confidence,
                'current_price': data.iloc[-1]['close'],
                'reasoning': reasoning,
                'analysis': analysis_results,
                'timeframe': self.timeframe
            }

            logger.info(
                f"Generated reasoning-based signal for {self.symbol}: {final_action} with {confidence:.2f} confidence")

            return signal

        except Exception as e:
            logger.error(f"Error generating reasoning-based signal for {self.symbol}: {e}")
            return {
                'action': 'HOLD',
                'confidence': 0.0,
                'reasoning': f"Error in signal generation: {str(e)}"
            }