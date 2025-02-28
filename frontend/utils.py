import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from datetime import datetime, timedelta


def format_currency(value):
    """Format a value as currency."""
    return f"${value:.2f}"


def format_percentage(value):
    """Format a value as percentage."""
    return f"{value:.2f}%"


def create_price_chart(data, symbol, timeframe='1Day', height=500):
    """Create a basic price chart for a stock.

    Args:
        data (pd.DataFrame): DataFrame with OHLCV data.
        symbol (str): Stock symbol.
        timeframe (str): Chart timeframe.
        height (int): Chart height in pixels.

    Returns:
        plotly.graph_objects.Figure: Chart figure.
    """
    fig = go.Figure(data=[go.Candlestick(
        x=data.index,
        open=data['open'],
        high=data['high'],
        low=data['low'],
        close=data['close'],
        name=symbol
    )])

    # Update layout
    timeframe_name = {
        '1Min': '1 Minute',
        '5Min': '5 Minutes',
        '15Min': '15 Minutes',
        '1Hour': '1 Hour',
        '1Day': 'Daily'
    }.get(timeframe, timeframe)

    fig.update_layout(
        title=f"{symbol} - {timeframe_name} Chart",
        xaxis_title="Date",
        yaxis_title="Price ($)",
        height=height,
        xaxis_rangeslider_visible=False
    )

    return fig


def calculate_sma(data, window):
    """Calculate Simple Moving Average."""
    return data.rolling(window=window).mean()


def calculate_ema(data, window):
    """Calculate Exponential Moving Average."""
    return data.ewm(span=window, adjust=False).mean()


def calculate_rsi(data, window=14):
    """Calculate Relative Strength Index."""
    delta = data.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    return rsi


def calculate_macd(data, fast=12, slow=26, signal=9):
    """Calculate MACD (Moving Average Convergence Divergence)."""
    ema_fast = calculate_ema(data, fast)
    ema_slow = calculate_ema(data, slow)
    macd_line = ema_fast - ema_slow
    signal_line = calculate_ema(macd_line, signal)
    histogram = macd_line - signal_line

    return macd_line, signal_line, histogram


def calculate_bollinger_bands(data, window=20, num_std=2):
    """Calculate Bollinger Bands."""
    sma = calculate_sma(data, window)
    std = data.rolling(window=window).std()
    upper_band = sma + (std * num_std)
    lower_band = sma - (std * num_std)

    return upper_band, sma, lower_band


def add_technical_indicators(fig, data, indicators):
    """Add technical indicators to a chart."""
    if 'sma20' in indicators:
        fig.add_trace(go.Scatter(
            x=data.index,
            y=indicators['sma20'],
            name='SMA 20',
            line=dict(color='blue', width=1)
        ))

    if 'sma50' in indicators:
        fig.add_trace(go.Scatter(
            x=data.index,
            y=indicators['sma50'],
            name='SMA 50',
            line=dict(color='orange', width=1)
        ))

    if 'sma200' in indicators:
        fig.add_trace(go.Scatter(
            x=data.index,
            y=indicators['sma200'],
            name='SMA 200',
            line=dict(color='purple', width=1)
        ))

    if 'upper_band' in indicators and 'middle_band' in indicators and 'lower_band' in indicators:
        fig.add_trace(go.Scatter(
            x=data.index,
            y=indicators['upper_band'],
            name='Upper Band',
            line=dict(color='rgba(0, 128, 0, 0.3)', width=1),
            mode='lines'
        ))

        fig.add_trace(go.Scatter(
            x=data.index,
            y=indicators['middle_band'],
            name='Middle Band',
            line=dict(color='rgba(0, 128, 0, 0.7)', width=1),
            mode='lines'
        ))

        fig.add_trace(go.Scatter(
            x=data.index,
            y=indicators['lower_band'],
            name='Lower Band',
            line=dict(color='rgba(0, 128, 0, 0.3)', width=1),
            fill='tonexty',
            fillcolor='rgba(0, 128, 0, 0.05)',
            mode='lines'
        ))

    return fig


def create_trade_markers(fig, trades, row=1, col=1):
    """Add trade markers to a chart."""
    if not trades:
        return fig

    buy_trades = [t for t in trades if t['trade_type'] == 'buy']
    sell_trades = [t for t in trades if t['trade_type'] == 'sell']

    if buy_trades:
        buy_times = [t['timestamp'] for t in buy_trades]
        buy_prices = [t['price'] for t in buy_trades]

        fig.add_trace(
            go.Scatter(
                x=buy_times,
                y=buy_prices,
                mode='markers',
                name='Buy',
                marker=dict(
                    symbol='triangle-up',
                    size=10,
                    color='green',
                    line=dict(width=1, color='green')
                )
            ),
            row=row, col=col
        )

    if sell_trades:
        sell_times = [t['timestamp'] for t in sell_trades]
        sell_prices = [t['price'] for t in sell_trades]

        fig.add_trace(
            go.Scatter(
                x=sell_times,
                y=sell_prices,
                mode='markers',
                name='Sell',
                marker=dict(
                    symbol='triangle-down',
                    size=10,
                    color='red',
                    line=dict(width=1, color='red')
                )
            ),
            row=row, col=col
        )

    return fig


def display_metrics_card(title, metrics):
    """Display a card with metrics."""
    st.subheader(title)

    # Create columns for the metrics
    cols = st.columns(len(metrics))

    for i, (label, value) in enumerate(metrics.items()):
        with cols[i]:
            st.metric(label, value)


def display_prediction_card(symbol, prediction):
    """Display a card with AI prediction information."""
    pred = prediction.get('prediction', 'UNKNOWN')
    confidence = prediction.get('confidence', 0.0)
    price_target = prediction.get('price_target')
    stop_loss = prediction.get('stop_loss')
    timestamp = prediction.get('timestamp', datetime.now())

    # Determine card color based on prediction
    if pred == 'BUY':
        card_color = "rgba(0, 128, 0, 0.1)"  # Green background
    elif pred == 'SELL':
        card_color = "rgba(255, 0, 0, 0.1)"  # Red background
    else:
        card_color = "rgba(128, 128, 128, 0.1)"  # Gray background

    # Create the card with HTML
    html = f"""
    <div style="background-color: {card_color}; padding: 15px; border-radius: 5px; margin-bottom: 10px;">
        <h3 style="margin-top: 0;">{symbol} - {pred}</h3>
        <p><strong>Confidence:</strong> {confidence * 100:.1f}%</p>
        <p><strong>Price Target:</strong> {format_currency(price_target) if price_target else 'N/A'}</p>
        <p><strong>Stop Loss:</strong> {format_currency(stop_loss) if stop_loss else 'N/A'}</p>
        <p><small>Generated: {timestamp.strftime('%Y-%m-%d %H:%M:%S')}</small></p>
    </div>
    """

    st.markdown(html, unsafe_allow_html=True)