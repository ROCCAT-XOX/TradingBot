# Trading AI Project

A machine learning-powered trading platform that supports both stocks and cryptocurrencies using Alpaca Markets API for real-time data and paper trading.

## Project Overview

This project is a complete trading system that:
1. Fetches real-time market data via Alpaca's API
2. Visualizes market data on interactive dashboards
3. Trains AI models to make trading decisions
4. Executes trades based on AI recommendations in a paper/live trading account
5. Supports comprehensive reasoning and decision-making processes
6. Handles both stock and cryptocurrency assets

## Key Features

- **Multi-Asset Support**: Trade both stocks and cryptocurrencies with the same platform
- **Advanced AI Models**: Uses various machine learning techniques including LSTM, reinforcement learning and reasoning models
- **Detailed Reasoning**: Provides comprehensive analysis and reasoning behind every trading decision
- **Real-time Data**: Connect to live market data for up-to-date information
- **Interactive Dashboards**: Multiple specialized dashboards for different aspects of the trading system
- **Paper Trading Mode**: Practice without risking real money
- **Training Mode**: Develop and test AI models before deployment
- **Performance Tracking**: Monitor the performance of your strategies and models

## Installation

### Prerequisites

- Python 3.8+
- PostgreSQL database
- Alpaca Markets account with API keys

### Installation Steps

1. Clone the repository:
git clone https://github.com/yourusername/trading-ai-project.git
cd trading-ai-project
Copy
2. Create a virtual environment:
python -m venv venv
Copy
3. Activate the virtual environment:

- On Windows:
  ```
  venv\Scripts\activate
  ```

- On macOS/Linux:
  ```
  source venv/bin/activate
  ```

4. Install dependencies:
pip install -r requirements.txt
Copy
5. Create a PostgreSQL database:
CREATE DATABASE trading_bot;
Copy
6. Configure settings:

Copy `config/settings.json.example` to `config/settings.json` and update with your Alpaca API keys and database credentials:

```json
{
    "ALPACA_API_KEY": "your_api_key_here",
    "ALPACA_API_SECRET": "your_api_secret_here",
    "ALPACA_API_BASE_URL": "https://paper-api.alpaca.markets",
    "ALPACA_DATA_URL": "https://data.alpaca.markets",

    "DB_HOST": "localhost",
    "DB_PORT": "5432",
    "DB_NAME": "trading_bot",
    "DB_USER": "postgres",
    "DB_PASSWORD": "your_db_password_here",

    "TRADING_SYMBOLS": ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"],
    "CRYPTO_SYMBOLS": ["BTC/USD", "ETH/USD", "SOL/USD", "DOGE/USD", "AVAX/USD"]
}

Initialize the database:
Copypython backend/database.py


Running the Application
The application has multiple components that can be started together or individually.
Starting Everything (Recommended for New Users)
To start all components at once, run:
Copypython run_all_dashboards.py --all
This will launch all dashboards, the trading bot, and connect to live data.
Starting Individual Components

Main Dashboard - Complete trading platform with charts, portfolio, and predictions:
Copypython start.py --dashboard

Bot Prediction Dashboard - Focused on bot trading signals and decision reasoning:
Copypython frontend/prediction_dashboard.py

Training Mode - For developing and testing AI models:
Copypython start_training_mode.py --bot

Simple Dashboard - Lightweight version for basics:
Copypython frontend/simple_dashboard.py

WebSocket Listener - For real-time market data:
Copypython start.py --websocket


Training AI Models
To train AI models for specific symbols:
Copypython backend/train.py --symbols AAPL,MSFT,GOOGL --epochs 50
Options:

--symbols: Comma-separated list of stock symbols (e.g., AAPL,MSFT,GOOGL)
--epochs: Number of training epochs
--timeframe: Data timeframe (1Day, 1Hour, etc.)
--days: Number of days of historical data to use
--unlimited: Enable unlimited training with early stopping

Dashboard Features
Main Dashboard

Stock/Crypto Charts: Interactive price charts with technical indicators
Portfolio Overview: Current positions and allocation
Trading Signals: AI-generated trading recommendations

Bot Prediction Dashboard

Prediction Overview: All current trading signals and confidence
Market Summary: Overall market condition analysis
Detailed Analysis: In-depth reasoning behind each trading decision
Asset Recommendations: Top opportunities based on AI analysis

Training Dashboard

Model Training Controls: Train and manage AI models
Performance Metrics: Track model accuracy and improvements
Bot Testing: Test models in paper trading mode

Trading Strategies
The system supports multiple trading strategies:

ML-Based Strategy: Uses LSTM neural networks to predict price movements
Technical Analysis Strategy: Based on technical indicators like RSI, MACD, etc.
Reasoning Strategy: Advanced decision-making with comprehensive market analysis

Working with Cryptocurrencies
The platform fully supports cryptocurrency trading through Alpaca's crypto API:

Configure the crypto symbols in settings.json:
jsonCopy"CRYPTO_SYMBOLS": ["BTC/USD", "ETH/USD", "SOL/USD", "DOGE/USD", "AVAX/USD"]

Use the CryptoAPI class for specialized crypto operations:
pythonCopyfrom backend.crypto_api import CryptoAPI

crypto_api = CryptoAPI(config_path='config/settings.json')
crypto_data = crypto_api.get_crypto_data(['BTC/USD', 'ETH/USD'])

Place crypto orders:
pythonCopy# Buy 0.01 BTC
crypto_api.place_crypto_order(symbol='BTC/USD', qty=0.01, side='buy')


Extending the Platform
Adding New Assets
Add new stocks or cryptocurrencies in the settings.json file:
jsonCopy"TRADING_SYMBOLS": ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "YOUR_NEW_STOCK"],
"CRYPTO_SYMBOLS": ["BTC/USD", "ETH/USD", "SOL/USD", "YOUR_NEW_CRYPTO/USD"]
Creating Custom Strategies
Create a new strategy by extending the TradingStrategy base class:
pythonCopyfrom backend.trade_bot import TradingStrategy

class MyCustomStrategy(TradingStrategy):
    def __init__(self, symbol, timeframe='1Day'):
        super().__init__(symbol, timeframe)
        # Initialize your strategy parameters
        
    def generate_signal(self, data):
        # Implement your trading logic
        # Return a signal dictionary
        return {
            'symbol': self.symbol,
            'action': 'BUY',  # or 'SELL' or 'HOLD'
            'confidence': 0.85,
            'reasoning': 'Detailed explanation of your decision'
        }
Adding Custom Visualizations
Add new visualizations to the dashboards by creating custom Plotly charts:
pythonCopyimport plotly.graph_objects as go

def create_custom_chart(data, symbol):