# Trading AI Project

A machine learning-powered stock trading bot that uses Alpaca Markets API for real-time data and paper trading.

## Project Overview

This project is a complete trading system that:
1. Fetches real-time stock data via Alpaca's API
2. Visualizes market data on a Streamlit dashboard
3. Will eventually train AI models to make trading decisions
4. Will execute trades based on AI recommendations in a paper trading account

## Project Structure

```
trading-ai-project/
│── backend/                   # Backend logic & AI
│   ├── data/                   # Raw data (optional, for local storage)
│   ├── __init__.py  
│   ├── models/                 # Trained AI models
│   ├── database.py             # PostgreSQL connection
│   ├── alpaca_api.py           # Alpaca API interface
│   ├── train.py                # AI training & Reinforcement Learning
│   ├── trade_bot.py            # Real-time trading strategy with AI
│   ├── websocket_listener.py   # WebSocket handler for live market data
│   ├── websocket_integration.py # Integration to process WebSocket data
│
│── frontend/                   # Visualization & UI
│   ├── simple_dashboard.py  
│   ├── app.py                  # Streamlit app
│   ├── dashboard.py            # Main dashboard with charts
│   ├── utils.py                # Helper functions for visualizations
│   
│── venv/  
│
│── config/                     # Configuration files
│   ├── settings.json           # API keys & environment variables
│
│── logs/                       # Logging for debugging
│
│── requirements.txt            # Python dependencies
│── README.md                   # Documentation
│── start.py                    # Main startup script
```

## Setup Instructions

### Prerequisites

- Python 3.8+
- PostgreSQL database
- Alpaca Markets account with API keys

### Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/trading-ai-project.git
   cd trading-ai-project
   ```

2. Create a virtual environment:
   ```
   python -m venv venv
   ```

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
   ```
   pip install -r requirements.txt
   ```

5. Create a PostgreSQL database:
   ```
   CREATE DATABASE trading_bot;
   ```

6. Configure settings:

   Copy `config/settings.json.example` to `config/settings.json` (if not already present) and update with your Alpaca API keys and database credentials:

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

       "TRADING_SYMBOLS": ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
   }
   ```

### Running the Application

The application has multiple components that can be started independently or together.

#### Starting Everything (Recommended for New Users)

To start both the WebSocket listener for real-time data and the dashboard, run:

```
python start.py --all
```

or simply:

```
python start.py
```

#### Starting Components Individually

1. To start only the Streamlit dashboard:
   ```
   python start.py --dashboard
   ```
   
   Alternatively, you can run Streamlit directly:
   ```
   streamlit run frontend/dashboard.py
   ```

2. To start only the WebSocket integration for real-time market data:
   ```
   python start.py --websocket
   ```
   
   Alternatively, you can run the WebSocket integration directly:
   ```
   python backend/websocket_integration.py
   ```
   
3. Training Bot
   Starte auch den Trading-Bot im Test-Modus
   ````
   python start_training_mode.py --bot
   ````
   Starte auch das einfache Dashboard
   ````
   python start_training_mode.py --bot --simple
   
   ````
   Starte alle zusammen
   ````
   python start_training_mode.py --bot --simple
 

### Dashboard Features

- **Stock Charts**: View historical price data with technical indicators
- **Portfolio Overview**: Track your current positions and portfolio allocation
- **AI Predictions**: View simulated trading signals (in the future, these will be based on actual AI models)

### WebSocket Integration

The WebSocket integration connects to Alpaca's streaming API to receive real-time market data, including:

- Trade updates
- Quote updates (bid/ask prices)
- Bar updates (OHLC price data)

This data is processed and stored in the database for later analysis and visualization on the dashboard.

## Development

### Adding New Features

1. **Creating a new visualization**: Add your visualization code to `frontend/dashboard.py` or create a new module in the `frontend` directory.

2. **Implementing a new trading strategy**: Create a new module in the `backend` directory and update `backend/trade_bot.py` to use your strategy.

3. **Training AI models**: Implement your training logic in `backend/train.py` and save trained models to the `backend/models` directory.

### Troubleshooting

If you encounter issues:

1. Check the log files in the `logs` directory for detailed error messages.
2. Verify your Alpaca API credentials and database connection settings in `config/settings.json`.
3. Make sure PostgreSQL is running and accessible with the provided credentials.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [Alpaca Markets](https://alpaca.markets/) for providing the trading API
- [Streamlit](https://streamlit.io/) for the interactive dashboard framework