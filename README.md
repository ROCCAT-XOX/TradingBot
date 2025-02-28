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
   
