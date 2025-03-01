import os
import sys
import json
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from pathlib import Path

# Add the parent directory to the path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

# Import our modules
from backend.alpaca_api import AlpacaAPI
from backend.database import Database, AIModel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(parent_dir, 'logs', 'training.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class StockDataset(Dataset):
    """PyTorch Dataset for stock price data."""

    def __init__(self, features, targets):
        """
        Initialize the dataset.

        Args:
            features (np.ndarray): Input features (X)
            targets (np.ndarray): Target values (y)
        """
        self.features = torch.tensor(features, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]


class LSTMModel(nn.Module):
    """Long Short-Term Memory (LSTM) model for time series prediction."""

    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.2):
        """
        Initialize the LSTM model.

        Args:
            input_size (int): Number of input features
            hidden_size (int): Size of the hidden state
            num_layers (int): Number of LSTM layers
            output_size (int): Number of output features
            dropout (float): Dropout rate
        """
        super(LSTMModel, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # Dropout layer
        self.dropout = nn.Dropout(dropout)

        # Fully connected layer
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        """Forward pass."""
        # Initialize hidden state and cell state
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))

        # Get the output from the last time step
        out = out[:, -1, :]

        # Apply dropout
        out = self.dropout(out)

        # Fully connected layer
        out = self.fc(out)

        return out


def prepare_sequence_data(data, sequence_length, prediction_horizon=1, target_col='close'):
    """
    Prepare sequence data for time series prediction.

    Args:
        data (pd.DataFrame): DataFrame with stock price data
        sequence_length (int): Number of time steps to use for each sequence
        prediction_horizon (int): Number of time steps to predict ahead
        target_col (str): Column to use as target variable

    Returns:
        tuple: (X, y) where X is the input sequences and y is the target values
    """
    # Make sure data is sorted by date
    data = data.sort_index()

    # Create sequences
    X, y = [], []

    for i in range(len(data) - sequence_length - prediction_horizon + 1):
        # Extract sequence of features
        sequence = data.iloc[i:i + sequence_length].values
        X.append(sequence)

        # Extract target value (using close price for simplicity)
        target = data.iloc[i + sequence_length + prediction_horizon - 1][target_col]
        y.append(target)

    return np.array(X), np.array(y).reshape(-1, 1)


def create_features(data):
    """
    Create technical indicators and features for the model.

    Args:
        data (pd.DataFrame): DataFrame with OHLCV data

    Returns:
        pd.DataFrame: DataFrame with additional features
    """
    # Create a copy to avoid modifying the original data
    df = data.copy()

    # Calculate returns
    df['returns'] = df['close'].pct_change()
    df['log_returns'] = np.log(df['close'] / df['close'].shift(1))

    # Simple Moving Averages
    df['sma5'] = df['close'].rolling(window=5).mean()
    df['sma10'] = df['close'].rolling(window=10).mean()
    df['sma20'] = df['close'].rolling(window=20).mean()
    df['sma50'] = df['close'].rolling(window=50).mean()

    # Exponential Moving Averages
    df['ema5'] = df['close'].ewm(span=5, adjust=False).mean()
    df['ema10'] = df['close'].ewm(span=10, adjust=False).mean()
    df['ema20'] = df['close'].ewm(span=20, adjust=False).mean()

    # Moving Average Convergence Divergence (MACD)
    df['ema12'] = df['close'].ewm(span=12, adjust=False).mean()
    df['ema26'] = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = df['ema12'] - df['ema26']
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']

    # Relative Strength Index (RSI)
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    df['rsi'] = 100 - (100 / (1 + rs))

    # Bollinger Bands
    df['middle_band'] = df['close'].rolling(window=20).mean()
    df['std'] = df['close'].rolling(window=20).std()
    df['upper_band'] = df['middle_band'] + (df['std'] * 2)
    df['lower_band'] = df['middle_band'] - (df['std'] * 2)
    df['bb_width'] = (df['upper_band'] - df['lower_band']) / df['middle_band']

    # Price relative to moving averages
    df['close_to_sma5'] = df['close'] / df['sma5']
    df['close_to_sma20'] = df['close'] / df['sma20']
    df['close_to_sma50'] = df['close'] / df['sma50']

    # Momentum indicators
    df['momentum_5'] = df['close'] / df['close'].shift(5) - 1
    df['momentum_10'] = df['close'] / df['close'].shift(10) - 1
    df['momentum_20'] = df['close'] / df['close'].shift(20) - 1

    # Average True Range (ATR) - volatility indicator
    df['high_low'] = df['high'] - df['low']
    df['high_close'] = np.abs(df['high'] - df['close'].shift(1))
    df['low_close'] = np.abs(df['low'] - df['close'].shift(1))
    df['tr'] = df[['high_low', 'high_close', 'low_close']].max(axis=1)
    df['atr'] = df['tr'].rolling(window=14).mean()

    # Volume features
    df['volume_sma5'] = df['volume'].rolling(window=5).mean()
    df['volume_ratio'] = df['volume'] / df['volume_sma5']

    # Day of week, month (time-based features)
    if isinstance(df.index, pd.DatetimeIndex):
        df['day_of_week'] = df.index.dayofweek
        df['month'] = df.index.month
        df['quarter'] = df.index.quarter

    # Drop NaN values (from rolling calculations)
    df = df.dropna()

    return df


def fetch_training_data(alpaca_api, symbols, timeframe='1Day', days=365):
    """
    Fetch historical data for model training.

    Args:
        alpaca_api (AlpacaAPI): Initialized Alpaca API client
        symbols (list): List of stock symbols
        timeframe (str): Timeframe for the data (e.g., '1Day', '1Hour')
        days (int): Number of days of historical data to fetch

    Returns:
        dict: Dictionary of DataFrames with historical data
    """
    start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
    end_date = datetime.now().strftime('%Y-%m-%d')

    try:
        logger.info(f"Fetching historical data for {len(symbols)} symbols from {start_date} to {end_date}")
        return alpaca_api.get_historical_bars(symbols, timeframe=timeframe, start=start_date, end=end_date)
    except Exception as e:
        logger.error(f"Error fetching historical data: {e}")
        return {}


def scale_features(train_data, test_data=None, feature_cols=None):
    """
    Scale features using StandardScaler.

    Args:
        train_data (pd.DataFrame): Training data
        test_data (pd.DataFrame, optional): Test data
        feature_cols (list, optional): List of feature columns to scale

    Returns:
        tuple: (scaled_train_data, scaled_test_data, scaler)
    """
    # If feature columns not specified, use all columns
    if feature_cols is None:
        feature_cols = train_data.columns.tolist()

    # Initialize scaler
    scaler = StandardScaler()

    # Fit scaler on training data
    scaler.fit(train_data[feature_cols])

    # Scale training data
    train_scaled = train_data.copy()
    train_scaled[feature_cols] = scaler.transform(train_data[feature_cols])

    # Scale test data if provided
    test_scaled = None
    if test_data is not None:
        test_scaled = test_data.copy()
        test_scaled[feature_cols] = scaler.transform(test_data[feature_cols])

    return train_scaled, test_scaled, scaler


def train_model(model, train_loader, valid_loader, num_epochs, learning_rate, device, early_stopping=False, patience=20,
                min_delta=0.0001):
    """
    Train the PyTorch model with optional early stopping.

    Args:
        model (nn.Module): PyTorch model
        train_loader (DataLoader): Training data loader
        valid_loader (DataLoader): Validation data loader
        num_epochs (int): Number of training epochs
        learning_rate (float): Learning rate
        device (torch.device): Device to train on
        early_stopping (bool): Whether to use early stopping
        patience (int): Number of epochs to wait for improvement before stopping
        min_delta (float): Minimum change to qualify as improvement

    Returns:
        tuple: (trained_model, training_history)
    """
    # Loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )

    # Training history
    history = {
        'train_loss': [],
        'valid_loss': []
    }

    # Early stopping variables
    best_valid_loss = float('inf')
    best_model_state = None
    no_improvement_count = 0

    # Move model to device
    model = model.to(device)

    # Training loop
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0

        for inputs, targets in train_loader:
            # Move data to device
            inputs, targets = inputs.to(device), targets.to(device)

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        # Calculate average training loss
        train_loss /= len(train_loader)
        history['train_loss'].append(train_loss)

        # Validation phase
        model.eval()
        valid_loss = 0.0

        with torch.no_grad():
            for inputs, targets in valid_loader:
                # Move data to device
                inputs, targets = inputs.to(device), targets.to(device)

                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs, targets)

                valid_loss += loss.item()

        # Calculate average validation loss
        valid_loss /= len(valid_loader)
        history['valid_loss'].append(valid_loss)

        # Update learning rate
        scheduler.step(valid_loss)

        # Log progress
        logger.info(f"Epoch {epoch + 1}/{num_epochs} - Train Loss: {train_loss:.6f}, Valid Loss: {valid_loss:.6f}")

        # Early stopping check
        if early_stopping:
            # Check if this is the best model so far
            if valid_loss < best_valid_loss - min_delta:
                best_valid_loss = valid_loss
                no_improvement_count = 0
                # Save the best model state
                best_model_state = model.state_dict().copy()
                logger.info(f"New best model at epoch {epoch + 1} with validation loss: {valid_loss:.6f}")
            else:
                no_improvement_count += 1
                logger.info(
                    f"No improvement for {no_improvement_count} epochs (best: {best_valid_loss:.6f}, current: {valid_loss:.6f})")

                # Stop training if no improvement for 'patience' epochs
                if no_improvement_count >= patience:
                    logger.info(f"Early stopping triggered after {epoch + 1} epochs")
                    if best_model_state:
                        # Restore the best model
                        model.load_state_dict(best_model_state)
                        logger.info("Restored best model")
                    break

    # If using early stopping and we have a best model state, load it
    if early_stopping and best_model_state and not (no_improvement_count >= patience):
        model.load_state_dict(best_model_state)
        logger.info("Training completed. Loaded best model.")
    else:
        logger.info("Training completed")

    return model, history


def evaluate_model(model, test_loader, device):
    """
    Evaluate the model on test data.

    Args:
        model (nn.Module): Trained PyTorch model
        test_loader (DataLoader): Test data loader
        device (torch.device): Device to evaluate on

    Returns:
        tuple: (predictions, actual_values, metrics)
    """
    model.eval()
    predictions = []
    actual_values = []

    with torch.no_grad():
        for inputs, targets in test_loader:
            # Move data to device
            inputs, targets = inputs.to(device), targets.to(device)

            # Forward pass
            outputs = model(inputs)

            # Store predictions and actual values
            predictions.append(outputs.cpu().numpy())
            actual_values.append(targets.cpu().numpy())

    # Concatenate predictions and actual values
    predictions = np.concatenate(predictions)
    actual_values = np.concatenate(actual_values)

    # Calculate metrics
    mse = np.mean((predictions - actual_values) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(predictions - actual_values))

    metrics = {
        'mse': mse,
        'rmse': rmse,
        'mae': mae
    }

    return predictions, actual_values, metrics


def save_model(model, model_name, model_path, scaler=None, metadata=None):
    """
    Save the trained model.

    Args:
        model (nn.Module): Trained PyTorch model
        model_name (str): Name of the model
        model_path (str): Path to save the model
        scaler (StandardScaler, optional): Feature scaler
        metadata (dict, optional): Additional metadata

    Returns:
        str: Path to the saved model
    """
    # Create directory if it doesn't exist
    os.makedirs(model_path, exist_ok=True)

    # Save model state
    model_file = os.path.join(model_path, f"{model_name}.pth")
    torch.save(model.state_dict(), model_file)

    # Save scaler if provided
    if scaler is not None:
        scaler_file = os.path.join(model_path, f"{model_name}_scaler.pkl")
        import pickle
        with open(scaler_file, 'wb') as f:
            pickle.dump(scaler, f)

    # Save metadata if provided
    if metadata is not None:
        # Convert NumPy types to Python native types for JSON serialization
        def convert_numpy_types(obj):
            if isinstance(obj, dict):
                return {k: convert_numpy_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            elif isinstance(obj, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64,
                                  np.uint8, np.uint16, np.uint32, np.uint64)):
                return int(obj)
            elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, (np.ndarray,)):
                return convert_numpy_types(obj.tolist())
            else:
                return obj

        # Convert NumPy types in metadata to standard Python types
        metadata_serializable = convert_numpy_types(metadata)

        metadata_file = os.path.join(model_path, f"{model_name}_metadata.json")
        with open(metadata_file, 'w') as f:
            json.dump(metadata_serializable, f, indent=4)

    logger.info(f"Model saved to {model_file}")
    return model_file


def train_trading_model(config=None):
    import sys
    import traceback

    # System and environment logging
    logger.info(f"Python executable: {sys.executable}")
    logger.info(f"Python version: {sys.version}")
    logger.info(f"Current working directory: {os.getcwd()}")
    logger.info(f"Parent directory: {parent_dir}")

    # Load configuration
    try:
        if config is None:
            config_path = os.path.join(parent_dir, 'config', 'settings.json')
            logger.info(f"Attempting to load configuration from: {config_path}")

            if not os.path.exists(config_path):
                logger.error(f"Configuration file not found: {config_path}")
                raise FileNotFoundError(f"Configuration file not found: {config_path}")

            with open(config_path, 'r') as f:
                config = json.load(f)

    except (FileNotFoundError, json.JSONDecodeError) as e:
        logger.error(f"Configuration loading error: {e}")
        # Fallback to default configuration
        config = {
            'TRADING_SYMBOLS': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA'],
            'DEFAULT_TIMEFRAME': '1Day',
            'TRAINING_PERIOD_DAYS': 365
        }

    # Extract training parameters
    symbols = config.get('TRADING_SYMBOLS', ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA'])
    timeframe = config.get('DEFAULT_TIMEFRAME', '1Day')
    training_days = config.get('TRAINING_PERIOD_DAYS', 365)

    logger.info(f"Training parameters:")
    logger.info(f"  Symbols: {symbols}")
    logger.info(f"  Timeframe: {timeframe}")
    logger.info(f"  Training Days: {training_days}")

    # Initialize database and Alpaca API
    try:
        db = Database()
        alpaca_api = AlpacaAPI(config_path=os.path.join(parent_dir, 'config', 'settings.json'))
    except Exception as e:
        logger.error(f"Failed to initialize database or Alpaca API: {e}")
        return None

    # Set device (GPU if available, otherwise CPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # Fetch training data for all symbols
    try:
        all_data = fetch_training_data(alpaca_api, symbols, timeframe, training_days)
        logger.info(f"Fetched training data for symbols: {list(all_data.keys())}")
    except Exception as e:
        logger.error(f"Failed to fetch training data: {e}")
        return None

    # ML model parameters
    model_params = {
        'sequence_length': 20,
        'prediction_horizon': 1,
        'hidden_size': 128,
        'num_layers': 2,
        'dropout': 0.2,
        'batch_size': 64,
        'learning_rate': 0.001,
        'num_epochs': 5000
    }

    # Update model parameters from AI_MODEL_SETTINGS if available
    ai_model_settings = config.get('AI_MODEL_SETTINGS', {})
    model_params.update({
        'batch_size': ai_model_settings.get('batch_size', model_params['batch_size']),
        'learning_rate': ai_model_settings.get('learning_rate', model_params['learning_rate']),
        'num_epochs': ai_model_settings.get('num_epochs', model_params['num_epochs'])
    })

    # Tracking for saved models and metrics
    saved_models = []
    model_metrics = {}

    # Train models for each symbol
    for symbol in symbols:
        logger.info(f"Starting training for symbol: {symbol}")

        try:
            # Get data for the current symbol
            data = all_data[symbol]

            # Create features
            data_with_features = create_features(data)

            # Drop rows with NaN values
            data_with_features = data_with_features.dropna()

            if len(data_with_features) < 100:
                logger.warning(f"Not enough data for {symbol} after feature creation, skipping")
                continue

            # Define feature columns (exclude target)
            feature_cols = [col for col in data_with_features.columns if col != 'close']

            # Split data into train, validation, and test sets (70/15/15)
            train_data, temp_data = train_test_split(data_with_features, test_size=0.3, shuffle=False)
            valid_data, test_data = train_test_split(temp_data, test_size=0.5, shuffle=False)

            # Scale features
            train_scaled, valid_scaled, scaler = scale_features(train_data, valid_data, feature_cols)
            _, test_scaled, _ = scale_features(train_data, test_data, feature_cols)

            # Prepare sequence data for LSTM
            sequence_length = model_params['sequence_length']
            prediction_horizon = model_params['prediction_horizon']

            # Prepare training data
            X_train, y_train = prepare_sequence_data(
                train_scaled, sequence_length, prediction_horizon, 'close'
            )
            X_valid, y_valid = prepare_sequence_data(
                valid_scaled, sequence_length, prediction_horizon, 'close'
            )
            X_test, y_test = prepare_sequence_data(
                test_scaled, sequence_length, prediction_horizon, 'close'
            )

            # Create datasets
            train_dataset = StockDataset(X_train, y_train)
            valid_dataset = StockDataset(X_valid, y_valid)
            test_dataset = StockDataset(X_test, y_test)

            # Create data loaders
            train_loader = DataLoader(
                train_dataset, batch_size=model_params['batch_size'], shuffle=True
            )
            valid_loader = DataLoader(
                valid_dataset, batch_size=model_params['batch_size'], shuffle=False
            )
            test_loader = DataLoader(
                test_dataset, batch_size=model_params['batch_size'], shuffle=False
            )

            # Create model
            input_size = X_train.shape[2]  # Number of features
            model = LSTMModel(
                input_size=input_size,
                hidden_size=model_params['hidden_size'],
                num_layers=model_params['num_layers'],
                output_size=1,
                dropout=model_params['dropout']
            )

            # Train model
            trained_model, history = train_model(
                model,
                train_loader,
                valid_loader,
                num_epochs=model_params['num_epochs'],
                learning_rate=model_params['learning_rate'],
                device=device
            )

            # Evaluate model
            predictions, actual_values, metrics = evaluate_model(trained_model, test_loader, device)

            logger.info(f"Model evaluation for {symbol}:")
            logger.info(f"  MSE: {metrics['mse']:.6f}")
            logger.info(f"  RMSE: {metrics['rmse']:.6f}")
            logger.info(f"  MAE: {metrics['mae']:.6f}")

            # Save model
            model_path = os.path.join(parent_dir, 'backend', 'models')
            model_name = f"{symbol}_lstm_{datetime.now().strftime('%Y%m%d')}"

            model_file = save_model(
                trained_model,
                model_name,
                model_path,
                scaler=scaler,
                metadata={
                    'symbol': symbol,
                    'timeframe': timeframe,
                    'training_data_start': str(train_data.index[0]),
                    'training_data_end': str(train_data.index[-1]),
                    'model_params': model_params,
                    'feature_columns': feature_cols,
                    'metrics': metrics,
                    'created_at': datetime.now().isoformat()
                }
            )

            # Save model in database
            model_db = db.add_model(
                name=f"LSTM_{symbol}",
                version=datetime.now().strftime('%Y%m%d'),
                model_type="LSTM",
                description=f"LSTM model for predicting {symbol} prices",
                is_active=True,
                performance_metrics={
                    'mse': float(metrics['mse']),
                    'rmse': float(metrics['rmse']),
                    'mae': float(metrics['mae'])
                },
                model_path=model_file
            )

            saved_models.append(symbol)
            model_metrics[symbol] = metrics

            logger.info(f"Successfully trained and saved model for {symbol}")

        except Exception as e:
            logger.error(f"Error training model for {symbol}: {e}")
            import traceback
            traceback.print_exc()

    # Log final results
    logger.info(f"Training completed. Models saved for: {saved_models}")

    return {
        'saved_models': saved_models,
        'model_metrics': model_metrics
    }


def parse_arguments():
    """Parse command line arguments for training."""
    import argparse

    parser = argparse.ArgumentParser(description='Train trading models for selected symbols')
    parser.add_argument('--symbols', type=str, help='Comma-separated list of symbols to train (e.g., AAPL,MSFT,GOOGL)')
    parser.add_argument('--epochs', type=int, default=10000, help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--timeframe', type=str, default='1Day', help='Data timeframe (e.g., 1Day, 1Hour)')
    parser.add_argument('--days', type=int, default=365, help='Number of days of historical data to use')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--unlimited', action='store_true', help='Enable unlimited training with early stopping')
    parser.add_argument('--patience', type=int, default=20, help='Early stopping patience (epochs without improvement)')
    parser.add_argument('--config', type=str, help='Path to custom configuration file')

    return parser.parse_args()


if __name__ == "__main__":
    # Parse command line arguments
    args = parse_arguments()

    # Create models directory if it doesn't exist
    models_dir = os.path.join(parent_dir, 'backend', 'models')
    os.makedirs(models_dir, exist_ok=True)

    # Dictionary für alle Modellmetriken
    all_model_metrics = {}

    # Load configuration
    if args.config:
        with open(args.config, 'r') as f:
            config = json.load(f)
    else:
        config_path = os.path.join(parent_dir, 'config', 'settings.json')
        with open(config_path, 'r') as f:
            config = json.load(f)

    # Override config with command line arguments
    if args.symbols:
        config['TRADING_SYMBOLS'] = args.symbols.split(',')

    # Update AI model settings
    if 'AI_MODEL_SETTINGS' not in config:
        config['AI_MODEL_SETTINGS'] = {}

    if args.unlimited:
        # Sehr hohe Epochenzahl für quasi-unbegrenztes Training mit Early Stopping
        config['AI_MODEL_SETTINGS']['num_epochs'] = 10000
        config['AI_MODEL_SETTINGS']['early_stopping'] = True
        config['AI_MODEL_SETTINGS']['patience'] = args.patience
    elif args.epochs:
        config['AI_MODEL_SETTINGS']['num_epochs'] = args.epochs

    if args.learning_rate:
        config['AI_MODEL_SETTINGS']['learning_rate'] = args.learning_rate

    if args.batch_size:
        config['AI_MODEL_SETTINGS']['batch_size'] = args.batch_size

    if args.timeframe:
        config['DEFAULT_TIMEFRAME'] = args.timeframe

    if args.days:
        config['TRAINING_PERIOD_DAYS'] = args.days

    # Get symbols from config
    symbols = config.get('TRADING_SYMBOLS', ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA','NVD'])

    try:
        # Run training with the updated config
        train_trading_model(config)

        # Trainingsabschluss-Information loggen und notifizieren
        from backend.training_completion_utils import log_training_completion

        log_training_completion(
            symbols=symbols,
            model_path=models_dir,
            model_metrics=all_model_metrics,
            config=config
        )

        logger.info("Training completed successfully!")
    except Exception as e:
        logger.error(f"Training failed with error: {e}")