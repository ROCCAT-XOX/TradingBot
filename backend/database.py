import os
import json
import logging
import math
import pandas as pd
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Boolean, Text, MetaData, Table, \
    ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

Base = declarative_base()


class Stock(Base):
    """Model for storing stock information."""
    __tablename__ = 'stocks'

    id = Column(Integer, primary_key=True)
    symbol = Column(String(10), unique=True, nullable=False)
    name = Column(String(100), nullable=True)
    sector = Column(String(50), nullable=True)
    industry = Column(String(100), nullable=True)
    created_at = Column(DateTime, default=datetime.now)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)

    # Relationships
    price_data = relationship("PriceData", back_populates="stock")
    trades = relationship("Trade", back_populates="stock")
    predictions = relationship("ModelPrediction", back_populates="stock")

    def __repr__(self):
        return f"<Stock(symbol='{self.symbol}', name='{self.name}')>"


class PriceData(Base):
    """Model for storing historical price data."""
    __tablename__ = 'price_data'

    id = Column(Integer, primary_key=True)
    stock_id = Column(Integer, ForeignKey('stocks.id'), nullable=False)
    timestamp = Column(DateTime, nullable=False)
    open = Column(Float, nullable=False)
    high = Column(Float, nullable=False)
    low = Column(Float, nullable=False)
    close = Column(Float, nullable=False)
    volume = Column(Integer, nullable=False)
    timeframe = Column(String(10), nullable=False)  # e.g., '1Min', '1Hour', '1Day'

    # Relationships
    stock = relationship("Stock", back_populates="price_data")

    def __repr__(self):
        return f"<PriceData(symbol='{self.stock.symbol}', timestamp='{self.timestamp}', close='{self.close}')>"


class Trade(Base):
    """Model for storing trades executed by the trading bot."""
    __tablename__ = 'trades'

    id = Column(Integer, primary_key=True)
    stock_id = Column(Integer, ForeignKey('stocks.id'), nullable=False)
    trade_type = Column(String(10), nullable=False)  # 'buy' or 'sell'
    quantity = Column(Integer, nullable=False)
    price = Column(Float, nullable=False)
    timestamp = Column(DateTime, default=datetime.now, nullable=False)
    order_id = Column(String(50), nullable=True)  # Alpaca order ID
    strategy = Column(String(50), nullable=True)  # Strategy that generated this trade
    model_id = Column(Integer, ForeignKey('models.id'), nullable=True)  # AI model that made the decision
    confidence = Column(Float, nullable=True)  # Confidence score of the AI prediction
    profit_loss = Column(Float, nullable=True)  # Realized profit/loss (for sell trades)
    notes = Column(Text, nullable=True)

    # Relationships
    stock = relationship("Stock", back_populates="trades")
    model = relationship("AIModel", back_populates="trades")

    def __repr__(self):
        return f"<Trade(symbol='{self.stock.symbol}', type='{self.trade_type}', price='{self.price}', quantity='{self.quantity}')>"


class AIModel(Base):
    """Model for storing information about trained AI models."""
    __tablename__ = 'models'

    id = Column(Integer, primary_key=True)
    name = Column(String(100), nullable=False)
    version = Column(String(20), nullable=False)
    model_type = Column(String(50), nullable=False)  # e.g., 'PPO', 'DQN', 'LSTM'
    description = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.now, nullable=False)
    is_active = Column(Boolean, default=False)
    performance_metrics = Column(Text, nullable=True)  # JSON string with metrics
    model_path = Column(String(255), nullable=True)  # Path to the saved model file

    # Relationships
    trades = relationship("Trade", back_populates="model")
    predictions = relationship("ModelPrediction", back_populates="model")

    def __repr__(self):
        return f"<AIModel(name='{self.name}', version='{self.version}', active='{self.is_active}')>"


class ModelPrediction(Base):
    """Model for storing AI model predictions."""
    __tablename__ = 'model_predictions'

    id = Column(Integer, primary_key=True)
    stock_id = Column(Integer, ForeignKey('stocks.id'), nullable=False)
    model_id = Column(Integer, ForeignKey('models.id'), nullable=True)
    timestamp = Column(DateTime, default=datetime.now, nullable=False)
    predicted_value = Column(Float, nullable=False)
    actual_value = Column(Float, nullable=True)  # Updated after we know the actual value
    confidence = Column(Float, nullable=True)
    error_percent = Column(Float, nullable=True)  # Difference between predicted and actual
    is_correct = Column(Boolean, nullable=True)  # Whether prediction was correct within threshold
    notes = Column(Text, nullable=True)

    # Relationships
    stock = relationship("Stock", back_populates="predictions")
    model = relationship("AIModel", back_populates="predictions")

    def __repr__(self):
        return f"<ModelPrediction(stock='{self.stock.symbol if self.stock else None}', predicted='{self.predicted_value}', actual='{self.actual_value}')>"


class Database:
    """Handler for database operations."""

    def __init__(self, config_path=None):
        """Initialize the database connection.

        Args:
            config_path (str, optional): Path to the settings.json file.
                If None, will use environment variables.
        """
        # Get the project root directory
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        # Create the data directory if it doesn't exist
        data_dir = os.path.join(project_root, 'backend', 'data')
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)

        # Define the SQLite database path
        db_path = os.path.join(data_dir, 'trading_bot.db')
        logger.info(f"Using SQLite database at: {db_path}")

        # Create the SQLite database URL
        self.database_url = f"sqlite:///{db_path}"

        try:
            # Create the engine
            self.engine = create_engine(self.database_url)

            # Create all tables if they don't exist
            Base.metadata.create_all(self.engine)

            # Create a session factory
            self.Session = sessionmaker(bind=self.engine)

            logger.info("SQLite database connection established")
        except Exception as e:
            logger.error(f"Error connecting to SQLite database: {e}")
            raise

    def get_session(self):
        """Get a new database session.

        Returns:
            Session: SQLAlchemy session object.
        """
        return self.Session()

    def add_stock(self, symbol, name=None, sector=None, industry=None):
        """Add a new stock to the database.

        Args:
            symbol (str): Stock symbol.
            name (str, optional): Company name.
            sector (str, optional): Company sector.
            industry (str, optional): Company industry.

        Returns:
            Stock: The created Stock object.
        """
        session = self.get_session()
        try:
            # Check if stock already exists
            stock = session.query(Stock).filter_by(symbol=symbol).first()
            if stock:
                logger.info(f"Stock {symbol} already exists in database")
                return stock

            # Create a new stock
            stock = Stock(
                symbol=symbol,
                name=name,
                sector=sector,
                industry=industry
            )

            session.add(stock)
            session.commit()
            logger.info(f"Added stock {symbol} to database")
            return stock
        except Exception as e:
            session.rollback()
            logger.error(f"Error adding stock {symbol}: {e}")
            raise
        finally:
            session.close()

    def add_price_data(self, stock_symbol, dataframe, timeframe='1Day'):
        """Add price data for a stock from a pandas DataFrame.

        Args:
            stock_symbol (str): Stock symbol.
            dataframe (pd.DataFrame): DataFrame with price data.
            timeframe (str, optional): Timeframe of the data.

        Returns:
            int: Number of records added.
        """
        session = self.get_session()
        try:
            # Get the stock
            stock = session.query(Stock).filter_by(symbol=stock_symbol).first()
            if not stock:
                logger.warning(f"Stock {stock_symbol} not found, adding it")
                stock = self.add_stock(stock_symbol)

            # Prepare the data
            count = 0
            for _, row in dataframe.iterrows():
                # Check if this timestamp already exists
                existing = session.query(PriceData).filter_by(
                    stock_id=stock.id,
                    timestamp=row.name if isinstance(row.name, datetime) else pd.to_datetime(row.name),
                    timeframe=timeframe
                ).first()

                if existing:
                    # Update existing record
                    existing.open = row['open']
                    existing.high = row['high']
                    existing.low = row['low']
                    existing.close = row['close']
                    existing.volume = row['volume']
                else:
                    # Create a new record
                    price_data = PriceData(
                        stock_id=stock.id,
                        timestamp=row.name if isinstance(row.name, datetime) else pd.to_datetime(row.name),
                        open=row['open'],
                        high=row['high'],
                        low=row['low'],
                        close=row['close'],
                        volume=row['volume'],
                        timeframe=timeframe
                    )
                    session.add(price_data)
                    count += 1

            session.commit()
            logger.info(f"Added {count} price records for {stock_symbol}")
            return count
        except Exception as e:
            session.rollback()
            logger.error(f"Error adding price data for {stock_symbol}: {e}")
            raise
        finally:
            session.close()

    def add_trade(self, stock_symbol, trade_type, quantity, price, timestamp=None, order_id=None,
                  strategy=None, model_id=None, confidence=None, profit_loss=None, notes=None):
        """Add a trade to the database.

        Args:
            stock_symbol (str): Stock symbol.
            trade_type (str): 'buy' or 'sell'.
            quantity (int): Number of shares.
            price (float): Trade price.
            timestamp (datetime, optional): Trade timestamp.
            order_id (str, optional): Alpaca order ID.
            strategy (str, optional): Trading strategy.
            model_id (int, optional): AI model ID.
            confidence (float, optional): AI prediction confidence.
            profit_loss (float, optional): Realized profit/loss.
            notes (str, optional): Additional notes.

        Returns:
            Trade: The created Trade object.
        """
        session = self.get_session()
        try:
            # Get the stock
            stock = session.query(Stock).filter_by(symbol=stock_symbol).first()
            if not stock:
                logger.warning(f"Stock {stock_symbol} not found, adding it")
                stock = self.add_stock(stock_symbol)

            # Create a new trade
            trade = Trade(
                stock_id=stock.id,
                trade_type=trade_type,
                quantity=quantity,
                price=price,
                timestamp=timestamp or datetime.now(),
                order_id=order_id,
                strategy=strategy,
                model_id=model_id,
                confidence=confidence,
                profit_loss=profit_loss,
                notes=notes
            )

            session.add(trade)
            session.commit()
            logger.info(f"Added {trade_type} trade for {stock_symbol} at ${price:.2f} x {quantity}")
            return trade
        except Exception as e:
            session.rollback()
            logger.error(f"Error adding trade for {stock_symbol}: {e}")
            raise
        finally:
            session.close()

    def add_model(self, name, version, model_type, description=None, is_active=False,
                  performance_metrics=None, model_path=None):
        """Add a new AI model to the database.

        Args:
            name (str): Model name.
            version (str): Model version.
            model_type (str): Model type (e.g., 'PPO', 'DQN').
            description (str, optional): Model description.
            is_active (bool, optional): Whether this is the active model.
            performance_metrics (dict, optional): Performance metrics (will be stored as JSON).
            model_path (str, optional): Path to the saved model file.

        Returns:
            AIModel: The created AIModel object.
        """
        session = self.get_session()
        try:
            # Convert performance metrics to JSON if provided
            if performance_metrics and isinstance(performance_metrics, dict):
                performance_metrics = json.dumps(performance_metrics)

            # Create a new AI model
            model = AIModel(
                name=name,
                version=version,
                model_type=model_type,
                description=description,
                is_active=is_active,
                performance_metrics=performance_metrics,
                model_path=model_path
            )

            # If this is active, deactivate other models
            if is_active:
                active_models = session.query(AIModel).filter_by(is_active=True).all()
                for active_model in active_models:
                    active_model.is_active = False

            session.add(model)
            session.commit()
            logger.info(f"Added AI model {name} v{version}")
            return model
        except Exception as e:
            session.rollback()
            logger.error(f"Error adding AI model {name}: {e}")
            raise
        finally:
            session.close()

    def get_stock_price_data(self, symbol, timeframe='1Day', start_date=None, end_date=None):
        """Get historical price data for a stock.

        Args:
            symbol (str): Stock symbol.
            timeframe (str, optional): Timeframe of the data.
            start_date (datetime, optional): Start date for filtering.
            end_date (datetime, optional): End date for filtering.

        Returns:
            pd.DataFrame: DataFrame with price data.
        """
        session = self.get_session()
        try:
            # Get the stock
            stock = session.query(Stock).filter_by(symbol=symbol).first()
            if not stock:
                logger.warning(f"Stock {symbol} not found in database")
                return pd.DataFrame()

            # Build the query
            query = session.query(PriceData).filter_by(stock_id=stock.id, timeframe=timeframe)

            if start_date:
                query = query.filter(PriceData.timestamp >= start_date)
            if end_date:
                query = query.filter(PriceData.timestamp <= end_date)

            # Get the data
            price_data = query.order_by(PriceData.timestamp).all()

            # Convert to DataFrame
            if not price_data:
                return pd.DataFrame()

            data = {
                'open': [p.open for p in price_data],
                'high': [p.high for p in price_data],
                'low': [p.low for p in price_data],
                'close': [p.close for p in price_data],
                'volume': [p.volume for p in price_data]
            }

            df = pd.DataFrame(data, index=[p.timestamp for p in price_data])
            return df
        except Exception as e:
            logger.error(f"Error getting price data for {symbol}: {e}")
            raise
        finally:
            session.close()

    def get_stock_trades(self, symbol=None, start_date=None, end_date=None, trade_type=None, limit=None):
        """Get trades for a stock or all stocks.

        Args:
            symbol (str, optional): Stock symbol. If None, get trades for all stocks.
            start_date (datetime, optional): Start date for filtering.
            end_date (datetime, optional): End date for filtering.
            trade_type (str, optional): Filter by trade type ('buy' or 'sell').
            limit (int, optional): Limit number of results.

        Returns:
            list: List of Trade objects.
        """
        session = self.get_session()
        try:
            # Build the query
            query = session.query(Trade)

            if symbol:
                stock = session.query(Stock).filter_by(symbol=symbol).first()
                if not stock:
                    logger.warning(f"Stock {symbol} not found in database")
                    return []
                query = query.filter_by(stock_id=stock.id)

            if start_date:
                query = query.filter(Trade.timestamp >= start_date)
            if end_date:
                query = query.filter(Trade.timestamp <= end_date)
            if trade_type:
                query = query.filter_by(trade_type=trade_type)

            # Order by timestamp (newest first)
            query = query.order_by(Trade.timestamp.desc())

            # Limit results if specified
            if limit:
                query = query.limit(limit)

            # Get the trades
            trades = query.all()
            return trades
        except Exception as e:
            logger.error(f"Error getting trades: {e}")
            raise
        finally:
            session.close()

    def get_active_model(self):
        """Get the currently active AI model.

        Returns:
            AIModel: The active AIModel object, or None if no active model.
        """
        session = self.get_session()
        try:
            return session.query(AIModel).filter_by(is_active=True).first()
        except Exception as e:
            logger.error(f"Error getting active model: {e}")
            raise
        finally:
            session.close()

    def get_model_by_symbol(self, symbol):
        """Get the AI model for a specific stock symbol.

        Args:
            symbol (str): Stock symbol

        Returns:
            AIModel: AI model for the symbol
        """
        session = self.get_session()
        try:
            # Look for models matching the symbol
            models = session.query(AIModel).filter(AIModel.name.like(f'%{symbol}%')).order_by(
                AIModel.created_at.desc()).all()
            return models[0] if models else None
        except Exception as e:
            logger.error(f"Error getting model for {symbol}: {e}")
            return None
        finally:
            session.close()

    def save_model_prediction(self, symbol, prediction, confidence, timestamp=None, model_id=None):
        """Save a model prediction to the database.

        Args:
            symbol (str): Stock symbol
            prediction (float): Predicted price
            confidence (float): Prediction confidence
            timestamp (datetime, optional): Prediction timestamp
            model_id (int, optional): AI model ID

        Returns:
            ModelPrediction: The created ModelPrediction object
        """
        session = self.get_session()
        try:
            # Get the stock
            stock = session.query(Stock).filter_by(symbol=symbol).first()
            if not stock:
                logger.warning(f"Stock {symbol} not found, adding it")
                stock = self.add_stock(symbol)

            # Get active model if model_id not provided
            if not model_id:
                active_model = self.get_active_model()
                model_id = active_model.id if active_model else None

            # Create new prediction
            prediction_obj = ModelPrediction(
                stock_id=stock.id,
                model_id=model_id,
                predicted_value=prediction,
                confidence=confidence,
                timestamp=timestamp or datetime.now(),
                is_correct=None  # Will be updated later when we know the actual result
            )

            session.add(prediction_obj)
            session.commit()

            logger.info(f"Saved prediction for {symbol}: {prediction} with {confidence:.2f} confidence")
            return prediction_obj
        except Exception as e:
            session.rollback()
            logger.error(f"Error saving prediction for {symbol}: {e}")
            raise
        finally:
            session.close()

    def update_prediction_accuracy(self, prediction_id, actual_value):
        """Update the accuracy of a prediction.

        Args:
            prediction_id (int): Prediction ID
            actual_value (float): Actual price value

        Returns:
            ModelPrediction: The updated ModelPrediction object
        """
        session = self.get_session()
        try:
            # Get the prediction
            prediction = session.query(ModelPrediction).filter_by(id=prediction_id).first()
            if not prediction:
                logger.warning(f"Prediction {prediction_id} not found")
                return None

            # Calculate accuracy
            predicted = prediction.predicted_value
            error_percent = abs(predicted - actual_value) / actual_value * 100

            # Update prediction
            prediction.actual_value = actual_value
            prediction.error_percent = error_percent
            prediction.is_correct = error_percent <= 1.0  # Consider correct if within 1%

            session.commit()
            return prediction
        except Exception as e:
            session.rollback()
            logger.error(f"Error updating prediction {prediction_id}: {e}")
            raise
        finally:
            session.close()

    def get_model_performance(self, model_id=None, symbol=None, days=30):
        """Get the performance of a model.

        Args:
            model_id (int, optional): Model ID
            symbol (str, optional): Stock symbol
            days (int, optional): Number of days to look back

        Returns:
            dict: Performance metrics
        """
        session = self.get_session()
        try:
            # Build query for predictions
            query = session.query(ModelPrediction)

            if model_id:
                query = query.filter_by(model_id=model_id)
            elif symbol:
                # Get stock by symbol
                stock = session.query(Stock).filter_by(symbol=symbol).first()
                if not stock:
                    logger.warning(f"Stock {symbol} not found")
                    return {}
                query = query.filter_by(stock_id=stock.id)

            # Filter by date
            if days:
                start_date = datetime.now() - timedelta(days=days)
                query = query.filter(ModelPrediction.timestamp >= start_date)

            # Get predictions
            predictions = query.all()

            if not predictions:
                return {}

            # Calculate metrics
            total_predictions = len(predictions)
            correct_predictions = sum(1 for p in predictions if p.is_correct)
            accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0

            # Calculate RMSE for predictions with actual values
            predictions_with_actuals = [p for p in predictions if p.actual_value is not None]
            if predictions_with_actuals:
                mse = sum((p.predicted_value - p.actual_value) ** 2 for p in predictions_with_actuals) / len(
                    predictions_with_actuals)
                rmse = math.sqrt(mse)
            else:
                mse = rmse = None

            return {
                'total_predictions': total_predictions,
                'correct_predictions': correct_predictions,
                'accuracy': accuracy,
                'mse': mse,
                'rmse': rmse
            }
        except Exception as e:
            logger.error(f"Error getting model performance: {e}")
            return {}
        finally:
            session.close()


if __name__ == "__main__":
    # Simple test of the database connection
    import os
    import sys

    # Set up logging for the test
    logging.basicConfig(level=logging.INFO)

    try:
        # Initialize the database
        db = Database()

        # Test adding a stock
        stock = db.add_stock('AAPL', 'Apple Inc.', 'Technology', 'Consumer Electronics')
        print(f"Added stock: {stock}")

        # Test adding some dummy price data
        data = {
            'open': [150.0, 151.0, 149.5],
            'high': [152.5, 153.0, 150.2],
            'low': [148.0, 149.5, 147.8],
            'close': [151.5, 149.8, 148.5],
            'volume': [10000000, 12000000, 9000000]
        }

        dates = pd.date_range(end=pd.Timestamp.now(), periods=3, freq='D')
        df = pd.DataFrame(data, index=dates)

        count = db.add_price_data('AAPL', df)
        print(f"Added {count} price records")

        # Test adding a trade
        trade = db.add_trade('AAPL', 'buy', 10, 150.25, notes="Test trade")
        print(f"Added trade: {trade}")

        # Test adding an AI model
        model = db.add_model('DQN_Model', '1.0', 'DQN', 'Initial DQN model', is_active=True)
        print(f"Added model: {model}")

        # Test retrieving data
        price_data = db.get_stock_price_data('AAPL')
        print(f"Retrieved {len(price_data)} price records")
        print(price_data.head())

        trades = db.get_stock_trades('AAPL')
        print(f"Retrieved {len(trades)} trades")
        for t in trades:
            print(t)

        active_model = db.get_active_model()
        print(f"Active model: {active_model}")

        # Test adding a prediction
        prediction = db.save_model_prediction('AAPL', 155.0, 0.8, model_id=model.id)
        print(f"Added prediction: {prediction}")

        # Test updating prediction accuracy
        updated_prediction = db.update_prediction_accuracy(prediction.id, 156.0)
        print(f"Updated prediction: {updated_prediction}")

        # Test getting model performance
        performance = db.get_model_performance(model_id=model.id)
        print(f"Model performance: {performance}")

        print("\nDatabase test successful!")

    except Exception as e:
        print(f"Error testing database: {e}")
        sys.exit(1)