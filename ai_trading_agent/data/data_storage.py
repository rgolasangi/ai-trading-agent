"""
Data Storage Module for AI Trading Agent
Handles storage and retrieval of collected data
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
import json
import pickle
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Text, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.dialects.postgresql import JSON
import pymongo
import redis
import os

from utils.logger import get_logger
from config.config import Config

logger = get_logger(__name__)

# SQLAlchemy Base
Base = declarative_base()

# Database Models
class MarketData(Base):
    __tablename__ = 'market_data'
    
    id = Column(Integer, primary_key=True)
    symbol = Column(String(50), nullable=False)
    timestamp = Column(DateTime, nullable=False)
    open_price = Column(Float)
    high_price = Column(Float)
    low_price = Column(Float)
    close_price = Column(Float)
    volume = Column(Float)
    data_type = Column(String(20))  # 'underlying', 'option'
    additional_data = Column(JSON)

class OptionsData(Base):
    __tablename__ = 'options_data'
    
    id = Column(Integer, primary_key=True)
    symbol = Column(String(50), nullable=False)
    strike = Column(Float, nullable=False)
    expiry = Column(DateTime, nullable=False)
    option_type = Column(String(2), nullable=False)  # CE or PE
    timestamp = Column(DateTime, nullable=False)
    last_price = Column(Float)
    bid = Column(Float)
    ask = Column(Float)
    volume = Column(Float)
    open_interest = Column(Float)
    implied_volatility = Column(Float)
    delta = Column(Float)
    gamma = Column(Float)
    theta = Column(Float)
    vega = Column(Float)
    rho = Column(Float)

class NewsData(Base):
    __tablename__ = 'news_data'
    
    id = Column(Integer, primary_key=True)
    title = Column(Text, nullable=False)
    content = Column(Text)
    url = Column(Text)
    source = Column(String(100))
    published_at = Column(DateTime)
    timestamp = Column(DateTime, nullable=False)
    sentiment_score = Column(Float)
    keywords = Column(JSON)

class SocialMediaData(Base):
    __tablename__ = 'social_media_data'
    
    id = Column(Integer, primary_key=True)
    platform = Column(String(50), nullable=False)
    post_id = Column(String(100), nullable=False)
    text = Column(Text, nullable=False)
    created_at = Column(DateTime)
    timestamp = Column(DateTime, nullable=False)
    sentiment_score = Column(Float)
    engagement_metrics = Column(JSON)

class TradingSignals(Base):
    __tablename__ = 'trading_signals'
    
    id = Column(Integer, primary_key=True)
    symbol = Column(String(50), nullable=False)
    signal_type = Column(String(20), nullable=False)  # BUY, SELL, HOLD
    confidence = Column(Float, nullable=False)
    timestamp = Column(DateTime, nullable=False)
    model_version = Column(String(50))
    features_used = Column(JSON)
    executed = Column(Boolean, default=False)

class PostgreSQLStorage:
    """PostgreSQL storage for structured data"""
    
    def __init__(self):
        self.engine = None
        self.Session = None
        self.initialize_connection()
    
    def initialize_connection(self):
        """Initialize PostgreSQL connection"""
        try:
            self.engine = create_engine(Config.DATABASE_URL)
            Base.metadata.create_all(self.engine)
            self.Session = sessionmaker(bind=self.engine)
            logger.info("PostgreSQL connection initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing PostgreSQL connection: {e}")
    
    def store_market_data(self, data: Dict[str, Any]) -> bool:
        """Store market data"""
        try:
            session = self.Session()
            
            # Store underlying data
            for symbol in ['nifty_underlying', 'banknifty_underlying']:
                if symbol in data and data[symbol]:
                    underlying_data = data[symbol]
                    market_record = MarketData(
                        symbol=underlying_data['symbol'],
                        timestamp=underlying_data['timestamp'],
                        open_price=underlying_data.get('open'),
                        high_price=underlying_data.get('high'),
                        low_price=underlying_data.get('low'),
                        close_price=underlying_data.get('price'),
                        volume=underlying_data.get('volume'),
                        data_type='underlying'
                    )
                    session.add(market_record)
            
            session.commit()
            session.close()
            logger.info("Market data stored successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error storing market data: {e}")
            if session:
                session.rollback()
                session.close()
            return False
    
    def store_options_data(self, options_df: pd.DataFrame, symbol: str) -> bool:
        """Store options data"""
        try:
            if options_df.empty:
                return True
            
            session = self.Session()
            
            for _, row in options_df.iterrows():
                option_record = OptionsData(
                    symbol=symbol,
                    strike=row.get('strike', 0),
                    expiry=pd.to_datetime(row.get('expiry')),
                    option_type=row.get('instrument_type', ''),
                    timestamp=row.get('timestamp', datetime.now()),
                    last_price=row.get('last_price', 0),
                    bid=row.get('bid', 0),
                    ask=row.get('ask', 0),
                    volume=row.get('volume', 0),
                    open_interest=row.get('oi', 0)
                )
                session.add(option_record)
            
            session.commit()
            session.close()
            logger.info(f"Options data for {symbol} stored successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error storing options data for {symbol}: {e}")
            if session:
                session.rollback()
                session.close()
            return False
    
    def store_news_data(self, news_df: pd.DataFrame) -> bool:
        """Store news data"""
        try:
            if news_df.empty:
                return True
            
            session = self.Session()
            
            for _, row in news_df.iterrows():
                news_record = NewsData(
                    title=row.get('title', ''),
                    content=row.get('full_text', ''),
                    url=row.get('url', ''),
                    source=row.get('source', ''),
                    published_at=row.get('published_at'),
                    timestamp=row.get('timestamp', datetime.now())
                )
                session.add(news_record)
            
            session.commit()
            session.close()
            logger.info("News data stored successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error storing news data: {e}")
            if session:
                session.rollback()
                session.close()
            return False
    
    def store_social_media_data(self, social_df: pd.DataFrame) -> bool:
        """Store social media data"""
        try:
            if social_df.empty:
                return True
            
            session = self.Session()
            
            for _, row in social_df.iterrows():
                social_record = SocialMediaData(
                    platform='twitter',
                    post_id=str(row.get('id', '')),
                    text=row.get('text', ''),
                    created_at=row.get('created_at'),
                    timestamp=row.get('timestamp', datetime.now()),
                    engagement_metrics={
                        'retweet_count': row.get('retweet_count', 0),
                        'like_count': row.get('like_count', 0),
                        'reply_count': row.get('reply_count', 0)
                    }
                )
                session.add(social_record)
            
            session.commit()
            session.close()
            logger.info("Social media data stored successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error storing social media data: {e}")
            if session:
                session.rollback()
                session.close()
            return False
    
    def get_latest_market_data(self, symbol: str, limit: int = 100) -> pd.DataFrame:
        """Get latest market data"""
        try:
            session = self.Session()
            
            query = session.query(MarketData).filter(
                MarketData.symbol == symbol
            ).order_by(MarketData.timestamp.desc()).limit(limit)
            
            data = []
            for record in query:
                data.append({
                    'symbol': record.symbol,
                    'timestamp': record.timestamp,
                    'open': record.open_price,
                    'high': record.high_price,
                    'low': record.low_price,
                    'close': record.close_price,
                    'volume': record.volume
                })
            
            session.close()
            return pd.DataFrame(data)
            
        except Exception as e:
            logger.error(f"Error retrieving market data for {symbol}: {e}")
            return pd.DataFrame()
    
    def get_latest_options_data(self, symbol: str, limit: int = 1000) -> pd.DataFrame:
        """Get latest options data"""
        try:
            session = self.Session()
            
            query = session.query(OptionsData).filter(
                OptionsData.symbol == symbol
            ).order_by(OptionsData.timestamp.desc()).limit(limit)
            
            data = []
            for record in query:
                data.append({
                    'symbol': record.symbol,
                    'strike': record.strike,
                    'expiry': record.expiry,
                    'option_type': record.option_type,
                    'timestamp': record.timestamp,
                    'last_price': record.last_price,
                    'bid': record.bid,
                    'ask': record.ask,
                    'volume': record.volume,
                    'open_interest': record.open_interest,
                    'delta': record.delta,
                    'gamma': record.gamma,
                    'theta': record.theta,
                    'vega': record.vega,
                    'rho': record.rho
                })
            
            session.close()
            return pd.DataFrame(data)
            
        except Exception as e:
            logger.error(f"Error retrieving options data for {symbol}: {e}")
            return pd.DataFrame()

class MongoDBStorage:
    """MongoDB storage for unstructured data"""
    
    def __init__(self):
        self.client = None
        self.db = None
        self.initialize_connection()
    
    def initialize_connection(self):
        """Initialize MongoDB connection"""
        try:
            self.client = pymongo.MongoClient(Config.MONGODB_URL)
            self.db = self.client.trading_db
            logger.info("MongoDB connection initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing MongoDB connection: {e}")
    
    def store_raw_data(self, collection_name: str, data: Dict[str, Any]) -> bool:
        """Store raw data in MongoDB"""
        try:
            if not self.db:
                return False
            
            collection = self.db[collection_name]
            data['_timestamp'] = datetime.now()
            result = collection.insert_one(data)
            
            logger.info(f"Raw data stored in {collection_name}: {result.inserted_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error storing raw data in {collection_name}: {e}")
            return False
    
    def get_raw_data(self, collection_name: str, query: Dict = None, limit: int = 100) -> List[Dict]:
        """Get raw data from MongoDB"""
        try:
            if not self.db:
                return []
            
            collection = self.db[collection_name]
            query = query or {}
            
            cursor = collection.find(query).sort('_timestamp', -1).limit(limit)
            return list(cursor)
            
        except Exception as e:
            logger.error(f"Error retrieving raw data from {collection_name}: {e}")
            return []

class RedisCache:
    """Redis cache for real-time data"""
    
    def __init__(self):
        self.client = None
        self.initialize_connection()
    
    def initialize_connection(self):
        """Initialize Redis connection"""
        try:
            self.client = redis.from_url(Config.REDIS_URL)
            self.client.ping()
            logger.info("Redis connection initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing Redis connection: {e}")
    
    def set_data(self, key: str, data: Any, expiry: int = 3600) -> bool:
        """Set data in Redis cache"""
        try:
            if not self.client:
                return False
            
            serialized_data = json.dumps(data, default=str)
            self.client.setex(key, expiry, serialized_data)
            return True
            
        except Exception as e:
            logger.error(f"Error setting data in Redis for key {key}: {e}")
            return False
    
    def get_data(self, key: str) -> Optional[Any]:
        """Get data from Redis cache"""
        try:
            if not self.client:
                return None
            
            data = self.client.get(key)
            if data:
                return json.loads(data)
            return None
            
        except Exception as e:
            logger.error(f"Error getting data from Redis for key {key}: {e}")
            return None
    
    def delete_data(self, key: str) -> bool:
        """Delete data from Redis cache"""
        try:
            if not self.client:
                return False
            
            self.client.delete(key)
            return True
            
        except Exception as e:
            logger.error(f"Error deleting data from Redis for key {key}: {e}")
            return False

class FileStorage:
    """File-based storage for models and large datasets"""
    
    def __init__(self, base_path: str = "data/files"):
        self.base_path = base_path
        os.makedirs(base_path, exist_ok=True)
    
    def save_dataframe(self, df: pd.DataFrame, filename: str, format: str = 'parquet') -> bool:
        """Save DataFrame to file"""
        try:
            filepath = os.path.join(self.base_path, filename)
            
            if format == 'parquet':
                df.to_parquet(filepath)
            elif format == 'csv':
                df.to_csv(filepath, index=False)
            elif format == 'pickle':
                df.to_pickle(filepath)
            else:
                raise ValueError(f"Unsupported format: {format}")
            
            logger.info(f"DataFrame saved to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving DataFrame to {filename}: {e}")
            return False
    
    def load_dataframe(self, filename: str, format: str = 'parquet') -> pd.DataFrame:
        """Load DataFrame from file"""
        try:
            filepath = os.path.join(self.base_path, filename)
            
            if not os.path.exists(filepath):
                logger.warning(f"File not found: {filepath}")
                return pd.DataFrame()
            
            if format == 'parquet':
                return pd.read_parquet(filepath)
            elif format == 'csv':
                return pd.read_csv(filepath)
            elif format == 'pickle':
                return pd.read_pickle(filepath)
            else:
                raise ValueError(f"Unsupported format: {format}")
            
        except Exception as e:
            logger.error(f"Error loading DataFrame from {filename}: {e}")
            return pd.DataFrame()
    
    def save_model(self, model: Any, filename: str) -> bool:
        """Save model to file"""
        try:
            filepath = os.path.join(self.base_path, filename)
            
            with open(filepath, 'wb') as f:
                pickle.dump(model, f)
            
            logger.info(f"Model saved to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving model to {filename}: {e}")
            return False
    
    def load_model(self, filename: str) -> Optional[Any]:
        """Load model from file"""
        try:
            filepath = os.path.join(self.base_path, filename)
            
            if not os.path.exists(filepath):
                logger.warning(f"Model file not found: {filepath}")
                return None
            
            with open(filepath, 'rb') as f:
                model = pickle.load(f)
            
            logger.info(f"Model loaded from {filepath}")
            return model
            
        except Exception as e:
            logger.error(f"Error loading model from {filename}: {e}")
            return None

class DataStorageManager:
    """Main data storage manager that coordinates all storage backends"""
    
    def __init__(self):
        self.postgres = PostgreSQLStorage()
        self.mongodb = MongoDBStorage()
        self.redis = RedisCache()
        self.file_storage = FileStorage()
        self.logger = get_logger(__name__)
    
    def store_collected_data(self, data: Dict[str, Any]) -> bool:
        """Store all collected data using appropriate storage backends"""
        try:
            success = True
            
            # Store in PostgreSQL
            if not self.postgres.store_market_data(data):
                success = False
            
            if 'nifty_options' in data:
                if not self.postgres.store_options_data(data['nifty_options'], 'NIFTY'):
                    success = False
            
            if 'banknifty_options' in data:
                if not self.postgres.store_options_data(data['banknifty_options'], 'BANKNIFTY'):
                    success = False
            
            if 'news' in data:
                if not self.postgres.store_news_data(data['news']):
                    success = False
            
            if 'social_media' in data:
                if not self.postgres.store_social_media_data(data['social_media']):
                    success = False
            
            # Store raw data in MongoDB
            if not self.mongodb.store_raw_data('market_snapshots', data):
                success = False
            
            # Cache latest data in Redis
            cache_key = f"latest_data_{datetime.now().strftime('%Y%m%d_%H%M')}"
            if not self.redis.set_data(cache_key, data, expiry=3600):
                success = False
            
            # Save DataFrames to files for backup
            for key, value in data.items():
                if isinstance(value, pd.DataFrame) and not value.empty:
                    filename = f"{key}_{datetime.now().strftime('%Y%m%d_%H%M')}.parquet"
                    if not self.file_storage.save_dataframe(value, filename):
                        success = False
            
            if success:
                self.logger.info("All data stored successfully across all backends")
            else:
                self.logger.warning("Some data storage operations failed")
            
            return success
            
        except Exception as e:
            self.logger.error(f"Error in data storage manager: {e}")
            return False
    
    def get_latest_data(self, data_type: str, symbol: str = None) -> Union[pd.DataFrame, Dict, None]:
        """Get latest data of specified type"""
        try:
            if data_type == 'market_data' and symbol:
                return self.postgres.get_latest_market_data(symbol)
            elif data_type == 'options_data' and symbol:
                return self.postgres.get_latest_options_data(symbol)
            elif data_type == 'cached_data':
                # Get from Redis cache
                cache_keys = self.redis.client.keys("latest_data_*")
                if cache_keys:
                    latest_key = sorted(cache_keys)[-1].decode('utf-8')
                    return self.redis.get_data(latest_key)
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error retrieving {data_type} data: {e}")
            return None

if __name__ == "__main__":
    # Test the storage system
    storage_manager = DataStorageManager()
    
    # Test data
    test_data = {
        'nifty_underlying': {
            'symbol': 'NIFTY 50',
            'price': 19500.0,
            'open': 19450.0,
            'high': 19550.0,
            'low': 19400.0,
            'volume': 1000000,
            'timestamp': datetime.now()
        },
        'timestamp': datetime.now()
    }
    
    # Test storage
    result = storage_manager.store_collected_data(test_data)
    print(f"Storage test result: {result}")
    
    # Test retrieval
    latest_data = storage_manager.get_latest_data('market_data', 'NIFTY 50')
    print(f"Retrieved data shape: {latest_data.shape if isinstance(latest_data, pd.DataFrame) else type(latest_data)}")

