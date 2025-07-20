"""
Configuration settings for the AI Trading Agent
"""
import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # Zerodha API Configuration
    ZERODHA_API_KEY = os.getenv('ZERODHA_API_KEY')
    ZERODHA_API_SECRET = os.getenv('ZERODHA_API_SECRET')
    ZERODHA_ACCESS_TOKEN = os.getenv('ZERODHA_ACCESS_TOKEN')
    
    # Database Configuration
    DATABASE_URL = os.getenv('DATABASE_URL', 'postgresql://user:password@localhost/trading_db')
    MONGODB_URL = os.getenv('MONGODB_URL', 'mongodb://localhost:27017/trading_db')
    REDIS_URL = os.getenv('REDIS_URL', 'redis://localhost:6379/0')
    
    # External APIs
    ALPHA_VANTAGE_API_KEY = os.getenv('ALPHA_VANTAGE_API_KEY')
    FRED_API_KEY = os.getenv('FRED_API_KEY')
    NEWS_API_KEY = os.getenv('NEWS_API_KEY')
    TWITTER_BEARER_TOKEN = os.getenv('TWITTER_BEARER_TOKEN')
    
    # Trading Configuration
    INITIAL_CAPITAL = float(os.getenv('INITIAL_CAPITAL', 100000))
    MAX_POSITION_SIZE = float(os.getenv('MAX_POSITION_SIZE', 0.1))  # 10% of capital
    STOP_LOSS_PERCENTAGE = float(os.getenv('STOP_LOSS_PERCENTAGE', 0.02))  # 2%
    TAKE_PROFIT_PERCENTAGE = float(os.getenv('TAKE_PROFIT_PERCENTAGE', 0.05))  # 5%
    
    # Risk Management
    MAX_DAILY_LOSS = float(os.getenv('MAX_DAILY_LOSS', 0.05))  # 5% of capital
    MAX_DRAWDOWN = float(os.getenv('MAX_DRAWDOWN', 0.15))  # 15% of capital
    
    # Model Configuration
    MODEL_UPDATE_FREQUENCY = int(os.getenv('MODEL_UPDATE_FREQUENCY', 24))  # hours
    LOOKBACK_PERIOD = int(os.getenv('LOOKBACK_PERIOD', 252))  # trading days
    
    # Data Collection
    DATA_UPDATE_INTERVAL = int(os.getenv('DATA_UPDATE_INTERVAL', 60))  # seconds
    NEWS_UPDATE_INTERVAL = int(os.getenv('NEWS_UPDATE_INTERVAL', 300))  # seconds
    
    # Logging
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
    LOG_FILE = os.getenv('LOG_FILE', 'logs/trading_agent.log')
    
    # Message Queue
    RABBITMQ_URL = os.getenv('RABBITMQ_URL', 'amqp://localhost:5672')
    
    # Dashboard
    DASHBOARD_HOST = os.getenv('DASHBOARD_HOST', '0.0.0.0')
    DASHBOARD_PORT = int(os.getenv('DASHBOARD_PORT', 5000))
    
    # Instruments to trade
    NIFTY_SYMBOLS = ['NIFTY', 'BANKNIFTY']
    
    # Options parameters
    OPTION_EXPIRY_DAYS = [0, 7, 14, 21, 28]  # Days to expiry to consider
    STRIKE_RANGE = 10  # Number of strikes above and below ATM
    
    @classmethod
    def validate_config(cls):
        """Validate that all required configuration is present"""
        required_vars = [
            'ZERODHA_API_KEY',
            'ZERODHA_API_SECRET',
            'ZERODHA_ACCESS_TOKEN'
        ]
        
        missing_vars = []
        for var in required_vars:
            if not getattr(cls, var):
                missing_vars.append(var)
        
        if missing_vars:
            raise ValueError(f"Missing required environment variables: {missing_vars}")
        
        return True

