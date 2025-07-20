"""
Data Collection Module for AI Trading Agent
Handles collection of market data, news, and social media data
"""
import asyncio
import aiohttp
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import yfinance as yf
from kiteconnect import KiteConnect
import requests
from fredapi import Fred
import time
import json

from utils.logger import get_logger
from config.config import Config

logger = get_logger(__name__)

class MarketDataCollector:
    """Collects market data from various sources"""
    
    def __init__(self):
        self.kite = None
        self.fred = None
        self.initialize_apis()
    
    def initialize_apis(self):
        """Initialize API connections"""
        try:
            # Initialize Zerodha Kite Connect
            if Config.ZERODHA_API_KEY and Config.ZERODHA_ACCESS_TOKEN:
                self.kite = KiteConnect(api_key=Config.ZERODHA_API_KEY)
                self.kite.set_access_token(Config.ZERODHA_ACCESS_TOKEN)
                logger.info("Zerodha Kite Connect initialized successfully")
            
            # Initialize FRED API
            if Config.FRED_API_KEY:
                self.fred = Fred(api_key=Config.FRED_API_KEY)
                logger.info("FRED API initialized successfully")
                
        except Exception as e:
            logger.error(f"Error initializing APIs: {e}")
    
    async def get_nifty_options_data(self, symbol: str = "NIFTY") -> Dict[str, Any]:
        """
        Get Nifty/BankNifty options data from Zerodha
        
        Args:
            symbol: NIFTY or BANKNIFTY
            
        Returns:
            Dictionary containing options data
        """
        try:
            if not self.kite:
                logger.error("Kite Connect not initialized")
                return {}
            
            # Get instruments list
            instruments = self.kite.instruments("NFO")
            
            # Filter for the specified symbol options
            options_data = []
            current_date = datetime.now().date()
            
            for instrument in instruments:
                if (instrument['name'] == symbol and 
                    instrument['instrument_type'] in ['CE', 'PE']):
                    
                    # Get expiry date
                    expiry_date = datetime.strptime(instrument['expiry'], '%Y-%m-%d').date()
                    days_to_expiry = (expiry_date - current_date).days
                    
                    # Filter by expiry days
                    if days_to_expiry in Config.OPTION_EXPIRY_DAYS:
                        options_data.append({
                            'instrument_token': instrument['instrument_token'],
                            'tradingsymbol': instrument['tradingsymbol'],
                            'name': instrument['name'],
                            'expiry': instrument['expiry'],
                            'strike': instrument['strike'],
                            'instrument_type': instrument['instrument_type'],
                            'days_to_expiry': days_to_expiry
                        })
            
            # Get quotes for options
            if options_data:
                tokens = [str(opt['instrument_token']) for opt in options_data[:100]]  # Limit to avoid API limits
                quotes = self.kite.quote(tokens)
                
                for opt in options_data:
                    token = str(opt['instrument_token'])
                    if token in quotes:
                        quote_data = quotes[token]
                        opt.update({
                            'last_price': quote_data.get('last_price', 0),
                            'bid': quote_data.get('depth', {}).get('buy', [{}])[0].get('price', 0),
                            'ask': quote_data.get('depth', {}).get('sell', [{}])[0].get('price', 0),
                            'volume': quote_data.get('volume', 0),
                            'oi': quote_data.get('oi', 0),
                            'timestamp': datetime.now()
                        })
            
            logger.info(f"Collected {len(options_data)} {symbol} options data points")
            return {
                'symbol': symbol,
                'options': options_data,
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            logger.error(f"Error collecting {symbol} options data: {e}")
            return {}
    
    async def get_underlying_data(self, symbol: str = "NIFTY 50") -> Dict[str, Any]:
        """
        Get underlying index data
        
        Args:
            symbol: Index symbol
            
        Returns:
            Dictionary containing index data
        """
        try:
            # Use yfinance as backup for index data
            if symbol == "NIFTY 50":
                ticker = "^NSEI"
            elif symbol == "BANK NIFTY":
                ticker = "^NSEBANK"
            else:
                ticker = symbol
            
            stock = yf.Ticker(ticker)
            hist = stock.history(period="1d", interval="1m")
            
            if not hist.empty:
                latest = hist.iloc[-1]
                return {
                    'symbol': symbol,
                    'price': latest['Close'],
                    'open': latest['Open'],
                    'high': latest['High'],
                    'low': latest['Low'],
                    'volume': latest['Volume'],
                    'timestamp': datetime.now()
                }
            
        except Exception as e:
            logger.error(f"Error collecting underlying data for {symbol}: {e}")
        
        return {}
    
    async def get_historical_data(self, symbol: str, days: int = 252) -> pd.DataFrame:
        """
        Get historical data for backtesting and model training
        
        Args:
            symbol: Trading symbol
            days: Number of days of historical data
            
        Returns:
            DataFrame with historical data
        """
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            if self.kite and symbol in ['NIFTY', 'BANKNIFTY']:
                # Get from Zerodha if available
                instrument_token = self._get_instrument_token(symbol)
                if instrument_token:
                    historical_data = self.kite.historical_data(
                        instrument_token=instrument_token,
                        from_date=start_date,
                        to_date=end_date,
                        interval="day"
                    )
                    return pd.DataFrame(historical_data)
            
            # Fallback to yfinance
            if symbol == "NIFTY":
                ticker = "^NSEI"
            elif symbol == "BANKNIFTY":
                ticker = "^NSEBANK"
            else:
                ticker = symbol
            
            stock = yf.Ticker(ticker)
            hist = stock.history(start=start_date, end=end_date)
            
            return hist
            
        except Exception as e:
            logger.error(f"Error collecting historical data for {symbol}: {e}")
            return pd.DataFrame()
    
    def _get_instrument_token(self, symbol: str) -> Optional[int]:
        """Get instrument token for a symbol"""
        try:
            if not self.kite:
                return None
            
            instruments = self.kite.instruments("NSE")
            for instrument in instruments:
                if instrument['tradingsymbol'] == symbol:
                    return instrument['instrument_token']
            
        except Exception as e:
            logger.error(f"Error getting instrument token for {symbol}: {e}")
        
        return None
    
    async def get_economic_indicators(self) -> Dict[str, Any]:
        """
        Get economic indicators from FRED
        
        Returns:
            Dictionary containing economic indicators
        """
        try:
            if not self.fred:
                logger.warning("FRED API not initialized")
                return {}
            
            indicators = {
                'GDP': 'GDP',
                'INFLATION': 'CPIAUCSL',
                'UNEMPLOYMENT': 'UNRATE',
                'INTEREST_RATE': 'FEDFUNDS',
                'VIX': 'VIXCLS'
            }
            
            data = {}
            for name, series_id in indicators.items():
                try:
                    series = self.fred.get_series(series_id, limit=1)
                    if not series.empty:
                        data[name] = {
                            'value': series.iloc[-1],
                            'date': series.index[-1],
                            'timestamp': datetime.now()
                        }
                except Exception as e:
                    logger.warning(f"Could not fetch {name}: {e}")
            
            return data
            
        except Exception as e:
            logger.error(f"Error collecting economic indicators: {e}")
            return {}

class NewsDataCollector:
    """Collects news data from various sources"""
    
    def __init__(self):
        self.session = None
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def get_financial_news(self, keywords: List[str] = None) -> List[Dict[str, Any]]:
        """
        Get financial news from various sources
        
        Args:
            keywords: List of keywords to search for
            
        Returns:
            List of news articles
        """
        if keywords is None:
            keywords = ['nifty', 'bank nifty', 'indian stock market', 'rbi', 'inflation']
        
        news_articles = []
        
        # NewsAPI
        if Config.NEWS_API_KEY:
            news_articles.extend(await self._get_news_api_data(keywords))
        
        # Add other news sources here
        # news_articles.extend(await self._get_rss_feeds())
        
        return news_articles
    
    async def _get_news_api_data(self, keywords: List[str]) -> List[Dict[str, Any]]:
        """Get news from NewsAPI"""
        try:
            if not self.session:
                return []
            
            articles = []
            for keyword in keywords:
                url = "https://newsapi.org/v2/everything"
                params = {
                    'q': keyword,
                    'language': 'en',
                    'sortBy': 'publishedAt',
                    'pageSize': 10,
                    'apiKey': Config.NEWS_API_KEY
                }
                
                async with self.session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        for article in data.get('articles', []):
                            articles.append({
                                'title': article.get('title', ''),
                                'description': article.get('description', ''),
                                'content': article.get('content', ''),
                                'url': article.get('url', ''),
                                'published_at': article.get('publishedAt', ''),
                                'source': article.get('source', {}).get('name', ''),
                                'keyword': keyword,
                                'timestamp': datetime.now()
                            })
                
                # Rate limiting
                await asyncio.sleep(1)
            
            return articles
            
        except Exception as e:
            logger.error(f"Error collecting news from NewsAPI: {e}")
            return []

class SocialMediaDataCollector:
    """Collects social media data for sentiment analysis"""
    
    def __init__(self):
        self.session = None
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def get_twitter_data(self, keywords: List[str] = None) -> List[Dict[str, Any]]:
        """
        Get Twitter data for sentiment analysis
        
        Args:
            keywords: List of keywords to search for
            
        Returns:
            List of tweets
        """
        if keywords is None:
            keywords = ['#nifty', '#banknifty', '#stockmarket', '#trading']
        
        tweets = []
        
        if not Config.TWITTER_BEARER_TOKEN:
            logger.warning("Twitter Bearer Token not configured")
            return tweets
        
        try:
            for keyword in keywords:
                url = "https://api.twitter.com/2/tweets/search/recent"
                headers = {
                    'Authorization': f'Bearer {Config.TWITTER_BEARER_TOKEN}'
                }
                params = {
                    'query': keyword,
                    'max_results': 10,
                    'tweet.fields': 'created_at,public_metrics,context_annotations'
                }
                
                async with self.session.get(url, headers=headers, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        for tweet in data.get('data', []):
                            tweets.append({
                                'id': tweet.get('id'),
                                'text': tweet.get('text', ''),
                                'created_at': tweet.get('created_at', ''),
                                'public_metrics': tweet.get('public_metrics', {}),
                                'keyword': keyword,
                                'timestamp': datetime.now()
                            })
                
                # Rate limiting
                await asyncio.sleep(2)
            
        except Exception as e:
            logger.error(f"Error collecting Twitter data: {e}")
        
        return tweets

class DataPreprocessor:
    """Preprocesses collected data for analysis"""
    
    def __init__(self):
        self.logger = get_logger(__name__)
    
    def preprocess_options_data(self, options_data: Dict[str, Any]) -> pd.DataFrame:
        """
        Preprocess options data
        
        Args:
            options_data: Raw options data
            
        Returns:
            Preprocessed DataFrame
        """
        try:
            if not options_data.get('options'):
                return pd.DataFrame()
            
            df = pd.DataFrame(options_data['options'])
            
            # Calculate additional metrics
            df['mid_price'] = (df['bid'] + df['ask']) / 2
            df['bid_ask_spread'] = df['ask'] - df['bid']
            df['spread_percentage'] = (df['bid_ask_spread'] / df['mid_price']) * 100
            
            # Moneyness calculation (requires underlying price)
            # This would be enhanced with actual underlying price
            df['moneyness'] = df['strike'] / 100  # Placeholder
            
            # Time to expiry in years
            df['time_to_expiry'] = df['days_to_expiry'] / 365.25
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error preprocessing options data: {e}")
            return pd.DataFrame()
    
    def preprocess_news_data(self, news_data: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Preprocess news data
        
        Args:
            news_data: Raw news data
            
        Returns:
            Preprocessed DataFrame
        """
        try:
            if not news_data:
                return pd.DataFrame()
            
            df = pd.DataFrame(news_data)
            
            # Clean text data
            df['title'] = df['title'].fillna('')
            df['description'] = df['description'].fillna('')
            df['content'] = df['content'].fillna('')
            
            # Combine text fields
            df['full_text'] = df['title'] + ' ' + df['description'] + ' ' + df['content']
            
            # Convert timestamps
            df['published_at'] = pd.to_datetime(df['published_at'], errors='coerce')
            
            # Remove duplicates
            df = df.drop_duplicates(subset=['title', 'url'])
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error preprocessing news data: {e}")
            return pd.DataFrame()
    
    def preprocess_social_media_data(self, social_data: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Preprocess social media data
        
        Args:
            social_data: Raw social media data
            
        Returns:
            Preprocessed DataFrame
        """
        try:
            if not social_data:
                return pd.DataFrame()
            
            df = pd.DataFrame(social_data)
            
            # Clean text data
            df['text'] = df['text'].fillna('')
            
            # Convert timestamps
            df['created_at'] = pd.to_datetime(df['created_at'], errors='coerce')
            
            # Extract engagement metrics
            if 'public_metrics' in df.columns:
                df['retweet_count'] = df['public_metrics'].apply(
                    lambda x: x.get('retweet_count', 0) if isinstance(x, dict) else 0
                )
                df['like_count'] = df['public_metrics'].apply(
                    lambda x: x.get('like_count', 0) if isinstance(x, dict) else 0
                )
                df['reply_count'] = df['public_metrics'].apply(
                    lambda x: x.get('reply_count', 0) if isinstance(x, dict) else 0
                )
            
            # Remove duplicates
            df = df.drop_duplicates(subset=['id'])
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error preprocessing social media data: {e}")
            return pd.DataFrame()

# Main data collection orchestrator
class DataCollectionOrchestrator:
    """Orchestrates all data collection activities"""
    
    def __init__(self):
        self.market_collector = MarketDataCollector()
        self.preprocessor = DataPreprocessor()
        self.logger = get_logger(__name__)
    
    async def collect_all_data(self) -> Dict[str, Any]:
        """
        Collect all required data
        
        Returns:
            Dictionary containing all collected and preprocessed data
        """
        try:
            self.logger.info("Starting data collection...")
            
            # Collect market data
            nifty_options = await self.market_collector.get_nifty_options_data("NIFTY")
            banknifty_options = await self.market_collector.get_nifty_options_data("BANKNIFTY")
            nifty_underlying = await self.market_collector.get_underlying_data("NIFTY 50")
            banknifty_underlying = await self.market_collector.get_underlying_data("BANK NIFTY")
            economic_indicators = await self.market_collector.get_economic_indicators()
            
            # Collect news data
            async with NewsDataCollector() as news_collector:
                news_data = await news_collector.get_financial_news()
            
            # Collect social media data
            async with SocialMediaDataCollector() as social_collector:
                social_data = await social_collector.get_twitter_data()
            
            # Preprocess data
            processed_data = {
                'nifty_options': self.preprocessor.preprocess_options_data(nifty_options),
                'banknifty_options': self.preprocessor.preprocess_options_data(banknifty_options),
                'nifty_underlying': nifty_underlying,
                'banknifty_underlying': banknifty_underlying,
                'economic_indicators': economic_indicators,
                'news': self.preprocessor.preprocess_news_data(news_data),
                'social_media': self.preprocessor.preprocess_social_media_data(social_data),
                'timestamp': datetime.now()
            }
            
            self.logger.info("Data collection completed successfully")
            return processed_data
            
        except Exception as e:
            self.logger.error(f"Error in data collection orchestrator: {e}")
            return {}

if __name__ == "__main__":
    # Test the data collection
    async def test_data_collection():
        orchestrator = DataCollectionOrchestrator()
        data = await orchestrator.collect_all_data()
        print(f"Collected data keys: {list(data.keys())}")
        
        for key, value in data.items():
            if isinstance(value, pd.DataFrame):
                print(f"{key}: {len(value)} rows")
            elif isinstance(value, dict):
                print(f"{key}: {len(value)} items")
            else:
                print(f"{key}: {type(value)}")
    
    asyncio.run(test_data_collection())

