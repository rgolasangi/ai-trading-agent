"""
Sentiment Analysis Agent for AI Trading Agent
Analyzes sentiment from news articles and social media posts
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import re
import asyncio
from textblob import TextBlob
import nltk
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch
from sklearn.preprocessing import StandardScaler
import pickle
import os

from utils.logger import get_logger
from config.config import Config

logger = get_logger(__name__)

class SentimentAnalyzer:
    """Core sentiment analysis functionality"""
    
    def __init__(self):
        self.finbert_analyzer = None
        self.general_analyzer = None
        self.tokenizer = None
        self.model = None
        self.scaler = StandardScaler()
        self.initialize_models()
    
    def initialize_models(self):
        """Initialize sentiment analysis models"""
        try:
            # Download required NLTK data
            nltk.download('punkt', quiet=True)
            nltk.download('vader_lexicon', quiet=True)
            nltk.download('stopwords', quiet=True)
            
            # Initialize FinBERT for financial sentiment analysis
            try:
                self.finbert_analyzer = pipeline(
                    "sentiment-analysis",
                    model="ProsusAI/finbert",
                    tokenizer="ProsusAI/finbert",
                    device=0 if torch.cuda.is_available() else -1
                )
                logger.info("FinBERT model initialized successfully")
            except Exception as e:
                logger.warning(f"Could not initialize FinBERT: {e}")
            
            # Initialize general sentiment analyzer as fallback
            try:
                self.general_analyzer = pipeline(
                    "sentiment-analysis",
                    model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                    device=0 if torch.cuda.is_available() else -1
                )
                logger.info("General sentiment analyzer initialized successfully")
            except Exception as e:
                logger.warning(f"Could not initialize general analyzer: {e}")
                
        except Exception as e:
            logger.error(f"Error initializing sentiment models: {e}")
    
    def preprocess_text(self, text: str) -> str:
        """
        Preprocess text for sentiment analysis
        
        Args:
            text: Raw text to preprocess
            
        Returns:
            Preprocessed text
        """
        if not text or not isinstance(text, str):
            return ""
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove user mentions and hashtags (keep the text)
        text = re.sub(r'@\w+|#', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Remove special characters but keep punctuation for sentiment
        text = re.sub(r'[^\w\s.,!?;:]', '', text)
        
        return text
    
    def analyze_sentiment_finbert(self, text: str) -> Dict[str, float]:
        """
        Analyze sentiment using FinBERT
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary with sentiment scores
        """
        try:
            if not self.finbert_analyzer or not text:
                return {'positive': 0.0, 'negative': 0.0, 'neutral': 0.0, 'compound': 0.0}
            
            # Truncate text if too long
            max_length = 512
            if len(text) > max_length:
                text = text[:max_length]
            
            result = self.finbert_analyzer(text)
            
            # Convert to standardized format
            label = result[0]['label'].lower()
            score = result[0]['score']
            
            sentiment_scores = {'positive': 0.0, 'negative': 0.0, 'neutral': 0.0}
            
            if label == 'positive':
                sentiment_scores['positive'] = score
                sentiment_scores['compound'] = score
            elif label == 'negative':
                sentiment_scores['negative'] = score
                sentiment_scores['compound'] = -score
            else:
                sentiment_scores['neutral'] = score
                sentiment_scores['compound'] = 0.0
            
            return sentiment_scores
            
        except Exception as e:
            logger.error(f"Error in FinBERT sentiment analysis: {e}")
            return {'positive': 0.0, 'negative': 0.0, 'neutral': 0.0, 'compound': 0.0}
    
    def analyze_sentiment_textblob(self, text: str) -> Dict[str, float]:
        """
        Analyze sentiment using TextBlob as fallback
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary with sentiment scores
        """
        try:
            if not text:
                return {'positive': 0.0, 'negative': 0.0, 'neutral': 0.0, 'compound': 0.0}
            
            blob = TextBlob(text)
            polarity = blob.sentiment.polarity  # -1 to 1
            subjectivity = blob.sentiment.subjectivity  # 0 to 1
            
            # Convert polarity to positive/negative/neutral scores
            if polarity > 0.1:
                positive = polarity
                negative = 0.0
                neutral = 1 - abs(polarity)
            elif polarity < -0.1:
                positive = 0.0
                negative = abs(polarity)
                neutral = 1 - abs(polarity)
            else:
                positive = 0.0
                negative = 0.0
                neutral = 1.0
            
            return {
                'positive': positive,
                'negative': negative,
                'neutral': neutral,
                'compound': polarity,
                'subjectivity': subjectivity
            }
            
        except Exception as e:
            logger.error(f"Error in TextBlob sentiment analysis: {e}")
            return {'positive': 0.0, 'negative': 0.0, 'neutral': 0.0, 'compound': 0.0}
    
    def analyze_sentiment(self, text: str) -> Dict[str, float]:
        """
        Main sentiment analysis method
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary with sentiment scores
        """
        if not text:
            return {'positive': 0.0, 'negative': 0.0, 'neutral': 0.0, 'compound': 0.0}
        
        # Preprocess text
        processed_text = self.preprocess_text(text)
        
        # Try FinBERT first for financial text
        finbert_scores = self.analyze_sentiment_finbert(processed_text)
        
        # If FinBERT fails or gives neutral results, use TextBlob
        if finbert_scores['compound'] == 0.0:
            textblob_scores = self.analyze_sentiment_textblob(processed_text)
            # Combine scores with weights
            combined_scores = {
                'positive': (finbert_scores['positive'] + textblob_scores['positive']) / 2,
                'negative': (finbert_scores['negative'] + textblob_scores['negative']) / 2,
                'neutral': (finbert_scores['neutral'] + textblob_scores['neutral']) / 2,
                'compound': (finbert_scores['compound'] + textblob_scores['compound']) / 2
            }
            return combined_scores
        
        return finbert_scores

class SentimentAgent:
    """Sentiment Analysis Agent"""
    
    def __init__(self):
        self.analyzer = SentimentAnalyzer()
        self.logger = get_logger(__name__)
        self.sentiment_history = []
        self.market_keywords = [
            'nifty', 'bank nifty', 'sensex', 'stock market', 'trading', 'investment',
            'bull market', 'bear market', 'volatility', 'rbi', 'inflation', 'gdp',
            'interest rate', 'monetary policy', 'fiscal policy', 'budget', 'election'
        ]
    
    def calculate_relevance_score(self, text: str) -> float:
        """
        Calculate how relevant the text is to financial markets
        
        Args:
            text: Text to analyze
            
        Returns:
            Relevance score between 0 and 1
        """
        if not text:
            return 0.0
        
        text_lower = text.lower()
        keyword_count = sum(1 for keyword in self.market_keywords if keyword in text_lower)
        
        # Normalize by text length and keyword count
        relevance = min(keyword_count / len(self.market_keywords), 1.0)
        
        # Boost score if multiple keywords are present
        if keyword_count > 1:
            relevance *= 1.2
        
        return min(relevance, 1.0)
    
    def analyze_news_sentiment(self, news_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze sentiment of news articles
        
        Args:
            news_df: DataFrame containing news articles
            
        Returns:
            Dictionary with aggregated sentiment analysis
        """
        try:
            if news_df.empty:
                return self._get_empty_sentiment_result()
            
            sentiments = []
            
            for _, row in news_df.iterrows():
                # Combine title and content for analysis
                text = f"{row.get('title', '')} {row.get('full_text', '')}"
                
                # Calculate sentiment
                sentiment_scores = self.analyzer.analyze_sentiment(text)
                
                # Calculate relevance
                relevance = self.calculate_relevance_score(text)
                
                # Weight sentiment by relevance
                weighted_sentiment = {
                    'positive': sentiment_scores['positive'] * relevance,
                    'negative': sentiment_scores['negative'] * relevance,
                    'neutral': sentiment_scores['neutral'] * relevance,
                    'compound': sentiment_scores['compound'] * relevance,
                    'relevance': relevance,
                    'timestamp': row.get('timestamp', datetime.now()),
                    'source': row.get('source', 'unknown')
                }
                
                sentiments.append(weighted_sentiment)
            
            # Aggregate sentiments
            return self._aggregate_sentiments(sentiments, 'news')
            
        except Exception as e:
            self.logger.error(f"Error analyzing news sentiment: {e}")
            return self._get_empty_sentiment_result()
    
    def analyze_social_sentiment(self, social_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze sentiment of social media posts
        
        Args:
            social_df: DataFrame containing social media posts
            
        Returns:
            Dictionary with aggregated sentiment analysis
        """
        try:
            if social_df.empty:
                return self._get_empty_sentiment_result()
            
            sentiments = []
            
            for _, row in social_df.iterrows():
                text = row.get('text', '')
                
                # Calculate sentiment
                sentiment_scores = self.analyzer.analyze_sentiment(text)
                
                # Calculate relevance
                relevance = self.calculate_relevance_score(text)
                
                # Weight by engagement metrics if available
                engagement_weight = 1.0
                if 'like_count' in row and 'retweet_count' in row:
                    engagement_weight = 1 + np.log1p(row['like_count'] + row['retweet_count']) / 10
                
                # Weight sentiment by relevance and engagement
                weighted_sentiment = {
                    'positive': sentiment_scores['positive'] * relevance * engagement_weight,
                    'negative': sentiment_scores['negative'] * relevance * engagement_weight,
                    'neutral': sentiment_scores['neutral'] * relevance * engagement_weight,
                    'compound': sentiment_scores['compound'] * relevance * engagement_weight,
                    'relevance': relevance,
                    'engagement_weight': engagement_weight,
                    'timestamp': row.get('timestamp', datetime.now())
                }
                
                sentiments.append(weighted_sentiment)
            
            # Aggregate sentiments
            return self._aggregate_sentiments(sentiments, 'social')
            
        except Exception as e:
            self.logger.error(f"Error analyzing social sentiment: {e}")
            return self._get_empty_sentiment_result()
    
    def _aggregate_sentiments(self, sentiments: List[Dict], source_type: str) -> Dict[str, Any]:
        """
        Aggregate individual sentiment scores
        
        Args:
            sentiments: List of sentiment dictionaries
            source_type: Type of source ('news' or 'social')
            
        Returns:
            Aggregated sentiment analysis
        """
        if not sentiments:
            return self._get_empty_sentiment_result()
        
        # Calculate weighted averages
        total_weight = sum(s.get('relevance', 1.0) * s.get('engagement_weight', 1.0) for s in sentiments)
        
        if total_weight == 0:
            return self._get_empty_sentiment_result()
        
        weighted_positive = sum(s['positive'] for s in sentiments) / total_weight
        weighted_negative = sum(s['negative'] for s in sentiments) / total_weight
        weighted_neutral = sum(s['neutral'] for s in sentiments) / total_weight
        weighted_compound = sum(s['compound'] for s in sentiments) / total_weight
        
        # Calculate sentiment trend (last 24 hours vs previous)
        now = datetime.now()
        recent_sentiments = [s for s in sentiments if (now - s['timestamp']).total_seconds() < 86400]
        older_sentiments = [s for s in sentiments if (now - s['timestamp']).total_seconds() >= 86400]
        
        trend = 0.0
        if recent_sentiments and older_sentiments:
            recent_avg = np.mean([s['compound'] for s in recent_sentiments])
            older_avg = np.mean([s['compound'] for s in older_sentiments])
            trend = recent_avg - older_avg
        
        # Calculate confidence based on sample size and agreement
        confidence = min(len(sentiments) / 100, 1.0)  # More samples = higher confidence
        
        # Calculate volatility of sentiment
        compound_scores = [s['compound'] for s in sentiments]
        volatility = np.std(compound_scores) if len(compound_scores) > 1 else 0.0
        
        result = {
            'source_type': source_type,
            'sentiment_scores': {
                'positive': weighted_positive,
                'negative': weighted_negative,
                'neutral': weighted_neutral,
                'compound': weighted_compound
            },
            'sentiment_label': self._get_sentiment_label(weighted_compound),
            'confidence': confidence,
            'trend': trend,
            'volatility': volatility,
            'sample_size': len(sentiments),
            'timestamp': datetime.now(),
            'raw_sentiments': sentiments[-10:]  # Keep last 10 for debugging
        }
        
        # Store in history
        self.sentiment_history.append(result)
        
        # Keep only last 1000 entries
        if len(self.sentiment_history) > 1000:
            self.sentiment_history = self.sentiment_history[-1000:]
        
        return result
    
    def _get_sentiment_label(self, compound_score: float) -> str:
        """Get sentiment label from compound score"""
        if compound_score >= 0.05:
            return 'positive'
        elif compound_score <= -0.05:
            return 'negative'
        else:
            return 'neutral'
    
    def _get_empty_sentiment_result(self) -> Dict[str, Any]:
        """Return empty sentiment result"""
        return {
            'source_type': 'unknown',
            'sentiment_scores': {
                'positive': 0.0,
                'negative': 0.0,
                'neutral': 1.0,
                'compound': 0.0
            },
            'sentiment_label': 'neutral',
            'confidence': 0.0,
            'trend': 0.0,
            'volatility': 0.0,
            'sample_size': 0,
            'timestamp': datetime.now(),
            'raw_sentiments': []
        }
    
    def get_overall_sentiment(self, news_sentiment: Dict, social_sentiment: Dict) -> Dict[str, Any]:
        """
        Combine news and social sentiment for overall market sentiment
        
        Args:
            news_sentiment: News sentiment analysis result
            social_sentiment: Social sentiment analysis result
            
        Returns:
            Combined sentiment analysis
        """
        try:
            # Weight news sentiment higher than social sentiment for financial decisions
            news_weight = 0.7
            social_weight = 0.3
            
            # Get sentiment scores
            news_scores = news_sentiment.get('sentiment_scores', {})
            social_scores = social_sentiment.get('sentiment_scores', {})
            
            # Calculate weighted average
            combined_scores = {
                'positive': (news_scores.get('positive', 0) * news_weight + 
                           social_scores.get('positive', 0) * social_weight),
                'negative': (news_scores.get('negative', 0) * news_weight + 
                           social_scores.get('negative', 0) * social_weight),
                'neutral': (news_scores.get('neutral', 0) * news_weight + 
                          social_scores.get('neutral', 0) * social_weight),
                'compound': (news_scores.get('compound', 0) * news_weight + 
                           social_scores.get('compound', 0) * social_weight)
            }
            
            # Calculate combined confidence
            combined_confidence = (
                news_sentiment.get('confidence', 0) * news_weight +
                social_sentiment.get('confidence', 0) * social_weight
            )
            
            # Calculate combined trend
            combined_trend = (
                news_sentiment.get('trend', 0) * news_weight +
                social_sentiment.get('trend', 0) * social_weight
            )
            
            return {
                'sentiment_scores': combined_scores,
                'sentiment_label': self._get_sentiment_label(combined_scores['compound']),
                'confidence': combined_confidence,
                'trend': combined_trend,
                'news_sentiment': news_sentiment,
                'social_sentiment': social_sentiment,
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            self.logger.error(f"Error combining sentiments: {e}")
            return self._get_empty_sentiment_result()
    
    def get_sentiment_signals(self, overall_sentiment: Dict) -> Dict[str, Any]:
        """
        Generate trading signals based on sentiment analysis
        
        Args:
            overall_sentiment: Combined sentiment analysis
            
        Returns:
            Dictionary with trading signals
        """
        try:
            compound_score = overall_sentiment.get('sentiment_scores', {}).get('compound', 0)
            confidence = overall_sentiment.get('confidence', 0)
            trend = overall_sentiment.get('trend', 0)
            
            # Generate signals based on sentiment
            signal_strength = 0.0
            signal_direction = 'neutral'
            
            # Strong positive sentiment
            if compound_score > 0.3 and confidence > 0.5:
                signal_strength = min(compound_score * confidence, 1.0)
                signal_direction = 'bullish'
            
            # Strong negative sentiment
            elif compound_score < -0.3 and confidence > 0.5:
                signal_strength = min(abs(compound_score) * confidence, 1.0)
                signal_direction = 'bearish'
            
            # Trend-based signals
            if abs(trend) > 0.2:
                if trend > 0:
                    signal_direction = 'bullish' if signal_direction != 'bearish' else 'neutral'
                else:
                    signal_direction = 'bearish' if signal_direction != 'bullish' else 'neutral'
                
                signal_strength = max(signal_strength, abs(trend))
            
            return {
                'signal_direction': signal_direction,
                'signal_strength': signal_strength,
                'confidence': confidence,
                'reasoning': {
                    'compound_score': compound_score,
                    'trend': trend,
                    'sentiment_label': overall_sentiment.get('sentiment_label', 'neutral')
                },
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            self.logger.error(f"Error generating sentiment signals: {e}")
            return {
                'signal_direction': 'neutral',
                'signal_strength': 0.0,
                'confidence': 0.0,
                'reasoning': {},
                'timestamp': datetime.now()
            }

if __name__ == "__main__":
    # Test the sentiment agent
    import asyncio
    
    async def test_sentiment_agent():
        agent = SentimentAgent()
        
        # Test with sample news data
        news_data = pd.DataFrame([
            {
                'title': 'Nifty hits new high as markets rally',
                'full_text': 'The Nifty index reached a new all-time high today as investors showed strong confidence in the market outlook.',
                'timestamp': datetime.now(),
                'source': 'test'
            },
            {
                'title': 'Bank Nifty falls on RBI concerns',
                'full_text': 'Banking stocks declined after RBI expressed concerns about rising inflation and potential rate hikes.',
                'timestamp': datetime.now(),
                'source': 'test'
            }
        ])
        
        # Test with sample social data
        social_data = pd.DataFrame([
            {
                'text': 'Bullish on #Nifty! Great momentum in the markets today 🚀',
                'timestamp': datetime.now(),
                'like_count': 50,
                'retweet_count': 20
            },
            {
                'text': 'Worried about the market volatility. #BankNifty looking weak',
                'timestamp': datetime.now(),
                'like_count': 10,
                'retweet_count': 5
            }
        ])
        
        # Analyze sentiments
        news_sentiment = agent.analyze_news_sentiment(news_data)
        social_sentiment = agent.analyze_social_sentiment(social_data)
        overall_sentiment = agent.get_overall_sentiment(news_sentiment, social_sentiment)
        signals = agent.get_sentiment_signals(overall_sentiment)
        
        print("News Sentiment:", news_sentiment['sentiment_label'])
        print("Social Sentiment:", social_sentiment['sentiment_label'])
        print("Overall Sentiment:", overall_sentiment['sentiment_label'])
        print("Trading Signal:", signals['signal_direction'], "with strength", signals['signal_strength'])
    
    asyncio.run(test_sentiment_agent())

