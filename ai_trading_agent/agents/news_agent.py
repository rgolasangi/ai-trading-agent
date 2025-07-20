"""
News Analysis Agent for AI Trading Agent
Analyzes news articles for market-relevant information and events
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import re
import asyncio
import spacy
from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import os
import json

from utils.logger import get_logger
from config.config import Config

logger = get_logger(__name__)

class NewsAnalyzer:
    """Core news analysis functionality"""
    
    def __init__(self):
        self.nlp = None
        self.ner_pipeline = None
        self.summarizer = None
        self.tfidf_vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        self.event_keywords = self._load_event_keywords()
        self.initialize_models()
    
    def initialize_models(self):
        """Initialize NLP models"""
        try:
            # Load spaCy model for NER and text processing
            try:
                self.nlp = spacy.load("en_core_web_sm")
                logger.info("spaCy model loaded successfully")
            except OSError:
                logger.warning("spaCy model not found. Installing...")
                os.system("python -m spacy download en_core_web_sm")
                self.nlp = spacy.load("en_core_web_sm")
            
            # Initialize NER pipeline for financial entities
            try:
                self.ner_pipeline = pipeline(
                    "ner",
                    model="dbmdz/bert-large-cased-finetuned-conll03-english",
                    aggregation_strategy="simple",
                    device=0 if torch.cuda.is_available() else -1
                )
                logger.info("NER pipeline initialized successfully")
            except Exception as e:
                logger.warning(f"Could not initialize NER pipeline: {e}")
            
            # Initialize summarization pipeline
            try:
                self.summarizer = pipeline(
                    "summarization",
                    model="facebook/bart-large-cnn",
                    device=0 if torch.cuda.is_available() else -1
                )
                logger.info("Summarization pipeline initialized successfully")
            except Exception as e:
                logger.warning(f"Could not initialize summarization pipeline: {e}")
                
        except Exception as e:
            logger.error(f"Error initializing news analysis models: {e}")
    
    def _load_event_keywords(self) -> Dict[str, List[str]]:
        """Load predefined event keywords for classification"""
        return {
            'monetary_policy': [
                'interest rate', 'rbi', 'repo rate', 'reverse repo', 'monetary policy',
                'inflation', 'cpi', 'wpi', 'rate hike', 'rate cut', 'liquidity'
            ],
            'earnings': [
                'earnings', 'quarterly results', 'profit', 'revenue', 'guidance',
                'eps', 'dividend', 'financial results', 'q1', 'q2', 'q3', 'q4'
            ],
            'corporate_actions': [
                'merger', 'acquisition', 'ipo', 'rights issue', 'bonus shares',
                'stock split', 'buyback', 'delisting', 'spin-off'
            ],
            'regulatory': [
                'sebi', 'regulation', 'compliance', 'policy', 'government',
                'tax', 'gst', 'budget', 'amendment', 'circular'
            ],
            'market_structure': [
                'trading halt', 'circuit breaker', 'volatility', 'volume',
                'liquidity crisis', 'market crash', 'bull market', 'bear market'
            ],
            'global_events': [
                'fed', 'federal reserve', 'ecb', 'brexit', 'trade war',
                'geopolitical', 'oil prices', 'gold prices', 'dollar index'
            ],
            'sector_specific': [
                'banking', 'it', 'pharma', 'auto', 'fmcg', 'metals',
                'energy', 'infrastructure', 'telecom', 'real estate'
            ]
        }
    
    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """
        Extract named entities from text
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary of entity types and their values
        """
        entities = {
            'organizations': [],
            'persons': [],
            'locations': [],
            'financial_instruments': [],
            'dates': [],
            'money': []
        }
        
        try:
            if self.nlp:
                doc = self.nlp(text)
                
                for ent in doc.ents:
                    if ent.label_ in ['ORG', 'COMPANY']:
                        entities['organizations'].append(ent.text)
                    elif ent.label_ in ['PERSON']:
                        entities['persons'].append(ent.text)
                    elif ent.label_ in ['GPE', 'LOC']:
                        entities['locations'].append(ent.text)
                    elif ent.label_ in ['DATE', 'TIME']:
                        entities['dates'].append(ent.text)
                    elif ent.label_ in ['MONEY', 'PERCENT']:
                        entities['money'].append(ent.text)
            
            # Use transformer-based NER as backup
            if self.ner_pipeline:
                try:
                    ner_results = self.ner_pipeline(text[:512])  # Limit text length
                    for entity in ner_results:
                        entity_type = entity['entity_group']
                        entity_text = entity['word']
                        
                        if entity_type == 'ORG':
                            entities['organizations'].append(entity_text)
                        elif entity_type == 'PER':
                            entities['persons'].append(entity_text)
                        elif entity_type == 'LOC':
                            entities['locations'].append(entity_text)
                except Exception as e:
                    logger.warning(f"Error in transformer NER: {e}")
            
            # Remove duplicates and clean
            for key in entities:
                entities[key] = list(set([ent.strip() for ent in entities[key] if ent.strip()]))
            
            return entities
            
        except Exception as e:
            logger.error(f"Error extracting entities: {e}")
            return entities
    
    def classify_event_type(self, text: str) -> Dict[str, float]:
        """
        Classify the type of event described in the text
        
        Args:
            text: Text to classify
            
        Returns:
            Dictionary with event type probabilities
        """
        text_lower = text.lower()
        event_scores = {}
        
        for event_type, keywords in self.event_keywords.items():
            score = 0
            for keyword in keywords:
                if keyword in text_lower:
                    # Weight by keyword importance (longer keywords get higher weight)
                    weight = len(keyword.split())
                    score += weight
            
            # Normalize by number of keywords and text length
            normalized_score = score / (len(keywords) * len(text.split()) / 100)
            event_scores[event_type] = min(normalized_score, 1.0)
        
        return event_scores
    
    def extract_key_phrases(self, text: str, num_phrases: int = 5) -> List[str]:
        """
        Extract key phrases from text using TF-IDF
        
        Args:
            text: Text to analyze
            num_phrases: Number of key phrases to extract
            
        Returns:
            List of key phrases
        """
        try:
            if not text:
                return []
            
            # Preprocess text
            sentences = text.split('.')
            if len(sentences) < 2:
                return [text[:100]]  # Return truncated text if too short
            
            # Fit TF-IDF vectorizer
            tfidf_matrix = self.tfidf_vectorizer.fit_transform(sentences)
            feature_names = self.tfidf_vectorizer.get_feature_names_out()
            
            # Get top features
            scores = tfidf_matrix.sum(axis=0).A1
            top_indices = scores.argsort()[-num_phrases:][::-1]
            
            key_phrases = [feature_names[i] for i in top_indices if scores[i] > 0]
            
            return key_phrases
            
        except Exception as e:
            logger.error(f"Error extracting key phrases: {e}")
            return []
    
    def summarize_text(self, text: str, max_length: int = 150) -> str:
        """
        Summarize text using transformer model
        
        Args:
            text: Text to summarize
            max_length: Maximum length of summary
            
        Returns:
            Summarized text
        """
        try:
            if not self.summarizer or not text:
                # Fallback to simple truncation
                return text[:max_length] + "..." if len(text) > max_length else text
            
            # Ensure text is long enough for summarization
            if len(text.split()) < 50:
                return text
            
            # Truncate if too long for model
            if len(text) > 1024:
                text = text[:1024]
            
            summary = self.summarizer(
                text,
                max_length=max_length,
                min_length=30,
                do_sample=False
            )
            
            return summary[0]['summary_text']
            
        except Exception as e:
            logger.error(f"Error summarizing text: {e}")
            return text[:max_length] + "..." if len(text) > max_length else text
    
    def calculate_market_impact_score(self, text: str, entities: Dict) -> float:
        """
        Calculate potential market impact score based on content
        
        Args:
            text: News text
            entities: Extracted entities
            
        Returns:
            Market impact score between 0 and 1
        """
        impact_score = 0.0
        
        # Check for high-impact keywords
        high_impact_keywords = [
            'crash', 'rally', 'surge', 'plunge', 'soar', 'tumble',
            'record high', 'record low', 'all-time high', 'all-time low',
            'emergency', 'crisis', 'bailout', 'default', 'bankruptcy'
        ]
        
        text_lower = text.lower()
        for keyword in high_impact_keywords:
            if keyword in text_lower:
                impact_score += 0.2
        
        # Check for financial figures
        money_entities = entities.get('money', [])
        if money_entities:
            impact_score += min(len(money_entities) * 0.1, 0.3)
        
        # Check for major organizations
        major_orgs = ['rbi', 'sebi', 'government', 'ministry', 'nse', 'bse']
        org_entities = [org.lower() for org in entities.get('organizations', [])]
        for org in major_orgs:
            if any(org in entity for entity in org_entities):
                impact_score += 0.15
        
        # Check for market-specific terms
        market_terms = ['nifty', 'sensex', 'bank nifty', 'stock market', 'share market']
        for term in market_terms:
            if term in text_lower:
                impact_score += 0.1
        
        return min(impact_score, 1.0)

class NewsAgent:
    """News Analysis Agent"""
    
    def __init__(self):
        self.analyzer = NewsAnalyzer()
        self.logger = get_logger(__name__)
        self.news_history = []
        self.processed_articles = set()  # Track processed articles to avoid duplicates
    
    def analyze_news_article(self, article: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze a single news article
        
        Args:
            article: Dictionary containing article data
            
        Returns:
            Dictionary with analysis results
        """
        try:
            # Extract text content
            title = article.get('title', '')
            content = article.get('content', '') or article.get('full_text', '')
            full_text = f"{title} {content}"
            
            if not full_text.strip():
                return self._get_empty_analysis()
            
            # Check if already processed
            article_id = f"{title}_{article.get('url', '')}"
            if article_id in self.processed_articles:
                return self._get_empty_analysis()
            
            self.processed_articles.add(article_id)
            
            # Extract entities
            entities = self.analyzer.extract_entities(full_text)
            
            # Classify event type
            event_classification = self.analyzer.classify_event_type(full_text)
            
            # Extract key phrases
            key_phrases = self.analyzer.extract_key_phrases(full_text)
            
            # Summarize
            summary = self.analyzer.summarize_text(full_text)
            
            # Calculate market impact
            market_impact = self.analyzer.calculate_market_impact_score(full_text, entities)
            
            # Determine primary event type
            primary_event = max(event_classification.items(), key=lambda x: x[1])[0] if event_classification else 'general'
            
            analysis_result = {
                'article_id': article_id,
                'title': title,
                'url': article.get('url', ''),
                'source': article.get('source', ''),
                'published_at': article.get('published_at'),
                'timestamp': datetime.now(),
                'entities': entities,
                'event_classification': event_classification,
                'primary_event_type': primary_event,
                'key_phrases': key_phrases,
                'summary': summary,
                'market_impact_score': market_impact,
                'relevance_score': self._calculate_relevance_score(full_text, entities),
                'urgency_score': self._calculate_urgency_score(article, full_text),
                'full_text_length': len(full_text)
            }
            
            return analysis_result
            
        except Exception as e:
            self.logger.error(f"Error analyzing news article: {e}")
            return self._get_empty_analysis()
    
    def analyze_news_batch(self, news_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze a batch of news articles
        
        Args:
            news_df: DataFrame containing news articles
            
        Returns:
            Dictionary with batch analysis results
        """
        try:
            if news_df.empty:
                return self._get_empty_batch_analysis()
            
            analyzed_articles = []
            
            for _, row in news_df.iterrows():
                article_analysis = self.analyze_news_article(row.to_dict())
                if article_analysis.get('market_impact_score', 0) > 0:
                    analyzed_articles.append(article_analysis)
            
            if not analyzed_articles:
                return self._get_empty_batch_analysis()
            
            # Aggregate analysis
            batch_analysis = self._aggregate_news_analysis(analyzed_articles)
            
            # Store in history
            self.news_history.append(batch_analysis)
            
            # Keep only last 500 entries
            if len(self.news_history) > 500:
                self.news_history = self.news_history[-500:]
            
            return batch_analysis
            
        except Exception as e:
            self.logger.error(f"Error analyzing news batch: {e}")
            return self._get_empty_batch_analysis()
    
    def _calculate_relevance_score(self, text: str, entities: Dict) -> float:
        """Calculate relevance score for financial markets"""
        relevance_score = 0.0
        
        # Market-specific keywords
        market_keywords = [
            'nifty', 'sensex', 'bank nifty', 'stock market', 'share market',
            'bse', 'nse', 'trading', 'investment', 'portfolio', 'equity'
        ]
        
        text_lower = text.lower()
        for keyword in market_keywords:
            if keyword in text_lower:
                relevance_score += 0.15
        
        # Financial organizations
        financial_orgs = ['rbi', 'sebi', 'hdfc', 'icici', 'sbi', 'axis', 'kotak']
        org_entities = [org.lower() for org in entities.get('organizations', [])]
        for org in financial_orgs:
            if any(org in entity for entity in org_entities):
                relevance_score += 0.1
        
        # Economic indicators
        economic_terms = [
            'gdp', 'inflation', 'cpi', 'wpi', 'fiscal deficit', 'current account',
            'foreign investment', 'fii', 'dii', 'mutual fund'
        ]
        for term in economic_terms:
            if term in text_lower:
                relevance_score += 0.1
        
        return min(relevance_score, 1.0)
    
    def _calculate_urgency_score(self, article: Dict, text: str) -> float:
        """Calculate urgency score based on timing and content"""
        urgency_score = 0.0
        
        # Check publication time
        published_at = article.get('published_at')
        if published_at:
            try:
                if isinstance(published_at, str):
                    published_at = pd.to_datetime(published_at)
                
                time_diff = datetime.now() - published_at.replace(tzinfo=None)
                hours_old = time_diff.total_seconds() / 3600
                
                # More recent = higher urgency
                if hours_old < 1:
                    urgency_score += 0.5
                elif hours_old < 6:
                    urgency_score += 0.3
                elif hours_old < 24:
                    urgency_score += 0.1
            except:
                pass
        
        # Check for urgent keywords
        urgent_keywords = [
            'breaking', 'urgent', 'alert', 'immediate', 'emergency',
            'just in', 'developing', 'live', 'update'
        ]
        
        text_lower = text.lower()
        for keyword in urgent_keywords:
            if keyword in text_lower:
                urgency_score += 0.2
        
        return min(urgency_score, 1.0)
    
    def _aggregate_news_analysis(self, analyzed_articles: List[Dict]) -> Dict[str, Any]:
        """Aggregate analysis from multiple articles"""
        if not analyzed_articles:
            return self._get_empty_batch_analysis()
        
        # Calculate weighted averages
        total_weight = sum(article.get('relevance_score', 0) for article in analyzed_articles)
        
        if total_weight == 0:
            return self._get_empty_batch_analysis()
        
        # Aggregate event types
        event_aggregation = {}
        for article in analyzed_articles:
            for event_type, score in article.get('event_classification', {}).items():
                if event_type not in event_aggregation:
                    event_aggregation[event_type] = 0
                event_aggregation[event_type] += score * article.get('relevance_score', 0)
        
        # Normalize event scores
        for event_type in event_aggregation:
            event_aggregation[event_type] /= total_weight
        
        # Get top entities
        all_entities = {'organizations': [], 'persons': [], 'locations': []}
        for article in analyzed_articles:
            entities = article.get('entities', {})
            for entity_type in all_entities:
                all_entities[entity_type].extend(entities.get(entity_type, []))
        
        # Count entity frequencies
        entity_counts = {}
        for entity_type, entity_list in all_entities.items():
            entity_counts[entity_type] = {}
            for entity in entity_list:
                entity_counts[entity_type][entity] = entity_counts[entity_type].get(entity, 0) + 1
        
        # Get top entities
        top_entities = {}
        for entity_type, counts in entity_counts.items():
            top_entities[entity_type] = sorted(counts.items(), key=lambda x: x[1], reverse=True)[:5]
        
        # Calculate overall scores
        avg_market_impact = np.mean([article.get('market_impact_score', 0) for article in analyzed_articles])
        avg_relevance = np.mean([article.get('relevance_score', 0) for article in analyzed_articles])
        avg_urgency = np.mean([article.get('urgency_score', 0) for article in analyzed_articles])
        
        # Determine dominant event type
        dominant_event = max(event_aggregation.items(), key=lambda x: x[1])[0] if event_aggregation else 'general'
        
        return {
            'timestamp': datetime.now(),
            'article_count': len(analyzed_articles),
            'event_aggregation': event_aggregation,
            'dominant_event_type': dominant_event,
            'top_entities': top_entities,
            'average_market_impact': avg_market_impact,
            'average_relevance': avg_relevance,
            'average_urgency': avg_urgency,
            'high_impact_articles': [
                article for article in analyzed_articles 
                if article.get('market_impact_score', 0) > 0.5
            ],
            'recent_articles': analyzed_articles[-5:],  # Last 5 articles
            'summary_insights': self._generate_summary_insights(analyzed_articles, event_aggregation)
        }
    
    def _generate_summary_insights(self, articles: List[Dict], event_aggregation: Dict) -> List[str]:
        """Generate summary insights from news analysis"""
        insights = []
        
        # Event-based insights
        top_events = sorted(event_aggregation.items(), key=lambda x: x[1], reverse=True)[:3]
        for event_type, score in top_events:
            if score > 0.3:
                insights.append(f"High activity in {event_type.replace('_', ' ')} events (score: {score:.2f})")
        
        # Impact-based insights
        high_impact_count = len([a for a in articles if a.get('market_impact_score', 0) > 0.5])
        if high_impact_count > 0:
            insights.append(f"{high_impact_count} high-impact articles detected")
        
        # Urgency-based insights
        urgent_count = len([a for a in articles if a.get('urgency_score', 0) > 0.5])
        if urgent_count > 0:
            insights.append(f"{urgent_count} urgent news items requiring attention")
        
        # Entity-based insights
        for article in articles:
            entities = article.get('entities', {})
            if 'rbi' in [org.lower() for org in entities.get('organizations', [])]:
                insights.append("RBI-related news detected - potential monetary policy impact")
                break
        
        return insights[:5]  # Return top 5 insights
    
    def get_news_signals(self, news_analysis: Dict) -> Dict[str, Any]:
        """
        Generate trading signals based on news analysis
        
        Args:
            news_analysis: News analysis results
            
        Returns:
            Dictionary with trading signals
        """
        try:
            signal_strength = 0.0
            signal_direction = 'neutral'
            confidence = 0.0
            
            # Get key metrics
            avg_market_impact = news_analysis.get('average_market_impact', 0)
            avg_urgency = news_analysis.get('average_urgency', 0)
            dominant_event = news_analysis.get('dominant_event_type', 'general')
            event_scores = news_analysis.get('event_aggregation', {})
            
            # Event-based signals
            if dominant_event == 'monetary_policy' and event_scores.get('monetary_policy', 0) > 0.5:
                if avg_market_impact > 0.6:
                    signal_strength = 0.8
                    signal_direction = 'bearish'  # Monetary policy changes often create uncertainty
                    confidence = 0.7
            
            elif dominant_event == 'earnings' and event_scores.get('earnings', 0) > 0.4:
                if avg_market_impact > 0.5:
                    signal_strength = 0.6
                    signal_direction = 'bullish'  # Good earnings generally positive
                    confidence = 0.6
            
            elif dominant_event == 'regulatory' and event_scores.get('regulatory', 0) > 0.5:
                signal_strength = 0.5
                signal_direction = 'bearish'  # Regulatory changes create uncertainty
                confidence = 0.5
            
            # Impact and urgency based adjustments
            if avg_market_impact > 0.7 and avg_urgency > 0.6:
                signal_strength = min(signal_strength * 1.5, 1.0)
                confidence = min(confidence * 1.3, 1.0)
            
            # High-impact article count
            high_impact_count = len(news_analysis.get('high_impact_articles', []))
            if high_impact_count > 2:
                signal_strength = min(signal_strength + 0.2, 1.0)
            
            return {
                'signal_direction': signal_direction,
                'signal_strength': signal_strength,
                'confidence': confidence,
                'reasoning': {
                    'dominant_event': dominant_event,
                    'average_market_impact': avg_market_impact,
                    'average_urgency': avg_urgency,
                    'high_impact_articles': high_impact_count,
                    'insights': news_analysis.get('summary_insights', [])
                },
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            self.logger.error(f"Error generating news signals: {e}")
            return {
                'signal_direction': 'neutral',
                'signal_strength': 0.0,
                'confidence': 0.0,
                'reasoning': {},
                'timestamp': datetime.now()
            }
    
    def _get_empty_analysis(self) -> Dict[str, Any]:
        """Return empty analysis result"""
        return {
            'market_impact_score': 0.0,
            'relevance_score': 0.0,
            'urgency_score': 0.0,
            'entities': {'organizations': [], 'persons': [], 'locations': []},
            'event_classification': {},
            'timestamp': datetime.now()
        }
    
    def _get_empty_batch_analysis(self) -> Dict[str, Any]:
        """Return empty batch analysis result"""
        return {
            'timestamp': datetime.now(),
            'article_count': 0,
            'event_aggregation': {},
            'dominant_event_type': 'general',
            'top_entities': {},
            'average_market_impact': 0.0,
            'average_relevance': 0.0,
            'average_urgency': 0.0,
            'high_impact_articles': [],
            'recent_articles': [],
            'summary_insights': []
        }

if __name__ == "__main__":
    # Test the news agent
    import asyncio
    
    async def test_news_agent():
        agent = NewsAgent()
        
        # Test with sample news data
        news_data = pd.DataFrame([
            {
                'title': 'RBI raises repo rate by 25 basis points',
                'content': 'The Reserve Bank of India announced a 25 basis point increase in the repo rate to combat rising inflation. This decision is expected to impact borrowing costs across the economy.',
                'url': 'https://example.com/rbi-rate-hike',
                'source': 'Economic Times',
                'published_at': datetime.now() - timedelta(hours=2),
                'timestamp': datetime.now()
            },
            {
                'title': 'Nifty hits record high on strong earnings',
                'content': 'The Nifty index reached a new all-time high today as investors cheered strong quarterly earnings from major companies. Banking and IT stocks led the rally.',
                'url': 'https://example.com/nifty-high',
                'source': 'Business Standard',
                'published_at': datetime.now() - timedelta(hours=1),
                'timestamp': datetime.now()
            }
        ])
        
        # Analyze news
        news_analysis = agent.analyze_news_batch(news_data)
        signals = agent.get_news_signals(news_analysis)
        
        print("News Analysis Results:")
        print(f"Article Count: {news_analysis['article_count']}")
        print(f"Dominant Event: {news_analysis['dominant_event_type']}")
        print(f"Average Market Impact: {news_analysis['average_market_impact']:.2f}")
        print(f"Summary Insights: {news_analysis['summary_insights']}")
        print(f"\nTrading Signal: {signals['signal_direction']} with strength {signals['signal_strength']:.2f}")
        print(f"Confidence: {signals['confidence']:.2f}")
    
    asyncio.run(test_news_agent())

