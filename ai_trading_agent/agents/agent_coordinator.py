"""
Agent Coordinator for AI Trading Agent
Coordinates communication between different agents
"""
import asyncio
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import pandas as pd
from dataclasses import dataclass, asdict
import uuid

from utils.logger import get_logger
from config.config import Config
from .sentiment_agent import SentimentAgent
from .news_agent import NewsAgent

logger = get_logger(__name__)

@dataclass
class AgentMessage:
    """Message structure for inter-agent communication"""
    id: str
    sender: str
    receiver: str
    message_type: str
    payload: Dict[str, Any]
    timestamp: datetime
    priority: int = 1  # 1=low, 2=medium, 3=high

class MessageQueue:
    """Simple message queue for agent communication"""
    
    def __init__(self):
        self.queue = asyncio.Queue()
        self.subscribers = {}
        self.message_history = []
    
    async def publish(self, message: AgentMessage):
        """Publish a message to the queue"""
        await self.queue.put(message)
        self.message_history.append(message)
        
        # Keep only last 1000 messages
        if len(self.message_history) > 1000:
            self.message_history = self.message_history[-1000:]
    
    async def subscribe(self, agent_name: str, callback):
        """Subscribe an agent to receive messages"""
        self.subscribers[agent_name] = callback
    
    async def process_messages(self):
        """Process messages in the queue"""
        while True:
            try:
                message = await self.queue.get()
                
                # Deliver to specific receiver or broadcast
                if message.receiver in self.subscribers:
                    await self.subscribers[message.receiver](message)
                elif message.receiver == "broadcast":
                    for subscriber_name, callback in self.subscribers.items():
                        if subscriber_name != message.sender:
                            await callback(message)
                
                self.queue.task_done()
                
            except Exception as e:
                logger.error(f"Error processing message: {e}")

class AgentCoordinator:
    """Coordinates all agents and their communications"""
    
    def __init__(self):
        self.sentiment_agent = SentimentAgent()
        self.news_agent = NewsAgent()
        self.message_queue = MessageQueue()
        self.logger = get_logger(__name__)
        
        # Agent states
        self.agent_states = {
            'sentiment_agent': {'status': 'idle', 'last_update': datetime.now()},
            'news_agent': {'status': 'idle', 'last_update': datetime.now()},
            'coordinator': {'status': 'active', 'last_update': datetime.now()}
        }
        
        # Analysis results cache
        self.analysis_cache = {
            'sentiment': None,
            'news': None,
            'combined': None,
            'last_update': None
        }
        
        # Initialize message processing
        asyncio.create_task(self.message_queue.process_messages())
        
        # Subscribe agents to message queue
        asyncio.create_task(self._setup_subscriptions())
    
    async def _setup_subscriptions(self):
        """Setup message subscriptions for agents"""
        await self.message_queue.subscribe('sentiment_agent', self._handle_sentiment_message)
        await self.message_queue.subscribe('news_agent', self._handle_news_message)
        await self.message_queue.subscribe('coordinator', self._handle_coordinator_message)
    
    async def _handle_sentiment_message(self, message: AgentMessage):
        """Handle messages for sentiment agent"""
        try:
            if message.message_type == 'analysis_request':
                # Process sentiment analysis request
                payload = message.payload
                news_data = payload.get('news_data')
                social_data = payload.get('social_data')
                
                if news_data is not None:
                    news_sentiment = self.sentiment_agent.analyze_news_sentiment(news_data)
                else:
                    news_sentiment = self.sentiment_agent._get_empty_sentiment_result()
                
                if social_data is not None:
                    social_sentiment = self.sentiment_agent.analyze_social_sentiment(social_data)
                else:
                    social_sentiment = self.sentiment_agent._get_empty_sentiment_result()
                
                overall_sentiment = self.sentiment_agent.get_overall_sentiment(news_sentiment, social_sentiment)
                signals = self.sentiment_agent.get_sentiment_signals(overall_sentiment)
                
                # Send results back
                response_message = AgentMessage(
                    id=str(uuid.uuid4()),
                    sender='sentiment_agent',
                    receiver='coordinator',
                    message_type='analysis_result',
                    payload={
                        'request_id': message.id,
                        'news_sentiment': news_sentiment,
                        'social_sentiment': social_sentiment,
                        'overall_sentiment': overall_sentiment,
                        'signals': signals
                    },
                    timestamp=datetime.now(),
                    priority=2
                )
                
                await self.message_queue.publish(response_message)
                
        except Exception as e:
            self.logger.error(f"Error handling sentiment message: {e}")
    
    async def _handle_news_message(self, message: AgentMessage):
        """Handle messages for news agent"""
        try:
            if message.message_type == 'analysis_request':
                # Process news analysis request
                payload = message.payload
                news_data = payload.get('news_data')
                
                if news_data is not None:
                    news_analysis = self.news_agent.analyze_news_batch(news_data)
                    signals = self.news_agent.get_news_signals(news_analysis)
                else:
                    news_analysis = self.news_agent._get_empty_batch_analysis()
                    signals = {
                        'signal_direction': 'neutral',
                        'signal_strength': 0.0,
                        'confidence': 0.0,
                        'reasoning': {},
                        'timestamp': datetime.now()
                    }
                
                # Send results back
                response_message = AgentMessage(
                    id=str(uuid.uuid4()),
                    sender='news_agent',
                    receiver='coordinator',
                    message_type='analysis_result',
                    payload={
                        'request_id': message.id,
                        'news_analysis': news_analysis,
                        'signals': signals
                    },
                    timestamp=datetime.now(),
                    priority=2
                )
                
                await self.message_queue.publish(response_message)
                
        except Exception as e:
            self.logger.error(f"Error handling news message: {e}")
    
    async def _handle_coordinator_message(self, message: AgentMessage):
        """Handle messages for coordinator"""
        try:
            if message.message_type == 'analysis_result':
                # Store analysis results
                sender = message.sender
                payload = message.payload
                
                if sender == 'sentiment_agent':
                    self.analysis_cache['sentiment'] = payload
                elif sender == 'news_agent':
                    self.analysis_cache['news'] = payload
                
                self.analysis_cache['last_update'] = datetime.now()
                
                # Update agent state
                self.agent_states[sender]['status'] = 'completed'
                self.agent_states[sender]['last_update'] = datetime.now()
                
                self.logger.info(f"Received analysis result from {sender}")
                
        except Exception as e:
            self.logger.error(f"Error handling coordinator message: {e}")
    
    async def analyze_market_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Coordinate analysis of market data across all agents
        
        Args:
            data: Dictionary containing market data
            
        Returns:
            Combined analysis results
        """
        try:
            self.logger.info("Starting coordinated market data analysis...")
            
            # Reset agent states
            for agent in self.agent_states:
                if agent != 'coordinator':
                    self.agent_states[agent]['status'] = 'processing'
                    self.agent_states[agent]['last_update'] = datetime.now()
            
            # Extract data for different agents
            news_data = data.get('news', pd.DataFrame())
            social_data = data.get('social_media', pd.DataFrame())
            
            # Create analysis requests
            requests = []
            
            # Sentiment analysis request
            sentiment_request = AgentMessage(
                id=str(uuid.uuid4()),
                sender='coordinator',
                receiver='sentiment_agent',
                message_type='analysis_request',
                payload={
                    'news_data': news_data,
                    'social_data': social_data
                },
                timestamp=datetime.now(),
                priority=2
            )
            requests.append(sentiment_request)
            
            # News analysis request
            news_request = AgentMessage(
                id=str(uuid.uuid4()),
                sender='coordinator',
                receiver='news_agent',
                message_type='analysis_request',
                payload={
                    'news_data': news_data
                },
                timestamp=datetime.now(),
                priority=2
            )
            requests.append(news_request)
            
            # Send requests
            for request in requests:
                await self.message_queue.publish(request)
            
            # Wait for all agents to complete (with timeout)
            timeout = 30  # seconds
            start_time = datetime.now()
            
            while (datetime.now() - start_time).total_seconds() < timeout:
                all_completed = all(
                    self.agent_states[agent]['status'] == 'completed'
                    for agent in ['sentiment_agent', 'news_agent']
                )
                
                if all_completed:
                    break
                
                await asyncio.sleep(0.5)
            
            # Combine results
            combined_analysis = self._combine_analysis_results()
            
            self.logger.info("Coordinated market data analysis completed")
            return combined_analysis
            
        except Exception as e:
            self.logger.error(f"Error in coordinated analysis: {e}")
            return self._get_empty_combined_analysis()
    
    def _combine_analysis_results(self) -> Dict[str, Any]:
        """Combine analysis results from all agents"""
        try:
            sentiment_results = self.analysis_cache.get('sentiment', {})
            news_results = self.analysis_cache.get('news', {})
            
            # Extract signals
            sentiment_signals = sentiment_results.get('signals', {})
            news_signals = news_results.get('signals', {})
            
            # Combine signals with weights
            sentiment_weight = 0.4
            news_weight = 0.6
            
            # Calculate combined signal strength
            sentiment_strength = sentiment_signals.get('signal_strength', 0)
            news_strength = news_signals.get('signal_strength', 0)
            
            combined_strength = (
                sentiment_strength * sentiment_weight +
                news_strength * news_weight
            )
            
            # Determine combined direction
            sentiment_direction = sentiment_signals.get('signal_direction', 'neutral')
            news_direction = news_signals.get('signal_direction', 'neutral')
            
            combined_direction = self._resolve_signal_direction(
                sentiment_direction, news_direction,
                sentiment_strength, news_strength
            )
            
            # Calculate combined confidence
            sentiment_confidence = sentiment_signals.get('confidence', 0)
            news_confidence = news_signals.get('confidence', 0)
            
            combined_confidence = (
                sentiment_confidence * sentiment_weight +
                news_confidence * news_weight
            )
            
            # Generate trading recommendation
            recommendation = self._generate_trading_recommendation(
                combined_direction, combined_strength, combined_confidence
            )
            
            combined_analysis = {
                'timestamp': datetime.now(),
                'sentiment_analysis': sentiment_results,
                'news_analysis': news_results,
                'combined_signals': {
                    'signal_direction': combined_direction,
                    'signal_strength': combined_strength,
                    'confidence': combined_confidence,
                    'recommendation': recommendation
                },
                'agent_states': self.agent_states.copy(),
                'analysis_summary': self._generate_analysis_summary(
                    sentiment_results, news_results, combined_direction, combined_strength
                )
            }
            
            # Cache combined results
            self.analysis_cache['combined'] = combined_analysis
            
            return combined_analysis
            
        except Exception as e:
            self.logger.error(f"Error combining analysis results: {e}")
            return self._get_empty_combined_analysis()
    
    def _resolve_signal_direction(self, sentiment_dir: str, news_dir: str, 
                                sentiment_strength: float, news_strength: float) -> str:
        """Resolve conflicting signal directions"""
        if sentiment_dir == news_dir:
            return sentiment_dir
        
        # If directions conflict, use the stronger signal
        if sentiment_strength > news_strength:
            return sentiment_dir
        elif news_strength > sentiment_strength:
            return news_dir
        else:
            return 'neutral'  # Equal strength but different directions
    
    def _generate_trading_recommendation(self, direction: str, strength: float, confidence: float) -> str:
        """Generate trading recommendation based on signals"""
        if direction == 'neutral' or strength < 0.3 or confidence < 0.4:
            return 'HOLD'
        
        if direction == 'bullish':
            if strength > 0.7 and confidence > 0.7:
                return 'STRONG_BUY'
            elif strength > 0.5 and confidence > 0.5:
                return 'BUY'
            else:
                return 'WEAK_BUY'
        
        elif direction == 'bearish':
            if strength > 0.7 and confidence > 0.7:
                return 'STRONG_SELL'
            elif strength > 0.5 and confidence > 0.5:
                return 'SELL'
            else:
                return 'WEAK_SELL'
        
        return 'HOLD'
    
    def _generate_analysis_summary(self, sentiment_results: Dict, news_results: Dict,
                                 direction: str, strength: float) -> List[str]:
        """Generate human-readable analysis summary"""
        summary = []
        
        # Sentiment summary
        sentiment_signals = sentiment_results.get('signals', {})
        if sentiment_signals.get('signal_strength', 0) > 0.3:
            summary.append(
                f"Sentiment analysis indicates {sentiment_signals.get('signal_direction', 'neutral')} "
                f"sentiment with {sentiment_signals.get('signal_strength', 0):.2f} strength"
            )
        
        # News summary
        news_signals = news_results.get('signals', {})
        news_analysis = news_results.get('news_analysis', {})
        if news_signals.get('signal_strength', 0) > 0.3:
            dominant_event = news_analysis.get('dominant_event_type', 'general')
            summary.append(
                f"News analysis shows {news_signals.get('signal_direction', 'neutral')} "
                f"bias driven by {dominant_event.replace('_', ' ')} events"
            )
        
        # Combined summary
        if strength > 0.5:
            summary.append(
                f"Combined analysis suggests {direction} market outlook "
                f"with {strength:.2f} signal strength"
            )
        
        # Insights from news
        insights = news_analysis.get('summary_insights', [])
        summary.extend(insights[:2])  # Add top 2 insights
        
        return summary[:5]  # Return top 5 summary points
    
    def _get_empty_combined_analysis(self) -> Dict[str, Any]:
        """Return empty combined analysis"""
        return {
            'timestamp': datetime.now(),
            'sentiment_analysis': {},
            'news_analysis': {},
            'combined_signals': {
                'signal_direction': 'neutral',
                'signal_strength': 0.0,
                'confidence': 0.0,
                'recommendation': 'HOLD'
            },
            'agent_states': self.agent_states.copy(),
            'analysis_summary': ['No significant signals detected']
        }
    
    def get_agent_status(self) -> Dict[str, Any]:
        """Get status of all agents"""
        return {
            'agent_states': self.agent_states.copy(),
            'message_queue_size': self.message_queue.queue.qsize(),
            'last_analysis': self.analysis_cache.get('last_update'),
            'cache_status': {
                'sentiment': self.analysis_cache['sentiment'] is not None,
                'news': self.analysis_cache['news'] is not None,
                'combined': self.analysis_cache['combined'] is not None
            }
        }

if __name__ == "__main__":
    # Test the agent coordinator
    import asyncio
    
    async def test_coordinator():
        coordinator = AgentCoordinator()
        
        # Test with sample data
        test_data = {
            'news': pd.DataFrame([
                {
                    'title': 'RBI raises repo rate',
                    'content': 'The Reserve Bank of India raised the repo rate by 25 basis points',
                    'timestamp': datetime.now(),
                    'source': 'test'
                }
            ]),
            'social_media': pd.DataFrame([
                {
                    'text': 'Bullish on #Nifty today! 🚀',
                    'timestamp': datetime.now(),
                    'like_count': 10,
                    'retweet_count': 5
                }
            ])
        }
        
        # Run coordinated analysis
        results = await coordinator.analyze_market_data(test_data)
        
        print("Coordinated Analysis Results:")
        print(f"Combined Signal: {results['combined_signals']['signal_direction']}")
        print(f"Signal Strength: {results['combined_signals']['signal_strength']:.2f}")
        print(f"Recommendation: {results['combined_signals']['recommendation']}")
        print(f"Summary: {results['analysis_summary']}")
        
        # Check agent status
        status = coordinator.get_agent_status()
        print(f"\nAgent Status: {status}")
    
    asyncio.run(test_coordinator())

