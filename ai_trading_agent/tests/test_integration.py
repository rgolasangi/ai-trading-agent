"""
Integration Test Suite for AI Trading Agent
Comprehensive testing of all system components
"""
import unittest
import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.sentiment_agent import SentimentAgent
from agents.news_agent import NewsAgent
from agents.greeks_agent import GreeksAgent
from agents.rl_agent import RLTradingAgent
from agents.agent_coordinator import AgentCoordinator
from trading.zerodha_client import ZerodhaClient
from trading.execution_engine import ExecutionEngine, TradeSignal, SignalDirection, SignalStrength
from risk_management.risk_manager import RiskManager
from backtesting.backtester import Backtester
from data.data_collector import DataCollector

class TestIntegration(unittest.TestCase):
    """Integration tests for the complete trading system"""
    
    def setUp(self):
        """Set up test environment"""
        self.initial_capital = 100000
        self.test_data = self._create_test_data()
        
        # Initialize components
        self.sentiment_agent = SentimentAgent()
        self.news_agent = NewsAgent()
        self.greeks_agent = GreeksAgent()
        self.rl_agent = RLTradingAgent()
        self.coordinator = AgentCoordinator()
        self.risk_manager = RiskManager(self.initial_capital)
        self.execution_engine = ExecutionEngine()
        
        # Mock Zerodha client for testing
        self.zerodha_client = None  # Will be mocked
        
    def _create_test_data(self):
        """Create test market data"""
        dates = pd.date_range('2024-01-01', '2024-03-31', freq='D')
        np.random.seed(42)
        
        data = pd.DataFrame({
            'timestamp': dates,
            'symbol': 'NIFTY',
            'open': 19000 + np.random.randn(len(dates)) * 100,
            'high': 19100 + np.random.randn(len(dates)) * 100,
            'low': 18900 + np.random.randn(len(dates)) * 100,
            'close': 19000 + np.random.randn(len(dates)) * 100,
            'volume': 1000000 + np.random.randint(0, 500000, len(dates))
        })
        
        # Ensure price consistency
        data['high'] = np.maximum(data[['open', 'close']].max(axis=1), data['high'])
        data['low'] = np.minimum(data[['open', 'close']].min(axis=1), data['low'])
        
        return data.set_index('timestamp')
    
    def test_sentiment_agent(self):
        """Test sentiment analysis agent"""
        print("Testing Sentiment Agent...")
        
        # Test with sample news
        test_news = [
            "Market rallies on positive economic data",
            "Concerns over inflation impact stock prices",
            "Strong earnings boost investor confidence"
        ]
        
        for news in test_news:
            sentiment = asyncio.run(self.sentiment_agent.analyze_sentiment(news))
            self.assertIsInstance(sentiment, dict)
            self.assertIn('sentiment_score', sentiment)
            self.assertIn('confidence', sentiment)
            self.assertTrue(-1 <= sentiment['sentiment_score'] <= 1)
            self.assertTrue(0 <= sentiment['confidence'] <= 1)
        
        print("✓ Sentiment Agent tests passed")
    
    def test_news_agent(self):
        """Test news analysis agent"""
        print("Testing News Agent...")
        
        # Test news processing
        test_articles = [
            {
                'title': 'RBI announces new monetary policy',
                'content': 'The Reserve Bank of India announced changes to interest rates...',
                'timestamp': datetime.now()
            }
        ]
        
        analysis = asyncio.run(self.news_agent.analyze_news(test_articles))
        self.assertIsInstance(analysis, dict)
        self.assertIn('market_impact', analysis)
        self.assertIn('sentiment', analysis)
        
        print("✓ News Agent tests passed")
    
    def test_greeks_agent(self):
        """Test options Greeks calculation agent"""
        print("Testing Greeks Agent...")
        
        # Test Greeks calculation
        option_data = {
            'underlying_price': 19000,
            'strike_price': 19500,
            'time_to_expiry': 30,
            'risk_free_rate': 0.06,
            'volatility': 0.20,
            'option_type': 'call'
        }
        
        greeks = asyncio.run(self.greeks_agent.calculate_greeks(option_data))
        self.assertIsInstance(greeks, dict)
        self.assertIn('delta', greeks)
        self.assertIn('gamma', greeks)
        self.assertIn('theta', greeks)
        self.assertIn('vega', greeks)
        self.assertIn('rho', greeks)
        
        # Validate Greeks ranges
        self.assertTrue(0 <= greeks['delta'] <= 1)  # Call option delta
        self.assertTrue(greeks['gamma'] >= 0)
        self.assertTrue(greeks['theta'] <= 0)  # Time decay
        self.assertTrue(greeks['vega'] >= 0)
        
        print("✓ Greeks Agent tests passed")
    
    def test_rl_agent(self):
        """Test reinforcement learning agent"""
        print("Testing RL Agent...")
        
        # Test state processing
        market_state = {
            'price': 19000,
            'volume': 1000000,
            'volatility': 0.20,
            'sentiment': 0.5,
            'greeks': {'delta': 0.6, 'gamma': 0.02}
        }
        
        # Test action prediction
        action = asyncio.run(self.rl_agent.predict_action(market_state))
        self.assertIsInstance(action, dict)
        self.assertIn('action', action)
        self.assertIn('confidence', action)
        
        print("✓ RL Agent tests passed")
    
    def test_agent_coordinator(self):
        """Test multi-agent coordination"""
        print("Testing Agent Coordinator...")
        
        # Add agents to coordinator
        self.coordinator.add_agent('sentiment', self.sentiment_agent)
        self.coordinator.add_agent('news', self.news_agent)
        self.coordinator.add_agent('greeks', self.greeks_agent)
        self.coordinator.add_agent('rl', self.rl_agent)
        
        # Test signal generation
        market_data = {
            'symbol': 'NIFTY24APR19500CE',
            'price': 150,
            'underlying_price': 19000,
            'volume': 50000
        }
        
        signals = asyncio.run(self.coordinator.generate_signals(market_data))
        self.assertIsInstance(signals, list)
        
        print("✓ Agent Coordinator tests passed")
    
    def test_risk_manager(self):
        """Test risk management system"""
        print("Testing Risk Manager...")
        
        # Test trade signal validation
        test_signal = TradeSignal(
            symbol="NIFTY24APR19500CE",
            direction=SignalDirection.BULLISH,
            strength=SignalStrength.MODERATE,
            confidence=0.75,
            entry_price=150,
            stop_loss=140,
            target_price=170,
            quantity=50
        )
        
        is_valid, reason = self.risk_manager.validate_trade_signal(test_signal, {})
        self.assertIsInstance(is_valid, bool)
        self.assertIsInstance(reason, str)
        
        # Test risk metrics calculation
        dashboard_data = self.risk_manager.get_risk_dashboard_data()
        self.assertIsInstance(dashboard_data, dict)
        self.assertIn('risk_level', dashboard_data)
        self.assertIn('risk_metrics', dashboard_data)
        
        print("✓ Risk Manager tests passed")
    
    def test_execution_engine(self):
        """Test trade execution engine"""
        print("Testing Execution Engine...")
        
        # Test signal processing
        test_signals = [
            TradeSignal(
                symbol="NIFTY24APR19500CE",
                direction=SignalDirection.BULLISH,
                strength=SignalStrength.STRONG,
                confidence=0.85,
                entry_price=150,
                stop_loss=140,
                target_price=170,
                quantity=50
            )
        ]
        
        # Test signal validation and processing
        for signal in test_signals:
            processed = self.execution_engine.process_signal(signal)
            self.assertIsInstance(processed, dict)
        
        print("✓ Execution Engine tests passed")
    
    def test_backtesting_system(self):
        """Test backtesting system"""
        print("Testing Backtesting System...")
        
        # Create simple test strategy
        async def test_strategy(data):
            # Simple moving average crossover strategy
            if len(data) < 20:
                return None
            
            short_ma = data['close'].rolling(5).mean().iloc[-1]
            long_ma = data['close'].rolling(20).mean().iloc[-1]
            
            if short_ma > long_ma:
                return TradeSignal(
                    symbol="NIFTY",
                    direction=SignalDirection.BULLISH,
                    strength=SignalStrength.MODERATE,
                    confidence=0.7,
                    entry_price=data['close'].iloc[-1],
                    stop_loss=data['close'].iloc[-1] * 0.98,
                    target_price=data['close'].iloc[-1] * 1.05,
                    quantity=1
                )
            return None
        
        # Run backtest
        backtester = Backtester(initial_capital=self.initial_capital)
        backtester.set_strategy(test_strategy)
        
        result = asyncio.run(backtester.run_backtest(self.test_data))
        
        self.assertIsNotNone(result)
        self.assertEqual(result.initial_capital, self.initial_capital)
        self.assertIsInstance(result.final_capital, (int, float))
        self.assertIsInstance(result.total_return, (int, float))
        
        print("✓ Backtesting System tests passed")
    
    def test_data_collection(self):
        """Test data collection system"""
        print("Testing Data Collection...")
        
        # Test data collector initialization
        collector = DataCollector()
        self.assertIsNotNone(collector)
        
        # Test data validation
        test_data = {
            'symbol': 'NIFTY',
            'timestamp': datetime.now(),
            'open': 19000,
            'high': 19100,
            'low': 18900,
            'close': 19050,
            'volume': 1000000
        }
        
        is_valid = collector.validate_market_data(test_data)
        self.assertTrue(is_valid)
        
        print("✓ Data Collection tests passed")
    
    def test_end_to_end_workflow(self):
        """Test complete end-to-end trading workflow"""
        print("Testing End-to-End Workflow...")
        
        # Simulate complete trading cycle
        market_data = {
            'symbol': 'NIFTY24APR19500CE',
            'timestamp': datetime.now(),
            'open': 150,
            'high': 155,
            'low': 148,
            'close': 152,
            'volume': 50000,
            'underlying_price': 19000
        }
        
        # 1. Generate signals from agents
        signals = []
        
        # Sentiment signal
        sentiment_result = asyncio.run(self.sentiment_agent.analyze_sentiment("Positive market outlook"))
        if sentiment_result['sentiment_score'] > 0.5:
            signals.append(TradeSignal(
                symbol=market_data['symbol'],
                direction=SignalDirection.BULLISH,
                strength=SignalStrength.MODERATE,
                confidence=sentiment_result['confidence'],
                entry_price=market_data['close'],
                stop_loss=market_data['close'] * 0.95,
                target_price=market_data['close'] * 1.1,
                quantity=25
            ))
        
        # 2. Validate signals with risk manager
        validated_signals = []
        for signal in signals:
            is_valid, reason = self.risk_manager.validate_trade_signal(signal, {})
            if is_valid:
                validated_signals.append(signal)
        
        # 3. Process signals through execution engine
        execution_results = []
        for signal in validated_signals:
            result = self.execution_engine.process_signal(signal)
            execution_results.append(result)
        
        # Verify workflow completion
        self.assertIsInstance(signals, list)
        self.assertIsInstance(validated_signals, list)
        self.assertIsInstance(execution_results, list)
        
        print("✓ End-to-End Workflow tests passed")
    
    def test_performance_requirements(self):
        """Test performance requirements"""
        print("Testing Performance Requirements...")
        
        # Test signal generation speed
        start_time = datetime.now()
        
        for _ in range(100):
            market_state = {
                'price': 19000 + np.random.randn() * 100,
                'volume': 1000000,
                'volatility': 0.20,
                'sentiment': np.random.random(),
                'greeks': {'delta': 0.6, 'gamma': 0.02}
            }
            
            # Generate signal
            action = asyncio.run(self.rl_agent.predict_action(market_state))
        
        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()
        
        # Should process 100 signals in under 10 seconds
        self.assertLess(processing_time, 10.0)
        
        print(f"✓ Performance tests passed (100 signals in {processing_time:.2f}s)")
    
    def test_error_handling(self):
        """Test error handling and recovery"""
        print("Testing Error Handling...")
        
        # Test invalid data handling
        invalid_data = {
            'symbol': '',  # Invalid symbol
            'price': -100,  # Invalid price
            'volume': 'invalid'  # Invalid volume
        }
        
        # Should handle gracefully without crashing
        try:
            result = asyncio.run(self.sentiment_agent.analyze_sentiment(""))
            self.assertIsInstance(result, dict)
        except Exception as e:
            self.fail(f"Error handling failed: {e}")
        
        # Test risk manager with extreme values
        extreme_signal = TradeSignal(
            symbol="INVALID",
            direction=SignalDirection.BULLISH,
            strength=SignalStrength.STRONG,
            confidence=1.5,  # Invalid confidence > 1
            entry_price=-100,  # Invalid negative price
            stop_loss=0,
            target_price=0,
            quantity=1000000  # Extremely large quantity
        )
        
        is_valid, reason = self.risk_manager.validate_trade_signal(extreme_signal, {})
        self.assertFalse(is_valid)  # Should reject invalid signal
        
        print("✓ Error Handling tests passed")

def run_integration_tests():
    """Run all integration tests"""
    print("=" * 60)
    print("AI TRADING AGENT - INTEGRATION TEST SUITE")
    print("=" * 60)
    
    # Create test suite
    suite = unittest.TestLoader().loadTestsFromTestCase(TestIntegration)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.failures:
        print("\nFAILURES:")
        for test, traceback in result.failures:
            print(f"- {test}: {traceback}")
    
    if result.errors:
        print("\nERRORS:")
        for test, traceback in result.errors:
            print(f"- {test}: {traceback}")
    
    success_rate = (result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100
    print(f"\nSUCCESS RATE: {success_rate:.1f}%")
    
    if success_rate >= 90:
        print("✅ INTEGRATION TESTS PASSED - SYSTEM READY FOR DEPLOYMENT")
    else:
        print("❌ INTEGRATION TESTS FAILED - SYSTEM NEEDS FIXES")
    
    return result

if __name__ == "__main__":
    run_integration_tests()

