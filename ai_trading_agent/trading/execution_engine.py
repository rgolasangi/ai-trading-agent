"""
Trade Execution Engine for AI Trading Agent
Coordinates between AI agents and trading systems
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import asyncio
import json
import uuid

from utils.logger import get_logger
from config.config import Config
from .zerodha_client import ZerodhaClient
from .order_manager import OrderManager, OrderType, TransactionType, ProductType
from .portfolio_manager import PortfolioManager
from agents.agent_coordinator import AgentCoordinator

logger = get_logger(__name__)

class SignalStrength(Enum):
    """Signal strength levels"""
    WEAK = "WEAK"
    MODERATE = "MODERATE"
    STRONG = "STRONG"
    VERY_STRONG = "VERY_STRONG"

class SignalDirection(Enum):
    """Signal direction"""
    BULLISH = "BULLISH"
    BEARISH = "BEARISH"
    NEUTRAL = "NEUTRAL"

@dataclass
class TradeSignal:
    """Trade signal data structure"""
    signal_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    symbol: str = ""
    direction: SignalDirection = SignalDirection.NEUTRAL
    strength: SignalStrength = SignalStrength.WEAK
    confidence: float = 0.0
    entry_price: float = 0.0
    stop_loss: float = 0.0
    target_price: float = 0.0
    quantity: int = 0
    strategy_name: str = ""
    source_agents: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    expires_at: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert signal to dictionary"""
        return {
            'signal_id': self.signal_id,
            'symbol': self.symbol,
            'direction': self.direction.value,
            'strength': self.strength.value,
            'confidence': self.confidence,
            'entry_price': self.entry_price,
            'stop_loss': self.stop_loss,
            'target_price': self.target_price,
            'quantity': self.quantity,
            'strategy_name': self.strategy_name,
            'source_agents': self.source_agents,
            'metadata': self.metadata,
            'created_at': self.created_at.isoformat(),
            'expires_at': self.expires_at.isoformat() if self.expires_at else None
        }

class ExecutionEngine:
    """Trade Execution Engine"""
    
    def __init__(self, zerodha_client: ZerodhaClient, order_manager: OrderManager,
                 portfolio_manager: PortfolioManager, agent_coordinator: AgentCoordinator):
        
        self.zerodha_client = zerodha_client
        self.order_manager = order_manager
        self.portfolio_manager = portfolio_manager
        self.agent_coordinator = agent_coordinator
        self.logger = get_logger(__name__)
        
        # Execution parameters
        self.min_confidence_threshold = 0.6  # Minimum confidence to execute
        self.max_position_size_pct = 0.1  # Maximum 10% of portfolio per position
        self.execution_delay = 1.0  # Delay between executions (seconds)
        
        # Signal tracking
        self.active_signals: Dict[str, TradeSignal] = {}
        self.executed_signals: List[TradeSignal] = []
        self.signal_history: List[Dict[str, Any]] = []
        
        # Strategy weights
        self.strategy_weights = {
            'sentiment_analysis': 0.3,
            'news_analysis': 0.3,
            'greeks_analysis': 0.2,
            'rl_agent': 0.2
        }
        
        # Performance tracking
        self.execution_stats = {
            'total_signals': 0,
            'executed_signals': 0,
            'successful_trades': 0,
            'failed_trades': 0,
            'total_pnl': 0.0
        }
        
        # Start execution loop
        asyncio.create_task(self._execution_loop())
        asyncio.create_task(self._signal_cleanup())
    
    async def process_market_data(self, market_data: Dict[str, Any]) -> Optional[TradeSignal]:
        """
        Process market data through all agents and generate trade signal
        
        Args:
            market_data: Current market data
            
        Returns:
            Generated trade signal if any
        """
        try:
            self.logger.info("Processing market data through agent coordinator...")
            
            # Get analysis from all agents
            analysis_results = await self.agent_coordinator.analyze_market_data(market_data)
            
            if not analysis_results:
                return None
            
            # Extract signals from different agents
            sentiment_signals = analysis_results.get('sentiment_analysis', {}).get('signals', {})
            news_signals = analysis_results.get('news_analysis', {}).get('signals', {})
            combined_signals = analysis_results.get('combined_signals', {})
            
            # Generate trade signal
            trade_signal = await self._generate_trade_signal(
                market_data, sentiment_signals, news_signals, combined_signals
            )
            
            if trade_signal:
                self.active_signals[trade_signal.signal_id] = trade_signal
                self.signal_history.append({
                    'timestamp': datetime.now(),
                    'signal': trade_signal.to_dict(),
                    'market_data': market_data
                })
                
                self.execution_stats['total_signals'] += 1
                self.logger.info(f"Trade signal generated: {trade_signal.signal_id}")
            
            return trade_signal
            
        except Exception as e:
            self.logger.error(f"Error processing market data: {e}")
            return None
    
    async def _generate_trade_signal(self, market_data: Dict[str, Any],
                                   sentiment_signals: Dict[str, Any],
                                   news_signals: Dict[str, Any],
                                   combined_signals: Dict[str, Any]) -> Optional[TradeSignal]:
        """Generate trade signal from agent outputs"""
        try:
            # Extract signal components
            sentiment_direction = sentiment_signals.get('signal_direction', 'neutral')
            sentiment_strength = sentiment_signals.get('signal_strength', 0)
            sentiment_confidence = sentiment_signals.get('confidence', 0)
            
            news_direction = news_signals.get('signal_direction', 'neutral')
            news_strength = news_signals.get('signal_strength', 0)
            news_confidence = news_signals.get('confidence', 0)
            
            combined_direction = combined_signals.get('signal_direction', 'neutral')
            combined_strength = combined_signals.get('signal_strength', 0)
            combined_confidence = combined_signals.get('confidence', 0)
            
            # Calculate weighted signal
            total_weight = sum(self.strategy_weights.values())
            
            weighted_strength = (
                sentiment_strength * self.strategy_weights['sentiment_analysis'] +
                news_strength * self.strategy_weights['news_analysis'] +
                combined_strength * (self.strategy_weights['greeks_analysis'] + self.strategy_weights['rl_agent'])
            ) / total_weight
            
            weighted_confidence = (
                sentiment_confidence * self.strategy_weights['sentiment_analysis'] +
                news_confidence * self.strategy_weights['news_analysis'] +
                combined_confidence * (self.strategy_weights['greeks_analysis'] + self.strategy_weights['rl_agent'])
            ) / total_weight
            
            # Determine overall direction
            direction_scores = {
                'bullish': 0,
                'bearish': 0,
                'neutral': 0
            }
            
            for direction, weight in [
                (sentiment_direction, self.strategy_weights['sentiment_analysis']),
                (news_direction, self.strategy_weights['news_analysis']),
                (combined_direction, self.strategy_weights['greeks_analysis'] + self.strategy_weights['rl_agent'])
            ]:
                if direction.lower() in direction_scores:
                    direction_scores[direction.lower()] += weight
            
            overall_direction = max(direction_scores.items(), key=lambda x: x[1])[0]
            
            # Check if signal meets minimum criteria
            if weighted_confidence < self.min_confidence_threshold or weighted_strength < 0.3:
                return None
            
            # Determine signal strength category
            if weighted_strength >= 0.8:
                strength = SignalStrength.VERY_STRONG
            elif weighted_strength >= 0.6:
                strength = SignalStrength.STRONG
            elif weighted_strength >= 0.4:
                strength = SignalStrength.MODERATE
            else:
                strength = SignalStrength.WEAK
            
            # Get symbol and current price
            symbol = market_data.get('symbol', 'NIFTY2312519500CE')  # Default for testing
            current_price = market_data.get('last_price', market_data.get('underlying_price', 19500))
            
            # Calculate entry price, stop loss, and target
            if overall_direction == 'bullish':
                signal_direction = SignalDirection.BULLISH
                entry_price = current_price * 1.001  # Slight premium for market orders
                stop_loss = current_price * 0.98  # 2% stop loss
                target_price = current_price * 1.05  # 5% target
            elif overall_direction == 'bearish':
                signal_direction = SignalDirection.BEARISH
                entry_price = current_price * 0.999  # Slight discount for market orders
                stop_loss = current_price * 1.02  # 2% stop loss
                target_price = current_price * 0.95  # 5% target
            else:
                return None  # No signal for neutral
            
            # Calculate position size
            quantity = self.portfolio_manager.calculate_position_size(symbol, entry_price)
            
            # Create trade signal
            trade_signal = TradeSignal(
                symbol=symbol,
                direction=signal_direction,
                strength=strength,
                confidence=weighted_confidence,
                entry_price=entry_price,
                stop_loss=stop_loss,
                target_price=target_price,
                quantity=quantity,
                strategy_name="multi_agent_strategy",
                source_agents=['sentiment_agent', 'news_agent', 'greeks_agent'],
                metadata={
                    'sentiment_signals': sentiment_signals,
                    'news_signals': news_signals,
                    'combined_signals': combined_signals,
                    'market_data': market_data
                },
                expires_at=datetime.now() + timedelta(minutes=30)  # Signal expires in 30 minutes
            )
            
            return trade_signal
            
        except Exception as e:
            self.logger.error(f"Error generating trade signal: {e}")
            return None
    
    async def execute_signal(self, signal: TradeSignal) -> bool:
        """
        Execute a trade signal
        
        Args:
            signal: Trade signal to execute
            
        Returns:
            True if execution successful
        """
        try:
            # Validate signal
            if not await self._validate_signal(signal):
                return False
            
            # Check risk limits
            transaction_type = "BUY" if signal.direction == SignalDirection.BULLISH else "SELL"
            is_allowed, reason = self.portfolio_manager.check_risk_limits(
                signal.symbol, signal.quantity, signal.entry_price, transaction_type
            )
            
            if not is_allowed:
                self.logger.warning(f"Signal execution blocked: {reason}")
                return False
            
            # Place main order
            main_order = await self.order_manager.place_order(
                symbol=signal.symbol,
                quantity=signal.quantity,
                order_type=OrderType.MARKET,
                transaction_type=TransactionType.BUY if signal.direction == SignalDirection.BULLISH else TransactionType.SELL,
                product_type=ProductType.MIS,
                metadata={
                    'signal_id': signal.signal_id,
                    'strategy': signal.strategy_name,
                    'source': 'execution_engine'
                }
            )
            
            if not main_order:
                self.logger.error(f"Failed to place main order for signal {signal.signal_id}")
                return False
            
            # Place stop loss order
            sl_transaction = TransactionType.SELL if signal.direction == SignalDirection.BULLISH else TransactionType.BUY
            stop_loss_order = await self.order_manager.place_order(
                symbol=signal.symbol,
                quantity=signal.quantity,
                order_type=OrderType.STOP_LOSS,
                transaction_type=sl_transaction,
                trigger_price=signal.stop_loss,
                product_type=ProductType.MIS,
                metadata={
                    'signal_id': signal.signal_id,
                    'parent_order': main_order.order_id,
                    'order_type': 'stop_loss'
                }
            )
            
            # Place target order
            target_order = await self.order_manager.place_order(
                symbol=signal.symbol,
                quantity=signal.quantity,
                order_type=OrderType.LIMIT,
                transaction_type=sl_transaction,
                price=signal.target_price,
                product_type=ProductType.MIS,
                metadata={
                    'signal_id': signal.signal_id,
                    'parent_order': main_order.order_id,
                    'order_type': 'target'
                }
            )
            
            # Update signal metadata
            signal.metadata.update({
                'main_order_id': main_order.order_id,
                'stop_loss_order_id': stop_loss_order.order_id if stop_loss_order else None,
                'target_order_id': target_order.order_id if target_order else None,
                'executed_at': datetime.now()
            })
            
            # Move to executed signals
            self.executed_signals.append(signal)
            if signal.signal_id in self.active_signals:
                del self.active_signals[signal.signal_id]
            
            self.execution_stats['executed_signals'] += 1
            self.logger.info(f"Signal executed successfully: {signal.signal_id}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error executing signal {signal.signal_id}: {e}")
            return False
    
    async def _validate_signal(self, signal: TradeSignal) -> bool:
        """Validate signal before execution"""
        try:
            # Check if signal has expired
            if signal.expires_at and datetime.now() > signal.expires_at:
                self.logger.warning(f"Signal expired: {signal.signal_id}")
                return False
            
            # Check if market is open
            if not self.zerodha_client.is_market_open():
                self.logger.warning("Market is closed")
                return False
            
            # Check minimum confidence
            if signal.confidence < self.min_confidence_threshold:
                self.logger.warning(f"Signal confidence too low: {signal.confidence}")
                return False
            
            # Check if we already have a position in this symbol
            current_position = self.portfolio_manager.get_portfolio().positions.get(signal.symbol)
            if current_position and abs(current_position.quantity) > 0:
                self.logger.warning(f"Already have position in {signal.symbol}")
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error validating signal: {e}")
            return False
    
    async def _execution_loop(self):
        """Main execution loop"""
        while True:
            try:
                # Process active signals
                signals_to_execute = []
                
                for signal in self.active_signals.values():
                    if await self._should_execute_signal(signal):
                        signals_to_execute.append(signal)
                
                # Execute signals
                for signal in signals_to_execute:
                    await self.execute_signal(signal)
                    await asyncio.sleep(self.execution_delay)
                
                await asyncio.sleep(5)  # Check every 5 seconds
                
            except Exception as e:
                self.logger.error(f"Error in execution loop: {e}")
                await asyncio.sleep(10)
    
    async def _should_execute_signal(self, signal: TradeSignal) -> bool:
        """Determine if a signal should be executed"""
        try:
            # Check basic validation
            if not await self._validate_signal(signal):
                return False
            
            # Check if signal strength is sufficient
            if signal.strength in [SignalStrength.WEAK]:
                return False
            
            # Check if confidence is high enough for immediate execution
            if signal.confidence >= 0.8:
                return True
            
            # For moderate confidence, wait for confirmation
            if signal.confidence >= 0.6:
                # Check if signal has been active for at least 2 minutes
                signal_age = (datetime.now() - signal.created_at).total_seconds()
                return signal_age >= 120
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error checking signal execution criteria: {e}")
            return False
    
    async def _signal_cleanup(self):
        """Clean up expired signals"""
        while True:
            try:
                current_time = datetime.now()
                expired_signals = []
                
                for signal_id, signal in self.active_signals.items():
                    if signal.expires_at and current_time > signal.expires_at:
                        expired_signals.append(signal_id)
                
                # Remove expired signals
                for signal_id in expired_signals:
                    del self.active_signals[signal_id]
                    self.logger.info(f"Removed expired signal: {signal_id}")
                
                await asyncio.sleep(60)  # Clean up every minute
                
            except Exception as e:
                self.logger.error(f"Error in signal cleanup: {e}")
                await asyncio.sleep(300)
    
    def get_execution_status(self) -> Dict[str, Any]:
        """Get execution engine status"""
        return {
            'active_signals': len(self.active_signals),
            'executed_signals': len(self.executed_signals),
            'execution_stats': self.execution_stats,
            'strategy_weights': self.strategy_weights,
            'min_confidence_threshold': self.min_confidence_threshold,
            'last_signal_time': max([s.created_at for s in self.active_signals.values()]) if self.active_signals else None
        }
    
    def update_strategy_weights(self, new_weights: Dict[str, float]):
        """Update strategy weights"""
        total_weight = sum(new_weights.values())
        if abs(total_weight - 1.0) > 0.01:
            self.logger.warning(f"Strategy weights don't sum to 1.0: {total_weight}")
            return False
        
        self.strategy_weights.update(new_weights)
        self.logger.info(f"Strategy weights updated: {self.strategy_weights}")
        return True
    
    def get_signal_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get signal history"""
        return self.signal_history[-limit:]
    
    async def emergency_stop(self):
        """Emergency stop - close all positions and cancel all orders"""
        try:
            self.logger.warning("Emergency stop initiated")
            
            # Cancel all pending orders
            for order in await self.order_manager.get_orders():
                if order.get('status') in ['OPEN', 'PENDING']:
                    await self.order_manager.cancel_order(order.get('order_id'))
            
            # Close all positions
            await self.order_manager.close_all_positions()
            
            # Clear active signals
            self.active_signals.clear()
            
            self.logger.info("Emergency stop completed")
            
        except Exception as e:
            self.logger.error(f"Error during emergency stop: {e}")

if __name__ == "__main__":
    # Test the execution engine
    import asyncio
    from .zerodha_client import ZerodhaClient
    from agents.agent_coordinator import AgentCoordinator
    
    async def test_execution_engine():
        # Create dummy clients
        zerodha_client = ZerodhaClient("dummy_key", "dummy_secret", "dummy_token")
        order_manager = OrderManager(zerodha_client)
        portfolio_manager = PortfolioManager(order_manager)
        agent_coordinator = AgentCoordinator()
        
        # Create execution engine
        execution_engine = ExecutionEngine(
            zerodha_client, order_manager, portfolio_manager, agent_coordinator
        )
        
        # Test signal generation
        market_data = {
            'symbol': 'NIFTY2312519500CE',
            'last_price': 150,
            'underlying_price': 19550,
            'volume': 5000,
            'sentiment_score': 0.3,
            'news_score': 0.2
        }
        
        try:
            signal = await execution_engine.process_market_data(market_data)
            print(f"Generated signal: {signal.signal_id if signal else 'None'}")
        except Exception as e:
            print(f"Signal generation test failed (expected): {e}")
        
        # Test execution status
        status = execution_engine.get_execution_status()
        print(f"Execution status: {status}")
        
        # Test strategy weight update
        new_weights = {
            'sentiment_analysis': 0.4,
            'news_analysis': 0.3,
            'greeks_analysis': 0.2,
            'rl_agent': 0.1
        }
        success = execution_engine.update_strategy_weights(new_weights)
        print(f"Weight update successful: {success}")
    
    asyncio.run(test_execution_engine())

