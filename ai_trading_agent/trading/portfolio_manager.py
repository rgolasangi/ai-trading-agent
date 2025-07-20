"""
Portfolio Management System for AI Trading Agent
Manages portfolio allocation, risk, and performance tracking
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
import asyncio
import json
import pickle

from utils.logger import get_logger
from config.config import Config
from .order_manager import OrderManager, Position, Order

logger = get_logger(__name__)

@dataclass
class Portfolio:
    """Portfolio data structure"""
    portfolio_id: str = ""
    name: str = ""
    initial_capital: float = 0.0
    current_value: float = 0.0
    cash_balance: float = 0.0
    invested_amount: float = 0.0
    total_pnl: float = 0.0
    daily_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    positions: Dict[str, Position] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert portfolio to dictionary"""
        return {
            'portfolio_id': self.portfolio_id,
            'name': self.name,
            'initial_capital': self.initial_capital,
            'current_value': self.current_value,
            'cash_balance': self.cash_balance,
            'invested_amount': self.invested_amount,
            'total_pnl': self.total_pnl,
            'daily_pnl': self.daily_pnl,
            'unrealized_pnl': self.unrealized_pnl,
            'realized_pnl': self.realized_pnl,
            'positions': {k: v.to_dict() for k, v in self.positions.items()},
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'metadata': self.metadata
        }

class RiskMetrics:
    """Risk metrics calculator"""
    
    @staticmethod
    def calculate_var(returns: np.ndarray, confidence_level: float = 0.95) -> float:
        """Calculate Value at Risk"""
        if len(returns) == 0:
            return 0.0
        return np.percentile(returns, (1 - confidence_level) * 100)
    
    @staticmethod
    def calculate_sharpe_ratio(returns: np.ndarray, risk_free_rate: float = 0.06) -> float:
        """Calculate Sharpe ratio"""
        if len(returns) == 0 or np.std(returns) == 0:
            return 0.0
        
        excess_returns = returns - risk_free_rate / 252  # Daily risk-free rate
        return np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)
    
    @staticmethod
    def calculate_max_drawdown(portfolio_values: np.ndarray) -> float:
        """Calculate maximum drawdown"""
        if len(portfolio_values) == 0:
            return 0.0
        
        peak = np.maximum.accumulate(portfolio_values)
        drawdown = (portfolio_values - peak) / peak
        return np.min(drawdown)
    
    @staticmethod
    def calculate_beta(portfolio_returns: np.ndarray, market_returns: np.ndarray) -> float:
        """Calculate portfolio beta"""
        if len(portfolio_returns) == 0 or len(market_returns) == 0:
            return 0.0
        
        covariance = np.cov(portfolio_returns, market_returns)[0, 1]
        market_variance = np.var(market_returns)
        
        return covariance / market_variance if market_variance != 0 else 0.0

class PortfolioManager:
    """Portfolio Management System"""
    
    def __init__(self, order_manager: OrderManager, initial_capital: float = 100000):
        self.order_manager = order_manager
        self.logger = get_logger(__name__)
        
        # Portfolio setup
        self.portfolio = Portfolio(
            portfolio_id="main_portfolio",
            name="AI Trading Portfolio",
            initial_capital=initial_capital,
            current_value=initial_capital,
            cash_balance=initial_capital
        )
        
        # Risk management parameters
        self.max_position_size = 0.1  # 10% of portfolio per position
        self.max_sector_exposure = 0.3  # 30% per sector
        self.max_daily_loss = 0.05  # 5% daily loss limit
        self.max_drawdown = 0.15  # 15% maximum drawdown
        self.stop_loss_percentage = 0.02  # 2% stop loss
        self.take_profit_percentage = 0.05  # 5% take profit
        
        # Performance tracking
        self.daily_returns = []
        self.portfolio_history = []
        self.trade_history = []
        
        # Greeks exposure limits
        self.max_delta_exposure = 0.5
        self.max_gamma_exposure = 0.1
        self.max_vega_exposure = 100
        self.max_theta_decay = 500  # Daily theta decay limit
        
        # Start monitoring
        asyncio.create_task(self._monitor_portfolio())
    
    async def update_portfolio(self):
        """Update portfolio with latest positions and prices"""
        try:
            # Get latest positions from order manager
            positions = self.order_manager.get_all_positions()
            
            # Update portfolio positions
            self.portfolio.positions = positions
            
            # Calculate portfolio metrics
            total_position_value = 0
            total_unrealized_pnl = 0
            total_realized_pnl = 0
            
            for position in positions.values():
                position_value = abs(position.quantity) * position.last_price
                total_position_value += position_value
                total_unrealized_pnl += position.unrealized_pnl
                total_realized_pnl += position.realized_pnl
            
            # Update portfolio values
            self.portfolio.invested_amount = total_position_value
            self.portfolio.unrealized_pnl = total_unrealized_pnl
            self.portfolio.realized_pnl = total_realized_pnl
            self.portfolio.total_pnl = total_unrealized_pnl + total_realized_pnl
            self.portfolio.current_value = self.portfolio.cash_balance + total_position_value + total_unrealized_pnl
            self.portfolio.daily_pnl = self.order_manager.get_daily_pnl()
            self.portfolio.updated_at = datetime.now()
            
            # Store portfolio snapshot
            self.portfolio_history.append({
                'timestamp': datetime.now(),
                'portfolio_value': self.portfolio.current_value,
                'cash_balance': self.portfolio.cash_balance,
                'invested_amount': self.portfolio.invested_amount,
                'total_pnl': self.portfolio.total_pnl,
                'daily_pnl': self.portfolio.daily_pnl,
                'position_count': len(positions)
            })
            
            # Keep only last 1000 snapshots
            if len(self.portfolio_history) > 1000:
                self.portfolio_history = self.portfolio_history[-1000:]
            
        except Exception as e:
            self.logger.error(f"Error updating portfolio: {e}")
    
    def calculate_position_size(self, symbol: str, entry_price: float, 
                              risk_percentage: float = None) -> int:
        """
        Calculate optimal position size based on risk management
        
        Args:
            symbol: Trading symbol
            entry_price: Entry price
            risk_percentage: Risk percentage (default: max_position_size)
            
        Returns:
            Recommended position size
        """
        try:
            risk_pct = risk_percentage or self.max_position_size
            
            # Calculate maximum position value
            max_position_value = self.portfolio.current_value * risk_pct
            
            # Calculate position size
            position_size = int(max_position_value / entry_price)
            
            # Ensure minimum lot size (assuming 50 for options)
            lot_size = 50
            position_size = (position_size // lot_size) * lot_size
            
            return max(position_size, lot_size)
            
        except Exception as e:
            self.logger.error(f"Error calculating position size: {e}")
            return 50  # Default lot size
    
    def check_risk_limits(self, symbol: str, quantity: int, price: float,
                         transaction_type: str) -> Tuple[bool, str]:
        """
        Check if a trade violates risk limits
        
        Args:
            symbol: Trading symbol
            quantity: Trade quantity
            price: Trade price
            transaction_type: BUY or SELL
            
        Returns:
            Tuple of (is_allowed, reason)
        """
        try:
            # Check daily loss limit
            if self.portfolio.daily_pnl <= -self.portfolio.current_value * self.max_daily_loss:
                return False, "Daily loss limit exceeded"
            
            # Check maximum drawdown
            if len(self.portfolio_history) > 0:
                portfolio_values = [h['portfolio_value'] for h in self.portfolio_history]
                max_dd = RiskMetrics.calculate_max_drawdown(np.array(portfolio_values))
                if max_dd <= -self.max_drawdown:
                    return False, "Maximum drawdown limit exceeded"
            
            # Check position size limit
            trade_value = quantity * price
            if trade_value > self.portfolio.current_value * self.max_position_size:
                return False, "Position size limit exceeded"
            
            # Check cash availability for buy orders
            if transaction_type == "BUY" and trade_value > self.portfolio.cash_balance:
                return False, "Insufficient cash balance"
            
            # Check concentration risk
            current_position = self.portfolio.positions.get(symbol)
            if current_position:
                new_position_value = abs(current_position.quantity * current_position.average_price)
                if transaction_type == "BUY":
                    new_position_value += trade_value
                
                if new_position_value > self.portfolio.current_value * self.max_position_size:
                    return False, "Position concentration limit exceeded"
            
            return True, "Risk checks passed"
            
        except Exception as e:
            self.logger.error(f"Error checking risk limits: {e}")
            return False, f"Risk check error: {e}"
    
    def check_greeks_limits(self, portfolio_greeks: Dict[str, float]) -> Tuple[bool, str]:
        """
        Check if portfolio Greeks are within limits
        
        Args:
            portfolio_greeks: Portfolio Greeks values
            
        Returns:
            Tuple of (is_within_limits, reason)
        """
        try:
            delta = portfolio_greeks.get('total_delta', 0)
            gamma = portfolio_greeks.get('total_gamma', 0)
            vega = portfolio_greeks.get('total_vega', 0)
            theta = portfolio_greeks.get('total_theta', 0)
            
            # Check delta exposure
            if abs(delta) > self.max_delta_exposure:
                return False, f"Delta exposure limit exceeded: {delta:.3f}"
            
            # Check gamma exposure
            if gamma > self.max_gamma_exposure:
                return False, f"Gamma exposure limit exceeded: {gamma:.3f}"
            
            # Check vega exposure
            if abs(vega) > self.max_vega_exposure:
                return False, f"Vega exposure limit exceeded: {vega:.1f}"
            
            # Check theta decay
            if abs(theta) > self.max_theta_decay:
                return False, f"Theta decay limit exceeded: {theta:.1f}"
            
            return True, "Greeks limits check passed"
            
        except Exception as e:
            self.logger.error(f"Error checking Greeks limits: {e}")
            return False, f"Greeks check error: {e}"
    
    def get_portfolio_allocation(self) -> Dict[str, float]:
        """
        Get portfolio allocation by asset/sector
        
        Returns:
            Dictionary with allocation percentages
        """
        try:
            allocation = {}
            total_value = self.portfolio.current_value
            
            if total_value == 0:
                return allocation
            
            # Cash allocation
            allocation['cash'] = self.portfolio.cash_balance / total_value
            
            # Position allocations
            for symbol, position in self.portfolio.positions.items():
                position_value = abs(position.quantity) * position.last_price
                allocation[symbol] = position_value / total_value
            
            return allocation
            
        except Exception as e:
            self.logger.error(f"Error calculating portfolio allocation: {e}")
            return {}
    
    def calculate_performance_metrics(self) -> Dict[str, float]:
        """
        Calculate portfolio performance metrics
        
        Returns:
            Dictionary with performance metrics
        """
        try:
            if len(self.portfolio_history) < 2:
                return {}
            
            # Calculate returns
            portfolio_values = np.array([h['portfolio_value'] for h in self.portfolio_history])
            returns = np.diff(portfolio_values) / portfolio_values[:-1]
            
            # Calculate metrics
            metrics = {
                'total_return': (self.portfolio.current_value - self.portfolio.initial_capital) / self.portfolio.initial_capital,
                'daily_return': returns[-1] if len(returns) > 0 else 0,
                'avg_daily_return': np.mean(returns) if len(returns) > 0 else 0,
                'volatility': np.std(returns) * np.sqrt(252) if len(returns) > 0 else 0,
                'sharpe_ratio': RiskMetrics.calculate_sharpe_ratio(returns),
                'max_drawdown': RiskMetrics.calculate_max_drawdown(portfolio_values),
                'var_95': RiskMetrics.calculate_var(returns, 0.95),
                'var_99': RiskMetrics.calculate_var(returns, 0.99),
                'win_rate': self._calculate_win_rate(),
                'profit_factor': self._calculate_profit_factor(),
                'calmar_ratio': self._calculate_calmar_ratio(returns, portfolio_values)
            }
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error calculating performance metrics: {e}")
            return {}
    
    def _calculate_win_rate(self) -> float:
        """Calculate win rate from trade history"""
        if not self.trade_history:
            return 0.0
        
        winning_trades = sum(1 for trade in self.trade_history if trade.get('pnl', 0) > 0)
        return winning_trades / len(self.trade_history)
    
    def _calculate_profit_factor(self) -> float:
        """Calculate profit factor"""
        if not self.trade_history:
            return 0.0
        
        gross_profit = sum(trade.get('pnl', 0) for trade in self.trade_history if trade.get('pnl', 0) > 0)
        gross_loss = abs(sum(trade.get('pnl', 0) for trade in self.trade_history if trade.get('pnl', 0) < 0))
        
        return gross_profit / gross_loss if gross_loss > 0 else float('inf')
    
    def _calculate_calmar_ratio(self, returns: np.ndarray, portfolio_values: np.ndarray) -> float:
        """Calculate Calmar ratio"""
        if len(returns) == 0:
            return 0.0
        
        annual_return = np.mean(returns) * 252
        max_dd = abs(RiskMetrics.calculate_max_drawdown(portfolio_values))
        
        return annual_return / max_dd if max_dd > 0 else 0.0
    
    async def rebalance_portfolio(self, target_allocation: Dict[str, float]) -> List[str]:
        """
        Rebalance portfolio to target allocation
        
        Args:
            target_allocation: Target allocation percentages
            
        Returns:
            List of order IDs for rebalancing trades
        """
        try:
            current_allocation = self.get_portfolio_allocation()
            order_ids = []
            
            for symbol, target_pct in target_allocation.items():
                if symbol == 'cash':
                    continue
                
                current_pct = current_allocation.get(symbol, 0)
                difference = target_pct - current_pct
                
                # Only rebalance if difference is significant (>1%)
                if abs(difference) > 0.01:
                    target_value = target_pct * self.portfolio.current_value
                    current_position = self.portfolio.positions.get(symbol)
                    
                    if current_position:
                        current_value = abs(current_position.quantity) * current_position.last_price
                        value_difference = target_value - current_value
                        
                        # Calculate quantity to trade
                        quantity_to_trade = int(abs(value_difference) / current_position.last_price)
                        
                        if quantity_to_trade > 0:
                            transaction_type = "BUY" if value_difference > 0 else "SELL"
                            
                            # Place rebalancing order
                            order = await self.order_manager.place_order(
                                symbol=symbol,
                                quantity=quantity_to_trade,
                                order_type="MARKET",
                                transaction_type=transaction_type,
                                metadata={'action': 'rebalance'}
                            )
                            
                            if order:
                                order_ids.append(order.order_id)
            
            self.logger.info(f"Portfolio rebalancing: {len(order_ids)} orders placed")
            return order_ids
            
        except Exception as e:
            self.logger.error(f"Error rebalancing portfolio: {e}")
            return []
    
    async def _monitor_portfolio(self):
        """Monitor portfolio and trigger alerts"""
        while True:
            try:
                await self.update_portfolio()
                
                # Check for risk limit violations
                if self.portfolio.daily_pnl <= -self.portfolio.current_value * self.max_daily_loss:
                    self.logger.warning("Daily loss limit reached - consider closing positions")
                
                # Check drawdown
                if len(self.portfolio_history) > 10:
                    portfolio_values = [h['portfolio_value'] for h in self.portfolio_history[-10:]]
                    recent_dd = RiskMetrics.calculate_max_drawdown(np.array(portfolio_values))
                    
                    if recent_dd <= -self.max_drawdown * 0.8:  # 80% of max drawdown
                        self.logger.warning(f"Approaching maximum drawdown: {recent_dd:.2%}")
                
                await asyncio.sleep(30)  # Monitor every 30 seconds
                
            except Exception as e:
                self.logger.error(f"Error monitoring portfolio: {e}")
                await asyncio.sleep(60)
    
    def get_portfolio_summary(self) -> Dict[str, Any]:
        """Get comprehensive portfolio summary"""
        try:
            performance_metrics = self.calculate_performance_metrics()
            allocation = self.get_portfolio_allocation()
            
            return {
                'portfolio': self.portfolio.to_dict(),
                'performance_metrics': performance_metrics,
                'allocation': allocation,
                'position_count': len(self.portfolio.positions),
                'last_updated': self.portfolio.updated_at.isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error generating portfolio summary: {e}")
            return {}
    
    def save_portfolio_state(self, filepath: str):
        """Save portfolio state to file"""
        try:
            portfolio_data = {
                'portfolio': self.portfolio.to_dict(),
                'portfolio_history': self.portfolio_history,
                'trade_history': self.trade_history,
                'daily_returns': self.daily_returns
            }
            
            with open(filepath, 'w') as f:
                json.dump(portfolio_data, f, indent=2, default=str)
            
            self.logger.info(f"Portfolio state saved to {filepath}")
            
        except Exception as e:
            self.logger.error(f"Error saving portfolio state: {e}")
    
    def load_portfolio_state(self, filepath: str):
        """Load portfolio state from file"""
        try:
            with open(filepath, 'r') as f:
                portfolio_data = json.load(f)
            
            # Restore portfolio data
            self.portfolio_history = portfolio_data.get('portfolio_history', [])
            self.trade_history = portfolio_data.get('trade_history', [])
            self.daily_returns = portfolio_data.get('daily_returns', [])
            
            self.logger.info(f"Portfolio state loaded from {filepath}")
            
        except Exception as e:
            self.logger.error(f"Error loading portfolio state: {e}")

if __name__ == "__main__":
    # Test the portfolio manager
    import asyncio
    from .zerodha_client import ZerodhaClient
    
    async def test_portfolio_manager():
        # Create dummy clients
        zerodha_client = ZerodhaClient("dummy_key", "dummy_secret", "dummy_token")
        order_manager = OrderManager(zerodha_client)
        
        # Create portfolio manager
        portfolio_manager = PortfolioManager(order_manager, initial_capital=100000)
        
        # Test position size calculation
        position_size = portfolio_manager.calculate_position_size("NIFTY2312519500CE", 100.0)
        print(f"Recommended position size: {position_size}")
        
        # Test risk checks
        is_allowed, reason = portfolio_manager.check_risk_limits("TEST", 100, 1000, "BUY")
        print(f"Risk check: {is_allowed} - {reason}")
        
        # Test portfolio allocation
        allocation = portfolio_manager.get_portfolio_allocation()
        print(f"Portfolio allocation: {allocation}")
        
        # Test performance metrics
        metrics = portfolio_manager.calculate_performance_metrics()
        print(f"Performance metrics: {metrics}")
        
        # Test portfolio summary
        summary = portfolio_manager.get_portfolio_summary()
        print(f"Portfolio summary keys: {list(summary.keys())}")
    
    asyncio.run(test_portfolio_manager())

