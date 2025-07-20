"""
Backtesting Engine for AI Trading Agent
Simulates trading strategies on historical data
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Callable
from dataclasses import dataclass, field
import asyncio
import json
import pickle
from concurrent.futures import ThreadPoolExecutor
import warnings
warnings.filterwarnings('ignore')

from utils.logger import get_logger
from config.config import Config
from trading.execution_engine import TradeSignal, SignalDirection, SignalStrength
from trading.order_manager import Order, Position, OrderType, TransactionType, ProductType, OrderStatus

logger = get_logger(__name__)

@dataclass
class BacktestTrade:
    """Backtest trade record"""
    trade_id: str = ""
    symbol: str = ""
    entry_time: datetime = field(default_factory=datetime.now)
    exit_time: Optional[datetime] = None
    entry_price: float = 0.0
    exit_price: float = 0.0
    quantity: int = 0
    direction: str = ""  # LONG or SHORT
    pnl: float = 0.0
    pnl_percentage: float = 0.0
    commission: float = 0.0
    slippage: float = 0.0
    max_favorable_excursion: float = 0.0
    max_adverse_excursion: float = 0.0
    hold_time: timedelta = field(default_factory=timedelta)
    exit_reason: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert trade to dictionary"""
        return {
            'trade_id': self.trade_id,
            'symbol': self.symbol,
            'entry_time': self.entry_time.isoformat(),
            'exit_time': self.exit_time.isoformat() if self.exit_time else None,
            'entry_price': self.entry_price,
            'exit_price': self.exit_price,
            'quantity': self.quantity,
            'direction': self.direction,
            'pnl': self.pnl,
            'pnl_percentage': self.pnl_percentage,
            'commission': self.commission,
            'slippage': self.slippage,
            'max_favorable_excursion': self.max_favorable_excursion,
            'max_adverse_excursion': self.max_adverse_excursion,
            'hold_time': str(self.hold_time),
            'exit_reason': self.exit_reason,
            'metadata': self.metadata
        }

@dataclass
class BacktestResult:
    """Backtest result summary"""
    start_date: datetime = field(default_factory=datetime.now)
    end_date: datetime = field(default_factory=datetime.now)
    initial_capital: float = 100000.0
    final_capital: float = 100000.0
    total_return: float = 0.0
    total_return_percentage: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_percentage: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    largest_win: float = 0.0
    largest_loss: float = 0.0
    avg_hold_time: timedelta = field(default_factory=timedelta)
    total_commission: float = 0.0
    total_slippage: float = 0.0
    trades: List[BacktestTrade] = field(default_factory=list)
    equity_curve: List[Dict[str, Any]] = field(default_factory=list)
    daily_returns: List[float] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary"""
        return {
            'start_date': self.start_date.isoformat(),
            'end_date': self.end_date.isoformat(),
            'initial_capital': self.initial_capital,
            'final_capital': self.final_capital,
            'total_return': self.total_return,
            'total_return_percentage': self.total_return_percentage,
            'max_drawdown': self.max_drawdown,
            'max_drawdown_percentage': self.max_drawdown_percentage,
            'sharpe_ratio': self.sharpe_ratio,
            'sortino_ratio': self.sortino_ratio,
            'calmar_ratio': self.calmar_ratio,
            'win_rate': self.win_rate,
            'profit_factor': self.profit_factor,
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'avg_win': self.avg_win,
            'avg_loss': self.avg_loss,
            'largest_win': self.largest_win,
            'largest_loss': self.largest_loss,
            'avg_hold_time': str(self.avg_hold_time),
            'total_commission': self.total_commission,
            'total_slippage': self.total_slippage,
            'trades': [trade.to_dict() for trade in self.trades],
            'equity_curve': self.equity_curve,
            'daily_returns': self.daily_returns,
            'metadata': self.metadata
        }

class Backtester:
    """Comprehensive Backtesting Engine"""
    
    def __init__(self, initial_capital: float = 100000.0):
        self.logger = get_logger(__name__)
        
        # Backtest parameters
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.cash_balance = initial_capital
        
        # Trading parameters
        self.commission_rate = 0.0003  # 0.03% commission
        self.slippage_rate = 0.0001  # 0.01% slippage
        self.margin_requirement = 0.2  # 20% margin for options
        
        # Position tracking
        self.positions: Dict[str, Position] = {}
        self.open_orders: Dict[str, Order] = {}
        self.trade_history: List[BacktestTrade] = []
        self.equity_curve: List[Dict[str, Any]] = []
        self.daily_returns: List[float] = []
        
        # Performance tracking
        self.peak_capital = initial_capital
        self.max_drawdown = 0.0
        self.current_drawdown = 0.0
        
        # Risk management
        self.max_position_size = 0.1  # 10% of capital per position
        self.max_daily_loss = 0.05  # 5% daily loss limit
        self.stop_loss_percentage = 0.02  # 2% stop loss
        self.take_profit_percentage = 0.05  # 5% take profit
        
        # Strategy function
        self.strategy_function: Optional[Callable] = None
        
    def set_strategy(self, strategy_function: Callable):
        """Set the trading strategy function"""
        self.strategy_function = strategy_function
    
    def set_commission(self, commission_rate: float):
        """Set commission rate"""
        self.commission_rate = commission_rate
    
    def set_slippage(self, slippage_rate: float):
        """Set slippage rate"""
        self.slippage_rate = slippage_rate
    
    def set_risk_parameters(self, max_position_size: float = None,
                           max_daily_loss: float = None,
                           stop_loss_percentage: float = None,
                           take_profit_percentage: float = None):
        """Set risk management parameters"""
        if max_position_size:
            self.max_position_size = max_position_size
        if max_daily_loss:
            self.max_daily_loss = max_daily_loss
        if stop_loss_percentage:
            self.stop_loss_percentage = stop_loss_percentage
        if take_profit_percentage:
            self.take_profit_percentage = take_profit_percentage
    
    async def run_backtest(self, data: pd.DataFrame, start_date: datetime = None,
                          end_date: datetime = None) -> BacktestResult:
        """
        Run backtest on historical data
        
        Args:
            data: Historical market data
            start_date: Backtest start date
            end_date: Backtest end date
            
        Returns:
            Backtest results
        """
        try:
            self.logger.info("Starting backtest...")
            
            # Reset state
            self._reset_state()
            
            # Filter data by date range
            if start_date:
                data = data[data.index >= start_date]
            if end_date:
                data = data[data.index <= end_date]
            
            if data.empty:
                raise ValueError("No data available for the specified date range")
            
            # Sort data by timestamp
            data = data.sort_index()
            
            # Run simulation
            for timestamp, row in data.iterrows():
                await self._process_bar(timestamp, row)
            
            # Close all open positions at the end
            await self._close_all_positions(data.iloc[-1])
            
            # Calculate final results
            result = self._calculate_results(data.index[0], data.index[-1])
            
            self.logger.info(f"Backtest completed. Total return: {result.total_return_percentage:.2f}%")
            return result
            
        except Exception as e:
            self.logger.error(f"Error running backtest: {e}")
            raise
    
    async def _process_bar(self, timestamp: datetime, market_data: pd.Series):
        """Process a single bar of market data"""
        try:
            # Update positions with current prices
            await self._update_positions(timestamp, market_data)
            
            # Check stop losses and take profits
            await self._check_exit_conditions(timestamp, market_data)
            
            # Generate trading signals using strategy
            if self.strategy_function:
                signals = await self._generate_signals(timestamp, market_data)
                
                # Execute signals
                for signal in signals:
                    await self._execute_signal(timestamp, signal, market_data)
            
            # Update equity curve
            self._update_equity_curve(timestamp)
            
            # Update drawdown
            self._update_drawdown()
            
        except Exception as e:
            self.logger.error(f"Error processing bar at {timestamp}: {e}")
    
    async def _generate_signals(self, timestamp: datetime, market_data: pd.Series) -> List[TradeSignal]:
        """Generate trading signals using the strategy function"""
        try:
            # Prepare market data for strategy
            strategy_data = {
                'timestamp': timestamp,
                'symbol': market_data.get('symbol', 'NIFTY'),
                'open': market_data.get('open', 0),
                'high': market_data.get('high', 0),
                'low': market_data.get('low', 0),
                'close': market_data.get('close', 0),
                'volume': market_data.get('volume', 0),
                'current_capital': self.current_capital,
                'positions': self.positions,
                'cash_balance': self.cash_balance
            }
            
            # Call strategy function
            signals = await self.strategy_function(strategy_data)
            
            # Ensure signals is a list
            if not isinstance(signals, list):
                signals = [signals] if signals else []
            
            return signals
            
        except Exception as e:
            self.logger.error(f"Error generating signals: {e}")
            return []
    
    async def _execute_signal(self, timestamp: datetime, signal: TradeSignal, market_data: pd.Series):
        """Execute a trading signal"""
        try:
            # Validate signal
            if not self._validate_signal(signal):
                return
            
            # Calculate position size
            position_size = self._calculate_position_size(signal, market_data)
            if position_size <= 0:
                return
            
            # Calculate entry price with slippage
            entry_price = self._apply_slippage(signal.entry_price, signal.direction)
            
            # Calculate commission
            commission = position_size * entry_price * self.commission_rate
            
            # Check if we have enough capital
            required_capital = position_size * entry_price + commission
            if signal.direction == SignalDirection.BULLISH:
                required_capital *= self.margin_requirement
            
            if required_capital > self.cash_balance:
                self.logger.warning(f"Insufficient capital for signal: {signal.signal_id}")
                return
            
            # Create trade record
            trade = BacktestTrade(
                trade_id=signal.signal_id,
                symbol=signal.symbol,
                entry_time=timestamp,
                entry_price=entry_price,
                quantity=position_size,
                direction="LONG" if signal.direction == SignalDirection.BULLISH else "SHORT",
                commission=commission,
                slippage=abs(entry_price - signal.entry_price),
                metadata={
                    'signal': signal.to_dict(),
                    'stop_loss': signal.stop_loss,
                    'target_price': signal.target_price
                }
            )
            
            # Update positions
            if signal.symbol in self.positions:
                position = self.positions[signal.symbol]
                # Update existing position
                total_quantity = position.quantity + (position_size if signal.direction == SignalDirection.BULLISH else -position_size)
                if total_quantity == 0:
                    del self.positions[signal.symbol]
                else:
                    # Calculate new average price
                    total_value = position.quantity * position.average_price + position_size * entry_price
                    position.quantity = total_quantity
                    position.average_price = total_value / total_quantity
                    position.updated_at = timestamp
            else:
                # Create new position
                self.positions[signal.symbol] = Position(
                    symbol=signal.symbol,
                    quantity=position_size if signal.direction == SignalDirection.BULLISH else -position_size,
                    average_price=entry_price,
                    last_price=entry_price,
                    created_at=timestamp,
                    updated_at=timestamp
                )
            
            # Update cash balance
            self.cash_balance -= required_capital
            
            # Add to trade history
            self.trade_history.append(trade)
            
            self.logger.debug(f"Signal executed: {signal.signal_id} at {entry_price}")
            
        except Exception as e:
            self.logger.error(f"Error executing signal: {e}")
    
    def _validate_signal(self, signal: TradeSignal) -> bool:
        """Validate trading signal"""
        try:
            # Check if signal has required fields
            if not signal.symbol or signal.quantity <= 0:
                return False
            
            # Check confidence threshold
            if signal.confidence < 0.6:
                return False
            
            # Check if we already have a large position in this symbol
            current_position = self.positions.get(signal.symbol)
            if current_position:
                position_value = abs(current_position.quantity) * current_position.average_price
                if position_value > self.current_capital * self.max_position_size:
                    return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error validating signal: {e}")
            return False
    
    def _calculate_position_size(self, signal: TradeSignal, market_data: pd.Series) -> int:
        """Calculate position size based on risk management"""
        try:
            # Maximum position value
            max_position_value = self.current_capital * self.max_position_size
            
            # Calculate position size
            position_size = int(max_position_value / signal.entry_price)
            
            # Ensure minimum lot size (50 for options)
            lot_size = 50
            position_size = (position_size // lot_size) * lot_size
            
            return max(position_size, lot_size)
            
        except Exception as e:
            self.logger.error(f"Error calculating position size: {e}")
            return 0
    
    def _apply_slippage(self, price: float, direction: SignalDirection) -> float:
        """Apply slippage to price"""
        slippage = price * self.slippage_rate
        if direction == SignalDirection.BULLISH:
            return price + slippage  # Pay more when buying
        else:
            return price - slippage  # Receive less when selling
    
    async def _update_positions(self, timestamp: datetime, market_data: pd.Series):
        """Update position values with current market prices"""
        try:
            current_price = market_data.get('close', market_data.get('last_price', 0))
            
            for symbol, position in self.positions.items():
                if symbol == market_data.get('symbol', symbol):
                    position.last_price = current_price
                    position.updated_at = timestamp
                    
                    # Calculate unrealized P&L
                    if position.quantity > 0:  # Long position
                        position.unrealized_pnl = position.quantity * (current_price - position.average_price)
                    else:  # Short position
                        position.unrealized_pnl = abs(position.quantity) * (position.average_price - current_price)
            
        except Exception as e:
            self.logger.error(f"Error updating positions: {e}")
    
    async def _check_exit_conditions(self, timestamp: datetime, market_data: pd.Series):
        """Check stop loss and take profit conditions"""
        try:
            current_price = market_data.get('close', market_data.get('last_price', 0))
            positions_to_close = []
            
            for symbol, position in self.positions.items():
                if symbol != market_data.get('symbol', symbol):
                    continue
                
                # Find corresponding trade
                open_trade = None
                for trade in reversed(self.trade_history):
                    if trade.symbol == symbol and trade.exit_time is None:
                        open_trade = trade
                        break
                
                if not open_trade:
                    continue
                
                # Get stop loss and target from trade metadata
                stop_loss = open_trade.metadata.get('stop_loss', 0)
                target_price = open_trade.metadata.get('target_price', 0)
                
                exit_reason = None
                
                # Check stop loss
                if stop_loss > 0:
                    if position.quantity > 0 and current_price <= stop_loss:
                        exit_reason = "stop_loss"
                    elif position.quantity < 0 and current_price >= stop_loss:
                        exit_reason = "stop_loss"
                
                # Check take profit
                if target_price > 0 and not exit_reason:
                    if position.quantity > 0 and current_price >= target_price:
                        exit_reason = "take_profit"
                    elif position.quantity < 0 and current_price <= target_price:
                        exit_reason = "take_profit"
                
                # Check maximum adverse excursion (trailing stop)
                if not exit_reason:
                    current_pnl_pct = position.unrealized_pnl / (abs(position.quantity) * position.average_price)
                    if current_pnl_pct <= -self.stop_loss_percentage:
                        exit_reason = "trailing_stop"
                
                if exit_reason:
                    positions_to_close.append((symbol, exit_reason))
            
            # Close positions
            for symbol, exit_reason in positions_to_close:
                await self._close_position(timestamp, symbol, current_price, exit_reason)
                
        except Exception as e:
            self.logger.error(f"Error checking exit conditions: {e}")
    
    async def _close_position(self, timestamp: datetime, symbol: str, exit_price: float, exit_reason: str):
        """Close a position"""
        try:
            position = self.positions.get(symbol)
            if not position:
                return
            
            # Find corresponding open trade
            open_trade = None
            for trade in reversed(self.trade_history):
                if trade.symbol == symbol and trade.exit_time is None:
                    open_trade = trade
                    break
            
            if not open_trade:
                return
            
            # Apply slippage to exit price
            direction = SignalDirection.BEARISH if position.quantity > 0 else SignalDirection.BULLISH
            exit_price_with_slippage = self._apply_slippage(exit_price, direction)
            
            # Calculate commission
            commission = abs(position.quantity) * exit_price_with_slippage * self.commission_rate
            
            # Calculate P&L
            if position.quantity > 0:  # Long position
                pnl = position.quantity * (exit_price_with_slippage - position.average_price) - commission - open_trade.commission
            else:  # Short position
                pnl = abs(position.quantity) * (position.average_price - exit_price_with_slippage) - commission - open_trade.commission
            
            pnl_percentage = pnl / (abs(position.quantity) * position.average_price)
            
            # Update trade record
            open_trade.exit_time = timestamp
            open_trade.exit_price = exit_price_with_slippage
            open_trade.pnl = pnl
            open_trade.pnl_percentage = pnl_percentage
            open_trade.commission += commission
            open_trade.slippage += abs(exit_price_with_slippage - exit_price)
            open_trade.hold_time = timestamp - open_trade.entry_time
            open_trade.exit_reason = exit_reason
            
            # Update cash balance
            if position.quantity > 0:
                self.cash_balance += abs(position.quantity) * exit_price_with_slippage - commission
            else:
                # For short positions, return margin and add profit/loss
                margin_used = abs(position.quantity) * position.average_price * self.margin_requirement
                self.cash_balance += margin_used + pnl
            
            # Remove position
            del self.positions[symbol]
            
            self.logger.debug(f"Position closed: {symbol} P&L: {pnl:.2f} ({pnl_percentage:.2%})")
            
        except Exception as e:
            self.logger.error(f"Error closing position: {e}")
    
    async def _close_all_positions(self, final_data: pd.Series):
        """Close all remaining positions at the end of backtest"""
        try:
            final_price = final_data.get('close', final_data.get('last_price', 0))
            
            for symbol in list(self.positions.keys()):
                await self._close_position(final_data.name, symbol, final_price, "end_of_backtest")
                
        except Exception as e:
            self.logger.error(f"Error closing all positions: {e}")
    
    def _update_equity_curve(self, timestamp: datetime):
        """Update equity curve"""
        try:
            # Calculate total portfolio value
            total_position_value = 0
            total_unrealized_pnl = 0
            
            for position in self.positions.values():
                position_value = abs(position.quantity) * position.last_price
                total_position_value += position_value
                total_unrealized_pnl += position.unrealized_pnl
            
            total_equity = self.cash_balance + total_position_value + total_unrealized_pnl
            
            # Update current capital
            self.current_capital = total_equity
            
            # Add to equity curve
            self.equity_curve.append({
                'timestamp': timestamp,
                'equity': total_equity,
                'cash': self.cash_balance,
                'positions_value': total_position_value,
                'unrealized_pnl': total_unrealized_pnl,
                'position_count': len(self.positions)
            })
            
            # Calculate daily return
            if len(self.equity_curve) > 1:
                prev_equity = self.equity_curve[-2]['equity']
                daily_return = (total_equity - prev_equity) / prev_equity
                self.daily_returns.append(daily_return)
            
        except Exception as e:
            self.logger.error(f"Error updating equity curve: {e}")
    
    def _update_drawdown(self):
        """Update drawdown calculations"""
        try:
            if self.current_capital > self.peak_capital:
                self.peak_capital = self.current_capital
                self.current_drawdown = 0.0
            else:
                self.current_drawdown = (self.peak_capital - self.current_capital) / self.peak_capital
                if self.current_drawdown > self.max_drawdown:
                    self.max_drawdown = self.current_drawdown
                    
        except Exception as e:
            self.logger.error(f"Error updating drawdown: {e}")
    
    def _calculate_results(self, start_date: datetime, end_date: datetime) -> BacktestResult:
        """Calculate final backtest results"""
        try:
            # Basic metrics
            total_return = self.current_capital - self.initial_capital
            total_return_percentage = total_return / self.initial_capital
            
            # Trade statistics
            completed_trades = [t for t in self.trade_history if t.exit_time is not None]
            winning_trades = [t for t in completed_trades if t.pnl > 0]
            losing_trades = [t for t in completed_trades if t.pnl < 0]
            
            total_trades = len(completed_trades)
            winning_count = len(winning_trades)
            losing_count = len(losing_trades)
            
            win_rate = winning_count / total_trades if total_trades > 0 else 0
            
            avg_win = np.mean([t.pnl for t in winning_trades]) if winning_trades else 0
            avg_loss = np.mean([t.pnl for t in losing_trades]) if losing_trades else 0
            
            largest_win = max([t.pnl for t in winning_trades]) if winning_trades else 0
            largest_loss = min([t.pnl for t in losing_trades]) if losing_trades else 0
            
            # Profit factor
            gross_profit = sum([t.pnl for t in winning_trades])
            gross_loss = abs(sum([t.pnl for t in losing_trades]))
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
            
            # Risk metrics
            returns_array = np.array(self.daily_returns) if self.daily_returns else np.array([0])
            
            # Sharpe ratio
            if len(returns_array) > 1 and np.std(returns_array) > 0:
                sharpe_ratio = np.mean(returns_array) / np.std(returns_array) * np.sqrt(252)
            else:
                sharpe_ratio = 0
            
            # Sortino ratio
            negative_returns = returns_array[returns_array < 0]
            if len(negative_returns) > 1 and np.std(negative_returns) > 0:
                sortino_ratio = np.mean(returns_array) / np.std(negative_returns) * np.sqrt(252)
            else:
                sortino_ratio = 0
            
            # Calmar ratio
            calmar_ratio = (total_return_percentage * 252) / self.max_drawdown if self.max_drawdown > 0 else 0
            
            # Average hold time
            hold_times = [t.hold_time for t in completed_trades if t.hold_time]
            avg_hold_time = np.mean(hold_times) if hold_times else timedelta()
            
            # Commission and slippage
            total_commission = sum([t.commission for t in self.trade_history])
            total_slippage = sum([t.slippage for t in self.trade_history])
            
            # Create result object
            result = BacktestResult(
                start_date=start_date,
                end_date=end_date,
                initial_capital=self.initial_capital,
                final_capital=self.current_capital,
                total_return=total_return,
                total_return_percentage=total_return_percentage,
                max_drawdown=self.max_drawdown * self.initial_capital,
                max_drawdown_percentage=self.max_drawdown,
                sharpe_ratio=sharpe_ratio,
                sortino_ratio=sortino_ratio,
                calmar_ratio=calmar_ratio,
                win_rate=win_rate,
                profit_factor=profit_factor,
                total_trades=total_trades,
                winning_trades=winning_count,
                losing_trades=losing_count,
                avg_win=avg_win,
                avg_loss=avg_loss,
                largest_win=largest_win,
                largest_loss=largest_loss,
                avg_hold_time=avg_hold_time,
                total_commission=total_commission,
                total_slippage=total_slippage,
                trades=self.trade_history,
                equity_curve=self.equity_curve,
                daily_returns=self.daily_returns
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error calculating results: {e}")
            return BacktestResult()
    
    def _reset_state(self):
        """Reset backtester state"""
        self.current_capital = self.initial_capital
        self.cash_balance = self.initial_capital
        self.positions.clear()
        self.open_orders.clear()
        self.trade_history.clear()
        self.equity_curve.clear()
        self.daily_returns.clear()
        self.peak_capital = self.initial_capital
        self.max_drawdown = 0.0
        self.current_drawdown = 0.0
    
    def save_results(self, result: BacktestResult, filepath: str):
        """Save backtest results to file"""
        try:
            with open(filepath, 'w') as f:
                json.dump(result.to_dict(), f, indent=2, default=str)
            
            self.logger.info(f"Backtest results saved to {filepath}")
            
        except Exception as e:
            self.logger.error(f"Error saving results: {e}")
    
    def load_results(self, filepath: str) -> BacktestResult:
        """Load backtest results from file"""
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            # Convert back to BacktestResult object
            result = BacktestResult()
            for key, value in data.items():
                if hasattr(result, key):
                    setattr(result, key, value)
            
            self.logger.info(f"Backtest results loaded from {filepath}")
            return result
            
        except Exception as e:
            self.logger.error(f"Error loading results: {e}")
            return BacktestResult()

if __name__ == "__main__":
    # Test the backtester
    import asyncio
    
    async def simple_strategy(data):
        """Simple test strategy"""
        # Generate random signals for testing
        import random
        
        if random.random() > 0.95:  # 5% chance of signal
            signal = TradeSignal(
                symbol=data['symbol'],
                direction=SignalDirection.BULLISH if random.random() > 0.5 else SignalDirection.BEARISH,
                strength=SignalStrength.MODERATE,
                confidence=0.7,
                entry_price=data['close'],
                stop_loss=data['close'] * 0.98,
                target_price=data['close'] * 1.05,
                quantity=50
            )
            return signal
        
        return None
    
    async def test_backtester():
        # Create sample data
        dates = pd.date_range('2023-01-01', '2023-12-31', freq='D')
        np.random.seed(42)
        
        data = pd.DataFrame({
            'symbol': 'NIFTY',
            'open': 19000 + np.random.randn(len(dates)) * 100,
            'high': 19100 + np.random.randn(len(dates)) * 100,
            'low': 18900 + np.random.randn(len(dates)) * 100,
            'close': 19000 + np.random.randn(len(dates)) * 100,
            'volume': 1000000 + np.random.randint(0, 500000, len(dates))
        }, index=dates)
        
        # Ensure high >= close >= low and high >= open >= low
        data['high'] = np.maximum(data[['open', 'close']].max(axis=1), data['high'])
        data['low'] = np.minimum(data[['open', 'close']].min(axis=1), data['low'])
        
        # Create backtester
        backtester = Backtester(initial_capital=100000)
        backtester.set_strategy(simple_strategy)
        
        # Run backtest
        result = await backtester.run_backtest(data)
        
        print(f"Backtest Results:")
        print(f"Total Return: {result.total_return_percentage:.2%}")
        print(f"Sharpe Ratio: {result.sharpe_ratio:.2f}")
        print(f"Max Drawdown: {result.max_drawdown_percentage:.2%}")
        print(f"Win Rate: {result.win_rate:.2%}")
        print(f"Total Trades: {result.total_trades}")
        print(f"Profit Factor: {result.profit_factor:.2f}")
    
    asyncio.run(test_backtester())

