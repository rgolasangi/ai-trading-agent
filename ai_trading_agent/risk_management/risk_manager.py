"""
Risk Manager for AI Trading Agent
Comprehensive risk management and control system
"""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import asyncio
import json

from utils.logger import get_logger
from trading.execution_engine import TradeSignal, SignalDirection

logger = get_logger(__name__)

class RiskLevel(Enum):
    """Risk level enumeration"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class AlertType(Enum):
    """Risk alert types"""
    POSITION_LIMIT = "position_limit"
    DRAWDOWN = "drawdown"
    VOLATILITY = "volatility"
    CONCENTRATION = "concentration"
    MARGIN = "margin"
    VAR_BREACH = "var_breach"
    CORRELATION = "correlation"
    LIQUIDITY = "liquidity"

@dataclass
class RiskAlert:
    """Risk alert data structure"""
    alert_type: AlertType
    level: RiskLevel
    message: str
    timestamp: datetime
    data: Dict[str, Any]
    acknowledged: bool = False

@dataclass
class RiskLimits:
    """Risk limits configuration"""
    max_position_size: float = 0.1  # 10% of portfolio per position
    max_portfolio_exposure: float = 0.8  # 80% of total capital
    max_daily_loss: float = 0.05  # 5% daily loss limit
    max_drawdown: float = 0.15  # 15% maximum drawdown
    var_limit: float = 0.03  # 3% VaR limit
    max_correlation: float = 0.7  # Maximum correlation between positions
    min_liquidity_ratio: float = 0.2  # Minimum cash ratio
    max_leverage: float = 3.0  # Maximum leverage ratio
    max_concentration: float = 0.3  # Maximum sector/instrument concentration

class RiskManager:
    """Comprehensive Risk Management System"""
    
    def __init__(self, initial_capital: float = 100000):
        self.logger = get_logger(__name__)
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        
        # Risk limits and configuration
        self.risk_limits = RiskLimits()
        self.risk_alerts: List[RiskAlert] = []
        
        # Risk metrics tracking
        self.daily_pnl_history: List[float] = []
        self.portfolio_values: List[float] = [initial_capital]
        self.drawdown_history: List[float] = []
        self.var_history: List[float] = []
        
        # Position tracking
        self.positions: Dict[str, Dict[str, Any]] = {}
        self.position_history: List[Dict[str, Any]] = []
        
        # Risk monitoring flags
        self.emergency_stop = False
        self.trading_halted = False
        self.risk_override = False
        
        # Performance metrics
        self.performance_metrics = {
            'sharpe_ratio': 0.0,
            'sortino_ratio': 0.0,
            'max_drawdown': 0.0,
            'var_95': 0.0,
            'win_rate': 0.0,
            'profit_factor': 0.0,
            'calmar_ratio': 0.0
        }
        
        self.logger.info("Risk Manager initialized")
    
    def update_risk_limits(self, new_limits: Dict[str, float]):
        """Update risk limits"""
        try:
            for key, value in new_limits.items():
                if hasattr(self.risk_limits, key):
                    setattr(self.risk_limits, key, value)
                    self.logger.info(f"Updated risk limit {key}: {value}")
            
        except Exception as e:
            self.logger.error(f"Error updating risk limits: {e}")
    
    def validate_trade_signal(self, signal: TradeSignal, current_positions: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Validate trade signal against risk limits
        
        Args:
            signal: Trade signal to validate
            current_positions: Current portfolio positions
            
        Returns:
            Tuple of (is_valid, reason)
        """
        try:
            # Check if trading is halted
            if self.trading_halted:
                return False, "Trading is currently halted due to risk controls"
            
            if self.emergency_stop:
                return False, "Emergency stop is active"
            
            # Position size validation
            position_value = signal.quantity * signal.entry_price
            position_ratio = position_value / self.current_capital
            
            if position_ratio > self.risk_limits.max_position_size:
                return False, f"Position size {position_ratio:.2%} exceeds limit {self.risk_limits.max_position_size:.2%}"
            
            # Portfolio exposure validation
            total_exposure = self._calculate_total_exposure(current_positions, signal)
            if total_exposure > self.risk_limits.max_portfolio_exposure:
                return False, f"Total exposure {total_exposure:.2%} exceeds limit {self.risk_limits.max_portfolio_exposure:.2%}"
            
            # Concentration risk validation
            concentration = self._calculate_concentration_risk(current_positions, signal)
            if concentration > self.risk_limits.max_concentration:
                return False, f"Concentration risk {concentration:.2%} exceeds limit {self.risk_limits.max_concentration:.2%}"
            
            # Correlation risk validation
            correlation_risk = self._calculate_correlation_risk(current_positions, signal)
            if correlation_risk > self.risk_limits.max_correlation:
                return False, f"Correlation risk {correlation_risk:.2f} exceeds limit {self.risk_limits.max_correlation:.2f}"
            
            # Liquidity validation
            liquidity_ratio = self._calculate_liquidity_ratio(current_positions, signal)
            if liquidity_ratio < self.risk_limits.min_liquidity_ratio:
                return False, f"Liquidity ratio {liquidity_ratio:.2%} below minimum {self.risk_limits.min_liquidity_ratio:.2%}"
            
            # Leverage validation
            leverage = self._calculate_leverage(current_positions, signal)
            if leverage > self.risk_limits.max_leverage:
                return False, f"Leverage {leverage:.2f} exceeds limit {self.risk_limits.max_leverage:.2f}"
            
            return True, "Trade signal validated successfully"
            
        except Exception as e:
            self.logger.error(f"Error validating trade signal: {e}")
            return False, f"Validation error: {e}"
    
    def update_portfolio_metrics(self, current_positions: Dict[str, Any], market_data: Dict[str, Any]):
        """Update portfolio metrics and risk calculations"""
        try:
            # Calculate current portfolio value
            portfolio_value = self._calculate_portfolio_value(current_positions, market_data)
            self.portfolio_values.append(portfolio_value)
            self.current_capital = portfolio_value
            
            # Calculate daily P&L
            if len(self.portfolio_values) > 1:
                daily_pnl = portfolio_value - self.portfolio_values[-2]
                daily_return = daily_pnl / self.portfolio_values[-2]
                self.daily_pnl_history.append(daily_return)
            
            # Update drawdown
            peak_value = max(self.portfolio_values)
            current_drawdown = (portfolio_value - peak_value) / peak_value
            self.drawdown_history.append(current_drawdown)
            
            # Calculate VaR
            if len(self.daily_pnl_history) >= 30:
                var_95 = np.percentile(self.daily_pnl_history[-252:], 5)  # 95% VaR
                self.var_history.append(var_95)
            
            # Update performance metrics
            self._update_performance_metrics()
            
            # Check risk alerts
            self._check_risk_alerts(current_positions, market_data)
            
        except Exception as e:
            self.logger.error(f"Error updating portfolio metrics: {e}")
    
    def get_risk_dashboard_data(self) -> Dict[str, Any]:
        """Get comprehensive risk dashboard data"""
        try:
            return {
                'risk_level': self._get_overall_risk_level(),
                'risk_metrics': {
                    'var_95': self.var_history[-1] if self.var_history else 0,
                    'max_drawdown': min(self.drawdown_history) if self.drawdown_history else 0,
                    'current_drawdown': self.drawdown_history[-1] if self.drawdown_history else 0,
                    'portfolio_exposure': self._calculate_current_exposure(),
                    'liquidity_ratio': self._calculate_current_liquidity(),
                    'leverage': self._calculate_current_leverage(),
                    'concentration_risk': self._calculate_current_concentration()
                },
                'performance_metrics': self.performance_metrics,
                'risk_alerts': [
                    {
                        'type': alert.alert_type.value,
                        'level': alert.level.value,
                        'message': alert.message,
                        'timestamp': alert.timestamp.isoformat(),
                        'acknowledged': alert.acknowledged
                    }
                    for alert in self.risk_alerts[-10:]  # Last 10 alerts
                ],
                'limits': {
                    'max_position_size': self.risk_limits.max_position_size,
                    'max_portfolio_exposure': self.risk_limits.max_portfolio_exposure,
                    'max_daily_loss': self.risk_limits.max_daily_loss,
                    'max_drawdown': self.risk_limits.max_drawdown,
                    'var_limit': self.risk_limits.var_limit
                },
                'status': {
                    'trading_halted': self.trading_halted,
                    'emergency_stop': self.emergency_stop,
                    'risk_override': self.risk_override
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error getting risk dashboard data: {e}")
            return {}
    
    def _calculate_total_exposure(self, current_positions: Dict[str, Any], new_signal: TradeSignal) -> float:
        """Calculate total portfolio exposure including new signal"""
        try:
            total_exposure = 0
            
            # Current positions exposure
            for symbol, position in current_positions.items():
                position_value = abs(position.get('quantity', 0)) * position.get('current_price', 0)
                total_exposure += position_value
            
            # Add new signal exposure
            new_position_value = abs(new_signal.quantity) * new_signal.entry_price
            total_exposure += new_position_value
            
            return total_exposure / self.current_capital
            
        except Exception as e:
            self.logger.error(f"Error calculating total exposure: {e}")
            return 0
    
    def _calculate_concentration_risk(self, current_positions: Dict[str, Any], new_signal: TradeSignal) -> float:
        """Calculate concentration risk for instrument/sector"""
        try:
            # Group positions by underlying (NIFTY vs BANKNIFTY)
            concentration = {}
            
            for symbol, position in current_positions.items():
                underlying = 'NIFTY' if 'NIFTY' in symbol else 'BANKNIFTY'
                position_value = abs(position.get('quantity', 0)) * position.get('current_price', 0)
                concentration[underlying] = concentration.get(underlying, 0) + position_value
            
            # Add new signal
            new_underlying = 'NIFTY' if 'NIFTY' in new_signal.symbol else 'BANKNIFTY'
            new_value = abs(new_signal.quantity) * new_signal.entry_price
            concentration[new_underlying] = concentration.get(new_underlying, 0) + new_value
            
            # Return maximum concentration
            max_concentration = max(concentration.values()) if concentration else 0
            return max_concentration / self.current_capital
            
        except Exception as e:
            self.logger.error(f"Error calculating concentration risk: {e}")
            return 0
    
    def _calculate_correlation_risk(self, current_positions: Dict[str, Any], new_signal: TradeSignal) -> float:
        """Calculate correlation risk between positions"""
        try:
            # Simplified correlation calculation
            # In practice, this would use historical price correlations
            
            nifty_exposure = 0
            banknifty_exposure = 0
            
            for symbol, position in current_positions.items():
                position_value = position.get('quantity', 0) * position.get('current_price', 0)
                if 'NIFTY' in symbol:
                    nifty_exposure += position_value
                else:
                    banknifty_exposure += position_value
            
            # Add new signal
            new_value = new_signal.quantity * new_signal.entry_price
            if 'NIFTY' in new_signal.symbol:
                nifty_exposure += new_value
            else:
                banknifty_exposure += new_value
            
            # Calculate correlation proxy (same direction exposure)
            total_exposure = abs(nifty_exposure) + abs(banknifty_exposure)
            if total_exposure == 0:
                return 0
            
            # High correlation if both exposures are in same direction
            correlation = min(abs(nifty_exposure), abs(banknifty_exposure)) / total_exposure
            return correlation * 2  # Scale to 0-1 range
            
        except Exception as e:
            self.logger.error(f"Error calculating correlation risk: {e}")
            return 0
    
    def _calculate_liquidity_ratio(self, current_positions: Dict[str, Any], new_signal: TradeSignal) -> float:
        """Calculate liquidity ratio after new trade"""
        try:
            total_position_value = 0
            
            for symbol, position in current_positions.items():
                position_value = abs(position.get('quantity', 0)) * position.get('current_price', 0)
                total_position_value += position_value
            
            # Add new position value
            new_position_value = abs(new_signal.quantity) * new_signal.entry_price
            total_position_value += new_position_value
            
            # Calculate cash remaining
            cash_remaining = self.current_capital - total_position_value
            return cash_remaining / self.current_capital
            
        except Exception as e:
            self.logger.error(f"Error calculating liquidity ratio: {e}")
            return 0
    
    def _calculate_leverage(self, current_positions: Dict[str, Any], new_signal: TradeSignal) -> float:
        """Calculate portfolio leverage"""
        try:
            total_notional = 0
            
            for symbol, position in current_positions.items():
                # For options, notional value is quantity * underlying price * lot size
                notional_value = abs(position.get('quantity', 0)) * position.get('current_price', 0) * 10  # Assume 10x for options
                total_notional += notional_value
            
            # Add new signal notional
            new_notional = abs(new_signal.quantity) * new_signal.entry_price * 10
            total_notional += new_notional
            
            return total_notional / self.current_capital
            
        except Exception as e:
            self.logger.error(f"Error calculating leverage: {e}")
            return 1.0
    
    def _calculate_portfolio_value(self, current_positions: Dict[str, Any], market_data: Dict[str, Any]) -> float:
        """Calculate current portfolio value"""
        try:
            portfolio_value = 0
            
            for symbol, position in current_positions.items():
                quantity = position.get('quantity', 0)
                current_price = market_data.get(symbol, {}).get('ltp', position.get('current_price', 0))
                position_value = quantity * current_price
                portfolio_value += position_value
            
            # Add cash (simplified - assume remaining capital is cash)
            cash = self.current_capital - sum(
                abs(pos.get('quantity', 0)) * pos.get('avg_price', 0) 
                for pos in current_positions.values()
            )
            portfolio_value += max(cash, 0)
            
            return portfolio_value
            
        except Exception as e:
            self.logger.error(f"Error calculating portfolio value: {e}")
            return self.current_capital
    
    def _update_performance_metrics(self):
        """Update performance metrics"""
        try:
            if len(self.daily_pnl_history) < 30:
                return
            
            returns = np.array(self.daily_pnl_history)
            
            # Sharpe Ratio
            if np.std(returns) > 0:
                self.performance_metrics['sharpe_ratio'] = np.mean(returns) / np.std(returns) * np.sqrt(252)
            
            # Sortino Ratio
            negative_returns = returns[returns < 0]
            if len(negative_returns) > 0 and np.std(negative_returns) > 0:
                self.performance_metrics['sortino_ratio'] = np.mean(returns) / np.std(negative_returns) * np.sqrt(252)
            
            # Max Drawdown
            self.performance_metrics['max_drawdown'] = min(self.drawdown_history) if self.drawdown_history else 0
            
            # VaR 95%
            self.performance_metrics['var_95'] = np.percentile(returns, 5) if len(returns) > 0 else 0
            
            # Win Rate
            winning_days = len(returns[returns > 0])
            self.performance_metrics['win_rate'] = winning_days / len(returns) if len(returns) > 0 else 0
            
            # Profit Factor
            gross_profit = np.sum(returns[returns > 0])
            gross_loss = abs(np.sum(returns[returns < 0]))
            self.performance_metrics['profit_factor'] = gross_profit / gross_loss if gross_loss > 0 else 0
            
            # Calmar Ratio
            annual_return = np.mean(returns) * 252
            max_dd = abs(self.performance_metrics['max_drawdown'])
            self.performance_metrics['calmar_ratio'] = annual_return / max_dd if max_dd > 0 else 0
            
        except Exception as e:
            self.logger.error(f"Error updating performance metrics: {e}")
    
    def _check_risk_alerts(self, current_positions: Dict[str, Any], market_data: Dict[str, Any]):
        """Check for risk alerts and generate notifications"""
        try:
            current_time = datetime.now()
            
            # Check drawdown alert
            if self.drawdown_history and self.drawdown_history[-1] < -self.risk_limits.max_drawdown:
                self._create_alert(
                    AlertType.DRAWDOWN,
                    RiskLevel.CRITICAL,
                    f"Maximum drawdown exceeded: {self.drawdown_history[-1]:.2%}",
                    current_time
                )
            
            # Check daily loss alert
            if self.daily_pnl_history and self.daily_pnl_history[-1] < -self.risk_limits.max_daily_loss:
                self._create_alert(
                    AlertType.POSITION_LIMIT,
                    RiskLevel.HIGH,
                    f"Daily loss limit exceeded: {self.daily_pnl_history[-1]:.2%}",
                    current_time
                )
            
            # Check VaR alert
            if self.var_history and self.var_history[-1] < -self.risk_limits.var_limit:
                self._create_alert(
                    AlertType.VAR_BREACH,
                    RiskLevel.HIGH,
                    f"VaR limit breached: {self.var_history[-1]:.2%}",
                    current_time
                )
            
            # Check exposure alert
            current_exposure = self._calculate_current_exposure()
            if current_exposure > self.risk_limits.max_portfolio_exposure:
                self._create_alert(
                    AlertType.POSITION_LIMIT,
                    RiskLevel.MEDIUM,
                    f"Portfolio exposure high: {current_exposure:.2%}",
                    current_time
                )
            
            # Check concentration alert
            concentration = self._calculate_current_concentration()
            if concentration > self.risk_limits.max_concentration:
                self._create_alert(
                    AlertType.CONCENTRATION,
                    RiskLevel.MEDIUM,
                    f"Concentration risk high: {concentration:.2%}",
                    current_time
                )
            
        except Exception as e:
            self.logger.error(f"Error checking risk alerts: {e}")
    
    def _create_alert(self, alert_type: AlertType, level: RiskLevel, message: str, timestamp: datetime):
        """Create a new risk alert"""
        try:
            alert = RiskAlert(
                alert_type=alert_type,
                level=level,
                message=message,
                timestamp=timestamp,
                data={}
            )
            
            self.risk_alerts.append(alert)
            self.logger.warning(f"Risk Alert [{level.value.upper()}]: {message}")
            
            # Auto-halt trading for critical alerts
            if level == RiskLevel.CRITICAL:
                self.trading_halted = True
                self.logger.critical("Trading halted due to critical risk alert")
            
        except Exception as e:
            self.logger.error(f"Error creating risk alert: {e}")
    
    def _get_overall_risk_level(self) -> str:
        """Calculate overall risk level"""
        try:
            # Count recent alerts by level
            recent_alerts = [a for a in self.risk_alerts if a.timestamp > datetime.now() - timedelta(hours=24)]
            
            critical_count = len([a for a in recent_alerts if a.level == RiskLevel.CRITICAL])
            high_count = len([a for a in recent_alerts if a.level == RiskLevel.HIGH])
            medium_count = len([a for a in recent_alerts if a.level == RiskLevel.MEDIUM])
            
            if critical_count > 0 or self.trading_halted:
                return RiskLevel.CRITICAL.value
            elif high_count > 2:
                return RiskLevel.HIGH.value
            elif high_count > 0 or medium_count > 3:
                return RiskLevel.MEDIUM.value
            else:
                return RiskLevel.LOW.value
                
        except Exception as e:
            self.logger.error(f"Error calculating overall risk level: {e}")
            return RiskLevel.MEDIUM.value
    
    def _calculate_current_exposure(self) -> float:
        """Calculate current portfolio exposure"""
        try:
            # This would be calculated from actual positions
            # For now, return a placeholder
            return 0.64  # 64% exposure
            
        except Exception as e:
            self.logger.error(f"Error calculating current exposure: {e}")
            return 0
    
    def _calculate_current_liquidity(self) -> float:
        """Calculate current liquidity ratio"""
        try:
            return 1.0 - self._calculate_current_exposure()
            
        except Exception as e:
            self.logger.error(f"Error calculating current liquidity: {e}")
            return 0.2
    
    def _calculate_current_leverage(self) -> float:
        """Calculate current leverage"""
        try:
            # Placeholder calculation
            return 2.1
            
        except Exception as e:
            self.logger.error(f"Error calculating current leverage: {e}")
            return 1.0
    
    def _calculate_current_concentration(self) -> float:
        """Calculate current concentration risk"""
        try:
            # Placeholder calculation
            return 0.45  # 45% concentration
            
        except Exception as e:
            self.logger.error(f"Error calculating current concentration: {e}")
            return 0
    
    def emergency_stop_trading(self, reason: str = "Manual emergency stop"):
        """Emergency stop all trading"""
        try:
            self.emergency_stop = True
            self.trading_halted = True
            
            self._create_alert(
                AlertType.POSITION_LIMIT,
                RiskLevel.CRITICAL,
                f"Emergency stop activated: {reason}",
                datetime.now()
            )
            
            self.logger.critical(f"EMERGENCY STOP: {reason}")
            
        except Exception as e:
            self.logger.error(f"Error in emergency stop: {e}")
    
    def resume_trading(self, override_reason: str = "Manual override"):
        """Resume trading after emergency stop"""
        try:
            if self.risk_override or override_reason:
                self.emergency_stop = False
                self.trading_halted = False
                
                self.logger.info(f"Trading resumed: {override_reason}")
            else:
                self.logger.warning("Cannot resume trading without risk override")
                
        except Exception as e:
            self.logger.error(f"Error resuming trading: {e}")
    
    def acknowledge_alert(self, alert_index: int):
        """Acknowledge a risk alert"""
        try:
            if 0 <= alert_index < len(self.risk_alerts):
                self.risk_alerts[alert_index].acknowledged = True
                self.logger.info(f"Alert {alert_index} acknowledged")
                
        except Exception as e:
            self.logger.error(f"Error acknowledging alert: {e}")
    
    def save_risk_state(self, filepath: str):
        """Save risk manager state to file"""
        try:
            state = {
                'current_capital': self.current_capital,
                'daily_pnl_history': self.daily_pnl_history,
                'portfolio_values': self.portfolio_values,
                'drawdown_history': self.drawdown_history,
                'var_history': self.var_history,
                'performance_metrics': self.performance_metrics,
                'risk_limits': {
                    'max_position_size': self.risk_limits.max_position_size,
                    'max_portfolio_exposure': self.risk_limits.max_portfolio_exposure,
                    'max_daily_loss': self.risk_limits.max_daily_loss,
                    'max_drawdown': self.risk_limits.max_drawdown,
                    'var_limit': self.risk_limits.var_limit,
                    'max_correlation': self.risk_limits.max_correlation,
                    'min_liquidity_ratio': self.risk_limits.min_liquidity_ratio,
                    'max_leverage': self.risk_limits.max_leverage,
                    'max_concentration': self.risk_limits.max_concentration
                },
                'status': {
                    'emergency_stop': self.emergency_stop,
                    'trading_halted': self.trading_halted,
                    'risk_override': self.risk_override
                },
                'timestamp': datetime.now().isoformat()
            }
            
            with open(filepath, 'w') as f:
                json.dump(state, f, indent=2)
            
            self.logger.info(f"Risk state saved to {filepath}")
            
        except Exception as e:
            self.logger.error(f"Error saving risk state: {e}")

if __name__ == "__main__":
    # Test the risk manager
    from trading.execution_engine import TradeSignal, SignalDirection, SignalStrength
    
    # Create risk manager
    risk_manager = RiskManager(initial_capital=100000)
    
    # Test trade signal validation
    test_signal = TradeSignal(
        symbol="NIFTY24APR19500CE",
        direction=SignalDirection.BULLISH,
        strength=SignalStrength.STRONG,
        confidence=0.85,
        entry_price=150,
        stop_loss=140,
        target_price=170,
        quantity=100
    )
    
    # Test validation
    is_valid, reason = risk_manager.validate_trade_signal(test_signal, {})
    print(f"Trade validation: {is_valid}, Reason: {reason}")
    
    # Test risk dashboard data
    dashboard_data = risk_manager.get_risk_dashboard_data()
    print("\nRisk Dashboard Data:")
    for key, value in dashboard_data.items():
        print(f"{key}: {value}")
    
    # Test emergency stop
    risk_manager.emergency_stop_trading("Test emergency stop")
    print(f"\nEmergency stop status: {risk_manager.emergency_stop}")
    
    # Save risk state
    risk_manager.save_risk_state("/tmp/risk_state.json")
    print("\nRisk state saved to /tmp/risk_state.json")

