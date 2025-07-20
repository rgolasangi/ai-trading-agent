"""
Performance Analyzer for AI Trading Agent
Advanced performance metrics and analysis tools
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import warnings
warnings.filterwarnings('ignore')

from utils.logger import get_logger
from .backtester import BacktestResult, BacktestTrade

logger = get_logger(__name__)

class PerformanceAnalyzer:
    """Advanced Performance Analysis Tools"""
    
    def __init__(self):
        self.logger = get_logger(__name__)
        
        # Set plotting style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
    
    def analyze_performance(self, result: BacktestResult) -> Dict[str, Any]:
        """
        Comprehensive performance analysis
        
        Args:
            result: Backtest result to analyze
            
        Returns:
            Dictionary with detailed performance metrics
        """
        try:
            analysis = {
                'basic_metrics': self._calculate_basic_metrics(result),
                'risk_metrics': self._calculate_risk_metrics(result),
                'trade_analysis': self._analyze_trades(result),
                'time_analysis': self._analyze_time_patterns(result),
                'drawdown_analysis': self._analyze_drawdowns(result),
                'rolling_metrics': self._calculate_rolling_metrics(result),
                'benchmark_comparison': self._compare_to_benchmark(result)
            }
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Error analyzing performance: {e}")
            return {}
    
    def _calculate_basic_metrics(self, result: BacktestResult) -> Dict[str, float]:
        """Calculate basic performance metrics"""
        try:
            metrics = {
                'total_return': result.total_return,
                'total_return_pct': result.total_return_percentage,
                'annualized_return': self._annualize_return(result),
                'volatility': self._calculate_volatility(result),
                'sharpe_ratio': result.sharpe_ratio,
                'sortino_ratio': result.sortino_ratio,
                'calmar_ratio': result.calmar_ratio,
                'max_drawdown': result.max_drawdown,
                'max_drawdown_pct': result.max_drawdown_percentage,
                'win_rate': result.win_rate,
                'profit_factor': result.profit_factor,
                'expectancy': self._calculate_expectancy(result),
                'kelly_criterion': self._calculate_kelly_criterion(result)
            }
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error calculating basic metrics: {e}")
            return {}
    
    def _calculate_risk_metrics(self, result: BacktestResult) -> Dict[str, float]:
        """Calculate risk-adjusted metrics"""
        try:
            returns = np.array(result.daily_returns)
            
            metrics = {
                'var_95': np.percentile(returns, 5) if len(returns) > 0 else 0,
                'var_99': np.percentile(returns, 1) if len(returns) > 0 else 0,
                'cvar_95': np.mean(returns[returns <= np.percentile(returns, 5)]) if len(returns) > 0 else 0,
                'cvar_99': np.mean(returns[returns <= np.percentile(returns, 1)]) if len(returns) > 0 else 0,
                'skewness': self._calculate_skewness(returns),
                'kurtosis': self._calculate_kurtosis(returns),
                'tail_ratio': self._calculate_tail_ratio(returns),
                'gain_to_pain_ratio': self._calculate_gain_to_pain_ratio(returns),
                'ulcer_index': self._calculate_ulcer_index(result),
                'sterling_ratio': self._calculate_sterling_ratio(result)
            }
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error calculating risk metrics: {e}")
            return {}
    
    def _analyze_trades(self, result: BacktestResult) -> Dict[str, Any]:
        """Analyze individual trades"""
        try:
            completed_trades = [t for t in result.trades if t.exit_time is not None]
            
            if not completed_trades:
                return {}
            
            # Trade statistics
            pnls = [t.pnl for t in completed_trades]
            pnl_pcts = [t.pnl_percentage for t in completed_trades]
            hold_times = [t.hold_time.total_seconds() / 3600 for t in completed_trades]  # Hours
            
            winning_trades = [t for t in completed_trades if t.pnl > 0]
            losing_trades = [t for t in completed_trades if t.pnl < 0]
            
            analysis = {
                'total_trades': len(completed_trades),
                'winning_trades': len(winning_trades),
                'losing_trades': len(losing_trades),
                'win_rate': len(winning_trades) / len(completed_trades),
                'avg_win': np.mean([t.pnl for t in winning_trades]) if winning_trades else 0,
                'avg_loss': np.mean([t.pnl for t in losing_trades]) if losing_trades else 0,
                'avg_win_pct': np.mean([t.pnl_percentage for t in winning_trades]) if winning_trades else 0,
                'avg_loss_pct': np.mean([t.pnl_percentage for t in losing_trades]) if losing_trades else 0,
                'largest_win': max(pnls) if pnls else 0,
                'largest_loss': min(pnls) if pnls else 0,
                'avg_trade': np.mean(pnls) if pnls else 0,
                'median_trade': np.median(pnls) if pnls else 0,
                'std_trade': np.std(pnls) if pnls else 0,
                'avg_hold_time_hours': np.mean(hold_times) if hold_times else 0,
                'median_hold_time_hours': np.median(hold_times) if hold_times else 0,
                'consecutive_wins': self._calculate_consecutive_wins(completed_trades),
                'consecutive_losses': self._calculate_consecutive_losses(completed_trades),
                'trade_distribution': self._analyze_trade_distribution(pnl_pcts)
            }
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Error analyzing trades: {e}")
            return {}
    
    def _analyze_time_patterns(self, result: BacktestResult) -> Dict[str, Any]:
        """Analyze time-based patterns"""
        try:
            if not result.equity_curve:
                return {}
            
            # Convert to DataFrame
            df = pd.DataFrame(result.equity_curve)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
            
            # Calculate returns
            df['returns'] = df['equity'].pct_change()
            
            # Time-based analysis
            df['hour'] = df.index.hour
            df['day_of_week'] = df.index.dayofweek
            df['month'] = df.index.month
            df['quarter'] = df.index.quarter
            
            analysis = {
                'hourly_returns': df.groupby('hour')['returns'].mean().to_dict(),
                'daily_returns': df.groupby('day_of_week')['returns'].mean().to_dict(),
                'monthly_returns': df.groupby('month')['returns'].mean().to_dict(),
                'quarterly_returns': df.groupby('quarter')['returns'].mean().to_dict(),
                'best_hour': df.groupby('hour')['returns'].mean().idxmax(),
                'worst_hour': df.groupby('hour')['returns'].mean().idxmin(),
                'best_day': df.groupby('day_of_week')['returns'].mean().idxmax(),
                'worst_day': df.groupby('day_of_week')['returns'].mean().idxmin(),
                'volatility_by_hour': df.groupby('hour')['returns'].std().to_dict(),
                'volatility_by_day': df.groupby('day_of_week')['returns'].std().to_dict()
            }
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Error analyzing time patterns: {e}")
            return {}
    
    def _analyze_drawdowns(self, result: BacktestResult) -> Dict[str, Any]:
        """Analyze drawdown patterns"""
        try:
            if not result.equity_curve:
                return {}
            
            equity_values = [point['equity'] for point in result.equity_curve]
            timestamps = [point['timestamp'] for point in result.equity_curve]
            
            # Calculate drawdowns
            peak = np.maximum.accumulate(equity_values)
            drawdowns = (equity_values - peak) / peak
            
            # Find drawdown periods
            drawdown_periods = []
            in_drawdown = False
            start_idx = 0
            
            for i, dd in enumerate(drawdowns):
                if dd < -0.01 and not in_drawdown:  # Start of drawdown (>1%)
                    in_drawdown = True
                    start_idx = i
                elif dd >= 0 and in_drawdown:  # End of drawdown
                    in_drawdown = False
                    drawdown_periods.append({
                        'start': timestamps[start_idx],
                        'end': timestamps[i],
                        'duration': i - start_idx,
                        'max_drawdown': min(drawdowns[start_idx:i+1]),
                        'recovery_time': i - start_idx
                    })
            
            analysis = {
                'max_drawdown': min(drawdowns) if drawdowns else 0,
                'avg_drawdown': np.mean([dd for dd in drawdowns if dd < 0]) if drawdowns else 0,
                'drawdown_periods': len(drawdown_periods),
                'avg_drawdown_duration': np.mean([p['duration'] for p in drawdown_periods]) if drawdown_periods else 0,
                'max_drawdown_duration': max([p['duration'] for p in drawdown_periods]) if drawdown_periods else 0,
                'avg_recovery_time': np.mean([p['recovery_time'] for p in drawdown_periods]) if drawdown_periods else 0,
                'time_underwater': sum([p['duration'] for p in drawdown_periods]) / len(equity_values) if drawdown_periods else 0
            }
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Error analyzing drawdowns: {e}")
            return {}
    
    def _calculate_rolling_metrics(self, result: BacktestResult, window: int = 30) -> Dict[str, Any]:
        """Calculate rolling performance metrics"""
        try:
            if not result.daily_returns or len(result.daily_returns) < window:
                return {}
            
            returns = np.array(result.daily_returns)
            
            # Rolling calculations
            rolling_returns = []
            rolling_volatility = []
            rolling_sharpe = []
            rolling_max_dd = []
            
            for i in range(window, len(returns)):
                window_returns = returns[i-window:i]
                
                # Rolling return
                rolling_returns.append(np.mean(window_returns) * 252)
                
                # Rolling volatility
                rolling_volatility.append(np.std(window_returns) * np.sqrt(252))
                
                # Rolling Sharpe
                if np.std(window_returns) > 0:
                    rolling_sharpe.append(np.mean(window_returns) / np.std(window_returns) * np.sqrt(252))
                else:
                    rolling_sharpe.append(0)
                
                # Rolling max drawdown
                cumulative = np.cumprod(1 + window_returns)
                peak = np.maximum.accumulate(cumulative)
                dd = (cumulative - peak) / peak
                rolling_max_dd.append(min(dd))
            
            analysis = {
                'rolling_return_mean': np.mean(rolling_returns),
                'rolling_return_std': np.std(rolling_returns),
                'rolling_volatility_mean': np.mean(rolling_volatility),
                'rolling_volatility_std': np.std(rolling_volatility),
                'rolling_sharpe_mean': np.mean(rolling_sharpe),
                'rolling_sharpe_std': np.std(rolling_sharpe),
                'rolling_max_dd_mean': np.mean(rolling_max_dd),
                'rolling_max_dd_std': np.std(rolling_max_dd),
                'stability_ratio': np.mean(rolling_sharpe) / np.std(rolling_sharpe) if np.std(rolling_sharpe) > 0 else 0
            }
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Error calculating rolling metrics: {e}")
            return {}
    
    def _compare_to_benchmark(self, result: BacktestResult, benchmark_return: float = 0.12) -> Dict[str, float]:
        """Compare performance to benchmark"""
        try:
            strategy_return = result.total_return_percentage
            
            # Calculate tracking error
            if result.daily_returns:
                benchmark_daily = benchmark_return / 252
                tracking_error = np.std(np.array(result.daily_returns) - benchmark_daily) * np.sqrt(252)
                information_ratio = (strategy_return - benchmark_return) / tracking_error if tracking_error > 0 else 0
            else:
                tracking_error = 0
                information_ratio = 0
            
            analysis = {
                'benchmark_return': benchmark_return,
                'strategy_return': strategy_return,
                'excess_return': strategy_return - benchmark_return,
                'tracking_error': tracking_error,
                'information_ratio': information_ratio,
                'treynor_ratio': strategy_return / 1.0,  # Assuming beta = 1
                'jensen_alpha': strategy_return - benchmark_return
            }
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Error comparing to benchmark: {e}")
            return {}
    
    def _annualize_return(self, result: BacktestResult) -> float:
        """Calculate annualized return"""
        try:
            days = (result.end_date - result.start_date).days
            if days <= 0:
                return 0
            
            years = days / 365.25
            return (result.final_capital / result.initial_capital) ** (1 / years) - 1
            
        except Exception as e:
            self.logger.error(f"Error calculating annualized return: {e}")
            return 0
    
    def _calculate_volatility(self, result: BacktestResult) -> float:
        """Calculate annualized volatility"""
        try:
            if not result.daily_returns:
                return 0
            
            return np.std(result.daily_returns) * np.sqrt(252)
            
        except Exception as e:
            self.logger.error(f"Error calculating volatility: {e}")
            return 0
    
    def _calculate_expectancy(self, result: BacktestResult) -> float:
        """Calculate trade expectancy"""
        try:
            completed_trades = [t for t in result.trades if t.exit_time is not None]
            
            if not completed_trades:
                return 0
            
            winning_trades = [t for t in completed_trades if t.pnl > 0]
            losing_trades = [t for t in completed_trades if t.pnl < 0]
            
            if not winning_trades or not losing_trades:
                return 0
            
            win_rate = len(winning_trades) / len(completed_trades)
            avg_win = np.mean([t.pnl for t in winning_trades])
            avg_loss = abs(np.mean([t.pnl for t in losing_trades]))
            
            return (win_rate * avg_win) - ((1 - win_rate) * avg_loss)
            
        except Exception as e:
            self.logger.error(f"Error calculating expectancy: {e}")
            return 0
    
    def _calculate_kelly_criterion(self, result: BacktestResult) -> float:
        """Calculate Kelly criterion for optimal position sizing"""
        try:
            completed_trades = [t for t in result.trades if t.exit_time is not None]
            
            if not completed_trades:
                return 0
            
            winning_trades = [t for t in completed_trades if t.pnl > 0]
            losing_trades = [t for t in completed_trades if t.pnl < 0]
            
            if not winning_trades or not losing_trades:
                return 0
            
            win_rate = len(winning_trades) / len(completed_trades)
            avg_win_pct = np.mean([t.pnl_percentage for t in winning_trades])
            avg_loss_pct = abs(np.mean([t.pnl_percentage for t in losing_trades]))
            
            if avg_loss_pct == 0:
                return 0
            
            kelly = win_rate - ((1 - win_rate) / (avg_win_pct / avg_loss_pct))
            return max(0, min(kelly, 0.25))  # Cap at 25%
            
        except Exception as e:
            self.logger.error(f"Error calculating Kelly criterion: {e}")
            return 0
    
    def _calculate_skewness(self, returns: np.ndarray) -> float:
        """Calculate skewness of returns"""
        try:
            if len(returns) < 3:
                return 0
            
            mean_return = np.mean(returns)
            std_return = np.std(returns)
            
            if std_return == 0:
                return 0
            
            skewness = np.mean(((returns - mean_return) / std_return) ** 3)
            return skewness
            
        except Exception as e:
            self.logger.error(f"Error calculating skewness: {e}")
            return 0
    
    def _calculate_kurtosis(self, returns: np.ndarray) -> float:
        """Calculate kurtosis of returns"""
        try:
            if len(returns) < 4:
                return 0
            
            mean_return = np.mean(returns)
            std_return = np.std(returns)
            
            if std_return == 0:
                return 0
            
            kurtosis = np.mean(((returns - mean_return) / std_return) ** 4) - 3
            return kurtosis
            
        except Exception as e:
            self.logger.error(f"Error calculating kurtosis: {e}")
            return 0
    
    def _calculate_tail_ratio(self, returns: np.ndarray) -> float:
        """Calculate tail ratio (95th percentile / 5th percentile)"""
        try:
            if len(returns) < 20:
                return 0
            
            p95 = np.percentile(returns, 95)
            p5 = np.percentile(returns, 5)
            
            return abs(p95 / p5) if p5 != 0 else 0
            
        except Exception as e:
            self.logger.error(f"Error calculating tail ratio: {e}")
            return 0
    
    def _calculate_gain_to_pain_ratio(self, returns: np.ndarray) -> float:
        """Calculate gain-to-pain ratio"""
        try:
            if len(returns) == 0:
                return 0
            
            total_return = np.sum(returns)
            pain = np.sum(np.abs(returns[returns < 0]))
            
            return total_return / pain if pain > 0 else 0
            
        except Exception as e:
            self.logger.error(f"Error calculating gain-to-pain ratio: {e}")
            return 0
    
    def _calculate_ulcer_index(self, result: BacktestResult) -> float:
        """Calculate Ulcer Index"""
        try:
            if not result.equity_curve:
                return 0
            
            equity_values = np.array([point['equity'] for point in result.equity_curve])
            peak = np.maximum.accumulate(equity_values)
            drawdowns = (equity_values - peak) / peak
            
            ulcer_index = np.sqrt(np.mean(drawdowns ** 2))
            return ulcer_index
            
        except Exception as e:
            self.logger.error(f"Error calculating Ulcer Index: {e}")
            return 0
    
    def _calculate_sterling_ratio(self, result: BacktestResult) -> float:
        """Calculate Sterling ratio"""
        try:
            annualized_return = self._annualize_return(result)
            avg_drawdown = np.mean([abs(dd) for dd in [point['equity'] for point in result.equity_curve] if dd < 0])
            
            return annualized_return / avg_drawdown if avg_drawdown > 0 else 0
            
        except Exception as e:
            self.logger.error(f"Error calculating Sterling ratio: {e}")
            return 0
    
    def _calculate_consecutive_wins(self, trades: List[BacktestTrade]) -> int:
        """Calculate maximum consecutive wins"""
        try:
            max_consecutive = 0
            current_consecutive = 0
            
            for trade in trades:
                if trade.pnl > 0:
                    current_consecutive += 1
                    max_consecutive = max(max_consecutive, current_consecutive)
                else:
                    current_consecutive = 0
            
            return max_consecutive
            
        except Exception as e:
            self.logger.error(f"Error calculating consecutive wins: {e}")
            return 0
    
    def _calculate_consecutive_losses(self, trades: List[BacktestTrade]) -> int:
        """Calculate maximum consecutive losses"""
        try:
            max_consecutive = 0
            current_consecutive = 0
            
            for trade in trades:
                if trade.pnl < 0:
                    current_consecutive += 1
                    max_consecutive = max(max_consecutive, current_consecutive)
                else:
                    current_consecutive = 0
            
            return max_consecutive
            
        except Exception as e:
            self.logger.error(f"Error calculating consecutive losses: {e}")
            return 0
    
    def _analyze_trade_distribution(self, pnl_percentages: List[float]) -> Dict[str, float]:
        """Analyze distribution of trade P&L percentages"""
        try:
            if not pnl_percentages:
                return {}
            
            pnl_array = np.array(pnl_percentages)
            
            return {
                'mean': np.mean(pnl_array),
                'median': np.median(pnl_array),
                'std': np.std(pnl_array),
                'min': np.min(pnl_array),
                'max': np.max(pnl_array),
                'q25': np.percentile(pnl_array, 25),
                'q75': np.percentile(pnl_array, 75),
                'skewness': self._calculate_skewness(pnl_array),
                'kurtosis': self._calculate_kurtosis(pnl_array)
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing trade distribution: {e}")
            return {}
    
    def generate_performance_report(self, result: BacktestResult, filepath: str):
        """Generate comprehensive performance report"""
        try:
            analysis = self.analyze_performance(result)
            
            report = {
                'backtest_summary': {
                    'start_date': result.start_date.isoformat(),
                    'end_date': result.end_date.isoformat(),
                    'initial_capital': result.initial_capital,
                    'final_capital': result.final_capital,
                    'duration_days': (result.end_date - result.start_date).days
                },
                'performance_analysis': analysis,
                'generated_at': datetime.now().isoformat()
            }
            
            with open(filepath, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            self.logger.info(f"Performance report generated: {filepath}")
            
        except Exception as e:
            self.logger.error(f"Error generating performance report: {e}")

if __name__ == "__main__":
    # Test the performance analyzer
    from .backtester import BacktestResult, BacktestTrade
    
    # Create sample backtest result
    result = BacktestResult(
        start_date=datetime(2023, 1, 1),
        end_date=datetime(2023, 12, 31),
        initial_capital=100000,
        final_capital=120000,
        total_return=20000,
        total_return_percentage=0.2,
        daily_returns=[0.001, -0.002, 0.003, -0.001, 0.002] * 50  # Sample returns
    )
    
    # Create analyzer
    analyzer = PerformanceAnalyzer()
    
    # Analyze performance
    analysis = analyzer.analyze_performance(result)
    
    print("Performance Analysis Results:")
    for category, metrics in analysis.items():
        print(f"\n{category.upper()}:")
        if isinstance(metrics, dict):
            for key, value in metrics.items():
                print(f"  {key}: {value}")
        else:
            print(f"  {metrics}")
    
    # Generate report
    analyzer.generate_performance_report(result, "/tmp/performance_report.json")
    print("\nPerformance report generated: /tmp/performance_report.json")

