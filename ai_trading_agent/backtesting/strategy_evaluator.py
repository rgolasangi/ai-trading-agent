"""
Strategy Evaluator for AI Trading Agent
Framework for evaluating and comparing trading strategies
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Callable
import asyncio
import json
import warnings
warnings.filterwarnings('ignore')

from utils.logger import get_logger
from .backtester import Backtester, BacktestResult
from .performance_analyzer import PerformanceAnalyzer

logger = get_logger(__name__)

class StrategyEvaluator:
    """Strategy Evaluation and Comparison Framework"""
    
    def __init__(self):
        self.logger = get_logger(__name__)
        self.performance_analyzer = PerformanceAnalyzer()
        
        # Evaluation criteria weights
        self.evaluation_weights = {
            'total_return': 0.25,
            'sharpe_ratio': 0.20,
            'max_drawdown': 0.15,
            'win_rate': 0.10,
            'profit_factor': 0.10,
            'calmar_ratio': 0.10,
            'stability': 0.10
        }
        
        # Strategy results storage
        self.strategy_results: Dict[str, BacktestResult] = {}
        self.strategy_scores: Dict[str, float] = {}
        
    def evaluate_strategy(self, strategy_name: str, strategy_function: Callable,
                         data: pd.DataFrame, initial_capital: float = 100000,
                         **backtest_params) -> Dict[str, Any]:
        """
        Evaluate a single trading strategy
        
        Args:
            strategy_name: Name of the strategy
            strategy_function: Strategy function to evaluate
            data: Historical market data
            initial_capital: Initial capital for backtesting
            **backtest_params: Additional backtesting parameters
            
        Returns:
            Strategy evaluation results
        """
        try:
            self.logger.info(f"Evaluating strategy: {strategy_name}")
            
            # Create backtester
            backtester = Backtester(initial_capital=initial_capital)
            backtester.set_strategy(strategy_function)
            
            # Set backtest parameters
            if 'commission_rate' in backtest_params:
                backtester.set_commission(backtest_params['commission_rate'])
            if 'slippage_rate' in backtest_params:
                backtester.set_slippage(backtest_params['slippage_rate'])
            
            # Run backtest
            result = asyncio.run(backtester.run_backtest(data))
            
            # Store result
            self.strategy_results[strategy_name] = result
            
            # Analyze performance
            analysis = self.performance_analyzer.analyze_performance(result)
            
            # Calculate strategy score
            score = self._calculate_strategy_score(result, analysis)
            self.strategy_scores[strategy_name] = score
            
            evaluation = {
                'strategy_name': strategy_name,
                'backtest_result': result,
                'performance_analysis': analysis,
                'strategy_score': score,
                'evaluation_timestamp': datetime.now()
            }
            
            self.logger.info(f"Strategy {strategy_name} evaluated. Score: {score:.3f}")
            return evaluation
            
        except Exception as e:
            self.logger.error(f"Error evaluating strategy {strategy_name}: {e}")
            return {}
    
    def compare_strategies(self, strategy_evaluations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Compare multiple strategies
        
        Args:
            strategy_evaluations: List of strategy evaluation results
            
        Returns:
            Strategy comparison results
        """
        try:
            if len(strategy_evaluations) < 2:
                raise ValueError("Need at least 2 strategies to compare")
            
            comparison = {
                'strategy_count': len(strategy_evaluations),
                'comparison_metrics': {},
                'rankings': {},
                'statistical_tests': {},
                'correlation_analysis': {},
                'risk_return_analysis': {}
            }
            
            # Extract metrics for comparison
            metrics_data = {}
            for eval_result in strategy_evaluations:
                strategy_name = eval_result['strategy_name']
                result = eval_result['backtest_result']
                analysis = eval_result['performance_analysis']
                
                metrics_data[strategy_name] = {
                    'total_return': result.total_return_percentage,
                    'sharpe_ratio': result.sharpe_ratio,
                    'max_drawdown': result.max_drawdown_percentage,
                    'win_rate': result.win_rate,
                    'profit_factor': result.profit_factor,
                    'calmar_ratio': result.calmar_ratio,
                    'volatility': analysis.get('basic_metrics', {}).get('volatility', 0),
                    'strategy_score': eval_result['strategy_score']
                }
            
            # Create comparison metrics
            comparison['comparison_metrics'] = self._create_comparison_table(metrics_data)
            
            # Rank strategies
            comparison['rankings'] = self._rank_strategies(metrics_data)
            
            # Statistical significance tests
            comparison['statistical_tests'] = self._perform_statistical_tests(strategy_evaluations)
            
            # Correlation analysis
            comparison['correlation_analysis'] = self._analyze_correlations(strategy_evaluations)
            
            # Risk-return analysis
            comparison['risk_return_analysis'] = self._analyze_risk_return(metrics_data)
            
            return comparison
            
        except Exception as e:
            self.logger.error(f"Error comparing strategies: {e}")
            return {}
    
    def optimize_strategy_parameters(self, strategy_function: Callable, data: pd.DataFrame,
                                   parameter_ranges: Dict[str, List], 
                                   optimization_metric: str = 'sharpe_ratio') -> Dict[str, Any]:
        """
        Optimize strategy parameters using grid search
        
        Args:
            strategy_function: Strategy function to optimize
            data: Historical market data
            parameter_ranges: Dictionary of parameter ranges to test
            optimization_metric: Metric to optimize for
            
        Returns:
            Optimization results
        """
        try:
            self.logger.info("Starting strategy parameter optimization...")
            
            # Generate parameter combinations
            param_combinations = self._generate_parameter_combinations(parameter_ranges)
            
            optimization_results = []
            best_score = float('-inf')
            best_params = None
            
            for i, params in enumerate(param_combinations):
                self.logger.info(f"Testing parameter combination {i+1}/{len(param_combinations)}: {params}")
                
                # Create parameterized strategy
                def parameterized_strategy(data):
                    return strategy_function(data, **params)
                
                # Evaluate strategy
                backtester = Backtester()
                backtester.set_strategy(parameterized_strategy)
                result = asyncio.run(backtester.run_backtest(data))
                
                # Get optimization metric value
                metric_value = self._get_metric_value(result, optimization_metric)
                
                optimization_results.append({
                    'parameters': params,
                    'metric_value': metric_value,
                    'result': result
                })
                
                # Track best parameters
                if metric_value > best_score:
                    best_score = metric_value
                    best_params = params
            
            # Analyze optimization results
            optimization_analysis = {
                'best_parameters': best_params,
                'best_score': best_score,
                'optimization_metric': optimization_metric,
                'total_combinations_tested': len(param_combinations),
                'parameter_sensitivity': self._analyze_parameter_sensitivity(optimization_results, parameter_ranges),
                'optimization_surface': self._create_optimization_surface(optimization_results, parameter_ranges),
                'all_results': optimization_results
            }
            
            self.logger.info(f"Optimization completed. Best {optimization_metric}: {best_score:.4f}")
            return optimization_analysis
            
        except Exception as e:
            self.logger.error(f"Error optimizing strategy parameters: {e}")
            return {}
    
    def walk_forward_analysis(self, strategy_function: Callable, data: pd.DataFrame,
                            train_period_days: int = 252, test_period_days: int = 63,
                            step_days: int = 21) -> Dict[str, Any]:
        """
        Perform walk-forward analysis
        
        Args:
            strategy_function: Strategy function to test
            data: Historical market data
            train_period_days: Training period length in days
            test_period_days: Testing period length in days
            step_days: Step size in days
            
        Returns:
            Walk-forward analysis results
        """
        try:
            self.logger.info("Starting walk-forward analysis...")
            
            # Ensure data is sorted by date
            data = data.sort_index()
            
            walk_forward_results = []
            start_date = data.index[0]
            end_date = data.index[-1]
            
            current_date = start_date + timedelta(days=train_period_days)
            
            while current_date + timedelta(days=test_period_days) <= end_date:
                # Define training and testing periods
                train_start = current_date - timedelta(days=train_period_days)
                train_end = current_date
                test_start = current_date
                test_end = current_date + timedelta(days=test_period_days)
                
                # Extract training and testing data
                train_data = data[(data.index >= train_start) & (data.index < train_end)]
                test_data = data[(data.index >= test_start) & (data.index < test_end)]
                
                if len(train_data) == 0 or len(test_data) == 0:
                    current_date += timedelta(days=step_days)
                    continue
                
                # Train strategy (if applicable)
                # For now, we'll just test the strategy on out-of-sample data
                
                # Test strategy
                backtester = Backtester()
                backtester.set_strategy(strategy_function)
                test_result = asyncio.run(backtester.run_backtest(test_data))
                
                walk_forward_results.append({
                    'train_start': train_start,
                    'train_end': train_end,
                    'test_start': test_start,
                    'test_end': test_end,
                    'test_result': test_result,
                    'out_of_sample_return': test_result.total_return_percentage,
                    'out_of_sample_sharpe': test_result.sharpe_ratio,
                    'out_of_sample_max_dd': test_result.max_drawdown_percentage
                })
                
                current_date += timedelta(days=step_days)
            
            # Analyze walk-forward results
            analysis = self._analyze_walk_forward_results(walk_forward_results)
            
            self.logger.info(f"Walk-forward analysis completed. {len(walk_forward_results)} periods tested.")
            return analysis
            
        except Exception as e:
            self.logger.error(f"Error in walk-forward analysis: {e}")
            return {}
    
    def monte_carlo_analysis(self, strategy_function: Callable, data: pd.DataFrame,
                           num_simulations: int = 1000, bootstrap_length: int = 252) -> Dict[str, Any]:
        """
        Perform Monte Carlo analysis using bootstrap resampling
        
        Args:
            strategy_function: Strategy function to test
            data: Historical market data
            num_simulations: Number of Monte Carlo simulations
            bootstrap_length: Length of each bootstrap sample
            
        Returns:
            Monte Carlo analysis results
        """
        try:
            self.logger.info(f"Starting Monte Carlo analysis with {num_simulations} simulations...")
            
            simulation_results = []
            
            for i in range(num_simulations):
                if i % 100 == 0:
                    self.logger.info(f"Completed {i}/{num_simulations} simulations")
                
                # Bootstrap sample
                bootstrap_data = self._bootstrap_sample(data, bootstrap_length)
                
                # Run backtest on bootstrap sample
                backtester = Backtester()
                backtester.set_strategy(strategy_function)
                result = asyncio.run(backtester.run_backtest(bootstrap_data))
                
                simulation_results.append({
                    'simulation_id': i,
                    'total_return': result.total_return_percentage,
                    'sharpe_ratio': result.sharpe_ratio,
                    'max_drawdown': result.max_drawdown_percentage,
                    'win_rate': result.win_rate,
                    'profit_factor': result.profit_factor
                })
            
            # Analyze Monte Carlo results
            analysis = self._analyze_monte_carlo_results(simulation_results)
            
            self.logger.info("Monte Carlo analysis completed.")
            return analysis
            
        except Exception as e:
            self.logger.error(f"Error in Monte Carlo analysis: {e}")
            return {}
    
    def _calculate_strategy_score(self, result: BacktestResult, analysis: Dict[str, Any]) -> float:
        """Calculate overall strategy score"""
        try:
            basic_metrics = analysis.get('basic_metrics', {})
            
            # Normalize metrics to 0-1 scale
            normalized_metrics = {
                'total_return': min(max(result.total_return_percentage / 0.5, 0), 1),  # Cap at 50%
                'sharpe_ratio': min(max(result.sharpe_ratio / 3.0, 0), 1),  # Cap at 3.0
                'max_drawdown': max(1 - abs(result.max_drawdown_percentage) / 0.3, 0),  # Penalty for >30% DD
                'win_rate': result.win_rate,
                'profit_factor': min(max((result.profit_factor - 1) / 2, 0), 1),  # Normalize around 1.0
                'calmar_ratio': min(max(result.calmar_ratio / 2.0, 0), 1),  # Cap at 2.0
                'stability': min(max(basic_metrics.get('stability_ratio', 0) / 2.0, 0), 1)  # Cap at 2.0
            }
            
            # Calculate weighted score
            score = sum(normalized_metrics[metric] * weight 
                       for metric, weight in self.evaluation_weights.items()
                       if metric in normalized_metrics)
            
            return score
            
        except Exception as e:
            self.logger.error(f"Error calculating strategy score: {e}")
            return 0
    
    def _create_comparison_table(self, metrics_data: Dict[str, Dict[str, float]]) -> pd.DataFrame:
        """Create comparison table of strategy metrics"""
        try:
            df = pd.DataFrame(metrics_data).T
            df = df.round(4)
            return df
            
        except Exception as e:
            self.logger.error(f"Error creating comparison table: {e}")
            return pd.DataFrame()
    
    def _rank_strategies(self, metrics_data: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, int]]:
        """Rank strategies by different metrics"""
        try:
            rankings = {}
            
            for metric in ['total_return', 'sharpe_ratio', 'win_rate', 'profit_factor', 'strategy_score']:
                if metric in list(metrics_data.values())[0]:
                    # Sort strategies by metric (descending for positive metrics)
                    sorted_strategies = sorted(metrics_data.items(), 
                                             key=lambda x: x[1][metric], 
                                             reverse=True)
                    
                    rankings[metric] = {strategy: rank + 1 
                                      for rank, (strategy, _) in enumerate(sorted_strategies)}
            
            # Special handling for max_drawdown (ascending - lower is better)
            if 'max_drawdown' in list(metrics_data.values())[0]:
                sorted_strategies = sorted(metrics_data.items(), 
                                         key=lambda x: abs(x[1]['max_drawdown']))
                rankings['max_drawdown'] = {strategy: rank + 1 
                                          for rank, (strategy, _) in enumerate(sorted_strategies)}
            
            return rankings
            
        except Exception as e:
            self.logger.error(f"Error ranking strategies: {e}")
            return {}
    
    def _perform_statistical_tests(self, strategy_evaluations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Perform statistical significance tests"""
        try:
            from scipy import stats
            
            tests = {}
            
            # Extract daily returns for each strategy
            strategy_returns = {}
            for eval_result in strategy_evaluations:
                strategy_name = eval_result['strategy_name']
                returns = eval_result['backtest_result'].daily_returns
                strategy_returns[strategy_name] = returns
            
            # Pairwise t-tests
            strategy_names = list(strategy_returns.keys())
            pairwise_tests = {}
            
            for i in range(len(strategy_names)):
                for j in range(i + 1, len(strategy_names)):
                    strategy1 = strategy_names[i]
                    strategy2 = strategy_names[j]
                    
                    returns1 = strategy_returns[strategy1]
                    returns2 = strategy_returns[strategy2]
                    
                    if len(returns1) > 1 and len(returns2) > 1:
                        t_stat, p_value = stats.ttest_ind(returns1, returns2)
                        
                        pairwise_tests[f"{strategy1}_vs_{strategy2}"] = {
                            't_statistic': t_stat,
                            'p_value': p_value,
                            'significant': p_value < 0.05
                        }
            
            tests['pairwise_t_tests'] = pairwise_tests
            
            return tests
            
        except Exception as e:
            self.logger.error(f"Error performing statistical tests: {e}")
            return {}
    
    def _analyze_correlations(self, strategy_evaluations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze correlations between strategy returns"""
        try:
            # Extract daily returns
            returns_data = {}
            for eval_result in strategy_evaluations:
                strategy_name = eval_result['strategy_name']
                returns = eval_result['backtest_result'].daily_returns
                returns_data[strategy_name] = returns
            
            # Create DataFrame
            min_length = min(len(returns) for returns in returns_data.values())
            
            df_returns = pd.DataFrame({
                strategy: returns[:min_length] 
                for strategy, returns in returns_data.items()
            })
            
            # Calculate correlation matrix
            correlation_matrix = df_returns.corr()
            
            analysis = {
                'correlation_matrix': correlation_matrix.to_dict(),
                'average_correlation': correlation_matrix.values[np.triu_indices_from(correlation_matrix.values, k=1)].mean(),
                'max_correlation': correlation_matrix.values[np.triu_indices_from(correlation_matrix.values, k=1)].max(),
                'min_correlation': correlation_matrix.values[np.triu_indices_from(correlation_matrix.values, k=1)].min()
            }
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Error analyzing correlations: {e}")
            return {}
    
    def _analyze_risk_return(self, metrics_data: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
        """Analyze risk-return characteristics"""
        try:
            analysis = {
                'efficient_frontier': [],
                'risk_return_ratios': {},
                'risk_adjusted_rankings': {}
            }
            
            # Calculate risk-adjusted metrics
            for strategy, metrics in metrics_data.items():
                volatility = metrics.get('volatility', 0)
                total_return = metrics.get('total_return', 0)
                
                analysis['risk_return_ratios'][strategy] = {
                    'return_to_risk': total_return / volatility if volatility > 0 else 0,
                    'return': total_return,
                    'risk': volatility
                }
            
            # Rank by risk-adjusted return
            sorted_strategies = sorted(analysis['risk_return_ratios'].items(),
                                     key=lambda x: x[1]['return_to_risk'],
                                     reverse=True)
            
            analysis['risk_adjusted_rankings'] = {
                strategy: rank + 1 
                for rank, (strategy, _) in enumerate(sorted_strategies)
            }
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Error analyzing risk-return: {e}")
            return {}
    
    def _generate_parameter_combinations(self, parameter_ranges: Dict[str, List]) -> List[Dict[str, Any]]:
        """Generate all parameter combinations for grid search"""
        try:
            import itertools
            
            param_names = list(parameter_ranges.keys())
            param_values = list(parameter_ranges.values())
            
            combinations = []
            for combination in itertools.product(*param_values):
                param_dict = dict(zip(param_names, combination))
                combinations.append(param_dict)
            
            return combinations
            
        except Exception as e:
            self.logger.error(f"Error generating parameter combinations: {e}")
            return []
    
    def _get_metric_value(self, result: BacktestResult, metric_name: str) -> float:
        """Get metric value from backtest result"""
        try:
            metric_mapping = {
                'total_return': result.total_return_percentage,
                'sharpe_ratio': result.sharpe_ratio,
                'calmar_ratio': result.calmar_ratio,
                'sortino_ratio': result.sortino_ratio,
                'max_drawdown': -result.max_drawdown_percentage,  # Negative because lower is better
                'win_rate': result.win_rate,
                'profit_factor': result.profit_factor
            }
            
            return metric_mapping.get(metric_name, 0)
            
        except Exception as e:
            self.logger.error(f"Error getting metric value: {e}")
            return 0
    
    def _analyze_parameter_sensitivity(self, optimization_results: List[Dict[str, Any]], 
                                     parameter_ranges: Dict[str, List]) -> Dict[str, Any]:
        """Analyze parameter sensitivity"""
        try:
            sensitivity = {}
            
            for param_name in parameter_ranges.keys():
                param_values = []
                metric_values = []
                
                for result in optimization_results:
                    param_values.append(result['parameters'][param_name])
                    metric_values.append(result['metric_value'])
                
                # Calculate correlation between parameter and metric
                if len(set(param_values)) > 1:  # Only if parameter varies
                    correlation = np.corrcoef(param_values, metric_values)[0, 1]
                    sensitivity[param_name] = {
                        'correlation': correlation,
                        'sensitivity': abs(correlation)
                    }
            
            return sensitivity
            
        except Exception as e:
            self.logger.error(f"Error analyzing parameter sensitivity: {e}")
            return {}
    
    def _create_optimization_surface(self, optimization_results: List[Dict[str, Any]], 
                                   parameter_ranges: Dict[str, List]) -> Dict[str, Any]:
        """Create optimization surface data"""
        try:
            # For 2D optimization surface (first two parameters)
            param_names = list(parameter_ranges.keys())[:2]
            
            if len(param_names) < 2:
                return {}
            
            surface_data = {
                'param1_name': param_names[0],
                'param2_name': param_names[1],
                'param1_values': [],
                'param2_values': [],
                'metric_values': []
            }
            
            for result in optimization_results:
                surface_data['param1_values'].append(result['parameters'][param_names[0]])
                surface_data['param2_values'].append(result['parameters'][param_names[1]])
                surface_data['metric_values'].append(result['metric_value'])
            
            return surface_data
            
        except Exception as e:
            self.logger.error(f"Error creating optimization surface: {e}")
            return {}
    
    def _analyze_walk_forward_results(self, walk_forward_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze walk-forward analysis results"""
        try:
            returns = [r['out_of_sample_return'] for r in walk_forward_results]
            sharpe_ratios = [r['out_of_sample_sharpe'] for r in walk_forward_results]
            max_drawdowns = [r['out_of_sample_max_dd'] for r in walk_forward_results]
            
            analysis = {
                'total_periods': len(walk_forward_results),
                'avg_out_of_sample_return': np.mean(returns),
                'std_out_of_sample_return': np.std(returns),
                'avg_sharpe_ratio': np.mean(sharpe_ratios),
                'std_sharpe_ratio': np.std(sharpe_ratios),
                'avg_max_drawdown': np.mean(max_drawdowns),
                'consistency_score': len([r for r in returns if r > 0]) / len(returns),
                'worst_period_return': min(returns),
                'best_period_return': max(returns),
                'periods_data': walk_forward_results
            }
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Error analyzing walk-forward results: {e}")
            return {}
    
    def _bootstrap_sample(self, data: pd.DataFrame, sample_length: int) -> pd.DataFrame:
        """Create bootstrap sample from data"""
        try:
            # Random sampling with replacement
            sample_indices = np.random.choice(len(data), size=sample_length, replace=True)
            bootstrap_data = data.iloc[sample_indices].copy()
            
            # Reset index to maintain chronological order
            bootstrap_data.reset_index(drop=True, inplace=True)
            bootstrap_data.index = pd.date_range(start=data.index[0], periods=sample_length, freq='D')
            
            return bootstrap_data
            
        except Exception as e:
            self.logger.error(f"Error creating bootstrap sample: {e}")
            return pd.DataFrame()
    
    def _analyze_monte_carlo_results(self, simulation_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze Monte Carlo simulation results"""
        try:
            returns = [r['total_return'] for r in simulation_results]
            sharpe_ratios = [r['sharpe_ratio'] for r in simulation_results]
            max_drawdowns = [r['max_drawdown'] for r in simulation_results]
            
            analysis = {
                'num_simulations': len(simulation_results),
                'return_statistics': {
                    'mean': np.mean(returns),
                    'std': np.std(returns),
                    'min': np.min(returns),
                    'max': np.max(returns),
                    'percentile_5': np.percentile(returns, 5),
                    'percentile_95': np.percentile(returns, 95),
                    'probability_positive': len([r for r in returns if r > 0]) / len(returns)
                },
                'sharpe_statistics': {
                    'mean': np.mean(sharpe_ratios),
                    'std': np.std(sharpe_ratios),
                    'min': np.min(sharpe_ratios),
                    'max': np.max(sharpe_ratios),
                    'percentile_5': np.percentile(sharpe_ratios, 5),
                    'percentile_95': np.percentile(sharpe_ratios, 95)
                },
                'drawdown_statistics': {
                    'mean': np.mean(max_drawdowns),
                    'std': np.std(max_drawdowns),
                    'min': np.min(max_drawdowns),
                    'max': np.max(max_drawdowns),
                    'percentile_5': np.percentile(max_drawdowns, 5),
                    'percentile_95': np.percentile(max_drawdowns, 95)
                },
                'confidence_intervals': {
                    'return_95_ci': [np.percentile(returns, 2.5), np.percentile(returns, 97.5)],
                    'sharpe_95_ci': [np.percentile(sharpe_ratios, 2.5), np.percentile(sharpe_ratios, 97.5)]
                },
                'simulation_data': simulation_results
            }
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Error analyzing Monte Carlo results: {e}")
            return {}
    
    def save_evaluation_results(self, results: Dict[str, Any], filepath: str):
        """Save evaluation results to file"""
        try:
            with open(filepath, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            self.logger.info(f"Evaluation results saved to {filepath}")
            
        except Exception as e:
            self.logger.error(f"Error saving evaluation results: {e}")

if __name__ == "__main__":
    # Test the strategy evaluator
    import asyncio
    
    async def test_strategy_1(data):
        """Simple test strategy 1"""
        import random
        if random.random() > 0.98:
            from trading.execution_engine import TradeSignal, SignalDirection, SignalStrength
            return TradeSignal(
                symbol=data['symbol'],
                direction=SignalDirection.BULLISH,
                strength=SignalStrength.MODERATE,
                confidence=0.7,
                entry_price=data['close'],
                stop_loss=data['close'] * 0.98,
                target_price=data['close'] * 1.05,
                quantity=50
            )
        return None
    
    async def test_strategy_2(data):
        """Simple test strategy 2"""
        import random
        if random.random() > 0.97:
            from trading.execution_engine import TradeSignal, SignalDirection, SignalStrength
            return TradeSignal(
                symbol=data['symbol'],
                direction=SignalDirection.BEARISH,
                strength=SignalStrength.STRONG,
                confidence=0.8,
                entry_price=data['close'],
                stop_loss=data['close'] * 1.02,
                target_price=data['close'] * 0.95,
                quantity=50
            )
        return None
    
    # Create sample data
    dates = pd.date_range('2023-01-01', '2023-06-30', freq='D')
    np.random.seed(42)
    
    data = pd.DataFrame({
        'symbol': 'NIFTY',
        'open': 19000 + np.random.randn(len(dates)) * 100,
        'high': 19100 + np.random.randn(len(dates)) * 100,
        'low': 18900 + np.random.randn(len(dates)) * 100,
        'close': 19000 + np.random.randn(len(dates)) * 100,
        'volume': 1000000 + np.random.randint(0, 500000, len(dates))
    }, index=dates)
    
    # Ensure price consistency
    data['high'] = np.maximum(data[['open', 'close']].max(axis=1), data['high'])
    data['low'] = np.minimum(data[['open', 'close']].min(axis=1), data['low'])
    
    # Create evaluator
    evaluator = StrategyEvaluator()
    
    # Evaluate strategies
    eval1 = evaluator.evaluate_strategy("Strategy_1", test_strategy_1, data)
    eval2 = evaluator.evaluate_strategy("Strategy_2", test_strategy_2, data)
    
    # Compare strategies
    comparison = evaluator.compare_strategies([eval1, eval2])
    
    print("Strategy Evaluation Results:")
    print(f"Strategy 1 Score: {eval1.get('strategy_score', 0):.3f}")
    print(f"Strategy 2 Score: {eval2.get('strategy_score', 0):.3f}")
    
    print("\nStrategy Rankings:")
    rankings = comparison.get('rankings', {})
    for metric, ranking in rankings.items():
        print(f"{metric}: {ranking}")
    
    # Save results
    evaluator.save_evaluation_results(comparison, "/tmp/strategy_comparison.json")
    print("\nComparison results saved to /tmp/strategy_comparison.json")

