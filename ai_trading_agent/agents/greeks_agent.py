"""
Options Greeks Calculation and Analysis Agent for AI Trading Agent
Calculates and analyzes options Greeks (Delta, Gamma, Theta, Vega, Rho)
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import math
from scipy.stats import norm
from scipy.optimize import minimize_scalar
import warnings
warnings.filterwarnings('ignore')

from utils.logger import get_logger
from config.config import Config

logger = get_logger(__name__)

class BlackScholesCalculator:
    """Black-Scholes model implementation for options pricing and Greeks"""
    
    def __init__(self):
        self.logger = get_logger(__name__)
    
    def black_scholes_price(self, S: float, K: float, T: float, r: float, 
                           sigma: float, option_type: str = 'call') -> float:
        """
        Calculate Black-Scholes option price
        
        Args:
            S: Current stock price
            K: Strike price
            T: Time to expiration (in years)
            r: Risk-free rate
            sigma: Volatility
            option_type: 'call' or 'put'
            
        Returns:
            Option price
        """
        try:
            if T <= 0:
                # Handle expiration case
                if option_type.lower() == 'call':
                    return max(S - K, 0)
                else:
                    return max(K - S, 0)
            
            d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
            d2 = d1 - sigma * np.sqrt(T)
            
            if option_type.lower() == 'call':
                price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
            else:  # put
                price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
            
            return max(price, 0)  # Ensure non-negative price
            
        except Exception as e:
            self.logger.error(f"Error calculating Black-Scholes price: {e}")
            return 0.0
    
    def calculate_delta(self, S: float, K: float, T: float, r: float, 
                       sigma: float, option_type: str = 'call') -> float:
        """
        Calculate Delta (price sensitivity to underlying price change)
        
        Returns:
            Delta value
        """
        try:
            if T <= 0:
                if option_type.lower() == 'call':
                    return 1.0 if S > K else 0.0
                else:
                    return -1.0 if S < K else 0.0
            
            d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
            
            if option_type.lower() == 'call':
                delta = norm.cdf(d1)
            else:  # put
                delta = norm.cdf(d1) - 1
            
            return delta
            
        except Exception as e:
            self.logger.error(f"Error calculating Delta: {e}")
            return 0.0
    
    def calculate_gamma(self, S: float, K: float, T: float, r: float, sigma: float) -> float:
        """
        Calculate Gamma (rate of change of Delta)
        
        Returns:
            Gamma value
        """
        try:
            if T <= 0:
                return 0.0
            
            d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
            gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
            
            return gamma
            
        except Exception as e:
            self.logger.error(f"Error calculating Gamma: {e}")
            return 0.0
    
    def calculate_theta(self, S: float, K: float, T: float, r: float, 
                       sigma: float, option_type: str = 'call') -> float:
        """
        Calculate Theta (time decay)
        
        Returns:
            Theta value (per day)
        """
        try:
            if T <= 0:
                return 0.0
            
            d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
            d2 = d1 - sigma * np.sqrt(T)
            
            if option_type.lower() == 'call':
                theta = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) 
                        - r * K * np.exp(-r * T) * norm.cdf(d2))
            else:  # put
                theta = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) 
                        + r * K * np.exp(-r * T) * norm.cdf(-d2))
            
            # Convert to per-day theta
            theta_per_day = theta / 365
            
            return theta_per_day
            
        except Exception as e:
            self.logger.error(f"Error calculating Theta: {e}")
            return 0.0
    
    def calculate_vega(self, S: float, K: float, T: float, r: float, sigma: float) -> float:
        """
        Calculate Vega (sensitivity to volatility)
        
        Returns:
            Vega value
        """
        try:
            if T <= 0:
                return 0.0
            
            d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
            vega = S * norm.pdf(d1) * np.sqrt(T) / 100  # Per 1% change in volatility
            
            return vega
            
        except Exception as e:
            self.logger.error(f"Error calculating Vega: {e}")
            return 0.0
    
    def calculate_rho(self, S: float, K: float, T: float, r: float, 
                     sigma: float, option_type: str = 'call') -> float:
        """
        Calculate Rho (sensitivity to interest rate)
        
        Returns:
            Rho value
        """
        try:
            if T <= 0:
                return 0.0
            
            d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
            d2 = d1 - sigma * np.sqrt(T)
            
            if option_type.lower() == 'call':
                rho = K * T * np.exp(-r * T) * norm.cdf(d2) / 100  # Per 1% change in rate
            else:  # put
                rho = -K * T * np.exp(-r * T) * norm.cdf(-d2) / 100
            
            return rho
            
        except Exception as e:
            self.logger.error(f"Error calculating Rho: {e}")
            return 0.0
    
    def calculate_all_greeks(self, S: float, K: float, T: float, r: float, 
                           sigma: float, option_type: str = 'call') -> Dict[str, float]:
        """
        Calculate all Greeks for an option
        
        Returns:
            Dictionary with all Greeks
        """
        return {
            'price': self.black_scholes_price(S, K, T, r, sigma, option_type),
            'delta': self.calculate_delta(S, K, T, r, sigma, option_type),
            'gamma': self.calculate_gamma(S, K, T, r, sigma),
            'theta': self.calculate_theta(S, K, T, r, sigma, option_type),
            'vega': self.calculate_vega(S, K, T, r, sigma),
            'rho': self.calculate_rho(S, K, T, r, sigma, option_type)
        }
    
    def implied_volatility(self, market_price: float, S: float, K: float, T: float, 
                          r: float, option_type: str = 'call', 
                          max_iterations: int = 100, tolerance: float = 1e-6) -> float:
        """
        Calculate implied volatility using Newton-Raphson method
        
        Args:
            market_price: Market price of the option
            S: Current stock price
            K: Strike price
            T: Time to expiration
            r: Risk-free rate
            option_type: 'call' or 'put'
            max_iterations: Maximum iterations
            tolerance: Convergence tolerance
            
        Returns:
            Implied volatility
        """
        try:
            if T <= 0 or market_price <= 0:
                return 0.0
            
            # Initial guess
            sigma = 0.2
            
            for i in range(max_iterations):
                # Calculate price and vega
                price = self.black_scholes_price(S, K, T, r, sigma, option_type)
                vega = self.calculate_vega(S, K, T, r, sigma)
                
                # Price difference
                price_diff = price - market_price
                
                # Check convergence
                if abs(price_diff) < tolerance:
                    return sigma
                
                # Newton-Raphson update
                if vega != 0:
                    sigma = sigma - price_diff / (vega * 100)  # vega is per 1% change
                else:
                    break
                
                # Ensure sigma stays positive
                sigma = max(sigma, 0.001)
                sigma = min(sigma, 5.0)  # Cap at 500%
            
            return sigma
            
        except Exception as e:
            self.logger.error(f"Error calculating implied volatility: {e}")
            return 0.2  # Return default volatility

class VolatilitySurfaceAnalyzer:
    """Analyzes implied volatility surface for trading opportunities"""
    
    def __init__(self):
        self.logger = get_logger(__name__)
    
    def build_volatility_surface(self, options_data: pd.DataFrame, 
                                underlying_price: float) -> Dict[str, Any]:
        """
        Build implied volatility surface from options data
        
        Args:
            options_data: DataFrame with options data
            underlying_price: Current underlying price
            
        Returns:
            Volatility surface data
        """
        try:
            if options_data.empty:
                return {}
            
            bs_calc = BlackScholesCalculator()
            surface_data = []
            
            for _, row in options_data.iterrows():
                try:
                    # Extract option parameters
                    strike = row.get('strike', 0)
                    expiry = pd.to_datetime(row.get('expiry'))
                    option_type = row.get('option_type', '').lower()
                    market_price = row.get('last_price', 0)
                    
                    if market_price <= 0 or strike <= 0:
                        continue
                    
                    # Calculate time to expiry
                    time_to_expiry = (expiry - datetime.now()).total_seconds() / (365.25 * 24 * 3600)
                    
                    if time_to_expiry <= 0:
                        continue
                    
                    # Calculate moneyness
                    moneyness = strike / underlying_price
                    
                    # Calculate implied volatility
                    risk_free_rate = 0.06  # Assume 6% risk-free rate
                    implied_vol = bs_calc.implied_volatility(
                        market_price, underlying_price, strike, time_to_expiry,
                        risk_free_rate, option_type
                    )
                    
                    surface_data.append({
                        'strike': strike,
                        'expiry': expiry,
                        'time_to_expiry': time_to_expiry,
                        'moneyness': moneyness,
                        'option_type': option_type,
                        'market_price': market_price,
                        'implied_volatility': implied_vol,
                        'volume': row.get('volume', 0),
                        'open_interest': row.get('open_interest', 0)
                    })
                    
                except Exception as e:
                    continue
            
            if not surface_data:
                return {}
            
            surface_df = pd.DataFrame(surface_data)
            
            # Analyze surface characteristics
            analysis = self._analyze_surface_characteristics(surface_df)
            
            return {
                'surface_data': surface_df,
                'analysis': analysis,
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            self.logger.error(f"Error building volatility surface: {e}")
            return {}
    
    def _analyze_surface_characteristics(self, surface_df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze volatility surface characteristics"""
        try:
            analysis = {}
            
            # Volatility smile/skew analysis
            atm_options = surface_df[
                (surface_df['moneyness'] >= 0.95) & 
                (surface_df['moneyness'] <= 1.05)
            ]
            
            if not atm_options.empty:
                atm_vol = atm_options['implied_volatility'].mean()
                analysis['atm_volatility'] = atm_vol
            else:
                analysis['atm_volatility'] = surface_df['implied_volatility'].mean()
            
            # Skew analysis (OTM puts vs OTM calls)
            otm_puts = surface_df[
                (surface_df['option_type'] == 'pe') & 
                (surface_df['moneyness'] < 0.95)
            ]
            otm_calls = surface_df[
                (surface_df['option_type'] == 'ce') & 
                (surface_df['moneyness'] > 1.05)
            ]
            
            if not otm_puts.empty and not otm_calls.empty:
                put_vol = otm_puts['implied_volatility'].mean()
                call_vol = otm_calls['implied_volatility'].mean()
                analysis['volatility_skew'] = put_vol - call_vol
            else:
                analysis['volatility_skew'] = 0.0
            
            # Term structure analysis
            expiry_groups = surface_df.groupby('time_to_expiry')['implied_volatility'].mean()
            if len(expiry_groups) > 1:
                analysis['term_structure_slope'] = (
                    expiry_groups.iloc[-1] - expiry_groups.iloc[0]
                ) / (expiry_groups.index[-1] - expiry_groups.index[0])
            else:
                analysis['term_structure_slope'] = 0.0
            
            # Volatility range
            analysis['vol_range'] = {
                'min': surface_df['implied_volatility'].min(),
                'max': surface_df['implied_volatility'].max(),
                'std': surface_df['implied_volatility'].std()
            }
            
            # Liquidity analysis
            high_volume_options = surface_df[surface_df['volume'] > surface_df['volume'].quantile(0.75)]
            if not high_volume_options.empty:
                analysis['liquid_vol_avg'] = high_volume_options['implied_volatility'].mean()
            else:
                analysis['liquid_vol_avg'] = analysis['atm_volatility']
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Error analyzing surface characteristics: {e}")
            return {}

class GreeksAgent:
    """Options Greeks Calculation and Analysis Agent"""
    
    def __init__(self):
        self.bs_calculator = BlackScholesCalculator()
        self.vol_analyzer = VolatilitySurfaceAnalyzer()
        self.logger = get_logger(__name__)
        self.greeks_history = []
        self.risk_free_rate = 0.06  # Default 6% risk-free rate
    
    def calculate_portfolio_greeks(self, options_data: pd.DataFrame, 
                                 underlying_price: float) -> Dict[str, Any]:
        """
        Calculate Greeks for entire options portfolio
        
        Args:
            options_data: DataFrame with options data
            underlying_price: Current underlying price
            
        Returns:
            Portfolio Greeks analysis
        """
        try:
            if options_data.empty:
                return self._get_empty_greeks_result()
            
            portfolio_greeks = {
                'total_delta': 0.0,
                'total_gamma': 0.0,
                'total_theta': 0.0,
                'total_vega': 0.0,
                'total_rho': 0.0,
                'option_details': []
            }
            
            for _, row in options_data.iterrows():
                try:
                    # Extract option parameters
                    strike = row.get('strike', 0)
                    expiry = pd.to_datetime(row.get('expiry'))
                    option_type = row.get('option_type', '').lower()
                    market_price = row.get('last_price', 0)
                    volume = row.get('volume', 0)
                    open_interest = row.get('open_interest', 0)
                    
                    if strike <= 0:
                        continue
                    
                    # Calculate time to expiry
                    time_to_expiry = (expiry - datetime.now()).total_seconds() / (365.25 * 24 * 3600)
                    
                    if time_to_expiry <= 0:
                        continue
                    
                    # Calculate implied volatility
                    if market_price > 0:
                        implied_vol = self.bs_calculator.implied_volatility(
                            market_price, underlying_price, strike, time_to_expiry,
                            self.risk_free_rate, option_type
                        )
                    else:
                        implied_vol = 0.2  # Default volatility
                    
                    # Calculate all Greeks
                    greeks = self.bs_calculator.calculate_all_greeks(
                        underlying_price, strike, time_to_expiry,
                        self.risk_free_rate, implied_vol, option_type
                    )
                    
                    # Add option details
                    option_detail = {
                        'strike': strike,
                        'expiry': expiry,
                        'option_type': option_type,
                        'time_to_expiry': time_to_expiry,
                        'market_price': market_price,
                        'theoretical_price': greeks['price'],
                        'implied_volatility': implied_vol,
                        'delta': greeks['delta'],
                        'gamma': greeks['gamma'],
                        'theta': greeks['theta'],
                        'vega': greeks['vega'],
                        'rho': greeks['rho'],
                        'volume': volume,
                        'open_interest': open_interest,
                        'moneyness': strike / underlying_price,
                        'intrinsic_value': max(underlying_price - strike, 0) if option_type == 'ce' else max(strike - underlying_price, 0),
                        'time_value': market_price - max(underlying_price - strike, 0) if option_type == 'ce' else market_price - max(strike - underlying_price, 0)
                    }
                    
                    portfolio_greeks['option_details'].append(option_detail)
                    
                    # Aggregate portfolio Greeks (weighted by open interest)
                    weight = max(open_interest, 1)  # Minimum weight of 1
                    portfolio_greeks['total_delta'] += greeks['delta'] * weight
                    portfolio_greeks['total_gamma'] += greeks['gamma'] * weight
                    portfolio_greeks['total_theta'] += greeks['theta'] * weight
                    portfolio_greeks['total_vega'] += greeks['vega'] * weight
                    portfolio_greeks['total_rho'] += greeks['rho'] * weight
                    
                except Exception as e:
                    self.logger.warning(f"Error processing option: {e}")
                    continue
            
            # Normalize by total weight
            total_weight = sum(max(opt['open_interest'], 1) for opt in portfolio_greeks['option_details'])
            if total_weight > 0:
                portfolio_greeks['total_delta'] /= total_weight
                portfolio_greeks['total_gamma'] /= total_weight
                portfolio_greeks['total_theta'] /= total_weight
                portfolio_greeks['total_vega'] /= total_weight
                portfolio_greeks['total_rho'] /= total_weight
            
            # Add analysis
            portfolio_greeks.update(self._analyze_portfolio_greeks(portfolio_greeks))
            
            # Add volatility surface analysis
            vol_surface = self.vol_analyzer.build_volatility_surface(options_data, underlying_price)
            portfolio_greeks['volatility_surface'] = vol_surface
            
            portfolio_greeks['timestamp'] = datetime.now()
            portfolio_greeks['underlying_price'] = underlying_price
            
            # Store in history
            self.greeks_history.append(portfolio_greeks)
            if len(self.greeks_history) > 100:
                self.greeks_history = self.greeks_history[-100:]
            
            return portfolio_greeks
            
        except Exception as e:
            self.logger.error(f"Error calculating portfolio Greeks: {e}")
            return self._get_empty_greeks_result()
    
    def _analyze_portfolio_greeks(self, portfolio_greeks: Dict) -> Dict[str, Any]:
        """Analyze portfolio Greeks for risk and opportunities"""
        try:
            analysis = {}
            
            # Delta analysis
            total_delta = portfolio_greeks['total_delta']
            analysis['delta_exposure'] = {
                'value': total_delta,
                'interpretation': self._interpret_delta(total_delta),
                'risk_level': 'high' if abs(total_delta) > 0.5 else 'medium' if abs(total_delta) > 0.2 else 'low'
            }
            
            # Gamma analysis
            total_gamma = portfolio_greeks['total_gamma']
            analysis['gamma_exposure'] = {
                'value': total_gamma,
                'interpretation': self._interpret_gamma(total_gamma),
                'risk_level': 'high' if total_gamma > 0.1 else 'medium' if total_gamma > 0.05 else 'low'
            }
            
            # Theta analysis
            total_theta = portfolio_greeks['total_theta']
            analysis['theta_decay'] = {
                'value': total_theta,
                'daily_decay': total_theta,
                'interpretation': self._interpret_theta(total_theta),
                'risk_level': 'high' if abs(total_theta) > 50 else 'medium' if abs(total_theta) > 20 else 'low'
            }
            
            # Vega analysis
            total_vega = portfolio_greeks['total_vega']
            analysis['vega_exposure'] = {
                'value': total_vega,
                'interpretation': self._interpret_vega(total_vega),
                'risk_level': 'high' if abs(total_vega) > 100 else 'medium' if abs(total_vega) > 50 else 'low'
            }
            
            # Option concentration analysis
            option_details = portfolio_greeks['option_details']
            if option_details:
                # Expiry concentration
                expiry_counts = {}
                for opt in option_details:
                    expiry_date = opt['expiry'].date()
                    expiry_counts[expiry_date] = expiry_counts.get(expiry_date, 0) + 1
                
                analysis['expiry_concentration'] = {
                    'most_concentrated_expiry': max(expiry_counts.items(), key=lambda x: x[1])[0],
                    'concentration_ratio': max(expiry_counts.values()) / len(option_details)
                }
                
                # Strike concentration
                strikes = [opt['strike'] for opt in option_details]
                analysis['strike_range'] = {
                    'min_strike': min(strikes),
                    'max_strike': max(strikes),
                    'range_percentage': (max(strikes) - min(strikes)) / portfolio_greeks['underlying_price']
                }
                
                # Moneyness distribution
                moneyness_values = [opt['moneyness'] for opt in option_details]
                analysis['moneyness_distribution'] = {
                    'itm_count': len([m for m in moneyness_values if m < 0.95]),
                    'atm_count': len([m for m in moneyness_values if 0.95 <= m <= 1.05]),
                    'otm_count': len([m for m in moneyness_values if m > 1.05])
                }
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Error analyzing portfolio Greeks: {e}")
            return {}
    
    def _interpret_delta(self, delta: float) -> str:
        """Interpret delta value"""
        if delta > 0.5:
            return "Strong bullish exposure - portfolio gains significantly if underlying rises"
        elif delta > 0.2:
            return "Moderate bullish exposure - portfolio benefits from upward moves"
        elif delta > -0.2:
            return "Delta neutral - portfolio relatively insensitive to small price moves"
        elif delta > -0.5:
            return "Moderate bearish exposure - portfolio benefits from downward moves"
        else:
            return "Strong bearish exposure - portfolio gains significantly if underlying falls"
    
    def _interpret_gamma(self, gamma: float) -> str:
        """Interpret gamma value"""
        if gamma > 0.1:
            return "High gamma - delta will change rapidly with underlying price moves"
        elif gamma > 0.05:
            return "Moderate gamma - some acceleration in delta changes"
        else:
            return "Low gamma - delta changes slowly with underlying price"
    
    def _interpret_theta(self, theta: float) -> str:
        """Interpret theta value"""
        if theta < -50:
            return "High time decay - portfolio loses significant value daily"
        elif theta < -20:
            return "Moderate time decay - portfolio loses value over time"
        elif theta < 0:
            return "Low time decay - minimal daily value loss"
        else:
            return "Positive theta - portfolio gains value from time decay"
    
    def _interpret_vega(self, vega: float) -> str:
        """Interpret vega value"""
        if abs(vega) > 100:
            return "High volatility sensitivity - portfolio significantly affected by volatility changes"
        elif abs(vega) > 50:
            return "Moderate volatility sensitivity - portfolio moderately affected by volatility"
        else:
            return "Low volatility sensitivity - portfolio relatively insensitive to volatility changes"
    
    def get_greeks_signals(self, greeks_analysis: Dict) -> Dict[str, Any]:
        """
        Generate trading signals based on Greeks analysis
        
        Args:
            greeks_analysis: Greeks analysis results
            
        Returns:
            Dictionary with trading signals
        """
        try:
            signals = {
                'delta_hedge_signal': 'none',
                'gamma_scalping_signal': 'none',
                'volatility_signal': 'none',
                'time_decay_signal': 'none',
                'overall_signal': 'neutral',
                'signal_strength': 0.0,
                'recommendations': []
            }
            
            # Delta hedging signals
            delta_exposure = greeks_analysis.get('delta_exposure', {})
            delta_value = delta_exposure.get('value', 0)
            
            if abs(delta_value) > 0.5:
                signals['delta_hedge_signal'] = 'hedge_required'
                signals['recommendations'].append(f"Consider delta hedging - current delta: {delta_value:.3f}")
            
            # Gamma scalping signals
            gamma_exposure = greeks_analysis.get('gamma_exposure', {})
            gamma_value = gamma_exposure.get('value', 0)
            
            if gamma_value > 0.1:
                signals['gamma_scalping_signal'] = 'favorable'
                signals['recommendations'].append("High gamma - favorable for scalping strategies")
            
            # Volatility signals
            vol_surface = greeks_analysis.get('volatility_surface', {})
            if vol_surface:
                vol_analysis = vol_surface.get('analysis', {})
                vol_skew = vol_analysis.get('volatility_skew', 0)
                
                if abs(vol_skew) > 0.05:
                    signals['volatility_signal'] = 'skew_opportunity'
                    signals['recommendations'].append(f"Volatility skew detected: {vol_skew:.3f}")
            
            # Time decay signals
            theta_decay = greeks_analysis.get('theta_decay', {})
            theta_value = theta_decay.get('value', 0)
            
            if theta_value < -50:
                signals['time_decay_signal'] = 'high_decay'
                signals['recommendations'].append("High time decay - consider closing near-expiry positions")
            elif theta_value > 10:
                signals['time_decay_signal'] = 'positive_decay'
                signals['recommendations'].append("Positive theta - time decay working in favor")
            
            # Overall signal strength
            risk_factors = 0
            if delta_exposure.get('risk_level') == 'high':
                risk_factors += 1
            if gamma_exposure.get('risk_level') == 'high':
                risk_factors += 1
            if theta_decay.get('risk_level') == 'high':
                risk_factors += 1
            
            signals['signal_strength'] = min(risk_factors / 3, 1.0)
            
            if risk_factors >= 2:
                signals['overall_signal'] = 'high_risk'
            elif risk_factors == 1:
                signals['overall_signal'] = 'moderate_risk'
            else:
                signals['overall_signal'] = 'low_risk'
            
            signals['timestamp'] = datetime.now()
            
            return signals
            
        except Exception as e:
            self.logger.error(f"Error generating Greeks signals: {e}")
            return {
                'delta_hedge_signal': 'none',
                'gamma_scalping_signal': 'none',
                'volatility_signal': 'none',
                'time_decay_signal': 'none',
                'overall_signal': 'neutral',
                'signal_strength': 0.0,
                'recommendations': [],
                'timestamp': datetime.now()
            }
    
    def _get_empty_greeks_result(self) -> Dict[str, Any]:
        """Return empty Greeks result"""
        return {
            'total_delta': 0.0,
            'total_gamma': 0.0,
            'total_theta': 0.0,
            'total_vega': 0.0,
            'total_rho': 0.0,
            'option_details': [],
            'timestamp': datetime.now(),
            'underlying_price': 0.0
        }

if __name__ == "__main__":
    # Test the Greeks agent
    import asyncio
    
    async def test_greeks_agent():
        agent = GreeksAgent()
        
        # Test with sample options data
        options_data = pd.DataFrame([
            {
                'strike': 19500,
                'expiry': datetime.now() + timedelta(days=7),
                'option_type': 'ce',
                'last_price': 150,
                'volume': 1000,
                'open_interest': 5000
            },
            {
                'strike': 19500,
                'expiry': datetime.now() + timedelta(days=7),
                'option_type': 'pe',
                'last_price': 120,
                'volume': 800,
                'open_interest': 4000
            },
            {
                'strike': 19600,
                'expiry': datetime.now() + timedelta(days=14),
                'option_type': 'ce',
                'last_price': 80,
                'volume': 500,
                'open_interest': 3000
            }
        ])
        
        underlying_price = 19550
        
        # Calculate Greeks
        greeks_analysis = agent.calculate_portfolio_greeks(options_data, underlying_price)
        signals = agent.get_greeks_signals(greeks_analysis)
        
        print("Greeks Analysis Results:")
        print(f"Total Delta: {greeks_analysis['total_delta']:.4f}")
        print(f"Total Gamma: {greeks_analysis['total_gamma']:.4f}")
        print(f"Total Theta: {greeks_analysis['total_theta']:.2f}")
        print(f"Total Vega: {greeks_analysis['total_vega']:.2f}")
        
        print(f"\nDelta Exposure: {greeks_analysis.get('delta_exposure', {}).get('interpretation', 'N/A')}")
        print(f"Overall Signal: {signals['overall_signal']}")
        print(f"Recommendations: {signals['recommendations']}")
    
    asyncio.run(test_greeks_agent())

