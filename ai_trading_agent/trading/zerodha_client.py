"""
Zerodha API Client for AI Trading Agent
Handles all interactions with Zerodha Kite Connect API
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import time
import asyncio
from kiteconnect import KiteConnect
from kiteconnect.exceptions import KiteException
import json
import os

from utils.logger import get_logger
from config.config import Config

logger = get_logger(__name__)

class ZerodhaClient:
    """Zerodha Kite Connect API Client"""
    
    def __init__(self, api_key: str, api_secret: str, access_token: str = None):
        self.api_key = api_key
        self.api_secret = api_secret
        self.access_token = access_token
        
        # Initialize KiteConnect
        self.kite = KiteConnect(api_key=api_key)
        
        if access_token:
            self.kite.set_access_token(access_token)
        
        self.logger = get_logger(__name__)
        self.instruments = {}
        self.positions = {}
        self.orders = {}
        
        # Rate limiting
        self.last_request_time = 0
        self.min_request_interval = 0.1  # 100ms between requests
        
        # Initialize instruments
        asyncio.create_task(self._load_instruments())
    
    def generate_session(self, request_token: str) -> Dict[str, str]:
        """
        Generate access token using request token
        
        Args:
            request_token: Request token from Zerodha login
            
        Returns:
            Session data with access token
        """
        try:
            data = self.kite.generate_session(request_token, api_secret=self.api_secret)
            self.access_token = data["access_token"]
            self.kite.set_access_token(self.access_token)
            
            self.logger.info("Zerodha session generated successfully")
            return data
            
        except Exception as e:
            self.logger.error(f"Error generating Zerodha session: {e}")
            raise
    
    async def _rate_limit(self):
        """Implement rate limiting"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < self.min_request_interval:
            await asyncio.sleep(self.min_request_interval - time_since_last)
        
        self.last_request_time = time.time()
    
    async def _load_instruments(self):
        """Load and cache instrument data"""
        try:
            await self._rate_limit()
            
            # Get instruments for NSE and NFO
            nse_instruments = self.kite.instruments("NSE")
            nfo_instruments = self.kite.instruments("NFO")
            
            # Combine and index by trading symbol
            all_instruments = nse_instruments + nfo_instruments
            
            for instrument in all_instruments:
                symbol = instrument['tradingsymbol']
                self.instruments[symbol] = instrument
            
            self.logger.info(f"Loaded {len(self.instruments)} instruments")
            
        except Exception as e:
            self.logger.error(f"Error loading instruments: {e}")
    
    def get_instrument_token(self, symbol: str) -> Optional[int]:
        """Get instrument token for a symbol"""
        instrument = self.instruments.get(symbol)
        return instrument['instrument_token'] if instrument else None
    
    async def get_quote(self, symbols: List[str]) -> Dict[str, Any]:
        """
        Get real-time quotes for symbols
        
        Args:
            symbols: List of trading symbols
            
        Returns:
            Quote data
        """
        try:
            await self._rate_limit()
            
            # Convert symbols to instrument tokens
            instruments = []
            for symbol in symbols:
                token = self.get_instrument_token(symbol)
                if token:
                    instruments.append(f"NSE:{symbol}" if symbol in self.instruments and 
                                     self.instruments[symbol]['exchange'] == 'NSE' else f"NFO:{symbol}")
            
            if not instruments:
                return {}
            
            quotes = self.kite.quote(instruments)
            return quotes
            
        except Exception as e:
            self.logger.error(f"Error getting quotes: {e}")
            return {}
    
    async def get_historical_data(self, symbol: str, from_date: datetime, 
                                to_date: datetime, interval: str = "minute") -> pd.DataFrame:
        """
        Get historical data for a symbol
        
        Args:
            symbol: Trading symbol
            from_date: Start date
            to_date: End date
            interval: Data interval (minute, 3minute, 5minute, 15minute, 30minute, 60minute, day)
            
        Returns:
            Historical data as DataFrame
        """
        try:
            await self._rate_limit()
            
            instrument_token = self.get_instrument_token(symbol)
            if not instrument_token:
                self.logger.error(f"Instrument token not found for {symbol}")
                return pd.DataFrame()
            
            data = self.kite.historical_data(
                instrument_token=instrument_token,
                from_date=from_date,
                to_date=to_date,
                interval=interval
            )
            
            if data:
                df = pd.DataFrame(data)
                df['date'] = pd.to_datetime(df['date'])
                df.set_index('date', inplace=True)
                return df
            else:
                return pd.DataFrame()
                
        except Exception as e:
            self.logger.error(f"Error getting historical data for {symbol}: {e}")
            return pd.DataFrame()
    
    async def place_order(self, symbol: str, quantity: int, order_type: str,
                         transaction_type: str, product: str = "MIS",
                         price: float = None, trigger_price: float = None,
                         validity: str = "DAY") -> Optional[str]:
        """
        Place an order
        
        Args:
            symbol: Trading symbol
            quantity: Order quantity
            order_type: Order type (MARKET, LIMIT, SL, SL-M)
            transaction_type: BUY or SELL
            product: Product type (MIS, CNC, NRML)
            price: Limit price (for LIMIT orders)
            trigger_price: Trigger price (for SL orders)
            validity: Order validity (DAY, IOC)
            
        Returns:
            Order ID if successful
        """
        try:
            await self._rate_limit()
            
            order_params = {
                "tradingsymbol": symbol,
                "exchange": self._get_exchange(symbol),
                "transaction_type": transaction_type,
                "quantity": quantity,
                "order_type": order_type,
                "product": product,
                "validity": validity
            }
            
            if price:
                order_params["price"] = price
            
            if trigger_price:
                order_params["trigger_price"] = trigger_price
            
            order_id = self.kite.place_order(**order_params)
            
            self.logger.info(f"Order placed successfully: {order_id}")
            return order_id
            
        except KiteException as e:
            self.logger.error(f"Kite API error placing order: {e}")
            return None
        except Exception as e:
            self.logger.error(f"Error placing order: {e}")
            return None
    
    async def modify_order(self, order_id: str, quantity: int = None,
                          price: float = None, order_type: str = None,
                          trigger_price: float = None) -> bool:
        """
        Modify an existing order
        
        Args:
            order_id: Order ID to modify
            quantity: New quantity
            price: New price
            order_type: New order type
            trigger_price: New trigger price
            
        Returns:
            True if successful
        """
        try:
            await self._rate_limit()
            
            modify_params = {"order_id": order_id}
            
            if quantity:
                modify_params["quantity"] = quantity
            if price:
                modify_params["price"] = price
            if order_type:
                modify_params["order_type"] = order_type
            if trigger_price:
                modify_params["trigger_price"] = trigger_price
            
            self.kite.modify_order(**modify_params)
            
            self.logger.info(f"Order {order_id} modified successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Error modifying order {order_id}: {e}")
            return False
    
    async def cancel_order(self, order_id: str) -> bool:
        """
        Cancel an order
        
        Args:
            order_id: Order ID to cancel
            
        Returns:
            True if successful
        """
        try:
            await self._rate_limit()
            
            self.kite.cancel_order(order_id=order_id)
            
            self.logger.info(f"Order {order_id} cancelled successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Error cancelling order {order_id}: {e}")
            return False
    
    async def get_orders(self) -> List[Dict[str, Any]]:
        """
        Get all orders for the day
        
        Returns:
            List of orders
        """
        try:
            await self._rate_limit()
            
            orders = self.kite.orders()
            return orders
            
        except Exception as e:
            self.logger.error(f"Error getting orders: {e}")
            return []
    
    async def get_positions(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Get current positions
        
        Returns:
            Dictionary with net and day positions
        """
        try:
            await self._rate_limit()
            
            positions = self.kite.positions()
            return positions
            
        except Exception as e:
            self.logger.error(f"Error getting positions: {e}")
            return {"net": [], "day": []}
    
    async def get_holdings(self) -> List[Dict[str, Any]]:
        """
        Get holdings
        
        Returns:
            List of holdings
        """
        try:
            await self._rate_limit()
            
            holdings = self.kite.holdings()
            return holdings
            
        except Exception as e:
            self.logger.error(f"Error getting holdings: {e}")
            return []
    
    async def get_margins(self) -> Dict[str, Any]:
        """
        Get margin information
        
        Returns:
            Margin data
        """
        try:
            await self._rate_limit()
            
            margins = self.kite.margins()
            return margins
            
        except Exception as e:
            self.logger.error(f"Error getting margins: {e}")
            return {}
    
    def _get_exchange(self, symbol: str) -> str:
        """Get exchange for a symbol"""
        instrument = self.instruments.get(symbol)
        if instrument:
            return instrument['exchange']
        
        # Default logic based on symbol pattern
        if any(keyword in symbol.upper() for keyword in ['CE', 'PE', 'FUT']):
            return "NFO"
        else:
            return "NSE"
    
    async def get_option_chain(self, underlying: str, expiry: str = None) -> List[Dict[str, Any]]:
        """
        Get option chain for an underlying
        
        Args:
            underlying: Underlying symbol (e.g., NIFTY, BANKNIFTY)
            expiry: Expiry date (YYYY-MM-DD format)
            
        Returns:
            List of option contracts
        """
        try:
            option_contracts = []
            
            for symbol, instrument in self.instruments.items():
                if (instrument['name'] == underlying and 
                    instrument['instrument_type'] in ['CE', 'PE']):
                    
                    if expiry and instrument['expiry'].strftime('%Y-%m-%d') != expiry:
                        continue
                    
                    option_contracts.append(instrument)
            
            return option_contracts
            
        except Exception as e:
            self.logger.error(f"Error getting option chain for {underlying}: {e}")
            return []
    
    async def get_ltp(self, symbols: List[str]) -> Dict[str, float]:
        """
        Get Last Traded Price for symbols
        
        Args:
            symbols: List of trading symbols
            
        Returns:
            Dictionary with symbol: LTP mapping
        """
        try:
            quotes = await self.get_quote(symbols)
            ltp_data = {}
            
            for instrument, quote_data in quotes.items():
                symbol = instrument.split(':')[1]  # Remove exchange prefix
                ltp_data[symbol] = quote_data.get('last_price', 0)
            
            return ltp_data
            
        except Exception as e:
            self.logger.error(f"Error getting LTP: {e}")
            return {}
    
    async def get_market_depth(self, symbols: List[str]) -> Dict[str, Any]:
        """
        Get market depth (order book) for symbols
        
        Args:
            symbols: List of trading symbols
            
        Returns:
            Market depth data
        """
        try:
            quotes = await self.get_quote(symbols)
            depth_data = {}
            
            for instrument, quote_data in quotes.items():
                symbol = instrument.split(':')[1]
                depth_data[symbol] = {
                    'buy': quote_data.get('depth', {}).get('buy', []),
                    'sell': quote_data.get('depth', {}).get('sell', [])
                }
            
            return depth_data
            
        except Exception as e:
            self.logger.error(f"Error getting market depth: {e}")
            return {}
    
    def is_market_open(self) -> bool:
        """
        Check if market is currently open
        
        Returns:
            True if market is open
        """
        try:
            now = datetime.now()
            
            # Check if it's a weekday (Monday = 0, Sunday = 6)
            if now.weekday() > 4:  # Saturday or Sunday
                return False
            
            # Market hours: 9:15 AM to 3:30 PM
            market_open = now.replace(hour=9, minute=15, second=0, microsecond=0)
            market_close = now.replace(hour=15, minute=30, second=0, microsecond=0)
            
            return market_open <= now <= market_close
            
        except Exception as e:
            self.logger.error(f"Error checking market status: {e}")
            return False
    
    async def get_profile(self) -> Dict[str, Any]:
        """
        Get user profile
        
        Returns:
            User profile data
        """
        try:
            await self._rate_limit()
            
            profile = self.kite.profile()
            return profile
            
        except Exception as e:
            self.logger.error(f"Error getting profile: {e}")
            return {}
    
    async def close_all_positions(self) -> List[str]:
        """
        Close all open positions
        
        Returns:
            List of order IDs for closing orders
        """
        try:
            positions = await self.get_positions()
            order_ids = []
            
            for position in positions.get('net', []):
                if position['quantity'] != 0:
                    # Determine transaction type (opposite of current position)
                    transaction_type = "SELL" if position['quantity'] > 0 else "BUY"
                    quantity = abs(position['quantity'])
                    
                    order_id = await self.place_order(
                        symbol=position['tradingsymbol'],
                        quantity=quantity,
                        order_type="MARKET",
                        transaction_type=transaction_type,
                        product=position['product']
                    )
                    
                    if order_id:
                        order_ids.append(order_id)
            
            self.logger.info(f"Placed {len(order_ids)} orders to close positions")
            return order_ids
            
        except Exception as e:
            self.logger.error(f"Error closing all positions: {e}")
            return []

if __name__ == "__main__":
    # Test the Zerodha client
    import asyncio
    
    async def test_zerodha_client():
        # Note: These are dummy credentials for testing
        api_key = "your_api_key"
        api_secret = "your_api_secret"
        access_token = "your_access_token"
        
        client = ZerodhaClient(api_key, api_secret, access_token)
        
        # Test market status
        print(f"Market Open: {client.is_market_open()}")
        
        # Test getting quotes (this will fail without valid credentials)
        try:
            quotes = await client.get_quote(["NIFTY 50"])
            print(f"Quotes: {quotes}")
        except Exception as e:
            print(f"Quote test failed (expected): {e}")
        
        # Test getting option chain
        try:
            options = await client.get_option_chain("NIFTY")
            print(f"Found {len(options)} option contracts")
        except Exception as e:
            print(f"Option chain test failed (expected): {e}")
    
    asyncio.run(test_zerodha_client())

