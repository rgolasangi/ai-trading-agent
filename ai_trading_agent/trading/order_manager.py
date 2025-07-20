"""
Order Management System for AI Trading Agent
Handles order lifecycle, tracking, and risk management
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import asyncio
import uuid
import json

from utils.logger import get_logger
from config.config import Config
from .zerodha_client import ZerodhaClient

logger = get_logger(__name__)

class OrderStatus(Enum):
    """Order status enumeration"""
    PENDING = "PENDING"
    OPEN = "OPEN"
    COMPLETE = "COMPLETE"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"
    MODIFIED = "MODIFIED"

class OrderType(Enum):
    """Order type enumeration"""
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP_LOSS = "SL"
    STOP_LOSS_MARKET = "SL-M"

class TransactionType(Enum):
    """Transaction type enumeration"""
    BUY = "BUY"
    SELL = "SELL"

class ProductType(Enum):
    """Product type enumeration"""
    MIS = "MIS"  # Margin Intraday Square-off
    CNC = "CNC"  # Cash and Carry
    NRML = "NRML"  # Normal

@dataclass
class Order:
    """Order data structure"""
    order_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    symbol: str = ""
    quantity: int = 0
    price: float = 0.0
    trigger_price: float = 0.0
    order_type: OrderType = OrderType.MARKET
    transaction_type: TransactionType = TransactionType.BUY
    product_type: ProductType = ProductType.MIS
    status: OrderStatus = OrderStatus.PENDING
    exchange_order_id: str = ""
    filled_quantity: int = 0
    average_price: float = 0.0
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    parent_order_id: str = ""
    stop_loss_order_id: str = ""
    target_order_id: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert order to dictionary"""
        return {
            'order_id': self.order_id,
            'symbol': self.symbol,
            'quantity': self.quantity,
            'price': self.price,
            'trigger_price': self.trigger_price,
            'order_type': self.order_type.value,
            'transaction_type': self.transaction_type.value,
            'product_type': self.product_type.value,
            'status': self.status.value,
            'exchange_order_id': self.exchange_order_id,
            'filled_quantity': self.filled_quantity,
            'average_price': self.average_price,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'parent_order_id': self.parent_order_id,
            'stop_loss_order_id': self.stop_loss_order_id,
            'target_order_id': self.target_order_id,
            'metadata': self.metadata
        }

@dataclass
class Position:
    """Position data structure"""
    symbol: str = ""
    quantity: int = 0
    average_price: float = 0.0
    last_price: float = 0.0
    pnl: float = 0.0
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    product_type: ProductType = ProductType.MIS
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert position to dictionary"""
        return {
            'symbol': self.symbol,
            'quantity': self.quantity,
            'average_price': self.average_price,
            'last_price': self.last_price,
            'pnl': self.pnl,
            'unrealized_pnl': self.unrealized_pnl,
            'realized_pnl': self.realized_pnl,
            'product_type': self.product_type.value,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'metadata': self.metadata
        }

class OrderManager:
    """Order Management System"""
    
    def __init__(self, zerodha_client: ZerodhaClient):
        self.zerodha_client = zerodha_client
        self.logger = get_logger(__name__)
        
        # Order and position tracking
        self.orders: Dict[str, Order] = {}
        self.positions: Dict[str, Position] = {}
        self.order_history: List[Order] = []
        
        # Risk management parameters
        self.max_order_value = 100000  # Maximum order value
        self.max_position_size = 50000  # Maximum position size per symbol
        self.max_daily_loss = 10000  # Maximum daily loss
        self.max_orders_per_minute = 10  # Rate limiting
        
        # Tracking
        self.daily_pnl = 0.0
        self.total_orders_today = 0
        self.last_order_time = datetime.now()
        self.order_count_last_minute = 0
        
        # Start monitoring tasks
        asyncio.create_task(self._monitor_orders())
        asyncio.create_task(self._update_positions())
    
    async def place_order(self, symbol: str, quantity: int, order_type: OrderType,
                         transaction_type: TransactionType, price: float = None,
                         trigger_price: float = None, product_type: ProductType = ProductType.MIS,
                         metadata: Dict[str, Any] = None) -> Optional[Order]:
        """
        Place a new order with risk checks
        
        Args:
            symbol: Trading symbol
            quantity: Order quantity
            order_type: Type of order
            transaction_type: BUY or SELL
            price: Limit price (for LIMIT orders)
            trigger_price: Trigger price (for SL orders)
            product_type: Product type
            metadata: Additional metadata
            
        Returns:
            Order object if successful
        """
        try:
            # Risk checks
            if not await self._pre_order_risk_checks(symbol, quantity, price or 0):
                return None
            
            # Create order object
            order = Order(
                symbol=symbol,
                quantity=quantity,
                price=price or 0.0,
                trigger_price=trigger_price or 0.0,
                order_type=order_type,
                transaction_type=transaction_type,
                product_type=product_type,
                metadata=metadata or {}
            )
            
            # Place order with Zerodha
            exchange_order_id = await self.zerodha_client.place_order(
                symbol=symbol,
                quantity=quantity,
                order_type=order_type.value,
                transaction_type=transaction_type.value,
                product=product_type.value,
                price=price,
                trigger_price=trigger_price
            )
            
            if exchange_order_id:
                order.exchange_order_id = exchange_order_id
                order.status = OrderStatus.OPEN
                
                # Store order
                self.orders[order.order_id] = order
                self.order_history.append(order)
                
                # Update counters
                self.total_orders_today += 1
                self.last_order_time = datetime.now()
                
                self.logger.info(f"Order placed successfully: {order.order_id}")
                return order
            else:
                order.status = OrderStatus.REJECTED
                self.logger.error(f"Order rejected by exchange: {order.order_id}")
                return None
                
        except Exception as e:
            self.logger.error(f"Error placing order: {e}")
            return None
    
    async def modify_order(self, order_id: str, quantity: int = None,
                          price: float = None, trigger_price: float = None) -> bool:
        """
        Modify an existing order
        
        Args:
            order_id: Order ID to modify
            quantity: New quantity
            price: New price
            trigger_price: New trigger price
            
        Returns:
            True if successful
        """
        try:
            order = self.orders.get(order_id)
            if not order:
                self.logger.error(f"Order not found: {order_id}")
                return False
            
            if order.status not in [OrderStatus.OPEN, OrderStatus.PENDING]:
                self.logger.error(f"Cannot modify order in status: {order.status}")
                return False
            
            # Modify order with Zerodha
            success = await self.zerodha_client.modify_order(
                order_id=order.exchange_order_id,
                quantity=quantity,
                price=price,
                trigger_price=trigger_price
            )
            
            if success:
                # Update order details
                if quantity:
                    order.quantity = quantity
                if price:
                    order.price = price
                if trigger_price:
                    order.trigger_price = trigger_price
                
                order.status = OrderStatus.MODIFIED
                order.updated_at = datetime.now()
                
                self.logger.info(f"Order modified successfully: {order_id}")
                return True
            else:
                self.logger.error(f"Failed to modify order: {order_id}")
                return False
                
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
            order = self.orders.get(order_id)
            if not order:
                self.logger.error(f"Order not found: {order_id}")
                return False
            
            if order.status not in [OrderStatus.OPEN, OrderStatus.PENDING]:
                self.logger.error(f"Cannot cancel order in status: {order.status}")
                return False
            
            # Cancel order with Zerodha
            success = await self.zerodha_client.cancel_order(order.exchange_order_id)
            
            if success:
                order.status = OrderStatus.CANCELLED
                order.updated_at = datetime.now()
                
                self.logger.info(f"Order cancelled successfully: {order_id}")
                return True
            else:
                self.logger.error(f"Failed to cancel order: {order_id}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error cancelling order {order_id}: {e}")
            return False
    
    async def place_bracket_order(self, symbol: str, quantity: int,
                                 transaction_type: TransactionType, entry_price: float,
                                 stop_loss: float, target: float,
                                 product_type: ProductType = ProductType.MIS) -> Optional[Dict[str, Order]]:
        """
        Place a bracket order (entry + stop loss + target)
        
        Args:
            symbol: Trading symbol
            quantity: Order quantity
            transaction_type: BUY or SELL
            entry_price: Entry price
            stop_loss: Stop loss price
            target: Target price
            product_type: Product type
            
        Returns:
            Dictionary with entry, stop loss, and target orders
        """
        try:
            # Place entry order
            entry_order = await self.place_order(
                symbol=symbol,
                quantity=quantity,
                order_type=OrderType.LIMIT,
                transaction_type=transaction_type,
                price=entry_price,
                product_type=product_type,
                metadata={'order_group': 'bracket', 'order_role': 'entry'}
            )
            
            if not entry_order:
                return None
            
            # Determine stop loss and target transaction types
            sl_transaction = TransactionType.SELL if transaction_type == TransactionType.BUY else TransactionType.BUY
            target_transaction = TransactionType.SELL if transaction_type == TransactionType.BUY else TransactionType.BUY
            
            # Place stop loss order
            stop_loss_order = await self.place_order(
                symbol=symbol,
                quantity=quantity,
                order_type=OrderType.STOP_LOSS,
                transaction_type=sl_transaction,
                trigger_price=stop_loss,
                product_type=product_type,
                metadata={'order_group': 'bracket', 'order_role': 'stop_loss', 'parent_order': entry_order.order_id}
            )
            
            # Place target order
            target_order = await self.place_order(
                symbol=symbol,
                quantity=quantity,
                order_type=OrderType.LIMIT,
                transaction_type=target_transaction,
                price=target,
                product_type=product_type,
                metadata={'order_group': 'bracket', 'order_role': 'target', 'parent_order': entry_order.order_id}
            )
            
            # Link orders
            if stop_loss_order:
                entry_order.stop_loss_order_id = stop_loss_order.order_id
            if target_order:
                entry_order.target_order_id = target_order.order_id
            
            bracket_orders = {
                'entry': entry_order,
                'stop_loss': stop_loss_order,
                'target': target_order
            }
            
            self.logger.info(f"Bracket order placed for {symbol}")
            return bracket_orders
            
        except Exception as e:
            self.logger.error(f"Error placing bracket order: {e}")
            return None
    
    async def _pre_order_risk_checks(self, symbol: str, quantity: int, price: float) -> bool:
        """
        Perform pre-order risk checks
        
        Args:
            symbol: Trading symbol
            quantity: Order quantity
            price: Order price
            
        Returns:
            True if order passes risk checks
        """
        try:
            # Check daily loss limit
            if self.daily_pnl <= -self.max_daily_loss:
                self.logger.warning(f"Daily loss limit exceeded: {self.daily_pnl}")
                return False
            
            # Check order value limit
            order_value = quantity * price if price > 0 else quantity * 100  # Estimate for market orders
            if order_value > self.max_order_value:
                self.logger.warning(f"Order value exceeds limit: {order_value}")
                return False
            
            # Check position size limit
            current_position = self.positions.get(symbol)
            if current_position:
                new_position_value = abs(current_position.quantity * current_position.average_price + order_value)
                if new_position_value > self.max_position_size:
                    self.logger.warning(f"Position size limit exceeded for {symbol}: {new_position_value}")
                    return False
            
            # Check rate limiting
            now = datetime.now()
            if (now - self.last_order_time).total_seconds() < 60:
                self.order_count_last_minute += 1
                if self.order_count_last_minute > self.max_orders_per_minute:
                    self.logger.warning("Order rate limit exceeded")
                    return False
            else:
                self.order_count_last_minute = 1
            
            # Check market hours
            if not self.zerodha_client.is_market_open():
                self.logger.warning("Market is closed")
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error in risk checks: {e}")
            return False
    
    async def _monitor_orders(self):
        """Monitor order status updates"""
        while True:
            try:
                # Get latest orders from Zerodha
                exchange_orders = await self.zerodha_client.get_orders()
                
                # Update order statuses
                for exchange_order in exchange_orders:
                    exchange_order_id = exchange_order.get('order_id')
                    
                    # Find corresponding internal order
                    internal_order = None
                    for order in self.orders.values():
                        if order.exchange_order_id == exchange_order_id:
                            internal_order = order
                            break
                    
                    if internal_order:
                        # Update order status
                        old_status = internal_order.status
                        new_status = self._map_exchange_status(exchange_order.get('status'))
                        
                        if old_status != new_status:
                            internal_order.status = new_status
                            internal_order.filled_quantity = exchange_order.get('filled_quantity', 0)
                            internal_order.average_price = exchange_order.get('average_price', 0)
                            internal_order.updated_at = datetime.now()
                            
                            self.logger.info(f"Order status updated: {internal_order.order_id} -> {new_status}")
                            
                            # Handle order completion
                            if new_status == OrderStatus.COMPLETE:
                                await self._handle_order_completion(internal_order)
                
                await asyncio.sleep(5)  # Check every 5 seconds
                
            except Exception as e:
                self.logger.error(f"Error monitoring orders: {e}")
                await asyncio.sleep(10)
    
    async def _update_positions(self):
        """Update position information"""
        while True:
            try:
                # Get latest positions from Zerodha
                exchange_positions = await self.zerodha_client.get_positions()
                
                # Update internal positions
                for pos_data in exchange_positions.get('net', []):
                    symbol = pos_data.get('tradingsymbol')
                    quantity = pos_data.get('quantity', 0)
                    
                    if quantity != 0:  # Only track non-zero positions
                        position = self.positions.get(symbol)
                        if not position:
                            position = Position(symbol=symbol)
                            self.positions[symbol] = position
                        
                        position.quantity = quantity
                        position.average_price = pos_data.get('average_price', 0)
                        position.last_price = pos_data.get('last_price', 0)
                        position.pnl = pos_data.get('pnl', 0)
                        position.unrealized_pnl = pos_data.get('unrealised', 0)
                        position.realized_pnl = pos_data.get('realised', 0)
                        position.updated_at = datetime.now()
                    else:
                        # Remove zero positions
                        if symbol in self.positions:
                            del self.positions[symbol]
                
                # Update daily P&L
                self.daily_pnl = sum(pos.pnl for pos in self.positions.values())
                
                await asyncio.sleep(10)  # Update every 10 seconds
                
            except Exception as e:
                self.logger.error(f"Error updating positions: {e}")
                await asyncio.sleep(30)
    
    def _map_exchange_status(self, exchange_status: str) -> OrderStatus:
        """Map exchange order status to internal status"""
        status_mapping = {
            'OPEN': OrderStatus.OPEN,
            'COMPLETE': OrderStatus.COMPLETE,
            'CANCELLED': OrderStatus.CANCELLED,
            'REJECTED': OrderStatus.REJECTED,
            'MODIFY_PENDING': OrderStatus.MODIFIED,
            'CANCEL_PENDING': OrderStatus.CANCELLED
        }
        
        return status_mapping.get(exchange_status, OrderStatus.PENDING)
    
    async def _handle_order_completion(self, order: Order):
        """Handle order completion logic"""
        try:
            # Update position
            position = self.positions.get(order.symbol)
            if not position:
                position = Position(symbol=order.symbol)
                self.positions[order.symbol] = position
            
            # Calculate new position
            if order.transaction_type == TransactionType.BUY:
                new_quantity = position.quantity + order.filled_quantity
                if position.quantity == 0:
                    new_avg_price = order.average_price
                else:
                    total_value = (position.quantity * position.average_price + 
                                 order.filled_quantity * order.average_price)
                    new_avg_price = total_value / new_quantity
            else:  # SELL
                new_quantity = position.quantity - order.filled_quantity
                new_avg_price = position.average_price  # Keep same average price
            
            position.quantity = new_quantity
            position.average_price = new_avg_price
            position.updated_at = datetime.now()
            
            # Remove position if quantity becomes zero
            if new_quantity == 0:
                del self.positions[order.symbol]
            
            self.logger.info(f"Position updated for {order.symbol}: {new_quantity} @ {new_avg_price}")
            
        except Exception as e:
            self.logger.error(f"Error handling order completion: {e}")
    
    def get_order_status(self, order_id: str) -> Optional[OrderStatus]:
        """Get order status"""
        order = self.orders.get(order_id)
        return order.status if order else None
    
    def get_position(self, symbol: str) -> Optional[Position]:
        """Get position for a symbol"""
        return self.positions.get(symbol)
    
    def get_all_positions(self) -> Dict[str, Position]:
        """Get all positions"""
        return self.positions.copy()
    
    def get_daily_pnl(self) -> float:
        """Get daily P&L"""
        return self.daily_pnl
    
    def get_order_history(self, symbol: str = None) -> List[Order]:
        """Get order history"""
        if symbol:
            return [order for order in self.order_history if order.symbol == symbol]
        return self.order_history.copy()
    
    async def close_all_positions(self) -> List[str]:
        """Close all open positions"""
        try:
            order_ids = []
            
            for symbol, position in self.positions.items():
                if position.quantity != 0:
                    transaction_type = TransactionType.SELL if position.quantity > 0 else TransactionType.BUY
                    
                    order = await self.place_order(
                        symbol=symbol,
                        quantity=abs(position.quantity),
                        order_type=OrderType.MARKET,
                        transaction_type=transaction_type,
                        product_type=position.product_type,
                        metadata={'action': 'close_all_positions'}
                    )
                    
                    if order:
                        order_ids.append(order.order_id)
            
            self.logger.info(f"Placed {len(order_ids)} orders to close all positions")
            return order_ids
            
        except Exception as e:
            self.logger.error(f"Error closing all positions: {e}")
            return []

if __name__ == "__main__":
    # Test the order manager
    import asyncio
    
    async def test_order_manager():
        # Create dummy Zerodha client
        zerodha_client = ZerodhaClient("dummy_key", "dummy_secret", "dummy_token")
        
        # Create order manager
        order_manager = OrderManager(zerodha_client)
        
        # Test order creation (will fail without valid credentials)
        try:
            order = await order_manager.place_order(
                symbol="NIFTY2312519500CE",
                quantity=50,
                order_type=OrderType.LIMIT,
                transaction_type=TransactionType.BUY,
                price=100.0
            )
            print(f"Order created: {order.order_id if order else 'Failed'}")
        except Exception as e:
            print(f"Order test failed (expected): {e}")
        
        # Test risk checks
        risk_passed = await order_manager._pre_order_risk_checks("TEST", 100, 1000)
        print(f"Risk checks passed: {risk_passed}")
        
        # Test position tracking
        positions = order_manager.get_all_positions()
        print(f"Current positions: {len(positions)}")
    
    asyncio.run(test_order_manager())

