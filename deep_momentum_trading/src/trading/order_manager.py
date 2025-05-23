"""
Enhanced order manager for Deep Momentum Trading System with ARM64 optimizations.

This module provides comprehensive order management capabilities including order lifecycle
management, status tracking, risk controls, and ARM64-specific performance optimizations.
"""

import time
import threading
import asyncio
from typing import Dict, List, Optional, Any, Union, Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from decimal import Decimal
from enum import Enum
import json
import platform
from collections import defaultdict, deque
import uuid

# Internal imports
from .alpaca_client import AlpacaClient, OrderRequest, OrderType, OrderSideEnum, TimeInForceEnum
from ..utils.logger import get_logger, PerformanceLogger, TradingLogger
from ..utils.decorators import performance_monitor, retry_with_backoff, rate_limit
from ..utils.exceptions import OrderError, RiskError, ValidationError
from ..utils.validators import validate_trading_data, ValidationConfig
from ..utils.constants import TRADING_CONSTANTS
from ..utils.shared_memory import create_shared_dict, SharedDict

# ARM64 detection
IS_ARM64 = platform.machine().lower() in ['arm64', 'aarch64']

logger = get_logger(__name__)
perf_logger = PerformanceLogger(logger)
trading_logger = TradingLogger(logger)

class OrderStatus(Enum):
    """Enhanced order status enumeration."""
    PENDING = "pending"
    SUBMITTED = "submitted"
    ACCEPTED = "accepted"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"
    REPLACED = "replaced"
    PENDING_CANCEL = "pending_cancel"
    PENDING_REPLACE = "pending_replace"

class OrderPriority(Enum):
    """Order priority levels."""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    URGENT = 4
    CRITICAL = 5

@dataclass
class OrderConfig:
    """Configuration for order manager."""
    default_order_type: OrderType = OrderType.MARKET
    default_time_in_force: TimeInForceEnum = TimeInForceEnum.DAY
    max_active_orders: int = 1000
    max_order_size: float = 100000.0
    min_order_size: float = 1.0
    enable_risk_checks: bool = True
    enable_position_limits: bool = True
    enable_order_validation: bool = True
    enable_performance_monitoring: bool = True
    enable_arm64_optimizations: bool = True
    order_timeout_seconds: float = 300.0  # 5 minutes
    max_daily_orders: int = 10000
    max_order_value: float = 1000000.0
    enable_shared_memory: bool = True
    shared_memory_size: int = 10000
    enable_order_batching: bool = True
    batch_size: int = 50
    batch_timeout_ms: float = 100.0

@dataclass
class Order:
    """Enhanced order data structure."""
    order_id: str
    symbol: str
    quantity: Union[float, Decimal]
    side: OrderSideEnum
    order_type: OrderType
    time_in_force: TimeInForceEnum
    status: OrderStatus = OrderStatus.PENDING
    priority: OrderPriority = OrderPriority.NORMAL
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    submitted_at: Optional[datetime] = None
    filled_at: Optional[datetime] = None
    cancelled_at: Optional[datetime] = None
    expired_at: Optional[datetime] = None
    filled_quantity: float = 0.0
    remaining_quantity: Optional[float] = None
    avg_fill_price: Optional[float] = None
    limit_price: Optional[Union[float, Decimal]] = None
    stop_price: Optional[Union[float, Decimal]] = None
    trail_price: Optional[Union[float, Decimal]] = None
    trail_percent: Optional[float] = None
    commission: float = 0.0
    fees: float = 0.0
    client_order_id: Optional[str] = None
    parent_order_id: Optional[str] = None
    child_order_ids: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None
    retry_count: int = 0
    last_update: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def __post_init__(self):
        """Post-initialization processing."""
        if self.remaining_quantity is None:
            self.remaining_quantity = float(self.quantity)
        
        if not self.client_order_id:
            self.client_order_id = f"order_{uuid.uuid4().hex[:8]}"
    
    @property
    def is_active(self) -> bool:
        """Check if order is in an active state."""
        return self.status in [
            OrderStatus.PENDING,
            OrderStatus.SUBMITTED,
            OrderStatus.ACCEPTED,
            OrderStatus.PARTIALLY_FILLED,
            OrderStatus.PENDING_CANCEL,
            OrderStatus.PENDING_REPLACE
        ]
    
    @property
    def is_terminal(self) -> bool:
        """Check if order is in a terminal state."""
        return self.status in [
            OrderStatus.FILLED,
            OrderStatus.CANCELLED,
            OrderStatus.REJECTED,
            OrderStatus.EXPIRED
        ]
    
    @property
    def fill_percentage(self) -> float:
        """Calculate fill percentage."""
        if float(self.quantity) == 0:
            return 0.0
        return (self.filled_quantity / float(self.quantity)) * 100.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert order to dictionary."""
        return {
            "order_id": self.order_id,
            "symbol": self.symbol,
            "quantity": float(self.quantity),
            "side": self.side.value,
            "order_type": self.order_type.value,
            "time_in_force": self.time_in_force.value,
            "status": self.status.value,
            "priority": self.priority.value,
            "created_at": self.created_at.isoformat(),
            "submitted_at": self.submitted_at.isoformat() if self.submitted_at else None,
            "filled_at": self.filled_at.isoformat() if self.filled_at else None,
            "filled_quantity": self.filled_quantity,
            "remaining_quantity": self.remaining_quantity,
            "avg_fill_price": self.avg_fill_price,
            "limit_price": float(self.limit_price) if self.limit_price else None,
            "stop_price": float(self.stop_price) if self.stop_price else None,
            "commission": self.commission,
            "fees": self.fees,
            "client_order_id": self.client_order_id,
            "tags": self.tags,
            "metadata": self.metadata,
            "error_message": self.error_message,
            "retry_count": self.retry_count,
            "last_update": self.last_update.isoformat()
        }

class OrderManager:
    """Enhanced order manager with ARM64 optimizations."""
    
    def __init__(self, 
                 broker_client: Optional[AlpacaClient] = None,
                 config: Optional[OrderConfig] = None):
        self.broker_client = broker_client
        self.config = config or OrderConfig()
        
        # Order storage
        self.active_orders: Dict[str, Order] = {}
        self.order_history: deque = deque(maxlen=10000)  # Circular buffer for history
        self.orders_by_symbol: Dict[str, List[str]] = defaultdict(list)
        self.orders_by_status: Dict[OrderStatus, List[str]] = defaultdict(list)
        
        # Thread safety
        self._order_lock = threading.RLock()
        
        # Performance monitoring
        self._setup_performance_monitoring()
        self._setup_shared_memory()
        self._setup_validation()
        
        # ARM64 optimizations
        self.is_arm64 = IS_ARM64
        if self.is_arm64 and self.config.enable_arm64_optimizations:
            self._apply_arm64_optimizations()
        
        # Order batching
        self._batch_queue = deque()
        self._batch_timer = None
        self._batch_lock = threading.Lock()
        
        # Daily limits tracking
        self._daily_order_count = 0
        self._daily_order_value = 0.0
        self._last_reset_date = datetime.now(timezone.utc).date()
        
        logger.info(f"OrderManager initialized (ARM64: {self.is_arm64})")
    
    def _setup_performance_monitoring(self):
        """Setup performance monitoring."""
        if self.config.enable_performance_monitoring:
            self._performance_stats = {
                "orders_created": 0,
                "orders_submitted": 0,
                "orders_filled": 0,
                "orders_cancelled": 0,
                "orders_rejected": 0,
                "total_volume": 0.0,
                "total_value": 0.0,
                "avg_fill_time_ms": 0.0,
                "total_fill_time_ms": 0.0,
                "fill_rate": 0.0
            }
        else:
            self._performance_stats = None
    
    def _setup_shared_memory(self):
        """Setup shared memory for high-performance order tracking."""
        if self.config.enable_shared_memory:
            try:
                self.shared_orders = create_shared_dict(
                    name="order_manager_orders",
                    max_items=self.config.shared_memory_size
                )
            except Exception as e:
                logger.warning(f"Failed to setup shared memory: {e}")
                self.shared_orders = None
        else:
            self.shared_orders = None
    
    def _setup_validation(self):
        """Setup order validation."""
        if self.config.enable_order_validation:
            self.validation_config = ValidationConfig(level="normal")
        else:
            self.validation_config = None
    
    def _apply_arm64_optimizations(self):
        """Apply ARM64-specific optimizations."""
        # Optimize batch sizes for ARM64 SIMD operations
        if self.config.batch_size % 64 != 0:
            self.config.batch_size = ((self.config.batch_size // 64) + 1) * 64
        
        # Reduce batch timeout for ARM64 performance
        self.config.batch_timeout_ms = max(10.0, self.config.batch_timeout_ms * 0.5)
        
        logger.debug(f"Applied ARM64 optimizations: batch_size={self.config.batch_size}")
    
    def _reset_daily_limits_if_needed(self):
        """Reset daily limits if new day."""
        current_date = datetime.now(timezone.utc).date()
        if current_date > self._last_reset_date:
            self._daily_order_count = 0
            self._daily_order_value = 0.0
            self._last_reset_date = current_date
            logger.info("Daily order limits reset")
    
    def _validate_order_limits(self, order: Order):
        """Validate order against configured limits."""
        self._reset_daily_limits_if_needed()
        
        # Check daily order count
        if self._daily_order_count >= self.config.max_daily_orders:
            raise OrderError(f"Daily order limit exceeded: {self.config.max_daily_orders}")
        
        # Check order size limits
        quantity = float(order.quantity)
        if quantity < self.config.min_order_size:
            raise OrderError(f"Order size {quantity} below minimum {self.config.min_order_size}")
        
        if quantity > self.config.max_order_size:
            raise OrderError(f"Order size {quantity} exceeds maximum {self.config.max_order_size}")
        
        # Check order value (approximate)
        if order.limit_price:
            order_value = quantity * float(order.limit_price)
            if order_value > self.config.max_order_value:
                raise OrderError(f"Order value {order_value} exceeds maximum {self.config.max_order_value}")
            
            if self._daily_order_value + order_value > self.config.max_order_value * 10:
                raise OrderError("Daily order value limit exceeded")
        
        # Check active order count
        if len(self.active_orders) >= self.config.max_active_orders:
            raise OrderError(f"Maximum active orders exceeded: {self.config.max_active_orders}")
    
    def _validate_order_data(self, order: Order):
        """Validate order data."""
        if not self.validation_config:
            return
        
        order_data = {
            "symbol": order.symbol,
            "quantity": float(order.quantity),
            "side": order.side.value,
            "order_type": order.order_type.value
        }
        
        validation_errors = validate_trading_data(order_data, self.validation_config)
        if validation_errors:
            error_messages = [error.message for error in validation_errors]
            raise ValidationError(f"Order validation failed: {'; '.join(error_messages)}")
    
    @performance_monitor
    def create_order(self, 
                    symbol: str,
                    quantity: Union[float, Decimal],
                    side: Union[OrderSideEnum, str],
                    order_type: Optional[Union[OrderType, str]] = None,
                    time_in_force: Optional[Union[TimeInForceEnum, str]] = None,
                    limit_price: Optional[Union[float, Decimal]] = None,
                    stop_price: Optional[Union[float, Decimal]] = None,
                    trail_price: Optional[Union[float, Decimal]] = None,
                    trail_percent: Optional[float] = None,
                    priority: Union[OrderPriority, str] = OrderPriority.NORMAL,
                    client_order_id: Optional[str] = None,
                    tags: Optional[List[str]] = None,
                    metadata: Optional[Dict[str, Any]] = None) -> Order:
        """Create a new order with comprehensive validation."""
        
        # Convert string enums to proper enums
        if isinstance(side, str):
            side = OrderSideEnum(side.lower())
        
        if isinstance(order_type, str):
            order_type = OrderType(order_type.lower())
        elif order_type is None:
            order_type = self.config.default_order_type
        
        if isinstance(time_in_force, str):
            time_in_force = TimeInForceEnum(time_in_force.lower())
        elif time_in_force is None:
            time_in_force = self.config.default_time_in_force
        
        if isinstance(priority, str):
            priority = OrderPriority[priority.upper()]
        
        # Generate order ID
        order_id = f"ord_{int(time.time() * 1000000)}_{uuid.uuid4().hex[:8]}"
        
        # Create order object
        order = Order(
            order_id=order_id,
            symbol=symbol.upper(),
            quantity=quantity,
            side=side,
            order_type=order_type,
            time_in_force=time_in_force,
            priority=priority,
            limit_price=limit_price,
            stop_price=stop_price,
            trail_price=trail_price,
            trail_percent=trail_percent,
            client_order_id=client_order_id,
            tags=tags or [],
            metadata=metadata or {}
        )
        
        # Validate order
        if self.config.enable_risk_checks:
            self._validate_order_limits(order)
        
        if self.config.enable_order_validation:
            self._validate_order_data(order)
        
        # Store order
        with self._order_lock:
            self.active_orders[order_id] = order
            self.orders_by_symbol[symbol.upper()].append(order_id)
            self.orders_by_status[OrderStatus.PENDING].append(order_id)
            
            # Update shared memory
            if self.shared_orders:
                try:
                    self.shared_orders.put(order_id, json.dumps(order.to_dict()).encode())
                except Exception as e:
                    logger.warning(f"Failed to update shared memory: {e}")
        
        # Update performance stats
        if self._performance_stats:
            self._performance_stats["orders_created"] += 1
        
        # Update daily counters
        self._daily_order_count += 1
        if order.limit_price:
            self._daily_order_value += float(quantity) * float(order.limit_price)
        
        logger.info(f"Order created: {order_id} {symbol} {side.value} {quantity}")
        
        return order
    
    @performance_monitor
    def submit_order(self, order: Order) -> bool:
        """Submit order to broker."""
        if not self.broker_client:
            raise OrderError("No broker client configured")
        
        if order.status != OrderStatus.PENDING:
            raise OrderError(f"Order {order.order_id} is not in pending state")
        
        try:
            # Create broker order request
            order_request = OrderRequest(
                symbol=order.symbol,
                quantity=order.quantity,
                side=order.side,
                order_type=order.order_type,
                time_in_force=order.time_in_force,
                limit_price=order.limit_price,
                stop_price=order.stop_price,
                trail_price=order.trail_price,
                trail_percent=order.trail_percent,
                client_order_id=order.client_order_id
            )
            
            # Submit to broker
            start_time = time.time()
            broker_response = self.broker_client.place_order(order_request)
            submission_time_ms = (time.time() - start_time) * 1000
            
            # Update order status
            with self._order_lock:
                order.status = OrderStatus.SUBMITTED
                order.submitted_at = datetime.now(timezone.utc)
                order.last_update = datetime.now(timezone.utc)
                
                # Update broker order ID if different
                if broker_response.get("order_id") != order.order_id:
                    order.metadata["broker_order_id"] = broker_response["order_id"]
                
                # Update status tracking
                self._update_order_status_tracking(order, OrderStatus.PENDING, OrderStatus.SUBMITTED)
            
            # Update performance stats
            if self._performance_stats:
                self._performance_stats["orders_submitted"] += 1
            
            # Log performance
            perf_logger.log_latency("order_submission", submission_time_ms, symbol=order.symbol)
            
            # Log trading event
            trading_logger.log_order({
                "order_id": order.order_id,
                "symbol": order.symbol,
                "quantity": float(order.quantity),
                "side": order.side.value,
                "order_type": order.order_type.value,
                "status": "submitted"
            })
            
            logger.info(f"Order submitted: {order.order_id}")
            return True
            
        except Exception as e:
            # Update order with error
            with self._order_lock:
                order.status = OrderStatus.REJECTED
                order.error_message = str(e)
                order.last_update = datetime.now(timezone.utc)
                self._update_order_status_tracking(order, OrderStatus.PENDING, OrderStatus.REJECTED)
            
            logger.error(f"Failed to submit order {order.order_id}: {e}")
            raise OrderError(f"Order submission failed: {e}")
    
    def _update_order_status_tracking(self, order: Order, old_status: OrderStatus, new_status: OrderStatus):
        """Update order status tracking indices."""
        # Remove from old status list
        if order.order_id in self.orders_by_status[old_status]:
            self.orders_by_status[old_status].remove(order.order_id)
        
        # Add to new status list
        self.orders_by_status[new_status].append(order.order_id)
        
        # Move to history if terminal
        if order.is_terminal:
            self.order_history.append(order)
            if order.order_id in self.active_orders:
                del self.active_orders[order.order_id]
            
            # Remove from symbol tracking
            if order.order_id in self.orders_by_symbol[order.symbol]:
                self.orders_by_symbol[order.symbol].remove(order.order_id)
    
    @performance_monitor
    def update_order_status(self, 
                           order_id: str,
                           status: Union[OrderStatus, str],
                           filled_quantity: Optional[float] = None,
                           avg_fill_price: Optional[float] = None,
                           commission: Optional[float] = None,
                           fees: Optional[float] = None,
                           error_message: Optional[str] = None) -> bool:
        """Update order status with comprehensive tracking."""
        
        if isinstance(status, str):
            status = OrderStatus(status.lower())
        
        with self._order_lock:
            order = self.active_orders.get(order_id)
            if not order:
                # Check history
                for hist_order in self.order_history:
                    if hist_order.order_id == order_id:
                        order = hist_order
                        break
                
                if not order:
                    logger.warning(f"Order {order_id} not found for status update")
                    return False
            
            old_status = order.status
            order.status = status
            order.last_update = datetime.now(timezone.utc)
            
            # Update fill information
            if filled_quantity is not None:
                order.filled_quantity = filled_quantity
                order.remaining_quantity = float(order.quantity) - filled_quantity
                
                if filled_quantity > 0 and not order.filled_at:
                    order.filled_at = datetime.now(timezone.utc)
            
            if avg_fill_price is not None:
                order.avg_fill_price = avg_fill_price
            
            if commission is not None:
                order.commission = commission
            
            if fees is not None:
                order.fees = fees
            
            if error_message is not None:
                order.error_message = error_message
            
            # Update status-specific timestamps
            if status == OrderStatus.CANCELLED:
                order.cancelled_at = datetime.now(timezone.utc)
            elif status == OrderStatus.EXPIRED:
                order.expired_at = datetime.now(timezone.utc)
            
            # Update tracking
            self._update_order_status_tracking(order, old_status, status)
            
            # Update shared memory
            if self.shared_orders:
                try:
                    self.shared_orders.put(order_id, json.dumps(order.to_dict()).encode())
                except Exception as e:
                    logger.warning(f"Failed to update shared memory: {e}")
        
        # Update performance stats
        if self._performance_stats:
            if status == OrderStatus.FILLED:
                self._performance_stats["orders_filled"] += 1
                self._performance_stats["total_volume"] += order.filled_quantity
                if order.avg_fill_price:
                    self._performance_stats["total_value"] += order.filled_quantity * order.avg_fill_price
                
                # Calculate fill time
                if order.submitted_at and order.filled_at:
                    fill_time_ms = (order.filled_at - order.submitted_at).total_seconds() * 1000
                    self._performance_stats["total_fill_time_ms"] += fill_time_ms
                    filled_orders = self._performance_stats["orders_filled"]
                    self._performance_stats["avg_fill_time_ms"] = (
                        self._performance_stats["total_fill_time_ms"] / filled_orders
                    )
            
            elif status == OrderStatus.CANCELLED:
                self._performance_stats["orders_cancelled"] += 1
            elif status == OrderStatus.REJECTED:
                self._performance_stats["orders_rejected"] += 1
            
            # Calculate fill rate
            total_orders = (
                self._performance_stats["orders_filled"] + 
                self._performance_stats["orders_cancelled"] + 
                self._performance_stats["orders_rejected"]
            )
            if total_orders > 0:
                self._performance_stats["fill_rate"] = (
                    self._performance_stats["orders_filled"] / total_orders * 100
                )
        
        logger.info(f"Order {order_id} status updated to {status.value}")
        return True
    
    @performance_monitor
    def cancel_order(self, order_id: str) -> bool:
        """Cancel an active order."""
        with self._order_lock:
            order = self.active_orders.get(order_id)
            if not order:
                logger.warning(f"Cannot cancel order {order_id}: not found in active orders")
                return False
            
            if not order.is_active:
                logger.warning(f"Cannot cancel order {order_id}: not in active state ({order.status.value})")
                return False
            
            # Update status to pending cancel
            order.status = OrderStatus.PENDING_CANCEL
            order.last_update = datetime.now(timezone.utc)
        
        # Cancel with broker
        if self.broker_client:
            try:
                success = self.broker_client.cancel_order(order_id)
                if success:
                    self.update_order_status(order_id, OrderStatus.CANCELLED)
                    logger.info(f"Order {order_id} cancelled successfully")
                    return True
                else:
                    # Revert status
                    with self._order_lock:
                        order.status = OrderStatus.SUBMITTED
                    logger.error(f"Failed to cancel order {order_id}")
                    return False
            except Exception as e:
                # Revert status
                with self._order_lock:
                    order.status = OrderStatus.SUBMITTED
                logger.error(f"Error cancelling order {order_id}: {e}")
                return False
        else:
            # No broker client - just mark as cancelled
            self.update_order_status(order_id, OrderStatus.CANCELLED)
            return True
    
    def get_order(self, order_id: str) -> Optional[Order]:
        """Get order by ID."""
        with self._order_lock:
            # Check active orders first
            order = self.active_orders.get(order_id)
            if order:
                return order
            
            # Check history
            for hist_order in self.order_history:
                if hist_order.order_id == order_id:
                    return hist_order
        
        return None
    
    def get_orders_by_symbol(self, symbol: str) -> List[Order]:
        """Get all orders for a symbol."""
        symbol = symbol.upper()
        orders = []
        
        with self._order_lock:
            order_ids = self.orders_by_symbol.get(symbol, [])
            for order_id in order_ids:
                order = self.active_orders.get(order_id)
                if order:
                    orders.append(order)
        
        return orders
    
    def get_orders_by_status(self, status: Union[OrderStatus, str]) -> List[Order]:
        """Get all orders with specific status."""
        if isinstance(status, str):
            status = OrderStatus(status.lower())
        
        orders = []
        with self._order_lock:
            order_ids = self.orders_by_status.get(status, [])
            for order_id in order_ids:
                order = self.active_orders.get(order_id)
                if order:
                    orders.append(order)
        
        return orders
    
    def get_active_orders(self) -> List[Order]:
        """Get all active orders."""
        with self._order_lock:
            return list(self.active_orders.values())
    
    def get_order_history(self, limit: Optional[int] = None) -> List[Order]:
        """Get order history."""
        history = list(self.order_history)
        if limit:
            return history[-limit:]
        return history
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get order manager performance statistics."""
        if not self._performance_stats:
            return {"performance_monitoring": "disabled"}
        
        stats = self._performance_stats.copy()
        stats.update({
            "active_orders": len(self.active_orders),
            "total_orders": len(self.active_orders) + len(self.order_history),
            "daily_order_count": self._daily_order_count,
            "daily_order_value": self._daily_order_value,
            "arm64_optimized": self.is_arm64 and self.config.enable_arm64_optimizations,
            "shared_memory_enabled": self.config.enable_shared_memory
        })
        
        return stats
    
    def get_status(self) -> Dict[str, Any]:
        """Get order manager status."""
        return {
            "active_orders": len(self.active_orders),
            "total_orders": len(self.active_orders) + len(self.order_history),
            "daily_order_count": self._daily_order_count,
            "arm64_optimized": self.is_arm64 and self.config.enable_arm64_optimizations,
            "performance_monitoring": self.config.enable_performance_monitoring,
            "shared_memory_enabled": self.config.enable_shared_memory,
            "risk_checks_enabled": self.config.enable_risk_checks
        }
    
    def cleanup_expired_orders(self):
        """Cleanup expired orders."""
        current_time = datetime.now(timezone.utc)
        expired_orders = []
        
        with self._order_lock:
            for order in self.active_orders.values():
                if order.is_active:
                    # Check if order has expired based on timeout
                    time_since_creation = (current_time - order.created_at).total_seconds()
                    if time_since_creation > self.config.order_timeout_seconds:
                        expired_orders.append(order.order_id)
        
        # Mark expired orders
        for order_id in expired_orders:
            self.update_order_status(order_id, OrderStatus.EXPIRED)
            logger.info(f"Order {order_id} marked as expired")
    
    def shutdown(self):
        """Shutdown order manager and cleanup resources."""
        # Cancel batch timer
        if self._batch_timer:
            self._batch_timer.cancel()
        
        # Cleanup shared memory
        if self.shared_orders:
            self.shared_orders.close()
        
        logger.info("OrderManager shutdown completed")

# Export all public components
__all__ = [
    "OrderManager",
    "OrderConfig",
    "Order",
    "OrderStatus",
    "OrderPriority",
    "IS_ARM64"
]
