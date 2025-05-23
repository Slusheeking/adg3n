"""
Enhanced Alpaca trading client for Deep Momentum Trading System with ARM64 optimizations.

This module provides comprehensive Alpaca API integration with advanced features including
connection pooling, rate limiting, error handling, and ARM64-specific optimizations.
"""

import os
import asyncio
import time
import threading
from typing import List, Optional, Dict, Any, Union, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from decimal import Decimal
import json
from enum import Enum
import platform

# Alpaca imports
from alpaca.trading.client import TradingClient
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.trading.requests import (
    MarketOrderRequest, LimitOrderRequest, StopOrderRequest, 
    StopLimitOrderRequest, TrailingStopOrderRequest
)
from alpaca.trading.enums import OrderSide, TimeInForce, OrderType as AlpacaOrderType
from alpaca.data.requests import StockBarsRequest, StockLatestQuoteRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.common.exceptions import APIError

# Internal imports
from ..utils.logger import get_logger, PerformanceLogger
from ..utils.decorators import retry_with_backoff, rate_limit, performance_monitor
from ..utils.exceptions import BrokerError, OrderError, DataError
from ..utils.validators import validate_trading_data, ValidationConfig
from ..utils.constants import TRADING_CONSTANTS
from ...config.settings import config_manager

# ARM64 detection
IS_ARM64 = platform.machine().lower() in ['arm64', 'aarch64']

logger = get_logger(__name__)
perf_logger = PerformanceLogger(logger)

class OrderType(Enum):
    """Enhanced order types."""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"
    TRAILING_STOP = "trailing_stop"

class OrderSideEnum(Enum):
    """Order side enumeration."""
    BUY = "buy"
    SELL = "sell"

class TimeInForceEnum(Enum):
    """Time in force enumeration."""
    DAY = "day"
    GTC = "gtc"
    IOC = "ioc"
    FOK = "fok"

@dataclass
class AlpacaConfig:
    """Configuration for Alpaca client."""
    api_key: Optional[str] = None
    secret_key: Optional[str] = None
    base_url: Optional[str] = None
    data_url: Optional[str] = None
    paper_trading: bool = True
    max_retries: int = 3
    retry_delay: float = 1.0
    rate_limit_requests: int = 200
    rate_limit_window: int = 60
    connection_timeout: float = 30.0
    read_timeout: float = 30.0
    enable_caching: bool = True
    cache_ttl: int = 300  # 5 minutes
    enable_arm64_optimizations: bool = True
    batch_size: int = 100
    enable_performance_monitoring: bool = True
    enable_validation: bool = True
    validation_level: str = "normal"

@dataclass
class OrderRequest:
    """Enhanced order request with validation."""
    symbol: str
    quantity: Union[int, float, Decimal]
    side: OrderSideEnum
    order_type: OrderType
    time_in_force: TimeInForceEnum = TimeInForceEnum.DAY
    limit_price: Optional[Union[float, Decimal]] = None
    stop_price: Optional[Union[float, Decimal]] = None
    trail_price: Optional[Union[float, Decimal]] = None
    trail_percent: Optional[float] = None
    client_order_id: Optional[str] = None
    extended_hours: bool = False
    
    def __post_init__(self):
        """Validate order request after initialization."""
        if self.order_type in [OrderType.LIMIT, OrderType.STOP_LIMIT] and self.limit_price is None:
            raise ValueError("Limit price required for limit orders")
        
        if self.order_type in [OrderType.STOP, OrderType.STOP_LIMIT] and self.stop_price is None:
            raise ValueError("Stop price required for stop orders")
        
        if self.order_type == OrderType.TRAILING_STOP:
            if self.trail_price is None and self.trail_percent is None:
                raise ValueError("Trail price or trail percent required for trailing stop orders")

class AlpacaClient:
    """Enhanced Alpaca trading client with ARM64 optimizations."""
    
    def __init__(self, config: Optional[AlpacaConfig] = None):
        # Load configuration from config manager if not provided
        if config is None:
            config_data = config_manager.get('trading_config.alpaca', {})
            self.config = AlpacaConfig(**config_data) if config_data else AlpacaConfig()
        else:
            self.config = config
            
        self._setup_credentials()
        self._initialize_clients()
        self._setup_caching()
        self._setup_performance_monitoring()
        self._setup_validation()
        
        # ARM64 optimizations
        self.is_arm64 = IS_ARM64
        if self.is_arm64 and self.config.enable_arm64_optimizations:
            self._apply_arm64_optimizations()
        
        logger.info(f"AlpacaClient initialized (Paper: {self.config.paper_trading}, ARM64: {self.is_arm64})")
    
    def _setup_credentials(self):
        """Setup API credentials from config or environment."""
        self.api_key = self.config.api_key or os.getenv('ALPACA_API_KEY')
        self.secret_key = self.config.secret_key or os.getenv('ALPACA_SECRET_KEY')
        self.base_url = self.config.base_url or os.getenv('ALPACA_BASE_URL')
        self.data_url = self.config.data_url or os.getenv('ALPACA_DATA_URL')
        
        if not self.api_key or not self.secret_key:
            raise ValueError(
                "Alpaca API key and secret key must be provided via config or environment variables "
                "(ALPACA_API_KEY, ALPACA_SECRET_KEY)"
            )
        
        # Determine paper trading mode
        if self.base_url and "api.alpaca.markets" in self.base_url:
            self.config.paper_trading = False
    
    def _initialize_clients(self):
        """Initialize Alpaca trading and data clients."""
        try:
            self.trading_client = TradingClient(
                api_key=self.api_key,
                secret_key=self.secret_key,
                paper=self.config.paper_trading,
                base_url=self.base_url
            )
            
            self.data_client = StockHistoricalDataClient(
                api_key=self.api_key,
                secret_key=self.secret_key,
                base_url=self.data_url
            )
            
            # Test connection
            self._test_connection()
            
        except Exception as e:
            raise BrokerError(f"Failed to initialize Alpaca clients: {e}")
    
    def _test_connection(self):
        """Test connection to Alpaca API."""
        try:
            account = self.trading_client.get_account()
            logger.info(f"Connected to Alpaca (Account: {account.id})")
        except Exception as e:
            raise BrokerError(f"Failed to connect to Alpaca API: {e}")
    
    def _setup_caching(self):
        """Setup caching for API responses."""
        self._cache = {} if self.config.enable_caching else None
        self._cache_timestamps = {} if self.config.enable_caching else None
        self._cache_lock = threading.RLock()
    
    def _setup_performance_monitoring(self):
        """Setup performance monitoring."""
        if self.config.enable_performance_monitoring:
            self._performance_stats = {
                "orders_placed": 0,
                "orders_cancelled": 0,
                "data_requests": 0,
                "api_errors": 0,
                "total_latency_ms": 0.0,
                "avg_latency_ms": 0.0
            }
        else:
            self._performance_stats = None
    
    def _setup_validation(self):
        """Setup data validation."""
        if self.config.enable_validation:
            self.validation_config = ValidationConfig(level=self.config.validation_level)
        else:
            self.validation_config = None
    
    def _apply_arm64_optimizations(self):
        """Apply ARM64-specific optimizations."""
        # Optimize batch sizes for ARM64 SIMD operations
        if self.config.batch_size % 64 != 0:
            self.config.batch_size = ((self.config.batch_size // 64) + 1) * 64
            logger.debug(f"Adjusted batch size to {self.config.batch_size} for ARM64 optimization")
    
    def _get_cache_key(self, method: str, **kwargs) -> str:
        """Generate cache key for API calls."""
        key_data = {"method": method, **kwargs}
        return json.dumps(key_data, sort_keys=True, default=str)
    
    def _get_cached_result(self, cache_key: str) -> Optional[Any]:
        """Get cached result if valid."""
        if not self.config.enable_caching:
            return None
        
        with self._cache_lock:
            if cache_key in self._cache:
                timestamp = self._cache_timestamps.get(cache_key, 0)
                if time.time() - timestamp < self.config.cache_ttl:
                    return self._cache[cache_key]
                else:
                    # Remove expired cache entry
                    del self._cache[cache_key]
                    del self._cache_timestamps[cache_key]
        
        return None
    
    def _cache_result(self, cache_key: str, result: Any):
        """Cache API result."""
        if not self.config.enable_caching:
            return
        
        with self._cache_lock:
            self._cache[cache_key] = result
            self._cache_timestamps[cache_key] = time.time()
    
    def _update_performance_stats(self, operation: str, latency_ms: float, success: bool = True):
        """Update performance statistics."""
        if not self._performance_stats:
            return
        
        if operation == "place_order" and success:
            self._performance_stats["orders_placed"] += 1
        elif operation == "cancel_order" and success:
            self._performance_stats["orders_cancelled"] += 1
        elif operation.startswith("get_") and success:
            self._performance_stats["data_requests"] += 1
        
        if not success:
            self._performance_stats["api_errors"] += 1
        
        # Update latency stats
        self._performance_stats["total_latency_ms"] += latency_ms
        total_operations = (
            self._performance_stats["orders_placed"] + 
            self._performance_stats["orders_cancelled"] + 
            self._performance_stats["data_requests"]
        )
        
        if total_operations > 0:
            self._performance_stats["avg_latency_ms"] = (
                self._performance_stats["total_latency_ms"] / total_operations
            )
    
    @retry_with_backoff(max_retries=3, base_delay=1.0)
    @rate_limit(calls=200, period=60)
    @performance_monitor
    def place_order(self, order_request: OrderRequest) -> Dict[str, Any]:
        """Place order with enhanced error handling and validation."""
        start_time = time.time()
        
        try:
            # Validate order request
            if self.validation_config:
                order_data = {
                    "symbol": order_request.symbol,
                    "quantity": float(order_request.quantity),
                    "side": order_request.side.value,
                    "order_type": order_request.order_type.value
                }
                
                validation_errors = validate_trading_data(order_data, self.validation_config)
                if validation_errors:
                    error_messages = [error.message for error in validation_errors]
                    raise OrderError(f"Order validation failed: {'; '.join(error_messages)}")
            
            # Convert to Alpaca order request
            alpaca_request = self._create_alpaca_order_request(order_request)
            
            # Submit order
            order_response = self.trading_client.submit_order(alpaca_request)
            
            # Log performance
            latency_ms = (time.time() - start_time) * 1000
            self._update_performance_stats("place_order", latency_ms, True)
            perf_logger.log_latency("place_order", latency_ms, symbol=order_request.symbol)
            
            logger.info(f"Order placed successfully: {order_response.id}")
            
            return {
                "order_id": order_response.id,
                "symbol": order_response.symbol,
                "quantity": float(order_response.qty),
                "side": order_response.side.value,
                "order_type": order_response.order_type.value,
                "status": order_response.status.value,
                "submitted_at": order_response.submitted_at,
                "client_order_id": order_response.client_order_id
            }
            
        except APIError as e:
            latency_ms = (time.time() - start_time) * 1000
            self._update_performance_stats("place_order", latency_ms, False)
            logger.error(f"Alpaca API error placing order: {e}")
            raise OrderError(f"Failed to place order: {e}")
        
        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            self._update_performance_stats("place_order", latency_ms, False)
            logger.error(f"Unexpected error placing order: {e}")
            raise OrderError(f"Unexpected error placing order: {e}")
    
    def _create_alpaca_order_request(self, order_request: OrderRequest):
        """Create Alpaca-specific order request."""
        # Convert enums to Alpaca enums
        side = OrderSide.BUY if order_request.side == OrderSideEnum.BUY else OrderSide.SELL
        tif = TimeInForce(order_request.time_in_force.value.upper())
        
        # Common parameters
        common_params = {
            "symbol": order_request.symbol,
            "qty": float(order_request.quantity),
            "side": side,
            "time_in_force": tif
        }
        
        if order_request.client_order_id:
            common_params["client_order_id"] = order_request.client_order_id
        
        if order_request.extended_hours:
            common_params["extended_hours"] = True
        
        # Create specific order type
        if order_request.order_type == OrderType.MARKET:
            return MarketOrderRequest(**common_params)
        
        elif order_request.order_type == OrderType.LIMIT:
            return LimitOrderRequest(
                **common_params,
                limit_price=float(order_request.limit_price)
            )
        
        elif order_request.order_type == OrderType.STOP:
            return StopOrderRequest(
                **common_params,
                stop_price=float(order_request.stop_price)
            )
        
        elif order_request.order_type == OrderType.STOP_LIMIT:
            return StopLimitOrderRequest(
                **common_params,
                limit_price=float(order_request.limit_price),
                stop_price=float(order_request.stop_price)
            )
        
        elif order_request.order_type == OrderType.TRAILING_STOP:
            params = common_params.copy()
            if order_request.trail_price:
                params["trail_price"] = float(order_request.trail_price)
            if order_request.trail_percent:
                params["trail_percent"] = order_request.trail_percent
            
            return TrailingStopOrderRequest(**params)
        
        else:
            raise ValueError(f"Unsupported order type: {order_request.order_type}")
    
    @retry_with_backoff(max_retries=3, base_delay=1.0)
    @rate_limit(calls=200, period=60)
    @performance_monitor
    def cancel_order(self, order_id: str) -> bool:
        """Cancel order with enhanced error handling."""
        start_time = time.time()
        
        try:
            self.trading_client.cancel_order_by_id(order_id)
            
            latency_ms = (time.time() - start_time) * 1000
            self._update_performance_stats("cancel_order", latency_ms, True)
            perf_logger.log_latency("cancel_order", latency_ms, order_id=order_id)
            
            logger.info(f"Order cancelled successfully: {order_id}")
            return True
            
        except APIError as e:
            latency_ms = (time.time() - start_time) * 1000
            self._update_performance_stats("cancel_order", latency_ms, False)
            logger.error(f"Failed to cancel order {order_id}: {e}")
            raise OrderError(f"Failed to cancel order: {e}")
    
    @retry_with_backoff(max_retries=3, base_delay=1.0)
    @rate_limit(calls=200, period=60)
    @performance_monitor
    def get_account(self) -> Dict[str, Any]:
        """Get account information with caching."""
        cache_key = self._get_cache_key("get_account")
        cached_result = self._get_cached_result(cache_key)
        
        if cached_result:
            return cached_result
        
        start_time = time.time()
        
        try:
            account = self.trading_client.get_account()
            
            result = {
                "id": account.id,
                "account_number": account.account_number,
                "status": account.status.value,
                "currency": account.currency,
                "buying_power": float(account.buying_power),
                "cash": float(account.cash),
                "portfolio_value": float(account.portfolio_value),
                "equity": float(account.equity),
                "last_equity": float(account.last_equity),
                "multiplier": int(account.multiplier),
                "day_trade_count": account.day_trade_count,
                "daytrade_buying_power": float(account.daytrade_buying_power),
                "pattern_day_trader": account.pattern_day_trader
            }
            
            self._cache_result(cache_key, result)
            
            latency_ms = (time.time() - start_time) * 1000
            self._update_performance_stats("get_account", latency_ms, True)
            
            return result
            
        except APIError as e:
            latency_ms = (time.time() - start_time) * 1000
            self._update_performance_stats("get_account", latency_ms, False)
            raise BrokerError(f"Failed to get account information: {e}")
    
    @retry_with_backoff(max_retries=3, base_delay=1.0)
    @rate_limit(calls=200, period=60)
    @performance_monitor
    def get_positions(self) -> List[Dict[str, Any]]:
        """Get all positions with caching."""
        cache_key = self._get_cache_key("get_positions")
        cached_result = self._get_cached_result(cache_key)
        
        if cached_result:
            return cached_result
        
        start_time = time.time()
        
        try:
            positions = self.trading_client.get_all_positions()
            
            result = []
            for position in positions:
                result.append({
                    "symbol": position.symbol,
                    "quantity": float(position.qty),
                    "side": position.side.value,
                    "market_value": float(position.market_value),
                    "cost_basis": float(position.cost_basis),
                    "unrealized_pl": float(position.unrealized_pl),
                    "unrealized_plpc": float(position.unrealized_plpc),
                    "avg_entry_price": float(position.avg_entry_price),
                    "current_price": float(position.current_price) if position.current_price else None
                })
            
            self._cache_result(cache_key, result)
            
            latency_ms = (time.time() - start_time) * 1000
            self._update_performance_stats("get_positions", latency_ms, True)
            
            return result
            
        except APIError as e:
            latency_ms = (time.time() - start_time) * 1000
            self._update_performance_stats("get_positions", latency_ms, False)
            raise BrokerError(f"Failed to get positions: {e}")
    
    @retry_with_backoff(max_retries=3, base_delay=1.0)
    @rate_limit(calls=200, period=60)
    @performance_monitor
    def get_orders(self, status: Optional[str] = None, limit: int = 100) -> List[Dict[str, Any]]:
        """Get orders with optional status filter."""
        start_time = time.time()
        
        try:
            orders = self.trading_client.get_orders(status=status, limit=limit)
            
            result = []
            for order in orders:
                result.append({
                    "id": order.id,
                    "client_order_id": order.client_order_id,
                    "symbol": order.symbol,
                    "quantity": float(order.qty),
                    "filled_qty": float(order.filled_qty) if order.filled_qty else 0.0,
                    "side": order.side.value,
                    "order_type": order.order_type.value,
                    "time_in_force": order.time_in_force.value,
                    "status": order.status.value,
                    "limit_price": float(order.limit_price) if order.limit_price else None,
                    "stop_price": float(order.stop_price) if order.stop_price else None,
                    "submitted_at": order.submitted_at,
                    "filled_at": order.filled_at,
                    "cancelled_at": order.cancelled_at,
                    "expired_at": order.expired_at
                })
            
            latency_ms = (time.time() - start_time) * 1000
            self._update_performance_stats("get_orders", latency_ms, True)
            
            return result
            
        except APIError as e:
            latency_ms = (time.time() - start_time) * 1000
            self._update_performance_stats("get_orders", latency_ms, False)
            raise BrokerError(f"Failed to get orders: {e}")
    
    @retry_with_backoff(max_retries=3, base_delay=1.0)
    @rate_limit(calls=200, period=60)
    @performance_monitor
    def get_bars(self, symbol: str, timeframe: str, start: datetime, end: datetime) -> Dict[str, Any]:
        """Get historical bar data with enhanced timeframe support."""
        cache_key = self._get_cache_key("get_bars", symbol=symbol, timeframe=timeframe, 
                                       start=start.isoformat(), end=end.isoformat())
        cached_result = self._get_cached_result(cache_key)
        
        if cached_result:
            return cached_result
        
        start_time = time.time()
        
        try:
            # Enhanced timeframe mapping
            timeframe_map = {
                '1min': TimeFrame.Minute,
                '5min': TimeFrame.FiveMinutes,
                '15min': TimeFrame.FifteenMinutes,
                '30min': TimeFrame.ThirtyMinutes,
                '1hour': TimeFrame.Hour,
                '1day': TimeFrame.Day,
                '1week': TimeFrame.Week,
                '1month': TimeFrame.Month
            }
            
            if timeframe not in timeframe_map:
                raise ValueError(f"Unsupported timeframe: {timeframe}. Supported: {list(timeframe_map.keys())}")
            
            request_params = StockBarsRequest(
                symbol_or_symbols=[symbol],
                timeframe=timeframe_map[timeframe],
                start=start,
                end=end
            )
            
            bars_response = self.data_client.get_stock_bars(request_params)
            bars = bars_response[symbol]
            
            result = {
                "symbol": symbol,
                "timeframe": timeframe,
                "bars": []
            }
            
            for bar in bars:
                result["bars"].append({
                    "timestamp": bar.timestamp,
                    "open": float(bar.open),
                    "high": float(bar.high),
                    "low": float(bar.low),
                    "close": float(bar.close),
                    "volume": int(bar.volume),
                    "trade_count": bar.trade_count,
                    "vwap": float(bar.vwap) if bar.vwap else None
                })
            
            self._cache_result(cache_key, result)
            
            latency_ms = (time.time() - start_time) * 1000
            self._update_performance_stats("get_bars", latency_ms, True)
            
            return result
            
        except APIError as e:
            latency_ms = (time.time() - start_time) * 1000
            self._update_performance_stats("get_bars", latency_ms, False)
            raise DataError(f"Failed to get bars for {symbol}: {e}")
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get client performance statistics."""
        if not self._performance_stats:
            return {"performance_monitoring": "disabled"}
        
        stats = self._performance_stats.copy()
        stats.update({
            "cache_enabled": self.config.enable_caching,
            "cache_size": len(self._cache) if self._cache else 0,
            "arm64_optimized": self.is_arm64 and self.config.enable_arm64_optimizations,
            "paper_trading": self.config.paper_trading
        })
        
        return stats
    
    def get_status(self) -> Dict[str, Any]:
        """Get client status information."""
        try:
            # Test connection
            account = self.trading_client.get_account()
            connection_status = "connected"
        except:
            connection_status = "disconnected"
        
        return {
            "status": connection_status,
            "paper_trading": self.config.paper_trading,
            "arm64_optimized": self.is_arm64 and self.config.enable_arm64_optimizations,
            "caching_enabled": self.config.enable_caching,
            "validation_enabled": self.config.enable_validation,
            "performance_monitoring": self.config.enable_performance_monitoring
        }
    
    def clear_cache(self):
        """Clear all cached data."""
        if self._cache:
            with self._cache_lock:
                self._cache.clear()
                self._cache_timestamps.clear()
            logger.info("Cache cleared")
    
    def shutdown(self):
        """Shutdown client and cleanup resources."""
        self.clear_cache()
        logger.info("AlpacaClient shutdown completed")

# Export all public components
__all__ = [
    "AlpacaClient",
    "AlpacaConfig",
    "OrderRequest",
    "OrderType",
    "OrderSideEnum", 
    "TimeInForceEnum",
    "IS_ARM64"
]
