"""
Enhanced trade logger for Deep Momentum Trading System with ARM64 optimizations.

This module provides comprehensive trade logging capabilities including real-time
trade tracking, performance analytics, audit trails, and ARM64-specific
optimizations for high-frequency trading systems.
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
import numpy as np
import pandas as pd
from pathlib import Path

# Internal imports
from ..storage.sqlite_storage import SQLiteTransactionStorage, TradeRecord
from ..storage.hdf5_storage import HDF5Storage
from ..storage.parquet_storage import ParquetStorage
from ..utils.logger import get_logger, PerformanceLogger
from ..utils.decorators import performance_monitor, retry_with_backoff
from ..utils.exceptions import LoggingError, StorageError, ValidationError
from ..utils.validators import validate_trading_data, ValidationConfig
from ..utils.constants import TRADING_CONSTANTS
from ..utils.shared_memory import create_shared_array, create_shared_dict, SharedArray, SharedDict
from ..utils.helpers import calculate_returns, calculate_sharpe_ratio

# ARM64 detection
IS_ARM64 = platform.machine().lower() in ['arm64', 'aarch64']

logger = get_logger(__name__)
perf_logger = PerformanceLogger(logger)

class TradeType(Enum):
    """Trade type enumeration."""
    BUY = "buy"
    SELL = "sell"
    SHORT = "short"
    COVER = "cover"

class TradeStatus(Enum):
    """Trade status enumeration."""
    PENDING = "pending"
    EXECUTED = "executed"
    CANCELLED = "cancelled"
    FAILED = "failed"

class LogLevel(Enum):
    """Logging level enumeration."""
    BASIC = "basic"
    DETAILED = "detailed"
    COMPREHENSIVE = "comprehensive"

@dataclass
class TradeLoggerConfig:
    """Configuration for trade logger."""
    db_path: str = "deep_momentum_trading/data/trades.db"
    hdf5_path: str = "deep_momentum_trading/data/trades.h5"
    parquet_path: str = "deep_momentum_trading/data/trades.parquet"
    log_level: LogLevel = LogLevel.DETAILED
    enable_real_time_analytics: bool = True
    enable_performance_monitoring: bool = True
    enable_arm64_optimizations: bool = True
    enable_shared_memory: bool = True
    shared_memory_size: int = 10000
    batch_size: int = 100
    flush_interval_seconds: float = 5.0
    enable_compression: bool = True
    enable_validation: bool = True
    max_memory_trades: int = 100000
    enable_audit_trail: bool = True
    backup_interval_hours: int = 24
    retention_days: int = 365

@dataclass
class LoggedTrade:
    """Enhanced trade data structure."""
    timestamp: Union[int, datetime]
    symbol: str
    side: Union[str, TradeType]
    quantity: Union[float, Decimal]
    price: Union[float, Decimal]
    commission: float = 0.0
    fees: float = 0.0
    order_id: str = ""
    trade_id: str = ""
    execution_id: str = ""
    strategy: Optional[str] = None
    pnl: Optional[float] = None
    unrealized_pnl: Optional[float] = None
    position_size_before: Optional[float] = None
    position_size_after: Optional[float] = None
    portfolio_value_before: Optional[float] = None
    portfolio_value_after: Optional[float] = None
    execution_time_ms: Optional[float] = None
    slippage_bps: Optional[float] = None
    market_impact_bps: Optional[float] = None
    venue: Optional[str] = None
    liquidity_flag: Optional[str] = None
    trade_type: TradeType = TradeType.BUY
    status: TradeStatus = TradeStatus.EXECUTED
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def __post_init__(self):
        """Post-initialization processing."""
        # Convert timestamp to datetime if needed
        if isinstance(self.timestamp, int):
            self.timestamp = datetime.fromtimestamp(self.timestamp, tz=timezone.utc)
        elif isinstance(self.timestamp, datetime) and self.timestamp.tzinfo is None:
            self.timestamp = self.timestamp.replace(tzinfo=timezone.utc)
        
        # Convert side to TradeType enum
        if isinstance(self.side, str):
            self.trade_type = TradeType(self.side.lower())
        else:
            self.trade_type = self.side
        
        # Generate IDs if not provided
        if not self.trade_id:
            self.trade_id = f"trade_{int(time.time() * 1000000)}"
        
        if not self.execution_id:
            self.execution_id = f"exec_{int(time.time() * 1000000)}"
    
    @property
    def trade_value(self) -> float:
        """Calculate trade value."""
        return float(self.quantity) * float(self.price)
    
    @property
    def total_cost(self) -> float:
        """Calculate total cost including fees."""
        return self.trade_value + self.commission + self.fees
    
    @property
    def is_buy(self) -> bool:
        """Check if trade is a buy."""
        return self.trade_type in [TradeType.BUY, TradeType.COVER]
    
    @property
    def is_sell(self) -> bool:
        """Check if trade is a sell."""
        return self.trade_type in [TradeType.SELL, TradeType.SHORT]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert trade to dictionary."""
        return {
            "trade_id": self.trade_id,
            "execution_id": self.execution_id,
            "timestamp": self.timestamp.isoformat(),
            "symbol": self.symbol,
            "side": self.trade_type.value,
            "quantity": float(self.quantity),
            "price": float(self.price),
            "trade_value": self.trade_value,
            "commission": self.commission,
            "fees": self.fees,
            "total_cost": self.total_cost,
            "order_id": self.order_id,
            "strategy": self.strategy,
            "pnl": self.pnl,
            "unrealized_pnl": self.unrealized_pnl,
            "position_size_before": self.position_size_before,
            "position_size_after": self.position_size_after,
            "portfolio_value_before": self.portfolio_value_before,
            "portfolio_value_after": self.portfolio_value_after,
            "execution_time_ms": self.execution_time_ms,
            "slippage_bps": self.slippage_bps,
            "market_impact_bps": self.market_impact_bps,
            "venue": self.venue,
            "liquidity_flag": self.liquidity_flag,
            "status": self.status.value,
            "tags": self.tags,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat()
        }

@dataclass
class TradingMetrics:
    """Trading performance metrics."""
    total_trades: int = 0
    total_volume: float = 0.0
    total_value: float = 0.0
    total_pnl: float = 0.0
    total_commission: float = 0.0
    total_fees: float = 0.0
    winning_trades: int = 0
    losing_trades: int = 0
    avg_trade_size: float = 0.0
    avg_trade_value: float = 0.0
    avg_pnl_per_trade: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    avg_execution_time_ms: float = 0.0
    avg_slippage_bps: float = 0.0
    trades_per_hour: float = 0.0
    unique_symbols: int = 0
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

class TradeLogger:
    """Enhanced trade logger with ARM64 optimizations."""
    
    def __init__(self, config: Optional[TradeLoggerConfig] = None):
        self.config = config or TradeLoggerConfig()
        
        # Initialize storage backends
        self._setup_storage()
        
        # In-memory trade buffer
        self.trade_buffer: deque = deque(maxlen=self.config.max_memory_trades)
        self.pending_trades: deque = deque()
        
        # Thread safety
        self._buffer_lock = threading.RLock()
        self._flush_lock = threading.Lock()
        
        # Performance monitoring
        self._setup_performance_monitoring()
        self._setup_shared_memory()
        self._setup_validation()
        
        # ARM64 optimizations
        self.is_arm64 = IS_ARM64
        if self.is_arm64 and self.config.enable_arm64_optimizations:
            self._apply_arm64_optimizations()
        
        # Background processing
        self._running = False
        self._flush_thread = None
        self._analytics_thread = None
        
        # Real-time metrics
        self._current_metrics = TradingMetrics()
        self._metrics_lock = threading.RLock()
        
        # Start background processing
        self.start_background_processing()
        
        logger.info(f"TradeLogger initialized (ARM64: {self.is_arm64})")
    
    def _setup_storage(self):
        """Setup storage backends."""
        try:
            # SQLite for transactional data
            self.db_storage = SQLiteTransactionStorage(self.config.db_path)
            
            # HDF5 for time series analytics
            if self.config.enable_real_time_analytics:
                self.hdf5_storage = HDF5Storage(self.config.hdf5_path)
            else:
                self.hdf5_storage = None
            
            # Parquet for data science workflows
            self.parquet_storage = ParquetStorage(self.config.parquet_path)
            
        except Exception as e:
            logger.error(f"Failed to setup storage: {e}")
            raise LoggingError(f"Storage initialization failed: {e}")
    
    def _setup_performance_monitoring(self):
        """Setup performance monitoring."""
        if self.config.enable_performance_monitoring:
            self._performance_stats = {
                "trades_logged": 0,
                "trades_per_second": 0.0,
                "avg_log_time_ms": 0.0,
                "total_log_time_ms": 0.0,
                "buffer_size": 0,
                "flush_count": 0,
                "storage_errors": 0
            }
        else:
            self._performance_stats = None
    
    def _setup_shared_memory(self):
        """Setup shared memory for high-performance data sharing."""
        if self.config.enable_shared_memory:
            try:
                self.shared_trades = create_shared_dict(
                    name="trade_logger_trades",
                    max_items=self.config.shared_memory_size
                )
                self.shared_metrics = create_shared_array(
                    name="trade_logger_metrics",
                    size=50,  # Store key metrics
                    dtype=np.float64
                )
            except Exception as e:
                logger.warning(f"Failed to setup shared memory: {e}")
                self.shared_trades = None
                self.shared_metrics = None
        else:
            self.shared_trades = None
            self.shared_metrics = None
    
    def _setup_validation(self):
        """Setup trade validation."""
        if self.config.enable_validation:
            self.validation_config = ValidationConfig(level="normal")
        else:
            self.validation_config = None
    
    def _apply_arm64_optimizations(self):
        """Apply ARM64-specific optimizations."""
        # Optimize batch sizes for ARM64 SIMD operations
        if self.config.batch_size % 64 != 0:
            self.config.batch_size = ((self.config.batch_size // 64) + 1) * 64
        
        # Reduce flush interval for ARM64 performance
        self.config.flush_interval_seconds = max(1.0, self.config.flush_interval_seconds * 0.5)
        
        logger.debug(f"Applied ARM64 optimizations: batch_size={self.config.batch_size}")
    
    def start_background_processing(self):
        """Start background processing threads."""
        if self._running:
            return
        
        self._running = True
        
        # Start flush thread
        self._flush_thread = threading.Thread(target=self._flush_loop, daemon=True)
        self._flush_thread.start()
        
        # Start analytics thread
        if self.config.enable_real_time_analytics:
            self._analytics_thread = threading.Thread(target=self._analytics_loop, daemon=True)
            self._analytics_thread.start()
        
        logger.info("Background processing started")
    
    def stop_background_processing(self):
        """Stop background processing threads."""
        if not self._running:
            return
        
        self._running = False
        
        # Wait for threads to finish
        if self._flush_thread and self._flush_thread.is_alive():
            self._flush_thread.join(timeout=5.0)
        
        if self._analytics_thread and self._analytics_thread.is_alive():
            self._analytics_thread.join(timeout=5.0)
        
        # Final flush
        self._flush_pending_trades()
        
        logger.info("Background processing stopped")
    
    def _flush_loop(self):
        """Background flush loop."""
        while self._running:
            try:
                self._flush_pending_trades()
                time.sleep(self.config.flush_interval_seconds)
            except Exception as e:
                logger.error(f"Error in flush loop: {e}")
                time.sleep(self.config.flush_interval_seconds * 2)
    
    def _analytics_loop(self):
        """Background analytics loop."""
        while self._running:
            try:
                self._update_real_time_metrics()
                self._update_shared_memory()
                time.sleep(10.0)  # Update every 10 seconds
            except Exception as e:
                logger.error(f"Error in analytics loop: {e}")
                time.sleep(30.0)
    
    @performance_monitor
    def log_trade(self, trade: LoggedTrade):
        """Log a single trade with comprehensive validation and processing."""
        start_time = time.time()
        
        try:
            # Validate trade data
            if self.config.enable_validation:
                self._validate_trade(trade)
            
            # Add to buffer
            with self._buffer_lock:
                self.trade_buffer.append(trade)
                self.pending_trades.append(trade)
            
            # Update performance stats
            if self._performance_stats:
                log_time_ms = (time.time() - start_time) * 1000
                self._performance_stats["trades_logged"] += 1
                self._performance_stats["total_log_time_ms"] += log_time_ms
                self._performance_stats["avg_log_time_ms"] = (
                    self._performance_stats["total_log_time_ms"] / 
                    self._performance_stats["trades_logged"]
                )
                self._performance_stats["buffer_size"] = len(self.pending_trades)
            
            # Log performance metric
            perf_logger.log_latency("trade_logging", (time.time() - start_time) * 1000, symbol=trade.symbol)
            
            logger.debug(f"Trade logged: {trade.trade_id} {trade.symbol} {trade.trade_type.value} {trade.quantity}")
            
        except Exception as e:
            logger.error(f"Failed to log trade {trade.trade_id}: {e}")
            if self._performance_stats:
                self._performance_stats["storage_errors"] += 1
            raise LoggingError(f"Trade logging failed: {e}")
    
    def _validate_trade(self, trade: LoggedTrade):
        """Validate trade data."""
        trade_data = {
            "symbol": trade.symbol,
            "quantity": float(trade.quantity),
            "price": float(trade.price),
            "side": trade.trade_type.value
        }
        
        validation_errors = validate_trading_data(trade_data, self.validation_config)
        if validation_errors:
            error_messages = [error.message for error in validation_errors]
            raise ValidationError(f"Trade validation failed: {'; '.join(error_messages)}")
    
    def _flush_pending_trades(self):
        """Flush pending trades to storage."""
        if not self.pending_trades:
            return
        
        with self._flush_lock:
            # Get batch of trades to flush
            batch = []
            batch_size = min(self.config.batch_size, len(self.pending_trades))
            
            for _ in range(batch_size):
                if self.pending_trades:
                    batch.append(self.pending_trades.popleft())
            
            if not batch:
                return
            
            # Flush to storage backends
            try:
                self._flush_to_sqlite(batch)
                
                if self.hdf5_storage:
                    self._flush_to_hdf5(batch)
                
                self._flush_to_parquet(batch)
                
                # Update performance stats
                if self._performance_stats:
                    self._performance_stats["flush_count"] += 1
                    self._performance_stats["buffer_size"] = len(self.pending_trades)
                
                logger.debug(f"Flushed {len(batch)} trades to storage")
                
            except Exception as e:
                # Put trades back in queue for retry
                with self._buffer_lock:
                    for trade in reversed(batch):
                        self.pending_trades.appendleft(trade)
                
                logger.error(f"Failed to flush trades: {e}")
                if self._performance_stats:
                    self._performance_stats["storage_errors"] += 1
    
    def _flush_to_sqlite(self, trades: List[LoggedTrade]):
        """Flush trades to SQLite storage."""
        for trade in trades:
            trade_record = TradeRecord(
                timestamp=int(trade.timestamp.timestamp()),
                symbol=trade.symbol,
                side=trade.trade_type.value,
                quantity=float(trade.quantity),
                price=float(trade.price),
                commission=trade.commission,
                order_id=trade.order_id,
                strategy=trade.strategy
            )
            
            metadata = trade.to_dict()
            self.db_storage.insert_trade(trade_record, metadata)
    
    def _flush_to_hdf5(self, trades: List[LoggedTrade]):
        """Flush trades to HDF5 storage."""
        if not self.hdf5_storage:
            return
        
        # Convert trades to DataFrame
        trade_data = []
        for trade in trades:
            trade_data.append(trade.to_dict())
        
        df = pd.DataFrame(trade_data)
        
        # Store in HDF5 with compression
        dataset_name = f"trades_{datetime.now(timezone.utc).strftime('%Y%m%d')}"
        self.hdf5_storage.store_dataframe(df, dataset_name, compress=self.config.enable_compression)
    
    def _flush_to_parquet(self, trades: List[LoggedTrade]):
        """Flush trades to Parquet storage."""
        # Convert trades to DataFrame
        trade_data = []
        for trade in trades:
            trade_data.append(trade.to_dict())
        
        df = pd.DataFrame(trade_data)
        
        # Append to Parquet file
        self.parquet_storage.append_dataframe(df, partition_cols=['symbol'])
    
    def _update_real_time_metrics(self):
        """Update real-time trading metrics."""
        with self._metrics_lock:
            trades = list(self.trade_buffer)
            
            if not trades:
                return
            
            # Calculate metrics
            total_trades = len(trades)
            total_volume = sum(float(t.quantity) for t in trades)
            total_value = sum(t.trade_value for t in trades)
            total_pnl = sum(t.pnl or 0 for t in trades)
            total_commission = sum(t.commission for t in trades)
            total_fees = sum(t.fees for t in trades)
            
            winning_trades = len([t for t in trades if (t.pnl or 0) > 0])
            losing_trades = len([t for t in trades if (t.pnl or 0) < 0])
            
            # Calculate derived metrics
            avg_trade_size = total_volume / total_trades if total_trades > 0 else 0
            avg_trade_value = total_value / total_trades if total_trades > 0 else 0
            avg_pnl_per_trade = total_pnl / total_trades if total_trades > 0 else 0
            win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
            
            # Calculate profit factor
            gross_profit = sum(t.pnl for t in trades if (t.pnl or 0) > 0)
            gross_loss = abs(sum(t.pnl for t in trades if (t.pnl or 0) < 0))
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
            
            # Calculate execution metrics
            execution_times = [t.execution_time_ms for t in trades if t.execution_time_ms]
            avg_execution_time_ms = np.mean(execution_times) if execution_times else 0
            
            slippages = [t.slippage_bps for t in trades if t.slippage_bps]
            avg_slippage_bps = np.mean(slippages) if slippages else 0
            
            # Calculate time-based metrics
            if trades:
                time_span_hours = (trades[-1].timestamp - trades[0].timestamp).total_seconds() / 3600
                trades_per_hour = total_trades / time_span_hours if time_span_hours > 0 else 0
            else:
                trades_per_hour = 0
            
            unique_symbols = len(set(t.symbol for t in trades))
            
            # Update metrics
            self._current_metrics = TradingMetrics(
                total_trades=total_trades,
                total_volume=total_volume,
                total_value=total_value,
                total_pnl=total_pnl,
                total_commission=total_commission,
                total_fees=total_fees,
                winning_trades=winning_trades,
                losing_trades=losing_trades,
                avg_trade_size=avg_trade_size,
                avg_trade_value=avg_trade_value,
                avg_pnl_per_trade=avg_pnl_per_trade,
                win_rate=win_rate,
                profit_factor=profit_factor,
                avg_execution_time_ms=avg_execution_time_ms,
                avg_slippage_bps=avg_slippage_bps,
                trades_per_hour=trades_per_hour,
                unique_symbols=unique_symbols
            )
    
    def _update_shared_memory(self):
        """Update shared memory with current data."""
        if not self.shared_trades or not self.shared_metrics:
            return
        
        try:
            # Update recent trades in shared memory
            recent_trades = list(self.trade_buffer)[-100:]  # Last 100 trades
            for i, trade in enumerate(recent_trades):
                trade_data = json.dumps(trade.to_dict()).encode()
                self.shared_trades.put(f"trade_{i}", trade_data)
            
            # Update metrics in shared memory
            with self.shared_metrics.write_lock() as array:
                array[0] = self._current_metrics.total_trades
                array[1] = self._current_metrics.total_volume
                array[2] = self._current_metrics.total_value
                array[3] = self._current_metrics.total_pnl
                array[4] = self._current_metrics.win_rate
                array[5] = self._current_metrics.profit_factor
                array[6] = self._current_metrics.avg_execution_time_ms
                array[7] = self._current_metrics.trades_per_hour
                # Add more metrics as needed
        
        except Exception as e:
            logger.warning(f"Failed to update shared memory: {e}")
    
    @performance_monitor
    def get_trades_by_date_range(self, 
                                start_timestamp: Union[int, datetime], 
                                end_timestamp: Union[int, datetime]) -> List[LoggedTrade]:
        """Retrieve trades from storage within date range."""
        try:
            # Convert datetime to timestamp if needed
            if isinstance(start_timestamp, datetime):
                start_timestamp = int(start_timestamp.timestamp())
            if isinstance(end_timestamp, datetime):
                end_timestamp = int(end_timestamp.timestamp())
            
            # Get trades from SQLite
            df = self.db_storage.get_trades_by_date_range(start_timestamp, end_timestamp)
            
            trades = []
            for _, row in df.iterrows():
                metadata = {}
                if row.get('metadata'):
                    try:
                        metadata = json.loads(row['metadata'])
                    except json.JSONDecodeError:
                        logger.warning(f"Could not decode metadata for trade {row['order_id']}")
                
                trade = LoggedTrade(
                    timestamp=datetime.fromtimestamp(row['timestamp'], tz=timezone.utc),
                    symbol=row['symbol'],
                    side=row['side'],
                    quantity=row['quantity'],
                    price=row['price'],
                    commission=row['commission'],
                    order_id=row['order_id'],
                    strategy=row.get('strategy'),
                    metadata=metadata
                )
                trades.append(trade)
            
            logger.info(f"Retrieved {len(trades)} trades from history")
            return trades
            
        except Exception as e:
            logger.error(f"Failed to retrieve trades: {e}")
            return []
    
    def get_portfolio_positions_as_of(self, as_of_timestamp: Union[int, datetime]) -> Dict[str, float]:
        """Calculate portfolio positions as of specific timestamp."""
        try:
            if isinstance(as_of_timestamp, datetime):
                as_of_timestamp = int(as_of_timestamp.timestamp())
            
            positions = self.db_storage.get_portfolio_positions(as_of_timestamp)
            logger.info(f"Retrieved positions as of {as_of_timestamp}: {len(positions)} symbols")
            return positions
            
        except Exception as e:
            logger.error(f"Failed to get portfolio positions: {e}")
            return {}
    
    def get_trading_metrics(self) -> TradingMetrics:
        """Get current trading metrics."""
        with self._metrics_lock:
            return self._current_metrics
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get trade logger performance statistics."""
        if not self._performance_stats:
            return {"performance_monitoring": "disabled"}
        
        stats = self._performance_stats.copy()
        
        # Calculate trades per second
        if stats["total_log_time_ms"] > 0:
            stats["trades_per_second"] = (stats["trades_logged"] * 1000) / stats["total_log_time_ms"]
        
        stats.update({
            "buffer_size": len(self.trade_buffer),
            "pending_trades": len(self.pending_trades),
            "arm64_optimized": self.is_arm64 and self.config.enable_arm64_optimizations,
            "shared_memory_enabled": self.config.enable_shared_memory,
            "background_processing": self._running
        })
        
        return stats
    
    def get_status(self) -> Dict[str, Any]:
        """Get trade logger status."""
        return {
            "total_trades_logged": self._performance_stats["trades_logged"] if self._performance_stats else 0,
            "buffer_size": len(self.trade_buffer),
            "pending_trades": len(self.pending_trades),
            "background_processing": self._running,
            "arm64_optimized": self.is_arm64 and self.config.enable_arm64_optimizations,
            "performance_monitoring": self.config.enable_performance_monitoring,
            "shared_memory_enabled": self.config.enable_shared_memory,
            "real_time_analytics": self.config.enable_real_time_analytics
        }
    
    def export_trades_to_csv(self, 
                           output_path: str,
                           start_date: Optional[datetime] = None,
                           end_date: Optional[datetime] = None,
                           symbols: Optional[List[str]] = None) -> bool:
        """Export trades to CSV file."""
        try:
            # Get date range
            if not start_date:
                start_date = datetime.now(timezone.utc) - timedelta(days=30)
            if not end_date:
                end_date = datetime.now(timezone.utc)
            
            # Get trades
            trades = self.get_trades_by_date_range(start_date, end_date)
            
            # Filter by symbols if specified
            if symbols:
                trades = [t for t in trades if t.symbol in symbols]
            
            # Convert to DataFrame and export
            trade_data = [trade.to_dict() for trade in trades]
            df = pd.DataFrame(trade_data)
            df.to_csv(output_path, index=False)
            
            logger.info(f"Exported {len(trades)} trades to {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to export trades to CSV: {e}")
            return False
    
    def shutdown(self):
        """Shutdown trade logger and cleanup resources."""
        self.stop_background_processing()
        
        # Cleanup shared memory
        if self.shared_trades:
            self.shared_trades.close()
        if self.shared_metrics:
            self.shared_metrics.close()
        
        logger.info("TradeLogger shutdown completed")

# Export all public components
__all__ = [
    "TradeLogger",
    "TradeLoggerConfig",
    "LoggedTrade",
    "TradingMetrics",
    "TradeType",
    "TradeStatus",
    "LogLevel",
    "IS_ARM64"
]
