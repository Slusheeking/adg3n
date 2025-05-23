"""
Enhanced position manager for Deep Momentum Trading System with ARM64 optimizations.

This module provides comprehensive position management capabilities including real-time
position tracking, risk management, portfolio optimization, and ARM64-specific
performance enhancements for high-frequency trading systems.
"""

import time
import threading
import asyncio
from typing import Dict, Optional, List, Any, Union, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from decimal import Decimal
from enum import Enum
import json
import platform
from collections import defaultdict, deque
import numpy as np

# Internal imports
from .alpaca_client import AlpacaClient
from ..utils.logger import get_logger, PerformanceLogger, TradingLogger
from ..utils.decorators import performance_monitor, retry_with_backoff, rate_limit
from ..utils.exceptions import PositionError, RiskError, ValidationError
from ..utils.validators import validate_trading_data, ValidationConfig
from ..utils.constants import TRADING_CONSTANTS
from ..utils.shared_memory import create_shared_array, create_shared_dict, SharedArray, SharedDict
from ..utils.helpers import calculate_returns, calculate_volatility, calculate_sharpe_ratio

# ARM64 detection
IS_ARM64 = platform.machine().lower() in ['arm64', 'aarch64']

logger = get_logger(__name__)
perf_logger = PerformanceLogger(logger)
trading_logger = TradingLogger(logger)

class PositionSide(Enum):
    """Position side enumeration."""
    LONG = "long"
    SHORT = "short"
    FLAT = "flat"

class PositionStatus(Enum):
    """Position status enumeration."""
    ACTIVE = "active"
    CLOSING = "closing"
    CLOSED = "closed"
    SUSPENDED = "suspended"

@dataclass
class PositionConfig:
    """Configuration for position manager."""
    daily_capital_limit: float = 50000.0
    max_position_size_percent: float = 0.005  # 0.5% of capital per position
    max_simultaneous_positions: int = 15000
    min_position_value: float = 5.0
    max_position_value: float = 2500.0  # $50K * 0.5% = $250, but allow up to $2500
    enable_dynamic_reallocation: bool = True
    enable_risk_management: bool = True
    enable_performance_monitoring: bool = True
    enable_arm64_optimizations: bool = True
    position_timeout_hours: float = 24.0
    max_daily_trades: int = 50000
    max_sector_exposure: float = 0.20  # 20% max exposure per sector
    enable_shared_memory: bool = True
    shared_memory_size: int = 20000
    sync_interval_seconds: float = 30.0
    enable_real_time_pnl: bool = True
    stop_loss_percent: float = 0.05  # 5% stop loss
    take_profit_percent: float = 0.10  # 10% take profit
    enable_position_sizing: bool = True
    volatility_adjustment: bool = True

@dataclass
class Position:
    """Enhanced position data structure."""
    symbol: str
    quantity: Union[float, Decimal]
    side: PositionSide
    avg_entry_price: float
    current_price: Optional[float] = None
    market_value: Optional[float] = None
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    cost_basis: Optional[float] = None
    status: PositionStatus = PositionStatus.ACTIVE
    entry_time: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_update: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    stop_loss_price: Optional[float] = None
    take_profit_price: Optional[float] = None
    sector: Optional[str] = None
    industry: Optional[str] = None
    volatility: Optional[float] = None
    beta: Optional[float] = None
    correlation: Optional[float] = None
    position_id: str = field(default_factory=lambda: f"pos_{int(time.time() * 1000000)}")
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Post-initialization processing."""
        if self.cost_basis is None:
            self.cost_basis = float(self.quantity) * self.avg_entry_price
        
        if self.market_value is None and self.current_price:
            self.market_value = float(self.quantity) * self.current_price
    
    @property
    def is_long(self) -> bool:
        """Check if position is long."""
        return self.side == PositionSide.LONG
    
    @property
    def is_short(self) -> bool:
        """Check if position is short."""
        return self.side == PositionSide.SHORT
    
    @property
    def is_flat(self) -> bool:
        """Check if position is flat."""
        return self.side == PositionSide.FLAT or abs(float(self.quantity)) < 1e-6
    
    @property
    def pnl_percent(self) -> float:
        """Calculate P&L percentage."""
        if self.cost_basis and self.cost_basis != 0:
            return (self.unrealized_pnl / abs(self.cost_basis)) * 100
        return 0.0
    
    @property
    def holding_period_hours(self) -> float:
        """Calculate holding period in hours."""
        return (self.last_update - self.entry_time).total_seconds() / 3600
    
    def update_price(self, new_price: float):
        """Update current price and recalculate metrics."""
        self.current_price = new_price
        self.market_value = float(self.quantity) * new_price
        
        if self.is_long:
            self.unrealized_pnl = (new_price - self.avg_entry_price) * float(self.quantity)
        elif self.is_short:
            self.unrealized_pnl = (self.avg_entry_price - new_price) * float(self.quantity)
        
        self.last_update = datetime.now(timezone.utc)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert position to dictionary."""
        return {
            "position_id": self.position_id,
            "symbol": self.symbol,
            "quantity": float(self.quantity),
            "side": self.side.value,
            "avg_entry_price": self.avg_entry_price,
            "current_price": self.current_price,
            "market_value": self.market_value,
            "unrealized_pnl": self.unrealized_pnl,
            "realized_pnl": self.realized_pnl,
            "cost_basis": self.cost_basis,
            "status": self.status.value,
            "entry_time": self.entry_time.isoformat(),
            "last_update": self.last_update.isoformat(),
            "stop_loss_price": self.stop_loss_price,
            "take_profit_price": self.take_profit_price,
            "pnl_percent": self.pnl_percent,
            "holding_period_hours": self.holding_period_hours,
            "sector": self.sector,
            "industry": self.industry,
            "volatility": self.volatility,
            "beta": self.beta,
            "tags": self.tags,
            "metadata": self.metadata
        }

@dataclass
class PortfolioMetrics:
    """Portfolio performance metrics."""
    total_positions: int = 0
    total_market_value: float = 0.0
    total_unrealized_pnl: float = 0.0
    total_realized_pnl: float = 0.0
    daily_pnl: float = 0.0
    capital_utilization: float = 0.0
    available_capital: float = 0.0
    allocated_capital: float = 0.0
    long_positions: int = 0
    short_positions: int = 0
    avg_position_size: float = 0.0
    largest_position: float = 0.0
    smallest_position: float = 0.0
    portfolio_beta: float = 0.0
    portfolio_volatility: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    sector_exposure: Dict[str, float] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

class PositionManager:
    """Enhanced position manager with ARM64 optimizations."""
    
    def __init__(self, 
                 broker_client: Optional[AlpacaClient] = None,
                 config: Optional[PositionConfig] = None):
        self.broker_client = broker_client
        self.config = config or PositionConfig()
        
        # Position storage
        self.positions: Dict[str, Position] = {}
        self.position_history: deque = deque(maxlen=100000)  # Large history for analysis
        self.positions_by_sector: Dict[str, List[str]] = defaultdict(list)
        
        # Capital management
        self.allocated_capital = 0.0
        self.available_capital = self.config.daily_capital_limit
        self.daily_trades = 0
        self.last_reset_date = datetime.now(timezone.utc).date()
        
        # Account information
        self.cash = 0.0
        self.total_equity = 0.0
        self.buying_power = 0.0
        
        # Thread safety
        self._position_lock = threading.RLock()
        
        # Performance monitoring
        self._setup_performance_monitoring()
        self._setup_shared_memory()
        
        # ARM64 optimizations
        self.is_arm64 = IS_ARM64
        if self.is_arm64 and self.config.enable_arm64_optimizations:
            self._apply_arm64_optimizations()
        
        # Background sync
        self._sync_thread = None
        self._running = False
        
        # Initialize with broker sync
        if self.broker_client:
            self.sync_with_broker()
        
        logger.info(f"PositionManager initialized (ARM64: {self.is_arm64})")
    
    def _setup_performance_monitoring(self):
        """Setup performance monitoring."""
        if self.config.enable_performance_monitoring:
            self._performance_stats = {
                "positions_opened": 0,
                "positions_closed": 0,
                "total_volume": 0.0,
                "total_pnl": 0.0,
                "winning_trades": 0,
                "losing_trades": 0,
                "avg_holding_time_hours": 0.0,
                "max_concurrent_positions": 0,
                "capital_efficiency": 0.0
            }
        else:
            self._performance_stats = None
    
    def _setup_shared_memory(self):
        """Setup shared memory for high-performance position tracking."""
        if self.config.enable_shared_memory:
            try:
                self.shared_positions = create_shared_dict(
                    name="position_manager_positions",
                    max_items=self.config.shared_memory_size
                )
                self.shared_metrics = create_shared_array(
                    name="position_manager_metrics",
                    size=100,  # Store key metrics
                    dtype=np.float64
                )
            except Exception as e:
                logger.warning(f"Failed to setup shared memory: {e}")
                self.shared_positions = None
                self.shared_metrics = None
        else:
            self.shared_positions = None
            self.shared_metrics = None
    
    def _apply_arm64_optimizations(self):
        """Apply ARM64-specific optimizations."""
        # Optimize sync intervals for ARM64 performance
        self.config.sync_interval_seconds = max(5.0, self.config.sync_interval_seconds * 0.5)
        
        # Optimize shared memory size for ARM64 cache efficiency
        if self.config.shared_memory_size % 64 != 0:
            self.config.shared_memory_size = ((self.config.shared_memory_size // 64) + 1) * 64
        
        logger.debug(f"Applied ARM64 optimizations: sync_interval={self.config.sync_interval_seconds}s")
    
    def start_background_sync(self):
        """Start background synchronization with broker."""
        if self._running or not self.broker_client:
            return
        
        self._running = True
        self._sync_thread = threading.Thread(target=self._sync_loop, daemon=True)
        self._sync_thread.start()
        logger.info("Background sync started")
    
    def stop_background_sync(self):
        """Stop background synchronization."""
        if not self._running:
            return
        
        self._running = False
        if self._sync_thread and self._sync_thread.is_alive():
            self._sync_thread.join(timeout=5.0)
        logger.info("Background sync stopped")
    
    def _sync_loop(self):
        """Background sync loop."""
        while self._running:
            try:
                self.sync_with_broker()
                time.sleep(self.config.sync_interval_seconds)
            except Exception as e:
                logger.error(f"Error in sync loop: {e}")
                time.sleep(self.config.sync_interval_seconds * 2)  # Longer delay on error
    
    def _reset_daily_limits_if_needed(self):
        """Reset daily limits if new day."""
        current_date = datetime.now(timezone.utc).date()
        if current_date > self.last_reset_date:
            self.daily_trades = 0
            self.available_capital = self.config.daily_capital_limit
            self.allocated_capital = 0.0
            self.last_reset_date = current_date
            logger.info("Daily limits reset")
    
    @performance_monitor
    @retry_with_backoff(max_retries=3, base_delay=1.0)
    def sync_with_broker(self):
        """Sync positions and account data with broker."""
        if not self.broker_client:
            logger.warning("No broker client configured for sync")
            return
        
        try:
            # Get account information
            account_data = self.broker_client.get_account()
            self.cash = account_data["cash"]
            self.total_equity = account_data["portfolio_value"]
            self.buying_power = account_data["buying_power"]
            
            # Get positions from broker
            broker_positions = self.broker_client.get_positions()
            
            with self._position_lock:
                # Update existing positions and add new ones
                broker_symbols = set()
                
                for pos_data in broker_positions:
                    symbol = pos_data["symbol"]
                    broker_symbols.add(symbol)
                    
                    quantity = pos_data["quantity"]
                    if abs(quantity) < 1e-6:
                        # Position is closed
                        if symbol in self.positions:
                            self._close_position(symbol, "broker_sync")
                        continue
                    
                    # Determine side
                    side = PositionSide.LONG if quantity > 0 else PositionSide.SHORT
                    
                    if symbol in self.positions:
                        # Update existing position
                        position = self.positions[symbol]
                        position.quantity = abs(quantity)
                        position.side = side
                        position.market_value = pos_data["market_value"]
                        position.unrealized_pnl = pos_data["unrealized_pl"]
                        position.cost_basis = pos_data["cost_basis"]
                        position.avg_entry_price = pos_data["avg_entry_price"]
                        if pos_data.get("current_price"):
                            position.update_price(pos_data["current_price"])
                    else:
                        # Create new position
                        position = Position(
                            symbol=symbol,
                            quantity=abs(quantity),
                            side=side,
                            avg_entry_price=pos_data["avg_entry_price"],
                            current_price=pos_data.get("current_price"),
                            market_value=pos_data["market_value"],
                            unrealized_pnl=pos_data["unrealized_pl"],
                            cost_basis=pos_data["cost_basis"]
                        )
                        self.positions[symbol] = position
                
                # Remove positions that no longer exist at broker
                local_symbols = set(self.positions.keys())
                closed_symbols = local_symbols - broker_symbols
                
                for symbol in closed_symbols:
                    self._close_position(symbol, "broker_sync")
            
            # Update shared memory
            self._update_shared_memory()
            
            logger.info(f"Synced with broker: {len(self.positions)} positions, ${self.total_equity:,.2f} equity")
            
        except Exception as e:
            logger.error(f"Failed to sync with broker: {e}")
            raise
    
    def _close_position(self, symbol: str, reason: str = "manual"):
        """Close a position and move to history."""
        if symbol not in self.positions:
            return
        
        position = self.positions[symbol]
        position.status = PositionStatus.CLOSED
        
        # Move to history
        self.position_history.append(position)
        del self.positions[symbol]
        
        # Remove from sector tracking
        if position.sector and symbol in self.positions_by_sector[position.sector]:
            self.positions_by_sector[position.sector].remove(symbol)
        
        # Release capital
        if position.cost_basis:
            self.release_capital(abs(position.cost_basis))
        
        # Update performance stats
        if self._performance_stats:
            self._performance_stats["positions_closed"] += 1
            if position.unrealized_pnl > 0:
                self._performance_stats["winning_trades"] += 1
            else:
                self._performance_stats["losing_trades"] += 1
            
            self._performance_stats["total_pnl"] += position.unrealized_pnl
        
        # Log trading event
        trading_logger.log_position({
            "symbol": symbol,
            "action": "closed",
            "reason": reason,
            "pnl": position.unrealized_pnl,
            "holding_period_hours": position.holding_period_hours
        })
        
        logger.info(f"Position closed: {symbol} (reason: {reason}, P&L: ${position.unrealized_pnl:.2f})")
    
    @performance_monitor
    def update_position(self, 
                       symbol: str, 
                       quantity: Union[float, Decimal], 
                       side: Union[str, PositionSide],
                       price: Optional[float] = None,
                       metadata: Optional[Dict[str, Any]] = None):
        """Update position with new trade."""
        self._reset_daily_limits_if_needed()
        
        if isinstance(side, str):
            side_enum = PositionSide.LONG if side.lower() == "buy" else PositionSide.SHORT
        else:
            side_enum = side
        
        quantity = abs(float(quantity))
        
        with self._position_lock:
            if symbol in self.positions:
                # Update existing position
                position = self.positions[symbol]
                
                if side_enum == position.side:
                    # Adding to position
                    old_quantity = float(position.quantity)
                    old_cost = position.cost_basis or 0
                    
                    new_quantity = old_quantity + quantity
                    new_cost = old_cost + (quantity * (price or position.avg_entry_price))
                    
                    position.quantity = new_quantity
                    position.cost_basis = new_cost
                    position.avg_entry_price = new_cost / new_quantity if new_quantity > 0 else 0
                    
                else:
                    # Reducing or reversing position
                    old_quantity = float(position.quantity)
                    
                    if quantity >= old_quantity:
                        # Close and potentially reverse
                        realized_pnl = self._calculate_realized_pnl(position, old_quantity, price)
                        position.realized_pnl += realized_pnl
                        
                        remaining_quantity = quantity - old_quantity
                        if remaining_quantity > 0:
                            # Reverse position
                            position.quantity = remaining_quantity
                            position.side = side_enum
                            position.avg_entry_price = price or position.current_price or 0
                            position.cost_basis = remaining_quantity * position.avg_entry_price
                            position.unrealized_pnl = 0
                        else:
                            # Close position
                            self._close_position(symbol, "trade_update")
                            return
                    else:
                        # Partial close
                        realized_pnl = self._calculate_realized_pnl(position, quantity, price)
                        position.realized_pnl += realized_pnl
                        position.quantity = old_quantity - quantity
                        position.cost_basis = (position.cost_basis or 0) * (position.quantity / old_quantity)
                
                position.last_update = datetime.now(timezone.utc)
                if price:
                    position.update_price(price)
                
            else:
                # Create new position
                if quantity < self.config.min_position_value / (price or 100):
                    logger.warning(f"Position size too small for {symbol}")
                    return
                
                position = Position(
                    symbol=symbol,
                    quantity=quantity,
                    side=side_enum,
                    avg_entry_price=price or 0,
                    current_price=price,
                    metadata=metadata or {}
                )
                
                # Set stop loss and take profit if enabled
                if price and self.config.enable_risk_management:
                    if side_enum == PositionSide.LONG:
                        position.stop_loss_price = price * (1 - self.config.stop_loss_percent)
                        position.take_profit_price = price * (1 + self.config.take_profit_percent)
                    else:
                        position.stop_loss_price = price * (1 + self.config.stop_loss_percent)
                        position.take_profit_price = price * (1 - self.config.take_profit_percent)
                
                self.positions[symbol] = position
                
                # Allocate capital
                if position.cost_basis:
                    self.allocate_capital(position.cost_basis)
                
                # Update performance stats
                if self._performance_stats:
                    self._performance_stats["positions_opened"] += 1
                    self._performance_stats["total_volume"] += quantity
                    self._performance_stats["max_concurrent_positions"] = max(
                        self._performance_stats["max_concurrent_positions"],
                        len(self.positions)
                    )
                
                # Log trading event
                trading_logger.log_position({
                    "symbol": symbol,
                    "action": "opened",
                    "quantity": quantity,
                    "side": side_enum.value,
                    "price": price
                })
        
        # Update shared memory
        self._update_shared_memory()
        
        self.daily_trades += 1
        logger.info(f"Position updated: {symbol} {side_enum.value} {quantity}")
    
    def _calculate_realized_pnl(self, position: Position, quantity: float, price: Optional[float]) -> float:
        """Calculate realized P&L for position close/reduction."""
        if not price:
            return 0.0
        
        if position.side == PositionSide.LONG:
            return (price - position.avg_entry_price) * quantity
        else:
            return (position.avg_entry_price - price) * quantity
    
    def can_open_new_position(self, symbol: str, position_value: float) -> bool:
        """Check if new position can be opened within limits."""
        self._reset_daily_limits_if_needed()
        
        # Check daily trade limit
        if self.daily_trades >= self.config.max_daily_trades:
            logger.warning(f"Daily trade limit reached: {self.config.max_daily_trades}")
            return False
        
        # Check position count limit
        if len(self.positions) >= self.config.max_simultaneous_positions:
            logger.warning(f"Max positions reached: {self.config.max_simultaneous_positions}")
            return False
        
        # Check position size limits
        if position_value < self.config.min_position_value:
            logger.warning(f"Position value ${position_value:.2f} below minimum ${self.config.min_position_value}")
            return False
        
        if position_value > self.config.max_position_value:
            logger.warning(f"Position value ${position_value:.2f} exceeds maximum ${self.config.max_position_value}")
            return False
        
        # Check available capital
        if position_value > self.available_capital:
            logger.warning(f"Insufficient capital: need ${position_value:.2f}, have ${self.available_capital:.2f}")
            return False
        
        # Check sector exposure if configured
        if self.config.enable_risk_management:
            # This would require sector classification - simplified for now
            pass
        
        return True
    
    def allocate_capital(self, amount: float) -> bool:
        """Allocate capital for new position."""
        if amount <= self.available_capital:
            self.allocated_capital += amount
            self.available_capital -= amount
            logger.debug(f"Allocated ${amount:.2f}, Available: ${self.available_capital:.2f}")
            return True
        return False
    
    def release_capital(self, amount: float):
        """Release capital when position is closed."""
        self.allocated_capital = max(0, self.allocated_capital - amount)
        self.available_capital = min(self.config.daily_capital_limit, self.available_capital + amount)
        logger.debug(f"Released ${amount:.2f}, Available: ${self.available_capital:.2f}")
    
    def get_position(self, symbol: str) -> Optional[Position]:
        """Get position for symbol."""
        return self.positions.get(symbol)
    
    def get_current_positions(self) -> Dict[str, float]:
        """Get current positions as symbol -> quantity mapping."""
        return {symbol: float(pos.quantity) for symbol, pos in self.positions.items()}
    
    def get_all_positions(self) -> List[Position]:
        """Get all active positions."""
        return list(self.positions.values())
    
    def get_positions_by_side(self, side: PositionSide) -> List[Position]:
        """Get positions by side (long/short)."""
        return [pos for pos in self.positions.values() if pos.side == side]
    
    def get_portfolio_metrics(self) -> PortfolioMetrics:
        """Calculate comprehensive portfolio metrics."""
        with self._position_lock:
            positions = list(self.positions.values())
        
        if not positions:
            return PortfolioMetrics(
                available_capital=self.available_capital,
                allocated_capital=self.allocated_capital
            )
        
        # Basic metrics
        total_positions = len(positions)
        total_market_value = sum(pos.market_value or 0 for pos in positions)
        total_unrealized_pnl = sum(pos.unrealized_pnl for pos in positions)
        total_realized_pnl = sum(pos.realized_pnl for pos in positions)
        daily_pnl = total_unrealized_pnl + total_realized_pnl
        
        # Position distribution
        long_positions = len([p for p in positions if p.side == PositionSide.LONG])
        short_positions = len([p for p in positions if p.side == PositionSide.SHORT])
        
        # Position sizes
        position_values = [pos.market_value or 0 for pos in positions if pos.market_value]
        avg_position_size = np.mean(position_values) if position_values else 0
        largest_position = max(position_values) if position_values else 0
        smallest_position = min(position_values) if position_values else 0
        
        # Capital utilization
        capital_utilization = (self.allocated_capital / self.config.daily_capital_limit) * 100
        
        # Risk metrics (simplified)
        returns = [pos.pnl_percent for pos in positions if pos.pnl_percent != 0]
        portfolio_volatility = np.std(returns) if len(returns) > 1 else 0
        
        # Performance metrics
        winning_positions = len([p for p in positions if p.unrealized_pnl > 0])
        win_rate = (winning_positions / total_positions * 100) if total_positions > 0 else 0
        
        # Sector exposure
        sector_exposure = defaultdict(float)
        for pos in positions:
            if pos.sector and pos.market_value:
                sector_exposure[pos.sector] += pos.market_value
        
        # Convert to percentages
        if total_market_value > 0:
            sector_exposure = {
                sector: (value / total_market_value * 100) 
                for sector, value in sector_exposure.items()
            }
        
        return PortfolioMetrics(
            total_positions=total_positions,
            total_market_value=total_market_value,
            total_unrealized_pnl=total_unrealized_pnl,
            total_realized_pnl=total_realized_pnl,
            daily_pnl=daily_pnl,
            capital_utilization=capital_utilization,
            available_capital=self.available_capital,
            allocated_capital=self.allocated_capital,
            long_positions=long_positions,
            short_positions=short_positions,
            avg_position_size=avg_position_size,
            largest_position=largest_position,
            smallest_position=smallest_position,
            portfolio_volatility=portfolio_volatility,
            win_rate=win_rate,
            sector_exposure=dict(sector_exposure)
        )
    
    def _update_shared_memory(self):
        """Update shared memory with current positions."""
        if not self.shared_positions:
            return
        
        try:
            # Update position data
            for symbol, position in self.positions.items():
                position_data = json.dumps(position.to_dict()).encode()
                self.shared_positions.put(symbol, position_data)
            
            # Update metrics array
            if self.shared_metrics:
                metrics = self.get_portfolio_metrics()
                with self.shared_metrics.write_lock() as array:
                    array[0] = metrics.total_positions
                    array[1] = metrics.total_market_value
                    array[2] = metrics.total_unrealized_pnl
                    array[3] = metrics.capital_utilization
                    array[4] = metrics.available_capital
                    # Add more metrics as needed
        
        except Exception as e:
            logger.warning(f"Failed to update shared memory: {e}")
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get position manager performance statistics."""
        if not self._performance_stats:
            return {"performance_monitoring": "disabled"}
        
        stats = self._performance_stats.copy()
        
        # Calculate additional metrics
        total_trades = stats["positions_opened"] + stats["positions_closed"]
        if total_trades > 0:
            stats["win_rate"] = (stats["winning_trades"] / stats["positions_closed"] * 100) if stats["positions_closed"] > 0 else 0
            stats["avg_pnl_per_trade"] = stats["total_pnl"] / stats["positions_closed"] if stats["positions_closed"] > 0 else 0
        
        stats.update({
            "current_positions": len(self.positions),
            "daily_trades": self.daily_trades,
            "capital_utilization": (self.allocated_capital / self.config.daily_capital_limit * 100),
            "arm64_optimized": self.is_arm64 and self.config.enable_arm64_optimizations,
            "shared_memory_enabled": self.config.enable_shared_memory
        })
        
        return stats
    
    def get_status(self) -> Dict[str, Any]:
        """Get position manager status."""
        return {
            "total_positions": len(self.positions),
            "allocated_capital": self.allocated_capital,
            "available_capital": self.available_capital,
            "daily_trades": self.daily_trades,
            "total_equity": self.total_equity,
            "cash": self.cash,
            "buying_power": self.buying_power,
            "arm64_optimized": self.is_arm64 and self.config.enable_arm64_optimizations,
            "background_sync_running": self._running,
            "performance_monitoring": self.config.enable_performance_monitoring,
            "shared_memory_enabled": self.config.enable_shared_memory
        }
    
    def shutdown(self):
        """Shutdown position manager and cleanup resources."""
        self.stop_background_sync()
        
        # Cleanup shared memory
        if self.shared_positions:
            self.shared_positions.close()
        if self.shared_metrics:
            self.shared_metrics.close()
        
        logger.info("PositionManager shutdown completed")

# Export all public components
__all__ = [
    "PositionManager",
    "PositionConfig",
    "Position",
    "PositionSide",
    "PositionStatus",
    "PortfolioMetrics",
    "IS_ARM64"
]
