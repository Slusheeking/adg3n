"""
Enhanced trading engine for Deep Momentum Trading System with ARM64 optimizations.

This module provides the main trading engine that orchestrates all trading components
including execution, risk management, position management, and real-time monitoring
with ARM64-specific performance optimizations.
"""

import time
import os
import yaml
import asyncio
import threading
from typing import Dict, List, Optional, Any, Union, Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from decimal import Decimal
from enum import Enum
import json
import platform
from pathlib import Path
import numpy as np

# Internal imports
from ..communication.zmq_subscriber import ZMQSubscriber
from ..communication.zmq_publisher import ZMQPublisher
from ..utils.logger import get_logger, PerformanceLogger, TradingLogger
from ..utils.decorators import performance_monitor, retry_with_backoff
from ..utils.exceptions import TradingError, ConfigurationError, SystemError
from ..utils.validators import validate_trading_data, ValidationConfig
from ..utils.constants import TRADING_CONSTANTS
from ..utils.shared_memory import create_shared_array, create_shared_dict, SharedArray, SharedDict

# Trading components
from .alpaca_client import AlpacaClient, AlpacaConfig
from .execution_engine import ExecutionEngine, ExecutionConfig, ExecutionRequest
from .order_manager import OrderManager, OrderConfig
from .position_manager import PositionManager, PositionConfig
from .trade_logger import TradeLogger, TradeLoggerConfig, LoggedTrade

# Risk and infrastructure
from ..risk.risk_manager import RiskManager
from ..infrastructure.health_check import HealthMonitor
from ..data.polygon_client import PolygonClient

# ARM64 detection
IS_ARM64 = platform.machine().lower() in ['arm64', 'aarch64']

logger = get_logger(__name__)
perf_logger = PerformanceLogger(logger)
trading_logger = TradingLogger(logger)

class TradingEngineState(Enum):
    """Trading engine state enumeration."""
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    ERROR = "error"
    MAINTENANCE = "maintenance"

class TradingMode(Enum):
    """Trading mode enumeration."""
    LIVE = "live"
    PAPER = "paper"
    SIMULATION = "simulation"
    BACKTEST = "backtest"

@dataclass
class TradingConfig:
    """Enhanced trading configuration."""
    # Basic configuration
    trading_config_path: str = "config/trading_config.yaml"
    risk_config_path: str = "config/risk_config.yaml"
    trading_mode: TradingMode = TradingMode.PAPER
    
    # Performance targets
    daily_capital_limit: float = 50000.0
    target_daily_return_min: float = 0.03  # 3%
    target_daily_return_max: float = 0.06  # 6%
    target_sharpe_min: float = 4.0
    max_drawdown_percent: float = 0.05  # 5%
    
    # System configuration
    update_interval_seconds: float = 5.0
    health_check_interval_seconds: float = 60.0
    enable_real_time_monitoring: bool = True
    enable_performance_tracking: bool = True
    enable_arm64_optimizations: bool = True
    enable_shared_memory: bool = True
    shared_memory_size: int = 20000
    
    # Risk management
    enable_risk_management: bool = True
    max_position_count: int = 15000
    max_position_size_percent: float = 0.005
    enable_stop_loss: bool = True
    enable_take_profit: bool = True
    
    # Execution settings
    enable_smart_routing: bool = True
    max_order_size: float = 10000.0
    min_trade_threshold: float = 0.01
    execution_timeout_seconds: float = 30.0
    
    # Logging and monitoring
    enable_comprehensive_logging: bool = True
    log_level: str = "INFO"
    enable_audit_trail: bool = True
    backup_interval_hours: int = 24

@dataclass
class TradingMetrics:
    """Real-time trading metrics."""
    total_trades: int = 0
    successful_trades: int = 0
    failed_trades: int = 0
    total_volume: float = 0.0
    total_pnl: float = 0.0
    daily_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    current_positions: int = 0
    capital_utilization: float = 0.0
    avg_execution_time_ms: float = 0.0
    orders_per_second: float = 0.0
    uptime_hours: float = 0.0
    last_update: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

class TradingEngine:
    """Enhanced trading engine with ARM64 optimizations."""
    
    def __init__(self, config: Optional[TradingConfig] = None):
        self.config = config or TradingConfig()
        
        # Load configurations
        self.trading_config_data = self._load_config(self.config.trading_config_path)
        self.risk_config_data = self._load_config(self.config.risk_config_path)
        
        # Engine state
        self.state = TradingEngineState.STOPPED
        self.start_time = None
        self.is_running = False
        
        # Thread safety
        self._state_lock = threading.RLock()
        self._metrics_lock = threading.RLock()
        
        # Initialize components
        self._initialize_clients()
        self._initialize_managers()
        self._initialize_communication()
        self._initialize_monitoring()
        
        # Performance tracking
        self._setup_performance_monitoring()
        self._setup_shared_memory()
        
        # ARM64 optimizations
        self.is_arm64 = IS_ARM64
        if self.is_arm64 and self.config.enable_arm64_optimizations:
            self._apply_arm64_optimizations()
        
        # Real-time metrics
        self._current_metrics = TradingMetrics()
        
        # Background tasks
        self._background_tasks = []
        self._task_executor = None
        
        logger.info(f"TradingEngine initialized (ARM64: {self.is_arm64}, Mode: {self.config.trading_mode.value})")
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            config_file = Path(config_path)
            if not config_file.exists():
                logger.warning(f"Config file not found: {config_path}, using defaults")
                return {}
            
            with open(config_file, 'r') as f:
                return yaml.safe_load(f) or {}
        except Exception as e:
            logger.error(f"Failed to load config from {config_path}: {e}")
            raise ConfigurationError(f"Configuration loading failed: {e}")
    
    def _initialize_clients(self):
        """Initialize trading clients."""
        try:
            # Initialize Alpaca client
            alpaca_config = self.trading_config_data.get('alpaca', {})
            
            alpaca_client_config = AlpacaConfig(
                api_key=os.getenv(alpaca_config.get('api_key_env', 'ALPACA_API_KEY')),
                secret_key=os.getenv(alpaca_config.get('secret_key_env', 'ALPACA_SECRET_KEY')),
                base_url=alpaca_config.get('base_url'),
                data_url=alpaca_config.get('data_url'),
                paper_trading=(self.config.trading_mode == TradingMode.PAPER),
                enable_arm64_optimizations=self.config.enable_arm64_optimizations
            )
            
            self.alpaca_client = AlpacaClient(alpaca_client_config)
            
            # Initialize Polygon client if configured
            polygon_config = self.trading_config_data.get('polygon', {})
            if polygon_config:
                polygon_api_key = os.getenv(polygon_config.get('api_key_env', 'POLYGON_API_KEY'))
                if polygon_api_key:
                    self.polygon_client = PolygonClient(polygon_api_key)
                else:
                    self.polygon_client = None
                    logger.warning("Polygon API key not found, Polygon client disabled")
            else:
                self.polygon_client = None
            
        except Exception as e:
            logger.error(f"Failed to initialize clients: {e}")
            raise SystemError(f"Client initialization failed: {e}")
    
    def _initialize_managers(self):
        """Initialize trading managers."""
        try:
            # Order manager configuration
            order_config = OrderConfig(
                default_order_type=self.trading_config_data.get('order_management', {}).get('default_order_type', 'market'),
                max_active_orders=self.config.max_position_count,
                max_order_size=self.config.max_order_size,
                enable_arm64_optimizations=self.config.enable_arm64_optimizations,
                enable_shared_memory=self.config.enable_shared_memory
            )
            
            self.order_manager = OrderManager(
                broker_client=self.alpaca_client,
                config=order_config
            )
            
            # Position manager configuration
            position_config = PositionConfig(
                daily_capital_limit=self.config.daily_capital_limit,
                max_position_size_percent=self.config.max_position_size_percent,
                max_simultaneous_positions=self.config.max_position_count,
                enable_arm64_optimizations=self.config.enable_arm64_optimizations,
                enable_shared_memory=self.config.enable_shared_memory,
                enable_risk_management=self.config.enable_risk_management
            )
            
            self.position_manager = PositionManager(
                broker_client=self.alpaca_client,
                config=position_config
            )
            
            # Trade logger configuration
            trade_logger_config = TradeLoggerConfig(
                db_path=self.trading_config_data.get('trade_logging', {}).get('database_path', 'data/trades.db'),
                enable_arm64_optimizations=self.config.enable_arm64_optimizations,
                enable_shared_memory=self.config.enable_shared_memory,
                enable_real_time_analytics=self.config.enable_real_time_monitoring
            )
            
            self.trade_logger = TradeLogger(trade_logger_config)
            
            # Execution engine configuration
            execution_config = ExecutionConfig(
                min_trade_threshold=self.config.min_trade_threshold,
                max_order_size=self.config.max_order_size,
                enable_smart_routing=self.config.enable_smart_routing,
                enable_arm64_optimizations=self.config.enable_arm64_optimizations,
                order_timeout_seconds=self.config.execution_timeout_seconds
            )
            
            self.execution_engine = ExecutionEngine(
                broker_client=self.alpaca_client,
                order_manager=self.order_manager,
                position_manager=self.position_manager,
                config=execution_config
            )
            
        except Exception as e:
            logger.error(f"Failed to initialize managers: {e}")
            raise SystemError(f"Manager initialization failed: {e}")
    
    def _initialize_communication(self):
        """Initialize ZMQ communication."""
        try:
            # Get ZMQ ports from configuration
            risk_config = self.risk_config_data.get('zmq_ports', {})
            trading_config = self.trading_config_data.get('zmq_ports', {})
            
            risk_predictions_port = risk_config.get('predictions_subscriber', 5557)
            risk_approved_port = risk_config.get('risk_approved_predictions_publisher', 5558)
            execution_results_port = trading_config.get('execution_results_publisher', 5559)
            
            # Initialize risk manager
            if self.config.enable_risk_management:
                self.risk_manager = RiskManager(
                    risk_config=self.risk_config_data,
                    position_manager=self.position_manager,
                    alpaca_client=self.alpaca_client,
                    risk_predictions_port=risk_predictions_port,
                    risk_approved_predictions_port=risk_approved_port
                )
            else:
                self.risk_manager = None
            
            # ZMQ subscribers and publishers
            self.risk_subscriber = ZMQSubscriber(
                publishers=[f"tcp://localhost:{risk_approved_port}"],
                topics=["risk_approved_predictions"]
            )
            
            self.execution_publisher = ZMQPublisher(port=execution_results_port)
            
        except Exception as e:
            logger.error(f"Failed to initialize communication: {e}")
            raise SystemError(f"Communication initialization failed: {e}")
    
    def _initialize_monitoring(self):
        """Initialize health monitoring."""
        try:
            if self.config.enable_real_time_monitoring:
                self.health_monitor = HealthMonitor(
                    check_interval_seconds=self.config.health_check_interval_seconds,
                    alpaca_client=self.alpaca_client,
                    polygon_client=self.polygon_client
                )
            else:
                self.health_monitor = None
        
        except Exception as e:
            logger.error(f"Failed to initialize monitoring: {e}")
            raise SystemError(f"Monitoring initialization failed: {e}")
    
    def _setup_performance_monitoring(self):
        """Setup performance monitoring."""
        if self.config.enable_performance_tracking:
            self._performance_stats = {
                "engine_starts": 0,
                "total_runtime_hours": 0.0,
                "predictions_processed": 0,
                "orders_generated": 0,
                "execution_success_rate": 0.0,
                "avg_processing_time_ms": 0.0,
                "system_errors": 0,
                "last_error_time": None
            }
        else:
            self._performance_stats = None
    
    def _setup_shared_memory(self):
        """Setup shared memory for high-performance data sharing."""
        if self.config.enable_shared_memory:
            try:
                self.shared_metrics = create_shared_array(
                    name="trading_engine_metrics",
                    size=100,  # Store key metrics
                    dtype=np.float64
                )
                self.shared_state = create_shared_dict(
                    name="trading_engine_state",
                    max_items=1000
                )
            except Exception as e:
                logger.warning(f"Failed to setup shared memory: {e}")
                self.shared_metrics = None
                self.shared_state = None
        else:
            self.shared_metrics = None
            self.shared_state = None
    
    def _apply_arm64_optimizations(self):
        """Apply ARM64-specific optimizations."""
        # Optimize update intervals for ARM64 performance
        self.config.update_interval_seconds = max(1.0, self.config.update_interval_seconds * 0.5)
        
        # Optimize health check intervals
        self.config.health_check_interval_seconds = max(30.0, self.config.health_check_interval_seconds * 0.75)
        
        logger.debug(f"Applied ARM64 optimizations: update_interval={self.config.update_interval_seconds}s")
    
    @performance_monitor
    async def start(self):
        """Start the trading engine with comprehensive initialization."""
        with self._state_lock:
            if self.state != TradingEngineState.STOPPED:
                logger.warning(f"Cannot start engine in state: {self.state.value}")
                return False
            
            self.state = TradingEngineState.STARTING
        
        try:
            logger.info("Starting trading engine...")
            
            # Start components
            await self._start_components()
            
            # Setup message handlers
            self.risk_subscriber.add_handler("risk_approved_predictions", self._process_risk_approved_predictions)
            self.risk_subscriber.start()
            
            # Start background tasks
            await self._start_background_tasks()
            
            # Update state
            with self._state_lock:
                self.state = TradingEngineState.RUNNING
                self.is_running = True
                self.start_time = datetime.now(timezone.utc)
            
            # Update performance stats
            if self._performance_stats:
                self._performance_stats["engine_starts"] += 1
            
            logger.info(f"Trading engine started successfully (Mode: {self.config.trading_mode.value})")
            
            # Main execution loop
            await self._main_loop()
            
        except Exception as e:
            logger.error(f"Failed to start trading engine: {e}")
            with self._state_lock:
                self.state = TradingEngineState.ERROR
            raise TradingError(f"Engine startup failed: {e}")
    
    async def _start_components(self):
        """Start all trading components."""
        # Start execution engine
        self.execution_engine.start()
        
        # Start position manager background sync
        self.position_manager.start_background_sync()
        
        # Start trade logger background processing
        # (Already started in constructor)
        
        # Start risk manager if enabled
        if self.risk_manager:
            await self.risk_manager.start()
    
    async def _start_background_tasks(self):
        """Start background monitoring tasks."""
        self._task_executor = asyncio.create_task(self._background_task_manager())
    
    async def _background_task_manager(self):
        """Manage background tasks."""
        tasks = []
        
        # Health monitoring task
        if self.health_monitor:
            tasks.append(asyncio.create_task(self._health_monitoring_task()))
        
        # Metrics update task
        if self.config.enable_performance_tracking:
            tasks.append(asyncio.create_task(self._metrics_update_task()))
        
        # Position sync task
        tasks.append(asyncio.create_task(self._position_sync_task()))
        
        # Wait for all tasks
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
    
    async def _health_monitoring_task(self):
        """Background health monitoring."""
        while self.is_running:
            try:
                if self.health_monitor:
                    await self.health_monitor.check_alpaca_health()
                    if self.polygon_client:
                        await self.health_monitor.check_polygon_health()
                
                await asyncio.sleep(self.config.health_check_interval_seconds)
                
            except Exception as e:
                logger.error(f"Health monitoring error: {e}")
                await asyncio.sleep(self.config.health_check_interval_seconds * 2)
    
    async def _metrics_update_task(self):
        """Background metrics update."""
        while self.is_running:
            try:
                self._update_real_time_metrics()
                self._update_shared_memory()
                await asyncio.sleep(10.0)  # Update every 10 seconds
                
            except Exception as e:
                logger.error(f"Metrics update error: {e}")
                await asyncio.sleep(30.0)
    
    async def _position_sync_task(self):
        """Background position synchronization."""
        while self.is_running:
            try:
                # Sync is handled by position manager background thread
                await asyncio.sleep(60.0)  # Check every minute
                
            except Exception as e:
                logger.error(f"Position sync error: {e}")
                await asyncio.sleep(120.0)
    
    async def _main_loop(self):
        """Main trading engine loop."""
        logger.info("Trading engine main loop started")
        
        try:
            while self.is_running:
                # Update metrics
                self._update_real_time_metrics()
                
                # Check system health
                await self._check_system_health()
                
                # Process any pending tasks
                await self._process_pending_tasks()
                
                # Sleep for update interval
                await asyncio.sleep(self.config.update_interval_seconds)
                
        except asyncio.CancelledError:
            logger.info("Main loop cancelled")
        except Exception as e:
            logger.error(f"Error in main loop: {e}")
            if self._performance_stats:
                self._performance_stats["system_errors"] += 1
                self._performance_stats["last_error_time"] = datetime.now(timezone.utc)
        finally:
            await self._cleanup()
    
    async def _check_system_health(self):
        """Check overall system health."""
        # Check component health
        components_healthy = True
        
        # Check execution engine
        if not self.execution_engine.get_status()["running"]:
            logger.warning("Execution engine not running")
            components_healthy = False
        
        # Check position manager
        if self.position_manager.get_status()["background_sync_running"] is False:
            logger.warning("Position manager sync not running")
            components_healthy = False
        
        # Update state if unhealthy
        if not components_healthy:
            with self._state_lock:
                if self.state == TradingEngineState.RUNNING:
                    self.state = TradingEngineState.ERROR
                    logger.error("System health check failed")
    
    async def _process_pending_tasks(self):
        """Process any pending system tasks."""
        # Cleanup expired orders
        try:
            self.order_manager.cleanup_expired_orders()
        except Exception as e:
            logger.error(f"Error cleaning up orders: {e}")
    
    def _process_risk_approved_predictions(self, topic: str, message: Dict[str, Any]):
        """Process risk-approved predictions from risk manager."""
        start_time = time.time()
        
        try:
            logger.info(f"Received risk-approved predictions from topic: {topic}")
            predictions = message.get('data', {})
            
            if not predictions:
                logger.warning("Received empty predictions from risk manager")
                return
            
            # Update performance stats
            if self._performance_stats:
                self._performance_stats["predictions_processed"] += 1
            
            # Get current positions
            current_positions = self.position_manager.get_current_positions()
            
            # Generate execution requests
            execution_requests = self.execution_engine.generate_execution_requests(predictions, current_positions)
            
            if execution_requests:
                # Submit execution requests
                for request in execution_requests:
                    try:
                        request_id = self.execution_engine.execute_request(request, priority=0.7)
                        logger.debug(f"Submitted execution request: {request_id}")
                    except Exception as e:
                        logger.error(f"Failed to submit execution request: {e}")
                
                # Update performance stats
                if self._performance_stats:
                    self._performance_stats["orders_generated"] += len(execution_requests)
                
                logger.info(f"Generated and submitted {len(execution_requests)} execution requests")
            else:
                logger.info("No execution requests generated from predictions")
            
            # Log performance
            processing_time_ms = (time.time() - start_time) * 1000
            perf_logger.log_latency("prediction_processing", processing_time_ms)
            
            # Update average processing time
            if self._performance_stats:
                total_predictions = self._performance_stats["predictions_processed"]
                current_avg = self._performance_stats["avg_processing_time_ms"]
                self._performance_stats["avg_processing_time_ms"] = (
                    (current_avg * (total_predictions - 1) + processing_time_ms) / total_predictions
                )
            
        except Exception as e:
            logger.error(f"Error processing risk-approved predictions: {e}")
            if self._performance_stats:
                self._performance_stats["system_errors"] += 1
    
    def _update_real_time_metrics(self):
        """Update real-time trading metrics."""
        with self._metrics_lock:
            try:
                # Get metrics from components
                order_stats = self.order_manager.get_performance_stats()
                position_metrics = self.position_manager.get_portfolio_metrics()
                execution_stats = self.execution_engine.get_performance_stats()
                trade_metrics = self.trade_logger.get_trading_metrics()
                
                # Calculate uptime
                uptime_hours = 0.0
                if self.start_time:
                    uptime_hours = (datetime.now(timezone.utc) - self.start_time).total_seconds() / 3600
                
                # Update current metrics
                self._current_metrics = TradingMetrics(
                    total_trades=trade_metrics.total_trades,
                    successful_trades=trade_metrics.winning_trades,
                    failed_trades=trade_metrics.losing_trades,
                    total_volume=trade_metrics.total_volume,
                    total_pnl=trade_metrics.total_pnl,
                    daily_pnl=position_metrics.daily_pnl,
                    unrealized_pnl=position_metrics.total_unrealized_pnl,
                    realized_pnl=position_metrics.total_realized_pnl,
                    win_rate=trade_metrics.win_rate,
                    profit_factor=trade_metrics.profit_factor,
                    current_positions=position_metrics.total_positions,
                    capital_utilization=position_metrics.capital_utilization,
                    avg_execution_time_ms=execution_stats.get("avg_execution_time_ms", 0),
                    orders_per_second=execution_stats.get("orders_per_second", 0),
                    uptime_hours=uptime_hours
                )
                
            except Exception as e:
                logger.error(f"Error updating metrics: {e}")
    
    def _update_shared_memory(self):
        """Update shared memory with current metrics."""
        if not self.shared_metrics or not self.shared_state:
            return
        
        try:
            # Update metrics array
            with self.shared_metrics.write_lock() as array:
                array[0] = self._current_metrics.total_trades
                array[1] = self._current_metrics.total_pnl
                array[2] = self._current_metrics.daily_pnl
                array[3] = self._current_metrics.win_rate
                array[4] = self._current_metrics.current_positions
                array[5] = self._current_metrics.capital_utilization
                array[6] = self._current_metrics.uptime_hours
                # Add more metrics as needed
            
            # Update state dictionary
            state_data = {
                "state": self.state.value,
                "trading_mode": self.config.trading_mode.value,
                "is_running": self.is_running,
                "start_time": self.start_time.isoformat() if self.start_time else None,
                "last_update": datetime.now(timezone.utc).isoformat()
            }
            
            self.shared_state.put("engine_state", json.dumps(state_data).encode())
            
        except Exception as e:
            logger.warning(f"Failed to update shared memory: {e}")
    
    def stop(self):
        """Stop the trading engine gracefully."""
        with self._state_lock:
            if self.state == TradingEngineState.STOPPED:
                logger.warning("Trading engine is already stopped")
                return
            
            self.state = TradingEngineState.STOPPING
            self.is_running = False
        
        logger.info("Stopping trading engine...")
        
        # Cancel background tasks
        if self._task_executor and not self._task_executor.done():
            self._task_executor.cancel()
        
        # Stop ZMQ communication
        self.risk_subscriber.stop()
        
        # Stop components
        self.execution_engine.stop()
        self.position_manager.stop_background_sync()
        
        # Stop risk manager
        if self.risk_manager:
            asyncio.create_task(self.risk_manager.stop())
        
        with self._state_lock:
            self.state = TradingEngineState.STOPPED
        
        logger.info("Trading engine stopped")
    
    async def _cleanup(self):
        """Cleanup resources."""
        try:
            # Shutdown components
            self.execution_engine.shutdown()
            self.position_manager.shutdown()
            self.trade_logger.shutdown()
            
            # Cleanup shared memory
            if self.shared_metrics:
                self.shared_metrics.close()
            if self.shared_state:
                self.shared_state.close()
            
            logger.info("Trading engine cleanup completed")
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
    
    def get_current_metrics(self) -> TradingMetrics:
        """Get current trading metrics."""
        with self._metrics_lock:
            return self._current_metrics
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get trading engine performance statistics."""
        if not self._performance_stats:
            return {"performance_monitoring": "disabled"}
        
        stats = self._performance_stats.copy()
        
        # Add current runtime
        if self.start_time:
            runtime_hours = (datetime.now(timezone.utc) - self.start_time).total_seconds() / 3600
            stats["current_runtime_hours"] = runtime_hours
            stats["total_runtime_hours"] += runtime_hours
        
        # Add component stats
        stats.update({
            "current_state": self.state.value,
            "trading_mode": self.config.trading_mode.value,
            "arm64_optimized": self.is_arm64 and self.config.enable_arm64_optimizations,
            "shared_memory_enabled": self.config.enable_shared_memory,
            "components": {
                "alpaca_client": self.alpaca_client.get_status(),
                "execution_engine": self.execution_engine.get_status(),
                "order_manager": self.order_manager.get_status(),
                "position_manager": self.position_manager.get_status(),
                "trade_logger": self.trade_logger.get_status()
            }
        })
        
        return stats
    
    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive trading engine status."""
        return {
            "state": self.state.value,
            "trading_mode": self.config.trading_mode.value,
            "is_running": self.is_running,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "uptime_hours": self._current_metrics.uptime_hours,
            "current_positions": self._current_metrics.current_positions,
            "daily_pnl": self._current_metrics.daily_pnl,
            "total_trades": self._current_metrics.total_trades,
            "win_rate": self._current_metrics.win_rate,
            "capital_utilization": self._current_metrics.capital_utilization,
            "arm64_optimized": self.is_arm64 and self.config.enable_arm64_optimizations,
            "performance_monitoring": self.config.enable_performance_tracking,
            "shared_memory_enabled": self.config.enable_shared_memory,
            "risk_management_enabled": self.config.enable_risk_management
        }
    
    def get_current_portfolio_value(self) -> float:
        """Get current portfolio value."""
        return self.position_manager.get_portfolio_metrics().total_market_value

# Export all public components
__all__ = [
    "TradingEngine",
    "TradingConfig",
    "TradingMetrics",
    "TradingEngineState",
    "TradingMode",
    "IS_ARM64"
]
