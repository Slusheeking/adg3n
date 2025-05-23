"""
Enhanced execution engine for Deep Momentum Trading System with ARM64 optimizations.

This module provides comprehensive order execution capabilities with advanced features
including smart order routing, execution algorithms, latency optimization, and
ARM64-specific performance enhancements.
"""

import time
import asyncio
import threading
from typing import Dict, List, Optional, Any, Callable, Union, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from decimal import Decimal
from enum import Enum
import numpy as np
import platform
from concurrent.futures import ThreadPoolExecutor, as_completed
import queue
import json

# Internal imports
from .alpaca_client import AlpacaClient, OrderRequest, OrderType, OrderSideEnum, TimeInForceEnum
from .order_manager import OrderManager
from .position_manager import PositionManager
from ..communication.zmq_publisher import ZMQPublisher
from ..communication.zmq_subscriber import ZMQSubscriber
from ..utils.logger import get_logger, PerformanceLogger, TradingLogger
from ..utils.decorators import performance_monitor, retry_with_backoff, rate_limit
from ..utils.exceptions import ExecutionError, OrderError, RiskError
from ..utils.validators import validate_trading_data, ValidationConfig
from ..utils.constants import TRADING_CONSTANTS
from ..utils.shared_memory import create_shared_array, SharedArray

# ARM64 detection
IS_ARM64 = platform.machine().lower() in ['arm64', 'aarch64']

logger = get_logger(__name__)
perf_logger = PerformanceLogger(logger)
trading_logger = TradingLogger(logger)

class ExecutionAlgorithm(Enum):
    """Execution algorithm types."""
    MARKET = "market"
    TWAP = "twap"  # Time-Weighted Average Price
    VWAP = "vwap"  # Volume-Weighted Average Price
    IMPLEMENTATION_SHORTFALL = "implementation_shortfall"
    ICEBERG = "iceberg"
    SMART_ROUTING = "smart_routing"

class ExecutionStatus(Enum):
    """Execution status types."""
    PENDING = "pending"
    SUBMITTED = "submitted"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    FAILED = "failed"

@dataclass
class ExecutionConfig:
    """Configuration for execution engine."""
    min_trade_threshold: float = 0.01
    max_order_size: float = 10000.0
    default_algorithm: ExecutionAlgorithm = ExecutionAlgorithm.SMART_ROUTING
    enable_smart_routing: bool = True
    enable_latency_optimization: bool = True
    enable_arm64_optimizations: bool = True
    max_concurrent_orders: int = 100
    order_timeout_seconds: float = 30.0
    retry_failed_orders: bool = True
    max_retries: int = 3
    retry_delay: float = 1.0
    enable_performance_monitoring: bool = True
    enable_risk_checks: bool = True
    enable_position_tracking: bool = True
    batch_execution_size: int = 50
    execution_delay_ms: float = 10.0  # Minimum delay between executions
    zmq_publisher_port: int = 5559
    zmq_subscriber_port: int = 5560
    enable_shared_memory: bool = True
    shared_memory_size: int = 10000

@dataclass
class ExecutionRequest:
    """Enhanced execution request."""
    symbol: str
    quantity: Union[float, Decimal]
    side: OrderSideEnum
    algorithm: ExecutionAlgorithm = ExecutionAlgorithm.SMART_ROUTING
    order_type: OrderType = OrderType.MARKET
    time_in_force: TimeInForceEnum = TimeInForceEnum.DAY
    limit_price: Optional[Union[float, Decimal]] = None
    stop_price: Optional[Union[float, Decimal]] = None
    urgency: float = 0.5  # 0.0 = patient, 1.0 = aggressive
    max_participation_rate: float = 0.1  # Maximum % of volume
    client_order_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate execution request."""
        if self.quantity <= 0:
            raise ValueError("Quantity must be positive")
        
        if not 0.0 <= self.urgency <= 1.0:
            raise ValueError("Urgency must be between 0.0 and 1.0")
        
        if not 0.0 <= self.max_participation_rate <= 1.0:
            raise ValueError("Max participation rate must be between 0.0 and 1.0")

@dataclass
class ExecutionResult:
    """Execution result with comprehensive information."""
    request_id: str
    symbol: str
    status: ExecutionStatus
    submitted_quantity: float
    filled_quantity: float = 0.0
    remaining_quantity: float = 0.0
    avg_fill_price: Optional[float] = None
    order_ids: List[str] = field(default_factory=list)
    execution_time_ms: float = 0.0
    slippage_bps: Optional[float] = None
    implementation_shortfall_bps: Optional[float] = None
    error_message: Optional[str] = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = field(default_factory=dict)

class ExecutionEngine:
    """Enhanced execution engine with ARM64 optimizations."""
    
    def __init__(self, 
                 broker_client: AlpacaClient,
                 order_manager: OrderManager,
                 position_manager: Optional[PositionManager] = None,
                 config: Optional[ExecutionConfig] = None):
        self.broker_client = broker_client
        self.order_manager = order_manager
        self.position_manager = position_manager
        self.config = config or ExecutionConfig()
        
        # Initialize components
        self._setup_communication()
        self._setup_performance_monitoring()
        self._setup_shared_memory()
        self._setup_execution_queue()
        
        # ARM64 optimizations
        self.is_arm64 = IS_ARM64
        if self.is_arm64 and self.config.enable_arm64_optimizations:
            self._apply_arm64_optimizations()
        
        # Execution state
        self._running = False
        self._execution_thread = None
        self._pending_executions = {}
        self._execution_lock = threading.RLock()
        
        logger.info(f"ExecutionEngine initialized (ARM64: {self.is_arm64})")
    
    def _setup_communication(self):
        """Setup ZMQ communication."""
        self.zmq_publisher = ZMQPublisher(port=self.config.zmq_publisher_port)
        self.zmq_subscriber = ZMQSubscriber(port=self.config.zmq_subscriber_port)
        
        # Subscribe to risk-approved predictions
        self.zmq_subscriber.subscribe("risk_approved_predictions", self._handle_risk_approved_predictions)
        self.zmq_subscriber.subscribe("execution_requests", self._handle_execution_requests)
    
    def _setup_performance_monitoring(self):
        """Setup performance monitoring."""
        if self.config.enable_performance_monitoring:
            self._performance_stats = {
                "total_executions": 0,
                "successful_executions": 0,
                "failed_executions": 0,
                "total_volume": 0.0,
                "total_execution_time_ms": 0.0,
                "avg_execution_time_ms": 0.0,
                "avg_slippage_bps": 0.0,
                "orders_per_second": 0.0
            }
        else:
            self._performance_stats = None
    
    def _setup_shared_memory(self):
        """Setup shared memory for high-performance data sharing."""
        if self.config.enable_shared_memory:
            try:
                self.shared_positions = create_shared_array(
                    name="execution_positions",
                    size=self.config.shared_memory_size,
                    dtype=np.float64
                )
                self.shared_orders = create_shared_array(
                    name="execution_orders", 
                    size=self.config.shared_memory_size,
                    dtype=np.float64
                )
            except Exception as e:
                logger.warning(f"Failed to setup shared memory: {e}")
                self.shared_positions = None
                self.shared_orders = None
        else:
            self.shared_positions = None
            self.shared_orders = None
    
    def _setup_execution_queue(self):
        """Setup execution queue with ARM64 optimizations."""
        self.execution_queue = queue.PriorityQueue(maxsize=self.config.max_concurrent_orders)
        self.result_queue = queue.Queue()
        
        # Thread pool for concurrent execution
        max_workers = min(self.config.max_concurrent_orders, 20)
        if self.is_arm64:
            # Optimize for ARM64 core count
            import os
            max_workers = min(max_workers, os.cpu_count() or 4)
        
        self.executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="execution")
    
    def _apply_arm64_optimizations(self):
        """Apply ARM64-specific optimizations."""
        # Optimize batch sizes for ARM64 SIMD operations
        if self.config.batch_execution_size % 64 != 0:
            self.config.batch_execution_size = ((self.config.batch_execution_size // 64) + 1) * 64
        
        # Reduce execution delay for ARM64 performance
        self.config.execution_delay_ms = max(1.0, self.config.execution_delay_ms * 0.5)
        
        logger.debug(f"Applied ARM64 optimizations: batch_size={self.config.batch_execution_size}")
    
    @performance_monitor
    def start(self):
        """Start the execution engine."""
        if self._running:
            logger.warning("ExecutionEngine is already running")
            return
        
        self._running = True
        self._execution_thread = threading.Thread(target=self._execution_loop, daemon=True)
        self._execution_thread.start()
        
        # Start ZMQ subscriber
        self.zmq_subscriber.start()
        
        logger.info("ExecutionEngine started")
    
    def stop(self):
        """Stop the execution engine."""
        if not self._running:
            return
        
        self._running = False
        
        # Stop ZMQ subscriber
        self.zmq_subscriber.stop()
        
        # Wait for execution thread to finish
        if self._execution_thread and self._execution_thread.is_alive():
            self._execution_thread.join(timeout=5.0)
        
        # Shutdown executor
        self.executor.shutdown(wait=True)
        
        logger.info("ExecutionEngine stopped")
    
    def _execution_loop(self):
        """Main execution loop."""
        logger.info("Execution loop started")
        
        while self._running:
            try:
                # Process execution queue
                self._process_execution_queue()
                
                # Process results
                self._process_execution_results()
                
                # Small delay to prevent CPU spinning
                time.sleep(self.config.execution_delay_ms / 1000.0)
                
            except Exception as e:
                logger.error(f"Error in execution loop: {e}")
                time.sleep(1.0)  # Longer delay on error
        
        logger.info("Execution loop stopped")
    
    def _process_execution_queue(self):
        """Process pending execution requests."""
        batch_requests = []
        
        # Collect batch of requests
        for _ in range(self.config.batch_execution_size):
            try:
                priority, request = self.execution_queue.get_nowait()
                batch_requests.append((priority, request))
            except queue.Empty:
                break
        
        if not batch_requests:
            return
        
        # Execute batch
        if self.is_arm64 and len(batch_requests) > 1:
            # Use parallel execution for ARM64
            self._execute_batch_parallel(batch_requests)
        else:
            # Sequential execution
            self._execute_batch_sequential(batch_requests)
    
    def _execute_batch_parallel(self, batch_requests: List[Tuple[float, ExecutionRequest]]):
        """Execute batch of requests in parallel (ARM64 optimized)."""
        futures = []
        
        for priority, request in batch_requests:
            future = self.executor.submit(self._execute_single_request, request)
            futures.append((future, request))
        
        # Collect results
        for future, request in futures:
            try:
                result = future.result(timeout=self.config.order_timeout_seconds)
                self.result_queue.put(result)
            except Exception as e:
                error_result = ExecutionResult(
                    request_id=getattr(request, 'client_order_id', 'unknown'),
                    symbol=request.symbol,
                    status=ExecutionStatus.FAILED,
                    submitted_quantity=float(request.quantity),
                    error_message=str(e)
                )
                self.result_queue.put(error_result)
    
    def _execute_batch_sequential(self, batch_requests: List[Tuple[float, ExecutionRequest]]):
        """Execute batch of requests sequentially."""
        for priority, request in batch_requests:
            try:
                result = self._execute_single_request(request)
                self.result_queue.put(result)
            except Exception as e:
                error_result = ExecutionResult(
                    request_id=getattr(request, 'client_order_id', 'unknown'),
                    symbol=request.symbol,
                    status=ExecutionStatus.FAILED,
                    submitted_quantity=float(request.quantity),
                    error_message=str(e)
                )
                self.result_queue.put(error_result)
    
    def _execute_single_request(self, request: ExecutionRequest) -> ExecutionResult:
        """Execute a single execution request."""
        start_time = time.time()
        request_id = request.client_order_id or f"exec_{int(time.time() * 1000000)}"
        
        try:
            # Validate request
            if self.config.enable_risk_checks:
                self._validate_execution_request(request)
            
            # Choose execution algorithm
            if request.algorithm == ExecutionAlgorithm.MARKET:
                result = self._execute_market_order(request, request_id)
            elif request.algorithm == ExecutionAlgorithm.TWAP:
                result = self._execute_twap_order(request, request_id)
            elif request.algorithm == ExecutionAlgorithm.VWAP:
                result = self._execute_vwap_order(request, request_id)
            elif request.algorithm == ExecutionAlgorithm.SMART_ROUTING:
                result = self._execute_smart_routing(request, request_id)
            else:
                # Default to market order
                result = self._execute_market_order(request, request_id)
            
            # Calculate execution metrics
            execution_time_ms = (time.time() - start_time) * 1000
            result.execution_time_ms = execution_time_ms
            
            # Update performance stats
            self._update_performance_stats(result, execution_time_ms)
            
            # Log execution
            trading_logger.log_trade({
                "symbol": request.symbol,
                "quantity": float(request.quantity),
                "side": request.side.value,
                "algorithm": request.algorithm.value,
                "execution_time_ms": execution_time_ms,
                "status": result.status.value
            })
            
            return result
            
        except Exception as e:
            execution_time_ms = (time.time() - start_time) * 1000
            logger.error(f"Execution failed for {request.symbol}: {e}")
            
            return ExecutionResult(
                request_id=request_id,
                symbol=request.symbol,
                status=ExecutionStatus.FAILED,
                submitted_quantity=float(request.quantity),
                execution_time_ms=execution_time_ms,
                error_message=str(e)
            )
    
    def _validate_execution_request(self, request: ExecutionRequest):
        """Validate execution request."""
        # Basic validation
        if request.quantity <= 0:
            raise ExecutionError("Quantity must be positive")
        
        if request.quantity > self.config.max_order_size:
            raise ExecutionError(f"Order size {request.quantity} exceeds maximum {self.config.max_order_size}")
        
        # Position validation
        if self.position_manager and self.config.enable_position_tracking:
            current_position = self.position_manager.get_position(request.symbol)
            # Add position-based validation logic here
    
    def _execute_market_order(self, request: ExecutionRequest, request_id: str) -> ExecutionResult:
        """Execute market order."""
        order_request = OrderRequest(
            symbol=request.symbol,
            quantity=request.quantity,
            side=request.side,
            order_type=OrderType.MARKET,
            time_in_force=request.time_in_force,
            client_order_id=request_id
        )
        
        order_response = self.broker_client.place_order(order_request)
        
        return ExecutionResult(
            request_id=request_id,
            symbol=request.symbol,
            status=ExecutionStatus.SUBMITTED,
            submitted_quantity=float(request.quantity),
            order_ids=[order_response["order_id"]]
        )
    
    def _execute_twap_order(self, request: ExecutionRequest, request_id: str) -> ExecutionResult:
        """Execute TWAP (Time-Weighted Average Price) order."""
        # Simplified TWAP implementation
        # In production, this would split the order over time
        return self._execute_market_order(request, request_id)
    
    def _execute_vwap_order(self, request: ExecutionRequest, request_id: str) -> ExecutionResult:
        """Execute VWAP (Volume-Weighted Average Price) order."""
        # Simplified VWAP implementation
        # In production, this would consider volume patterns
        return self._execute_market_order(request, request_id)
    
    def _execute_smart_routing(self, request: ExecutionRequest, request_id: str) -> ExecutionResult:
        """Execute with smart routing algorithm."""
        # Choose best execution algorithm based on market conditions
        if request.urgency > 0.8:
            # High urgency - use market order
            return self._execute_market_order(request, request_id)
        elif request.quantity > 1000:
            # Large order - use TWAP
            return self._execute_twap_order(request, request_id)
        else:
            # Default to market order
            return self._execute_market_order(request, request_id)
    
    def _process_execution_results(self):
        """Process execution results."""
        results_processed = 0
        
        while results_processed < self.config.batch_execution_size:
            try:
                result = self.result_queue.get_nowait()
                self._handle_execution_result(result)
                results_processed += 1
            except queue.Empty:
                break
    
    def _handle_execution_result(self, result: ExecutionResult):
        """Handle execution result."""
        # Update position manager
        if self.position_manager and result.status == ExecutionStatus.FILLED:
            side = "buy" if result.filled_quantity > 0 else "sell"
            self.position_manager.update_position(
                result.symbol, 
                abs(result.filled_quantity), 
                side
            )
        
        # Publish result
        self.zmq_publisher.send({
            "type": "execution_result",
            "data": {
                "request_id": result.request_id,
                "symbol": result.symbol,
                "status": result.status.value,
                "filled_quantity": result.filled_quantity,
                "avg_fill_price": result.avg_fill_price,
                "execution_time_ms": result.execution_time_ms,
                "timestamp": result.timestamp.isoformat()
            }
        })
        
        logger.info(f"Execution result: {result.symbol} {result.status.value}")
    
    def _update_performance_stats(self, result: ExecutionResult, execution_time_ms: float):
        """Update performance statistics."""
        if not self._performance_stats:
            return
        
        self._performance_stats["total_executions"] += 1
        
        if result.status in [ExecutionStatus.FILLED, ExecutionStatus.SUBMITTED]:
            self._performance_stats["successful_executions"] += 1
        else:
            self._performance_stats["failed_executions"] += 1
        
        self._performance_stats["total_volume"] += result.submitted_quantity
        self._performance_stats["total_execution_time_ms"] += execution_time_ms
        
        # Calculate averages
        total_execs = self._performance_stats["total_executions"]
        self._performance_stats["avg_execution_time_ms"] = (
            self._performance_stats["total_execution_time_ms"] / total_execs
        )
        
        # Calculate orders per second
        if execution_time_ms > 0:
            self._performance_stats["orders_per_second"] = 1000.0 / execution_time_ms
    
    @performance_monitor
    def execute_request(self, request: ExecutionRequest, priority: float = 0.5) -> str:
        """Submit execution request."""
        if not self._running:
            raise ExecutionError("ExecutionEngine is not running")
        
        request_id = request.client_order_id or f"exec_{int(time.time() * 1000000)}"
        
        try:
            # Add to execution queue (lower priority value = higher priority)
            self.execution_queue.put((1.0 - priority, request), timeout=1.0)
            
            # Store pending execution
            with self._execution_lock:
                self._pending_executions[request_id] = {
                    "request": request,
                    "timestamp": datetime.now(timezone.utc),
                    "status": "pending"
                }
            
            logger.info(f"Execution request queued: {request.symbol} {request.side.value} {request.quantity}")
            return request_id
            
        except queue.Full:
            raise ExecutionError("Execution queue is full")
    
    def _handle_risk_approved_predictions(self, message: Dict[str, Any]):
        """Handle risk-approved predictions."""
        try:
            predictions = message.get("data", {})
            current_positions = self.position_manager.get_current_positions() if self.position_manager else {}
            
            execution_requests = self.generate_execution_requests(predictions, current_positions)
            
            for request in execution_requests:
                self.execute_request(request, priority=0.7)  # High priority for predictions
            
            logger.info(f"Generated {len(execution_requests)} execution requests from predictions")
            
        except Exception as e:
            logger.error(f"Error handling risk-approved predictions: {e}")
    
    def _handle_execution_requests(self, message: Dict[str, Any]):
        """Handle direct execution requests."""
        try:
            request_data = message.get("data", {})
            
            request = ExecutionRequest(
                symbol=request_data["symbol"],
                quantity=request_data["quantity"],
                side=OrderSideEnum(request_data["side"]),
                algorithm=ExecutionAlgorithm(request_data.get("algorithm", "smart_routing")),
                urgency=request_data.get("urgency", 0.5)
            )
            
            priority = request_data.get("priority", 0.5)
            self.execute_request(request, priority)
            
        except Exception as e:
            logger.error(f"Error handling execution request: {e}")
    
    def generate_execution_requests(self, predictions: Dict[str, Any], 
                                  current_positions: Dict[str, float]) -> List[ExecutionRequest]:
        """Generate execution requests from model predictions."""
        requests = []
        
        for symbol, prediction in predictions.items():
            target_position = prediction.get("position", 0) * prediction.get("confidence", 1.0)
            current_position = current_positions.get(symbol, 0)
            
            position_diff = target_position - current_position
            
            if abs(position_diff) > self.config.min_trade_threshold:
                side = OrderSideEnum.BUY if position_diff > 0 else OrderSideEnum.SELL
                quantity = abs(position_diff)
                
                # Determine urgency based on prediction confidence
                confidence = prediction.get("confidence", 0.5)
                urgency = min(0.9, confidence * 1.5)  # Scale confidence to urgency
                
                request = ExecutionRequest(
                    symbol=symbol,
                    quantity=quantity,
                    side=side,
                    algorithm=self.config.default_algorithm,
                    urgency=urgency,
                    metadata={"prediction": prediction}
                )
                
                requests.append(request)
        
        return requests
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get execution engine performance statistics."""
        if not self._performance_stats:
            return {"performance_monitoring": "disabled"}
        
        stats = self._performance_stats.copy()
        stats.update({
            "pending_executions": len(self._pending_executions),
            "queue_size": self.execution_queue.qsize(),
            "arm64_optimized": self.is_arm64 and self.config.enable_arm64_optimizations,
            "shared_memory_enabled": self.config.enable_shared_memory
        })
        
        return stats
    
    def get_status(self) -> Dict[str, Any]:
        """Get execution engine status."""
        return {
            "running": self._running,
            "pending_executions": len(self._pending_executions),
            "queue_size": self.execution_queue.qsize(),
            "arm64_optimized": self.is_arm64 and self.config.enable_arm64_optimizations,
            "performance_monitoring": self.config.enable_performance_monitoring,
            "shared_memory_enabled": self.config.enable_shared_memory
        }
    
    def shutdown(self):
        """Shutdown execution engine and cleanup resources."""
        self.stop()
        
        # Cleanup shared memory
        if self.shared_positions:
            self.shared_positions.close()
        if self.shared_orders:
            self.shared_orders.close()
        
        logger.info("ExecutionEngine shutdown completed")

# Export all public components
__all__ = [
    "ExecutionEngine",
    "ExecutionConfig",
    "ExecutionRequest",
    "ExecutionResult",
    "ExecutionAlgorithm",
    "ExecutionStatus",
    "IS_ARM64"
]
