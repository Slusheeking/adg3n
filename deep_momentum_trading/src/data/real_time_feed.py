"""
Real-time data feed system with enhanced error handling, performance monitoring,
and ARM64 optimizations for the Deep Momentum Trading System.

This module provides a comprehensive real-time data pipeline that ingests market data
from Polygon.io, processes it through feature engineering, and distributes it via ZeroMQ
with advanced buffering, caching, and error recovery mechanisms.

Critical Improvements:
- Fixed race conditions in buffer management
- ARM64-optimized batch processing for GH200
- Memory leak prevention with task lifecycle management
- Circuit breaker pattern for error recovery
- Whole market subscription capabilities
- GH200 memory pool management
- Real-time performance monitoring
"""

import asyncio
import os
import time
import threading
import platform
import psutil
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass, field
from collections import deque, defaultdict
from concurrent.futures import ThreadPoolExecutor

from deep_momentum_trading.src.data.polygon_client import AdvancedPolygonClient, PolygonConfig
from deep_momentum_trading.src.data.data_preprocessing import DataPreprocessor
from deep_momentum_trading.src.data.feature_engineering import FeatureEngineeringProcess
from deep_momentum_trading.src.data.memory_cache import UnifiedMemoryManager
from deep_momentum_trading.src.communication.zmq_publisher import ZMQPublisher
from deep_momentum_trading.src.communication.arm64_data_publisher import ARM64DataPublisher
from deep_momentum_trading.src.communication.arm64_comm_monitor import ARM64CommunicationMonitor
from deep_momentum_trading.src.utils.logger import get_logger

logger = get_logger(__name__)


class CircuitBreaker:
    """Circuit breaker for automatic error recovery."""
    
    def __init__(self, failure_threshold: int = 5, recovery_timeout: float = 60.0):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = 0
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
    
    async def call(self, func: Callable, *args, **kwargs):
        """Execute function with circuit breaker protection."""
        if self.state == "OPEN":
            if time.time() - self.last_failure_time > self.recovery_timeout:
                self.state = "HALF_OPEN"
            else:
                raise Exception("Circuit breaker is OPEN")
        
        try:
            result = await func(*args, **kwargs)
            if self.state == "HALF_OPEN":
                self.state = "CLOSED"
                self.failure_count = 0
            return result
        except Exception as e:
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            if self.failure_count >= self.failure_threshold:
                self.state = "OPEN"
            raise e


class ARM64BatchProcessor:
    """ARM64-optimized batch processing for market data."""
    
    def __init__(self, batch_size: int = 1000):
        self.batch_size = batch_size
        self.numpy_buffer = np.zeros((batch_size, 6), dtype=np.float64)  # OHLCV + timestamp
        self.current_idx = 0
        
    def add_data_point(self, ohlcv_data: Dict[str, Any]) -> Optional[np.ndarray]:
        """Add data point to ARM64-optimized buffer."""
        try:
            # Direct numpy assignment (faster than DataFrame)
            self.numpy_buffer[self.current_idx] = [
                ohlcv_data['timestamp'],
                ohlcv_data['open'],
                ohlcv_data['high'], 
                ohlcv_data['low'],
                ohlcv_data['close'],
                ohlcv_data['volume']
            ]
            self.current_idx += 1
            
            if self.current_idx >= self.batch_size:
                # Return batch and reset
                batch = self.numpy_buffer[:self.current_idx].copy()
                self.current_idx = 0
                return batch
            
        except Exception as e:
            logger.error(f"ARM64 batch processing error: {e}")
            return None
        
        return None


class GH200MemoryPool:
    """Leverage GH200's 624GB unified memory architecture."""
    
    def __init__(self, total_memory_gb: int = 600):
        self.total_memory_bytes = total_memory_gb * 1024**3
        self.allocated_memory = 0
        self.memory_pools = {
            'market_data': {},  # Symbol -> data buffer
            'features': {},     # Symbol -> feature buffer  
            'models': {},       # Model parameters
            'working': {}       # Temporary computations
        }
    
    def allocate_symbol_buffer(self, symbol: str, size_mb: int = 10) -> Optional[np.ndarray]:
        """Allocate unified memory buffer for symbol data."""
        size_bytes = size_mb * 1024**2
        
        if self.allocated_memory + size_bytes > self.total_memory_bytes:
            self._garbage_collect()
        
        try:
            # Use memory-mapped allocation for ARM64 efficiency
            buffer = np.memmap(
                f'/tmp/dmn_{symbol}_buffer.dat',
                dtype=np.float64,
                mode='w+',
                shape=(size_bytes // 8,)  # 8 bytes per float64
            )
            
            self.memory_pools['market_data'][symbol] = buffer
            self.allocated_memory += size_bytes
            
            return buffer
        except Exception as e:
            logger.error(f"Failed to allocate buffer for {symbol}: {e}")
            return None
    
    def _garbage_collect(self):
        """Clean up unused memory pools."""
        try:
            # Remove empty or unused buffers
            for pool_name, pool in self.memory_pools.items():
                symbols_to_remove = []
                for symbol, buffer in pool.items():
                    if hasattr(buffer, 'size') and buffer.size == 0:
                        symbols_to_remove.append(symbol)
                
                for symbol in symbols_to_remove:
                    del pool[symbol]
                    logger.debug(f"Garbage collected buffer for {symbol} in {pool_name}")
        except Exception as e:
            logger.error(f"Error during garbage collection: {e}")


@dataclass
class RealTimeFeedConfig:
    """Configuration for RealTimeFeed with ARM64 optimizations and enhanced performance monitoring."""
    polygon_api_key: Optional[str] = None
    zmq_market_data_port: int = 5555
    zmq_features_port: int = 5556
    memory_cache_max_gb: float = 200.0
    buffer_flush_interval: float = 1.0  # seconds
    buffer_max_size: int = 100
    enable_compression: bool = True
    enable_batching: bool = True
    batch_size: int = 1000
    max_reconnect_attempts: int = 10
    reconnect_delay: float = 1.0
    enable_performance_tracking: bool = True
    enable_error_recovery: bool = True
    enable_arm64_optimizations: bool = True
    use_arm64_publisher: bool = True
    enable_communication_monitoring: bool = True
    enable_whole_market_feed: bool = False
    gh200_memory_pool_gb: int = 600
    circuit_breaker_threshold: int = 5
    circuit_breaker_timeout: float = 60.0
    data_preprocessor_config: Dict[str, Any] = field(default_factory=dict)
    feature_engineering_config: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FeedStatistics:
    """Statistics for real-time feed performance monitoring."""
    messages_processed: int = 0
    messages_failed: int = 0
    bytes_processed: int = 0
    symbols_active: int = 0
    avg_processing_time_ms: float = 0.0
    buffer_overflow_count: int = 0
    reconnection_count: int = 0
    last_message_time: float = 0.0
    circuit_breaker_trips: int = 0
    memory_usage_mb: float = 0.0
    start_time: float = field(default_factory=time.time)
    
    def get_uptime_seconds(self) -> float:
        """Get feed uptime in seconds."""
        return time.time() - self.start_time
    
    def get_messages_per_second(self) -> float:
        """Get average messages per second."""
        uptime = self.get_uptime_seconds()
        return self.messages_processed / uptime if uptime > 0 else 0.0


class RealTimeFeed:
    """
    Enhanced real-time data feed system with comprehensive error handling,
    performance monitoring, and ARM64 optimizations.
    
    Manages the end-to-end real-time data pipeline, from ingestion
    (via Polygon.io WebSocket) through preprocessing, feature engineering,
    caching, and distribution to other system components via ZeroMQ.
    """

    def __init__(self, 
                 config: Optional[RealTimeFeedConfig] = None,
                 polygon_api_key: Optional[str] = None,
                 zmq_market_data_port: int = 5555,
                 zmq_features_port: int = 5556,
                 memory_cache_max_gb: float = 200.0,
                 data_preprocessor_config: Optional[Dict[str, Any]] = None,
                 feature_engineering_config: Optional[Dict[str, Any]] = None):
        """
        Initialize the RealTimeFeed with enhanced configuration and error handling.

        Args:
            config: RealTimeFeedConfig object (preferred)
            polygon_api_key: API key for Polygon.io (fallback)
            zmq_market_data_port: ZMQ port for publishing raw market data
            zmq_features_port: ZMQ port for publishing engineered features
            memory_cache_max_gb: Max memory for unified cache in GB
            data_preprocessor_config: Configuration for DataPreprocessor
            feature_engineering_config: Configuration for FeatureEngineeringProcess
            
        Raises:
            ValueError: If configuration is invalid
            ConnectionError: If unable to initialize components
        """
        # Configuration handling with fallback support
        if config is not None:
            self.config = config
        else:
            self.config = RealTimeFeedConfig(
                polygon_api_key=polygon_api_key,
                zmq_market_data_port=zmq_market_data_port,
                zmq_features_port=zmq_features_port,
                memory_cache_max_gb=memory_cache_max_gb,
                data_preprocessor_config=data_preprocessor_config or {},
                feature_engineering_config=feature_engineering_config or {}
            )
        
        # Initialize circuit breaker for error recovery
        self.circuit_breaker = CircuitBreaker(
            failure_threshold=self.config.circuit_breaker_threshold,
            recovery_timeout=self.config.circuit_breaker_timeout
        )
        
        # Initialize ARM64 batch processor
        self.arm64_processor = ARM64BatchProcessor(batch_size=self.config.batch_size)
        
        # Initialize GH200 memory pool
        self.gh200_memory_pool = GH200MemoryPool(
            total_memory_gb=self.config.gh200_memory_pool_gb
        )
        
        # Initialize components with error handling
        try:
            polygon_config = PolygonConfig(api_key=self.config.polygon_api_key)
            self.polygon_client = AdvancedPolygonClient(config=polygon_config)
            logger.info("Advanced Polygon client initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Polygon client: {e}")
            raise ConnectionError(f"Polygon client initialization failed: {e}")
        
        try:
            self.memory_cache = UnifiedMemoryManager(max_memory_gb=self.config.memory_cache_max_gb)
            logger.info(f"Memory cache initialized with {self.config.memory_cache_max_gb}GB capacity")
        except Exception as e:
            logger.error(f"Failed to initialize memory cache: {e}")
            raise
        
        try:
            self.data_preprocessor = DataPreprocessor(**(self.config.data_preprocessor_config))
            logger.info("Data preprocessor initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize data preprocessor: {e}")
            raise
        
        try:
            self.feature_engineer = FeatureEngineeringProcess(
                zmq_subscriber_port=self.config.zmq_market_data_port,
                zmq_publisher_port=self.config.zmq_features_port,
                memory_cache_max_gb=self.config.memory_cache_max_gb
            )
            logger.info("Feature engineering process initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize feature engineering: {e}")
            raise
        
        # Initialize publisher with ARM64 optimizations
        try:
            if self.config.use_arm64_publisher:
                self.raw_data_publisher = ARM64DataPublisher(
                    port=self.config.zmq_market_data_port,
                    enable_compression=self.config.enable_compression
                )
                logger.info(f"ARM64 data publisher initialized on port {self.config.zmq_market_data_port}")
            else:
                self.raw_data_publisher = ZMQPublisher(port=self.config.zmq_market_data_port)
                logger.info(f"Standard ZMQ publisher initialized on port {self.config.zmq_market_data_port}")
        except Exception as e:
            logger.error(f"Failed to initialize data publisher: {e}")
            raise
        
        # State management
        self.is_running = False
        self.subscribed_symbols: List[str] = []
        self.ohlcv_buffers: Dict[str, deque] = defaultdict(lambda: deque(maxlen=self.config.buffer_max_size))
        self.buffer_lock = asyncio.Lock()
        self.processing_lock = threading.RLock()
        
        # Performance tracking with ARM64 monitoring
        self.statistics = FeedStatistics()
        self.error_counts: Dict[str, int] = defaultdict(int)
        self.processing_times: deque = deque(maxlen=1000)  # Last 1000 processing times
        
        # ARM64 communication monitoring
        self.comm_monitor = None
        if self.config.enable_communication_monitoring:
            try:
                self.comm_monitor = ARM64CommunicationMonitor(
                    monitoring_interval=1.0,
                    enable_system_monitoring=True
                )
                # Register the publisher for monitoring
                if hasattr(self.raw_data_publisher, 'get_statistics'):
                    self.comm_monitor.register_publisher(
                        "real_time_feed_publisher",
                        self.raw_data_publisher.get_statistics
                    )
                logger.info("ARM64 communication monitoring initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize communication monitoring: {e}")
                self.comm_monitor = None
        
        # Threading and async management with task lifecycle
        self.executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="RealTimeFeed")
        self.background_tasks: List[asyncio.Task] = []
        
        logger.info(f"RealTimeFeed initialized with {len(self.subscribed_symbols)} symbols")

    async def _create_managed_task(self, coro, name: str = None) -> asyncio.Task:
        """Create task with automatic cleanup."""
        task = asyncio.create_task(coro, name=name)
        self.background_tasks.append(task)
        
        # Add cleanup callback
        task.add_done_callback(lambda t: self._cleanup_task(t))
        return task

    def _cleanup_task(self, task: asyncio.Task):
        """Remove completed tasks from background_tasks."""
        try:
            if task in self.background_tasks:
                self.background_tasks.remove(task)
            if task.exception():
                logger.error(f"Background task failed: {task.exception()}")
        except Exception as e:
            logger.error(f"Error cleaning up task: {e}")

    async def start_whole_market_feed(self):
        """Subscribe to entire US equity market with single WebSocket."""
        try:
            await self.polygon_client._connect_websocket()
            
            # Single subscription for everything
            subscribe_message = {
                "action": "subscribe",
                "params": "T.*,Q.*,AM.*,A.*"  # All trades, quotes, minute & second bars
            }
            
            await self.polygon_client.websocket.send(json.dumps(subscribe_message))
            logger.info("Subscribed to entire US equity market")
            
            # Start streaming
            stream_task = await self._create_managed_task(
                self.polygon_client.stream_data(),
                name="whole_market_stream"
            )
            
        except Exception as e:
            logger.error(f"Failed to start whole market feed: {e}")
            raise

    async def _performance_monitor(self):
        """Monitor system performance for GH200 optimization."""
        while self.is_running:
            try:
                # Monitor memory usage
                memory_info = psutil.virtual_memory()
                self.statistics.memory_usage_mb = memory_info.used / (1024 * 1024)
                
                if memory_info.percent > 90:
                    logger.warning(f"High memory usage: {memory_info.percent}%")
                    await self._emergency_buffer_flush()
                
                # Monitor processing latency
                if self.statistics.avg_processing_time_ms > 10:  # 10ms threshold
                    logger.warning(f"High processing latency: {self.statistics.avg_processing_time_ms}ms")
                    await self._optimize_buffers()
                
                # Monitor ARM64 specific metrics
                if self.comm_monitor:
                    comm_stats = self.comm_monitor.get_communication_stats()
                    if comm_stats.get('message_loss_rate', 0) > 0.01:  # 1% loss
                        logger.warning("High message loss detected")
                        await self._adjust_buffer_sizes()
                
                await asyncio.sleep(1.0)
                
            except Exception as e:
                logger.error(f"Performance monitoring error: {e}")

    async def _emergency_buffer_flush(self):
        """Emergency flush of all buffers."""
        try:
            async with self.buffer_lock:
                symbols_to_flush = list(self.ohlcv_buffers.keys())
            
            flush_tasks = []
            for symbol in symbols_to_flush:
                if self.ohlcv_buffers[symbol]:
                    task = await self._create_managed_task(
                        self._flush_buffer(symbol),
                        name=f"emergency_flush_{symbol}"
                    )
                    flush_tasks.append(task)
            
            # Wait for all flushes to complete
            if flush_tasks:
                await asyncio.gather(*flush_tasks, return_exceptions=True)
                
        except Exception as e:
            logger.error(f"Emergency buffer flush error: {e}")

    async def _optimize_buffers(self):
        """Optimize buffer sizes based on current performance."""
        try:
            # Reduce buffer sizes if processing is slow
            if self.statistics.avg_processing_time_ms > 10:
                new_size = max(50, self.config.buffer_max_size // 2)
                for symbol in self.ohlcv_buffers:
                    if len(self.ohlcv_buffers[symbol]) > new_size:
                        # Flush excess data
                        await self._flush_buffer(symbol)
                logger.info(f"Optimized buffer sizes to {new_size}")
        except Exception as e:
            logger.error(f"Buffer optimization error: {e}")

    async def _adjust_buffer_sizes(self):
        """Adjust buffer sizes based on message loss rate."""
        try:
            # Increase buffer sizes if message loss is detected
            new_size = min(1000, self.config.buffer_max_size * 2)
            self.config.buffer_max_size = new_size
            logger.info(f"Adjusted buffer sizes to {new_size} due to message loss")
        except Exception as e:
            logger.error(f"Buffer size adjustment error: {e}")

    async def _handle_trade_message(self, data_point) -> None:
        """Handle trade messages from Polygon WebSocket."""
        try:
            message = {
                'ev': 'T',
                'sym': data_point.symbol,
                't': data_point.timestamp,
                'p': data_point.price,
                's': data_point.size,
                'x': data_point.exchange,
                'c': data_point.conditions or []
            }
            await self.circuit_breaker.call(self._process_incoming_message, message)
        except Exception as e:
            logger.error(f"Error handling trade message: {e}")
            self.error_counts['trade_handler_error'] += 1

    async def _handle_quote_message(self, data_point) -> None:
        """Handle quote messages from Polygon WebSocket."""
        try:
            message = {
                'ev': 'Q',
                'sym': data_point.symbol,
                't': data_point.timestamp,
                'bp': data_point.bid_price,
                'ap': data_point.ask_price,
                'bs': data_point.bid_size,
                'as': data_point.ask_size,
                'x': data_point.exchange
            }
            await self.circuit_breaker.call(self._process_incoming_message, message)
        except Exception as e:
            logger.error(f"Error handling quote message: {e}")
            self.error_counts['quote_handler_error'] += 1

    async def _handle_aggregate_message(self, data_point) -> None:
        """Handle aggregate (minute bar) messages from Polygon WebSocket."""
        try:
            message = {
                'ev': 'AM',
                'sym': data_point.symbol,
                't': data_point.timestamp,
                'o': data_point.open_price,
                'h': data_point.high_price,
                'l': data_point.low_price,
                'c': data_point.close_price,
                'v': data_point.volume,
                'vw': data_point.vwap
            }
            await self.circuit_breaker.call(self._process_incoming_message, message)
        except Exception as e:
            logger.error(f"Error handling aggregate message: {e}")
            self.error_counts['aggregate_handler_error'] += 1

    async def _process_incoming_message(self, message: Dict[str, Any]) -> None:
        """
        Enhanced callback function to process incoming messages from Polygon.io WebSocket.
        Buffers messages and triggers processing periodically with comprehensive error handling.
        Fixed race conditions with atomic buffer operations.

        Args:
            message (Dict[str, Any]): The incoming data message from Polygon.io.
        """
        if not self.is_running:
            return

        start_time = time.time()
        
        try:
            event_type = message.get('ev')
            symbol = message.get('sym')

            if not symbol:
                logger.warning(f"Received message without symbol: {message}")
                self.statistics.messages_failed += 1
                self.error_counts['missing_symbol'] += 1
                return

            # Update statistics
            self.statistics.messages_processed += 1
            self.statistics.symbols_active = len(set(self.subscribed_symbols))
            self.statistics.last_message_time = time.time()

            # Cache raw message in unified memory and publish with ARM64 optimizations
            try:
                self.memory_cache.store_market_data(symbol, message)
                
                # Use ARM64-optimized publishing if available
                if hasattr(self.raw_data_publisher, 'publish_market_data'):
                    if isinstance(self.raw_data_publisher, ARM64DataPublisher):
                        # ARM64 publisher expects symbol and data separately
                        success = self.raw_data_publisher.publish_market_data(symbol, message)
                        if not success:
                            self.error_counts['arm64_publish_error'] += 1
                    else:
                        # Standard ZMQ publisher
                        self.raw_data_publisher.publish_market_data(message)
                else:
                    logger.warning("Publisher does not support publish_market_data method")
                    self.error_counts['publisher_method_error'] += 1
                    
                # Track communication performance
                if self.comm_monitor:
                    self.comm_monitor.track_message_latency(start_time, time.time())
                    
            except Exception as e:
                logger.error(f"Failed to cache/publish message for {symbol}: {e}")
                self.error_counts['cache_publish_error'] += 1

            # Buffer OHLCV-relevant messages for batch processing with atomic operations
            if event_type == 'AM':  # Aggregate Minute Bar
                try:
                    # Convert Polygon.io 'AM' message to a format suitable for DataPreprocessor
                    ohlcv_data = {
                        'timestamp': message.get('t'),
                        'open': message.get('o'),
                        'high': message.get('h'),
                        'low': message.get('l'),
                        'close': message.get('c'),
                        'volume': message.get('v')
                    }
                    
                    # FIXED: Atomic buffer operations to prevent race conditions
                    async with self.buffer_lock:
                        if symbol not in self.ohlcv_buffers:
                            self.ohlcv_buffers[symbol] = deque(maxlen=self.config.buffer_max_size)
                        
                        buffer = self.ohlcv_buffers[symbol]
                        buffer.append(ohlcv_data)
                        
                        # Atomic check and clear
                        if len(buffer) >= self.config.buffer_max_size:
                            # Move data atomically
                            buffered_data = list(buffer)
                            buffer.clear()
                            # Process without holding lock
                            await self._create_managed_task(
                                self._process_buffered_data(symbol, buffered_data),
                                name=f"process_buffer_{symbol}"
                            )
                            
                except Exception as e:
                    logger.error(f"Failed to process OHLCV data for {symbol}: {e}")
                    self.error_counts['ohlcv_processing_error'] += 1
                    
            elif event_type in ['T', 'Q']:  # Trades and Quotes
                # For now, we focus on 'AM' for OHLCV, but tick data could be processed here too
                logger.debug(f"Received {event_type} event for {symbol}")
            else:
                logger.debug(f"Unhandled Polygon.io event type: {event_type} for {symbol}")

            # Track processing time
            processing_time = (time.time() - start_time) * 1000  # Convert to milliseconds
            self.processing_times.append(processing_time)
            
            # Update average processing time
            if self.processing_times:
                self.statistics.avg_processing_time_ms = sum(self.processing_times) / len(self.processing_times)

        except Exception as e:
            logger.error(f"Critical error processing message: {e}")
            self.statistics.messages_failed += 1
            self.error_counts['critical_processing_error'] += 1

    async def _process_buffered_data(self, symbol: str, buffered_data: List[Dict[str, Any]]) -> None:
        """
        Process buffered data using ARM64 optimizations.
        
        Args:
            symbol: Symbol to process
            buffered_data: List of buffered OHLCV data points
        """
        try:
            if not buffered_data:
                return

            logger.info(f"Processing {len(buffered_data)} buffered messages for {symbol}")
            
            # Use ARM64 batch processor for performance
            if self.config.enable_arm64_optimizations:
                batch_data = []
                for data_point in buffered_data:
                    batch = self.arm64_processor.add_data_point(data_point)
                    if batch is not None:
                        batch_data.append(batch)
                
                # Process any remaining data
                if self.arm64_processor.current_idx > 0:
                    remaining_batch = self.arm64_processor.numpy_buffer[:self.arm64_processor.current_idx].copy()
                    batch_data.append(remaining_batch)
                    self.arm64_processor.current_idx = 0
                
                # Convert numpy batches to DataFrame for preprocessing
                if batch_data:
                    combined_data = np.vstack(batch_data)
                    raw_df = pd.DataFrame(combined_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                else:
                    raw_df = pd.DataFrame(buffered_data)
            else:
                # Standard DataFrame conversion
                raw_df = pd.DataFrame(buffered_data)
            
            # Preprocess data
            preprocessed_df = self.data_preprocessor.preprocess(raw_df.copy())

            if not preprocessed_df.empty:
                # Engineer features and publish
                self.feature_engineer.process_and_publish_features(symbol, preprocessed_df)
            else:
                logger.warning(f"Preprocessing resulted in empty DataFrame for {symbol}. Skipping feature engineering.")
                self.error_counts['empty_preprocessing_result'] += 1
                
        except Exception as e:
            logger.error(f"Error processing buffered data for {symbol}: {e}")
            self.error_counts['buffer_processing_error'] += 1

    async def _flush_buffer(self, symbol: str) -> None:
        """
        Flushes the buffered OHLCV data for a given symbol, preprocesses it,
        and then engineers features with comprehensive error handling.
        
        Args:
            symbol: Symbol to flush buffer for
        """
        try:
            async with self.buffer_lock:
                if symbol not in self.ohlcv_buffers or not self.ohlcv_buffers[symbol]:
                    return
                
                buffered_data = list(self.ohlcv_buffers[symbol])
                self.ohlcv_buffers[symbol].clear()  # Clear buffer after copying

            if buffered_data:
                await self._process_buffered_data(symbol, buffered_data)
                    
        except Exception as e:
            logger.error(f"Critical error in buffer flush for {symbol}: {e}")
            self.error_counts['buffer_flush_error'] += 1

    async def _periodic_buffer_flush(self):
        """Periodically flushes all buffers."""
        while self.is_running:
            await asyncio.sleep(self.config.buffer_flush_interval)
            async with self.buffer_lock:
                symbols_to_flush = list(self.ohlcv_buffers.keys())
            for symbol in symbols_to_flush:
                if self.ohlcv_buffers[symbol]: # Only flush if there's data
                    await self._create_managed_task(
                        self._flush_buffer(symbol),
                        name=f"periodic_flush_{symbol}"
                    )

    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive feed statistics including ARM64 metrics."""
        basic_stats = {
            'feed_stats': {
                'messages_processed': self.statistics.messages_processed,
                'messages_failed': self.statistics.messages_failed,
                'bytes_processed': self.statistics.bytes_processed,
                'symbols_active': self.statistics.symbols_active,
                'avg_processing_time_ms': self.statistics.avg_processing_time_ms,
                'buffer_overflow_count': self.statistics.buffer_overflow_count,
                'reconnection_count': self.statistics.reconnection_count,
                'circuit_breaker_trips': self.statistics.circuit_breaker_trips,
                'memory_usage_mb': self.statistics.memory_usage_mb,
                'uptime_seconds': self.statistics.get_uptime_seconds(),
                'messages_per_second': self.statistics.get_messages_per_second()
            },
            'error_counts': dict(self.error_counts),
            'circuit_breaker_state': self.circuit_breaker.state
        }
        
        # Add ARM64 communication stats if available
        if self.comm_monitor:
            try:
                comm_stats = self.comm_monitor.get_communication_stats()
                basic_stats['arm64_communication'] = comm_stats
            except Exception as e:
                logger.error(f"Error getting communication stats: {e}")
        
        # Add publisher stats if available
        if hasattr(self.raw_data_publisher, 'get_statistics'):
            try:
                publisher_stats = self.raw_data_publisher.get_statistics()
                basic_stats['publisher_stats'] = publisher_stats
            except Exception as e:
                logger.error(f"Error getting publisher stats: {e}")
        
        return basic_stats
    
    def get_error_counts(self) -> Dict[str, int]:
        """Get current error counts."""
        return dict(self.error_counts)
    
    def reset_statistics(self) -> None:
        """Reset statistics counters."""
        self.statistics = FeedStatistics()
        self.error_counts.clear()
        self.processing_times.clear()
        logger.info("Feed statistics reset")

    async def start(self, symbols: List[str]):
        """
        Starts the real-time data feed with enhanced error handling.

        Args:
            symbols (List[str]): List of ticker symbols to subscribe to.
            
        Raises:
            RuntimeError: If feed is already running or fails to start
        """
        if self.is_running:
            logger.warning("RealTimeFeed is already running.")
            return

        try:
            self.is_running = True
            self.subscribed_symbols = symbols
            self.statistics = FeedStatistics()  # Reset statistics on start
            
            logger.info(f"Starting RealTimeFeed for symbols: {symbols}")

            # Start ARM64 communication monitoring
            if self.comm_monitor:
                self.comm_monitor.start_monitoring()
                logger.info("ARM64 communication monitoring started")
            
            # Start performance monitoring
            await self._create_managed_task(
                self._performance_monitor(),
                name="performance_monitor"
            )
            
            # Start periodic buffer flushing
            await self._create_managed_task(
                self._periodic_buffer_flush(),
                name="periodic_buffer_flush"
            )

            # Start Polygon.io WebSocket streaming
            try:
                if self.config.enable_whole_market_feed:
                    await self.start_whole_market_feed()
                else:
                    # Subscribe to specific symbols
                    await self.polygon_client._connect_websocket()
                    await self.polygon_client.subscribe_to_trades(symbols)
                    await self.polygon_client.subscribe_to_quotes(symbols)
                    await self.polygon_client.subscribe_to_minute_aggregates(symbols)
                
                # Add message handlers
                self.polygon_client.add_handler("T", self._handle_trade_message)
                self.polygon_client.add_handler("Q", self._handle_quote_message)
                self.polygon_client.add_handler("AM", self._handle_aggregate_message)
                
                # Start streaming
                await self._create_managed_task(
                    self.polygon_client.stream_data(),
                    name="polygon_stream"
                )
                
                logger.info(f"Successfully started Polygon WebSocket streaming for {len(symbols)} symbols")
                
            except Exception as e:
                logger.error(f"Failed to start Polygon WebSocket stream: {e}")
                await self.stop()
                raise RuntimeError(f"Failed to start data feed: {e}")
                
        except Exception as e:
            logger.error(f"Failed to start RealTimeFeed: {e}")
            self.is_running = False
            raise

    async def stop(self):
        """Stops the real-time data feed with proper cleanup."""
        if not self.is_running:
            logger.warning("RealTimeFeed is not running.")
            return

        try:
            self.is_running = False
            logger.info("Stopping RealTimeFeed...")
            
            # Cancel background tasks with proper cleanup
            for task in self.background_tasks:
                if not task.done():
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass
            self.background_tasks.clear()
            
            # Flush remaining buffers
            try:
                async with self.buffer_lock:
                    symbols_to_flush = list(self.ohlcv_buffers.keys())
                for symbol in symbols_to_flush:
                    if self.ohlcv_buffers[symbol]:
                        await self._flush_buffer(symbol)
            except Exception as e:
                logger.error(f"Error flushing buffers during shutdown: {e}")
            
            # Stop ARM64 communication monitoring
            if self.comm_monitor:
                try:
                    self.comm_monitor.stop_monitoring()
                    logger.info("ARM64 communication monitoring stopped")
                except Exception as e:
                    logger.error(f"Error stopping communication monitoring: {e}")
            
            # Close connections
            try:
                await self.polygon_client.close()
            except Exception as e:
                logger.error(f"Error closing Polygon client: {e}")
                
            try:
                if hasattr(self.raw_data_publisher, 'close'):
                    self.raw_data_publisher.close()
                logger.info("Data publisher closed successfully")
            except Exception as e:
                logger.error(f"Error closing data publisher: {e}")
                
            try:
                self.feature_engineer.stop()
            except Exception as e:
                logger.error(f"Error stopping feature engineer: {e}")
                
            # Shutdown executor
            try:
                self.executor.shutdown(wait=True, timeout=5.0)
            except Exception as e:
                logger.error(f"Error shutting down executor: {e}")
                
            logger.info("RealTimeFeed stopped successfully")
            
        except Exception as e:
            logger.error(f"Error during RealTimeFeed shutdown: {e}")
            raise


if __name__ == "__main__":
    # Example Usage
    async def main():
        # Ensure POLYGON_API_KEY is set in your environment variables
        # export POLYGON_API_KEY="YOUR_POLYGON_API_KEY"
        api_key = os.getenv('POLYGON_API_KEY')
        if not api_key:
            logger.error("POLYGON_API_KEY environment variable not set. Cannot run example.")
            return

        config = RealTimeFeedConfig(
            polygon_api_key=api_key,
            zmq_market_data_port=5555,
            zmq_features_port=5556,
            memory_cache_max_gb=0.1,  # Small cache for testing
            enable_arm64_optimizations=True,
            use_arm64_publisher=True,
            enable_communication_monitoring=True,
            enable_whole_market_feed=False,  # Set to True for whole market
            gh200_memory_pool_gb=600,
            circuit_breaker_threshold=5,
            circuit_breaker_timeout=60.0,
            data_preprocessor_config={
                'resample_interval': '1min',
                'fill_method': 'ffill'
            },
            feature_engineering_config={}  # Default config
        )
        
        feed = RealTimeFeed(config=config)

        symbols_to_stream = ["AAPL", "MSFT"]

        try:
            logger.info("ARM64-optimized RealTimeFeed running. Press Ctrl+C to stop.")
            await feed.start(symbols_to_stream)
            
            # Periodically log ARM64 performance stats
            import signal
            import sys
            
            def signal_handler(signum, frame):
                logger.info("Received interrupt signal")
                sys.exit(0)
            
            signal.signal(signal.SIGINT, signal_handler)
            
            # Keep running and log stats every 30 seconds
            while True:
                await asyncio.sleep(30)
                stats = feed.get_statistics()
                logger.info(f"ARM64 Feed Performance: {stats.get('feed_stats', {})}")
                
        except KeyboardInterrupt:
            logger.info("Stopping ARM64-optimized RealTimeFeed...")
        except SystemExit:
            logger.info("System exit requested...")
        finally:
            # Log final statistics
            try:
                final_stats = feed.get_statistics()
                logger.info(f"Final ARM64 Feed Statistics: {final_stats}")
            except Exception as e:
                logger.error(f"Error getting final stats: {e}")
            await feed.stop()

    asyncio.run(main())
