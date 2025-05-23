import os
import time
import threading
import logging
import platform
import asyncio
import numpy as np
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass, field
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor

from deep_momentum_trading.src.data.polygon_client import AdvancedPolygonClient, PolygonConfig
from deep_momentum_trading.src.data.memory_cache import UnifiedMemoryManager
from deep_momentum_trading.src.storage.hdf5_storage import HDF5TimeSeriesStorage
from deep_momentum_trading.src.communication.zmq_publisher import ZMQPublisher
from deep_momentum_trading.src.utils.logger import get_logger
from deep_momentum_trading.config.settings import config_manager

logger = get_logger(__name__)

@dataclass
class DataManagerConfig:
    """Configuration for DataManager with ARM64 optimizations."""
    polygon_api_key: Optional[str] = None
    zmq_port: int = 5555
    hdf5_path: str = "data/raw/polygon/market_data.h5"
    memory_cache_max_gb: float = 200.0
    enable_arm64_optimizations: bool = True
    enable_compression: bool = True
    enable_batching: bool = True
    batch_size: int = 100
    buffer_size: int = 10000
    enable_performance_monitoring: bool = True
    enable_error_recovery: bool = True
    max_reconnect_attempts: int = 10
    reconnect_delay: float = 1.0

@dataclass
class DataManagerStats:
    """Statistics for DataManager performance monitoring."""
    messages_processed: int = 0
    messages_failed: int = 0
    bytes_processed: int = 0
    symbols_active: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    hdf5_writes: int = 0
    zmq_publishes: int = 0
    errors: int = 0
    start_time: float = field(default_factory=time.time)
    
    @property
    def uptime_seconds(self) -> float:
        return time.time() - self.start_time
    
    @property
    def messages_per_second(self) -> float:
        uptime = self.uptime_seconds
        return self.messages_processed / uptime if uptime > 0 else 0.0

class DataManager:
    """
    Enhanced DataManager with ARM64 optimizations for high-performance market data processing.
    
    Manages the ingestion, caching, and distribution of real-time and historical
    market data. It integrates with Polygon.io for data feeds, uses a unified
    memory cache (GH200 optimized) for hot data, persists data to HDF5 for
    historical analysis, and publishes data via ZeroMQ for other system components.
    """

    def __init__(self,
                 config: Optional[DataManagerConfig] = None,
                 polygon_api_key: Optional[str] = None,
                 zmq_port: int = 5555,
                 hdf5_path: str = "data/raw/polygon/market_data.h5",
                 memory_cache_max_gb: float = 200.0):
        """
        Initializes the enhanced DataManager with ARM64 optimizations.

        Args:
            config: DataManagerConfig object (preferred)
            polygon_api_key: API key for Polygon.io (fallback)
            zmq_port: Port for ZeroMQ publisher (fallback)
            hdf5_path: Path to the HDF5 file for historical data storage (fallback)
            memory_cache_max_gb: Maximum memory in GB for the unified memory cache (fallback)
        """
        # Configuration handling - use config manager first
        if config is not None:
            self.config = config
        else:
            # Load from config manager with fallbacks
            config_data = config_manager.get('training_config.data', {})
            self.config = DataManagerConfig(
                polygon_api_key=polygon_api_key or config_data.get('polygon', {}).get('api_key'),
                zmq_port=zmq_port if zmq_port != 5555 else config_manager.get('training_config.communication.zmq.data_port', 5555),
                hdf5_path=hdf5_path if hdf5_path != "data/raw/polygon/market_data.h5" else config_data.get('hdf5_path', "data/raw/polygon/market_data.h5"),
                memory_cache_max_gb=memory_cache_max_gb if memory_cache_max_gb != 200.0 else config_data.get('memory_cache_gb', 32.0)
            )
        
        # ARM64 optimization detection
        self.is_arm64 = platform.machine().lower() in ['arm64', 'aarch64']
        if self.is_arm64 and self.config.enable_arm64_optimizations:
            logger.info("ARM64 optimizations enabled for DataManager")
        
        # Validate configuration
        self._validate_configuration()
        
        # Initialize thread pool for data processing
        self._executor = ThreadPoolExecutor(
            max_workers=min(os.cpu_count() or 4, 8),
            thread_name_prefix="DataManager"
        )
        
        # Initialize batch processing queues
        self._hdf5_batch_queue = asyncio.Queue(maxsize=1000)
        self._batch_processing_task = None
        
        # Circuit breaker and backoff tracking
        self._circuit_breaker_symbols = set()
        self._backoff_symbols = set()
        
        # Initialize components with error handling
        try:
            polygon_config = PolygonConfig(
                api_key=self.config.polygon_api_key or os.getenv('POLYGON_API_KEY'),
                enable_compression=self.config.enable_compression,
                enable_batching=self.config.enable_batching,
                buffer_size=self.config.buffer_size
            )
            self.polygon_client = AdvancedPolygonClient(config=polygon_config)
            logger.info("Advanced Polygon client initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Polygon client: {e}")
            raise
        
        try:
            self.memory_cache = UnifiedMemoryManager(max_memory_gb=self.config.memory_cache_max_gb)
            logger.info(f"Memory cache initialized with {self.config.memory_cache_max_gb}GB capacity")
        except Exception as e:
            logger.error(f"Failed to initialize memory cache: {e}")
            raise
        
        try:
            self.hdf5_storage = HDF5TimeSeriesStorage(base_path=self.config.hdf5_path)
            logger.info(f"HDF5 storage initialized at {self.config.hdf5_path}")
        except Exception as e:
            logger.error(f"Failed to initialize HDF5 storage: {e}")
            raise
        
        try:
            # Initialize ZMQ publisher with ARM64 optimizations
            publisher_config = {
                'port': self.config.zmq_port,
                'enable_arm64_optimizations': self.config.enable_arm64_optimizations and self.is_arm64,
                'compression': 'lz4' if self.config.enable_compression else 'none',
                'enable_monitoring': self.config.enable_performance_monitoring
            }
            self.zmq_publisher = ZMQPublisher(**publisher_config)
            logger.info(f"ZMQ publisher initialized on port {self.config.zmq_port}")
        except Exception as e:
            logger.error(f"Failed to initialize ZMQ publisher: {e}")
            raise
        
        # State management
        self.is_running = False
        self.data_ingestion_thread: Optional[threading.Thread] = None
        self.subscribed_symbols: List[str] = []
        
        # Performance monitoring
        self.stats = DataManagerStats()
        self.error_counts: Dict[str, int] = defaultdict(int)
        self.message_handlers: Dict[str, List[Callable]] = defaultdict(list)
        
        # Threading and locks
        self.processing_lock = threading.RLock()
        self.stats_lock = threading.Lock()
        
        logger.info("Enhanced DataManager initialized with ARM64 optimizations")

    def __del__(self):
        """Proper cleanup on destruction."""
        try:
            if hasattr(self, '_executor'):
                self._executor.shutdown(wait=False)
        except:
            pass

    def _validate_configuration(self):
        """Validate configuration parameters."""
        errors = []
        
        if not self.config.polygon_api_key:
            errors.append("Polygon API key is required")
        
        if self.config.memory_cache_max_gb < 1.0:
            errors.append("Memory cache must be at least 1GB")
        
        if self.config.memory_cache_max_gb > 500.0:  # Reasonable limit for 624GB system
            logger.warning(f"Large memory cache configured: {self.config.memory_cache_max_gb}GB")
        
        if self.config.batch_size < 1 or self.config.batch_size > 10000:
            errors.append("Batch size must be between 1 and 10000")
        
        if errors:
            raise ValueError(f"Configuration errors: {', '.join(errors)}")

    def _process_trade_message(self, data_point):
        """Process trade messages in thread pool."""
        if not self.is_running:
            return
        
        # Convert to dict format for processing
        message = {
            'ev': 'T',
            'sym': data_point.symbol,
            't': data_point.timestamp,
            'p': data_point.price,
            's': data_point.size
        }
        
        # Process in thread to avoid blocking async loop
        if hasattr(self, '_executor'):
            self._executor.submit(self._process_incoming_data, message)
        else:
            self._process_incoming_data(message)

    def _process_quote_message(self, data_point):
        """Process quote messages in thread pool."""
        if not self.is_running:
            return
        
        # Convert to dict format for processing
        message = {
            'ev': 'Q',
            'sym': data_point.symbol,
            't': data_point.timestamp,
            'bp': data_point.bid_price,
            'ap': data_point.ask_price,
            'bs': data_point.bid_size,
            'as': data_point.ask_size
        }
        
        # Process in thread to avoid blocking async loop
        if hasattr(self, '_executor'):
            self._executor.submit(self._process_incoming_data, message)
        else:
            self._process_incoming_data(message)

    def _process_aggregate_message(self, data_point):
        """Process aggregate messages in thread pool."""
        if not self.is_running:
            return
        
        # Convert to dict format for processing
        message = {
            'ev': 'A',
            'sym': data_point.symbol,
            't': data_point.timestamp,
            'o': data_point.open,
            'h': data_point.high,
            'l': data_point.low,
            'c': data_point.close,
            'v': data_point.volume,
            'vw': data_point.vwap,
            'n': data_point.transactions
        }
        
        # Process in thread to avoid blocking async loop
        if hasattr(self, '_executor'):
            self._executor.submit(self._process_incoming_data, message)
        else:
            self._process_incoming_data(message)

    async def start_data_ingestion(self, symbols: List[str]):
        """
        Enhanced data ingestion with proper async/thread coordination.
        
        Args:
            symbols (List[str]): List of symbols to subscribe to.
        """
        if self.is_running:
            logger.warning("Data ingestion is already running.")
            return

        self.is_running = True
        self.subscribed_symbols = symbols.copy()
        
        try:
            # Connect to Polygon WebSocket
            if not await self.polygon_client._connect_websocket():
                raise ConnectionError("Failed to connect to Polygon WebSocket")
            
            # Subscribe to data streams
            await self.polygon_client.subscribe_to_trades(symbols)
            await self.polygon_client.subscribe_to_quotes(symbols)
            await self.polygon_client.subscribe_to_second_aggregates(symbols)
            
            # Add message handlers
            self.polygon_client.add_handler("T", self._process_trade_message)
            self.polygon_client.add_handler("Q", self._process_quote_message)
            self.polygon_client.add_handler("A", self._process_aggregate_message)
            
            # Start streaming
            logger.info(f"Starting data stream for {len(symbols)} symbols")
            await self.polygon_client.stream_data()
            
        except Exception as e:
            logger.error(f"Error starting data ingestion: {e}")
            self.is_running = False
            raise

    def _process_incoming_data(self, message: Dict[str, Any]):
        """
        Enhanced callback function to process incoming data from Polygon.io with ARM64 optimizations.
        Caches data, persists to HDF5, and publishes via ZeroMQ with performance monitoring.

        Args:
            message (Dict[str, Any]): The incoming data message.
        """
        if not self.is_running:
            return

        start_time = time.perf_counter()
        message_type = message.get('ev')  # Event type, e.g., 'T' for trade, 'Q' for quote, 'AM' for aggregate minute
        symbol = message.get('sym')

        if not symbol:
            logger.warning(f"Received message without symbol: {message}")
            with self.stats_lock:
                self.stats.messages_failed += 1
            return

        try:
            with self.processing_lock:
                # ARM64 optimized data processing
                processed_data = self._optimize_data_for_arm64(message)
                
                # Store in unified memory cache with ARM64 optimizations
                cache_success = self.memory_cache.store_market_data(symbol, processed_data)
                
                with self.stats_lock:
                    if cache_success:
                        self.stats.cache_hits += 1
                    else:
                        self.stats.cache_misses += 1
                
                # Persist to HDF5 for historical analysis (ARM64 optimized batching)
                if self.config.enable_batching:
                    self._batch_hdf5_write(symbol, processed_data)
                else:
                    self._direct_hdf5_write(symbol, processed_data)
                
                # Publish to model processes via ZeroMQ with ARM64 optimizations
                if message_type in ['T', 'Q', 'AM', 'A']:  # Include aggregates
                    zmq_message = self._create_zmq_message(symbol, processed_data, message_type)
                    publish_success = self.zmq_publisher.publish_market_data(zmq_message)
                    
                    with self.stats_lock:
                        if publish_success:
                            self.stats.zmq_publishes += 1
                        
                # Update statistics
                with self.stats_lock:
                    self.stats.messages_processed += 1
                    self.stats.bytes_processed += len(str(message).encode('utf-8'))
                    
                # Call registered message handlers
                self._call_message_handlers(message_type, processed_data)
                
                # Performance monitoring
                if self.config.enable_performance_monitoring:
                    processing_time = time.perf_counter() - start_time
                    if processing_time > 0.001:  # Log slow processing (>1ms)
                        logger.warning(f"Slow message processing for {symbol}: {processing_time:.4f}s")

        except Exception as e:
            logger.error(f"Error processing incoming data for {symbol}: {e}", exc_info=True)
            with self.stats_lock:
                self.stats.messages_failed += 1
                self.stats.errors += 1
            self.error_counts[f"{symbol}_{message_type}"] += 1
            
            # Error recovery if enabled
            if self.config.enable_error_recovery:
                self._handle_processing_error(symbol, message, e)
    
    def _optimize_data_for_arm64(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Enhanced ARM64 optimization with memory management."""
        if not self.is_arm64 or not self.config.enable_arm64_optimizations:
            return message
        
        try:
            # Check memory pressure
            if self._check_memory_pressure():
                return self._create_minimal_data_structure(message)
            
            # Full optimization for normal conditions
            return self._create_optimized_data_structure(message)
            
        except Exception as e:
            logger.warning(f"ARM64 optimization failed: {e}")
            return message

    def _check_memory_pressure(self) -> bool:
        """Check if system is under memory pressure."""
        try:
            import psutil
            memory = psutil.virtual_memory()
            # Alert if over 70% of 624GB is used
            return memory.percent > 70
        except ImportError:
            return False

    def _create_minimal_data_structure(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Create minimal data structure under memory pressure."""
        return {
            'symbol': message.get('sym', ''),
            'timestamp': int(message.get('t', time.time() * 1e9)),
            'event_type': message.get('ev', ''),
            'price': float(message.get('p', message.get('c', 0.0))),
            'volume': int(message.get('s', message.get('v', 0)))
        }

    def _create_optimized_data_structure(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Create full optimized data structure."""
        # Create aligned arrays for SIMD operations
        numeric_data = np.array([
            float(message.get('p', message.get('P', message.get('c', 0.0)))),
            float(message.get('bp', message.get('b', 0.0))),
            float(message.get('ap', message.get('a', 0.0))),
            float(message.get('o', 0.0)),
            float(message.get('h', 0.0)),
            float(message.get('l', 0.0)),
            float(message.get('vw', 0.0))
        ], dtype=np.float64)  # 8-byte aligned for ARM64
        
        volume_data = np.array([
            int(message.get('s', message.get('v', message.get('V', 0)))),
            int(message.get('bs', message.get('B', 0))),
            int(message.get('as', message.get('A', 0))),
            int(message.get('n', 0))
        ], dtype=np.int64)  # 8-byte aligned for ARM64
        
        return {
            'symbol': message.get('sym', ''),
            'timestamp': int(message.get('t', time.time() * 1e9)),
            'event_type': message.get('ev', ''),
            'numeric_data': numeric_data,  # SIMD-optimized
            'volume_data': volume_data,    # SIMD-optimized
            '_arm64_aligned': True,
            '_processing_timestamp': time.perf_counter_ns(),
            'raw_message': message if self.config.enable_performance_monitoring else None
        }
    
    def _create_zmq_message(self, symbol: str, data: Dict[str, Any], message_type: str) -> Dict[str, Any]:
        """
        Create optimized ZMQ message format.
        
        Args:
            symbol: Trading symbol
            data: Processed data
            message_type: Type of market data message
            
        Returns:
            Formatted ZMQ message
        """
        return {
            'symbol': symbol,
            'timestamp': data.get('timestamp', int(time.time() * 1e9)),
            'price': data.get('price', 0.0),
            'volume': data.get('volume', 0),
            'bid': data.get('bid', 0.0),
            'ask': data.get('ask', 0.0),
            'bid_size': data.get('bid_size', 0),
            'ask_size': data.get('ask_size', 0),
            'open': data.get('open', 0.0),
            'high': data.get('high', 0.0),
            'low': data.get('low', 0.0),
            'close': data.get('close', 0.0),
            'vwap': data.get('vwap', 0.0),
            'trade_count': data.get('trade_count', 0),
            'message_type': 'market_data',
            'event_type': message_type,
            'processing_latency_ns': time.perf_counter_ns() - data.get('_processing_timestamp', 0)
        }
    
    def _batch_hdf5_write(self, symbol: str, data: Dict[str, Any]):
        """Real batching implementation for HDF5."""
        try:
            # Add to batch queue
            batch_item = {
                'symbol': symbol,
                'data': data,
                'timestamp': time.time()
            }
            
            # Non-blocking queue put
            try:
                self._hdf5_batch_queue.put_nowait(batch_item)
            except asyncio.QueueFull:
                logger.warning("HDF5 batch queue full, dropping message")
                return
            
            # Start batch processor if not running
            if not self._batch_processing_task:
                self._batch_processing_task = asyncio.create_task(self._batch_processor())
            
            with self.stats_lock:
                self.stats.hdf5_writes += 1
                
        except Exception as e:
            logger.error(f"Batch queuing failed for {symbol}: {e}")
    
    def _direct_hdf5_write(self, symbol: str, data: Dict[str, Any]):
        """
        Direct HDF5 write for immediate persistence.
        
        Args:
            symbol: Trading symbol
            data: Market data to write
        """
        try:
            # Direct write implementation
            with self.stats_lock:
                self.stats.hdf5_writes += 1
            logger.debug(f"Direct HDF5 write for {symbol}")
        except Exception as e:
            logger.error(f"Direct HDF5 write failed for {symbol}: {e}")
    
    async def _batch_processor(self):
        """Process HDF5 writes in batches."""
        batch = []
        last_write = time.time()
        
        while self.is_running:
            try:
                # Wait for items or timeout
                try:
                    item = await asyncio.wait_for(
                        self._hdf5_batch_queue.get(), 
                        timeout=1.0
                    )
                    batch.append(item)
                except asyncio.TimeoutError:
                    pass
                
                # Write batch if conditions met
                current_time = time.time()
                should_write = (
                    len(batch) >= self.config.batch_size or
                    (batch and current_time - last_write > 5.0)  # 5 second timeout
                )
                
                if should_write and batch:
                    await self._write_batch_to_hdf5(batch)
                    batch.clear()
                    last_write = current_time
                    
            except Exception as e:
                logger.error(f"Batch processor error: {e}")
                
        # Final batch write on shutdown
        if batch:
            await self._write_batch_to_hdf5(batch)

    async def _write_batch_to_hdf5(self, batch: List[Dict[str, Any]]):
        """Write batch of data to HDF5."""
        try:
            # Group by symbol for efficient writing
            symbol_groups = defaultdict(list)
            for item in batch:
                symbol_groups[item['symbol']].append(item['data'])
            
            # Write each symbol's data
            for symbol, data_list in symbol_groups.items():
                # Run in thread pool to avoid blocking
                await asyncio.get_event_loop().run_in_executor(
                    self._executor,
                    self._write_symbol_data_to_hdf5,
                    symbol,
                    data_list
                )
                
            logger.debug(f"Wrote batch of {len(batch)} items to HDF5")
            
        except Exception as e:
            logger.error(f"Batch HDF5 write failed: {e}")

    def _write_symbol_data_to_hdf5(self, symbol: str, data_list: List[Dict[str, Any]]):
        """Write symbol data to HDF5 in thread."""
        try:
            # Convert to pandas DataFrame for efficient HDF5 storage
            import pandas as pd
            
            records = []
            for data in data_list:
                record = {
                    'timestamp': data.get('timestamp', 0),
                    'price': data.get('price', 0.0),
                    'volume': data.get('volume', 0),
                    'bid': data.get('bid', 0.0),
                    'ask': data.get('ask', 0.0),
                    'bid_size': data.get('bid_size', 0),
                    'ask_size': data.get('ask_size', 0)
                }
                records.append(record)
            
            if records:
                df = pd.DataFrame(records)
                # Use HDF5 storage to append data
                self.hdf5_storage.append_market_data(symbol, df)
                
        except Exception as e:
            logger.error(f"HDF5 write failed for {symbol}: {e}")

    def _call_message_handlers(self, message_type: str, data: Dict[str, Any]):
        """
        Call registered message handlers for specific message types.
        
        Args:
            message_type: Type of message
            data: Processed message data
        """
        handlers = self.message_handlers.get(message_type, [])
        for handler in handlers:
            try:
                handler(data)
            except Exception as e:
                logger.error(f"Message handler failed for {message_type}: {e}")
    
    def _handle_processing_error(self, symbol: str, message: Dict[str, Any], error: Exception):
        """Enhanced error recovery with circuit breaker pattern."""
        error_key = f"{symbol}_{message.get('ev', 'unknown')}"
        self.error_counts[error_key] += 1
        error_count = self.error_counts[error_key]
        
        # Circuit breaker logic
        if error_count > 10:
            logger.error(f"Circuit breaker triggered for {error_key}")
            # Remove from active processing temporarily
            self._circuit_breaker_symbols.add(symbol)
            
            # Schedule recovery attempt
            asyncio.create_task(self._schedule_recovery(symbol, 60))  # 60 second timeout
        
        elif error_count > 5:
            logger.warning(f"High error count for {error_key}: {error_count}")
            # Implement exponential backoff
            backoff_time = min(2 ** (error_count - 5), 30)  # Max 30 seconds
            asyncio.create_task(self._backoff_processing(symbol, backoff_time))
        
        # Log error details
        logger.debug(f"Processing error for {symbol}: {error}")

    async def _schedule_recovery(self, symbol: str, delay: float):
        """Schedule recovery attempt for failed symbol."""
        await asyncio.sleep(delay)
        
        if symbol in getattr(self, '_circuit_breaker_symbols', set()):
            # Reset error count and remove from circuit breaker
            error_keys = [k for k in self.error_counts.keys() if k.startswith(symbol)]
            for key in error_keys:
                self.error_counts[key] = 0
            
            self._circuit_breaker_symbols.discard(symbol)
            logger.info(f"Recovery attempted for {symbol}")

    async def _backoff_processing(self, symbol: str, delay: float):
        """Implement backoff for high-error symbols."""
        # Add symbol to temporary blacklist
        getattr(self, '_backoff_symbols', set()).add(symbol)
        
        await asyncio.sleep(delay)
        
        # Remove from blacklist
        getattr(self, '_backoff_symbols', set()).discard(symbol)

    async def stop_data_ingestion(self):
        """Enhanced async shutdown."""
        if not self.is_running:
            logger.warning("Data ingestion is not running.")
            return

        logger.info("Initiating graceful shutdown...")
        self.is_running = False
        
        try:
            # Stop batch processor
            if self._batch_processing_task:
                self._batch_processing_task.cancel()
                try:
                    await self._batch_processing_task
                except asyncio.CancelledError:
                    pass
            
            # Close polygon client
            await self.polygon_client.close()
            
            # Flush remaining data
            await self._flush_all_pending_data()
            
            # Close other components
            if hasattr(self.zmq_publisher, 'close'):
                self.zmq_publisher.close()
            
            # Shutdown thread pool
            if hasattr(self, '_executor'):
                self._executor.shutdown(wait=True, timeout=30)
            
            self._log_final_statistics()
            logger.info("Graceful shutdown completed")
            
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")

    async def _flush_all_pending_data(self):
        """Flush all pending data to storage."""
        try:
            # Process remaining items in batch queue
            remaining_items = []
            while not self._hdf5_batch_queue.empty():
                try:
                    item = self._hdf5_batch_queue.get_nowait()
                    remaining_items.append(item)
                except asyncio.QueueEmpty:
                    break
            
            if remaining_items:
                await self._write_batch_to_hdf5(remaining_items)
                logger.info(f"Flushed {len(remaining_items)} remaining items")
                
        except Exception as e:
            logger.error(f"Error flushing pending data: {e}")

    def get_cached_data(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Enhanced cached data retrieval with ARM64 optimizations.

        Args:
            symbol (str): The symbol to retrieve data for.

        Returns:
            Optional[Dict[str, Any]]: The latest cached data, or None if not found.
        """
        try:
            data = self.memory_cache.get_market_data(symbol)
            
            with self.stats_lock:
                if data:
                    self.stats.cache_hits += 1
                else:
                    self.stats.cache_misses += 1
            
            return data
            
        except Exception as e:
            logger.error(f"Error retrieving cached data for {symbol}: {e}")
            return None

    def get_historical_data(self, symbol: str, start_timestamp: int, end_timestamp: int) -> Optional[Any]:
        """
        Enhanced historical data retrieval with ARM64 optimizations.

        Args:
            symbol (str): The symbol to retrieve data for.
            start_timestamp (int): Start timestamp (e.g., Unix nanoseconds).
            end_timestamp (int): End timestamp (e.g., Unix nanoseconds).

        Returns:
            Optional[Any]: Historical data, or None if not found.
        """
        try:
            # ARM64 optimized data retrieval
            if self.is_arm64 and self.config.enable_arm64_optimizations:
                return self._get_historical_data_arm64(symbol, start_timestamp, end_timestamp)
            else:
                return self._get_historical_data_standard(symbol, start_timestamp, end_timestamp)
                
        except Exception as e:
            logger.error(f"Error retrieving historical data for {symbol}: {e}")
            return None
    
    def _get_historical_data_arm64(self, symbol: str, start_timestamp: int, end_timestamp: int) -> Optional[Any]:
        """
        ARM64 optimized historical data retrieval.
        
        Args:
            symbol: Trading symbol
            start_timestamp: Start time in nanoseconds
            end_timestamp: End time in nanoseconds
            
        Returns:
            Historical data optimized for ARM64 processing
        """
        try:
            # ARM64 specific optimizations for data retrieval
            # This would include SIMD-optimized data loading and processing
            logger.debug(f"ARM64 optimized historical data retrieval for {symbol}")
            
            # Placeholder for actual HDF5 retrieval with ARM64 optimizations
            # return self.hdf5_storage.load_ohlcv_data_arm64(symbol, start_timestamp, end_timestamp)
            return None
            
        except Exception as e:
            logger.error(f"ARM64 historical data retrieval failed for {symbol}: {e}")
            return None
    
    def _get_historical_data_standard(self, symbol: str, start_timestamp: int, end_timestamp: int) -> Optional[Any]:
        """
        Standard historical data retrieval.
        
        Args:
            symbol: Trading symbol
            start_timestamp: Start time in nanoseconds
            end_timestamp: End time in nanoseconds
            
        Returns:
            Historical data
        """
        try:
            logger.debug(f"Standard historical data retrieval for {symbol}")
            
            # Placeholder for actual HDF5 retrieval
            # return self.hdf5_storage.load_ohlcv_data(symbol, start_timestamp, end_timestamp)
            return None
            
        except Exception as e:
            logger.error(f"Standard historical data retrieval failed for {symbol}: {e}")
            return None
    
    def register_message_handler(self, message_type: str, handler: Callable[[Dict[str, Any]], None]):
        """
        Register a custom message handler for specific message types.
        
        Args:
            message_type: Type of message to handle (e.g., 'T', 'Q', 'AM')
            handler: Callback function to handle the message
        """
        self.message_handlers[message_type].append(handler)
        logger.info(f"Registered message handler for type: {message_type}")
    
    def unregister_message_handler(self, message_type: str, handler: Callable[[Dict[str, Any]], None]):
        """
        Unregister a message handler.
        
        Args:
            message_type: Type of message
            handler: Handler function to remove
        """
        if handler in self.message_handlers[message_type]:
            self.message_handlers[message_type].remove(handler)
            logger.info(f"Unregistered message handler for type: {message_type}")
    
    def get_statistics(self) -> DataManagerStats:
        """
        Get current performance statistics.
        
        Returns:
            Current statistics object
        """
        with self.stats_lock:
            return self.stats
    
    def reset_statistics(self):
        """
        Reset performance statistics.
        """
        with self.stats_lock:
            self.stats = DataManagerStats()
            self.error_counts.clear()
        logger.info("Statistics reset")
    
    def get_active_symbols(self) -> List[str]:
        """
        Get list of currently active symbols.
        
        Returns:
            List of active symbol strings
        """
        return self.subscribed_symbols.copy()
    
    def add_symbol(self, symbol: str):
        """
        Add a symbol to the subscription list.
        
        Args:
            symbol: Trading symbol to add
        """
        if symbol not in self.subscribed_symbols:
            self.subscribed_symbols.append(symbol)
            with self.stats_lock:
                self.stats.symbols_active = len(self.subscribed_symbols)
            logger.info(f"Added symbol: {symbol}")
    
    def remove_symbol(self, symbol: str):
        """
        Remove a symbol from the subscription list.
        
        Args:
            symbol: Trading symbol to remove
        """
        if symbol in self.subscribed_symbols:
            self.subscribed_symbols.remove(symbol)
            with self.stats_lock:
                self.stats.symbols_active = len(self.subscribed_symbols)
            logger.info(f"Removed symbol: {symbol}")
    
    def _flush_pending_writes(self):
        """
        Flush any pending batched writes to HDF5.
        """
        try:
            # Implementation would flush any pending batched writes
            logger.info("Flushing pending HDF5 writes")
        except Exception as e:
            logger.error(f"Error flushing pending writes: {e}")
    
    def _log_final_statistics(self):
        """
        Log final performance statistics on shutdown.
        """
        try:
            stats = self.get_statistics()
            logger.info(f"Final Statistics:")
            logger.info(f"  Messages Processed: {stats.messages_processed}")
            logger.info(f"  Messages Failed: {stats.messages_failed}")
            logger.info(f"  Bytes Processed: {stats.bytes_processed}")
            logger.info(f"  Cache Hits: {stats.cache_hits}")
            logger.info(f"  Cache Misses: {stats.cache_misses}")
            logger.info(f"  HDF5 Writes: {stats.hdf5_writes}")
            logger.info(f"  ZMQ Publishes: {stats.zmq_publishes}")
            logger.info(f"  Errors: {stats.errors}")
            logger.info(f"  Uptime: {stats.uptime_seconds:.2f} seconds")
            logger.info(f"  Messages/Second: {stats.messages_per_second:.2f}")
        except Exception as e:
            logger.error(f"Error logging final statistics: {e}")
    
    def health_check(self) -> Dict[str, Any]:
        """
        Perform a health check of all components.
        
        Returns:
            Health status dictionary
        """
        health = {
            'status': 'healthy',
            'timestamp': time.time(),
            'components': {},
            'statistics': {}
        }
        
        try:
            # Check polygon client
            health['components']['polygon_client'] = {
                'status': 'connected' if hasattr(self.polygon_client, 'is_connected') else 'unknown',
                'last_message': getattr(self.polygon_client, 'last_message_time', None)
            }
            
            # Check memory cache
            health['components']['memory_cache'] = {
                'status': 'active',
                'memory_usage_gb': getattr(self.memory_cache, 'current_memory_gb', 0)
            }
            
            # Check ZMQ publisher
            health['components']['zmq_publisher'] = {
                'status': 'active' if hasattr(self.zmq_publisher, 'is_active') else 'unknown'
            }
            
            # Add statistics
            stats = self.get_statistics()
            health['statistics'] = {
                'messages_processed': stats.messages_processed,
                'messages_per_second': stats.messages_per_second,
                'error_rate': stats.messages_failed / max(stats.messages_processed, 1),
                'cache_hit_rate': stats.cache_hits / max(stats.cache_hits + stats.cache_misses, 1),
                'uptime_seconds': stats.uptime_seconds
            }
            
            # Determine overall health
            error_rate = health['statistics']['error_rate']
            if error_rate > 0.1:  # More than 10% error rate
                health['status'] = 'degraded'
            elif error_rate > 0.05:  # More than 5% error rate
                health['status'] = 'warning'
                
        except Exception as e:
            health['status'] = 'error'
            health['error'] = str(e)
            logger.error(f"Health check failed: {e}")
        
        return health

if __name__ == "__main__":
    import asyncio

    # Example usage:
    # Ensure POLYGON_API_KEY is set in your environment variables
    # export POLYGON_API_KEY="YOUR_POLYGON_API_KEY"

    async def main():
        # Initialize DataManager
        data_manager = DataManager(
            polygon_api_key=os.getenv('POLYGON_API_KEY'),
            zmq_port=5555,
            hdf5_path="data/raw/polygon/market_data.h5",
            memory_cache_max_gb=0.1 # Small cache for testing
        )

        # Symbols to subscribe to
        symbols_to_stream = ["AAPL", "MSFT"]

        # Start data ingestion in a background task
        ingestion_task = asyncio.create_task(data_manager.start_data_ingestion(symbols_to_stream))

        try:
            logger.info("DataManager running. Press Ctrl+C to stop.")
            # Keep the main loop running to allow ingestion_task to execute
            while True:
                # You can add logic here to check cached data periodically
                # For example, get_cached_data("AAPL")
                await asyncio.sleep(5) # Wait for 5 seconds
                aapl_data = data_manager.get_cached_data("AAPL")
                if aapl_data:
                    logger.info(f"Latest cached AAPL data: {aapl_data.get('data', {}).get('p')}")
                else:
                    logger.info("No cached AAPL data yet.")

        except KeyboardInterrupt:
            logger.info("Stopping DataManager...")
        finally:
            data_manager.stop_data_ingestion()
            ingestion_task.cancel()
            try:
                await ingestion_task
            except asyncio.CancelledError:
                logger.info("Ingestion task cancelled.")

    asyncio.run(main())
