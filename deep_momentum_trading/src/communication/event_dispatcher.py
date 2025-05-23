import zmq
import threading
import time
import platform
import asyncio
from typing import Dict, List, Callable, Any, Optional, Union
from dataclasses import dataclass, field
from enum import Enum
import logging
import msgpack
import json
from collections import defaultdict, deque
from datetime import datetime
import uuid

# Compression imports with fallbacks
try:
    import lz4.frame as lz4
    HAS_LZ4 = True
except ImportError:
    HAS_LZ4 = False

try:
    import zstandard as zstd
    HAS_ZSTD = True
except ImportError:
    HAS_ZSTD = False

from deep_momentum_trading.src.utils.logger import get_logger

logger = get_logger(__name__)

class EventPriority(Enum):
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4

@dataclass
class Event:
    event_type: str
    data: Dict[str, Any]
    priority: EventPriority = EventPriority.NORMAL
    timestamp: int = field(default_factory=lambda: int(time.time() * 1e9))
    source: str = "unknown"
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    correlation_id: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 3

@dataclass
class DispatcherStats:
    """Statistics for event dispatcher performance monitoring"""
    events_dispatched: int = 0
    events_processed: int = 0
    events_failed: int = 0
    events_retried: int = 0
    bytes_transmitted: int = 0
    processing_time: float = 0.0
    compression_time: float = 0.0
    start_time: datetime = None
    
    def __post_init__(self):
        if self.start_time is None:
            self.start_time = datetime.now()
    
    @property
    def uptime_seconds(self) -> float:
        return (datetime.now() - self.start_time).total_seconds()
    
    @property
    def events_per_second(self) -> float:
        uptime = self.uptime_seconds
        return self.events_dispatched / uptime if uptime > 0 else 0.0

class EventMetrics:
    """Monitor event processing performance with ARM64 optimizations"""
    
    def __init__(self, window_size: int = 1000):
        self.window_size = window_size
        self.processing_times = deque(maxlen=window_size)
        self.event_counts_by_type = defaultdict(int)
        self.event_counts_by_priority = defaultdict(int)
        self.failed_events = deque(maxlen=100)  # Keep last 100 failures
        
        # ARM64 optimization flags
        self.is_arm64 = platform.machine().lower() in ['arm64', 'aarch64']
        if self.is_arm64:
            logger.info("ARM64 architecture detected for event dispatcher")
    
    def record_event(self, event: Event, processing_time: float, success: bool = True):
        """Record event processing metrics"""
        self.processing_times.append(processing_time)
        self.event_counts_by_type[event.event_type] += 1
        self.event_counts_by_priority[event.priority.name] += 1
        
        if not success:
            self.failed_events.append({
                'event_id': event.event_id,
                'event_type': event.event_type,
                'timestamp': event.timestamp,
                'error_time': time.time()
            })
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get comprehensive event processing metrics"""
        if not self.processing_times:
            return {}
        
        times_list = list(self.processing_times)
        
        return {
            'performance': {
                'avg_processing_time_ms': sum(times_list) / len(times_list) * 1000,
                'p95_processing_time_ms': sorted(times_list)[int(len(times_list) * 0.95)] * 1000,
                'p99_processing_time_ms': sorted(times_list)[int(len(times_list) * 0.99)] * 1000,
                'total_events': len(times_list),
            },
            'event_types': dict(self.event_counts_by_type),
            'event_priorities': dict(self.event_counts_by_priority),
            'recent_failures': list(self.failed_events)[-10:],  # Last 10 failures
            'arm64_optimized': self.is_arm64
        }

class EventDispatcher:
    """Enhanced event dispatcher with ARM64 optimizations and compression support"""
    
    def __init__(self,
                 port: int = 5559,
                 enable_compression: bool = True,
                 compression_type: str = "auto",
                 enable_monitoring: bool = True,
                 enable_arm64_optimizations: bool = True,
                 buffer_size: int = 10000,
                 enable_async: bool = False):
        """
        Initialize enhanced event dispatcher.
        
        Args:
            port: Port for event publishing
            enable_compression: Whether to enable message compression
            compression_type: Compression type ('lz4', 'zstd', 'auto')
            enable_monitoring: Whether to enable performance monitoring
            enable_arm64_optimizations: Whether to enable ARM64 optimizations
            buffer_size: Size of event buffers
            enable_async: Whether to enable async event processing
        """
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PUB)
        
        # ARM64 optimization detection
        self.is_arm64 = platform.machine().lower() in ['arm64', 'aarch64']
        self.enable_arm64_optimizations = enable_arm64_optimizations and self.is_arm64
        
        if self.enable_arm64_optimizations:
            logger.info("ARM64 optimizations enabled for event dispatcher")
            # ARM64-optimized socket settings
            self.socket.setsockopt(zmq.SNDHWM, buffer_size * 2)
            self.socket.setsockopt(zmq.SNDBUF, 1024 * 1024)  # 1MB send buffer
        else:
            self.socket.setsockopt(zmq.SNDHWM, buffer_size)
        
        self.socket.bind(f"tcp://*:{port}")
        
        # Compression setup
        self.enable_compression = enable_compression
        self.compression_type = compression_type
        self._setup_compression()
        
        # Event handling
        self.event_handlers: Dict[str, List[Callable]] = defaultdict(list)
        self.async_handlers: Dict[str, List[Callable]] = defaultdict(list)
        
        # State management
        self.is_running = False
        self.processing_thread: Optional[threading.Thread] = None
        self.enable_async = enable_async
        self.async_loop: Optional[asyncio.AbstractEventLoop] = None
        
        # Priority queues for different event types
        self.priority_queues = {
            EventPriority.CRITICAL: deque(),
            EventPriority.HIGH: deque(),
            EventPriority.NORMAL: deque(),
            EventPriority.LOW: deque()
        }
        
        # Retry queue for failed events
        self.retry_queue = deque()
        
        # Statistics and monitoring
        self.enable_monitoring = enable_monitoring
        self.stats = DispatcherStats() if enable_monitoring else None
        self.metrics = EventMetrics() if enable_monitoring else None
        
        # Performance optimization
        self.batch_size = 100 if self.enable_arm64_optimizations else 50
        self.event_batch = []
        self.last_batch_time = time.time()
        
        logger.info(f"Enhanced EventDispatcher initialized on port {port} with "
                   f"ARM64={self.enable_arm64_optimizations}, compression={self.enable_compression}")
    
    def _setup_compression(self):
        """Setup compression handlers based on availability and preference"""
        self.compressors = {}
        
        if not self.enable_compression:
            return
        
        # Setup LZ4 if available
        if HAS_LZ4:
            self.compressors['lz4'] = self._compress_lz4
            if self.compression_type in ['auto', 'lz4']:
                logger.info("LZ4 compression available for events")
        
        # Setup ZSTD if available
        if HAS_ZSTD:
            self.compressors['zstd'] = self._compress_zstd
            if self.compression_type in ['auto', 'zstd']:
                logger.info("ZSTD compression available for events")
        
        if not self.compressors:
            logger.warning("No compression libraries available, disabling compression")
            self.enable_compression = False
    
    def _compress_lz4(self, data: bytes) -> bytes:
        """Compress data using LZ4 with ARM64 optimizations"""
        try:
            if self.enable_arm64_optimizations:
                # ARM64-optimized compression
                return lz4.compress(data, compression_level=1, auto_flush=True)
            else:
                return lz4.compress(data)
        except Exception as e:
            logger.error(f"LZ4 compression failed: {e}")
            return data
    
    def _compress_zstd(self, data: bytes) -> bytes:
        """Compress data using ZSTD with ARM64 optimizations"""
        try:
            cctx = zstd.ZstdCompressor(level=1 if self.enable_arm64_optimizations else 3)
            return cctx.compress(data)
        except Exception as e:
            logger.error(f"ZSTD compression failed: {e}")
            return data
    
    def _compress_message(self, data: bytes) -> tuple[bytes, str]:
        """Compress message data and return compressed data with compression type"""
        if not self.enable_compression:
            return data, 'none'
        
        # Choose compression method
        if self.compression_type == 'auto':
            # Prefer LZ4 for speed, ZSTD for compression ratio
            compression_method = 'lz4' if 'lz4' in self.compressors else 'zstd'
        else:
            compression_method = self.compression_type
        
        if compression_method in self.compressors:
            start_time = time.time()
            compressed_data = self.compressors[compression_method](data)
            
            if self.enable_monitoring and self.stats:
                self.stats.compression_time += time.time() - start_time
            
            return compressed_data, compression_method
        
        return data, 'none'
        
    def register_handler(self, event_type: str, handler: Callable, async_handler: bool = False):
        """Register event handler with optional async support"""
        if async_handler:
            self.async_handlers[event_type].append(handler)
            logger.info(f"Registered async handler for event type: {event_type}")
        else:
            self.event_handlers[event_type].append(handler)
            logger.info(f"Registered sync handler for event type: {event_type}")
    
    def register_training_handler(self, handler: Callable):
        """Register handler specifically for training events"""
        self.register_handler("training", handler)
    
    def register_system_handler(self, handler: Callable):
        """Register handler specifically for system events"""
        self.register_handler("system", handler)
    
    def register_market_handler(self, handler: Callable):
        """Register handler specifically for market events"""
        self.register_handler("market", handler)
    
    def dispatch_event(self, event: Event):
        """Enhanced event dispatch with compression and ARM64 optimizations"""
        start_time = time.time()
        
        # Add to priority queue for local processing
        self.priority_queues[event.priority].append(event)
        
        # Prepare message for ZMQ publishing
        topic = f"event.{event.event_type}"
        message_data = {
            'event_type': event.event_type,
            'data': event.data,
            'priority': event.priority.value,
            'timestamp': event.timestamp,
            'source': event.source,
            'event_id': event.event_id,
            'correlation_id': event.correlation_id,
            'retry_count': event.retry_count
        }
        
        # Serialize message
        serialized_message = msgpack.packb(message_data, use_bin_type=True)
        
        # Compress if enabled
        compressed_message, compression_type = self._compress_message(serialized_message)
        
        # Create metadata
        metadata = {
            'compression': compression_type,
            'original_size': len(serialized_message),
            'compressed_size': len(compressed_message)
        }
        
        try:
            # Send with compression metadata
            parts = [
                topic.encode('utf-8'),
                compressed_message,
                json.dumps(metadata).encode('utf-8')
            ]
            
            if self.enable_arm64_optimizations:
                # ARM64 batch processing
                self.event_batch.append(parts)
                
                current_time = time.time()
                if (len(self.event_batch) >= self.batch_size or
                    current_time - self.last_batch_time > 0.05):  # 50ms timeout
                    self._send_event_batch()
                    self.last_batch_time = current_time
            else:
                # Direct sending
                self.socket.send_multipart(parts, zmq.NOBLOCK)
            
            # Update statistics
            if self.enable_monitoring and self.stats:
                self.stats.events_dispatched += 1
                self.stats.bytes_transmitted += len(compressed_message)
            
            logger.debug(f"Dispatched event '{event.event_type}' with ID '{event.event_id}' "
                        f"(compression: {compression_type})")
                        
        except zmq.Again:
            logger.warning(f"Failed to dispatch event '{event.event_type}' (queue full)")
            # Add to retry queue
            if event.retry_count < event.max_retries:
                event.retry_count += 1
                self.retry_queue.append(event)
                if self.enable_monitoring and self.stats:
                    self.stats.events_retried += 1
            else:
                if self.enable_monitoring and self.stats:
                    self.stats.events_failed += 1
                    
        except Exception as e:
            logger.error(f"Error dispatching event '{event.event_type}': {e}", exc_info=True)
            if self.enable_monitoring and self.stats:
                self.stats.events_failed += 1
    
    def _send_event_batch(self):
        """Send a batch of events for ARM64 optimization"""
        try:
            for parts in self.event_batch:
                self.socket.send_multipart(parts, zmq.NOBLOCK)
            self.event_batch.clear()
        except Exception as e:
            logger.error(f"Error sending event batch: {e}")
    
    def start_processing(self, enable_async: bool = None):
        """Start enhanced event processing with optional async support."""
        if self.is_running:
            logger.warning("Event processing is already running.")
            return
        
        self.is_running = True
        
        # Use provided async setting or default from initialization
        use_async = enable_async if enable_async is not None else self.enable_async
        
        if use_async and self.async_handlers:
            # Start async event loop in separate thread
            self.processing_thread = threading.Thread(
                target=self._start_async_loop, daemon=True
            )
            self.processing_thread.start()
            logger.info("Enhanced event processing started with async support.")
        else:
            # Standard synchronous processing
            self.processing_thread = threading.Thread(
                target=self._process_events, daemon=True
            )
            self.processing_thread.start()
            logger.info("Enhanced event processing started in sync mode.")
    
    def _start_async_loop(self):
        """Start async event loop for async event processing"""
        self.async_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.async_loop)
        
        try:
            self.async_loop.run_until_complete(self._async_process_events())
        except Exception as e:
            logger.error(f"Async event loop error: {e}")
        finally:
            self.async_loop.close()
    
    async def _async_process_events(self):
        """Async event processing loop"""
        while self.is_running:
            try:
                event_processed = False
                
                # Process retry queue first
                if self.retry_queue:
                    event = self.retry_queue.popleft()
                    await self._handle_event_async(event)
                    event_processed = True
                
                # Process by priority (critical first)
                if not event_processed:
                    for priority in [EventPriority.CRITICAL, EventPriority.HIGH,
                                   EventPriority.NORMAL, EventPriority.LOW]:
                        if self.priority_queues[priority]:
                            event = self.priority_queues[priority].popleft()
                            await self._handle_event_async(event)
                            event_processed = True
                            break
                
                # Send any pending batched events
                if self.event_batch and time.time() - self.last_batch_time > 0.05:
                    self._send_event_batch()
                    self.last_batch_time = time.time()
                
                if not event_processed:
                    await asyncio.sleep(0.001)  # 1ms sleep if no events
                    
            except Exception as e:
                logger.error(f"Error in async event processing: {e}")
                await asyncio.sleep(0.01)
    
    def _process_events(self):
        """Enhanced synchronous event processing loop with ARM64 optimizations."""
        event_batch = []
        last_batch_time = time.time()
        
        while self.is_running:
            try:
                event_processed = False
                
                # Process retry queue first
                if self.retry_queue:
                    event = self.retry_queue.popleft()
                    self._handle_event(event)
                    event_processed = True
                
                # Process by priority (critical first)
                if not event_processed:
                    for priority in [EventPriority.CRITICAL, EventPriority.HIGH,
                                   EventPriority.NORMAL, EventPriority.LOW]:
                        if self.priority_queues[priority]:
                            event = self.priority_queues[priority].popleft()
                            
                            if self.enable_arm64_optimizations:
                                # Batch processing for ARM64
                                event_batch.append(event)
                                
                                current_time = time.time()
                                if (len(event_batch) >= self.batch_size or
                                    current_time - last_batch_time > 0.1):  # 100ms timeout
                                    self._handle_event_batch(event_batch)
                                    event_batch.clear()
                                    last_batch_time = current_time
                            else:
                                # Single event processing
                                self._handle_event(event)
                            
                            event_processed = True
                            break
                
                # Send any pending ZMQ batched events
                if self.event_batch and time.time() - self.last_batch_time > 0.05:
                    self._send_event_batch()
                    self.last_batch_time = time.time()
                
                # Process any remaining batch events on timeout
                if event_batch and time.time() - last_batch_time > 0.1:
                    self._handle_event_batch(event_batch)
                    event_batch.clear()
                    last_batch_time = time.time()
                
                if not event_processed:
                    time.sleep(0.001)  # 1ms sleep if no events
                    
            except Exception as e:
                logger.error(f"Error in event processing loop: {e}")
                time.sleep(0.01)
        
        # Process any remaining events when stopping
        if event_batch:
            self._handle_event_batch(event_batch)
        if self.event_batch:
            self._send_event_batch()
    
    def _handle_event_batch(self, event_batch: List[Event]):
        """Handle a batch of events for ARM64 optimization"""
        try:
            for event in event_batch:
                self._handle_event(event)
        except Exception as e:
            logger.error(f"Error handling event batch: {e}")
    
    def _handle_event(self, event: Event):
        """Enhanced event handling with monitoring and error recovery."""
        start_time = time.time()
        success = True
        
        handlers = self.event_handlers.get(event.event_type, [])
        
        if not handlers:
            logger.debug(f"No handlers registered for event type: {event.event_type}")
            return

        for handler in handlers:
            try:
                handler(event)
                logger.debug(f"Handler processed event '{event.event_type}' (ID: {event.event_id})")
            except Exception as e:
                logger.error(f"Event handler error for {event.event_type} (ID: {event.event_id}): {e}",
                           exc_info=True)
                success = False
                
                # Add to retry queue if retries available
                if event.retry_count < event.max_retries:
                    event.retry_count += 1
                    self.retry_queue.append(event)
                    if self.enable_monitoring and self.stats:
                        self.stats.events_retried += 1
        
        # Update statistics
        if self.enable_monitoring:
            processing_time = time.time() - start_time
            if self.stats:
                self.stats.events_processed += 1
                self.stats.processing_time += processing_time
                if not success:
                    self.stats.events_failed += 1
            
            if self.metrics:
                self.metrics.record_event(event, processing_time, success)
    
    async def _handle_event_async(self, event: Event):
        """Handle event asynchronously"""
        start_time = time.time()
        success = True
        
        # Handle sync handlers
        sync_handlers = self.event_handlers.get(event.event_type, [])
        for handler in sync_handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(event)
                else:
                    # Run sync handler in executor
                    loop = asyncio.get_event_loop()
                    await loop.run_in_executor(None, handler, event)
            except Exception as e:
                logger.error(f"Sync handler error for {event.event_type}: {e}")
                success = False
        
        # Handle async handlers
        async_handlers = self.async_handlers.get(event.event_type, [])
        if async_handlers:
            tasks = []
            for handler in async_handlers:
                try:
                    if asyncio.iscoroutinefunction(handler):
                        tasks.append(handler(event))
                    else:
                        loop = asyncio.get_event_loop()
                        tasks.append(loop.run_in_executor(None, handler, event))
                except Exception as e:
                    logger.error(f"Async handler error for {event.event_type}: {e}")
                    success = False
            
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
        
        # Update statistics
        if self.enable_monitoring:
            processing_time = time.time() - start_time
            if self.stats:
                self.stats.events_processed += 1
                self.stats.processing_time += processing_time
                if not success:
                    self.stats.events_failed += 1
            
            if self.metrics:
                self.metrics.record_event(event, processing_time, success)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive dispatcher statistics"""
        if not self.enable_monitoring:
            return {'monitoring_disabled': True}
        
        stats = {
            'dispatcher_stats': {
                'events_dispatched': self.stats.events_dispatched,
                'events_processed': self.stats.events_processed,
                'events_failed': self.stats.events_failed,
                'events_retried': self.stats.events_retried,
                'bytes_transmitted': self.stats.bytes_transmitted,
                'processing_time': self.stats.processing_time,
                'compression_time': self.stats.compression_time,
                'uptime_seconds': self.stats.uptime_seconds,
                'events_per_second': self.stats.events_per_second,
            },
            'configuration': {
                'arm64_optimizations': self.enable_arm64_optimizations,
                'compression_enabled': self.enable_compression,
                'compression_type': self.compression_type,
                'batch_size': self.batch_size,
                'async_enabled': self.enable_async,
            },
            'queue_status': {
                'critical_queue': len(self.priority_queues[EventPriority.CRITICAL]),
                'high_queue': len(self.priority_queues[EventPriority.HIGH]),
                'normal_queue': len(self.priority_queues[EventPriority.NORMAL]),
                'low_queue': len(self.priority_queues[EventPriority.LOW]),
                'retry_queue': len(self.retry_queue),
            }
        }
        
        if self.metrics:
            stats['metrics'] = self.metrics.get_metrics()
        
        return stats
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get health status of the event dispatcher"""
        total_queued = sum(len(q) for q in self.priority_queues.values())
        
        return {
            'is_running': self.is_running,
            'thread_alive': self.processing_thread.is_alive() if self.processing_thread else False,
            'async_running': self.async_loop.is_running() if self.async_loop else False,
            'total_queued_events': total_queued,
            'retry_queue_size': len(self.retry_queue),
            'error_rate': (self.stats.events_failed / max(self.stats.events_processed, 1)
                          if self.stats else 0),
            'uptime_seconds': self.stats.uptime_seconds if self.stats else 0,
            'events_per_second': self.stats.events_per_second if self.stats else 0,
        }

    def stop(self):
        """Gracefully stops the enhanced event dispatcher."""
        if not self.is_running:
            logger.warning("EventDispatcher is not running.")
            return
        
        self.is_running = False
        
        # Send any remaining batched events
        if self.event_batch:
            self._send_event_batch()
        
        # Stop async loop if running
        if self.async_loop and self.async_loop.is_running():
            self.async_loop.call_soon_threadsafe(self.async_loop.stop)
        
        # Wait for processing thread to finish
        if self.processing_thread and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=5.0)
            if self.processing_thread.is_alive():
                logger.warning("Event processing thread did not terminate gracefully.")
        
        # Close socket and context
        self.socket.close()
        self.context.term()
        
        # Log final statistics
        if self.enable_monitoring and self.stats:
            logger.info(f"Enhanced EventDispatcher stopped. Statistics: "
                       f"Dispatched: {self.stats.events_dispatched}, "
                       f"Processed: {self.stats.events_processed}, "
                       f"Failed: {self.stats.events_failed}, "
                       f"Retried: {self.stats.events_retried}, "
                       f"Uptime: {self.stats.uptime_seconds:.1f}s, "
                       f"Rate: {self.stats.events_per_second:.1f} events/s")
        else:
            logger.info("Enhanced EventDispatcher stopped.")

if __name__ == "__main__":
    # Enhanced Example Usage with ARM64 optimizations, compression, and async support
    
    def market_handler(event: Event):
        logger.info(f"Market Handler: {event.event_type} - Symbol: {event.data.get('symbol')}, "
                   f"Price: {event.data.get('price')}")

    def training_handler(event: Event):
        logger.info(f"Training Handler: {event.event_type} - Epoch: {event.data.get('epoch')}, "
                   f"Loss: {event.data.get('loss'):.4f}")

    async def async_system_handler(event: Event):
        """Example async handler for system events"""
        await asyncio.sleep(0.001)  # Simulate async processing
        logger.info(f"Async System Handler: {event.event_type} - {event.data.get('status')}")

    def critical_handler(event: Event):
        logger.warning(f"CRITICAL EVENT: {event.event_type} - {event.data.get('message')}")
        # Simulate occasional handler error for retry testing
        if event.retry_count == 0 and "test_retry" in event.data:
            raise ValueError("Simulated handler error for retry testing!")

    # Create enhanced dispatcher with ARM64 optimizations
    dispatcher = EventDispatcher(
        port=5560,
        enable_compression=True,
        compression_type="auto",  # Auto-detect LZ4/ZSTD
        enable_monitoring=True,
        enable_arm64_optimizations=True,
        buffer_size=20000,  # Larger buffer for high-throughput
        enable_async=True
    )

    # Register various types of handlers
    dispatcher.register_market_handler(market_handler)
    dispatcher.register_training_handler(training_handler)
    dispatcher.register_system_handler(async_system_handler)
    dispatcher.register_handler("critical_alert", critical_handler)
    
    # Register async handler
    dispatcher.register_handler("system", async_system_handler, async_handler=True)

    # Start with async support
    dispatcher.start_processing(enable_async=True)

    # Dispatch various events to demonstrate features
    events = [
        Event(
            event_type="market_update",
            data={"symbol": "AAPL", "price": 175.50, "volume": 1000000},
            priority=EventPriority.NORMAL,
            source="market_feed"
        ),
        Event(
            event_type="training_progress",
            data={"epoch": 10, "loss": 0.0234, "accuracy": 0.9876},
            priority=EventPriority.HIGH,
            source="training_system"
        ),
        Event(
            event_type="system_health",
            data={"component": "data_manager", "status": "healthy", "cpu": 45.2},
            priority=EventPriority.LOW,
            source="health_monitor"
        ),
        Event(
            event_type="critical_alert",
            data={"message": "Risk limit breached", "symbol": "XYZ", "test_retry": True},
            priority=EventPriority.CRITICAL,
            source="risk_manager"
        ),
        Event(
            event_type="market_update",
            data={"symbol": "GOOG", "price": 120.10, "volume": 750000},
            priority=EventPriority.HIGH,
            source="market_feed"
        )
    ]

    # Dispatch events
    for event in events:
        dispatcher.dispatch_event(event)
        time.sleep(0.1)  # Small delay between events

    try:
        logger.info("Enhanced EventDispatcher running with ARM64 optimizations...")
        logger.info("Features: compression, async handlers, retry logic, monitoring")
        
        # Periodically log statistics
        start_time = time.time()
        while True:
            time.sleep(5)  # Log stats every 5 seconds
            
            if time.time() - start_time > 5:
                stats = dispatcher.get_statistics()
                health = dispatcher.get_health_status()
                
                logger.info(f"Stats: Dispatched={stats['dispatcher_stats']['events_dispatched']}, "
                           f"Processed={stats['dispatcher_stats']['events_processed']}, "
                           f"Failed={stats['dispatcher_stats']['events_failed']}, "
                           f"Rate={stats['dispatcher_stats']['events_per_second']:.1f}/s")
                logger.info(f"Health: Running={health['is_running']}, "
                           f"Queued={health['total_queued_events']}, "
                           f"Error Rate={health['error_rate']:.4f}")
                
                start_time = time.time()
                
    except KeyboardInterrupt:
        logger.info("Stopping enhanced event dispatcher...")
    finally:
        # Get final statistics before stopping
        final_stats = dispatcher.get_statistics()
        logger.info(f"Final Event Statistics: {final_stats}")
        dispatcher.stop()
