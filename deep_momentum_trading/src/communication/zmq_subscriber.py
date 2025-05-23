import zmq
import msgpack
import threading
import time
import logging
import asyncio
import platform
from typing import Callable, Dict, List, Optional, Any, Tuple, Union
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime
import json

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

@dataclass
class SubscriberStats:
    """Statistics for subscriber performance monitoring"""
    messages_received: int = 0
    bytes_received: int = 0
    decompression_time: float = 0.0
    processing_time: float = 0.0
    errors: int = 0
    start_time: datetime = None
    
    def __post_init__(self):
        if self.start_time is None:
            self.start_time = datetime.now()
    
    @property
    def uptime_seconds(self) -> float:
        return (datetime.now() - self.start_time).total_seconds()
    
    @property
    def messages_per_second(self) -> float:
        uptime = self.uptime_seconds
        return self.messages_received / uptime if uptime > 0 else 0.0
    
    @property
    def bytes_per_second(self) -> float:
        uptime = self.uptime_seconds
        return self.bytes_received / uptime if uptime > 0 else 0.0

class PerformanceMonitor:
    """Monitor subscriber performance with ARM64 optimizations"""
    
    def __init__(self, window_size: int = 1000):
        self.window_size = window_size
        self.latencies = deque(maxlen=window_size)
        self.throughput_samples = deque(maxlen=window_size)
        self.last_sample_time = time.time()
        self.sample_count = 0
        
        # ARM64 optimization flags
        self.is_arm64 = platform.machine().lower() in ['arm64', 'aarch64']
        if self.is_arm64:
            logger.info("ARM64 architecture detected - enabling optimizations")
    
    def record_message(self, latency: float, size: int):
        """Record message processing metrics"""
        current_time = time.time()
        self.latencies.append(latency)
        
        # Calculate throughput
        time_diff = current_time - self.last_sample_time
        if time_diff >= 1.0:  # Sample every second
            throughput = self.sample_count / time_diff
            self.throughput_samples.append(throughput)
            self.last_sample_time = current_time
            self.sample_count = 0
        
        self.sample_count += 1
    
    def get_stats(self) -> Dict[str, float]:
        """Get current performance statistics"""
        if not self.latencies:
            return {}
        
        latencies_list = list(self.latencies)
        throughput_list = list(self.throughput_samples)
        
        return {
            'avg_latency_ms': sum(latencies_list) / len(latencies_list) * 1000,
            'p95_latency_ms': sorted(latencies_list)[int(len(latencies_list) * 0.95)] * 1000,
            'p99_latency_ms': sorted(latencies_list)[int(len(latencies_list) * 0.99)] * 1000,
            'avg_throughput': sum(throughput_list) / len(throughput_list) if throughput_list else 0,
            'max_throughput': max(throughput_list) if throughput_list else 0,
        }

class ZMQSubscriber:
    """High-performance ZeroMQ subscriber with ARM64 optimizations and compression support"""
    
    def __init__(self,
                 publishers: Optional[List[str]] = None,
                 topics: Optional[List[str]] = None,
                 enable_compression: bool = True,
                 compression_type: str = "auto",
                 enable_monitoring: bool = True,
                 buffer_size: int = 10000,
                 enable_arm64_optimizations: bool = True):
        """
        Initializes the enhanced ZMQSubscriber.

        Args:
            publishers: List of publisher addresses to connect to
            topics: List of topics to subscribe to. If None, subscribes to all
            enable_compression: Whether to enable message decompression
            compression_type: Compression type ('lz4', 'zstd', 'auto')
            enable_monitoring: Whether to enable performance monitoring
            buffer_size: Size of the receive buffer
            enable_arm64_optimizations: Whether to enable ARM64 optimizations
        """
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.SUB)
        
        # ARM64 optimization detection
        self.is_arm64 = platform.machine().lower() in ['arm64', 'aarch64']
        self.enable_arm64_optimizations = enable_arm64_optimizations and self.is_arm64
        
        if self.enable_arm64_optimizations:
            logger.info("ARM64 optimizations enabled for subscriber")
        
        # Enhanced socket configuration for ARM64
        if self.enable_arm64_optimizations:
            # ARM64-optimized settings
            self.socket.setsockopt(zmq.RCVHWM, buffer_size * 2)  # Larger buffer for ARM64
            self.socket.setsockopt(zmq.RCVBUF, 1024 * 1024)     # 1MB receive buffer
            self.socket.setsockopt(zmq.MAXMSGSIZE, 10 * 1024 * 1024)  # 10MB max message
        else:
            # Standard settings
            self.socket.setsockopt(zmq.RCVHWM, buffer_size)
        
        # Common optimizations
        self.socket.setsockopt(zmq.TCP_KEEPALIVE, 1)
        self.socket.setsockopt(zmq.TCP_KEEPALIVE_IDLE, 300)
        self.socket.setsockopt(zmq.TCP_KEEPALIVE_INTVL, 300)
        self.socket.setsockopt(zmq.RCVTIMEO, 1000)
        
        # Compression setup
        self.enable_compression = enable_compression
        self.compression_type = compression_type
        self._setup_compression()
        
        # Connect to publishers
        self.publishers = publishers if publishers is not None else []
        for publisher in self.publishers:
            self.socket.connect(publisher)
            logger.info(f"Connected to publisher: {publisher}")
        
        # Subscribe to topics
        self.topics = topics if topics is not None else []
        if self.topics:
            for topic in self.topics:
                self.socket.setsockopt(zmq.SUBSCRIBE, topic.encode('utf-8'))
                logger.info(f"Subscribed to topic: {topic}")
        else:
            self.socket.setsockopt(zmq.SUBSCRIBE, b"")
            logger.info("Subscribed to all topics.")
        
        # Message handling
        self.message_handlers: Dict[str, List[Callable]] = defaultdict(list)
        self.async_handlers: Dict[str, List[Callable]] = defaultdict(list)
        
        # State management
        self.is_running = False
        self.processing_thread: Optional[threading.Thread] = None
        self.async_loop: Optional[asyncio.AbstractEventLoop] = None
        self.async_task: Optional[asyncio.Task] = None
        
        # Statistics and monitoring
        self.stats = SubscriberStats()
        self.performance_monitor = PerformanceMonitor() if enable_monitoring else None
        self.enable_monitoring = enable_monitoring
        
        # Message queue for batch processing
        self.message_queue = deque(maxlen=buffer_size)
        self.batch_size = 100 if self.enable_arm64_optimizations else 50
        
        logger.info(f"Enhanced ZMQ Subscriber initialized with ARM64={self.enable_arm64_optimizations}, "
                   f"compression={self.enable_compression}")
    
    def _setup_compression(self):
        """Setup compression handlers based on availability and preference"""
        self.decompressors = {}
        
        if not self.enable_compression:
            return
        
        # Setup LZ4 if available
        if HAS_LZ4:
            self.decompressors['lz4'] = self._decompress_lz4
            if self.compression_type in ['auto', 'lz4']:
                logger.info("LZ4 decompression available")
        
        # Setup ZSTD if available
        if HAS_ZSTD:
            self.decompressors['zstd'] = self._decompress_zstd
            if self.compression_type in ['auto', 'zstd']:
                logger.info("ZSTD decompression available")
        
        if not self.decompressors:
            logger.warning("No compression libraries available, disabling compression")
            self.enable_compression = False
    
    def _decompress_lz4(self, data: bytes) -> bytes:
        """Decompress LZ4 data with ARM64 optimizations"""
        try:
            if self.enable_arm64_optimizations:
                # ARM64-optimized decompression
                return lz4.decompress(data, return_bytearray=False)
            else:
                return lz4.decompress(data)
        except Exception as e:
            logger.error(f"LZ4 decompression failed: {e}")
            raise
    
    def _decompress_zstd(self, data: bytes) -> bytes:
        """Decompress ZSTD data with ARM64 optimizations"""
        try:
            dctx = zstd.ZstdDecompressor()
            if self.enable_arm64_optimizations:
                # ARM64-optimized decompression
                return dctx.decompress(data, max_output_size=10*1024*1024)
            else:
                return dctx.decompress(data)
        except Exception as e:
            logger.error(f"ZSTD decompression failed: {e}")
            raise
    
    def _decompress_message(self, data: bytes, compression_type: str) -> bytes:
        """Decompress message data"""
        if not self.enable_compression or compression_type == 'none':
            return data
        
        start_time = time.time()
        try:
            if compression_type in self.decompressors:
                result = self.decompressors[compression_type](data)
                if self.enable_monitoring:
                    self.stats.decompression_time += time.time() - start_time
                return result
            else:
                logger.warning(f"Unknown compression type: {compression_type}")
                return data
        except Exception as e:
            logger.error(f"Decompression failed for {compression_type}: {e}")
            return data
        
    def add_handler(self, topic_prefix: str, handler: Callable, async_handler: bool = False):
        """
        Add a message handler for a specific topic prefix.

        Args:
            topic_prefix: The prefix of the topic to handle
            handler: The callback function to execute
            async_handler: Whether this is an async handler
        """
        if async_handler:
            self.async_handlers[topic_prefix].append(handler)
            logger.info(f"Added async handler for topic prefix: {topic_prefix}")
        else:
            self.message_handlers[topic_prefix].append(handler)
            logger.info(f"Added sync handler for topic prefix: {topic_prefix}")
    
    def add_training_handler(self, handler: Callable):
        """Add handler specifically for training events"""
        self.add_handler("training.", handler)
    
    def add_system_handler(self, handler: Callable):
        """Add handler specifically for system events"""
        self.add_handler("system.", handler)
    
    def add_market_data_handler(self, handler: Callable):
        """Add handler specifically for market data"""
        self.add_handler("market_data.", handler)
    
    def start(self, enable_async: bool = False):
        """Start message processing with optional async support."""
        if self.is_running:
            logger.warning("ZMQ Subscriber is already running.")
            return
        
        self.is_running = True
        
        if enable_async and self.async_handlers:
            # Start async event loop in separate thread
            self.processing_thread = threading.Thread(
                target=self._start_async_loop, daemon=True
            )
            self.processing_thread.start()
            logger.info("ZMQ Subscriber started with async support.")
        else:
            # Standard synchronous processing
            self.processing_thread = threading.Thread(
                target=self._message_loop, daemon=True
            )
            self.processing_thread.start()
            logger.info("ZMQ Subscriber started in sync mode.")
    
    def _start_async_loop(self):
        """Start async event loop for async message processing"""
        self.async_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.async_loop)
        
        try:
            self.async_task = self.async_loop.create_task(self._async_message_loop())
            self.async_loop.run_until_complete(self.async_task)
        except Exception as e:
            logger.error(f"Async loop error: {e}")
        finally:
            self.async_loop.close()
    
    async def _async_message_loop(self):
        """Async message processing loop"""
        while self.is_running:
            try:
                # Use asyncio-compatible ZMQ polling
                await asyncio.sleep(0.001)  # Small delay to prevent busy waiting
                
                # Check for messages with timeout
                try:
                    parts = self.socket.recv_multipart(zmq.NOBLOCK)
                    await self._process_message_async(parts)
                except zmq.Again:
                    continue
                    
            except Exception as e:
                logger.error(f"Error in async message loop: {e}")
                await asyncio.sleep(0.01)
    
    async def _process_message_async(self, parts: List[bytes]):
        """Process message asynchronously"""
        try:
            topic_str = parts[0].decode('utf-8')
            message_data = parts[1]
            
            # Handle compression metadata if present
            compression_type = 'none'
            if len(parts) > 2:
                metadata = json.loads(parts[2].decode('utf-8'))
                compression_type = metadata.get('compression', 'none')
            
            # Decompress if needed
            if self.enable_compression and compression_type != 'none':
                message_data = self._decompress_message(message_data, compression_type)
            
            # Deserialize message
            message = msgpack.unpackb(message_data, raw=False)
            
            # Route to async handlers
            await self._route_message_async(topic_str, message)
            
            # Update statistics
            if self.enable_monitoring:
                self.stats.messages_received += 1
                self.stats.bytes_received += len(message_data)
                
        except Exception as e:
            logger.error(f"Error processing async message: {e}")
            if self.enable_monitoring:
                self.stats.errors += 1
    
    async def _route_message_async(self, topic: str, message: Dict[str, Any]):
        """Route message to async handlers"""
        tasks = []
        
        for topic_prefix, handlers in self.async_handlers.items():
            if topic.startswith(topic_prefix):
                for handler in handlers:
                    try:
                        if asyncio.iscoroutinefunction(handler):
                            tasks.append(handler(topic, message))
                        else:
                            # Run sync handler in executor
                            loop = asyncio.get_event_loop()
                            tasks.append(loop.run_in_executor(None, handler, topic, message))
                    except Exception as e:
                        logger.error(f"Error creating async task for {topic}: {e}")
        
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
    
    def _message_loop(self):
        """Enhanced synchronous message processing loop with ARM64 optimizations"""
        batch_messages = []
        last_batch_time = time.time()
        
        while self.is_running:
            try:
                start_time = time.time()
                
                # Receive message parts
                parts = self.socket.recv_multipart()
                topic_str = parts[0].decode('utf-8')
                message_data = parts[1]
                
                # Handle compression metadata if present
                compression_type = 'none'
                if len(parts) > 2:
                    try:
                        metadata = json.loads(parts[2].decode('utf-8'))
                        compression_type = metadata.get('compression', 'none')
                    except (json.JSONDecodeError, UnicodeDecodeError):
                        logger.warning("Failed to parse message metadata")
                
                # Decompress if needed
                original_size = len(message_data)
                if self.enable_compression and compression_type != 'none':
                    message_data = self._decompress_message(message_data, compression_type)
                
                # Deserialize message
                message = msgpack.unpackb(message_data, raw=False)
                
                # ARM64 batch processing optimization
                if self.enable_arm64_optimizations:
                    batch_messages.append((topic_str, message, start_time))
                    
                    # Process batch when full or timeout reached
                    current_time = time.time()
                    if (len(batch_messages) >= self.batch_size or
                        current_time - last_batch_time > 0.1):  # 100ms timeout
                        self._process_message_batch(batch_messages)
                        batch_messages.clear()
                        last_batch_time = current_time
                else:
                    # Single message processing
                    self._route_message(topic_str, message)
                
                # Update statistics
                if self.enable_monitoring:
                    processing_time = time.time() - start_time
                    self.stats.messages_received += 1
                    self.stats.bytes_received += original_size
                    self.stats.processing_time += processing_time
                    
                    if self.performance_monitor:
                        self.performance_monitor.record_message(processing_time, original_size)
                
                logger.debug(f"Processed message on topic '{topic_str}' "
                           f"(compression: {compression_type})")
                
            except zmq.Again:
                # Process any remaining batch messages on timeout
                if batch_messages:
                    self._process_message_batch(batch_messages)
                    batch_messages.clear()
                    last_batch_time = time.time()
                continue
                
            except Exception as e:
                logger.error(f"Error processing message: {e}", exc_info=True)
                if self.enable_monitoring:
                    self.stats.errors += 1
                time.sleep(0.01)
        
        # Process any remaining messages when stopping
        if batch_messages:
            self._process_message_batch(batch_messages)
    
    def _process_message_batch(self, batch_messages: List[Tuple[str, Dict[str, Any], float]]):
        """Process a batch of messages for ARM64 optimization"""
        try:
            for topic_str, message, start_time in batch_messages:
                self._route_message(topic_str, message)
        except Exception as e:
            logger.error(f"Error processing message batch: {e}")
    
    def _route_message(self, topic: str, message: Dict[str, Any]):
        """
        Routes the received message to all registered handlers whose topic prefix matches.

        Args:
            topic (str): The full topic string of the received message.
            message (Dict[str, Any]): The deserialized message content.
        """
        handled = False
        for topic_prefix, handlers in self.message_handlers.items():
            if topic.startswith(topic_prefix):
                for handler in handlers:
                    try:
                        handler(topic, message)
                        handled = True
                    except Exception as e:
                        logger.error(f"Handler error for topic {topic} (message: {message}): {e}", exc_info=True)
        if not handled:
            logger.debug(f"No handler found for topic: {topic}")
    
    def stop(self):
        """Gracefully stops the message processing and closes the socket."""
        if not self.is_running:
            logger.warning("ZMQ Subscriber is not running.")
            return
        
        self.is_running = False
        
        # Cancel async task if running
        if self.async_task and not self.async_task.done():
            self.async_task.cancel()
        
        # Stop async loop if running
        if self.async_loop and self.async_loop.is_running():
            self.async_loop.call_soon_threadsafe(self.async_loop.stop)
        
        # Wait for processing thread to finish
        if self.processing_thread and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=5.0)
            if self.processing_thread.is_alive():
                logger.warning("Subscriber processing thread did not terminate gracefully.")
        
        # Close socket and context
        self.socket.close()
        self.context.term()
        
        # Log final statistics
        if self.enable_monitoring:
            logger.info(f"ZMQ Subscriber stopped. Statistics: "
                       f"Messages: {self.stats.messages_received}, "
                       f"Bytes: {self.stats.bytes_received}, "
                       f"Errors: {self.stats.errors}, "
                       f"Uptime: {self.stats.uptime_seconds:.1f}s, "
                       f"Rate: {self.stats.messages_per_second:.1f} msg/s")
        else:
            logger.info("ZMQ Subscriber stopped.")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive subscriber statistics"""
        stats = {
            'basic_stats': {
                'messages_received': self.stats.messages_received,
                'bytes_received': self.stats.bytes_received,
                'errors': self.stats.errors,
                'uptime_seconds': self.stats.uptime_seconds,
                'messages_per_second': self.stats.messages_per_second,
                'bytes_per_second': self.stats.bytes_per_second,
                'decompression_time': self.stats.decompression_time,
                'processing_time': self.stats.processing_time,
            },
            'configuration': {
                'arm64_optimizations': self.enable_arm64_optimizations,
                'compression_enabled': self.enable_compression,
                'compression_type': self.compression_type,
                'batch_size': self.batch_size,
                'publishers': self.publishers,
                'topics': self.topics,
            }
        }
        
        if self.performance_monitor:
            stats['performance'] = self.performance_monitor.get_stats()
        
        return stats
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get health status of the subscriber"""
        return {
            'is_running': self.is_running,
            'thread_alive': self.processing_thread.is_alive() if self.processing_thread else False,
            'async_running': self.async_loop.is_running() if self.async_loop else False,
            'error_rate': self.stats.errors / max(self.stats.messages_received, 1),
            'uptime_seconds': self.stats.uptime_seconds,
            'last_message_time': time.time(),  # Could be enhanced to track actual last message
        }
    
    def reset_statistics(self):
        """Reset all statistics counters"""
        self.stats = SubscriberStats()
        if self.performance_monitor:
            self.performance_monitor = PerformanceMonitor()
        logger.info("Subscriber statistics reset")

if __name__ == "__main__":
    # Example Usage
    # This example requires a ZMQPublisher running on port 5555
    # You can run the zmq_publisher.py example in a separate terminal first.

    def market_data_handler(topic: str, message: Dict[str, Any]):
        logger.info(f"Market Data Handler: Topic='{topic}', Symbol='{message.get('symbol')}', Price='{message.get('price')}'")

    def trading_signal_handler(topic: str, message: Dict[str, Any]):
        logger.info(f"Trading Signal Handler: Topic='{topic}', Symbol='{message.get('symbol')}', Signal='{message.get('signal')}'")

    # Create a subscriber instance
    # Connect to the publisher running on localhost:5555
    subscriber = ZMQSubscriber(
        publishers=["tcp://localhost:5555"],
        topics=["market_data.", "trading_signal."] # Subscribe to these prefixes
    )

    # Add handlers for specific topic prefixes
    subscriber.add_handler("market_data.", market_data_handler)
    subscriber.add_handler("trading_signal.", trading_signal_handler)

    # Start the subscriber in a background thread
    subscriber.start()

    try:
        logger.info("ZMQ Subscriber running. Waiting for messages (Ctrl+C to stop)...")
        # Keep the main thread alive
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Stopping subscriber example.")
    finally:
        subscriber.stop()
