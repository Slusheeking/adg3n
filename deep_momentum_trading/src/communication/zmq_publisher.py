import zmq
import zmq.asyncio
import time
import logging
import asyncio
import threading
import platform
import torch
import numpy as np
from typing import Dict, Any, List, Optional, Callable, Union, Tuple
from dataclasses import dataclass, asdict, field
from collections import defaultdict, deque
import msgpack
import lz4.frame
import zstandard as zstd
from concurrent.futures import ThreadPoolExecutor

from deep_momentum_trading.src.utils.logger import get_logger

logger = get_logger(__name__)

@dataclass
class MarketDataMessage:
    symbol: str
    timestamp: int
    price: float
    volume: int
    bid: float
    ask: float
    message_type: str = "market_data"

@dataclass
class TradingSignalMessage:
    symbol: str
    timestamp: int
    signal: float  # -1 to 1
    confidence: float  # 0 to 1
    strategy: str
    message_type: str = "trading_signal"
    # Add a generic 'features' field to allow passing arbitrary feature data
    features: Dict[str, Any] = field(default_factory=dict)

@dataclass
class TrainingMetricsMessage:
    """Training metrics message structure"""
    training_id: str
    timestamp: int
    epoch: int
    train_loss: float
    val_loss: float
    learning_rate: float
    metrics: Dict[str, Any] = field(default_factory=dict)
    message_type: str = "training_metrics"

@dataclass
class SystemEventMessage:
    """System event message structure"""
    event_type: str
    timestamp: int
    source: str
    data: Dict[str, Any] = field(default_factory=dict)
    severity: str = "info"
    message_type: str = "system_event"

@dataclass
class MessageStats:
    """Statistics for message throughput and performance with ARM64 support."""
    messages_sent: int = 0
    messages_received: int = 0
    bytes_sent: int = 0
    bytes_received: int = 0
    avg_latency_ms: float = 0.0
    errors: int = 0
    last_activity: float = 0.0
    tensor_operations: int = 0
    tensor_serialization_time: float = 0.0
    compression_time: float = 0.0
    start_time: float = field(default_factory=time.time)
    
    @property
    def uptime_seconds(self) -> float:
        return time.time() - self.start_time
    
    @property
    def messages_per_second(self) -> float:
        uptime = self.uptime_seconds
        return self.messages_sent / uptime if uptime > 0 else 0.0

class CompressionHandler:
    """Handles message compression and decompression with ARM64 optimizations."""
    
    def __init__(self, compression_type: str = "lz4", enable_arm64_optimizations: bool = True):
        self.compression_type = compression_type.lower()
        self.is_arm64 = platform.machine().lower() in ['arm64', 'aarch64']
        self.enable_arm64_optimizations = enable_arm64_optimizations and self.is_arm64
        
        if self.compression_type == "zstd":
            self.compressor = zstd.ZstdCompressor(level=3, threads=-1)  # Use all available cores
            self.decompressor = zstd.ZstdDecompressor()
        
        # ARM64-specific compression settings
        if self.enable_arm64_optimizations:
            self.compression_level = lz4.frame.COMPRESSIONLEVEL_MINHC
        else:
            self.compression_level = lz4.frame.COMPRESSIONLEVEL_DEFAULT
        
    def compress(self, data: bytes) -> bytes:
        """Compress data using the configured algorithm with ARM64 optimizations."""
        try:
            if self.compression_type == "lz4":
                if self.enable_arm64_optimizations:
                    # ARM64-optimized compression
                    return lz4.frame.compress(
                        data,
                        compression_level=self.compression_level,
                        auto_flush=True
                    )
                else:
                    return lz4.frame.compress(data, compression_level=1)  # Fast compression
            elif self.compression_type == "zstd":
                return self.compressor.compress(data)
            else:
                return data
        except Exception as e:
            logger.error(f"Compression error: {e}")
            return data
    
    def decompress(self, data: bytes) -> bytes:
        """Decompress data using the configured algorithm."""
        try:
            if self.compression_type == "lz4":
                return lz4.frame.decompress(data)
            elif self.compression_type == "zstd":
                return self.decompressor.decompress(data)
            else:
                return data
        except Exception as e:
            logger.error(f"Decompression error: {e}")
            return data

class ZMQPublisher:
    """Enhanced high-performance ZeroMQ publisher with ARM64 optimizations"""
    
    def __init__(self,
                 port: int,
                 high_water_mark: int = 10000,
                 compression: str = "lz4",
                 enable_monitoring: bool = True,
                 thread_pool_size: int = 4,
                 enable_arm64_optimizations: bool = True,
                 tensor_pool_size: int = 1000):
        """
        Initialize enhanced ZMQ publisher.
        
        Args:
            port (int): Port to bind to
            high_water_mark (int): High water mark for send queue
            compression (str): Compression algorithm ('none', 'lz4', 'zstd')
            enable_monitoring (bool): Enable performance monitoring
            thread_pool_size (int): Size of thread pool for async operations
        """
        self.port = port
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PUB)
        
        # ARM64 optimization detection
        self.is_arm64 = platform.machine().lower() in ['arm64', 'aarch64']
        self.enable_arm64_optimizations = enable_arm64_optimizations and self.is_arm64
        
        if self.enable_arm64_optimizations:
            logger.info("ARM64 optimizations enabled for ZMQ publisher")
        
        # ARM64-specific socket optimizations
        if self.enable_arm64_optimizations:
            self.socket.setsockopt(zmq.SNDHWM, high_water_mark * 2)  # Higher water mark
            self.socket.setsockopt(zmq.SNDBUF, 2 * 1024 * 1024)  # 2MB send buffer
        else:
            self.socket.setsockopt(zmq.SNDHWM, high_water_mark)
            self.socket.setsockopt(zmq.SNDBUF, 1024 * 1024)  # 1MB send buffer
        
        # Common optimizations
        self.socket.setsockopt(zmq.LINGER, 0)  # Do not linger on close
        self.socket.setsockopt(zmq.TCP_KEEPALIVE, 1)  # Enable TCP keep-alives
        self.socket.setsockopt(zmq.TCP_KEEPALIVE_IDLE, 300)  # 5 minutes
        self.socket.setsockopt(zmq.TCP_KEEPALIVE_INTVL, 30)  # 30 seconds
        self.socket.setsockopt(zmq.TCP_KEEPALIVE_CNT, 3)  # 3 probes
        self.socket.setsockopt(zmq.IMMEDIATE, 1)  # Immediate delivery
        
        # Bind socket
        self.socket.bind(f"tcp://*:{port}")
        
        # Initialize compression with ARM64 support
        self.compression_handler = CompressionHandler(
            compression, 
            enable_arm64_optimizations=self.enable_arm64_optimizations
        ) if compression != "none" else None
        
        # Tensor support for ARM64
        self.tensor_pool_size = tensor_pool_size
        self.tensor_pool = {}
        self.tensor_cache = {}
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.compression_threshold = 1024  # Compress messages > 1KB
        
        # Performance monitoring
        self.enable_monitoring = enable_monitoring
        self.stats = MessageStats()
        self.topic_stats = defaultdict(lambda: MessageStats())
        self.latency_buffer = deque(maxlen=1000)
        
        # Thread pool for async operations
        self.thread_pool = ThreadPoolExecutor(max_workers=thread_pool_size)
        
        # Message batching for high throughput
        self.batch_size = 100
        self.batch_timeout_ms = 10
        self.message_batch = []
        self.last_batch_time = time.time()
        self.batch_lock = threading.Lock()
        
        # Start monitoring thread if enabled
        if self.enable_monitoring:
            self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
            self.monitoring_thread.start()
        
        logger.info(f"Enhanced ZMQ Publisher started on port {port} with compression: {compression}")
        logger.info(f"ARM64 optimizations: {self.enable_arm64_optimizations}, Tensor support: enabled")
    
    def _monitoring_loop(self):
        """Background monitoring loop for performance metrics."""
        while True:
            try:
                time.sleep(30)  # Update every 30 seconds
                
                # Calculate average latency
                if self.latency_buffer:
                    self.stats.avg_latency_ms = sum(self.latency_buffer) / len(self.latency_buffer)
                
                # Log performance stats
                if self.stats.messages_sent > 0:
                    logger.debug(f"ZMQ Publisher Stats - Messages: {self.stats.messages_sent}, "
                               f"Bytes: {self.stats.bytes_sent}, Avg Latency: {self.stats.avg_latency_ms:.2f}ms")
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
    
    def _send_message(self, topic: str, message_data: Dict[str, Any], compress: bool = True):
        """
        Internal method to send messages with compression and monitoring.
        
        Args:
            topic (str): Message topic
            message_data (Dict[str, Any]): Message data
            compress (bool): Whether to compress the message
        """
        start_time = time.time()
        
        try:
            # Serialize message
            message = msgpack.packb(message_data, use_bin_type=True)
            
            # Compress if enabled
            if compress and self.compression_handler:
                message = self.compression_handler.compress(message)
            
            # Send message
            self.socket.send_multipart([
                topic.encode('utf-8'),
                message
            ], zmq.NOBLOCK)
            
            # Update statistics
            if self.enable_monitoring:
                latency_ms = (time.time() - start_time) * 1000
                self.latency_buffer.append(latency_ms)
                
                self.stats.messages_sent += 1
                self.stats.bytes_sent += len(message)
                self.stats.last_activity = time.time()
                
                # Update topic-specific stats
                topic_base = topic.split('.')[0]
                self.topic_stats[topic_base].messages_sent += 1
                self.topic_stats[topic_base].bytes_sent += len(message)
            
            logger.debug(f"Published message to topic: {topic}")
            
        except zmq.Again:
            self.stats.errors += 1
            logger.warning(f"Failed to publish to topic {topic} (queue full)")
            raise
        except Exception as e:
            self.stats.errors += 1
            logger.error(f"Error publishing to topic {topic}: {e}")
            raise
    
    def publish_market_data(self, data: Dict[str, Any]):
        """
        Publish market data with topic-based routing and compression.
        Expects a dictionary that can be converted to MarketDataMessage.
        """
        try:
            # Create MarketDataMessage instance from dictionary
            market_data_msg = MarketDataMessage(
                symbol=data.get('symbol'),
                timestamp=data.get('timestamp'),
                price=data.get('price'),
                volume=data.get('volume'),
                bid=data.get('bid', 0.0),
                ask=data.get('ask', 0.0),
                message_type=data.get('message_type', "market_data")
            )
            
            topic = f"market_data.{market_data_msg.symbol}"
            self._send_message(topic, asdict(market_data_msg))
            
        except Exception as e:
            logger.error(f"Error publishing market data for {data.get('symbol')}: {e}", exc_info=True)
    
    def publish_trading_signal(self, signal: Dict[str, Any]):
        """
        Publish trading signals to execution engines with enhanced features.
        Expects a dictionary that can be converted to TradingSignalMessage.
        """
        try:
            # Create TradingSignalMessage instance from dictionary
            trading_signal_msg = TradingSignalMessage(
                symbol=signal.get('symbol'),
                timestamp=signal.get('timestamp'),
                signal=signal.get('signal'),
                confidence=signal.get('confidence'),
                strategy=signal.get('strategy'),
                message_type=signal.get('message_type', "trading_signal"),
                features=signal.get('features', {})
            )
            
            topic = f"trading_signal.{trading_signal_msg.symbol}"
            self._send_message(topic, asdict(trading_signal_msg))
            
        except Exception as e:
            logger.error(f"Error publishing trading signal for {signal.get('symbol')}: {e}", exc_info=True)
    
    def publish_training_metrics(self, metrics: Dict[str, Any]):
        """
        Publish training metrics for MLOps monitoring.
        Expects a dictionary that can be converted to TrainingMetricsMessage.
        """
        try:
            training_metrics_msg = TrainingMetricsMessage(
                training_id=metrics.get('training_id'),
                timestamp=metrics.get('timestamp', int(time.time() * 1e9)),
                epoch=metrics.get('epoch'),
                train_loss=metrics.get('train_loss'),
                val_loss=metrics.get('val_loss'),
                learning_rate=metrics.get('learning_rate'),
                metrics=metrics.get('metrics', {}),
                message_type=metrics.get('message_type', "training_metrics")
            )
            
            topic = f"training_metrics.{training_metrics_msg.training_id}"
            self._send_message(topic, asdict(training_metrics_msg))
            
        except Exception as e:
            logger.error(f"Error publishing training metrics: {e}", exc_info=True)
    
    def publish_system_event(self, event: Dict[str, Any]):
        """
        Publish system events for monitoring and alerting.
        Expects a dictionary that can be converted to SystemEventMessage.
        """
        try:
            system_event_msg = SystemEventMessage(
                event_type=event.get('event_type'),
                timestamp=event.get('timestamp', int(time.time() * 1e9)),
                source=event.get('source', 'unknown'),
                data=event.get('data', {}),
                severity=event.get('severity', 'info'),
                message_type=event.get('message_type', "system_event")
            )
            
            topic = f"system_event.{system_event_msg.event_type}"
            self._send_message(topic, asdict(system_event_msg))
            
        except Exception as e:
            logger.error(f"Error publishing system event: {e}", exc_info=True)
    
    async def publish_async(self, topic: str, data: Dict[str, Any]):
        """
        Asynchronous message publishing for high-throughput scenarios.
        
        Args:
            topic (str): Message topic
            data (Dict[str, Any]): Message data
        """
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(self.thread_pool, self._send_message, topic, data)
    
    def publish_batch(self, messages: List[Tuple[str, Dict[str, Any]]]):
        """
        Batch publish multiple messages for improved throughput.
        
        Args:
            messages (List[Tuple[str, Dict[str, Any]]]): List of (topic, data) tuples
        """
        try:
            for topic, data in messages:
                self._send_message(topic, data, compress=False)  # Skip compression for batch
            
            logger.debug(f"Published batch of {len(messages)} messages")
            
        except Exception as e:
            logger.error(f"Error in batch publish: {e}", exc_info=True)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive publisher performance statistics."""
        avg_latency = sum(self.latency_buffer) / len(self.latency_buffer) if self.latency_buffer else 0
        p95_latency = sorted(self.latency_buffer)[int(len(self.latency_buffer) * 0.95)] if self.latency_buffer else 0
        
        return {
            'publisher_stats': {
                'messages_sent': self.stats.messages_sent,
                'bytes_sent': self.stats.bytes_sent,
                'tensor_operations': self.stats.tensor_operations,
                'messages_per_second': self.stats.messages_per_second,
                'errors': self.stats.errors,
                'uptime_seconds': self.stats.uptime_seconds,
            },
            'performance_metrics': {
                'avg_latency_ms': avg_latency,
                'p95_latency_ms': p95_latency,
                'tensor_serialization_time': self.stats.tensor_serialization_time,
                'compression_time': self.stats.compression_time,
            },
            'configuration': {
                'port': self.port,
                'arm64_optimized': self.enable_arm64_optimizations,
                'compression_enabled': self.compression_handler is not None,
                'tensor_pool_size': self.tensor_pool_size,
                'device': str(self.device),
            },
            'cache_stats': {
                'tensor_cache_size': len(self.tensor_cache),
                'tensor_pool_size': len(self.tensor_pool),
            },
            'topic_stats': dict(self.topic_stats) if hasattr(self, 'topic_stats') else {}
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get publisher performance statistics (legacy method)."""
        return self.get_statistics()
    
    def reset_stats(self):
        """Reset performance statistics."""
        self.stats = MessageStats()
        self.topic_stats.clear()
        self.latency_buffer.clear()
    
    def publish_tensor_data(self, symbol: str, tensor_data: Dict[str, torch.Tensor], 
                           metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Publish tensor data with ARM64 optimization"""
        start_time = time.perf_counter()
        
        try:
            # Convert tensors to serializable format
            serialized_tensors = {}
            tensor_start = time.perf_counter()
            
            for key, tensor in tensor_data.items():
                if isinstance(tensor, torch.Tensor):
                    serialized_tensors[key] = self._serialize_tensor(tensor)
                else:
                    serialized_tensors[key] = tensor
            
            tensor_time = time.perf_counter() - tensor_start
            self.stats.tensor_serialization_time += tensor_time
            self.stats.tensor_operations += len(tensor_data)
            
            # Create message
            message = {
                'symbol': symbol,
                'timestamp': int(time.time() * 1e9),
                'tensors': serialized_tensors,
                'metadata': metadata or {},
                'arm64_optimized': self.enable_arm64_optimizations
            }
            
            # Serialize and compress
            compression_start = time.perf_counter()
            packed_data = msgpack.packb(message)
            
            topic = f"tensor_data.{symbol}"
            
            # Compress large messages
            if self.compression_handler and len(packed_data) > self.compression_threshold:
                packed_data = self.compression_handler.compress(packed_data)
                topic = f"tensor_data_compressed.{symbol}"
            
            compression_time = time.perf_counter() - compression_start
            self.stats.compression_time += compression_time
            
            # Publish message
            self.socket.send_multipart([
                topic.encode('utf-8'),
                packed_data
            ])
            
            # Update statistics
            total_time = time.perf_counter() - start_time
            self.stats.messages_sent += 1
            self.stats.bytes_sent += len(packed_data)
            self.latency_buffer.append(total_time * 1000)  # Convert to ms
            
            logger.debug(f"Published tensor data for {symbol} in {total_time*1000:.2f}ms")
            return True
            
        except Exception as e:
            self.stats.errors += 1
            logger.error(f"Error publishing tensor data for {symbol}: {e}")
            return False
    
    def _serialize_tensor(self, tensor: torch.Tensor) -> Dict[str, Any]:
        """Serialize PyTorch tensor with ARM64 optimization"""
        try:
            # Move to CPU if on GPU for serialization
            if tensor.is_cuda:
                cpu_tensor = tensor.cpu()
            else:
                cpu_tensor = tensor
            
            # Convert to numpy for serialization
            numpy_array = cpu_tensor.numpy()
            
            # Generate tensor template ID for caching
            template_id = f"{numpy_array.shape}_{numpy_array.dtype}"
            
            # ARM64-specific optimizations
            if self.enable_arm64_optimizations:
                # Use more efficient serialization for common types
                if numpy_array.dtype in [np.float32, np.float16]:
                    # Optimize for ARM64 SIMD operations
                    data_bytes = numpy_array.ascontiguousarray().tobytes()
                else:
                    data_bytes = numpy_array.tobytes()
            else:
                data_bytes = numpy_array.tobytes()
            
            # Cache tensor template if enabled
            if template_id not in self.tensor_cache and len(self.tensor_cache) < self.tensor_pool_size:
                self.tensor_cache[template_id] = {
                    'shape': numpy_array.shape,
                    'dtype': str(numpy_array.dtype)
                }
            
            return {
                '__tensor__': True,
                'data': data_bytes,
                'shape': numpy_array.shape,
                'dtype': str(numpy_array.dtype),
                'device': 'cuda' if tensor.is_cuda else 'cpu',
                'template_id': template_id,
                'arm64_optimized': self.enable_arm64_optimizations
            }
            
        except Exception as e:
            logger.error(f"Tensor serialization error: {e}")
            raise
    
    def close(self):
        """Clean shutdown of the publisher socket and context."""
        # Log final statistics
        if hasattr(self, 'stats'):
            logger.info(f"ZMQ Publisher closing. Final stats: Messages sent: {self.stats.messages_sent}, "
                       f"Tensor operations: {self.stats.tensor_operations}, "
                       f"ARM64 optimized: {self.enable_arm64_optimizations}")
        
        # Clear tensor cache
        if hasattr(self, 'tensor_cache'):
            self.tensor_cache.clear()
        if hasattr(self, 'tensor_pool'):
            self.tensor_pool.clear()
        
        self.socket.close()
        self.context.term()
        logger.info("ZMQ Publisher closed successfully")

if __name__ == "__main__":
    # Example Usage
    publisher = ZMQPublisher(
        port=5555,
        enable_arm64_optimizations=True,
        tensor_pool_size=1000
    )

    # Example Market Data
    market_data_aapl = {
        "symbol": "AAPL",
        "timestamp": int(time.time() * 1e9),
        "price": 175.50,
        "volume": 1500,
        "bid": 175.45,
        "ask": 175.55
    }
    publisher.publish_market_data(market_data_aapl)

    # Example Trading Signal
    trading_signal_msft = {
        "symbol": "MSFT",
        "timestamp": int(time.time() * 1e9),
        "signal": 0.85,
        "confidence": 0.92,
        "strategy": "deep_momentum_lstm",
        "features": {"rsi": 70.1, "macd_hist": 0.5}
    }
    publisher.publish_trading_signal(trading_signal_msft)

    # Example tensor data publishing
    tensor_data = {
        'features': torch.randn(50, 20),
        'predictions': torch.randn(1, 3)
    }
    publisher.publish_tensor_data("TSLA", tensor_data, {'model': 'lstm_v1'})
    
    # Display performance stats
    stats = publisher.get_statistics()
    logger.info(f"Publisher statistics: {stats}")
    
    # Keep publisher alive for a short period to allow messages to be sent
    time.sleep(1) 
    publisher.close()
