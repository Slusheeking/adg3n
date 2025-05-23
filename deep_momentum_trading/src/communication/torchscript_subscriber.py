# TorchScript Model Subscriber for ARM64 Optimization
# Implements high-performance tensor deserialization for model inference
# Optimized for NVIDIA GH200 ARM64 platform

import zmq
import torch
import numpy as np
import time
import threading
import platform
import msgpack
import lz4.frame
from typing import Dict, List, Any, Optional, Callable
from collections import deque
from dataclasses import dataclass, field

from deep_momentum_trading.src.utils.logger import get_logger

logger = get_logger(__name__)

@dataclass
class TorchScriptSubscriberStats:
    """Statistics for TorchScript subscriber performance monitoring"""
    messages_received: int = 0
    bytes_received: int = 0
    tensor_deserialization_time: float = 0.0
    decompression_time: float = 0.0
    processing_time: float = 0.0
    cuda_transfer_time: float = 0.0
    errors: int = 0
    start_time: float = field(default_factory=time.time)
    
    @property
    def uptime_seconds(self) -> float:
        return time.time() - self.start_time
    
    @property
    def messages_per_second(self) -> float:
        uptime = self.uptime_seconds
        return self.messages_received / uptime if uptime > 0 else 0.0

class TorchScriptModelSubscriber:
    """ARM64-optimized subscriber for TorchScript model workers"""
    
    def __init__(self, publisher_port: int = 5556, model_id: str = "lstm_1",
                 enable_cuda_streams: bool = True, buffer_size: int = 10000):
        """
        Initialize TorchScript model subscriber.
        
        Args:
            publisher_port: Port to connect to for feature data
            model_id: Unique identifier for this model worker
            enable_cuda_streams: Enable CUDA streams for async processing
            buffer_size: Size of receive buffer
        """
        self.publisher_port = publisher_port
        self.model_id = model_id
        
        # ZeroMQ setup
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.SUB)
        
        # ARM64 optimization detection
        self.is_arm64 = platform.machine().lower() in ['arm64', 'aarch64']
        if self.is_arm64:
            logger.info("ARM64 architecture detected for TorchScript subscriber")
        
        # ARM64 socket optimizations
        if self.is_arm64:
            self.socket.setsockopt(zmq.RCVHWM, buffer_size * 2)
            self.socket.setsockopt(zmq.RCVBUF, 2 * 1024 * 1024)  # 2MB buffer
        else:
            self.socket.setsockopt(zmq.RCVHWM, buffer_size)
            self.socket.setsockopt(zmq.RCVBUF, 1024 * 1024)
        
        # Subscribe to feature topics
        self.socket.setsockopt(zmq.SUBSCRIBE, b"features.")
        self.socket.setsockopt(zmq.SUBSCRIBE, b"features_compressed.")
        self.socket.connect(f"tcp://localhost:{publisher_port}")
        
        # CUDA optimization setup
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.enable_cuda_streams = enable_cuda_streams and torch.cuda.is_available()
        
        if self.enable_cuda_streams:
            self.cuda_stream = torch.cuda.Stream()
            logger.info("CUDA streams enabled for async tensor processing")
        else:
            self.cuda_stream = None
        
        # Tensor cache and memory management
        self.tensor_cache = {}
        self.cache_size_limit = 1000
        
        # Message handlers
        self.feature_handlers: List[Callable] = []
        self.prediction_handlers: List[Callable] = []
        
        # Processing state
        self.is_running = False
        self.processing_thread: Optional[threading.Thread] = None
        
        # Performance monitoring
        self.stats = TorchScriptSubscriberStats()
        self.latency_buffer = deque(maxlen=1000)
        
        # Thread safety
        self.lock = threading.RLock()
        
        logger.info(f"TorchScript subscriber initialized for model {model_id}")
        logger.info(f"Connected to publisher on port {publisher_port}")
    
    def add_feature_handler(self, handler: Callable[[str, torch.Tensor, Dict], None]):
        """Add handler for feature updates"""
        self.feature_handlers.append(handler)
        logger.info(f"Added feature handler for model {self.model_id}")
    
    def add_prediction_handler(self, handler: Callable[[str, torch.Tensor, Dict], None]):
        """Add handler for prediction processing"""
        self.prediction_handlers.append(handler)
        logger.info(f"Added prediction handler for model {self.model_id}")
    
    def start(self):
        """Start message processing"""
        if self.is_running:
            logger.warning("TorchScript subscriber already running")
            return
        
        self.is_running = True
        self.processing_thread = threading.Thread(
            target=self._processing_loop, 
            daemon=True
        )
        self.processing_thread.start()
        
        logger.info(f"TorchScript subscriber started for model {self.model_id}")
    
    def _processing_loop(self):
        """Main processing loop for receiving and handling messages"""
        while self.is_running:
            try:
                start_time = time.perf_counter()
                
                # Receive message with timeout
                parts = self.socket.recv_multipart(zmq.NOBLOCK)
                
                if len(parts) >= 2:
                    topic = parts[0].decode('utf-8')
                    message_data = parts[1]
                    
                    # Handle compressed messages
                    is_compressed = "compressed" in topic
                    
                    # Decompress if needed
                    if is_compressed:
                        decompress_start = time.perf_counter()
                        message_data = lz4.frame.decompress(message_data)
                        self.stats.decompression_time += time.perf_counter() - decompress_start
                    
                    # Process the message
                    self._process_message(topic, message_data, start_time)
                
            except zmq.Again:
                # No message available, continue
                time.sleep(0.001)  # 1ms sleep
                continue
            except Exception as e:
                logger.error(f"Error in processing loop: {e}")
                self.stats.errors += 1
                time.sleep(0.01)
    
    def _process_message(self, topic: str, message_data: bytes, start_time: float):
        """Process individual message"""
        try:
            # Deserialize message
            message = msgpack.unpackb(message_data, raw=False)
            
            # Extract message components
            symbol = message.get('symbol', 'unknown')
            timestamp = message.get('timestamp', 0)
            
            processing_start = time.perf_counter()
            
            if topic.startswith('features'):
                self._handle_feature_message(symbol, message, timestamp)
            elif topic.startswith('predictions'):
                self._handle_prediction_message(symbol, message, timestamp)
            
            processing_time = time.perf_counter() - processing_start
            self.stats.processing_time += processing_time
            
            # Update statistics
            total_time = time.perf_counter() - start_time
            self.stats.messages_received += 1
            self.stats.bytes_received += len(message_data)
            self.latency_buffer.append(total_time * 1000)  # Convert to ms
            
            logger.debug(f"Processed {topic} for {symbol} in {total_time*1000:.2f}ms")
            
        except Exception as e:
            logger.error(f"Error processing message from topic {topic}: {e}")
            self.stats.errors += 1
    
    def _handle_feature_message(self, symbol: str, message: Dict[str, Any], timestamp: int):
        """Handle feature update messages"""
        try:
            # Deserialize tensor with CUDA stream optimization
            features_data = message.get('features')
            if not features_data:
                return
            
            tensor_start = time.perf_counter()
            
            if self.enable_cuda_streams:
                with torch.cuda.stream(self.cuda_stream):
                    features = self._deserialize_tensor(features_data)
                    if not features.is_cuda:
                        features = features.cuda(non_blocking=True)
            else:
                features = self._deserialize_tensor(features_data)
                if torch.cuda.is_available() and not features.is_cuda:
                    features = features.cuda()
            
            tensor_time = time.perf_counter() - tensor_start
            self.stats.tensor_deserialization_time += tensor_time
            
            # Prepare message data
            feature_data = {
                'symbol': symbol,
                'timestamp': timestamp,
                'features': features,
                'confidence': message.get('confidence', 1.0),
                'model_version': message.get('model_version', 'unknown'),
                'arm64_optimized': message.get('arm64_optimized', False)
            }
            
            # Call feature handlers
            for handler in self.feature_handlers:
                try:
                    handler(symbol, features, feature_data)
                except Exception as e:
                    logger.error(f"Error in feature handler: {e}")
            
        except Exception as e:
            logger.error(f"Error handling feature message for {symbol}: {e}")
            self.stats.errors += 1
    
    def _handle_prediction_message(self, symbol: str, message: Dict[str, Any], timestamp: int):
        """Handle prediction messages"""
        try:
            prediction_data = message.get('prediction')
            if not prediction_data:
                return
            
            # Deserialize prediction tensor
            prediction = self._deserialize_tensor(prediction_data)
            
            if torch.cuda.is_available() and not prediction.is_cuda:
                if self.enable_cuda_streams:
                    with torch.cuda.stream(self.cuda_stream):
                        prediction = prediction.cuda(non_blocking=True)
                else:
                    prediction = prediction.cuda()
            
            # Prepare prediction data
            pred_data = {
                'symbol': symbol,
                'timestamp': timestamp,
                'prediction': prediction,
                'model_id': message.get('model_id', 'unknown'),
                'confidence': message.get('confidence', 1.0),
                'arm64_optimized': message.get('arm64_optimized', False)
            }
            
            # Call prediction handlers
            for handler in self.prediction_handlers:
                try:
                    handler(symbol, prediction, pred_data)
                except Exception as e:
                    logger.error(f"Error in prediction handler: {e}")
            
        except Exception as e:
            logger.error(f"Error handling prediction message for {symbol}: {e}")
            self.stats.errors += 1
    
    def _deserialize_tensor(self, tensor_data: bytes) -> torch.Tensor:
        """Efficient tensor deserialization for ARM64"""
        try:
            # Unpack tensor metadata
            tensor_info = msgpack.unpackb(tensor_data, raw=False)
            
            # Extract tensor components
            data_bytes = tensor_info['data']
            shape = tuple(tensor_info['shape'])
            dtype_str = tensor_info['dtype']
            original_device = tensor_info.get('device', 'cpu')
            tensor_id = tensor_info.get('tensor_id', None)
            
            # Check cache first
            if tensor_id and tensor_id in self.tensor_cache:
                cached_tensor = self.tensor_cache[tensor_id]
                if cached_tensor.shape == shape:
                    return cached_tensor.clone()
            
            # Reconstruct numpy array
            numpy_dtype = getattr(np, dtype_str.split('.')[-1])
            numpy_array = np.frombuffer(data_bytes, dtype=numpy_dtype).reshape(shape)
            
            # Convert to PyTorch tensor
            tensor = torch.from_numpy(numpy_array.copy())  # Copy to avoid memory issues
            
            # Cache tensor if it has an ID
            if tensor_id and len(self.tensor_cache) < self.cache_size_limit:
                self.tensor_cache[tensor_id] = tensor.clone()
            
            return tensor
            
        except Exception as e:
            logger.error(f"Tensor deserialization error: {e}")
            raise
    
    def receive_features(self) -> Optional[Dict[str, Any]]:
        """Receive and deserialize features for TorchScript processing (polling interface)"""
        try:
            # Non-blocking receive
            parts = self.socket.recv_multipart(zmq.NOBLOCK)
            
            if len(parts) >= 2:
                topic = parts[0].decode('utf-8')
                message_data = parts[1]
                
                # Handle compression
                if "compressed" in topic:
                    message_data = lz4.frame.decompress(message_data)
                
                # Unpack message
                message = msgpack.unpackb(message_data, raw=False)
                
                # Deserialize tensor with CUDA stream
                features_data = message.get('features')
                if features_data:
                    if self.enable_cuda_streams:
                        with torch.cuda.stream(self.cuda_stream):
                            features = self._deserialize_tensor(features_data)
                            if not features.is_cuda:
                                features = features.cuda(non_blocking=True)
                    else:
                        features = self._deserialize_tensor(features_data)
                        if torch.cuda.is_available() and not features.is_cuda:
                            features = features.cuda()
                    
                    return {
                        'symbol': message['symbol'],
                        'timestamp': message['timestamp'],
                        'features': features,
                        'confidence': message['confidence'],
                        'model_version': message.get('model_version', 'unknown')
                    }
            
            return None
            
        except zmq.Again:
            return None
        except Exception as e:
            logger.error(f"Error receiving features: {e}")
            return None
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive subscriber statistics"""
        
        avg_latency = sum(self.latency_buffer) / len(self.latency_buffer) if self.latency_buffer else 0
        p95_latency = sorted(self.latency_buffer)[int(len(self.latency_buffer) * 0.95)] if self.latency_buffer else 0
        
        return {
            'subscriber_stats': {
                'messages_received': self.stats.messages_received,
                'bytes_received': self.stats.bytes_received,
                'messages_per_second': self.stats.messages_per_second,
                'errors': self.stats.errors,
                'uptime_seconds': self.stats.uptime_seconds,
            },
            'performance_metrics': {
                'avg_latency_ms': avg_latency,
                'p95_latency_ms': p95_latency,
                'tensor_deserialization_time': self.stats.tensor_deserialization_time,
                'decompression_time': self.stats.decompression_time,
                'processing_time': self.stats.processing_time,
                'cuda_transfer_time': self.stats.cuda_transfer_time,
            },
            'configuration': {
                'model_id': self.model_id,
                'publisher_port': self.publisher_port,
                'arm64_optimized': self.is_arm64,
                'cuda_streams_enabled': self.enable_cuda_streams,
                'device': str(self.device),
                'cache_size': len(self.tensor_cache),
            }
        }
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get health status of the subscriber"""
        
        error_rate = self.stats.errors / max(self.stats.messages_received, 1)
        
        return {
            'healthy': error_rate < 0.01 and self.is_running,
            'is_running': self.is_running,
            'error_rate': error_rate,
            'messages_per_second': self.stats.messages_per_second,
            'cuda_available': torch.cuda.is_available(),
            'cache_utilization': len(self.tensor_cache) / self.cache_size_limit,
        }
    
    def stop(self):
        """Stop message processing"""
        if not self.is_running:
            logger.warning("TorchScript subscriber not running")
            return
        
        self.is_running = False
        
        # Wait for processing thread to finish
        if self.processing_thread and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=5.0)
            if self.processing_thread.is_alive():
                logger.warning("Processing thread did not terminate gracefully")
        
        # Clear tensor cache
        self.tensor_cache.clear()
        
        # Log final statistics
        stats = self.get_statistics()
        logger.info(f"TorchScript subscriber stopped. Final stats: {stats['subscriber_stats']}")
    
    def close(self):
        """Clean shutdown of subscriber"""
        try:
            self.stop()
            
            # Close socket and context
            self.socket.close()
            self.context.term()
            
            logger.info(f"TorchScript subscriber closed for model {self.model_id}")
            
        except Exception as e:
            logger.error(f"Error closing TorchScript subscriber: {e}")

if __name__ == "__main__":
    # Example usage of TorchScript subscriber
    
    def feature_handler(symbol: str, features: torch.Tensor, data: Dict[str, Any]):
        """Example feature handler"""
        logger.info(f"Received features for {symbol}: shape={features.shape}, "
                   f"confidence={data['confidence']:.3f}")
        
        # Simulate model inference
        time.sleep(0.001)  # 1ms processing time
        
        # Example prediction
        prediction = torch.randn(1, 3)  # [buy, hold, sell] probabilities
        logger.info(f"Generated prediction for {symbol}: {prediction}")
    
    def prediction_handler(symbol: str, prediction: torch.Tensor, data: Dict[str, Any]):
        """Example prediction handler"""
        logger.info(f"Received prediction for {symbol}: {prediction}, "
                   f"model={data['model_id']}, confidence={data['confidence']:.3f}")
    
    # Create subscriber
    subscriber = TorchScriptModelSubscriber(
        publisher_port=5555,
        model_id="test_lstm_model",
        enable_cuda_streams=True
    )
    
    # Add handlers
    subscriber.add_feature_handler(feature_handler)
    subscriber.add_prediction_handler(prediction_handler)
    
    # Start processing
    subscriber.start()
    
    try:
        logger.info("TorchScript subscriber running...")
        logger.info("Waiting for feature messages from publisher...")
        
        # Run for a while to receive messages
        start_time = time.time()
        while time.time() - start_time < 30:  # Run for 30 seconds
            time.sleep(1)
            
            # Log statistics periodically
            if int(time.time() - start_time) % 10 == 0:
                stats = subscriber.get_statistics()
                logger.info(f"Stats: {stats['subscriber_stats']['messages_received']} messages, "
                           f"{stats['subscriber_stats']['messages_per_second']:.1f} msg/s")
        
    except KeyboardInterrupt:
        logger.info("Shutting down subscriber...")
    finally:
        subscriber.close()