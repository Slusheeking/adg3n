import zmq
import threading
import time
import logging
import platform
from collections import defaultdict, deque
from typing import Dict, Set, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime
import json

from deep_momentum_trading.src.utils.logger import get_logger

logger = get_logger(__name__)

@dataclass
class BrokerStats:
    """Statistics for message broker performance monitoring"""
    messages_routed: int = 0
    frontend_to_backend: int = 0
    backend_to_frontend: int = 0
    bytes_transferred: int = 0
    routing_errors: int = 0
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
        return self.messages_routed / uptime if uptime > 0 else 0.0

class WorkerMonitor:
    """Monitor worker performance and load balancing"""
    
    def __init__(self, window_size: int = 1000):
        self.worker_stats = defaultdict(lambda: {
            'messages_processed': 0,
            'last_seen': time.time(),
            'response_times': deque(maxlen=window_size),
            'errors': 0
        })
        self.load_balancing_queue = deque()
        
    def record_worker_activity(self, worker_id: str, response_time: float = None):
        """Record worker activity for load balancing"""
        stats = self.worker_stats[worker_id]
        stats['messages_processed'] += 1
        stats['last_seen'] = time.time()
        
        if response_time is not None:
            stats['response_times'].append(response_time)
    
    def get_worker_stats(self) -> Dict[str, Any]:
        """Get comprehensive worker statistics"""
        current_time = time.time()
        active_workers = []
        
        for worker_id, stats in self.worker_stats.items():
            if current_time - stats['last_seen'] < 60:  # Active within last minute
                avg_response_time = (
                    sum(stats['response_times']) / len(stats['response_times'])
                    if stats['response_times'] else 0
                )
                active_workers.append({
                    'worker_id': worker_id,
                    'messages_processed': stats['messages_processed'],
                    'avg_response_time': avg_response_time,
                    'last_seen': stats['last_seen'],
                    'errors': stats['errors']
                })
        
        return {
            'active_workers': len(active_workers),
            'total_workers': len(self.worker_stats),
            'worker_details': active_workers
        }

class MessageBroker:
    """Enhanced central message broker with ARM64 optimizations and monitoring"""
    
    def __init__(self,
                 frontend_port: int = 5555,
                 backend_port: int = 5556,
                 enable_monitoring: bool = True,
                 enable_arm64_optimizations: bool = True,
                 buffer_size: int = 10000):
        """
        Initialize enhanced message broker.
        
        Args:
            frontend_port: Port for client connections
            backend_port: Port for worker connections
            enable_monitoring: Whether to enable performance monitoring
            enable_arm64_optimizations: Whether to enable ARM64 optimizations
            buffer_size: Size of message buffers
        """
        self.context = zmq.Context()
        
        # ARM64 optimization detection
        self.is_arm64 = platform.machine().lower() in ['arm64', 'aarch64']
        self.enable_arm64_optimizations = enable_arm64_optimizations and self.is_arm64
        
        if self.enable_arm64_optimizations:
            logger.info("ARM64 optimizations enabled for message broker")
        
        # Frontend socket (clients connect here)
        self.frontend = self.context.socket(zmq.ROUTER)
        if self.enable_arm64_optimizations:
            self.frontend.setsockopt(zmq.SNDHWM, buffer_size * 2)
            self.frontend.setsockopt(zmq.RCVHWM, buffer_size * 2)
            self.frontend.setsockopt(zmq.SNDBUF, 1024 * 1024)  # 1MB send buffer
            self.frontend.setsockopt(zmq.RCVBUF, 1024 * 1024)  # 1MB receive buffer
        else:
            self.frontend.setsockopt(zmq.SNDHWM, buffer_size)
            self.frontend.setsockopt(zmq.RCVHWM, buffer_size)
        
        self.frontend.bind(f"tcp://*:{frontend_port}")
        
        # Backend socket (workers connect here)
        self.backend = self.context.socket(zmq.DEALER)
        if self.enable_arm64_optimizations:
            self.backend.setsockopt(zmq.SNDHWM, buffer_size * 2)
            self.backend.setsockopt(zmq.RCVHWM, buffer_size * 2)
            self.backend.setsockopt(zmq.SNDBUF, 1024 * 1024)
            self.backend.setsockopt(zmq.RCVBUF, 1024 * 1024)
        else:
            self.backend.setsockopt(zmq.SNDHWM, buffer_size)
            self.backend.setsockopt(zmq.RCVHWM, buffer_size)
        
        self.backend.bind(f"tcp://*:{backend_port}")
        
        # State management
        self.is_running = False
        self.broker_thread: Optional[threading.Thread] = None
        
        # Monitoring and statistics
        self.enable_monitoring = enable_monitoring
        self.stats = BrokerStats() if enable_monitoring else None
        self.worker_monitor = WorkerMonitor() if enable_monitoring else None
        
        # Performance optimization
        self.batch_size = 100 if self.enable_arm64_optimizations else 50
        self.message_queue = deque(maxlen=buffer_size)
        
        logger.info(f"Enhanced MessageBroker initialized with frontend:{frontend_port}, "
                   f"backend:{backend_port}, ARM64={self.enable_arm64_optimizations}")
        
    def start(self):
        """Start enhanced message broker with ARM64 optimizations."""
        if self.is_running:
            logger.warning("Message broker is already running.")
            return
        
        self.is_running = True
        self.broker_thread = threading.Thread(target=self._broker_loop, daemon=True)
        self.broker_thread.start()
        
        logger.info("Enhanced message broker started with ARM64 optimizations.")
    
    def _broker_loop(self):
        """Enhanced broker loop with ARM64 optimizations and monitoring."""
        poller = zmq.Poller()
        poller.register(self.frontend, zmq.POLLIN)
        poller.register(self.backend, zmq.POLLIN)
        
        # ARM64 batch processing
        message_batch = []
        last_batch_time = time.time()
        
        while self.is_running:
            try:
                start_time = time.time()
                sockets = dict(poller.poll(100))  # Shorter timeout for ARM64
                
                # Process frontend to backend messages
                if self.frontend in sockets and sockets[self.frontend] == zmq.POLLIN:
                    message = self.frontend.recv_multipart()
                    message_size = sum(len(part) for part in message)
                    
                    if self.enable_arm64_optimizations:
                        # Batch processing for ARM64
                        message_batch.append(('frontend_to_backend', message, message_size, start_time))
                        
                        # Process batch when full or timeout reached
                        current_time = time.time()
                        if (len(message_batch) >= self.batch_size or
                            current_time - last_batch_time > 0.05):  # 50ms timeout
                            self._process_message_batch(message_batch)
                            message_batch.clear()
                            last_batch_time = current_time
                    else:
                        # Direct processing
                        self.backend.send_multipart(message)
                        self._update_stats('frontend_to_backend', message_size, start_time)
                
                # Process backend to frontend messages
                if self.backend in sockets and sockets[self.backend] == zmq.POLLIN:
                    message = self.backend.recv_multipart()
                    message_size = sum(len(part) for part in message)
                    
                    if self.enable_arm64_optimizations:
                        message_batch.append(('backend_to_frontend', message, message_size, start_time))
                        
                        current_time = time.time()
                        if (len(message_batch) >= self.batch_size or
                            current_time - last_batch_time > 0.05):
                            self._process_message_batch(message_batch)
                            message_batch.clear()
                            last_batch_time = current_time
                    else:
                        self.frontend.send_multipart(message)
                        self._update_stats('backend_to_frontend', message_size, start_time)
                
                # Process any remaining batch messages on timeout
                if message_batch and time.time() - last_batch_time > 0.05:
                    self._process_message_batch(message_batch)
                    message_batch.clear()
                    last_batch_time = time.time()
                    
            except zmq.Again:
                # Process any remaining batch messages
                if message_batch:
                    self._process_message_batch(message_batch)
                    message_batch.clear()
                    last_batch_time = time.time()
                continue
                
            except Exception as e:
                logger.error(f"Broker error in main loop: {e}", exc_info=True)
                if self.enable_monitoring and self.stats:
                    self.stats.routing_errors += 1
                time.sleep(0.01)
        
        # Process any remaining messages when stopping
        if message_batch:
            self._process_message_batch(message_batch)
    
    def _process_message_batch(self, message_batch):
        """Process a batch of messages for ARM64 optimization"""
        try:
            for direction, message, message_size, start_time in message_batch:
                if direction == 'frontend_to_backend':
                    self.backend.send_multipart(message)
                else:  # backend_to_frontend
                    self.frontend.send_multipart(message)
                
                self._update_stats(direction, message_size, start_time)
                
        except Exception as e:
            logger.error(f"Error processing message batch: {e}")
    
    def _update_stats(self, direction: str, message_size: int, start_time: float):
        """Update broker statistics"""
        if not self.enable_monitoring or not self.stats:
            return
        
        self.stats.messages_routed += 1
        self.stats.bytes_transferred += message_size
        
        if direction == 'frontend_to_backend':
            self.stats.frontend_to_backend += 1
        else:
            self.stats.backend_to_frontend += 1
        
        # Log periodic statistics
        if self.stats.messages_routed % 1000 == 0:
            logger.debug(f"Broker stats: {self.stats.messages_routed} messages routed, "
                        f"{self.stats.messages_per_second:.1f} msg/s")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive broker statistics"""
        if not self.enable_monitoring:
            return {'monitoring_disabled': True}
        
        stats = {
            'broker_stats': {
                'messages_routed': self.stats.messages_routed,
                'frontend_to_backend': self.stats.frontend_to_backend,
                'backend_to_frontend': self.stats.backend_to_frontend,
                'bytes_transferred': self.stats.bytes_transferred,
                'routing_errors': self.stats.routing_errors,
                'uptime_seconds': self.stats.uptime_seconds,
                'messages_per_second': self.stats.messages_per_second,
            },
            'configuration': {
                'arm64_optimizations': self.enable_arm64_optimizations,
                'batch_size': self.batch_size,
                'monitoring_enabled': self.enable_monitoring,
            }
        }
        
        if self.worker_monitor:
            stats['worker_stats'] = self.worker_monitor.get_worker_stats()
        
        return stats
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get health status of the broker"""
        return {
            'is_running': self.is_running,
            'thread_alive': self.broker_thread.is_alive() if self.broker_thread else False,
            'error_rate': (self.stats.routing_errors / max(self.stats.messages_routed, 1)
                          if self.stats else 0),
            'uptime_seconds': self.stats.uptime_seconds if self.stats else 0,
            'messages_per_second': self.stats.messages_per_second if self.stats else 0,
        }
    
    def stop(self):
        """Gracefully stops the enhanced message broker."""
        if not self.is_running:
            logger.warning("Message broker is not running.")
            return
        
        self.is_running = False
        
        # Wait for broker thread to finish
        if self.broker_thread and self.broker_thread.is_alive():
            self.broker_thread.join(timeout=5.0)
            if self.broker_thread.is_alive():
                logger.warning("Broker thread did not terminate gracefully.")
        
        # Close sockets and context
        self.frontend.close()
        self.backend.close()
        self.context.term()
        
        # Log final statistics
        if self.enable_monitoring and self.stats:
            logger.info(f"Enhanced message broker stopped. Statistics: "
                       f"Messages: {self.stats.messages_routed}, "
                       f"Bytes: {self.stats.bytes_transferred}, "
                       f"Errors: {self.stats.routing_errors}, "
                       f"Uptime: {self.stats.uptime_seconds:.1f}s, "
                       f"Rate: {self.stats.messages_per_second:.1f} msg/s")
        else:
            logger.info("Enhanced message broker stopped.")

if __name__ == "__main__":
    # Enhanced Example Usage with ARM64 optimizations and monitoring
    # This demonstrates the enhanced message broker capabilities
    
    # Create enhanced broker with ARM64 optimizations
    broker = MessageBroker(
        frontend_port=5555,
        backend_port=5556,
        enable_monitoring=True,
        enable_arm64_optimizations=True,
        buffer_size=20000  # Larger buffer for high-throughput
    )
    
    broker.start()

    try:
        logger.info("Enhanced Message Broker running with ARM64 optimizations...")
        logger.info("Monitoring enabled - statistics will be logged periodically")
        
        # Periodically log statistics
        start_time = time.time()
        while True:
            time.sleep(10)  # Log stats every 10 seconds
            
            if time.time() - start_time > 10:
                stats = broker.get_statistics()
                health = broker.get_health_status()
                
                logger.info(f"Broker Stats: {stats['broker_stats']['messages_routed']} messages, "
                           f"{stats['broker_stats']['messages_per_second']:.1f} msg/s, "
                           f"{stats['broker_stats']['routing_errors']} errors")
                logger.info(f"Health: Running={health['is_running']}, "
                           f"Error Rate={health['error_rate']:.4f}")
                
                start_time = time.time()
                
    except KeyboardInterrupt:
        logger.info("Stopping enhanced message broker...")
    finally:
        # Get final statistics before stopping
        final_stats = broker.get_statistics()
        logger.info(f"Final Broker Statistics: {final_stats}")
        broker.stop()

    # To test with enhanced clients and workers:
    # 1. Use the enhanced ZMQPublisher/ZMQSubscriber with compression
    # 2. Connect DEALER workers to port 5556 with ARM64 optimizations
    # 3. Connect ROUTER clients to port 5555 with monitoring enabled
    # 4. The broker will automatically handle load balancing and performance optimization
