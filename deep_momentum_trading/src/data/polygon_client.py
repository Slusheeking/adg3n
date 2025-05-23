import os
import asyncio
import websockets
import json
import httpx
import time
import platform
import psutil
import pandas as pd
import numpy as np
import requests
from typing import List, Dict, Any, Callable, Optional, Union, Tuple, Set
from dataclasses import dataclass, asdict, field
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor
import threading
from collections import deque, defaultdict, OrderedDict
import warnings
import hashlib
import gzip
import lz4.frame
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import struct

from deep_momentum_trading.src.utils.logger import get_logger
from deep_momentum_trading.config.settings import config_manager

# Try to import GPU memory management for GH200
try:
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False
    cp = None

# Try to import xxhash for faster ARM64 hashing
try:
    import xxhash
    XXHASH_AVAILABLE = True
except ImportError:
    XXHASH_AVAILABLE = False

# Rate limiting utility
class RateLimiter:
    """Simple rate limiter for API calls."""
    
    def __init__(self, calls: int, period: int):
        self.calls = calls
        self.period = period
        self.call_times = deque(maxlen=calls)
        self.lock = threading.Lock()
    
    async def acquire(self):
        """Acquire permission to make a call."""
        current_time = time.time()
        
        with self.lock:
            # Remove old calls outside the period
            while self.call_times and current_time - self.call_times[0] > self.period:
                self.call_times.popleft()
            
            # If we're at the limit, wait
            if len(self.call_times) >= self.calls:
                sleep_time = self.period - (current_time - self.call_times[0])
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)
            
            self.call_times.append(current_time)

logger = get_logger(__name__)

@dataclass
class PolygonConfig:
    """Enhanced configuration for Polygon client with ARM64 optimizations and high-frequency support."""
    api_key: Optional[str] = None
    enable_second_data: bool = True
    enable_subsecond_data: bool = True
    enable_arm64_optimizations: bool = True
    enable_performance_monitoring: bool = True
    max_reconnect_attempts: int = 10
    reconnect_delay: float = 1.0
    rate_limit_per_minute: int = 5000  # Polygon rate limits
    buffer_size: int = 10000
    enable_data_validation: bool = True
    enable_latency_tracking: bool = True
    websocket_timeout: float = 30.0
    rest_timeout: float = 30.0
    enable_compression: bool = True
    compression_algorithm: str = 'lz4'  # 'gzip', 'lz4', 'none'
    enable_batching: bool = True
    batch_size: int = 100
    enable_caching: bool = True
    cache_ttl_seconds: int = 300
    enable_parallel_processing: bool = True
    max_workers: int = 8
    memory_alignment_bytes: int = 64  # ARM64 cache line size
    enable_numa_awareness: bool = True
    enable_prefetching: bool = True
    prefetch_size: int = 16
    enable_message_deduplication: bool = True
    deduplication_window_seconds: int = 60
    enable_adaptive_buffering: bool = True
    min_buffer_size: int = 1000
    max_buffer_size: int = 100000
    enable_binary_parsing: bool = True
    enable_unified_memory: bool = True
    total_memory_gb: int = 600  # GH200 memory capacity
    enable_market_wide_subscription: bool = True
    enable_memory_pressure_monitoring: bool = True
    memory_pressure_threshold: float = 85.0  # Percentage

class PolygonStats:
    """Enhanced comprehensive statistics tracking for Polygon client operations with ARM64 optimizations."""
    
    def __init__(self):
        self.start_time = time.time()
        self.lock = threading.Lock()
        
        # Request statistics
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0
        self.rate_limited_requests = 0
        
        # Data statistics
        self.total_trades_received = 0
        self.total_quotes_received = 0
        self.total_bars_generated = 0
        self.duplicate_messages_filtered = 0
        self.messages_received = 0
        self.messages_processed = 0
        self.messages_failed = 0
        self.bytes_received = 0
        self.bytes_processed = 0
        
        # WebSocket statistics
        self.ws_connections_established = 0
        self.ws_connections_failed = 0
        self.ws_reconnections = 0
        self.ws_messages_received = 0
        self.ws_messages_processed = 0
        
        # Performance statistics
        self.avg_processing_time_ns = 0
        self.max_processing_time_ns = 0
        self.min_processing_time_ns = float('inf')
        self.processing_times = deque(maxlen=1000)
        
        # Buffer statistics
        self.active_buffers = 0
        self.total_buffer_operations = 0
        self.buffer_resize_operations = 0
        
        # ARM64 specific statistics
        self.arm64_optimizations_used = 0
        self.numa_operations = 0
        self.simd_operations = 0
        self.cache_hits = 0
        self.cache_misses = 0
        self.compression_ratio = 0.0
        self.prefetch_operations = 0
        self.adaptive_buffer_adjustments = 0
    
    def record_request(self, success: bool, rate_limited: bool = False):
        """Record API request statistics."""
        with self.lock:
            self.total_requests += 1
            if success:
                self.successful_requests += 1
            else:
                self.failed_requests += 1
            if rate_limited:
                self.rate_limited_requests += 1
    
    def record_data_received(self, data_type: str, count: int = 1):
        """Record data reception statistics."""
        with self.lock:
            if data_type == "trade":
                self.total_trades_received += count
            elif data_type == "quote":
                self.total_quotes_received += count
            elif data_type == "bar":
                self.total_bars_generated += count
            
            self.messages_received += count
            self.messages_processed += count
    
    def record_websocket_event(self, event_type: str):
        """Record WebSocket event statistics."""
        with self.lock:
            if event_type == "connection_established":
                self.ws_connections_established += 1
            elif event_type == "connection_failed":
                self.ws_connections_failed += 1
            elif event_type == "reconnection":
                self.ws_reconnections += 1
            elif event_type == "message_received":
                self.ws_messages_received += 1
            elif event_type == "message_processed":
                self.ws_messages_processed += 1
    
    def record_processing_time(self, processing_time_ns: int):
        """Record message processing time."""
        with self.lock:
            self.processing_times.append(processing_time_ns)
            self.max_processing_time_ns = max(self.max_processing_time_ns, processing_time_ns)
            self.min_processing_time_ns = min(self.min_processing_time_ns, processing_time_ns)
            
            # Update running average
            if self.processing_times:
                self.avg_processing_time_ns = sum(self.processing_times) / len(self.processing_times)
    
    def record_duplicate_filtered(self):
        """Record duplicate message filtering."""
        with self.lock:
            self.duplicate_messages_filtered += 1
    
    def record_buffer_operation(self, operation_type: str):
        """Record buffer operation statistics."""
        with self.lock:
            self.total_buffer_operations += 1
            if operation_type == "resize":
                self.buffer_resize_operations += 1
            elif operation_type == "adaptive_adjustment":
                self.adaptive_buffer_adjustments += 1
    
    def record_arm64_operation(self, operation_type: str):
        """Record ARM64 specific operations."""
        with self.lock:
            self.arm64_optimizations_used += 1
            if operation_type == "numa":
                self.numa_operations += 1
            elif operation_type == "simd":
                self.simd_operations += 1
            elif operation_type == "prefetch":
                self.prefetch_operations += 1
    
    def record_cache_operation(self, hit: bool):
        """Record cache hit/miss statistics."""
        with self.lock:
            if hit:
                self.cache_hits += 1
            else:
                self.cache_misses += 1
    
    def update_compression_ratio(self, original_size: int, compressed_size: int):
        """Update compression ratio statistics."""
        with self.lock:
            if original_size > 0:
                ratio = compressed_size / original_size
                # Running average of compression ratio
                if self.compression_ratio == 0.0:
                    self.compression_ratio = ratio
                else:
                    self.compression_ratio = (self.compression_ratio + ratio) / 2
    
    @property
    def uptime_seconds(self) -> float:
        return time.time() - self.start_time
    
    @property
    def messages_per_second(self) -> float:
        return self.messages_processed / max(self.uptime_seconds, 1)
    
    @property
    def success_rate(self) -> float:
        total = self.messages_received
        return (self.messages_processed / max(total, 1)) if total > 0 else 0.0
    
    @property
    def cache_hit_rate(self) -> float:
        total = self.cache_hits + self.cache_misses
        return (self.cache_hits / max(total, 1)) if total > 0 else 0.0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics."""
        with self.lock:
            uptime = self.uptime_seconds
            
            return {
                "uptime_seconds": uptime,
                "requests": {
                    "total": self.total_requests,
                    "successful": self.successful_requests,
                    "failed": self.failed_requests,
                    "rate_limited": self.rate_limited_requests,
                    "success_rate": self.successful_requests / max(self.total_requests, 1),
                    "requests_per_second": self.total_requests / max(uptime, 1)
                },
                "data": {
                    "trades_received": self.total_trades_received,
                    "quotes_received": self.total_quotes_received,
                    "bars_generated": self.total_bars_generated,
                    "duplicates_filtered": self.duplicate_messages_filtered,
                    "messages_received": self.messages_received,
                    "messages_processed": self.messages_processed,
                    "messages_failed": self.messages_failed,
                    "bytes_received": self.bytes_received,
                    "bytes_processed": self.bytes_processed,
                    "trades_per_second": self.total_trades_received / max(uptime, 1),
                    "quotes_per_second": self.total_quotes_received / max(uptime, 1),
                    "success_rate": self.success_rate
                },
                "websocket": {
                    "connections_established": self.ws_connections_established,
                    "connections_failed": self.ws_connections_failed,
                    "reconnections": self.ws_reconnections,
                    "messages_received": self.ws_messages_received,
                    "messages_processed": self.ws_messages_processed,
                    "processing_rate": self.ws_messages_processed / max(self.ws_messages_received, 1)
                },
                "performance": {
                    "avg_processing_time_ms": self.avg_processing_time_ns / 1_000_000,
                    "max_processing_time_ms": self.max_processing_time_ns / 1_000_000,
                    "min_processing_time_ms": self.min_processing_time_ns / 1_000_000 if self.min_processing_time_ns != float('inf') else 0,
                    "processing_samples": len(self.processing_times),
                    "messages_per_second": self.messages_per_second
                },
                "buffers": {
                    "active_buffers": self.active_buffers,
                    "total_operations": self.total_buffer_operations,
                    "resize_operations": self.buffer_resize_operations,
                    "adaptive_adjustments": self.adaptive_buffer_adjustments
                },
                "cache": {
                    "hits": self.cache_hits,
                    "misses": self.cache_misses,
                    "hit_rate": self.cache_hit_rate,
                    "compression_ratio": self.compression_ratio
                },
                "arm64": {
                    "optimizations_used": self.arm64_optimizations_used,
                    "numa_operations": self.numa_operations,
                    "simd_operations": self.simd_operations,
                    "prefetch_operations": self.prefetch_operations
                }
            }

@dataclass
class MarketDataPoint:
    """Enhanced market data point with sub-second precision and ARM64 optimizations."""
    symbol: str
    timestamp: int  # Unix nanoseconds
    event_type: str  # T, Q, A, AS, AM
    data_type: str = "unknown"  # trade, quote, aggregate
    price: Optional[float] = None
    size: Optional[int] = None
    bid_price: Optional[float] = None
    ask_price: Optional[float] = None
    bid_size: Optional[int] = None
    ask_size: Optional[int] = None
    open_price: Optional[float] = None
    high_price: Optional[float] = None
    low_price: Optional[float] = None
    close_price: Optional[float] = None
    volume: Optional[int] = None
    vwap: Optional[float] = None
    exchange: Optional[str] = None
    conditions: Optional[List[int]] = None
    sequence_number: Optional[int] = None
    participant_timestamp: Optional[int] = None
    sip_timestamp: Optional[int] = None
    receive_timestamp: Optional[int] = None
    processing_latency_ns: Optional[int] = None
    message_hash: Optional[str] = None
    numa_node: Optional[int] = None
    
    def __post_init__(self):
        """Post-initialization processing with ARM64 optimizations."""
        # Set data_type based on event_type if not explicitly set
        if self.data_type == "unknown":
            if self.event_type == "T":
                self.data_type = "trade"
            elif self.event_type == "Q":
                self.data_type = "quote"
            elif self.event_type in ["A", "AS", "AM"]:
                self.data_type = "aggregate"
        
        # Set receive timestamp if not set
        if self.receive_timestamp is None:
            self.receive_timestamp = time.time_ns()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization with ARM64 optimizations."""
        return asdict(self)
    
    def calculate_hash(self) -> str:
        """Calculate hash for deduplication with ARM64 optimizations."""
        if self.message_hash is None:
            # Create hash from key fields for deduplication
            # Use more fields for better deduplication accuracy
            hash_fields = [
                str(self.symbol),
                str(self.timestamp),
                str(self.event_type),
                str(self.price) if self.price is not None else "None",
                str(self.size) if self.size is not None else "None",
                str(self.bid_price) if self.bid_price is not None else "None",
                str(self.ask_price) if self.ask_price is not None else "None"
            ]
            hash_data = "_".join(hash_fields)
            
            # Use faster hash for ARM64
            if platform.machine().lower() in ['arm64', 'aarch64']:
                # Use xxhash if available, fallback to md5
                try:
                    import xxhash
                    self.message_hash = xxhash.xxh64(hash_data.encode()).hexdigest()[:16]
                except ImportError:
                    self.message_hash = hashlib.md5(hash_data.encode()).hexdigest()[:16]
            else:
                self.message_hash = hashlib.md5(hash_data.encode()).hexdigest()[:16]
        
        return self.message_hash
    
    def is_valid(self) -> bool:
        """Validate data point with ARM64 optimized checks."""
        # Basic validation
        if not self.symbol or not self.timestamp or not self.event_type:
            return False
        
        # Type-specific validation
        if self.data_type == "trade":
            return self.price is not None and self.size is not None and self.price > 0 and self.size > 0
        elif self.data_type == "quote":
            return (self.bid_price is not None and self.ask_price is not None and
                   self.bid_price > 0 and self.ask_price > 0 and self.ask_price >= self.bid_price)
        elif self.data_type == "aggregate":
            return (self.open_price is not None and self.high_price is not None and
                   self.low_price is not None and self.close_price is not None and
                   all(p > 0 for p in [self.open_price, self.high_price, self.low_price, self.close_price]))
        
        return True
    
    def get_mid_price(self) -> Optional[float]:
        """Get mid price for quotes with ARM64 optimizations."""
        if self.data_type == "quote" and self.bid_price is not None and self.ask_price is not None:
            return (self.bid_price + self.ask_price) / 2.0
        elif self.data_type == "trade" and self.price is not None:
            return self.price
        elif self.data_type == "aggregate" and self.close_price is not None:
            return self.close_price
        return None
    
    def get_spread(self) -> Optional[float]:
        """Get bid-ask spread for quotes."""
        if (self.data_type == "quote" and
            self.bid_price is not None and self.ask_price is not None):
            return self.ask_price - self.bid_price
        return None
    
    def get_spread_bps(self) -> Optional[float]:
        """Get bid-ask spread in basis points."""
        spread = self.get_spread()
        mid_price = self.get_mid_price()
        if spread is not None and mid_price is not None and mid_price > 0:
            return (spread / mid_price) * 10000
        return None

class LatencyTracker:
    """Enhanced latency tracker with ARM64 optimizations and comprehensive metrics."""
    
    def __init__(self, max_samples: int = 1000, enable_arm64_optimizations: bool = True):
        self.max_samples = max_samples
        self.enable_arm64_optimizations = enable_arm64_optimizations
        self.is_arm64 = platform.machine().lower() in ['arm64', 'aarch64']
        
        # ARM64 optimized data structures
        if self.is_arm64 and enable_arm64_optimizations:
            # Use numpy arrays for better ARM64 SIMD performance
            self.latencies = np.zeros(max_samples, dtype=np.int64)
            self.current_index = 0
            self.sample_count = 0
        else:
            self.latencies = deque(maxlen=max_samples)
        
        self.lock = threading.Lock()
        self.stats_cache = {}
        self.last_stats_update = 0
        self.cache_ttl = 1.0  # 1 second cache TTL
    
    def record_latency(self, receive_time: int, market_time: int) -> None:
        """Record latency with ARM64 optimizations."""
        latency_ns = receive_time - market_time
        
        with self.lock:
            if self.is_arm64 and self.enable_arm64_optimizations:
                # ARM64 optimized circular buffer
                self.latencies[self.current_index] = latency_ns
                self.current_index = (self.current_index + 1) % self.max_samples
                self.sample_count = min(self.sample_count + 1, self.max_samples)
            else:
                self.latencies.append(latency_ns)
            
            # Invalidate cache
            self.stats_cache.clear()
    
    def get_stats(self) -> Dict[str, float]:
        """Get latency statistics with ARM64 optimizations and caching."""
        current_time = time.time()
        
        with self.lock:
            # Check cache
            if (self.stats_cache and
                current_time - self.last_stats_update < self.cache_ttl):
                return self.stats_cache
            
            if self.is_arm64 and self.enable_arm64_optimizations:
                if self.sample_count == 0:
                    return {}
                
                # ARM64 optimized statistics calculation using numpy
                valid_latencies = self.latencies[:self.sample_count]
                latencies_ms = valid_latencies / 1_000_000  # Convert to ms
                
                stats = {
                    "avg_latency_ms": float(np.mean(latencies_ms)),
                    "min_latency_ms": float(np.min(latencies_ms)),
                    "max_latency_ms": float(np.max(latencies_ms)),
                    "p50_latency_ms": float(np.percentile(latencies_ms, 50)),
                    "p95_latency_ms": float(np.percentile(latencies_ms, 95)),
                    "p99_latency_ms": float(np.percentile(latencies_ms, 99)),
                    "std_latency_ms": float(np.std(latencies_ms)),
                    "sample_count": self.sample_count,
                    "arm64_optimized": True
                }
            else:
                if not self.latencies:
                    return {}
                
                latencies_ms = [lat / 1_000_000 for lat in self.latencies]
                stats = {
                    "avg_latency_ms": np.mean(latencies_ms),
                    "min_latency_ms": np.min(latencies_ms),
                    "max_latency_ms": np.max(latencies_ms),
                    "p50_latency_ms": np.percentile(latencies_ms, 50),
                    "p95_latency_ms": np.percentile(latencies_ms, 95),
                    "p99_latency_ms": np.percentile(latencies_ms, 99),
                    "std_latency_ms": np.std(latencies_ms),
                    "sample_count": len(latencies_ms),
                    "arm64_optimized": False
                }
            
            # Cache results
            self.stats_cache = stats
            self.last_stats_update = current_time
            
            return stats

class GH200DataManager:
    """Leverage GH200's unified CPU-GPU memory architecture."""
    
    def __init__(self, total_memory_gb: int = 600):
        self.total_memory_gb = total_memory_gb
        self.cpu_memory_pool = {}
        self.gpu_memory_pool = {}
        self.unified_buffers = {}
        self.is_gh200 = platform.machine().lower() in ['arm64', 'aarch64'] and GPU_AVAILABLE
        
    def allocate_unified_buffer(self, symbol: str, size: int):
        """Allocate buffer accessible by both CPU and GPU."""
        try:
            if self.is_gh200 and GPU_AVAILABLE:
                # Allocate in unified memory space
                buffer = cp.cuda.memory.MemoryPool().malloc(size)
                self.unified_buffers[symbol] = buffer
                return buffer
            else:
                # Fallback to CPU-only
                return np.zeros(size, dtype=np.float32)
        except Exception:
            # Fallback to CPU-only
            return np.zeros(size, dtype=np.float32)
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory usage statistics."""
        stats = {
            "total_memory_gb": self.total_memory_gb,
            "cpu_buffers": len(self.cpu_memory_pool),
            "unified_buffers": len(self.unified_buffers),
            "is_gh200": self.is_gh200
        }
        
        if self.is_gh200 and GPU_AVAILABLE:
            try:
                meminfo = cp.cuda.runtime.memGetInfo()
                stats["gpu_memory_free"] = meminfo[0] / (1024**3)  # GB
                stats["gpu_memory_total"] = meminfo[1] / (1024**3)  # GB
                stats["gpu_memory_used"] = (meminfo[1] - meminfo[0]) / (1024**3)  # GB
            except Exception:
                pass
        
        return stats

class DataBuffer:
    """Enhanced high-performance buffer with ARM64 optimizations and adaptive sizing."""
    
    def __init__(self, symbol: str, buffer_size: int = 10000,
                 enable_arm64_optimizations: bool = True,
                 enable_adaptive_sizing: bool = True,
                 gh200_manager: Optional[GH200DataManager] = None):
        self.symbol = symbol
        self.initial_buffer_size = buffer_size
        self.current_buffer_size = buffer_size
        self.enable_arm64_optimizations = enable_arm64_optimizations
        self.enable_adaptive_sizing = enable_adaptive_sizing
        self.is_arm64 = platform.machine().lower() in ['arm64', 'aarch64']
        self.gh200_manager = gh200_manager
        
        # ARM64 optimized data structures
        if self.is_arm64 and enable_arm64_optimizations:
            # Use OrderedDict for better ARM64 cache performance
            self.trades = OrderedDict()
            self.quotes = OrderedDict()
            self.max_trades = buffer_size
            self.max_quotes = buffer_size
        else:
            self.trades = deque(maxlen=buffer_size)
            self.quotes = deque(maxlen=buffer_size)
        
        self.second_bars = OrderedDict()  # ARM64 optimized ordering
        self.lock = threading.RLock()
        
        # Performance monitoring
        self.add_operations = 0
        self.last_resize_time = time.time()
        self.resize_threshold = 1000  # Operations before considering resize
        
        # Message deduplication
        self.message_hashes = set()
        self.dedup_window_start = time.time()
        self.dedup_window_seconds = 60
        
        # NUMA awareness
        self.numa_node = self._get_current_numa_node() if enable_arm64_optimizations else None
        
        # GH200 unified memory buffer
        if self.gh200_manager:
            try:
                self.unified_buffer = self.gh200_manager.allocate_unified_buffer(symbol, buffer_size * 1024)
            except Exception:
                self.unified_buffer = None
        else:
            self.unified_buffer = None
    
    def _get_current_numa_node(self) -> Optional[int]:
        """Get current NUMA node for ARM64 optimization."""
        try:
            if self.is_arm64:
                # Simple NUMA detection for ARM64
                return os.sched_getaffinity(0).__iter__().__next__() % 4  # Assume 4 NUMA nodes
        except:
            pass
        return None
    
    def _should_resize_buffer(self) -> bool:
        """Determine if buffer should be resized based on usage patterns."""
        if not self.enable_adaptive_sizing:
            return False
        
        current_time = time.time()
        if current_time - self.last_resize_time < 60:  # Don't resize too frequently
            return False
        
        if self.is_arm64 and self.enable_arm64_optimizations:
            trade_usage = len(self.trades) / self.max_trades
            quote_usage = len(self.quotes) / self.max_quotes
        else:
            trade_usage = len(self.trades) / self.trades.maxlen if self.trades.maxlen else 0
            quote_usage = len(self.quotes) / self.quotes.maxlen if self.quotes.maxlen else 0
        
        # Resize if consistently over 80% or under 20% usage
        return trade_usage > 0.8 or quote_usage > 0.8 or (trade_usage < 0.2 and quote_usage < 0.2)
    
    def _resize_buffers(self):
        """Resize buffers based on usage patterns."""
        if not self.enable_adaptive_sizing:
            return
        
        try:
            if self.is_arm64 and self.enable_arm64_optimizations:
                trade_usage = len(self.trades) / self.max_trades
                quote_usage = len(self.quotes) / self.max_quotes
                
                if trade_usage > 0.8 or quote_usage > 0.8:
                    # Increase buffer size
                    new_size = min(self.current_buffer_size * 2, 100000)
                    self.max_trades = new_size
                    self.max_quotes = new_size
                    
                    # Trim if necessary
                    while len(self.trades) > self.max_trades:
                        self.trades.popitem(last=False)
                    while len(self.quotes) > self.max_quotes:
                        self.quotes.popitem(last=False)
                        
                elif trade_usage < 0.2 and quote_usage < 0.2:
                    # Decrease buffer size
                    new_size = max(self.current_buffer_size // 2, 1000)
                    self.max_trades = new_size
                    self.max_quotes = new_size
                
                self.current_buffer_size = max(self.max_trades, self.max_quotes)
                self.last_resize_time = time.time()
                
                logger.debug(f"Resized buffers for {self.symbol} to {self.current_buffer_size}")
        
        except Exception as e:
            logger.warning(f"Buffer resize failed for {self.symbol}: {e}")
    
    def _is_duplicate_message(self, data_point: MarketDataPoint) -> bool:
        """Check for duplicate messages using hash-based deduplication."""
        current_time = time.time()
        
        # Clean old hashes periodically
        if current_time - self.dedup_window_start > self.dedup_window_seconds:
            self.message_hashes.clear()
            self.dedup_window_start = current_time
        
        message_hash = data_point.calculate_hash()
        if message_hash in self.message_hashes:
            return True
        
        self.message_hashes.add(message_hash)
        return False
    
    def add_trade(self, data_point: MarketDataPoint) -> bool:
        """Add trade data with ARM64 optimizations and deduplication."""
        with self.lock:
            # Check for duplicates
            if self._is_duplicate_message(data_point):
                return False
            
            # Add ARM64 specific metadata
            if self.is_arm64 and self.enable_arm64_optimizations:
                data_point.numa_node = self.numa_node
                data_point.receive_timestamp = time.time_ns()
            
            if self.is_arm64 and self.enable_arm64_optimizations:
                # ARM64 optimized storage
                self.trades[data_point.timestamp] = data_point
                
                # Maintain size limit
                while len(self.trades) > self.max_trades:
                    self.trades.popitem(last=False)
            else:
                self.trades.append(data_point)
            
            self._update_second_bar(data_point)
            self.add_operations += 1
            
            # Check for buffer resize
            if self.add_operations % self.resize_threshold == 0:
                if self._should_resize_buffer():
                    self._resize_buffers()
            
            return True
    
    def add_quote(self, data_point: MarketDataPoint) -> bool:
        """Add quote data with ARM64 optimizations."""
        with self.lock:
            # Check for duplicates
            if self._is_duplicate_message(data_point):
                return False
            
            # Add ARM64 specific metadata
            if self.is_arm64 and self.enable_arm64_optimizations:
                data_point.numa_node = self.numa_node
                data_point.receive_timestamp = time.time_ns()
            
            if self.is_arm64 and self.enable_arm64_optimizations:
                # ARM64 optimized storage
                self.quotes[data_point.timestamp] = data_point
                
                # Maintain size limit
                while len(self.quotes) > self.max_quotes:
                    self.quotes.popitem(last=False)
            else:
                self.quotes.append(data_point)
            
            self.add_operations += 1
            return True
    
    def _update_second_bar(self, trade: MarketDataPoint) -> None:
        """Update second-level OHLCV bar with ARM64 optimizations."""
        if not trade.price or not trade.size:
            return
        
        # Get second timestamp (truncate to second)
        second_ts = (trade.timestamp // 1_000_000_000) * 1_000_000_000
        
        if second_ts not in self.second_bars:
            self.second_bars[second_ts] = {
                "symbol": self.symbol,
                "timestamp": second_ts,
                "open": trade.price,
                "high": trade.price,
                "low": trade.price,
                "close": trade.price,
                "volume": trade.size,
                "trade_count": 1,
                "vwap_sum": trade.price * trade.size,
                "first_trade_time": trade.timestamp,
                "last_trade_time": trade.timestamp,
                "numa_node": trade.numa_node if self.is_arm64 else None
            }
        else:
            bar = self.second_bars[second_ts]
            # ARM64 optimized min/max operations
            bar["high"] = max(bar["high"], trade.price)
            bar["low"] = min(bar["low"], trade.price)
            bar["close"] = trade.price
            bar["volume"] += trade.size
            bar["trade_count"] += 1
            bar["vwap_sum"] += trade.price * trade.size
            bar["last_trade_time"] = trade.timestamp
        
        # Maintain reasonable number of second bars (ARM64 cache friendly)
        max_bars = 86400 if self.is_arm64 else 3600  # 24h vs 1h
        while len(self.second_bars) > max_bars:
            self.second_bars.popitem(last=False)
    
    def _check_memory_pressure(self) -> bool:
        """Check system memory pressure and adjust buffers accordingly."""
        memory = psutil.virtual_memory()
        if memory.percent > 85:  # High memory usage
            self._reduce_buffer_sizes()
            return True
        return False
    
    def _reduce_buffer_sizes(self) -> None:
        """Reduce buffer sizes to free memory under pressure."""
        try:
            if self.is_arm64 and self.enable_arm64_optimizations:
                # Reduce ARM64 optimized buffers
                new_size = max(self.max_trades // 2, 1000)
                self.max_trades = new_size
                self.max_quotes = new_size
                
                # Trim existing data
                while len(self.trades) > self.max_trades:
                    self.trades.popitem(last=False)
                while len(self.quotes) > self.max_quotes:
                    self.quotes.popitem(last=False)
            else:
                # Reduce standard buffers
                new_size = max(self.current_buffer_size // 2, 1000)
                self.current_buffer_size = new_size
                
                # Create new smaller deques
                self.trades = deque(list(self.trades)[-new_size:], maxlen=new_size)
                self.quotes = deque(list(self.quotes)[-new_size:], maxlen=new_size)
            
            # Reduce second bars
            max_bars = min(len(self.second_bars), 3600)  # Keep max 1 hour
            while len(self.second_bars) > max_bars:
                self.second_bars.popitem(last=False)
                
            logger.info(f"Reduced buffer sizes for {self.symbol} due to memory pressure")
            
        except Exception as e:
            logger.warning(f"Failed to reduce buffer sizes for {self.symbol}: {e}")
    
    def get_second_bars(self, start_time: Optional[int] = None,
                       end_time: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get second-level bars with ARM64 optimizations."""
        with self.lock:
            bars = []
            
            # ARM64 optimized iteration
            for ts, bar_data in self.second_bars.items():
                if start_time and ts < start_time:
                    continue
                if end_time and ts > end_time:
                    break
                
                # Calculate VWAP with ARM64 optimized arithmetic
                bar_copy = bar_data.copy()
                if bar_copy["volume"] > 0:
                    bar_copy["vwap"] = bar_copy["vwap_sum"] / bar_copy["volume"]
                else:
                    bar_copy["vwap"] = bar_copy["close"]
                
                bars.append(bar_copy)
            
            return bars
    
    def get_latest_quote(self) -> Optional[MarketDataPoint]:
        """Get the most recent quote with ARM64 optimizations."""
        with self.lock:
            if self.is_arm64 and self.enable_arm64_optimizations:
                if not self.quotes:
                    return None
                # Get last item from OrderedDict
                return next(reversed(self.quotes.values()))
            else:
                return self.quotes[-1] if self.quotes else None
    
    def get_recent_trades(self, count: int = 100) -> List[MarketDataPoint]:
        """Get recent trades with ARM64 optimizations."""
        with self.lock:
            if self.is_arm64 and self.enable_arm64_optimizations:
                # ARM64 optimized slice from OrderedDict
                trades_list = list(self.trades.values())
                return trades_list[-count:] if trades_list else []
            else:
                return list(self.trades)[-count:]
    
    def get_buffer_stats(self) -> Dict[str, Any]:
        """Get buffer performance statistics."""
        with self.lock:
            if self.is_arm64 and self.enable_arm64_optimizations:
                trade_count = len(self.trades)
                quote_count = len(self.quotes)
                trade_capacity = self.max_trades
                quote_capacity = self.max_quotes
            else:
                trade_count = len(self.trades)
                quote_count = len(self.quotes)
                trade_capacity = self.trades.maxlen if self.trades.maxlen else 0
                quote_capacity = self.quotes.maxlen if self.quotes.maxlen else 0
            
            return {
                "symbol": self.symbol,
                "trade_count": trade_count,
                "quote_count": quote_count,
                "second_bars_count": len(self.second_bars),
                "trade_capacity": trade_capacity,
                "quote_capacity": quote_capacity,
                "trade_utilization": trade_count / max(trade_capacity, 1),
                "quote_utilization": quote_count / max(quote_capacity, 1),
                "add_operations": self.add_operations,
                "current_buffer_size": self.current_buffer_size,
                "arm64_optimized": self.is_arm64 and self.enable_arm64_optimizations,
                "numa_node": self.numa_node,
                "dedup_hashes": len(self.message_hashes)
            }

class AdvancedPolygonClient:
    """
    Advanced Polygon client with sub-second data support, latency tracking,
    and high-frequency trading optimizations.
    """

    WS_URL = "wss://socket.polygon.io/stocks"
    REST_BASE_URL = "https://api.polygon.io"
    
    # Supported timeframes for historical data
    TIMEFRAMES = {
        "second": "second",
        "minute": "minute", 
        "hour": "hour",
        "day": "day",
        "week": "week",
        "month": "month",
        "quarter": "quarter",
        "year": "year"
    }

    def __init__(self, config: Optional[PolygonConfig] = None):
        """
        Initialize Advanced Polygon Client.

        Args:
            config: Polygon configuration
        """
        # Load configuration from config manager if not provided
        if config is None:
            config_data = config_manager.get('training_config.data.polygon', {})
            self.config = PolygonConfig(**config_data) if config_data else PolygonConfig()
        else:
            self.config = config
        
        # API key handling
        self.api_key = self.config.api_key or os.getenv('POLYGON_API_KEY')
        if not self.api_key:
            raise ValueError("Polygon.io API key not provided and not found in environment variables (POLYGON_API_KEY).")

        # Connection management
        self.websocket: Optional[websockets.WebSocketClientProtocol] = None
        self.is_connected = False
        self.reconnect_attempts = 0
        
        # Data handling
        self.message_handlers: Dict[str, List[Callable]] = defaultdict(list)
        self.data_buffers: Dict[str, DataBuffer] = {}
        self.latency_tracker = LatencyTracker()
        
        # GH200 unified memory management
        self.gh200_manager = None
        if self.config.enable_unified_memory:
            try:
                self.gh200_manager = GH200DataManager(self.config.total_memory_gb)
                logger.info("GH200 unified memory manager initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize GH200 manager: {e}")
        
        # Performance tracking
        self.message_count = 0
        self.error_count = 0
        self.last_message_time = 0
        self.connection_start_time = 0
        
        # Threading
        self.executor = ThreadPoolExecutor(max_workers=self.config.max_workers)
        self.processing_lock = threading.RLock()
        
        # Rate limiting
        self.request_times = deque(maxlen=self.config.rate_limit_per_minute)
        self.rate_limit_lock = threading.Lock()
        
        # Statistics tracking
        self.stats = PolygonStats()
        
        logger.info("AdvancedPolygonClient initialized with sub-second data support and GH200 optimizations")

    async def _connect_websocket(self) -> bool:
        """Establish WebSocket connection with enhanced error handling."""
        try:
            logger.info(f"Connecting to WebSocket: {self.WS_URL}")
            
            # Connection with timeout and compression
            extra_headers = {}
            if self.config.enable_compression:
                extra_headers["Sec-WebSocket-Extensions"] = "permessage-deflate"
            
            self.websocket = await websockets.connect(
                self.WS_URL,
                timeout=self.config.websocket_timeout,
                extra_headers=extra_headers,
                ping_interval=20,
                ping_timeout=10
            )
            
            self.is_connected = True
            self.reconnect_attempts = 0
            self.connection_start_time = time.time_ns()
            
            logger.info("WebSocket connected successfully")
            
            # Authenticate
            await self._authenticate_websocket()
            return True
            
        except Exception as e:
            logger.error(f"WebSocket connection failed: {e}")
            self.is_connected = False
            return False

    async def _authenticate_websocket(self) -> bool:
        """Authenticate WebSocket connection."""
        try:
            auth_message = {
                "action": "auth",
                "params": self.api_key
            }
            
            await self.websocket.send(json.dumps(auth_message))
            response = json.loads(await self.websocket.recv())
            
            if response.get("status") == "auth_success":
                logger.info("WebSocket authentication successful")
                return True
            else:
                logger.error(f"WebSocket authentication failed: {response}")
                await self.close()
                return False
                
        except Exception as e:
            logger.error(f"Authentication error: {e}")
            return False

    async def _handle_reconnect(self) -> None:
        """Handle reconnection with exponential backoff."""
        if self.reconnect_attempts >= self.config.max_reconnect_attempts:
            logger.error("Max reconnection attempts reached")
            return
        
        self.reconnect_attempts += 1
        delay = self.config.reconnect_delay * (2 ** (self.reconnect_attempts - 1))
        delay = min(delay, 60)  # Cap at 60 seconds
        
        logger.warning(f"Reconnecting in {delay:.1f}s (attempt {self.reconnect_attempts})")
        await asyncio.sleep(delay)
        
        if await self._connect_websocket():
            # Re-subscribe to previous subscriptions
            await self._resubscribe()

    async def _resubscribe(self) -> None:
        """Re-subscribe to previous subscriptions after reconnection."""
        # This would store and restore previous subscriptions
        # Implementation depends on tracking subscription state
        pass

    def _check_rate_limit(self) -> bool:
        """Check if we're within rate limits."""
        current_time = time.time()
        
        with self.rate_limit_lock:
            # Remove requests older than 1 minute
            while (self.request_times and 
                   current_time - self.request_times[0] > 60):
                self.request_times.popleft()
            
            # Check if we can make another request
            if len(self.request_times) >= self.config.rate_limit_per_minute:
                return False
            
            self.request_times.append(current_time)
            return True

    def _parse_binary_message(self, data: bytes) -> List[MarketDataPoint]:
        """Parse binary Polygon messages for higher throughput."""
        try:
            # This is a placeholder for binary parsing implementation
            # Polygon supports binary formats for high-frequency clients
            # Implementation would depend on specific binary protocol
            logger.debug("Binary message parsing not yet implemented")
            return []
        except Exception as e:
            logger.error(f"Error parsing binary message: {e}")
            return []

    def _parse_message(self, raw_message: Dict[str, Any]) -> Optional[MarketDataPoint]:
        """Parse raw Polygon message into MarketDataPoint."""
        try:
            event_type = raw_message.get("ev")
            if not event_type:
                return None
            
            symbol = raw_message.get("sym")
            if not symbol:
                return None
            
            # Get timestamp with nanosecond precision
            timestamp = raw_message.get("t", 0)
            if isinstance(timestamp, (int, float)):
                # Convert to nanoseconds if needed
                if timestamp < 1e15:  # Likely milliseconds
                    timestamp = int(timestamp * 1_000_000)
                elif timestamp < 1e18:  # Likely microseconds
                    timestamp = int(timestamp * 1_000)
                else:  # Already nanoseconds
                    timestamp = int(timestamp)
            
            data_point = MarketDataPoint(
                symbol=symbol,
                timestamp=timestamp,
                event_type=event_type
            )
            
            # Parse based on event type
            if event_type == "T":  # Trade
                data_point.price = raw_message.get("p")
                data_point.size = raw_message.get("s")
                data_point.exchange = raw_message.get("x")
                data_point.conditions = raw_message.get("c", [])
                data_point.sequence_number = raw_message.get("q")
                data_point.participant_timestamp = raw_message.get("y")
                data_point.sip_timestamp = raw_message.get("f")
                
            elif event_type == "Q":  # Quote
                data_point.bid_price = raw_message.get("bp")
                data_point.ask_price = raw_message.get("ap")
                data_point.bid_size = raw_message.get("bs")
                data_point.ask_size = raw_message.get("as")
                data_point.exchange = raw_message.get("x")
                data_point.participant_timestamp = raw_message.get("y")
                data_point.sip_timestamp = raw_message.get("f")
                
            elif event_type in ["A", "AS", "AM"]:  # Aggregates (second, minute)
                data_point.open_price = raw_message.get("o")
                data_point.high_price = raw_message.get("h")
                data_point.low_price = raw_message.get("l")
                data_point.close_price = raw_message.get("c")
                data_point.volume = raw_message.get("v")
                data_point.vwap = raw_message.get("vw")
            
            return data_point
            
        except Exception as e:
            logger.error(f"Error parsing message: {e}")
            return None

    def _process_data_point(self, data_point: MarketDataPoint) -> None:
        """Process data point and update buffers."""
        try:
            # Track latency
            if self.config.enable_latency_tracking:
                receive_time = time.time_ns()
                if data_point.sip_timestamp:
                    self.latency_tracker.record_latency(receive_time, data_point.sip_timestamp)
                elif data_point.participant_timestamp:
                    self.latency_tracker.record_latency(receive_time, data_point.participant_timestamp)
            
            # Update data buffer
            if data_point.symbol not in self.data_buffers:
                self.data_buffers[data_point.symbol] = DataBuffer(
                    data_point.symbol, 
                    self.config.buffer_size
                )
            
            buffer = self.data_buffers[data_point.symbol]
            
            # Check memory pressure periodically
            if self.config.enable_memory_pressure_monitoring and self.message_count % 1000 == 0:
                buffer._check_memory_pressure()
            
            if data_point.event_type == "T":
                buffer.add_trade(data_point)
            elif data_point.event_type == "Q":
                buffer.add_quote(data_point)
            
            # Call registered handlers
            for handler in self.message_handlers.get(data_point.event_type, []):
                try:
                    handler(data_point)
                except Exception as e:
                    logger.error(f"Error in message handler: {e}")
            
            # Call general handlers
            for handler in self.message_handlers.get("*", []):
                try:
                    handler(data_point)
                except Exception as e:
                    logger.error(f"Error in general handler: {e}")
            
            self.message_count += 1
            self.last_message_time = time.time_ns()
            
        except Exception as e:
            logger.error(f"Error processing data point: {e}")
            self.error_count += 1

    async def subscribe_to_trades(self, symbols: List[str]) -> bool:
        """Subscribe to real-time trade data."""
        if not self.is_connected:
            logger.error("Not connected to WebSocket")
            return False
        
        try:
            subscriptions = [f"T.{symbol}" for symbol in symbols]
            subscribe_message = {
                "action": "subscribe",
                "params": ",".join(subscriptions)
            }
            
            await self.websocket.send(json.dumps(subscribe_message))
            logger.info(f"Subscribed to trades: {symbols}")
            return True
            
        except Exception as e:
            logger.error(f"Error subscribing to trades: {e}")
            return False

    async def subscribe_to_quotes(self, symbols: List[str]) -> bool:
        """Subscribe to real-time quote data."""
        if not self.is_connected:
            logger.error("Not connected to WebSocket")
            return False
        
        try:
            subscriptions = [f"Q.{symbol}" for symbol in symbols]
            subscribe_message = {
                "action": "subscribe", 
                "params": ",".join(subscriptions)
            }
            
            await self.websocket.send(json.dumps(subscribe_message))
            logger.info(f"Subscribed to quotes: {symbols}")
            return True
            
        except Exception as e:
            logger.error(f"Error subscribing to quotes: {e}")
            return False

    async def subscribe_to_second_aggregates(self, symbols: List[str]) -> bool:
        """Subscribe to second-level aggregate data."""
        if not self.is_connected:
            logger.error("Not connected to WebSocket")
            return False
        
        try:
            subscriptions = [f"A.{symbol}" for symbol in symbols]  # Second aggregates
            subscribe_message = {
                "action": "subscribe",
                "params": ",".join(subscriptions)
            }
            
            await self.websocket.send(json.dumps(subscribe_message))
            logger.info(f"Subscribed to second aggregates: {symbols}")
            return True
            
        except Exception as e:
            logger.error(f"Error subscribing to second aggregates: {e}")
            return False

    async def subscribe_to_minute_aggregates(self, symbols: List[str]) -> bool:
        """Subscribe to minute-level aggregate data."""
        if not self.is_connected:
            logger.error("Not connected to WebSocket")
            return False
        
        try:
            subscriptions = [f"AM.{symbol}" for symbol in symbols]  # Minute aggregates
            subscribe_message = {
                "action": "subscribe",
                "params": ",".join(subscriptions)
            }
            
            await self.websocket.send(json.dumps(subscribe_message))
            logger.info(f"Subscribed to minute aggregates: {symbols}")
            return True
            
        except Exception as e:
            logger.error(f"Error subscribing to minute aggregates: {e}")
            return False

    async def subscribe_to_all_market_data(self) -> bool:
        """Subscribe to entire US equity market with single call."""
        if not self.is_connected:
            logger.error("Not connected to WebSocket")
            return False
        
        try:
            subscribe_message = {
                "action": "subscribe",
                "params": "T.*,Q.*,A.*,AM.*"  # All trades, quotes, second & minute bars
            }
            await self.websocket.send(json.dumps(subscribe_message))
            logger.info("Subscribed to entire US equity market")
            return True
        except Exception as e:
            logger.error(f"Error subscribing to market: {e}")
            return False

    def add_handler(self, event_type: str, handler: Callable[[MarketDataPoint], None]) -> None:
        """Add message handler for specific event type."""
        self.message_handlers[event_type].append(handler)
        logger.info(f"Added handler for event type: {event_type}")

    def remove_handler(self, event_type: str, handler: Callable[[MarketDataPoint], None]) -> None:
        """Remove message handler."""
        if handler in self.message_handlers[event_type]:
            self.message_handlers[event_type].remove(handler)

    async def stream_data(self) -> None:
        """Start streaming data from WebSocket."""
        if not self.is_connected:
            if not await self._connect_websocket():
                return
        
        try:
            while self.is_connected:
                try:
                    message = await asyncio.wait_for(
                        self.websocket.recv(), 
                        timeout=self.config.websocket_timeout
                    )
                    
                    # Parse message
                    try:
                        messages = json.loads(message)
                        if not isinstance(messages, list):
                            messages = [messages]
                        
                        for msg in messages:
                            if msg.get("ev") == "status":
                                logger.info(f"Status: {msg}")
                                continue
                            
                            data_point = self._parse_message(msg)
                            if data_point:
                                # Process in thread pool for better performance
                                self.executor.submit(self._process_data_point, data_point)
                    
                    except json.JSONDecodeError as e:
                        logger.error(f"JSON decode error: {e}")
                        continue
                
                except asyncio.TimeoutError:
                    logger.warning("WebSocket receive timeout")
                    continue
                except websockets.exceptions.ConnectionClosed:
                    logger.warning("WebSocket connection closed")
                    self.is_connected = False
                    await self._handle_reconnect()
                    break
                    
        except Exception as e:
            logger.error(f"Error in data streaming: {e}")
            self.is_connected = False
            await self._handle_reconnect()

    async def get_historical_aggregates(self,
                                      symbol: str,
                                      timespan: str,
                                      multiplier: int = 1,
                                      from_date: str = None,
                                      to_date: str = None,
                                      adjusted: bool = True,
                                      sort: str = "asc",
                                      limit: int = 50000) -> Optional[pd.DataFrame]:
        """
        Get historical aggregate data with support for second-level timeframes.

        Args:
            symbol: Stock symbol
            timespan: second, minute, hour, day, week, month, quarter, year
            multiplier: Size of the timespan multiplier
            from_date: Start date (YYYY-MM-DD)
            to_date: End date (YYYY-MM-DD)
            adjusted: Whether to adjust for splits
            sort: Sort order (asc/desc)
            limit: Maximum number of results

        Returns:
            DataFrame with OHLCV data
        """
        if not self._check_rate_limit():
            logger.warning("Rate limit exceeded, waiting...")
            await asyncio.sleep(1)
            return None
        
        if timespan not in self.TIMEFRAMES:
            logger.error(f"Unsupported timespan: {timespan}")
            return None
        
        try:
            url = (f"{self.REST_BASE_URL}/v2/aggs/ticker/{symbol}/range/"
                   f"{multiplier}/{timespan}/{from_date}/{to_date}")
            
            params = {
                "adjusted": str(adjusted).lower(),
                "sort": sort,
                "limit": limit,
                "apikey": self.api_key
            }
            
            async with httpx.AsyncClient(timeout=self.config.rest_timeout) as client:
                response = await client.get(url, params=params)
                response.raise_for_status()
                
                data = response.json()
                
                if not data.get("results"):
                    logger.warning(f"No data returned for {symbol}")
                    return pd.DataFrame()
                
                # Convert to DataFrame
                df = pd.DataFrame(data["results"])
                
                # Convert timestamp to datetime
                df['timestamp'] = pd.to_datetime(df['t'], unit='ms')
                df.set_index('timestamp', inplace=True)
                
                # Rename columns to standard OHLCV format
                column_mapping = {
                    'o': 'open',
                    'h': 'high', 
                    'l': 'low',
                    'c': 'close',
                    'v': 'volume',
                    'vw': 'vwap',
                    'n': 'transactions'
                }
                df.rename(columns=column_mapping, inplace=True)
                
                # Add symbol column
                df['symbol'] = symbol
                
                logger.info(f"Retrieved {len(df)} {timespan} bars for {symbol}")
                return df
                
        except Exception as e:
            logger.error(f"Error getting historical data for {symbol}: {e}")
            return None

    def get_second_bars(self, symbol: str, 
                       start_time: Optional[int] = None,
                       end_time: Optional[int] = None) -> pd.DataFrame:
        """Get second-level bars from buffer."""
        if symbol not in self.data_buffers:
            return pd.DataFrame()
        
        bars = self.data_buffers[symbol].get_second_bars(start_time, end_time)
        if not bars:
            return pd.DataFrame()
        
        df = pd.DataFrame(bars)
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ns')
        df.set_index('timestamp', inplace=True)
        
        return df

    def get_latest_quote(self, symbol: str) -> Optional[MarketDataPoint]:
        """Get latest quote for symbol."""
        if symbol not in self.data_buffers:
            return None
        
        return self.data_buffers[symbol].get_latest_quote()

    def get_recent_trades(self, symbol: str, count: int = 100) -> List[MarketDataPoint]:
        """Get recent trades for symbol."""
        if symbol not in self.data_buffers:
            return []
        
        return self.data_buffers[symbol].get_recent_trades(count)

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get client performance statistics."""
        current_time = time.time_ns()
        uptime_seconds = (current_time - self.connection_start_time) / 1_000_000_000
        
        stats = {
            "is_connected": self.is_connected,
            "uptime_seconds": uptime_seconds,
            "message_count": self.message_count,
            "error_count": self.error_count,
            "messages_per_second": self.message_count / max(uptime_seconds, 1),
            "reconnect_attempts": self.reconnect_attempts,
            "buffered_symbols": len(self.data_buffers),
            "last_message_age_ms": (current_time - self.last_message_time) / 1_000_000 if self.last_message_time else 0
        }
        
        # Add latency stats
        if self.config.enable_latency_tracking:
            stats["latency"] = self.latency_tracker.get_stats()
        
        return stats
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get comprehensive system performance stats."""
        stats = {
            "cpu_usage": psutil.cpu_percent(interval=0.1),
            "memory_usage": psutil.virtual_memory()._asdict(),
            "network_io": psutil.net_io_counters()._asdict(),
            "disk_io": psutil.disk_io_counters()._asdict(),
        }
        
        # Add GPU memory stats if available
        if self.gh200_manager:
            stats.update(self.gh200_manager.get_memory_stats())
        
        return stats

    async def close(self) -> None:
        """Close WebSocket connection and cleanup."""
        if self.websocket and self.is_connected:
            logger.info("Closing WebSocket connection")
            await self.websocket.close()
            self.is_connected = False
            self.websocket = None
        
        # Shutdown executor
        self.executor.shutdown(wait=True)
        
        logger.info("AdvancedPolygonClient closed")

class DMNDataProcessor:
    """Process market data for Deep Momentum Networks."""
    
    def __init__(self, client: AdvancedPolygonClient):
        self.client = client
        self.feature_calculator = None  # Would be imported from feature engineering module
        
    async def process_for_dmn(self, data_point: MarketDataPoint):
        """Process data point for DMN feature engineering."""
        try:
            # Calculate momentum features
            if self.feature_calculator:
                features = self.feature_calculator.calculate_features(data_point)
                
                # Send to GPU for LSTM processing
                await self._send_to_gpu_pipeline(features)
        except Exception as e:
            logger.error(f"Error processing DMN data: {e}")
    
    async def _send_to_gpu_pipeline(self, features):
        """Send features to GPU pipeline for processing."""
        # Placeholder for GPU pipeline integration
        pass

class MarketWideDataManager:
    """Manage data for entire equity market."""
    
    def __init__(self, max_symbols: int = 10000):
        self.max_symbols = max_symbols
        self.active_symbols = set()
        self.priority_symbols = set()  # Most liquid stocks
        self.symbol_rotation_schedule = {}
        
    def optimize_symbol_coverage(self):
        """Dynamically adjust symbol coverage based on activity."""
        # Prioritize most active symbols
        # Rotate coverage for less active symbols
        pass
    
    def add_priority_symbol(self, symbol: str):
        """Add symbol to priority list."""
        self.priority_symbols.add(symbol)
        self.active_symbols.add(symbol)
    
    def get_active_symbols(self) -> Set[str]:
        """Get currently active symbols."""
        return self.active_symbols.copy()

class RealTimeRiskMonitor:
    """Real-time position risk monitoring."""
    
    def __init__(self):
        self.active_positions = {}
        self.risk_limits = {}
        
    async def monitor_position_risk(self, data_point: MarketDataPoint):
        """Real-time position risk monitoring."""
        if data_point.symbol in self.active_positions:
            current_price = data_point.get_mid_price()
            if current_price is None:
                return
                
            position = self.active_positions[data_point.symbol]
            
            # Calculate real-time P&L
            unrealized_pnl = (current_price - position.get('entry_price', 0)) * position.get('size', 0)
            
            # Risk checks
            risk_limit = self.risk_limits.get(data_point.symbol, float('inf'))
            if abs(unrealized_pnl) > risk_limit:
                await self._trigger_risk_action(data_point.symbol, unrealized_pnl)
    
    async def _trigger_risk_action(self, symbol: str, unrealized_pnl: float):
        """Trigger risk management action."""
        logger.warning(f"Risk limit exceeded for {symbol}: P&L {unrealized_pnl}")
        # Implement risk action (e.g., close position, reduce size)

# Legacy compatibility
PolygonClient = AdvancedPolygonClient

if __name__ == "__main__":
    # Example usage with sub-second data
    async def trade_handler(data_point: MarketDataPoint):
        """Handle trade data."""
        print(f"TRADE: {data_point.symbol} @ ${data_point.price} "
              f"size {data_point.size} at {data_point.timestamp}")

    async def quote_handler(data_point: MarketDataPoint):
        """Handle quote data."""
        print(f"QUOTE: {data_point.symbol} "
              f"${data_point.bid_price}x{data_point.bid_size} - "
              f"${data_point.ask_price}x{data_point.ask_size}")

    async def second_bar_handler(data_point: MarketDataPoint):
        """Handle second-level aggregate data."""
        print(f"SEC BAR: {data_point.symbol} "
              f"O:{data_point.open_price} H:{data_point.high_price} "
              f"L:{data_point.low_price} C:{data_point.close_price} "
              f"V:{data_point.volume}")

    async def main():
        # Check for API key
        api_key = os.getenv('POLYGON_API_KEY')
        if not api_key:
            logger.error("POLYGON_API_KEY environment variable not set")
            return

        # Create client with sub-second support
        config = PolygonConfig(
            api_key=api_key,
            enable_second_data=True,
            enable_subsecond_data=True,
            enable_latency_tracking=True
        )
        
        client = AdvancedPolygonClient(config)

        # Test historical second data
        logger.info("=== Testing Historical Second Data ===")
        second_data = await client.get_historical_aggregates(
            symbol="AAPL",
            timespan="second",
            multiplier=1,
            from_date="2023-12-01",
            to_date="2023-12-01",
            limit=100
        )
        
        if second_data is not None and not second_data.empty:
            logger.info(f"Retrieved {len(second_data)} second bars")
            logger.info(f"Sample data:\n{second_data.head()}")
        else:
            logger.warning("No second-level data retrieved")

        # Test minute data
        logger.info("=== Testing Historical Minute Data ===")
        minute_data = await client.get_historical_aggregates(
            symbol="AAPL",
            timespan="minute",
            multiplier=1,
            from_date="2023-12-01",
            to_date="2023-12-01",
            limit=100
        )
        
        if minute_data is not None and not minute_data.empty:
            logger.info(f"Retrieved {len(minute_data)} minute bars")

        # Test real-time streaming
        logger.info("=== Testing Real-time Streaming ===")
        
        # Add handlers
        client.add_handler("T", trade_handler)
        client.add_handler("Q", quote_handler)
        client.add_handler("A", second_bar_handler)

        # Connect and subscribe
        await client._connect_websocket()
        
        symbols = ["AAPL", "MSFT"]
        await client.subscribe_to_trades(symbols)
        await client.subscribe_to_quotes(symbols)
        await client.subscribe_to_second_aggregates(symbols)

        # Stream for 30 seconds
        streaming_task = asyncio.create_task(client.stream_data())
        
        try:
            await asyncio.sleep(30)
        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        finally:
            streaming_task.cancel()
            
            # Show performance stats
            stats = client.get_performance_stats()
            logger.info(f"Performance stats: {json.dumps(stats, indent=2)}")
            
            # Show second bars from buffer
            for symbol in symbols:
                second_bars = client.get_second_bars(symbol)
                if not second_bars.empty:
                    logger.info(f"{symbol} second bars: {len(second_bars)}")
            
            await client.close()

    asyncio.run(main())
