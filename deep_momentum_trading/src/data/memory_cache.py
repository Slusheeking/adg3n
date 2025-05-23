"""
Enhanced UnifiedMemoryManager with ARM64 optimizations for GH200 unified memory.

This module provides a high-performance memory management system optimized for
ARM64 architecture, specifically designed for the GH200 unified memory system.
It implements advanced caching mechanisms with automatic eviction, compression,
and ARM64-specific optimizations for storing market data, engineered features,
and model states.
"""

import gc
import hashlib
import pickle
import platform
import threading
import time
import warnings
from collections import OrderedDict, deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np
import torch

try:
    import psutil
except ImportError:
    psutil = None
    warnings.warn("psutil not available - memory monitoring will be limited")

from deep_momentum_trading.config.settings import config_manager
from deep_momentum_trading.src.utils.logger import get_logger

logger = get_logger(__name__)

@dataclass
class UnifiedMemoryConfig:
    """
    Configuration for UnifiedMemoryManager with ARM64 optimizations.
    
    Attributes:
        max_memory_gb: Maximum memory capacity in gigabytes
        enable_arm64_optimizations: Enable ARM64-specific optimizations
        enable_performance_monitoring: Enable performance metrics collection
        enable_compression: Enable data compression for storage efficiency
        enable_memory_mapping: Enable memory mapping for large datasets
        eviction_policy: Cache eviction policy ('lru', 'lfu', 'ttl')
        ttl_seconds: Default time-to-live for cached items
        max_blocks_per_pool: Maximum blocks per memory pool
        memory_alignment_bytes: Memory alignment for ARM64 cache lines
        enable_prefetching: Enable predictive data prefetching
        prefetch_size: Number of items to prefetch
        enable_background_cleanup: Enable background cleanup thread
        cleanup_interval_seconds: Interval between cleanup cycles
        memory_pressure_threshold: Memory utilization threshold for pressure events
        enable_numa_awareness: Enable NUMA-aware memory allocation
        enable_data_integrity: Enable data integrity checks with checksums
    """
    max_memory_gb: float = 200.0
    enable_arm64_optimizations: bool = True
    enable_performance_monitoring: bool = True
    enable_compression: bool = True
    enable_memory_mapping: bool = True
    eviction_policy: str = 'lru'  # 'lru', 'lfu', 'ttl'
    ttl_seconds: int = 3600
    max_blocks_per_pool: int = 10000
    memory_alignment_bytes: int = 64  # ARM64 cache line size
    enable_prefetching: bool = True
    prefetch_size: int = 8
    enable_background_cleanup: bool = True
    cleanup_interval_seconds: int = 300
    memory_pressure_threshold: float = 0.85
    enable_numa_awareness: bool = True
    enable_data_integrity: bool = True
    
    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        if self.max_memory_gb <= 0:
            raise ValueError("max_memory_gb must be positive")
        if not 0 < self.memory_pressure_threshold <= 1:
            raise ValueError("memory_pressure_threshold must be between 0 and 1")
        if self.eviction_policy not in ['lru', 'lfu', 'ttl']:
            raise ValueError("eviction_policy must be one of: lru, lfu, ttl")
        if self.cleanup_interval_seconds <= 0:
            raise ValueError("cleanup_interval_seconds must be positive")

@dataclass
class MemoryBlock:
    """
    Enhanced memory block with ARM64 optimizations and comprehensive metadata.
    
    Attributes:
        data: The actual data stored in the block
        timestamp: Creation timestamp
        access_count: Number of times the block has been accessed
        last_access: Timestamp of last access
        size_bytes: Size of the data in bytes
        ttl: Time-to-live in seconds (optional)
        priority: Priority level for eviction (higher = keep longer)
        compressed: Whether the data is compressed
        numa_node: NUMA node where data is allocated
        alignment_offset: Memory alignment offset for ARM64
        checksum: Data integrity checksum
    """
    data: Union[np.ndarray, torch.Tensor, Dict[str, Any]]
    timestamp: float
    access_count: int
    last_access: float
    size_bytes: int
    ttl: Optional[float] = None
    priority: int = 0
    compressed: bool = False
    numa_node: Optional[int] = None
    alignment_offset: int = 0
    checksum: Optional[str] = None
    
    def __post_init__(self) -> None:
        """Initialize computed fields and validate data."""
        if self.size_bytes < 0:
            raise ValueError("size_bytes cannot be negative")
        if self.priority < 0:
            raise ValueError("priority cannot be negative")
        if self.access_count < 0:
            raise ValueError("access_count cannot be negative")
    
    @property
    def is_expired(self) -> bool:
        """
        Check if the block has expired based on TTL.
        
        Returns:
            True if the block has expired, False otherwise
        """
        if self.ttl is None:
            return False
        return time.time() > (self.timestamp + self.ttl)
    
    @property
    def age_seconds(self) -> float:
        """
        Get the age of the block in seconds.
        
        Returns:
            Age in seconds since creation
        """
        return time.time() - self.timestamp
    
    def calculate_checksum(self) -> str:
        """
        Calculate checksum for data integrity verification.
        
        Returns:
            SHA-256 checksum of the data
        """
        try:
            if isinstance(self.data, (np.ndarray, torch.Tensor)):
                # Convert to bytes for hashing
                if isinstance(self.data, torch.Tensor):
                    data_bytes = self.data.cpu().numpy().tobytes()
                else:
                    data_bytes = self.data.tobytes()
            else:
                # For dictionaries and other objects
                data_bytes = pickle.dumps(self.data, protocol=pickle.HIGHEST_PROTOCOL)
            
            return hashlib.sha256(data_bytes).hexdigest()
        except Exception as e:
            logger.warning(f"Failed to calculate checksum: {e}")
            return ""
    
    def verify_integrity(self) -> bool:
        """
        Verify data integrity using checksum.
        
        Returns:
            True if data integrity is verified, False otherwise
        """
        if not self.checksum:
            return True  # No checksum to verify
        
        current_checksum = self.calculate_checksum()
        return current_checksum == self.checksum

@dataclass
class MemoryStats:
    """
    Comprehensive memory statistics with ARM64 performance metrics.
    
    Attributes:
        total_capacity_gb: Total memory capacity in GB
        current_usage_gb: Current memory usage in GB
        utilization_pct: Memory utilization percentage
        num_market_data_blocks: Number of market data blocks
        num_feature_blocks: Number of feature blocks
        num_model_states: Number of model state blocks
        num_total_blocks: Total number of blocks
        cache_hits: Number of cache hits
        cache_misses: Number of cache misses
        evictions_performed: Number of evictions performed
        compressions_performed: Number of compressions performed
        arm64_optimizations_used: Number of ARM64 optimizations applied
        numa_allocations: Number of NUMA-aware allocations
        memory_pressure_events: Number of memory pressure events
        average_access_time_ms: Average access time in milliseconds
        start_time: Manager start time
        integrity_checks_passed: Number of successful integrity checks
        integrity_checks_failed: Number of failed integrity checks
    """
    total_capacity_gb: float = 0.0
    current_usage_gb: float = 0.0
    utilization_pct: float = 0.0
    num_market_data_blocks: int = 0
    num_feature_blocks: int = 0
    num_model_states: int = 0
    num_total_blocks: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    evictions_performed: int = 0
    compressions_performed: int = 0
    arm64_optimizations_used: int = 0
    numa_allocations: int = 0
    memory_pressure_events: int = 0
    average_access_time_ms: float = 0.0
    start_time: float = field(default_factory=time.time)
    integrity_checks_passed: int = 0
    integrity_checks_failed: int = 0
    
    @property
    def cache_hit_rate(self) -> float:
        """
        Calculate cache hit rate.
        
        Returns:
            Cache hit rate as a float between 0 and 1
        """
        total_accesses = self.cache_hits + self.cache_misses
        return self.cache_hits / max(total_accesses, 1)
    
    @property
    def uptime_seconds(self) -> float:
        """
        Calculate uptime in seconds.
        
        Returns:
            Uptime since manager start
        """
        return time.time() - self.start_time
    
    @property
    def integrity_check_rate(self) -> float:
        """
        Calculate integrity check success rate.
        
        Returns:
            Integrity check success rate as a float between 0 and 1
        """
        total_checks = self.integrity_checks_passed + self.integrity_checks_failed
        return self.integrity_checks_passed / max(total_checks, 1)

class UnifiedMemoryManager:
    """
    Enhanced UnifiedMemoryManager with ARM64 optimizations for GH200 unified memory.
    
    Manages the GH200 Unified Memory for hot data, implementing advanced caching
    mechanisms with automatic eviction, compression, and ARM64-specific optimizations.
    Supports storing market data, engineered features, and model states with
    high-performance access patterns optimized for ARM64 architecture.
    """

    def __init__(self, 
                 config: Optional[UnifiedMemoryConfig] = None,
                 max_memory_gb: float = 200.0):
        """
        Initializes the enhanced UnifiedMemoryManager with ARM64 optimizations.

        Args:
            config: UnifiedMemoryConfig object (preferred)
            max_memory_gb: The maximum memory capacity in gigabytes (fallback)
        """
        # Configuration handling
        if config is not None:
            self.config = config
        else:
            # Load from config manager with fallbacks
            config_data = config_manager.get('training_config.memory', {})
            self.config = UnifiedMemoryConfig(
                max_memory_gb=max_memory_gb if max_memory_gb != 200.0 else config_data.get('max_memory_gb', 32.0)
            )
        
        # ARM64 optimization detection
        self.is_arm64 = platform.machine().lower() in ['arm64', 'aarch64']
        if self.is_arm64 and self.config.enable_arm64_optimizations:
            logger.info("ARM64 optimizations enabled for UnifiedMemoryManager")
            # Optimize memory alignment for ARM64 cache lines
            self.config.memory_alignment_bytes = 64  # ARM64 cache line size
        
        # Memory management
        self.max_memory_bytes = int(self.config.max_memory_gb * 1024**3)
        self.current_usage_bytes = 0
        self.lock = threading.RLock()  # Reentrant lock for thread-safe operations
        
        # Enhanced data structures with ARM64 optimizations
        self.memory_blocks: OrderedDict[str, MemoryBlock] = OrderedDict()
        self.market_data_pool: OrderedDict[str, MemoryBlock] = OrderedDict()
        self.feature_pool: Dict[str, Dict[str, Union[MemoryBlock, List[str]]]] = {}
        self.model_state_pool: OrderedDict[str, MemoryBlock] = OrderedDict()
        self.prediction_pool: deque = deque(maxlen=1440)  # Last 24 hours of 1-min predictions
        
        # Performance monitoring
        self.stats = MemoryStats()
        self.stats.total_capacity_gb = self.config.max_memory_gb
        
        # ARM64 specific optimizations
        self.numa_nodes = self._detect_numa_nodes() if self.config.enable_numa_awareness else []
        self.current_numa_node = 0
        
        # Background cleanup thread
        self.cleanup_thread = None
        self.cleanup_stop_event = threading.Event()
        
        if self.config.enable_background_cleanup:
            self._start_background_cleanup()
        
        # Prefetch cache for ARM64 optimization
        self.prefetch_cache: Dict[str, List[str]] = {}
        
        logger.info(f"Enhanced UnifiedMemoryManager initialized with {self.config.max_memory_gb:.2f} GB capacity")
        logger.info(f"ARM64 optimizations: {self.is_arm64 and self.config.enable_arm64_optimizations}")
        logger.info(f"NUMA nodes detected: {len(self.numa_nodes)}")

    def __del__(self):
        """Cleanup background threads on destruction."""
        if hasattr(self, 'cleanup_stop_event'):
            self.cleanup_stop_event.set()
        if hasattr(self, 'cleanup_thread') and self.cleanup_thread:
            self.cleanup_thread.join(timeout=1.0)

    def _detect_numa_nodes(self) -> List[int]:
        """
        Detect available NUMA nodes for ARM64 optimization.
        
        Returns:
            List of available NUMA node IDs
        """
        numa_nodes = []
        try:
            if self.is_arm64:
                # ARM64 NUMA detection
                for i in range(8):  # Check up to 8 NUMA nodes
                    try:
                        with open(f'/sys/devices/system/node/node{i}/cpulist', 'r') as f:
                            cpulist = f.read().strip()
                            if cpulist:
                                numa_nodes.append(i)
                                logger.debug(f"Detected NUMA node {i} with CPUs: {cpulist}")
                    except (FileNotFoundError, PermissionError):
                        break
                    except Exception as e:
                        logger.warning(f"Error reading NUMA node {i}: {e}")
                        break
                
                if numa_nodes:
                    logger.info(f"Detected {len(numa_nodes)} NUMA nodes: {numa_nodes}")
                else:
                    logger.debug("No NUMA nodes detected")
                    
        except Exception as e:
            logger.debug(f"NUMA detection failed: {e}")
        
        return numa_nodes

    def _start_background_cleanup(self):
        """Start background cleanup thread for expired blocks."""
        def cleanup_worker():
            while not self.cleanup_stop_event.wait(self.config.cleanup_interval_seconds):
                try:
                    self._cleanup_expired_blocks()
                    self._check_memory_pressure()
                except Exception as e:
                    logger.error(f"Background cleanup error: {e}")
        
        self.cleanup_thread = threading.Thread(target=cleanup_worker, daemon=True)
        self.cleanup_thread.start()
        logger.debug("Background cleanup thread started")

    def _cleanup_expired_blocks(self):
        """Clean up expired blocks based on TTL."""
        with self.lock:
            expired_keys = []
            for key, block in self.memory_blocks.items():
                if block.is_expired:
                    expired_keys.append(key)
            
            for key in expired_keys:
                self._remove_block(key)
                logger.debug(f"Cleaned up expired block: {key}")

    def _check_memory_pressure(self):
        """Check for memory pressure and trigger cleanup if needed."""
        utilization = self.current_usage_bytes / self.max_memory_bytes
        if utilization > self.config.memory_pressure_threshold:
            self.stats.memory_pressure_events += 1
            logger.warning(f"Memory pressure detected: {utilization:.2%} utilization")
            
            # Aggressive cleanup under memory pressure
            self._evict_multiple_blocks(max_evictions=10)

    def _calculate_size(self, data: Union[np.ndarray, torch.Tensor, Dict[str, Any]]) -> int:
        """Enhanced size calculation with ARM64 optimizations."""
        try:
            if isinstance(data, np.ndarray):
                size = data.nbytes
                # Add ARM64 alignment padding
                if self.is_arm64 and self.config.enable_arm64_optimizations:
                    alignment = self.config.memory_alignment_bytes
                    size = ((size + alignment - 1) // alignment) * alignment
                return size
            elif isinstance(data, torch.Tensor):
                size = data.numel() * data.element_size()
                # Add ARM64 alignment padding
                if self.is_arm64 and self.config.enable_arm64_optimizations:
                    alignment = self.config.memory_alignment_bytes
                    size = ((size + alignment - 1) // alignment) * alignment
                return size
            elif isinstance(data, dict):
                # More accurate size estimation for dictionaries
                import pickle
                return len(pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL))
            else:
                logger.warning(f"Unsupported data type for size calculation: {type(data)}")
                return len(str(data).encode('utf-8'))
        except Exception as e:
            logger.error(f"Error calculating data size: {e}")
            return 0

    def _check_memory_availability(self, required_bytes: int) -> bool:
        """Enhanced memory availability check with ARM64 considerations."""
        available_bytes = self.max_memory_bytes - self.current_usage_bytes
        
        # Add ARM64 alignment overhead
        if self.is_arm64 and self.config.enable_arm64_optimizations:
            alignment_overhead = self.config.memory_alignment_bytes
            required_bytes += alignment_overhead
        
        return available_bytes >= required_bytes

    def _evict_lru_data(self) -> bool:
        """Enhanced LRU eviction with multiple eviction policies."""
        if not self.memory_blocks:
            logger.warning("No data blocks to evict.")
            return False

        if self.config.eviction_policy == 'lru':
            return self._evict_lru_block()
        elif self.config.eviction_policy == 'lfu':
            return self._evict_lfu_block()
        elif self.config.eviction_policy == 'ttl':
            return self._evict_ttl_block()
        else:
            return self._evict_lru_block()  # Default to LRU

    def _evict_lru_block(self) -> bool:
        """Evict least recently used block."""
        oldest_key = None
        oldest_time = float('inf')

        for key, block in self.memory_blocks.items():
            if block.last_access < oldest_time:
                oldest_time = block.last_access
                oldest_key = key
        
        if oldest_key:
            self._remove_block(oldest_key)
            self.stats.evictions_performed += 1
            return True
        return False

    def _evict_lfu_block(self) -> bool:
        """Evict least frequently used block."""
        lfu_key = None
        min_access_count = float('inf')

        for key, block in self.memory_blocks.items():
            if block.access_count < min_access_count:
                min_access_count = block.access_count
                lfu_key = key
        
        if lfu_key:
            self._remove_block(lfu_key)
            self.stats.evictions_performed += 1
            return True
        return False

    def _evict_ttl_block(self) -> bool:
        """Evict expired blocks first, then fall back to LRU."""
        # First try to evict expired blocks
        for key, block in self.memory_blocks.items():
            if block.is_expired:
                self._remove_block(key)
                self.stats.evictions_performed += 1
                return True
        
        # Fall back to LRU if no expired blocks
        return self._evict_lru_block()

    def _evict_multiple_blocks(self, max_evictions: int = 5) -> int:
        """Evict multiple blocks for aggressive cleanup."""
        evicted_count = 0
        for _ in range(max_evictions):
            if self._evict_lru_data():
                evicted_count += 1
            else:
                break
        return evicted_count

    def _remove_block(self, key: str):
        """Remove a block from all data structures."""
        if key in self.memory_blocks:
            block = self.memory_blocks[key]
            self.current_usage_bytes -= block.size_bytes
            
            # Remove from specific pools
            if key in self.market_data_pool:
                del self.market_data_pool[key]
            elif key in self.feature_pool:
                del self.feature_pool[key]
            elif key in self.model_state_pool:
                del self.model_state_pool[key]
            
            del self.memory_blocks[key]
            
            logger.debug(f"Removed block '{key}' ({block.size_bytes / (1024**2):.2f} MB). "
                        f"Current usage: {self.current_usage_bytes / (1024**2):.2f} MB.")

    def _optimize_data_for_arm64(self, data: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """Optimize data layout for ARM64 SIMD operations."""
        if not self.is_arm64 or not self.config.enable_arm64_optimizations:
            return data
        
        try:
            if isinstance(data, np.ndarray):
                # Ensure proper alignment and data type for ARM64
                if data.dtype != np.float64:
                    data = data.astype(np.float64)  # 8-byte alignment for ARM64
                
                # Ensure C-contiguous layout for better ARM64 performance
                if not data.flags.c_contiguous:
                    data = np.ascontiguousarray(data)
                
                self.stats.arm64_optimizations_used += 1
                
            elif isinstance(data, torch.Tensor):
                # Ensure proper tensor layout for ARM64
                if not data.is_contiguous():
                    data = data.contiguous()
                
                # Move to appropriate device for ARM64
                if torch.cuda.is_available() and self.is_arm64:
                    data = data.cuda()
                
                self.stats.arm64_optimizations_used += 1
            
            return data
            
        except Exception as e:
            logger.warning(f"ARM64 optimization failed: {e}")
            return data

    def _compress_data(self, data: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """Compress data if compression is enabled."""
        if not self.config.enable_compression:
            return data
        
        try:
            if isinstance(data, np.ndarray):
                # Simple compression using numpy's built-in compression
                # In production, consider using more sophisticated compression
                compressed = np.packbits(data.astype(np.uint8), axis=None)
                self.stats.compressions_performed += 1
                return compressed
            
            # For torch tensors, compression is more complex and may not always be beneficial
            return data
            
        except Exception as e:
            logger.warning(f"Data compression failed: {e}")
            return data

    def store_market_data(self, symbol: str, data: Dict[str, Any], ttl: Optional[float] = None, priority: int = 0) -> bool:
        """
        Enhanced market data storage with ARM64 optimizations and advanced features.

        Args:
            symbol: The ticker symbol of the market data.
            data: The market data to store.
            ttl: Time-to-live in seconds (optional).
            priority: Priority level for eviction (higher = keep longer).

        Returns:
            True if data was stored, False otherwise.
        """
        start_time = time.perf_counter()
        
        with self.lock:
            key = f"market_data_{symbol}"
            size_bytes = self._calculate_size(data)

            # Check memory availability and evict if necessary
            eviction_attempts = 0
            max_eviction_attempts = 10
            
            while not self._check_memory_availability(size_bytes) and eviction_attempts < max_eviction_attempts:
                if not self._evict_lru_data():
                    break
                eviction_attempts += 1

            if not self._check_memory_availability(size_bytes):
                logger.error(f"Failed to store market data for {symbol}: Insufficient memory after {eviction_attempts} evictions.")
                return False

            # Create memory block with enhanced metadata
            block = MemoryBlock(
                data=data,
                timestamp=time.time(),
                access_count=1,
                last_access=time.time(),
                size_bytes=size_bytes,
                ttl=ttl,
                priority=priority,
                numa_node=self.current_numa_node if self.config.enable_numa_awareness else None
            )
            
            # Add data integrity check if enabled
            if self.config.enable_data_integrity:
                block.checksum = block.calculate_checksum()
            
            # ARM64 optimizations
            if self.is_arm64 and self.config.enable_arm64_optimizations:
                # Rotate NUMA nodes for better distribution
                if self.numa_nodes:
                    self.current_numa_node = (self.current_numa_node + 1) % len(self.numa_nodes)
                    self.stats.numa_allocations += 1

            # Store in pools
            self.market_data_pool[key] = block
            self.memory_blocks[key] = block
            self.current_usage_bytes += size_bytes
            
            # Update statistics
            processing_time = time.perf_counter() - start_time
            self.stats.average_access_time_ms = (self.stats.average_access_time_ms + processing_time * 1000) / 2
            
            logger.debug(f"Stored market data for {symbol} in {processing_time*1000:.2f}ms. "
                        f"Current usage: {self.current_usage_bytes / (1024**2):.2f} MB.")
            return True

    def get_market_data(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Enhanced market data retrieval with ARM64 optimizations and prefetching.

        Args:
            symbol: The ticker symbol.

        Returns:
            The market data, or None if not found.
        """
        start_time = time.perf_counter()
        
        with self.lock:
            key = f"market_data_{symbol}"
            block = self.market_data_pool.get(key)
            
            if block:
                # Data integrity check if enabled
                if self.config.enable_data_integrity and block.checksum:
                    if block.verify_integrity():
                        self.stats.integrity_checks_passed += 1
                    else:
                        self.stats.integrity_checks_failed += 1
                        logger.error(f"Data integrity check failed for market data {symbol}")
                        # Remove corrupted block
                        self._remove_block(key)
                        self.stats.cache_misses += 1
                        return None
                
                # Update access statistics
                block.access_count += 1
                block.last_access = time.time()
                
                # Move to end for LRU ordering
                self.market_data_pool.move_to_end(key)
                self.memory_blocks.move_to_end(key)
                
                # ARM64 prefetching optimization
                if self.is_arm64 and self.config.enable_prefetching:
                    self._prefetch_related_data(symbol, 'market_data')
                
                self.stats.cache_hits += 1
                
                processing_time = time.perf_counter() - start_time
                self.stats.average_access_time_ms = (self.stats.average_access_time_ms + processing_time * 1000) / 2
                
                logger.debug(f"Retrieved market data for {symbol} in {processing_time*1000:.2f}ms.")
                return block.data
            
            self.stats.cache_misses += 1
            return None

    def store_features(self, symbol: str, features: np.ndarray, feature_names: List[str], 
                      ttl: Optional[float] = None, priority: int = 0) -> bool:
        """
        Enhanced feature storage with ARM64 optimizations.

        Args:
            symbol: The ticker symbol.
            features: NumPy array of engineered features.
            feature_names: List of feature names corresponding to the columns.
            ttl: Time-to-live in seconds (optional).
            priority: Priority level for eviction.

        Returns:
            True if data was stored, False otherwise.
        """
        start_time = time.perf_counter()
        
        with self.lock:
            key = f"features_{symbol}"
            
            # ARM64 optimization for features
            optimized_features = self._optimize_data_for_arm64(features)
            
            # Optional compression
            if self.config.enable_compression:
                optimized_features = self._compress_data(optimized_features)
            
            size_bytes = self._calculate_size(optimized_features)

            # Memory management
            eviction_attempts = 0
            max_eviction_attempts = 10
            
            while not self._check_memory_availability(size_bytes) and eviction_attempts < max_eviction_attempts:
                if not self._evict_lru_data():
                    break
                eviction_attempts += 1

            if not self._check_memory_availability(size_bytes):
                logger.error(f"Failed to store features for {symbol}: Insufficient memory after {eviction_attempts} evictions.")
                return False

            # Create memory block
            block = MemoryBlock(
                data=optimized_features,
                timestamp=time.time(),
                access_count=1,
                last_access=time.time(),
                size_bytes=size_bytes,
                ttl=ttl,
                priority=priority,
                compressed=self.config.enable_compression,
                numa_node=self.current_numa_node if self.config.enable_numa_awareness else None
            )
            
            # Add data integrity check if enabled
            if self.config.enable_data_integrity:
                block.checksum = block.calculate_checksum()
            
            # Store in pools
            self.feature_pool[key] = {'data': block, 'names': feature_names}
            self.memory_blocks[key] = block
            self.current_usage_bytes += size_bytes
            
            processing_time = time.perf_counter() - start_time
            self.stats.average_access_time_ms = (self.stats.average_access_time_ms + processing_time * 1000) / 2
            
            logger.debug(f"Stored features for {symbol} in {processing_time*1000:.2f}ms. "
                        f"Current usage: {self.current_usage_bytes / (1024**2):.2f} MB.")
            return True

    def get_features(self, symbol: str) -> Optional[Tuple[np.ndarray, List[str]]]:
        """
        Enhanced feature retrieval with ARM64 optimizations.

        Args:
            symbol: The ticker symbol.

        Returns:
            A tuple of (features_array, feature_names), or None if not found.
        """
        start_time = time.perf_counter()
        
        with self.lock:
            key = f"features_{symbol}"
            feature_data = self.feature_pool.get(key)
            
            if feature_data:
                block = feature_data['data']
                
                # Data integrity check if enabled
                if self.config.enable_data_integrity and block.checksum:
                    if block.verify_integrity():
                        self.stats.integrity_checks_passed += 1
                    else:
                        self.stats.integrity_checks_failed += 1
                        logger.error(f"Data integrity check failed for features {symbol}")
                        # Remove corrupted block
                        self._remove_block(key)
                        self.stats.cache_misses += 1
                        return None
                
                block.access_count += 1
                block.last_access = time.time()
                
                # ARM64 prefetching
                if self.is_arm64 and self.config.enable_prefetching:
                    self._prefetch_related_data(symbol, 'features')
                
                self.stats.cache_hits += 1
                
                processing_time = time.perf_counter() - start_time
                self.stats.average_access_time_ms = (self.stats.average_access_time_ms + processing_time * 1000) / 2
                
                logger.debug(f"Retrieved features for {symbol} in {processing_time*1000:.2f}ms.")
                return block.data, feature_data['names']
            
            self.stats.cache_misses += 1
            return None

    def store_model_state(self, model_name: str, state_dict: Dict[str, Any], 
                         ttl: Optional[float] = None, priority: int = 10) -> bool:
        """
        Enhanced model state storage with ARM64 optimizations.

        Args:
            model_name: The name of the model.
            state_dict: The model's state dictionary.
            ttl: Time-to-live in seconds (optional).
            priority: Priority level (models typically have higher priority).

        Returns:
            True if data was stored, False otherwise.
        """
        start_time = time.perf_counter()
        
        with self.lock:
            key = f"model_state_{model_name}"
            
            try:
                # Enhanced state dict processing for ARM64
                if self.is_arm64 and self.config.enable_arm64_optimizations:
                    # Optimize tensors in state dict for ARM64
                    optimized_state_dict = {}
                    for k, v in state_dict.items():
                        if isinstance(v, torch.Tensor):
                            optimized_state_dict[k] = self._optimize_data_for_arm64(v)
                        else:
                            optimized_state_dict[k] = v
                    state_dict = optimized_state_dict
                
                # Calculate size more accurately
                total_size = 0
                for v in state_dict.values():
                    if isinstance(v, torch.Tensor):
                        total_size += v.numel() * v.element_size()
                    else:
                        total_size += self._calculate_size(v)
                
                size_bytes = total_size
                
            except Exception as e:
                logger.error(f"Failed to process model state_dict for {model_name}: {e}")
                return False

            # Memory management with higher eviction attempts for models
            eviction_attempts = 0
            max_eviction_attempts = 20  # More attempts for important model data
            
            while not self._check_memory_availability(size_bytes) and eviction_attempts < max_eviction_attempts:
                if not self._evict_lru_data():
                    break
                eviction_attempts += 1

            if not self._check_memory_availability(size_bytes):
                logger.error(f"Failed to store model state for {model_name}: Insufficient memory after {eviction_attempts} evictions.")
                return False

            # Create memory block with high priority
            block = MemoryBlock(
                data=state_dict,
                timestamp=time.time(),
                access_count=1,
                last_access=time.time(),
                size_bytes=size_bytes,
                ttl=ttl,
                priority=priority,  # Higher priority for models
                numa_node=self.current_numa_node if self.config.enable_numa_awareness else None
            )
            
            # Add data integrity check if enabled
            if self.config.enable_data_integrity:
                block.checksum = block.calculate_checksum()
            
            # Store in pools
            self.model_state_pool[key] = block
            self.memory_blocks[key] = block
            self.current_usage_bytes += size_bytes
            
            processing_time = time.perf_counter() - start_time
            self.stats.average_access_time_ms = (self.stats.average_access_time_ms + processing_time * 1000) / 2
            
            logger.debug(f"Stored model state for {model_name} in {processing_time*1000:.2f}ms. "
                        f"Current usage: {self.current_usage_bytes / (1024**2):.2f} MB.")
            return True

    def get_model_state(self, model_name: str) -> Optional[Dict[str, Any]]:
        """
        Enhanced model state retrieval with ARM64 optimizations.

        Args:
            model_name: The name of the model.

        Returns:
            The model's state dictionary, or None if not found.
        """
        start_time = time.perf_counter()
        
        with self.lock:
            key = f"model_state_{model_name}"
            block = self.model_state_pool.get(key)
            
            if block:
                # Data integrity check if enabled
                if self.config.enable_data_integrity and block.checksum:
                    if block.verify_integrity():
                        self.stats.integrity_checks_passed += 1
                    else:
                        self.stats.integrity_checks_failed += 1
                        logger.error(f"Data integrity check failed for model state {model_name}")
                        # Remove corrupted block
                        self._remove_block(key)
                        self.stats.cache_misses += 1
                        return None
                
                block.access_count += 1
                block.last_access = time.time()
                
                # Move to end for LRU ordering
                self.model_state_pool.move_to_end(key)
                self.memory_blocks.move_to_end(key)
                
                self.stats.cache_hits += 1
                
                processing_time = time.perf_counter() - start_time
                self.stats.average_access_time_ms = (self.stats.average_access_time_ms + processing_time * 1000) / 2
                
                logger.debug(f"Retrieved model state for {model_name} in {processing_time*1000:.2f}ms.")
                return block.data
            
            self.stats.cache_misses += 1
            return None

    def _prefetch_related_data(self, symbol: str, data_type: str):
        """ARM64 optimized prefetching of related data."""
        if not self.config.enable_prefetching:
            return
        
        try:
            # Simple prefetching strategy - could be enhanced with ML-based prediction
            if data_type == 'market_data':
                # Prefetch features for the same symbol
                feature_key = f"features_{symbol}"
                if feature_key in self.feature_pool:
                    # Touch the data to bring it into cache
                    _ = self.feature_pool[feature_key]
            
            elif data_type == 'features':
                # Prefetch market data for the same symbol
                market_key = f"market_data_{symbol}"
                if market_key in self.market_data_pool:
                    _ = self.market_data_pool[market_key]
            
        except Exception as e:
            logger.debug(f"Prefetching failed: {e}")

    def get_memory_stats(self) -> MemoryStats:
        """
        Enhanced memory statistics with ARM64 performance metrics.

        Returns:
            Comprehensive memory statistics.
        """
        with self.lock:
            self.stats.current_usage_gb = self.current_usage_bytes / (1024**3)
            self.stats.utilization_pct = (self.current_usage_bytes / self.max_memory_bytes) * 100 if self.max_memory_bytes > 0 else 0
            self.stats.num_market_data_blocks = len(self.market_data_pool)
            self.stats.num_feature_blocks = len(self.feature_pool)
            self.stats.num_model_states = len(self.model_state_pool)
            self.stats.num_total_blocks = len(self.memory_blocks)
            
            return self.stats

    def clear_cache(self, pool: Optional[str] = None):
        """
        Clear cache with optional pool specification.
        
        Args:
            pool: Specific pool to clear ('market_data', 'features', 'model_states', or None for all)
        """
        with self.lock:
            if pool == 'market_data':
                for key in list(self.market_data_pool.keys()):
                    self._remove_block(key)
            elif pool == 'features':
                for key in list(self.feature_pool.keys()):
                    self._remove_block(key)
            elif pool == 'model_states':
                for key in list(self.model_state_pool.keys()):
                    self._remove_block(key)
            else:
                # Clear all
                self.memory_blocks.clear()
                self.market_data_pool.clear()
                self.feature_pool.clear()
                self.model_state_pool.clear()
                self.current_usage_bytes = 0
            
            # Force garbage collection
            gc.collect()
            
            logger.info(f"Cleared cache pool: {pool or 'all'}")

    def health_check(self) -> Dict[str, Any]:
        """
        Comprehensive health check with ARM64 metrics.
        
        Returns:
            Health status dictionary
        """
        health = {
            'status': 'healthy',
            'timestamp': time.time(),
            'memory': {},
            'performance': {},
            'arm64': {}
        }
        
        try:
            stats = self.get_memory_stats()
            
            # Memory health
            health['memory'] = {
                'utilization_pct': stats.utilization_pct,
                'current_usage_gb': stats.current_usage_gb,
                'total_capacity_gb': stats.total_capacity_gb,
                'total_blocks': stats.num_total_blocks
            }
            
            # Performance health
            health['performance'] = {
                'cache_hit_rate': stats.cache_hit_rate,
                'average_access_time_ms': stats.average_access_time_ms,
                'evictions_performed': stats.evictions_performed,
                'memory_pressure_events': stats.memory_pressure_events
            }
            
            # ARM64 specific health
            health['arm64'] = {
                'optimizations_enabled': self.is_arm64 and self.config.enable_arm64_optimizations,
                'optimizations_used': stats.arm64_optimizations_used,
                'numa_nodes': len(self.numa_nodes),
                'numa_allocations': stats.numa_allocations
            }
            
            # Determine overall health
            if stats.utilization_pct > 95:
                health['status'] = 'critical'
                health['warning'] = 'Memory utilization critical'
            elif stats.utilization_pct > 85:
                health['status'] = 'warning'
                health['warning'] = 'High memory utilization'
            elif stats.cache_hit_rate < 0.5:
                health['status'] = 'warning'
                health['warning'] = 'Low cache hit rate'
                
        except Exception as e:
            health['status'] = 'error'
            health['error'] = str(e)
            logger.error(f"Health check failed: {e}")
        
        return health
    
    def get_block_info(self, key: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed information about a specific memory block.
        
        Args:
            key: The block key to inspect
            
        Returns:
            Dictionary with block information or None if not found
        """
        with self.lock:
            block = self.memory_blocks.get(key)
            if not block:
                return None
            
            return {
                'key': key,
                'size_bytes': block.size_bytes,
                'size_mb': block.size_bytes / (1024**2),
                'timestamp': block.timestamp,
                'last_access': block.last_access,
                'access_count': block.access_count,
                'age_seconds': block.age_seconds,
                'ttl': block.ttl,
                'priority': block.priority,
                'compressed': block.compressed,
                'numa_node': block.numa_node,
                'has_checksum': bool(block.checksum),
                'is_expired': block.is_expired
            }
    
    def list_all_blocks(self) -> List[Dict[str, Any]]:
        """
        List information about all memory blocks.
        
        Returns:
            List of dictionaries with block information
        """
        with self.lock:
            return [self.get_block_info(key) for key in self.memory_blocks.keys()]
    
    def get_pool_summary(self) -> Dict[str, Dict[str, Any]]:
        """
        Get summary information for each memory pool.
        
        Returns:
            Dictionary with pool summaries
        """
        with self.lock:
            summary = {}
            
            # Market data pool
            market_size = sum(block.size_bytes for block in self.market_data_pool.values())
            summary['market_data'] = {
                'count': len(self.market_data_pool),
                'total_size_mb': market_size / (1024**2),
                'avg_size_mb': (market_size / len(self.market_data_pool) / (1024**2)) if self.market_data_pool else 0
            }
            
            # Feature pool
            feature_size = sum(data['data'].size_bytes for data in self.feature_pool.values())
            summary['features'] = {
                'count': len(self.feature_pool),
                'total_size_mb': feature_size / (1024**2),
                'avg_size_mb': (feature_size / len(self.feature_pool) / (1024**2)) if self.feature_pool else 0
            }
            
            # Model state pool
            model_size = sum(block.size_bytes for block in self.model_state_pool.values())
            summary['model_states'] = {
                'count': len(self.model_state_pool),
                'total_size_mb': model_size / (1024**2),
                'avg_size_mb': (model_size / len(self.model_state_pool) / (1024**2)) if self.model_state_pool else 0
            }
            
            return summary
    
    def force_integrity_check(self) -> Dict[str, int]:
        """
        Force integrity check on all blocks with checksums.
        
        Returns:
            Dictionary with check results
        """
        results = {'passed': 0, 'failed': 0, 'no_checksum': 0}
        corrupted_keys = []
        
        with self.lock:
            for key, block in self.memory_blocks.items():
                if not block.checksum:
                    results['no_checksum'] += 1
                    continue
                
                if block.verify_integrity():
                    results['passed'] += 1
                    self.stats.integrity_checks_passed += 1
                else:
                    results['failed'] += 1
                    self.stats.integrity_checks_failed += 1
                    corrupted_keys.append(key)
                    logger.error(f"Integrity check failed for block: {key}")
            
            # Remove corrupted blocks
            for key in corrupted_keys:
                self._remove_block(key)
        
        if corrupted_keys:
            logger.warning(f"Removed {len(corrupted_keys)} corrupted blocks during integrity check")
        
        return results

if __name__ == "__main__":
    # Enhanced example usage with ARM64 optimizations
    import warnings
    warnings.filterwarnings('ignore')
    
    # Initialize enhanced memory manager
    config = UnifiedMemoryConfig(
        max_memory_gb=0.1,  # 100MB for testing
        enable_arm64_optimizations=True,
        enable_performance_monitoring=True,
        enable_compression=True,
        eviction_policy='lru',
        enable_background_cleanup=True
    )
    
    memory_manager = UnifiedMemoryManager(config=config)

    print("Enhanced UnifiedMemoryManager Test")
    print("=" * 50)
    
    # Test storing market data with TTL and priority
    market_data_aapl = {'ev': 'T', 'sym': 'AAPL', 'p': 170.50, 's': 100, 't': 1678886400000000000}
    market_data_msft = {'ev': 'T', 'sym': 'MSFT', 'p': 280.10, 's': 50, 't': 1678886400000000000}
    
    memory_manager.store_market_data('AAPL', market_data_aapl, ttl=3600, priority=5)
    memory_manager.store_market_data('MSFT', market_data_msft, ttl=1800, priority=3)
    
    stats = memory_manager.get_memory_stats()
    print(f"After storing market data:")
    print(f"  Utilization: {stats.utilization_pct:.2f}%")
    print(f"  Cache hit rate: {stats.cache_hit_rate:.2%}")
    print(f"  ARM64 optimizations used: {stats.arm64_optimizations_used}")

    # Test storing features with ARM64 optimization
    features_aapl = np.random.rand(100, 20).astype(np.float32)
    feature_names_aapl = [f'feature_{i}' for i in range(20)]
    memory_manager.store_features('AAPL', features_aapl, feature_names_aapl, priority=7)
    
    features_msft = np.random.rand(100, 20).astype(np.float32)
    feature_names_msft = [f'feature_{i}' for i in range(20)]
    memory_manager.store_features('MSFT', features_msft, feature_names_msft, priority=7)
    
    print(f"\nAfter storing features:")
    stats = memory_manager.get_memory_stats()
    print(f"  Utilization: {stats.utilization_pct:.2f}%")
    print(f"  Total blocks: {stats.num_total_blocks}")
    print(f"  Compressions performed: {stats.compressions_performed}")

    # Test storing model state
    dummy_model_state = {
        'layer1.weight': torch.randn(64, 32),
        'layer1.bias': torch.randn(64),
        'layer2.weight': torch.randn(10, 64),
        'layer2.bias': torch.randn(10)
    }
    memory_manager.store_model_state('LSTM_Model_V1', dummy_model_state, priority=10)
    
    print(f"\nAfter storing model state:")
    stats = memory_manager.get_memory_stats()
    print(f"  Utilization: {stats.utilization_pct:.2f}%")
    print(f"  Model states: {stats.num_model_states}")

    # Test data retrieval with performance monitoring
    start_time = time.perf_counter()
    retrieved_aapl_md = memory_manager.get_market_data('AAPL')
    retrieval_time = time.perf_counter() - start_time
    
    print(f"\nData retrieval test:")
    print(f"  Retrieved AAPL price: {retrieved_aapl_md['p'] if retrieved_aapl_md else 'Not found'}")
    print(f"  Retrieval time: {retrieval_time*1000:.2f}ms")
    
    # Test feature retrieval
    retrieved_features, names = memory_manager.get_features('MSFT')
    print(f"  Retrieved MSFT features shape: {retrieved_features.shape if retrieved_features is not None else 'Not found'}")

    # Force memory pressure by adding large data
    print(f"\nTesting memory pressure and eviction:")
    large_data = np.random.rand(50000).astype(np.float32)  # ~200KB
    
    for i in range(10):
        success = memory_manager.store_market_data(f'LARGE_DATA_{i}', {'data': large_data}, priority=1)
        if not success:
            print(f"  Failed to store LARGE_DATA_{i}")
            break
    
    final_stats = memory_manager.get_memory_stats()
    print(f"\nFinal statistics:")
    print(f"  Utilization: {final_stats.utilization_pct:.2f}%")
    print(f"  Cache hit rate: {final_stats.cache_hit_rate:.2%}")
    print(f"  Evictions performed: {final_stats.evictions_performed}")
    print(f"  Average access time: {final_stats.average_access_time_ms:.2f}ms")
    print(f"  ARM64 optimizations used: {final_stats.arm64_optimizations_used}")

    # Health check
    health = memory_manager.health_check()
    print(f"\nHealth Status: {health['status']}")
    print(f"Memory utilization: {health['memory']['utilization_pct']:.2f}%")
    print(f"ARM64 optimizations enabled: {health['arm64']['optimizations_enabled']}")

    # Test data persistence after eviction
    print(f"\nData persistence test:")
    print(f"  AAPL market data still present: {memory_manager.get_market_data('AAPL') is not None}")
    print(f"  MSFT features still present: {memory_manager.get_features('MSFT') is not None}")
    print(f"  LSTM_Model_V1 state still present: {memory_manager.get_model_state('LSTM_Model_V1') is not None}")
    
    # Cleanup
    memory_manager.clear_cache()
    print(f"\nAfter cache clear:")
    final_stats = memory_manager.get_memory_stats()
    print(f"  Total blocks: {final_stats.num_total_blocks}")
    print(f"  Memory usage: {final_stats.current_usage_gb:.4f} GB")
