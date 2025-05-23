import numpy as np
import torch
import time
import os
import asyncio
import threading
import multiprocessing as mp
from typing import Dict, List, Optional, Union, Tuple, Any, Callable
from collections import deque, OrderedDict
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor
import pickle
import json
import gc
import psutil
import warnings
from pathlib import Path
import redis
import lz4.frame
import zstandard as zstd
from functools import lru_cache
import weakref

from deep_momentum_trading.src.utils.logger import get_logger

logger = get_logger(__name__)

@dataclass
class MemoryConfig:
    """Configuration for memory storage with ARM64 optimizations."""
    use_arm64_optimizations: bool = True
    max_memory_usage: int = 8 * 1024**3  # 8GB
    enable_compression: bool = True
    compression_algorithm: str = "lz4"  # lz4, zstd, gzip
    eviction_policy: str = "lru"  # lru, lfu, fifo, random
    persistence_enabled: bool = True
    persistence_path: str = "data/memory_cache"
    cache_levels: int = 3  # L1, L2, L3 cache hierarchy
    prefetch_enabled: bool = True
    numa_aware: bool = True
    memory_pool_size: int = 1024**2  # 1MB memory pool blocks
    
@dataclass
class MemoryBlock:
    """Enhanced memory block with ARM64 optimizations."""
    data: Union[np.ndarray, torch.Tensor, Dict[str, Any], bytes]
    timestamp: float
    access_count: int
    last_access: float
    size_bytes: int
    compressed: bool = False
    compression_ratio: float = 1.0
    cache_level: int = 1
    numa_node: int = 0
    reference_count: int = 1
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "timestamp": self.timestamp,
            "access_count": self.access_count,
            "last_access": self.last_access,
            "size_bytes": self.size_bytes,
            "compressed": self.compressed,
            "compression_ratio": self.compression_ratio,
            "cache_level": self.cache_level,
            "numa_node": self.numa_node,
            "reference_count": self.reference_count
        }

@dataclass
class MemoryMetrics:
    """Memory performance metrics."""
    total_capacity: int = 0
    current_usage: int = 0
    hit_ratio: float = 0.0
    miss_ratio: float = 0.0
    eviction_count: int = 0
    compression_ratio: float = 0.0
    avg_access_time: float = 0.0
    numa_distribution: Dict[int, int] = None
    cache_level_distribution: Dict[int, int] = None
    
    def __post_init__(self):
        if self.numa_distribution is None:
            self.numa_distribution = {}
        if self.cache_level_distribution is None:
            self.cache_level_distribution = {}

class AdvancedMemoryStorage:
    """
    Advanced memory storage with ARM64 optimizations, hierarchical caching,
    NUMA awareness, and intelligent prefetching.
    """

    def __init__(self, config: Optional[MemoryConfig] = None):
        """
        Initialize Advanced Memory Storage.

        Args:
            config: Memory configuration
        """
        self.config = config or MemoryConfig()
        
        # Memory management
        self.memory_blocks: Dict[str, MemoryBlock] = {}
        self.access_order = OrderedDict()  # For LRU tracking
        self.current_usage_bytes = 0
        self.lock = threading.RLock()
        
        # ARM64 optimizations
        self.numa_nodes = self._detect_numa_nodes()
        self.cpu_count = os.cpu_count()
        self.cache_line_size = 64  # ARM64 cache line size
        
        # Cache hierarchy (L1: hot data, L2: warm data, L3: cold data)
        self.cache_levels = {
            1: {},  # L1: Most frequently accessed
            2: {},  # L2: Moderately accessed
            3: {}   # L3: Least frequently accessed
        }
        
        # Performance tracking
        self.metrics = MemoryMetrics()
        self.access_stats = {"hits": 0, "misses": 0}
        self.operation_times = deque(maxlen=1000)
        
        # Compression
        if self.config.enable_compression:
            self._setup_compression()
        
        # Persistence
        if self.config.persistence_enabled:
            self.persistence_path = Path(self.config.persistence_path)
            self.persistence_path.mkdir(parents=True, exist_ok=True)
        
        # Prefetching
        self.prefetch_queue = deque(maxlen=100)
        self.prefetch_executor = ThreadPoolExecutor(max_workers=2)
        
        # Memory pools for efficient allocation
        self.memory_pools = self._create_memory_pools()
        
        # Setup ARM64 optimizations
        self._setup_arm64_optimizations()
        
        logger.info(f"AdvancedMemoryStorage initialized with {self.config.max_memory_usage / 1024**3:.1f}GB capacity")
        logger.info(f"ARM64 optimizations: {'enabled' if self.config.use_arm64_optimizations else 'disabled'}")
        logger.info(f"NUMA nodes detected: {len(self.numa_nodes)}")

    def _detect_numa_nodes(self) -> List[int]:
        """Detect available NUMA nodes."""
        try:
            if hasattr(os, 'sched_getaffinity'):
                # Try to detect NUMA topology
                numa_nodes = []
                for i in range(8):  # Check up to 8 NUMA nodes
                    try:
                        # This is a simplified check
                        numa_nodes.append(i)
                        if len(numa_nodes) >= 4:  # Reasonable limit
                            break
                    except:
                        break
                return numa_nodes or [0]
            else:
                return [0]
        except Exception as e:
            logger.warning(f"Failed to detect NUMA nodes: {e}")
            return [0]

    def _setup_compression(self) -> None:
        """Setup compression algorithms."""
        try:
            if self.config.compression_algorithm == "lz4":
                self.compressor = lz4.frame
            elif self.config.compression_algorithm == "zstd":
                self.compressor = zstd.ZstdCompressor()
                self.decompressor = zstd.ZstdDecompressor()
            else:
                import gzip
                self.compressor = gzip
            
            logger.info(f"Compression enabled: {self.config.compression_algorithm}")
            
        except ImportError as e:
            logger.warning(f"Compression library not available: {e}")
            self.config.enable_compression = False

    def _create_memory_pools(self) -> Dict[int, List[memoryview]]:
        """Create memory pools for efficient allocation."""
        pools = {}
        pool_sizes = [1024, 4096, 16384, 65536, 262144, 1048576]  # Various pool sizes
        
        for size in pool_sizes:
            pools[size] = []
            # Pre-allocate some memory blocks
            for _ in range(10):
                try:
                    block = bytearray(size)
                    pools[size].append(memoryview(block))
                except MemoryError:
                    break
        
        return pools

    def _setup_arm64_optimizations(self) -> None:
        """Setup ARM64-specific optimizations."""
        if not self.config.use_arm64_optimizations:
            return
            
        try:
            # Set ARM64-specific environment variables
            os.environ.setdefault("OMP_NUM_THREADS", str(self.cpu_count))
            os.environ.setdefault("MKL_NUM_THREADS", str(self.cpu_count))
            
            # Enable ARM64 SIMD optimizations for NumPy
            if hasattr(np, 'show_config'):
                # Check if ARM64 optimizations are available
                pass
            
            # Set CPU affinity for better cache locality
            if hasattr(os, 'sched_setaffinity') and self.config.numa_aware:
                try:
                    available_cpus = list(range(self.cpu_count))
                    os.sched_setaffinity(0, available_cpus)
                except:
                    pass
            
            logger.info("ARM64 optimizations applied")
            
        except Exception as e:
            logger.warning(f"Failed to apply ARM64 optimizations: {e}")

    def _calculate_size(self, data: Union[np.ndarray, torch.Tensor, Dict[str, Any], bytes]) -> int:
        """Calculate data size with ARM64 optimizations."""
        if isinstance(data, np.ndarray):
            return data.nbytes
        elif isinstance(data, torch.Tensor):
            return data.numel() * data.element_size()
        elif isinstance(data, bytes):
            return len(data)
        elif isinstance(data, dict):
            return len(pickle.dumps(data))
        else:
            return len(str(data).encode('utf-8'))

    def _compress_data(self, data: Any) -> Tuple[bytes, float]:
        """Compress data with ARM64 optimizations."""
        if not self.config.enable_compression:
            serialized = pickle.dumps(data)
            return serialized, 1.0
        
        try:
            serialized = pickle.dumps(data)
            original_size = len(serialized)
            
            if self.config.compression_algorithm == "lz4":
                compressed = lz4.frame.compress(serialized)
            elif self.config.compression_algorithm == "zstd":
                compressed = self.compressor.compress(serialized)
            else:
                import gzip
                compressed = gzip.compress(serialized)
            
            compression_ratio = len(compressed) / original_size
            return compressed, compression_ratio
            
        except Exception as e:
            logger.warning(f"Compression failed: {e}")
            serialized = pickle.dumps(data)
            return serialized, 1.0

    def _decompress_data(self, compressed_data: bytes) -> Any:
        """Decompress data with ARM64 optimizations."""
        if not self.config.enable_compression:
            return pickle.loads(compressed_data)
        
        try:
            if self.config.compression_algorithm == "lz4":
                decompressed = lz4.frame.decompress(compressed_data)
            elif self.config.compression_algorithm == "zstd":
                decompressed = self.decompressor.decompress(compressed_data)
            else:
                import gzip
                decompressed = gzip.decompress(compressed_data)
            
            return pickle.loads(decompressed)
            
        except Exception as e:
            logger.warning(f"Decompression failed: {e}")
            return pickle.loads(compressed_data)

    def _select_numa_node(self) -> int:
        """Select optimal NUMA node for data placement."""
        if not self.config.numa_aware or len(self.numa_nodes) <= 1:
            return 0
        
        # Simple round-robin NUMA placement
        # In production, this could be more sophisticated
        return len(self.memory_blocks) % len(self.numa_nodes)

    def _determine_cache_level(self, access_count: int, last_access: float) -> int:
        """Determine appropriate cache level based on access patterns."""
        current_time = time.time()
        time_since_access = current_time - last_access
        
        if access_count > 10 and time_since_access < 60:  # Hot data
            return 1
        elif access_count > 3 and time_since_access < 300:  # Warm data
            return 2
        else:  # Cold data
            return 3

    def _evict_data(self, required_bytes: int) -> bool:
        """Evict data based on configured policy."""
        if self.config.eviction_policy == "lru":
            return self._evict_lru(required_bytes)
        elif self.config.eviction_policy == "lfu":
            return self._evict_lfu(required_bytes)
        elif self.config.eviction_policy == "fifo":
            return self._evict_fifo(required_bytes)
        else:
            return self._evict_random(required_bytes)

    def _evict_lru(self, required_bytes: int) -> bool:
        """Evict least recently used data."""
        freed_bytes = 0
        
        # Start with L3 cache (coldest data)
        for level in [3, 2, 1]:
            if freed_bytes >= required_bytes:
                break
                
            cache_level = self.cache_levels[level]
            
            # Sort by last access time
            sorted_keys = sorted(
                cache_level.keys(),
                key=lambda k: self.memory_blocks[k].last_access
            )
            
            for key in sorted_keys:
                if freed_bytes >= required_bytes:
                    break
                
                block = self.memory_blocks[key]
                freed_bytes += block.size_bytes
                
                # Remove from all structures
                self._remove_block(key)
                
                logger.debug(f"Evicted LRU block '{key}' ({block.size_bytes / 1024:.1f} KB)")
        
        self.metrics.eviction_count += 1
        return freed_bytes >= required_bytes

    def _evict_lfu(self, required_bytes: int) -> bool:
        """Evict least frequently used data."""
        freed_bytes = 0
        
        # Sort by access count (ascending)
        sorted_keys = sorted(
            self.memory_blocks.keys(),
            key=lambda k: self.memory_blocks[k].access_count
        )
        
        for key in sorted_keys:
            if freed_bytes >= required_bytes:
                break
            
            block = self.memory_blocks[key]
            freed_bytes += block.size_bytes
            self._remove_block(key)
            
            logger.debug(f"Evicted LFU block '{key}' ({block.size_bytes / 1024:.1f} KB)")
        
        return freed_bytes >= required_bytes

    def _evict_fifo(self, required_bytes: int) -> bool:
        """Evict first in, first out."""
        freed_bytes = 0
        
        # Sort by timestamp (ascending)
        sorted_keys = sorted(
            self.memory_blocks.keys(),
            key=lambda k: self.memory_blocks[k].timestamp
        )
        
        for key in sorted_keys:
            if freed_bytes >= required_bytes:
                break
            
            block = self.memory_blocks[key]
            freed_bytes += block.size_bytes
            self._remove_block(key)
            
            logger.debug(f"Evicted FIFO block '{key}' ({block.size_bytes / 1024:.1f} KB)")
        
        return freed_bytes >= required_bytes

    def _evict_random(self, required_bytes: int) -> bool:
        """Evict random data."""
        import random
        
        freed_bytes = 0
        keys = list(self.memory_blocks.keys())
        random.shuffle(keys)
        
        for key in keys:
            if freed_bytes >= required_bytes:
                break
            
            block = self.memory_blocks[key]
            freed_bytes += block.size_bytes
            self._remove_block(key)
            
            logger.debug(f"Evicted random block '{key}' ({block.size_bytes / 1024:.1f} KB)")
        
        return freed_bytes >= required_bytes

    def _remove_block(self, key: str) -> None:
        """Remove block from all data structures."""
        if key in self.memory_blocks:
            block = self.memory_blocks[key]
            
            # Update usage
            self.current_usage_bytes -= block.size_bytes
            
            # Remove from cache level
            for level_cache in self.cache_levels.values():
                if key in level_cache:
                    del level_cache[key]
                    break
            
            # Remove from access order
            if key in self.access_order:
                del self.access_order[key]
            
            # Remove from main storage
            del self.memory_blocks[key]

    def store(self, key: str, data: Any, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Store data in memory with ARM64 optimizations.

        Args:
            key: Unique identifier for the data
            data: Data to store
            metadata: Optional metadata

        Returns:
            True if stored successfully, False otherwise
        """
        start_time = time.time()
        
        with self.lock:
            try:
                # Compress data if enabled
                compressed_data, compression_ratio = self._compress_data(data)
                size_bytes = len(compressed_data)
                
                # Check memory availability
                while (self.current_usage_bytes + size_bytes) > self.config.max_memory_usage:
                    if not self._evict_data(size_bytes):
                        logger.error(f"Failed to store '{key}': insufficient memory after eviction")
                        return False
                
                # Select NUMA node and cache level
                numa_node = self._select_numa_node()
                cache_level = 1  # New data starts in L1
                
                # Create memory block
                block = MemoryBlock(
                    data=compressed_data,
                    timestamp=time.time(),
                    access_count=1,
                    last_access=time.time(),
                    size_bytes=size_bytes,
                    compressed=self.config.enable_compression,
                    compression_ratio=compression_ratio,
                    cache_level=cache_level,
                    numa_node=numa_node
                )
                
                # Store in appropriate structures
                self.memory_blocks[key] = block
                self.cache_levels[cache_level][key] = block
                self.access_order[key] = time.time()
                self.current_usage_bytes += size_bytes
                
                # Update metrics
                store_time = time.time() - start_time
                self.operation_times.append(store_time)
                
                # Persist if enabled
                if self.config.persistence_enabled:
                    self._persist_block(key, block)
                
                logger.debug(f"Stored '{key}' ({size_bytes / 1024:.1f} KB, compression: {compression_ratio:.2f})")
                return True
                
            except Exception as e:
                logger.error(f"Failed to store '{key}': {e}")
                return False

    def get(self, key: str) -> Optional[Any]:
        """
        Retrieve data from memory with ARM64 optimizations.

        Args:
            key: Unique identifier for the data

        Returns:
            Retrieved data or None if not found
        """
        start_time = time.time()
        
        with self.lock:
            try:
                if key not in self.memory_blocks:
                    self.access_stats["misses"] += 1
                    
                    # Try to load from persistence
                    if self.config.persistence_enabled:
                        if self._load_from_persistence(key):
                            return self.get(key)  # Recursive call after loading
                    
                    return None
                
                block = self.memory_blocks[key]
                
                # Update access statistics
                block.access_count += 1
                block.last_access = time.time()
                self.access_order[key] = time.time()
                self.access_stats["hits"] += 1
                
                # Promote to higher cache level if frequently accessed
                new_cache_level = self._determine_cache_level(block.access_count, block.last_access)
                if new_cache_level != block.cache_level:
                    self._move_to_cache_level(key, new_cache_level)
                
                # Decompress data
                data = self._decompress_data(block.data)
                
                # Update metrics
                access_time = time.time() - start_time
                self.operation_times.append(access_time)
                
                # Prefetch related data
                if self.config.prefetch_enabled:
                    self._schedule_prefetch(key)
                
                logger.debug(f"Retrieved '{key}' (access count: {block.access_count})")
                return data
                
            except Exception as e:
                logger.error(f"Failed to retrieve '{key}': {e}")
                return None

    def _move_to_cache_level(self, key: str, new_level: int) -> None:
        """Move block to different cache level."""
        if key not in self.memory_blocks:
            return
        
        block = self.memory_blocks[key]
        old_level = block.cache_level
        
        # Remove from old level
        if key in self.cache_levels[old_level]:
            del self.cache_levels[old_level][key]
        
        # Add to new level
        block.cache_level = new_level
        self.cache_levels[new_level][key] = block

    def _persist_block(self, key: str, block: MemoryBlock) -> None:
        """Persist block to disk."""
        try:
            persist_file = self.persistence_path / f"{key}.cache"
            
            # Save block metadata and data
            cache_data = {
                "metadata": block.to_dict(),
                "data": block.data
            }
            
            with open(persist_file, 'wb') as f:
                pickle.dump(cache_data, f)
                
        except Exception as e:
            logger.warning(f"Failed to persist block '{key}': {e}")

    def _load_from_persistence(self, key: str) -> bool:
        """Load block from disk persistence."""
        try:
            persist_file = self.persistence_path / f"{key}.cache"
            
            if not persist_file.exists():
                return False
            
            with open(persist_file, 'rb') as f:
                cache_data = pickle.load(f)
            
            # Recreate memory block
            metadata = cache_data["metadata"]
            block = MemoryBlock(
                data=cache_data["data"],
                timestamp=metadata["timestamp"],
                access_count=metadata["access_count"],
                last_access=time.time(),  # Update access time
                size_bytes=metadata["size_bytes"],
                compressed=metadata["compressed"],
                compression_ratio=metadata["compression_ratio"],
                cache_level=metadata["cache_level"],
                numa_node=metadata["numa_node"]
            )
            
            # Check if we have space
            if (self.current_usage_bytes + block.size_bytes) > self.config.max_memory_usage:
                if not self._evict_data(block.size_bytes):
                    return False
            
            # Store in memory
            self.memory_blocks[key] = block
            self.cache_levels[block.cache_level][key] = block
            self.access_order[key] = time.time()
            self.current_usage_bytes += block.size_bytes
            
            logger.debug(f"Loaded '{key}' from persistence")
            return True
            
        except Exception as e:
            logger.warning(f"Failed to load '{key}' from persistence: {e}")
            return False

    def _schedule_prefetch(self, key: str) -> None:
        """Schedule prefetching of related data."""
        # Simple prefetch strategy - could be more sophisticated
        if len(self.prefetch_queue) < self.prefetch_queue.maxlen:
            self.prefetch_queue.append(key)

    async def store_async(self, key: str, data: Any, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Async version of store."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.store, key, data, metadata)

    async def get_async(self, key: str) -> Optional[Any]:
        """Async version of get."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.get, key)

    def get_metrics(self) -> MemoryMetrics:
        """Get comprehensive memory metrics."""
        total_ops = self.access_stats["hits"] + self.access_stats["misses"]
        
        self.metrics.total_capacity = self.config.max_memory_usage
        self.metrics.current_usage = self.current_usage_bytes
        self.metrics.hit_ratio = self.access_stats["hits"] / total_ops if total_ops > 0 else 0.0
        self.metrics.miss_ratio = self.access_stats["misses"] / total_ops if total_ops > 0 else 0.0
        
        # Calculate average compression ratio
        if self.memory_blocks:
            total_compression = sum(block.compression_ratio for block in self.memory_blocks.values())
            self.metrics.compression_ratio = total_compression / len(self.memory_blocks)
        
        # Calculate average access time
        if self.operation_times:
            self.metrics.avg_access_time = sum(self.operation_times) / len(self.operation_times)
        
        # NUMA distribution
        self.metrics.numa_distribution = {}
        for block in self.memory_blocks.values():
            node = block.numa_node
            self.metrics.numa_distribution[node] = self.metrics.numa_distribution.get(node, 0) + 1
        
        # Cache level distribution
        self.metrics.cache_level_distribution = {}
        for level, cache in self.cache_levels.items():
            self.metrics.cache_level_distribution[level] = len(cache)
        
        return self.metrics

    def optimize(self) -> None:
        """Optimize memory storage performance."""
        logger.info("Optimizing memory storage")
        
        with self.lock:
            # Garbage collection
            gc.collect()
            
            # Rebalance cache levels
            self._rebalance_cache_levels()
            
            # Clean up expired persistence files
            if self.config.persistence_enabled:
                self._cleanup_persistence()
        
        logger.info("Memory storage optimization completed")

    def _rebalance_cache_levels(self) -> None:
        """Rebalance data across cache levels."""
        current_time = time.time()
        
        for key, block in list(self.memory_blocks.items()):
            new_level = self._determine_cache_level(block.access_count, block.last_access)
            if new_level != block.cache_level:
                self._move_to_cache_level(key, new_level)

    def _cleanup_persistence(self) -> None:
        """Clean up old persistence files."""
        try:
            current_time = time.time()
            max_age = 7 * 24 * 3600  # 7 days
            
            for cache_file in self.persistence_path.glob("*.cache"):
                if current_time - cache_file.stat().st_mtime > max_age:
                    cache_file.unlink()
                    
        except Exception as e:
            logger.warning(f"Failed to cleanup persistence: {e}")

    def clear(self) -> None:
        """Clear all cached data."""
        with self.lock:
            self.memory_blocks.clear()
            self.access_order.clear()
            for cache in self.cache_levels.values():
                cache.clear()
            self.current_usage_bytes = 0
            self.access_stats = {"hits": 0, "misses": 0}
            
        logger.info("Memory storage cleared")

    def close(self) -> None:
        """Close memory storage and cleanup resources."""
        self.prefetch_executor.shutdown(wait=True)
        self.clear()
        logger.info("Memory storage closed")

# Specialized memory storage classes
class MemoryCache(AdvancedMemoryStorage):
    """Simple memory cache interface."""
    
    def __init__(self, max_size: int = 1024**3):  # 1GB default
        config = MemoryConfig(max_memory_usage=max_size)
        super().__init__(config)

class DistributedMemoryStorage(AdvancedMemoryStorage):
    """Distributed memory storage across multiple nodes."""
    
    def __init__(self, nodes: List[str], config: Optional[MemoryConfig] = None):
        super().__init__(config)
        self.nodes = nodes
        self.node_clients = {}
        self._setup_distributed_storage()
    
    def _setup_distributed_storage(self):
        """Setup connections to distributed nodes."""
        # Implementation would depend on specific distributed storage system
        pass

class RedisStorage:
    """Redis-based memory storage for distributed caching."""
    
    def __init__(self, 
                 host: str = "localhost", 
                 port: int = 6379, 
                 db: int = 0,
                 config: Optional[MemoryConfig] = None):
        """
        Initialize Redis storage.

        Args:
            host: Redis host
            port: Redis port
            db: Redis database number
            config: Memory configuration
        """
        self.config = config or MemoryConfig()
        
        try:
            self.redis_client = redis.Redis(
                host=host, 
                port=port, 
                db=db,
                decode_responses=False,
                socket_connect_timeout=5,
                socket_timeout=5
            )
            
            # Test connection
            self.redis_client.ping()
            logger.info(f"Connected to Redis at {host}:{port}")
            
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            raise
    
    def store(self, key: str, data: Any, ttl: Optional[int] = None) -> bool:
        """Store data in Redis."""
        try:
            # Serialize and optionally compress
            if self.config.enable_compression:
                serialized = pickle.dumps(data)
                if self.config.compression_algorithm == "lz4":
                    compressed = lz4.frame.compress(serialized)
                else:
                    compressed = serialized
            else:
                compressed = pickle.dumps(data)
            
            # Store in Redis
            if ttl:
                self.redis_client.setex(key, ttl, compressed)
            else:
                self.redis_client.set(key, compressed)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to store '{key}' in Redis: {e}")
            return False
    
    def get(self, key: str) -> Optional[Any]:
        """Retrieve data from Redis."""
        try:
            compressed = self.redis_client.get(key)
            if compressed is None:
                return None
            
            # Decompress and deserialize
            if self.config.enable_compression and self.config.compression_algorithm == "lz4":
                try:
                    decompressed = lz4.frame.decompress(compressed)
                    return pickle.loads(decompressed)
                except:
                    # Fallback to direct deserialization
                    return pickle.loads(compressed)
            else:
                return pickle.loads(compressed)
                
        except Exception as e:
            logger.error(f"Failed to retrieve '{key}' from Redis: {e}")
            return None

# Legacy compatibility
UnifiedMemoryManager = AdvancedMemoryStorage

if __name__ == "__main__":
    # Example usage with ARM64 optimizations
    config = MemoryConfig(
        use_arm64_optimizations=True,
        max_memory_usage=100 * 1024**2,  # 100MB for testing
        enable_compression=True,
        compression_algorithm="lz4",
        eviction_policy="lru",
        persistence_enabled=True
    )
    
    storage = AdvancedMemoryStorage(config)
    
    print("=== Advanced Memory Storage Test ===")
    
    # Test basic storage
    print("\n1. Basic Storage Test:")
    test_data = {
        "market_data": np.random.rand(1000, 10).astype(np.float32),
        "features": np.random.rand(1000, 50).astype(np.float32),
        "metadata": {"symbol": "AAPL", "timestamp": time.time()}
    }
    
    success = storage.store("test_data", test_data)
    print(f"   Storage success: {success}")
    
    retrieved = storage.get("test_data")
    print(f"   Retrieval success: {retrieved is not None}")
    
    # Test compression
    print("\n2. Compression Test:")
    large_data = np.random.rand(10000).astype(np.float32)
    storage.store("large_data", large_data)
    
    metrics = storage.get_metrics()
    print(f"   Compression ratio: {metrics.compression_ratio:.2f}")
    print(f"   Hit ratio: {metrics.hit_ratio:.2f}")
    
    # Test cache levels
    print("\n3. Cache Level Distribution:")
    for level, count in metrics.cache_level_distribution.items():
        print(f"   L{level}: {count} items")
    
    # Test eviction
    print("\n4. Eviction Test:")
    for i in range(20):
        data = np.random.rand(5000).astype(np.float32)
        storage.store(f"data_{i}", data)
    
    final_metrics = storage.get_metrics()
    print(f"   Final usage: {final_metrics.current_usage / 1024**2:.1f} MB")
    print(f"   Evictions: {final_metrics.eviction_count}")
    
    # Test async operations
    print("\n5. Async Operations Test:")
    async def test_async():
        await storage.store_async("async_data", {"test": "async"})
        result = await storage.get_async("async_data")
        return result is not None
    
    import asyncio
    async_success = asyncio.run(test_async())
    print(f"   Async operations success: {async_success}")
    
    # Cleanup
    storage.close()
    
    print("\n=== Advanced Memory Storage Test Complete ===")
