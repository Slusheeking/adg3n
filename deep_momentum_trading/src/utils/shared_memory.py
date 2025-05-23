"""
Enhanced shared memory management for Deep Momentum Trading System with ARM64 optimizations.

This module provides high-performance shared memory capabilities for inter-process
communication, data sharing, and ARM64-optimized memory operations.
"""

import os
import sys
import mmap
import struct
import threading
import multiprocessing as mp
from multiprocessing import shared_memory
import numpy as np
import time
import weakref
import atexit
from typing import Dict, Any, Optional, Union, Tuple, List, Type, Callable
from dataclasses import dataclass, field
from enum import Enum
import pickle
import json
import hashlib
from pathlib import Path
import psutil
from contextlib import contextmanager
import fcntl
import tempfile

# ARM64 detection
import platform
IS_ARM64 = platform.machine().lower() in ['arm64', 'aarch64']

# Global registry for shared memory objects
_shared_memory_registry: Dict[str, Any] = {}
_registry_lock = threading.RLock()

class MemoryLayout(Enum):
    """Memory layout strategies for ARM64 optimization."""
    CONTIGUOUS = "contiguous"
    INTERLEAVED = "interleaved"
    NUMA_AWARE = "numa_aware"
    CACHE_ALIGNED = "cache_aligned"

class DataType(Enum):
    """Supported data types for shared memory."""
    FLOAT32 = np.float32
    FLOAT64 = np.float64
    INT32 = np.int32
    INT64 = np.int64
    UINT32 = np.uint32
    UINT64 = np.uint64
    BOOL = np.bool_
    BYTES = bytes
    STRING = str

@dataclass
class MemoryConfig:
    """Configuration for shared memory operations."""
    name: str
    size: int
    dtype: Union[DataType, np.dtype] = DataType.FLOAT64
    layout: MemoryLayout = MemoryLayout.CACHE_ALIGNED
    numa_node: Optional[int] = None
    cache_line_size: int = 64  # ARM64 typical cache line size
    alignment: int = 64  # ARM64 SIMD alignment
    enable_numa: bool = True
    enable_prefetch: bool = True
    enable_compression: bool = False
    compression_level: int = 6
    enable_checksums: bool = True
    max_readers: int = 100
    max_writers: int = 10
    timeout: float = 30.0
    auto_cleanup: bool = True

class SharedMemoryError(Exception):
    """Base exception for shared memory operations."""
    pass

class MemoryAllocationError(SharedMemoryError):
    """Raised when memory allocation fails."""
    pass

class MemoryAccessError(SharedMemoryError):
    """Raised when memory access fails."""
    pass

class MemoryCorruptionError(SharedMemoryError):
    """Raised when memory corruption is detected."""
    pass

class SharedArray:
    """High-performance shared numpy array with ARM64 optimizations."""
    
    def __init__(self, 
                 config: MemoryConfig,
                 create: bool = True,
                 initial_data: Optional[np.ndarray] = None):
        self.config = config
        self.name = config.name
        self._shm = None
        self._array = None
        self._lock = threading.RLock()
        self._readers = 0
        self._writers = 0
        self._checksum = None
        
        # ARM64 optimizations
        self._numa_node = self._detect_numa_node() if config.enable_numa else None
        self._cache_line_size = config.cache_line_size
        
        if create:
            self._create_shared_memory(initial_data)
        else:
            self._attach_shared_memory()
        
        # Register for cleanup
        self._register_cleanup()
    
    def _detect_numa_node(self) -> Optional[int]:
        """Detect optimal NUMA node for ARM64 systems."""
        if not IS_ARM64:
            return None
        
        try:
            # Try to get current process NUMA node
            pid = os.getpid()
            numa_info_path = f"/proc/{pid}/numa_maps"
            
            if os.path.exists(numa_info_path):
                with open(numa_info_path, 'r') as f:
                    for line in f:
                        if 'heap' in line and 'N' in line:
                            # Extract NUMA node from line like "N0=123 N1=456"
                            parts = line.split()
                            for part in parts:
                                if part.startswith('N') and '=' in part:
                                    node = int(part.split('=')[0][1:])
                                    return node
        except:
            pass
        
        return 0  # Default to node 0
    
    def _create_shared_memory(self, initial_data: Optional[np.ndarray] = None):
        """Create shared memory segment with ARM64 optimizations."""
        try:
            # Calculate size with alignment
            dtype = self.config.dtype.value if isinstance(self.config.dtype, DataType) else self.config.dtype
            element_size = np.dtype(dtype).itemsize
            
            # Align size to cache line boundaries for ARM64
            aligned_size = self._align_size(self.config.size * element_size)
            
            # Create shared memory
            self._shm = shared_memory.SharedMemory(
                name=self.name,
                create=True,
                size=aligned_size
            )
            
            # Create numpy array view
            self._array = np.ndarray(
                shape=(self.config.size,),
                dtype=dtype,
                buffer=self._shm.buf
            )
            
            # Initialize with data or zeros
            if initial_data is not None:
                if initial_data.size <= self.config.size:
                    self._array[:initial_data.size] = initial_data.flatten()
                    if initial_data.size < self.config.size:
                        self._array[initial_data.size:] = 0
                else:
                    self._array[:] = initial_data.flatten()[:self.config.size]
            else:
                self._array.fill(0)
            
            # Apply ARM64 optimizations
            self._apply_arm64_optimizations()
            
            # Calculate initial checksum
            if self.config.enable_checksums:
                self._update_checksum()
            
        except Exception as e:
            if self._shm:
                try:
                    self._shm.close()
                    self._shm.unlink()
                except:
                    pass
            raise MemoryAllocationError(f"Failed to create shared memory: {e}")
    
    def _attach_shared_memory(self):
        """Attach to existing shared memory segment."""
        try:
            self._shm = shared_memory.SharedMemory(name=self.name)
            
            dtype = self.config.dtype.value if isinstance(self.config.dtype, DataType) else self.config.dtype
            self._array = np.ndarray(
                shape=(self.config.size,),
                dtype=dtype,
                buffer=self._shm.buf
            )
            
            # Verify checksum if enabled
            if self.config.enable_checksums:
                self._verify_checksum()
            
        except Exception as e:
            raise MemoryAccessError(f"Failed to attach to shared memory: {e}")
    
    def _align_size(self, size: int) -> int:
        """Align size to cache line boundaries for ARM64."""
        alignment = self.config.alignment
        return ((size + alignment - 1) // alignment) * alignment
    
    def _apply_arm64_optimizations(self):
        """Apply ARM64-specific memory optimizations."""
        if not IS_ARM64:
            return
        
        try:
            # Set memory layout hints for ARM64
            if self.config.layout == MemoryLayout.NUMA_AWARE and self._numa_node is not None:
                # Try to bind memory to specific NUMA node
                try:
                    import numa
                    numa.set_preferred(self._numa_node)
                except ImportError:
                    pass
            
            # Prefetch data into cache if enabled
            if self.config.enable_prefetch:
                self._prefetch_memory()
            
            # Set memory access patterns for ARM64 cache optimization
            if hasattr(os, 'posix_madvise'):
                try:
                    # Advise sequential access pattern
                    os.posix_madvise(
                        self._shm.buf,
                        len(self._shm.buf),
                        os.POSIX_MADV_SEQUENTIAL
                    )
                except:
                    pass
        
        except Exception:
            # ARM64 optimizations are best-effort
            pass
    
    def _prefetch_memory(self):
        """Prefetch memory into ARM64 cache."""
        if not IS_ARM64:
            return
        
        try:
            # Touch memory pages to bring into cache
            cache_line_size = self._cache_line_size
            buffer_size = len(self._shm.buf)
            
            for offset in range(0, buffer_size, cache_line_size):
                # Read a byte to trigger cache load
                _ = self._shm.buf[offset]
        except:
            pass
    
    def _update_checksum(self):
        """Update checksum for corruption detection."""
        if self._array is not None:
            data_bytes = self._array.tobytes()
            self._checksum = hashlib.md5(data_bytes).hexdigest()
    
    def _verify_checksum(self):
        """Verify data integrity using checksum."""
        if self._array is not None and self._checksum is not None:
            current_checksum = hashlib.md5(self._array.tobytes()).hexdigest()
            if current_checksum != self._checksum:
                raise MemoryCorruptionError("Shared memory corruption detected")
    
    @contextmanager
    def read_lock(self):
        """Context manager for read access."""
        with self._lock:
            if self._writers > 0:
                raise MemoryAccessError("Cannot read while writers are active")
            if self._readers >= self.config.max_readers:
                raise MemoryAccessError("Maximum readers exceeded")
            
            self._readers += 1
        
        try:
            if self.config.enable_checksums:
                self._verify_checksum()
            yield self._array
        finally:
            with self._lock:
                self._readers -= 1
    
    @contextmanager
    def write_lock(self):
        """Context manager for write access."""
        with self._lock:
            if self._readers > 0 or self._writers > 0:
                raise MemoryAccessError("Cannot write while other operations are active")
            if self._writers >= self.config.max_writers:
                raise MemoryAccessError("Maximum writers exceeded")
            
            self._writers += 1
        
        try:
            yield self._array
        finally:
            with self._lock:
                self._writers -= 1
                if self.config.enable_checksums:
                    self._update_checksum()
    
    def read(self) -> np.ndarray:
        """Read data from shared memory."""
        with self.read_lock() as array:
            return array.copy()
    
    def write(self, data: np.ndarray):
        """Write data to shared memory."""
        with self.write_lock() as array:
            if data.size <= array.size:
                array[:data.size] = data.flatten()
                if data.size < array.size:
                    array[data.size:] = 0
            else:
                array[:] = data.flatten()[:array.size]
    
    def update(self, indices: Union[slice, np.ndarray], values: np.ndarray):
        """Update specific indices in shared memory."""
        with self.write_lock() as array:
            array[indices] = values
    
    def get_stats(self) -> Dict[str, Any]:
        """Get memory usage statistics."""
        return {
            "name": self.name,
            "size_bytes": len(self._shm.buf) if self._shm else 0,
            "dtype": str(self._array.dtype) if self._array is not None else None,
            "shape": self._array.shape if self._array is not None else None,
            "numa_node": self._numa_node,
            "readers": self._readers,
            "writers": self._writers,
            "checksum": self._checksum,
            "is_arm64": IS_ARM64
        }
    
    def _register_cleanup(self):
        """Register cleanup function."""
        def cleanup():
            try:
                self.close()
            except:
                pass
        
        atexit.register(cleanup)
        
        # Store weak reference in registry
        with _registry_lock:
            _shared_memory_registry[self.name] = weakref.ref(self)
    
    def close(self):
        """Close shared memory segment."""
        if self._shm:
            try:
                self._shm.close()
            except:
                pass
            self._shm = None
        
        self._array = None
        
        # Remove from registry
        with _registry_lock:
            _shared_memory_registry.pop(self.name, None)
    
    def unlink(self):
        """Unlink shared memory segment."""
        if self._shm:
            try:
                self._shm.unlink()
            except:
                pass
        self.close()
    
    def __del__(self):
        """Cleanup on deletion."""
        try:
            self.close()
        except:
            pass

class SharedQueue:
    """High-performance shared queue with ARM64 optimizations."""
    
    def __init__(self, 
                 name: str,
                 maxsize: int = 1000,
                 item_size: int = 1024,
                 create: bool = True):
        self.name = name
        self.maxsize = maxsize
        self.item_size = item_size
        
        # Calculate total size needed
        header_size = 64  # For metadata
        queue_size = maxsize * item_size
        total_size = header_size + queue_size
        
        # Create shared memory for queue
        self.config = MemoryConfig(
            name=f"{name}_queue",
            size=total_size,
            dtype=DataType.BYTES
        )
        
        if create:
            self._create_queue()
        else:
            self._attach_queue()
    
    def _create_queue(self):
        """Create shared queue."""
        try:
            self._shm = shared_memory.SharedMemory(
                name=self.config.name,
                create=True,
                size=self.config.size
            )
            
            # Initialize header
            header = struct.pack('QQQQ', 0, 0, self.maxsize, self.item_size)
            self._shm.buf[:64] = header
            
        except Exception as e:
            raise MemoryAllocationError(f"Failed to create shared queue: {e}")
    
    def _attach_queue(self):
        """Attach to existing shared queue."""
        try:
            self._shm = shared_memory.SharedMemory(name=self.config.name)
            
            # Read header
            header = struct.unpack('QQQQ', self._shm.buf[:64])
            self.head, self.tail, self.maxsize, self.item_size = header
            
        except Exception as e:
            raise MemoryAccessError(f"Failed to attach to shared queue: {e}")
    
    def put(self, item: bytes, timeout: Optional[float] = None) -> bool:
        """Put item in queue."""
        if len(item) > self.item_size:
            raise ValueError(f"Item size {len(item)} exceeds maximum {self.item_size}")
        
        start_time = time.time()
        
        while True:
            # Read current state
            header = struct.unpack('QQQQ', self._shm.buf[:64])
            head, tail, _, _ = header
            
            # Check if queue is full
            next_tail = (tail + 1) % self.maxsize
            if next_tail == head:
                if timeout is not None and time.time() - start_time > timeout:
                    return False
                time.sleep(0.001)  # Small delay
                continue
            
            # Write item
            offset = 64 + tail * self.item_size
            padded_item = item.ljust(self.item_size, b'\x00')
            self._shm.buf[offset:offset + self.item_size] = padded_item
            
            # Update tail
            new_header = struct.pack('QQQQ', head, next_tail, self.maxsize, self.item_size)
            self._shm.buf[:64] = new_header
            
            return True
    
    def get(self, timeout: Optional[float] = None) -> Optional[bytes]:
        """Get item from queue."""
        start_time = time.time()
        
        while True:
            # Read current state
            header = struct.unpack('QQQQ', self._shm.buf[:64])
            head, tail, _, _ = header
            
            # Check if queue is empty
            if head == tail:
                if timeout is not None and time.time() - start_time > timeout:
                    return None
                time.sleep(0.001)  # Small delay
                continue
            
            # Read item
            offset = 64 + head * self.item_size
            item_data = bytes(self._shm.buf[offset:offset + self.item_size])
            
            # Remove padding
            item = item_data.rstrip(b'\x00')
            
            # Update head
            next_head = (head + 1) % self.maxsize
            new_header = struct.pack('QQQQ', next_head, tail, self.maxsize, self.item_size)
            self._shm.buf[:64] = new_header
            
            return item
    
    def size(self) -> int:
        """Get current queue size."""
        header = struct.unpack('QQQQ', self._shm.buf[:64])
        head, tail, maxsize, _ = header
        return (tail - head) % maxsize
    
    def empty(self) -> bool:
        """Check if queue is empty."""
        return self.size() == 0
    
    def full(self) -> bool:
        """Check if queue is full."""
        header = struct.unpack('QQQQ', self._shm.buf[:64])
        head, tail, maxsize, _ = header
        return (tail + 1) % maxsize == head
    
    def close(self):
        """Close shared queue."""
        if hasattr(self, '_shm') and self._shm:
            try:
                self._shm.close()
            except:
                pass
    
    def unlink(self):
        """Unlink shared queue."""
        if hasattr(self, '_shm') and self._shm:
            try:
                self._shm.unlink()
            except:
                pass
        self.close()

class SharedDict:
    """High-performance shared dictionary with ARM64 optimizations."""
    
    def __init__(self, 
                 name: str,
                 max_items: int = 1000,
                 max_key_size: int = 256,
                 max_value_size: int = 4096,
                 create: bool = True):
        self.name = name
        self.max_items = max_items
        self.max_key_size = max_key_size
        self.max_value_size = max_value_size
        
        # Calculate sizes
        header_size = 64
        item_size = max_key_size + max_value_size + 16  # Extra for metadata
        total_size = header_size + max_items * item_size
        
        self.config = MemoryConfig(
            name=f"{name}_dict",
            size=total_size,
            dtype=DataType.BYTES
        )
        
        if create:
            self._create_dict()
        else:
            self._attach_dict()
    
    def _create_dict(self):
        """Create shared dictionary."""
        try:
            self._shm = shared_memory.SharedMemory(
                name=self.config.name,
                create=True,
                size=self.config.size
            )
            
            # Initialize header (count, max_items, max_key_size, max_value_size)
            header = struct.pack('QQQQ', 0, self.max_items, self.max_key_size, self.max_value_size)
            self._shm.buf[:64] = header
            
        except Exception as e:
            raise MemoryAllocationError(f"Failed to create shared dict: {e}")
    
    def _attach_dict(self):
        """Attach to existing shared dictionary."""
        try:
            self._shm = shared_memory.SharedMemory(name=self.config.name)
            
            # Read header
            header = struct.unpack('QQQQ', self._shm.buf[:64])
            _, self.max_items, self.max_key_size, self.max_value_size = header
            
        except Exception as e:
            raise MemoryAccessError(f"Failed to attach to shared dict: {e}")
    
    def _hash_key(self, key: str) -> int:
        """Hash key to find slot."""
        return hash(key) % self.max_items
    
    def _find_slot(self, key: str) -> Tuple[int, bool]:
        """Find slot for key (slot_index, exists)."""
        start_slot = self._hash_key(key)
        slot = start_slot
        
        while True:
            offset = 64 + slot * (self.max_key_size + self.max_value_size + 16)
            
            # Read slot metadata
            slot_data = self._shm.buf[offset:offset + 16]
            used, key_len, value_len = struct.unpack('QQQ', slot_data[:24])
            
            if not used:
                return slot, False
            
            # Check if key matches
            key_data = self._shm.buf[offset + 16:offset + 16 + key_len]
            if key_data.decode('utf-8') == key:
                return slot, True
            
            # Linear probing
            slot = (slot + 1) % self.max_items
            if slot == start_slot:
                raise MemoryAccessError("Dictionary is full")
    
    def put(self, key: str, value: bytes) -> bool:
        """Put key-value pair in dictionary."""
        if len(key.encode('utf-8')) > self.max_key_size:
            raise ValueError(f"Key size exceeds maximum {self.max_key_size}")
        if len(value) > self.max_value_size:
            raise ValueError(f"Value size exceeds maximum {self.max_value_size}")
        
        slot, exists = self._find_slot(key)
        
        # Calculate offset
        offset = 64 + slot * (self.max_key_size + self.max_value_size + 16)
        
        # Encode key and value
        key_bytes = key.encode('utf-8')
        
        # Write slot data
        slot_metadata = struct.pack('QQQ', 1, len(key_bytes), len(value))
        self._shm.buf[offset:offset + 24] = slot_metadata
        
        # Write key
        self._shm.buf[offset + 16:offset + 16 + len(key_bytes)] = key_bytes
        
        # Write value
        value_offset = offset + 16 + self.max_key_size
        self._shm.buf[value_offset:value_offset + len(value)] = value
        
        # Update count if new item
        if not exists:
            header = struct.unpack('QQQQ', self._shm.buf[:64])
            count = header[0] + 1
            new_header = struct.pack('QQQQ', count, self.max_items, self.max_key_size, self.max_value_size)
            self._shm.buf[:64] = new_header
        
        return True
    
    def get(self, key: str) -> Optional[bytes]:
        """Get value for key."""
        try:
            slot, exists = self._find_slot(key)
            if not exists:
                return None
            
            # Calculate offset
            offset = 64 + slot * (self.max_key_size + self.max_value_size + 16)
            
            # Read value length
            slot_data = self._shm.buf[offset:offset + 24]
            _, _, value_len = struct.unpack('QQQ', slot_data)
            
            # Read value
            value_offset = offset + 16 + self.max_key_size
            return bytes(self._shm.buf[value_offset:value_offset + value_len])
            
        except MemoryAccessError:
            return None
    
    def delete(self, key: str) -> bool:
        """Delete key from dictionary."""
        try:
            slot, exists = self._find_slot(key)
            if not exists:
                return False
            
            # Mark slot as unused
            offset = 64 + slot * (self.max_key_size + self.max_value_size + 16)
            slot_metadata = struct.pack('QQQ', 0, 0, 0)
            self._shm.buf[offset:offset + 24] = slot_metadata
            
            # Update count
            header = struct.unpack('QQQQ', self._shm.buf[:64])
            count = max(0, header[0] - 1)
            new_header = struct.pack('QQQQ', count, self.max_items, self.max_key_size, self.max_value_size)
            self._shm.buf[:64] = new_header
            
            return True
            
        except MemoryAccessError:
            return False
    
    def size(self) -> int:
        """Get number of items in dictionary."""
        header = struct.unpack('QQQQ', self._shm.buf[:64])
        return header[0]
    
    def close(self):
        """Close shared dictionary."""
        if hasattr(self, '_shm') and self._shm:
            try:
                self._shm.close()
            except:
                pass
    
    def unlink(self):
        """Unlink shared dictionary."""
        if hasattr(self, '_shm') and self._shm:
            try:
                self._shm.unlink()
            except:
                pass
        self.close()

def create_shared_array(name: str, 
                       size: int,
                       dtype: Union[DataType, np.dtype] = DataType.FLOAT64,
                       initial_data: Optional[np.ndarray] = None,
                       **kwargs) -> SharedArray:
    """Create shared array with ARM64 optimizations."""
    config = MemoryConfig(name=name, size=size, dtype=dtype, **kwargs)
    return SharedArray(config, create=True, initial_data=initial_data)

def attach_shared_array(name: str, 
                       size: int,
                       dtype: Union[DataType, np.dtype] = DataType.FLOAT64,
                       **kwargs) -> SharedArray:
    """Attach to existing shared array."""
    config = MemoryConfig(name=name, size=size, dtype=dtype, **kwargs)
    return SharedArray(config, create=False)

def create_shared_queue(name: str, 
                       maxsize: int = 1000,
                       item_size: int = 1024) -> SharedQueue:
    """Create shared queue."""
    return SharedQueue(name, maxsize, item_size, create=True)

def attach_shared_queue(name: str, 
                       maxsize: int = 1000,
                       item_size: int = 1024) -> SharedQueue:
    """Attach to existing shared queue."""
    return SharedQueue(name, maxsize, item_size, create=False)

def create_shared_dict(name: str,
                      max_items: int = 1000,
                      max_key_size: int = 256,
                      max_value_size: int = 4096) -> SharedDict:
    """Create shared dictionary."""
    return SharedDict(name, max_items, max_key_size, max_value_size, create=True)

def attach_shared_dict(name: str,
                      max_items: int = 1000,
                      max_key_size: int = 256,
                      max_value_size: int = 4096) -> SharedDict:
    """Attach to existing shared dictionary."""
    return SharedDict(name, max_items, max_key_size, max_value_size, create=False)

def get_system_memory_info() -> Dict[str, Any]:
    """Get comprehensive system memory information."""
    try:
        memory = psutil.virtual_memory()
        swap = psutil.swap_memory()
        
        info = {
            "total_memory_gb": round(memory.total / 1024**3, 2),
            "available_memory_gb": round(memory.available / 1024**3, 2),
            "used_memory_gb": round(memory.used / 1024**3, 2),
            "memory_percent": memory.percent,
            "swap_total_gb": round(swap.total / 1024**3, 2),
            "swap_used_gb": round(swap.used / 1024**3, 2),
            "swap_percent": swap.percent,
            "is_arm64": IS_ARM64
        }
        
        # Add NUMA information for ARM64
        if IS_ARM64:
            try:
                numa_nodes = []
                for node_dir in Path("/sys/devices/system/node").glob("node*"):
                    if node_dir.is_dir():
                        node_id = int(node_dir.name[4:])
                        numa_nodes.append(node_id)
                
                info["numa_nodes"] = sorted(numa_nodes)
                info["numa_enabled"] = len(numa_nodes) > 1
            except:
                info["numa_nodes"] = [0]
                info["numa_enabled"] = False
        
        return info
        
    except Exception as e:
        return {"error": str(e), "is_arm64": IS_ARM64}

def cleanup_shared_memory(name_pattern: Optional[str] = None) -> int:
    """Cleanup shared memory objects."""
    cleaned = 0
    
    with _registry_lock:
        to_remove = []
        
        for name, weak_ref in _shared_memory_registry.items():
            if name_pattern and name_pattern not in name:
                continue
            
            obj = weak_ref()
            if obj is None:
                # Object was garbage collected
                to_remove.append(name)
                cleaned += 1
            else:
                try:
                    obj.close()
                    to_remove.append(name)
                    cleaned += 1
                except:
                    pass
        
        for name in to_remove:
            _shared_memory_registry.pop(name, None)
    
    return cleaned

def list_shared_memory() -> List[Dict[str, Any]]:
    """List all active shared memory objects."""
    objects = []
    
    with _registry_lock:
        for name, weak_ref in _shared_memory_registry.items():
            obj = weak_ref()
            if obj is not None:
                try:
                    stats = obj.get_stats()
                    objects.append(stats)
                except:
                    pass
    
    return objects

# Register cleanup on exit
atexit.register(lambda: cleanup_shared_memory())

# Export all public components
__all__ = [
    "MemoryLayout",
    "DataType", 
    "MemoryConfig",
    "SharedMemoryError",
    "MemoryAllocationError",
    "MemoryAccessError",
    "MemoryCorruptionError",
    "SharedArray",
    "SharedQueue",
    "SharedDict",
    "create_shared_array",
    "attach_shared_array",
    "create_shared_queue",
    "attach_shared_queue", 
    "create_shared_dict",
    "attach_shared_dict",
    "get_system_memory_info",
    "cleanup_shared_memory",
    "list_shared_memory",
    "IS_ARM64"
]

if __name__ == "__main__":
    # Example usage with ARM64 optimizations
    print("Testing enhanced shared memory with ARM64 optimizations...")
    print(f"ARM64 detected: {IS_ARM64}")
    
    # Test shared array
    print("\n1. Testing SharedArray...")
    try:
        # Create shared array
        data = np.random.random(1000).astype(np.float64)
        shared_arr = create_shared_array(
            name="test_array",
            size=1000,
            dtype=DataType.FLOAT64,
            initial_data=data,
            enable_numa=True,
            enable_prefetch=True
        )
        
        print(f"Created shared array: {shared_arr.get_stats()}")
        
        # Test read/write
        with shared_arr.read_lock() as arr:
            print(f"First 5 elements: {arr[:5]}")
        
        with shared_arr.write_lock() as arr:
            arr[0] = 999.0
        
        print(f"Updated first element: {shared_arr.read()[0]}")
        
        shared_arr.unlink()
        print("SharedArray test completed successfully")
        
    except Exception as e:
        print(f"SharedArray test failed: {e}")
    
    # Test shared queue
    print("\n2. Testing SharedQueue...")
    try:
        queue = create_shared_queue("test_queue", maxsize=100, item_size=256)
        
        # Test put/get
        test_data = b"Hello, ARM64 optimized queue!"
        queue.put(test_data)
        
        retrieved = queue.get()
        print(f"Queue test - Put: {test_data}, Got: {retrieved}")
        print(f"Queue size: {queue.size()}, Empty: {queue.empty()}")
        
        queue.unlink()
        print("SharedQueue test completed successfully")
        
    except Exception as e:
        print(f"SharedQueue test failed: {e}")
    
    # Test shared dictionary
    print("\n3. Testing SharedDict...")
    try:
        shared_dict = create_shared_dict("test_dict", max_items=100)
        
        # Test put/get
        shared_dict.put("test_key", b"ARM64 optimized value")
        value = shared_dict.get("test_key")
        
        print(f"Dict test - Key: test_key, Value: {value}")
        print(f"Dict size: {shared_dict.size()}")
        
        shared_dict.unlink()
        print("SharedDict test completed successfully")
        
    except Exception as e:
        print(f"SharedDict test failed: {e}")
    
    # Test system info
    print("\n4. System Memory Information:")
    memory_info = get_system_memory_info()
    for key, value in memory_info.items():
        print(f"  {key}: {value}")
    
    print("\nEnhanced shared memory testing completed!")