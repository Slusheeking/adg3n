"""
Advanced Storage Package for Deep Momentum Trading System

This package provides high-performance storage solutions with ARM64 optimizations,
comprehensive data management, and production-ready features for the trading system.

Key Features:
- ARM64-optimized storage engines
- Multi-format support (SQLite, HDF5, Parquet, Memory)
- Connection pooling and caching
- Async operations and batch processing
- Data compression and encryption
- Performance monitoring and metrics
- Automatic backup and recovery
- Schema management and migrations
"""

import os
from typing import Dict, Any, Optional, Type, Union
from pathlib import Path
import warnings

# Core storage components
from .sqlite_storage import (
    AdvancedSQLiteStorage, 
    SQLiteTransactionStorage,
    SQLiteTimeSeriesStorage,
    SQLiteModelStorage
)
from .hdf5_storage import (
    AdvancedHDF5Storage,
    HDF5TimeSeriesStorage,
    HDF5ModelStorage,
    HDF5FeatureStorage
)
from .parquet_storage import (
    AdvancedParquetStorage,
    ParquetTimeSeriesStorage,
    ParquetFeatureStorage,
    ParquetAnalyticsStorage
)
from .memory_storage import (
    AdvancedMemoryStorage,
    MemoryCache,
    DistributedMemoryStorage,
    RedisStorage
)

# Storage factory and management
from .storage_factory import StorageFactory, StorageManager
from .storage_base import BaseStorage, StorageConfig, StorageMetrics
from .storage_utils import (
    StorageOptimizer,
    DataCompressor,
    StorageMonitor,
    BackupManager
)

# Get logger
from ..utils.logger import get_logger
logger = get_logger(__name__)

# Package version
__version__ = "2.0.0"

# Default storage configurations
DEFAULT_STORAGE_CONFIG = {
    "sqlite": {
        "use_arm64_optimizations": True,
        "enable_wal_mode": True,
        "connection_pool_size": 10,
        "cache_size": 100000,
        "enable_compression": True,
        "backup_enabled": True
    },
    "hdf5": {
        "use_arm64_optimizations": True,
        "compression": "lz4",
        "chunk_cache_size": 1024**3,  # 1GB
        "enable_swmr": True,
        "fletcher32": True
    },
    "parquet": {
        "use_arm64_optimizations": True,
        "compression": "snappy",
        "row_group_size": 100000,
        "enable_statistics": True,
        "use_dictionary": True
    },
    "memory": {
        "use_arm64_optimizations": True,
        "max_memory_usage": 8 * 1024**3,  # 8GB
        "enable_compression": True,
        "eviction_policy": "lru",
        "persistence_enabled": True
    }
}

# Storage type registry
STORAGE_TYPES = {
    "sqlite": AdvancedSQLiteStorage,
    "hdf5": AdvancedHDF5Storage,
    "parquet": AdvancedParquetStorage,
    "memory": AdvancedMemoryStorage,
    "redis": RedisStorage
}

# Global storage manager instance
_storage_manager: Optional[StorageManager] = None

def get_storage_manager() -> StorageManager:
    """Get the global storage manager instance."""
    global _storage_manager
    if _storage_manager is None:
        _storage_manager = StorageManager()
    return _storage_manager

def create_storage(storage_type: str, 
                  config: Optional[Dict[str, Any]] = None,
                  **kwargs) -> BaseStorage:
    """
    Create a storage instance with ARM64 optimizations.
    
    Args:
        storage_type: Type of storage ('sqlite', 'hdf5', 'parquet', 'memory', 'redis')
        config: Storage configuration
        **kwargs: Additional arguments
        
    Returns:
        Configured storage instance
    """
    if storage_type not in STORAGE_TYPES:
        raise ValueError(f"Unsupported storage type: {storage_type}")
    
    # Merge with default configuration
    default_config = DEFAULT_STORAGE_CONFIG.get(storage_type, {})
    final_config = {**default_config, **(config or {}), **kwargs}
    
    storage_class = STORAGE_TYPES[storage_type]
    return storage_class(config=final_config)

def get_storage(name: str) -> Optional[BaseStorage]:
    """Get a named storage instance from the global manager."""
    return get_storage_manager().get_storage(name)

def register_storage(name: str, storage: BaseStorage) -> None:
    """Register a storage instance with the global manager."""
    get_storage_manager().register_storage(name, storage)

def initialize_storage_system(config_path: Optional[str] = None) -> None:
    """
    Initialize the storage system with configuration.
    
    Args:
        config_path: Path to storage configuration file
    """
    manager = get_storage_manager()
    
    if config_path and os.path.exists(config_path):
        manager.load_config(config_path)
    else:
        # Initialize with default storages
        manager.initialize_default_storages()
    
    logger.info("Storage system initialized successfully")

def optimize_storage_performance() -> None:
    """Optimize storage performance with ARM64-specific tuning."""
    manager = get_storage_manager()
    optimizer = StorageOptimizer()
    
    for storage_name, storage in manager.get_all_storages().items():
        try:
            optimizer.optimize_storage(storage)
            logger.info(f"Optimized storage: {storage_name}")
        except Exception as e:
            logger.warning(f"Failed to optimize storage {storage_name}: {e}")

def get_storage_metrics() -> Dict[str, StorageMetrics]:
    """Get performance metrics for all registered storages."""
    manager = get_storage_manager()
    metrics = {}
    
    for name, storage in manager.get_all_storages().items():
        if hasattr(storage, 'get_metrics'):
            metrics[name] = storage.get_metrics()
    
    return metrics

def cleanup_storage_system() -> None:
    """Clean up the storage system and close all connections."""
    global _storage_manager
    if _storage_manager:
        _storage_manager.cleanup()
        _storage_manager = None
    
    logger.info("Storage system cleaned up")

# Environment setup
def setup_arm64_environment() -> None:
    """Setup ARM64-specific environment optimizations."""
    try:
        # Set ARM64-specific environment variables
        os.environ.setdefault("OMP_NUM_THREADS", str(os.cpu_count()))
        os.environ.setdefault("MKL_NUM_THREADS", str(os.cpu_count()))
        
        # Enable ARM64 optimizations for various libraries
        if hasattr(os, 'sched_setaffinity'):
            # Set CPU affinity for better performance
            available_cpus = list(range(os.cpu_count()))
            os.sched_setaffinity(0, available_cpus)
        
        logger.info("ARM64 environment optimizations applied")
        
    except Exception as e:
        logger.warning(f"Failed to apply ARM64 optimizations: {e}")

# Auto-initialize on import
try:
    setup_arm64_environment()
except Exception as e:
    warnings.warn(f"ARM64 setup failed: {e}")

# Export main components
__all__ = [
    # Storage classes
    "AdvancedSQLiteStorage",
    "SQLiteTransactionStorage", 
    "SQLiteTimeSeriesStorage",
    "SQLiteModelStorage",
    "AdvancedHDF5Storage",
    "HDF5TimeSeriesStorage",
    "HDF5ModelStorage", 
    "HDF5FeatureStorage",
    "AdvancedParquetStorage",
    "ParquetTimeSeriesStorage",
    "ParquetFeatureStorage",
    "ParquetAnalyticsStorage",
    "AdvancedMemoryStorage",
    "MemoryCache",
    "DistributedMemoryStorage",
    "RedisStorage",
    
    # Management classes
    "StorageFactory",
    "StorageManager",
    "BaseStorage",
    "StorageConfig",
    "StorageMetrics",
    
    # Utility classes
    "StorageOptimizer",
    "DataCompressor", 
    "StorageMonitor",
    "BackupManager",
    
    # Functions
    "create_storage",
    "get_storage",
    "register_storage",
    "get_storage_manager",
    "initialize_storage_system",
    "optimize_storage_performance",
    "get_storage_metrics",
    "cleanup_storage_system",
    
    # Constants
    "DEFAULT_STORAGE_CONFIG",
    "STORAGE_TYPES",
    "__version__"
]

# Package metadata
__author__ = "Deep Momentum Trading Team"
__email__ = "team@deepmomentum.ai"
__description__ = "Advanced storage system with ARM64 optimizations for high-performance trading"
__url__ = "https://github.com/deepmomentum/trading-system"
__license__ = "MIT"

# Compatibility check
def check_compatibility() -> bool:
    """Check system compatibility for ARM64 optimizations."""
    import platform
    import sys
    
    system_info = {
        "platform": platform.platform(),
        "architecture": platform.architecture(),
        "python_version": sys.version,
        "cpu_count": os.cpu_count()
    }
    
    logger.info(f"System compatibility check: {system_info}")
    
    # Check for ARM64 architecture
    arch = platform.machine().lower()
    is_arm64 = any(arm_indicator in arch for arm_indicator in ['arm64', 'aarch64', 'arm'])
    
    if is_arm64:
        logger.info("ARM64 architecture detected - optimizations enabled")
    else:
        logger.info("Non-ARM64 architecture - using standard optimizations")
    
    return True

# Run compatibility check on import
try:
    check_compatibility()
except Exception as e:
    logger.warning(f"Compatibility check failed: {e}")

logger.info(f"Deep Momentum Trading Storage Package v{__version__} loaded successfully")
