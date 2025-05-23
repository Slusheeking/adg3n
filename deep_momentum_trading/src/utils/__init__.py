"""
Enhanced utilities package for Deep Momentum Trading System.

This package provides comprehensive utilities with ARM64 optimizations,
high-performance computing features, and production-ready components.
"""

import os
import sys
import platform
from typing import Dict, Any, Optional, List
import warnings

# Version information
__version__ = "2.0.0"
__author__ = "Deep Momentum Trading Team"
__description__ = "Advanced utilities with ARM64 optimizations for high-frequency trading"

# ARM64 detection and optimization flags
SYSTEM_INFO = {
    "platform": platform.platform(),
    "machine": platform.machine(),
    "processor": platform.processor(),
    "python_version": platform.python_version(),
    "is_arm64": platform.machine().lower() in ['arm64', 'aarch64'],
    "cpu_count": os.cpu_count(),
    "supports_simd": False,
    "supports_numa": False
}

# Detect ARM64 SIMD support
try:
    if SYSTEM_INFO["is_arm64"]:
        # Check for NEON support (ARM64 SIMD)
        with open('/proc/cpuinfo', 'r') as f:
            cpuinfo = f.read()
            SYSTEM_INFO["supports_simd"] = 'neon' in cpuinfo.lower()
except:
    pass

# Detect NUMA support
try:
    import numa
    SYSTEM_INFO["supports_numa"] = numa.available()
except ImportError:
    try:
        # Alternative NUMA detection
        numa_nodes = len([d for d in os.listdir('/sys/devices/system/node/') 
                         if d.startswith('node')])
        SYSTEM_INFO["supports_numa"] = numa_nodes > 1
    except:
        pass

# Core imports with error handling
try:
    from .logger import get_logger, setup_logging, LogConfig
    from .constants import *
    from .exceptions import *
    from .validators import *
    from .decorators import *
    from .helpers import *
    from .shared_memory import *
    from .visuals import TrainingVisualizer, create_training_visualizations
    
    # Initialize logging
    logger = get_logger(__name__)
    logger.info(f"Utils package initialized v{__version__}")
    logger.info(f"System: {SYSTEM_INFO['platform']}")
    logger.info(f"ARM64 optimizations: {'enabled' if SYSTEM_INFO['is_arm64'] else 'disabled'}")
    logger.info(f"SIMD support: {'enabled' if SYSTEM_INFO['supports_simd'] else 'disabled'}")
    logger.info(f"NUMA support: {'enabled' if SYSTEM_INFO['supports_numa'] else 'disabled'}")
    
except ImportError as e:
    warnings.warn(f"Failed to import some utilities: {e}")

# Utility factory functions
def create_logger(name: str, config: Optional[Dict[str, Any]] = None):
    """Create a configured logger instance."""
    from .logger import get_logger, LogConfig
    
    if config:
        log_config = LogConfig(**config)
        return get_logger(name, log_config)
    return get_logger(name)

def create_validator(validator_type: str, **kwargs):
    """Create a validator instance."""
    from .validators import ValidatorFactory
    return ValidatorFactory.create_validator(validator_type, **kwargs)

def create_shared_memory(name: str, size: int, **kwargs):
    """Create a shared memory instance."""
    from .shared_memory import SharedMemoryManager
    return SharedMemoryManager.create_memory_block(name, size, **kwargs)

def get_system_info() -> Dict[str, Any]:
    """Get comprehensive system information."""
    return SYSTEM_INFO.copy()

def optimize_for_arm64() -> bool:
    """Apply ARM64-specific optimizations."""
    if not SYSTEM_INFO["is_arm64"]:
        return False
    
    try:
        # Set ARM64-specific environment variables
        os.environ.setdefault("OMP_NUM_THREADS", str(SYSTEM_INFO["cpu_count"]))
        os.environ.setdefault("MKL_NUM_THREADS", str(SYSTEM_INFO["cpu_count"]))
        os.environ.setdefault("OPENBLAS_NUM_THREADS", str(SYSTEM_INFO["cpu_count"]))
        
        # Enable ARM64 SIMD optimizations
        if SYSTEM_INFO["supports_simd"]:
            os.environ.setdefault("NPY_DISABLE_CPU_FEATURES", "")
            os.environ.setdefault("NNPACK_BACKEND", "auto")
        
        # NUMA optimizations
        if SYSTEM_INFO["supports_numa"]:
            os.environ.setdefault("NUMA_POLICY", "interleave")
        
        logger.info("ARM64 optimizations applied")
        return True
        
    except Exception as e:
        logger.warning(f"Failed to apply ARM64 optimizations: {e}")
        return False

# Performance monitoring
class PerformanceMonitor:
    """Global performance monitoring for utilities."""
    
    def __init__(self):
        self.metrics = {
            "function_calls": {},
            "execution_times": {},
            "memory_usage": {},
            "errors": {}
        }
    
    def record_call(self, function_name: str, execution_time: float):
        """Record function call metrics."""
        if function_name not in self.metrics["function_calls"]:
            self.metrics["function_calls"][function_name] = 0
            self.metrics["execution_times"][function_name] = []
        
        self.metrics["function_calls"][function_name] += 1
        self.metrics["execution_times"][function_name].append(execution_time)
    
    def record_error(self, function_name: str, error_type: str):
        """Record error metrics."""
        if function_name not in self.metrics["errors"]:
            self.metrics["errors"][function_name] = {}
        
        if error_type not in self.metrics["errors"][function_name]:
            self.metrics["errors"][function_name][error_type] = 0
        
        self.metrics["errors"][function_name][error_type] += 1
    
    def get_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        stats = {}
        
        for func_name, times in self.metrics["execution_times"].items():
            if times:
                stats[func_name] = {
                    "call_count": self.metrics["function_calls"][func_name],
                    "avg_time": sum(times) / len(times),
                    "min_time": min(times),
                    "max_time": max(times),
                    "total_time": sum(times)
                }
        
        return stats

# Global performance monitor
performance_monitor = PerformanceMonitor()

# Utility configuration
class UtilsConfig:
    """Global configuration for utilities package."""
    
    def __init__(self):
        self.enable_arm64_optimizations = SYSTEM_INFO["is_arm64"]
        self.enable_performance_monitoring = True
        self.enable_memory_optimization = True
        self.enable_simd = SYSTEM_INFO["supports_simd"]
        self.enable_numa = SYSTEM_INFO["supports_numa"]
        self.log_level = "INFO"
        self.max_memory_usage = 8 * 1024**3  # 8GB
        self.thread_pool_size = SYSTEM_INFO["cpu_count"]
    
    def apply_optimizations(self):
        """Apply all configured optimizations."""
        if self.enable_arm64_optimizations:
            optimize_for_arm64()
        
        # Additional optimizations can be added here

# Global configuration instance
config = UtilsConfig()

# Auto-apply optimizations on import
if config.enable_arm64_optimizations:
    config.apply_optimizations()

# Export all public components
__all__ = [
    # Version info
    "__version__",
    "__author__",
    "__description__",
    
    # System info
    "SYSTEM_INFO",
    "get_system_info",
    "optimize_for_arm64",
    
    # Factory functions
    "create_logger",
    "create_validator", 
    "create_shared_memory",
    
    # Performance monitoring
    "PerformanceMonitor",
    "performance_monitor",
    
    # Configuration
    "UtilsConfig",
    "config",
    
    # Core modules (imported from submodules)
    "get_logger",
    "setup_logging",
    "LogConfig",
    
    # Constants
    "TradingConstants",
    "ModelConstants", 
    "SystemConstants",
    
    # Exceptions
    "TradingSystemError",
    "ModelError",
    "DataError",
    "ValidationError",
    
    # Validators
    "DataValidator",
    "ModelValidator",
    "TradingValidator",
    "ValidatorFactory",
    
    # Decorators
    "timing_decorator",
    "retry_decorator",
    "cache_decorator",
    "arm64_optimized",
    "performance_monitor_decorator",
    
    # Helpers
    "format_currency",
    "calculate_returns",
    "normalize_data",
    "ARM64Optimizer",
    "PerformanceProfiler",
    
    # Shared memory
    "SharedMemoryManager",
    "SharedMemoryBlock",
    "DistributedMemory"
]

# Cleanup
def cleanup():
    """Cleanup utility resources."""
    try:
        # Cleanup shared memory
        from .shared_memory import SharedMemoryManager
        SharedMemoryManager.cleanup_all()
        
        # Log final performance stats
        if performance_monitor:
            stats = performance_monitor.get_stats()
            if stats:
                logger.info(f"Final performance stats: {stats}")
        
        logger.info("Utils package cleanup completed")
        
    except Exception as e:
        logger.warning(f"Error during cleanup: {e}")

# Register cleanup function
import atexit
atexit.register(cleanup)