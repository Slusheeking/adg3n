"""
Enhanced helper functions for Deep Momentum Trading System with ARM64 optimizations.

This module provides comprehensive utility functions for data processing,
financial calculations, performance optimization, and ARM64-specific operations.
"""

import os
import time
import threading
import multiprocessing as mp
from typing import Any, Dict, List, Optional, Union, Tuple, Callable
from datetime import datetime, timedelta, timezone
from decimal import Decimal, ROUND_HALF_UP
import numpy as np
import pandas as pd
import json
import hashlib
import pickle
import gzip
import lz4.frame
import zstandard as zstd
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import psutil
import gc
from dataclasses import dataclass
from functools import lru_cache, wraps
import warnings

from .logger import get_logger
from .constants import SystemArchitecture, TradingConstants, ARM64OptimizationConstants
from .decorators import timing_decorator, arm64_optimized

logger = get_logger(__name__)

# Financial calculation helpers
def format_currency(amount: Union[float, Decimal], 
                   currency: str = "USD",
                   precision: int = 2) -> str:
    """
    Format currency with proper precision and symbols.
    
    Args:
        amount: Amount to format
        currency: Currency code
        precision: Decimal precision
        
    Returns:
        Formatted currency string
    """
    if isinstance(amount, float):
        amount = Decimal(str(amount))
    
    # Round to specified precision
    rounded = amount.quantize(Decimal('0.' + '0' * precision), rounding=ROUND_HALF_UP)
    
    # Currency symbols
    symbols = {
        "USD": "$",
        "EUR": "€", 
        "GBP": "£",
        "JPY": "¥",
        "CAD": "C$",
        "AUD": "A$"
    }
    
    symbol = symbols.get(currency, currency)
    
    # Format with thousands separator
    formatted = f"{symbol}{rounded:,.{precision}f}"
    
    return formatted

@arm64_optimized(enable_simd=True)
def calculate_returns(prices: Union[np.ndarray, pd.Series],
                     method: str = "simple",
                     periods: int = 1) -> np.ndarray:
    """
    Calculate returns with ARM64 SIMD optimizations.
    
    Args:
        prices: Price series
        method: 'simple' or 'log' returns
        periods: Number of periods for return calculation
        
    Returns:
        Array of returns
    """
    if isinstance(prices, pd.Series):
        prices = prices.values
    
    if len(prices) < periods + 1:
        return np.array([])
    
    # ARM64 optimized calculations
    if SystemArchitecture.IS_ARM64:
        # Use ARM64 NEON optimizations
        prices = prices.astype(np.float32)  # Better NEON performance
    
    if method == "simple":
        returns = (prices[periods:] - prices[:-periods]) / prices[:-periods]
    elif method == "log":
        returns = np.log(prices[periods:] / prices[:-periods])
    else:
        raise ValueError(f"Unknown return method: {method}")
    
    return returns

@arm64_optimized(enable_simd=True)
def calculate_volatility(returns: Union[np.ndarray, pd.Series],
                        window: int = 252,
                        annualize: bool = True) -> float:
    """
    Calculate volatility with ARM64 optimizations.
    
    Args:
        returns: Return series
        window: Rolling window size
        annualize: Whether to annualize volatility
        
    Returns:
        Volatility value
    """
    if isinstance(returns, pd.Series):
        returns = returns.values
    
    if SystemArchitecture.IS_ARM64:
        returns = returns.astype(np.float32)
    
    # Calculate standard deviation
    volatility = np.std(returns, ddof=1)
    
    if annualize:
        volatility *= np.sqrt(window)
    
    return float(volatility)

def calculate_sharpe_ratio(returns: Union[np.ndarray, pd.Series],
                          risk_free_rate: float = 0.02,
                          periods_per_year: int = 252) -> float:
    """
    Calculate Sharpe ratio.
    
    Args:
        returns: Return series
        risk_free_rate: Risk-free rate (annual)
        periods_per_year: Number of periods per year
        
    Returns:
        Sharpe ratio
    """
    if isinstance(returns, pd.Series):
        returns = returns.values
    
    # Convert risk-free rate to period rate
    rf_period = risk_free_rate / periods_per_year
    
    # Calculate excess returns
    excess_returns = returns - rf_period
    
    # Calculate Sharpe ratio
    if np.std(excess_returns) == 0:
        return 0.0
    
    sharpe = np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(periods_per_year)
    
    return float(sharpe)

def calculate_max_drawdown(prices: Union[np.ndarray, pd.Series]) -> Tuple[float, int, int]:
    """
    Calculate maximum drawdown.
    
    Args:
        prices: Price series
        
    Returns:
        Tuple of (max_drawdown, start_index, end_index)
    """
    if isinstance(prices, pd.Series):
        prices = prices.values
    
    # Calculate running maximum
    running_max = np.maximum.accumulate(prices)
    
    # Calculate drawdown
    drawdown = (prices - running_max) / running_max
    
    # Find maximum drawdown
    max_dd_idx = np.argmin(drawdown)
    max_drawdown = drawdown[max_dd_idx]
    
    # Find start of drawdown period
    start_idx = np.argmax(running_max[:max_dd_idx + 1] == running_max[max_dd_idx])
    
    return float(max_drawdown), int(start_idx), int(max_dd_idx)

# Data processing helpers
@arm64_optimized(enable_simd=True)
def normalize_data(data: Union[np.ndarray, pd.DataFrame],
                  method: str = "zscore",
                  axis: int = 0) -> Union[np.ndarray, pd.DataFrame]:
    """
    Normalize data with ARM64 optimizations.
    
    Args:
        data: Data to normalize
        method: Normalization method ('zscore', 'minmax', 'robust')
        axis: Axis along which to normalize
        
    Returns:
        Normalized data
    """
    if isinstance(data, pd.DataFrame):
        return data.apply(lambda x: normalize_data(x.values, method, axis=0))
    
    if SystemArchitecture.IS_ARM64:
        data = data.astype(np.float32)
    
    if method == "zscore":
        mean = np.mean(data, axis=axis, keepdims=True)
        std = np.std(data, axis=axis, keepdims=True)
        normalized = (data - mean) / (std + 1e-8)
        
    elif method == "minmax":
        min_val = np.min(data, axis=axis, keepdims=True)
        max_val = np.max(data, axis=axis, keepdims=True)
        normalized = (data - min_val) / (max_val - min_val + 1e-8)
        
    elif method == "robust":
        median = np.median(data, axis=axis, keepdims=True)
        mad = np.median(np.abs(data - median), axis=axis, keepdims=True)
        normalized = (data - median) / (mad + 1e-8)
        
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    
    return normalized

def create_lagged_features(data: pd.DataFrame,
                          columns: List[str],
                          lags: List[int]) -> pd.DataFrame:
    """
    Create lagged features for time series data.
    
    Args:
        data: Input DataFrame
        columns: Columns to create lags for
        lags: List of lag periods
        
    Returns:
        DataFrame with lagged features
    """
    result = data.copy()
    
    for col in columns:
        if col not in data.columns:
            continue
            
        for lag in lags:
            lag_col = f"{col}_lag_{lag}"
            result[lag_col] = data[col].shift(lag)
    
    return result

def create_rolling_features(data: pd.DataFrame,
                           columns: List[str],
                           windows: List[int],
                           functions: List[str] = None) -> pd.DataFrame:
    """
    Create rolling window features.
    
    Args:
        data: Input DataFrame
        columns: Columns to create rolling features for
        windows: List of window sizes
        functions: List of functions to apply ('mean', 'std', 'min', 'max')
        
    Returns:
        DataFrame with rolling features
    """
    if functions is None:
        functions = ['mean', 'std']
    
    result = data.copy()
    
    for col in columns:
        if col not in data.columns:
            continue
            
        for window in windows:
            rolling = data[col].rolling(window=window)
            
            for func in functions:
                if hasattr(rolling, func):
                    feature_col = f"{col}_rolling_{window}_{func}"
                    result[feature_col] = getattr(rolling, func)()
    
    return result

# Performance optimization helpers
class ARM64Optimizer:
    """ARM64-specific optimization utilities."""
    
    @staticmethod
    def optimize_numpy_array(arr: np.ndarray) -> np.ndarray:
        """Optimize NumPy array for ARM64 NEON."""
        if not SystemArchitecture.IS_ARM64:
            return arr
        
        # Use float32 for better NEON performance
        if arr.dtype == np.float64:
            arr = arr.astype(np.float32)
        
        # Ensure memory alignment for SIMD
        if not arr.flags.aligned:
            arr = np.ascontiguousarray(arr)
        
        return arr
    
    @staticmethod
    def set_thread_affinity(cpu_list: Optional[List[int]] = None):
        """Set thread affinity for ARM64 optimization."""
        if not SystemArchitecture.IS_ARM64:
            return
        
        try:
            if cpu_list is None:
                cpu_list = list(range(SystemArchitecture.CPU_COUNT))
            
            os.sched_setaffinity(0, cpu_list)
            logger.debug(f"Set thread affinity to CPUs: {cpu_list}")
            
        except Exception as e:
            logger.warning(f"Failed to set thread affinity: {e}")
    
    @staticmethod
    def enable_numa_optimization():
        """Enable NUMA optimizations for ARM64."""
        if not SystemArchitecture.IS_ARM64 or SystemArchitecture.NUMA_NODES <= 1:
            return
        
        try:
            # Set NUMA memory policy
            os.environ["NUMA_POLICY"] = "interleave"
            os.environ["OMP_PROC_BIND"] = "true"
            os.environ["OMP_PLACES"] = "cores"
            
            logger.debug("NUMA optimizations enabled")
            
        except Exception as e:
            logger.warning(f"Failed to enable NUMA optimizations: {e}")

class PerformanceProfiler:
    """Performance profiling utilities."""
    
    def __init__(self):
        self.profiles = {}
        self.lock = threading.Lock()
    
    def profile_function(self, func_name: str):
        """Decorator to profile function performance."""
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                start_time = time.perf_counter()
                start_memory = self._get_memory_usage()
                
                try:
                    result = func(*args, **kwargs)
                    
                    end_time = time.perf_counter()
                    end_memory = self._get_memory_usage()
                    
                    execution_time = end_time - start_time
                    memory_delta = end_memory - start_memory
                    
                    with self.lock:
                        if func_name not in self.profiles:
                            self.profiles[func_name] = {
                                "call_count": 0,
                                "total_time": 0.0,
                                "total_memory": 0.0,
                                "min_time": float('inf'),
                                "max_time": 0.0
                            }
                        
                        profile = self.profiles[func_name]
                        profile["call_count"] += 1
                        profile["total_time"] += execution_time
                        profile["total_memory"] += memory_delta
                        profile["min_time"] = min(profile["min_time"], execution_time)
                        profile["max_time"] = max(profile["max_time"], execution_time)
                    
                    return result
                    
                except Exception as e:
                    logger.error(f"Error in profiled function {func_name}: {e}")
                    raise
            
            return wrapper
        return decorator
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        try:
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024
        except:
            return 0.0
    
    def get_profile_stats(self) -> Dict[str, Any]:
        """Get profiling statistics."""
        with self.lock:
            stats = {}
            for func_name, profile in self.profiles.items():
                if profile["call_count"] > 0:
                    stats[func_name] = {
                        "call_count": profile["call_count"],
                        "avg_time": profile["total_time"] / profile["call_count"],
                        "total_time": profile["total_time"],
                        "min_time": profile["min_time"],
                        "max_time": profile["max_time"],
                        "avg_memory": profile["total_memory"] / profile["call_count"]
                    }
            return stats
    
    def clear_profiles(self):
        """Clear all profiling data."""
        with self.lock:
            self.profiles.clear()

# Utility functions
def get_timestamp(timezone_aware: bool = True) -> Union[datetime, float]:
    """
    Get current timestamp.
    
    Args:
        timezone_aware: Whether to return timezone-aware datetime
        
    Returns:
        Timestamp as datetime or float
    """
    if timezone_aware:
        return datetime.now(timezone.utc)
    else:
        return time.time()

def format_timestamp(timestamp: Union[datetime, float],
                    format_str: str = "%Y-%m-%d %H:%M:%S") -> str:
    """
    Format timestamp to string.
    
    Args:
        timestamp: Timestamp to format
        format_str: Format string
        
    Returns:
        Formatted timestamp string
    """
    if isinstance(timestamp, float):
        timestamp = datetime.fromtimestamp(timestamp, tz=timezone.utc)
    
    return timestamp.strftime(format_str)

def parse_timestamp(timestamp_str: str,
                   format_str: str = "%Y-%m-%d %H:%M:%S") -> datetime:
    """
    Parse timestamp string to datetime.
    
    Args:
        timestamp_str: Timestamp string
        format_str: Format string
        
    Returns:
        Parsed datetime
    """
    return datetime.strptime(timestamp_str, format_str)

def generate_hash(data: Any, algorithm: str = "md5") -> str:
    """
    Generate hash for data.
    
    Args:
        data: Data to hash
        algorithm: Hash algorithm ('md5', 'sha1', 'sha256')
        
    Returns:
        Hash string
    """
    # Serialize data
    if isinstance(data, (dict, list)):
        data_str = json.dumps(data, sort_keys=True)
    else:
        data_str = str(data)
    
    # Generate hash
    if algorithm == "md5":
        return hashlib.md5(data_str.encode()).hexdigest()
    elif algorithm == "sha1":
        return hashlib.sha1(data_str.encode()).hexdigest()
    elif algorithm == "sha256":
        return hashlib.sha256(data_str.encode()).hexdigest()
    else:
        raise ValueError(f"Unsupported hash algorithm: {algorithm}")

def compress_data(data: bytes, algorithm: str = "lz4") -> bytes:
    """
    Compress data using specified algorithm.
    
    Args:
        data: Data to compress
        algorithm: Compression algorithm ('lz4', 'zstd', 'gzip')
        
    Returns:
        Compressed data
    """
    if algorithm == "lz4":
        return lz4.frame.compress(data)
    elif algorithm == "zstd":
        compressor = zstd.ZstdCompressor()
        return compressor.compress(data)
    elif algorithm == "gzip":
        return gzip.compress(data)
    else:
        raise ValueError(f"Unsupported compression algorithm: {algorithm}")

def decompress_data(compressed_data: bytes, algorithm: str = "lz4") -> bytes:
    """
    Decompress data using specified algorithm.
    
    Args:
        compressed_data: Compressed data
        algorithm: Compression algorithm ('lz4', 'zstd', 'gzip')
        
    Returns:
        Decompressed data
    """
    if algorithm == "lz4":
        return lz4.frame.decompress(compressed_data)
    elif algorithm == "zstd":
        decompressor = zstd.ZstdDecompressor()
        return decompressor.decompress(compressed_data)
    elif algorithm == "gzip":
        return gzip.decompress(compressed_data)
    else:
        raise ValueError(f"Unsupported compression algorithm: {algorithm}")

def serialize_object(obj: Any, format: str = "pickle") -> bytes:
    """
    Serialize object to bytes.
    
    Args:
        obj: Object to serialize
        format: Serialization format ('pickle', 'json')
        
    Returns:
        Serialized bytes
    """
    if format == "pickle":
        return pickle.dumps(obj)
    elif format == "json":
        return json.dumps(obj).encode('utf-8')
    else:
        raise ValueError(f"Unsupported serialization format: {format}")

def deserialize_object(data: bytes, format: str = "pickle") -> Any:
    """
    Deserialize object from bytes.
    
    Args:
        data: Serialized data
        format: Serialization format ('pickle', 'json')
        
    Returns:
        Deserialized object
    """
    if format == "pickle":
        return pickle.loads(data)
    elif format == "json":
        return json.loads(data.decode('utf-8'))
    else:
        raise ValueError(f"Unsupported serialization format: {format}")

def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """
    Safe division with default value for zero denominator.
    
    Args:
        numerator: Numerator
        denominator: Denominator
        default: Default value if denominator is zero
        
    Returns:
        Division result or default
    """
    if abs(denominator) < 1e-10:
        return default
    return numerator / denominator

def clamp(value: float, min_val: float, max_val: float) -> float:
    """
    Clamp value between min and max.
    
    Args:
        value: Value to clamp
        min_val: Minimum value
        max_val: Maximum value
        
    Returns:
        Clamped value
    """
    return max(min_val, min(value, max_val))

def is_market_open(timestamp: Optional[datetime] = None) -> bool:
    """
    Check if market is currently open.
    
    Args:
        timestamp: Timestamp to check (default: current time)
        
    Returns:
        True if market is open
    """
    if timestamp is None:
        timestamp = datetime.now(timezone.utc)
    
    # Convert to UTC if not timezone-aware
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=timezone.utc)
    
    # Check if weekday (Monday=0, Sunday=6)
    if timestamp.weekday() >= 5:  # Saturday or Sunday
        return False
    
    # Check market hours (9:30 AM - 4:00 PM EST)
    market_time = timestamp.astimezone(timezone.utc)
    hour = market_time.hour
    minute = market_time.minute
    
    # Market hours in UTC: 13:30 - 20:00 (EST + 4/5 hours depending on DST)
    market_start = 13 * 60 + 30  # 13:30 in minutes
    market_end = 20 * 60  # 20:00 in minutes
    current_time = hour * 60 + minute
    
    return market_start <= current_time <= market_end

def get_trading_session(timestamp: Optional[datetime] = None) -> str:
    """
    Get current trading session.
    
    Args:
        timestamp: Timestamp to check (default: current time)
        
    Returns:
        Trading session ('pre_market', 'regular', 'after_hours', 'closed')
    """
    if timestamp is None:
        timestamp = datetime.now(timezone.utc)
    
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=timezone.utc)
    
    # Check if weekday
    if timestamp.weekday() >= 5:
        return TradingConstants.SessionType.CLOSED.value
    
    hour = timestamp.hour
    minute = timestamp.minute
    current_time = hour * 60 + minute
    
    # Session times in UTC
    pre_market_start = 9 * 60  # 09:00
    regular_start = 13 * 60 + 30  # 13:30
    regular_end = 20 * 60  # 20:00
    after_hours_end = 25 * 60  # 01:00 next day (25:00)
    
    if pre_market_start <= current_time < regular_start:
        return TradingConstants.SessionType.PRE_MARKET.value
    elif regular_start <= current_time <= regular_end:
        return TradingConstants.SessionType.REGULAR.value
    elif regular_end < current_time <= after_hours_end:
        return TradingConstants.SessionType.AFTER_HOURS.value
    else:
        return TradingConstants.SessionType.CLOSED.value

def memory_usage() -> Dict[str, float]:
    """
    Get current memory usage statistics.
    
    Returns:
        Dictionary with memory usage info
    """
    try:
        process = psutil.Process()
        memory_info = process.memory_info()
        
        return {
            "rss_mb": memory_info.rss / 1024 / 1024,
            "vms_mb": memory_info.vms / 1024 / 1024,
            "percent": process.memory_percent(),
            "available_mb": psutil.virtual_memory().available / 1024 / 1024
        }
    except Exception as e:
        logger.warning(f"Failed to get memory usage: {e}")
        return {}

def cpu_usage() -> Dict[str, float]:
    """
    Get current CPU usage statistics.
    
    Returns:
        Dictionary with CPU usage info
    """
    try:
        return {
            "percent": psutil.cpu_percent(interval=1),
            "count": psutil.cpu_count(),
            "load_avg": os.getloadavg() if hasattr(os, 'getloadavg') else [0, 0, 0]
        }
    except Exception as e:
        logger.warning(f"Failed to get CPU usage: {e}")
        return {}

def cleanup_memory():
    """Force garbage collection and memory cleanup."""
    try:
        # Force garbage collection
        collected = gc.collect()
        
        # Additional cleanup for ARM64
        if SystemArchitecture.IS_ARM64:
            # Clear NumPy cache
            try:
                import numpy as np
                np.core._internal._clear_cache()
            except:
                pass
        
        logger.debug(f"Memory cleanup completed, collected {collected} objects")
        
    except Exception as e:
        logger.warning(f"Memory cleanup failed: {e}")

# Global instances
arm64_optimizer = ARM64Optimizer()
performance_profiler = PerformanceProfiler()

# Export all helper functions
__all__ = [
    # Financial calculations
    "format_currency",
    "calculate_returns",
    "calculate_volatility", 
    "calculate_sharpe_ratio",
    "calculate_max_drawdown",
    
    # Data processing
    "normalize_data",
    "create_lagged_features",
    "create_rolling_features",
    
    # Performance optimization
    "ARM64Optimizer",
    "PerformanceProfiler",
    "arm64_optimizer",
    "performance_profiler",
    
    # Utility functions
    "get_timestamp",
    "format_timestamp",
    "parse_timestamp",
    "generate_hash",
    "compress_data",
    "decompress_data",
    "serialize_object",
    "deserialize_object",
    "safe_divide",
    "clamp",
    "is_market_open",
    "get_trading_session",
    "memory_usage",
    "cpu_usage",
    "cleanup_memory"
]