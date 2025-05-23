"""
Enhanced decorators for Deep Momentum Trading System with ARM64 optimizations.

This module provides comprehensive decorators for performance monitoring,
caching, retry logic, ARM64 optimizations, and trading-specific functionality.
"""

import time
import functools
import threading
import asyncio
import inspect
import os
import gc
import psutil
import warnings
from typing import Any, Callable, Dict, Optional, Union, List, Tuple
from collections import defaultdict, deque
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import weakref
import pickle
import hashlib
import json

from .logger import get_logger
from .constants import SystemArchitecture, ARM64OptimizationConstants

logger = get_logger(__name__)

# Performance monitoring storage
_performance_data = defaultdict(lambda: {
    "call_count": 0,
    "total_time": 0.0,
    "min_time": float('inf'),
    "max_time": 0.0,
    "errors": 0,
    "last_called": None,
    "memory_usage": deque(maxlen=100),
    "execution_times": deque(maxlen=1000)
})

_performance_lock = threading.RLock()

@dataclass
class CacheEntry:
    """Cache entry with metadata."""
    value: Any
    timestamp: float
    access_count: int = 0
    last_access: float = 0.0
    size_bytes: int = 0

class LRUCache:
    """Thread-safe LRU cache with ARM64 optimizations."""
    
    def __init__(self, maxsize: int = 128, ttl: Optional[float] = None):
        self.maxsize = maxsize
        self.ttl = ttl
        self.cache: Dict[str, CacheEntry] = {}
        self.access_order = deque()
        self.lock = threading.RLock()
        self.hits = 0
        self.misses = 0
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        with self.lock:
            if key not in self.cache:
                self.misses += 1
                return None
            
            entry = self.cache[key]
            current_time = time.time()
            
            # Check TTL
            if self.ttl and (current_time - entry.timestamp) > self.ttl:
                del self.cache[key]
                self.access_order.remove(key)
                self.misses += 1
                return None
            
            # Update access info
            entry.access_count += 1
            entry.last_access = current_time
            
            # Move to end (most recently used)
            self.access_order.remove(key)
            self.access_order.append(key)
            
            self.hits += 1
            return entry.value
    
    def put(self, key: str, value: Any) -> None:
        """Put value in cache."""
        with self.lock:
            current_time = time.time()
            
            # Calculate size
            try:
                size_bytes = len(pickle.dumps(value))
            except:
                size_bytes = 0
            
            # Remove if exists
            if key in self.cache:
                self.access_order.remove(key)
            
            # Evict if necessary
            while len(self.cache) >= self.maxsize:
                oldest_key = self.access_order.popleft()
                del self.cache[oldest_key]
            
            # Add new entry
            entry = CacheEntry(
                value=value,
                timestamp=current_time,
                last_access=current_time,
                size_bytes=size_bytes
            )
            
            self.cache[key] = entry
            self.access_order.append(key)
    
    def clear(self) -> None:
        """Clear cache."""
        with self.lock:
            self.cache.clear()
            self.access_order.clear()
            self.hits = 0
            self.misses = 0
    
    def stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self.lock:
            total_requests = self.hits + self.misses
            hit_ratio = self.hits / total_requests if total_requests > 0 else 0.0
            
            total_size = sum(entry.size_bytes for entry in self.cache.values())
            
            return {
                "size": len(self.cache),
                "maxsize": self.maxsize,
                "hits": self.hits,
                "misses": self.misses,
                "hit_ratio": hit_ratio,
                "total_size_bytes": total_size
            }

def timing_decorator(include_args: bool = False, 
                    include_memory: bool = False,
                    enable_profiling: bool = False):
    """
    Decorator to measure function execution time with ARM64 optimizations.
    
    Args:
        include_args: Include function arguments in timing data
        include_memory: Track memory usage during execution
        enable_profiling: Enable detailed profiling
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            func_name = f"{func.__module__}.{func.__qualname__}"
            start_time = time.perf_counter_ns() if SystemArchitecture.IS_ARM64 else time.perf_counter()
            
            # Memory tracking
            start_memory = None
            if include_memory:
                try:
                    process = psutil.Process()
                    start_memory = process.memory_info().rss
                except:
                    pass
            
            try:
                # Execute function
                result = func(*args, **kwargs)
                
                # Calculate timing
                if SystemArchitecture.IS_ARM64:
                    execution_time = (time.perf_counter_ns() - start_time) / 1_000_000_000
                else:
                    execution_time = time.perf_counter() - start_time
                
                # Memory usage
                memory_delta = None
                if include_memory and start_memory:
                    try:
                        end_memory = psutil.Process().memory_info().rss
                        memory_delta = end_memory - start_memory
                    except:
                        pass
                
                # Update performance data
                with _performance_lock:
                    data = _performance_data[func_name]
                    data["call_count"] += 1
                    data["total_time"] += execution_time
                    data["min_time"] = min(data["min_time"], execution_time)
                    data["max_time"] = max(data["max_time"], execution_time)
                    data["last_called"] = time.time()
                    data["execution_times"].append(execution_time)
                    
                    if memory_delta is not None:
                        data["memory_usage"].append(memory_delta)
                
                # Log if slow
                if execution_time > 1.0:  # Log if > 1 second
                    logger.warning(f"Slow function {func_name}: {execution_time:.3f}s")
                
                return result
                
            except Exception as e:
                # Track errors
                with _performance_lock:
                    _performance_data[func_name]["errors"] += 1
                
                logger.error(f"Error in {func_name}: {e}")
                raise
        
        # Async version
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            func_name = f"{func.__module__}.{func.__qualname__}"
            start_time = time.perf_counter_ns() if SystemArchitecture.IS_ARM64 else time.perf_counter()
            
            try:
                result = await func(*args, **kwargs)
                
                if SystemArchitecture.IS_ARM64:
                    execution_time = (time.perf_counter_ns() - start_time) / 1_000_000_000
                else:
                    execution_time = time.perf_counter() - start_time
                
                with _performance_lock:
                    data = _performance_data[func_name]
                    data["call_count"] += 1
                    data["total_time"] += execution_time
                    data["min_time"] = min(data["min_time"], execution_time)
                    data["max_time"] = max(data["max_time"], execution_time)
                    data["last_called"] = time.time()
                    data["execution_times"].append(execution_time)
                
                return result
                
            except Exception as e:
                with _performance_lock:
                    _performance_data[func_name]["errors"] += 1
                raise
        
        # Return appropriate wrapper
        if inspect.iscoroutinefunction(func):
            return async_wrapper
        else:
            return wrapper
    
    return decorator

def retry_decorator(max_attempts: int = 3,
                   delay: float = 1.0,
                   backoff: float = 2.0,
                   exceptions: Tuple = (Exception,),
                   jitter: bool = True):
    """
    Decorator for retry logic with exponential backoff.
    
    Args:
        max_attempts: Maximum number of retry attempts
        delay: Initial delay between retries
        backoff: Backoff multiplier
        exceptions: Tuple of exceptions to catch
        jitter: Add random jitter to delay
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            import random
            
            last_exception = None
            
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    
                    if attempt == max_attempts - 1:
                        logger.error(f"Function {func.__name__} failed after {max_attempts} attempts")
                        raise
                    
                    # Calculate delay with backoff and jitter
                    current_delay = delay * (backoff ** attempt)
                    if jitter:
                        current_delay *= (0.5 + random.random() * 0.5)
                    
                    logger.warning(f"Attempt {attempt + 1} failed for {func.__name__}, "
                                 f"retrying in {current_delay:.2f}s: {e}")
                    time.sleep(current_delay)
            
            raise last_exception
        
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            import random
            
            last_exception = None
            
            for attempt in range(max_attempts):
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    
                    if attempt == max_attempts - 1:
                        raise
                    
                    current_delay = delay * (backoff ** attempt)
                    if jitter:
                        current_delay *= (0.5 + random.random() * 0.5)
                    
                    logger.warning(f"Async attempt {attempt + 1} failed for {func.__name__}, "
                                 f"retrying in {current_delay:.2f}s: {e}")
                    await asyncio.sleep(current_delay)
            
            raise last_exception
        
        if inspect.iscoroutinefunction(func):
            return async_wrapper
        else:
            return wrapper
    
    return decorator

def cache_decorator(maxsize: int = 128, 
                   ttl: Optional[float] = None,
                   key_func: Optional[Callable] = None,
                   ignore_args: Optional[List[str]] = None):
    """
    Decorator for function result caching with ARM64 optimizations.
    
    Args:
        maxsize: Maximum cache size
        ttl: Time to live for cache entries
        key_func: Custom key generation function
        ignore_args: Arguments to ignore in key generation
    """
    def decorator(func: Callable) -> Callable:
        cache = LRUCache(maxsize=maxsize, ttl=ttl)
        
        def generate_key(*args, **kwargs) -> str:
            """Generate cache key from arguments."""
            if key_func:
                return key_func(*args, **kwargs)
            
            # Filter ignored arguments
            if ignore_args:
                filtered_kwargs = {k: v for k, v in kwargs.items() 
                                 if k not in ignore_args}
            else:
                filtered_kwargs = kwargs
            
            # Create key from args and kwargs
            key_data = {
                "args": args,
                "kwargs": filtered_kwargs,
                "func": func.__name__
            }
            
            # Use hash for efficiency
            key_str = json.dumps(key_data, sort_keys=True, default=str)
            return hashlib.md5(key_str.encode()).hexdigest()
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Generate cache key
            cache_key = generate_key(*args, **kwargs)
            
            # Try to get from cache
            cached_result = cache.get(cache_key)
            if cached_result is not None:
                return cached_result
            
            # Execute function and cache result
            result = func(*args, **kwargs)
            cache.put(cache_key, result)
            
            return result
        
        # Add cache management methods
        wrapper.cache_clear = cache.clear
        wrapper.cache_stats = cache.stats
        wrapper.cache_info = cache.stats  # Compatibility with functools.lru_cache
        
        return wrapper
    
    return decorator

def arm64_optimized(enable_simd: bool = True,
                   enable_numa: bool = True,
                   thread_affinity: bool = True):
    """
    Decorator to apply ARM64-specific optimizations.
    
    Args:
        enable_simd: Enable SIMD optimizations
        enable_numa: Enable NUMA optimizations
        thread_affinity: Set thread affinity for ARM64
    """
    def decorator(func: Callable) -> Callable:
        if not SystemArchitecture.IS_ARM64:
            # Return unmodified function on non-ARM64 systems
            return func
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Set thread affinity if enabled
            if thread_affinity:
                try:
                    import os
                    # Set CPU affinity to all available cores
                    os.sched_setaffinity(0, range(SystemArchitecture.CPU_COUNT))
                except:
                    pass
            
            # NUMA optimizations
            if enable_numa and SystemArchitecture.NUMA_NODES > 1:
                try:
                    # Set memory policy for better NUMA performance
                    os.environ.setdefault("NUMA_POLICY", "interleave")
                except:
                    pass
            
            # SIMD optimizations
            if enable_simd:
                try:
                    # Enable ARM64 NEON optimizations
                    os.environ.setdefault("ARM_NEON_ENABLE", "1")
                    os.environ.setdefault("USE_NEON", "1")
                except:
                    pass
            
            return func(*args, **kwargs)
        
        return wrapper
    
    return decorator

def performance_monitor_decorator(threshold_ms: float = 100.0,
                                memory_threshold_mb: float = 100.0,
                                alert_callback: Optional[Callable] = None):
    """
    Decorator for comprehensive performance monitoring.
    
    Args:
        threshold_ms: Alert threshold for execution time
        memory_threshold_mb: Alert threshold for memory usage
        alert_callback: Callback function for alerts
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            func_name = f"{func.__module__}.{func.__qualname__}"
            
            # Start monitoring
            start_time = time.perf_counter()
            start_memory = None
            
            try:
                process = psutil.Process()
                start_memory = process.memory_info().rss
            except:
                pass
            
            try:
                result = func(*args, **kwargs)
                
                # Calculate metrics
                execution_time_ms = (time.perf_counter() - start_time) * 1000
                memory_usage_mb = 0
                
                if start_memory:
                    try:
                        end_memory = psutil.Process().memory_info().rss
                        memory_usage_mb = (end_memory - start_memory) / 1024 / 1024
                    except:
                        pass
                
                # Check thresholds and alert
                if execution_time_ms > threshold_ms:
                    message = f"Slow execution: {func_name} took {execution_time_ms:.2f}ms"
                    logger.warning(message)
                    
                    if alert_callback:
                        alert_callback("slow_execution", {
                            "function": func_name,
                            "execution_time_ms": execution_time_ms,
                            "threshold_ms": threshold_ms
                        })
                
                if memory_usage_mb > memory_threshold_mb:
                    message = f"High memory usage: {func_name} used {memory_usage_mb:.2f}MB"
                    logger.warning(message)
                    
                    if alert_callback:
                        alert_callback("high_memory", {
                            "function": func_name,
                            "memory_usage_mb": memory_usage_mb,
                            "threshold_mb": memory_threshold_mb
                        })
                
                return result
                
            except Exception as e:
                if alert_callback:
                    alert_callback("function_error", {
                        "function": func_name,
                        "error": str(e),
                        "error_type": type(e).__name__
                    })
                raise
        
        return wrapper
    
    return decorator

def trading_session_decorator(require_market_hours: bool = True,
                            allowed_sessions: Optional[List[str]] = None):
    """
    Decorator to enforce trading session requirements.
    
    Args:
        require_market_hours: Require market to be open
        allowed_sessions: List of allowed session types
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            from datetime import datetime, time as dt_time
            from .constants import TradingConstants
            
            now = datetime.utcnow().time()
            
            # Check market hours
            if require_market_hours:
                market_open = dt_time.fromisoformat(TradingConstants.MARKET_OPEN_UTC)
                market_close = dt_time.fromisoformat(TradingConstants.MARKET_CLOSE_UTC)
                
                if not (market_open <= now <= market_close):
                    raise ValueError(f"Function {func.__name__} can only be called during market hours")
            
            # Check allowed sessions
            if allowed_sessions:
                current_session = _get_current_session(now)
                if current_session not in allowed_sessions:
                    raise ValueError(f"Function {func.__name__} not allowed during {current_session} session")
            
            return func(*args, **kwargs)
        
        return wrapper
    
    return decorator

def rate_limit_decorator(calls_per_second: float = 10.0,
                        burst_size: int = 5):
    """
    Decorator for rate limiting function calls.
    
    Args:
        calls_per_second: Maximum calls per second
        burst_size: Maximum burst size
    """
    def decorator(func: Callable) -> Callable:
        call_times = deque()
        lock = threading.Lock()
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            current_time = time.time()
            
            with lock:
                # Remove old calls outside the time window
                window_start = current_time - 1.0  # 1 second window
                while call_times and call_times[0] < window_start:
                    call_times.popleft()
                
                # Check rate limit
                if len(call_times) >= calls_per_second:
                    sleep_time = 1.0 / calls_per_second
                    time.sleep(sleep_time)
                    current_time = time.time()
                
                # Check burst limit
                recent_calls = sum(1 for t in call_times if current_time - t < 0.1)  # 100ms window
                if recent_calls >= burst_size:
                    time.sleep(0.1)
                    current_time = time.time()
                
                call_times.append(current_time)
            
            return func(*args, **kwargs)
        
        return wrapper
    
    return decorator

def async_timeout_decorator(timeout_seconds: float = 30.0):
    """
    Decorator to add timeout to async functions.
    
    Args:
        timeout_seconds: Timeout in seconds
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            try:
                return await asyncio.wait_for(
                    func(*args, **kwargs), 
                    timeout=timeout_seconds
                )
            except asyncio.TimeoutError:
                logger.error(f"Function {func.__name__} timed out after {timeout_seconds}s")
                raise
        
        return wrapper
    
    return decorator

def memory_limit_decorator(max_memory_mb: float = 1000.0):
    """
    Decorator to monitor and limit memory usage.
    
    Args:
        max_memory_mb: Maximum memory usage in MB
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                process = psutil.Process()
                start_memory = process.memory_info().rss / 1024 / 1024  # MB
                
                if start_memory > max_memory_mb:
                    logger.warning(f"High memory usage before {func.__name__}: {start_memory:.2f}MB")
                    gc.collect()  # Force garbage collection
                
                result = func(*args, **kwargs)
                
                end_memory = process.memory_info().rss / 1024 / 1024  # MB
                memory_delta = end_memory - start_memory
                
                if memory_delta > max_memory_mb * 0.5:  # 50% of limit
                    logger.warning(f"Function {func.__name__} used {memory_delta:.2f}MB")
                
                return result
                
            except Exception as e:
                logger.error(f"Memory monitoring error in {func.__name__}: {e}")
                return func(*args, **kwargs)  # Fallback to normal execution
        
        return wrapper
    
    return decorator

# Utility functions
def _get_current_session(current_time) -> str:
    """Get current trading session type."""
    from datetime import time as dt_time
    from .constants import TradingConstants
    
    pre_market_start = dt_time.fromisoformat(TradingConstants.PRE_MARKET_START_UTC)
    market_open = dt_time.fromisoformat(TradingConstants.MARKET_OPEN_UTC)
    market_close = dt_time.fromisoformat(TradingConstants.MARKET_CLOSE_UTC)
    after_hours_end = dt_time.fromisoformat(TradingConstants.AFTER_HOURS_END_UTC)
    
    if pre_market_start <= current_time < market_open:
        return TradingConstants.SessionType.PRE_MARKET.value
    elif market_open <= current_time <= market_close:
        return TradingConstants.SessionType.REGULAR.value
    elif market_close < current_time <= after_hours_end:
        return TradingConstants.SessionType.AFTER_HOURS.value
    else:
        return TradingConstants.SessionType.CLOSED.value

def get_performance_stats() -> Dict[str, Any]:
    """Get global performance statistics."""
    with _performance_lock:
        stats = {}
        for func_name, data in _performance_data.items():
            if data["call_count"] > 0:
                avg_time = data["total_time"] / data["call_count"]
                stats[func_name] = {
                    "call_count": data["call_count"],
                    "total_time": data["total_time"],
                    "avg_time": avg_time,
                    "min_time": data["min_time"],
                    "max_time": data["max_time"],
                    "errors": data["errors"],
                    "last_called": data["last_called"]
                }
        return stats

def clear_performance_stats() -> None:
    """Clear all performance statistics."""
    with _performance_lock:
        _performance_data.clear()

# Export all decorators
__all__ = [
    "timing_decorator",
    "retry_decorator", 
    "cache_decorator",
    "arm64_optimized",
    "performance_monitor_decorator",
    "trading_session_decorator",
    "rate_limit_decorator",
    "async_timeout_decorator",
    "memory_limit_decorator",
    "get_performance_stats",
    "clear_performance_stats",
    "LRUCache",
    "CacheEntry"
]