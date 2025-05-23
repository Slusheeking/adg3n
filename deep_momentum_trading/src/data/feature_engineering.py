import pandas as pd
import numpy as np
import platform
import time
import os
import asyncio
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor
from collections import OrderedDict, deque
from functools import partial
import numba
import warnings

from deep_momentum_trading.src.utils.logger import get_logger
from deep_momentum_trading.src.data.memory_cache import UnifiedMemoryManager
from deep_momentum_trading.src.communication.zmq_publisher import ZMQPublisher
from deep_momentum_trading.config.settings import config_manager

logger = get_logger(__name__)

@dataclass
class FeatureEngineeringConfig:
    """Configuration for FeatureEngineeringProcess with GH200 and ARM64 optimizations."""
    zmq_subscriber_port: int = 5555
    zmq_publisher_port: int = 5556
    memory_cache_max_gb: float = 200.0
    enable_arm64_optimizations: bool = True
    enable_parallel_processing: bool = True
    max_workers: int = 4
    chunk_size: int = 10000
    enable_performance_monitoring: bool = True
    enable_caching: bool = True
    cache_size: int = 1000
    feature_calculation_timeout: float = 30.0
    enable_advanced_features: bool = True
    enable_real_time_features: bool = True
    # GH200 specific configurations
    gh200_memory_pool_gb: float = 50.0
    enable_gh200_optimizations: bool = True
    # Multi-symbol processing
    symbol_batch_size: int = 100
    enable_streaming_features: bool = True
    max_symbols: int = 10000
    lookback_periods: int = 200

@dataclass
class FeatureEngineeringStats:
    """Statistics for feature engineering performance monitoring."""
    features_calculated: int = 0
    processing_time_seconds: float = 0.0
    cache_hits: int = 0
    cache_misses: int = 0
    symbols_processed: int = 0
    errors: int = 0
    numba_compilation_time: float = 0.0
    arm64_optimizations_used: int = 0
    gh200_optimizations_used: int = 0
    memory_pool_allocations: int = 0
    streaming_updates: int = 0
    batch_operations: int = 0
    start_time: float = field(default_factory=time.time)
    
    @property
    def features_per_second(self) -> float:
        return self.features_calculated / max(self.processing_time_seconds, 0.001)
    
    @property
    def uptime_seconds(self) -> float:
        return time.time() - self.start_time

class GH200FeaturePool:
    """Leverage GH200's 624GB unified memory architecture for feature storage."""
    
    def __init__(self, pool_size_gb: float = 50.0):
        """Initialize GH200 unified memory pool."""
        self.pool_size_bytes = int(pool_size_gb * 1024**3)
        self.feature_pool = np.zeros(self.pool_size_bytes // 8, dtype=np.float64)
        self.current_offset = 0
        self.allocations = {}  # Track active allocations
        self.allocation_history = deque(maxlen=1000)  # Track allocation patterns
        
        logger.info(f"GH200 Feature Pool initialized with {pool_size_gb}GB unified memory")
        
    def allocate_feature_array(self, symbol: str, num_features: int, num_periods: int) -> np.ndarray:
        """Allocate feature array from unified memory pool."""
        required_size = num_features * num_periods
        
        if self.current_offset + required_size > len(self.feature_pool):
            self._garbage_collect()
        
        # Return memory view (zero-copy)
        start_idx = self.current_offset
        end_idx = start_idx + required_size
        
        feature_array = self.feature_pool[start_idx:end_idx].reshape(num_periods, num_features)
        self.allocations[symbol] = (start_idx, end_idx, time.time())
        self.current_offset = end_idx
        
        # Track allocation pattern
        self.allocation_history.append({
            'symbol': symbol,
            'size': required_size,
            'timestamp': time.time()
        })
        
        return feature_array
    
    def deallocate_symbol(self, symbol: str):
        """Deallocate memory for a specific symbol."""
        if symbol in self.allocations:
            del self.allocations[symbol]
    
    def _garbage_collect(self):
        """Compact memory pool when fragmented."""
        # Simple reset strategy - could implement proper compaction
        logger.info("GH200 memory pool garbage collection triggered")
        self.current_offset = 0
        self.allocations.clear()
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage statistics."""
        used_bytes = self.current_offset * 8  # 8 bytes per float64
        return {
            'used_gb': used_bytes / (1024**3),
            'total_gb': self.pool_size_bytes / (1024**3),
            'utilization_percent': (used_bytes / self.pool_size_bytes) * 100,
            'active_allocations': len(self.allocations)
        }

class ARM64FeatureCache:
    """ARM64 optimized feature cache with memory mapping."""
    
    def __init__(self, max_memory_gb: float = 10.0):
        """Initialize ARM64 optimized cache."""
        self.max_memory_bytes = int(max_memory_gb * 1024**3)
        self.cache_dir = "/tmp/dmn_feature_cache"
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # LRU cache with memory mapping
        self.cache_metadata = OrderedDict()
        self.memory_mapped_files = {}
        self.current_memory_usage = 0
        
        logger.info(f"ARM64 Feature Cache initialized with {max_memory_gb}GB capacity")
        
    def store_features(self, symbol: str, features: np.ndarray, 
                      feature_names: List[str]) -> str:
        """Store features using memory mapping for ARM64 efficiency."""
        cache_key = f"{symbol}_{int(time.time())}"
        
        # Create memory-mapped file
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.dat")
        
        # ARM64 optimized memory mapping
        mmap_features = np.memmap(
            cache_file, 
            dtype=np.float64,  # 8-byte aligned for ARM64
            mode='w+',
            shape=features.shape
        )
        mmap_features[:] = features[:]
        mmap_features.flush()
        
        # Update cache metadata
        self.cache_metadata[cache_key] = {
            'file_path': cache_file,
            'shape': features.shape,
            'feature_names': feature_names,
            'access_time': time.time(),
            'size_bytes': features.nbytes
        }
        
        self.current_memory_usage += features.nbytes
        self._evict_if_needed()
        
        return cache_key
    
    def get_features(self, cache_key: str) -> Optional[np.ndarray]:
        """Retrieve features with ARM64 optimized memory mapping."""
        if cache_key not in self.cache_metadata:
            return None
        
        metadata = self.cache_metadata[cache_key]
        
        # Update access time (LRU)
        metadata['access_time'] = time.time()
        self.cache_metadata.move_to_end(cache_key)
        
        # Return memory-mapped array
        return np.memmap(
            metadata['file_path'],
            dtype=np.float64,
            mode='r',
            shape=metadata['shape']
        )
    
    def _evict_if_needed(self):
        """Evict oldest entries if memory limit exceeded."""
        while self.current_memory_usage > self.max_memory_bytes and self.cache_metadata:
            oldest_key = next(iter(self.cache_metadata))
            metadata = self.cache_metadata[oldest_key]
            
            # Remove file and metadata
            try:
                os.remove(metadata['file_path'])
                self.current_memory_usage -= metadata['size_bytes']
                del self.cache_metadata[oldest_key]
            except Exception as e:
                logger.warning(f"Failed to evict cache entry {oldest_key}: {e}")

class StreamingFeatureCalculator:
    """Real-time streaming feature calculation for high-frequency data."""
    
    def __init__(self, max_symbols: int = 10000, lookback_periods: int = 200):
        self.max_symbols = max_symbols
        self.lookback_periods = lookback_periods
        
        # Pre-allocate circular buffers for all symbols
        self.price_buffers = {}
        self.feature_buffers = {}
        self.buffer_positions = {}
        self.last_processing_times = {}
        
        # ARM64 optimized feature calculation pipeline
        self.feature_pipeline = self._create_arm64_pipeline()
        
        logger.info(f"Streaming calculator initialized for {max_symbols} symbols")
    
    def _create_arm64_pipeline(self):
        """Create ARM64 optimized feature calculation pipeline."""
        return {
            'momentum': partial(_calculate_rsi_vectorized_arm64, window=14),
            'volatility': partial(_calculate_bollinger_bands_numba, window=20),
            'trend': partial(_calculate_macd_numba, fast_window=12, slow_window=26)
        }
    
    def _initialize_symbol_buffers(self, symbol: str):
        """Initialize circular buffers for a new symbol."""
        self.price_buffers[symbol] = np.zeros(self.lookback_periods, dtype=np.float64)
        self.feature_buffers[symbol] = {}
        self.buffer_positions[symbol] = 0
        self.last_processing_times[symbol] = 0
    
    def update_features_streaming(self, symbol: str, price: float, 
                                volume: float, timestamp: int) -> Dict[str, float]:
        """Update features for single price tick (streaming mode)."""
        if symbol not in self.price_buffers:
            self._initialize_symbol_buffers(symbol)
        
        # Update circular buffer (zero-copy)
        pos = self.buffer_positions[symbol]
        self.price_buffers[symbol][pos] = price
        self.buffer_positions[symbol] = (pos + 1) % self.lookback_periods
        
        # Calculate features only for latest data point
        return self._calculate_incremental_features(symbol, price, volume)
    
    def _calculate_incremental_features(self, symbol: str, price: float, 
                                      volume: float) -> Dict[str, float]:
        """Calculate features incrementally for latest tick."""
        buffer = self.price_buffers[symbol]
        
        # Only calculate what's needed for latest point
        features = {}
        
        # RSI (incremental calculation)
        if len(buffer) >= 14:
            features['rsi_14'] = self._calculate_rsi_incremental(buffer, price)
        
        # MACD (incremental calculation)
        if len(buffer) >= 26:
            features.update(self._calculate_macd_incremental(buffer, price))
        
        return features
    
    def _calculate_rsi_incremental(self, buffer: np.ndarray, latest_price: float) -> float:
        """Calculate RSI incrementally for latest price."""
        # Simplified incremental RSI calculation
        # In production, maintain running averages
        return _calculate_rsi_vectorized_arm64(buffer)[-1]
    
    def _calculate_macd_incremental(self, buffer: np.ndarray, latest_price: float) -> Dict[str, float]:
        """Calculate MACD incrementally for latest price."""
        macd, signal, hist = _calculate_macd_numba(buffer)
        return {
            'macd': macd[-1] if not np.isnan(macd[-1]) else 0.0,
            'macd_signal': signal[-1] if not np.isnan(signal[-1]) else 0.0,
            'macd_hist': hist[-1] if not np.isnan(hist[-1]) else 0.0
        }

class SymbolPriorityManager:
    """Manage feature calculation priority for 10,000+ symbols."""
    
    def __init__(self):
        self.priority_tiers = {
            'tier1': set(),  # Most liquid 1000 symbols
            'tier2': set(),  # Medium liquid 3000 symbols  
            'tier3': set()   # All other symbols
        }
        self.processing_intervals = {
            'tier1': 1.0,   # Process every second
            'tier2': 5.0,   # Process every 5 seconds
            'tier3': 30.0   # Process every 30 seconds
        }
        self.last_processing_times = {}
    
    def should_process_symbol(self, symbol: str) -> bool:
        """Determine if symbol should be processed now."""
        for tier, symbols in self.priority_tiers.items():
            if symbol in symbols:
                interval = self.processing_intervals[tier]
                last_processed = self.last_processing_times.get(symbol, 0)
                return time.time() - last_processed >= interval
        
        return False  # Unknown symbol, don't process
    
    def update_processing_time(self, symbol: str):
        """Update last processing time for symbol."""
        self.last_processing_times[symbol] = time.time()
    
    def add_symbol_to_tier(self, symbol: str, tier: str):
        """Add symbol to specific priority tier."""
        if tier in self.priority_tiers:
            self.priority_tiers[tier].add(symbol)

class MultiSymbolBatchProcessor:
    """Process multiple symbols simultaneously for ARM64 cache efficiency."""
    
    def __init__(self, batch_size: int = 100):
        self.batch_size = batch_size
        self.symbol_queue = deque()
        self.data_batch = {}
        
    async def process_symbol_batch(self, symbol_data_pairs: List[Tuple[str, pd.DataFrame]]):
        """Process multiple symbols in single ARM64 optimized batch."""
        
        # Group symbols by similar data characteristics for cache efficiency
        grouped_symbols = self._group_symbols_by_characteristics(symbol_data_pairs)
        
        # Process each group in parallel
        tasks = []
        for group in grouped_symbols:
            task = asyncio.create_task(self._process_symbol_group(group))
            tasks.append(task)
        
        # Wait for all groups to complete
        results = await asyncio.gather(*tasks)
        
        # Flatten results
        all_features = {}
        for group_result in results:
            all_features.update(group_result)
        
        return all_features
    
    def _group_symbols_by_characteristics(self, symbol_data_pairs):
        """Group symbols by data size/type for optimal ARM64 cache usage."""
        small_data = []  # < 1000 rows
        medium_data = []  # 1000-10000 rows  
        large_data = []  # > 10000 rows
        
        for symbol, data in symbol_data_pairs:
            if len(data) < 1000:
                small_data.append((symbol, data))
            elif len(data) < 10000:
                medium_data.append((symbol, data))
            else:
                large_data.append((symbol, data))
        
        return [small_data, medium_data, large_data]
    
    async def _process_symbol_group(self, symbol_group):
        """Process a group of symbols with similar characteristics."""
        # Placeholder for actual processing logic
        results = {}
        for symbol, data in symbol_group:
            # Process each symbol in the group
            results[symbol] = f"processed_{symbol}"
        return results

# Enhanced ARM64 SIMD Vectorization
@numba.jit(nopython=True, cache=True, fastmath=True, parallel=True)
def _calculate_rsi_vectorized_arm64(prices: np.ndarray, window: int = 14) -> np.ndarray:
    """ARM64 NEON optimized RSI with parallel processing."""
    if len(prices) < window:
        return np.full(len(prices), np.nan)
    
    # Vectorized delta calculation
    delta = np.diff(prices)
    gain = np.maximum(delta, 0.0)  # Vectorized max
    loss = np.maximum(-delta, 0.0)  # Vectorized max
    
    # Use numba parallel for ARM64 SIMD
    rsi_values = np.zeros(len(prices))
    
    # Vectorized initial window calculation
    initial_avg_gain = np.mean(gain[:window])
    initial_avg_loss = np.mean(loss[:window])
    
    # Parallel EMA-style calculation
    avg_gain = initial_avg_gain
    avg_loss = initial_avg_loss
    
    for i in numba.prange(window, len(prices)):  # Parallel loop
        # Exponential moving average calculation (vectorizable)
        alpha = 1.0 / window
        if i == window:
            avg_gain = initial_avg_gain
            avg_loss = initial_avg_loss
        else:
            avg_gain = alpha * gain[i-1] + (1-alpha) * avg_gain
            avg_loss = alpha * loss[i-1] + (1-alpha) * avg_loss
        
        rs = avg_gain / (avg_loss + 1e-10)
        rsi_values[i] = 100.0 - (100.0 / (1.0 + rs))
    
    rsi_values[:window] = np.nan
    return rsi_values

@numba.jit(nopython=True, cache=True, fastmath=True, parallel=True)
def _calculate_multiple_indicators_batch(prices: np.ndarray, high: np.ndarray, 
                                       low: np.ndarray, volume: np.ndarray) -> np.ndarray:
    """Calculate multiple indicators in single ARM64 SIMD pass."""
    n = len(prices)
    # Pre-allocate output array: [RSI, MACD, BB_upper, BB_middle, BB_lower, ATR]
    indicators = np.zeros((n, 6), dtype=np.float64)
    
    # Batch calculate all indicators using shared computations
    # This maximizes ARM64 cache efficiency and SIMD utilization
    
    # Shared price movements
    delta = np.diff(prices)
    gain = np.maximum(delta, 0.0)
    loss = np.maximum(-delta, 0.0)
    
    # Calculate all indicators in parallel
    indicators[:, 0] = _calculate_rsi_vectorized_arm64(prices, 14)
    # Add other indicators...
    
    return indicators

# Original Numba functions (keeping for compatibility)
@numba.jit(nopython=True, cache=True, fastmath=True)
def _calculate_rsi_numba(prices: np.ndarray, window: int = 14) -> np.ndarray:
    """ARM64 optimized RSI calculation using Numba JIT with fastmath."""
    if len(prices) < window:
        return np.full(len(prices), np.nan)

    delta = np.diff(prices)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)

    avg_gain = np.zeros_like(gain)
    avg_loss = np.zeros_like(loss)

    avg_gain[window - 1] = np.mean(gain[:window])
    avg_loss[window - 1] = np.mean(loss[:window])

    for i in range(window, len(gain)):
        avg_gain[i] = (avg_gain[i - 1] * (window - 1) + gain[i]) / window
        avg_loss[i] = (avg_loss[i - 1] * (window - 1) + loss[i]) / window

    rs = avg_gain[window - 1:] / (avg_loss[window - 1:] + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    return np.concatenate((np.full(window - 1, np.nan), rsi))

@numba.jit(nopython=True, cache=True, fastmath=True)
def _calculate_macd_numba(prices: np.ndarray, fast_window: int = 12, slow_window: int = 26, signal_window: int = 9) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """ARM64 optimized MACD calculation using Numba JIT with fastmath."""
    if len(prices) < slow_window:
        return np.full(len(prices), np.nan), np.full(len(prices), np.nan), np.full(len(prices), np.nan)

    ema_fast = np.zeros_like(prices)
    ema_slow = np.zeros_like(prices)

    # Initial EMA calculation
    ema_fast[fast_window - 1] = np.mean(prices[:fast_window])
    ema_slow[slow_window - 1] = np.mean(prices[:slow_window])

    alpha_fast = 2 / (fast_window + 1)
    alpha_slow = 2 / (slow_window + 1)

    for i in range(fast_window, len(prices)):
        ema_fast[i] = prices[i] * alpha_fast + ema_fast[i - 1] * (1 - alpha_fast)
    for i in range(slow_window, len(prices)):
        ema_slow[i] = prices[i] * alpha_slow + ema_slow[i - 1] * (1 - alpha_slow)

    macd = ema_fast - ema_slow
    
    # Signal line calculation
    signal_line = np.zeros_like(macd)
    if len(macd[slow_window - 1:]) >= signal_window:
        signal_line[slow_window - 1 + signal_window - 1] = np.mean(macd[slow_window - 1 : slow_window - 1 + signal_window])
        alpha_signal = 2 / (signal_window + 1)
        for i in range(slow_window - 1 + signal_window, len(macd)):
            signal_line[i] = macd[i] * alpha_signal + signal_line[i - 1] * (1 - alpha_signal)

    histogram = macd - signal_line
    
    # Fill leading NaNs
    macd[:slow_window - 1] = np.nan
    signal_line[:slow_window - 1 + signal_window - 1] = np.nan
    histogram[:slow_window - 1 + signal_window - 1] = np.nan

    return macd, signal_line, histogram

@numba.jit(nopython=True, cache=True, fastmath=True)
def _calculate_bollinger_bands_numba(prices: np.ndarray, window: int = 20, num_std: float = 2.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """ARM64 optimized Bollinger Bands calculation using Numba JIT with fastmath."""
    if len(prices) < window:
        return np.full(len(prices), np.nan), np.full(len(prices), np.nan), np.full(len(prices), np.nan)

    middle_band = np.full(len(prices), np.nan)
    upper_band = np.full(len(prices), np.nan)
    lower_band = np.full(len(prices), np.nan)

    for i in range(window - 1, len(prices)):
        window_prices = prices[i - window + 1 : i + 1]
        ma = np.mean(window_prices)
        std = np.std(window_prices)
        
        middle_band[i] = ma
        upper_band[i] = ma + (std * num_std)
        lower_band[i] = ma - (std * num_std)
        
    return upper_band, middle_band, lower_band

@numba.jit(nopython=True, cache=True, fastmath=True)
def _calculate_momentum_numba(prices: np.ndarray, window: int = 10) -> np.ndarray:
    """ARM64 optimized Momentum calculation using Numba JIT with fastmath."""
    if len(prices) < window:
        return np.full(len(prices), np.nan)
    
    momentum = np.full(len(prices), np.nan)
    for i in range(window, len(prices)):
        momentum[i] = prices[i] / prices[i - window] * 100
    return momentum

@numba.jit(nopython=True, cache=True, fastmath=True)
def _calculate_stochastic_numba(high: np.ndarray, low: np.ndarray, close: np.ndarray, window: int = 14) -> Tuple[np.ndarray, np.ndarray]:
    """ARM64 optimized Stochastic Oscillator calculation using Numba JIT."""
    if len(close) < window:
        return np.full(len(close), np.nan), np.full(len(close), np.nan)
    
    k_percent = np.full(len(close), np.nan)
    d_percent = np.full(len(close), np.nan)
    
    for i in range(window - 1, len(close)):
        window_high = np.max(high[i - window + 1 : i + 1])
        window_low = np.min(low[i - window + 1 : i + 1])
        
        if window_high != window_low:
            k_percent[i] = ((close[i] - window_low) / (window_high - window_low)) * 100
        else:
            k_percent[i] = 50.0  # Neutral value when no range
    
    # Calculate %D as 3-period SMA of %K
    for i in range(window + 1, len(close)):
        d_percent[i] = np.mean(k_percent[i-2:i+1])
    
    return k_percent, d_percent

@numba.jit(nopython=True, cache=True, fastmath=True)
def _calculate_atr_numba(high: np.ndarray, low: np.ndarray, close: np.ndarray, window: int = 14) -> np.ndarray:
    """ARM64 optimized Average True Range calculation using Numba JIT."""
    if len(close) < window + 1:
        return np.full(len(close), np.nan)
    
    true_range = np.full(len(close), np.nan)
    atr = np.full(len(close), np.nan)
    
    for i in range(1, len(close)):
        tr1 = high[i] - low[i]
        tr2 = abs(high[i] - close[i-1])
        tr3 = abs(low[i] - close[i-1])
        true_range[i] = max(tr1, tr2, tr3)
    
    # Calculate initial ATR as SMA
    atr[window] = np.mean(true_range[1:window+1])
    
    # Calculate subsequent ATR values using Wilder's smoothing
    for i in range(window + 1, len(close)):
        atr[i] = (atr[i-1] * (window - 1) + true_range[i]) / window
    
    return atr

@numba.jit(nopython=True, cache=True, fastmath=True)
def _calculate_williams_r_numba(high: np.ndarray, low: np.ndarray, close: np.ndarray, window: int = 14) -> np.ndarray:
    """ARM64 optimized Williams %R calculation using Numba JIT."""
    if len(close) < window:
        return np.full(len(close), np.nan)
    
    williams_r = np.full(len(close), np.nan)
    
    for i in range(window - 1, len(close)):
        window_high = np.max(high[i - window + 1 : i + 1])
        window_low = np.min(low[i - window + 1 : i + 1])
        
        if window_high != window_low:
            williams_r[i] = ((window_high - close[i]) / (window_high - window_low)) * -100
        else:
            williams_r[i] = -50.0  # Neutral value when no range
    
    return williams_r

class FeatureEngineeringProcess:
    """
    Enhanced FeatureEngineeringProcess with GH200 unified memory and ARM64 optimizations.
    
    Engineers features from raw market data using high-performance computations
    with GH200's 624GB unified memory architecture and ARM64 SIMD vectorization.
    """

    def __init__(self,
                 config: Optional[FeatureEngineeringConfig] = None,
                 zmq_subscriber_port: int = 5555,
                 zmq_publisher_port: int = 5556,
                 memory_cache_max_gb: float = 200.0):
        """
        Initialize enhanced FeatureEngineeringProcess with GH200 and ARM64 optimizations.

        Args:
            config: FeatureEngineeringConfig object (preferred)
            zmq_subscriber_port: Port for ZeroMQ subscriber (fallback)
            zmq_publisher_port: Port for ZeroMQ publisher (fallback)
            memory_cache_max_gb: Maximum memory in GB for cache (fallback)
        """
        # Configuration handling
        if config is not None:
            self.config = config
        else:
            # Load from config manager with fallbacks
            config_data = config_manager.get('training_config.feature_engineering', {})
            self.config = FeatureEngineeringConfig(
                zmq_subscriber_port=zmq_subscriber_port if zmq_subscriber_port != 5555 else config_data.get('subscriber_port', 5555),
                zmq_publisher_port=zmq_publisher_port if zmq_publisher_port != 5556 else config_data.get('publisher_port', 5556),
                memory_cache_max_gb=memory_cache_max_gb if memory_cache_max_gb != 200.0 else config_data.get('memory_cache_gb', 32.0)
            )
        
        # Hardware optimization detection
        self.is_arm64 = platform.machine().lower() in ['arm64', 'aarch64']
        self.is_gh200 = self._detect_gh200_hardware()
        
        if self.is_arm64 and self.config.enable_arm64_optimizations:
            logger.info("ARM64 optimizations enabled for FeatureEngineeringProcess")
            # Optimize chunk size for ARM64 cache lines
            self.config.chunk_size = min(self.config.chunk_size, 8192)
        
        if self.is_gh200 and self.config.enable_gh200_optimizations:
            logger.info("GH200 unified memory optimizations enabled")
        
        # Initialize GH200 memory pool
        if self.config.enable_gh200_optimizations:
            self.gh200_pool = GH200FeaturePool(self.config.gh200_memory_pool_gb)
        else:
            self.gh200_pool = None
        
        # Initialize ARM64 cache
        if self.is_arm64 and self.config.enable_arm64_optimizations:
            self.arm64_cache = ARM64FeatureCache(max_memory_gb=10.0)
        else:
            self.arm64_cache = None
        
        # Initialize streaming calculator
        if self.config.enable_streaming_features:
            self.streaming_calculator = StreamingFeatureCalculator(
                max_symbols=self.config.max_symbols,
                lookback_periods=self.config.lookback_periods
            )
        else:
            self.streaming_calculator = None
        
        # Initialize symbol priority manager
        self.priority_manager = SymbolPriorityManager()
        
        # Initialize batch processor
        self.batch_processor = MultiSymbolBatchProcessor(
            batch_size=self.config.symbol_batch_size
        )
        
        # Initialize components
        try:
            # Initialize ZMQ publisher with optimizations
            publisher_config = {
                'port': self.config.zmq_publisher_port,
                'enable_arm64_optimizations': self.config.enable_arm64_optimizations and self.is_arm64,
                'compression': 'lz4' if self.is_arm64 else 'none',
                'enable_monitoring': self.config.enable_performance_monitoring
            }
            self.zmq_publisher = ZMQPublisher(**publisher_config)
            logger.info(f"ZMQ publisher initialized on port {self.config.zmq_publisher_port}")
        except Exception as e:
            logger.error(f"Failed to initialize ZMQ publisher: {e}")
            raise
        
        try:
            self.feature_cache = UnifiedMemoryManager(max_memory_gb=self.config.memory_cache_max_gb)
            logger.info(f"Feature cache initialized with {self.config.memory_cache_max_gb}GB capacity")
        except Exception as e:
            logger.error(f"Failed to initialize feature cache: {e}")
            raise
        
        # State management
        self.is_running = False
        self.stats = FeatureEngineeringStats()
        self.processing_cache: Dict[str, Any] = {}
        
        # Thread pool for parallel processing
        if self.config.enable_parallel_processing:
            self.executor = ThreadPoolExecutor(max_workers=self.config.max_workers)
        else:
            self.executor = None
        
        # Pre-compile Numba functions for better performance
        self._warm_up_numba_functions()
        
        logger.info("Enhanced FeatureEngineeringProcess initialized with GH200 and ARM64 optimizations")

    def _detect_gh200_hardware(self) -> bool:
        """Detect if running on GH200 hardware."""
        try:
            # Check for GH200 specific indicators
            # This is a placeholder - actual detection would check GPU info
            import subprocess
            result = subprocess.run(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'], 
                                  capture_output=True, text=True, timeout=5)
            return 'H200' in result.stdout or 'GH200' in result.stdout
        except:
            return False

    def __del__(self):
        """Cleanup resources on destruction."""
        if hasattr(self, 'executor') and self.executor:
            self.executor.shutdown(wait=False)
        
        # Cleanup ARM64 cache
        if hasattr(self, 'arm64_cache') and self.arm64_cache:
            try:
                import shutil
                shutil.rmtree(self.arm64_cache.cache_dir, ignore_errors=True)
            except:
                pass

    def _warm_up_numba_functions(self):
        """Pre-compile Numba functions to avoid compilation overhead during runtime."""
        start_time = time.perf_counter()
        
        try:
            # Create dummy data for compilation
            dummy_prices = np.random.rand(100) * 100
            dummy_high = dummy_prices + np.random.rand(100) * 5
            dummy_low = dummy_prices - np.random.rand(100) * 5
            dummy_volume = np.random.rand(100) * 1000
            
            # Warm up all Numba functions
            _calculate_rsi_numba(dummy_prices)
            _calculate_rsi_vectorized_arm64(dummy_prices)
            _calculate_macd_numba(dummy_prices)
            _calculate_bollinger_bands_numba(dummy_prices)
            _calculate_momentum_numba(dummy_prices)
            _calculate_stochastic_numba(dummy_high, dummy_low, dummy_prices)
            _calculate_atr_numba(dummy_high, dummy_low, dummy_prices)
            _calculate_williams_r_numba(dummy_high, dummy_low, dummy_prices)
            _calculate_multiple_indicators_batch(dummy_prices, dummy_high, dummy_low, dummy_volume)
            
            compilation_time = time.perf_counter() - start_time
            self.stats.numba_compilation_time = compilation_time
            logger.info(f"Numba functions pre-compiled in {compilation_time:.4f}s")
            
        except Exception as e:
            logger.warning(f"Numba warm-up failed: {e}")

    def calculate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Enhanced feature calculation with GH200 unified memory and ARM64 optimizations.

        Args:
            df: DataFrame containing OHLCV data with a DatetimeIndex.

        Returns:
            DataFrame with engineered features.
        """
        if df.empty:
            logger.warning("Input DataFrame is empty, returning empty features DataFrame.")
            return pd.DataFrame()

        start_time = time.perf_counter()
        
        try:
            # Use GH200 memory pool if available
            if self.gh200_pool and len(df) > 1000:
                return self._calculate_features_gh200(df)
            
            # Use ARM64 vectorized calculations for medium datasets
            elif self.is_arm64 and self.config.enable_arm64_optimizations and len(df) > 100:
                return self._calculate_features_arm64_vectorized(df)
            
            # Standard calculation for small datasets
            else:
                return self._calculate_features_standard(df)
                
        except Exception as e:
            logger.error(f"Error calculating features: {e}")
            self.stats.errors += 1
            return pd.DataFrame()

    def _calculate_features_gh200(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate features using GH200 unified memory pool."""
        symbol = "temp_symbol"  # In practice, this would be passed as parameter
        num_features = 20  # Estimated number of features
        num_periods = len(df)
        
        # Allocate from GH200 memory pool
        feature_array = self.gh200_pool.allocate_feature_array(symbol, num_features, num_periods)
        
        # Extract price arrays
        close_prices = df['close'].values.astype(np.float64)
        high_prices = df['high'].values.astype(np.float64) if 'high' in df.columns else close_prices
        low_prices = df['low'].values.astype(np.float64) if 'low' in df.columns else close_prices
        volume = df['volume'].values.astype(np.float64) if 'volume' in df.columns else np.ones_like(close_prices)
        
        # Use batch calculation for maximum efficiency
        batch_indicators = _calculate_multiple_indicators_batch(close_prices, high_prices, low_prices, volume)
        
        # Create features DataFrame
        features_df = pd.DataFrame(index=df.index)
        
        # Map batch results to feature columns
        feature_names = ['rsi_14', 'macd', 'bb_upper', 'bb_middle', 'bb_lower', 'atr']
        for i, name in enumerate(feature_names):
            if i < batch_indicators.shape[1]:
                features_df[name] = batch_indicators[:, i]
        
        # Add additional features using standard calculations
        features_df = self._add_additional_features(df, features_df)
        
        # Update statistics
        self.stats.gh200_optimizations_used += 1
        self.stats.memory_pool_allocations += 1
        
        # Deallocate from pool
        self.gh200_pool.deallocate_symbol(symbol)
        
        return features_df

    def _calculate_features_arm64_vectorized(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate features using ARM64 vectorized operations."""
        features_df = pd.DataFrame(index=df.index)
        
        # Extract price arrays
        close_prices = df['close'].values.astype(np.float64)
        high_prices = df['high'].values.astype(np.float64) if 'high' in df.columns else close_prices
        low_prices = df['low'].values.astype(np.float64) if 'low' in df.columns else close_prices
        
        # Use ARM64 optimized vectorized calculations
        features_df['rsi_14'] = _calculate_rsi_vectorized_arm64(close_prices, window=14)
        features_df['rsi_28'] = _calculate_rsi_vectorized_arm64(close_prices, window=28)
        
        # Standard calculations for other indicators
        macd, signal, hist = _calculate_macd_numba(close_prices)
        features_df['macd'] = macd
        features_df['macd_signal'] = signal
        features_df['macd_hist'] = hist
        
        upper_bb, middle_bb, lower_bb = _calculate_bollinger_bands_numba(close_prices)
        features_df['bb_upper'] = upper_bb
        features_df['bb_middle'] = middle_bb
        features_df['bb_lower'] = lower_bb
        
        # Add additional features
        features_df = self._add_additional_features(df, features_df)
        
        # Update statistics
        self.stats.arm64_optimizations_used += 1
        
        return features_df

    def _calculate_features_standard(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standard feature calculation implementation."""
        features_df = pd.DataFrame(index=df.index)
        
        close_prices = df['close'].values.astype(np.float64)
        high_prices = df['high'].values.astype(np.float64) if 'high' in df.columns else close_prices
        low_prices = df['low'].values.astype(np.float64) if 'low' in df.columns else close_prices

        # Momentum Indicators
        features_df['rsi_14'] = _calculate_rsi_numba(close_prices, window=14)
        features_df['rsi_28'] = _calculate_rsi_numba(close_prices, window=28)

        macd, signal, hist = _calculate_macd_numba(close_prices)
        features_df['macd'] = macd
        features_df['macd_signal'] = signal
        features_df['macd_hist'] = hist

        features_df['momentum_10'] = _calculate_momentum_numba(close_prices, window=10)
        features_df['momentum_20'] = _calculate_momentum_numba(close_prices, window=20)

        # Volatility Indicators
        features_df['volatility_20'] = df['close'].rolling(window=20).std()
        features_df['volatility_60'] = df['close'].rolling(window=60).std()

        upper_bb, middle_bb, lower_bb = _calculate_bollinger_bands_numba(close_prices)
        features_df['bb_upper'] = upper_bb
        features_df['bb_middle'] = middle_bb
        features_df['bb_lower'] = lower_bb

        # Advanced indicators if enabled
        if self.config.enable_advanced_features:
            k_percent, d_percent = _calculate_stochastic_numba(high_prices, low_prices, close_prices)
            features_df['stoch_k'] = k_percent
            features_df['stoch_d'] = d_percent
            
            features_df['atr'] = _calculate_atr_numba(high_prices, low_prices, close_prices)
            features_df['williams_r'] = _calculate_williams_r_numba(high_prices, low_prices, close_prices)

        # Add additional features
        features_df = self._add_additional_features(df, features_df)

        return features_df

    def _add_additional_features(self, df: pd.DataFrame, features_df: pd.DataFrame) -> pd.DataFrame:
        """Add additional features common to all calculation methods."""
        # Volume Indicators
        if 'volume' in df.columns:
            features_df['volume_ma_10'] = df['volume'].rolling(window=10).mean()
            features_df['volume_ma_30'] = df['volume'].rolling(window=30).mean()
            features_df['on_balance_volume'] = (np.sign(df['close'].diff()) * df['volume']).cumsum()
            features_df['volume_rate_of_change'] = df['volume'].pct_change(periods=10)

        # Price-based features
        features_df['daily_return'] = df['close'].pct_change()
        features_df['log_return'] = np.log(df['close'] / df['close'].shift(1))
        
        if 'high' in df.columns and 'low' in df.columns:
            features_df['high_low_range'] = df['high'] - df['low']
            features_df['high_low_ratio'] = df['high'] / df['low']
        
        if 'open' in df.columns:
            features_df['open_close_range'] = df['close'] - df['open']
            features_df['gap'] = df['open'] - df['close'].shift(1)

        # Moving averages
        for window in [5, 10, 20, 50]:
            features_df[f'sma_{window}'] = df['close'].rolling(window=window).mean()
            features_df[f'ema_{window}'] = df['close'].ewm(span=window).mean()

        # Handle NaNs
        if self.is_arm64 and self.config.enable_arm64_optimizations:
            features_df = self._handle_nans_arm64(features_df)
        else:
            features_df = features_df.fillna(method='ffill').fillna(method='bfill')

        return features_df

    def _handle_nans_arm64(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """ARM64 optimized NaN handling with vectorized operations."""
        # Use vectorized operations for better ARM64 SIMD utilization
        numeric_columns = features_df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_columns:
            # Forward fill then backward fill
            features_df[col] = features_df[col].fillna(method='ffill').fillna(method='bfill')
        
        return features_df

    def update_features_streaming(self, symbol: str, price: float, volume: float, timestamp: int) -> Dict[str, float]:
        """Update features for single price tick using streaming calculator."""
        if not self.streaming_calculator:
            logger.warning("Streaming calculator not initialized")
            return {}
        
        # Check if symbol should be processed based on priority
        if not self.priority_manager.should_process_symbol(symbol):
            return {}
        
        # Update streaming features
        features = self.streaming_calculator.update_features_streaming(symbol, price, volume, timestamp)
        
        # Update processing time
        self.priority_manager.update_processing_time(symbol)
        
        # Update statistics
        self.stats.streaming_updates += 1
        
        return features

    async def process_symbol_batch_async(self, symbol_data_pairs: List[Tuple[str, pd.DataFrame]]) -> Dict[str, Any]:
        """Process multiple symbols asynchronously using batch processor."""
        if not self.batch_processor:
            logger.warning("Batch processor not initialized")
            return {}
        
        # Process batch
        results = await self.batch_processor.process_symbol_batch(symbol_data_pairs)
        
        # Update statistics
        self.stats.batch_operations += 1
        
        return results

    def process_and_publish_features(self, symbol: str, market_data_df: pd.DataFrame):
        """
        Enhanced feature processing and publishing with GH200 and ARM64 optimizations.

        Args:
            symbol: The symbol for which features are being processed.
            market_data_df: DataFrame containing preprocessed OHLCV data.
        """
        if market_data_df.empty:
            logger.warning(f"No market data for {symbol}, skipping feature engineering.")
            return

        try:
            start_time = time.perf_counter()
            
            # Check cache for recent features
            cache_key = f"features_{symbol}_{hash(str(market_data_df.index[-1]))}"
            
            if self.config.enable_caching and cache_key in self.processing_cache:
                engineered_features = self.processing_cache[cache_key]
                self.stats.cache_hits += 1
                logger.debug(f"Using cached features for {symbol}")
            else:
                engineered_features = self.calculate_features(market_data_df)
                if self.config.enable_caching and not engineered_features.empty:
                    self.processing_cache[cache_key] = engineered_features
                    # Limit cache size
                    if len(self.processing_cache) > self.config.cache_size:
                        oldest_key = next(iter(self.processing_cache))
                        del self.processing_cache[oldest_key]
                    self.stats.cache_misses += 1

            processing_time = time.perf_counter() - start_time
            logger.info(f"Feature engineering for {symbol} took {processing_time:.4f} seconds.")

            if engineered_features.empty:
                logger.warning(f"No features engineered for {symbol}.")
                return

            # Cache in ARM64 optimized storage if available
            if self.arm64_cache:
                try:
                    features_data = engineered_features.values.astype(np.float64)  # 8-byte alignment
                    cache_key = self.arm64_cache.store_features(
                        symbol=symbol,
                        features=features_data,
                        feature_names=engineered_features.columns.tolist()
                    )
                    logger.debug(f"Cached engineered features for {symbol} in ARM64 cache.")
                except Exception as e:
                    logger.error(f"Failed to cache features in ARM64 cache for {symbol}: {e}")

            # Cache in unified memory with optimizations
            try:
                if self.is_arm64 and self.config.enable_arm64_optimizations:
                    # ARM64 optimized storage format
                    features_data = engineered_features.values.astype(np.float64)  # Ensure 8-byte alignment
                else:
                    features_data = engineered_features.values
                
                self.feature_cache.store_features(
                    symbol=symbol,
                    features=features_data,
                    feature_names=engineered_features.columns.tolist()
                )
                logger.debug(f"Cached engineered features for {symbol}.")
            except Exception as e:
                logger.error(f"Failed to cache features for {symbol}: {e}")

            # Publish to model processes with enhanced message format
            try:
                latest_features = engineered_features.iloc[-1].to_dict()
                
                # Enhanced message format with optimizations
                message = {
                    'symbol': symbol,
                    'timestamp': int(market_data_df.index[-1].timestamp() * 1e9),
                    'signal': 0.0,  # Placeholder for feature data
                    'confidence': 0.0,  # Placeholder
                    'strategy': 'feature_engineering',
                    'features': latest_features,
                    'feature_count': len(latest_features),
                    'processing_time_ms': processing_time * 1000,
                    'arm64_optimized': self.is_arm64 and self.config.enable_arm64_optimizations,
                    'gh200_optimized': self.is_gh200 and self.config.enable_gh200_optimizations
                }
                
                self.zmq_publisher.publish_trading_signal(message)
                logger.debug(f"Published latest features for {symbol}.")
                
            except Exception as e:
                logger.error(f"Failed to publish features for {symbol}: {e}")

            # Update statistics
            self.stats.symbols_processed += 1
            self.stats.features_calculated += len(engineered_features.columns)
            self.stats.processing_time_seconds += processing_time

        except Exception as e:
            logger.error(f"Error in feature engineering for {symbol}: {e}", exc_info=True)
            self.stats.errors += 1

    def get_cached_features(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve cached features for a symbol.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Cached features or None if not found
        """
        try:
            return self.feature_cache.get_features(symbol)
        except Exception as e:
            logger.error(f"Error retrieving cached features for {symbol}: {e}")
            return None

    def get_statistics(self) -> FeatureEngineeringStats:
        """Get current feature engineering statistics."""
        return self.stats

    def reset_statistics(self):
        """Reset feature engineering statistics."""
        self.stats = FeatureEngineeringStats()
        logger.info("Feature engineering statistics reset")

    def clear_cache(self):
        """Clear the processing cache."""
        self.processing_cache.clear()
        if self.arm64_cache:
            # Clear ARM64 cache files
            try:
                import shutil
                shutil.rmtree(self.arm64_cache.cache_dir, ignore_errors=True)
                os.makedirs(self.arm64_cache.cache_dir, exist_ok=True)
            except Exception as e:
                logger.warning(f"Failed to clear ARM64 cache: {e}")
        logger.info("Feature engineering cache cleared")

    def health_check(self) -> Dict[str, Any]:
        """
        Perform a health check of the feature engineering process.
        
        Returns:
            Health status dictionary
        """
        health = {
            'status': 'healthy',
            'timestamp': time.time(),
            'components': {},
            'statistics': {},
            'optimizations': {}
        }
        
        try:
            # Check ZMQ publisher
            health['components']['zmq_publisher'] = {
                'status': 'active' if hasattr(self.zmq_publisher, 'is_active') else 'unknown'
            }
            
            # Check feature cache
            health['components']['feature_cache'] = {
                'status': 'active',
                'memory_usage_gb': getattr(self.feature_cache, 'current_memory_gb', 0)
            }
            
            # Check GH200 pool
            if self.gh200_pool:
                memory_usage = self.gh200_pool.get_memory_usage()
                health['components']['gh200_pool'] = {
                    'status': 'active',
                    'memory_usage': memory_usage
                }
            
            # Check ARM64 cache
            if self.arm64_cache:
                health['components']['arm64_cache'] = {
                    'status': 'active',
                    'memory_usage_gb': self.arm64_cache.current_memory_usage / (1024**3)
                }
            
            # Add optimization status
            health['optimizations'] = {
                'arm64_enabled': self.is_arm64 and self.config.enable_arm64_optimizations,
                'gh200_enabled': self.is_gh200 and self.config.enable_gh200_optimizations,
                'streaming_enabled': self.config.enable_streaming_features,
                'parallel_enabled': self.config.enable_parallel_processing
            }
            
            # Add statistics
            stats = self.get_statistics()
            health['statistics'] = {
                'features_calculated': stats.features_calculated,
                'features_per_second': stats.features_per_second,
                'symbols_processed': stats.symbols_processed,
                'error_rate': stats.errors / max(stats.symbols_processed, 1),
                'cache_hit_rate': stats.cache_hits / max(stats.cache_hits + stats.cache_misses, 1),
                'uptime_seconds': stats.uptime_seconds,
                'arm64_optimizations_used': stats.arm64_optimizations_used,
                'gh200_optimizations_used': stats.gh200_optimizations_used,
                'streaming_updates': stats.streaming_updates,
                'batch_operations': stats.batch_operations
            }
            
            # Determine overall health
            error_rate = health['statistics']['error_rate']
            if error_rate > 0.1:  # More than 10% error rate
                health['status'] = 'degraded'
            elif error_rate > 0.05:  # More than 5% error rate
                health['status'] = 'warning'
                
        except Exception as e:
            health['status'] = 'error'
            health['error'] = str(e)
            logger.error(f"Health check failed: {e}")
        
        return health

    def run(self):
        """
        Enhanced run method with GH200 and ARM64 optimizations.
        """
        logger.info("Enhanced FeatureEngineeringProcess ready with GH200 and ARM64 optimizations.")
        self.is_running = True
        
        # Log configuration
        logger.info(f"Configuration: ARM64={self.is_arm64}, GH200={self.is_gh200}, "
                   f"Parallel={self.config.enable_parallel_processing}, "
                   f"Workers={self.config.max_workers}, "
                   f"Streaming={self.config.enable_streaming_features}")

    def stop(self):
        """Enhanced stop method with proper cleanup and statistics logging."""
        logger.info("Stopping FeatureEngineeringProcess...")
        self.is_running = False
        
        try:
            # Log final statistics
            stats = self.get_statistics()
            logger.info("Final Feature Engineering Statistics:")
            logger.info(f"  Features Calculated: {stats.features_calculated}")
            logger.info(f"  Symbols Processed: {stats.symbols_processed}")
            logger.info(f"  Features/Second: {stats.features_per_second:.2f}")
            logger.info(f"  Cache Hit Rate: {stats.cache_hits / max(stats.cache_hits + stats.cache_misses, 1):.2%}")
            logger.info(f"  Error Rate: {stats.errors / max(stats.symbols_processed, 1):.2%}")
            logger.info(f"  ARM64 Optimizations Used: {stats.arm64_optimizations_used}")
            logger.info(f"  GH200 Optimizations Used: {stats.gh200_optimizations_used}")
            logger.info(f"  Streaming Updates: {stats.streaming_updates}")
            logger.info(f"  Batch Operations: {stats.batch_operations}")
            logger.info(f"  Uptime: {stats.uptime_seconds:.2f} seconds")
            
            # Close ZMQ publisher
            if hasattr(self.zmq_publisher, 'close'):
                self.zmq_publisher.close()
            
            # Shutdown thread pool
            if self.executor:
                self.executor.shutdown(wait=True)
            
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")
        
        logger.info("FeatureEngineeringProcess stopped successfully.")

if __name__ == "__main__":
    # Enhanced example usage with GH200 and ARM64 optimizations
    import warnings
    warnings.filterwarnings('ignore')
    
    # Create comprehensive test data
    np.random.seed(42)
    n_records = 10000  # Larger dataset to test optimizations
    
    data = {
        'timestamp': pd.to_datetime(pd.date_range(start='2023-01-01', periods=n_records, freq='1min')),
        'open': np.random.rand(n_records) * 100 + 100,
        'high': np.random.rand(n_records) * 100 + 105,
        'low': np.random.rand(n_records) * 100 + 95,
        'close': np.random.rand(n_records) * 100 + 100,
        'volume': np.random.randint(1000, 10000, n_records)
    }
    df = pd.DataFrame(data).set_index('timestamp')

    print("Original DataFrame info:")
    print(f"Shape: {df.shape}")
    print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

    # Initialize enhanced feature engineering process
    config = FeatureEngineeringConfig(
        enable_arm64_optimizations=True,
        enable_gh200_optimizations=True,
        enable_parallel_processing=True,
        max_workers=4,
        enable_performance_monitoring=True,
        enable_advanced_features=True,
        enable_streaming_features=True,
        gh200_memory_pool_gb=10.0
    )
    
    feature_engineer = FeatureEngineeringProcess(config=config)

    # Test feature calculation
    start_time = time.time()
    engineered_df = feature_engineer.calculate_features(df.copy())
    total_time = time.time() - start_time

    print(f"\nEngineered Features DataFrame info:")
    print(f"Shape: {engineered_df.shape}")
    print(f"Feature calculation time: {total_time:.4f}s")
    print(f"Features per second: {len(engineered_df.columns) / total_time:.2f}")

    # Display statistics
    stats = feature_engineer.get_statistics()
    print(f"\nPerformance Statistics:")
    print(f"Features calculated: {stats.features_calculated}")
    print(f"ARM64 optimizations used: {stats.arm64_optimizations_used}")
    print(f"GH200 optimizations used: {stats.gh200_optimizations_used}")
    print(f"Numba compilation time: {stats.numba_compilation_time:.4f}s")

    # Test streaming features
    if feature_engineer.streaming_calculator:
        print(f"\nTesting streaming features...")
        streaming_features = feature_engineer.update_features_streaming(
            "TEST", 100.5, 1000.0, int(time.time() * 1e9)
        )
        print(f"Streaming features: {streaming_features}")

    # Test processing and publishing
    print(f"\nTesting feature processing for symbol 'TEST'...")
    feature_engineer.process_and_publish_features("TEST", df.copy())
    
    # Health check
    health = feature_engineer.health_check()
    print(f"\nHealth Status: {health['status']}")
    print(f"Error Rate: {health['statistics']['error_rate']:.2%}")
    print(f"Optimizations: {health['optimizations']}")
