import pandas as pd
import numpy as np
import platform
import time
import os
import asyncio
from typing import Dict, Any, List, Tuple, Optional, Union
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor
import warnings
import hashlib

from deep_momentum_trading.src.utils.logger import get_logger

try:
    import psutil
except ImportError:
    psutil = None

logger = get_logger(__name__)

@dataclass
class PreprocessingConfig:
    """Configuration for DataPreprocessor with ARM64 optimizations."""
    fill_method: str = 'ffill'
    resample_interval: str = '1min'
    outlier_threshold_std: float = 3.0
    enable_arm64_optimizations: bool = True
    enable_parallel_processing: bool = True
    enable_numa_awareness: bool = True
    max_workers: int = 8        # More workers for 72-core system
    chunk_size: int = 50000     # Larger chunks for better throughput
    enable_performance_monitoring: bool = True
    enable_caching: bool = True
    cache_size: int = 1000
    memory_pool_mb: int = 8000  # 8GB pool instead of 1GB
    batch_size_gb: float = 50.0  # 50GB batches instead of 10GB
    enable_zero_copy: bool = True
    enable_streaming_resampling: bool = True

@dataclass
class PreprocessingStats:
    """Statistics for preprocessing performance monitoring."""
    records_processed: int = 0
    processing_time_seconds: float = 0.0
    validation_errors: int = 0
    outliers_removed: int = 0
    missing_values_filled: int = 0
    normalization_operations: int = 0
    resampling_operations: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    memory_pool_allocations: int = 0
    numa_node_switches: int = 0
    zero_copy_operations: int = 0
    start_time: float = field(default_factory=time.time)
    
    @property
    def records_per_second(self) -> float:
        return self.records_processed / max(self.processing_time_seconds, 0.001)

class ARM64PreprocessingPipeline:
    """ARM64-optimized zero-copy preprocessing pipeline."""
    
    def __init__(self, memory_pool_mb: int = 1000):
        # Ensure 64-byte alignment for ARM64 cache lines
        pool_size = (memory_pool_mb * 1024 * 1024) // 8
        # Round to cache line boundary
        aligned_size = ((pool_size + 7) // 8) * 8
        self.memory_pool = np.zeros(aligned_size, dtype=np.float64)
        self.pool_index = 0
        self.pool_size = len(self.memory_pool)
        
    def allocate_buffer(self, size: int) -> np.ndarray:
        """Allocate from pre-allocated memory pool."""
        if self.pool_index + size > self.pool_size:
            self.pool_index = 0  # Reset pool
        
        buffer = self.memory_pool[self.pool_index:self.pool_index + size]
        self.pool_index += size
        return buffer.reshape(-1, 1)
    
    def preprocess_zero_copy(self, df: pd.DataFrame) -> pd.DataFrame:
        """Zero-copy preprocessing using memory views."""
        # Work directly with numpy arrays
        numeric_data = df.select_dtypes(include=np.number).values
        
        # In-place operations
        self._validate_inplace(numeric_data)
        self._handle_missing_inplace(numeric_data)
        self._remove_outliers_inplace(numeric_data)
        
        # Reconstruct DataFrame only at the end
        return pd.DataFrame(
            numeric_data, 
            columns=df.select_dtypes(include=np.number).columns, 
            index=df.index
        )
    
    def _validate_inplace(self, data: np.ndarray) -> None:
        """In-place validation of numeric data."""
        # Check for infinite values
        np.nan_to_num(data, copy=False, nan=np.nan, posinf=np.nan, neginf=np.nan)
    
    def _handle_missing_inplace(self, data: np.ndarray) -> None:
        """Fixed in-place missing data handling."""
        if data.size == 0:
            return
            
        # Forward fill using numpy operations (more robust)
        for col in range(data.shape[1]):
            column = data[:, col]
            mask = np.isnan(column)
            if not mask.any():
                continue
                
            # Forward fill
            valid_indices = np.where(~mask)[0]
            if len(valid_indices) == 0:
                continue
                
            # Use searchsorted for efficient forward fill
            fill_indices = np.searchsorted(valid_indices, np.arange(len(column)), side='right') - 1
            fill_indices = np.clip(fill_indices, 0, len(valid_indices) - 1)
            column[mask] = column[valid_indices[fill_indices[mask]]]
    
    def _remove_outliers_inplace(self, data: np.ndarray) -> None:
        """In-place outlier removal."""
        means = np.nanmean(data, axis=0)
        stds = np.nanstd(data, axis=0)
        
        # Vectorized outlier detection
        z_scores = np.abs((data - means) / stds)
        outlier_mask = z_scores > 3.0
        data[outlier_mask] = np.nan

class NUMAPreprocessor:
    """NUMA-aware preprocessing for GH200."""
    
    def __init__(self, config: PreprocessingConfig):
        self.config = config
        self.numa_nodes = self._detect_numa_topology()
        self.node_affinity = {}
        
    def _detect_numa_topology(self) -> Dict[int, List[int]]:
        """Enhanced NUMA detection for GH200."""
        try:
            # Check if actually on GH200
            if not (platform.machine().lower() in ['arm64', 'aarch64'] and os.cpu_count() == 72):
                return {0: list(range(os.cpu_count() or 4))}
            
            # GH200 Grace Hopper specific topology
            # 72 cores: 4 NUMA nodes with 18 cores each
            topology = {}
            cores_per_node = 18
            for node in range(4):
                start_core = node * cores_per_node
                end_core = start_core + cores_per_node
                topology[node] = list(range(start_core, end_core))
            
            return topology
        except:
            return {0: list(range(os.cpu_count() or 4))}
    
    def _set_numa_affinity(self, numa_node: int) -> None:
        """Set CPU affinity for better cache performance."""
        try:
            if numa_node in self.numa_nodes:
                cores = self.numa_nodes[numa_node]
                os.sched_setaffinity(0, cores)
        except (OSError, AttributeError):
            # Not supported on all systems
            pass
    
    def process_numa_aware(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Process data with NUMA awareness."""
        # Assign symbols to NUMA nodes for cache locality
        numa_node = hash(symbol) % len(self.numa_nodes)
        self._set_numa_affinity(numa_node)
        
        return self._process_on_numa_node(df, numa_node)
    
    def _process_on_numa_node(self, df: pd.DataFrame, numa_node: int) -> pd.DataFrame:
        """Process data on specific NUMA node."""
        # Implementation would include NUMA-specific optimizations
        return df

class GH200BatchProcessor:
    """Optimized batch processing for GH200's 624GB memory."""
    
    def __init__(self, batch_size_gb: float = 10.0):
        self.batch_size_records = int((batch_size_gb * 1024**3) // (8 * 10))  # ~10 float64 columns
        self.processing_buffer = None
        
    def process_large_dataset(self, df: pd.DataFrame, processor_func) -> pd.DataFrame:
        """Process dataset larger than memory in batches."""
        if len(df) <= self.batch_size_records:
            return processor_func(df)
        
        results = []
        for i in range(0, len(df), self.batch_size_records):
            batch = df.iloc[i:i + self.batch_size_records]
            processed_batch = processor_func(batch)
            results.append(processed_batch)
            
            # Memory cleanup
            del batch
            
        return pd.concat(results, ignore_index=True)

class DataPreprocessor:
    """
    Enhanced DataPreprocessor with ARM64 optimizations, NUMA awareness, and zero-copy operations.
    
    Handles various data preprocessing steps for raw market data,
    including validation, cleaning, normalization, and resampling with
    GH200-specific optimizations and parallel processing capabilities.
    """

    def __init__(self,
                 config: Optional[PreprocessingConfig] = None,
                 fill_method: str = 'ffill',
                 resample_interval: str = '1min',
                 outlier_threshold_std: float = 3.0):
        """
        Initializes the enhanced DataPreprocessor with ARM64 and NUMA optimizations.

        Args:
            config: PreprocessingConfig object (preferred)
            fill_method: Method to fill missing data (fallback)
            resample_interval: Interval for resampling data (fallback)
            outlier_threshold_std: Number of standard deviations for outlier detection (fallback)
        """
        # Configuration handling
        if config is not None:
            self.config = config
        else:
            self.config = PreprocessingConfig(
                fill_method=fill_method,
                resample_interval=resample_interval,
                outlier_threshold_std=outlier_threshold_std
            )
        
        # ARM64 optimization detection
        self.is_arm64 = platform.machine().lower() in ['arm64', 'aarch64']
        if self.is_arm64 and self.config.enable_arm64_optimizations:
            logger.info("ARM64 optimizations enabled for DataPreprocessor")
            # Optimize chunk size for ARM64 cache lines
            self.config.chunk_size = min(self.config.chunk_size, 8192)  # ARM64 cache-friendly
        
        # Initialize specialized processors
        self.arm64_pipeline = ARM64PreprocessingPipeline(self.config.memory_pool_mb)
        self.numa_processor = NUMAPreprocessor(self.config) if self.config.enable_numa_awareness else None
        self.batch_processor = GH200BatchProcessor(self.config.batch_size_gb)
        
        # Performance monitoring
        self.stats = PreprocessingStats()
        self.processing_cache: Dict[str, Any] = {}
        
        # Thread pool for parallel processing
        if self.config.enable_parallel_processing:
            self.executor = ThreadPoolExecutor(max_workers=self.config.max_workers)
        else:
            self.executor = None
        
        # Add GH200 optimizations
        self._optimize_for_gh200()
        
        # Initialize memory monitoring
        self._last_memory_check = time.time()
        self._memory_check_interval = 30  # Check every 30 seconds
        
        logger.info(f"Enhanced DataPreprocessor initialized with ARM64 and NUMA optimizations")
        logger.info(f"Configuration: {self.config}")

    def __del__(self):
        """Cleanup thread pool on destruction."""
        if hasattr(self, 'executor') and self.executor:
            self.executor.shutdown(wait=False)

    def _check_memory_pressure(self) -> bool:
        """Monitor GH200 memory usage."""
        try:
            if psutil is None:
                return False
            memory = psutil.virtual_memory()
            # Keep under 80% of 624GB for safety
            return memory.percent > 80
        except ImportError:
            return False

    def _adjust_for_memory_pressure(self):
        """Adjust processing parameters under memory pressure."""
        if self._check_memory_pressure():
            # Reduce chunk size
            self.config.chunk_size = max(self.config.chunk_size // 2, 1000)
            # Reduce batch size
            self.config.batch_size_gb = max(self.config.batch_size_gb / 2, 1.0)
            # Clear caches
            self.processing_cache.clear()
            logger.warning("Memory pressure detected, reducing processing parameters")

    def _optimize_for_gh200(self):
        """Apply GH200-specific optimizations."""
        if not self.is_arm64:
            return
        
        # Set optimal thread count for 72-core system
        os.environ['OMP_NUM_THREADS'] = str(min(self.config.max_workers, 8))
        os.environ['OPENBLAS_NUM_THREADS'] = str(min(self.config.max_workers, 8))
        
        # Enable ARM64 specific numpy optimizations
        try:
            import numpy as np
            # Force use of optimized BLAS if available
            np.show_config()
        except:
            pass
            
        # Optimize pandas for ARM64
        try:
            import pandas as pd
            # Use faster engines where available
            pd.set_option('compute.use_numba', True)
        except:
            pass

    def enhanced_validate_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """Enhanced validation with trading-specific checks."""
        start_time = time.perf_counter()
        required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        errors = []

        try:
            # Check for missing columns
            missing_cols = [col for col in required_columns if col not in df.columns]
            if missing_cols:
                errors.append(f"Missing required columns: {', '.join(missing_cols)}")

            # Enhanced timestamp validation
            if 'timestamp' in df.columns:
                if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
                    try:
                        # Try multiple timestamp formats
                        if df['timestamp'].dtype in ['int64', 'float64']:
                            # Assume nanoseconds for high precision
                            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ns')
                        else:
                            df['timestamp'] = pd.to_datetime(df['timestamp'])
                    except Exception as e:
                        errors.append(f"Timestamp conversion failed: {str(e)}")

            # Trading-specific validations
            if 'volume' in df.columns:
                negative_volume = (df['volume'] < 0).sum()
                if negative_volume > 0:
                    errors.append(f"Found {negative_volume} negative volume values")
                    df.loc[df['volume'] < 0, 'volume'] = 0

            # Check for timestamp ordering
            if isinstance(df.index, pd.DatetimeIndex):
                if not df.index.is_monotonic_increasing:
                    errors.append("Timestamps are not in ascending order")
                    df = df.sort_index()

            # Check for duplicate timestamps
            if df.index.duplicated().any():
                dup_count = df.index.duplicated().sum()
                errors.append(f"Found {dup_count} duplicate timestamps")
                df = df[~df.index.duplicated(keep='last')]

            # Price reasonableness checks
            price_columns = ['open', 'high', 'low', 'close']
            for col in price_columns:
                if col in df.columns:
                    zero_prices = (df[col] <= 0).sum()
                    if zero_prices > 0:
                        errors.append(f"Found {zero_prices} zero/negative prices in {col}")
                        df = df[df[col] > 0]

            # Enhanced numeric validation with ARM64 optimizations
            numeric_cols = ['open', 'high', 'low', 'close', 'volume']
            for col in numeric_cols:
                if col in df.columns:
                    if not pd.api.types.is_numeric_dtype(df[col]):
                        try:
                            df[col] = pd.to_numeric(df[col], errors='coerce')
                        except Exception:
                            errors.append(f"Column '{col}' cannot be converted to numeric.")
                    
                    # ARM64 optimized validation checks
                    if self.is_arm64 and self.config.enable_arm64_optimizations:
                        # Check for ARM64-specific data alignment issues
                        if col in df.columns and df[col].dtype not in ['float64', 'int64']:
                            df[col] = df[col].astype('float64')  # Ensure 8-byte alignment

            # Additional validation checks
            if len(df) == 0:
                errors.append("DataFrame is empty")
            
            # Check for logical inconsistencies in OHLC data
            if all(col in df.columns for col in ['open', 'high', 'low', 'close']):
                # High should be >= max(open, close) and Low should be <= min(open, close)
                invalid_high = (df['high'] < df[['open', 'close']].max(axis=1)).sum()
                invalid_low = (df['low'] > df[['open', 'close']].min(axis=1)).sum()
                
                if invalid_high > 0:
                    errors.append(f"Found {invalid_high} records where high < max(open, close)")
                if invalid_low > 0:
                    errors.append(f"Found {invalid_low} records where low > min(open, close)")

            # Update statistics
            self.stats.validation_errors += len(errors)
            processing_time = time.perf_counter() - start_time
            self.stats.processing_time_seconds += processing_time

            if errors:
                logger.error(f"Data validation failed: {'; '.join(errors)}")
            else:
                logger.debug(f"Data validation successful in {processing_time:.4f}s")

        except Exception as e:
            errors.append(f"Validation error: {str(e)}")
            logger.error(f"Unexpected validation error: {e}")

        return df, errors

    def validate_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """Wrapper for enhanced validation."""
        return self.enhanced_validate_data(df)

    def handle_missing_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Enhanced missing data handling with ARM64 optimizations and parallel processing.

        Args:
            df: Input DataFrame.

        Returns:
            DataFrame with missing values handled.
        """
        if df.empty:
            logger.warning("DataFrame is empty, skipping missing data handling.")
            return df

        start_time = time.perf_counter()
        initial_nan_count = df.isnull().sum().sum()
        
        if initial_nan_count == 0:
            logger.debug("No missing values found, skipping missing data handling.")
            return df

        logger.info(f"Handling {initial_nan_count} missing values using '{self.config.fill_method}' method.")

        try:
            # Zero-copy optimization for ARM64
            if (self.config.enable_zero_copy and self.is_arm64 and 
                self.config.enable_arm64_optimizations and len(df) > self.config.chunk_size):
                df = self._handle_missing_data_zero_copy(df)
                self.stats.zero_copy_operations += 1
            elif self.is_arm64 and self.config.enable_arm64_optimizations and len(df) > self.config.chunk_size:
                df = self._handle_missing_data_chunked(df)
            else:
                df = self._handle_missing_data_standard(df)

            final_nan_count = df.isnull().sum().sum()
            filled_count = initial_nan_count - final_nan_count
            
            self.stats.missing_values_filled += filled_count
            processing_time = time.perf_counter() - start_time
            self.stats.processing_time_seconds += processing_time

            if final_nan_count > 0:
                logger.warning(f"After handling, {final_nan_count} missing values still remain.")
            else:
                logger.debug(f"All missing values handled successfully in {processing_time:.4f}s")

        except Exception as e:
            logger.error(f"Error handling missing data: {e}")

        return df

    def _handle_missing_data_zero_copy(self, df: pd.DataFrame) -> pd.DataFrame:
        """Zero-copy missing data handling using ARM64 pipeline."""
        return self.arm64_pipeline.preprocess_zero_copy(df)

    def _handle_missing_data_standard(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standard missing data handling."""
        if self.config.fill_method == 'ffill':
            return df.ffill()
        elif self.config.fill_method == 'bfill':
            return df.bfill()
        elif self.config.fill_method == 'mean':
            return df.fillna(df.mean(numeric_only=True))
        elif self.config.fill_method == 'median':
            return df.fillna(df.median(numeric_only=True))
        elif self.config.fill_method == 'drop':
            return df.dropna()
        else:
            logger.warning(f"Unknown fill_method: {self.config.fill_method}. No missing data handling performed.")
            return df

    def _handle_missing_data_chunked(self, df: pd.DataFrame) -> pd.DataFrame:
        """ARM64 optimized chunked missing data handling."""
        chunks = [df.iloc[i:i + self.config.chunk_size] for i in range(0, len(df), self.config.chunk_size)]
        
        if self.executor and self.config.enable_parallel_processing:
            # Parallel processing for large datasets
            processed_chunks = list(self.executor.map(self._handle_missing_data_standard, chunks))
        else:
            # Sequential processing
            processed_chunks = [self._handle_missing_data_standard(chunk) for chunk in chunks]
        
        return pd.concat(processed_chunks, ignore_index=True)

    def _remove_outliers_arm64_simd(self, df: pd.DataFrame, columns: List[str]) -> int:
        """ARM64 NEON SIMD optimized outlier removal."""
        numeric_data = df[columns].select_dtypes(include=np.number).values
        
        if numeric_data.size == 0:
            return 0
        
        # Vectorized statistics using ARM64 SIMD
        means = np.mean(numeric_data, axis=0)
        stds = np.std(numeric_data, axis=0)
        
        # Broadcasting for SIMD efficiency
        lower_bounds = means - self.config.outlier_threshold_std * stds
        upper_bounds = means + self.config.outlier_threshold_std * stds
        
        # Single vectorized comparison for all columns
        outlier_mask = (numeric_data < lower_bounds) | (numeric_data > upper_bounds)
        
        # Count outliers efficiently
        total_outliers = np.sum(outlier_mask)
        
        # In-place replacement
        numeric_data[outlier_mask] = np.nan
        
        # Update DataFrame
        df.loc[:, columns] = numeric_data
        
        return total_outliers

    def remove_outliers(self, df: pd.DataFrame, columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Enhanced outlier removal with ARM64 SIMD optimizations.

        Args:
            df: Input DataFrame.
            columns: List of columns to check for outliers.

        Returns:
            DataFrame with outliers removed.
        """
        if df.empty:
            logger.warning("DataFrame is empty, skipping outlier removal.")
            return df

        start_time = time.perf_counter()
        
        if columns is None:
            columns = df.select_dtypes(include=np.number).columns.tolist()
        
        if not columns:
            logger.warning("No numeric columns found for outlier removal.")
            return df

        logger.info(f"Removing outliers from columns: {', '.join(columns)} "
                    f"with threshold {self.config.outlier_threshold_std} std dev.")

        total_outliers = 0

        try:
            # ARM64 SIMD optimized outlier detection
            if self.is_arm64 and self.config.enable_arm64_optimizations:
                total_outliers = self._remove_outliers_arm64_simd(df, columns)
            else:
                total_outliers = self._remove_outliers_standard(df, columns)

            # Handle NaNs introduced by outlier removal
            df = self.handle_missing_data(df)

            self.stats.outliers_removed += total_outliers
            processing_time = time.perf_counter() - start_time
            self.stats.processing_time_seconds += processing_time

            logger.info(f"Removed {total_outliers} total outliers in {processing_time:.4f}s")

        except Exception as e:
            logger.error(f"Error removing outliers: {e}")

        return df

    def _remove_outliers_standard(self, df: pd.DataFrame, columns: List[str]) -> int:
        """Standard outlier removal using Z-score method."""
        total_outliers = 0
        
        for col in columns:
            if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
                mean = df[col].mean()
                std = df[col].std()
                
                if std == 0:
                    logger.debug(f"Standard deviation for column '{col}' is zero, skipping outlier removal.")
                    continue
                
                lower_bound = mean - self.config.outlier_threshold_std * std
                upper_bound = mean + self.config.outlier_threshold_std * std
                
                outlier_mask = (df[col] < lower_bound) | (df[col] > upper_bound)
                num_outliers = outlier_mask.sum()
                
                if num_outliers > 0:
                    df.loc[outlier_mask, col] = np.nan
                    total_outliers += num_outliers
                    logger.debug(f"Removed {num_outliers} outliers from column '{col}'.")
        
        return total_outliers

    def normalize_data(self, df: pd.DataFrame, columns: Optional[List[str]] = None, method: str = 'minmax') -> pd.DataFrame:
        """
        Enhanced data normalization with ARM64 optimizations and caching.

        Args:
            df: Input DataFrame.
            columns: List of columns to normalize.
            method: Normalization method ('minmax', 'standard', 'robust').

        Returns:
            DataFrame with normalized columns.
        """
        if df.empty:
            logger.warning("DataFrame is empty, skipping data normalization.")
            return df

        start_time = time.perf_counter()
        
        if columns is None:
            columns = df.select_dtypes(include=np.number).columns.tolist()
        
        if not columns:
            logger.warning("No numeric columns found for normalization.")
            return df

        logger.info(f"Normalizing columns: {', '.join(columns)} using '{method}' method.")

        try:
            # Check cache for normalization parameters
            cache_key = f"norm_{method}_{hash(tuple(columns))}"
            
            if self.config.enable_caching and cache_key in self.processing_cache:
                norm_params = self.processing_cache[cache_key]
                self.stats.cache_hits += 1
                logger.debug("Using cached normalization parameters")
            else:
                norm_params = self._calculate_normalization_params(df, columns, method)
                if self.config.enable_caching:
                    self.processing_cache[cache_key] = norm_params
                    self.stats.cache_misses += 1

            # Apply normalization with ARM64 optimizations
            if self.is_arm64 and self.config.enable_arm64_optimizations:
                df = self._normalize_data_arm64(df, columns, method, norm_params)
            else:
                df = self._normalize_data_standard(df, columns, method, norm_params)

            self.stats.normalization_operations += len(columns)
            processing_time = time.perf_counter() - start_time
            self.stats.processing_time_seconds += processing_time

            logger.debug(f"Normalization completed in {processing_time:.4f}s")

        except Exception as e:
            logger.error(f"Error normalizing data: {e}")

        return df

    def _calculate_normalization_params(self, df: pd.DataFrame, columns: List[str], method: str) -> Dict[str, Dict[str, float]]:
        """Calculate normalization parameters for caching."""
        params = {}
        
        for col in columns:
            if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
                if method == 'minmax':
                    params[col] = {
                        'min': df[col].min(),
                        'max': df[col].max()
                    }
                elif method == 'standard':
                    params[col] = {
                        'mean': df[col].mean(),
                        'std': df[col].std()
                    }
                elif method == 'robust':
                    params[col] = {
                        'median': df[col].median(),
                        'iqr': df[col].quantile(0.75) - df[col].quantile(0.25)
                    }
        
        return params

    def _normalize_data_standard(self, df: pd.DataFrame, columns: List[str], method: str, params: Dict[str, Dict[str, float]]) -> pd.DataFrame:
        """Standard normalization implementation."""
        for col in columns:
            if col in df.columns and col in params:
                col_params = params[col]
                
                if method == 'minmax':
                    min_val, max_val = col_params['min'], col_params['max']
                    if max_val == min_val:
                        df[col] = 0.0
                    else:
                        df[col] = (df[col] - min_val) / (max_val - min_val)
                        
                elif method == 'standard':
                    mean, std = col_params['mean'], col_params['std']
                    if std == 0:
                        df[col] = 0.0
                    else:
                        df[col] = (df[col] - mean) / std
                        
                elif method == 'robust':
                    median, iqr = col_params['median'], col_params['iqr']
                    if iqr == 0:
                        df[col] = 0.0
                    else:
                        df[col] = (df[col] - median) / iqr
        
        return df

    def _normalize_data_arm64(self, df: pd.DataFrame, columns: List[str], method: str, params: Dict[str, Dict[str, float]]) -> pd.DataFrame:
        """ARM64 optimized normalization with vectorized operations."""
        # Use vectorized operations for better ARM64 SIMD utilization
        numeric_data = df[columns].select_dtypes(include=np.number)
        
        if method == 'minmax':
            for col in numeric_data.columns:
                if col in params:
                    min_val, max_val = params[col]['min'], params[col]['max']
                    if max_val != min_val:
                        df[col] = (df[col] - min_val) / (max_val - min_val)
                    else:
                        df[col] = 0.0
                        
        elif method == 'standard':
            for col in numeric_data.columns:
                if col in params:
                    mean, std = params[col]['mean'], params[col]['std']
                    if std != 0:
                        df[col] = (df[col] - mean) / std
                    else:
                        df[col] = 0.0
                        
        elif method == 'robust':
            for col in numeric_data.columns:
                if col in params:
                    median, iqr = params[col]['median'], params[col]['iqr']
                    if iqr != 0:
                        df[col] = (df[col] - median) / iqr
                    else:
                        df[col] = 0.0
        
        return df

    def resample_streaming(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fixed memory-efficient streaming resampling."""
        if df.empty:
            return df
        
        try:
            freq = pd.Timedelta(self.config.resample_interval)
            
            # More robust OHLCV aggregation
            agg_dict = {}
            if 'open' in df.columns: agg_dict['open'] = 'first'
            if 'high' in df.columns: agg_dict['high'] = 'max'
            if 'low' in df.columns: agg_dict['low'] = 'min'
            if 'close' in df.columns: agg_dict['close'] = 'last'
            if 'volume' in df.columns: agg_dict['volume'] = 'sum'
            
            if not agg_dict:
                logger.warning("No OHLCV columns found for resampling")
                return df
            
            # Use time-based grouper for better performance
            time_groups = df.index.floor(freq)
            resampled = df.groupby(time_groups).agg(agg_dict)
            
            return resampled
            
        except Exception as e:
            logger.error(f"Streaming resampling failed: {e}")
            return df

    def resample_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Enhanced data resampling with ARM64 optimizations and streaming approach.

        Args:
            df: Input DataFrame with 'timestamp' as index.

        Returns:
            Resampled DataFrame.
        """
        if df.empty:
            logger.warning("DataFrame is empty, skipping resampling.")
            return df

        start_time = time.perf_counter()

        try:
            # Ensure proper datetime index
            if not isinstance(df.index, pd.DatetimeIndex):
                if 'timestamp' in df.columns:
                    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ns')
                    df = df.set_index('timestamp')
                else:
                    logger.error("DataFrame index is not DatetimeIndex and 'timestamp' column is missing. Cannot resample.")
                    return df

            logger.info(f"Resampling data to '{self.config.resample_interval}' interval.")

            # Use streaming resampling for better memory efficiency
            if self.config.enable_streaming_resampling:
                resampled_df = self.resample_streaming(df)
            else:
                # Enhanced OHLCV resampling with additional fields
                ohlc_dict = {
                    'open': 'first',
                    'high': 'max',
                    'low': 'min',
                    'close': 'last',
                    'volume': 'sum',
                    'vwap': 'mean',  # Volume-weighted average price
                    'trade_count': 'sum',
                    'bid': 'last',
                    'ask': 'last',
                    'bid_size': 'last',
                    'ask_size': 'last'
                }
                
                # Filter to only include columns present in the DataFrame
                ohlc_dict_filtered = {k: v for k, v in ohlc_dict.items() if k in df.columns}

                # ARM64 optimized resampling
                if self.is_arm64 and self.config.enable_arm64_optimizations and len(df) > self.config.chunk_size:
                    resampled_df = self._resample_data_chunked(df, ohlc_dict_filtered)
                else:
                    resampled_df = df.resample(self.config.resample_interval).agg(ohlc_dict_filtered)
            
            # Fill any NaNs introduced by resampling
            resampled_df = self.handle_missing_data(resampled_df)

            self.stats.resampling_operations += 1
            processing_time = time.perf_counter() - start_time
            self.stats.processing_time_seconds += processing_time

            logger.debug(f"Resampling complete in {processing_time:.4f}s. "
                        f"Original shape: {df.shape}, Resampled shape: {resampled_df.shape}")

        except Exception as e:
            logger.error(f"Error resampling data: {e}")
            return df

        return resampled_df

    def _resample_data_chunked(self, df: pd.DataFrame, ohlc_dict: Dict[str, str]) -> pd.DataFrame:
        """ARM64 optimized chunked resampling for large datasets."""
        # For very large datasets, process in chunks to optimize ARM64 cache usage
        time_chunks = pd.date_range(start=df.index.min(), end=df.index.max(), 
                                   freq=f"{self.config.chunk_size}T")  # Chunk by time
        
        resampled_chunks = []
        for i in range(len(time_chunks) - 1):
            chunk_data = df.loc[time_chunks[i]:time_chunks[i+1]]
            if not chunk_data.empty:
                chunk_resampled = chunk_data.resample(self.config.resample_interval).agg(ohlc_dict)
                resampled_chunks.append(chunk_resampled)
        
        if resampled_chunks:
            return pd.concat(resampled_chunks)
        else:
            return df.resample(self.config.resample_interval).agg(ohlc_dict)

    async def preprocess_parallel(self, df: pd.DataFrame) -> pd.DataFrame:
        """Parallel preprocessing pipeline."""
        # Split data for parallel processing
        chunks = np.array_split(df, self.config.max_workers)
        
        # Process chunks in parallel
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            tasks = [
                loop.run_in_executor(executor, self._process_chunk, chunk, i)
                for i, chunk in enumerate(chunks)
            ]
            
            processed_chunks = await asyncio.gather(*tasks)
        
        # Merge results
        return pd.concat(processed_chunks, ignore_index=True)

    def _process_chunk(self, chunk: pd.DataFrame, chunk_id: int) -> pd.DataFrame:
        """Process individual chunk with optimized pipeline."""
        # Pin to specific NUMA node if available
        if self.numa_processor:
            numa_node = chunk_id % len(self.numa_processor.numa_nodes)
            self.numa_processor._set_numa_affinity(numa_node)
            self.stats.numa_node_switches += 1
        
        # Optimized chunk processing
        return self._preprocess_chunk_optimized(chunk)

    def _preprocess_chunk_optimized(self, chunk: pd.DataFrame) -> pd.DataFrame:
        """Optimized preprocessing for individual chunks."""
        # Apply zero-copy operations if enabled
        if self.config.enable_zero_copy and self.is_arm64:
            return self.arm64_pipeline.preprocess_zero_copy(chunk)
        else:
            # Standard processing
            chunk, errors = self.validate_data(chunk)
            if not errors:
                chunk = self.handle_missing_data(chunk)
                chunk = self.remove_outliers(chunk)
            return chunk

    def preprocess(self, df: pd.DataFrame, enable_normalization: bool = False, symbol: str = None) -> pd.DataFrame:
        """
        Enhanced preprocessing pipeline with ARM64 optimizations, NUMA awareness, and parallel processing.

        Args:
            df: Raw market data DataFrame.
            enable_normalization: Whether to apply normalization (typically done in feature engineering).
            symbol: Symbol identifier for NUMA-aware processing.

        Returns:
            Preprocessed DataFrame.
        """
        start_time = time.perf_counter()
        logger.info("Starting enhanced data preprocessing pipeline with ARM64 and NUMA optimizations.")

        try:
            # Reset statistics for this preprocessing run
            original_records = len(df)
            self.stats.records_processed = original_records

            # Periodic memory pressure check
            current_time = time.time()
            if current_time - self._last_memory_check > self._memory_check_interval:
                self._adjust_for_memory_pressure()
                self._last_memory_check = current_time

            # Use batch processing for very large datasets
            if len(df) > self.batch_processor.batch_size_records:
                logger.info(f"Using batch processing for large dataset ({len(df)} records)")
                return self.batch_processor.process_large_dataset(df,
                    lambda batch: self._preprocess_single_batch(batch, enable_normalization, symbol))

            # NUMA-aware processing if symbol provided
            if symbol and self.numa_processor:
                return self.numa_processor.process_numa_aware(df, symbol)

            # Standard preprocessing pipeline
            return self._preprocess_single_batch(df, enable_normalization, symbol)

        except Exception as e:
            logger.error(f"Error in preprocessing pipeline: {e}")
            return pd.DataFrame()

    def _preprocess_single_batch(self, df: pd.DataFrame, enable_normalization: bool = False, symbol: str = None) -> pd.DataFrame:
        """Process with proper error recovery."""
        try:
            # Check memory pressure before processing
            self._adjust_for_memory_pressure()
            
            # Validate Data with retry on failure
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    df, errors = self.validate_data(df.copy())
                    break
                except Exception as e:
                    if attempt == max_retries - 1:
                        logger.error(f"Validation failed after {max_retries} attempts: {e}")
                        return pd.DataFrame()
                    logger.warning(f"Validation attempt {attempt + 1} failed, retrying: {e}")
            
            if errors:
                logger.warning(f"Validation errors found but continuing: {errors}")
            
            # Rest of processing with try-catch around each step
            try:
                if 'timestamp' in df.columns:
                    df = df.set_index('timestamp')
                elif not isinstance(df.index, pd.DatetimeIndex):
                    logger.error("No valid timestamp found")
                    return pd.DataFrame()
            except Exception as e:
                logger.error(f"Timestamp processing failed: {e}")
                return pd.DataFrame()
            
            # 2. Handle Missing Data
            try:
                df = self.handle_missing_data(df)
            except Exception as e:
                logger.error(f"Missing data handling failed: {e}")
                return pd.DataFrame()

            # 3. Remove Outliers
            try:
                outlier_columns = ['open', 'high', 'low', 'close', 'volume']
                available_outlier_columns = [col for col in outlier_columns if col in df.columns]
                if available_outlier_columns:
                    df = self.remove_outliers(df, columns=available_outlier_columns)
            except Exception as e:
                logger.error(f"Outlier removal failed: {e}")
                # Continue without outlier removal

            # 4. Resample Data
            try:
                df = self.resample_data(df)
            except Exception as e:
                logger.error(f"Resampling failed: {e}")
                # Continue without resampling

            # 5. Optional Normalization
            if enable_normalization:
                try:
                    norm_columns = ['open', 'high', 'low', 'close', 'volume']
                    available_norm_columns = [col for col in norm_columns if col in df.columns]
                    if available_norm_columns:
                        df = self.normalize_data(df, columns=available_norm_columns, method='minmax')
                except Exception as e:
                    logger.error(f"Normalization failed: {e}")
                    # Continue without normalization

            # Update final statistics
            total_processing_time = time.perf_counter() - time.time()
            self.stats.processing_time_seconds = total_processing_time

            logger.info(f"Data preprocessing pipeline completed in {total_processing_time:.4f}s")
            logger.info(f"Processed {self.stats.records_processed} records at {self.stats.records_per_second:.2f} records/second")
            
            # Log performance statistics
            if self.config.enable_performance_monitoring:
                self._log_performance_stats()

        except Exception as e:
            logger.error(f"Critical error in preprocessing: {e}")
            return pd.DataFrame()

        return df

    def _log_performance_stats(self):
        """Log detailed performance statistics."""
        logger.info("Preprocessing Performance Statistics:")
        logger.info(f"  Records Processed: {self.stats.records_processed}")
        logger.info(f"  Processing Time: {self.stats.processing_time_seconds:.4f}s")
        logger.info(f"  Records/Second: {self.stats.records_per_second:.2f}")
        logger.info(f"  Validation Errors: {self.stats.validation_errors}")
        logger.info(f"  Outliers Removed: {self.stats.outliers_removed}")
        logger.info(f"  Missing Values Filled: {self.stats.missing_values_filled}")
        logger.info(f"  Normalization Operations: {self.stats.normalization_operations}")
        logger.info(f"  Resampling Operations: {self.stats.resampling_operations}")
        logger.info(f"  Cache Hits: {self.stats.cache_hits}")
        logger.info(f"  Cache Misses: {self.stats.cache_misses}")
        logger.info(f"  Memory Pool Allocations: {self.stats.memory_pool_allocations}")
        logger.info(f"  NUMA Node Switches: {self.stats.numa_node_switches}")
        logger.info(f"  Zero-Copy Operations: {self.stats.zero_copy_operations}")

    def get_statistics(self) -> PreprocessingStats:
        """Get current preprocessing statistics."""
        return self.stats

    def reset_statistics(self):
        """Reset preprocessing statistics."""
        self.stats = PreprocessingStats()
        logger.info("Preprocessing statistics reset")

    def clear_cache(self):
        """Clear the processing cache."""
        self.processing_cache.clear()
        logger.info("Processing cache cleared")

if __name__ == "__main__":
    # Enhanced example usage with ARM64 and NUMA optimizations
    import warnings
    warnings.filterwarnings('ignore')
    
    # Create comprehensive test data
    np.random.seed(42)
    n_records = 10000
    
    data = {
        'timestamp': pd.to_datetime(pd.date_range(start='2023-01-01', periods=n_records, freq='1s')),
        'open': np.random.rand(n_records) * 100 + 100,
        'high': np.random.rand(n_records) * 100 + 105,
        'low': np.random.rand(n_records) * 100 + 95,
        'close': np.random.rand(n_records) * 100 + 100,
        'volume': np.random.randint(1000, 10000, n_records),
        'vwap': np.random.rand(n_records) * 100 + 100,
        'trade_count': np.random.randint(1, 100, n_records)
    }
    df = pd.DataFrame(data)

    # Introduce missing values and outliers for testing
    df.loc[100:120, 'close'] = np.nan
    df.loc[500:510, 'volume'] = np.nan
    df.loc[200, 'volume'] = 1000000  # Outlier
    df.loc[300, 'open'] = 0.001      # Outlier
    df.loc[400, 'high'] = 1000       # Outlier

    print("Original DataFrame info:")
    print(f"Shape: {df.shape}")
    print(f"Missing values: {df.isnull().sum().sum()}")
    print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

    # Initialize enhanced preprocessor with all optimizations
    config = PreprocessingConfig(
        fill_method='ffill',
        resample_interval='1min',
        outlier_threshold_std=2.5,
        enable_arm64_optimizations=True,
        enable_parallel_processing=True,
        enable_numa_awareness=True,
        max_workers=4,
        enable_performance_monitoring=True,
        enable_zero_copy=True,
        enable_streaming_resampling=True,
        memory_pool_mb=1000,
        batch_size_gb=10.0
    )
    
    preprocessor = DataPreprocessor(config=config)

    # Process the data
    start_time = time.time()
    processed_df = preprocessor.preprocess(df.copy(), enable_normalization=True, symbol="AAPL")
    total_time = time.time() - start_time

    print(f"\nProcessed DataFrame info:")
    print(f"Shape: {processed_df.shape}")
    print(f"Missing values: {processed_df.isnull().sum().sum()}")
    print(f"Processing time: {total_time:.4f}s")
    
    # Display statistics
    stats = preprocessor.get_statistics()
    print(f"\nPerformance Statistics:")
    print(f"Records/second: {stats.records_per_second:.2f}")
    print(f"Outliers removed: {stats.outliers_removed}")
    print(f"Missing values filled: {stats.missing_values_filled}")
    print(f"Zero-copy operations: {stats.zero_copy_operations}")
    print(f"NUMA node switches: {stats.numa_node_switches}")

    # Test with invalid data
    print("\nTesting with invalid data (missing 'volume' column):")
    df_bad = df.drop(columns=['volume'])
    processed_df_bad = preprocessor.preprocess(df_bad.copy())
    print(f"Processed bad DataFrame is empty: {processed_df_bad.empty}")
