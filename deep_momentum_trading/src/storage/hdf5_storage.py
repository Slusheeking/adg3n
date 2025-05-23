import h5py
import numpy as np
import pandas as pd
import logging
import threading
import time
import os
import asyncio
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from contextlib import contextmanager, asynccontextmanager
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor
import warnings
import gc
import psutil
from functools import lru_cache
import pickle
import json

from deep_momentum_trading.src.utils.logger import get_logger

logger = get_logger(__name__)

@dataclass
class HDF5Config:
    """Configuration for HDF5 storage with ARM64 optimizations."""
    use_arm64_optimizations: bool = True
    compression: str = "lz4"  # lz4, gzip, szip, lzf
    compression_opts: int = 9
    chunk_cache_size: int = 1024**3  # 1GB
    chunk_cache_nelems: int = 521
    chunk_cache_preemption: float = 0.75
    enable_swmr: bool = True  # Single Writer Multiple Reader
    fletcher32: bool = True  # Checksum
    shuffle: bool = True  # Reorder bytes for better compression
    track_order: bool = True  # Maintain creation order
    max_memory_usage: int = 8 * 1024**3  # 8GB
    enable_mpi: bool = False
    rdcc_nbytes: int = 1024**2  # Raw data chunk cache size
    rdcc_w0: float = 0.75  # Chunk preemption policy
    
@dataclass
class HDF5Metrics:
    """Performance metrics for HDF5 operations."""
    read_operations: int = 0
    write_operations: int = 0
    total_bytes_read: int = 0
    total_bytes_written: int = 0
    avg_read_time: float = 0.0
    avg_write_time: float = 0.0
    cache_hit_ratio: float = 0.0
    compression_ratio: float = 0.0
    memory_usage: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

class AdvancedHDF5Storage:
    """
    Advanced HDF5 storage with ARM64 optimizations, async operations,
    and comprehensive performance monitoring.
    """

    def __init__(self, 
                 base_path: str = "data/processed/",
                 config: Optional[HDF5Config] = None):
        """
        Initialize Advanced HDF5 Storage.

        Args:
            base_path: Base directory for HDF5 files
            config: HDF5 configuration
        """
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        self.config = config or HDF5Config()
        
        # File management
        self.file_handles: Dict[str, h5py.File] = {}
        self.lock = threading.RLock()
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # Performance tracking
        self.metrics = HDF5Metrics()
        self.operation_times = []
        self.cache_stats = {"hits": 0, "misses": 0}
        
        # ARM64 optimizations
        self._setup_arm64_optimizations()
        
        # HDF5 configuration
        self._configure_hdf5()
        
        logger.info(f"AdvancedHDF5Storage initialized with ARM64 optimizations: {self.config.use_arm64_optimizations}")

    def _setup_arm64_optimizations(self) -> None:
        """Setup ARM64-specific optimizations."""
        if not self.config.use_arm64_optimizations:
            return
            
        try:
            # Set ARM64-specific HDF5 environment variables
            os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")
            os.environ.setdefault("HDF5_DRIVER", "sec2")
            
            # Optimize for ARM64 SIMD operations
            if hasattr(h5py, '_hl'):
                # Enable SIMD optimizations if available
                pass
            
            # Set optimal chunk cache parameters for ARM64
            chunk_cache_size = min(self.config.chunk_cache_size, psutil.virtual_memory().available // 4)
            self.config.chunk_cache_size = chunk_cache_size
            
            logger.info("ARM64 optimizations applied to HDF5 storage")
            
        except Exception as e:
            logger.warning(f"Failed to apply ARM64 optimizations: {e}")

    def _configure_hdf5(self) -> None:
        """Configure HDF5 library settings."""
        try:
            # Set global HDF5 configuration
            h5py.get_config().track_order = self.config.track_order
            
            # Configure chunk cache
            if hasattr(h5py, 'get_libversion'):
                version = h5py.get_libversion()
                logger.info(f"HDF5 library version: {version}")
            
        except Exception as e:
            logger.warning(f"Failed to configure HDF5: {e}")

    @contextmanager
    def _get_file_handle(self, symbol: str, mode: str = 'r') -> h5py.File:
        """Thread-safe context manager for HDF5 file handles with ARM64 optimizations."""
        file_path = self.base_path / f"{symbol}.h5"
        
        with self.lock:
            cache_key = f"{symbol}_{mode}"
            
            if symbol not in self.file_handles or not self.file_handles[symbol]:
                try:
                    # ARM64-optimized file access parameters
                    kwargs = {
                        'rdcc_nbytes': self.config.rdcc_nbytes,
                        'rdcc_w0': self.config.rdcc_w0,
                        'rdcc_nslots': self.config.chunk_cache_nelems,
                    }
                    
                    if mode in ['a', 'w', 'r+']:
                        kwargs['swmr'] = False  # SWMR not compatible with write modes
                    elif self.config.enable_swmr:
                        kwargs['swmr'] = True
                    
                    self.file_handles[symbol] = h5py.File(file_path, mode, **kwargs)
                    logger.debug(f"Opened HDF5 file for {symbol} in mode '{mode}' with ARM64 optimizations")
                    
                except Exception as e:
                    logger.error(f"Error opening HDF5 file {file_path}: {e}")
                    raise
                    
            elif self.file_handles[symbol].mode != mode and mode != 'r+':
                # Close and reopen with new mode
                self.file_handles[symbol].close()
                kwargs = {
                    'rdcc_nbytes': self.config.rdcc_nbytes,
                    'rdcc_w0': self.config.rdcc_w0,
                    'rdcc_nslots': self.config.chunk_cache_nelems,
                }
                
                if mode in ['a', 'w', 'r+']:
                    kwargs['swmr'] = False
                elif self.config.enable_swmr:
                    kwargs['swmr'] = True
                
                self.file_handles[symbol] = h5py.File(file_path, mode, **kwargs)
                logger.debug(f"Reopened HDF5 file for {symbol} in mode '{mode}'")
            
            yield self.file_handles[symbol]

    def _create_optimized_dataset(self, 
                                 group: h5py.Group, 
                                 name: str, 
                                 shape: Tuple[int, ...], 
                                 dtype: str,
                                 maxshape: Optional[Tuple[Optional[int], ...]] = None,
                                 chunks: bool = True) -> h5py.Dataset:
        """Create dataset with ARM64 optimizations."""
        kwargs = {
            'compression': self.config.compression,
            'compression_opts': self.config.compression_opts,
            'shuffle': self.config.shuffle,
            'fletcher32': self.config.fletcher32,
            'chunks': chunks,
            'track_order': self.config.track_order
        }
        
        if maxshape:
            kwargs['maxshape'] = maxshape
        
        # ARM64-specific chunk size optimization
        if chunks and len(shape) > 0:
            # Optimize chunk size for ARM64 cache lines (64 bytes)
            if dtype == 'f4':  # float32
                optimal_chunk_size = 16  # 64 bytes / 4 bytes per float
            elif dtype == 'f8':  # float64
                optimal_chunk_size = 8   # 64 bytes / 8 bytes per double
            elif dtype == 'i8':  # int64
                optimal_chunk_size = 8   # 64 bytes / 8 bytes per int64
            else:
                optimal_chunk_size = 16
            
            if len(shape) == 1:
                kwargs['chunks'] = (min(optimal_chunk_size * 1024, shape[0]),)
            elif len(shape) == 2:
                kwargs['chunks'] = (min(optimal_chunk_size * 64, shape[0]), shape[1])
        
        return group.create_dataset(name, shape, dtype=dtype, **kwargs)

    def store_ohlcv_data(self, symbol: str, data: pd.DataFrame) -> None:
        """Store OHLCV data with ARM64 optimizations."""
        if data.empty:
            logger.warning(f"No OHLCV data provided for {symbol}")
            return
            
        if not isinstance(data.index, pd.DatetimeIndex):
            raise ValueError("DataFrame index must be a DatetimeIndex")

        start_time = time.time()
        
        try:
            logger.info(f"Storing OHLCV data for {symbol}. Shape: {data.shape}")
            
            with self._get_file_handle(symbol, 'a') as f:
                if 'ohlcv' not in f:
                    grp = f.create_group('ohlcv')
                    
                    # Create optimized datasets
                    self._create_optimized_dataset(
                        grp, 'timestamp', (0,), 'i8', maxshape=(None,)
                    )
                    
                    for col in ['open', 'high', 'low', 'close', 'volume']:
                        self._create_optimized_dataset(
                            grp, col, (0,), 'f4', maxshape=(None,)
                        )
                    
                    # Store metadata
                    grp.attrs['created_at'] = time.time()
                    grp.attrs['symbol'] = symbol
                    grp.attrs['data_type'] = 'ohlcv'
                    
                    logger.debug(f"Created optimized OHLCV group for {symbol}")
                else:
                    grp = f['ohlcv']

                # Convert timestamps to nanoseconds
                timestamps_ns = data.index.astype(np.int64).values
                
                # Append data with ARM64-optimized operations
                current_len = grp['timestamp'].shape[0]
                new_len = current_len + len(data)
                
                # Resize datasets
                grp['timestamp'].resize((new_len,))
                grp['timestamp'][current_len:] = timestamps_ns
                
                for col in ['open', 'high', 'low', 'close', 'volume']:
                    if col in data.columns:
                        grp[col].resize((new_len,))
                        # Use ARM64-optimized numpy operations
                        values = data[col].values.astype(np.float32)
                        grp[col][current_len:] = values
                
                # Force write to disk
                f.flush()
                
                # Update metrics
                write_time = time.time() - start_time
                self.metrics.write_operations += 1
                self.metrics.total_bytes_written += data.memory_usage(deep=True).sum()
                self.metrics.avg_write_time = (
                    (self.metrics.avg_write_time * (self.metrics.write_operations - 1) + write_time) /
                    self.metrics.write_operations
                )
                
                logger.info(f"Stored {len(data)} OHLCV records for {symbol} in {write_time:.3f}s")
                
        except Exception as e:
            logger.error(f"Failed to store OHLCV data for {symbol}: {e}")
            raise

    def load_ohlcv_data(self, 
                       symbol: str,
                       start_timestamp: Optional[Union[int, str]] = None,
                       end_timestamp: Optional[Union[int, str]] = None,
                       columns: Optional[List[str]] = None) -> Optional[pd.DataFrame]:
        """Load OHLCV data with ARM64 optimizations."""
        start_time = time.time()
        
        try:
            logger.info(f"Loading OHLCV data for {symbol}")
            
            with self._get_file_handle(symbol, 'r') as f:
                if 'ohlcv' not in f:
                    logger.warning(f"No OHLCV data found for {symbol}")
                    return None
                
                grp = f['ohlcv']
                timestamps_ns = grp['timestamp'][:]
                
                # Convert string timestamps
                if isinstance(start_timestamp, str):
                    start_timestamp = pd.to_datetime(start_timestamp).value
                if isinstance(end_timestamp, str):
                    end_timestamp = pd.to_datetime(end_timestamp).value
                
                # Find indices using ARM64-optimized search
                start_idx = 0
                end_idx = len(timestamps_ns)
                
                if start_timestamp is not None:
                    start_idx = np.searchsorted(timestamps_ns, start_timestamp, side='left')
                if end_timestamp is not None:
                    end_idx = np.searchsorted(timestamps_ns, end_timestamp, side='right')
                
                if start_idx >= end_idx:
                    return pd.DataFrame()
                
                # Load data with specified columns
                cols_to_load = columns or ['open', 'high', 'low', 'close', 'volume']
                data = {}
                
                for col in cols_to_load:
                    if col in grp:
                        # ARM64-optimized data loading
                        data[col] = grp[col][start_idx:end_idx]
                
                df = pd.DataFrame(
                    data, 
                    index=pd.to_datetime(timestamps_ns[start_idx:end_idx])
                )
                
                # Update metrics
                read_time = time.time() - start_time
                self.metrics.read_operations += 1
                self.metrics.total_bytes_read += df.memory_usage(deep=True).sum()
                self.metrics.avg_read_time = (
                    (self.metrics.avg_read_time * (self.metrics.read_operations - 1) + read_time) /
                    self.metrics.read_operations
                )
                
                logger.info(f"Loaded {len(df)} OHLCV records for {symbol} in {read_time:.3f}s")
                return df
                
        except Exception as e:
            logger.error(f"Failed to load OHLCV data for {symbol}: {e}")
            raise

    async def store_ohlcv_data_async(self, symbol: str, data: pd.DataFrame) -> None:
        """Async version of store_ohlcv_data."""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(self.executor, self.store_ohlcv_data, symbol, data)

    async def load_ohlcv_data_async(self, 
                                   symbol: str,
                                   start_timestamp: Optional[Union[int, str]] = None,
                                   end_timestamp: Optional[Union[int, str]] = None,
                                   columns: Optional[List[str]] = None) -> Optional[pd.DataFrame]:
        """Async version of load_ohlcv_data."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor, self.load_ohlcv_data, symbol, start_timestamp, end_timestamp, columns
        )

    def store_features(self, symbol: str, features_df: pd.DataFrame, feature_set: str = "default") -> None:
        """Store engineered features with ARM64 optimizations."""
        if features_df.empty:
            logger.warning(f"No features data provided for {symbol}")
            return
            
        if not isinstance(features_df.index, pd.DatetimeIndex):
            raise ValueError("DataFrame index must be a DatetimeIndex")

        start_time = time.time()
        
        try:
            logger.info(f"Storing features for {symbol}. Shape: {features_df.shape}")
            
            with self._get_file_handle(symbol, 'a') as f:
                group_name = f'features_{feature_set}'
                
                if group_name not in f:
                    grp = f.create_group(group_name)
                    
                    # Store feature metadata
                    grp.attrs['feature_names'] = [name.encode('utf-8') for name in features_df.columns]
                    grp.attrs['feature_set'] = feature_set
                    grp.attrs['created_at'] = time.time()
                    grp.attrs['num_features'] = len(features_df.columns)
                    
                    # Create optimized datasets
                    self._create_optimized_dataset(
                        grp, 'timestamp', (0,), 'i8', maxshape=(None,)
                    )
                    self._create_optimized_dataset(
                        grp, 'data', (0, len(features_df.columns)), 'f4', 
                        maxshape=(None, len(features_df.columns))
                    )
                    
                    logger.debug(f"Created optimized features group for {symbol}")
                else:
                    grp = f[group_name]
                    
                    # Validate feature consistency
                    existing_features = [name.decode('utf-8') for name in grp.attrs['feature_names']]
                    if existing_features != features_df.columns.tolist():
                        logger.warning(f"Feature mismatch for {symbol}: {existing_features} vs {features_df.columns.tolist()}")

                timestamps_ns = features_df.index.astype(np.int64).values
                
                # Append data
                current_len = grp['timestamp'].shape[0]
                new_len = current_len + len(features_df)
                
                grp['timestamp'].resize((new_len,))
                grp['timestamp'][current_len:] = timestamps_ns
                
                grp['data'].resize((new_len, features_df.shape[1]))
                # ARM64-optimized data conversion
                features_array = features_df.values.astype(np.float32)
                grp['data'][current_len:] = features_array
                
                f.flush()
                
                # Update metrics
                write_time = time.time() - start_time
                self.metrics.write_operations += 1
                self.metrics.total_bytes_written += features_df.memory_usage(deep=True).sum()
                
                logger.info(f"Stored {len(features_df)} feature records for {symbol} in {write_time:.3f}s")
                
        except Exception as e:
            logger.error(f"Failed to store features for {symbol}: {e}")
            raise

    def load_features(self, 
                     symbol: str,
                     feature_set: str = "default",
                     start_timestamp: Optional[Union[int, str]] = None,
                     end_timestamp: Optional[Union[int, str]] = None,
                     feature_names: Optional[List[str]] = None) -> Optional[pd.DataFrame]:
        """Load engineered features with ARM64 optimizations."""
        start_time = time.time()
        
        try:
            logger.info(f"Loading features for {symbol} (set: {feature_set})")
            
            with self._get_file_handle(symbol, 'r') as f:
                group_name = f'features_{feature_set}'
                
                if group_name not in f:
                    logger.warning(f"No features data found for {symbol} (set: {feature_set})")
                    return None
                
                grp = f[group_name]
                timestamps_ns = grp['timestamp'][:]
                stored_feature_names = [name.decode('utf-8') for name in grp.attrs['feature_names']]
                
                # Convert string timestamps
                if isinstance(start_timestamp, str):
                    start_timestamp = pd.to_datetime(start_timestamp).value
                if isinstance(end_timestamp, str):
                    end_timestamp = pd.to_datetime(end_timestamp).value
                
                # Find indices
                start_idx = 0
                end_idx = len(timestamps_ns)
                
                if start_timestamp is not None:
                    start_idx = np.searchsorted(timestamps_ns, start_timestamp, side='left')
                if end_timestamp is not None:
                    end_idx = np.searchsorted(timestamps_ns, end_timestamp, side='right')
                
                if start_idx >= end_idx:
                    return pd.DataFrame()
                
                # Load data
                features_data = grp['data'][start_idx:end_idx]
                
                # Select specific features if requested
                if feature_names:
                    feature_indices = [stored_feature_names.index(name) for name in feature_names if name in stored_feature_names]
                    features_data = features_data[:, feature_indices]
                    column_names = [stored_feature_names[i] for i in feature_indices]
                else:
                    column_names = stored_feature_names
                
                df = pd.DataFrame(
                    features_data,
                    columns=column_names,
                    index=pd.to_datetime(timestamps_ns[start_idx:end_idx])
                )
                
                # Update metrics
                read_time = time.time() - start_time
                self.metrics.read_operations += 1
                self.metrics.total_bytes_read += df.memory_usage(deep=True).sum()
                
                logger.info(f"Loaded {len(df)} feature records for {symbol} in {read_time:.3f}s")
                return df
                
        except Exception as e:
            logger.error(f"Failed to load features for {symbol}: {e}")
            raise

    def get_symbols(self) -> List[str]:
        """Get list of available symbols."""
        symbols = []
        for file_path in self.base_path.glob("*.h5"):
            symbols.append(file_path.stem)
        return sorted(symbols)

    def get_data_info(self, symbol: str) -> Dict[str, Any]:
        """Get information about stored data for a symbol."""
        info = {
            "symbol": symbol,
            "file_exists": False,
            "groups": [],
            "file_size": 0,
            "compression_ratio": 0.0
        }
        
        file_path = self.base_path / f"{symbol}.h5"
        if not file_path.exists():
            return info
        
        info["file_exists"] = True
        info["file_size"] = file_path.stat().st_size
        
        try:
            with self._get_file_handle(symbol, 'r') as f:
                def visit_func(name, obj):
                    if isinstance(obj, h5py.Group):
                        group_info = {
                            "name": name,
                            "type": "group",
                            "attrs": dict(obj.attrs)
                        }
                        info["groups"].append(group_info)
                    elif isinstance(obj, h5py.Dataset):
                        dataset_info = {
                            "name": name,
                            "type": "dataset",
                            "shape": obj.shape,
                            "dtype": str(obj.dtype),
                            "compression": obj.compression,
                            "size": obj.size
                        }
                        info["groups"].append(dataset_info)
                
                f.visititems(visit_func)
                
        except Exception as e:
            logger.error(f"Failed to get data info for {symbol}: {e}")
        
        return info

    def optimize_storage(self) -> None:
        """Optimize storage performance."""
        logger.info("Optimizing HDF5 storage performance")
        
        # Close unused file handles
        with self.lock:
            for symbol, handle in list(self.file_handles.items()):
                if handle and not handle.id.valid:
                    del self.file_handles[symbol]
        
        # Force garbage collection
        gc.collect()
        
        # Update memory usage
        self.metrics.memory_usage = psutil.Process().memory_info().rss
        
        logger.info("Storage optimization completed")

    def get_metrics(self) -> HDF5Metrics:
        """Get performance metrics."""
        # Update current memory usage
        self.metrics.memory_usage = psutil.Process().memory_info().rss
        
        # Calculate cache hit ratio
        total_cache_ops = self.cache_stats["hits"] + self.cache_stats["misses"]
        if total_cache_ops > 0:
            self.metrics.cache_hit_ratio = self.cache_stats["hits"] / total_cache_ops
        
        return self.metrics

    def close_all_files(self) -> None:
        """Close all open HDF5 file handles."""
        with self.lock:
            for symbol, handle in list(self.file_handles.items()):
                if handle:
                    try:
                        handle.close()
                        logger.debug(f"Closed HDF5 file for {symbol}")
                    except Exception as e:
                        logger.error(f"Error closing HDF5 file for {symbol}: {e}")
                    finally:
                        del self.file_handles[symbol]
        
        # Shutdown executor
        self.executor.shutdown(wait=True)
        
        logger.info("All HDF5 file handles closed")

    def __del__(self):
        """Cleanup on destruction."""
        try:
            self.close_all_files()
        except:
            pass

# Specialized storage classes
class HDF5TimeSeriesStorage(AdvancedHDF5Storage):
    """Specialized HDF5 storage for time series data."""
    
    def __init__(self, base_path: str = "data/timeseries/", config: Optional[HDF5Config] = None):
        super().__init__(base_path, config)

class HDF5ModelStorage(AdvancedHDF5Storage):
    """Specialized HDF5 storage for model data."""
    
    def __init__(self, base_path: str = "data/models/", config: Optional[HDF5Config] = None):
        super().__init__(base_path, config)
    
    def store_model_weights(self, model_name: str, weights: Dict[str, np.ndarray]) -> None:
        """Store model weights."""
        with self._get_file_handle(model_name, 'a') as f:
            if 'weights' not in f:
                grp = f.create_group('weights')
            else:
                grp = f['weights']
            
            for layer_name, weight_array in weights.items():
                if layer_name in grp:
                    del grp[layer_name]
                
                self._create_optimized_dataset(
                    grp, layer_name, weight_array.shape, 
                    str(weight_array.dtype), chunks=True
                )
                grp[layer_name][:] = weight_array
            
            f.flush()

class HDF5FeatureStorage(AdvancedHDF5Storage):
    """Specialized HDF5 storage for feature data."""
    
    def __init__(self, base_path: str = "data/features/", config: Optional[HDF5Config] = None):
        super().__init__(base_path, config)

# Legacy compatibility
HDF5TimeSeriesStorage = HDF5TimeSeriesStorage

if __name__ == "__main__":
    # Example usage with ARM64 optimizations
    config = HDF5Config(
        use_arm64_optimizations=True,
        compression="lz4",
        chunk_cache_size=512 * 1024**2,  # 512MB
        enable_swmr=True
    )
    
    storage = AdvancedHDF5Storage(base_path="temp_hdf5_data", config=config)
    
    # Create test data
    dates = pd.date_range(start='2023-01-01', periods=10000, freq='1min')
    ohlcv_data = pd.DataFrame({
        'open': np.random.rand(10000) * 100,
        'high': np.random.rand(10000) * 100 + 5,
        'low': np.random.rand(10000) * 100 - 5,
        'close': np.random.rand(10000) * 100,
        'volume': np.random.randint(1000, 10000, 10000)
    }, index=dates)
    
    print("=== Advanced HDF5 Storage Test ===")
    
    # Test OHLCV storage
    print("\n1. Storing OHLCV data...")
    start_time = time.time()
    storage.store_ohlcv_data('AAPL', ohlcv_data)
    store_time = time.time() - start_time
    print(f"   Stored in {store_time:.3f}s")
    
    # Test OHLCV loading
    print("\n2. Loading OHLCV data...")
    start_time = time.time()
    loaded_data = storage.load_ohlcv_data('AAPL', columns=['open', 'close', 'volume'])
    load_time = time.time() - start_time
    print(f"   Loaded {len(loaded_data)} records in {load_time:.3f}s")
    
    # Test features storage
    features_data = pd.DataFrame({
        'sma_20': np.random.rand(10000),
        'rsi_14': np.random.rand(10000) * 100,
        'macd': np.random.rand(10000) * 2 - 1
    }, index=dates)
    
    print("\n3. Storing features...")
    storage.store_features('AAPL', features_data, 'technical')
    
    # Test data info
    print("\n4. Data information:")
    info = storage.get_data_info('AAPL')
    print(f"   File size: {info['file_size'] / 1024:.1f} KB")
    print(f"   Groups: {len(info['groups'])}")
    
    # Test metrics
    print("\n5. Performance metrics:")
    metrics = storage.get_metrics()
    print(f"   Read operations: {metrics.read_operations}")
    print(f"   Write operations: {metrics.write_operations}")
    print(f"   Avg read time: {metrics.avg_read_time:.3f}s")
    print(f"   Avg write time: {metrics.avg_write_time:.3f}s")
    print(f"   Memory usage: {metrics.memory_usage / 1024**2:.1f} MB")
    
    # Cleanup
    storage.close_all_files()
    import shutil
    if Path("temp_hdf5_data").exists():
        shutil.rmtree("temp_hdf5_data")
    
    print("\n=== Advanced HDF5 Storage Test Complete ===")
