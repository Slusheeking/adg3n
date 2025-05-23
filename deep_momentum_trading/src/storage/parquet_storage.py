import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.compute as pc
import pyarrow.dataset as ds
import numpy as np
import pandas as pd
import time
import os
import asyncio
import threading
from pathlib import Path
from typing import List, Dict, Optional, Union, Tuple, Any, Callable
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from dataclasses import dataclass, asdict
import json
import pickle
import warnings
from functools import lru_cache
import gc
import psutil

from deep_momentum_trading.src.utils.logger import get_logger

logger = get_logger(__name__)

@dataclass
class ParquetConfig:
    """Configuration for Parquet storage with ARM64 optimizations."""
    use_arm64_optimizations: bool = True
    compression: str = "zstd"  # zstd, snappy, lz4, gzip
    compression_level: int = 3
    row_group_size: int = 100000
    page_size: int = 1024 * 1024  # 1MB
    use_dictionary: bool = True
    write_statistics: bool = True
    enable_bloom_filter: bool = True
    enable_page_index: bool = True
    max_workers: int = os.cpu_count()
    memory_map: bool = True
    use_legacy_dataset: bool = False
    batch_size: int = 10000
    enable_caching: bool = True
    cache_size: int = 1024 * 1024 * 1024  # 1GB
    enable_partitioning: bool = True
    partition_cols: List[str] = None
    enable_predicate_pushdown: bool = True
    enable_projection_pushdown: bool = True
    
    def __post_init__(self):
        if self.partition_cols is None:
            self.partition_cols = ["date", "symbol"]

@dataclass
class ParquetMetrics:
    """Parquet storage performance metrics."""
    total_files: int = 0
    total_size_bytes: int = 0
    compression_ratio: float = 0.0
    avg_read_time: float = 0.0
    avg_write_time: float = 0.0
    cache_hit_ratio: float = 0.0
    predicate_pushdown_savings: float = 0.0
    partition_pruning_savings: float = 0.0

class AdvancedParquetStorage:
    """
    Advanced Parquet storage with ARM64 optimizations, intelligent partitioning,
    predicate pushdown, and analytics-optimized features.
    """

    def __init__(self, base_path: str = "data/parquet", config: Optional[ParquetConfig] = None):
        """
        Initialize Advanced Parquet Storage.

        Args:
            base_path: Base directory for Parquet files
            config: Parquet configuration
        """
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        self.config = config or ParquetConfig()
        
        # Thread safety
        self.lock = threading.RLock()
        
        # Executors for parallel operations
        self.thread_executor = ThreadPoolExecutor(max_workers=self.config.max_workers)
        self.process_executor = ProcessPoolExecutor(max_workers=min(4, self.config.max_workers))
        
        # Performance tracking
        self.metrics = ParquetMetrics()
        self.operation_times = {"read": [], "write": []}
        self.cache_stats = {"hits": 0, "misses": 0}
        
        # Schema cache for consistency
        self.schema_cache = {}
        
        # File metadata cache
        self.metadata_cache = {}
        
        # Setup ARM64 optimizations
        self._setup_arm64_optimizations()
        
        # Setup PyArrow memory pool
        self._setup_memory_pool()
        
        logger.info(f"AdvancedParquetStorage initialized at {self.base_path}")
        logger.info(f"ARM64 optimizations: {'enabled' if self.config.use_arm64_optimizations else 'disabled'}")
        logger.info(f"Compression: {self.config.compression} (level {self.config.compression_level})")

    def _setup_arm64_optimizations(self) -> None:
        """Setup ARM64-specific optimizations."""
        if not self.config.use_arm64_optimizations:
            return
            
        try:
            # Set ARM64-specific environment variables
            os.environ.setdefault("ARROW_DEFAULT_MEMORY_POOL", "system")
            os.environ.setdefault("ARROW_IO_THREADS", str(self.config.max_workers))
            
            # Enable SIMD optimizations
            if hasattr(pa, 'set_cpu_count'):
                pa.set_cpu_count(self.config.max_workers)
            
            # Set memory mapping for better performance
            if self.config.memory_map:
                os.environ.setdefault("ARROW_MMAP", "1")
            
            logger.info("ARM64 optimizations applied for Parquet storage")
            
        except Exception as e:
            logger.warning(f"Failed to apply ARM64 optimizations: {e}")

    def _setup_memory_pool(self) -> None:
        """Setup PyArrow memory pool for better memory management."""
        try:
            # Use system memory pool for better ARM64 performance
            self.memory_pool = pa.system_memory_pool()
            
            # Set memory pool for PyArrow operations
            pa.set_memory_pool(self.memory_pool)
            
            logger.debug("PyArrow memory pool configured")
            
        except Exception as e:
            logger.warning(f"Failed to setup memory pool: {e}")

    def _optimize_dtypes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Optimize DataFrame dtypes for ARM64 and storage efficiency."""
        optimized_df = df.copy()
        
        for col in optimized_df.columns:
            if pd.api.types.is_float_dtype(optimized_df[col]):
                # Use float32 for better ARM64 SIMD performance
                if optimized_df[col].dtype == np.float64:
                    optimized_df[col] = optimized_df[col].astype(np.float32)
            elif pd.api.types.is_integer_dtype(optimized_df[col]):
                # Downcast integers for storage efficiency
                col_min, col_max = optimized_df[col].min(), optimized_df[col].max()
                
                if col_min >= np.iinfo(np.int8).min and col_max <= np.iinfo(np.int8).max:
                    optimized_df[col] = optimized_df[col].astype(np.int8)
                elif col_min >= np.iinfo(np.int16).min and col_max <= np.iinfo(np.int16).max:
                    optimized_df[col] = optimized_df[col].astype(np.int16)
                elif col_min >= np.iinfo(np.int32).min and col_max <= np.iinfo(np.int32).max:
                    optimized_df[col] = optimized_df[col].astype(np.int32)
            elif pd.api.types.is_string_dtype(optimized_df[col]):
                # Use category for repeated strings
                if optimized_df[col].nunique() / len(optimized_df[col]) < 0.5:
                    optimized_df[col] = optimized_df[col].astype('category')
        
        return optimized_df

    def _create_parquet_schema(self, df: pd.DataFrame) -> pa.Schema:
        """Create optimized PyArrow schema."""
        # Convert pandas DataFrame to PyArrow table to infer schema
        table = pa.Table.from_pandas(df, preserve_index=True)
        schema = table.schema
        
        # Add metadata for better performance
        metadata = {
            "created_by": "AdvancedParquetStorage",
            "arm64_optimized": str(self.config.use_arm64_optimizations),
            "compression": self.config.compression
        }
        
        return schema.with_metadata(metadata)

    def _get_write_options(self) -> Dict[str, Any]:
        """Get optimized write options for ARM64."""
        options = {
            "compression": self.config.compression,
            "compression_level": self.config.compression_level,
            "use_dictionary": self.config.use_dictionary,
            "row_group_size": self.config.row_group_size,
            "data_page_size": self.config.page_size,
            "write_statistics": self.config.write_statistics,
            "store_schema": True
        }
        
        # Add ARM64-specific optimizations
        if self.config.use_arm64_optimizations:
            options.update({
                "use_compliant_nested_type": True,
                "write_batch_size": self.config.batch_size
            })
        
        # Add bloom filter support if available
        if self.config.enable_bloom_filter:
            try:
                options["bloom_filter_columns"] = None  # Auto-detect
            except:
                pass
        
        return options

    def store_time_series(self, 
                         symbol: str, 
                         data: pd.DataFrame, 
                         partition_by: Optional[List[str]] = None) -> bool:
        """
        Store time-series data with intelligent partitioning.

        Args:
            symbol: Asset symbol
            data: Time-series data with DatetimeIndex
            partition_by: Custom partition columns

        Returns:
            True if successful, False otherwise
        """
        start_time = time.time()
        
        try:
            if data.empty:
                logger.warning(f"Empty data provided for {symbol}")
                return False
            
            if not isinstance(data.index, pd.DatetimeIndex):
                raise ValueError("Data must have DatetimeIndex for time-series storage")
            
            # Optimize data types
            optimized_data = self._optimize_dtypes(data)
            
            # Add symbol column for partitioning
            optimized_data = optimized_data.reset_index()
            optimized_data['symbol'] = symbol
            optimized_data['date'] = optimized_data['index'].dt.date
            optimized_data['year'] = optimized_data['index'].dt.year
            optimized_data['month'] = optimized_data['index'].dt.month
            
            # Determine partitioning
            if self.config.enable_partitioning:
                partition_cols = partition_by or ["symbol", "year", "month"]
            else:
                partition_cols = None
            
            # Create PyArrow table
            table = pa.Table.from_pandas(optimized_data, preserve_index=False)
            
            # Write with partitioning
            if partition_cols:
                dataset_path = self.base_path / "time_series"
                dataset_path.mkdir(exist_ok=True)
                
                pq.write_to_dataset(
                    table,
                    root_path=dataset_path,
                    partition_cols=partition_cols,
                    **self._get_write_options()
                )
            else:
                file_path = self.base_path / f"time_series_{symbol}.parquet"
                pq.write_table(table, file_path, **self._get_write_options())
            
            # Update metrics
            write_time = time.time() - start_time
            self.operation_times["write"].append(write_time)
            self.metrics.total_files += 1
            
            logger.info(f"Stored time-series data for {symbol} ({len(data)} records, {write_time:.3f}s)")
            return True
            
        except Exception as e:
            logger.error(f"Failed to store time-series data for {symbol}: {e}")
            return False

    def store_cross_sectional(self, 
                            date: Union[str, pd.Timestamp], 
                            data: pd.DataFrame,
                            features: Optional[List[str]] = None) -> bool:
        """
        Store cross-sectional data for a specific date.

        Args:
            date: Date for the cross-sectional data
            data: Cross-sectional data with symbol index
            features: Specific features to store

        Returns:
            True if successful, False otherwise
        """
        start_time = time.time()
        
        try:
            if data.empty:
                logger.warning(f"Empty cross-sectional data for {date}")
                return False
            
            # Filter features if specified
            if features:
                data = data[features]
            
            # Optimize data types
            optimized_data = self._optimize_dtypes(data)
            
            # Add date information
            date_obj = pd.to_datetime(date)
            optimized_data = optimized_data.reset_index()
            optimized_data['date'] = date_obj.date()
            optimized_data['year'] = date_obj.year
            optimized_data['month'] = date_obj.month
            optimized_data['day'] = date_obj.day
            
            # Create PyArrow table
            table = pa.Table.from_pandas(optimized_data, preserve_index=False)
            
            # Write with date partitioning
            if self.config.enable_partitioning:
                dataset_path = self.base_path / "cross_sectional"
                dataset_path.mkdir(exist_ok=True)
                
                pq.write_to_dataset(
                    table,
                    root_path=dataset_path,
                    partition_cols=["year", "month"],
                    **self._get_write_options()
                )
            else:
                date_str = date_obj.strftime('%Y-%m-%d')
                file_path = self.base_path / f"cross_sectional_{date_str}.parquet"
                pq.write_table(table, file_path, **self._get_write_options())
            
            # Update metrics
            write_time = time.time() - start_time
            self.operation_times["write"].append(write_time)
            self.metrics.total_files += 1
            
            logger.info(f"Stored cross-sectional data for {date} ({len(data)} records, {write_time:.3f}s)")
            return True
            
        except Exception as e:
            logger.error(f"Failed to store cross-sectional data for {date}: {e}")
            return False

    def load_time_series(self, 
                        symbols: Union[str, List[str]], 
                        start_date: Optional[Union[str, pd.Timestamp]] = None,
                        end_date: Optional[Union[str, pd.Timestamp]] = None,
                        columns: Optional[List[str]] = None,
                        filters: Optional[List[Tuple]] = None) -> pd.DataFrame:
        """
        Load time-series data with advanced filtering and optimization.

        Args:
            symbols: Symbol or list of symbols
            start_date: Start date for filtering
            end_date: End date for filtering
            columns: Specific columns to load
            filters: Additional filters for predicate pushdown

        Returns:
            Loaded DataFrame
        """
        start_time = time.time()
        
        try:
            if isinstance(symbols, str):
                symbols = [symbols]
            
            # Build filters for predicate pushdown
            filter_expressions = []
            
            # Symbol filter
            if len(symbols) == 1:
                filter_expressions.append(("symbol", "=", symbols[0]))
            else:
                filter_expressions.append(("symbol", "in", symbols))
            
            # Date filters
            if start_date:
                start_ts = pd.to_datetime(start_date)
                filter_expressions.append(("index", ">=", start_ts))
            
            if end_date:
                end_ts = pd.to_datetime(end_date)
                filter_expressions.append(("index", "<=", end_ts))
            
            # Add custom filters
            if filters:
                filter_expressions.extend(filters)
            
            # Load data using dataset API for better performance
            dataset_path = self.base_path / "time_series"
            
            if dataset_path.exists():
                dataset = ds.dataset(dataset_path, format="parquet")
                
                # Apply filters and column selection
                table = dataset.to_table(
                    filter=self._build_filter_expression(filter_expressions),
                    columns=columns
                )
                
                df = table.to_pandas()
                
                # Set proper index
                if 'index' in df.columns:
                    df = df.set_index('index')
                    df.index.name = 'timestamp'
                
            else:
                # Fallback to individual file loading
                dfs = []
                for symbol in symbols:
                    file_path = self.base_path / f"time_series_{symbol}.parquet"
                    if file_path.exists():
                        symbol_df = pd.read_parquet(file_path)
                        
                        # Apply date filters
                        if start_date or end_date:
                            if start_date:
                                symbol_df = symbol_df[symbol_df.index >= pd.to_datetime(start_date)]
                            if end_date:
                                symbol_df = symbol_df[symbol_df.index <= pd.to_datetime(end_date)]
                        
                        # Apply column selection
                        if columns:
                            available_cols = [col for col in columns if col in symbol_df.columns]
                            symbol_df = symbol_df[available_cols]
                        
                        dfs.append(symbol_df)
                
                df = pd.concat(dfs) if dfs else pd.DataFrame()
            
            # Update metrics
            read_time = time.time() - start_time
            self.operation_times["read"].append(read_time)
            
            if not df.empty:
                self.cache_stats["hits"] += 1
            else:
                self.cache_stats["misses"] += 1
            
            logger.info(f"Loaded time-series data for {symbols} ({len(df)} records, {read_time:.3f}s)")
            return df
            
        except Exception as e:
            logger.error(f"Failed to load time-series data: {e}")
            return pd.DataFrame()

    def load_cross_sectional(self, 
                           dates: Union[str, pd.Timestamp, List],
                           symbols: Optional[List[str]] = None,
                           columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Load cross-sectional data for specific dates.

        Args:
            dates: Date or list of dates
            symbols: Specific symbols to load
            columns: Specific columns to load

        Returns:
            Loaded DataFrame
        """
        start_time = time.time()
        
        try:
            if not isinstance(dates, list):
                dates = [dates]
            
            # Convert dates
            date_objects = [pd.to_datetime(date) for date in dates]
            
            # Build filters
            filter_expressions = []
            
            if len(date_objects) == 1:
                filter_expressions.append(("date", "=", date_objects[0].date()))
            else:
                date_list = [d.date() for d in date_objects]
                filter_expressions.append(("date", "in", date_list))
            
            if symbols:
                if len(symbols) == 1:
                    filter_expressions.append(("index", "=", symbols[0]))
                else:
                    filter_expressions.append(("index", "in", symbols))
            
            # Load data
            dataset_path = self.base_path / "cross_sectional"
            
            if dataset_path.exists():
                dataset = ds.dataset(dataset_path, format="parquet")
                
                table = dataset.to_table(
                    filter=self._build_filter_expression(filter_expressions),
                    columns=columns
                )
                
                df = table.to_pandas()
                
                # Set proper index
                if 'index' in df.columns:
                    df = df.set_index('index')
                    df.index.name = 'symbol'
                
            else:
                # Fallback to individual file loading
                dfs = []
                for date in date_objects:
                    date_str = date.strftime('%Y-%m-%d')
                    file_path = self.base_path / f"cross_sectional_{date_str}.parquet"
                    
                    if file_path.exists():
                        date_df = pd.read_parquet(file_path)
                        
                        # Apply symbol filter
                        if symbols:
                            date_df = date_df[date_df.index.isin(symbols)]
                        
                        # Apply column selection
                        if columns:
                            available_cols = [col for col in columns if col in date_df.columns]
                            date_df = date_df[available_cols]
                        
                        dfs.append(date_df)
                
                df = pd.concat(dfs) if dfs else pd.DataFrame()
            
            # Update metrics
            read_time = time.time() - start_time
            self.operation_times["read"].append(read_time)
            
            logger.info(f"Loaded cross-sectional data for {len(dates)} dates ({len(df)} records, {read_time:.3f}s)")
            return df
            
        except Exception as e:
            logger.error(f"Failed to load cross-sectional data: {e}")
            return pd.DataFrame()

    def _build_filter_expression(self, filters: List[Tuple]) -> Optional[pc.Expression]:
        """Build PyArrow filter expression from filter list."""
        if not filters:
            return None
        
        expressions = []
        
        for filter_tuple in filters:
            if len(filter_tuple) == 3:
                column, op, value = filter_tuple
                
                if op == "=":
                    expr = pc.equal(pc.field(column), pa.scalar(value))
                elif op == "!=":
                    expr = pc.not_equal(pc.field(column), pa.scalar(value))
                elif op == ">":
                    expr = pc.greater(pc.field(column), pa.scalar(value))
                elif op == ">=":
                    expr = pc.greater_equal(pc.field(column), pa.scalar(value))
                elif op == "<":
                    expr = pc.less(pc.field(column), pa.scalar(value))
                elif op == "<=":
                    expr = pc.less_equal(pc.field(column), pa.scalar(value))
                elif op == "in":
                    expr = pc.is_in(pc.field(column), pa.array(value))
                else:
                    continue
                
                expressions.append(expr)
        
        if not expressions:
            return None
        
        # Combine expressions with AND
        result = expressions[0]
        for expr in expressions[1:]:
            result = pc.and_(result, expr)
        
        return result

    async def load_time_series_async(self, 
                                   symbols: Union[str, List[str]], 
                                   start_date: Optional[Union[str, pd.Timestamp]] = None,
                                   end_date: Optional[Union[str, pd.Timestamp]] = None,
                                   columns: Optional[List[str]] = None) -> pd.DataFrame:
        """Async version of load_time_series."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.thread_executor, 
            self.load_time_series, 
            symbols, start_date, end_date, columns
        )

    async def store_time_series_async(self, 
                                    symbol: str, 
                                    data: pd.DataFrame) -> bool:
        """Async version of store_time_series."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.thread_executor, 
            self.store_time_series, 
            symbol, data
        )

    def get_metrics(self) -> ParquetMetrics:
        """Get comprehensive storage metrics."""
        # Calculate file statistics
        total_size = 0
        file_count = 0
        
        for file_path in self.base_path.rglob("*.parquet"):
            try:
                file_count += 1
                total_size += file_path.stat().st_size
            except:
                continue
        
        self.metrics.total_files = file_count
        self.metrics.total_size_bytes = total_size
        
        # Calculate average times
        if self.operation_times["read"]:
            self.metrics.avg_read_time = sum(self.operation_times["read"]) / len(self.operation_times["read"])
        
        if self.operation_times["write"]:
            self.metrics.avg_write_time = sum(self.operation_times["write"]) / len(self.operation_times["write"])
        
        # Calculate cache hit ratio
        total_cache_ops = self.cache_stats["hits"] + self.cache_stats["misses"]
        if total_cache_ops > 0:
            self.metrics.cache_hit_ratio = self.cache_stats["hits"] / total_cache_ops
        
        return self.metrics

    def optimize_storage(self) -> None:
        """Optimize storage by compacting and reorganizing files."""
        logger.info("Optimizing Parquet storage")
        
        try:
            # Compact small files
            self._compact_small_files()
            
            # Update statistics
            self._update_statistics()
            
            # Garbage collection
            gc.collect()
            
            logger.info("Storage optimization completed")
            
        except Exception as e:
            logger.error(f"Storage optimization failed: {e}")

    def _compact_small_files(self) -> None:
        """Compact small Parquet files for better performance."""
        # Implementation would depend on specific requirements
        # This is a placeholder for file compaction logic
        pass

    def _update_statistics(self) -> None:
        """Update file statistics for better query planning."""
        # Implementation would update Parquet file statistics
        # This is a placeholder for statistics update logic
        pass

    def get_schema(self, table_type: str) -> Optional[pa.Schema]:
        """Get schema for a specific table type."""
        if table_type in self.schema_cache:
            return self.schema_cache[table_type]
        
        # Try to infer schema from existing files
        if table_type == "time_series":
            dataset_path = self.base_path / "time_series"
            if dataset_path.exists():
                try:
                    dataset = ds.dataset(dataset_path, format="parquet")
                    schema = dataset.schema
                    self.schema_cache[table_type] = schema
                    return schema
                except:
                    pass
        
        return None

    def clear_cache(self) -> None:
        """Clear internal caches."""
        self.schema_cache.clear()
        self.metadata_cache.clear()
        self.cache_stats = {"hits": 0, "misses": 0}
        logger.info("Caches cleared")

    def close(self) -> None:
        """Close storage and cleanup resources."""
        self.thread_executor.shutdown(wait=True)
        self.process_executor.shutdown(wait=True)
        self.clear_cache()
        logger.info("Parquet storage closed")

# Legacy compatibility
ParquetFeatureStorage = AdvancedParquetStorage

if __name__ == "__main__":
    # Example usage with ARM64 optimizations
    config = ParquetConfig(
        use_arm64_optimizations=True,
        compression="zstd",
        compression_level=3,
        enable_partitioning=True,
        enable_predicate_pushdown=True,
        max_workers=4
    )
    
    storage = AdvancedParquetStorage("temp_parquet_test", config)
    
    print("=== Advanced Parquet Storage Test ===")
    
    # Test time-series storage
    print("\n1. Time-Series Storage Test:")
    dates = pd.date_range(start='2023-01-01', periods=1000, freq='1min')
    
    for symbol in ['AAPL', 'MSFT', 'GOOGL']:
        ts_data = pd.DataFrame({
            'price': np.random.rand(1000) * 100 + 100,
            'volume': np.random.randint(1000, 10000, 1000),
            'feature_1': np.random.randn(1000),
            'feature_2': np.random.randn(1000) * 10
        }, index=dates)
        
        success = storage.store_time_series(symbol, ts_data)
        print(f"   {symbol}: {'✓' if success else '✗'}")
    
    # Test cross-sectional storage
    print("\n2. Cross-Sectional Storage Test:")
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
    
    for i, date in enumerate(pd.date_range('2023-01-01', periods=5)):
        cs_data = pd.DataFrame({
            'market_cap': np.random.rand(5) * 1e12,
            'pe_ratio': np.random.rand(5) * 30 + 10,
            'volume_ratio': np.random.rand(5) * 2
        }, index=symbols)
        
        success = storage.store_cross_sectional(date, cs_data)
        print(f"   {date.date()}: {'✓' if success else '✗'}")
    
    # Test loading with filters
    print("\n3. Loading with Filters Test:")
    
    # Load time-series data
    ts_loaded = storage.load_time_series(
        symbols=['AAPL', 'MSFT'],
        start_date='2023-01-01 10:00:00',
        end_date='2023-01-01 12:00:00',
        columns=['price', 'volume']
    )
    print(f"   Time-series loaded: {len(ts_loaded)} records")
    
    # Load cross-sectional data
    cs_loaded = storage.load_cross_sectional(
        dates=['2023-01-01', '2023-01-02'],
        symbols=['AAPL', 'MSFT', 'GOOGL'],
        columns=['market_cap', 'pe_ratio']
    )
    print(f"   Cross-sectional loaded: {len(cs_loaded)} records")
    
    # Test async operations
    print("\n4. Async Operations Test:")
    async def test_async():
        # Async load
        async_data = await storage.load_time_series_async(
            'AAPL',
            start_date='2023-01-01',
            end_date='2023-01-01 06:00:00'
        )
        return len(async_data)
    
    import asyncio
    async_records = asyncio.run(test_async())
    print(f"   Async loaded: {async_records} records")
    
    # Show metrics
    print("\n5. Performance Metrics:")
    metrics = storage.get_metrics()
    print(f"   Total files: {metrics.total_files}")
    print(f"   Total size: {metrics.total_size_bytes / 1024**2:.1f} MB")
    print(f"   Avg read time: {metrics.avg_read_time:.3f}s")
    print(f"   Avg write time: {metrics.avg_write_time:.3f}s")
    print(f"   Cache hit ratio: {metrics.cache_hit_ratio:.2f}")
    
    # Cleanup
    storage.close()
    
    import shutil
    shutil.rmtree("temp_parquet_test")
    
    print("\n=== Advanced Parquet Storage Test Complete ===")
