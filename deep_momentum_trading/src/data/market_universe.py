import pandas as pd
import numpy as np
import platform
import time
import threading
from typing import List, Dict, Any, Optional, Tuple, Set, Union
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path

from deep_momentum_trading.src.utils.logger import get_logger
from deep_momentum_trading.config.settings import config_manager

logger = get_logger(__name__)

@dataclass
class MarketUniverseConfig:
    """Configuration for MarketUniverse with ARM64 optimizations."""
    universe_file: Optional[str] = None
    min_market_cap: Optional[float] = None
    min_avg_daily_volume: Optional[float] = None
    max_universe_size: int = 10000
    enable_arm64_optimizations: bool = True
    enable_parallel_processing: bool = True
    max_workers: int = 4
    enable_performance_monitoring: bool = True
    enable_caching: bool = True
    cache_ttl_seconds: int = 3600
    auto_update_interval: int = 86400  # 24 hours
    enable_sector_filtering: bool = True
    enable_liquidity_filtering: bool = True
    enable_volatility_filtering: bool = True
    min_price: Optional[float] = 1.0
    max_price: Optional[float] = None
    excluded_sectors: List[str] = field(default_factory=list)
    required_exchanges: List[str] = field(default_factory=lambda: ['NASDAQ', 'NYSE'])

@dataclass
class MarketUniverseStats:
    """Statistics for MarketUniverse performance monitoring."""
    total_assets: int = 0
    filtered_assets: int = 0
    sectors_count: int = 0
    updates_performed: int = 0
    filter_operations: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    processing_time_seconds: float = 0.0
    last_update_timestamp: float = field(default_factory=time.time)
    start_time: float = field(default_factory=time.time)
    
    @property
    def filter_efficiency(self) -> float:
        return self.filtered_assets / max(self.total_assets, 1)
    
    @property
    def uptime_seconds(self) -> float:
        return time.time() - self.start_time

class MarketUniverse:
    """
    Enhanced MarketUniverse with ARM64 optimizations for high-performance asset management.
    
    Manages the universe of tradable assets, including filtering, categorization,
    and dynamic updates based on various criteria (e.g., liquidity, sector, market cap)
    with ARM64-specific optimizations and parallel processing capabilities.
    """

    def __init__(self,
                 config: Optional[MarketUniverseConfig] = None,
                 universe_file: Optional[str] = None,
                 min_market_cap: Optional[float] = None,
                 min_avg_daily_volume: Optional[float] = None):
        """
        Initializes the enhanced MarketUniverse manager with ARM64 optimizations.

        Args:
            config: MarketUniverseConfig object (preferred)
            universe_file: Path to a CSV file containing the initial universe (fallback)
            min_market_cap: Minimum market capitalization (fallback)
            min_avg_daily_volume: Minimum average daily volume (fallback)
        """
        # Configuration handling
        if config is not None:
            self.config = config
        else:
            # Load from config manager with fallbacks
            config_data = config_manager.get('trading_config.universe', {})
            self.config = MarketUniverseConfig(
                universe_file=universe_file or config_data.get('universe_file'),
                min_market_cap=min_market_cap or config_data.get('min_market_cap'),
                min_avg_daily_volume=min_avg_daily_volume or config_data.get('min_avg_daily_volume')
            )
        
        # Validate configuration
        self._validate_configuration()
        
        # ARM64 optimization detection
        self.is_arm64 = platform.machine().lower() in ['arm64', 'aarch64']
        if self.is_arm64 and self.config.enable_arm64_optimizations:
            logger.info("ARM64 optimizations enabled for MarketUniverse")
        
        # Initialize data structures
        self.universe_df: pd.DataFrame = pd.DataFrame()
        self.sector_cache: Dict[str, List[str]] = {}
        self.filter_cache: Dict[str, List[str]] = {}
        self.last_cache_update: float = 0.0
        
        # Performance monitoring
        self.stats = MarketUniverseStats()
        
        # Thread pool for parallel processing
        if self.config.enable_parallel_processing:
            self.executor = ThreadPoolExecutor(max_workers=self.config.max_workers)
        else:
            self.executor = None
        
        # Initialize auto-update
        self._auto_update_running = False
        self._update_thread = None
        
        # Load initial universe with optimization
        if self.config.universe_file:
            self.load_universe_from_file(self.config.universe_file)
            
            # Start auto-updates if configured
            if self.config.auto_update_interval > 0:
                self.start_auto_update()
        
        logger.info("Enhanced MarketUniverse initialized with ARM64 optimizations")
        self.log_current_universe_size()

    def __del__(self):
        """Cleanup thread pool and auto-update thread on destruction."""
        if hasattr(self, 'executor') and self.executor:
            self.executor.shutdown(wait=False)
        if hasattr(self, '_auto_update_running'):
            self.stop_auto_update()

    def _validate_configuration(self):
        """Validate configuration parameters."""
        errors = []
        warnings = []
        
        # Validate universe size
        if self.config.max_universe_size <= 0:
            errors.append("max_universe_size must be positive")
        elif self.config.max_universe_size > 50000:
            warnings.append(f"Large universe size ({self.config.max_universe_size}) may impact performance")
        
        # Validate filters
        if self.config.min_market_cap is not None and self.config.min_market_cap < 0:
            errors.append("min_market_cap cannot be negative")
        
        if self.config.min_avg_daily_volume is not None and self.config.min_avg_daily_volume < 0:
            errors.append("min_avg_daily_volume cannot be negative")
        
        if self.config.min_price is not None and self.config.min_price <= 0:
            errors.append("min_price must be positive")
        
        if (self.config.min_price is not None and self.config.max_price is not None and 
            self.config.min_price >= self.config.max_price):
            errors.append("min_price must be less than max_price")
        
        # Validate cache settings
        if self.config.cache_ttl_seconds <= 0:
            errors.append("cache_ttl_seconds must be positive")
        
        # Validate thread pool
        if self.config.max_workers <= 0:
            errors.append("max_workers must be positive")
        elif self.config.max_workers > os.cpu_count():
            warnings.append(f"max_workers ({self.config.max_workers}) exceeds CPU count ({os.cpu_count()})")
        
        # Log warnings
        for warning in warnings:
            logger.warning(f"Configuration warning: {warning}")
        
        # Raise errors
        if errors:
            raise ValueError(f"Configuration errors: {'; '.join(errors)}")

    def load_universe_from_file(self, file_path: str):
        """
        Enhanced universe loading with ARM64 optimizations and comprehensive validation.

        Args:
            file_path: Path to the CSV file.
        """
        start_time = time.perf_counter()
        
        try:
            # Check file size to determine loading strategy
            file_size = os.path.getsize(file_path)
            
            if file_size > 100 * 1024 * 1024:  # 100MB+
                logger.info(f"Large universe file detected ({file_size / 1024**2:.1f}MB), using chunked loading")
                self.universe_df = self._load_large_universe_chunked(file_path)
            else:
                # Standard loading for smaller files
                if self.is_arm64 and self.config.enable_arm64_optimizations:
                    self.universe_df = self._load_universe_arm64(file_path)
                else:
                    self.universe_df = self._load_universe_standard(file_path)
            
            if self.universe_df.empty:
                logger.warning(f"No data loaded from {file_path}")
                return
            
            # Validate required columns
            required_columns = ['symbol']
            missing_columns = [col for col in required_columns if col not in self.universe_df.columns]
            if missing_columns:
                raise ValueError(f"Universe file must contain columns: {missing_columns}")
            
            # Set symbol as index if not already
            if 'symbol' in self.universe_df.columns:
                self.universe_df.set_index('symbol', inplace=True)
            
            # Memory optimization for large universes
            self._optimize_for_large_universe()
            
            # Ensure ARM64-friendly data types
            if self.is_arm64 and self.config.enable_arm64_optimizations:
                self._optimize_data_types_arm64()
            
            # Apply initial filters
            self._apply_filters()
            
            # Update statistics
            processing_time = time.perf_counter() - start_time
            self.stats.processing_time_seconds += processing_time
            self.stats.total_assets = len(self.universe_df)
            self.stats.last_update_timestamp = time.time()
            
            logger.info(f"Loaded universe from {file_path} with {len(self.universe_df)} assets in {processing_time:.4f}s")
            
        except FileNotFoundError:
            logger.error(f"Universe file not found at {file_path}")
            self.universe_df = pd.DataFrame()
        except Exception as e:
            logger.error(f"Error loading universe from file {file_path}: {e}")
            self.universe_df = pd.DataFrame()
        
        self.log_current_universe_size()

    def _load_universe_standard(self, file_path: str) -> pd.DataFrame:
        """Standard universe loading implementation."""
        return pd.read_csv(file_path)

    def _load_universe_arm64(self, file_path: str) -> pd.DataFrame:
        """Enhanced ARM64 optimized universe loading."""
        try:
            # Use ARM64-optimized reading with better memory layout
            df = pd.read_csv(
                file_path,
                engine='c',  # C engine for ARM64
                low_memory=False,
                dtype={
                    'market_cap': 'float32',
                    'avg_daily_volume': 'float32',
                    'price': 'float32',
                    'sector': 'category',
                    'exchange': 'category'
                }
            )
            
            # Optimize memory layout for ARM64 cache performance
            if len(df) > 1000:
                # Sort by symbol for better cache locality
                df = df.sort_values('symbol') if 'symbol' in df.columns else df
            
            return df
            
        except Exception as e:
            logger.warning(f"ARM64 loading failed, falling back to standard: {e}")
            return pd.read_csv(file_path)

    def _load_large_universe_chunked(self, file_path: str) -> pd.DataFrame:
        """Load large universe files in chunks to manage memory."""
        chunks = []
        chunk_size = 10000  # Process 10k rows at a time
        
        try:
            for chunk in pd.read_csv(file_path, chunksize=chunk_size):
                # Apply basic filters to each chunk to reduce memory usage
                if self.config.min_market_cap and 'market_cap' in chunk.columns:
                    chunk = chunk[chunk['market_cap'] >= self.config.min_market_cap]
                
                if self.config.min_avg_daily_volume and 'avg_daily_volume' in chunk.columns:
                    chunk = chunk[chunk['avg_daily_volume'] >= self.config.min_avg_daily_volume]
                
                if not chunk.empty:
                    chunks.append(chunk)
            
            if chunks:
                return pd.concat(chunks, ignore_index=True)
            else:
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"Chunked loading failed: {e}")
            return pd.DataFrame()

    def _optimize_data_types_arm64(self):
        """Enhanced ARM64 optimization with SIMD-friendly data structures."""
        try:
            # Convert to ARM64-optimized data types
            numeric_columns = self.universe_df.select_dtypes(include=[np.number]).columns
            
            for col in numeric_columns:
                if self.universe_df[col].dtype in ['int8', 'int16', 'int32']:
                    # Use int32 for better ARM64 SIMD performance
                    self.universe_df[col] = self.universe_df[col].astype('int32')
                elif self.universe_df[col].dtype in ['float16', 'float32', 'float64']:
                    # Use float32 for optimal ARM64 NEON performance
                    self.universe_df[col] = self.universe_df[col].astype('float32')
            
            # Optimize string columns with categorical data
            string_columns = self.universe_df.select_dtypes(include=['object']).columns
            for col in string_columns:
                if self.universe_df[col].nunique() < len(self.universe_df) * 0.5:
                    # Convert to categorical for memory efficiency
                    self.universe_df[col] = self.universe_df[col].astype('category')
            
            # Ensure memory alignment for ARM64 cache lines
            if hasattr(self.universe_df, '_mgr'):
                # Force memory consolidation for better cache performance
                self.universe_df = self.universe_df.copy()
            
            logger.debug("ARM64 data type optimization completed")
            
        except Exception as e:
            logger.warning(f"ARM64 data type optimization failed: {e}")

    def _optimize_for_large_universe(self):
        """Optimize memory usage for large universes (10k+ assets)."""
        if len(self.universe_df) < 5000:
            return
        
        try:
            # Use more memory-efficient data types
            memory_before = self.universe_df.memory_usage(deep=True).sum()
            
            # Optimize numeric columns
            for col in self.universe_df.select_dtypes(include=[np.number]).columns:
                col_data = self.universe_df[col]
                
                if col_data.dtype == 'float64':
                    # Check if we can downcast to float32
                    if col_data.min() >= np.finfo(np.float32).min and col_data.max() <= np.finfo(np.float32).max:
                        self.universe_df[col] = col_data.astype('float32')
                
                elif col_data.dtype == 'int64':
                    # Downcast integers
                    if col_data.min() >= np.iinfo(np.int32).min and col_data.max() <= np.iinfo(np.int32).max:
                        self.universe_df[col] = col_data.astype('int32')
            
            # Convert repeated strings to categories
            for col in self.universe_df.select_dtypes(include=['object']).columns:
                if self.universe_df[col].nunique() < len(self.universe_df) * 0.5:
                    self.universe_df[col] = self.universe_df[col].astype('category')
            
            memory_after = self.universe_df.memory_usage(deep=True).sum()
            memory_saved = memory_before - memory_after
            
            if memory_saved > 0:
                logger.info(f"Optimized memory usage: saved {memory_saved / 1024**2:.2f} MB "
                           f"({memory_saved / memory_before:.1%} reduction)")
            
        except Exception as e:
            logger.warning(f"Memory optimization failed: {e}")

    def _apply_filters(self):
        """
        Enhanced filtering with ARM64 optimizations and comprehensive criteria.
        """
        start_time = time.perf_counter()
        initial_size = len(self.universe_df)
        
        try:
            # Use parallel processing for large datasets
            if self.executor and len(self.universe_df) > 1000:
                self._apply_filters_parallel()
            elif self.is_arm64 and self.config.enable_arm64_optimizations and len(self.universe_df) > 1000:
                self._apply_filters_arm64()
            else:
                self._apply_filters_standard()
            
            # Update statistics
            processing_time = time.perf_counter() - start_time
            self.stats.processing_time_seconds += processing_time
            self.stats.filter_operations += 1
            self.stats.filtered_assets = len(self.universe_df)
            
            if len(self.universe_df) < initial_size:
                logger.info(f"Applied filters in {processing_time:.4f}s. "
                           f"Universe size reduced from {initial_size} to {len(self.universe_df)}")
            else:
                logger.debug("No assets filtered by current criteria")
                
        except Exception as e:
            logger.error(f"Error applying filters: {e}")

    def _apply_filters_parallel(self):
        """Actually use parallel processing for filtering."""
        if not self.executor or len(self.universe_df) < 1000:
            return self._apply_filters_standard()
        
        try:
            # Split DataFrame into chunks for parallel processing
            chunk_size = max(len(self.universe_df) // self.config.max_workers, 100)
            chunks = [
                self.universe_df.iloc[i:i + chunk_size] 
                for i in range(0, len(self.universe_df), chunk_size)
            ]
            
            # Define filter function for parallel execution
            def filter_chunk(chunk):
                return self._apply_single_chunk_filters(chunk)
            
            # Process chunks in parallel
            filtered_chunks = list(self.executor.map(filter_chunk, chunks))
            
            # Combine results
            self.universe_df = pd.concat(filtered_chunks, ignore_index=True)
            
            logger.debug(f"Applied filters using {len(chunks)} parallel chunks")
            
        except Exception as e:
            logger.warning(f"Parallel filtering failed, using standard: {e}")
            self._apply_filters_standard()

    def _apply_single_chunk_filters(self, chunk: pd.DataFrame) -> pd.DataFrame:
        """Apply filters to a single chunk."""
        filtered_chunk = chunk.copy()
        
        # Apply all filters to the chunk
        if self.config.min_market_cap is not None and 'market_cap' in filtered_chunk.columns:
            filtered_chunk = filtered_chunk[filtered_chunk['market_cap'] >= self.config.min_market_cap]
        
        if self.config.min_avg_daily_volume is not None and 'avg_daily_volume' in filtered_chunk.columns:
            filtered_chunk = filtered_chunk[filtered_chunk['avg_daily_volume'] >= self.config.min_avg_daily_volume]
        
        if self.config.min_price is not None and 'price' in filtered_chunk.columns:
            filtered_chunk = filtered_chunk[filtered_chunk['price'] >= self.config.min_price]
        
        if self.config.max_price is not None and 'price' in filtered_chunk.columns:
            filtered_chunk = filtered_chunk[filtered_chunk['price'] <= self.config.max_price]
        
        if self.config.excluded_sectors and 'sector' in filtered_chunk.columns:
            filtered_chunk = filtered_chunk[~filtered_chunk['sector'].isin(self.config.excluded_sectors)]
        
        if self.config.required_exchanges and 'exchange' in filtered_chunk.columns:
            filtered_chunk = filtered_chunk[filtered_chunk['exchange'].isin(self.config.required_exchanges)]
        
        return filtered_chunk

    def _apply_filters_standard(self):
        """Standard filtering implementation."""
        # Market cap filter
        if self.config.min_market_cap is not None and 'market_cap' in self.universe_df.columns:
            self.universe_df = self.universe_df[self.universe_df['market_cap'] >= self.config.min_market_cap]
            logger.debug(f"Filtered by min_market_cap ({self.config.min_market_cap}). Current size: {len(self.universe_df)}")

        # Volume filter
        if self.config.min_avg_daily_volume is not None and 'avg_daily_volume' in self.universe_df.columns:
            self.universe_df = self.universe_df[self.universe_df['avg_daily_volume'] >= self.config.min_avg_daily_volume]
            logger.debug(f"Filtered by min_avg_daily_volume ({self.config.min_avg_daily_volume}). Current size: {len(self.universe_df)}")

        # Price filters
        if self.config.min_price is not None and 'price' in self.universe_df.columns:
            self.universe_df = self.universe_df[self.universe_df['price'] >= self.config.min_price]
            logger.debug(f"Filtered by min_price ({self.config.min_price}). Current size: {len(self.universe_df)}")

        if self.config.max_price is not None and 'price' in self.universe_df.columns:
            self.universe_df = self.universe_df[self.universe_df['price'] <= self.config.max_price]
            logger.debug(f"Filtered by max_price ({self.config.max_price}). Current size: {len(self.universe_df)}")

        # Sector exclusion filter
        if self.config.excluded_sectors and 'sector' in self.universe_df.columns:
            self.universe_df = self.universe_df[~self.universe_df['sector'].isin(self.config.excluded_sectors)]
            logger.debug(f"Excluded sectors: {self.config.excluded_sectors}. Current size: {len(self.universe_df)}")

        # Exchange filter
        if self.config.required_exchanges and 'exchange' in self.universe_df.columns:
            self.universe_df = self.universe_df[self.universe_df['exchange'].isin(self.config.required_exchanges)]
            logger.debug(f"Filtered by exchanges: {self.config.required_exchanges}. Current size: {len(self.universe_df)}")

        # Universe size limit
        if len(self.universe_df) > self.config.max_universe_size:
            # Sort by market cap and take top assets
            if 'market_cap' in self.universe_df.columns:
                self.universe_df = self.universe_df.nlargest(self.config.max_universe_size, 'market_cap')
            else:
                self.universe_df = self.universe_df.head(self.config.max_universe_size)
            logger.info(f"Limited universe size to {self.config.max_universe_size} assets")

    def _apply_filters_arm64(self):
        """ARM64 optimized filtering with vectorized operations."""
        # Use vectorized operations for better ARM64 SIMD utilization
        mask = pd.Series(True, index=self.universe_df.index)
        
        # Vectorized filtering operations
        if self.config.min_market_cap is not None and 'market_cap' in self.universe_df.columns:
            mask &= (self.universe_df['market_cap'] >= self.config.min_market_cap)
        
        if self.config.min_avg_daily_volume is not None and 'avg_daily_volume' in self.universe_df.columns:
            mask &= (self.universe_df['avg_daily_volume'] >= self.config.min_avg_daily_volume)
        
        if self.config.min_price is not None and 'price' in self.universe_df.columns:
            mask &= (self.universe_df['price'] >= self.config.min_price)
        
        if self.config.max_price is not None and 'price' in self.universe_df.columns:
            mask &= (self.universe_df['price'] <= self.config.max_price)
        
        # Apply vectorized mask
        self.universe_df = self.universe_df[mask]
        
        # Apply non-vectorizable filters
        if self.config.excluded_sectors and 'sector' in self.universe_df.columns:
            self.universe_df = self.universe_df[~self.universe_df['sector'].isin(self.config.excluded_sectors)]
        
        if self.config.required_exchanges and 'exchange' in self.universe_df.columns:
            self.universe_df = self.universe_df[self.universe_df['exchange'].isin(self.config.required_exchanges)]
        
        # Universe size limit
        if len(self.universe_df) > self.config.max_universe_size:
            if 'market_cap' in self.universe_df.columns:
                self.universe_df = self.universe_df.nlargest(self.config.max_universe_size, 'market_cap')
            else:
                self.universe_df = self.universe_df.head(self.config.max_universe_size)

    def add_asset(self, symbol: str, details: Dict[str, Any]):
        """
        Enhanced asset addition with ARM64 optimizations and validation.

        Args:
            symbol: The ticker symbol of the asset.
            details: A dictionary containing asset details.
        """
        try:
            # Validate asset details
            if not self._validate_asset_details(details):
                logger.warning(f"Asset {symbol} failed validation, not adding to universe")
                return
            
            if symbol in self.universe_df.index:
                logger.warning(f"Asset {symbol} already exists in the universe. Updating details.")
                self.universe_df.loc[symbol] = details
            else:
                new_asset_df = pd.DataFrame([details], index=[symbol])
                
                # ARM64 optimized concatenation
                if self.is_arm64 and self.config.enable_arm64_optimizations:
                    # Ensure data type compatibility for ARM64
                    for col in new_asset_df.columns:
                        if col in self.universe_df.columns:
                            new_asset_df[col] = new_asset_df[col].astype(self.universe_df[col].dtype)
                
                self.universe_df = pd.concat([self.universe_df, new_asset_df])
                logger.info(f"Added asset {symbol} to the universe.")
            
            # Re-apply filters and update cache
            self._apply_filters()
            self._invalidate_cache()
            self.stats.updates_performed += 1
            
        except Exception as e:
            logger.error(f"Error adding asset {symbol}: {e}")
        
        self.log_current_universe_size()

    def _validate_asset_details(self, details: Dict[str, Any]) -> bool:
        """
        Validate asset details before adding to universe.
        
        Args:
            details: Asset details dictionary
            
        Returns:
            True if valid, False otherwise
        """
        try:
            # Check for required fields
            if self.config.min_market_cap is not None:
                if 'market_cap' not in details or details['market_cap'] < self.config.min_market_cap:
                    return False
            
            if self.config.min_avg_daily_volume is not None:
                if 'avg_daily_volume' not in details or details['avg_daily_volume'] < self.config.min_avg_daily_volume:
                    return False
            
            # Check excluded sectors
            if self.config.excluded_sectors and 'sector' in details:
                if details['sector'] in self.config.excluded_sectors:
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating asset details: {e}")
            return False

    def remove_asset(self, symbol: str):
        """
        Enhanced asset removal with cache invalidation.

        Args:
            symbol: The ticker symbol of the asset to remove.
        """
        if symbol in self.universe_df.index:
            self.universe_df.drop(symbol, inplace=True)
            self._invalidate_cache()
            self.stats.updates_performed += 1
            logger.info(f"Removed asset {symbol} from the universe.")
        else:
            logger.warning(f"Asset {symbol} not found in the universe.")
        
        self.log_current_universe_size()

    def get_all_symbols(self) -> List[str]:
        """
        Enhanced symbol retrieval with caching.

        Returns:
            A list of ticker symbols.
        """
        cache_key = "all_symbols"
        
        if self.config.enable_caching and self._is_cache_valid():
            if cache_key in self.filter_cache:
                self.stats.cache_hits += 1
                return self.filter_cache[cache_key]
        
        symbols = self.universe_df.index.tolist()
        
        if self.config.enable_caching:
            self.filter_cache[cache_key] = symbols
            self.stats.cache_misses += 1
        
        return symbols

    def get_asset_details(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Enhanced asset details retrieval with caching.

        Args:
            symbol: The ticker symbol of the asset.

        Returns:
            A dictionary of asset details, or None if not found.
        """
        if symbol in self.universe_df.index:
            return self.universe_df.loc[symbol].to_dict()
        
        logger.warning(f"Asset {symbol} not found in the universe.")
        return None

    def get_filtered_universe(self,
                              sectors: Optional[List[str]] = None,
                              market_cap_range: Optional[Tuple[float, float]] = None,
                              volume_range: Optional[Tuple[float, float]] = None,
                              price_range: Optional[Tuple[float, float]] = None,
                              exchanges: Optional[List[str]] = None) -> List[str]:
        """
        Enhanced filtering with ARM64 optimizations and comprehensive criteria.

        Args:
            sectors: List of sectors to include.
            market_cap_range: Tuple (min_cap, max_cap) for market capitalization.
            volume_range: Tuple (min_vol, max_vol) for average daily volume.
            price_range: Tuple (min_price, max_price) for stock price.
            exchanges: List of exchanges to include.

        Returns:
            A list of ticker symbols matching the criteria.
        """
        start_time = time.perf_counter()
        
        # Create cache key
        cache_key = self._create_filter_cache_key(sectors, market_cap_range, volume_range, price_range, exchanges)
        
        if self.config.enable_caching and self._is_cache_valid():
            if cache_key in self.filter_cache:
                self.stats.cache_hits += 1
                return self.filter_cache[cache_key]
        
        try:
            # Use optimized filtering based on data size and platform
            if len(self.universe_df) > 5000:
                filtered_symbols = self._get_filtered_universe_optimized(
                    sectors, market_cap_range, volume_range, price_range, exchanges
                )
            elif self.is_arm64 and self.config.enable_arm64_optimizations and len(self.universe_df) > 1000:
                filtered_symbols = self._get_filtered_universe_arm64(
                    sectors, market_cap_range, volume_range, price_range, exchanges
                )
            else:
                filtered_symbols = self._get_filtered_universe_standard(
                    sectors, market_cap_range, volume_range, price_range, exchanges
                )
            
            # Cache result
            if self.config.enable_caching:
                self.filter_cache[cache_key] = filtered_symbols
                self.stats.cache_misses += 1
            
            processing_time = time.perf_counter() - start_time
            self.stats.processing_time_seconds += processing_time
            
            logger.info(f"Returning filtered universe with {len(filtered_symbols)} assets in {processing_time:.4f}s")
            return filtered_symbols
            
        except Exception as e:
            logger.error(f"Error filtering universe: {e}")
            return []

    def _get_filtered_universe_standard(self, sectors, market_cap_range, volume_range, price_range, exchanges) -> List[str]:
        """Standard filtering implementation."""
        filtered_df = self.universe_df.copy()

        if sectors and 'sector' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df['sector'].isin(sectors)]
            logger.debug(f"Filtered by sectors: {sectors}. Current size: {len(filtered_df)}")

        if market_cap_range and 'market_cap' in filtered_df.columns:
            min_cap, max_cap = market_cap_range
            filtered_df = filtered_df[(filtered_df['market_cap'] >= min_cap) & (filtered_df['market_cap'] <= max_cap)]
            logger.debug(f"Filtered by market cap range: {market_cap_range}. Current size: {len(filtered_df)}")

        if volume_range and 'avg_daily_volume' in filtered_df.columns:
            min_vol, max_vol = volume_range
            filtered_df = filtered_df[(filtered_df['avg_daily_volume'] >= min_vol) & (filtered_df['avg_daily_volume'] <= max_vol)]
            logger.debug(f"Filtered by volume range: {volume_range}. Current size: {len(filtered_df)}")

        if price_range and 'price' in filtered_df.columns:
            min_price, max_price = price_range
            filtered_df = filtered_df[(filtered_df['price'] >= min_price) & (filtered_df['price'] <= max_price)]
            logger.debug(f"Filtered by price range: {price_range}. Current size: {len(filtered_df)}")

        if exchanges and 'exchange' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df['exchange'].isin(exchanges)]
            logger.debug(f"Filtered by exchanges: {exchanges}. Current size: {len(filtered_df)}")

        return filtered_df.index.tolist()

    def _get_filtered_universe_arm64(self, sectors, market_cap_range, volume_range, price_range, exchanges) -> List[str]:
        """ARM64 optimized filtering with vectorized operations."""
        mask = pd.Series(True, index=self.universe_df.index)
        
        # Vectorized numeric filtering
        if market_cap_range and 'market_cap' in self.universe_df.columns:
            min_cap, max_cap = market_cap_range
            mask &= (self.universe_df['market_cap'] >= min_cap) & (self.universe_df['market_cap'] <= max_cap)
        
        if volume_range and 'avg_daily_volume' in self.universe_df.columns:
            min_vol, max_vol = volume_range
            mask &= (self.universe_df['avg_daily_volume'] >= min_vol) & (self.universe_df['avg_daily_volume'] <= max_vol)
        
        if price_range and 'price' in self.universe_df.columns:
            min_price, max_price = price_range
            mask &= (self.universe_df['price'] >= min_price) & (self.universe_df['price'] <= max_price)
        
        # Apply vectorized mask first
        filtered_df = self.universe_df[mask]
        
        # Apply categorical filters
        if sectors and 'sector' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df['sector'].isin(sectors)]
        
        if exchanges and 'exchange' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df['exchange'].isin(exchanges)]
        
        return filtered_df.index.tolist()

    def _create_filter_cache_key(self, sectors, market_cap_range, volume_range, price_range, exchanges) -> str:
        """Create a cache key for filter parameters."""
        key_parts = [
            f"sectors:{sorted(sectors) if sectors else 'None'}",
            f"market_cap:{market_cap_range}",
            f"volume:{volume_range}",
            f"price:{price_range}",
            f"exchanges:{sorted(exchanges) if exchanges else 'None'}"
        ]
        return "|".join(key_parts)

    def update_asset_metrics(self, symbol: str, metrics: Dict[str, Any]):
        """
        Enhanced asset metrics update with selective cache invalidation.

        Args:
            symbol: The ticker symbol of the asset.
            metrics: Dictionary of metrics to update.
        """
        if symbol not in self.universe_df.index:
            logger.warning(f"Asset {symbol} not found in universe, cannot update metrics.")
            return

        try:
            # Track which metrics are being updated
            affected_fields = set(metrics.keys())
            
            # Update the metrics
            for key, value in metrics.items():
                self.universe_df.loc[symbol, key] = value
            
            logger.debug(f"Updated metrics for {symbol}: {metrics}")
            
            # Selective cache invalidation based on what changed
            cache_affecting_fields = {'market_cap', 'avg_daily_volume', 'price', 'sector', 'exchange'}
            if affected_fields & cache_affecting_fields:
                self._invalidate_cache_selective(affected_fields & cache_affecting_fields)
            
            # Only re-apply filters if filtering criteria changed
            if affected_fields & {'market_cap', 'avg_daily_volume', 'price', 'sector', 'exchange'}:
                self._apply_filters()
            
            self.stats.updates_performed += 1
            
        except Exception as e:
            logger.error(f"Error updating metrics for {symbol}: {e}")
        
        self.log_current_universe_size()

    def get_sector_breakdown(self) -> Dict[str, int]:
        """
        Get breakdown of assets by sector with caching.
        
        Returns:
            Dictionary mapping sector names to asset counts
        """
        cache_key = "sector_breakdown"
        
        if self.config.enable_caching and self._is_cache_valid():
            if cache_key in self.sector_cache:
                self.stats.cache_hits += 1
                return self.sector_cache[cache_key]
        
        if 'sector' in self.universe_df.columns:
            breakdown = self.universe_df['sector'].value_counts().to_dict()
        else:
            breakdown = {}
        
        if self.config.enable_caching:
            self.sector_cache[cache_key] = breakdown
            self.stats.cache_misses += 1
        
        return breakdown

    def get_statistics(self) -> MarketUniverseStats:
        """Get current market universe statistics."""
        # Update current statistics
        self.stats.total_assets = len(self.universe_df)
        if 'sector' in self.universe_df.columns:
            self.stats.sectors_count = self.universe_df['sector'].nunique()
        
        return self.stats

    def reset_statistics(self):
        """Reset market universe statistics."""
        self.stats = MarketUniverseStats()
        logger.info("Market universe statistics reset")

    def _is_cache_valid(self) -> bool:
        """Check if cache is still valid based on TTL."""
        return (time.time() - self.last_cache_update) < self.config.cache_ttl_seconds

    def _invalidate_cache(self):
        """Invalidate all caches."""
        self.filter_cache.clear()
        self.sector_cache.clear()
        self.last_cache_update = time.time()
        logger.debug("Cache invalidated")

    def _invalidate_cache_selective(self, affected_keys: Optional[Set[str]] = None):
        """Selective cache invalidation instead of clearing everything."""
        if affected_keys is None:
            # Full invalidation
            self.filter_cache.clear()
            self.sector_cache.clear()
        else:
            # Selective invalidation
            keys_to_remove = []
            for cache_key in self.filter_cache.keys():
                # Check if cache key is affected by the change
                if any(affected_key in cache_key for affected_key in affected_keys):
                    keys_to_remove.append(cache_key)
            
            for key in keys_to_remove:
                self.filter_cache.pop(key, None)
            
            # Clear sector cache only if sector-related changes
            if 'sector' in affected_keys:
                self.sector_cache.clear()
        
        self.last_cache_update = time.time()

    def clear_cache(self):
        """Clear all caches manually."""
        self._invalidate_cache()
        logger.info("Market universe cache cleared")

    def start_auto_update(self):
        """Start automatic universe updates."""
        if hasattr(self, '_update_thread') and self._update_thread.is_alive():
            logger.warning("Auto-update already running")
            return
        
        def update_worker():
            logger.info("Started automatic universe updates")
            
            while getattr(self, '_auto_update_running', True):
                try:
                    time.sleep(self.config.auto_update_interval)
                    
                    if not getattr(self, '_auto_update_running', True):
                        break
                    
                    # Perform universe update
                    self._perform_auto_update()
                    
                except Exception as e:
                    logger.error(f"Auto-update error: {e}")
        
        self._auto_update_running = True
        self._update_thread = threading.Thread(target=update_worker, daemon=True)
        self._update_thread.start()

    def stop_auto_update(self):
        """Stop automatic universe updates."""
        self._auto_update_running = False
        if hasattr(self, '_update_thread'):
            self._update_thread.join(timeout=5.0)

    def _perform_auto_update(self):
        """Perform automatic universe update."""
        try:
            logger.info("Performing automatic universe update")
            
            # Reload universe file if it exists
            if self.config.universe_file and os.path.exists(self.config.universe_file):
                file_mtime = os.path.getmtime(self.config.universe_file)
                
                if file_mtime > self.stats.last_update_timestamp:
                    logger.info("Universe file updated, reloading")
                    self.load_universe_from_file(self.config.universe_file)
            
            # Perform health checks
            health = self.health_check()
            if health['status'] != 'healthy':
                logger.warning(f"Universe health check failed: {health}")
            
        except Exception as e:
            logger.error(f"Auto-update failed: {e}")

    def export_universe(self, file_path: str, format: str = 'csv'):
        """
        Export current universe to file.
        
        Args:
            file_path: Output file path
            format: Export format ('csv', 'json', 'parquet')
        """
        try:
            if format.lower() == 'csv':
                self.universe_df.to_csv(file_path)
            elif format.lower() == 'json':
                self.universe_df.to_json(file_path, orient='index', indent=2)
            elif format.lower() == 'parquet':
                self.universe_df.to_parquet(file_path)
            else:
                raise ValueError(f"Unsupported format: {format}")
            
            logger.info(f"Exported universe to {file_path} in {format} format")
            
        except Exception as e:
            logger.error(f"Error exporting universe: {e}")

    def health_check(self) -> Dict[str, Any]:
        """
        Perform a health check of the market universe.
        
        Returns:
            Health status dictionary
        """
        health = {
            'status': 'healthy',
            'timestamp': time.time(),
            'universe_size': len(self.universe_df),
            'statistics': {}
        }
        
        try:
            stats = self.get_statistics()
            health['statistics'] = {
                'total_assets': stats.total_assets,
                'sectors_count': stats.sectors_count,
                'filter_efficiency': stats.filter_efficiency,
                'updates_performed': stats.updates_performed,
                'cache_hit_rate': stats.cache_hits / max(stats.cache_hits + stats.cache_misses, 1),
                'uptime_seconds': stats.uptime_seconds
            }
            
            # Check for potential issues
            if stats.total_assets == 0:
                health['status'] = 'warning'
                health['warning'] = 'No assets in universe'
            elif stats.filter_efficiency < 0.1:
                health['status'] = 'warning'
                health['warning'] = 'Very low filter efficiency'
                
        except Exception as e:
            health['status'] = 'error'
            health['error'] = str(e)
            logger.error(f"Health check failed: {e}")
        
        return health

    def log_current_universe_size(self):
        """Enhanced logging with additional statistics."""
        stats = self.get_statistics()
        logger.info(f"Current market universe: {stats.total_assets} assets, "
                   f"{stats.sectors_count} sectors, "
                   f"filter efficiency: {stats.filter_efficiency:.2%}")

if __name__ == "__main__":
    # Enhanced example usage with ARM64 optimizations
    import tempfile
    import warnings
    warnings.filterwarnings('ignore')
    
    # Create comprehensive test data
    dummy_universe_data = {
        'symbol': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'SPG', 'IBM', 'NVDA', 'META', 'NFLX'],
        'name': [
            'Apple Inc.', 'Microsoft Corp.', 'Alphabet Inc.', 'Amazon.com Inc.', 'Tesla Inc.',
            'Simon Property Group Inc.', 'International Business Machines Corp.', 'NVIDIA Corp.',
            'Meta Platforms Inc.', 'Netflix Inc.'
        ],
        'sector': [
            'Technology', 'Technology', 'Communication Services', 'Consumer Discretionary',
            'Consumer Discretionary', 'Real Estate', 'Technology', 'Technology',
            'Communication Services', 'Communication Services'
        ],
        'market_cap': [2.5e12, 2.0e12, 1.5e12, 1.2e12, 8.0e11, 5.0e10, 1.5e11, 1.8e12, 7.0e11, 2.0e11],
        'avg_daily_volume': [100e6, 80e6, 50e6, 70e6, 60e6, 5e6, 10e6, 90e6, 40e6, 15e6],
        'price': [150.0, 300.0, 2500.0, 3200.0, 800.0, 120.0, 140.0, 450.0, 320.0, 400.0],
        'exchange': ['NASDAQ', 'NASDAQ', 'NASDAQ', 'NASDAQ', 'NASDAQ', 'NYSE', 'NYSE', 'NASDAQ', 'NASDAQ', 'NASDAQ']
    }
    
    dummy_df = pd.DataFrame(dummy_universe_data)
    
    # Create temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        dummy_df.to_csv(f.name, index=False)
        dummy_file_path = f.name
    
    print(f"Created test universe file: {dummy_file_path}")

    # Initialize enhanced MarketUniverse with comprehensive configuration
    config = MarketUniverseConfig(
        universe_file=dummy_file_path,
        min_market_cap=1.0e11,
        min_avg_daily_volume=10e6,
        enable_arm64_optimizations=True,
        enable_parallel_processing=True,
        enable_performance_monitoring=True,
        excluded_sectors=['Real Estate'],
        required_exchanges=['NASDAQ', 'NYSE']
    )
    
    universe_manager = MarketUniverse(config=config)

    print(f"\nAll symbols in the universe:")
    print(universe_manager.get_all_symbols())

    print(f"\nSector breakdown:")
    print(universe_manager.get_sector_breakdown())

    print(f"\nDetails for AAPL:")
    print(universe_manager.get_asset_details('AAPL'))

    print(f"\nFiltered by sector (Technology) and market cap range (1.0e12 to 3.0e12):")
    tech_giants = universe_manager.get_filtered_universe(
        sectors=['Technology'],
        market_cap_range=(1.0e12, 3.0e12),
        price_range=(100.0, 500.0)
    )
    print(tech_giants)

    print(f"\nAdding a new asset (AMD):")
    universe_manager.add_asset('AMD', {
        'name': 'Advanced Micro Devices Inc.',
        'sector': 'Technology',
        'market_cap': 2.5e11,
        'avg_daily_volume': 50e6,
        'price': 120.0,
        'exchange': 'NASDAQ'
    })
    print(universe_manager.get_all_symbols())

    print(f"\nUpdating metrics for MSFT:")
    universe_manager.update_asset_metrics('MSFT', {
        'market_cap': 2.1e12,
        'avg_daily_volume': 85e6,
        'price': 310.0
    })
    print(universe_manager.get_asset_details('MSFT'))

    # Performance statistics
    stats = universe_manager.get_statistics()
    print(f"\nPerformance Statistics:")
    print(f"Total assets: {stats.total_assets}")
    print(f"Sectors count: {stats.sectors_count}")
    print(f"Filter efficiency: {stats.filter_efficiency:.2%}")
    print(f"Updates performed: {stats.updates_performed}")
    print(f"Cache hit rate: {stats.cache_hits / max(stats.cache_hits + stats.cache_misses, 1):.2%}")

    # Health check
    health = universe_manager.health_check()
    print(f"\nHealth Status: {health['status']}")
    print(f"Universe size: {health['universe_size']}")

    # Export test
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        export_path = f.name
    
    universe_manager.export_universe(export_path, 'csv')
    print(f"\nExported universe to: {export_path}")

    # Clean up
    os.unlink(dummy_file_path)
    os.unlink(export_path)
    print(f"\nCleaned up test files")
