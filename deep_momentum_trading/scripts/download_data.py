#!/usr/bin/env python3
"""
Enhanced Data Download Script with ARM64 Optimizations

This script provides comprehensive data downloading capabilities for the Deep Momentum Trading System
with ARM64-specific optimizations, parallel processing, and robust data management features.

Features:
- Polygon.io data downloading with high-frequency support
- ARM64-optimized parallel processing
- Incremental data updates and synchronization
- Data validation and quality checks
- Automatic retry mechanisms and error handling
- Progress monitoring and reporting
- Data compression and storage optimization
"""

import os
import sys
import argparse
import asyncio
import aiohttp
import time
import platform
import multiprocessing as mp
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import numpy as np
import logging
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import requests
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.polygon_client import PolygonClient
from src.data.data_manager import DataManager
from src.storage.hdf5_storage import HDF5Storage
from src.storage.parquet_storage import ParquetStorage
from src.utils.logger import get_logger
from src.utils.decorators import performance_monitor, error_handler
from src.utils.exceptions import DataDownloadError
from config.settings import config_manager

logger = get_logger(__name__)

class ARM64DownloadOptimizer:
    """ARM64-specific optimizations for data downloading"""
    
    def __init__(self):
        self.is_arm64 = platform.machine().lower() in ['arm64', 'aarch64']
        self.cpu_count = mp.cpu_count()
        self.optimal_workers = self._calculate_optimal_workers()
        self.optimal_batch_size = self._calculate_optimal_batch_size()
        
    def _calculate_optimal_workers(self) -> int:
        """Calculate optimal number of workers for ARM64"""
        if self.is_arm64:
            # ARM64 benefits from moderate parallelism for I/O operations
            return min(self.cpu_count, 16)
        return min(self.cpu_count, 20)
    
    def _calculate_optimal_batch_size(self) -> int:
        """Calculate optimal batch size for ARM64"""
        if self.is_arm64:
            # ARM64 cache-friendly batch sizes
            return 100
        return 200

class DataSource:
    """Data source configuration"""
    
    def __init__(self, name: str, **kwargs):
        self.name = name
        self.enabled = kwargs.get('enabled', True)
        self.api_key = kwargs.get('api_key')
        self.base_url = kwargs.get('base_url')
        self.rate_limit = kwargs.get('rate_limit', 5)  # requests per second
        self.timeout = kwargs.get('timeout', 30)
        self.retry_attempts = kwargs.get('retry_attempts', 3)
        self.retry_delay = kwargs.get('retry_delay', 1.0)

class DownloadConfig:
    """Data download configuration"""
    
    def __init__(self, **kwargs):
        # Time period
        self.start_date = kwargs.get('start_date', '2020-01-01')
        self.end_date = kwargs.get('end_date', datetime.now().strftime('%Y-%m-%d'))
        
        # Symbols and markets
        self.symbols = kwargs.get('symbols', ['SPY', 'QQQ', 'IWM'])
        self.markets = kwargs.get('markets', ['stocks', 'etfs'])
        self.exchanges = kwargs.get('exchanges', ['NYSE', 'NASDAQ'])
        
        # Data types - Only historical bars data supported
        self.data_types = kwargs.get('data_types', ['bars'])
        self.timeframes = kwargs.get('timeframes', ['1min', '5min', '1hour', '1day'])
        
        # Sources - Only Polygon.io supported
        self.sources = ['polygon']
        self.primary_source = 'polygon'
        
        # Processing
        self.parallel_processing = kwargs.get('parallel_processing', True)
        self.batch_size = kwargs.get('batch_size', 100)
        self.max_workers = kwargs.get('max_workers', None)
        
        # Storage
        self.storage_format = kwargs.get('storage_format', 'parquet')  # parquet, hdf5, csv
        self.compression = kwargs.get('compression', 'snappy')
        self.output_dir = kwargs.get('output_dir', 'data/raw')
        
        # Quality control
        self.validate_data = kwargs.get('validate_data', True)
        self.remove_duplicates = kwargs.get('remove_duplicates', True)
        self.fill_missing = kwargs.get('fill_missing', True)
        
        # Incremental updates
        self.incremental = kwargs.get('incremental', True)
        self.update_existing = kwargs.get('update_existing', True)

class DataDownloadEngine:
    """
    Enhanced data download engine with ARM64 optimizations - Polygon.io only
    """
    
    def __init__(self, config: DownloadConfig):
        self.config = config
        self.optimizer = ARM64DownloadOptimizer()
        
        # Initialize components
        self.data_manager = DataManager()
        self.polygon_client = PolygonClient()
        
        # Storage backends
        self.storage_backends = self._initialize_storage_backends()
        
        # Data sources - Only Polygon.io
        self.data_sources = self._initialize_data_sources()
        
        # Download state
        self.download_stats = {
            'total_symbols': 0,
            'downloaded_symbols': 0,
            'failed_symbols': 0,
            'total_records': 0,
            'start_time': None,
            'end_time': None
        }
        
        logger.info(f"DataDownloadEngine initialized with ARM64 optimizations: {self.optimizer.is_arm64}")
        logger.info("Using Polygon.io as the exclusive data source")
    
    def _initialize_storage_backends(self) -> Dict[str, Any]:
        """Initialize storage backends"""
        backends = {}
        
        if self.config.storage_format == 'parquet':
            backends['parquet'] = ParquetStorage(
                base_path=self.config.output_dir,
                compression=self.config.compression
            )
        elif self.config.storage_format == 'hdf5':
            backends['hdf5'] = HDF5Storage(
                base_path=self.config.output_dir,
                compression=self.config.compression
            )
        
        return backends
    
    def _initialize_data_sources(self) -> Dict[str, DataSource]:
        """Initialize data sources - Polygon.io only"""
        sources = {}
        
        # Polygon.io - Primary and only source
        polygon_api_key = os.getenv('POLYGON_API_KEY')
        if not polygon_api_key:
            raise ValueError("POLYGON_API_KEY environment variable is required")
            
        sources['polygon'] = DataSource(
            name='polygon',
            api_key=polygon_api_key,
            base_url='https://api.polygon.io',
            rate_limit=5,
            timeout=30
        )
        
        return sources
    
    @performance_monitor
    @error_handler
    async def download_data(self) -> Dict[str, Any]:
        """
        Download data with comprehensive pipeline
        
        Returns:
            Dict containing download results
        """
        logger.info("Starting enhanced data download from Polygon.io...")
        self.download_stats['start_time'] = time.time()
        
        try:
            # Prepare symbol list
            symbols = await self._prepare_symbol_list()
            self.download_stats['total_symbols'] = len(symbols)
            
            # Download data
            if self.config.parallel_processing:
                results = await self._download_parallel(symbols)
            else:
                results = await self._download_sequential(symbols)
            
            # Post-process and validate
            processed_results = await self._post_process_data(results)
            
            # Generate summary
            summary = self._generate_download_summary(processed_results)
            
            self.download_stats['end_time'] = time.time()
            execution_time = self.download_stats['end_time'] - self.download_stats['start_time']
            
            logger.info(f"Data download completed in {execution_time:.2f} seconds")
            
            return {
                'results': processed_results,
                'summary': summary,
                'stats': self.download_stats,
                'execution_time': execution_time
            }
            
        except Exception as e:
            logger.error(f"Data download failed: {e}")
            raise DataDownloadError(f"Data download execution failed: {e}")
    
    async def _prepare_symbol_list(self) -> List[str]:
        """Prepare and validate symbol list"""
        logger.info("Preparing symbol list...")
        
        symbols = self.config.symbols.copy()
        
        # Expand symbol list if needed
        if 'ALL_SP500' in symbols:
            sp500_symbols = await self._get_sp500_symbols()
            symbols.extend(sp500_symbols)
            symbols.remove('ALL_SP500')
        
        if 'ALL_NASDAQ100' in symbols:
            nasdaq100_symbols = await self._get_nasdaq100_symbols()
            symbols.extend(nasdaq100_symbols)
            symbols.remove('ALL_NASDAQ100')
        
        # Remove duplicates and validate
        symbols = list(set(symbols))
        validated_symbols = await self._validate_symbols(symbols)
        
        logger.info(f"Prepared {len(validated_symbols)} symbols for download")
        return validated_symbols
    
    async def _get_sp500_symbols(self) -> List[str]:
        """Get S&P 500 symbols"""
        try:
            # Download S&P 500 list from Wikipedia
            url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
            tables = pd.read_html(url)
            sp500_table = tables[0]
            symbols = sp500_table['Symbol'].tolist()
            
            # Clean symbols
            symbols = [symbol.replace('.', '-') for symbol in symbols]
            
            logger.info(f"Retrieved {len(symbols)} S&P 500 symbols")
            return symbols
            
        except Exception as e:
            logger.warning(f"Failed to retrieve S&P 500 symbols: {e}")
            return []
    
    async def _get_nasdaq100_symbols(self) -> List[str]:
        """Get NASDAQ 100 symbols"""
        try:
            # Download NASDAQ 100 list
            url = "https://en.wikipedia.org/wiki/NASDAQ-100"
            tables = pd.read_html(url)
            nasdaq100_table = tables[4]  # Usually the 5th table
            symbols = nasdaq100_table['Ticker'].tolist()
            
            logger.info(f"Retrieved {len(symbols)} NASDAQ 100 symbols")
            return symbols
            
        except Exception as e:
            logger.warning(f"Failed to retrieve NASDAQ 100 symbols: {e}")
            return []
    
    async def _validate_symbols(self, symbols: List[str]) -> List[str]:
        """Validate symbol list"""
        logger.info("Validating symbols...")
        
        valid_symbols = []
        
        # Basic validation
        for symbol in symbols:
            if len(symbol) <= 5 and symbol.isalpha():
                valid_symbols.append(symbol.upper())
            else:
                logger.warning(f"Invalid symbol format: {symbol}")
        
        # Additional validation against Polygon API
        validated_symbols = await self._validate_symbols_polygon(valid_symbols)
        
        logger.info(f"Validated {len(validated_symbols)} symbols")
        return validated_symbols
    
    async def _validate_symbols_polygon(self, symbols: List[str]) -> List[str]:
        """Validate symbols against Polygon API"""
        validated = []
        
        for symbol in symbols:
            try:
                # Check if symbol exists
                ticker_info = await self.polygon_client.get_ticker_details(symbol)
                if ticker_info:
                    validated.append(symbol)
            except Exception as e:
                logger.debug(f"Symbol {symbol} validation failed: {e}")
        
        return validated
    
    async def _download_parallel(self, symbols: List[str]) -> Dict[str, Any]:
        """Download data in parallel"""
        logger.info(f"Downloading data for {len(symbols)} symbols in parallel...")
        
        # Calculate optimal batch size
        batch_size = self.config.batch_size or self.optimizer.optimal_batch_size
        max_workers = self.config.max_workers or self.optimizer.optimal_workers
        
        # Split symbols into batches
        symbol_batches = [symbols[i:i + batch_size] for i in range(0, len(symbols), batch_size)]
        
        results = {}
        
        # Process batches in parallel
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            batch_futures = []
            
            for batch in symbol_batches:
                future = executor.submit(self._download_batch, batch)
                batch_futures.append(future)
            
            # Collect results with progress bar
            with tqdm(total=len(symbol_batches), desc="Downloading batches") as pbar:
                for future in batch_futures:
                    batch_result = future.result()
                    results.update(batch_result)
                    pbar.update(1)
        
        return results
    
    async def _download_sequential(self, symbols: List[str]) -> Dict[str, Any]:
        """Download data sequentially"""
        logger.info(f"Downloading data for {len(symbols)} symbols sequentially...")
        
        results = {}
        
        with tqdm(total=len(symbols), desc="Downloading symbols") as pbar:
            for symbol in symbols:
                try:
                    symbol_data = await self._download_symbol_data(symbol)
                    results[symbol] = symbol_data
                    self.download_stats['downloaded_symbols'] += 1
                except Exception as e:
                    logger.error(f"Failed to download data for {symbol}: {e}")
                    self.download_stats['failed_symbols'] += 1
                
                pbar.update(1)
        
        return results
    
    def _download_batch(self, symbols: List[str]) -> Dict[str, Any]:
        """Download batch of symbols (synchronous for thread pool)"""
        batch_results = {}
        
        for symbol in symbols:
            try:
                # Use asyncio.run for each symbol in the thread
                symbol_data = asyncio.run(self._download_symbol_data(symbol))
                batch_results[symbol] = symbol_data
                self.download_stats['downloaded_symbols'] += 1
            except Exception as e:
                logger.error(f"Failed to download data for {symbol}: {e}")
                self.download_stats['failed_symbols'] += 1
        
        return batch_results
    
    async def _download_symbol_data(self, symbol: str) -> Dict[str, pd.DataFrame]:
        """Download data for a single symbol from Polygon.io"""
        symbol_data = {}
        
        try:
            symbol_data = await self._download_from_polygon(symbol)
        except Exception as e:
            logger.error(f"Failed to download data for {symbol} from Polygon.io: {e}")
            raise DataDownloadError(f"Polygon.io download failed for symbol {symbol}: {e}")
        
        if not symbol_data:
            raise DataDownloadError(f"No data retrieved for symbol {symbol}")
        
        return symbol_data
    
    async def _download_from_polygon(self, symbol: str) -> Dict[str, pd.DataFrame]:
        """Download data from Polygon.io"""
        data = {}
        
        for timeframe in self.config.timeframes:
            try:
                # Download bars data
                if 'bars' in self.config.data_types:
                    bars = await self.polygon_client.get_historical_aggregates(
                        symbol=symbol,
                        timespan=timeframe.replace('min', 'minute').replace('hour', 'hour').replace('day', 'day'),
                        multiplier=1,
                        from_date=self.config.start_date,
                        to_date=self.config.end_date
                    )
                    if bars is not None and not bars.empty:
                        data[f'bars_{timeframe}'] = bars
                
                
            except Exception as e:
                logger.warning(f"Failed to download {timeframe} data for {symbol} from Polygon: {e}")
        
        return data
    
    async def _post_process_data(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Post-process downloaded data"""
        logger.info("Post-processing downloaded data...")
        
        processed_results = {}
        
        for symbol, symbol_data in results.items():
            try:
                processed_symbol_data = {}
                
                for data_key, df in symbol_data.items():
                    # Data validation
                    if self.config.validate_data:
                        df = self._validate_dataframe(df, symbol, data_key)
                    
                    # Remove duplicates
                    if self.config.remove_duplicates:
                        df = df.drop_duplicates()
                    
                    # Fill missing values
                    if self.config.fill_missing:
                        df = self._fill_missing_values(df)
                    
                    # Store processed data
                    processed_symbol_data[data_key] = df
                    
                    # Update record count
                    self.download_stats['total_records'] += len(df)
                
                processed_results[symbol] = processed_symbol_data
                
                # Save to storage
                await self._save_symbol_data(symbol, processed_symbol_data)
                
            except Exception as e:
                logger.error(f"Post-processing failed for {symbol}: {e}")
        
        logger.info("Post-processing completed")
        return processed_results
    
    def _validate_dataframe(self, df: pd.DataFrame, symbol: str, data_key: str) -> pd.DataFrame:
        """Validate dataframe quality"""
        if df.empty:
            logger.warning(f"Empty dataframe for {symbol}:{data_key}")
            return df
        
        # Check for required columns based on data type
        if 'bars' in data_key:
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                logger.warning(f"Missing columns for {symbol}:{data_key}: {missing_cols}")
        
        # Check for reasonable price ranges
        if 'bars' in data_key and 'close' in df.columns:
            close_prices = df['close']
            if close_prices.min() <= 0:
                logger.warning(f"Invalid prices detected for {symbol}:{data_key}")
                df = df[df['close'] > 0]
        
        # Check for reasonable volume
        if 'volume' in df.columns:
            if (df['volume'] < 0).any():
                logger.warning(f"Negative volume detected for {symbol}:{data_key}")
                df = df[df['volume'] >= 0]
        
        return df
    
    def _fill_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fill missing values in dataframe"""
        if df.empty:
            return df
        
        # Forward fill for price data
        price_cols = ['open', 'high', 'low', 'close']
        for col in price_cols:
            if col in df.columns:
                df[col] = df[col].fillna(method='ffill')
        
        # Fill volume with 0
        if 'volume' in df.columns:
            df['volume'] = df['volume'].fillna(0)
        
        return df
    
    async def _save_symbol_data(self, symbol: str, symbol_data: Dict[str, pd.DataFrame]):
        """Save symbol data to storage"""
        try:
            for data_key, df in symbol_data.items():
                if df.empty:
                    continue
                
                # Determine storage path
                storage_path = f"{symbol}/{data_key}"
                
                # Save to primary storage backend
                if self.config.storage_format == 'parquet' and 'parquet' in self.storage_backends:
                    await self.storage_backends['parquet'].save_dataframe(df, storage_path)
                elif self.config.storage_format == 'hdf5' and 'hdf5' in self.storage_backends:
                    await self.storage_backends['hdf5'].save_dataframe(df, storage_path)
                else:
                    # Fallback to CSV
                    output_dir = Path(self.config.output_dir) / symbol
                    output_dir.mkdir(parents=True, exist_ok=True)
                    csv_path = output_dir / f"{data_key}.csv"
                    df.to_csv(csv_path, index=True)
        
        except Exception as e:
            logger.error(f"Failed to save data for {symbol}: {e}")
    
    def _generate_download_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate download summary"""
        summary = {
            'total_symbols_requested': self.download_stats['total_symbols'],
            'successful_downloads': self.download_stats['downloaded_symbols'],
            'failed_downloads': self.download_stats['failed_symbols'],
            'success_rate': (
                self.download_stats['downloaded_symbols'] / 
                max(1, self.download_stats['total_symbols'])
            ),
            'total_records': self.download_stats['total_records'],
            'data_types_downloaded': {},
            'timeframes_downloaded': {},
            'storage_info': {
                'format': self.config.storage_format,
                'compression': self.config.compression,
                'output_directory': self.config.output_dir
            },
            'data_source': 'Polygon.io'
        }
        
        # Analyze data types and timeframes
        for symbol, symbol_data in results.items():
            for data_key in symbol_data.keys():
                # Count data types
                data_type = data_key.split('_')[0]
                if data_type not in summary['data_types_downloaded']:
                    summary['data_types_downloaded'][data_type] = 0
                summary['data_types_downloaded'][data_type] += 1
                
                # Count timeframes
                if '_' in data_key:
                    timeframe = data_key.split('_', 1)[1]
                    if timeframe not in summary['timeframes_downloaded']:
                        summary['timeframes_downloaded'][timeframe] = 0
                    summary['timeframes_downloaded'][timeframe] += 1
        
        return summary

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Enhanced Data Download Script - Polygon.io Only')
    
    # Time period
    parser.add_argument('--start-date', default='2020-01-01', help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', help='End date (YYYY-MM-DD), defaults to today')
    
    # Symbols
    parser.add_argument('--symbols', nargs='+', default=['SPY', 'QQQ', 'IWM'], 
                       help='Symbols to download')
    parser.add_argument('--symbol-file', help='File containing symbols (one per line)')
    parser.add_argument('--include-sp500', action='store_true', help='Include all S&P 500 symbols')
    parser.add_argument('--include-nasdaq100', action='store_true', help='Include all NASDAQ 100 symbols')
    
    # Data types
    parser.add_argument('--data-types', nargs='+', default=['bars'], 
                       choices=['bars'], help='Data types to download (only historical bars supported)')
    parser.add_argument('--timeframes', nargs='+', default=['1day'], 
                       choices=['1min', '5min', '15min', '30min', '1hour', '1day'], 
                       help='Timeframes to download')
    
    # Processing
    parser.add_argument('--parallel', action='store_true', help='Enable parallel processing')
    parser.add_argument('--batch-size', type=int, help='Batch size for parallel processing')
    parser.add_argument('--max-workers', type=int, help='Maximum number of workers')
    
    # Storage
    parser.add_argument('--storage-format', choices=['parquet', 'hdf5', 'csv'], 
                       default='parquet', help='Storage format')
    parser.add_argument('--compression', choices=['snappy', 'gzip', 'lz4'], 
                       default='snappy', help='Compression algorithm')
    parser.add_argument('--output-dir', default='data/raw', help='Output directory')
    
    # Options
    parser.add_argument('--incremental', action='store_true', help='Incremental download')
    parser.add_argument('--validate', action='store_true', help='Validate downloaded data')
    parser.add_argument('--verbose', action='store_true', help='Verbose logging')
    
    return parser.parse_args()

async def main():
    """Main download function"""
    args = parse_arguments()
    
    # Setup logging
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Prepare symbols
    symbols = args.symbols.copy()
    
    if args.symbol_file:
        with open(args.symbol_file, 'r') as f:
            file_symbols = [line.strip().upper() for line in f if line.strip()]
        symbols.extend(file_symbols)
    
    if args.include_sp500:
        symbols.append('ALL_SP500')
    
    if args.include_nasdaq100:
        symbols.append('ALL_NASDAQ100')
    
    # Create configuration
    config = DownloadConfig(
        start_date=args.start_date,
        end_date=args.end_date or datetime.now().strftime('%Y-%m-%d'),
        symbols=symbols,
        data_types=args.data_types,
        timeframes=args.timeframes,
        parallel_processing=args.parallel,
        batch_size=args.batch_size,
        max_workers=args.max_workers,
        storage_format=args.storage_format,
        compression=args.compression,
        output_dir=args.output_dir,
        incremental=args.incremental,
        validate_data=args.validate
    )
    
    # Initialize and run download
    engine = DataDownloadEngine(config)
    
    try:
        result = await engine.download_data()
        
        # Print summary
        summary = result['summary']
        print(f"\n=== Download Results ===")
        print(f"Data Source: {summary['data_source']}")
        print(f"Symbols Requested: {summary['total_symbols_requested']}")
        print(f"Successful Downloads: {summary['successful_downloads']}")
        print(f"Failed Downloads: {summary['failed_downloads']}")
        print(f"Success Rate: {summary['success_rate']:.1%}")
        print(f"Total Records: {summary['total_records']:,}")
        print(f"Execution Time: {result['execution_time']:.2f}s")
        print(f"Storage Format: {summary['storage_info']['format']}")
        print(f"Output Directory: {summary['storage_info']['output_directory']}")
        
        if summary['data_types_downloaded']:
            print(f"Data Types: {', '.join(summary['data_types_downloaded'].keys())}")
        
        if summary['timeframes_downloaded']:
            print(f"Timeframes: {', '.join(summary['timeframes_downloaded'].keys())}")
        
    except Exception as e:
        logger.error(f"Data download failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
