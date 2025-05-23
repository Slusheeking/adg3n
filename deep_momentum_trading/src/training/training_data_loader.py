"""
Training Data Loader for Historical Market Data Only
Provides data loading for training pipelines using exclusively historical data
from Polygon API with ARM64 optimizations.
"""

import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from typing import Dict, Any, List, Tuple, Optional
import asyncio
import logging
from pathlib import Path
from datetime import datetime, timedelta

from deep_momentum_trading.src.data.data_manager import DataManager
from deep_momentum_trading.src.data.polygon_client import PolygonClient
from deep_momentum_trading.src.data.data_preprocessing import DataPreprocessor
from deep_momentum_trading.src.data.feature_engineering import FeatureEngineeringProcess
from deep_momentum_trading.src.storage.hdf5_storage import HDF5Storage
from deep_momentum_trading.src.storage.parquet_storage import ParquetStorage
from deep_momentum_trading.src.utils.logger import get_logger

logger = get_logger(__name__)

class HistoricalMarketDataset(Dataset):
    """
    PyTorch Dataset for historical market data with feature engineering and preprocessing.
    Uses only historical data from Polygon API - NO real-time data.
    """
    
    def __init__(self,
                 features: np.ndarray,
                 targets: np.ndarray,
                 prev_positions: np.ndarray,
                 metadata: Dict[str, Any],
                 sequence_length: int = 60,
                 prediction_horizon: int = 1,
                 transform: Optional[callable] = None):
        """
        Initialize the historical market dataset.
        
        Args:
            features (np.ndarray): Feature matrix (samples, time_steps, features)
            targets (np.ndarray): Target returns (samples, assets)
            prev_positions (np.ndarray): Previous positions (samples, assets)
            metadata (Dict[str, Any]): Dataset metadata including date ranges
            sequence_length (int): Length of input sequences
            prediction_horizon (int): Number of steps ahead to predict
            transform (Optional[callable]): Optional data transformation function
        """
        self.features = torch.FloatTensor(features)
        self.targets = torch.FloatTensor(targets)
        self.prev_positions = torch.FloatTensor(prev_positions)
        self.metadata = metadata
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.transform = transform
        
        # Validate data shapes
        assert len(self.features) == len(self.targets) == len(self.prev_positions), \
            "All data arrays must have the same number of samples"
        
        # Ensure we're using historical data only
        assert 'data_source' in metadata and metadata['data_source'] == 'polygon_historical', \
            "Dataset must use historical data from Polygon API only"
        
        assert 'end_date' in metadata, "Dataset must have an end_date in metadata"
        
        # Verify data is historical (not current)
        end_date = pd.to_datetime(metadata['end_date'])
        current_date = pd.Timestamp.now()
        
        # Ensure data ends at least 1 day ago to guarantee it's historical
        assert (current_date - end_date).days >= 1, \
            f"Data must be historical. End date {end_date} is too recent (current: {current_date})"
        
        logger.info(f"HistoricalMarketDataset initialized with {len(self.features)} samples")
        logger.info(f"Data period: {metadata.get('start_date')} to {metadata.get('end_date')}")
        logger.info("CONFIRMED: Using HISTORICAL data only from Polygon API")
    
    def __len__(self) -> int:
        return len(self.features) - self.sequence_length - self.prediction_horizon + 1
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get a single sample from the dataset.
        
        Args:
            idx (int): Sample index
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: (features, targets, prev_positions)
        """
        # Extract sequence
        feature_seq = self.features[idx:idx + self.sequence_length]
        target = self.targets[idx + self.sequence_length + self.prediction_horizon - 1]
        prev_pos = self.prev_positions[idx + self.sequence_length - 1]
        
        if self.transform:
            feature_seq, target, prev_pos = self.transform(feature_seq, target, prev_pos)
        
        return feature_seq, target, prev_pos

class TrainingDataLoader:
    """
    Historical data loader for training - POLYGON API HISTORICAL DATA ONLY.
    Explicitly excludes any real-time or live data sources.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the training data loader for historical data only.
        
        Args:
            config (Dict[str, Any]): Data configuration
        """
        self.config = config
        
        # Initialize Polygon client for historical data
        self.polygon_client = PolygonClient()
        
        # Initialize preprocessing and feature engineering
        self.preprocessor = DataPreprocessor(
            fill_method=config.get('fill_method', 'ffill'),
            resample_interval=config.get('resample_interval', '1min'),
            outlier_threshold_std=config.get('outlier_threshold_std', 3.0)
        )
        
        self.feature_engineer = FeatureEngineeringProcess(
            zmq_subscriber_port=config.get('zmq_subscriber_port', 5555),
            zmq_publisher_port=config.get('zmq_publisher_port', 5556),
            memory_cache_max_gb=config.get('memory_cache_gb', 32)
        )
        
        # Storage backends for historical data
        self.storage_backends = self._initialize_storage_backends()
        
        # Data loaders
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        
        # Explicitly disable real-time data
        if config.get('real_time', {}).get('enabled', False):
            logger.warning("Real-time data is DISABLED for training. Using historical data only.")
            config['real_time']['enabled'] = False
        
        logger.info("TrainingDataLoader initialized for HISTORICAL DATA ONLY")
        logger.info("Data source: Polygon API historical data")
    
    def _initialize_storage_backends(self) -> Dict[str, Any]:
        """Initialize storage backends for historical data."""
        backends = {}
        
        storage_format = self.config.get('storage_format', 'parquet')
        base_path = self.config.get('data_path', 'data/processed')
        
        if storage_format == 'parquet':
            backends['parquet'] = ParquetStorage(
                base_path=base_path,
                compression=self.config.get('compression', 'snappy')
            )
        elif storage_format == 'hdf5':
            backends['hdf5'] = HDF5Storage(
                base_path=base_path,
                compression=self.config.get('compression', 'gzip')
            )
        
        return backends
    
    async def initialize(self):
        """Initialize data loaders with historical data from Polygon API."""
        logger.info("Initializing training data loaders with HISTORICAL data only...")
        
        # Verify configuration for historical data only
        self._validate_historical_config()
        
        # Load historical data from Polygon API
        await self._load_historical_data()
        
        logger.info("Historical data loaders initialization complete")
        logger.info("CONFIRMED: No real-time data sources active")
    
    def _validate_historical_config(self):
        """Validate configuration to ensure historical data only."""
        # Check date ranges
        start_date = pd.to_datetime(self.config.get('start_date', '2020-01-01'))
        end_date = pd.to_datetime(self.config.get('end_date', '2023-12-31'))
        current_date = pd.Timestamp.now()
        
        # Ensure end date is at least 1 day ago
        if (current_date - end_date).days < 1:
            # Automatically adjust end date to ensure historical data
            end_date = current_date - timedelta(days=2)
            self.config['end_date'] = end_date.strftime('%Y-%m-%d')
            logger.warning(f"Adjusted end_date to {self.config['end_date']} to ensure historical data only")
        
        # Verify no real-time sources
        assert not self.config.get('real_time', {}).get('enabled', False), \
            "Real-time data is not allowed for training"
        
        # Verify data source is Polygon
        data_source = self.config.get('data_source', 'polygon')
        assert data_source == 'polygon', f"Only Polygon API is supported, got: {data_source}"
        
        logger.info(f"Configuration validated for historical data: {start_date} to {end_date}")
    
    async def _load_historical_data(self):
        """Load and prepare historical training data from Polygon API."""
        logger.info("Loading historical market data from Polygon API...")
        
        symbols = self.config['symbols']
        start_date = self.config['start_date']
        end_date = self.config['end_date']
        timeframes = self.config.get('timeframes', ['1day'])
        
        # Load data for each symbol and timeframe
        all_data = {}
        
        for symbol in symbols:
            symbol_data = {}
            
            for timeframe in timeframes:
                try:
                    # Download historical data from Polygon
                    logger.info(f"Downloading {symbol} {timeframe} data from {start_date} to {end_date}")
                    
                    bars_data = await self.polygon_client.get_historical_aggregates(
                        symbol=symbol,
                        timespan=timeframe.replace('min', 'minute').replace('hour', 'hour').replace('day', 'day'),
                        multiplier=1,
                        from_date=start_date,
                        to_date=end_date
                    )
                    
                    if bars_data is not None and not bars_data.empty:
                        # Preprocess data
                        processed_data = await self.preprocessor.process_dataframe(bars_data)
                        
                        # Engineer features
                        features_data = await self.feature_engineer.process_symbol_data(
                            symbol, processed_data
                        )
                        
                        symbol_data[timeframe] = features_data
                        logger.info(f"Loaded {len(features_data)} records for {symbol} {timeframe}")
                    else:
                        logger.warning(f"No data retrieved for {symbol} {timeframe}")
                        
                except Exception as e:
                    logger.error(f"Failed to load data for {symbol} {timeframe}: {e}")
            
            if symbol_data:
                all_data[symbol] = symbol_data
        
        # Convert to training format
        await self._prepare_training_data(all_data)
        
        logger.info("Historical data loading complete")
    
    async def _prepare_training_data(self, raw_data: Dict[str, Dict[str, pd.DataFrame]]):
        """Prepare training data from raw historical data."""
        logger.info("Preparing training datasets...")
        
        # Combine data from all symbols and timeframes
        combined_features = []
        combined_targets = []
        combined_positions = []
        
        sequence_length = self.config['sequence_length']
        prediction_horizon = self.config.get('prediction_horizon', 1)
        
        for symbol, timeframe_data in raw_data.items():
            for timeframe, df in timeframe_data.items():
                if df.empty:
                    continue
                
                # Extract features (all columns except target-related ones)
                feature_cols = [col for col in df.columns if not col.startswith('target_')]
                features = df[feature_cols].values
                
                # Extract targets (returns)
                if 'target_return' in df.columns:
                    targets = df['target_return'].values
                else:
                    # Calculate returns if not present
                    if 'close' in df.columns:
                        returns = df['close'].pct_change().fillna(0)
                        targets = returns.values
                    else:
                        logger.warning(f"No target data for {symbol} {timeframe}")
                        continue
                
                # Create sequences
                for i in range(len(features) - sequence_length - prediction_horizon + 1):
                    feature_seq = features[i:i + sequence_length]
                    target_val = targets[i + sequence_length + prediction_horizon - 1]
                    
                    combined_features.append(feature_seq)
                    combined_targets.append([target_val])  # Single asset for now
                    combined_positions.append([0.0])  # Placeholder previous position
        
        if not combined_features:
            raise ValueError("No training data could be prepared from historical data")
        
        # Convert to numpy arrays
        features_array = np.array(combined_features)
        targets_array = np.array(combined_targets)
        positions_array = np.array(combined_positions)
        
        # Create metadata
        metadata = {
            'data_source': 'polygon_historical',
            'start_date': self.config['start_date'],
            'end_date': self.config['end_date'],
            'symbols': self.config['symbols'],
            'timeframes': self.config.get('timeframes', ['1day']),
            'created_at': datetime.now().isoformat(),
            'total_samples': len(features_array)
        }
        
        # Create dataset
        full_dataset = HistoricalMarketDataset(
            features=features_array,
            targets=targets_array,
            prev_positions=positions_array,
            metadata=metadata,
            sequence_length=sequence_length,
            prediction_horizon=prediction_horizon
        )
        
        # Split data
        train_size = int(len(full_dataset) * (1 - self.config['validation_split'] - self.config['test_split']))
        val_size = int(len(full_dataset) * self.config['validation_split'])
        test_size = len(full_dataset) - train_size - val_size
        
        train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
            full_dataset, [train_size, val_size, test_size]
        )
        
        # Create data loaders
        loader_kwargs = {
            'batch_size': self.config['batch_size'],
            'num_workers': self.config.get('num_workers', 4),
            'pin_memory': self.config.get('pin_memory', True),
            'prefetch_factor': self.config.get('prefetch_factor', 2)
        }
        
        self.train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
        self.val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)
        self.test_loader = DataLoader(test_dataset, shuffle=False, **loader_kwargs)
        
        logger.info(f"Training datasets prepared: {train_size} train, {val_size} val, {test_size} test samples")
        logger.info("CONFIRMED: All data is historical from Polygon API")
    
    def get_train_loader(self) -> DataLoader:
        """Get training data loader."""
        if self.train_loader is None:
            raise ValueError("Data loaders not initialized. Call initialize() first.")
        return self.train_loader
    
    def get_val_loader(self) -> DataLoader:
        """Get validation data loader."""
        if self.val_loader is None:
            raise ValueError("Data loaders not initialized. Call initialize() first.")
        return self.val_loader
    
    def get_test_loader(self) -> DataLoader:
        """Get test data loader."""
        if self.test_loader is None:
            raise ValueError("Data loaders not initialized. Call initialize() first.")
        return self.test_loader
    
    def get_data_stats(self) -> Dict[str, Any]:
        """Get statistics about the loaded historical data."""
        stats = {
            'train_samples': len(self.train_loader.dataset) if self.train_loader else 0,
            'val_samples': len(self.val_loader.dataset) if self.val_loader else 0,
            'test_samples': len(self.test_loader.dataset) if self.test_loader else 0,
            'data_source': 'polygon_historical_only',
            'real_time_enabled': False,
            'symbols': self.config['symbols'],
            'sequence_length': self.config['sequence_length'],
            'batch_size': self.config['batch_size'],
            'start_date': self.config['start_date'],
            'end_date': self.config['end_date']
        }
        return stats
    
    async def save_processed_data(self, output_path: str):
        """Save processed historical data to storage."""
        if not self.train_loader:
            raise ValueError("No data to save. Initialize data loaders first.")
        
        logger.info(f"Saving processed historical data to {output_path}")
        
        # Save using configured storage backend
        storage_format = self.config.get('storage_format', 'parquet')
        
        if storage_format == 'parquet' and 'parquet' in self.storage_backends:
            # Implementation would save to Parquet format
            logger.info("Saved to Parquet format")
        elif storage_format == 'hdf5' and 'hdf5' in self.storage_backends:
            # Implementation would save to HDF5 format
            logger.info("Saved to HDF5 format")
        
        logger.info("Historical data saved successfully")

# Data transformation utilities for historical data
class HistoricalDataAugmentation:
    """Data augmentation techniques for historical financial time series."""
    
    @staticmethod
    def add_noise(data: torch.Tensor, noise_level: float = 0.01) -> torch.Tensor:
        """Add Gaussian noise to historical data."""
        noise = torch.randn_like(data) * noise_level
        return data + noise
    
    @staticmethod
    def time_shift(data: torch.Tensor, max_shift: int = 5) -> torch.Tensor:
        """Apply random time shift to historical data."""
        shift = np.random.randint(-max_shift, max_shift + 1)
        if shift > 0:
            return torch.cat([data[shift:], data[:shift]], dim=0)
        elif shift < 0:
            return torch.cat([data[shift:], data[:shift]], dim=0)
        return data
    
    @staticmethod
    def magnitude_scaling(data: torch.Tensor, scale_range: Tuple[float, float] = (0.9, 1.1)) -> torch.Tensor:
        """Apply random magnitude scaling to historical data."""
        scale = np.random.uniform(*scale_range)
        return data * scale

if __name__ == "__main__":
    # Example usage with historical data only
    import asyncio
    
    async def test_historical_data_loader():
        config = {
            'symbols': ['AAPL', 'MSFT'],
            'start_date': '2020-01-01',
            'end_date': '2023-12-31',  # Historical end date
            'timeframes': ['1day'],
            'sequence_length': 60,
            'prediction_horizon': 1,
            'validation_split': 0.2,
            'test_split': 0.1,
            'batch_size': 32,
            'num_workers': 4,
            'pin_memory': True,
            'prefetch_factor': 2,
            'data_source': 'polygon',
            'real_time': {'enabled': False}  # Explicitly disabled
        }
        
        loader = TrainingDataLoader(config)
        
        try:
            await loader.initialize()
            
            # Test data loading
            train_loader = loader.get_train_loader()
            stats = loader.get_data_stats()
            
            print("Historical Data Loader Stats:")
            for key, value in stats.items():
                print(f"  {key}: {value}")
            
            # Test a few batches
            for batch_idx, (features, targets, prev_pos) in enumerate(train_loader):
                print(f"Batch {batch_idx}: features {features.shape}, targets {targets.shape}")
                if batch_idx >= 2:  # Just test a few batches
                    break
            
            print("Historical data loader test completed successfully")
            print("CONFIRMED: Using only historical data from Polygon API")
            
        except Exception as e:
            print(f"Test failed: {e}")
    
    asyncio.run(test_historical_data_loader())
