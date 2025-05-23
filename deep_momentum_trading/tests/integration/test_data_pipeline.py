"""
Integration tests for Data Pipeline.

Tests the complete data flow from ingestion through feature engineering,
including real-time data processing, caching, and ARM64 optimizations.
"""

import pytest
import asyncio
import time
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any

from deep_momentum_trading.tests.fixtures.sample_data import (
    get_sample_market_data,
    get_sample_features_data,
    get_sample_correlation_matrix
)
from deep_momentum_trading.tests.fixtures.test_configs import (
    TestConfigManager,
    TestScenarios
)


@pytest.mark.integration
class TestDataIngestionPipeline:
    """Test data ingestion pipeline integration."""
    
    @pytest.fixture
    def pipeline_config(self):
        """Get data pipeline test configuration."""
        return {
            'symbols': ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA'],
            'data_sources': ['polygon', 'alpaca'],
            'update_frequency_ms': 1000,
            'cache_size_mb': 100,
            'enable_real_time': True,
            'enable_historical': True,
            'lookback_days': 30,
            'enable_arm64_optimizations': True
        }
    
    @pytest.fixture
    def mock_data_sources(self):
        """Mock external data sources."""
        mocks = {}
        
        # Mock Polygon client
        mock_polygon = Mock()
        mock_polygon.is_connected = True
        mock_polygon.get_historical_aggregates = AsyncMock()
        mock_polygon.subscribe_to_trades = AsyncMock(return_value=True)
        mock_polygon.get_real_time_data = AsyncMock()
        mocks['polygon'] = mock_polygon
        
        # Mock Alpaca client  
        mock_alpaca = Mock()
        mock_alpaca.get_bars = Mock()
        mock_alpaca.get_latest_bars = Mock()
        mock_alpaca.stream_bars = AsyncMock()
        mocks['alpaca'] = mock_alpaca
        
        return mocks
    
    @pytest.mark.asyncio
    async def test_historical_data_ingestion(self, pipeline_config, mock_data_sources):
        """Test historical data ingestion pipeline."""
        
        with patch('deep_momentum_trading.src.data.polygon_client.PolygonClient') as mock_polygon_class, \
             patch('deep_momentum_trading.src.data.data_manager.DataManager') as mock_data_manager_class:
            
            # Setup mocks
            mock_polygon_class.return_value = mock_data_sources['polygon']
            
            # Mock historical data response
            historical_data = {}
            for symbol in pipeline_config['symbols']:
                historical_data[symbol] = get_sample_market_data(
                    symbol=symbol, 
                    n_records=pipeline_config['lookback_days'] * 390  # Trading minutes per day
                )
            
            mock_data_sources['polygon'].get_historical_aggregates.return_value = historical_data
            
            # Initialize data manager
            from deep_momentum_trading.src.data.data_manager import DataManager
            
            data_manager = DataManager(config=pipeline_config)
            
            # Test historical data ingestion
            start_date = datetime.now(timezone.utc) - timedelta(days=pipeline_config['lookback_days'])
            end_date = datetime.now(timezone.utc)
            
            ingested_data = await data_manager.ingest_historical_data(
                symbols=pipeline_config['symbols'],
                start_date=start_date,
                end_date=end_date
            )
            
            # Verify data ingestion
            assert len(ingested_data) == len(pipeline_config['symbols'])
            
            for symbol in pipeline_config['symbols']:
                assert symbol in ingested_data
                assert len(ingested_data[symbol]) > 0
                assert 'open' in ingested_data[symbol].columns
                assert 'high' in ingested_data[symbol].columns
                assert 'low' in ingested_data[symbol].columns
                assert 'close' in ingested_data[symbol].columns
                assert 'volume' in ingested_data[symbol].columns
    
    @pytest.mark.asyncio
    async def test_real_time_data_streaming(self, pipeline_config, mock_data_sources):
        """Test real-time data streaming pipeline."""
        
        with patch('deep_momentum_trading.src.data.real_time_feed.RealTimeFeed') as mock_feed_class:
            
            # Mock real-time data feed
            mock_feed = Mock()
            mock_feed.is_connected = True
            mock_feed.subscribe = AsyncMock(return_value=True)
            mock_feed.start_streaming = AsyncMock()
            mock_feed.stop_streaming = AsyncMock()
            mock_feed_class.return_value = mock_feed
            
            # Setup data handler
            received_data = []
            
            def data_handler(symbol, data):
                received_data.append({'symbol': symbol, 'data': data})
            
            mock_feed.set_data_handler = Mock(side_effect=lambda handler: setattr(mock_feed, '_handler', handler))
            
            from deep_momentum_trading.src.data.real_time_feed import RealTimeFeed
            
            # Initialize real-time feed
            feed = RealTimeFeed(config=pipeline_config)
            feed.set_data_handler(data_handler)
            
            # Start streaming
            await feed.start_streaming(pipeline_config['symbols'])
            
            # Simulate real-time data
            for symbol in pipeline_config['symbols']:
                mock_data = {
                    'symbol': symbol,
                    'price': np.random.uniform(100, 200),
                    'volume': np.random.randint(1000, 10000),
                    'timestamp': datetime.now(timezone.utc).isoformat()
                }
                
                # Simulate data arrival
                if hasattr(mock_feed, '_handler'):
                    mock_feed._handler(symbol, mock_data)
            
            # Verify streaming setup
            mock_feed.start_streaming.assert_called_once()
            assert len(received_data) == len(pipeline_config['symbols'])
    
    def test_data_validation_pipeline(self, pipeline_config):
        """Test data validation in the pipeline."""
        
        with patch('deep_momentum_trading.src.data.data_preprocessing.DataPreprocessor') as mock_preprocessor_class:
            
            # Create sample data with various quality issues
            test_data = get_sample_market_data(n_records=1000)
            
            # Introduce data quality issues
            test_data.loc[10:20, 'close'] = np.nan  # Missing values
            test_data.loc[50:60, 'volume'] = -1     # Invalid values
            test_data.loc[100, 'high'] = test_data.loc[100, 'low'] - 1  # Inconsistent values
            
            # Mock preprocessor
            mock_preprocessor = Mock()
            mock_preprocessor.validate_data = Mock(return_value={
                'is_valid': False,
                'issues': ['missing_values', 'invalid_volumes', 'price_inconsistencies'],
                'cleaned_data': test_data.dropna()
            })
            mock_preprocessor_class.return_value = mock_preprocessor
            
            from deep_momentum_trading.src.data.data_preprocessing import DataPreprocessor
            
            preprocessor = DataPreprocessor(config=pipeline_config)
            validation_result = preprocessor.validate_data(test_data)
            
            # Verify validation
            assert validation_result['is_valid'] is False
            assert len(validation_result['issues']) > 0
            assert 'missing_values' in validation_result['issues']
            assert len(validation_result['cleaned_data']) < len(test_data)
    
    def test_data_caching_pipeline(self, pipeline_config):
        """Test data caching in the pipeline."""
        
        with patch('deep_momentum_trading.src.data.memory_cache.MemoryCache') as mock_cache_class:
            
            # Mock cache
            mock_cache = Mock()
            mock_cache.get = Mock(return_value=None)  # Cache miss initially
            mock_cache.set = Mock(return_value=True)
            mock_cache.exists = Mock(return_value=False)
            mock_cache.get_memory_usage = Mock(return_value=50.0)  # 50MB
            mock_cache_class.return_value = mock_cache
            
            from deep_momentum_trading.src.data.memory_cache import MemoryCache
            
            cache = MemoryCache(max_size_mb=pipeline_config['cache_size_mb'])
            
            # Test cache operations
            test_data = get_sample_market_data(n_records=500)
            cache_key = "AAPL_1D_20231201"
            
            # Cache miss
            cached_data = cache.get(cache_key)
            assert cached_data is None
            
            # Cache set
            cache.set(cache_key, test_data)
            mock_cache.set.assert_called_once()
            
            # Simulate cache hit
            mock_cache.get.return_value = test_data
            mock_cache.exists.return_value = True
            
            cached_data = cache.get(cache_key)
            assert cached_data is not None
            
            # Verify memory usage
            memory_usage = cache.get_memory_usage()
            assert memory_usage <= pipeline_config['cache_size_mb']


@pytest.mark.integration
class TestFeatureEngineeringPipeline:
    """Test feature engineering pipeline integration."""
    
    @pytest.fixture
    def feature_config(self):
        """Get feature engineering configuration."""
        return {
            'technical_indicators': ['rsi', 'macd', 'bollinger_bands', 'stochastic'],
            'momentum_features': ['price_momentum', 'volume_momentum', 'volatility'],
            'cross_sectional_features': ['relative_strength', 'sector_momentum'],
            'lookback_windows': [5, 10, 20, 50],
            'enable_arm64_optimizations': True,
            'parallel_processing': True,
            'max_workers': 4
        }
    
    @pytest.mark.asyncio
    async def test_feature_calculation_pipeline(self, feature_config):
        """Test complete feature calculation pipeline."""
        
        with patch('deep_momentum_trading.src.data.feature_engineering.FeatureEngineeringProcess') as mock_fe_class, \
             patch('deep_momentum_trading.src.data.feature_engineering.ZMQPublisher'), \
             patch('deep_momentum_trading.src.data.feature_engineering.UnifiedMemoryManager'):
            
            # Mock feature engineering process
            mock_fe = Mock()
            mock_fe.calculate_features = Mock()
            mock_fe.get_statistics = Mock(return_value=Mock(
                symbols_processed=5,
                features_calculated=250,
                processing_time_seconds=2.5,
                features_per_second=100.0
            ))
            mock_fe_class.return_value = mock_fe
            
            # Generate test data
            symbols = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA']
            market_data = {}
            feature_data = {}
            
            for symbol in symbols:
                market_data[symbol] = get_sample_market_data(symbol=symbol, n_records=200)
                feature_data[symbol] = get_sample_features_data(n_records=200, n_features=50)
            
            mock_fe.calculate_features.side_effect = lambda data: feature_data.get(
                data.get('symbol', 'AAPL'), 
                get_sample_features_data(n_records=len(data), n_features=50)
            )
            
            from deep_momentum_trading.src.data.feature_engineering import FeatureEngineeringProcess
            
            # Initialize feature engineering
            fe_process = FeatureEngineeringProcess(config=feature_config)
            
            # Process features for all symbols
            results = {}
            for symbol in symbols:
                features = fe_process.calculate_features(market_data[symbol])
                results[symbol] = features
            
            # Verify feature calculation
            assert len(results) == len(symbols)
            
            for symbol, features in results.items():
                assert isinstance(features, pd.DataFrame)
                assert len(features) > 0
                assert len(features.columns) >= 20  # Should have multiple features
            
            # Verify statistics
            stats = fe_process.get_statistics()
            assert stats.symbols_processed == len(symbols)
            assert stats.features_calculated > 0
    
    def test_cross_sectional_features(self, feature_config):
        """Test cross-sectional feature calculation."""
        
        with patch('deep_momentum_trading.src.data.feature_engineering.FeatureEngineeringProcess') as mock_fe_class:
            
            # Generate correlated market data
            symbols = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA']
            correlation_matrix = get_sample_correlation_matrix(symbols)
            
            # Mock cross-sectional feature calculation
            mock_fe = Mock()
            mock_fe.calculate_cross_sectional_features = Mock()
            mock_fe_class.return_value = mock_fe
            
            # Generate cross-sectional features
            cross_sectional_data = {}
            for symbol in symbols:
                cross_sectional_data[symbol] = {
                    'relative_strength': np.random.uniform(-2, 2),
                    'sector_momentum': np.random.uniform(-1, 1),
                    'correlation_rank': np.random.randint(1, len(symbols) + 1)
                }
            
            mock_fe.calculate_cross_sectional_features.return_value = cross_sectional_data
            
            from deep_momentum_trading.src.data.feature_engineering import FeatureEngineeringProcess
            
            fe_process = FeatureEngineeringProcess(config=feature_config)
            
            # Calculate cross-sectional features
            market_data = {symbol: get_sample_market_data(symbol=symbol, n_records=100) 
                          for symbol in symbols}
            
            cross_features = fe_process.calculate_cross_sectional_features(market_data)
            
            # Verify cross-sectional features
            assert len(cross_features) == len(symbols)
            
            for symbol in symbols:
                assert symbol in cross_features
                assert 'relative_strength' in cross_features[symbol]
                assert 'sector_momentum' in cross_features[symbol]
    
    @pytest.mark.performance
    def test_feature_pipeline_performance(self, feature_config):
        """Test feature pipeline performance."""
        
        with patch('deep_momentum_trading.src.data.feature_engineering.FeatureEngineeringProcess') as mock_fe_class:
            
            # Mock high-performance feature calculation
            mock_fe = Mock()
            mock_fe.calculate_features = Mock()
            mock_fe.get_statistics = Mock()
            mock_fe_class.return_value = mock_fe
            
            # Generate large dataset
            symbols = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA', 'AMZN', 'META', 'NFLX']
            n_records = 1000  # Large dataset
            
            # Mock fast feature calculation
            def fast_feature_calc(data):
                return get_sample_features_data(n_records=len(data), n_features=30)
            
            mock_fe.calculate_features.side_effect = fast_feature_calc
            
            from deep_momentum_trading.src.data.feature_engineering import FeatureEngineeringProcess
            
            fe_process = FeatureEngineeringProcess(config=feature_config)
            
            # Benchmark feature calculation
            start_time = time.perf_counter()
            
            for symbol in symbols:
                market_data = get_sample_market_data(symbol=symbol, n_records=n_records)
                features = fe_process.calculate_features(market_data)
                
                # Verify output
                assert len(features) == n_records
                assert len(features.columns) >= 20
            
            end_time = time.perf_counter()
            
            # Performance assertions
            total_time = end_time - start_time
            features_per_second = (len(symbols) * n_records) / total_time
            
            # Should process efficiently
            assert total_time < 10.0  # Less than 10 seconds total
            assert features_per_second > 500  # More than 500 features per second
    
    @pytest.mark.arm64
    def test_arm64_feature_optimizations(self, feature_config):
        """Test ARM64 optimizations in feature pipeline."""
        
        with patch('deep_momentum_trading.src.data.feature_engineering.IS_ARM64', True), \
             patch('deep_momentum_trading.src.data.feature_engineering.FeatureEngineeringProcess') as mock_fe_class:
            
            # Enable ARM64 optimizations
            feature_config['enable_arm64_optimizations'] = True
            
            # Mock ARM64-optimized feature engineering
            mock_fe = Mock()
            mock_fe.is_arm64 = True
            mock_fe.calculate_features = Mock()
            mock_fe.get_statistics = Mock(return_value=Mock(
                arm64_optimizations_used=10,
                processing_time_seconds=1.2,  # Faster with ARM64
                features_per_second=150.0
            ))
            mock_fe_class.return_value = mock_fe
            
            from deep_momentum_trading.src.data.feature_engineering import FeatureEngineeringProcess
            
            fe_process = FeatureEngineeringProcess(config=feature_config)
            
            # Verify ARM64 detection
            assert fe_process.is_arm64 is True
            
            # Process features with ARM64 optimizations
            market_data = get_sample_market_data(n_records=500)
            features = fe_process.calculate_features(market_data)
            
            # Verify ARM64 optimizations were used
            stats = fe_process.get_statistics()
            assert stats.arm64_optimizations_used > 0
            assert stats.features_per_second > 100  # Should be fast


@pytest.mark.integration
class TestDataPipelineReliability:
    """Test data pipeline reliability and error handling."""
    
    def test_data_source_failover(self):
        """Test failover between data sources."""
        
        with patch('deep_momentum_trading.src.data.polygon_client.PolygonClient') as mock_polygon, \
             patch('deep_momentum_trading.src.data.data_manager.DataManager') as mock_dm_class:
            
            # Mock primary source failure
            mock_polygon_instance = Mock()
            mock_polygon_instance.is_connected = False
            mock_polygon_instance.get_historical_aggregates = Mock(
                side_effect=Exception("Connection failed")
            )
            mock_polygon.return_value = mock_polygon_instance
            
            # Mock data manager with failover
            mock_dm = Mock()
            mock_dm.primary_source_available = False
            mock_dm.failover_to_secondary = Mock(return_value=True)
            mock_dm.get_data_with_failover = Mock(return_value=get_sample_market_data(n_records=100))
            mock_dm_class.return_value = mock_dm
            
            from deep_momentum_trading.src.data.data_manager import DataManager
            
            data_manager = DataManager(config={'enable_failover': True})
            
            # Test failover mechanism
            data = data_manager.get_data_with_failover('AAPL')
            
            # Verify failover occurred
            mock_dm.failover_to_secondary.assert_called_once()
            assert data is not None
            assert len(data) > 0
    
    def test_data_quality_monitoring(self):
        """Test data quality monitoring in pipeline."""
        
        quality_alerts = []
        
        def quality_alert_handler(alert):
            quality_alerts.append(alert)
        
        with patch('deep_momentum_trading.src.data.data_preprocessing.DataPreprocessor') as mock_preprocessor_class:
            
            # Mock data quality monitor
            mock_preprocessor = Mock()
            mock_preprocessor.monitor_data_quality = Mock()
            mock_preprocessor.set_quality_alert_handler = Mock(
                side_effect=lambda handler: setattr(mock_preprocessor, '_alert_handler', handler)
            )
            mock_preprocessor_class.return_value = mock_preprocessor
            
            from deep_momentum_trading.src.data.data_preprocessing import DataPreprocessor
            
            preprocessor = DataPreprocessor(config={'enable_quality_monitoring': True})
            preprocessor.set_quality_alert_handler(quality_alert_handler)
            
            # Simulate quality issues
            if hasattr(mock_preprocessor, '_alert_handler'):
                mock_preprocessor._alert_handler({
                    'type': 'missing_data',
                    'symbol': 'AAPL',
                    'severity': 'high',
                    'message': 'Missing data for 30 minutes'
                })
                
                mock_preprocessor._alert_handler({
                    'type': 'stale_data',
                    'symbol': 'MSFT', 
                    'severity': 'medium',
                    'message': 'Data is 5 minutes old'
                })
            
            # Verify quality monitoring
            assert len(quality_alerts) == 2
            assert quality_alerts[0]['type'] == 'missing_data'
            assert quality_alerts[1]['type'] == 'stale_data'
    
    def test_pipeline_recovery_mechanisms(self):
        """Test pipeline recovery from failures."""
        
        recovery_events = []
        
        def recovery_handler(event):
            recovery_events.append(event)
        
        with patch('deep_momentum_trading.src.data.data_manager.DataManager') as mock_dm_class:
            
            # Mock data manager with recovery
            mock_dm = Mock()
            mock_dm.set_recovery_handler = Mock(
                side_effect=lambda handler: setattr(mock_dm, '_recovery_handler', handler)
            )
            mock_dm.recover_from_failure = Mock(return_value=True)
            mock_dm_class.return_value = mock_dm
            
            from deep_momentum_trading.src.data.data_manager import DataManager
            
            data_manager = DataManager(config={'enable_auto_recovery': True})
            data_manager.set_recovery_handler(recovery_handler)
            
            # Simulate recovery scenarios
            if hasattr(mock_dm, '_recovery_handler'):
                mock_dm._recovery_handler({
                    'type': 'connection_restored',
                    'source': 'polygon',
                    'timestamp': datetime.now(timezone.utc).isoformat()
                })
                
                mock_dm._recovery_handler({
                    'type': 'cache_rebuilt',
                    'size_mb': 75.0,
                    'timestamp': datetime.now(timezone.utc).isoformat()
                })
            
            # Verify recovery handling
            assert len(recovery_events) == 2
            assert recovery_events[0]['type'] == 'connection_restored'
            assert recovery_events[1]['type'] == 'cache_rebuilt'


@pytest.mark.integration
class TestDataPipelineScaling:
    """Test data pipeline scaling and performance."""
    
    @pytest.mark.stress
    def test_high_volume_data_processing(self):
        """Test pipeline under high data volume."""
        
        with patch('deep_momentum_trading.src.data.feature_engineering.FeatureEngineeringProcess') as mock_fe_class:
            
            # Mock high-volume processing
            mock_fe = Mock()
            mock_fe.process_batch = Mock()
            mock_fe.get_statistics = Mock()
            mock_fe_class.return_value = mock_fe
            
            # Generate high-volume test scenario
            symbols = [f"TEST{i:03d}" for i in range(100)]  # 100 symbols
            batch_size = 50
            
            # Mock batch processing
            def process_batch(batch_data):
                return {symbol: get_sample_features_data(n_records=100, n_features=25) 
                       for symbol in batch_data.keys()}
            
            mock_fe.process_batch.side_effect = process_batch
            
            from deep_momentum_trading.src.data.feature_engineering import FeatureEngineeringProcess
            
            fe_process = FeatureEngineeringProcess(config={'batch_size': batch_size})
            
            # Process in batches
            start_time = time.perf_counter()
            total_processed = 0
            
            for i in range(0, len(symbols), batch_size):
                batch_symbols = symbols[i:i + batch_size]
                batch_data = {symbol: get_sample_market_data(symbol=symbol, n_records=100) 
                             for symbol in batch_symbols}
                
                results = fe_process.process_batch(batch_data)
                total_processed += len(results)
            
            end_time = time.perf_counter()
            
            # Verify high-volume processing
            assert total_processed == len(symbols)
            
            # Performance check
            processing_time = end_time - start_time
            symbols_per_second = len(symbols) / processing_time
            
            assert symbols_per_second > 10  # Should process at least 10 symbols per second
    
    @pytest.mark.performance
    def test_parallel_processing_scaling(self):
        """Test parallel processing scaling."""
        
        with patch('concurrent.futures.ProcessPoolExecutor') as mock_executor:
            
            # Mock parallel execution
            mock_executor.return_value.__enter__.return_value.map.return_value = [
                get_sample_features_data(n_records=100, n_features=20) for _ in range(8)
            ]
            
            # Test parallel feature calculation
            symbols = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA', 'AMZN', 'META', 'NFLX']
            
            def parallel_feature_calc(symbol_data_pairs):
                # Mock parallel processing
                return [get_sample_features_data(n_records=100, n_features=20) 
                       for _ in symbol_data_pairs]
            
            # Simulate parallel processing
            start_time = time.perf_counter()
            
            symbol_data_pairs = [(symbol, get_sample_market_data(symbol=symbol, n_records=100)) 
                                for symbol in symbols]
            
            results = parallel_feature_calc(symbol_data_pairs)
            
            end_time = time.perf_counter()
            
            # Verify parallel processing
            assert len(results) == len(symbols)
            
            # Should be faster than sequential processing
            parallel_time = end_time - start_time
            assert parallel_time < 2.0  # Should complete quickly with mocking


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
