"""
Unit tests for Feature Engineering Process.

Tests the feature engineering component with ARM64 optimizations,
performance benchmarks, and comprehensive feature calculation validation.
"""

import pytest
import numpy as np
import pandas as pd
import torch
import time
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timezone, timedelta

from deep_momentum_trading.src.data.feature_engineering import (
    FeatureEngineeringProcess,
    FeatureEngineeringConfig,
    FeatureEngineeringStats,
    _calculate_rsi_numba,
    _calculate_macd_numba,
    _calculate_bollinger_bands_numba,
    _calculate_momentum_numba,
    _calculate_stochastic_numba,
    _calculate_atr_numba,
    _calculate_williams_r_numba
)
from deep_momentum_trading.tests.fixtures.sample_data import get_sample_market_data
from deep_momentum_trading.tests.fixtures.test_configs import TestFeatureConfig


class TestFeatureEngineeringConfig:
    """Test FeatureEngineeringConfig class."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = FeatureEngineeringConfig()
        
        assert config.zmq_subscriber_port == 5555
        assert config.zmq_publisher_port == 5556
        assert config.memory_cache_max_gb == 200.0
        assert config.enable_arm64_optimizations is True
        assert config.enable_parallel_processing is True
        assert config.max_workers == 4
        assert config.chunk_size == 10000
        assert config.enable_performance_monitoring is True
        assert config.enable_caching is True
        assert config.cache_size == 1000
        assert config.feature_calculation_timeout == 30.0
        assert config.enable_advanced_features is True
        assert config.enable_real_time_features is True
    
    def test_custom_config(self):
        """Test custom configuration values."""
        config = FeatureEngineeringConfig(
            zmq_subscriber_port=6666,
            zmq_publisher_port=6667,
            memory_cache_max_gb=50.0,
            enable_arm64_optimizations=False,
            max_workers=8
        )
        
        assert config.zmq_subscriber_port == 6666
        assert config.zmq_publisher_port == 6667
        assert config.memory_cache_max_gb == 50.0
        assert config.enable_arm64_optimizations is False
        assert config.max_workers == 8


class TestFeatureEngineeringStats:
    """Test FeatureEngineeringStats class."""
    
    def test_default_stats(self):
        """Test default statistics values."""
        stats = FeatureEngineeringStats()
        
        assert stats.features_calculated == 0
        assert stats.processing_time_seconds == 0.0
        assert stats.cache_hits == 0
        assert stats.cache_misses == 0
        assert stats.symbols_processed == 0
        assert stats.errors == 0
        assert stats.numba_compilation_time == 0.0
        assert stats.arm64_optimizations_used == 0
        assert isinstance(stats.start_time, float)
    
    def test_features_per_second_calculation(self):
        """Test features per second calculation."""
        stats = FeatureEngineeringStats()
        stats.features_calculated = 100
        stats.processing_time_seconds = 10.0
        
        assert stats.features_per_second == 10.0
        
        # Test division by zero protection
        stats.processing_time_seconds = 0.0
        assert stats.features_per_second == 100000.0  # 100 / 0.001
    
    def test_uptime_calculation(self):
        """Test uptime calculation."""
        stats = FeatureEngineeringStats()
        time.sleep(0.01)  # Small delay
        
        uptime = stats.uptime_seconds
        assert uptime > 0
        assert uptime < 1.0  # Should be very small


class TestNumbaFunctions:
    """Test Numba-compiled feature calculation functions."""
    
    @pytest.fixture
    def sample_prices(self):
        """Generate sample price data for testing."""
        np.random.seed(42)
        base_price = 100.0
        returns = np.random.normal(0.001, 0.02, 100)
        prices = [base_price]
        
        for ret in returns:
            prices.append(prices[-1] * (1 + ret))
        
        return np.array(prices[1:])
    
    @pytest.fixture
    def sample_ohlc(self, sample_prices):
        """Generate sample OHLC data."""
        high = sample_prices * (1 + np.abs(np.random.normal(0, 0.01, len(sample_prices))))
        low = sample_prices * (1 - np.abs(np.random.normal(0, 0.01, len(sample_prices))))
        return high, low, sample_prices
    
    def test_calculate_rsi_numba(self, sample_prices):
        """Test RSI calculation with Numba optimization."""
        rsi = _calculate_rsi_numba(sample_prices, window=14)
        
        # Check output shape
        assert len(rsi) == len(sample_prices)
        
        # Check RSI bounds (0-100)
        valid_rsi = rsi[~np.isnan(rsi)]
        assert np.all(valid_rsi >= 0)
        assert np.all(valid_rsi <= 100)
        
        # Check that first 13 values are NaN (window-1)
        assert np.all(np.isnan(rsi[:13]))
        
        # Check that we have valid values after window period
        assert not np.isnan(rsi[20])
    
    def test_calculate_macd_numba(self, sample_prices):
        """Test MACD calculation with Numba optimization."""
        macd, signal, histogram = _calculate_macd_numba(sample_prices)
        
        # Check output shapes
        assert len(macd) == len(sample_prices)
        assert len(signal) == len(sample_prices)
        assert len(histogram) == len(sample_prices)
        
        # Check that histogram = macd - signal (where both are valid)
        valid_mask = ~(np.isnan(macd) | np.isnan(signal))
        if np.any(valid_mask):
            np.testing.assert_array_almost_equal(
                histogram[valid_mask], 
                macd[valid_mask] - signal[valid_mask],
                decimal=10
            )
    
    def test_calculate_bollinger_bands_numba(self, sample_prices):
        """Test Bollinger Bands calculation with Numba optimization."""
        upper, middle, lower = _calculate_bollinger_bands_numba(sample_prices, window=20, num_std=2.0)
        
        # Check output shapes
        assert len(upper) == len(sample_prices)
        assert len(middle) == len(sample_prices)
        assert len(lower) == len(sample_prices)
        
        # Check band relationships (where all are valid)
        valid_mask = ~(np.isnan(upper) | np.isnan(middle) | np.isnan(lower))
        if np.any(valid_mask):
            assert np.all(upper[valid_mask] >= middle[valid_mask])
            assert np.all(middle[valid_mask] >= lower[valid_mask])
    
    def test_calculate_momentum_numba(self, sample_prices):
        """Test Momentum calculation with Numba optimization."""
        momentum = _calculate_momentum_numba(sample_prices, window=10)
        
        # Check output shape
        assert len(momentum) == len(sample_prices)
        
        # Check that first 10 values are NaN
        assert np.all(np.isnan(momentum[:10]))
        
        # Check momentum calculation (price[t] / price[t-window] * 100)
        if len(sample_prices) > 15:
            expected = (sample_prices[15] / sample_prices[5]) * 100
            np.testing.assert_almost_equal(momentum[15], expected, decimal=10)
    
    def test_calculate_stochastic_numba(self, sample_ohlc):
        """Test Stochastic Oscillator calculation with Numba optimization."""
        high, low, close = sample_ohlc
        k_percent, d_percent = _calculate_stochastic_numba(high, low, close, window=14)
        
        # Check output shapes
        assert len(k_percent) == len(close)
        assert len(d_percent) == len(close)
        
        # Check %K bounds (0-100)
        valid_k = k_percent[~np.isnan(k_percent)]
        if len(valid_k) > 0:
            assert np.all(valid_k >= 0)
            assert np.all(valid_k <= 100)
    
    def test_calculate_atr_numba(self, sample_ohlc):
        """Test Average True Range calculation with Numba optimization."""
        high, low, close = sample_ohlc
        atr = _calculate_atr_numba(high, low, close, window=14)
        
        # Check output shape
        assert len(atr) == len(close)
        
        # Check that ATR is positive (where valid)
        valid_atr = atr[~np.isnan(atr)]
        if len(valid_atr) > 0:
            assert np.all(valid_atr >= 0)
    
    def test_calculate_williams_r_numba(self, sample_ohlc):
        """Test Williams %R calculation with Numba optimization."""
        high, low, close = sample_ohlc
        williams_r = _calculate_williams_r_numba(high, low, close, window=14)
        
        # Check output shape
        assert len(williams_r) == len(close)
        
        # Check Williams %R bounds (-100 to 0)
        valid_wr = williams_r[~np.isnan(williams_r)]
        if len(valid_wr) > 0:
            assert np.all(valid_wr >= -100)
            assert np.all(valid_wr <= 0)
    
    @pytest.mark.performance
    def test_numba_performance(self, sample_prices):
        """Test performance of Numba-compiled functions."""
        # Warm up
        _calculate_rsi_numba(sample_prices[:50])
        
        # Benchmark RSI calculation
        start_time = time.perf_counter()
        for _ in range(100):
            _calculate_rsi_numba(sample_prices)
        end_time = time.perf_counter()
        
        avg_time = (end_time - start_time) / 100
        assert avg_time < 0.01  # Should be very fast with Numba


class TestFeatureEngineeringProcess:
    """Test FeatureEngineeringProcess class."""
    
    @pytest.fixture
    def test_config(self):
        """Get test configuration."""
        return TestFeatureConfig()
    
    @pytest.fixture
    def mock_dependencies(self):
        """Mock external dependencies."""
        with patch('deep_momentum_trading.src.data.feature_engineering.ZMQPublisher') as mock_publisher, \
             patch('deep_momentum_trading.src.data.feature_engineering.UnifiedMemoryManager') as mock_memory:
            
            mock_publisher_instance = Mock()
            mock_publisher.return_value = mock_publisher_instance
            
            mock_memory_instance = Mock()
            mock_memory.return_value = mock_memory_instance
            
            yield {
                'publisher': mock_publisher_instance,
                'memory': mock_memory_instance
            }
    
    def test_initialization(self, test_config, mock_dependencies):
        """Test FeatureEngineeringProcess initialization."""
        with patch('deep_momentum_trading.config.settings.config_manager') as mock_config:
            mock_config.get.return_value = {}
            
            process = FeatureEngineeringProcess(config=test_config)
            
            assert process.config == test_config
            assert isinstance(process.stats, FeatureEngineeringStats)
            assert process.processing_cache == {}
            assert not process.is_running
    
    def test_initialization_with_fallback_params(self, mock_dependencies):
        """Test initialization with fallback parameters."""
        with patch('deep_momentum_trading.config.settings.config_manager') as mock_config:
            mock_config.get.return_value = {
                'subscriber_port': 7777,
                'publisher_port': 7778,
                'memory_cache_gb': 64.0
            }
            
            process = FeatureEngineeringProcess(
                zmq_subscriber_port=8888,
                zmq_publisher_port=8889,
                memory_cache_max_gb=128.0
            )
            
            # Should use provided parameters over config
            assert process.config.zmq_subscriber_port == 8888
            assert process.config.zmq_publisher_port == 8889
            assert process.config.memory_cache_max_gb == 128.0
    
    def test_calculate_features_basic(self, test_config, mock_dependencies):
        """Test basic feature calculation."""
        with patch('deep_momentum_trading.config.settings.config_manager') as mock_config:
            mock_config.get.return_value = {}
            
            process = FeatureEngineeringProcess(config=test_config)
            
            # Generate sample data
            market_data = get_sample_market_data(n_records=100)
            
            # Calculate features
            features = process.calculate_features(market_data)
            
            # Verify output
            assert isinstance(features, pd.DataFrame)
            assert len(features) == len(market_data)
            assert len(features.columns) > 0
            
            # Check for expected feature columns
            expected_features = ['rsi_14', 'rsi_28', 'macd', 'macd_signal', 'macd_hist']
            for feature in expected_features:
                assert feature in features.columns
    
    def test_calculate_features_empty_input(self, test_config, mock_dependencies):
        """Test feature calculation with empty input."""
        with patch('deep_momentum_trading.config.settings.config_manager') as mock_config:
            mock_config.get.return_value = {}
            
            process = FeatureEngineeringProcess(config=test_config)
            
            # Test with empty DataFrame
            empty_df = pd.DataFrame()
            features = process.calculate_features(empty_df)
            
            assert isinstance(features, pd.DataFrame)
            assert len(features) == 0
    
    def test_calculate_features_advanced(self, test_config, mock_dependencies):
        """Test advanced feature calculation."""
        test_config.enable_advanced_features = True
        
        with patch('deep_momentum_trading.config.settings.config_manager') as mock_config:
            mock_config.get.return_value = {}
            
            process = FeatureEngineeringProcess(config=test_config)
            
            # Generate sample data with OHLCV
            market_data = get_sample_market_data(n_records=100)
            
            # Calculate features
            features = process.calculate_features(market_data)
            
            # Check for advanced features
            advanced_features = ['stoch_k', 'stoch_d', 'atr', 'williams_r']
            for feature in advanced_features:
                assert feature in features.columns
    
    @pytest.mark.performance
    def test_calculate_features_performance(self, test_config, mock_dependencies):
        """Test feature calculation performance."""
        with patch('deep_momentum_trading.config.settings.config_manager') as mock_config:
            mock_config.get.return_value = {}
            
            process = FeatureEngineeringProcess(config=test_config)
            
            # Generate larger dataset
            market_data = get_sample_market_data(n_records=1000)
            
            # Benchmark feature calculation
            start_time = time.perf_counter()
            features = process.calculate_features(market_data)
            end_time = time.perf_counter()
            
            calculation_time = end_time - start_time
            
            # Verify performance
            assert calculation_time < 5.0  # Should complete within 5 seconds
            assert len(features) == len(market_data)
            assert len(features.columns) > 20  # Should have many features
    
    def test_process_and_publish_features(self, test_config, mock_dependencies):
        """Test feature processing and publishing."""
        with patch('deep_momentum_trading.config.settings.config_manager') as mock_config:
            mock_config.get.return_value = {}
            
            process = FeatureEngineeringProcess(config=test_config)
            
            # Generate sample data
            market_data = get_sample_market_data(n_records=50)
            symbol = "AAPL"
            
            # Process and publish
            process.process_and_publish_features(symbol, market_data)
            
            # Verify caching was called
            mock_dependencies['memory'].store_features.assert_called_once()
            
            # Verify publishing was called
            mock_dependencies['publisher'].publish_trading_signal.assert_called_once()
            
            # Check statistics update
            assert process.stats.symbols_processed == 1
    
    def test_process_and_publish_features_empty_data(self, test_config, mock_dependencies):
        """Test processing with empty market data."""
        with patch('deep_momentum_trading.config.settings.config_manager') as mock_config:
            mock_config.get.return_value = {}
            
            process = FeatureEngineeringProcess(config=test_config)
            
            # Test with empty DataFrame
            empty_df = pd.DataFrame()
            symbol = "AAPL"
            
            # Process and publish (should handle gracefully)
            process.process_and_publish_features(symbol, empty_df)
            
            # Verify no caching or publishing occurred
            mock_dependencies['memory'].store_features.assert_not_called()
            mock_dependencies['publisher'].publish_trading_signal.assert_not_called()
    
    def test_caching_functionality(self, test_config, mock_dependencies):
        """Test feature caching functionality."""
        test_config.enable_caching = True
        
        with patch('deep_momentum_trading.config.settings.config_manager') as mock_config:
            mock_config.get.return_value = {}
            
            process = FeatureEngineeringProcess(config=test_config)
            
            # Generate sample data
            market_data = get_sample_market_data(n_records=50)
            symbol = "AAPL"
            
            # First call - should calculate and cache
            process.process_and_publish_features(symbol, market_data)
            assert process.stats.cache_misses == 1
            
            # Second call with same data - should use cache
            process.process_and_publish_features(symbol, market_data)
            assert process.stats.cache_hits == 1
    
    def test_get_cached_features(self, test_config, mock_dependencies):
        """Test retrieving cached features."""
        with patch('deep_momentum_trading.config.settings.config_manager') as mock_config:
            mock_config.get.return_value = {}
            
            process = FeatureEngineeringProcess(config=test_config)
            
            # Mock return value
            expected_features = {'feature1': 1.0, 'feature2': 2.0}
            mock_dependencies['memory'].get_features.return_value = expected_features
            
            # Get cached features
            result = process.get_cached_features("AAPL")
            
            assert result == expected_features
            mock_dependencies['memory'].get_features.assert_called_once_with("AAPL")
    
    def test_get_cached_features_error_handling(self, test_config, mock_dependencies):
        """Test error handling in get_cached_features."""
        with patch('deep_momentum_trading.config.settings.config_manager') as mock_config:
            mock_config.get.return_value = {}
            
            process = FeatureEngineeringProcess(config=test_config)
            
            # Mock exception
            mock_dependencies['memory'].get_features.side_effect = Exception("Cache error")
            
            # Get cached features (should handle error gracefully)
            result = process.get_cached_features("AAPL")
            
            assert result is None
    
    def test_statistics_management(self, test_config, mock_dependencies):
        """Test statistics management methods."""
        with patch('deep_momentum_trading.config.settings.config_manager') as mock_config:
            mock_config.get.return_value = {}
            
            process = FeatureEngineeringProcess(config=test_config)
            
            # Update some statistics
            process.stats.features_calculated = 100
            process.stats.symbols_processed = 10
            
            # Get statistics
            stats = process.get_statistics()
            assert stats.features_calculated == 100
            assert stats.symbols_processed == 10
            
            # Reset statistics
            process.reset_statistics()
            new_stats = process.get_statistics()
            assert new_stats.features_calculated == 0
            assert new_stats.symbols_processed == 0
    
    def test_cache_management(self, test_config, mock_dependencies):
        """Test cache management methods."""
        with patch('deep_momentum_trading.config.settings.config_manager') as mock_config:
            mock_config.get.return_value = {}
            
            process = FeatureEngineeringProcess(config=test_config)
            
            # Add some items to cache
            process.processing_cache['key1'] = 'value1'
            process.processing_cache['key2'] = 'value2'
            
            assert len(process.processing_cache) == 2
            
            # Clear cache
            process.clear_cache()
            assert len(process.processing_cache) == 0
    
    def test_health_check(self, test_config, mock_dependencies):
        """Test health check functionality."""
        with patch('deep_momentum_trading.config.settings.config_manager') as mock_config:
            mock_config.get.return_value = {}
            
            process = FeatureEngineeringProcess(config=test_config)
            
            # Mock memory manager
            mock_dependencies['memory'].current_memory_gb = 1.5
            
            # Perform health check
            health = process.health_check()
            
            assert health['status'] == 'healthy'
            assert 'timestamp' in health
            assert 'components' in health
            assert 'statistics' in health
            
            # Check components
            assert 'zmq_publisher' in health['components']
            assert 'feature_cache' in health['components']
            
            # Check statistics
            assert 'features_calculated' in health['statistics']
            assert 'error_rate' in health['statistics']
    
    @pytest.mark.arm64
    def test_arm64_optimizations(self, test_config, mock_dependencies):
        """Test ARM64-specific optimizations."""
        test_config.enable_arm64_optimizations = True
        
        with patch('deep_momentum_trading.config.settings.config_manager') as mock_config, \
             patch('platform.machine') as mock_machine:
            
            mock_config.get.return_value = {}
            mock_machine.return_value = 'arm64'  # Simulate ARM64 environment
            
            process = FeatureEngineeringProcess(config=test_config)
            
            # Verify ARM64 detection
            assert process.is_arm64 is True
            
            # Generate sample data
            market_data = get_sample_market_data(n_records=100)
            
            # Calculate features (should use ARM64 optimizations)
            features = process.calculate_features(market_data)
            
            assert isinstance(features, pd.DataFrame)
            assert len(features) > 0
            
            # Check that ARM64 optimizations were used
            assert process.stats.arm64_optimizations_used > 0
    
    def test_run_and_stop(self, test_config, mock_dependencies):
        """Test run and stop methods."""
        with patch('deep_momentum_trading.config.settings.config_manager') as mock_config:
            mock_config.get.return_value = {}
            
            process = FeatureEngineeringProcess(config=test_config)
            
            # Initially not running
            assert not process.is_running
            
            # Start running
            process.run()
            assert process.is_running
            
            # Stop running
            process.stop()
            assert not process.is_running


@pytest.mark.integration
class TestFeatureEngineeringIntegration:
    """Integration tests for feature engineering."""
    
    def test_end_to_end_feature_calculation(self):
        """Test end-to-end feature calculation process."""
        # Use minimal config for integration test
        config = TestFeatureConfig()
        config.enable_parallel_processing = False
        config.enable_performance_monitoring = False
        
        with patch('deep_momentum_trading.src.data.feature_engineering.ZMQPublisher') as mock_publisher, \
             patch('deep_momentum_trading.src.data.feature_engineering.UnifiedMemoryManager') as mock_memory, \
             patch('deep_momentum_trading.config.settings.config_manager') as mock_config:
            
            mock_config.get.return_value = {}
            
            # Initialize process
            process = FeatureEngineeringProcess(config=config)
            
            # Generate realistic market data
            symbols = ["AAPL", "MSFT", "GOOGL"]
            
            for symbol in symbols:
                market_data = get_sample_market_data(symbol=symbol, n_records=200)
                
                # Process features
                features = process.calculate_features(market_data)
                
                # Verify comprehensive feature set
                assert len(features.columns) >= 20
                assert len(features) == len(market_data)
                
                # Verify no excessive NaN values
                nan_ratio = features.isnull().sum().sum() / (len(features) * len(features.columns))
                assert nan_ratio < 0.3  # Less than 30% NaN values
                
                # Process and publish
                process.process_and_publish_features(symbol, market_data)
            
            # Verify statistics
            stats = process.get_statistics()
            assert stats.symbols_processed == len(symbols)
            assert stats.features_calculated > 0


if __name__ == "__main__":
    pytest.main([__file__])
