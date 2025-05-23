"""
Pytest configuration and shared fixtures for Deep Momentum Trading System tests.

This module provides comprehensive test fixtures, configurations, and utilities
for testing the entire trading system with ARM64 optimizations.
"""

import pytest
import asyncio
import numpy as np
import pandas as pd
import torch
import os
import tempfile
import shutil
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any, Optional
from unittest.mock import Mock, AsyncMock, patch
import warnings

# Suppress warnings during testing
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Test configuration
pytest_plugins = ["pytest_asyncio"]

@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

@pytest.fixture(scope="session")
def test_config():
    """Global test configuration."""
    return {
        "test_data_size": 1000,
        "test_symbols": ["AAPL", "MSFT", "GOOGL", "TSLA", "NVDA"],
        "test_timeframe": "1min",
        "test_start_date": "2023-01-01",
        "test_end_date": "2023-12-31",
        "enable_arm64_tests": True,
        "enable_performance_tests": True,
        "test_timeout": 30.0,
        "mock_api_responses": True
    }

@pytest.fixture(scope="session")
def temp_dir():
    """Create temporary directory for test files."""
    temp_dir = tempfile.mkdtemp(prefix="deep_momentum_test_")
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)

@pytest.fixture
def mock_env_vars():
    """Mock environment variables for testing."""
    env_vars = {
        "POLYGON_API_KEY": "test_polygon_key",
        "ALPACA_API_KEY": "test_alpaca_key",
        "ALPACA_SECRET_KEY": "test_alpaca_secret",
        "ALPACA_BASE_URL": "https://paper-api.alpaca.markets",
        "REDIS_URL": "redis://localhost:6379",
        "DATABASE_URL": "sqlite:///test.db"
    }
    
    with patch.dict(os.environ, env_vars):
        yield env_vars

@pytest.fixture
def sample_market_data():
    """Generate sample market data for testing."""
    np.random.seed(42)
    n_records = 1000
    
    # Generate realistic price data with trends and volatility
    base_price = 100.0
    returns = np.random.normal(0.0005, 0.02, n_records)  # Daily returns
    prices = [base_price]
    
    for ret in returns:
        prices.append(prices[-1] * (1 + ret))
    
    prices = np.array(prices[1:])
    
    # Generate OHLCV data
    data = {
        'timestamp': pd.date_range(start='2023-01-01', periods=n_records, freq='1min'),
        'open': prices * (1 + np.random.normal(0, 0.001, n_records)),
        'high': prices * (1 + np.abs(np.random.normal(0, 0.005, n_records))),
        'low': prices * (1 - np.abs(np.random.normal(0, 0.005, n_records))),
        'close': prices,
        'volume': np.random.randint(1000, 100000, n_records),
        'vwap': prices * (1 + np.random.normal(0, 0.0005, n_records))
    }
    
    df = pd.DataFrame(data)
    df.set_index('timestamp', inplace=True)
    
    return df

@pytest.fixture
def sample_features_data():
    """Generate sample feature data for testing."""
    np.random.seed(42)
    n_records = 1000
    n_features = 50
    
    # Generate realistic feature data
    features = {}
    feature_names = [
        'rsi_14', 'rsi_28', 'macd', 'macd_signal', 'macd_hist',
        'momentum_10', 'momentum_20', 'volatility_20', 'volatility_60',
        'bb_upper', 'bb_middle', 'bb_lower', 'stoch_k', 'stoch_d',
        'atr', 'williams_r', 'volume_ma_10', 'volume_ma_30',
        'on_balance_volume', 'volume_rate_of_change', 'daily_return',
        'log_return', 'high_low_range', 'high_low_ratio', 'open_close_range',
        'gap', 'sma_5', 'sma_10', 'sma_20', 'sma_50',
        'ema_5', 'ema_10', 'ema_20', 'ema_50'
    ]
    
    # Add more features to reach n_features
    while len(feature_names) < n_features:
        feature_names.append(f'feature_{len(feature_names)}')
    
    for i, name in enumerate(feature_names[:n_features]):
        if 'rsi' in name or 'stoch' in name:
            # RSI and Stochastic: 0-100 range
            features[name] = np.random.uniform(20, 80, n_records)
        elif 'return' in name:
            # Returns: small values around 0
            features[name] = np.random.normal(0, 0.02, n_records)
        elif 'volume' in name:
            # Volume features: positive values
            features[name] = np.random.exponential(10000, n_records)
        else:
            # General features: normalized values
            features[name] = np.random.normal(0, 1, n_records)
    
    df = pd.DataFrame(features)
    df.index = pd.date_range(start='2023-01-01', periods=n_records, freq='1min')
    
    return df

@pytest.fixture
def sample_predictions():
    """Generate sample model predictions for testing."""
    symbols = ["AAPL", "MSFT", "GOOGL", "TSLA", "NVDA"]
    predictions = {}
    
    np.random.seed(42)
    for symbol in symbols:
        predictions[symbol] = {
            'symbol': symbol,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'position': np.random.uniform(-1, 1),  # Position signal
            'confidence': np.random.uniform(0.1, 0.9),  # Confidence score
            'expected_return': np.random.uniform(-0.05, 0.05),  # Expected return
            'volatility': np.random.uniform(0.1, 0.3),  # Expected volatility
            'strategy': 'deep_momentum_lstm',
            'model_version': '1.0.0'
        }
    
    return predictions

@pytest.fixture
def sample_positions():
    """Generate sample portfolio positions for testing."""
    symbols = ["AAPL", "MSFT", "GOOGL", "TSLA", "NVDA"]
    positions = {}
    
    np.random.seed(42)
    for symbol in symbols:
        positions[symbol] = {
            'symbol': symbol,
            'quantity': np.random.randint(-100, 100),
            'market_value': np.random.uniform(1000, 10000),
            'cost_basis': np.random.uniform(50, 200),
            'unrealized_pnl': np.random.uniform(-500, 500),
            'side': 'long' if np.random.random() > 0.5 else 'short'
        }
    
    return positions

@pytest.fixture
def mock_polygon_client():
    """Mock Polygon client for testing."""
    mock_client = Mock()
    
    # Mock WebSocket connection
    mock_client.is_connected = True
    mock_client._connect_websocket = AsyncMock(return_value=True)
    mock_client.close = AsyncMock()
    
    # Mock data subscription methods
    mock_client.subscribe_to_trades = AsyncMock(return_value=True)
    mock_client.subscribe_to_quotes = AsyncMock(return_value=True)
    mock_client.subscribe_to_second_aggregates = AsyncMock(return_value=True)
    
    # Mock historical data
    mock_client.get_historical_aggregates = AsyncMock()
    
    return mock_client

@pytest.fixture
def mock_alpaca_client():
    """Mock Alpaca client for testing."""
    mock_client = Mock()
    
    # Mock account information
    mock_client.get_account = Mock(return_value={
        'equity': 100000.0,
        'buying_power': 50000.0,
        'cash': 25000.0,
        'portfolio_value': 100000.0
    })
    
    # Mock positions
    mock_client.get_positions = Mock(return_value=[])
    mock_client.get_position = Mock(return_value=None)
    
    # Mock orders
    mock_client.submit_order = Mock()
    mock_client.get_orders = Mock(return_value=[])
    mock_client.cancel_order = Mock()
    
    # Mock market data
    mock_client.get_bars = Mock(return_value={})
    mock_client.get_latest_quote = Mock(return_value=None)
    
    return mock_client

@pytest.fixture
def mock_zmq_publisher():
    """Mock ZMQ publisher for testing."""
    mock_publisher = Mock()
    mock_publisher.send = Mock()
    mock_publisher.publish_trading_signal = Mock()
    mock_publisher.close = Mock()
    return mock_publisher

@pytest.fixture
def mock_zmq_subscriber():
    """Mock ZMQ subscriber for testing."""
    mock_subscriber = Mock()
    mock_subscriber.start = Mock()
    mock_subscriber.stop = Mock()
    mock_subscriber.add_handler = Mock()
    mock_subscriber.remove_handler = Mock()
    return mock_subscriber

@pytest.fixture
def sample_torch_model_input():
    """Generate sample input for PyTorch models."""
    batch_size = 32
    sequence_length = 60
    input_features = 200
    
    return torch.randn(batch_size, sequence_length, input_features)

@pytest.fixture
def sample_model_config():
    """Sample model configuration for testing."""
    return {
        'input_size': 200,
        'hidden_size': 512,
        'num_layers': 4,
        'dropout': 0.2,
        'device': 'cpu',
        'enable_arm64_optimizations': True
    }

@pytest.fixture
def sample_risk_config():
    """Sample risk management configuration."""
    return {
        'max_portfolio_var': 0.02,
        'max_position_concentration': 0.005,
        'max_sector_concentration': 0.25,
        'max_simultaneous_positions': 15000,
        'daily_capital_limit': 50000.0,
        'min_position_value': 5.0,
        'max_position_value': 2500.0,
        'enable_real_time_monitoring': True,
        'enable_arm64_optimizations': True
    }

@pytest.fixture
def sample_feature_config():
    """Sample feature engineering configuration."""
    return {
        'zmq_subscriber_port': 5555,
        'zmq_publisher_port': 5556,
        'memory_cache_max_gb': 32.0,
        'enable_arm64_optimizations': True,
        'enable_parallel_processing': True,
        'max_workers': 4,
        'enable_advanced_features': True
    }

@pytest.fixture
def mock_memory_manager():
    """Mock unified memory manager for testing."""
    mock_manager = Mock()
    mock_manager.store_features = Mock()
    mock_manager.get_features = Mock(return_value=None)
    mock_manager.clear_cache = Mock()
    mock_manager.get_memory_usage = Mock(return_value=0.0)
    return mock_manager

@pytest.fixture
def sample_correlation_matrix():
    """Generate sample correlation matrix for testing."""
    symbols = ["AAPL", "MSFT", "GOOGL", "TSLA", "NVDA"]
    n_symbols = len(symbols)
    
    # Generate realistic correlation matrix
    np.random.seed(42)
    correlations = np.random.uniform(0.1, 0.8, (n_symbols, n_symbols))
    correlations = (correlations + correlations.T) / 2  # Make symmetric
    np.fill_diagonal(correlations, 1.0)  # Diagonal = 1
    
    return pd.DataFrame(correlations, index=symbols, columns=symbols)

@pytest.fixture
def sample_portfolio_metrics():
    """Generate sample portfolio risk metrics."""
    return {
        'total_var': 0.015,
        'component_var': {
            'AAPL': 0.003,
            'MSFT': 0.004,
            'GOOGL': 0.003,
            'TSLA': 0.003,
            'NVDA': 0.002
        },
        'correlation_risk': 0.25,
        'liquidity_risk': 5.0,
        'concentration_risk': 0.15,
        'sector_risk': {
            'Technology': 0.6,
            'Consumer': 0.2,
            'Healthcare': 0.1,
            'Finance': 0.1
        },
        'stress_test_results': {
            'market_crash': -0.08,
            'liquidity_crisis': -0.05,
            'correlation_spike': -0.03
        }
    }

@pytest.fixture(autouse=True)
def setup_test_logging():
    """Setup logging for tests."""
    import logging
    logging.getLogger().setLevel(logging.WARNING)  # Reduce log noise during tests

@pytest.fixture
def performance_benchmark():
    """Performance benchmarking utilities."""
    import time
    
    class PerformanceBenchmark:
        def __init__(self):
            self.start_time = None
            self.end_time = None
        
        def start(self):
            self.start_time = time.perf_counter()
        
        def stop(self):
            self.end_time = time.perf_counter()
            return self.elapsed_time
        
        @property
        def elapsed_time(self):
            if self.start_time and self.end_time:
                return self.end_time - self.start_time
            return None
        
        def assert_performance(self, max_time_seconds: float):
            """Assert that operation completed within time limit."""
            assert self.elapsed_time is not None, "Benchmark not completed"
            assert self.elapsed_time <= max_time_seconds, \
                f"Operation took {self.elapsed_time:.4f}s, expected <= {max_time_seconds}s"
    
    return PerformanceBenchmark()

# Pytest markers for test categorization
def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line("markers", "unit: Unit tests")
    config.addinivalue_line("markers", "integration: Integration tests")
    config.addinivalue_line("markers", "performance: Performance tests")
    config.addinivalue_line("markers", "arm64: ARM64 optimization tests")
    config.addinivalue_line("markers", "slow: Slow running tests")
    config.addinivalue_line("markers", "gpu: Tests requiring GPU")
    config.addinivalue_line("markers", "network: Tests requiring network access")

# Test collection hooks
def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers based on test names."""
    for item in items:
        # Add markers based on test file location
        if "unit" in str(item.fspath):
            item.add_marker(pytest.mark.unit)
        elif "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
        
        # Add markers based on test names
        if "performance" in item.name or "benchmark" in item.name:
            item.add_marker(pytest.mark.performance)
        
        if "arm64" in item.name:
            item.add_marker(pytest.mark.arm64)
        
        if "slow" in item.name or "stress" in item.name:
            item.add_marker(pytest.mark.slow)

# Async test utilities
@pytest.fixture
async def async_test_timeout():
    """Provide timeout for async tests."""
    return 30.0  # 30 second timeout for async tests

# Database fixtures for integration tests
@pytest.fixture
def test_database_url(temp_dir):
    """Provide test database URL."""
    return f"sqlite:///{temp_dir}/test.db"

# Mock external services
@pytest.fixture
def mock_external_services():
    """Mock all external service dependencies."""
    mocks = {}
    
    # Mock Redis
    with patch('redis.Redis') as mock_redis:
        mock_redis.return_value.ping.return_value = True
        mock_redis.return_value.get.return_value = None
        mock_redis.return_value.set.return_value = True
        mocks['redis'] = mock_redis
        
        # Mock other external services as needed
        yield mocks

# ARM64 specific fixtures
@pytest.fixture
def arm64_optimization_config():
    """Configuration for ARM64 optimization tests."""
    return {
        'enable_arm64_optimizations': True,
        'enable_mixed_precision': True,
        'enable_cuda_graphs': True,
        'enable_torchscript': True,
        'optimization_level': 'aggressive'
    }

# Test data validation utilities
@pytest.fixture
def data_validator():
    """Data validation utilities for tests."""
    class DataValidator:
        @staticmethod
        def validate_market_data(df: pd.DataFrame):
            """Validate market data format."""
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            assert all(col in df.columns for col in required_columns)
            assert df.index.name == 'timestamp' or 'timestamp' in df.columns
            assert len(df) > 0
            assert not df.isnull().all().any()
        
        @staticmethod
        def validate_features(df: pd.DataFrame):
            """Validate feature data format."""
            assert len(df) > 0
            assert df.select_dtypes(include=[np.number]).shape[1] > 0
        
        @staticmethod
        def validate_predictions(predictions: Dict):
            """Validate prediction format."""
            required_keys = ['position', 'confidence']
            for symbol, pred in predictions.items():
                assert all(key in pred for key in required_keys)
                assert -1 <= pred['position'] <= 1
                assert 0 <= pred['confidence'] <= 1
    
    return DataValidator()
