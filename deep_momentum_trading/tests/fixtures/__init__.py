"""
Test Fixtures Package for Deep Momentum Trading System
=====================================================

This package provides comprehensive test fixtures, sample data generators,
and configuration utilities for testing all components of the Deep Momentum
Trading System with ARM64/GH200 optimizations.

Modules:
--------
- sample_data: Realistic market data and model prediction generators
- test_configs: Comprehensive test configuration management
"""

from .sample_data import (
    MarketDataConfig,
    SampleDataGenerator,
    get_sample_market_data,
    get_sample_features_data,
    get_sample_multi_symbol_data,
    get_sample_predictions,
    get_sample_portfolio,
    get_sample_torch_data
)

from .test_configs import (
    TestModelConfig,
    TestRiskConfig,
    TestFeatureConfig,
    TestTradingConfig,
    TestDataConfig,
    TestCommunicationConfig,
    TestMonitoringConfig,
    TestInfrastructureConfig,
    TestStorageConfig,
    TestTrainingConfig,
    TestConfigManager,
    EnhancedTestConfigManager,
    TestScenarios,
    TestEnvironments,
    validate_test_config
)

# Version information
__version__ = "1.0.0"
__author__ = "Deep Momentum Trading Team"

# Export all public APIs
__all__ = [
    # Sample Data Components
    'MarketDataConfig',
    'SampleDataGenerator',
    'get_sample_market_data',
    'get_sample_features_data',
    'get_sample_multi_symbol_data',
    'get_sample_predictions',
    'get_sample_portfolio',
    'get_sample_torch_data',
    'get_sample_risk_scenarios',
    'get_sample_correlation_matrix',
    'get_sample_time_series_data',
    'get_sample_order_book_data',
    'get_sample_trade_data',
    
    # Test Configuration Components
    'TestModelConfig',
    'TestRiskConfig',
    'TestFeatureConfig',
    'TestTradingConfig',
    'TestDataConfig',
    'TestCommunicationConfig',
    'TestMonitoringConfig',
    'TestInfrastructureConfig',
    'TestStorageConfig',
    'TestTrainingConfig',
    'TestConfigManager',
    'EnhancedTestConfigManager',
    'TestScenarios',
    'TestEnvironments',
    'validate_test_config',
]

# Convenience functions for quick test setup
def get_minimal_test_setup():
    """Get minimal test setup with sample data and config."""
    config_manager = TestScenarios.get_minimal_config()
    sample_data = get_sample_market_data(n_records=100)
    return config_manager, sample_data

def get_performance_test_setup():
    """Get performance test setup with larger datasets."""
    config_manager = TestScenarios.get_performance_config()
    sample_data = get_sample_multi_symbol_data(["AAPL", "MSFT", "GOOGL"])
    return config_manager, sample_data

def get_integration_test_setup():
    """Get integration test setup with full configuration."""
    config_manager = TestScenarios.get_integration_config()
    market_data = get_sample_multi_symbol_data(["AAPL", "MSFT", "GOOGL", "TSLA", "NVDA"])
    predictions = get_sample_predictions(n_predictions=50)
    portfolio = get_sample_portfolio(total_value=50000)
    
    return {
        'config': config_manager,
        'market_data': market_data,
        'predictions': predictions,
        'portfolio': portfolio
    }

# Add convenience functions to __all__
__all__.extend([
    'get_minimal_test_setup',
    'get_performance_test_setup', 
    'get_integration_test_setup'
])