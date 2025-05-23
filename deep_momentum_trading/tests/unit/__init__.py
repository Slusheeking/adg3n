"""
Deep Momentum Trading - Unit Tests

This package contains unit tests for individual components of the Deep Momentum Trading system.
Unit tests focus on testing isolated functionality without external dependencies.

Test Modules:
- test_data/: Data processing and feature engineering tests
- test_models/: Machine learning model tests
- test_risk/: Risk management component tests
- test_storage/: Data storage backend tests
- test_trading/: Trading execution component tests

Testing Approach:
- Mock external dependencies (APIs, databases, file systems)
- Test individual functions and classes in isolation
- Verify edge cases and error handling
- Ensure proper input validation and output formatting
- Test performance characteristics for critical components

Test Coverage Goals:
- Minimum 90% code coverage for all modules
- 100% coverage for critical trading and risk management functions
- Comprehensive edge case testing
- Performance regression testing

Usage:
    Run all unit tests:
        pytest tests/unit/
    
    Run specific module tests:
        pytest tests/unit/test_data/
        pytest tests/unit/test_models/
        pytest tests/unit/test_risk/
    
    Run with coverage reporting:
        pytest tests/unit/ --cov=src/ --cov-report=html
"""

__version__ = "1.0.0"
__author__ = "Deep Momentum Trading Team"

# Unit test configuration
UNIT_TEST_CONFIG = {
    "mock_external_apis": True,
    "use_test_data": True,
    "enable_performance_tests": True,
    "test_timeout_seconds": 60,
    "coverage_threshold": 0.90
}

# Test data paths
TEST_DATA_PATHS = {
    "sample_market_data": "tests/fixtures/sample_market_data.csv",
    "test_predictions": "tests/fixtures/test_predictions.json",
    "mock_portfolio": "tests/fixtures/mock_portfolio.json",
    "test_configs": "tests/fixtures/test_configs.py"
}

# Mock configuration for external services
MOCK_SERVICES = {
    "polygon_api": True,
    "alpaca_api": True,
    "external_data_feeds": True,
    "email_notifications": True,
    "file_storage": False  # Use real file operations for testing
}