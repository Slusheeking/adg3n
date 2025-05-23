"""
Deep Momentum Trading - Test Suite

This package contains comprehensive tests for the Deep Momentum Trading system,
including unit tests, integration tests, and end-to-end testing scenarios.

Test Structure:
- unit/: Unit tests for individual components
- integration/: Integration tests for component interactions
- fixtures/: Test data and configuration fixtures

Test Categories:
- Data pipeline testing (data ingestion, preprocessing, feature engineering)
- Model testing (LSTM, Transformer, ensemble systems)
- Trading engine testing (order management, execution, position tracking)
- Risk management testing (VaR, portfolio optimization, correlation monitoring)
- Infrastructure testing (communication, storage, monitoring)

Usage:
    Run all tests:
        pytest tests/
    
    Run specific test categories:
        pytest tests/unit/
        pytest tests/integration/
    
    Run with coverage:
        pytest tests/ --cov=src/
"""

__version__ = "1.0.0"
__author__ = "Deep Momentum Trading Team"

# Test configuration
TEST_CONFIG = {
    "enable_integration_tests": True,
    "enable_performance_tests": True,
    "test_data_path": "tests/fixtures/",
    "mock_external_apis": True,
    "test_timeout_seconds": 300
}

# Test markers for pytest
PYTEST_MARKERS = [
    "unit: Unit tests for individual components",
    "integration: Integration tests for component interactions", 
    "performance: Performance and load testing",
    "slow: Tests that take longer than 30 seconds",
    "external: Tests requiring external API connections"
]