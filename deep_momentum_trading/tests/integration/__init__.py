"""
Integration tests for Deep Momentum Trading System.

This module contains integration tests that verify the interaction between
different components of the trading system, including data flow, communication,
and end-to-end trading pipeline functionality.
"""

__version__ = "1.0.0"
__author__ = "Deep Momentum Trading Team"

# Integration test categories
INTEGRATION_TEST_CATEGORIES = [
    "communication",
    "data_pipeline", 
    "trading_pipeline",
    "end_to_end"
]

# Test configuration constants
DEFAULT_TEST_TIMEOUT = 30.0
DEFAULT_INTEGRATION_SYMBOLS = ["AAPL", "MSFT", "GOOGL", "TSLA", "NVDA"]
DEFAULT_TEST_PORTFOLIO_VALUE = 1000000.0

# Integration test markers
INTEGRATION_MARKERS = {
    "slow": "Tests that take longer than 5 seconds",
    "external": "Tests that require external services",
    "network": "Tests that require network connectivity",
    "realtime": "Tests that simulate real-time data flow"
}
