"""
Unit tests for VaR Calculator.

Tests the Value at Risk calculation system with multiple methodologies,
Monte Carlo simulation, and ARM64 performance optimizations.
"""

import pytest
import numpy as np
import pandas as pd
import time
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any

from deep_momentum_trading.src.risk.var_calculator import (
    VaRCalculator,
    VaRConfig,
    VaRResult,
    VaRMethod,
    VaRTimeHorizon
)
from deep_momentum_trading.tests.fixtures.sample_data import (
    get_sample_market_data,
    get_sample_correlation_matrix,
    get_sample_portfolio
)
from deep_momentum_trading.tests.fixtures.test_configs import TestRiskConfig


class TestVaRConfig:
    """Test VaRConfig class."""
    
    def test_default_config(self):
        """Test default VaR configuration."""
        config = VaRConfig()
        
        assert config.confidence_level == 0.95
        assert config.time_horizon == VaRTimeHorizon.DAILY
        assert config.method == VaRMethod.HISTORICAL
        assert config.lookback_days == 252
        assert config.monte_carlo_simulations == 10000
        assert config.enable_expected_shortfall is True
        assert config.enable_component_var is True
        assert config.enable_marginal_var is True
        assert config.enable_stress_testing is True
        assert config.enable_backtesting is True
        assert config.backtesting_window == 250
        assert config.enable_arm64_optimizations is True
        assert config.calculation_timeout == 30.0
        assert config.enable_parallel_processing is True
        assert config.max_workers == 4
    
    def test_custom_config(self):
        """Test custom VaR configuration."""
        config = VaRConfig(
            confidence_level=0.99,
            method=VaRMethod.MONTE_CARLO,
            lookback_days=500,
            monte_carlo_simulations=50000,
            enable_parallel_processing=False
        )
        
        assert config.confidence_level == 0.99
        assert config.method == VaRMethod.MONTE_CARLO
        assert config.lookback_days == 500
        assert config.monte_carlo_simulations == 50000
        assert config.enable_parallel_processing is False


class TestVaRResult:
    """Test VaRResult class."""
    
    def test_var_result_creation(self):
        """Test creating VaR result."""
        result = VaRResult(
            portfolio_var=0.025,
            expected_shortfall=0.035,
            component_var={"AAPL": 0.01, "MSFT": 0.008},
            marginal_var={"AAPL": 0.012, "MSFT": 0.009},
            method=VaRMethod.HISTORICAL,
            confidence_level=0.95,
            time_horizon=VaRTimeHorizon.DAILY,
            calculation_time=1.5,
            stress_test_results={"market_crash": -0.08},
            backtest_results={"violations": 12, "total_observations": 250}
        )
        
        assert result.portfolio_var == 0.025
        assert result.expected_shortfall == 0.035
        assert result.component_var["AAPL"] == 0.01
        assert result.marginal_var["MSFT"] == 0.009
        assert result.method == VaRMethod.HISTORICAL
        assert result.confidence_level == 0.95
        assert result.time_horizon == VaRTimeHorizon.DAILY
        assert result.calculation_time == 1.5
        assert result.stress_test_results["market_crash"] == -0.08
        assert result.backtest_results["violations"] == 12
        assert isinstance(result.timestamp, datetime)


class TestVaRCalculator:
    """Test VaRCalculator class."""
    
    @pytest.fixture
    def test_config(self):
        """Get test VaR configuration."""
        return VaRConfig(
            confidence_level=0.95,
            lookback_days=100,  # Shorter for testing
            monte_carlo_simulations=1000,  # Fewer for testing
            calculation_timeout=10.0,
            enable_arm64_optimizations=True
        )
    
    @pytest.fixture
    def sample_data(self):
        """Generate sample VaR calculation data."""
        symbols = ["AAPL", "MSFT", "GOOGL", "TSLA", "NVDA"]
        
        # Generate historical returns
        returns_data = {}
        for symbol in symbols:
            market_data = get_sample_market_data(symbol=symbol, n_records=252)
            returns = market_data['close'].pct_change().dropna()
            returns_data[symbol] = returns
        
        returns_df = pd.DataFrame(returns_data)
        
        # Portfolio weights
        weights = {"AAPL": 0.3, "MSFT": 0.25, "GOOGL": 0.2, "TSLA": 0.15, "NVDA": 0.1}
        
        # Portfolio value
        portfolio_value = 1000000.0  # $1M portfolio
        
        return {
            'returns': returns_df,
            'weights': weights,
            'portfolio_value': portfolio_value,
            'symbols': symbols
        }
    
    @pytest.fixture
    def var_calculator(self, test_config):
        """Create VaRCalculator instance."""
        return VaRCalculator(config=test_config)
    
    def test_initialization(self, test_config):
        """Test VaRCalculator initialization."""
        calculator = VaRCalculator(config=test_config)
        
        assert calculator.config == test_config
        assert calculator.calculation_history == []
        assert calculator.performance_stats == {
            'total_calculations': 0,
            'successful_calculations': 0,
            'failed_calculations': 0,
            'avg_calculation_time': 0.0,
            'arm64_optimizations_used': 0
        }
    
    def test_initialization_with_dict_config(self):
        """Test initialization with dictionary configuration."""
        config_dict = {
            'confidence_level': 0.99,
            'method': 'parametric',
            'lookback_days': 300
        }
        
        calculator = VaRCalculator(config=config_dict)
        
        assert calculator.config.confidence_level == 0.99
        assert calculator.config.method == VaRMethod.PARAMETRIC
        assert calculator.config.lookback_days == 300
    
    @pytest.mark.arm64
    def test_arm64_optimizations(self, test_config):
        """Test ARM64 optimization detection."""
        with patch('deep_momentum_trading.src.risk.var_calculator.IS_ARM64', True):
            calculator = VaRCalculator(config=test_config)
            
            assert calculator.is_arm64 is True
            # ARM64 should enable optimizations
            assert calculator.config.enable_parallel_processing is True
            assert calculator.config.max_workers >= 4
    
    def test_validate_inputs(self, var_calculator, sample_data):
        """Test input validation."""
        # Valid inputs
        is_valid, message = var_calculator._validate_inputs(
            sample_data['returns'],
            sample_data['weights'],
            sample_data['portfolio_value']
        )
        assert is_valid is True
        assert message == ""
        
        # Invalid portfolio value
        is_valid, message = var_calculator._validate_inputs(
            sample_data['returns'],
            sample_data['weights'],
            -1000.0  # Negative value
        )
        assert is_valid is False
        assert "portfolio value" in message.lower()
        
        # Mismatched weights
        bad_weights = {"AAPL": 0.5, "UNKNOWN": 0.5}
        is_valid, message = var_calculator._validate_inputs(
            sample_data['returns'],
            bad_weights,
            sample_data['portfolio_value']
        )
        assert is_valid is False
        assert "weight" in message.lower()
        
        # Weights don't sum to 1
        bad_weights = {"AAPL": 0.3, "MSFT": 0.3}  # Sum = 0.6
        is_valid, message = var_calculator._validate_inputs(
            sample_data['returns'],
            bad_weights,
            sample_data['portfolio_value']
        )
        assert is_valid is False
        assert "sum to 1" in message.lower()
    
    def test_calculate_portfolio_returns(self, var_calculator, sample_data):
        """Test portfolio returns calculation."""
        portfolio_returns = var_calculator._calculate_portfolio_returns(
            sample_data['returns'],
            sample_data['weights']
        )
        
        assert isinstance(portfolio_returns, pd.Series)
        assert len(portfolio_returns) == len(sample_data['returns'])
        assert not portfolio_returns.isnull().all()
        
        # Check that returns are reasonable
        assert portfolio_returns.std() > 0
        assert -0.2 < portfolio_returns.min() < 0.2  # Reasonable daily return range
    
    def test_historical_var(self, var_calculator, sample_data):
        """Test historical VaR calculation."""
        var_calculator.config.method = VaRMethod.HISTORICAL
        
        result = var_calculator.calculate_portfolio_var(
            sample_data['returns'],
            sample_data['weights'],
            sample_data['portfolio_value']
        )
        
        assert isinstance(result, VaRResult)
        assert result.method == VaRMethod.HISTORICAL
        assert result.portfolio_var > 0
        assert result.confidence_level == var_calculator.config.confidence_level
        assert result.expected_shortfall > result.portfolio_var  # ES should be higher than VaR
    
    def test_parametric_var(self, var_calculator, sample_data):
        """Test parametric VaR calculation."""
        var_calculator.config.method = VaRMethod.PARAMETRIC
        
        result = var_calculator.calculate_portfolio_var(
            sample_data['returns'],
            sample_data['weights'],
            sample_data['portfolio_value']
        )
        
        assert isinstance(result, VaRResult)
        assert result.method == VaRMethod.PARAMETRIC
        assert result.portfolio_var > 0
        assert result.expected_shortfall > 0
    
    def test_monte_carlo_var(self, var_calculator, sample_data):
        """Test Monte Carlo VaR calculation."""
        var_calculator.config.method = VaRMethod.MONTE_CARLO
        var_calculator.config.monte_carlo_simulations = 1000  # Reduce for testing
        
        with patch('numpy.random.multivariate_normal') as mock_random:
            # Mock random number generation for consistent testing
            mock_returns = np.random.normal(0, 0.02, (1000, len(sample_data['symbols'])))
            mock_random.return_value = mock_returns
            
            result = var_calculator.calculate_portfolio_var(
                sample_data['returns'],
                sample_data['weights'],
                sample_data['portfolio_value']
            )
            
            assert isinstance(result, VaRResult)
            assert result.method == VaRMethod.MONTE_CARLO
            assert result.portfolio_var > 0
    
    def test_component_var_calculation(self, var_calculator, sample_data):
        """Test component VaR calculation."""
        var_calculator.config.enable_component_var = True
        
        result = var_calculator.calculate_portfolio_var(
            sample_data['returns'],
            sample_data['weights'],
            sample_data['portfolio_value']
        )
        
        assert result.component_var is not None
        assert len(result.component_var) == len(sample_data['weights'])
        
        # Component VaRs should sum approximately to portfolio VaR
        total_component_var = sum(result.component_var.values())
        assert abs(total_component_var - result.portfolio_var) < result.portfolio_var * 0.1
    
    def test_marginal_var_calculation(self, var_calculator, sample_data):
        """Test marginal VaR calculation."""
        var_calculator.config.enable_marginal_var = True
        
        result = var_calculator.calculate_portfolio_var(
            sample_data['returns'],
            sample_data['weights'],
            sample_data['portfolio_value']
        )
        
        assert result.marginal_var is not None
        assert len(result.marginal_var) == len(sample_data['weights'])
        
        # All marginal VaRs should be positive
        for symbol, mvar in result.marginal_var.items():
            assert mvar > 0
    
    def test_expected_shortfall_calculation(self, var_calculator, sample_data):
        """Test Expected Shortfall (CVaR) calculation."""
        var_calculator.config.enable_expected_shortfall = True
        
        result = var_calculator.calculate_portfolio_var(
            sample_data['returns'],
            sample_data['weights'],
            sample_data['portfolio_value']
        )
        
        assert result.expected_shortfall is not None
        assert result.expected_shortfall > result.portfolio_var  # ES should be higher than VaR
        assert result.expected_shortfall > 0
    
    def test_stress_testing(self, var_calculator, sample_data):
        """Test stress testing functionality."""
        var_calculator.config.enable_stress_testing = True
        
        result = var_calculator.calculate_portfolio_var(
            sample_data['returns'],
            sample_data['weights'],
            sample_data['portfolio_value']
        )
        
        assert result.stress_test_results is not None
        assert len(result.stress_test_results) > 0
        
        # Stress test results should be negative (losses)
        for scenario, loss in result.stress_test_results.items():
            assert loss < 0
    
    def test_backtesting(self, var_calculator, sample_data):
        """Test VaR backtesting."""
        var_calculator.config.enable_backtesting = True
        var_calculator.config.backtesting_window = 50  # Shorter for testing
        
        result = var_calculator.calculate_portfolio_var(
            sample_data['returns'],
            sample_data['weights'],
            sample_data['portfolio_value']
        )
        
        assert result.backtest_results is not None
        assert 'violations' in result.backtest_results
        assert 'total_observations' in result.backtest_results
        assert 'violation_rate' in result.backtest_results
        
        # Violation rate should be reasonable
        violation_rate = result.backtest_results['violation_rate']
        expected_rate = 1 - var_calculator.config.confidence_level
        assert 0 <= violation_rate <= 0.2  # Should be reasonable
    
    def test_different_confidence_levels(self, var_calculator, sample_data):
        """Test VaR calculation with different confidence levels."""
        confidence_levels = [0.90, 0.95, 0.99]
        var_results = []
        
        for confidence in confidence_levels:
            var_calculator.config.confidence_level = confidence
            
            result = var_calculator.calculate_portfolio_var(
                sample_data['returns'],
                sample_data['weights'],
                sample_data['portfolio_value']
            )
            
            var_results.append(result.portfolio_var)
        
        # Higher confidence should result in higher VaR
        assert var_results[0] < var_results[1] < var_results[2]
    
    def test_different_time_horizons(self, var_calculator, sample_data):
        """Test VaR calculation with different time horizons."""
        horizons = [VaRTimeHorizon.DAILY, VaRTimeHorizon.WEEKLY, VaRTimeHorizon.MONTHLY]
        var_results = []
        
        for horizon in horizons:
            var_calculator.config.time_horizon = horizon
            
            result = var_calculator.calculate_portfolio_var(
                sample_data['returns'],
                sample_data['weights'],
                sample_data['portfolio_value']
            )
            
            var_results.append(result.portfolio_var)
        
        # Longer horizons should generally result in higher VaR
        assert var_results[0] < var_results[2]  # Daily < Monthly
    
    def test_calculate_with_invalid_inputs(self, var_calculator):
        """Test VaR calculation with invalid inputs."""
        # Empty returns
        empty_returns = pd.DataFrame()
        weights = {"AAPL": 1.0}
        portfolio_value = 100000.0
        
        result = var_calculator.calculate_portfolio_var(empty_returns, weights, portfolio_value)
        assert result is None
        
        # Invalid weights
        returns = pd.DataFrame({"AAPL": [0.01, 0.02, -0.01]})
        invalid_weights = {"MSFT": 1.0}  # Symbol not in returns
        
        result = var_calculator.calculate_portfolio_var(returns, invalid_weights, portfolio_value)
        assert result is None
    
    def test_performance_tracking(self, var_calculator, sample_data):
        """Test performance statistics tracking."""
        # Initial stats
        stats = var_calculator.get_performance_stats()
        assert stats['total_calculations'] == 0
        
        # Perform calculation
        result = var_calculator.calculate_portfolio_var(
            sample_data['returns'],
            sample_data['weights'],
            sample_data['portfolio_value']
        )
        
        # Updated stats
        updated_stats = var_calculator.get_performance_stats()
        assert updated_stats['total_calculations'] == 1
        assert updated_stats['successful_calculations'] == 1
        assert updated_stats['avg_calculation_time'] > 0
    
    def test_calculation_history(self, var_calculator, sample_data):
        """Test calculation history tracking."""
        # Perform calculation
        result = var_calculator.calculate_portfolio_var(
            sample_data['returns'],
            sample_data['weights'],
            sample_data['portfolio_value']
        )
        
        # Check history
        history = var_calculator.get_calculation_history()
        assert len(history) == 1
        assert history[0] == result
    
    @pytest.mark.performance
    def test_calculation_performance(self, var_calculator, sample_data):
        """Test VaR calculation performance."""
        # Benchmark calculation time
        start_time = time.perf_counter()
        
        # Run multiple calculations
        for _ in range(5):
            var_calculator.calculate_portfolio_var(
                sample_data['returns'],
                sample_data['weights'],
                sample_data['portfolio_value']
            )
        
        end_time = time.perf_counter()
        avg_time = (end_time - start_time) / 5
        
        # Should complete within reasonable time
        assert avg_time < 2.0  # Less than 2 seconds per calculation
    
    def test_parallel_processing(self, var_calculator, sample_data):
        """Test parallel processing for Monte Carlo simulation."""
        var_calculator.config.method = VaRMethod.MONTE_CARLO
        var_calculator.config.enable_parallel_processing = True
        var_calculator.config.monte_carlo_simulations = 5000
        
        with patch('concurrent.futures.ProcessPoolExecutor') as mock_executor:
            # Mock parallel execution
            mock_executor.return_value.__enter__.return_value.map.return_value = [
                np.random.normal(0, 0.02, 1000) for _ in range(5)
            ]
            
            result = var_calculator.calculate_portfolio_var(
                sample_data['returns'],
                sample_data['weights'],
                sample_data['portfolio_value']
            )
            
            assert isinstance(result, VaRResult)
            # Verify parallel processing was attempted
            mock_executor.assert_called_once()
    
    def test_clear_history(self, var_calculator, sample_data):
        """Test clearing calculation history."""
        # Perform calculation to create history
        var_calculator.calculate_portfolio_var(
            sample_data['returns'],
            sample_data['weights'],
            sample_data['portfolio_value']
        )
        
        assert len(var_calculator.calculation_history) == 1
        
        # Clear history
        var_calculator.clear_history()
        assert len(var_calculator.calculation_history) == 0
    
    def test_get_var_breakdown(self, var_calculator, sample_data):
        """Test VaR breakdown by asset."""
        var_calculator.config.enable_component_var = True
        var_calculator.config.enable_marginal_var = True
        
        result = var_calculator.calculate_portfolio_var(
            sample_data['returns'],
            sample_data['weights'],
            sample_data['portfolio_value']
        )
        
        breakdown = var_calculator.get_var_breakdown(result)
        
        assert 'portfolio_var' in breakdown
        assert 'component_var' in breakdown
        assert 'marginal_var' in breakdown
        assert 'weights' in breakdown
        
        # Check that all symbols are included
        for symbol in sample_data['symbols']:
            assert symbol in breakdown['component_var']
            assert symbol in breakdown['marginal_var']
            assert symbol in breakdown['weights']


@pytest.mark.integration
class TestVaRCalculatorIntegration:
    """Integration tests for VaR calculator."""
    
    def test_realistic_var_calculation(self):
        """Test realistic VaR calculation scenario."""
        # Create realistic portfolio
        symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
        
        # Generate correlated returns
        returns_data = {}
        correlation_matrix = get_sample_correlation_matrix(symbols)
        
        # Generate returns with realistic correlation structure
        np.random.seed(42)  # For reproducible results
        mean_returns = np.array([0.0008, 0.0006, 0.0005, 0.0009, 0.0012])  # Daily returns
        volatilities = np.array([0.025, 0.022, 0.028, 0.030, 0.045])  # Daily volatilities
        
        # Create covariance matrix
        cov_matrix = np.outer(volatilities, volatilities) * correlation_matrix.values
        
        # Generate correlated returns
        n_days = 500
        returns_matrix = np.random.multivariate_normal(mean_returns, cov_matrix, n_days)
        
        for i, symbol in enumerate(symbols):
            returns_data[symbol] = returns_matrix[:, i]
        
        returns_df = pd.DataFrame(returns_data, index=pd.date_range('2023-01-01', periods=n_days))
        
        # Portfolio weights
        weights = {"AAPL": 0.3, "MSFT": 0.25, "GOOGL": 0.2, "AMZN": 0.15, "TSLA": 0.1}
        portfolio_value = 5000000.0  # $5M portfolio
        
        # Test different VaR methods
        methods = [VaRMethod.HISTORICAL, VaRMethod.PARAMETRIC, VaRMethod.MONTE_CARLO]
        
        for method in methods:
            config = VaRConfig(
                method=method,
                confidence_level=0.95,
                lookback_days=252,
                monte_carlo_simulations=5000 if method == VaRMethod.MONTE_CARLO else 1000,
                enable_component_var=True,
                enable_expected_shortfall=True,
                enable_stress_testing=True
            )
            
            calculator = VaRCalculator(config=config)
            
            if method == VaRMethod.MONTE_CARLO:
                with patch('numpy.random.multivariate_normal') as mock_random:
                    # Mock for consistent testing
                    mock_random.return_value = np.random.normal(0, 0.02, (5000, len(symbols)))
                    
                    result = calculator.calculate_portfolio_var(returns_df, weights, portfolio_value)
            else:
                result = calculator.calculate_portfolio_var(returns_df, weights, portfolio_value)
            
            # Verify result quality
            assert isinstance(result, VaRResult)
            assert result.portfolio_var > 0
            assert result.portfolio_var < portfolio_value * 0.2  # Reasonable upper bound
            assert result.expected_shortfall > result.portfolio_var
            
            # Verify component VaR
            assert len(result.component_var) == len(weights)
            component_sum = sum(result.component_var.values())
            assert abs(component_sum - result.portfolio_var) < result.portfolio_var * 0.15
            
            # Verify stress test results
            assert len(result.stress_test_results) > 0
            for scenario_loss in result.stress_test_results.values():
                assert scenario_loss < 0  # Should be losses


if __name__ == "__main__":
    pytest.main([__file__])
