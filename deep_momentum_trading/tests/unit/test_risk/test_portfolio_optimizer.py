"""
Unit tests for Portfolio Optimizer.

Tests the portfolio optimization system with modern portfolio theory,
risk-return optimization, and ARM64 performance enhancements.
"""

import pytest
import numpy as np
import pandas as pd
import time
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any

from deep_momentum_trading.src.risk.portfolio_optimizer import (
    PortfolioOptimizer,
    OptimizationConfig,
    OptimizationResult,
    OptimizationObjective,
    ConstraintType,
    OptimizationConstraint
)
from deep_momentum_trading.tests.fixtures.sample_data import (
    get_sample_correlation_matrix,
    get_sample_predictions,
    get_sample_portfolio
)
from deep_momentum_trading.tests.fixtures.test_configs import TestRiskConfig


class TestOptimizationConfig:
    """Test OptimizationConfig class."""
    
    def test_default_config(self):
        """Test default optimization configuration."""
        config = OptimizationConfig()
        
        assert config.objective == OptimizationObjective.SHARPE_RATIO
        assert config.max_position_weight == 0.1
        assert config.min_position_weight == -0.05
        assert config.max_turnover == 0.3
        assert config.transaction_cost == 0.001
        assert config.risk_aversion == 1.0
        assert config.lookback_days == 252
        assert config.rebalance_threshold == 0.05
        assert config.enable_long_only is False
        assert config.enable_sector_constraints is True
        assert config.max_sector_weight == 0.3
        assert config.enable_arm64_optimizations is True
        assert config.optimization_timeout == 30.0
        assert config.solver == 'ECOS'
    
    def test_custom_config(self):
        """Test custom optimization configuration."""
        config = OptimizationConfig(
            objective=OptimizationObjective.MIN_VARIANCE,
            max_position_weight=0.15,
            enable_long_only=True,
            risk_aversion=2.0,
            solver='OSQP'
        )
        
        assert config.objective == OptimizationObjective.MIN_VARIANCE
        assert config.max_position_weight == 0.15
        assert config.enable_long_only is True
        assert config.risk_aversion == 2.0
        assert config.solver == 'OSQP'


class TestOptimizationConstraint:
    """Test OptimizationConstraint class."""
    
    def test_constraint_creation(self):
        """Test creating optimization constraints."""
        constraint = OptimizationConstraint(
            constraint_type=ConstraintType.POSITION_LIMIT,
            symbols=["AAPL", "MSFT"],
            lower_bound=-0.05,
            upper_bound=0.1,
            description="Position limits for tech stocks"
        )
        
        assert constraint.constraint_type == ConstraintType.POSITION_LIMIT
        assert constraint.symbols == ["AAPL", "MSFT"]
        assert constraint.lower_bound == -0.05
        assert constraint.upper_bound == 0.1
        assert constraint.description == "Position limits for tech stocks"
    
    def test_sector_constraint(self):
        """Test sector constraint creation."""
        constraint = OptimizationConstraint(
            constraint_type=ConstraintType.SECTOR_LIMIT,
            symbols=["AAPL", "MSFT", "GOOGL"],
            upper_bound=0.4,
            description="Technology sector limit"
        )
        
        assert constraint.constraint_type == ConstraintType.SECTOR_LIMIT
        assert len(constraint.symbols) == 3
        assert constraint.upper_bound == 0.4


class TestOptimizationResult:
    """Test OptimizationResult class."""
    
    def test_result_creation(self):
        """Test creating optimization result."""
        weights = {"AAPL": 0.3, "MSFT": 0.2, "GOOGL": 0.15}
        
        result = OptimizationResult(
            optimal_weights=weights,
            expected_return=0.12,
            expected_volatility=0.18,
            sharpe_ratio=0.67,
            max_drawdown=0.08,
            turnover=0.25,
            optimization_time=1.5,
            solver_status="optimal",
            objective_value=0.67
        )
        
        assert result.optimal_weights == weights
        assert result.expected_return == 0.12
        assert result.expected_volatility == 0.18
        assert result.sharpe_ratio == 0.67
        assert result.max_drawdown == 0.08
        assert result.turnover == 0.25
        assert result.optimization_time == 1.5
        assert result.solver_status == "optimal"
        assert result.objective_value == 0.67
        assert isinstance(result.timestamp, datetime)


class TestPortfolioOptimizer:
    """Test PortfolioOptimizer class."""
    
    @pytest.fixture
    def test_config(self):
        """Get test optimization configuration."""
        return OptimizationConfig(
            max_position_weight=0.2,
            min_position_weight=-0.1,
            optimization_timeout=10.0,
            enable_arm64_optimizations=True
        )
    
    @pytest.fixture
    def sample_data(self):
        """Generate sample optimization data."""
        symbols = ["AAPL", "MSFT", "GOOGL", "TSLA", "NVDA"]
        
        # Expected returns
        expected_returns = pd.Series({
            "AAPL": 0.12,
            "MSFT": 0.10,
            "GOOGL": 0.08,
            "TSLA": 0.15,
            "NVDA": 0.18
        })
        
        # Covariance matrix
        correlation_matrix = get_sample_correlation_matrix(symbols)
        volatilities = pd.Series({
            "AAPL": 0.25,
            "MSFT": 0.22,
            "GOOGL": 0.28,
            "TSLA": 0.45,
            "NVDA": 0.40
        })
        
        # Convert correlation to covariance
        covariance_matrix = correlation_matrix.copy()
        for i, symbol_i in enumerate(symbols):
            for j, symbol_j in enumerate(symbols):
                covariance_matrix.iloc[i, j] *= volatilities[symbol_i] * volatilities[symbol_j]
        
        return {
            'expected_returns': expected_returns,
            'covariance_matrix': covariance_matrix,
            'symbols': symbols
        }
    
    @pytest.fixture
    def optimizer(self, test_config):
        """Create PortfolioOptimizer instance."""
        with patch('deep_momentum_trading.src.risk.portfolio_optimizer.cvxpy') as mock_cvxpy:
            # Mock cvxpy for testing
            mock_cvxpy.Variable = Mock()
            mock_cvxpy.Problem = Mock()
            mock_cvxpy.Maximize = Mock()
            mock_cvxpy.Minimize = Mock()
            
            return PortfolioOptimizer(config=test_config)
    
    def test_initialization(self, test_config):
        """Test PortfolioOptimizer initialization."""
        with patch('deep_momentum_trading.src.risk.portfolio_optimizer.cvxpy'):
            optimizer = PortfolioOptimizer(config=test_config)
            
            assert optimizer.config == test_config
            assert optimizer.optimization_history == []
            assert optimizer.current_weights == {}
            assert optimizer.performance_stats == {
                'total_optimizations': 0,
                'successful_optimizations': 0,
                'failed_optimizations': 0,
                'avg_optimization_time': 0.0,
                'arm64_optimizations_used': 0
            }
    
    def test_initialization_with_dict_config(self):
        """Test initialization with dictionary configuration."""
        config_dict = {
            'max_position_weight': 0.15,
            'risk_aversion': 1.5,
            'enable_long_only': True
        }
        
        with patch('deep_momentum_trading.src.risk.portfolio_optimizer.cvxpy'):
            optimizer = PortfolioOptimizer(config=config_dict)
            
            assert optimizer.config.max_position_weight == 0.15
            assert optimizer.config.risk_aversion == 1.5
            assert optimizer.config.enable_long_only is True
    
    @pytest.mark.arm64
    def test_arm64_optimizations(self, test_config):
        """Test ARM64 optimization detection."""
        with patch('deep_momentum_trading.src.risk.portfolio_optimizer.IS_ARM64', True), \
             patch('deep_momentum_trading.src.risk.portfolio_optimizer.cvxpy'):
            
            optimizer = PortfolioOptimizer(config=test_config)
            
            assert optimizer.is_arm64 is True
            # ARM64 should reduce timeout and enable parallel processing
            assert optimizer.config.optimization_timeout <= 30.0
    
    def test_validate_inputs(self, optimizer, sample_data):
        """Test input validation."""
        # Valid inputs
        is_valid, message = optimizer._validate_inputs(
            sample_data['expected_returns'],
            sample_data['covariance_matrix']
        )
        assert is_valid is True
        assert message == ""
        
        # Mismatched dimensions
        bad_returns = sample_data['expected_returns'].iloc[:3]  # Only 3 symbols
        is_valid, message = optimizer._validate_inputs(
            bad_returns,
            sample_data['covariance_matrix']  # 5 symbols
        )
        assert is_valid is False
        assert "dimension mismatch" in message.lower()
        
        # Non-positive definite covariance matrix
        bad_cov = sample_data['covariance_matrix'].copy()
        bad_cov.iloc[0, 0] = -1.0  # Make it non-positive definite
        is_valid, message = optimizer._validate_inputs(
            sample_data['expected_returns'],
            bad_cov
        )
        assert is_valid is False
        assert "positive definite" in message.lower()
    
    def test_create_constraints_basic(self, optimizer, sample_data):
        """Test basic constraint creation."""
        symbols = sample_data['symbols']
        n_assets = len(symbols)
        
        # Mock cvxpy Variable
        with patch('deep_momentum_trading.src.risk.portfolio_optimizer.cvxpy.Variable') as mock_var:
            mock_weights = Mock()
            mock_var.return_value = mock_weights
            
            constraints = optimizer._create_constraints(mock_weights, symbols)
            
            # Should have basic constraints
            assert len(constraints) >= 1  # At least budget constraint
    
    def test_create_constraints_with_limits(self, optimizer, sample_data):
        """Test constraint creation with position limits."""
        symbols = sample_data['symbols']
        
        # Add custom constraints
        optimizer.constraints = [
            OptimizationConstraint(
                constraint_type=ConstraintType.POSITION_LIMIT,
                symbols=["AAPL"],
                lower_bound=-0.05,
                upper_bound=0.15
            )
        ]
        
        with patch('deep_momentum_trading.src.risk.portfolio_optimizer.cvxpy.Variable') as mock_var:
            mock_weights = Mock()
            mock_var.return_value = mock_weights
            
            constraints = optimizer._create_constraints(mock_weights, symbols)
            
            # Should include custom constraints
            assert len(constraints) >= 2
    
    def test_optimize_sharpe_ratio(self, optimizer, sample_data):
        """Test Sharpe ratio optimization."""
        optimizer.config.objective = OptimizationObjective.SHARPE_RATIO
        
        # Mock cvxpy optimization
        with patch('deep_momentum_trading.src.risk.portfolio_optimizer.cvxpy') as mock_cvxpy:
            # Mock successful optimization
            mock_problem = Mock()
            mock_problem.solve.return_value = 1.5  # Optimal value
            mock_problem.status = "optimal"
            
            mock_weights = Mock()
            mock_weights.value = np.array([0.2, 0.3, 0.2, 0.15, 0.15])
            
            mock_cvxpy.Variable.return_value = mock_weights
            mock_cvxpy.Problem.return_value = mock_problem
            mock_cvxpy.Maximize = Mock()
            
            result = optimizer.optimize(
                sample_data['expected_returns'],
                sample_data['covariance_matrix']
            )
            
            assert isinstance(result, OptimizationResult)
            assert result.solver_status == "optimal"
            assert len(result.optimal_weights) == len(sample_data['symbols'])
            assert abs(sum(result.optimal_weights.values()) - 1.0) < 1e-6  # Budget constraint
    
    def test_optimize_min_variance(self, optimizer, sample_data):
        """Test minimum variance optimization."""
        optimizer.config.objective = OptimizationObjective.MIN_VARIANCE
        
        with patch('deep_momentum_trading.src.risk.portfolio_optimizer.cvxpy') as mock_cvxpy:
            mock_problem = Mock()
            mock_problem.solve.return_value = 0.05  # Minimum variance
            mock_problem.status = "optimal"
            
            mock_weights = Mock()
            mock_weights.value = np.array([0.25, 0.25, 0.25, 0.125, 0.125])
            
            mock_cvxpy.Variable.return_value = mock_weights
            mock_cvxpy.Problem.return_value = mock_problem
            mock_cvxpy.Minimize = Mock()
            
            result = optimizer.optimize(
                sample_data['expected_returns'],
                sample_data['covariance_matrix']
            )
            
            assert isinstance(result, OptimizationResult)
            assert result.solver_status == "optimal"
            assert result.expected_volatility > 0
    
    def test_optimize_max_return(self, optimizer, sample_data):
        """Test maximum return optimization."""
        optimizer.config.objective = OptimizationObjective.MAX_RETURN
        
        with patch('deep_momentum_trading.src.risk.portfolio_optimizer.cvxpy') as mock_cvxpy:
            mock_problem = Mock()
            mock_problem.solve.return_value = 0.15  # Maximum return
            mock_problem.status = "optimal"
            
            mock_weights = Mock()
            mock_weights.value = np.array([0.0, 0.0, 0.0, 0.5, 0.5])  # High return assets
            
            mock_cvxpy.Variable.return_value = mock_weights
            mock_cvxpy.Problem.return_value = mock_problem
            mock_cvxpy.Maximize = Mock()
            
            result = optimizer.optimize(
                sample_data['expected_returns'],
                sample_data['covariance_matrix']
            )
            
            assert isinstance(result, OptimizationResult)
            assert result.expected_return > 0.1  # Should be high
    
    def test_optimize_with_current_portfolio(self, optimizer, sample_data):
        """Test optimization with current portfolio (turnover constraint)."""
        # Set current portfolio
        current_portfolio = {"AAPL": 0.3, "MSFT": 0.2, "GOOGL": 0.2, "TSLA": 0.15, "NVDA": 0.15}
        
        with patch('deep_momentum_trading.src.risk.portfolio_optimizer.cvxpy') as mock_cvxpy:
            mock_problem = Mock()
            mock_problem.solve.return_value = 1.2
            mock_problem.status = "optimal"
            
            mock_weights = Mock()
            mock_weights.value = np.array([0.25, 0.25, 0.2, 0.15, 0.15])  # Small changes
            
            mock_cvxpy.Variable.return_value = mock_weights
            mock_cvxpy.Problem.return_value = mock_problem
            
            result = optimizer.optimize(
                sample_data['expected_returns'],
                sample_data['covariance_matrix'],
                current_portfolio=current_portfolio
            )
            
            assert isinstance(result, OptimizationResult)
            assert result.turnover >= 0.0  # Should calculate turnover
    
    def test_optimize_failure_handling(self, optimizer, sample_data):
        """Test optimization failure handling."""
        with patch('deep_momentum_trading.src.risk.portfolio_optimizer.cvxpy') as mock_cvxpy:
            # Mock failed optimization
            mock_problem = Mock()
            mock_problem.solve.return_value = float('inf')
            mock_problem.status = "infeasible"
            
            mock_cvxpy.Problem.return_value = mock_problem
            
            result = optimizer.optimize(
                sample_data['expected_returns'],
                sample_data['covariance_matrix']
            )
            
            assert result is None
            assert optimizer.performance_stats['failed_optimizations'] == 1
    
    def test_optimize_with_invalid_inputs(self, optimizer):
        """Test optimization with invalid inputs."""
        # Empty returns
        empty_returns = pd.Series(dtype=float)
        empty_cov = pd.DataFrame()
        
        result = optimizer.optimize(empty_returns, empty_cov)
        assert result is None
        
        # Mismatched dimensions
        returns = pd.Series([0.1, 0.12], index=["AAPL", "MSFT"])
        cov = pd.DataFrame(np.eye(3), index=["AAPL", "MSFT", "GOOGL"], columns=["AAPL", "MSFT", "GOOGL"])
        
        result = optimizer.optimize(returns, cov)
        assert result is None
    
    def test_calculate_portfolio_metrics(self, optimizer, sample_data):
        """Test portfolio metrics calculation."""
        weights = {"AAPL": 0.2, "MSFT": 0.3, "GOOGL": 0.2, "TSLA": 0.15, "NVDA": 0.15}
        
        metrics = optimizer._calculate_portfolio_metrics(
            weights,
            sample_data['expected_returns'],
            sample_data['covariance_matrix']
        )
        
        assert 'expected_return' in metrics
        assert 'expected_volatility' in metrics
        assert 'sharpe_ratio' in metrics
        assert metrics['expected_return'] > 0
        assert metrics['expected_volatility'] > 0
        assert isinstance(metrics['sharpe_ratio'], float)
    
    def test_calculate_turnover(self, optimizer):
        """Test turnover calculation."""
        current_weights = {"AAPL": 0.3, "MSFT": 0.2, "GOOGL": 0.2, "TSLA": 0.15, "NVDA": 0.15}
        new_weights = {"AAPL": 0.25, "MSFT": 0.25, "GOOGL": 0.2, "TSLA": 0.15, "NVDA": 0.15}
        
        turnover = optimizer._calculate_turnover(current_weights, new_weights)
        
        assert turnover >= 0.0
        assert turnover <= 2.0  # Maximum possible turnover
        
        # Test with identical weights
        turnover_zero = optimizer._calculate_turnover(current_weights, current_weights)
        assert turnover_zero == 0.0
    
    def test_rebalance_check(self, optimizer):
        """Test rebalance threshold check."""
        current_weights = {"AAPL": 0.3, "MSFT": 0.2, "GOOGL": 0.2, "TSLA": 0.15, "NVDA": 0.15}
        
        # Small change - should not rebalance
        small_change_weights = {"AAPL": 0.31, "MSFT": 0.19, "GOOGL": 0.2, "TSLA": 0.15, "NVDA": 0.15}
        should_rebalance = optimizer.should_rebalance(current_weights, small_change_weights)
        assert should_rebalance is False
        
        # Large change - should rebalance
        large_change_weights = {"AAPL": 0.4, "MSFT": 0.1, "GOOGL": 0.2, "TSLA": 0.15, "NVDA": 0.15}
        should_rebalance = optimizer.should_rebalance(current_weights, large_change_weights)
        assert should_rebalance is True
    
    def test_get_optimization_history(self, optimizer, sample_data):
        """Test optimization history tracking."""
        # Perform optimization
        with patch('deep_momentum_trading.src.risk.portfolio_optimizer.cvxpy') as mock_cvxpy:
            mock_problem = Mock()
            mock_problem.solve.return_value = 1.5
            mock_problem.status = "optimal"
            
            mock_weights = Mock()
            mock_weights.value = np.array([0.2, 0.3, 0.2, 0.15, 0.15])
            
            mock_cvxpy.Variable.return_value = mock_weights
            mock_cvxpy.Problem.return_value = mock_problem
            
            result = optimizer.optimize(
                sample_data['expected_returns'],
                sample_data['covariance_matrix']
            )
            
            # Check history
            history = optimizer.get_optimization_history()
            assert len(history) == 1
            assert history[0] == result
    
    def test_get_performance_stats(self, optimizer, sample_data):
        """Test performance statistics."""
        # Initial stats
        stats = optimizer.get_performance_stats()
        assert stats['total_optimizations'] == 0
        
        # Perform optimization
        with patch('deep_momentum_trading.src.risk.portfolio_optimizer.cvxpy') as mock_cvxpy:
            mock_problem = Mock()
            mock_problem.solve.return_value = 1.5
            mock_problem.status = "optimal"
            
            mock_weights = Mock()
            mock_weights.value = np.array([0.2, 0.3, 0.2, 0.15, 0.15])
            
            mock_cvxpy.Variable.return_value = mock_weights
            mock_cvxpy.Problem.return_value = mock_problem
            
            optimizer.optimize(
                sample_data['expected_returns'],
                sample_data['covariance_matrix']
            )
            
            # Updated stats
            updated_stats = optimizer.get_performance_stats()
            assert updated_stats['total_optimizations'] == 1
            assert updated_stats['successful_optimizations'] == 1
    
    @pytest.mark.performance
    def test_optimization_performance(self, optimizer, sample_data):
        """Test optimization performance."""
        # Benchmark optimization time
        with patch('deep_momentum_trading.src.risk.portfolio_optimizer.cvxpy') as mock_cvxpy:
            mock_problem = Mock()
            mock_problem.solve.return_value = 1.5
            mock_problem.status = "optimal"
            
            mock_weights = Mock()
            mock_weights.value = np.array([0.2, 0.3, 0.2, 0.15, 0.15])
            
            mock_cvxpy.Variable.return_value = mock_weights
            mock_cvxpy.Problem.return_value = mock_problem
            
            start_time = time.perf_counter()
            
            # Run multiple optimizations
            for _ in range(10):
                optimizer.optimize(
                    sample_data['expected_returns'],
                    sample_data['covariance_matrix']
                )
            
            end_time = time.perf_counter()
            avg_time = (end_time - start_time) / 10
            
            # Should be fast with mocked solver
            assert avg_time < 0.1  # Less than 100ms per optimization
    
    def test_clear_history(self, optimizer, sample_data):
        """Test clearing optimization history."""
        # Perform optimization to create history
        with patch('deep_momentum_trading.src.risk.portfolio_optimizer.cvxpy') as mock_cvxpy:
            mock_problem = Mock()
            mock_problem.solve.return_value = 1.5
            mock_problem.status = "optimal"
            
            mock_weights = Mock()
            mock_weights.value = np.array([0.2, 0.3, 0.2, 0.15, 0.15])
            
            mock_cvxpy.Variable.return_value = mock_weights
            mock_cvxpy.Problem.return_value = mock_problem
            
            optimizer.optimize(
                sample_data['expected_returns'],
                sample_data['covariance_matrix']
            )
            
            assert len(optimizer.optimization_history) == 1
            
            # Clear history
            optimizer.clear_history()
            assert len(optimizer.optimization_history) == 0


@pytest.mark.integration
class TestPortfolioOptimizerIntegration:
    """Integration tests for portfolio optimizer."""
    
    def test_realistic_optimization_scenario(self):
        """Test realistic portfolio optimization scenario."""
        # Create realistic market scenario
        symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "META", "NFLX"]
        
        # Generate expected returns based on historical patterns
        expected_returns = pd.Series({
            "AAPL": 0.12, "MSFT": 0.11, "GOOGL": 0.09, "AMZN": 0.13,
            "TSLA": 0.20, "NVDA": 0.25, "META": 0.08, "NFLX": 0.10
        })
        
        # Create realistic correlation structure
        correlation_matrix = get_sample_correlation_matrix(symbols)
        
        # Add sector correlations (tech stocks more correlated)
        tech_stocks = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META"]
        for i, stock1 in enumerate(tech_stocks):
            for j, stock2 in enumerate(tech_stocks):
                if i != j:
                    correlation_matrix.loc[stock1, stock2] = 0.6  # Higher correlation
        
        # Convert to covariance matrix
        volatilities = pd.Series({
            "AAPL": 0.25, "MSFT": 0.22, "GOOGL": 0.28, "AMZN": 0.30,
            "TSLA": 0.50, "NVDA": 0.45, "META": 0.35, "NFLX": 0.40
        })
        
        covariance_matrix = correlation_matrix.copy()
        for i, symbol_i in enumerate(symbols):
            for j, symbol_j in enumerate(symbols):
                covariance_matrix.iloc[i, j] *= volatilities[symbol_i] * volatilities[symbol_j]
        
        # Test different optimization objectives
        objectives = [
            OptimizationObjective.SHARPE_RATIO,
            OptimizationObjective.MIN_VARIANCE,
            OptimizationObjective.MAX_RETURN
        ]
        
        for objective in objectives:
            config = OptimizationConfig(
                objective=objective,
                max_position_weight=0.3,
                enable_sector_constraints=True,
                max_sector_weight=0.6
            )
            
            with patch('deep_momentum_trading.src.risk.portfolio_optimizer.cvxpy') as mock_cvxpy:
                mock_problem = Mock()
                mock_problem.solve.return_value = 1.0
                mock_problem.status = "optimal"
                
                # Generate reasonable weights based on objective
                if objective == OptimizationObjective.MIN_VARIANCE:
                    weights = np.array([0.2, 0.2, 0.15, 0.15, 0.05, 0.05, 0.1, 0.1])
                elif objective == OptimizationObjective.MAX_RETURN:
                    weights = np.array([0.1, 0.1, 0.1, 0.1, 0.25, 0.25, 0.05, 0.05])
                else:  # Sharpe ratio
                    weights = np.array([0.15, 0.15, 0.12, 0.13, 0.15, 0.15, 0.08, 0.07])
                
                mock_weights = Mock()
                mock_weights.value = weights
                
                mock_cvxpy.Variable.return_value = mock_weights
                mock_cvxpy.Problem.return_value = mock_problem
                
                optimizer = PortfolioOptimizer(config=config)
                result = optimizer.optimize(expected_returns, covariance_matrix)
                
                # Verify result quality
                assert isinstance(result, OptimizationResult)
                assert result.solver_status == "optimal"
                assert len(result.optimal_weights) == len(symbols)
                
                # Check budget constraint
                total_weight = sum(result.optimal_weights.values())
                assert abs(total_weight - 1.0) < 1e-6
                
                # Check position limits
                for weight in result.optimal_weights.values():
                    assert weight <= config.max_position_weight + 1e-6
                    assert weight >= config.min_position_weight - 1e-6


if __name__ == "__main__":
    pytest.main([__file__])
