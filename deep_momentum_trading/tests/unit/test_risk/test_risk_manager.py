"""
Unit tests for Risk Manager.

Tests the comprehensive risk management system with ARM64 optimizations,
portfolio risk assessment, and real-time risk monitoring capabilities.
"""

import pytest
import asyncio
import numpy as np
import pandas as pd
import time
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from datetime import datetime, timezone, timedelta
from typing import Dict, Any

from deep_momentum_trading.src.risk.risk_manager import (
    RiskManager,
    RiskConfig,
    RiskAssessment,
    PortfolioRiskMetrics,
    RiskLevel,
    RiskAction
)
from deep_momentum_trading.tests.fixtures.sample_data import (
    get_sample_predictions,
    get_sample_portfolio,
    get_sample_market_data
)
from deep_momentum_trading.tests.fixtures.test_configs import TestRiskConfig


class TestRiskConfig:
    """Test RiskConfig class."""
    
    def test_default_config(self):
        """Test default risk configuration values."""
        config = RiskConfig()
        
        assert config.max_portfolio_var == 0.02
        assert config.max_portfolio_volatility == 0.20
        assert config.max_drawdown_limit == 0.10
        assert config.max_position_concentration == 0.005
        assert config.max_sector_concentration == 0.25
        assert config.max_illiquid_percentage == 10.0
        assert config.max_simultaneous_positions == 15000
        assert config.daily_capital_limit == 50000.0
        assert config.min_position_value == 5.0
        assert config.max_position_value == 2500.0
        assert config.enable_real_time_monitoring is True
        assert config.risk_check_interval_seconds == 30.0
        assert config.enable_stress_testing is True
        assert config.enable_arm64_optimizations is True
        assert config.enable_parallel_processing is True
        assert config.max_parallel_workers == 4
        assert config.enable_emergency_stop is True
        assert config.emergency_var_threshold == 0.05
        assert config.emergency_drawdown_threshold == 0.15
    
    def test_custom_config(self):
        """Test custom risk configuration values."""
        config = RiskConfig(
            max_portfolio_var=0.03,
            daily_capital_limit=100000.0,
            max_simultaneous_positions=20000,
            enable_real_time_monitoring=False,
            enable_arm64_optimizations=False
        )
        
        assert config.max_portfolio_var == 0.03
        assert config.daily_capital_limit == 100000.0
        assert config.max_simultaneous_positions == 20000
        assert config.enable_real_time_monitoring is False
        assert config.enable_arm64_optimizations is False


class TestRiskAssessment:
    """Test RiskAssessment class."""
    
    def test_risk_assessment_creation(self):
        """Test creating a risk assessment."""
        assessment = RiskAssessment(
            symbol="AAPL",
            action=RiskAction.APPROVE,
            risk_level=RiskLevel.LOW,
            confidence_adjustment=1.0,
            position_adjustment=1.0,
            reasons=["All checks passed"],
            risk_metrics={"var": 0.01, "concentration": 0.002}
        )
        
        assert assessment.symbol == "AAPL"
        assert assessment.action == RiskAction.APPROVE
        assert assessment.risk_level == RiskLevel.LOW
        assert assessment.confidence_adjustment == 1.0
        assert assessment.position_adjustment == 1.0
        assert assessment.reasons == ["All checks passed"]
        assert assessment.risk_metrics["var"] == 0.01
        assert isinstance(assessment.timestamp, datetime)
    
    def test_risk_assessment_with_defaults(self):
        """Test risk assessment with default timestamp."""
        assessment = RiskAssessment(
            symbol="MSFT",
            action=RiskAction.REJECT,
            risk_level=RiskLevel.HIGH,
            confidence_adjustment=0.0,
            position_adjustment=0.0,
            reasons=["Position too large"],
            risk_metrics={}
        )
        
        # Timestamp should be set automatically
        assert assessment.timestamp is not None
        assert isinstance(assessment.timestamp, datetime)
        assert assessment.timestamp.tzinfo == timezone.utc


class TestPortfolioRiskMetrics:
    """Test PortfolioRiskMetrics class."""
    
    def test_portfolio_risk_metrics_creation(self):
        """Test creating portfolio risk metrics."""
        metrics = PortfolioRiskMetrics(
            total_var=0.015,
            component_var={"AAPL": 0.005, "MSFT": 0.004},
            correlation_risk=0.25,
            liquidity_risk=5.0,
            concentration_risk=0.15,
            sector_risk={"Technology": 0.6, "Finance": 0.4},
            stress_test_results={"market_crash": -0.08},
            risk_level=RiskLevel.MEDIUM,
            emergency_status=False
        )
        
        assert metrics.total_var == 0.015
        assert metrics.component_var["AAPL"] == 0.005
        assert metrics.correlation_risk == 0.25
        assert metrics.liquidity_risk == 5.0
        assert metrics.concentration_risk == 0.15
        assert metrics.sector_risk["Technology"] == 0.6
        assert metrics.stress_test_results["market_crash"] == -0.08
        assert metrics.risk_level == RiskLevel.MEDIUM
        assert metrics.emergency_status is False
        assert isinstance(metrics.last_updated, datetime)


class TestRiskManager:
    """Test RiskManager class."""
    
    @pytest.fixture
    def test_config(self):
        """Get test risk configuration."""
        return TestRiskConfig()
    
    @pytest.fixture
    def mock_dependencies(self):
        """Mock external dependencies."""
        mocks = {}
        
        # Mock position manager
        mock_position_manager = Mock()
        mock_position_manager.sync_with_broker = Mock()
        mock_position_manager.get_current_positions = Mock(return_value={})
        mock_position_manager.get_total_equity = Mock(return_value=100000.0)
        mock_position_manager.get_available_capital = Mock(return_value=50000.0)
        mocks['position_manager'] = mock_position_manager
        
        # Mock Alpaca client
        mock_alpaca_client = Mock()
        mock_alpaca_client.get_bars = Mock(return_value={})
        mocks['alpaca_client'] = mock_alpaca_client
        
        # Mock ZMQ components
        mock_subscriber = Mock()
        mock_subscriber.add_handler = Mock()
        mock_subscriber.start = Mock()
        mock_subscriber.stop = Mock()
        mocks['subscriber'] = mock_subscriber
        
        mock_publisher = Mock()
        mock_publisher.send = Mock()
        mocks['publisher'] = mock_publisher
        
        return mocks
    
    @pytest.fixture
    def risk_manager(self, test_config, mock_dependencies):
        """Create a RiskManager instance with mocked dependencies."""
        with patch('deep_momentum_trading.src.risk.risk_manager.ZMQSubscriber') as mock_sub_class, \
             patch('deep_momentum_trading.src.risk.risk_manager.ZMQPublisher') as mock_pub_class, \
             patch('deep_momentum_trading.src.risk.risk_manager.CorrelationMonitor') as mock_corr, \
             patch('deep_momentum_trading.src.risk.risk_manager.LiquidityMonitor') as mock_liq, \
             patch('deep_momentum_trading.src.risk.risk_manager.PortfolioOptimizer') as mock_opt, \
             patch('deep_momentum_trading.src.risk.risk_manager.VaRCalculator') as mock_var:
            
            # Setup mock returns
            mock_sub_class.return_value = mock_dependencies['subscriber']
            mock_pub_class.return_value = mock_dependencies['publisher']
            
            # Mock risk components
            mock_corr.return_value = Mock()
            mock_liq.return_value = Mock()
            mock_opt.return_value = Mock()
            mock_var.return_value = Mock()
            
            manager = RiskManager(
                risk_config=test_config,
                position_manager=mock_dependencies['position_manager'],
                alpaca_client=mock_dependencies['alpaca_client']
            )
            
            return manager
    
    def test_initialization(self, test_config, mock_dependencies):
        """Test RiskManager initialization."""
        with patch('deep_momentum_trading.src.risk.risk_manager.ZMQSubscriber'), \
             patch('deep_momentum_trading.src.risk.risk_manager.ZMQPublisher'), \
             patch('deep_momentum_trading.src.risk.risk_manager.CorrelationMonitor'), \
             patch('deep_momentum_trading.src.risk.risk_manager.LiquidityMonitor'), \
             patch('deep_momentum_trading.src.risk.risk_manager.PortfolioOptimizer'), \
             patch('deep_momentum_trading.src.risk.risk_manager.VaRCalculator'):
            
            manager = RiskManager(
                risk_config=test_config,
                position_manager=mock_dependencies['position_manager'],
                alpaca_client=mock_dependencies['alpaca_client']
            )
            
            assert manager.config == test_config
            assert manager.position_manager == mock_dependencies['position_manager']
            assert manager.alpaca_client == mock_dependencies['alpaca_client']
            assert not manager.is_running
            assert not manager.emergency_stop_active
            assert manager.current_portfolio_risk is None
            assert len(manager.risk_history) == 0
            assert len(manager.assessment_history) == 0
    
    def test_initialization_with_dict_config(self, mock_dependencies):
        """Test initialization with dictionary configuration."""
        config_dict = {
            'global_limits': {
                'max_portfolio_var': 0.03,
                'max_simultaneous_positions': 20000,
                'daily_capital_limit': 75000.0
            }
        }
        
        with patch('deep_momentum_trading.src.risk.risk_manager.ZMQSubscriber'), \
             patch('deep_momentum_trading.src.risk.risk_manager.ZMQPublisher'), \
             patch('deep_momentum_trading.src.risk.risk_manager.CorrelationMonitor'), \
             patch('deep_momentum_trading.src.risk.risk_manager.LiquidityMonitor'), \
             patch('deep_momentum_trading.src.risk.risk_manager.PortfolioOptimizer'), \
             patch('deep_momentum_trading.src.risk.risk_manager.VaRCalculator'):
            
            manager = RiskManager(risk_config=config_dict)
            
            assert manager.config.max_portfolio_var == 0.03
            assert manager.config.max_simultaneous_positions == 20000
            assert manager.config.daily_capital_limit == 75000.0
    
    @pytest.mark.arm64
    def test_arm64_optimizations(self, test_config, mock_dependencies):
        """Test ARM64 optimization detection and application."""
        with patch('deep_momentum_trading.src.risk.risk_manager.IS_ARM64', True), \
             patch('deep_momentum_trading.src.risk.risk_manager.ZMQSubscriber'), \
             patch('deep_momentum_trading.src.risk.risk_manager.ZMQPublisher'), \
             patch('deep_momentum_trading.src.risk.risk_manager.CorrelationMonitor'), \
             patch('deep_momentum_trading.src.risk.risk_manager.LiquidityMonitor'), \
             patch('deep_momentum_trading.src.risk.risk_manager.PortfolioOptimizer'), \
             patch('deep_momentum_trading.src.risk.risk_manager.VaRCalculator'):
            
            manager = RiskManager(risk_config=test_config)
            
            assert manager.is_arm64 is True
            # ARM64 optimizations should reduce intervals and timeouts
            assert manager.config.risk_check_interval_seconds <= 30.0
            assert manager.config.calculation_timeout_seconds <= 30.0
            assert manager.config.max_parallel_workers >= 4
    
    @pytest.mark.asyncio
    async def test_start_and_stop(self, risk_manager):
        """Test starting and stopping the risk manager."""
        # Mock component start/stop methods
        risk_manager.correlation_monitor.start = Mock()
        risk_manager.correlation_monitor.stop = Mock()
        risk_manager.liquidity_monitor.start = Mock()
        risk_manager.liquidity_monitor.stop = Mock()
        risk_manager.portfolio_optimizer.start = Mock()
        risk_manager.portfolio_optimizer.stop = Mock()
        risk_manager.var_calculator.start = Mock()
        risk_manager.var_calculator.stop = Mock()
        
        # Initially not running
        assert not risk_manager.is_running
        
        # Start the manager
        await risk_manager.start()
        assert risk_manager.is_running
        
        # Verify components were started
        risk_manager.correlation_monitor.start.assert_called_once()
        risk_manager.liquidity_monitor.start.assert_called_once()
        risk_manager.portfolio_optimizer.start.assert_called_once()
        risk_manager.var_calculator.start.assert_called_once()
        
        # Stop the manager
        await risk_manager.stop()
        assert not risk_manager.is_running
        
        # Verify components were stopped
        risk_manager.correlation_monitor.stop.assert_called_once()
        risk_manager.liquidity_monitor.stop.assert_called_once()
        risk_manager.portfolio_optimizer.stop.assert_called_once()
        risk_manager.var_calculator.stop.assert_called_once()
    
    def test_get_current_prices(self, risk_manager):
        """Test getting current prices for symbols."""
        symbols = ["AAPL", "MSFT", "GOOGL"]
        
        # Mock Alpaca client response
        mock_bars = {
            "AAPL": [Mock(close=150.0)],
            "MSFT": [Mock(close=300.0)],
            "GOOGL": [Mock(close=2500.0)]
        }
        risk_manager.alpaca_client.get_bars.return_value = mock_bars
        
        prices = risk_manager._get_current_prices(symbols)
        
        assert prices["AAPL"] == 150.0
        assert prices["MSFT"] == 300.0
        assert prices["GOOGL"] == 2500.0
    
    def test_get_current_prices_fallback(self, risk_manager):
        """Test price fallback when Alpaca client fails."""
        symbols = ["AAPL", "MSFT"]
        
        # Mock Alpaca client to return empty or fail
        risk_manager.alpaca_client.get_bars.return_value = {}
        
        prices = risk_manager._get_current_prices(symbols)
        
        # Should return fallback prices
        assert prices["AAPL"] == 100.0
        assert prices["MSFT"] == 100.0
    
    def test_get_current_prices_no_client(self, test_config):
        """Test price handling when no Alpaca client is available."""
        with patch('deep_momentum_trading.src.risk.risk_manager.ZMQSubscriber'), \
             patch('deep_momentum_trading.src.risk.risk_manager.ZMQPublisher'), \
             patch('deep_momentum_trading.src.risk.risk_manager.CorrelationMonitor'), \
             patch('deep_momentum_trading.src.risk.risk_manager.LiquidityMonitor'), \
             patch('deep_momentum_trading.src.risk.risk_manager.PortfolioOptimizer'), \
             patch('deep_momentum_trading.src.risk.risk_manager.VaRCalculator'):
            
            manager = RiskManager(risk_config=test_config, alpaca_client=None)
            
            symbols = ["AAPL", "MSFT"]
            prices = manager._get_current_prices(symbols)
            
            # Should return fallback prices
            assert prices["AAPL"] == 100.0
            assert prices["MSFT"] == 100.0
    
    def test_calculate_concentration_risk(self, risk_manager):
        """Test concentration risk calculation."""
        positions = {
            "AAPL": 100,
            "MSFT": 50,
            "GOOGL": 25
        }
        prices = {
            "AAPL": 150.0,
            "MSFT": 300.0,
            "GOOGL": 2500.0
        }
        total_equity = 100000.0
        
        concentration_risk = risk_manager._calculate_concentration_risk(
            positions, prices, total_equity
        )
        
        # Should calculate based on position weights
        assert isinstance(concentration_risk, float)
        assert concentration_risk >= 0.0
        assert concentration_risk <= 1.0
    
    def test_calculate_concentration_risk_empty_positions(self, risk_manager):
        """Test concentration risk with empty positions."""
        concentration_risk = risk_manager._calculate_concentration_risk(
            {}, {}, 100000.0
        )
        
        assert concentration_risk == 0.0
    
    def test_determine_portfolio_risk_level(self, risk_manager):
        """Test portfolio risk level determination."""
        # Low risk scenario
        risk_level = risk_manager._determine_portfolio_risk_level(
            var=0.005, correlation_risk=0.1, liquidity_risk=2.0, concentration_risk=0.05
        )
        assert risk_level == RiskLevel.LOW
        
        # Medium risk scenario
        risk_level = risk_manager._determine_portfolio_risk_level(
            var=0.015, correlation_risk=0.4, liquidity_risk=15.0, concentration_risk=0.2
        )
        assert risk_level == RiskLevel.MEDIUM
        
        # High risk scenario
        risk_level = risk_manager._determine_portfolio_risk_level(
            var=0.025, correlation_risk=0.7, liquidity_risk=30.0, concentration_risk=0.4
        )
        assert risk_level == RiskLevel.HIGH
        
        # Critical risk scenario
        risk_level = risk_manager._determine_portfolio_risk_level(
            var=0.04, correlation_risk=0.9, liquidity_risk=50.0, concentration_risk=0.8
        )
        assert risk_level == RiskLevel.CRITICAL
    
    def test_assess_single_prediction_approve(self, risk_manager):
        """Test single prediction assessment - approval case."""
        prediction = {
            'confidence': 0.8,
            'position': 0.5,
            'expected_return': 0.03
        }
        
        current_positions = {"AAPL": 50}
        current_prices = {"AAPL": 150.0}
        total_equity = 100000.0
        available_capital = 50000.0
        
        # Mock portfolio risk
        portfolio_risk = PortfolioRiskMetrics(
            total_var=0.01, component_var={}, correlation_risk=0.2,
            liquidity_risk=5.0, concentration_risk=0.1, sector_risk={},
            stress_test_results={}, risk_level=RiskLevel.LOW, emergency_status=False
        )
        
        assessment = risk_manager._assess_single_prediction(
            "AAPL", prediction, current_positions, current_prices,
            total_equity, available_capital, portfolio_risk
        )
        
        assert assessment.symbol == "AAPL"
        assert assessment.action == RiskAction.APPROVE
        assert assessment.risk_level == RiskLevel.LOW
        assert assessment.confidence_adjustment == 1.0
        assert assessment.position_adjustment == 1.0
        assert "All risk checks passed" in assessment.reasons
    
    def test_assess_single_prediction_reject_low_confidence(self, risk_manager):
        """Test single prediction assessment - rejection due to low confidence."""
        prediction = {
            'confidence': 0.05,  # Below minimum threshold
            'position': 0.5,
            'expected_return': 0.03
        }
        
        current_positions = {}
        current_prices = {"AAPL": 150.0}
        total_equity = 100000.0
        available_capital = 50000.0
        
        portfolio_risk = PortfolioRiskMetrics(
            total_var=0.01, component_var={}, correlation_risk=0.2,
            liquidity_risk=5.0, concentration_risk=0.1, sector_risk={},
            stress_test_results={}, risk_level=RiskLevel.LOW, emergency_status=False
        )
        
        assessment = risk_manager._assess_single_prediction(
            "AAPL", prediction, current_positions, current_prices,
            total_equity, available_capital, portfolio_risk
        )
        
        assert assessment.action == RiskAction.REJECT
        assert assessment.confidence_adjustment == 0.0
        assert assessment.position_adjustment == 0.0
        assert any("Confidence" in reason for reason in assessment.reasons)
    
    def test_assess_single_prediction_scale_down_large_position(self, risk_manager):
        """Test single prediction assessment - scale down due to large position."""
        prediction = {
            'confidence': 0.8,
            'position': 10.0,  # Very large position
            'expected_return': 0.03
        }
        
        current_positions = {}
        current_prices = {"AAPL": 150.0}
        total_equity = 100000.0
        available_capital = 50000.0
        
        portfolio_risk = PortfolioRiskMetrics(
            total_var=0.01, component_var={}, correlation_risk=0.2,
            liquidity_risk=5.0, concentration_risk=0.1, sector_risk={},
            stress_test_results={}, risk_level=RiskLevel.LOW, emergency_status=False
        )
        
        assessment = risk_manager._assess_single_prediction(
            "AAPL", prediction, current_positions, current_prices,
            total_equity, available_capital, portfolio_risk
        )
        
        assert assessment.action == RiskAction.SCALE_DOWN
        assert assessment.position_adjustment < 1.0
        assert any("Position value" in reason for reason in assessment.reasons)
    
    def test_assess_single_prediction_emergency_stop(self, risk_manager):
        """Test single prediction assessment - emergency stop."""
        prediction = {
            'confidence': 0.8,
            'position': 0.5,
            'expected_return': 0.03
        }
        
        current_positions = {}
        current_prices = {"AAPL": 150.0}
        total_equity = 100000.0
        available_capital = 50000.0
        
        # Emergency portfolio risk
        portfolio_risk = PortfolioRiskMetrics(
            total_var=0.01, component_var={}, correlation_risk=0.2,
            liquidity_risk=5.0, concentration_risk=0.1, sector_risk={},
            stress_test_results={}, risk_level=RiskLevel.CRITICAL, emergency_status=True
        )
        
        assessment = risk_manager._assess_single_prediction(
            "AAPL", prediction, current_positions, current_prices,
            total_equity, available_capital, portfolio_risk
        )
        
        assert assessment.action == RiskAction.EMERGENCY_STOP
        assert assessment.risk_level == RiskLevel.CRITICAL
        assert assessment.confidence_adjustment == 0.0
        assert assessment.position_adjustment == 0.0
        assert any("emergency stop" in reason.lower() for reason in assessment.reasons)
    
    def test_assess_portfolio_risk(self, risk_manager):
        """Test portfolio risk assessment."""
        current_positions = {
            "AAPL": 100,
            "MSFT": 50
        }
        current_prices = {
            "AAPL": 150.0,
            "MSFT": 300.0
        }
        total_equity = 100000.0
        
        # Mock risk component responses
        risk_manager.var_calculator.calculate_portfolio_var = Mock(return_value=0.015)
        risk_manager.correlation_monitor.assess_portfolio_correlation_risk = Mock(
            return_value={'overall_correlation_risk': 0.25}
        )
        risk_manager.liquidity_monitor.assess_portfolio_liquidity_risk = Mock(
            return_value={'illiquid_percentage': 5.0}
        )
        
        portfolio_risk = risk_manager._assess_portfolio_risk(
            current_positions, current_prices, total_equity
        )
        
        assert isinstance(portfolio_risk, PortfolioRiskMetrics)
        assert portfolio_risk.total_var == 0.015
        assert portfolio_risk.correlation_risk == 0.25
        assert portfolio_risk.liquidity_risk == 5.0
        assert isinstance(portfolio_risk.concentration_risk, float)
        assert isinstance(portfolio_risk.risk_level, RiskLevel)
        assert isinstance(portfolio_risk.emergency_status, bool)
    
    def test_apply_risk_decisions(self, risk_manager):
        """Test applying risk decisions to predictions."""
        # Create sample assessments
        assessments = [
            RiskAssessment(
                symbol="AAPL",
                action=RiskAction.APPROVE,
                risk_level=RiskLevel.LOW,
                confidence_adjustment=1.0,
                position_adjustment=1.0,
                reasons=["Approved"],
                risk_metrics={}
            ),
            RiskAssessment(
                symbol="MSFT",
                action=RiskAction.SCALE_DOWN,
                risk_level=RiskLevel.MEDIUM,
                confidence_adjustment=0.8,
                position_adjustment=0.7,
                reasons=["Scaled down"],
                risk_metrics={}
            ),
            RiskAssessment(
                symbol="GOOGL",
                action=RiskAction.REJECT,
                risk_level=RiskLevel.HIGH,
                confidence_adjustment=0.0,
                position_adjustment=0.0,
                reasons=["Rejected"],
                risk_metrics={}
            )
        ]
        
        predictions = {
            "AAPL": {"confidence": 0.8, "position": 0.5},
            "MSFT": {"confidence": 0.7, "position": 0.6},
            "GOOGL": {"confidence": 0.6, "position": 0.4}
        }
        
        approved_predictions = risk_manager._apply_risk_decisions(assessments, predictions)
        
        # Should have AAPL (approved) and MSFT (scaled)
        assert len(approved_predictions) == 2
        assert "AAPL" in approved_predictions
        assert "MSFT" in approved_predictions
        assert "GOOGL" not in approved_predictions
        
        # Check adjustments
        assert approved_predictions["AAPL"]["confidence"] == 0.8  # No adjustment
        assert approved_predictions["MSFT"]["confidence"] == 0.7 * 0.8  # Scaled
        assert approved_predictions["MSFT"]["position"] == 0.6 * 0.7  # Scaled
        
        # Check statistics update
        assert risk_manager.performance_stats["approved_predictions"] == 1
        assert risk_manager.performance_stats["scaled_predictions"] == 1
        assert risk_manager.performance_stats["rejected_predictions"] == 1
    
    def test_get_risk_metrics(self, risk_manager):
        """Test getting current risk metrics."""
        # Set up some portfolio risk
        risk_manager.current_portfolio_risk = PortfolioRiskMetrics(
            total_var=0.015,
            component_var={"AAPL": 0.005},
            correlation_risk=0.25,
            liquidity_risk=5.0,
            concentration_risk=0.15,
            sector_risk={"Technology": 0.6},
            stress_test_results={"market_crash": -0.08},
            risk_level=RiskLevel.MEDIUM,
            emergency_status=False
        )
        
        # Update some performance stats
        risk_manager.performance_stats["total_assessments"] = 100
        risk_manager.performance_stats["approved_predictions"] = 80
        
        metrics = risk_manager.get_risk_metrics()
        
        assert "portfolio_risk" in metrics
        assert "performance_stats" in metrics
        assert "emergency_stop_active" in metrics
        assert "recent_assessments" in metrics
        
        assert metrics["portfolio_risk"]["total_var"] == 0.015
        assert metrics["portfolio_risk"]["risk_level"] == "medium"
        assert metrics["performance_stats"]["total_assessments"] == 100
        assert metrics["emergency_stop_active"] is False
    
    def test_get_risk_metrics_no_data(self, risk_manager):
        """Test getting risk metrics when no data is available."""
        metrics = risk_manager.get_risk_metrics()
        
        assert metrics["status"] == "no_risk_data"
    
    def test_get_status(self, risk_manager):
        """Test getting risk manager status."""
        # Mock component status
        risk_manager.correlation_monitor.get_status = Mock(return_value={"status": "active"})
        risk_manager.liquidity_monitor.get_status = Mock(return_value={"status": "active"})
        risk_manager.portfolio_optimizer.get_status = Mock(return_value={"status": "active"})
        risk_manager.var_calculator.get_status = Mock(return_value={"status": "active"})
        
        status = risk_manager.get_status()
        
        assert "running" in status
        assert "emergency_stop_active" in status
        assert "arm64_optimized" in status
        assert "shared_memory_enabled" in status
        assert "components" in status
        assert "performance_stats" in status
        assert "risk_assessments_count" in status
        assert "portfolio_risk_level" in status
        
        # Check components
        assert "correlation_monitor" in status["components"]
        assert "liquidity_monitor" in status["components"]
        assert "portfolio_optimizer" in status["components"]
        assert "var_calculator" in status["components"]
    
    @pytest.mark.performance
    def test_risk_assessment_performance(self, risk_manager):
        """Test risk assessment performance with multiple predictions."""
        # Generate large number of predictions
        symbols = ["AAPL", "MSFT", "GOOGL", "TSLA", "NVDA"] * 20  # 100 symbols
        predictions = {}
        
        for i, symbol in enumerate(symbols):
            predictions[f"{symbol}_{i}"] = {
                'confidence': np.random.uniform(0.1, 0.9),
                'position': np.random.uniform(-1, 1),
                'expected_return': np.random.uniform(-0.05, 0.05)
            }
        
        # Mock portfolio risk
        portfolio_risk = PortfolioRiskMetrics(
            total_var=0.01, component_var={}, correlation_risk=0.2,
            liquidity_risk=5.0, concentration_risk=0.1, sector_risk={},
            stress_test_results={}, risk_level=RiskLevel.LOW, emergency_status=False
        )
        
        # Benchmark assessment performance
        start_time = time.perf_counter()
        
        assessments = risk_manager._assess_prediction_risks(
            predictions, {}, {}, 100000.0, 50000.0, portfolio_risk
        )
        
        end_time = time.perf_counter()
        assessment_time = end_time - start_time
        
        # Should handle 100 predictions quickly
        assert assessment_time < 2.0  # Less than 2 seconds
        assert len(assessments) == len(predictions)
        
        # All assessments should be valid
        for assessment in assessments:
            assert isinstance(assessment, RiskAssessment)
            assert assessment.symbol in predictions
            assert isinstance(assessment.action, RiskAction)
            assert isinstance(assessment.risk_level, RiskLevel)


@pytest.mark.integration
class TestRiskManagerIntegration:
    """Integration tests for risk manager."""
    
    @pytest.mark.asyncio
    async def test_full_risk_assessment_workflow(self):
        """Test complete risk assessment workflow."""
        # Use test configuration
        config = TestRiskConfig()
        config.enable_real_time_monitoring = False  # Disable for test
        config.enable_parallel_processing = False   # Disable for test
        
        with patch('deep_momentum_trading.src.risk.risk_manager.ZMQSubscriber') as mock_sub, \
             patch('deep_momentum_trading.src.risk.risk_manager.ZMQPublisher') as mock_pub, \
             patch('deep_momentum_trading.src.risk.risk_manager.CorrelationMonitor') as mock_corr, \
             patch('deep_momentum_trading.src.risk.risk_manager.LiquidityMonitor') as mock_liq, \
             patch('deep_momentum_trading.src.risk.risk_manager.PortfolioOptimizer') as mock_opt, \
             patch('deep_momentum_trading.src.risk.risk_manager.VaRCalculator') as mock_var:
            
            # Setup mocks
            mock_sub.return_value = Mock()
            mock_pub.return_value = Mock()
            mock_corr.return_value = Mock()
            mock_liq.return_value = Mock()
            mock_opt.return_value = Mock()
            mock_var.return_value = Mock()
            
            # Mock position manager
            mock_position_manager = Mock()
            mock_position_manager.sync_with_broker = Mock()
            mock_position_manager.get_current_positions = Mock(return_value={
                "AAPL": 100,
                "MSFT": 50
            })
            mock_position_manager.get_total_equity = Mock(return_value=100000.0)
            mock_position_manager.get_available_capital = Mock(return_value=50000.0)
            
            # Create risk manager
            risk_manager = RiskManager(
                risk_config=config,
                position_manager=mock_position_manager
            )
            
            # Mock risk component responses
            risk_manager.var_calculator.calculate_portfolio_var = Mock(return_value=0.015)
            risk_manager.correlation_monitor.assess_portfolio_correlation_risk = Mock(
                return_value={'overall_correlation_risk': 0.25}
            )
            risk_manager.liquidity_monitor.assess_portfolio_liquidity_risk = Mock(
                return_value={'illiquid_percentage': 5.0}
            )
            
            # Generate sample predictions
            predictions = get_sample_predictions(n_predictions=10)
            predictions_dict = {pred['symbol']: pred for pred in predictions}
            
            # Create mock message
            message = {
                'data': predictions_dict
            }
            
            # Process predictions through risk manager
            risk_manager._process_predictions("predictions", message)
            
            # Verify risk assessment was performed
            assert len(risk_manager.assessment_history) > 0
            assert risk_manager.performance_stats["total_assessments"] > 0
            
            # Verify portfolio risk was assessed
            assert risk_manager.current_portfolio_risk is not None
            assert isinstance(risk_manager.current_portfolio_risk, PortfolioRiskMetrics)
    
    def test_emergency_stop_workflow(self):
        """Test emergency stop activation and deactivation."""
        config = TestRiskConfig()
        config.emergency_var_threshold = 0.02  # Low threshold for testing
        
        with patch('deep_momentum_trading.src.risk.risk_manager.ZMQSubscriber'), \
             patch('deep_momentum_trading.src.risk.risk_manager.ZMQPublisher'), \
             patch('deep_momentum_trading.src.risk.risk_manager.CorrelationMonitor'), \
             patch('deep_momentum_trading.src.risk.risk_manager.LiquidityMonitor'), \
             patch('deep_momentum_trading.src.risk.risk_manager.PortfolioOptimizer'), \
             patch('deep_momentum_trading.src.risk.risk_manager.VaRCalculator'):
            
            risk_manager = RiskManager(risk_config=config)
            
            # Initially no emergency stop
            assert not risk_manager.emergency_stop_active
            
            # Set high-risk portfolio metrics
            risk_manager.current_portfolio_risk = PortfolioRiskMetrics(
                total_var=0.05,  # Above threshold
                component_var={},
                correlation_risk=0.8,
                liquidity_risk=20.0,
                concentration_risk=0.6,
                sector_risk={},
                stress_test_results={},
                risk_level=RiskLevel.CRITICAL,
                emergency_status=True
            )
            
            # Check emergency conditions (would normally be called by background task)
            import asyncio
            asyncio.run(risk_manager._check_emergency_conditions())
            
            # Emergency stop should be activated
            assert risk_manager.emergency_stop_active
            
            # Test prediction processing during emergency stop
            predictions = {"AAPL": {"confidence": 0.8, "position": 0.5}}
            message = {"data": predictions}
            
            # Should reject all predictions during emergency stop
            risk_manager._process_predictions("predictions", message)
            
            # All assessments should be emergency stops
            emergency_assessments = [
                a for a in risk_manager.assessment_history 
                if a.action == RiskAction.EMERGENCY_STOP
            ]
            assert len(emergency_assessments) > 0


if __name__ == "__main__":
    pytest.main([__file__])
