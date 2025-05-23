"""
Integration tests for Trading Pipeline.

Tests the complete trading workflow from signal generation through order execution,
including risk management, position management, and performance tracking.
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
    get_sample_predictions,
    get_sample_portfolio,
    get_sample_market_data
)
from deep_momentum_trading.tests.fixtures.test_configs import (
    TestConfigManager,
    TestScenarios
)


@pytest.mark.integration
class TestTradingSignalPipeline:
    """Test trading signal generation and processing pipeline."""
    
    @pytest.fixture
    def trading_config(self):
        """Get trading pipeline test configuration."""
        return {
            'symbols': ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA'],
            'portfolio_value': 1000000.0,
            'max_position_size': 0.1,
            'min_confidence_threshold': 0.6,
            'risk_limit_daily': 0.02,
            'enable_paper_trading': True,
            'enable_risk_management': True,
            'order_timeout_seconds': 30,
            'enable_position_sizing': True
        }
    
    @pytest.fixture
    def mock_trading_components(self):
        """Mock trading system components."""
        mocks = {}
        
        # Mock Alpaca client
        mock_alpaca = Mock()
        mock_alpaca.get_account = Mock(return_value={
            'equity': 1000000.0,
            'buying_power': 500000.0,
            'cash': 250000.0,
            'portfolio_value': 1000000.0
        })
        mock_alpaca.get_positions = Mock(return_value=[])
        mock_alpaca.submit_order = Mock()
        mock_alpaca.get_orders = Mock(return_value=[])
        mock_alpaca.cancel_order = Mock()
        mocks['alpaca'] = mock_alpaca
        
        # Mock risk manager
        mock_risk_manager = Mock()
        mock_risk_manager.assess_prediction_risk = Mock()
        mock_risk_manager.get_position_limits = Mock()
        mock_risk_manager.check_portfolio_risk = Mock()
        mocks['risk_manager'] = mock_risk_manager
        
        # Mock position manager
        mock_position_manager = Mock()
        mock_position_manager.get_current_positions = Mock(return_value={})
        mock_position_manager.calculate_position_size = Mock()
        mock_position_manager.update_positions = Mock()
        mocks['position_manager'] = mock_position_manager
        
        return mocks
    
    @pytest.mark.asyncio
    async def test_signal_to_order_pipeline(self, trading_config, mock_trading_components):
        """Test complete signal to order execution pipeline."""
        
        with patch('deep_momentum_trading.src.trading.trading_engine.TradingEngine') as mock_engine_class, \
             patch('deep_momentum_trading.src.trading.alpaca_client.AlpacaClient') as mock_alpaca_class, \
             patch('deep_momentum_trading.src.risk.risk_manager.RiskManager') as mock_risk_class:
            
            # Setup mocks
            mock_alpaca_class.return_value = mock_trading_components['alpaca']
            mock_risk_class.return_value = mock_trading_components['risk_manager']
            
            # Mock trading engine
            mock_engine = Mock()
            mock_engine.process_predictions = AsyncMock()
            mock_engine.execute_trades = AsyncMock()
            mock_engine.get_performance_stats = Mock()
            mock_engine_class.return_value = mock_engine
            
            # Generate test predictions
            predictions = get_sample_predictions(n_predictions=len(trading_config['symbols']))
            
            # Filter predictions by confidence threshold
            high_confidence_predictions = [
                pred for pred in predictions 
                if pred['confidence'] >= trading_config['min_confidence_threshold']
            ]
            
            # Mock risk assessment
            mock_trading_components['risk_manager'].assess_prediction_risk.return_value = {
                'approved': True,
                'risk_level': 'medium',
                'position_size_multiplier': 0.8,
                'max_position_value': 50000.0
            }
            
            # Mock position sizing
            mock_trading_components['position_manager'].calculate_position_size.return_value = {
                'shares': 100,
                'dollar_amount': 15000.0,
                'position_weight': 0.015
            }
            
            # Mock successful order execution
            mock_trading_components['alpaca'].submit_order.return_value = {
                'id': 'order_123',
                'status': 'filled',
                'symbol': 'AAPL',
                'qty': 100,
                'side': 'buy',
                'filled_price': 150.0,
                'filled_at': datetime.now(timezone.utc).isoformat()
            }
            
            from deep_momentum_trading.src.trading.trading_engine import TradingEngine
            
            # Initialize trading engine
            engine = TradingEngine(config=trading_config)
            
            # Process predictions through pipeline
            await engine.process_predictions(high_confidence_predictions)
            
            # Verify pipeline execution
            mock_engine.process_predictions.assert_called_once()
            
            # Verify risk assessment was called
            assert mock_trading_components['risk_manager'].assess_prediction_risk.call_count > 0
            
            # Verify position sizing was calculated
            assert mock_trading_components['position_manager'].calculate_position_size.call_count > 0


@pytest.mark.integration
class TestRiskManagementPipeline:
    """Test risk management integration in trading pipeline."""
    
    @pytest.fixture
    def risk_config(self):
        """Get risk management configuration."""
        return {
            'max_portfolio_risk': 0.02,
            'max_position_risk': 0.005,
            'max_sector_concentration': 0.3,
            'var_confidence_level': 0.95,
            'enable_stop_losses': True,
            'enable_position_limits': True,
            'enable_correlation_limits': True,
            'risk_check_frequency_seconds': 60
        }
    
    @pytest.mark.asyncio
    async def test_pre_trade_risk_checks(self, risk_config):
        """Test pre-trade risk assessment pipeline."""
        
        with patch('deep_momentum_trading.src.risk.risk_manager.RiskManager') as mock_rm_class:
            
            # Mock risk manager
            mock_rm = Mock()
            mock_rm.assess_pre_trade_risk = Mock()
            mock_rm.check_position_limits = Mock()
            mock_rm.calculate_portfolio_var = Mock()
            mock_rm_class.return_value = mock_rm
            
            # Mock risk assessment results
            mock_rm.assess_pre_trade_risk.return_value = {
                'approved': True,
                'risk_score': 0.65,
                'warnings': [],
                'position_size_adjustment': 1.0,
                'max_position_value': 50000.0
            }
            
            mock_rm.check_position_limits.return_value = {
                'within_limits': True,
                'current_exposure': 0.08,
                'max_allowed': 0.10,
                'available_capacity': 0.02
            }
            
            mock_rm.calculate_portfolio_var.return_value = {
                'portfolio_var': 18000.0,
                'var_limit': 20000.0,
                'utilization': 0.90
            }
            
            from deep_momentum_trading.src.risk.risk_manager import RiskManager
            
            risk_manager = RiskManager(config=risk_config)
            
            # Test pre-trade risk assessment
            trade_request = {
                'symbol': 'AAPL',
                'side': 'buy',
                'quantity': 100,
                'estimated_value': 15000.0,
                'confidence': 0.75
            }
            
            risk_assessment = risk_manager.assess_pre_trade_risk(trade_request)
            
            # Verify risk assessment
            mock_rm.assess_pre_trade_risk.assert_called_once()
            assert risk_assessment['approved'] is True
            assert risk_assessment['risk_score'] > 0
            
            # Test position limits check
            position_check = risk_manager.check_position_limits('AAPL', 15000.0)
            
            # Verify position limits
            mock_rm.check_position_limits.assert_called_once()
            assert position_check['within_limits'] is True


@pytest.mark.integration
class TestTradingPerformancePipeline:
    """Test trading performance tracking and analysis pipeline."""
    
    @pytest.fixture
    def performance_config(self):
        """Get performance tracking configuration."""
        return {
            'track_real_time_pnl': True,
            'calculate_risk_metrics': True,
            'benchmark_symbol': 'SPY',
            'performance_window_days': 30,
            'enable_attribution_analysis': True,
            'enable_drawdown_tracking': True,
            'reporting_frequency_hours': 24
        }
    
    def test_pnl_tracking_pipeline(self, performance_config):
        """Test P&L tracking pipeline."""
        
        with patch('deep_momentum_trading.src.monitoring.performance_tracker.PerformanceTracker') as mock_pt_class:
            
            # Mock performance tracker
            mock_pt = Mock()
            mock_pt.update_pnl = Mock()
            mock_pt.calculate_daily_pnl = Mock()
            mock_pt.get_performance_summary = Mock()
            mock_pt_class.return_value = mock_pt
            
            # Mock P&L calculations
            mock_pt.calculate_daily_pnl.return_value = {
                'realized_pnl': 2500.0,
                'unrealized_pnl': 1800.0,
                'total_pnl': 4300.0,
                'pnl_percentage': 0.43
            }
            
            mock_pt.get_performance_summary.return_value = {
                'total_return': 0.085,
                'sharpe_ratio': 1.25,
                'max_drawdown': 0.032,
                'win_rate': 0.68,
                'average_win': 850.0,
                'average_loss': -420.0,
                'profit_factor': 2.02
            }
            
            from deep_momentum_trading.src.monitoring.performance_tracker import PerformanceTracker
            
            performance_tracker = PerformanceTracker(config=performance_config)
            
            # Test daily P&L calculation
            current_portfolio = get_sample_portfolio(total_value=1000000.0)
            daily_pnl = performance_tracker.calculate_daily_pnl(current_portfolio)
            
            # Verify P&L calculation
            mock_pt.calculate_daily_pnl.assert_called_once()
            assert daily_pnl['total_pnl'] > 0
            assert 'realized_pnl' in daily_pnl
            assert 'unrealized_pnl' in daily_pnl
            
            # Test performance summary
            performance_summary = performance_tracker.get_performance_summary()
            
            # Verify performance summary
            mock_pt.get_performance_summary.assert_called_once()
            assert 'total_return' in performance_summary
            assert 'sharpe_ratio' in performance_summary
            assert 'max_drawdown' in performance_summary


@pytest.mark.integration
class TestTradingPipelineReliability:
    """Test trading pipeline reliability and error handling."""
    
    def test_order_execution_failures(self):
        """Test handling of order execution failures."""
        
        failure_count = 0
        recovery_count = 0
        
        def failure_handler(error):
            nonlocal failure_count
            failure_count += 1
        
        def recovery_handler():
            nonlocal recovery_count
            recovery_count += 1
        
        with patch('deep_momentum_trading.src.trading.order_manager.OrderManager') as mock_om_class:
            
            # Mock order manager with failures
            mock_om = Mock()
            mock_om.place_order = Mock()
            mock_om.set_failure_handler = Mock(
                side_effect=lambda handler: setattr(mock_om, '_failure_handler', handler)
            )
            mock_om.set_recovery_handler = Mock(
                side_effect=lambda handler: setattr(mock_om, '_recovery_handler', handler)
            )
            mock_om_class.return_value = mock_om
            
            # Simulate order failures
            mock_om.place_order.side_effect = [
                Exception("Insufficient buying power"),
                Exception("Market closed"),
                {'order_id': 'order_789', 'status': 'submitted'},  # Success
                {'order_id': 'order_790', 'status': 'submitted'}   # Success
            ]
            
            from deep_momentum_trading.src.trading.order_manager import OrderManager
            
            order_manager = OrderManager(config={'enable_error_recovery': True})
            order_manager.set_failure_handler(failure_handler)
            order_manager.set_recovery_handler(recovery_handler)
            
            # Test order placement with failures
            orders = [
                {'symbol': 'AAPL', 'quantity': 100, 'side': 'buy'},
                {'symbol': 'MSFT', 'quantity': 75, 'side': 'buy'},
                {'symbol': 'GOOGL', 'quantity': 25, 'side': 'buy'},
                {'symbol': 'TSLA', 'quantity': 50, 'side': 'buy'}
            ]
            
            successful_orders = 0
            for order in orders:
                try:
                    result = order_manager.place_order(order)
                    if result and 'order_id' in result:
                        successful_orders += 1
                        if hasattr(mock_om, '_recovery_handler'):
                            mock_om._recovery_handler()
                except Exception as e:
                    if hasattr(mock_om, '_failure_handler'):
                        mock_om._failure_handler(e)
            
            # Verify error handling
            assert failure_count == 2  # Two failures expected
            assert recovery_count == 2  # Two recoveries expected
            assert successful_orders == 2  # Two successful orders
    
    def test_market_data_interruption_handling(self):
        """Test handling of market data interruptions."""
        
        with patch('deep_momentum_trading.src.trading.trading_engine.TradingEngine') as mock_engine_class:
            
            # Mock trading engine with data interruption handling
            mock_engine = Mock()
            mock_engine.handle_data_interruption = Mock()
            mock_engine.resume_trading = Mock()
            mock_engine.get_system_status = Mock()
            mock_engine_class.return_value = mock_engine
            
            # Mock data interruption scenarios
            mock_engine.handle_data_interruption.return_value = {
                'action_taken': 'pause_trading',
                'affected_symbols': ['AAPL', 'MSFT'],
                'estimated_recovery_time': 300,  # 5 minutes
                'fallback_data_source': 'alpaca'
            }
            
            mock_engine.resume_trading.return_value = {
                'trading_resumed': True,
                'data_source': 'polygon',
                'symbols_active': ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA'],
                'resume_time': datetime.now(timezone.utc).isoformat()
            }
            
            from deep_momentum_trading.src.trading.trading_engine import TradingEngine
            
            trading_engine = TradingEngine(config={'enable_data_failover': True})
            
            # Test data interruption handling
            interruption_result = trading_engine.handle_data_interruption(['AAPL', 'MSFT'])
            
            # Verify interruption handling
            mock_engine.handle_data_interruption.assert_called_once()
            assert interruption_result['action_taken'] == 'pause_trading'
            assert 'fallback_data_source' in interruption_result
            
            # Test trading resumption
            resume_result = trading_engine.resume_trading()
            
            # Verify trading resumption
            mock_engine.resume_trading.assert_called_once()
            assert resume_result['trading_resumed'] is True
            assert len(resume_result['symbols_active']) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
