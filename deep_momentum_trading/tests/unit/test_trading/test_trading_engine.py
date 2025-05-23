"""
Unit tests for Trading Engine.

Tests the main trading engine functionality including signal processing,
order execution, risk management integration, and performance tracking.
"""

import pytest
import asyncio
import time
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from datetime import datetime, timezone, timedelta
import pandas as pd
import numpy as np
from typing import Dict, List, Any

from deep_momentum_trading.tests.fixtures.sample_data import (
    get_sample_predictions,
    get_sample_portfolio,
    get_sample_market_data
)
from deep_momentum_trading.tests.fixtures.test_configs import TestConfigManager


class TestTradingEngine:
    """Test trading engine functionality."""
    
    @pytest.fixture
    def trading_config(self):
        """Get trading engine configuration."""
        return {
            'symbols': ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA'],
            'portfolio_value': 1000000.0,
            'max_position_size': 0.1,
            'min_confidence_threshold': 0.6,
            'risk_limit_daily': 0.02,
            'enable_paper_trading': True,
            'enable_risk_management': True,
            'enable_position_sizing': True,
            'order_timeout_seconds': 30,
            'max_orders_per_minute': 50,
            'enable_performance_tracking': True,
            'enable_real_time_monitoring': True,
            'trading_hours': {
                'start': '09:30',
                'end': '16:00',
                'timezone': 'US/Eastern'
            }
        }
    
    @pytest.fixture
    def mock_components(self):
        """Mock trading engine components."""
        components = {}
        
        # Mock Alpaca client
        mock_alpaca = Mock()
        mock_alpaca.get_account_info.return_value = {
            'portfolio_value': 1000000.0,
            'buying_power': 500000.0,
            'cash': 250000.0
        }
        mock_alpaca.get_positions.return_value = []
        mock_alpaca.submit_order.return_value = {
            'id': 'order_123',
            'status': 'accepted'
        }
        components['alpaca'] = mock_alpaca
        
        # Mock order manager
        mock_order_manager = Mock()
        mock_order_manager.place_order.return_value = {
            'order_id': 'order_456',
            'status': 'submitted'
        }
        mock_order_manager.monitor_orders.return_value = {
            'active_orders': 5,
            'filled_orders': 2
        }
        components['order_manager'] = mock_order_manager
        
        # Mock risk manager
        mock_risk_manager = Mock()
        mock_risk_manager.assess_prediction_risk.return_value = {
            'approved': True,
            'risk_level': 'medium',
            'position_size_multiplier': 0.8
        }
        mock_risk_manager.check_portfolio_risk.return_value = {
            'within_limits': True,
            'current_risk': 0.015
        }
        components['risk_manager'] = mock_risk_manager
        
        # Mock position manager
        mock_position_manager = Mock()
        mock_position_manager.calculate_position_size.return_value = {
            'shares': 100,
            'dollar_amount': 15000.0
        }
        mock_position_manager.get_current_positions.return_value = {}
        components['position_manager'] = mock_position_manager
        
        # Mock performance tracker
        mock_performance_tracker = Mock()
        mock_performance_tracker.update_performance.return_value = {
            'pnl_updated': True,
            'metrics_calculated': True
        }
        components['performance_tracker'] = mock_performance_tracker
        
        return components
    
    def test_trading_engine_initialization(self, trading_config, mock_components):
        """Test trading engine initialization."""
        
        with patch('deep_momentum_trading.src.trading.trading_engine.TradingEngine') as mock_te_class:
            mock_te = Mock()
            mock_te.config = trading_config
            mock_te.alpaca_client = mock_components['alpaca']
            mock_te.order_manager = mock_components['order_manager']
            mock_te.risk_manager = mock_components['risk_manager']
            mock_te.position_manager = mock_components['position_manager']
            mock_te.performance_tracker = mock_components['performance_tracker']
            mock_te.is_running = False
            mock_te_class.return_value = mock_te
            
            from deep_momentum_trading.src.trading.trading_engine import TradingEngine
            
            engine = TradingEngine(config=trading_config)
            
            # Verify initialization
            assert engine.config == trading_config
            assert engine.is_running is False
    
    @pytest.mark.asyncio
    async def test_process_predictions(self, trading_config, mock_components):
        """Test prediction processing pipeline."""
        
        with patch('deep_momentum_trading.src.trading.trading_engine.TradingEngine') as mock_te_class:
            mock_te = Mock()
            mock_te.process_predictions = AsyncMock()
            mock_te_class.return_value = mock_te
            
            # Mock prediction processing
            mock_te.process_predictions.return_value = {
                'predictions_processed': 5,
                'signals_generated': 3,
                'orders_placed': 2,
                'processing_time_ms': 150.5
            }
            
            from deep_momentum_trading.src.trading.trading_engine import TradingEngine
            
            engine = TradingEngine(config=trading_config)
            
            # Generate test predictions
            predictions = get_sample_predictions(n_predictions=5)
            
            # Process predictions
            result = await engine.process_predictions(predictions)
            
            # Verify processing
            mock_te.process_predictions.assert_called_once()
            assert result['predictions_processed'] == 5
            assert result['signals_generated'] == 3
            assert result['orders_placed'] == 2
    
    def test_signal_generation(self, trading_config, mock_components):
        """Test trading signal generation."""
        
        with patch('deep_momentum_trading.src.trading.trading_engine.TradingEngine') as mock_te_class:
            mock_te = Mock()
            mock_te.generate_signals = Mock()
            mock_te_class.return_value = mock_te
            
            # Mock signal generation
            mock_te.generate_signals.return_value = [
                {
                    'symbol': 'AAPL',
                    'signal': 'buy',
                    'confidence': 0.75,
                    'target_quantity': 100,
                    'signal_strength': 0.8
                },
                {
                    'symbol': 'MSFT',
                    'signal': 'sell',
                    'confidence': 0.68,
                    'target_quantity': 50,
                    'signal_strength': 0.7
                }
            ]
            
            from deep_momentum_trading.src.trading.trading_engine import TradingEngine
            
            engine = TradingEngine(config=trading_config)
            
            # Generate signals from predictions
            predictions = get_sample_predictions(n_predictions=3)
            signals = engine.generate_signals(predictions)
            
            # Verify signal generation
            mock_te.generate_signals.assert_called_once()
            assert len(signals) == 2
            assert signals[0]['symbol'] == 'AAPL'
            assert signals[0]['signal'] == 'buy'
            assert signals[1]['symbol'] == 'MSFT'
            assert signals[1]['signal'] == 'sell'
    
    def test_order_execution(self, trading_config, mock_components):
        """Test order execution pipeline."""
        
        with patch('deep_momentum_trading.src.trading.trading_engine.TradingEngine') as mock_te_class:
            mock_te = Mock()
            mock_te.execute_orders = Mock()
            mock_te_class.return_value = mock_te
            
            # Mock order execution
            mock_te.execute_orders.return_value = {
                'orders_submitted': 3,
                'successful_orders': 2,
                'failed_orders': 1,
                'total_value': 45000.0,
                'execution_time_ms': 250.3
            }
            
            from deep_momentum_trading.src.trading.trading_engine import TradingEngine
            
            engine = TradingEngine(config=trading_config)
            
            # Test order execution
            signals = [
                {
                    'symbol': 'AAPL',
                    'signal': 'buy',
                    'target_quantity': 100,
                    'confidence': 0.75
                },
                {
                    'symbol': 'MSFT',
                    'signal': 'sell',
                    'target_quantity': 75,
                    'confidence': 0.68
                },
                {
                    'symbol': 'GOOGL',
                    'signal': 'buy',
                    'target_quantity': 25,
                    'confidence': 0.82
                }
            ]
            
            execution_result = engine.execute_orders(signals)
            
            # Verify order execution
            mock_te.execute_orders.assert_called_once()
            assert execution_result['orders_submitted'] == 3
            assert execution_result['successful_orders'] == 2
            assert execution_result['failed_orders'] == 1
    
    def test_risk_integration(self, trading_config, mock_components):
        """Test risk management integration."""
        
        with patch('deep_momentum_trading.src.trading.trading_engine.TradingEngine') as mock_te_class:
            mock_te = Mock()
            mock_te.apply_risk_management = Mock()
            mock_te_class.return_value = mock_te
            
            # Mock risk management application
            mock_te.apply_risk_management.return_value = {
                'signals_approved': 2,
                'signals_rejected': 1,
                'position_sizes_adjusted': 1,
                'risk_warnings': ['High correlation detected']
            }
            
            from deep_momentum_trading.src.trading.trading_engine import TradingEngine
            
            engine = TradingEngine(config=trading_config)
            
            # Test risk management
            signals = [
                {'symbol': 'AAPL', 'signal': 'buy', 'confidence': 0.75},
                {'symbol': 'MSFT', 'signal': 'buy', 'confidence': 0.68},
                {'symbol': 'TSLA', 'signal': 'buy', 'confidence': 0.55}  # Below threshold
            ]
            
            risk_result = engine.apply_risk_management(signals)
            
            # Verify risk management
            mock_te.apply_risk_management.assert_called_once()
            assert risk_result['signals_approved'] == 2
            assert risk_result['signals_rejected'] == 1
            assert len(risk_result['risk_warnings']) > 0
    
    def test_position_sizing(self, trading_config, mock_components):
        """Test position sizing logic."""
        
        with patch('deep_momentum_trading.src.trading.trading_engine.TradingEngine') as mock_te_class:
            mock_te = Mock()
            mock_te.calculate_position_sizes = Mock()
            mock_te_class.return_value = mock_te
            
            # Mock position sizing
            mock_te.calculate_position_sizes.return_value = {
                'AAPL': {
                    'target_shares': 100,
                    'target_value': 15000.0,
                    'position_weight': 0.015,
                    'risk_adjusted': True
                },
                'MSFT': {
                    'target_shares': 75,
                    'target_value': 21000.0,
                    'position_weight': 0.021,
                    'risk_adjusted': True
                }
            }
            
            from deep_momentum_trading.src.trading.trading_engine import TradingEngine
            
            engine = TradingEngine(config=trading_config)
            
            # Test position sizing
            signals = [
                {'symbol': 'AAPL', 'signal': 'buy', 'confidence': 0.75},
                {'symbol': 'MSFT', 'signal': 'buy', 'confidence': 0.68}
            ]
            
            position_sizes = engine.calculate_position_sizes(signals)
            
            # Verify position sizing
            mock_te.calculate_position_sizes.assert_called_once()
            assert 'AAPL' in position_sizes
            assert 'MSFT' in position_sizes
            assert position_sizes['AAPL']['target_shares'] == 100
            assert position_sizes['MSFT']['target_shares'] == 75
    
    @pytest.mark.asyncio
    async def test_real_time_monitoring(self, trading_config, mock_components):
        """Test real-time monitoring functionality."""
        
        with patch('deep_momentum_trading.src.trading.trading_engine.TradingEngine') as mock_te_class:
            mock_te = Mock()
            mock_te.start_monitoring = AsyncMock()
            mock_te.stop_monitoring = AsyncMock()
            mock_te.get_system_status = Mock()
            mock_te_class.return_value = mock_te
            
            # Mock monitoring
            mock_te.start_monitoring.return_value = {
                'monitoring_started': True,
                'components_active': 5,
                'start_time': datetime.now(timezone.utc).isoformat()
            }
            
            mock_te.get_system_status.return_value = {
                'engine_status': 'running',
                'active_orders': 3,
                'portfolio_value': 1015000.0,
                'daily_pnl': 15000.0,
                'risk_utilization': 0.75,
                'last_update': datetime.now(timezone.utc).isoformat()
            }
            
            from deep_momentum_trading.src.trading.trading_engine import TradingEngine
            
            engine = TradingEngine(config=trading_config)
            
            # Test monitoring start
            start_result = await engine.start_monitoring()
            
            # Verify monitoring start
            mock_te.start_monitoring.assert_called_once()
            assert start_result['monitoring_started'] is True
            assert start_result['components_active'] == 5
            
            # Test system status
            status = engine.get_system_status()
            
            # Verify system status
            mock_te.get_system_status.assert_called_once()
            assert status['engine_status'] == 'running'
            assert status['active_orders'] == 3
            assert status['daily_pnl'] == 15000.0
    
    def test_performance_tracking(self, trading_config, mock_components):
        """Test performance tracking integration."""
        
        with patch('deep_momentum_trading.src.trading.trading_engine.TradingEngine') as mock_te_class:
            mock_te = Mock()
            mock_te.update_performance = Mock()
            mock_te.get_performance_summary = Mock()
            mock_te_class.return_value = mock_te
            
            # Mock performance tracking
            mock_te.update_performance.return_value = {
                'performance_updated': True,
                'trades_processed': 5,
                'pnl_calculated': True
            }
            
            mock_te.get_performance_summary.return_value = {
                'total_return': 0.085,
                'sharpe_ratio': 1.25,
                'max_drawdown': 0.032,
                'win_rate': 0.68,
                'total_trades': 150,
                'profitable_trades': 102
            }
            
            from deep_momentum_trading.src.trading.trading_engine import TradingEngine
            
            engine = TradingEngine(config=trading_config)
            
            # Test performance update
            trade_data = [
                {
                    'symbol': 'AAPL',
                    'quantity': 100,
                    'price': 150.0,
                    'side': 'buy',
                    'timestamp': datetime.now(timezone.utc).isoformat()
                }
            ]
            
            update_result = engine.update_performance(trade_data)
            
            # Verify performance update
            mock_te.update_performance.assert_called_once()
            assert update_result['performance_updated'] is True
            assert update_result['trades_processed'] == 5
            
            # Test performance summary
            summary = engine.get_performance_summary()
            
            # Verify performance summary
            mock_te.get_performance_summary.assert_called_once()
            assert summary['total_return'] == 0.085
            assert summary['win_rate'] == 0.68
    
    def test_trading_hours_validation(self, trading_config, mock_components):
        """Test trading hours validation."""
        
        with patch('deep_momentum_trading.src.trading.trading_engine.TradingEngine') as mock_te_class:
            mock_te = Mock()
            mock_te.is_market_open = Mock()
            mock_te.get_market_hours = Mock()
            mock_te_class.return_value = mock_te
            
            # Mock market hours
            mock_te.is_market_open.return_value = True
            mock_te.get_market_hours.return_value = {
                'market_open': True,
                'open_time': '09:30:00',
                'close_time': '16:00:00',
                'timezone': 'US/Eastern',
                'current_time': '14:30:00'
            }
            
            from deep_momentum_trading.src.trading.trading_engine import TradingEngine
            
            engine = TradingEngine(config=trading_config)
            
            # Test market hours check
            is_open = engine.is_market_open()
            market_hours = engine.get_market_hours()
            
            # Verify market hours validation
            mock_te.is_market_open.assert_called_once()
            mock_te.get_market_hours.assert_called_once()
            assert is_open is True
            assert market_hours['market_open'] is True
    
    def test_error_handling(self, trading_config, mock_components):
        """Test error handling and recovery."""
        
        with patch('deep_momentum_trading.src.trading.trading_engine.TradingEngine') as mock_te_class:
            mock_te = Mock()
            mock_te.handle_error = Mock()
            mock_te.recover_from_error = Mock()
            mock_te_class.return_value = mock_te
            
            # Mock error handling
            mock_te.handle_error.return_value = {
                'error_handled': True,
                'error_type': 'api_connection_error',
                'recovery_action': 'retry_with_backoff',
                'retry_count': 1
            }
            
            mock_te.recover_from_error.return_value = {
                'recovery_successful': True,
                'system_restored': True,
                'recovery_time_seconds': 5.2
            }
            
            from deep_momentum_trading.src.trading.trading_engine import TradingEngine
            
            engine = TradingEngine(config=trading_config)
            
            # Test error handling
            error = Exception("API connection failed")
            error_result = engine.handle_error(error)
            
            # Verify error handling
            mock_te.handle_error.assert_called_once()
            assert error_result['error_handled'] is True
            assert error_result['error_type'] == 'api_connection_error'
            
            # Test recovery
            recovery_result = engine.recover_from_error(error_result)
            
            # Verify recovery
            mock_te.recover_from_error.assert_called_once()
            assert recovery_result['recovery_successful'] is True
    
    @pytest.mark.asyncio
    async def test_engine_lifecycle(self, trading_config, mock_components):
        """Test trading engine lifecycle management."""
        
        with patch('deep_momentum_trading.src.trading.trading_engine.TradingEngine') as mock_te_class:
            mock_te = Mock()
            mock_te.start = AsyncMock()
            mock_te.stop = AsyncMock()
            mock_te.pause = AsyncMock()
            mock_te.resume = AsyncMock()
            mock_te.is_running = False
            mock_te_class.return_value = mock_te
            
            # Mock lifecycle operations
            mock_te.start.return_value = {
                'engine_started': True,
                'start_time': datetime.now(timezone.utc).isoformat(),
                'components_initialized': 5
            }
            
            mock_te.stop.return_value = {
                'engine_stopped': True,
                'stop_time': datetime.now(timezone.utc).isoformat(),
                'cleanup_completed': True
            }
            
            mock_te.pause.return_value = {
                'engine_paused': True,
                'active_orders_preserved': True
            }
            
            mock_te.resume.return_value = {
                'engine_resumed': True,
                'operations_restored': True
            }
            
            from deep_momentum_trading.src.trading.trading_engine import TradingEngine
            
            engine = TradingEngine(config=trading_config)
            
            # Test engine start
            start_result = await engine.start()
            mock_te.start.assert_called_once()
            assert start_result['engine_started'] is True
            
            # Test engine pause
            pause_result = await engine.pause()
            mock_te.pause.assert_called_once()
            assert pause_result['engine_paused'] is True
            
            # Test engine resume
            resume_result = await engine.resume()
            mock_te.resume.assert_called_once()
            assert resume_result['engine_resumed'] is True
            
            # Test engine stop
            stop_result = await engine.stop()
            mock_te.stop.assert_called_once()
            assert stop_result['engine_stopped'] is True
    
    def test_configuration_validation(self, trading_config, mock_components):
        """Test configuration validation."""
        
        with patch('deep_momentum_trading.src.trading.trading_engine.TradingEngine') as mock_te_class:
            mock_te = Mock()
            mock_te.validate_config = Mock()
            mock_te_class.return_value = mock_te
            
            # Mock config validation
            mock_te.validate_config.return_value = {
                'config_valid': True,
                'warnings': ['High risk limit detected'],
                'errors': [],
                'recommendations': ['Consider lowering max_position_size']
            }
            
            from deep_momentum_trading.src.trading.trading_engine import TradingEngine
            
            engine = TradingEngine(config=trading_config)
            
            # Test config validation
            validation_result = engine.validate_config()
            
            # Verify config validation
            mock_te.validate_config.assert_called_once()
            assert validation_result['config_valid'] is True
            assert len(validation_result['warnings']) > 0
            assert len(validation_result['recommendations']) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
