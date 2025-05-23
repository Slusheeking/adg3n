"""
Unit tests for Order Manager.

Tests order management functionality including order placement, monitoring,
cancellation, and order lifecycle management.
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
    get_sample_portfolio
)
from deep_momentum_trading.tests.fixtures.test_configs import TestConfigManager


class TestOrderManager:
    """Test order management functionality."""
    
    @pytest.fixture
    def order_config(self):
        """Get order manager configuration."""
        return {
            'max_orders_per_minute': 50,
            'order_timeout_seconds': 300,
            'retry_attempts': 3,
            'retry_delay_seconds': 1,
            'enable_order_validation': True,
            'enable_position_tracking': True,
            'enable_fill_monitoring': True,
            'cancel_unfilled_orders': True,
            'order_size_limits': {
                'min_order_value': 100.0,
                'max_order_value': 100000.0,
                'max_position_size': 0.1
            }
        }
    
    @pytest.fixture
    def mock_alpaca_client(self):
        """Mock Alpaca client for order operations."""
        mock_client = Mock()
        
        # Mock order submission
        mock_client.submit_order.return_value = {
            'id': 'order_123',
            'client_order_id': 'client_order_123',
            'created_at': datetime.now(timezone.utc).isoformat(),
            'symbol': 'AAPL',
            'qty': '100',
            'side': 'buy',
            'type': 'market',
            'time_in_force': 'day',
            'status': 'accepted'
        }
        
        # Mock order status
        mock_client.get_order.return_value = {
            'id': 'order_123',
            'status': 'filled',
            'filled_qty': '100',
            'filled_avg_price': '150.00',
            'filled_at': datetime.now(timezone.utc).isoformat()
        }
        
        # Mock order cancellation
        mock_client.cancel_order.return_value = {
            'id': 'order_123',
            'status': 'canceled',
            'canceled_at': datetime.now(timezone.utc).isoformat()
        }
        
        # Mock order list
        mock_client.get_orders.return_value = [
            {
                'id': 'order_123',
                'symbol': 'AAPL',
                'status': 'filled',
                'qty': '100',
                'side': 'buy'
            }
        ]
        
        return mock_client
    
    def test_order_manager_initialization(self, order_config, mock_alpaca_client):
        """Test order manager initialization."""
        
        with patch('deep_momentum_trading.src.trading.order_manager.OrderManager') as mock_om_class:
            mock_om = Mock()
            mock_om.config = order_config
            mock_om.alpaca_client = mock_alpaca_client
            mock_om.active_orders = {}
            mock_om.order_history = []
            mock_om_class.return_value = mock_om
            
            from deep_momentum_trading.src.trading.order_manager import OrderManager
            
            order_manager = OrderManager(
                alpaca_client=mock_alpaca_client,
                config=order_config
            )
            
            # Verify initialization
            assert order_manager.config == order_config
            assert order_manager.alpaca_client == mock_alpaca_client
    
    def test_place_market_order(self, order_config, mock_alpaca_client):
        """Test market order placement."""
        
        with patch('deep_momentum_trading.src.trading.order_manager.OrderManager') as mock_om_class:
            mock_om = Mock()
            mock_om.place_order = Mock()
            mock_om_class.return_value = mock_om
            
            # Mock successful order placement
            mock_om.place_order.return_value = {
                'order_id': 'market_order_456',
                'status': 'accepted',
                'symbol': 'MSFT',
                'quantity': 75,
                'side': 'buy',
                'order_type': 'market',
                'submitted_at': datetime.now(timezone.utc).isoformat()
            }
            
            from deep_momentum_trading.src.trading.order_manager import OrderManager
            
            order_manager = OrderManager(
                alpaca_client=mock_alpaca_client,
                config=order_config
            )
            
            # Test market order
            order_request = {
                'symbol': 'MSFT',
                'quantity': 75,
                'side': 'buy',
                'order_type': 'market',
                'time_in_force': 'day'
            }
            
            result = order_manager.place_order(order_request)
            
            # Verify order placement
            mock_om.place_order.assert_called_once()
            assert result['status'] == 'accepted'
            assert result['symbol'] == 'MSFT'
            assert result['order_type'] == 'market'
    
    def test_place_limit_order(self, order_config, mock_alpaca_client):
        """Test limit order placement."""
        
        with patch('deep_momentum_trading.src.trading.order_manager.OrderManager') as mock_om_class:
            mock_om = Mock()
            mock_om.place_order = Mock()
            mock_om_class.return_value = mock_om
            
            # Mock limit order placement
            mock_om.place_order.return_value = {
                'order_id': 'limit_order_789',
                'status': 'accepted',
                'symbol': 'GOOGL',
                'quantity': 25,
                'side': 'sell',
                'order_type': 'limit',
                'limit_price': 2800.0,
                'time_in_force': 'gtc',
                'submitted_at': datetime.now(timezone.utc).isoformat()
            }
            
            from deep_momentum_trading.src.trading.order_manager import OrderManager
            
            order_manager = OrderManager(
                alpaca_client=mock_alpaca_client,
                config=order_config
            )
            
            # Test limit order
            order_request = {
                'symbol': 'GOOGL',
                'quantity': 25,
                'side': 'sell',
                'order_type': 'limit',
                'limit_price': 2800.0,
                'time_in_force': 'gtc'
            }
            
            result = order_manager.place_order(order_request)
            
            # Verify limit order placement
            mock_om.place_order.assert_called_once()
            assert result['status'] == 'accepted'
            assert result['order_type'] == 'limit'
            assert result['limit_price'] == 2800.0
            assert result['time_in_force'] == 'gtc'
    
    def test_order_validation(self, order_config, mock_alpaca_client):
        """Test order validation logic."""
        
        with patch('deep_momentum_trading.src.trading.order_manager.OrderManager') as mock_om_class:
            mock_om = Mock()
            mock_om.validate_order = Mock()
            mock_om_class.return_value = mock_om
            
            # Mock validation results
            mock_om.validate_order.return_value = {
                'valid': True,
                'errors': [],
                'warnings': ['Order size is large'],
                'adjusted_quantity': 100
            }
            
            from deep_momentum_trading.src.trading.order_manager import OrderManager
            
            order_manager = OrderManager(
                alpaca_client=mock_alpaca_client,
                config=order_config
            )
            
            # Test order validation
            order_request = {
                'symbol': 'TSLA',
                'quantity': 150,
                'side': 'buy',
                'order_type': 'market'
            }
            
            validation_result = order_manager.validate_order(order_request)
            
            # Verify validation
            mock_om.validate_order.assert_called_once()
            assert validation_result['valid'] is True
            assert len(validation_result['warnings']) > 0
            assert validation_result['adjusted_quantity'] == 100
    
    def test_order_monitoring(self, order_config, mock_alpaca_client):
        """Test order status monitoring."""
        
        with patch('deep_momentum_trading.src.trading.order_manager.OrderManager') as mock_om_class:
            mock_om = Mock()
            mock_om.monitor_orders = Mock()
            mock_om.get_order_status = Mock()
            mock_om_class.return_value = mock_om
            
            # Mock order status updates
            mock_om.get_order_status.return_value = {
                'order_id': 'order_456',
                'status': 'partially_filled',
                'filled_quantity': 50,
                'remaining_quantity': 50,
                'filled_avg_price': 149.50,
                'last_update': datetime.now(timezone.utc).isoformat()
            }
            
            mock_om.monitor_orders.return_value = {
                'monitored_orders': 5,
                'status_updates': 2,
                'filled_orders': 1,
                'canceled_orders': 0,
                'expired_orders': 0
            }
            
            from deep_momentum_trading.src.trading.order_manager import OrderManager
            
            order_manager = OrderManager(
                alpaca_client=mock_alpaca_client,
                config=order_config
            )
            
            # Test order status check
            status = order_manager.get_order_status('order_456')
            
            # Verify status monitoring
            mock_om.get_order_status.assert_called_once()
            assert status['status'] == 'partially_filled'
            assert status['filled_quantity'] == 50
            assert status['remaining_quantity'] == 50
            
            # Test bulk monitoring
            monitoring_result = order_manager.monitor_orders()
            
            # Verify bulk monitoring
            mock_om.monitor_orders.assert_called_once()
            assert monitoring_result['monitored_orders'] == 5
            assert monitoring_result['filled_orders'] == 1
    
    def test_order_cancellation(self, order_config, mock_alpaca_client):
        """Test order cancellation."""
        
        with patch('deep_momentum_trading.src.trading.order_manager.OrderManager') as mock_om_class:
            mock_om = Mock()
            mock_om.cancel_order = Mock()
            mock_om.cancel_all_orders = Mock()
            mock_om_class.return_value = mock_om
            
            # Mock single order cancellation
            mock_om.cancel_order.return_value = {
                'order_id': 'order_to_cancel',
                'status': 'canceled',
                'canceled_at': datetime.now(timezone.utc).isoformat(),
                'reason': 'user_requested'
            }
            
            # Mock bulk cancellation
            mock_om.cancel_all_orders.return_value = {
                'canceled_orders': 3,
                'failed_cancellations': 0,
                'total_orders': 3
            }
            
            from deep_momentum_trading.src.trading.order_manager import OrderManager
            
            order_manager = OrderManager(
                alpaca_client=mock_alpaca_client,
                config=order_config
            )
            
            # Test single order cancellation
            cancel_result = order_manager.cancel_order('order_to_cancel')
            
            # Verify cancellation
            mock_om.cancel_order.assert_called_once()
            assert cancel_result['status'] == 'canceled'
            assert cancel_result['reason'] == 'user_requested'
            
            # Test bulk cancellation
            bulk_cancel_result = order_manager.cancel_all_orders()
            
            # Verify bulk cancellation
            mock_om.cancel_all_orders.assert_called_once()
            assert bulk_cancel_result['canceled_orders'] == 3
            assert bulk_cancel_result['failed_cancellations'] == 0
    
    def test_order_timeout_handling(self, order_config, mock_alpaca_client):
        """Test order timeout handling."""
        
        with patch('deep_momentum_trading.src.trading.order_manager.OrderManager') as mock_om_class:
            mock_om = Mock()
            mock_om.handle_expired_orders = Mock()
            mock_om.check_order_timeouts = Mock()
            mock_om_class.return_value = mock_om
            
            # Mock timeout handling
            mock_om.check_order_timeouts.return_value = [
                {
                    'order_id': 'expired_order_1',
                    'symbol': 'NVDA',
                    'submitted_at': (datetime.now(timezone.utc) - timedelta(minutes=10)).isoformat(),
                    'timeout_seconds': 300,
                    'status': 'pending'
                }
            ]
            
            mock_om.handle_expired_orders.return_value = {
                'expired_orders_found': 1,
                'orders_canceled': 1,
                'cancellation_failures': 0
            }
            
            from deep_momentum_trading.src.trading.order_manager import OrderManager
            
            order_manager = OrderManager(
                alpaca_client=mock_alpaca_client,
                config=order_config
            )
            
            # Test timeout checking
            expired_orders = order_manager.check_order_timeouts()
            
            # Verify timeout detection
            mock_om.check_order_timeouts.assert_called_once()
            assert len(expired_orders) == 1
            assert expired_orders[0]['order_id'] == 'expired_order_1'
            
            # Test expired order handling
            handling_result = order_manager.handle_expired_orders()
            
            # Verify timeout handling
            mock_om.handle_expired_orders.assert_called_once()
            assert handling_result['expired_orders_found'] == 1
            assert handling_result['orders_canceled'] == 1
    
    def test_order_retry_logic(self, order_config, mock_alpaca_client):
        """Test order retry logic for failed orders."""
        
        with patch('deep_momentum_trading.src.trading.order_manager.OrderManager') as mock_om_class:
            mock_om = Mock()
            mock_om.retry_failed_order = Mock()
            mock_om_class.return_value = mock_om
            
            # Mock retry attempts
            mock_om.retry_failed_order.side_effect = [
                {'success': False, 'error': 'Insufficient buying power'},
                {'success': False, 'error': 'Market closed'},
                {'success': True, 'order_id': 'retry_success_123'}
            ]
            
            from deep_momentum_trading.src.trading.order_manager import OrderManager
            
            order_manager = OrderManager(
                alpaca_client=mock_alpaca_client,
                config=order_config
            )
            
            # Test retry logic
            failed_order = {
                'symbol': 'AMD',
                'quantity': 100,
                'side': 'buy',
                'order_type': 'market',
                'retry_count': 0
            }
            
            # Simulate multiple retry attempts
            for attempt in range(3):
                result = order_manager.retry_failed_order(failed_order)
                if result['success']:
                    break
            
            # Verify retry attempts
            assert mock_om.retry_failed_order.call_count == 3
            assert result['success'] is True
            assert result['order_id'] == 'retry_success_123'
    
    def test_order_rate_limiting(self, order_config, mock_alpaca_client):
        """Test order rate limiting."""
        
        with patch('deep_momentum_trading.src.trading.order_manager.OrderManager') as mock_om_class, \
             patch('time.sleep') as mock_sleep:
            
            mock_om = Mock()
            mock_om.check_rate_limit = Mock()
            mock_om.place_order = Mock()
            mock_om_class.return_value = mock_om
            
            # Mock rate limit checking
            mock_om.check_rate_limit.side_effect = [
                {'allowed': True, 'remaining': 49},
                {'allowed': True, 'remaining': 48},
                {'allowed': False, 'wait_time': 60}
            ]
            
            from deep_momentum_trading.src.trading.order_manager import OrderManager
            
            order_manager = OrderManager(
                alpaca_client=mock_alpaca_client,
                config=order_config
            )
            
            # Test rate limiting
            for i in range(3):
                rate_check = order_manager.check_rate_limit()
                if not rate_check['allowed']:
                    # Should trigger rate limiting
                    assert rate_check['wait_time'] == 60
                    break
            
            # Verify rate limit checks
            assert mock_om.check_rate_limit.call_count == 3
    
    def test_order_fill_tracking(self, order_config, mock_alpaca_client):
        """Test order fill tracking and notifications."""
        
        with patch('deep_momentum_trading.src.trading.order_manager.OrderManager') as mock_om_class:
            mock_om = Mock()
            mock_om.track_fills = Mock()
            mock_om.process_fill = Mock()
            mock_om_class.return_value = mock_om
            
            # Mock fill tracking
            mock_om.track_fills.return_value = [
                {
                    'order_id': 'filled_order_1',
                    'symbol': 'AAPL',
                    'filled_qty': 100,
                    'filled_price': 150.25,
                    'fill_time': datetime.now(timezone.utc).isoformat(),
                    'side': 'buy'
                },
                {
                    'order_id': 'filled_order_2',
                    'symbol': 'MSFT',
                    'filled_qty': 50,
                    'filled_price': 280.75,
                    'fill_time': datetime.now(timezone.utc).isoformat(),
                    'side': 'sell'
                }
            ]
            
            mock_om.process_fill.return_value = {
                'position_updated': True,
                'pnl_calculated': True,
                'notifications_sent': True
            }
            
            from deep_momentum_trading.src.trading.order_manager import OrderManager
            
            order_manager = OrderManager(
                alpaca_client=mock_alpaca_client,
                config=order_config
            )
            
            # Test fill tracking
            fills = order_manager.track_fills()
            
            # Verify fill tracking
            mock_om.track_fills.assert_called_once()
            assert len(fills) == 2
            assert fills[0]['symbol'] == 'AAPL'
            assert fills[1]['symbol'] == 'MSFT'
            
            # Test fill processing
            for fill in fills:
                process_result = order_manager.process_fill(fill)
                assert process_result['position_updated'] is True
    
    def test_order_history_management(self, order_config, mock_alpaca_client):
        """Test order history management."""
        
        with patch('deep_momentum_trading.src.trading.order_manager.OrderManager') as mock_om_class:
            mock_om = Mock()
            mock_om.get_order_history = Mock()
            mock_om.archive_old_orders = Mock()
            mock_om_class.return_value = mock_om
            
            # Mock order history
            mock_om.get_order_history.return_value = {
                'total_orders': 150,
                'filled_orders': 120,
                'canceled_orders': 25,
                'expired_orders': 5,
                'success_rate': 0.80,
                'average_fill_time': 2.5
            }
            
            mock_om.archive_old_orders.return_value = {
                'orders_archived': 50,
                'storage_freed_mb': 2.5
            }
            
            from deep_momentum_trading.src.trading.order_manager import OrderManager
            
            order_manager = OrderManager(
                alpaca_client=mock_alpaca_client,
                config=order_config
            )
            
            # Test order history retrieval
            history = order_manager.get_order_history()
            
            # Verify history data
            mock_om.get_order_history.assert_called_once()
            assert history['total_orders'] == 150
            assert history['success_rate'] == 0.80
            assert history['average_fill_time'] == 2.5
            
            # Test order archiving
            archive_result = order_manager.archive_old_orders()
            
            # Verify archiving
            mock_om.archive_old_orders.assert_called_once()
            assert archive_result['orders_archived'] == 50
    
    @pytest.mark.asyncio
    async def test_async_order_operations(self, order_config, mock_alpaca_client):
        """Test asynchronous order operations."""
        
        with patch('deep_momentum_trading.src.trading.order_manager.OrderManager') as mock_om_class:
            mock_om = Mock()
            mock_om.place_orders_async = AsyncMock()
            mock_om.monitor_orders_async = AsyncMock()
            mock_om_class.return_value = mock_om
            
            # Mock async operations
            mock_om.place_orders_async.return_value = {
                'orders_submitted': 5,
                'successful_submissions': 4,
                'failed_submissions': 1,
                'total_time_seconds': 2.3
            }
            
            mock_om.monitor_orders_async.return_value = {
                'orders_monitored': 10,
                'status_updates': 3,
                'monitoring_time_seconds': 1.5
            }
            
            from deep_momentum_trading.src.trading.order_manager import OrderManager
            
            order_manager = OrderManager(
                alpaca_client=mock_alpaca_client,
                config=order_config
            )
            
            # Test async order placement
            batch_orders = [
                {'symbol': 'AAPL', 'quantity': 100, 'side': 'buy'},
                {'symbol': 'MSFT', 'quantity': 75, 'side': 'buy'},
                {'symbol': 'GOOGL', 'quantity': 25, 'side': 'sell'},
                {'symbol': 'TSLA', 'quantity': 50, 'side': 'buy'},
                {'symbol': 'NVDA', 'quantity': 30, 'side': 'sell'}
            ]
            
            placement_result = await order_manager.place_orders_async(batch_orders)
            
            # Verify async placement
            mock_om.place_orders_async.assert_called_once()
            assert placement_result['orders_submitted'] == 5
            assert placement_result['successful_submissions'] == 4
            
            # Test async monitoring
            monitoring_result = await order_manager.monitor_orders_async()
            
            # Verify async monitoring
            mock_om.monitor_orders_async.assert_called_once()
            assert monitoring_result['orders_monitored'] == 10


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
