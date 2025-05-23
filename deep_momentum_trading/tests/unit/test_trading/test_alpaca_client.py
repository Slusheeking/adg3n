"""
Unit tests for Alpaca Client.

Tests the Alpaca trading client functionality including account management,
order placement, position tracking, and market data integration.
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from datetime import datetime, timezone, timedelta
import pandas as pd
import numpy as np
from typing import Dict, List, Any

from deep_momentum_trading.tests.fixtures.sample_data import (
    get_sample_market_data,
    get_sample_portfolio
)
from deep_momentum_trading.tests.fixtures.test_configs import TestConfigManager


class TestAlpacaClient:
    """Test Alpaca client functionality."""
    
    @pytest.fixture
    def alpaca_config(self):
        """Get Alpaca client configuration."""
        return {
            'api_key': 'test_api_key',
            'secret_key': 'test_secret_key',
            'base_url': 'https://paper-api.alpaca.markets',
            'data_url': 'https://data.alpaca.markets',
            'enable_paper_trading': True,
            'timeout_seconds': 30,
            'retry_attempts': 3,
            'rate_limit_per_minute': 200
        }
    
    @pytest.fixture
    def mock_alpaca_api(self):
        """Mock Alpaca API responses."""
        mock_api = Mock()
        
        # Mock account information
        mock_api.get_account.return_value = {
            'id': 'test_account_id',
            'account_number': '123456789',
            'status': 'ACTIVE',
            'currency': 'USD',
            'buying_power': '500000.00',
            'regt_buying_power': '500000.00',
            'daytrading_buying_power': '1000000.00',
            'cash': '250000.00',
            'portfolio_value': '1000000.00',
            'equity': '1000000.00',
            'last_equity': '995000.00',
            'multiplier': '2',
            'created_at': '2023-01-01T00:00:00Z',
            'trading_blocked': False,
            'transfers_blocked': False,
            'account_blocked': False,
            'pattern_day_trader': False
        }
        
        # Mock positions
        mock_api.list_positions.return_value = [
            {
                'asset_id': 'asset_1',
                'symbol': 'AAPL',
                'exchange': 'NASDAQ',
                'asset_class': 'us_equity',
                'qty': '100',
                'side': 'long',
                'market_value': '15000.00',
                'cost_basis': '14500.00',
                'unrealized_pl': '500.00',
                'unrealized_plpc': '0.0345',
                'avg_entry_price': '145.00',
                'current_price': '150.00'
            },
            {
                'asset_id': 'asset_2',
                'symbol': 'MSFT',
                'exchange': 'NASDAQ',
                'asset_class': 'us_equity',
                'qty': '75',
                'side': 'long',
                'market_value': '21000.00',
                'cost_basis': '20250.00',
                'unrealized_pl': '750.00',
                'unrealized_plpc': '0.037',
                'avg_entry_price': '270.00',
                'current_price': '280.00'
            }
        ]
        
        # Mock orders
        mock_api.list_orders.return_value = [
            {
                'id': 'order_1',
                'client_order_id': 'client_order_1',
                'created_at': '2023-12-01T10:00:00Z',
                'updated_at': '2023-12-01T10:00:05Z',
                'submitted_at': '2023-12-01T10:00:00Z',
                'filled_at': '2023-12-01T10:00:05Z',
                'expired_at': None,
                'canceled_at': None,
                'failed_at': None,
                'asset_id': 'asset_3',
                'symbol': 'GOOGL',
                'asset_class': 'us_equity',
                'qty': '25',
                'filled_qty': '25',
                'type': 'market',
                'side': 'buy',
                'time_in_force': 'day',
                'limit_price': None,
                'stop_price': None,
                'status': 'filled',
                'extended_hours': False,
                'legs': None,
                'trail_percent': None,
                'trail_price': None,
                'hwm': None,
                'filled_avg_price': '2800.00'
            }
        ]
        
        return mock_api
    
    def test_client_initialization(self, alpaca_config):
        """Test Alpaca client initialization."""
        
        with patch('deep_momentum_trading.src.trading.alpaca_client.tradeapi') as mock_tradeapi:
            mock_tradeapi.REST.return_value = Mock()
            
            from deep_momentum_trading.src.trading.alpaca_client import AlpacaClient
            
            client = AlpacaClient(config=alpaca_config)
            
            # Verify client initialization
            assert client.config == alpaca_config
            assert client.paper_trading == alpaca_config['enable_paper_trading']
            
            # Verify API client creation
            mock_tradeapi.REST.assert_called_once_with(
                key_id=alpaca_config['api_key'],
                secret_key=alpaca_config['secret_key'],
                base_url=alpaca_config['base_url'],
                api_version='v2'
            )
    
    def test_get_account_info(self, alpaca_config, mock_alpaca_api):
        """Test account information retrieval."""
        
        with patch('deep_momentum_trading.src.trading.alpaca_client.tradeapi') as mock_tradeapi:
            mock_tradeapi.REST.return_value = mock_alpaca_api
            
            from deep_momentum_trading.src.trading.alpaca_client import AlpacaClient
            
            client = AlpacaClient(config=alpaca_config)
            account_info = client.get_account_info()
            
            # Verify account information
            assert account_info['status'] == 'ACTIVE'
            assert float(account_info['portfolio_value']) == 1000000.0
            assert float(account_info['buying_power']) == 500000.0
            assert float(account_info['cash']) == 250000.0
            assert account_info['pattern_day_trader'] is False
            
            # Verify API call
            mock_alpaca_api.get_account.assert_called_once()
    
    def test_get_positions(self, alpaca_config, mock_alpaca_api):
        """Test position retrieval."""
        
        with patch('deep_momentum_trading.src.trading.alpaca_client.tradeapi') as mock_tradeapi:
            mock_tradeapi.REST.return_value = mock_alpaca_api
            
            from deep_momentum_trading.src.trading.alpaca_client import AlpacaClient
            
            client = AlpacaClient(config=alpaca_config)
            positions = client.get_positions()
            
            # Verify positions
            assert len(positions) == 2
            
            aapl_position = next(pos for pos in positions if pos['symbol'] == 'AAPL')
            assert int(aapl_position['qty']) == 100
            assert float(aapl_position['market_value']) == 15000.0
            assert float(aapl_position['unrealized_pl']) == 500.0
            
            msft_position = next(pos for pos in positions if pos['symbol'] == 'MSFT')
            assert int(msft_position['qty']) == 75
            assert float(msft_position['market_value']) == 21000.0
            
            # Verify API call
            mock_alpaca_api.list_positions.assert_called_once()
    
    def test_submit_order(self, alpaca_config, mock_alpaca_api):
        """Test order submission."""
        
        # Mock successful order submission
        mock_alpaca_api.submit_order.return_value = {
            'id': 'new_order_123',
            'client_order_id': 'client_new_order_123',
            'created_at': datetime.now(timezone.utc).isoformat(),
            'symbol': 'TSLA',
            'qty': '50',
            'side': 'buy',
            'type': 'market',
            'time_in_force': 'day',
            'status': 'accepted'
        }
        
        with patch('deep_momentum_trading.src.trading.alpaca_client.tradeapi') as mock_tradeapi:
            mock_tradeapi.REST.return_value = mock_alpaca_api
            
            from deep_momentum_trading.src.trading.alpaca_client import AlpacaClient
            
            client = AlpacaClient(config=alpaca_config)
            
            # Test market order
            order_request = {
                'symbol': 'TSLA',
                'qty': 50,
                'side': 'buy',
                'type': 'market',
                'time_in_force': 'day'
            }
            
            order_result = client.submit_order(order_request)
            
            # Verify order submission
            assert order_result['symbol'] == 'TSLA'
            assert order_result['qty'] == '50'
            assert order_result['side'] == 'buy'
            assert order_result['status'] == 'accepted'
            
            # Verify API call
            mock_alpaca_api.submit_order.assert_called_once_with(
                symbol='TSLA',
                qty=50,
                side='buy',
                type='market',
                time_in_force='day'
            )
    
    def test_submit_limit_order(self, alpaca_config, mock_alpaca_api):
        """Test limit order submission."""
        
        # Mock limit order submission
        mock_alpaca_api.submit_order.return_value = {
            'id': 'limit_order_456',
            'symbol': 'NVDA',
            'qty': '30',
            'side': 'sell',
            'type': 'limit',
            'limit_price': '500.00',
            'time_in_force': 'gtc',
            'status': 'accepted'
        }
        
        with patch('deep_momentum_trading.src.trading.alpaca_client.tradeapi') as mock_tradeapi:
            mock_tradeapi.REST.return_value = mock_alpaca_api
            
            from deep_momentum_trading.src.trading.alpaca_client import AlpacaClient
            
            client = AlpacaClient(config=alpaca_config)
            
            # Test limit order
            order_request = {
                'symbol': 'NVDA',
                'qty': 30,
                'side': 'sell',
                'type': 'limit',
                'limit_price': 500.00,
                'time_in_force': 'gtc'
            }
            
            order_result = client.submit_order(order_request)
            
            # Verify limit order
            assert order_result['symbol'] == 'NVDA'
            assert order_result['type'] == 'limit'
            assert order_result['limit_price'] == '500.00'
            assert order_result['time_in_force'] == 'gtc'
            
            # Verify API call
            mock_alpaca_api.submit_order.assert_called_once_with(
                symbol='NVDA',
                qty=30,
                side='sell',
                type='limit',
                limit_price=500.00,
                time_in_force='gtc'
            )
    
    def test_get_orders(self, alpaca_config, mock_alpaca_api):
        """Test order retrieval."""
        
        with patch('deep_momentum_trading.src.trading.alpaca_client.tradeapi') as mock_tradeapi:
            mock_tradeapi.REST.return_value = mock_alpaca_api
            
            from deep_momentum_trading.src.trading.alpaca_client import AlpacaClient
            
            client = AlpacaClient(config=alpaca_config)
            orders = client.get_orders()
            
            # Verify orders
            assert len(orders) == 1
            order = orders[0]
            assert order['symbol'] == 'GOOGL'
            assert order['status'] == 'filled'
            assert int(order['qty']) == 25
            assert float(order['filled_avg_price']) == 2800.0
            
            # Verify API call
            mock_alpaca_api.list_orders.assert_called_once()
    
    def test_cancel_order(self, alpaca_config, mock_alpaca_api):
        """Test order cancellation."""
        
        # Mock order cancellation
        mock_alpaca_api.cancel_order.return_value = {
            'id': 'order_to_cancel',
            'status': 'canceled',
            'canceled_at': datetime.now(timezone.utc).isoformat()
        }
        
        with patch('deep_momentum_trading.src.trading.alpaca_client.tradeapi') as mock_tradeapi:
            mock_tradeapi.REST.return_value = mock_alpaca_api
            
            from deep_momentum_trading.src.trading.alpaca_client import AlpacaClient
            
            client = AlpacaClient(config=alpaca_config)
            result = client.cancel_order('order_to_cancel')
            
            # Verify cancellation
            assert result['status'] == 'canceled'
            assert 'canceled_at' in result
            
            # Verify API call
            mock_alpaca_api.cancel_order.assert_called_once_with('order_to_cancel')
    
    def test_get_market_data(self, alpaca_config):
        """Test market data retrieval."""
        
        # Mock market data response
        mock_bars = Mock()
        mock_bars.df = pd.DataFrame({
            'open': [150.0, 151.0, 152.0],
            'high': [152.0, 153.0, 154.0],
            'low': [149.0, 150.0, 151.0],
            'close': [151.0, 152.0, 153.0],
            'volume': [1000000, 1100000, 1200000]
        }, index=pd.date_range('2023-12-01', periods=3, freq='D'))
        
        with patch('deep_momentum_trading.src.trading.alpaca_client.tradeapi') as mock_tradeapi:
            mock_api = Mock()
            mock_api.get_bars.return_value = mock_bars
            mock_tradeapi.REST.return_value = mock_api
            
            from deep_momentum_trading.src.trading.alpaca_client import AlpacaClient
            
            client = AlpacaClient(config=alpaca_config)
            market_data = client.get_market_data('AAPL', '1Day', limit=3)
            
            # Verify market data
            assert len(market_data) == 3
            assert 'open' in market_data.columns
            assert 'high' in market_data.columns
            assert 'low' in market_data.columns
            assert 'close' in market_data.columns
            assert 'volume' in market_data.columns
            
            # Verify API call
            mock_api.get_bars.assert_called_once()
    
    def test_error_handling(self, alpaca_config):
        """Test error handling in Alpaca client."""
        
        with patch('deep_momentum_trading.src.trading.alpaca_client.tradeapi') as mock_tradeapi:
            mock_api = Mock()
            mock_api.get_account.side_effect = Exception("API Error")
            mock_tradeapi.REST.return_value = mock_api
            
            from deep_momentum_trading.src.trading.alpaca_client import AlpacaClient
            
            client = AlpacaClient(config=alpaca_config)
            
            # Test error handling
            with pytest.raises(Exception):
                client.get_account_info()
    
    def test_rate_limiting(self, alpaca_config):
        """Test rate limiting functionality."""
        
        with patch('deep_momentum_trading.src.trading.alpaca_client.tradeapi') as mock_tradeapi, \
             patch('time.sleep') as mock_sleep:
            
            mock_api = Mock()
            mock_api.get_account.return_value = {'status': 'ACTIVE'}
            mock_tradeapi.REST.return_value = mock_api
            
            from deep_momentum_trading.src.trading.alpaca_client import AlpacaClient
            
            client = AlpacaClient(config=alpaca_config)
            
            # Simulate multiple rapid calls
            for _ in range(5):
                client.get_account_info()
            
            # Verify rate limiting is considered
            assert mock_api.get_account.call_count == 5
    
    def test_paper_trading_mode(self, alpaca_config):
        """Test paper trading mode configuration."""
        
        # Test paper trading enabled
        paper_config = alpaca_config.copy()
        paper_config['enable_paper_trading'] = True
        paper_config['base_url'] = 'https://paper-api.alpaca.markets'
        
        with patch('deep_momentum_trading.src.trading.alpaca_client.tradeapi') as mock_tradeapi:
            mock_tradeapi.REST.return_value = Mock()
            
            from deep_momentum_trading.src.trading.alpaca_client import AlpacaClient
            
            client = AlpacaClient(config=paper_config)
            
            # Verify paper trading configuration
            assert client.paper_trading is True
            mock_tradeapi.REST.assert_called_with(
                key_id=paper_config['api_key'],
                secret_key=paper_config['secret_key'],
                base_url='https://paper-api.alpaca.markets',
                api_version='v2'
            )
    
    def test_live_trading_mode(self, alpaca_config):
        """Test live trading mode configuration."""
        
        # Test live trading
        live_config = alpaca_config.copy()
        live_config['enable_paper_trading'] = False
        live_config['base_url'] = 'https://api.alpaca.markets'
        
        with patch('deep_momentum_trading.src.trading.alpaca_client.tradeapi') as mock_tradeapi:
            mock_tradeapi.REST.return_value = Mock()
            
            from deep_momentum_trading.src.trading.alpaca_client import AlpacaClient
            
            client = AlpacaClient(config=live_config)
            
            # Verify live trading configuration
            assert client.paper_trading is False
            mock_tradeapi.REST.assert_called_with(
                key_id=live_config['api_key'],
                secret_key=live_config['secret_key'],
                base_url='https://api.alpaca.markets',
                api_version='v2'
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
