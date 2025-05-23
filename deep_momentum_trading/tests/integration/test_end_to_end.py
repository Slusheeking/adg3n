"""
End-to-end integration tests for Deep Momentum Trading System.

Tests the complete trading pipeline from data ingestion through feature engineering,
model predictions, risk management, and trade execution with ARM64 optimizations.
"""

import pytest
import asyncio
import numpy as np
import pandas as pd
import time
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any

from deep_momentum_trading.tests.fixtures.sample_data import (
    get_sample_market_data,
    get_sample_features_data,
    get_sample_predictions,
    get_sample_portfolio,
    get_sample_torch_data
)
from deep_momentum_trading.tests.fixtures.test_configs import (
    TestConfigManager,
    TestScenarios,
    TestEnvironments
)


@pytest.mark.integration
class TestEndToEndTradingPipeline:
    """Test complete trading pipeline integration."""
    
    @pytest.fixture
    def integration_config(self):
        """Get integration test configuration."""
        return TestScenarios.get_integration_config()
    
    @pytest.fixture
    def mock_external_services(self):
        """Mock all external services for integration testing."""
        mocks = {}
        
        # Mock Polygon client
        mock_polygon = Mock()
        mock_polygon.is_connected = True
        mock_polygon.subscribe_to_trades = AsyncMock(return_value=True)
        mock_polygon.get_historical_aggregates = AsyncMock()
        mocks['polygon_client'] = mock_polygon
        
        # Mock Alpaca client
        mock_alpaca = Mock()
        mock_alpaca.get_account = Mock(return_value={
            'equity': 100000.0,
            'buying_power': 50000.0,
            'cash': 25000.0
        })
        mock_alpaca.get_positions = Mock(return_value=[])
        mock_alpaca.submit_order = Mock()
        mock_alpaca.get_bars = Mock(return_value={})
        mocks['alpaca_client'] = mock_alpaca
        
        # Mock Redis
        mock_redis = Mock()
        mock_redis.ping = Mock(return_value=True)
        mock_redis.get = Mock(return_value=None)
        mock_redis.set = Mock(return_value=True)
        mocks['redis'] = mock_redis
        
        return mocks
    
    @pytest.mark.asyncio
    async def test_complete_trading_pipeline(self, integration_config, mock_external_services):
        """Test the complete trading pipeline from data to execution."""
        
        # Test symbols
        symbols = ["AAPL", "MSFT", "GOOGL"]
        
        with patch('deep_momentum_trading.src.data.polygon_client.PolygonClient') as mock_polygon_class, \
             patch('deep_momentum_trading.src.trading.alpaca_client.AlpacaClient') as mock_alpaca_class, \
             patch('deep_momentum_trading.src.data.feature_engineering.ZMQPublisher'), \
             patch('deep_momentum_trading.src.data.feature_engineering.UnifiedMemoryManager'), \
             patch('deep_momentum_trading.src.risk.risk_manager.ZMQSubscriber'), \
             patch('deep_momentum_trading.src.risk.risk_manager.ZMQPublisher'), \
             patch('deep_momentum_trading.src.trading.trading_engine.ZMQSubscriber'), \
             patch('deep_momentum_trading.src.trading.trading_engine.ZMQPublisher'), \
             patch('deep_momentum_trading.config.settings.config_manager') as mock_config:
            
            # Setup mock returns
            mock_polygon_class.return_value = mock_external_services['polygon_client']
            mock_alpaca_class.return_value = mock_external_services['alpaca_client']
            mock_config.get.return_value = {}
            
            # Step 1: Data Ingestion and Feature Engineering
            from deep_momentum_trading.src.data.feature_engineering import FeatureEngineeringProcess
            
            feature_processor = FeatureEngineeringProcess(config=integration_config.feature_config)
            
            # Generate market data for each symbol
            market_data_dict = {}
            features_dict = {}
            
            for symbol in symbols:
                # Generate realistic market data
                market_data = get_sample_market_data(symbol=symbol, n_records=200)
                market_data_dict[symbol] = market_data
                
                # Calculate features
                features = feature_processor.calculate_features(market_data)
                features_dict[symbol] = features
                
                # Verify feature quality
                assert len(features) == len(market_data)
                assert len(features.columns) >= 20  # Should have comprehensive features
                
                # Verify no excessive NaN values
                nan_ratio = features.isnull().sum().sum() / (len(features) * len(features.columns))
                assert nan_ratio < 0.3  # Less than 30% NaN
            
            # Step 2: Model Predictions (Mocked)
            # In real system, this would come from trained models
            predictions_dict = {}
            
            for symbol in symbols:
                # Generate realistic predictions based on features
                latest_features = features_dict[symbol].iloc[-1:].fillna(0)
                
                # Mock model prediction
                prediction = {
                    'symbol': symbol,
                    'timestamp': datetime.now(timezone.utc).isoformat(),
                    'position': np.random.uniform(-0.5, 0.5),
                    'confidence': np.random.uniform(0.3, 0.9),
                    'expected_return': np.random.uniform(-0.02, 0.03),
                    'volatility': np.random.uniform(0.1, 0.25),
                    'strategy': 'deep_momentum_lstm',
                    'model_version': '1.0.0'
                }
                predictions_dict[symbol] = prediction
            
            # Step 3: Risk Management
            from deep_momentum_trading.src.risk.risk_manager import RiskManager
            
            # Mock risk components
            with patch('deep_momentum_trading.src.risk.risk_manager.CorrelationMonitor') as mock_corr, \
                 patch('deep_momentum_trading.src.risk.risk_manager.LiquidityMonitor') as mock_liq, \
                 patch('deep_momentum_trading.src.risk.risk_manager.PortfolioOptimizer') as mock_opt, \
                 patch('deep_momentum_trading.src.risk.risk_manager.VaRCalculator') as mock_var:
                
                # Setup risk component mocks
                mock_corr.return_value = Mock()
                mock_liq.return_value = Mock()
                mock_opt.return_value = Mock()
                mock_var.return_value = Mock()
                
                # Mock position manager
                mock_position_manager = Mock()
                mock_position_manager.get_current_positions = Mock(return_value={})
                mock_position_manager.get_total_equity = Mock(return_value=100000.0)
                mock_position_manager.get_available_capital = Mock(return_value=50000.0)
                
                risk_manager = RiskManager(
                    risk_config=integration_config.risk_config,
                    position_manager=mock_position_manager,
                    alpaca_client=mock_external_services['alpaca_client']
                )
                
                # Mock risk assessment responses
                risk_manager.var_calculator.calculate_portfolio_var = Mock(return_value=0.015)
                risk_manager.correlation_monitor.assess_portfolio_correlation_risk = Mock(
                    return_value={'overall_correlation_risk': 0.25}
                )
                risk_manager.liquidity_monitor.assess_portfolio_liquidity_risk = Mock(
                    return_value={'illiquid_percentage': 5.0}
                )
                
                # Process predictions through risk manager
                risk_manager._process_predictions("predictions", {"data": predictions_dict})
                
                # Verify risk assessments were created
                assert len(risk_manager.assessment_history) > 0
                assert risk_manager.performance_stats["total_assessments"] > 0
                
                # Get approved predictions
                approved_count = risk_manager.performance_stats.get("approved_predictions", 0)
                scaled_count = risk_manager.performance_stats.get("scaled_predictions", 0)
                rejected_count = risk_manager.performance_stats.get("rejected_predictions", 0)
                
                # Should have processed all predictions
                assert approved_count + scaled_count + rejected_count == len(predictions_dict)
            
            # Step 4: Trade Execution (Mocked)
            from deep_momentum_trading.src.trading.trading_engine import TradingEngine
            
            with patch('deep_momentum_trading.src.trading.trading_engine.OrderManager') as mock_order_mgr, \
                 patch('deep_momentum_trading.src.trading.trading_engine.PositionManager') as mock_pos_mgr:
                
                # Setup trading component mocks
                mock_order_mgr.return_value = Mock()
                mock_pos_mgr.return_value = Mock()
                
                trading_engine = TradingEngine(
                    trading_config=integration_config.trading_config,
                    alpaca_client=mock_external_services['alpaca_client']
                )
                
                # Mock successful order placement
                trading_engine.order_manager.place_order = Mock(return_value={
                    'id': 'test_order_123',
                    'status': 'filled',
                    'symbol': 'AAPL',
                    'qty': 10,
                    'side': 'buy'
                })
                
                # Process approved predictions (simulate)
                for symbol, prediction in predictions_dict.items():
                    if prediction['confidence'] > 0.5:  # Only trade high confidence
                        # Calculate position size
                        position_size = abs(prediction['position']) * 1000  # $1000 base
                        
                        # Place mock order
                        order_result = trading_engine.order_manager.place_order(
                            symbol=symbol,
                            qty=int(position_size / 150),  # Assume $150 per share
                            side='buy' if prediction['position'] > 0 else 'sell',
                            order_type='market'
                        )
                        
                        # Verify order was placed
                        assert order_result is not None
                        assert 'id' in order_result
                        assert order_result['symbol'] == symbol
            
            # Step 5: Performance Verification
            # Verify feature processing performance
            feature_stats = feature_processor.get_statistics()
            assert feature_stats.symbols_processed == len(symbols)
            assert feature_stats.features_calculated > 0
            
            # Verify risk management performance
            risk_metrics = risk_manager.get_risk_metrics()
            assert 'portfolio_risk' in risk_metrics or 'status' in risk_metrics
            
            # Verify system integration
            assert len(market_data_dict) == len(symbols)
            assert len(features_dict) == len(symbols)
            assert len(predictions_dict) == len(symbols)
    
    @pytest.mark.performance
    async def test_pipeline_performance_benchmark(self, integration_config):
        """Benchmark the complete pipeline performance."""
        
        # Use larger dataset for performance testing
        symbols = ["AAPL", "MSFT", "GOOGL", "TSLA", "NVDA", "AMZN", "META", "NFLX"]
        n_records = 500  # Larger dataset
        
        with patch('deep_momentum_trading.src.data.feature_engineering.ZMQPublisher'), \
             patch('deep_momentum_trading.src.data.feature_engineering.UnifiedMemoryManager'), \
             patch('deep_momentum_trading.config.settings.config_manager') as mock_config:
            
            mock_config.get.return_value = {}
            
            # Initialize feature processor
            from deep_momentum_trading.src.data.feature_engineering import FeatureEngineeringProcess
            feature_processor = FeatureEngineeringProcess(config=integration_config.feature_config)
            
            # Benchmark feature calculation
            start_time = time.perf_counter()
            
            for symbol in symbols:
                market_data = get_sample_market_data(symbol=symbol, n_records=n_records)
                features = feature_processor.calculate_features(market_data)
                
                # Verify output quality
                assert len(features) == len(market_data)
                assert len(features.columns) >= 15
            
            end_time = time.perf_counter()
            total_time = end_time - start_time
            
            # Performance assertions
            avg_time_per_symbol = total_time / len(symbols)
            features_per_second = (len(symbols) * n_records) / total_time
            
            # Should process efficiently
            assert avg_time_per_symbol < 2.0  # Less than 2 seconds per symbol
            assert features_per_second > 1000   # More than 1000 features per second
            
            # Verify statistics
            stats = feature_processor.get_statistics()
            assert stats.symbols_processed == len(symbols)
            assert stats.processing_time_seconds > 0
            assert stats.features_per_second > 0
    
    @pytest.mark.stress
    async def test_pipeline_stress_test(self, integration_config):
        """Stress test the pipeline with high load."""
        
        # High load scenario
        symbols = [f"TEST{i:03d}" for i in range(50)]  # 50 symbols
        n_records = 1000  # Large dataset per symbol
        
        with patch('deep_momentum_trading.src.data.feature_engineering.ZMQPublisher'), \
             patch('deep_momentum_trading.src.data.feature_engineering.UnifiedMemoryManager'), \
             patch('deep_momentum_trading.config.settings.config_manager') as mock_config:
            
            mock_config.get.return_value = {}
            
            # Use stress test configuration
            stress_config = TestScenarios.get_stress_test_config()
            
            from deep_momentum_trading.src.data.feature_engineering import FeatureEngineeringProcess
            feature_processor = FeatureEngineeringProcess(config=stress_config.feature_config)
            
            # Process in batches to avoid memory issues
            batch_size = 10
            total_processed = 0
            
            for i in range(0, len(symbols), batch_size):
                batch_symbols = symbols[i:i + batch_size]
                
                for symbol in batch_symbols:
                    market_data = get_sample_market_data(symbol=symbol, n_records=n_records)
                    features = feature_processor.calculate_features(market_data)
                    
                    # Basic validation
                    assert len(features) == len(market_data)
                    total_processed += 1
                
                # Clear cache periodically to manage memory
                if i % (batch_size * 2) == 0:
                    feature_processor.clear_cache()
            
            # Verify all symbols were processed
            assert total_processed == len(symbols)
            
            # Check final statistics
            stats = feature_processor.get_statistics()
            assert stats.symbols_processed == len(symbols)
    
    def test_configuration_validation(self):
        """Test configuration validation across components."""
        
        # Test different configuration scenarios
        configs = [
            TestScenarios.get_minimal_config(),
            TestScenarios.get_performance_config(),
            TestScenarios.get_integration_config(),
            TestEnvironments.get_ci_config(),
            TestEnvironments.get_local_dev_config()
        ]
        
        for config in configs:
            # Validate configuration consistency
            from deep_momentum_trading.tests.fixtures.test_configs import validate_test_config
            issues = validate_test_config(config)
            
            # Should have no validation issues
            assert len(issues) == 0, f"Configuration issues: {issues}"
            
            # Test component initialization with config
            with patch('deep_momentum_trading.src.data.feature_engineering.ZMQPublisher'), \
                 patch('deep_momentum_trading.src.data.feature_engineering.UnifiedMemoryManager'), \
                 patch('deep_momentum_trading.config.settings.config_manager') as mock_config:
                
                mock_config.get.return_value = {}
                
                # Should initialize without errors
                from deep_momentum_trading.src.data.feature_engineering import FeatureEngineeringProcess
                feature_processor = FeatureEngineeringProcess(config=config.feature_config)
                
                assert feature_processor.config == config.feature_config
    
    @pytest.mark.arm64
    async def test_arm64_optimization_integration(self, integration_config):
        """Test ARM64 optimizations across the pipeline."""
        
        with patch('deep_momentum_trading.src.data.feature_engineering.IS_ARM64', True), \
             patch('deep_momentum_trading.src.risk.risk_manager.IS_ARM64', True), \
             patch('platform.machine', return_value='arm64'), \
             patch('deep_momentum_trading.src.data.feature_engineering.ZMQPublisher'), \
             patch('deep_momentum_trading.src.data.feature_engineering.UnifiedMemoryManager'), \
             patch('deep_momentum_trading.config.settings.config_manager') as mock_config:
            
            mock_config.get.return_value = {}
            
            # Enable ARM64 optimizations
            integration_config.feature_config.enable_arm64_optimizations = True
            integration_config.risk_config.enable_arm64_optimizations = True
            
            # Test feature engineering with ARM64
            from deep_momentum_trading.src.data.feature_engineering import FeatureEngineeringProcess
            feature_processor = FeatureEngineeringProcess(config=integration_config.feature_config)
            
            # Verify ARM64 detection
            assert feature_processor.is_arm64 is True
            
            # Process data with ARM64 optimizations
            symbols = ["AAPL", "MSFT", "GOOGL"]
            
            for symbol in symbols:
                market_data = get_sample_market_data(symbol=symbol, n_records=200)
                features = feature_processor.calculate_features(market_data)
                
                # Verify ARM64 optimizations were used
                assert feature_processor.stats.arm64_optimizations_used > 0
                
                # Verify output quality
                assert len(features) == len(market_data)
                assert len(features.columns) >= 15
    
    def test_error_handling_and_recovery(self, integration_config):
        """Test error handling and recovery mechanisms."""
        
        with patch('deep_momentum_trading.src.data.feature_engineering.ZMQPublisher'), \
             patch('deep_momentum_trading.src.data.feature_engineering.UnifiedMemoryManager'), \
             patch('deep_momentum_trading.config.settings.config_manager') as mock_config:
            
            mock_config.get.return_value = {}
            
            from deep_momentum_trading.src.data.feature_engineering import FeatureEngineeringProcess
            feature_processor = FeatureEngineeringProcess(config=integration_config.feature_config)
            
            # Test with invalid data
            invalid_data = pd.DataFrame()  # Empty DataFrame
            features = feature_processor.calculate_features(invalid_data)
            
            # Should handle gracefully
            assert isinstance(features, pd.DataFrame)
            assert len(features) == 0
            
            # Test with corrupted data
            corrupted_data = pd.DataFrame({
                'open': [np.nan] * 100,
                'high': [np.inf] * 100,
                'low': [-np.inf] * 100,
                'close': [None] * 100,
                'volume': [0] * 100
            })
            
            # Should handle corrupted data
            features = feature_processor.calculate_features(corrupted_data)
            assert isinstance(features, pd.DataFrame)
            
            # Test memory management under stress
            large_data = get_sample_market_data(n_records=10000)  # Large dataset
            features = feature_processor.calculate_features(large_data)
            
            # Should complete without memory errors
            assert len(features) == len(large_data)
    
    def test_health_monitoring_integration(self, integration_config):
        """Test health monitoring across components."""
        
        with patch('deep_momentum_trading.src.data.feature_engineering.ZMQPublisher'), \
             patch('deep_momentum_trading.src.data.feature_engineering.UnifiedMemoryManager') as mock_memory, \
             patch('deep_momentum_trading.config.settings.config_manager') as mock_config:
            
            mock_config.get.return_value = {}
            
            # Mock memory manager
            mock_memory_instance = Mock()
            mock_memory_instance.current_memory_gb = 2.5
            mock_memory.return_value = mock_memory_instance
            
            from deep_momentum_trading.src.data.feature_engineering import FeatureEngineeringProcess
            feature_processor = FeatureEngineeringProcess(config=integration_config.feature_config)
            
            # Test health check
            health = feature_processor.health_check()
            
            # Verify health check structure
            assert 'status' in health
            assert 'timestamp' in health
            assert 'components' in health
            assert 'statistics' in health
            
            # Verify component health
            assert 'zmq_publisher' in health['components']
            assert 'feature_cache' in health['components']
            
            # Verify statistics
            assert 'features_calculated' in health['statistics']
            assert 'error_rate' in health['statistics']
            
            # Process some data to update health metrics
            market_data = get_sample_market_data(n_records=100)
            features = feature_processor.calculate_features(market_data)
            
            # Check updated health
            updated_health = feature_processor.health_check()
            assert updated_health['statistics']['features_calculated'] > 0


@pytest.mark.integration
class TestDataFlowIntegration:
    """Test data flow between components."""
    
    def test_feature_to_prediction_flow(self):
        """Test data flow from features to predictions."""
        
        # Generate sample features
        features_data = get_sample_features_data(n_records=100, n_features=50)
        
        # Convert to model input format
        torch_data = get_sample_torch_data(batch_size=1, sequence_length=60, n_features=50)
        
        # Verify data compatibility
        assert torch_data['input_features'].shape[2] == 50  # Feature dimension
        assert torch_data['input_features'].shape[1] == 60  # Sequence length
        
        # Mock model prediction
        mock_prediction = {
            'position': float(torch_data['target_positions'][0].item()),
            'confidence': float(torch_data['target_confidence'][0].item()),
            'expected_return': float(torch_data['target_returns'][0].item())
        }
        
        # Verify prediction format
        assert -1 <= mock_prediction['position'] <= 1
        assert 0 <= mock_prediction['confidence'] <= 1
        assert isinstance(mock_prediction['expected_return'], float)
    
    def test_prediction_to_risk_flow(self):
        """Test data flow from predictions to risk management."""
        
        # Generate sample predictions
        predictions = get_sample_predictions(n_predictions=10)
        
        # Convert to risk manager format
        predictions_dict = {pred['symbol']: pred for pred in predictions}
        
        # Verify prediction format for risk manager
        for symbol, prediction in predictions_dict.items():
            assert 'confidence' in prediction
            assert 'position' in prediction
            assert 'expected_return' in prediction
            assert isinstance(prediction['confidence'], (int, float))
            assert isinstance(prediction['position'], (int, float))
            assert isinstance(prediction['expected_return'], (int, float))
    
    def test_risk_to_trading_flow(self):
        """Test data flow from risk management to trading."""
        
        # Generate sample portfolio
        portfolio = get_sample_portfolio(total_value=100000.0)
        
        # Verify portfolio format for trading engine
        for symbol, position in portfolio.items():
            assert 'quantity' in position
            assert 'market_value' in position
            assert 'side' in position
            assert isinstance(position['quantity'], (int, float))
            assert isinstance(position['market_value'], (int, float))
            assert position['side'] in ['long', 'short']


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
