import pytest
import torch
import torch.nn as nn
import numpy as np
from unittest.mock import MagicMock, patch
from deep_momentum_trading.src.models.ensemble_system import EnsembleMomentumSystem, ModelPerformanceTracker, MarketRegimeDetector, MetaLearningNetwork
from deep_momentum_trading.src.models.deep_momentum_lstm import DeepMomentumLSTM
from deep_momentum_trading.src.models.transformer_momentum import TransformerMomentumNetwork

# Fixtures for common components
@pytest.fixture
def mock_lstm_model():
    """Mock DeepMomentumLSTM model."""
    mock_model = MagicMock(spec=DeepMomentumLSTM)
    mock_model.return_value.forward.return_value = (
        torch.randn(1, 1), # positions
        torch.rand(1, 1),  # confidence
        torch.rand(1, 1),  # volatility
        torch.rand(1, 5, 5), # attention_weights
        (torch.randn(1,1,20), torch.randn(1,1,20)) # hidden_states
    )
    mock_model.return_value.eval.return_value = mock_model.return_value # Allow chaining .eval()
    mock_model.return_value.get_feature_importance.return_value = torch.rand(50)
    return mock_model

@pytest.fixture
def mock_transformer_model():
    """Mock TransformerMomentumNetwork model."""
    mock_model = MagicMock(spec=TransformerMomentumNetwork)
    mock_model.return_value.forward.return_value = {
        'positions': torch.randn(1, 1),
        'confidence': torch.rand(1, 1),
        'volatility': torch.rand(1, 1),
        'regime': torch.rand(1, 4),
        'attention_weights': [torch.rand(1, 8, 5, 5)],
        'final_hidden': torch.rand(1, 50)
    }
    mock_model.return_value.eval.return_value = mock_model.return_value # Allow chaining .eval()
    mock_model.return_value.get_feature_importance.return_value = torch.rand(50)
    return mock_model

@pytest.fixture
def ensemble_model_configs(mock_lstm_model, mock_transformer_model):
    """Provides sample model configurations for the ensemble."""
    return {
        'lstm_model_1': {
            'type': 'lstm',
            'input_size': 50, 'hidden_size': 20, 'num_layers': 1, 'dropout': 0.0
        },
        'transformer_model_1': {
            'type': 'transformer',
            'input_size': 50, 'd_model': 20, 'num_heads': 2, 'num_layers': 1, 'dropout': 0.0
        }
    }

@pytest.fixture
def ensemble_instance(ensemble_model_configs):
    """Provides an initialized EnsembleMomentumSystem."""
    # Patch the actual model classes with mocks during instantiation
    with patch('deep_momentum_trading.src.models.deep_momentum_lstm.DeepMomentumLSTM', new=MagicMock(return_value=MagicMock(spec=DeepMomentumLSTM))) as MockLSTM, \
         patch('deep_momentum_trading.src.models.transformer_momentum.TransformerMomentumNetwork', new=MagicMock(return_value=MagicMock(spec=TransformerMomentumNetwork))) as MockTransformer:
        
        # Configure the return values of the mocked models' forward passes
        MockLSTM.return_value.eval.return_value = MockLSTM.return_value
        MockLSTM.return_value.forward.return_value = (
            torch.randn(1, 1), torch.rand(1, 1), torch.rand(1, 1), torch.rand(1, 5, 5), (torch.randn(1,1,20), torch.randn(1,1,20))
        )
        MockLSTM.return_value.get_feature_importance.return_value = torch.rand(50)

        MockTransformer.return_value.eval.return_value = MockTransformer.return_value
        MockTransformer.return_value.forward.return_value = {
            'positions': torch.randn(1, 1), 'confidence': torch.rand(1, 1), 'volatility': torch.rand(1, 1),
            'regime': torch.rand(1, 4), 'attention_weights': [torch.rand(1, 8, 5, 5)], 'final_hidden': torch.rand(1, 50)
        }
        MockTransformer.return_value.get_feature_importance.return_value = torch.rand(50)

        ensemble = EnsembleMomentumSystem(
            model_configs=ensemble_model_configs,
            ensemble_method='adaptive_meta_learning',
            performance_tracking=True,
            market_feature_dim=50
        )
        # Replace actual models with the mocked instances
        ensemble.models['lstm_model_1'] = MockLSTM.return_value
        ensemble.models['transformer_model_1'] = MockTransformer.return_value
        return ensemble

# Tests for ModelPerformanceTracker
def test_performance_tracker_init():
    tracker = ModelPerformanceTracker(lookback_window=10, update_frequency=2)
    assert tracker.lookback_window == 10
    assert tracker.update_frequency == 2
    assert not tracker.model_metrics

def test_performance_tracker_update_performance():
    tracker = ModelPerformanceTracker(lookback_window=5, update_frequency=1)
    
    # Simulate some data
    predictions = torch.tensor([[0.5], [-0.5], [0.5], [-0.5], [0.5]])
    actuals = torch.tensor([[0.01], [-0.01], [0.01], [-0.01], [0.01]])
    returns = torch.tensor([[0.01], [-0.01], [0.01], [-0.01], [0.01]])

    for i in range(5):
        tracker.update_performance("test_model", predictions[i:i+1], actuals[i:i+1], returns[i:i+1])
    
    metrics = tracker.model_metrics["test_model"]
    assert len(metrics.recent_performance) == 5
    assert metrics.prediction_accuracy > 0 # Should be some accuracy
    assert metrics.last_updated > 0

def test_performance_tracker_get_model_weights():
    tracker = ModelPerformanceTracker(lookback_window=5, update_frequency=1)
    
    # Add some performance data for two models
    for i in range(30): # Enough data for stats
        tracker.update_performance("model_A", torch.tensor([[0.5]]), torch.tensor([[0.01]]), torch.tensor([[0.01]]))
        tracker.update_performance("model_B", torch.tensor([[-0.5]]), torch.tensor([[0.01]]), torch.tensor([[0.01]])) # Worse performance
    
    weights = tracker.get_model_weights(["model_A", "model_B"])
    assert weights.shape == (2,)
    assert torch.sum(weights).item() == pytest.approx(1.0)
    assert weights[0].item() > weights[1].item() # Model A should have higher weight

# Tests for MarketRegimeDetector
def test_market_regime_detector_forward():
    detector = MarketRegimeDetector(input_size=50)
    dummy_features = torch.randn(4, 50) # batch_size, feature_dim
    output = detector(dummy_features)
    assert output.shape == (4, 4) # batch_size, num_regimes
    assert torch.allclose(output.sum(dim=-1), torch.tensor(1.0)) # Softmax output

def test_market_regime_detector_with_seq_len_input():
    detector = MarketRegimeDetector(input_size=50)
    dummy_features = torch.randn(4, 10, 50) # batch_size, seq_len, feature_dim
    output = detector(dummy_features)
    assert output.shape == (4, 4) # Should take last timestep

# Tests for MetaLearningNetwork
def test_meta_learning_network_forward():
    meta_net = MetaLearningNetwork(num_models=2, market_feature_dim=50, performance_feature_dim=10)
    dummy_market_features = torch.randn(4, 50)
    dummy_performance_features = torch.randn(4, 10)
    dummy_model_predictions = torch.randn(4, 2, 3) # batch, num_models, (pos, conf, vol)
    
    output = meta_net(dummy_market_features, dummy_performance_features, dummy_model_predictions)
    assert 'model_weights' in output
    assert 'uncertainty' in output
    assert output['model_weights'].shape == (4, 2) # batch, num_models
    assert output['uncertainty'].shape == (4, 1) # batch, 1
    assert torch.allclose(output['model_weights'].sum(dim=-1), torch.tensor(1.0))

# Tests for EnsembleMomentumSystem
def test_ensemble_momentum_system_init(ensemble_instance):
    """Test EnsembleMomentumSystem initialization."""
    assert len(ensemble_instance.models) == 2
    assert 'lstm_model_1' in ensemble_instance.models
    assert 'transformer_model_1' in ensemble_instance.models
    assert ensemble_instance.performance_tracker is not None
    assert ensemble_instance.meta_learner is not None

def test_ensemble_momentum_system_forward(ensemble_instance):
    """Test forward pass of EnsembleMomentumSystem."""
    batch_size = 2
    seq_len = 10
    input_size = 50
    dummy_features = torch.randn(batch_size, seq_len, input_size)
    dummy_actual_returns = torch.randn(batch_size, 1) * 0.01

    output = ensemble_instance(dummy_features, actual_returns=dummy_actual_returns, return_individual_outputs=True)
    
    assert 'positions' in output
    assert 'confidence' in output
    assert 'volatility' in output
    assert 'weights' in output
    assert 'market_regime_probs' in output
    assert 'individual_outputs' in output

    assert output['positions'].shape == (batch_size, 1)
    assert output['confidence'].shape == (batch_size, 1)
    assert output['volatility'].shape == (batch_size, 1)
    assert output['weights'].shape == (batch_size, 2, 1) # batch, num_models, 1
    assert output['market_regime_probs'].shape == (batch_size, 4)

    assert 'lstm_model_1' in output['individual_outputs']
    assert 'transformer_model_1' in output['individual_outputs']

def test_ensemble_momentum_system_performance_update(ensemble_instance):
    """Test that performance tracker is updated during forward pass."""
    batch_size = 2
    seq_len = 10
    input_size = 50
    dummy_features = torch.randn(batch_size, seq_len, input_size)
    dummy_actual_returns = torch.randn(batch_size, 1) * 0.01

    initial_lstm_sharpe = ensemble_instance.performance_tracker.model_metrics['lstm_model_1'].sharpe_ratio
    initial_ensemble_sharpe = ensemble_instance.performance_tracker.model_metrics['Ensemble'].sharpe_ratio

    # Run forward pass multiple times to trigger performance updates
    for _ in range(ensemble_instance.performance_tracker.update_frequency + 5):
        ensemble_instance(dummy_features, actual_returns=dummy_actual_returns)
    
    updated_lstm_sharpe = ensemble_instance.performance_tracker.model_metrics['lstm_model_1'].sharpe_ratio
    updated_ensemble_sharpe = ensemble_instance.performance_tracker.model_metrics['Ensemble'].sharpe_ratio

    assert updated_lstm_sharpe != initial_lstm_sharpe
    assert updated_ensemble_sharpe != initial_ensemble_sharpe

def test_ensemble_momentum_system_get_explanations(ensemble_instance):
    """Test explanation generation."""
    single_sample_features = torch.randn(1, 10, 50) # batch_size=1 for explanations
    explanations = ensemble_instance.get_model_explanations(single_sample_features)
    
    assert 'model_contributions' in explanations
    assert 'market_regime' in explanations
    assert 'feature_importance' in explanations # Should be present if LSTM mock provides it

    assert 'lstm_model_1' in explanations['model_contributions']
    assert 'transformer_model_1' in explanations['model_contributions']
    assert isinstance(explanations['market_regime'], dict)
    assert isinstance(explanations['feature_importance'], list) # Converted to list for JSON serialization

def test_ensemble_momentum_system_save_load_state(ensemble_instance, tmp_path):
    """Test saving and loading ensemble state."""
    filepath = tmp_path / "ensemble_state.pth"
    ensemble_instance.save_ensemble_state(filepath)

    new_ensemble = EnsembleMomentumSystem(
        model_configs=ensemble_instance.ensemble_model_configs,
        ensemble_method='adaptive_meta_learning',
        performance_tracking=True,
        market_feature_dim=50
    )
    new_ensemble.load_ensemble_state(filepath)

    # Verify that model states are loaded
    assert torch.allclose(
        ensemble_instance.models['lstm_model_1'].state_dict()['lstm_layers.0.weight_ih_l0'],
        new_ensemble.models['lstm_model_1'].state_dict()['lstm_layers.0.weight_ih_l0']
    )
    # Verify performance tracker metrics are loaded
    assert new_ensemble.performance_tracker.model_metrics['lstm_model_1'].sharpe_ratio == \
           ensemble_instance.performance_tracker.model_metrics['lstm_model_1'].sharpe_ratio
