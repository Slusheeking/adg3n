import pytest
import torch
import torch.nn as nn
from deep_momentum_trading.src.models.deep_momentum_lstm import DeepMomentumLSTM

@pytest.fixture
def lstm_model_config():
    """Provides a sample configuration for DeepMomentumLSTM."""
    return {
        'input_size': 10,
        'hidden_size': 20,
        'num_layers': 2,
        'dropout': 0.1,
        'device': 'cpu' # Use CPU for testing unless specific GPU tests are needed
    }

@pytest.fixture
def lstm_model(lstm_model_config):
    """Provides an initialized DeepMomentumLSTM model."""
    return DeepMomentumLSTM(**lstm_model_config)

def test_model_initialization(lstm_model_config):
    """Test that the model initializes without errors and has correct layer counts."""
    model = DeepMomentumLSTM(**lstm_model_config)
    assert isinstance(model, nn.Module)
    assert len(model.lstm_layers) == lstm_model_config['num_layers']
    assert len(model.layer_norms) == lstm_model_config['num_layers']
    assert model.position_head is not None
    assert model.confidence_head is not None
    assert model.volatility_head is not None

def test_model_forward_pass(lstm_model, lstm_model_config):
    """Test the forward pass with dummy input."""
    batch_size = 4
    seq_len = 5
    dummy_input = torch.randn(batch_size, seq_len, lstm_model_config['input_size'])
    
    positions, confidence, volatility, attention_weights, hidden_states = lstm_model(dummy_input)
    
    assert positions.shape == (batch_size, 1)
    assert confidence.shape == (batch_size, 1)
    assert volatility.shape == (batch_size, 1)
    assert attention_weights.shape == (batch_size, seq_len, seq_len) # Assuming self-attention output
    assert len(hidden_states) == lstm_model_config['num_layers']
    assert hidden_states[0][0].shape == (1, batch_size, lstm_model_config['hidden_size']) # h_n shape

def test_model_forward_pass_with_hidden_states(lstm_model, lstm_model_config):
    """Test forward pass with provided hidden states."""
    batch_size = 4
    seq_len = 1
    dummy_input = torch.randn(batch_size, seq_len, lstm_model_config['input_size'])
    
    # Initial forward pass to get hidden states
    _, _, _, _, initial_hidden_states = lstm_model(dummy_input)
    
    # Subsequent forward pass with hidden states
    positions, confidence, volatility, attention_weights, new_hidden_states = lstm_model(dummy_input, initial_hidden_states)
    
    assert positions.shape == (batch_size, 1)
    assert confidence.shape == (batch_size, 1)
    assert volatility.shape == (batch_size, 1)
    assert len(new_hidden_states) == lstm_model_config['num_layers']

def test_predict_step(lstm_model, lstm_model_config):
    """Test the predict_step method for single-timestep inference."""
    batch_size = 4
    single_timestep_input = torch.randn(batch_size, 1, lstm_model_config['input_size'])
    
    predictions = lstm_model.predict_step(single_timestep_input)
    
    assert 'positions' in predictions
    assert 'confidence' in predictions
    assert 'volatility' in predictions
    assert 'hidden_states' in predictions
    
    assert predictions['positions'].shape == (batch_size, 1)
    assert predictions['confidence'].shape == (batch_size, 1)
    assert predictions['volatility'].shape == (batch_size, 1)
    assert len(predictions['hidden_states']) == lstm_model_config['num_layers']

def test_get_model_size(lstm_model):
    """Test parameter counting."""
    num_params = lstm_model.get_model_size()
    assert num_params > 0 # Model should have parameters

def test_get_memory_usage(lstm_model):
    """Test memory usage estimation."""
    mem_usage = lstm_model.get_memory_usage()
    assert mem_usage > 0 # Model should consume some memory

def test_feature_importance(lstm_model, lstm_model_config):
    """Test feature importance calculation."""
    batch_size = 2
    seq_len = 3
    dummy_input = torch.randn(batch_size, seq_len, lstm_model_config['input_size'], requires_grad=True)
    
    importance = lstm_model.get_feature_importance(dummy_input)
    
    assert importance.shape == (lstm_model_config['input_size'],)
    assert torch.sum(importance) >= 0 # Importance scores should be non-negative
