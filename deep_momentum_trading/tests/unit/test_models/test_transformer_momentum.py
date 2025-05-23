import pytest
import torch
import torch.nn as nn
from deep_momentum_trading.src.models.transformer_momentum import TransformerMomentumNetwork, HierarchicalTransformer

@pytest.fixture
def transformer_model_config():
    """Provides a sample configuration for TransformerMomentumNetwork."""
    return {
        'input_size': 50,
        'd_model': 64,
        'num_heads': 4,
        'num_layers': 2,
        'd_ff': 128,
        'max_seq_len': 100,
        'dropout': 0.1
    }

@pytest.fixture
def transformer_model(transformer_model_config):
    """Provides an initialized TransformerMomentumNetwork model."""
    return TransformerMomentumNetwork(**transformer_model_config)

def test_transformer_model_initialization(transformer_model_config):
    """Test that the TransformerMomentumNetwork initializes without errors."""
    model = TransformerMomentumNetwork(**transformer_model_config)
    assert isinstance(model, nn.Module)
    assert len(model.encoder_layers) == transformer_model_config['num_layers']
    assert model.position_head is not None
    assert model.confidence_head is not None
    assert model.volatility_head is not None
    assert model.regime_head is not None

def test_transformer_model_forward_pass(transformer_model, transformer_model_config):
    """Test the forward pass with dummy input."""
    batch_size = 4
    seq_len = 10
    dummy_input = torch.randn(batch_size, seq_len, transformer_model_config['input_size'])
    seq_lengths = torch.randint(low=5, high=seq_len + 1, size=(batch_size,))
    
    outputs = transformer_model(dummy_input, seq_lengths)
    
    assert 'positions' in outputs
    assert 'confidence' in outputs
    assert 'volatility' in outputs
    assert 'regime' in outputs
    assert 'attention_weights' in outputs
    assert 'final_hidden' in outputs

    assert outputs['positions'].shape == (batch_size, 1)
    assert outputs['confidence'].shape == (batch_size, 1)
    assert outputs['volatility'].shape == (batch_size, 1)
    assert outputs['regime'].shape == (batch_size, 4) # 4 regimes
    assert len(outputs['attention_weights']) == transformer_model_config['num_layers']
    assert outputs['final_hidden'].shape == (batch_size, transformer_model_config['d_model'])

def test_transformer_model_predict_step(transformer_model, transformer_model_config):
    """Test the predict_step method."""
    batch_size = 4
    single_timestep_input = torch.randn(batch_size, 1, transformer_model_config['input_size'])
    seq_lengths = torch.ones(batch_size, dtype=torch.long) # Length 1 for single step
    
    predictions = transformer_model.predict_step(single_timestep_input, seq_lengths)
    
    assert 'positions' in predictions
    assert 'confidence' in predictions
    assert 'volatility' in predictions
    assert 'regime' in predictions
    assert predictions['positions'].shape == (batch_size, 1)

def test_transformer_model_get_model_size(transformer_model):
    """Test parameter counting."""
    num_params = transformer_model.get_model_size()
    assert num_params > 0

def test_transformer_model_get_memory_usage(transformer_model):
    """Test memory usage estimation."""
    mem_usage = transformer_model.get_memory_usage()
    assert mem_usage > 0

def test_transformer_model_feature_importance(transformer_model, transformer_model_config):
    """Test feature importance calculation."""
    batch_size = 2
    seq_len = 3
    dummy_input = torch.randn(batch_size, seq_len, transformer_model_config['input_size'], requires_grad=True)
    seq_lengths = torch.tensor([seq_len, seq_len], dtype=torch.long)
    
    importance = transformer_model.get_feature_importance(dummy_input, seq_lengths)
    
    assert importance.shape == (transformer_model_config['input_size'],)
    assert torch.sum(importance) >= 0

# HierarchicalTransformer Tests
@pytest.fixture
def hierarchical_transformer_config():
    """Provides a sample configuration for HierarchicalTransformer."""
    return {
        'timeframe_configs': {
            'daily': {
                'input_size': 50, 'd_model': 32, 'num_heads': 2, 'num_layers': 1, 'd_ff': 64, 'max_seq_len': 20
            },
            'hourly': {
                'input_size': 50, 'd_model': 32, 'num_heads': 2, 'num_layers': 1, 'd_ff': 64, 'max_seq_len': 40
            }
        },
        'fusion_d_model': 64,
        'fusion_num_heads': 4,
        'fusion_num_layers': 1
    }

@pytest.fixture
def hierarchical_transformer(hierarchical_transformer_config):
    """Provides an initialized HierarchicalTransformer model."""
    return HierarchicalTransformer(**hierarchical_transformer_config)

def test_hierarchical_transformer_initialization(hierarchical_transformer_config):
    """Test HierarchicalTransformer initialization."""
    model = HierarchicalTransformer(**hierarchical_transformer_config)
    assert isinstance(model, nn.Module)
    assert 'daily' in model.timeframe_transformers
    assert 'hourly' in model.timeframe_transformers
    assert model.fusion_transformer is not None
    assert model.fusion_transformer.input_size == len(hierarchical_transformer_config['timeframe_configs']) * 3

def test_hierarchical_transformer_forward_pass(hierarchical_transformer, hierarchical_transformer_config):
    """Test forward pass of HierarchicalTransformer."""
    batch_size = 2
    timeframe_data = {
        'daily': torch.randn(batch_size, 20, 50),
        'hourly': torch.randn(batch_size, 40, 50)
    }
    timeframe_seq_lengths = {
        'daily': torch.randint(low=10, high=20 + 1, size=(batch_size,)),
        'hourly': torch.randint(low=20, high=40 + 1, size=(batch_size,))
    }

    outputs = hierarchical_transformer(timeframe_data, timeframe_seq_lengths)
    
    assert 'fused_predictions' in outputs
    assert 'timeframe_outputs' in outputs
    
    fused_preds = outputs['fused_predictions']
    assert fused_preds['positions'].shape == (batch_size, 1)
    assert fused_preds['confidence'].shape == (batch_size, 1)
    assert fused_preds['volatility'].shape == (batch_size, 1)
    assert fused_preds['regime'].shape == (batch_size, 4)

    assert 'daily' in outputs['timeframe_outputs']
    assert 'hourly' in outputs['timeframe_outputs']
