import pytest
import torch
import numpy as np
from deep_momentum_trading.src.storage.memory_storage import UnifiedMemoryManager, MemoryBlock

@pytest.fixture
def memory_manager_small():
    """Provides a UnifiedMemoryManager with a small capacity for testing eviction."""
    return UnifiedMemoryManager(max_memory_gb=0.001) # 1 MB capacity

@pytest.fixture
def memory_manager_large():
    """Provides a UnifiedMemoryManager with a larger capacity."""
    return UnifiedMemoryManager(max_memory_gb=0.1) # 100 MB capacity

def test_memory_manager_init(memory_manager_small):
    """Test initialization of UnifiedMemoryManager."""
    assert memory_manager_small.max_memory_bytes == 1 * 1024 * 1024 # 1 MB
    assert memory_manager_small.current_usage_bytes == 0
    assert not memory_manager_small.memory_blocks

def test_calculate_size():
    """Test _calculate_size method for different data types."""
    manager = UnifiedMemoryManager(max_memory_gb=0.001)
    
    # NumPy array
    np_array = np.zeros((10, 10), dtype=np.float32)
    assert manager._calculate_size(np_array) == np_array.nbytes

    # PyTorch Tensor
    torch_tensor = torch.zeros(10, 10, dtype=torch.float32)
    assert manager._calculate_size(torch_tensor) == torch_tensor.numel() * torch_tensor.element_size()

    # Dictionary
    test_dict = {"key": "value", "number": 123}
    # Estimate size for dicts is based on string representation, so it's approximate
    assert manager._calculate_size(test_dict) > 0

def test_store_market_data_success(memory_manager_large):
    """Test storing market data successfully."""
    data = {"symbol": "AAPL", "price": 150.0}
    stored = memory_manager_large.store_market_data("AAPL", data)
    assert stored
    assert "market_data_AAPL" in memory_manager_large.memory_blocks
    assert memory_manager_large.current_usage_bytes > 0

def test_get_market_data(memory_manager_large):
    """Test retrieving market data."""
    data = {"symbol": "GOOG", "price": 2000.0}
    memory_manager_large.store_market_data("GOOG", data)
    retrieved_data = memory_manager_large.get_market_data("GOOG")
    assert retrieved_data == data
    assert memory_manager_large.memory_blocks["market_data_GOOG"].access_count == 2 # Initial store + get

def test_store_features_success(memory_manager_large):
    """Test storing features successfully."""
    features = np.random.rand(10, 5).astype(np.float32)
    feature_names = [f"feat_{i}" for i in range(5)]
    stored = memory_manager_large.store_features("MSFT", features, feature_names)
    assert stored
    assert "features_MSFT" in memory_manager_large.memory_blocks
    assert memory_manager_large.current_usage_bytes > 0

def test_get_features(memory_manager_large):
    """Test retrieving features."""
    features = np.random.rand(5, 3).astype(np.float32)
    feature_names = ["f1", "f2", "f3"]
    memory_manager_large.store_features("AMZN", features, feature_names)
    retrieved_features, retrieved_names = memory_manager_large.get_features("AMZN")
    assert np.array_equal(retrieved_features, features)
    assert retrieved_names == feature_names

def test_store_model_state_success(memory_manager_large):
    """Test storing model state successfully."""
    model_state = {"layer1.weight": torch.randn(10, 5), "layer1.bias": torch.randn(10)}
    stored = memory_manager_large.store_model_state("MyModel", model_state)
    assert stored
    assert "model_state_MyModel" in memory_manager_large.memory_blocks
    assert memory_manager_large.current_usage_bytes > 0

def test_get_model_state(memory_manager_large):
    """Test retrieving model state."""
    model_state = {"param_A": torch.randn(5, 5)}
    memory_manager_large.store_model_state("AnotherModel", model_state)
    retrieved_state = memory_manager_large.get_model_state("AnotherModel")
    assert torch.equal(retrieved_state["param_A"], model_state["param_A"])

def test_eviction_lru(memory_manager_small):
    """Test LRU eviction mechanism."""
    # Store data that fills up the cache
    data1 = {"sym": "A", "val": "a" * 500} # Approx 500 bytes
    data2 = {"sym": "B", "val": "b" * 500}
    data3 = {"sym": "C", "val": "c" * 500}
    
    # Store A, B, C
    memory_manager_small.store_market_data("A", data1)
    memory_manager_small.store_market_data("B", data2)
    memory_manager_small.store_market_data("C", data3)
    
    # Access A to make it Most Recently Used
    memory_manager_small.get_market_data("A")
    
    # Store D, which should trigger eviction of B (LRU after A was accessed)
    data4 = {"sym": "D", "val": "d" * 500}
    memory_manager_small.store_market_data("D", data4)
    
    # B should be evicted
    assert memory_manager_small.get_market_data("B") is None
    assert memory_manager_small.get_market_data("A") is not None
    assert memory_manager_small.get_market_data("C") is not None
    assert memory_manager_small.get_market_data("D") is not None

def test_get_memory_stats(memory_manager_small):
    """Test memory statistics reporting."""
    stats = memory_manager_small.get_memory_stats()
    assert stats['total_capacity_gb'] == 0.001
    assert stats['current_usage_gb'] >= 0
    assert stats['utilization_pct'] >= 0
    assert stats['num_market_data_blocks'] >= 0
    assert stats['num_feature_blocks'] == 0
    assert stats['num_model_states'] == 0
    assert stats['num_total_blocks'] >= 0

def test_store_large_data_eviction(memory_manager_small):
    """Test storing data larger than capacity, forcing multiple evictions."""
    # Capacity is 1MB
    large_data_item = {"data": "x" * 200 * 1024} # 200KB per item
    
    # Store 5 items (1MB total)
    for i in range(5):
        memory_manager_small.store_market_data(f"item_{i}", large_data_item)
    
    # Access item_0 to make it MRU
    memory_manager_small.get_market_data("item_0")

    # Store a new item, should evict item_1
    memory_manager_small.store_market_data("item_5", large_data_item)
    
    assert memory_manager_small.get_market_data("item_1") is None
    assert memory_manager_small.get_market_data("item_0") is not None
    assert memory_manager_small.get_market_data("item_5") is not None
