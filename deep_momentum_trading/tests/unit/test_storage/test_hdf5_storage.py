import pytest
import pandas as pd
import numpy as np
import h5py
import os
from pathlib import Path
from deep_momentum_trading.src.storage.hdf5_storage import HDF5TimeSeriesStorage

@pytest.fixture
def temp_hdf5_dir(tmp_path):
    """Provides a temporary directory for HDF5 files and cleans up after tests."""
    test_dir = tmp_path / "test_hdf5_data"
    test_dir.mkdir()
    yield str(test_dir)
    # Cleanup is handled by tmp_path fixture

@pytest.fixture
def hdf5_storage(temp_hdf5_dir):
    """Provides an HDF5TimeSeriesStorage instance."""
    return HDF5TimeSeriesStorage(base_path=temp_hdf5_dir)

@pytest.fixture
def sample_ohlcv_df():
    """Provides a sample OHLCV DataFrame."""
    dates = pd.date_range(start='2023-01-01', periods=10, freq='min')
    return pd.DataFrame({
        'open': np.random.rand(10) * 100,
        'high': np.random.rand(10) * 100 + 5,
        'low': np.random.rand(10) * 100 - 5,
        'close': np.random.rand(10) * 100,
        'volume': np.random.randint(100, 1000, 10)
    }, index=dates)

@pytest.fixture
def sample_features_df():
    """Provides a sample features DataFrame."""
    dates = pd.date_range(start='2023-01-01', periods=10, freq='min')
    return pd.DataFrame({
        'feat_A': np.random.rand(10),
        'feat_B': np.random.rand(10) * 10,
        'feat_C': np.random.rand(10) * 100
    }, index=dates)

def test_store_ohlcv_data(hdf5_storage, sample_ohlcv_df):
    """Test storing OHLCV data."""
    symbol = "TEST_OHLCV"
    hdf5_storage.store_ohlcv_data(symbol, sample_ohlcv_df)
    
    file_path = Path(hdf5_storage.base_path) / f"{symbol}.h5"
    assert file_path.exists()
    
    with h5py.File(file_path, 'r') as f:
        assert 'ohlcv' in f
        assert 'timestamp' in f['ohlcv']
        assert 'close' in f['ohlcv']
        assert f['ohlcv']['timestamp'].shape[0] == len(sample_ohlcv_df)
        assert np.allclose(f['ohlcv']['close'][:], sample_ohlcv_df['close'].values)

def test_load_ohlcv_data(hdf5_storage, sample_ohlcv_df):
    """Test loading OHLCV data."""
    symbol = "LOAD_OHLCV"
    hdf5_storage.store_ohlcv_data(symbol, sample_ohlcv_df)
    
    loaded_df = hdf5_storage.load_ohlcv_data(symbol)
    assert loaded_df is not None
    pd.testing.assert_frame_equal(loaded_df, sample_ohlcv_df)

def test_load_ohlcv_data_range(hdf5_storage, sample_ohlcv_df):
    """Test loading OHLCV data within a specific time range."""
    symbol = "LOAD_OHLCV_RANGE"
    hdf5_storage.store_ohlcv_data(symbol, sample_ohlcv_df)
    
    start_ts = sample_ohlcv_df.index[2]
    end_ts = sample_ohlcv_df.index[7]
    
    loaded_df = hdf5_storage.load_ohlcv_data(symbol, start_ts, end_ts)
    expected_df = sample_ohlcv_df.loc[start_ts:end_ts]
    
    assert loaded_df is not None
    pd.testing.assert_frame_equal(loaded_df, expected_df)

def test_store_features(hdf5_storage, sample_features_df):
    """Test storing features data."""
    symbol = "TEST_FEATURES"
    hdf5_storage.store_features(symbol, sample_features_df)
    
    file_path = Path(hdf5_storage.base_path) / f"{symbol}.h5"
    assert file_path.exists()
    
    with h5py.File(file_path, 'r') as f:
        assert 'features' in f
        assert 'data' in f['features']
        assert 'timestamp' in f['features']
        assert 'feature_names' in f['features'].attrs
        assert f['features']['data'].shape == sample_features_df.shape
        assert f['features'].attrs['feature_names'] == sample_features_df.columns.tolist()
        assert np.allclose(f['features']['data'][:], sample_features_df.values)

def test_load_features(hdf5_storage, sample_features_df):
    """Test loading features data."""
    symbol = "LOAD_FEATURES"
    hdf5_storage.store_features(symbol, sample_features_df)
    
    loaded_df = hdf5_storage.load_features(symbol)
    assert loaded_df is not None
    pd.testing.assert_frame_equal(loaded_df, sample_features_df)

def test_load_features_range(hdf5_storage, sample_features_df):
    """Test loading features data within a specific time range."""
    symbol = "LOAD_FEATURES_RANGE"
    hdf5_storage.store_features(symbol, sample_features_df)
    
    start_ts = sample_features_df.index[1]
    end_ts = sample_features_df.index[5]
    
    loaded_df = hdf5_storage.load_features(symbol, start_ts, end_ts)
    expected_df = sample_features_df.loc[start_ts:end_ts]
    
    assert loaded_df is not None
    pd.testing.assert_frame_equal(loaded_df, expected_df)

def test_append_ohlcv_data(hdf5_storage, sample_ohlcv_df):
    """Test appending OHLCV data to an existing dataset."""
    symbol = "APPEND_OHLCV"
    first_half = sample_ohlcv_df.iloc[:5]
    second_half = sample_ohlcv_df.iloc[5:]
    
    hdf5_storage.store_ohlcv_data(symbol, first_half)
    hdf5_storage.store_ohlcv_data(symbol, second_half) # Append
    
    loaded_df = hdf5_storage.load_ohlcv_data(symbol)
    pd.testing.assert_frame_equal(loaded_df, sample_ohlcv_df)

def test_append_features_data(hdf5_storage, sample_features_df):
    """Test appending features data to an existing dataset."""
    symbol = "APPEND_FEATURES"
    first_half = sample_features_df.iloc[:5]
    second_half = sample_features_df.iloc[5:]
    
    hdf5_storage.store_features(symbol, first_half)
    hdf5_storage.store_features(symbol, second_half) # Append
    
    loaded_df = hdf5_storage.load_features(symbol)
    pd.testing.assert_frame_equal(loaded_df, sample_features_df)

def test_close_all_files(hdf5_storage, sample_ohlcv_df):
    """Test closing all open file handles."""
    symbol1 = "CLOSE_TEST_1"
    symbol2 = "CLOSE_TEST_2"
    hdf5_storage.store_ohlcv_data(symbol1, sample_ohlcv_df)
    hdf5_storage.store_features(symbol2, sample_features_df)
    
    # Ensure files are open
    assert symbol1 in hdf5_storage.file_handles
    assert symbol2 in hdf5_storage.file_handles
    
    hdf5_storage.close_all_files()
    
    assert not hdf5_storage.file_handles # Should be empty
    # Attempting to access a closed file should raise an error or re-open
    # For this test, we just check the handle dictionary is empty.
