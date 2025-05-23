import pytest
import pandas as pd
import numpy as np
from deep_momentum_trading.src.data.data_preprocessing import DataPreprocessor

@pytest.fixture
def sample_dataframe():
    """Provides a sample DataFrame for testing."""
    data = {
        'timestamp': pd.to_datetime(['2023-01-01 09:30:00', '2023-01-01 09:31:00', '2023-01-01 09:32:00',
                                     '2023-01-01 09:33:00', '2023-01-01 09:34:00']),
        'open': [100.0, 101.0, 102.0, 103.0, 104.0],
        'high': [100.5, 101.5, 102.5, 103.5, 104.5],
        'low': [99.5, 100.5, 101.5, 102.5, 103.5],
        'close': [100.2, 101.2, 102.2, 103.2, 104.2],
        'volume': [1000, 1100, 1200, 1300, 1400]
    }
    return pd.DataFrame(data)

@pytest.fixture
def preprocessor():
    """Provides a DataPreprocessor instance with default settings."""
    return DataPreprocessor()

def test_validate_data_valid(sample_dataframe, preprocessor):
    """Test data validation with a valid DataFrame."""
    df, errors = preprocessor.validate_data(sample_dataframe.copy())
    assert not errors
    assert isinstance(df, pd.DataFrame)
    assert pd.api.types.is_datetime64_any_dtype(df['timestamp'])

def test_validate_data_missing_columns(sample_dataframe, preprocessor):
    """Test data validation with missing required columns."""
    df_missing = sample_dataframe.drop(columns=['volume'])
    df, errors = preprocessor.validate_data(df_missing.copy())
    assert "Missing required columns: volume" in errors

def test_validate_data_invalid_timestamp(sample_dataframe, preprocessor):
    """Test data validation with invalid timestamp format."""
    df_invalid_ts = sample_dataframe.copy()
    df_invalid_ts['timestamp'] = ['not-a-date'] * len(df_invalid_ts)
    df, errors = preprocessor.validate_data(df_invalid_ts.copy())
    assert "Timestamp column is not in a valid datetime format." in errors

def test_handle_missing_data_ffill(sample_dataframe, preprocessor):
    """Test handling missing data with ffill method."""
    df_missing = sample_dataframe.copy()
    df_missing.loc[2, 'close'] = np.nan
    df_missing.loc[3, 'volume'] = np.nan
    
    preprocessor_ffill = DataPreprocessor(fill_method='ffill')
    processed_df = preprocessor_ffill.handle_missing_data(df_missing)
    
    assert not processed_df.isnull().any().any()
    assert processed_df.loc[2, 'close'] == df_missing.loc[1, 'close']
    assert processed_df.loc[3, 'volume'] == df_missing.loc[2, 'volume'] # ffill from previous valid

def test_handle_missing_data_drop(sample_dataframe):
    """Test handling missing data with drop method."""
    df_missing = sample_dataframe.copy()
    df_missing.loc[2, 'close'] = np.nan
    
    preprocessor_drop = DataPreprocessor(fill_method='drop')
    processed_df = preprocessor_drop.handle_missing_data(df_missing)
    
    assert len(processed_df) == len(sample_dataframe) - 1
    assert 2 not in processed_df.index

def test_remove_outliers(sample_dataframe, preprocessor):
    """Test outlier removal using Z-score."""
    df_outlier = sample_dataframe.copy()
    df_outlier.loc[2, 'close'] = 1000.0 # Introduce an outlier
    df_outlier.loc[4, 'volume'] = 100000 # Another outlier
    
    processed_df = preprocessor.remove_outliers(df_outlier.copy(), columns=['close', 'volume'])
    
    # After outlier removal, the values should be NaN and then filled by handle_missing_data (ffill by default)
    # So, the outlier value itself should not be present.
    assert processed_df.loc[2, 'close'] != 1000.0
    assert processed_df.loc[4, 'volume'] != 100000

def test_normalize_data_minmax(sample_dataframe, preprocessor):
    """Test Min-Max normalization."""
    df_norm = sample_dataframe.copy()
    processed_df = preprocessor.normalize_data(df_norm, columns=['open', 'close'], method='minmax')
    
    assert processed_df['open'].min() == pytest.approx(0.0)
    assert processed_df['open'].max() == pytest.approx(1.0)
    assert processed_df['close'].min() == pytest.approx(0.0)
    assert processed_df['close'].max() == pytest.approx(1.0)

def test_normalize_data_standard(sample_dataframe, preprocessor):
    """Test Standard (Z-score) normalization."""
    df_norm = sample_dataframe.copy()
    processed_df = preprocessor.normalize_data(df_norm, columns=['open', 'close'], method='standard')
    
    assert processed_df['open'].mean() == pytest.approx(0.0)
    assert processed_df['open'].std() == pytest.approx(1.0)
    assert processed_df['close'].mean() == pytest.approx(0.0)
    assert processed_df['close'].std() == pytest.approx(1.0)

def test_resample_data(sample_dataframe, preprocessor):
    """Test data resampling."""
    df_resample = sample_dataframe.set_index('timestamp')
    preprocessor_resample = DataPreprocessor(resample_interval='2min')
    processed_df = preprocessor_resample.resample_data(df_resample)
    
    assert len(processed_df) == 3 # 09:30, 09:32, 09:34
    assert processed_df.index[0] == pd.to_datetime('2023-01-01 09:30:00')
    assert processed_df.loc['2023-01-01 09:30:00', 'open'] == 100.0
    assert processed_df.loc['2023-01-01 09:30:00', 'close'] == 101.2
    assert processed_df.loc['2023-01-01 09:30:00', 'high'] == 101.5
    assert processed_df.loc['2023-01-01 09:30:00', 'low'] == 99.5
    assert processed_df.loc['2023-01-01 09:30:00', 'volume'] == 2100 # 1000 + 1100

def test_preprocess_pipeline(sample_dataframe, preprocessor):
    """Test the full preprocessing pipeline."""
    df_test = sample_dataframe.copy()
    df_test.loc[1, 'close'] = np.nan # Missing value
    df_test.loc[3, 'open'] = 0.001 # Outlier
    
    processed_df = preprocessor.preprocess(df_test)
    
    assert not processed_df.empty
    assert isinstance(processed_df.index, pd.DatetimeIndex)
    assert not processed_df.isnull().any().any() # No NaNs after ffill
    
    # Check if outlier was handled (value should be different from 0.001)
    assert processed_df.loc[pd.to_datetime('2023-01-01 09:33:00'), 'open'] != 0.001
