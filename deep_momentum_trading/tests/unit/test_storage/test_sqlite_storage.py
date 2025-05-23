import pytest
import pandas as pd
import numpy as np
import sqlite3
import os
import time
from deep_momentum_trading.src.storage.sqlite_storage import SQLiteTransactionStorage, TradeRecord, PerformanceMetric, ModelMetadata

@pytest.fixture
def temp_db_path(tmp_path):
    """Provides a temporary database file path and cleans up after tests."""
    db_file = tmp_path / "test_trading_history.db"
    yield str(db_file)
    if os.path.exists(db_file):
        os.remove(str(db_file))

@pytest.fixture
def sqlite_storage(temp_db_path):
    """Provides an initialized SQLiteTransactionStorage instance."""
    return SQLiteTransactionStorage(db_path=temp_db_path)

def test_init_creates_tables(sqlite_storage):
    """Test that initialization creates the necessary tables."""
    with sqlite_storage.get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = [row[0] for row in cursor.fetchall()]
        assert "trades" in tables
        assert "performance_metrics" in tables
        assert "model_metadata" in tables

def test_insert_trade(sqlite_storage):
    """Test inserting a single trade record."""
    trade = TradeRecord(
        timestamp=int(time.time() * 1e9), symbol="AAPL", side="buy",
        quantity=10.0, price=150.0, order_id="ORD123", strategy="TestStrat"
    )
    sqlite_storage.insert_trade(trade)
    
    trades_df = sqlite_storage.get_trades_by_date_range(0, int(time.time() * 1e9) + 1e9)
    assert len(trades_df) == 1
    assert trades_df.iloc[0]['symbol'] == "AAPL"
    assert trades_df.iloc[0]['order_id'] == "ORD123"

def test_insert_trade_duplicate_order_id(sqlite_storage):
    """Test inserting a trade with a duplicate order_id."""
    trade1 = TradeRecord(
        timestamp=int(time.time() * 1e9), symbol="AAPL", side="buy",
        quantity=10.0, price=150.0, order_id="DUPLICATE_ID", strategy="TestStrat"
    )
    trade2 = TradeRecord(
        timestamp=int(time.time() * 1e9), symbol="MSFT", side="sell",
        quantity=5.0, price=200.0, order_id="DUPLICATE_ID", strategy="TestStrat"
    )
    sqlite_storage.insert_trade(trade1)
    sqlite_storage.insert_trade(trade2) # This should fail silently or log a warning
    
    trades_df = sqlite_storage.get_trades_by_date_range(0, int(time.time() * 1e9) + 1e9)
    assert len(trades_df) == 1 # Only the first trade should be inserted

def test_get_trades_by_date_range(sqlite_storage):
    """Test retrieving trades within a date range."""
    trade1 = TradeRecord(timestamp=1672531200000000000, symbol="AAPL", side="buy", quantity=10, price=150) # Jan 1, 2023
    trade2 = TradeRecord(timestamp=1672617600000000000, symbol="MSFT", side="sell", quantity=5, price=200) # Jan 2, 2023
    trade3 = TradeRecord(timestamp=1672704000000000000, symbol="GOOG", side="buy", quantity=2, price=100) # Jan 3, 2023
    
    sqlite_storage.insert_trade(trade1)
    sqlite_storage.insert_trade(trade2)
    sqlite_storage.insert_trade(trade3)

    start_ts = 1672531200000000000 # Jan 1
    end_ts = 1672617600000000000 # Jan 2
    trades_df = sqlite_storage.get_trades_by_date_range(start_ts, end_ts)
    
    assert len(trades_df) == 2
    assert "AAPL" in trades_df['symbol'].values
    assert "MSFT" in trades_df['symbol'].values
    assert "GOOG" not in trades_df['symbol'].values

def test_get_portfolio_positions(sqlite_storage):
    """Test calculating portfolio positions."""
    trade1 = TradeRecord(timestamp=1, symbol="AAPL", side="buy", quantity=10, price=150)
    trade2 = TradeRecord(timestamp=2, symbol="MSFT", side="buy", quantity=5, price=200)
    trade3 = TradeRecord(timestamp=3, symbol="AAPL", side="sell", quantity=3, price=155)
    trade4 = TradeRecord(timestamp=4, symbol="MSFT", side="sell", quantity=5, price=205) # Close out MSFT
    
    sqlite_storage.insert_trade(trade1)
    sqlite_storage.insert_trade(trade2)
    sqlite_storage.insert_trade(trade3)
    sqlite_storage.insert_trade(trade4)

    positions = sqlite_storage.get_portfolio_positions(5) # As of latest timestamp
    assert positions == {"AAPL": 7.0} # 10 - 3 = 7, MSFT is 0

def test_insert_performance_metric(sqlite_storage):
    """Test inserting a performance metric."""
    metric = PerformanceMetric(
        timestamp=int(time.time() * 1e9), strategy="Momentum",
        metric_name="sharpe_ratio", metric_value=1.5
    )
    sqlite_storage.insert_performance_metric(metric)
    
    metrics_df = sqlite_storage.get_performance_metrics(strategy="Momentum")
    assert len(metrics_df) == 1
    assert metrics_df.iloc[0]['metric_name'] == "sharpe_ratio"
    assert metrics_df.iloc[0]['metric_value'] == 1.5

def test_get_performance_metrics(sqlite_storage):
    """Test retrieving performance metrics with filters."""
    metric1 = PerformanceMetric(timestamp=1, strategy="Momentum", metric_name="sharpe", metric_value=1.0)
    metric2 = PerformanceMetric(timestamp=2, strategy="Momentum", metric_name="return", metric_value=0.05)
    metric3 = PerformanceMetric(timestamp=3, strategy="Reversion", metric_name="sharpe", metric_value=0.8)
    
    sqlite_storage.insert_performance_metric(metric1)
    sqlite_storage.insert_performance_metric(metric2)
    sqlite_storage.insert_performance_metric(metric3)

    momentum_metrics = sqlite_storage.get_performance_metrics(strategy="Momentum")
    assert len(momentum_metrics) == 2
    assert "Reversion" not in momentum_metrics['strategy'].values

def test_insert_model_metadata(sqlite_storage):
    """Test inserting model metadata."""
    metadata = ModelMetadata(
        model_name="LSTM_V1", version="1.0.0",
        parameters={"layers": 3}, performance_metrics={"accuracy": 0.9}
    )
    sqlite_storage.insert_model_metadata(metadata)
    
    loaded_metadata_df = sqlite_storage.get_model_metadata(model_name="LSTM_V1")
    assert len(loaded_metadata_df) == 1
    assert loaded_metadata_df.iloc[0]['model_name'] == "LSTM_V1"
    assert loaded_metadata_df.iloc[0]['parameters'] == {"layers": 3}

def test_insert_model_metadata_upsert(sqlite_storage):
    """Test UPSERT functionality for model metadata."""
    metadata1 = ModelMetadata(
        model_name="LSTM_V1", version="1.0.0",
        parameters={"layers": 3}, performance_metrics={"accuracy": 0.9}
    )
    metadata2 = ModelMetadata( # Same model_name, same version, updated metrics
        model_name="LSTM_V1", version="1.0.0",
        parameters={"layers": 3, "hidden": 64}, performance_metrics={"accuracy": 0.92}
    )
    sqlite_storage.insert_model_metadata(metadata1)
    sqlite_storage.insert_model_metadata(metadata2)
    
    loaded_metadata_df = sqlite_storage.get_model_metadata(model_name="LSTM_V1")
    assert len(loaded_metadata_df) == 1
    assert loaded_metadata_df.iloc[0]['performance_metrics'] == {"accuracy": 0.92}
    assert loaded_metadata_df.iloc[0]['parameters'] == {"layers": 3, "hidden": 64}

def test_close_connection(sqlite_storage):
    """Test closing the database connection."""
    # Perform an operation to ensure connection is open
    sqlite_storage.insert_trade(TradeRecord(timestamp=1, symbol="A", side="buy", quantity=1, price=1))
    
    # Check if connection attribute exists
    assert hasattr(sqlite_storage.local, 'connection')
    
    sqlite_storage.close_connection()
    
    # Check if connection attribute is removed
    assert not hasattr(sqlite_storage.local, 'connection')
    
    # Attempting to use it again should create a new connection
    sqlite_storage.insert_trade(TradeRecord(timestamp=2, symbol="B", side="buy", quantity=1, price=1))
    assert hasattr(sqlite_storage.local, 'connection')
