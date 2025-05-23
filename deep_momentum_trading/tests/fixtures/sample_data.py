"""
Sample data generators for Deep Momentum Trading System tests.

This module provides comprehensive sample data generation utilities for testing
all components of the trading system with realistic market data patterns.
"""

import numpy as np
import pandas as pd
import torch
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any, Optional, Tuple
import random
from dataclasses import dataclass

# Set random seeds for reproducible test data
np.random.seed(42)
random.seed(42)
torch.manual_seed(42)

@dataclass
class MarketDataConfig:
    """Configuration for market data generation."""
    n_records: int = 1000
    base_price: float = 100.0
    volatility: float = 0.02
    trend: float = 0.0005
    start_date: str = "2023-01-01"
    frequency: str = "1min"
    add_gaps: bool = True
    add_outliers: bool = True

class SampleDataGenerator:
    """Comprehensive sample data generator for testing."""
    
    def __init__(self, config: Optional[MarketDataConfig] = None):
        self.config = config or MarketDataConfig()
    
    def generate_market_data(self, symbol: str = "AAPL") -> pd.DataFrame:
        """Generate realistic OHLCV market data."""
        n = self.config.n_records
        
        # Generate price series with trend and volatility
        returns = np.random.normal(
            self.config.trend, 
            self.config.volatility, 
            n
        )
        
        # Add some autocorrelation for realism
        for i in range(1, len(returns)):
            returns[i] += 0.1 * returns[i-1]
        
        # Calculate prices
        prices = [self.config.base_price]
        for ret in returns:
            prices.append(prices[-1] * (1 + ret))
        
        close_prices = np.array(prices[1:])
        
        # Generate OHLC from close prices
        open_prices = np.roll(close_prices, 1)
        open_prices[0] = self.config.base_price
        
        # Add intraday volatility
        intraday_vol = np.random.uniform(0.001, 0.01, n)
        high_prices = close_prices * (1 + intraday_vol)
        low_prices = close_prices * (1 - intraday_vol)
        
        # Ensure OHLC consistency
        for i in range(n):
            high_prices[i] = max(high_prices[i], open_prices[i], close_prices[i])
            low_prices[i] = min(low_prices[i], open_prices[i], close_prices[i])
        
        # Generate volume with realistic patterns
        base_volume = 50000
        volume_trend = np.random.exponential(1.0, n)
        volume = (base_volume * volume_trend).astype(int)
        
        # Add volume spikes during price movements
        price_changes = np.abs(np.diff(close_prices, prepend=close_prices[0]))
        volume_multiplier = 1 + (price_changes / np.mean(price_changes))
        volume = (volume * volume_multiplier).astype(int)
        
        # Calculate VWAP
        typical_price = (high_prices + low_prices + close_prices) / 3
        vwap = np.cumsum(typical_price * volume) / np.cumsum(volume)
        
        # Create DataFrame
        timestamps = pd.date_range(
            start=self.config.start_date,
            periods=n,
            freq=self.config.frequency
        )
        
        data = {
            'open': open_prices,
            'high': high_prices,
            'low': low_prices,
            'close': close_prices,
            'volume': volume,
            'vwap': vwap,
            'symbol': symbol
        }
        
        df = pd.DataFrame(data, index=timestamps)
        df.index.name = 'timestamp'
        
        # Add market gaps and outliers if configured
        if self.config.add_gaps:
            df = self._add_market_gaps(df)
        
        if self.config.add_outliers:
            df = self._add_outliers(df)
        
        return df
    
    def _add_market_gaps(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add realistic market gaps to the data."""
        n_gaps = max(1, len(df) // 100)  # 1% of data points
        gap_indices = np.random.choice(len(df), n_gaps, replace=False)
        
        for idx in gap_indices:
            if idx > 0:
                gap_size = np.random.uniform(0.02, 0.05)  # 2-5% gap
                direction = np.random.choice([-1, 1])
                
                df.iloc[idx:, df.columns.get_loc('open')] *= (1 + direction * gap_size)
                df.iloc[idx:, df.columns.get_loc('high')] *= (1 + direction * gap_size)
                df.iloc[idx:, df.columns.get_loc('low')] *= (1 + direction * gap_size)
                df.iloc[idx:, df.columns.get_loc('close')] *= (1 + direction * gap_size)
        
        return df
    
    def _add_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add realistic outliers to the data."""
        n_outliers = max(1, len(df) // 200)  # 0.5% of data points
        outlier_indices = np.random.choice(len(df), n_outliers, replace=False)
        
        for idx in outlier_indices:
            outlier_multiplier = np.random.uniform(1.1, 1.3)  # 10-30% outlier
            direction = np.random.choice([-1, 1])
            
            if direction > 0:
                df.iloc[idx, df.columns.get_loc('high')] *= outlier_multiplier
            else:
                df.iloc[idx, df.columns.get_loc('low')] /= outlier_multiplier
            
            # Increase volume for outliers
            df.iloc[idx, df.columns.get_loc('volume')] *= 2
        
        return df
    
    def generate_multi_symbol_data(self, symbols: List[str]) -> Dict[str, pd.DataFrame]:
        """Generate market data for multiple symbols with correlations."""
        data = {}
        base_returns = np.random.normal(0.0005, 0.02, self.config.n_records)
        
        for i, symbol in enumerate(symbols):
            # Add symbol-specific characteristics
            symbol_returns = base_returns.copy()
            
            # Add correlation with base returns
            correlation = np.random.uniform(0.3, 0.8)
            noise = np.random.normal(0, 0.01, self.config.n_records)
            symbol_returns = correlation * base_returns + (1 - correlation) * noise
            
            # Adjust base price and volatility per symbol
            config = MarketDataConfig(
                n_records=self.config.n_records,
                base_price=self.config.base_price * (0.5 + i * 0.3),
                volatility=self.config.volatility * (0.8 + i * 0.1),
                start_date=self.config.start_date,
                frequency=self.config.frequency
            )
            
            generator = SampleDataGenerator(config)
            data[symbol] = generator.generate_market_data(symbol)
        
        return data
    
    def generate_features_data(self, market_data: pd.DataFrame) -> pd.DataFrame:
        """Generate realistic technical features from market data."""
        features = {}
        
        # Price-based features
        features['returns'] = market_data['close'].pct_change()
        features['log_returns'] = np.log(market_data['close'] / market_data['close'].shift(1))
        features['price_change'] = market_data['close'].diff()
        features['price_range'] = market_data['high'] - market_data['low']
        features['body_size'] = np.abs(market_data['close'] - market_data['open'])
        features['upper_shadow'] = market_data['high'] - np.maximum(market_data['open'], market_data['close'])
        features['lower_shadow'] = np.minimum(market_data['open'], market_data['close']) - market_data['low']
        
        # Moving averages
        for window in [5, 10, 20, 50]:
            features[f'sma_{window}'] = market_data['close'].rolling(window).mean()
            features[f'ema_{window}'] = market_data['close'].ewm(span=window).mean()
            features[f'price_to_sma_{window}'] = market_data['close'] / features[f'sma_{window}']
        
        # Volatility features
        for window in [10, 20, 50]:
            features[f'volatility_{window}'] = market_data['close'].rolling(window).std()
            features[f'volatility_ratio_{window}'] = features[f'volatility_{window}'] / features['volatility_50']
        
        # Volume features
        features['volume_sma_10'] = market_data['volume'].rolling(10).mean()
        features['volume_sma_20'] = market_data['volume'].rolling(20).mean()
        features['volume_ratio'] = market_data['volume'] / features['volume_sma_20']
        features['price_volume'] = market_data['close'] * market_data['volume']
        
        # Technical indicators (simplified versions)
        features['rsi_14'] = self._calculate_rsi(market_data['close'], 14)
        features['rsi_28'] = self._calculate_rsi(market_data['close'], 28)
        
        # MACD
        ema_12 = market_data['close'].ewm(span=12).mean()
        ema_26 = market_data['close'].ewm(span=26).mean()
        features['macd'] = ema_12 - ema_26
        features['macd_signal'] = features['macd'].ewm(span=9).mean()
        features['macd_histogram'] = features['macd'] - features['macd_signal']
        
        # Bollinger Bands
        sma_20 = market_data['close'].rolling(20).mean()
        std_20 = market_data['close'].rolling(20).std()
        features['bb_upper'] = sma_20 + (2 * std_20)
        features['bb_lower'] = sma_20 - (2 * std_20)
        features['bb_position'] = (market_data['close'] - features['bb_lower']) / (features['bb_upper'] - features['bb_lower'])
        
        # Momentum features
        for period in [5, 10, 20]:
            features[f'momentum_{period}'] = market_data['close'] / market_data['close'].shift(period) - 1
            features[f'roc_{period}'] = market_data['close'].pct_change(period)
        
        # Statistical features
        for window in [10, 20]:
            features[f'skewness_{window}'] = market_data['close'].rolling(window).skew()
            features[f'kurtosis_{window}'] = market_data['close'].rolling(window).kurt()
        
        # Create DataFrame
        features_df = pd.DataFrame(features, index=market_data.index)
        
        # Forward fill and backward fill NaN values
        features_df = features_df.ffill().bfill()
        
        return features_df
    
    def _calculate_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """Calculate RSI indicator."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def generate_model_predictions(self, symbols: List[str], n_predictions: int = 100) -> List[Dict[str, Any]]:
        """Generate realistic model predictions."""
        predictions = []
        
        for _ in range(n_predictions):
            symbol = np.random.choice(symbols)
            
            # Generate correlated position and confidence
            base_signal = np.random.normal(0, 0.5)
            position = np.tanh(base_signal)  # Bounded between -1 and 1
            confidence = min(0.95, max(0.05, np.abs(base_signal) / 2))  # Higher confidence for stronger signals
            
            prediction = {
                'symbol': symbol,
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'position': position,
                'confidence': confidence,
                'expected_return': position * confidence * np.random.uniform(0.01, 0.05),
                'volatility': np.random.uniform(0.1, 0.3),
                'strategy': 'deep_momentum_lstm',
                'model_version': '1.0.0',
                'features_used': np.random.randint(50, 200),
                'prediction_horizon': '1h'
            }
            
            predictions.append(prediction)
        
        return predictions
    
    def generate_portfolio_positions(self, symbols: List[str], total_value: float = 100000) -> Dict[str, Dict[str, Any]]:
        """Generate realistic portfolio positions."""
        positions = {}
        remaining_value = total_value
        
        # Randomly select subset of symbols to have positions
        n_positions = np.random.randint(len(symbols) // 2, len(symbols))
        selected_symbols = np.random.choice(symbols, n_positions, replace=False)
        
        for i, symbol in enumerate(selected_symbols):
            if i == len(selected_symbols) - 1:
                # Last position gets remaining value
                position_value = remaining_value
            else:
                # Random allocation between 5% and 20% of total
                position_value = np.random.uniform(0.05, 0.2) * total_value
                remaining_value -= position_value
            
            # Random price for position calculation
            price = np.random.uniform(50, 300)
            quantity = int(position_value / price)
            
            # Random side (long/short)
            side = np.random.choice(['long', 'short'])
            if side == 'short':
                quantity = -quantity
            
            # Calculate unrealized P&L
            current_price = price * (1 + np.random.normal(0, 0.05))
            unrealized_pnl = quantity * (current_price - price)
            
            positions[symbol] = {
                'symbol': symbol,
                'quantity': quantity,
                'market_value': quantity * current_price,
                'cost_basis': price,
                'current_price': current_price,
                'unrealized_pnl': unrealized_pnl,
                'side': side,
                'entry_time': datetime.now(timezone.utc) - timedelta(hours=np.random.randint(1, 168)),
                'position_id': f"pos_{symbol}_{np.random.randint(1000, 9999)}"
            }
        
        return positions
    
    def generate_risk_scenarios(self) -> Dict[str, Dict[str, float]]:
        """Generate risk scenario data for stress testing."""
        scenarios = {
            'market_crash': {
                'equity_shock': -0.20,
                'volatility_multiplier': 2.5,
                'correlation_increase': 0.3,
                'liquidity_decrease': 0.5
            },
            'interest_rate_shock': {
                'equity_shock': -0.10,
                'volatility_multiplier': 1.5,
                'correlation_increase': 0.2,
                'sector_rotation': 0.15
            },
            'liquidity_crisis': {
                'equity_shock': -0.15,
                'volatility_multiplier': 2.0,
                'bid_ask_spread_increase': 3.0,
                'volume_decrease': 0.7
            },
            'sector_rotation': {
                'tech_shock': -0.25,
                'finance_boost': 0.15,
                'correlation_decrease': -0.2,
                'volatility_multiplier': 1.3
            },
            'flash_crash': {
                'equity_shock': -0.30,
                'recovery_time_minutes': 15,
                'volatility_spike': 5.0,
                'volume_spike': 10.0
            }
        }
        
        return scenarios
    
    def generate_correlation_matrix(self, symbols: List[str]) -> pd.DataFrame:
        """Generate realistic correlation matrix for symbols."""
        n_symbols = len(symbols)
        
        # Start with random correlations
        correlations = np.random.uniform(0.1, 0.8, (n_symbols, n_symbols))
        
        # Make symmetric
        correlations = (correlations + correlations.T) / 2
        
        # Set diagonal to 1
        np.fill_diagonal(correlations, 1.0)
        
        # Ensure positive semi-definite
        eigenvals, eigenvecs = np.linalg.eigh(correlations)
        eigenvals = np.maximum(eigenvals, 0.01)  # Ensure positive eigenvalues
        correlations = eigenvecs @ np.diag(eigenvals) @ eigenvecs.T
        
        # Normalize diagonal back to 1
        d = np.sqrt(np.diag(correlations))
        correlations = correlations / np.outer(d, d)
        
        return pd.DataFrame(correlations, index=symbols, columns=symbols)
    
    def generate_torch_tensors(self, batch_size: int = 32, sequence_length: int = 60, 
                             n_features: int = 200) -> Dict[str, torch.Tensor]:
        """Generate PyTorch tensors for model testing."""
        tensors = {
            'input_features': torch.randn(batch_size, sequence_length, n_features),
            'target_positions': torch.randn(batch_size, 1),
            'target_confidence': torch.sigmoid(torch.randn(batch_size, 1)),
            'target_returns': torch.randn(batch_size, 1) * 0.05,
            'mask': torch.ones(batch_size, sequence_length, dtype=torch.bool)
        }
        
        # Add some realistic patterns
        # Trend in features
        for i in range(batch_size):
            trend = torch.linspace(-0.1, 0.1, sequence_length).unsqueeze(1)
            tensors['input_features'][i] += trend * torch.randn(1, n_features) * 0.1
        
        # Correlation between position and confidence
        position_strength = torch.abs(tensors['target_positions'])
        tensors['target_confidence'] = torch.sigmoid(position_strength + torch.randn_like(position_strength) * 0.2)
        
        return tensors

# Convenience functions for common test data
def get_sample_market_data(symbol: str = "AAPL", n_records: int = 1000) -> pd.DataFrame:
    """Get sample market data for testing."""
    config = MarketDataConfig(n_records=n_records)
    generator = SampleDataGenerator(config)
    return generator.generate_market_data(symbol)

def get_sample_features_data(n_records: int = 1000, n_features: int = 50) -> pd.DataFrame:
    """Get sample features data for testing."""
    market_data = get_sample_market_data(n_records=n_records)
    generator = SampleDataGenerator()
    return generator.generate_features_data(market_data)

def get_sample_multi_symbol_data(symbols: List[str] = None) -> Dict[str, pd.DataFrame]:
    """Get sample multi-symbol market data for testing."""
    if symbols is None:
        symbols = ["AAPL", "MSFT", "GOOGL", "TSLA", "NVDA"]
    
    generator = SampleDataGenerator()
    return generator.generate_multi_symbol_data(symbols)

def get_sample_predictions(symbols: List[str] = None, n_predictions: int = 100) -> List[Dict[str, Any]]:
    """Get sample model predictions for testing."""
    if symbols is None:
        symbols = ["AAPL", "MSFT", "GOOGL", "TSLA", "NVDA"]
    
    generator = SampleDataGenerator()
    return generator.generate_model_predictions(symbols, n_predictions)

def get_sample_portfolio(symbols: List[str] = None, total_value: float = 100000) -> Dict[str, Dict[str, Any]]:
    """Get sample portfolio positions for testing."""
    if symbols is None:
        symbols = ["AAPL", "MSFT", "GOOGL", "TSLA", "NVDA"]
    
    generator = SampleDataGenerator()
    return generator.generate_portfolio_positions(symbols, total_value)

def get_sample_torch_data(batch_size: int = 32, sequence_length: int = 60,
                         n_features: int = 200) -> Dict[str, torch.Tensor]:
    """Get sample PyTorch tensors for testing."""
    generator = SampleDataGenerator()
    return generator.generate_torch_tensors(batch_size, sequence_length, n_features)

def get_sample_risk_scenarios() -> Dict[str, Dict[str, float]]:
    """Get sample risk scenarios for testing."""
    generator = SampleDataGenerator()
    return generator.generate_risk_scenarios()

def get_sample_correlation_matrix(symbols: List[str] = None) -> pd.DataFrame:
    """Get sample correlation matrix for testing."""
    if symbols is None:
        symbols = ["AAPL", "MSFT", "GOOGL", "TSLA", "NVDA"]
    
    generator = SampleDataGenerator()
    return generator.generate_correlation_matrix(symbols)

def get_sample_time_series_data(n_series: int = 5, n_points: int = 1000) -> Dict[str, pd.Series]:
    """Get sample time series data for testing."""
    data = {}
    timestamps = pd.date_range(start="2023-01-01", periods=n_points, freq="1min")
    
    for i in range(n_series):
        # Generate correlated time series
        base_trend = np.cumsum(np.random.normal(0, 0.01, n_points))
        noise = np.random.normal(0, 0.05, n_points)
        values = base_trend + noise
        
        data[f"series_{i+1}"] = pd.Series(values, index=timestamps)
    
    return data

def get_sample_order_book_data(symbol: str = "AAPL", depth: int = 10) -> Dict[str, Any]:
    """Get sample order book data for testing."""
    mid_price = 150.0
    spread = 0.01
    
    bids = []
    asks = []
    
    for i in range(depth):
        bid_price = mid_price - spread/2 - i * 0.01
        ask_price = mid_price + spread/2 + i * 0.01
        
        bid_size = np.random.randint(100, 1000)
        ask_size = np.random.randint(100, 1000)
        
        bids.append({"price": bid_price, "size": bid_size})
        asks.append({"price": ask_price, "size": ask_size})
    
    return {
        "symbol": symbol,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "bids": bids,
        "asks": asks,
        "mid_price": mid_price,
        "spread": spread
    }

def get_sample_trade_data(symbol: str = "AAPL", n_trades: int = 100) -> List[Dict[str, Any]]:
    """Get sample trade data for testing."""
    trades = []
    base_price = 150.0
    
    for i in range(n_trades):
        price = base_price * (1 + np.random.normal(0, 0.01))
        size = np.random.randint(100, 10000)
        side = np.random.choice(["buy", "sell"])
        
        trade = {
            "symbol": symbol,
            "timestamp": datetime.now(timezone.utc) - timedelta(minutes=n_trades-i),
            "price": price,
            "size": size,
            "side": side,
            "trade_id": f"trade_{i+1:06d}"
        }
        trades.append(trade)
    
    return trades

# Export main classes and functions
__all__ = [
    'MarketDataConfig',
    'SampleDataGenerator',
    'get_sample_market_data',
    'get_sample_features_data',
    'get_sample_multi_symbol_data',
    'get_sample_predictions',
    'get_sample_portfolio',
    'get_sample_torch_data',
    'get_sample_risk_scenarios',
    'get_sample_correlation_matrix',
    'get_sample_time_series_data',
    'get_sample_order_book_data',
    'get_sample_trade_data'
]
