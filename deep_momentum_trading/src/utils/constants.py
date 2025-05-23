"""
Enhanced constants for Deep Momentum Trading System with ARM64 optimizations.

This module provides comprehensive constants for trading, models, system configuration,
and ARM64-specific optimizations.
"""

import os
import platform
from typing import Dict, Any, List, Tuple
from enum import Enum, IntEnum
from dataclasses import dataclass

# System Architecture Constants
class SystemArchitecture:
    """System architecture and optimization constants."""
    
    # Platform detection
    IS_ARM64 = platform.machine().lower() in ['arm64', 'aarch64']
    IS_X86_64 = platform.machine().lower() in ['x86_64', 'amd64']
    CPU_COUNT = os.cpu_count()
    
    # ARM64 specific constants
    ARM64_CACHE_LINE_SIZE = 64  # bytes
    ARM64_PAGE_SIZE = 4096      # bytes
    ARM64_NEON_REGISTER_SIZE = 128  # bits
    ARM64_SVE_MAX_VECTOR_LENGTH = 2048  # bits
    
    # Memory alignment for ARM64 SIMD
    MEMORY_ALIGNMENT = 64 if IS_ARM64 else 32
    
    # NUMA configuration
    NUMA_NODES = max(1, len([d for d in os.listdir('/sys/devices/system/node/') 
                            if d.startswith('node')]) if os.path.exists('/sys/devices/system/node/') else 1)
    
    # Threading constants
    MAX_THREADS = CPU_COUNT
    IO_THREADS = min(4, CPU_COUNT)
    COMPUTE_THREADS = max(1, CPU_COUNT - IO_THREADS)

class TradingConstants:
    """Trading-specific constants and configurations."""
    
    # Market hours (UTC)
    MARKET_OPEN_UTC = "13:30:00"  # 9:30 AM EST
    MARKET_CLOSE_UTC = "20:00:00"  # 4:00 PM EST
    PRE_MARKET_START_UTC = "09:00:00"  # 5:00 AM EST
    AFTER_HOURS_END_UTC = "01:00:00"  # 9:00 PM EST (next day)
    
    # Trading session types
    class SessionType(Enum):
        PRE_MARKET = "pre_market"
        REGULAR = "regular"
        AFTER_HOURS = "after_hours"
        CLOSED = "closed"
    
    # Order types
    class OrderType(Enum):
        MARKET = "market"
        LIMIT = "limit"
        STOP = "stop"
        STOP_LIMIT = "stop_limit"
        TRAILING_STOP = "trailing_stop"
        ICEBERG = "iceberg"
        TWAP = "twap"
        VWAP = "vwap"
    
    # Order sides
    class OrderSide(Enum):
        BUY = "buy"
        SELL = "sell"
        BUY_TO_COVER = "buy_to_cover"
        SELL_SHORT = "sell_short"
    
    # Time in force
    class TimeInForce(Enum):
        DAY = "day"
        GTC = "gtc"  # Good Till Canceled
        IOC = "ioc"  # Immediate or Cancel
        FOK = "fok"  # Fill or Kill
        GTD = "gtd"  # Good Till Date
    
    # Position sizes and limits
    MAX_POSITION_SIZE = 1000000  # $1M per position
    MIN_ORDER_SIZE = 1  # 1 share minimum
    MAX_ORDER_SIZE = 100000  # 100K shares maximum
    MAX_PORTFOLIO_VALUE = 10000000  # $10M total portfolio
    
    # Risk management
    MAX_DAILY_LOSS = 0.02  # 2% max daily loss
    MAX_POSITION_RISK = 0.05  # 5% max risk per position
    MAX_SECTOR_EXPOSURE = 0.25  # 25% max sector exposure
    MAX_LEVERAGE = 2.0  # 2:1 maximum leverage
    
    # Commission and fees
    COMMISSION_PER_SHARE = 0.005  # $0.005 per share
    MIN_COMMISSION = 1.00  # $1.00 minimum
    SEC_FEE_RATE = 0.0000278  # SEC fee rate
    FINRA_TAF_RATE = 0.000145  # FINRA TAF rate
    
    # Market data frequencies
    TICK_FREQUENCY = "1ms"  # Millisecond ticks
    SECOND_FREQUENCY = "1s"
    MINUTE_FREQUENCY = "1min"
    HOUR_FREQUENCY = "1h"
    DAILY_FREQUENCY = "1d"
    
    # Supported exchanges
    EXCHANGES = [
        "NYSE", "NASDAQ", "AMEX", "ARCA", "BATS", "IEX",
        "EDGX", "EDGA", "BYX", "BZX", "PSX", "CHX"
    ]
    
    # Market data providers
    DATA_PROVIDERS = {
        "polygon": "wss://socket.polygon.io/stocks",
        "alpaca": "wss://stream.data.alpaca.markets/v2/stocks"
    }

class ModelConstants:
    """Machine learning model constants and configurations."""
    
    # Model types
    class ModelType(Enum):
        LSTM = "lstm"
        GRU = "gru"
        TRANSFORMER = "transformer"
        CNN_LSTM = "cnn_lstm"
        ATTENTION_LSTM = "attention_lstm"
        DEEP_MOMENTUM = "deep_momentum"
        ENSEMBLE = "ensemble"
    
    # Model architectures
    LSTM_ARCHITECTURES = {
        "small": {"layers": 2, "units": 64, "dropout": 0.2},
        "medium": {"layers": 3, "units": 128, "dropout": 0.3},
        "large": {"layers": 4, "units": 256, "dropout": 0.4},
        "xlarge": {"layers": 6, "units": 512, "dropout": 0.5}
    }
    
    # Training parameters
    DEFAULT_BATCH_SIZE = 64 if SystemArchitecture.IS_ARM64 else 32
    DEFAULT_EPOCHS = 100
    DEFAULT_LEARNING_RATE = 0.001
    DEFAULT_PATIENCE = 10
    DEFAULT_VALIDATION_SPLIT = 0.2
    
    # ARM64 optimized batch sizes
    ARM64_BATCH_SIZES = {
        "small_model": 128,
        "medium_model": 64,
        "large_model": 32,
        "xlarge_model": 16
    }
    
    # Feature engineering
    TECHNICAL_INDICATORS = [
        "sma", "ema", "rsi", "macd", "bollinger_bands",
        "stochastic", "williams_r", "cci", "atr", "adx",
        "obv", "vwap", "momentum", "roc", "trix"
    ]
    
    LOOKBACK_PERIODS = [5, 10, 20, 50, 100, 200]
    
    # Model evaluation metrics
    REGRESSION_METRICS = [
        "mse", "rmse", "mae", "mape", "r2", "explained_variance"
    ]
    
    CLASSIFICATION_METRICS = [
        "accuracy", "precision", "recall", "f1", "auc", "log_loss"
    ]
    
    TRADING_METRICS = [
        "sharpe_ratio", "sortino_ratio", "calmar_ratio",
        "max_drawdown", "total_return", "annual_return",
        "volatility", "beta", "alpha", "information_ratio"
    ]
    
    # Model persistence
    MODEL_SAVE_FORMAT = "pytorch"  # pytorch, onnx, tensorrt
    CHECKPOINT_FREQUENCY = 10  # Save every 10 epochs
    MAX_CHECKPOINTS = 5  # Keep last 5 checkpoints
    
    # Distributed training
    DISTRIBUTED_BACKEND = "nccl" if SystemArchitecture.IS_ARM64 else "gloo"
    GRADIENT_ACCUMULATION_STEPS = 4
    MIXED_PRECISION = True  # Enable for ARM64

class DataConstants:
    """Data processing and storage constants."""
    
    # Data types and precision
    PRICE_PRECISION = 4  # 4 decimal places for prices
    VOLUME_PRECISION = 0  # Whole numbers for volume
    RATIO_PRECISION = 6  # 6 decimal places for ratios
    
    # ARM64 optimized data types
    if SystemArchitecture.IS_ARM64:
        FLOAT_TYPE = "float32"  # ARM64 NEON optimized
        INT_TYPE = "int32"
    else:
        FLOAT_TYPE = "float64"
        INT_TYPE = "int64"
    
    # Data validation ranges
    PRICE_RANGE = (0.01, 100000.0)  # $0.01 to $100K
    VOLUME_RANGE = (1, 1000000000)  # 1 to 1B shares
    RATIO_RANGE = (-10.0, 10.0)  # -1000% to +1000%
    
    # Missing data handling
    class MissingDataStrategy(Enum):
        DROP = "drop"
        FORWARD_FILL = "ffill"
        BACKWARD_FILL = "bfill"
        INTERPOLATE = "interpolate"
        MEAN_FILL = "mean"
        MEDIAN_FILL = "median"
        ZERO_FILL = "zero"
    
    # Data storage formats
    class StorageFormat(Enum):
        PARQUET = "parquet"
        HDF5 = "hdf5"
        CSV = "csv"
        FEATHER = "feather"
        PICKLE = "pickle"
        SQLITE = "sqlite"
    
    # Compression algorithms
    COMPRESSION_ALGORITHMS = {
        "lz4": {"speed": "fast", "ratio": "medium"},
        "zstd": {"speed": "medium", "ratio": "high"},
        "snappy": {"speed": "fast", "ratio": "low"},
        "gzip": {"speed": "slow", "ratio": "high"}
    }
    
    # Data pipeline constants
    CHUNK_SIZE = 10000  # Process 10K records at a time
    BUFFER_SIZE = 100000  # 100K record buffer
    MAX_MEMORY_USAGE = 8 * 1024**3  # 8GB max memory
    
    # Real-time data
    WEBSOCKET_TIMEOUT = 30.0  # seconds
    RECONNECT_ATTEMPTS = 5
    HEARTBEAT_INTERVAL = 30.0  # seconds
    MESSAGE_QUEUE_SIZE = 100000

class SystemConstants:
    """System-level constants and configurations."""
    
    # Logging configuration
    class LogLevel(IntEnum):
        DEBUG = 10
        INFO = 20
        WARNING = 30
        ERROR = 40
        CRITICAL = 50
    
    DEFAULT_LOG_LEVEL = LogLevel.INFO
    LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
    
    # File paths
    DATA_DIR = "data"
    MODELS_DIR = "models"
    LOGS_DIR = "logs"
    CONFIG_DIR = "config"
    CACHE_DIR = "cache"
    TEMP_DIR = "temp"
    
    # Cache configuration
    CACHE_TTL = 3600  # 1 hour default TTL
    MAX_CACHE_SIZE = 1024 * 1024 * 1024  # 1GB
    CACHE_CLEANUP_INTERVAL = 300  # 5 minutes
    
    # Network configuration
    DEFAULT_TIMEOUT = 30.0  # seconds
    MAX_RETRIES = 3
    RETRY_BACKOFF = 2.0  # exponential backoff multiplier
    CONNECTION_POOL_SIZE = 10
    
    # Security
    API_KEY_LENGTH = 32
    SESSION_TIMEOUT = 3600  # 1 hour
    MAX_LOGIN_ATTEMPTS = 5
    PASSWORD_MIN_LENGTH = 8
    
    # Performance monitoring
    METRICS_COLLECTION_INTERVAL = 60  # seconds
    PERFORMANCE_HISTORY_SIZE = 1000  # Keep last 1000 measurements
    ALERT_THRESHOLDS = {
        "cpu_usage": 80.0,  # 80%
        "memory_usage": 85.0,  # 85%
        "disk_usage": 90.0,  # 90%
        "latency_ms": 100.0  # 100ms
    }

class ARM64OptimizationConstants:
    """ARM64-specific optimization constants."""
    
    # NEON SIMD constants
    NEON_REGISTER_COUNT = 32
    NEON_REGISTER_WIDTH = 128  # bits
    NEON_FLOAT32_ELEMENTS = 4  # per register
    NEON_FLOAT64_ELEMENTS = 2  # per register
    
    # SVE (Scalable Vector Extension) constants
    SVE_MIN_VECTOR_LENGTH = 128  # bits
    SVE_MAX_VECTOR_LENGTH = 2048  # bits
    SVE_VECTOR_LENGTH_INCREMENT = 128  # bits
    
    # Memory optimization
    CACHE_LINE_SIZE = 64  # bytes
    PAGE_SIZE = 4096  # bytes
    HUGE_PAGE_SIZE = 2 * 1024 * 1024  # 2MB
    
    # Threading optimization
    ARM64_THREAD_AFFINITY = True
    ARM64_NUMA_BINDING = True
    ARM64_CPU_SCALING = "performance"
    
    # Compiler optimizations
    ARM64_COMPILER_FLAGS = [
        "-march=armv8-a",
        "-mtune=cortex-a76",
        "-O3",
        "-ffast-math",
        "-funroll-loops",
        "-fvectorize"
    ]
    
    # Library optimizations
    ARM64_BLAS_LIBRARY = "openblas"  # openblas, atlas, mkl
    ARM64_LAPACK_LIBRARY = "openblas"
    ARM64_FFT_LIBRARY = "fftw3"

# Environment-specific constants
@dataclass
class EnvironmentConfig:
    """Environment-specific configuration."""
    
    # Development
    DEV_CONFIG = {
        "debug": True,
        "log_level": SystemConstants.LogLevel.DEBUG,
        "cache_enabled": False,
        "profiling_enabled": True,
        "mock_data": True
    }
    
    # Testing
    TEST_CONFIG = {
        "debug": True,
        "log_level": SystemConstants.LogLevel.INFO,
        "cache_enabled": False,
        "profiling_enabled": False,
        "mock_data": True,
        "test_data_size": 1000
    }
    
    # Staging
    STAGING_CONFIG = {
        "debug": False,
        "log_level": SystemConstants.LogLevel.INFO,
        "cache_enabled": True,
        "profiling_enabled": True,
        "mock_data": False
    }
    
    # Production
    PROD_CONFIG = {
        "debug": False,
        "log_level": SystemConstants.LogLevel.WARNING,
        "cache_enabled": True,
        "profiling_enabled": False,
        "mock_data": False,
        "monitoring_enabled": True,
        "alerting_enabled": True
    }

# Global configuration based on environment
ENVIRONMENT = os.getenv("TRADING_ENV", "development").lower()
ENV_CONFIGS = {
    "development": EnvironmentConfig.DEV_CONFIG,
    "testing": EnvironmentConfig.TEST_CONFIG,
    "staging": EnvironmentConfig.STAGING_CONFIG,
    "production": EnvironmentConfig.PROD_CONFIG
}

CURRENT_CONFIG = ENV_CONFIGS.get(ENVIRONMENT, EnvironmentConfig.DEV_CONFIG)

# Export all constants
__all__ = [
    "SystemArchitecture",
    "TradingConstants", 
    "ModelConstants",
    "DataConstants",
    "SystemConstants",
    "ARM64OptimizationConstants",
    "EnvironmentConfig",
    "ENVIRONMENT",
    "CURRENT_CONFIG"
]