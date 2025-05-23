# Data Package - ARM64 Optimized Market Data Pipeline
# Implements high-performance data ingestion, processing, and distribution
# Optimized for NVIDIA GH200 ARM64 platform with Polygon.io integration

import platform

# Core data components
from .data_manager import DataManager
from .data_preprocessing import DataPreprocessor
from .feature_engineering import FeatureEngineeringProcess
from .market_universe import MarketUniverse
from .memory_cache import UnifiedMemoryManager, MemoryBlock
from .polygon_client import (
    AdvancedPolygonClient, 
    PolygonClient,  # Legacy compatibility
    PolygonConfig,
    MarketDataPoint,
    LatencyTracker,
    DataBuffer
)
from .real_time_feed import RealTimeFeed, RealTimeFeedConfig, FeedStatistics

# ARM64-specific optimizations
try:
    # Import ARM64-specific modules if available
    from .arm64_data_processor import ARM64DataProcessor
    from .arm64_feature_engine import ARM64FeatureEngine
    from .arm64_memory_optimizer import ARM64MemoryOptimizer
    ARM64_DATA_MODULES_AVAILABLE = True
except ImportError:
    ARM64_DATA_MODULES_AVAILABLE = False

__all__ = [
    # Core data management
    'DataManager',
    'DataPreprocessor', 
    'FeatureEngineeringProcess',
    'MarketUniverse',
    
    # Memory and caching
    'UnifiedMemoryManager',
    'MemoryBlock',
    
    # Polygon.io integration
    'AdvancedPolygonClient',
    'PolygonClient',
    'PolygonConfig',
    'MarketDataPoint',
    'LatencyTracker',
    'DataBuffer',
    
    # Real-time data feed
    'RealTimeFeed',
    'RealTimeFeedConfig', 
    'FeedStatistics',
]

# Add ARM64 modules to exports if available
if ARM64_DATA_MODULES_AVAILABLE:
    __all__.extend([
        'ARM64DataProcessor',
        'ARM64FeatureEngine', 
        'ARM64MemoryOptimizer'
    ])

# Version and platform info
__version__ = '1.0.0'
__platform__ = 'ARM64-optimized'

# ARM64 optimization status
IS_ARM64 = platform.machine().lower() in ['arm64', 'aarch64']
ARM64_OPTIMIZATIONS_ENABLED = IS_ARM64 and ARM64_DATA_MODULES_AVAILABLE

# Data pipeline configuration
DEFAULT_CONFIG = {
    'polygon_integration': True,
    'arm64_optimizations': ARM64_OPTIMIZATIONS_ENABLED,
    'memory_cache_gb': 200.0,
    'enable_compression': True,
    'enable_batching': True,
    'sub_second_data': True,
    'latency_tracking': True
}

def get_data_pipeline_info():
    """Get information about the data pipeline capabilities."""
    return {
        'version': __version__,
        'platform': __platform__,
        'arm64_enabled': ARM64_OPTIMIZATIONS_ENABLED,
        'polygon_integration': True,
        'sub_second_support': True,
        'memory_optimization': True,
        'real_time_processing': True,
        'feature_engineering': True
    }
