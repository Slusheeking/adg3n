"""
Deep Momentum Trading System - Main Package
============================================

An advanced deep learning momentum trading system leveraging NVIDIA GH200 Grace Hopper 
architecture to achieve superior risk-adjusted returns through LSTM-based momentum 
detection and Sharpe ratio optimization.

This package provides:
- Advanced neural architectures (LSTM, Transformer, Ensemble)
- Real-time data processing and feature engineering
- Risk management and portfolio optimization
- High-frequency trading execution
- ARM64/GH200 optimizations
- Comprehensive monitoring and alerting

Author: Deep Momentum Trading Team
License: MIT
Version: 1.0.0
"""

import sys
import platform
import warnings
from typing import Dict, Any, Optional
import logging

# Package metadata
__version__ = "1.0.0"
__author__ = "Deep Momentum Trading Team"
__email__ = "contact@deepmomentum.ai"
__license__ = "MIT"
__description__ = "Advanced deep learning momentum trading system with ARM64/GH200 optimizations"

# Minimum Python version check
MIN_PYTHON_VERSION = (3, 11)
if sys.version_info < MIN_PYTHON_VERSION:
    raise RuntimeError(
        f"Deep Momentum Trading System requires Python {MIN_PYTHON_VERSION[0]}.{MIN_PYTHON_VERSION[1]}+ "
        f"but you are running Python {sys.version_info.major}.{sys.version_info.minor}"
    )

# ARM64/GH200 detection
def detect_arm64() -> bool:
    """Detect if running on ARM64 architecture."""
    machine = platform.machine().lower()
    return machine in ['arm64', 'aarch64']

def detect_gh200() -> bool:
    """Detect if running on NVIDIA GH200 Grace Hopper."""
    try:
        # Check for GH200-specific indicators
        with open('/proc/cpuinfo', 'r') as f:
            cpuinfo = f.read().lower()
            return 'neoverse-v2' in cpuinfo or 'grace' in cpuinfo
    except (FileNotFoundError, PermissionError):
        return False

# System information
SYSTEM_INFO = {
    'python_version': f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
    'platform': platform.platform(),
    'machine': platform.machine(),
    'processor': platform.processor(),
    'is_arm64': detect_arm64(),
    'is_gh200': detect_gh200(),
    'package_version': __version__
}

# Configure logging for the package
def setup_logging(level: str = "INFO") -> logging.Logger:
    """Setup package-wide logging configuration."""
    logger = logging.getLogger(__name__)
    
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    logger.setLevel(getattr(logging, level.upper()))
    return logger

# Package logger
logger = setup_logging()

# ARM64/GH200 optimization warnings
if SYSTEM_INFO['is_arm64']:
    logger.info("✅ ARM64 architecture detected - optimizations enabled")
    if SYSTEM_INFO['is_gh200']:
        logger.info("🚀 NVIDIA GH200 Grace Hopper detected - maximum performance mode")
else:
    logger.warning("⚠️  Non-ARM64 architecture detected - some optimizations may not be available")

# Lazy imports for better startup performance
_LAZY_IMPORTS = {}

def _lazy_import(module_name: str, package: str = None):
    """Lazy import helper to improve startup time."""
    if module_name not in _LAZY_IMPORTS:
        if package:
            full_name = f"{package}.{module_name}"
        else:
            full_name = module_name
        
        try:
            _LAZY_IMPORTS[module_name] = __import__(full_name, fromlist=[''])
        except ImportError as e:
            logger.warning(f"Failed to import {full_name}: {e}")
            _LAZY_IMPORTS[module_name] = None
    
    return _LAZY_IMPORTS[module_name]

# Core module imports with error handling
def get_models():
    """Get models module with lazy loading."""
    return _lazy_import('models', 'src')

def get_data():
    """Get data module with lazy loading."""
    return _lazy_import('data', 'src')

def get_trading():
    """Get trading module with lazy loading."""
    return _lazy_import('trading', 'src')

def get_risk():
    """Get risk module with lazy loading."""
    return _lazy_import('risk', 'src')

def get_training():
    """Get training module with lazy loading."""
    return _lazy_import('training', 'src')

def get_monitoring():
    """Get monitoring module with lazy loading."""
    return _lazy_import('monitoring', 'src')

def get_communication():
    """Get communication module with lazy loading."""
    return _lazy_import('communication', 'src')

def get_storage():
    """Get storage module with lazy loading."""
    return _lazy_import('storage', 'src')

def get_infrastructure():
    """Get infrastructure module with lazy loading."""
    return _lazy_import('infrastructure', 'src')

def get_utils():
    """Get utils module with lazy loading."""
    return _lazy_import('utils', 'src')

# Convenience functions for common operations
def get_system_info() -> Dict[str, Any]:
    """Get comprehensive system information."""
    info = SYSTEM_INFO.copy()
    
    # Add runtime information
    try:
        import torch
        info['torch_version'] = torch.__version__
        info['cuda_available'] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info['cuda_version'] = torch.version.cuda
            info['gpu_count'] = torch.cuda.device_count()
            info['gpu_names'] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    except ImportError:
        info['torch_available'] = False
    
    try:
        import numpy as np
        info['numpy_version'] = np.__version__
    except ImportError:
        info['numpy_available'] = False
    
    try:
        import pandas as pd
        info['pandas_version'] = pd.__version__
    except ImportError:
        info['pandas_available'] = False
    
    return info

def validate_environment() -> Dict[str, bool]:
    """Validate the environment for optimal performance."""
    validation = {
        'python_version_ok': sys.version_info >= MIN_PYTHON_VERSION,
        'arm64_optimized': SYSTEM_INFO['is_arm64'],
        'gh200_detected': SYSTEM_INFO['is_gh200'],
    }
    
    # Check required packages
    required_packages = ['torch', 'numpy', 'pandas', 'scipy', 'sklearn']
    for package in required_packages:
        try:
            __import__(package)
            validation[f'{package}_available'] = True
        except ImportError:
            validation[f'{package}_available'] = False
    
    # Check optional packages
    optional_packages = ['numba', 'h5py', 'pyzmq', 'websockets']
    for package in optional_packages:
        try:
            __import__(package)
            validation[f'{package}_available'] = True
        except ImportError:
            validation[f'{package}_available'] = False
    
    return validation

def print_system_info():
    """Print comprehensive system information."""
    info = get_system_info()
    validation = validate_environment()
    
    print("=" * 60)
    print("DEEP MOMENTUM TRADING SYSTEM")
    print("=" * 60)
    print(f"Version: {__version__}")
    print(f"Python: {info['python_version']}")
    print(f"Platform: {info['platform']}")
    print(f"Architecture: {info['machine']}")
    
    if info['is_arm64']:
        print("✅ ARM64 optimizations: ENABLED")
        if info['is_gh200']:
            print("🚀 GH200 Grace Hopper: DETECTED")
    else:
        print("⚠️  ARM64 optimizations: DISABLED")
    
    if info.get('torch_available', True):
        print(f"PyTorch: {info.get('torch_version', 'Unknown')}")
        if info.get('cuda_available', False):
            print(f"CUDA: {info.get('cuda_version', 'Unknown')} ({info.get('gpu_count', 0)} GPUs)")
        else:
            print("CUDA: Not available")
    
    print("\nPackage Status:")
    for key, value in validation.items():
        if key.endswith('_available'):
            package = key.replace('_available', '')
            status = "✅" if value else "❌"
            print(f"  {status} {package}")
    
    print("=" * 60)

# Package initialization
def initialize_package(config: Optional[Dict[str, Any]] = None):
    """Initialize the Deep Momentum Trading System package."""
    config = config or {}
    
    # Set up logging level
    log_level = config.get('log_level', 'INFO')
    setup_logging(log_level)
    
    # Print system info if requested
    if config.get('show_system_info', False):
        print_system_info()
    
    # Validate environment if requested
    if config.get('validate_environment', True):
        validation = validate_environment()
        failed_checks = [k for k, v in validation.items() if not v and k.endswith('_available')]
        
        if failed_checks:
            missing_packages = [k.replace('_available', '') for k in failed_checks]
            logger.warning(f"Missing packages: {', '.join(missing_packages)}")
    
    # ARM64 specific initialization
    if SYSTEM_INFO['is_arm64']:
        logger.info("Initializing ARM64 optimizations...")
        
        # Set environment variables for optimal performance
        import os
        os.environ.setdefault('OMP_NUM_THREADS', str(min(8, os.cpu_count() or 1)))
        os.environ.setdefault('MKL_NUM_THREADS', str(min(8, os.cpu_count() or 1)))
        
        if SYSTEM_INFO['is_gh200']:
            logger.info("Applying GH200 Grace Hopper optimizations...")
            # GH200-specific optimizations
            os.environ.setdefault('CUDA_DEVICE_ORDER', 'PCI_BUS_ID')
            os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'max_split_size_mb:512')

# Auto-initialize on import
try:
    initialize_package()
except Exception as e:
    logger.warning(f"Package initialization warning: {e}")

# Export public API
__all__ = [
    # Metadata
    '__version__',
    '__author__',
    '__email__',
    '__license__',
    '__description__',
    
    # System information
    'SYSTEM_INFO',
    'get_system_info',
    'validate_environment',
    'print_system_info',
    
    # Module accessors
    'get_models',
    'get_data',
    'get_trading',
    'get_risk',
    'get_training',
    'get_monitoring',
    'get_communication',
    'get_storage',
    'get_infrastructure',
    'get_utils',
    
    # Utilities
    'setup_logging',
    'initialize_package',
    'detect_arm64',
    'detect_gh200',
]

# Deprecation warnings for old imports
def __getattr__(name: str):
    """Handle deprecated imports with warnings."""
    deprecated_imports = {
        'models': 'Use get_models() instead',
        'data': 'Use get_data() instead',
        'trading': 'Use get_trading() instead',
        'risk': 'Use get_risk() instead',
        'training': 'Use get_training() instead',
        'monitoring': 'Use get_monitoring() instead',
        'communication': 'Use get_communication() instead',
        'storage': 'Use get_storage() instead',
        'infrastructure': 'Use get_infrastructure() instead',
        'utils': 'Use get_utils() instead',
    }
    
    if name in deprecated_imports:
        warnings.warn(
            f"Direct import of '{name}' is deprecated. {deprecated_imports[name]}",
            DeprecationWarning,
            stacklevel=2
        )
        return globals()[f'get_{name}']()
    
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")