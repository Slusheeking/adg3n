"""
Deep Momentum Trading Models Package

ARM64-optimized neural network models for high-frequency trading on NVIDIA GH200 platform.
Includes TorchScript compilation, CUDA graph optimization, and distributed hyperparameter tuning.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, List, Union

# Core model implementations
from .deep_momentum_lstm import DeepMomentumLSTM
from .transformer_momentum import TransformerMomentumNetwork
from .ensemble_system import EnsembleMomentumSystem, MarketRegimeDetector, ModelPerformanceTracker
from .meta_learner import MetaLearningOptimizer, OnlineMetaLearner, HierarchicalMetaLearner

# Loss functions
from .loss_functions import (
    SharpeOptimizedLoss,
    TurnoverRegularization, 
    RiskAdjustedLoss,
    InformationRatioLoss,
    CalmarRatioLoss,
    SortinoRatioLoss,
    CombinedLoss
)

# Model management and utilities
from .model_registry import ModelRegistry, ModelFactory, ModelConfig, ModelMetadata, global_registry
from .model_utils import (
    initialize_weights,
    clip_gradients,
    save_model,
    load_model,
    count_parameters,
    get_model_memory_usage
)

# ARM64 optimization utilities (will be created)
try:
    from .arm64_optimizations import (
        ARM64ModelOptimizer,
        TorchScriptCompiler,
        CUDAGraphManager,
        MixedPrecisionManager,
        ZeroCopyMemoryManager,
        DistributedHyperparameterOptimizer,
        ARM64PerformanceProfiler
    )
    ARM64_OPTIMIZATIONS_AVAILABLE = True
except ImportError:
    ARM64_OPTIMIZATIONS_AVAILABLE = False

__version__ = "2.0.0"
__author__ = "Deep Momentum Trading Team"

# Configure logging for the models package
from ..utils.logger import get_logger
logger = get_logger(__name__)

# ARM64 platform detection and optimization
def is_arm64_platform() -> bool:
    """Check if running on ARM64 platform"""
    import platform
    return platform.machine().lower() in ['arm64', 'aarch64']

def is_gh200_available() -> bool:
    """Check if NVIDIA GH200 Grace Hopper is available"""
    if not torch.cuda.is_available():
        return False
    
    try:
        # Check for Grace Hopper specific capabilities
        device_props = torch.cuda.get_device_properties(0)
        # GH200 has specific compute capability and unified memory features
        return (device_props.major >= 9 and 
                hasattr(torch.cuda, 'get_unified_memory_size'))
    except:
        return False

# Global optimization settings
ARM64_OPTIMIZED = is_arm64_platform()
GH200_AVAILABLE = is_gh200_available()
TORCHSCRIPT_ENABLED = True
CUDA_GRAPHS_ENABLED = GH200_AVAILABLE
MIXED_PRECISION_ENABLED = True
ZERO_COPY_ENABLED = GH200_AVAILABLE

# Model factory with ARM64 optimizations
def create_optimized_model(model_name: str, 
                          config: Optional[Dict[str, Any]] = None,
                          enable_torchscript: bool = TORCHSCRIPT_ENABLED,
                          enable_cuda_graphs: bool = CUDA_GRAPHS_ENABLED,
                          enable_mixed_precision: bool = MIXED_PRECISION_ENABLED) -> nn.Module:
    """
    Create an ARM64-optimized model instance with advanced optimizations.
    
    Args:
        model_name: Name of the model to create
        config: Optional model configuration
        enable_torchscript: Enable TorchScript compilation
        enable_cuda_graphs: Enable CUDA graph optimization
        enable_mixed_precision: Enable mixed precision training
        
    Returns:
        Optimized model instance
    """
    # Create base model
    if config:
        model_config = ModelConfig(**config)
        model = ModelFactory.create_model(model_config)
    else:
        model = global_registry.create_model(model_name)
    
    # Apply ARM64 optimizations if available
    if ARM64_OPTIMIZED and ARM64_OPTIMIZATIONS_AVAILABLE:
        optimizer = ARM64ModelOptimizer()
        model = optimizer.optimize_model(
            model,
            enable_torchscript=enable_torchscript,
            enable_cuda_graphs=enable_cuda_graphs,
            enable_mixed_precision=enable_mixed_precision
        )
        
        logger.info(f"Applied ARM64 optimizations to {model_name}")
    
    return model

# Export all public components
__all__ = [
    # Core models
    'DeepMomentumLSTM',
    'TransformerMomentumNetwork', 
    'EnsembleMomentumSystem',
    'MarketRegimeDetector',
    'ModelPerformanceTracker',
    
    # Meta-learning
    'MetaLearningOptimizer',
    'OnlineMetaLearner',
    'HierarchicalMetaLearner',
    
    # Loss functions
    'SharpeOptimizedLoss',
    'TurnoverRegularization',
    'RiskAdjustedLoss', 
    'InformationRatioLoss',
    'CalmarRatioLoss',
    'SortinoRatioLoss',
    'CombinedLoss',
    
    # Model management
    'ModelRegistry',
    'ModelFactory',
    'ModelConfig',
    'ModelMetadata',
    'global_registry',
    
    # Utilities
    'initialize_weights',
    'clip_gradients',
    'save_model',
    'load_model',
    'count_parameters',
    'get_model_memory_usage',
    
    # Factory functions
    'create_optimized_model',
    
    # Platform detection
    'is_arm64_platform',
    'is_gh200_available',
    
    # Global settings
    'ARM64_OPTIMIZED',
    'GH200_AVAILABLE',
    'TORCHSCRIPT_ENABLED',
    'CUDA_GRAPHS_ENABLED',
    'MIXED_PRECISION_ENABLED',
    'ZERO_COPY_ENABLED'
]

# Conditionally add ARM64 optimization exports if available
if ARM64_OPTIMIZATIONS_AVAILABLE:
    __all__.extend([
        'ARM64ModelOptimizer',
        'TorchScriptCompiler',
        'CUDAGraphManager',
        'MixedPrecisionManager',
        'ZeroCopyMemoryManager',
        'DistributedHyperparameterOptimizer',
        'ARM64PerformanceProfiler'
    ])

# Initialize ARM64 optimizations on import
if ARM64_OPTIMIZED:
    logger.info("ARM64 platform detected - enabling optimizations")
    if GH200_AVAILABLE:
        logger.info("NVIDIA GH200 Grace Hopper detected - enabling advanced features")
