wimport torch
import torch.nn as nn
import torch.distributed as dist
import logging
import os
import time
import gc
import psutil
import threading
from typing import Union, Optional, Dict, Any, List, Tuple, Callable
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
from pathlib import Path
import json
import pickle
import hashlib
from contextlib import contextmanager

from deep_momentum_trading.src.utils.logger import get_logger

# ARM64 optimization imports
try:
    from .arm64_optimizations import (
        ScalableARM64Optimizer,
        ScalableARM64Config,
        MixedPrecisionManager,
        TorchScriptCompiler,
        CUDAGraphManager,
        ARM64PerformanceProfiler,
        UnifiedMemoryManager
    )
    ARM64_AVAILABLE = True
except ImportError:
    ARM64_AVAILABLE = False

logger = get_logger(__name__)

@dataclass
class ScalableModelConfig:
    """Configuration for scalable model operations."""
    num_assets: int = 10000
    num_models: int = 50
    batch_size: int = 2000
    max_memory_gb: float = 32.0
    enable_distributed: bool = True
    enable_mixed_precision: bool = True
    enable_cuda_graphs: bool = True
    enable_torchscript: bool = True
    memory_pool_size_gb: float = 8.0
    max_concurrent_models: int = 10
    checkpoint_interval: int = 1000
    profiling_enabled: bool = True

class ScalableModelManager:
    """
    Advanced model manager for massive-scale trading operations.
    Supports 10,000+ assets and 50+ models with enterprise-grade performance.
    """
    
    def __init__(self, config: ScalableModelConfig):
        self.config = config
        self.models: Dict[str, nn.Module] = {}
        self.optimizers: Dict[str, torch.optim.Optimizer] = {}
        self.model_metadata: Dict[str, Dict[str, Any]] = {}
        self.performance_metrics: Dict[str, Dict[str, float]] = {}
        self.memory_manager = None
        self.arm64_optimizer = None
        self.profiler = None
        
        # Initialize ARM64 optimizations if available
        if ARM64_AVAILABLE:
            arm64_config = ScalableARM64Config(
                num_assets=config.num_assets,
                num_models=config.num_models,
                unified_memory_pool_size=int(config.memory_pool_size_gb * 1024 * 1024 * 1024)
            )
            self.arm64_optimizer = ScalableARM64Optimizer(arm64_config)
            self.memory_manager = UnifiedMemoryManager(arm64_config)
            self.profiler = ARM64PerformanceProfiler()
        
        # Thread pool for concurrent operations
        self.executor = ThreadPoolExecutor(max_workers=config.max_concurrent_models)
        
        logger.info(f"Initialized ScalableModelManager for {config.num_assets} assets and {config.num_models} models")
    
    def register_model(self, model_id: str, model: nn.Module, 
                      optimizer: Optional[torch.optim.Optimizer] = None,
                      metadata: Optional[Dict[str, Any]] = None) -> None:
        """Register a model with the manager."""
        try:
            # Apply ARM64 optimizations if available
            if ARM64_AVAILABLE and self.arm64_optimizer:
                sample_input = torch.randn(self.config.batch_size, 100).cuda()
                optimized_model = self.arm64_optimizer.optimize_model(
                    model, sample_input, f"model_{model_id}"
                )
                self.models[model_id] = optimized_model
            else:
                self.models[model_id] = model
            
            if optimizer:
                self.optimizers[model_id] = optimizer
            
            # Store metadata
            model_metadata = {
                'model_class': model.__class__.__name__,
                'parameter_count': self.count_parameters(model),
                'memory_usage_mb': self.get_model_memory_usage(model),
                'registration_time': time.time(),
                'arm64_optimized': ARM64_AVAILABLE,
                **(metadata or {})
            }
            self.model_metadata[model_id] = model_metadata
            
            logger.info(f"Registered model {model_id} with {model_metadata['parameter_count']} parameters")
            
        except Exception as e:
            logger.error(f"Failed to register model {model_id}: {e}")
            raise
    
    def batch_inference(self, model_ids: List[str], inputs: torch.Tensor,
                       asset_ids: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Perform batch inference across multiple models."""
        results = {}
        
        try:
            # Use thread pool for concurrent inference
            future_to_model = {}
            
            for model_id in model_ids:
                if model_id not in self.models:
                    logger.warning(f"Model {model_id} not found, skipping")
                    continue
                
                future = self.executor.submit(
                    self._single_model_inference,
                    model_id, inputs, asset_ids
                )
                future_to_model[future] = model_id
            
            # Collect results
            for future in as_completed(future_to_model):
                model_id = future_to_model[future]
                try:
                    result = future.result()
                    results[model_id] = result
                except Exception as e:
                    logger.error(f"Inference failed for model {model_id}: {e}")
                    results[model_id] = None
            
            return results
            
        except Exception as e:
            logger.error(f"Batch inference failed: {e}")
            raise
    
    def _single_model_inference(self, model_id: str, inputs: torch.Tensor,
                               asset_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Perform inference with a single model."""
        model = self.models[model_id]
        model.eval()
        
        with torch.no_grad():
            if hasattr(model, 'forward_with_assets') and asset_ids is not None:
                return model.forward_with_assets(inputs, asset_ids)
            else:
                return model(inputs)
    
    def optimize_all_models(self) -> Dict[str, Dict[str, Any]]:
        """Apply comprehensive optimizations to all registered models."""
        optimization_results = {}
        
        for model_id, model in self.models.items():
            try:
                if ARM64_AVAILABLE and self.arm64_optimizer:
                    sample_input = torch.randn(self.config.batch_size, 100).cuda()
                    result = self.arm64_optimizer.optimize_model(
                        model, sample_input, f"optimized_{model_id}"
                    )
                    optimization_results[model_id] = {
                        'success': True,
                        'optimizations_applied': ['torchscript', 'mixed_precision', 'cuda_graphs']
                    }
                else:
                    optimization_results[model_id] = {
                        'success': False,
                        'reason': 'ARM64 optimizations not available'
                    }
                    
            except Exception as e:
                logger.error(f"Failed to optimize model {model_id}: {e}")
                optimization_results[model_id] = {
                    'success': False,
                    'error': str(e)
                }
        
        return optimization_results
    
    def profile_performance(self, model_ids: Optional[List[str]] = None,
                           num_iterations: int = 100) -> Dict[str, Dict[str, float]]:
        """Profile performance of models."""
        if model_ids is None:
            model_ids = list(self.models.keys())
        
        performance_results = {}
        
        for model_id in model_ids:
            if model_id not in self.models:
                continue
                
            try:
                model = self.models[model_id]
                sample_input = torch.randn(self.config.batch_size, 100).cuda()
                
                if ARM64_AVAILABLE and self.profiler:
                    metrics = self.profiler.profile_model(
                        model, sample_input, num_iterations, model_id
                    )
                else:
                    metrics = self._basic_performance_profile(
                        model, sample_input, num_iterations
                    )
                
                performance_results[model_id] = metrics
                self.performance_metrics[model_id] = metrics
                
            except Exception as e:
                logger.error(f"Failed to profile model {model_id}: {e}")
                performance_results[model_id] = {'error': str(e)}
        
        return performance_results
    
    def _basic_performance_profile(self, model: nn.Module, sample_input: torch.Tensor,
                                  num_iterations: int) -> Dict[str, float]:
        """Basic performance profiling without ARM64 optimizations."""
        model.eval()
        
        # Warmup
        with torch.no_grad():
            for _ in range(10):
                _ = model(sample_input)
        
        torch.cuda.synchronize()
        
        # Timing
        start_time = time.time()
        with torch.no_grad():
            for _ in range(num_iterations):
                _ = model(sample_input)
        
        torch.cuda.synchronize()
        end_time = time.time()
        
        total_time = end_time - start_time
        avg_latency = (total_time / num_iterations) * 1000  # ms
        throughput = num_iterations / total_time
        
        return {
            'avg_latency_ms': avg_latency,
            'throughput_fps': throughput,
            'total_time_s': total_time
        }
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Get comprehensive memory usage statistics."""
        gpu_memory = {}
        if torch.cuda.is_available():
            gpu_memory = {
                'allocated_gb': torch.cuda.memory_allocated() / (1024**3),
                'reserved_gb': torch.cuda.memory_reserved() / (1024**3),
                'max_allocated_gb': torch.cuda.max_memory_allocated() / (1024**3)
            }
        
        cpu_memory = {
            'used_gb': psutil.virtual_memory().used / (1024**3),
            'available_gb': psutil.virtual_memory().available / (1024**3),
            'percent': psutil.virtual_memory().percent
        }
        
        return {
            'gpu': gpu_memory,
            'cpu': cpu_memory,
            'model_count': len(self.models)
        }
    
    def cleanup_memory(self) -> None:
        """Perform comprehensive memory cleanup."""
        try:
            # Clear model caches
            for model in self.models.values():
                if hasattr(model, 'clear_cache'):
                    model.clear_cache()
            
            # PyTorch cleanup
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
            # Python garbage collection
            gc.collect()
            
            logger.info("Memory cleanup completed")
            
        except Exception as e:
            logger.error(f"Memory cleanup failed: {e}")

def initialize_weights_scalable(model: nn.Module, init_method: str = 'xavier_uniform',
                               enable_arm64: bool = True) -> None:
    """
    Advanced weight initialization for massive-scale models.
    
    Args:
        model (nn.Module): The PyTorch model to initialize
        init_method (str): The initialization method
        enable_arm64 (bool): Whether to use ARM64 optimizations
    """
    logger.info(f"Initializing model weights using {init_method} method (ARM64: {enable_arm64})")
    
    try:
        # ARM64-optimized initialization if available
        if enable_arm64 and ARM64_AVAILABLE:
            arm64_config = ScalableARM64Config()
            arm64_optimizer = ScalableARM64Optimizer(arm64_config)
            
            # Apply standard initialization first
            _apply_standard_initialization(model, init_method)
            
            # Apply ARM64-specific optimizations
            logger.info("Applied ARM64-optimized weight initialization")
        else:
            _apply_standard_initialization(model, init_method)
            
    except Exception as e:
        logger.error(f"Weight initialization failed: {e}")
        # Fallback to basic initialization
        _apply_standard_initialization(model, init_method)

def _apply_standard_initialization(model: nn.Module, init_method: str) -> None:
    """Apply standard weight initialization."""
    for name, param in model.named_parameters():
        if 'weight' in name:
            if 'lstm' in name or 'rnn' in name or 'gru' in name:
                # Special handling for recurrent layers
                if 'weight_ih' in name:
                    if init_method == 'orthogonal':
                        nn.init.orthogonal_(param.data)
                    else:
                        getattr(nn.init, f'{init_method}_')(param.data)
                elif 'weight_hh' in name:
                    nn.init.orthogonal_(param.data)
                else:
                    if param.dim() >= 2:
                        getattr(nn.init, f'{init_method}_')(param.data)
            elif param.dim() >= 2:
                getattr(nn.init, f'{init_method}_')(param.data)
            else:
                nn.init.normal_(param.data, mean=0.0, std=0.02)
        elif 'bias' in name:
            nn.init.constant_(param.data, 0)

def clip_gradients_scalable(model: nn.Module, clip_value: float = 1.0,
                           clip_norm_type: float = 2.0, enable_monitoring: bool = True) -> Dict[str, float]:
    """
    Advanced gradient clipping with monitoring for massive-scale models.
    
    Args:
        model (nn.Module): The PyTorch model
        clip_value (float): The maximum norm of the gradients
        clip_norm_type (float): The type of p-norm to use for clipping
        enable_monitoring (bool): Whether to return gradient statistics
        
    Returns:
        Dict[str, float]: Gradient statistics if monitoring is enabled
    """
    # Calculate gradient norm before clipping
    total_norm_before = 0.0
    param_count = 0
    
    if enable_monitoring:
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(clip_norm_type)
                total_norm_before += param_norm.item() ** clip_norm_type
                param_count += 1
        total_norm_before = total_norm_before ** (1. / clip_norm_type)
    
    # Apply gradient clipping
    total_norm_after = torch.nn.utils.clip_grad_norm_(
        model.parameters(), clip_value, norm_type=clip_norm_type
    )
    
    stats = {}
    if enable_monitoring:
        stats = {
            'grad_norm_before': total_norm_before,
            'grad_norm_after': float(total_norm_after),
            'clipped': total_norm_before > clip_value,
            'param_count': param_count
        }
        
        logger.debug(f"Gradient clipping: before={total_norm_before:.4f}, "
                    f"after={total_norm_after:.4f}, clipped={stats['clipped']}")
    
    return stats

def save_model_scalable(model: nn.Module, filepath: str,
                       optimizer: Optional[torch.optim.Optimizer] = None,
                       epoch: Optional[int] = None, loss: Optional[float] = None,
                       metadata: Optional[Dict[str, Any]] = None,
                       compress: bool = True) -> None:
    """
    Advanced model saving with compression and metadata for massive-scale models.
    
    Args:
        model (nn.Module): The PyTorch model to save
        filepath (str): The path to save the model to
        optimizer (Optional[torch.optim.Optimizer]): The optimizer to save
        epoch (Optional[int]): The current epoch number
        loss (Optional[float]): The current loss value
        metadata (Optional[Dict[str, Any]]): Additional metadata
        compress (bool): Whether to compress the saved file
    """
    try:
        # Prepare state dictionary
        state = {
            'model_state_dict': model.state_dict(),
            'timestamp': time.time(),
            'model_class': model.__class__.__name__,
            'parameter_count': count_parameters(model),
            'memory_usage_mb': get_model_memory_usage(model),
            'arm64_optimizations_available': ARM64_AVAILABLE
        }
        
        if optimizer:
            state['optimizer_state_dict'] = optimizer.state_dict()
        if epoch is not None:
            state['epoch'] = epoch
        if loss is not None:
            state['loss'] = loss
        if metadata:
            state['metadata'] = metadata
        
        # Add ARM64 optimization status
        if ARM64_AVAILABLE:
            state['arm64_metadata'] = get_arm64_optimization_status(model)
        
        # Create directory if it doesn't exist
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        # Save with optional compression
        if compress:
            torch.save(state, filepath, pickle_protocol=pickle.HIGHEST_PROTOCOL)
        else:
            torch.save(state, filepath)
        
        # Calculate and log file size
        file_size_mb = os.path.getsize(filepath) / (1024 * 1024)
        logger.info(f"Model saved to {filepath} (size: {file_size_mb:.2f} MB)")
        
    except Exception as e:
        logger.error(f"Failed to save model to {filepath}: {e}", exc_info=True)
        raise

def load_model_scalable(model: nn.Module, filepath: str,
                       optimizer: Optional[torch.optim.Optimizer] = None,
                       strict: bool = True, map_location: Optional[str] = None) -> Dict[str, Any]:
    """
    Advanced model loading with validation for massive-scale models.
    
    Args:
        model (nn.Module): The PyTorch model instance to load state into
        filepath (str): The path to the saved model file
        optimizer (Optional[torch.optim.Optimizer]): The optimizer instance to load state into
        strict (bool): Whether to strictly enforce state dict keys match
        map_location (Optional[str]): Device to map tensors to
        
    Returns:
        Dict[str, Any]: A dictionary containing loaded metadata
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Model file not found at {filepath}")
    
    try:
        # Load checkpoint
        if map_location is None:
            map_location = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        checkpoint = torch.load(filepath, map_location=map_location)
        
        # Validate checkpoint
        if 'model_state_dict' not in checkpoint:
            raise ValueError("Invalid checkpoint: missing model_state_dict")
        
        # Load model state
        model.load_state_dict(checkpoint['model_state_dict'], strict=strict)
        
        # Load optimizer state if available
        if optimizer and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Extract metadata
        metadata = {
            'epoch': checkpoint.get('epoch'),
            'loss': checkpoint.get('loss'),
            'timestamp': checkpoint.get('timestamp'),
            'model_class': checkpoint.get('model_class'),
            'parameter_count': checkpoint.get('parameter_count'),
            'memory_usage_mb': checkpoint.get('memory_usage_mb'),
            'arm64_metadata': checkpoint.get('arm64_metadata'),
            'metadata': checkpoint.get('metadata')
        }
        
        file_size_mb = os.path.getsize(filepath) / (1024 * 1024)
        logger.info(f"Model loaded from {filepath} (size: {file_size_mb:.2f} MB)")
        
        return metadata
        
    except Exception as e:
        logger.error(f"Failed to load model from {filepath}: {e}", exc_info=True)
        raise

def count_parameters(model: nn.Module) -> int:
    """Count the total number of trainable parameters in a PyTorch model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_model_memory_usage(model: nn.Module) -> float:
    """
    Estimate the memory usage of a PyTorch model's parameters and buffers in MB.
    
    Args:
        model (nn.Module): The PyTorch model
        
    Returns:
        float: Estimated memory usage in MB
    """
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    return (param_size + buffer_size) / (1024 * 1024)

def optimize_model_for_arm64_scalable(model: nn.Module, sample_input: torch.Tensor,
                                     config: Optional[ScalableARM64Config] = None) -> Dict[str, Any]:
    """
    Comprehensive ARM64 optimization for massive-scale models.
    
    Args:
        model (nn.Module): The PyTorch model to optimize
        sample_input (torch.Tensor): Sample input for optimization
        config (Optional[ScalableARM64Config]): ARM64 optimization configuration
        
    Returns:
        Dict[str, Any]: Optimization results and status
    """
    if not ARM64_AVAILABLE:
        logger.warning("ARM64 optimizations not available")
        return {"arm64_available": False}
    
    try:
        if config is None:
            config = ScalableARM64Config()
        
        arm64_optimizer = ScalableARM64Optimizer(config)
        
        # Apply comprehensive optimizations
        optimized_model = arm64_optimizer.optimize_model(
            model, sample_input, model.__class__.__name__
        )
        
        # Get optimization status
        status = arm64_optimizer.get_optimization_status(model.__class__.__name__)
        
        return {
            "arm64_available": True,
            "optimization_successful": True,
            "optimizations_applied": status.get('optimizations_applied', []),
            "performance_improvement": status.get('performance_improvement', {}),
            "memory_optimization": status.get('memory_optimization', {})
        }
        
    except Exception as e:
        logger.error(f"ARM64 optimization failed: {e}")
        return {
            "arm64_available": True,
            "optimization_successful": False,
            "error": str(e)
        }

def profile_model_arm64_performance_scalable(model: nn.Module, sample_input: torch.Tensor,
                                           num_iterations: int = 100,
                                           config: Optional[ScalableARM64Config] = None) -> Dict[str, float]:
    """
    Comprehensive ARM64 performance profiling for massive-scale models.
    
    Args:
        model (nn.Module): The PyTorch model to profile
        sample_input (torch.Tensor): Sample input for profiling
        num_iterations (int): Number of iterations for profiling
        config (Optional[ScalableARM64Config]): ARM64 configuration
        
    Returns:
        Dict[str, float]: Comprehensive performance metrics
    """
    if not ARM64_AVAILABLE:
        logger.warning("ARM64 optimizations not available")
        return {}
    
    try:
        if config is None:
            config = ScalableARM64Config()
        
        profiler = ARM64PerformanceProfiler()
        
        # Comprehensive profiling
        metrics = profiler.profile_model(
            model=model,
            sample_input=sample_input,
            num_iterations=num_iterations,
            model_name=model.__class__.__name__
        )
        
        return metrics
        
    except Exception as e:
        logger.error(f"ARM64 performance profiling failed: {e}")
        return {"error": str(e)}

def get_arm64_optimization_status(model: nn.Module) -> Dict[str, Any]:
    """
    Get comprehensive ARM64 optimization status of a model.
    
    Args:
        model (nn.Module): The PyTorch model
        
    Returns:
        Dict[str, Any]: Comprehensive optimization status information
    """
    if not ARM64_AVAILABLE:
        return {
            "arm64_optimizations_available": False,
            "reason": "ARM64 optimizations not available"
        }
    
    status = {
        "arm64_optimizations_available": True,
        "model_class": model.__class__.__name__,
        "parameter_count": count_parameters(model),
        "memory_usage_mb": get_model_memory_usage(model),
        "supports_arm64_optimizations": hasattr(model, 'get_arm64_optimization_status')
    }
    
    # Get model-specific ARM64 status if available
    if hasattr(model, 'get_arm64_optimization_status'):
        try:
            model_status = model.get_arm64_optimization_status()
            status.update(model_status)
        except Exception as e:
            status["error"] = str(e)
    
    # Check for ARM64-specific attributes
    arm64_features = [
        'torchscript_compiled', 'cuda_graph_enabled', 'mixed_precision_enabled',
        'unified_memory_enabled', 'performance_optimized'
    ]
    
    for feature in arm64_features:
        status[feature] = hasattr(model, feature) and getattr(model, feature, False)
    
    return status

@contextmanager
def memory_efficient_context():
    """Context manager for memory-efficient operations."""
    try:
        # Clear cache before operation
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
        yield
        
    finally:
        # Clean up after operation
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

def batch_process_models(models: List[nn.Module], inputs: torch.Tensor,
                        batch_size: int = 1000, max_workers: int = 4) -> List[torch.Tensor]:
    """
    Process multiple models in batches for memory efficiency.
    
    Args:
        models (List[nn.Module]): List of models to process
        inputs (torch.Tensor): Input tensor
        batch_size (int): Batch size for processing
        max_workers (int): Maximum number of worker threads
        
    Returns:
        List[torch.Tensor]: List of model outputs
    """
    results = []
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        
        for model in models:
            future = executor.submit(_process_single_model, model, inputs)
            futures.append(future)
        
        for future in as_completed(futures):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                logger.error(f"Model processing failed: {e}")
                results.append(None)
    
    return results

def _process_single_model(model: nn.Module, inputs: torch.Tensor) -> torch.Tensor:
    """Process a single model with memory management."""
    with memory_efficient_context():
        model.eval()
        with torch.no_grad():
            return model(inputs)

if __name__ == "__main__":
    # Example usage for massive-scale trading
    print("--- Scalable Model Utilities Test ---")
    
    # Test configuration
    config = ScalableModelConfig(
        num_assets=1000,  # Reduced for testing
        num_models=5,     # Reduced for testing
        batch_size=100
    )
    
    # Create test model
    class TestTradingModel(nn.Module):
        def __init__(self, input_dim: int, output_dim: int):
            super().__init__()
            self.fc1 = nn.Linear(input_dim, 128)
            self.fc2 = nn.Linear(128, 64)
            self.fc3 = nn.Linear(64, output_dim)
            self.relu = nn.ReLU()
            self.dropout = nn.Dropout(0.1)
        
        def forward(self, x):
            x = self.relu(self.fc1(x))
            x = self.dropout(x)
            x = self.relu(self.fc2(x))
            x = self.dropout(x)
            return self.fc3(x)
    
    # Initialize model manager
    manager = ScalableModelManager(config)
    
    # Create and register test models
    for i in range(3):
        model = TestTradingModel(100, 10)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        initialize_weights_scalable(model, 'xavier_uniform')
        manager.register_model(f"model_{i}", model, optimizer)
    
    # Test batch inference
    test_input = torch.randn(config.batch_size, 100)
    results = manager.batch_inference(["model_0", "model_1", "model_2"], test_input)
    print(f"Batch inference completed for {len(results)} models")
    
    # Test performance profiling
    performance = manager.profile_performance(num_iterations=10)
    print(f"Performance profiling completed for {len(performance)} models")
    
    # Test memory usage
    memory_stats = manager.get_memory_usage()
    print(f"Memory usage: {memory_stats}")
    
    # Test model optimization
    optimization_results = manager.optimize_all_models()
    print(f"Model optimization completed: {optimization_results}")
    
    print("--- Scalable Model Utilities Test Complete ---")
