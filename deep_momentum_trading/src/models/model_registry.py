import torch
import torch.nn as nn
from typing import Dict, List, Optional, Any, Callable, Type, Union, Tuple
import json
import os
import time
import logging
from pathlib import Path
from dataclasses import dataclass, asdict, field
import inspect
from collections import defaultdict, deque
import copy
import hashlib
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import pickle
import asyncio
from enum import Enum
import uuid
import numpy as np

from deep_momentum_trading.src.utils.logger import get_logger
from deep_momentum_trading.config.settings import config_manager

# Import all model classes
from deep_momentum_trading.src.models.deep_momentum_lstm import DeepMomentumLSTM
from deep_momentum_trading.src.models.transformer_momentum import TransformerMomentumNetwork
from deep_momentum_trading.src.models.ensemble_system import EnsembleMomentumSystem
from deep_momentum_trading.src.models.meta_learner import (
    ScalableOnlineMetaLearner, OnlineMetaLearner, 
    ScalableMetaLearningConfig, create_production_meta_learner
)

# Import loss functions
from deep_momentum_trading.src.models.loss_functions import (
    ScalableCombinedLoss, ScalableSharpeOptimizedLoss, ScalableLossConfig,
    create_production_loss_function, SharpeOptimizedLoss, TurnoverRegularization, 
    RiskAdjustedLoss, CombinedLoss
)

# ARM64 optimization imports
try:
    from .arm64_optimizations import (
        ARM64ModelOptimizer,
        ARM64OptimizationConfig,
        MixedPrecisionManager,
        TorchScriptCompiler,
        CUDAGraphManager,
        ARM64PerformanceProfiler
    )
    ARM64_AVAILABLE = True
except ImportError:
    ARM64_AVAILABLE = False

logger = get_logger(__name__)

# ===== SCALABLE MODEL REGISTRY ENUMS AND CONFIGS =====

class ModelStatus(Enum):
    """Model status enumeration"""
    REGISTERED = "registered"
    TRAINING = "training"
    TRAINED = "trained"
    DEPLOYED = "deployed"
    ARCHIVED = "archived"
    FAILED = "failed"

class OptimizationLevel(Enum):
    """Optimization level enumeration"""
    NONE = "none"
    BASIC = "basic"
    ADVANCED = "advanced"
    PRODUCTION = "production"

@dataclass
class ScalableModelConfig:
    """Enhanced configuration for massive-scale model management"""
    model_type: str
    parameters: Dict[str, Any]
    description: str = ""
    version: str = "1.0"
    created_at: float = field(default_factory=time.time)
    tags: List[str] = field(default_factory=list)
    
    # Scalability parameters
    max_assets: int = 10000
    batch_size: int = 1000
    target_sharpe: float = 4.0
    max_drawdown_target: float = 0.05
    
    # Optimization settings
    optimization_level: OptimizationLevel = OptimizationLevel.BASIC
    enable_mixed_precision: bool = True
    enable_gradient_checkpointing: bool = True
    enable_model_parallelism: bool = True
    
    # Performance requirements
    max_inference_latency_ms: float = 100.0
    min_throughput_samples_per_sec: float = 1000.0
    memory_limit_gb: float = 32.0
    
    # Deployment settings
    deployment_targets: List[str] = field(default_factory=lambda: ["cuda", "cpu"])
    auto_scaling_enabled: bool = True
    health_check_interval_sec: int = 60

@dataclass
class ScalableModelMetadata:
    """Enhanced metadata for scalable model management"""
    name: str
    config: ScalableModelConfig
    model_path: Optional[str] = None
    checkpoint_path: Optional[str] = None
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    last_updated: float = field(default_factory=time.time)
    model_size_mb: float = 0.0
    parameter_count: int = 0
    training_history: List[Dict] = field(default_factory=list)
    
    # Enhanced metadata
    model_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    status: ModelStatus = ModelStatus.REGISTERED
    deployment_info: Dict[str, Any] = field(default_factory=dict)
    optimization_info: Dict[str, Any] = field(default_factory=dict)
    health_metrics: Dict[str, float] = field(default_factory=dict)
    
    # Performance tracking
    inference_stats: Dict[str, float] = field(default_factory=dict)
    memory_usage: Dict[str, float] = field(default_factory=dict)
    error_count: int = 0
    success_count: int = 0
    
    # Versioning
    parent_model_id: Optional[str] = None
    child_model_ids: List[str] = field(default_factory=list)
    model_hash: Optional[str] = None

# ===== SCALABLE MODEL FACTORY =====

class ScalableModelFactory:
    """Enhanced factory for creating scalable model instances"""
    
    _model_constructors: Dict[str, Type[nn.Module]] = {
        'deep_momentum_lstm': DeepMomentumLSTM,
        'transformer_momentum': TransformerMomentumNetwork,
        'ensemble_momentum': EnsembleMomentumSystem,
        'scalable_meta_learner': ScalableOnlineMetaLearner,
        'online_meta_learner': OnlineMetaLearner
    }
    
    _loss_constructors: Dict[str, Type[nn.Module]] = {
        'scalable_combined': ScalableCombinedLoss,
        'scalable_sharpe': ScalableSharpeOptimizedLoss,
        'sharpe_optimized': SharpeOptimizedLoss,
        'turnover_regularization': TurnoverRegularization,
        'risk_adjusted': RiskAdjustedLoss,
        'combined_loss': CombinedLoss
    }
    
    _optimization_configs: Dict[str, Dict[str, Any]] = {
        'production': {
            'use_mixed_precision': True,
            'use_gradient_checkpointing': True,
            'use_model_parallelism': True,
            'compile_for_inference': True,
            'enable_cuda_graphs': True
        },
        'development': {
            'use_mixed_precision': False,
            'use_gradient_checkpointing': False,
            'use_model_parallelism': False,
            'compile_for_inference': False,
            'enable_cuda_graphs': False
        }
    }
    
    def __init__(self):
        self._model_cache = {}
        self._lock = threading.RLock()
        
    @classmethod
    def register_model(cls, name: str, model_class: Type[nn.Module]):
        """Register a new model class"""
        cls._model_constructors[name] = model_class
        logger.info(f"Registered model: {name} -> {model_class.__name__}")
    
    @classmethod
    def register_loss(cls, name: str, loss_class: Type[nn.Module]):
        """Register a new loss function class"""
        cls._loss_constructors[name] = loss_class
        logger.info(f"Registered loss function: {name} -> {loss_class.__name__}")
    
    def create_model(self, config: ScalableModelConfig, device: str = 'cuda') -> nn.Module:
        """Create optimized model instance"""
        model_type = config.model_type.lower()
        
        if model_type not in self._model_constructors:
            raise ValueError(f"Unknown model type: {model_type}")
        
        model_class = self._model_constructors[model_type]
        
        try:
            # Validate parameters
            self._validate_parameters(model_class, config.parameters)
            
            # Create base model
            if model_type == 'scalable_meta_learner':
                # Special handling for meta-learner
                meta_config = ScalableMetaLearningConfig(
                    num_models=config.parameters.get('num_models', 50),
                    num_assets=config.max_assets,
                    **config.parameters
                )
                model = ScalableOnlineMetaLearner(meta_config, device)
            else:
                model = model_class(**config.parameters)
            
            # Apply optimizations based on configuration
            model = self._apply_optimizations(model, config, device)
            
            logger.info(f"Created optimized model: {model_type} with {self._count_parameters(model):,} parameters")
            return model
            
        except Exception as e:
            logger.error(f"Failed to create model {model_type}: {str(e)}", exc_info=True)
            raise
    
    def create_loss_function(self, loss_type: str, config: Optional[ScalableLossConfig] = None, **kwargs) -> nn.Module:
        """Create optimized loss function"""
        loss_type = loss_type.lower()
        
        if loss_type not in self._loss_constructors:
            raise ValueError(f"Unknown loss function type: {loss_type}")
        
        if loss_type.startswith('scalable_') and config is None:
            # Create default scalable config
            config = ScalableLossConfig(
                num_assets=kwargs.get('num_assets', 10000),
                target_sharpe=kwargs.get('target_sharpe', 4.0),
                max_drawdown_target=kwargs.get('max_drawdown_target', 0.05)
            )
        
        loss_class = self._loss_constructors[loss_type]
        
        if config and hasattr(loss_class, '__init__') and 'config' in inspect.signature(loss_class.__init__).parameters:
            return loss_class(config)
        else:
            return loss_class(**kwargs)
    
    def _apply_optimizations(self, model: nn.Module, config: ScalableModelConfig, device: str) -> nn.Module:
        """Apply various optimizations to the model"""
        
        # Move to device first
        model = model.to(device)
        
        # Apply mixed precision if enabled
        if config.enable_mixed_precision and device.startswith('cuda'):
            if hasattr(model, 'enable_mixed_precision'):
                model.enable_mixed_precision()
            else:
                # Wrap with autocast-compatible forward
                original_forward = model.forward
                def mixed_precision_forward(*args, **kwargs):
                    with torch.cuda.amp.autocast():
                        return original_forward(*args, **kwargs)
                model.forward = mixed_precision_forward
        
        # Apply gradient checkpointing if enabled
        if config.enable_gradient_checkpointing:
            if hasattr(model, 'enable_gradient_checkpointing'):
                model.enable_gradient_checkpointing()
        
        # Apply model parallelism if enabled and multiple GPUs available
        if config.enable_model_parallelism and torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
        
        # Compile for inference if optimization level is production
        if config.optimization_level == OptimizationLevel.PRODUCTION and hasattr(torch, 'compile'):
            model = torch.compile(model, mode='max-autotune')
        
        return model
    
    def _validate_parameters(self, target_class: Type[Any], parameters: Dict[str, Any]):
        """Validate parameters against constructor signature"""
        sig = inspect.signature(target_class.__init__)
        all_params = {name: param for name, param in sig.parameters.items() if name != 'self'}
        
        missing_required = [
            name for name, param in all_params.items()
            if param.default == inspect.Parameter.empty and name not in parameters
        ]
        if missing_required:
            raise ValueError(f"Missing required parameters for {target_class.__name__}: {missing_required}")
    
    def _count_parameters(self, model: nn.Module) -> int:
        """Count trainable parameters"""
        if hasattr(model, 'module'):  # Handle DataParallel
            model = model.module
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    def get_available_models(self) -> List[str]:
        """Get available model types"""
        return list(self._model_constructors.keys())
    
    def get_available_losses(self) -> List[str]:
        """Get available loss function types"""
        return list(self._loss_constructors.keys())

# ===== SCALABLE MODEL REGISTRY =====

class ScalableModelRegistry:
    """Production-ready model registry for massive-scale trading systems"""
    
    def __init__(self, 
                 registry_path: str = "scalable_model_registry.json",
                 enable_arm64_optimizations: bool = True,
                 max_concurrent_operations: int = 10,
                 enable_model_versioning: bool = True):
        
        self.registry_path = Path(registry_path)
        self.models: Dict[str, ScalableModelMetadata] = {}
        self.factory = ScalableModelFactory()
        self.enable_arm64_optimizations = enable_arm64_optimizations and ARM64_AVAILABLE
        self.max_concurrent_operations = max_concurrent_operations
        self.enable_model_versioning = enable_model_versioning
        
        # Thread safety
        self._lock = threading.RLock()
        self._executor = ThreadPoolExecutor(max_workers=max_concurrent_operations)
        
        # Performance monitoring
        self._performance_history = deque(maxlen=10000)
        self._health_check_interval = 60  # seconds
        self._last_health_check = time.time()
        
        # ARM64 optimization components
        if self.enable_arm64_optimizations:
            self.arm64_optimizer = ARM64ModelOptimizer()
            self.performance_profiler = ARM64PerformanceProfiler()
            logger.info("ARM64 optimizations enabled for ScalableModelRegistry")
        
        # Model caching
        self._model_cache = {}
        self._cache_size_limit = 10  # Maximum cached models
        
        # Load existing registry
        if self.registry_path.exists():
            self.load_registry()
        
        logger.info(f"ScalableModelRegistry initialized with {len(self.models)} models")
    
    def register_model(self, 
                      name: str, 
                      config: ScalableModelConfig,
                      model_path: Optional[str] = None,
                      checkpoint_path: Optional[str] = None,
                      performance_metrics: Optional[Dict[str, float]] = None,
                      parent_model_name: Optional[str] = None) -> str:
        """Register a new model with enhanced metadata"""
        
        with self._lock:
            if name in self.models and not self.enable_model_versioning:
                raise ValueError(f"Model '{name}' already exists and versioning is disabled")
            
            # Handle versioning
            if name in self.models and self.enable_model_versioning:
                # Create new version
                version_num = len([m for m in self.models.keys() if m.startswith(f"{name}_v")]) + 1
                versioned_name = f"{name}_v{version_num}"
                logger.info(f"Creating new version: {versioned_name}")
                name = versioned_name
            
            # Create metadata
            metadata = ScalableModelMetadata(
                name=name,
                config=config,
                model_path=model_path,
                checkpoint_path=checkpoint_path,
                performance_metrics=performance_metrics or {}
            )
            
            # Set parent relationship if specified
            if parent_model_name and parent_model_name in self.models:
                metadata.parent_model_id = self.models[parent_model_name].model_id
                self.models[parent_model_name].child_model_ids.append(metadata.model_id)
            
            # Validate configuration by creating test instance
            try:
                test_model = self.factory.create_model(config)
                metadata.parameter_count = self.factory._count_parameters(test_model)
                metadata.model_size_mb = self._estimate_model_size(test_model)
                metadata.model_hash = self._compute_model_hash(test_model)
                del test_model  # Free memory
            except Exception as e:
                logger.error(f"Failed to validate model configuration for '{name}': {e}")
                raise ValueError(f"Invalid model configuration: {e}")
            
            # Store metadata
            self.models[name] = metadata
            self.save_registry()
            
            logger.info(f"Registered model: '{name}' ({metadata.parameter_count:,} parameters, {metadata.model_size_mb:.2f} MB)")
            return name
    
    def create_model(self, 
                    name: str, 
                    load_checkpoint: bool = True,
                    device: str = 'cuda',
                    use_cache: bool = True) -> nn.Module:
        """Create model instance with caching and optimization"""
        
        if name not in self.models:
            raise ValueError(f"Model '{name}' not found in registry")
        
        # Check cache first
        cache_key = f"{name}_{device}_{load_checkpoint}"
        if use_cache and cache_key in self._model_cache:
            logger.debug(f"Returning cached model: {name}")
            return self._model_cache[cache_key]
        
        metadata = self.models[name]
        
        try:
            # Create model instance
            model = self.factory.create_model(metadata.config, device)
            
            # Load checkpoint if requested
            if load_checkpoint and metadata.checkpoint_path:
                self._load_checkpoint_safe(model, metadata.checkpoint_path)
            
            # Cache model if enabled
            if use_cache:
                self._cache_model(cache_key, model)
            
            # Update usage statistics
            with self._lock:
                metadata.success_count += 1
                metadata.last_updated = time.time()
            
            return model
            
        except Exception as e:
            with self._lock:
                metadata.error_count += 1
            logger.error(f"Failed to create model '{name}': {e}")
            raise
    
    def batch_create_models(self, 
                           model_names: List[str], 
                           device: str = 'cuda',
                           max_workers: Optional[int] = None) -> Dict[str, nn.Module]:
        """Create multiple models concurrently"""
        
        max_workers = max_workers or min(len(model_names), self.max_concurrent_operations)
        results = {}
        
        def create_single_model(name):
            try:
                return name, self.create_model(name, device=device)
            except Exception as e:
                logger.error(f"Failed to create model '{name}': {e}")
                return name, None
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_name = {executor.submit(create_single_model, name): name for name in model_names}
            
            for future in as_completed(future_to_name):
                name, model = future.result()
                if model is not None:
                    results[name] = model
        
        logger.info(f"Batch created {len(results)}/{len(model_names)} models successfully")
        return results
    
    def update_model_performance(self, 
                               name: str, 
                               metrics: Dict[str, float],
                               training_epoch: Optional[int] = None):
        """Update performance metrics with history tracking"""
        
        if name not in self.models:
            raise ValueError(f"Model '{name}' not found in registry")
        
        with self._lock:
            metadata = self.models[name]
            
            # Update current metrics
            metadata.performance_metrics.update(metrics)
            metadata.last_updated = time.time()
            
            # Add to training history
            history_entry = {
                'timestamp': time.time(),
                'epoch': training_epoch,
                'metrics': metrics.copy()
            }
            metadata.training_history.append(history_entry)
            
            # Limit history size
            if len(metadata.training_history) > 1000:
                metadata.training_history = metadata.training_history[-1000:]
            
            # Update status based on performance
            if 'sharpe_ratio' in metrics and metrics['sharpe_ratio'] > metadata.config.target_sharpe:
                metadata.status = ModelStatus.TRAINED
            
            self.save_registry()
        
        logger.info(f"Updated performance metrics for '{name}': {metrics}")
    
    def save_model_checkpoint(self, 
                             name: str, 
                             model: nn.Module, 
                             optimizer: Optional[torch.optim.Optimizer] = None,
                             scheduler: Optional[Any] = None,
                             epoch: Optional[int] = None,
                             loss: Optional[float] = None,
                             additional_data: Optional[Dict[str, Any]] = None,
                             checkpoint_dir: str = "deep_momentum_trading/data/models/checkpoints") -> str:
        """Save comprehensive model checkpoint"""
        
        if name not in self.models:
            raise ValueError(f"Model '{name}' not found in registry")
        
        # Ensure checkpoint directory exists
        checkpoint_path = Path(checkpoint_dir)
        checkpoint_path.mkdir(parents=True, exist_ok=True)
        
        # Create checkpoint filename with timestamp
        timestamp = int(time.time())
        checkpoint_file = checkpoint_path / f"{name}_checkpoint_{timestamp}.pt"
        
        # Prepare comprehensive checkpoint data
        checkpoint_data = {
            'model_state_dict': model.state_dict(),
            'model_config': asdict(self.models[name].config),
            'model_metadata': asdict(self.models[name]),
            'timestamp': timestamp,
            'parameter_count': self.factory._count_parameters(model),
            'model_hash': self._compute_model_hash(model),
            'pytorch_version': torch.__version__,
            'device': str(next(model.parameters()).device) if list(model.parameters()) else 'cpu'
        }
        
        # Add optional components
        if optimizer:
            checkpoint_data['optimizer_state_dict'] = optimizer.state_dict()
            checkpoint_data['optimizer_type'] = type(optimizer).__name__
        
        if scheduler:
            checkpoint_data['scheduler_state_dict'] = scheduler.state_dict()
            checkpoint_data['scheduler_type'] = type(scheduler).__name__
        
        if epoch is not None:
            checkpoint_data['epoch'] = epoch
        
        if loss is not None:
            checkpoint_data['loss'] = loss
        
        if additional_data:
            checkpoint_data['additional_data'] = additional_data
        
        # Save checkpoint with error handling
        try:
            torch.save(checkpoint_data, checkpoint_file)
            
            # Update metadata
            with self._lock:
                self.models[name].checkpoint_path = str(checkpoint_file)
                self.models[name].last_updated = time.time()
                self.models[name].status = ModelStatus.TRAINED
                self.save_registry()
            
            logger.info(f"Saved comprehensive checkpoint for '{name}': {checkpoint_file}")
            return str(checkpoint_file)
            
        except Exception as e:
            logger.error(f"Failed to save checkpoint for '{name}': {e}")
            raise
    
    def deploy_model(self, 
                    name: str, 
                    deployment_config: Dict[str, Any],
                    health_check_callback: Optional[Callable] = None) -> bool:
        """Deploy model with monitoring and health checks"""
        
        if name not in self.models:
            raise ValueError(f"Model '{name}' not found in registry")
        
        try:
            metadata = self.models[name]
            
            # Create deployment-optimized model
            model = self.create_model(name, device=deployment_config.get('device', 'cuda'))
            
            # Apply deployment-specific optimizations
            if deployment_config.get('enable_torchscript', False):
                model = torch.jit.script(model)
            
            # Update deployment info
            with self._lock:
                metadata.deployment_info = {
                    'deployed_at': time.time(),
                    'deployment_config': deployment_config,
                    'deployment_id': str(uuid.uuid4())
                }
                metadata.status = ModelStatus.DEPLOYED
                self.save_registry()
            
            # Start health monitoring if callback provided
            if health_check_callback:
                self._start_health_monitoring(name, health_check_callback)
            
            logger.info(f"Successfully deployed model '{name}'")
            return True
            
        except Exception as e:
            logger.error(f"Failed to deploy model '{name}': {e}")
            with self._lock:
                self.models[name].status = ModelStatus.FAILED
            return False
    
    def get_model_lineage(self, name: str) -> Dict[str, Any]:
        """Get complete model lineage (parents and children)"""
        
        if name not in self.models:
            raise ValueError(f"Model '{name}' not found in registry")
        
        metadata = self.models[name]
        lineage = {
            'model_id': metadata.model_id,
            'name': name,
            'parents': [],
            'children': [],
            'generation': 0
        }
        
        # Find parents
        current_parent_id = metadata.parent_model_id
        generation = 0
        while current_parent_id:
            parent_model = next((m for m in self.models.values() if m.model_id == current_parent_id), None)
            if parent_model:
                lineage['parents'].append({
                    'model_id': parent_model.model_id,
                    'name': parent_model.name,
                    'generation': generation + 1
                })
                current_parent_id = parent_model.parent_model_id
                generation += 1
            else:
                break
        
        lineage['generation'] = generation
        
        # Find children
        def find_children(model_id: str, depth: int = 0) -> List[Dict]:
            children = []
            for model in self.models.values():
                if model.parent_model_id == model_id:
                    child_info = {
                        'model_id': model.model_id,
                        'name': model.name,
                        'generation': depth + 1,
                        'children': find_children(model.model_id, depth + 1)
                    }
                    children.append(child_info)
            return children
        
        lineage['children'] = find_children(metadata.model_id)
        
        return lineage
    
    def optimize_model_for_production(self, 
                                    name: str, 
                                    sample_input: torch.Tensor,
                                    optimization_config: Optional[Dict[str, Any]] = None) -> bool:
        """Comprehensive production optimization"""
        
        if name not in self.models:
            raise ValueError(f"Model '{name}' not found in registry")
        
        try:
            model = self.create_model(name, use_cache=False)
            optimization_results = {}
            
            # ARM64 optimizations
            if self.enable_arm64_optimizations:
                if hasattr(model, 'compile_for_arm64'):
                    model.compile_for_arm64(sample_input)
                    optimization_results['arm64_compiled'] = True
                
                if hasattr(model, 'create_cuda_graph'):
                    model.create_cuda_graph(sample_input)
                    optimization_results['cuda_graph_created'] = True
            
            # TorchScript compilation
            try:
                scripted_model = torch.jit.script(model)
                optimization_results['torchscript_compiled'] = True
            except Exception as e:
                logger.warning(f"TorchScript compilation failed for '{name}': {e}")
                optimization_results['torchscript_compiled'] = False
            
            # Performance profiling
            if self.enable_arm64_optimizations:
                perf_metrics = self.performance_profiler.profile_model(
                    model=model,
                    sample_input=sample_input,
                    num_iterations=100,
                    model_name=name
                )
                optimization_results['performance_metrics'] = perf_metrics
            
            # Update optimization info
            with self._lock:
                self.models[name].optimization_info = {
                    'optimized_at': time.time(),
                    'optimization_results': optimization_results,
                    'optimization_level': OptimizationLevel.PRODUCTION.value
                }
                self.models[name].performance_metrics.update({
                    f'opt_{k}': v for k, v in optimization_results.get('performance_metrics', {}).items()
                })
                self.save_registry()
            
            logger.info(f"Production optimization completed for '{name}': {optimization_results}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to optimize model '{name}' for production: {e}")
            return False
    
    def get_registry_analytics(self) -> Dict[str, Any]:
        """Get comprehensive registry analytics"""
        
        with self._lock:
            analytics = {
                'total_models': len(self.models),
                'model_status_distribution': defaultdict(int),
                'model_type_distribution': defaultdict(int),
                'optimization_level_distribution': defaultdict(int),
                'total_parameters': 0,
                'total_size_mb': 0.0,
                'performance_statistics': defaultdict(list),
                'deployment_statistics': {
                    'deployed_models': 0,
                    'failed_models': 0,
                    'average_success_rate': 0.0
                },
                'lineage_statistics': {
                    'root_models': 0,
                    'max_generation': 0,
                    'total_lineages': 0
                }
            }
            
            success_rates = []
            max_generation = 0
            
            for metadata in self.models.values():
                # Basic statistics
                analytics['model_status_distribution'][metadata.status.value] += 1
                analytics['model_type_distribution'][metadata.config.model_type] += 1
                analytics['optimization_level_distribution'][metadata.config.optimization_level.value] += 1
                analytics['total_parameters'] += metadata.parameter_count
                analytics['total_size_mb'] += metadata.model_size_mb
                
                # Performance statistics
                for metric, value in metadata.performance_metrics.items():
                    analytics['performance_statistics'][metric].append(value)
                
                # Deployment statistics
                if metadata.status == ModelStatus.DEPLOYED:
                    analytics['deployment_statistics']['deployed_models'] += 1
                elif metadata.status == ModelStatus.FAILED:
                    analytics['deployment_statistics']['failed_models'] += 1
                
                # Success rate
                total_operations = metadata.success_count + metadata.error_count
                if total_operations > 0:
                    success_rate = metadata.success_count / total_operations
                    success_rates.append(success_rate)
                
                # Lineage statistics
                if not metadata.parent_model_id:
                    analytics['lineage_statistics']['root_models'] += 1
                
                # Calculate generation depth
                lineage = self.get_model_lineage(metadata.name)
                generation = lineage['generation']
                max_generation = max(max_generation, generation)
            
            analytics['lineage_statistics']['max_generation'] = max_generation
            analytics['lineage_statistics']['total_lineages'] = analytics['lineage_statistics']['root_models']
            
            # Calculate average success rate
            if success_rates:
                analytics['deployment_statistics']['average_success_rate'] = sum(success_rates) / len(success_rates)
            
            # Calculate performance statistics summaries
            for metric, values in analytics['performance_statistics'].items():
                if values:
                    analytics['performance_statistics'][metric] = {
                        'mean': np.mean(values),
                        'std': np.std(values),
                        'min': np.min(values),
                        'max': np.max(values),
                        'count': len(values)
                    }
            
            return analytics
    
    def cleanup_old_models(self,
                          max_age_days: int = 30,
                          keep_deployed: bool = True,
                          keep_best_performers: int = 5) -> int:
        """Cleanup old models based on age and performance"""
        
        current_time = time.time()
        max_age_seconds = max_age_days * 24 * 3600
        removed_count = 0
        
        # Get best performers to keep
        best_performers = []
        if keep_best_performers > 0:
            models_with_sharpe = [
                (name, metadata) for name, metadata in self.models.items()
                if 'sharpe_ratio' in metadata.performance_metrics
            ]
            models_with_sharpe.sort(
                key=lambda x: x[1].performance_metrics['sharpe_ratio'],
                reverse=True
            )
            best_performers = [name for name, _ in models_with_sharpe[:keep_best_performers]]
        
        models_to_remove = []
        
        with self._lock:
            for name, metadata in self.models.items():
                # Skip if deployed and keep_deployed is True
                if keep_deployed and metadata.status == ModelStatus.DEPLOYED:
                    continue
                
                # Skip if in best performers
                if name in best_performers:
                    continue
                
                # Check age
                age_seconds = current_time - metadata.last_updated
                if age_seconds > max_age_seconds:
                    models_to_remove.append(name)
            
            # Remove old models
            for name in models_to_remove:
                # Clean up checkpoint files
                metadata = self.models[name]
                if metadata.checkpoint_path and Path(metadata.checkpoint_path).exists():
                    try:
                        Path(metadata.checkpoint_path).unlink()
                    except Exception as e:
                        logger.warning(f"Failed to remove checkpoint for '{name}': {e}")
                
                # Remove from registry
                del self.models[name]
                removed_count += 1
            
            if removed_count > 0:
                self.save_registry()
        
        logger.info(f"Cleaned up {removed_count} old models")
        return removed_count
    
    def export_model_configs(self, output_path: str) -> bool:
        """Export all model configurations to a file"""
        
        try:
            export_data = {
                'export_timestamp': time.time(),
                'total_models': len(self.models),
                'models': {}
            }
            
            for name, metadata in self.models.items():
                export_data['models'][name] = {
                    'config': asdict(metadata.config),
                    'metadata': {
                        'model_id': metadata.model_id,
                        'status': metadata.status.value,
                        'parameter_count': metadata.parameter_count,
                        'model_size_mb': metadata.model_size_mb,
                        'performance_metrics': metadata.performance_metrics,
                        'last_updated': metadata.last_updated,
                        'parent_model_id': metadata.parent_model_id,
                        'child_model_ids': metadata.child_model_ids
                    }
                }
            
            with open(output_path, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)
            
            logger.info(f"Exported {len(self.models)} model configurations to {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to export model configurations: {e}")
            return False
    
    def import_model_configs(self, input_path: str, overwrite: bool = False) -> int:
        """Import model configurations from a file"""
        
        try:
            with open(input_path, 'r') as f:
                import_data = json.load(f)
            
            imported_count = 0
            
            for name, model_data in import_data.get('models', {}).items():
                if name in self.models and not overwrite:
                    logger.warning(f"Model '{name}' already exists, skipping")
                    continue
                
                try:
                    # Reconstruct config
                    config_data = model_data['config']
                    config = ScalableModelConfig(
                        model_type=config_data['model_type'],
                        parameters=config_data['parameters'],
                        description=config_data.get('description', ''),
                        version=config_data.get('version', '1.0'),
                        created_at=config_data.get('created_at', time.time()),
                        tags=config_data.get('tags', []),
                        max_assets=config_data.get('max_assets', 10000),
                        batch_size=config_data.get('batch_size', 1000),
                        target_sharpe=config_data.get('target_sharpe', 4.0),
                        max_drawdown_target=config_data.get('max_drawdown_target', 0.05),
                        optimization_level=OptimizationLevel(config_data.get('optimization_level', 'basic')),
                        enable_mixed_precision=config_data.get('enable_mixed_precision', True),
                        enable_gradient_checkpointing=config_data.get('enable_gradient_checkpointing', True),
                        enable_model_parallelism=config_data.get('enable_model_parallelism', True),
                        max_inference_latency_ms=config_data.get('max_inference_latency_ms', 100.0),
                        min_throughput_samples_per_sec=config_data.get('min_throughput_samples_per_sec', 1000.0),
                        memory_limit_gb=config_data.get('memory_limit_gb', 32.0),
                        deployment_targets=config_data.get('deployment_targets', ['cuda', 'cpu']),
                        auto_scaling_enabled=config_data.get('auto_scaling_enabled', True),
                        health_check_interval_sec=config_data.get('health_check_interval_sec', 60)
                    )
                    
                    # Register the model
                    self.register_model(name, config)
                    imported_count += 1
                    
                except Exception as e:
                    logger.error(f"Failed to import model '{name}': {e}")
            
            logger.info(f"Imported {imported_count} model configurations")
            return imported_count
            
        except Exception as e:
            logger.error(f"Failed to import model configurations: {e}")
            return 0
    
    # ===== PRIVATE HELPER METHODS =====
    
    def _estimate_model_size(self, model: nn.Module) -> float:
        """Estimate model size in MB"""
        param_size = 0
        buffer_size = 0
        
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
        
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        
        size_mb = (param_size + buffer_size) / 1024 / 1024
        return size_mb
    
    def _compute_model_hash(self, model: nn.Module) -> str:
        """Compute hash of model architecture and parameters"""
        model_str = str(model)
        param_str = ""
        
        for name, param in model.named_parameters():
            param_str += f"{name}:{param.shape}:{param.dtype}"
        
        combined_str = model_str + param_str
        return hashlib.md5(combined_str.encode()).hexdigest()
    
    def _load_checkpoint_safe(self, model: nn.Module, checkpoint_path: str):
        """Safely load checkpoint with error handling"""
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
            
            logger.debug(f"Loaded checkpoint: {checkpoint_path}")
            
        except Exception as e:
            logger.error(f"Failed to load checkpoint {checkpoint_path}: {e}")
            raise
    
    def _cache_model(self, cache_key: str, model: nn.Module):
        """Cache model with size limit"""
        if len(self._model_cache) >= self._cache_size_limit:
            # Remove oldest cached model
            oldest_key = next(iter(self._model_cache))
            del self._model_cache[oldest_key]
        
        self._model_cache[cache_key] = model
    
    def _start_health_monitoring(self, name: str, health_check_callback: Callable):
        """Start health monitoring for deployed model"""
        def health_monitor():
            while self.models[name].status == ModelStatus.DEPLOYED:
                try:
                    health_result = health_check_callback(name)
                    
                    with self._lock:
                        self.models[name].health_metrics.update(health_result)
                        self.models[name].last_updated = time.time()
                    
                    time.sleep(self._health_check_interval)
                    
                except Exception as e:
                    logger.error(f"Health check failed for '{name}': {e}")
                    with self._lock:
                        self.models[name].status = ModelStatus.FAILED
                    break
        
        # Start health monitoring in background thread
        health_thread = threading.Thread(target=health_monitor, daemon=True)
        health_thread.start()
    
    def save_registry(self):
        """Save registry to disk with error handling"""
        try:
            # Create backup of existing registry
            if self.registry_path.exists():
                backup_path = self.registry_path.with_suffix('.bak')
                self.registry_path.rename(backup_path)
            
            # Prepare serializable data
            registry_data = {
                'version': '2.0',
                'timestamp': time.time(),
                'models': {}
            }
            
            for name, metadata in self.models.items():
                registry_data['models'][name] = asdict(metadata)
            
            # Save to file
            with open(self.registry_path, 'w') as f:
                json.dump(registry_data, f, indent=2, default=str)
            
            logger.debug(f"Saved registry with {len(self.models)} models")
            
        except Exception as e:
            logger.error(f"Failed to save registry: {e}")
            # Restore backup if save failed
            backup_path = self.registry_path.with_suffix('.bak')
            if backup_path.exists():
                backup_path.rename(self.registry_path)
            raise
    
    def load_registry(self):
        """Load registry from disk with error handling"""
        try:
            with open(self.registry_path, 'r') as f:
                registry_data = json.load(f)
            
            models_data = registry_data.get('models', {})
            
            for name, metadata_dict in models_data.items():
                try:
                    # Reconstruct config
                    config_dict = metadata_dict['config']
                    config = ScalableModelConfig(
                        model_type=config_dict['model_type'],
                        parameters=config_dict['parameters'],
                        description=config_dict.get('description', ''),
                        version=config_dict.get('version', '1.0'),
                        created_at=config_dict.get('created_at', time.time()),
                        tags=config_dict.get('tags', []),
                        max_assets=config_dict.get('max_assets', 10000),
                        batch_size=config_dict.get('batch_size', 1000),
                        target_sharpe=config_dict.get('target_sharpe', 4.0),
                        max_drawdown_target=config_dict.get('max_drawdown_target', 0.05),
                        optimization_level=OptimizationLevel(config_dict.get('optimization_level', 'basic')),
                        enable_mixed_precision=config_dict.get('enable_mixed_precision', True),
                        enable_gradient_checkpointing=config_dict.get('enable_gradient_checkpointing', True),
                        enable_model_parallelism=config_dict.get('enable_model_parallelism', True),
                        max_inference_latency_ms=config_dict.get('max_inference_latency_ms', 100.0),
                        min_throughput_samples_per_sec=config_dict.get('min_throughput_samples_per_sec', 1000.0),
                        memory_limit_gb=config_dict.get('memory_limit_gb', 32.0),
                        deployment_targets=config_dict.get('deployment_targets', ['cuda', 'cpu']),
                        auto_scaling_enabled=config_dict.get('auto_scaling_enabled', True),
                        health_check_interval_sec=config_dict.get('health_check_interval_sec', 60)
                    )
                    
                    # Reconstruct metadata
                    metadata = ScalableModelMetadata(
                        name=name,
                        config=config,
                        model_path=metadata_dict.get('model_path'),
                        checkpoint_path=metadata_dict.get('checkpoint_path'),
                        performance_metrics=metadata_dict.get('performance_metrics', {}),
                        last_updated=metadata_dict.get('last_updated', time.time()),
                        model_size_mb=metadata_dict.get('model_size_mb', 0.0),
                        parameter_count=metadata_dict.get('parameter_count', 0),
                        training_history=metadata_dict.get('training_history', []),
                        model_id=metadata_dict.get('model_id', str(uuid.uuid4())),
                        status=ModelStatus(metadata_dict.get('status', 'registered')),
                        deployment_info=metadata_dict.get('deployment_info', {}),
                        optimization_info=metadata_dict.get('optimization_info', {}),
                        health_metrics=metadata_dict.get('health_metrics', {}),
                        inference_stats=metadata_dict.get('inference_stats', {}),
                        memory_usage=metadata_dict.get('memory_usage', {}),
                        error_count=metadata_dict.get('error_count', 0),
                        success_count=metadata_dict.get('success_count', 0),
                        parent_model_id=metadata_dict.get('parent_model_id'),
                        child_model_ids=metadata_dict.get('child_model_ids', []),
                        model_hash=metadata_dict.get('model_hash')
                    )
                    
                    self.models[name] = metadata
                    
                except Exception as e:
                    logger.error(f"Failed to load model '{name}' from registry: {e}")
            
            logger.info(f"Loaded registry with {len(self.models)} models")
            
        except Exception as e:
            logger.error(f"Failed to load registry: {e}")
            self.models = {}
    
    def get_model_info(self, name: str) -> Dict[str, Any]:
        """Get comprehensive model information"""
        if name not in self.models:
            raise ValueError(f"Model '{name}' not found in registry")
        
        metadata = self.models[name]
        
        return {
            'name': name,
            'model_id': metadata.model_id,
            'status': metadata.status.value,
            'config': asdict(metadata.config),
            'performance_metrics': metadata.performance_metrics,
            'parameter_count': metadata.parameter_count,
            'model_size_mb': metadata.model_size_mb,
            'last_updated': metadata.last_updated,
            'training_history_length': len(metadata.training_history),
            'deployment_info': metadata.deployment_info,
            'optimization_info': metadata.optimization_info,
            'health_metrics': metadata.health_metrics,
            'success_rate': (
                metadata.success_count / (metadata.success_count + metadata.error_count)
                if (metadata.success_count + metadata.error_count) > 0 else 0.0
            ),
            'lineage': self.get_model_lineage(name)
        }
    
    def list_models(self,
                   status_filter: Optional[ModelStatus] = None,
                   model_type_filter: Optional[str] = None,
                   sort_by: str = 'last_updated',
                   reverse: bool = True) -> List[Dict[str, Any]]:
        """List models with filtering and sorting"""
        
        models_list = []
        
        for name, metadata in self.models.items():
            # Apply filters
            if status_filter and metadata.status != status_filter:
                continue
            
            if model_type_filter and metadata.config.model_type != model_type_filter:
                continue
            
            model_info = {
                'name': name,
                'model_id': metadata.model_id,
                'status': metadata.status.value,
                'model_type': metadata.config.model_type,
                'parameter_count': metadata.parameter_count,
                'model_size_mb': metadata.model_size_mb,
                'last_updated': metadata.last_updated,
                'performance_metrics': metadata.performance_metrics
            }
            
            models_list.append(model_info)
        
        # Sort models
        if sort_by in ['last_updated', 'parameter_count', 'model_size_mb']:
            models_list.sort(key=lambda x: x[sort_by], reverse=reverse)
        elif sort_by == 'name':
            models_list.sort(key=lambda x: x['name'], reverse=reverse)
        elif sort_by == 'sharpe_ratio':
            models_list.sort(
                key=lambda x: x['performance_metrics'].get('sharpe_ratio', 0.0),
                reverse=reverse
            )
        
        return models_list
    
    def __del__(self):
        """Cleanup resources"""
        try:
            if hasattr(self, '_executor'):
                self._executor.shutdown(wait=False)
        except Exception:
            pass


# ===== CONVENIENCE FUNCTIONS =====

def create_production_model_registry(
    registry_path: str = "production_model_registry.json",
    enable_arm64_optimizations: bool = True,
    max_concurrent_operations: int = 20
) -> ScalableModelRegistry:
    """Create a production-ready model registry with optimal settings"""
    
    return ScalableModelRegistry(
        registry_path=registry_path,
        enable_arm64_optimizations=enable_arm64_optimizations,
        max_concurrent_operations=max_concurrent_operations,
        enable_model_versioning=True
    )


def create_development_model_registry(
    registry_path: str = "dev_model_registry.json"
) -> ScalableModelRegistry:
    """Create a development model registry with basic settings"""
    
    return ScalableModelRegistry(
        registry_path=registry_path,
        enable_arm64_optimizations=False,
        max_concurrent_operations=5,
        enable_model_versioning=True
    )


# ===== EXPORTS =====

__all__ = [
    'ModelStatus',
    'OptimizationLevel',
    'ScalableModelConfig',
    'ScalableModelMetadata',
    'ScalableModelFactory',
    'ScalableModelRegistry',
    'create_production_model_registry',
    'create_development_model_registry'
]
