"""
Enhanced Training Package for Deep Momentum Networks + LSTM Trading System

This package provides comprehensive training infrastructure with ARM64 optimizations,
distributed training capabilities, and advanced MLOps features.

Key Components:
- DistributedTrainer: Multi-GPU and multi-node training with ARM64 optimizations
- TrainingDataLoader: Real-time data integration with streaming capabilities
- TrainingMonitor: Comprehensive MLOps monitoring and alerting
- TrainingPipeline: End-to-end training orchestration
- HyperparameterTuning: Advanced hyperparameter optimization
- ModelSelection: Automated model selection and validation
- Trainer: Core training engine with ARM64 optimizations
- Validation: Comprehensive model validation and testing

Features:
- ARM64 architecture optimizations
- Distributed training (multi-GPU/multi-node)
- Real-time data streaming
- Comprehensive monitoring and alerting
- Automated hyperparameter tuning
- Advanced model selection
- Production-ready deployment
"""

from .trainer import Trainer, TrainerConfig
from .distributed_trainer import (
    DistributedTrainer,
    DistributedTrainingConfig,
    TrainingNode,
    DistributedTrainingStats
)
from .training_data_loader import (
    TrainingDataLoader,
    StreamingDataset,
    DataLoaderConfig,
    DataLoaderStats
)
from .training_monitor import (
    TrainingMonitor,
    TrainingMetrics,
    MonitoringConfig,
    AlertConfig,
    TrainingAlert
)
from .training_pipeline import (
    TrainingPipeline,
    PipelineConfig,
    PipelineStage,
    PipelineStats
)
from .hyperparameter_tuning import (
    HyperparameterTuner,
    TuningConfig,
    TuningStrategy,
    TuningResults
)
from .model_selection import (
    ModelSelector,
    SelectionConfig,
    SelectionCriteria,
    ModelCandidate,
    SelectionResults
)
from .validation import (
    ModelValidator,
    ValidationConfig,
    ValidationSuite,
    ValidationResults,
    ValidationMetrics
)

# Version information
__version__ = "1.0.0"
__author__ = "Deep Momentum Trading Team"

# Package metadata
__all__ = [
    # Core training components
    "Trainer",
    "TrainerConfig",
    
    # Distributed training
    "DistributedTrainer",
    "DistributedTrainingConfig",
    "TrainingNode",
    "DistributedTrainingStats",
    
    # Data loading
    "TrainingDataLoader",
    "StreamingDataset",
    "DataLoaderConfig",
    "DataLoaderStats",
    
    # Monitoring
    "TrainingMonitor",
    "TrainingMetrics",
    "MonitoringConfig",
    "AlertConfig",
    "TrainingAlert",
    
    # Pipeline
    "TrainingPipeline",
    "PipelineConfig",
    "PipelineStage",
    "PipelineStats",
    
    # Hyperparameter tuning
    "HyperparameterTuner",
    "TuningConfig",
    "TuningStrategy",
    "TuningResults",
    
    # Model selection
    "ModelSelector",
    "SelectionConfig",
    "SelectionCriteria",
    "ModelCandidate",
    "SelectionResults",
    
    # Validation
    "ModelValidator",
    "ValidationConfig",
    "ValidationSuite",
    "ValidationResults",
    "ValidationMetrics",
]

# Configuration defaults
DEFAULT_CONFIG = {
    "arm64_optimizations": True,
    "distributed_training": False,
    "monitoring_enabled": True,
    "real_time_data": True,
    "compression_enabled": True,
    "batch_size": 32,
    "learning_rate": 0.001,
    "epochs": 100,
    "validation_split": 0.2,
    "early_stopping": True,
    "checkpoint_frequency": 10,
}

def get_default_config():
    """Get default training configuration"""
    return DEFAULT_CONFIG.copy()

def create_training_pipeline(config=None):
    """
    Create a complete training pipeline with default configuration.
    
    Args:
        config: Optional configuration dictionary
        
    Returns:
        TrainingPipeline: Configured training pipeline
    """
    if config is None:
        config = get_default_config()
    
    pipeline_config = PipelineConfig(**config)
    return TrainingPipeline(pipeline_config)

def create_distributed_trainer(config=None):
    """
    Create a distributed trainer with default configuration.
    
    Args:
        config: Optional configuration dictionary
        
    Returns:
        DistributedTrainer: Configured distributed trainer
    """
    if config is None:
        config = get_default_config()
    
    trainer_config = DistributedTrainingConfig(**config)
    return DistributedTrainer(trainer_config)

# Package initialization
def initialize_training_environment():
    """Initialize the training environment with optimal settings"""
    import torch
    import platform
    
    # Detect ARM64 architecture
    is_arm64 = platform.machine().lower() in ['arm64', 'aarch64']
    
    if is_arm64:
        # ARM64-specific optimizations
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        
    # Set optimal thread counts
    if torch.get_num_threads() < 4:
        torch.set_num_threads(4)
    
    return {
        'arm64_detected': is_arm64,
        'torch_threads': torch.get_num_threads(),
        'cuda_available': torch.cuda.is_available(),
        'cuda_devices': torch.cuda.device_count() if torch.cuda.is_available() else 0
    }

# Auto-initialize on import
_ENVIRONMENT_INFO = initialize_training_environment()

def get_environment_info():
    """Get training environment information"""
    return _ENVIRONMENT_INFO.copy()