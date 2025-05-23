#!/usr/bin/env python3
"""
Enhanced Model Training Script with ARM64 Optimizations

This script provides comprehensive model training capabilities for the Deep Momentum Trading System
with ARM64-specific optimizations, Polygon.io data integration, and production-ready features.

Features:
- ARM64-optimized training pipeline
- Polygon.io data integration (no Yahoo Finance)
- Distributed training support
- Real-time model validation
- Automated hyperparameter tuning
- Model deployment and versioning
- Performance monitoring and reporting
"""

import os
import sys
import argparse
import asyncio
import signal
import time
import platform
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional
import logging
import json
import yaml
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor
import multiprocessing as cpu_mp

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.training.training_pipeline import TrainingPipeline
from src.training.distributed_trainer import DistributedTrainer
from src.training.training_monitor import TrainingMonitor
from src.training.training_data_loader import TrainingDataLoader
from src.training.hyperparameter_tuning import HyperparameterTuner
from src.training.model_selection import ModelSelector
from src.training.validation import ModelValidator
from src.models.model_registry import ModelRegistry
from src.models.ensemble_system import EnsembleSystem
from src.models.arm64_optimizations import ARM64Optimizer
from src.data.data_manager import DataManager
from src.data.polygon_client import PolygonClient
from src.communication.message_broker import MessageBroker
from src.infrastructure.health_check import HealthChecker
from src.monitoring.alert_system import AlertSystem
from src.monitoring.performance_tracker import PerformanceTracker
from src.utils.logger import get_logger
from src.utils.decorators import performance_monitor, error_handler
from src.utils.exceptions import TrainingError
from config.settings import config_manager

logger = get_logger(__name__)

class ARM64TrainingOptimizer:
    """ARM64-specific optimizations for model training"""
    
    def __init__(self):
        self.is_arm64 = platform.machine().lower() in ['arm64', 'aarch64']
        self.cpu_count = cpu_mp.cpu_count()
        self.optimal_workers = self._calculate_optimal_workers()
        self.memory_optimization = self._get_memory_optimization()
        
    def _calculate_optimal_workers(self) -> int:
        """Calculate optimal number of workers for ARM64"""
        if self.is_arm64:
            # ARM64 benefits from moderate parallelism for training
            return min(self.cpu_count, 8)
        return min(self.cpu_count, 12)
    
    def _get_memory_optimization(self) -> Dict[str, Any]:
        """Get ARM64 memory optimization settings"""
        if self.is_arm64:
            return {
                'batch_size': 32,
                'gradient_accumulation_steps': 4,
                'max_sequence_length': 256,
                'cache_size': 1024
            }
        return {
            'batch_size': 64,
            'gradient_accumulation_steps': 2,
            'max_sequence_length': 512,
            'cache_size': 2048
        }

class TrainingConfig:
    """Training configuration"""
    
    def __init__(self, **kwargs):
        # Training mode
        self.training_mode = kwargs.get('training_mode', 'full')  # full, incremental, validation_only
        
        # Data configuration
        self.symbols = kwargs.get('symbols', ['SPY', 'QQQ', 'IWM'])
        self.data_source = 'polygon'  # Only Polygon, no Yahoo
        self.lookback_days = kwargs.get('lookback_days', 252)  # 1 year
        self.validation_split = kwargs.get('validation_split', 0.2)
        self.test_split = kwargs.get('test_split', 0.1)
        
        # Model configuration
        self.model_types = kwargs.get('model_types', ['lstm', 'transformer', 'ensemble'])
        self.enable_ensemble = kwargs.get('enable_ensemble', True)
        self.ensemble_size = kwargs.get('ensemble_size', 5)
        
        # Training parameters
        self.epochs = kwargs.get('epochs', 100)
        self.early_stopping_patience = kwargs.get('early_stopping_patience', 10)
        self.learning_rate = kwargs.get('learning_rate', 0.001)
        self.batch_size = kwargs.get('batch_size', None)  # Will be set by ARM64 optimizer
        
        # Optimization
        self.enable_hyperparameter_tuning = kwargs.get('enable_hyperparameter_tuning', True)
        self.tuning_trials = kwargs.get('tuning_trials', 50)
        self.enable_distributed = kwargs.get('enable_distributed', False)
        self.world_size = kwargs.get('world_size', 1)
        
        # Validation
        self.enable_cross_validation = kwargs.get('enable_cross_validation', True)
        self.cv_folds = kwargs.get('cv_folds', 5)
        self.validation_metrics = kwargs.get('validation_metrics', ['sharpe_ratio', 'max_drawdown', 'accuracy'])
        
        # Deployment
        self.auto_deploy = kwargs.get('auto_deploy', False)
        self.deployment_threshold = kwargs.get('deployment_threshold', 0.5)  # Minimum Sharpe ratio
        
        # Performance
        self.max_workers = kwargs.get('max_workers', None)
        self.enable_mixed_precision = kwargs.get('enable_mixed_precision', True)
        self.enable_gradient_checkpointing = kwargs.get('enable_gradient_checkpointing', True)
        
        # Monitoring
        self.enable_monitoring = kwargs.get('enable_monitoring', True)
        self.save_checkpoints = kwargs.get('save_checkpoints', True)
        self.checkpoint_interval = kwargs.get('checkpoint_interval', 10)  # epochs

class ModelTrainingEngine:
    """
    Enhanced model training engine with ARM64 optimizations
    """
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.optimizer = ARM64TrainingOptimizer()
        
        # Apply ARM64 optimizations to config
        if not self.config.batch_size:
            self.config.batch_size = self.optimizer.memory_optimization['batch_size']
        
        # Initialize components
        self.polygon_client = PolygonClient()
        self.data_manager = DataManager()
        self.model_registry = ModelRegistry()
        self.ensemble_system = EnsembleSystem()
        self.arm64_optimizer = ARM64Optimizer()
        
        # Training components
        self.training_pipeline = None
        self.distributed_trainer = None
        self.hyperparameter_tuner = None
        self.model_selector = None
        self.model_validator = None
        
        # Infrastructure
        self.message_broker = MessageBroker()
        self.health_checker = HealthChecker()
        self.alert_system = AlertSystem()
        self.performance_tracker = PerformanceTracker()
        
        # Training state
        self.training_id = self._generate_training_id()
        self.is_running = False
        self.current_model = None
        self.training_results = {}
        
        # Statistics
        self.stats = {
            'models_trained': 0,
            'best_model_performance': 0.0,
            'total_training_time': 0.0,
            'data_points_processed': 0,
            'validation_accuracy': 0.0,
            'start_time': None
        }
        
        # Signal handling
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        logger.info(f"ModelTrainingEngine initialized with ARM64 optimizations: {self.optimizer.is_arm64}")
    
    def _generate_training_id(self) -> str:
        """Generate unique training ID"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        return f"training_{timestamp}"
    
    @performance_monitor
    @error_handler
    async def start_training(self) -> Dict[str, Any]:
        """
        Start comprehensive model training
        
        Returns:
            Dict containing training results
        """
        logger.info(f"Starting enhanced model training session: {self.training_id}")
        self.stats['start_time'] = time.time()
        
        try:
            # Initialize infrastructure
            await self._initialize_infrastructure()
            
            # Prepare training data
            await self._prepare_training_data()
            
            # Initialize training components
            await self._initialize_training_components()
            
            # Execute training based on mode
            if self.config.training_mode == 'validation_only':
                results = await self._run_validation_only()
            elif self.config.training_mode == 'incremental':
                results = await self._run_incremental_training()
            else:
                results = await self._run_full_training()
            
            # Deploy models if configured
            if self.config.auto_deploy:
                await self._auto_deploy_models(results)
            
            # Generate training report
            await self._generate_training_report(results)
            
            self.stats['total_training_time'] = time.time() - self.stats['start_time']
            
            logger.info(f"Training session {self.training_id} completed successfully")
            
            return {
                'status': 'completed',
                'training_id': self.training_id,
                'results': results,
                'stats': self.stats
            }
            
        except Exception as e:
            logger.error(f"Training session {self.training_id} failed: {e}")
            await self._cleanup()
            raise TrainingError(f"Training session failed: {e}")
    
    async def _initialize_infrastructure(self):
        """Initialize infrastructure components"""
        logger.info("Initializing training infrastructure...")
        
        # Initialize message broker
        await self.message_broker.start()
        
        # Initialize health checker
        await self.health_checker.start()
        
        # Initialize alert system
        await self.alert_system.start()
        
        # Initialize performance tracker
        await self.performance_tracker.start()
        
        logger.info("Training infrastructure initialized")
    
    async def _prepare_training_data(self):
        """Prepare training data from Polygon"""
        logger.info("Preparing training data from Polygon...")
        
        # Initialize Polygon client
        await self.polygon_client.connect()
        
        # Initialize data manager
        await self.data_manager.initialize()
        
        # Download historical data for all symbols
        end_date = datetime.now()
        start_date = end_date - timedelta(days=self.config.lookback_days)
        
        for symbol in self.config.symbols:
            logger.info(f"Downloading data for {symbol}...")
            
            # Download bars data
            bars_data = await self.polygon_client.get_historical_bars(
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
                timespan='day'
            )
            
            # Store data
            await self.data_manager.store_historical_data(symbol, bars_data)
            
            self.stats['data_points_processed'] += len(bars_data)
        
        logger.info(f"Training data prepared for {len(self.config.symbols)} symbols")
    
    async def _initialize_training_components(self):
        """Initialize training components"""
        logger.info("Initializing training components...")
        
        # Initialize model registry
        await self.model_registry.initialize()
        
        # Initialize ARM64 optimizer
        await self.arm64_optimizer.initialize()
        
        # Initialize training pipeline
        self.training_pipeline = TrainingPipeline(
            data_manager=self.data_manager,
            model_registry=self.model_registry,
            arm64_optimizer=self.arm64_optimizer
        )
        
        # Initialize distributed trainer if enabled
        if self.config.enable_distributed:
            self.distributed_trainer = DistributedTrainer(
                training_pipeline=self.training_pipeline,
                world_size=self.config.world_size
            )
        
        # Initialize hyperparameter tuner
        if self.config.enable_hyperparameter_tuning:
            self.hyperparameter_tuner = HyperparameterTuner(
                training_pipeline=self.training_pipeline,
                trials=self.config.tuning_trials
            )
        
        # Initialize model selector
        self.model_selector = ModelSelector(
            model_registry=self.model_registry,
            validation_metrics=self.config.validation_metrics
        )
        
        # Initialize model validator
        self.model_validator = ModelValidator(
            data_manager=self.data_manager,
            cross_validation=self.config.enable_cross_validation,
            cv_folds=self.config.cv_folds
        )
        
        # Initialize ensemble system
        if self.config.enable_ensemble:
            await self.ensemble_system.initialize(
                model_registry=self.model_registry,
                ensemble_size=self.config.ensemble_size
            )
        
        logger.info("Training components initialized")
    
    async def _run_full_training(self) -> Dict[str, Any]:
        """Run full training pipeline"""
        logger.info("Running full training pipeline...")
        
        results = {
            'models': {},
            'best_model': None,
            'ensemble_model': None,
            'validation_results': {}
        }
        
        # Train individual models
        for model_type in self.config.model_types:
            if model_type == 'ensemble':
                continue  # Handle ensemble separately
            
            logger.info(f"Training {model_type} model...")
            
            model_results = await self._train_single_model(model_type)
            results['models'][model_type] = model_results
            
            self.stats['models_trained'] += 1
        
        # Select best model
        if results['models']:
            best_model_name = await self.model_selector.select_best_model(
                list(results['models'].keys())
            )
            results['best_model'] = results['models'][best_model_name]
            
            # Update best performance
            best_performance = results['best_model'].get('validation_metrics', {}).get('sharpe_ratio', 0.0)
            self.stats['best_model_performance'] = best_performance
        
        # Train ensemble model if enabled
        if self.config.enable_ensemble and len(results['models']) > 1:
            logger.info("Training ensemble model...")
            
            ensemble_results = await self._train_ensemble_model(list(results['models'].keys()))
            results['ensemble_model'] = ensemble_results
        
        # Run comprehensive validation
        validation_results = await self._run_comprehensive_validation(results)
        results['validation_results'] = validation_results
        
        logger.info("Full training pipeline completed")
        return results
    
    async def _train_single_model(self, model_type: str) -> Dict[str, Any]:
        """Train a single model"""
        try:
            model_name = f"{model_type}_{self.training_id}"
            
            # Prepare model configuration
            model_config = self._get_model_config(model_type)
            
            # Apply ARM64 optimizations
            model_config = await self.arm64_optimizer.optimize_model_config(model_config)
            
            # Hyperparameter tuning if enabled
            if self.config.enable_hyperparameter_tuning:
                logger.info(f"Tuning hyperparameters for {model_type}...")
                
                best_params = await self.hyperparameter_tuner.tune_hyperparameters(
                    model_type=model_type,
                    base_config=model_config
                )
                
                model_config.update(best_params)
            
            # Train model
            if self.config.enable_distributed and self.distributed_trainer:
                training_results = await self.distributed_trainer.train_model(
                    model_name=model_name,
                    model_config=model_config,
                    epochs=self.config.epochs
                )
            else:
                training_results = await self.training_pipeline.train_model(
                    model_name=model_name,
                    model_config=model_config,
                    epochs=self.config.epochs
                )
            
            # Validate model
            validation_results = await self.model_validator.validate_model(
                model_name=model_name,
                symbols=self.config.symbols
            )
            
            # Combine results
            results = {
                'model_name': model_name,
                'model_type': model_type,
                'training_results': training_results,
                'validation_results': validation_results,
                'config': model_config
            }
            
            # Update validation accuracy
            accuracy = validation_results.get('accuracy', 0.0)
            self.stats['validation_accuracy'] = max(self.stats['validation_accuracy'], accuracy)
            
            logger.info(f"Model {model_name} training completed")
            return results
            
        except Exception as e:
            logger.error(f"Error training {model_type} model: {e}")
            raise
    
    def _get_model_config(self, model_type: str) -> Dict[str, Any]:
        """Get model configuration"""
        base_config = {
            'symbols': self.config.symbols,
            'batch_size': self.config.batch_size,
            'learning_rate': self.config.learning_rate,
            'early_stopping_patience': self.config.early_stopping_patience,
            'enable_mixed_precision': self.config.enable_mixed_precision,
            'enable_gradient_checkpointing': self.config.enable_gradient_checkpointing
        }
        
        if model_type == 'lstm':
            base_config.update({
                'hidden_size': 128,
                'num_layers': 3,
                'dropout': 0.2,
                'sequence_length': 60
            })
        elif model_type == 'transformer':
            base_config.update({
                'hidden_size': 256,
                'num_heads': 8,
                'num_layers': 6,
                'dropout': 0.1,
                'sequence_length': 120
            })
        
        return base_config
    
    async def _train_ensemble_model(self, model_names: List[str]) -> Dict[str, Any]:
        """Train ensemble model"""
        try:
            logger.info("Training ensemble model...")
            
            ensemble_name = f"ensemble_{self.training_id}"
            
            # Create ensemble
            ensemble_results = await self.ensemble_system.create_ensemble(
                model_names=model_names,
                ensemble_name=ensemble_name,
                ensemble_method='weighted_average'
            )
            
            # Validate ensemble
            validation_results = await self.model_validator.validate_model(
                model_name=ensemble_name,
                symbols=self.config.symbols
            )
            
            results = {
                'ensemble_name': ensemble_name,
                'member_models': model_names,
                'ensemble_results': ensemble_results,
                'validation_results': validation_results
            }
            
            logger.info("Ensemble model training completed")
            return results
            
        except Exception as e:
            logger.error(f"Error training ensemble model: {e}")
            raise
    
    async def _run_incremental_training(self) -> Dict[str, Any]:
        """Run incremental training on existing models"""
        logger.info("Running incremental training...")
        
        # Get existing models
        existing_models = await self.model_registry.list_models()
        
        if not existing_models:
            logger.warning("No existing models found, running full training instead")
            return await self._run_full_training()
        
        results = {'updated_models': {}}
        
        # Update each existing model
        for model_name in existing_models:
            logger.info(f"Incrementally training {model_name}...")
            
            # Load existing model
            model = await self.model_registry.load_model(model_name)
            
            # Incremental training
            training_results = await self.training_pipeline.incremental_train(
                model=model,
                model_name=model_name,
                epochs=self.config.epochs // 4  # Fewer epochs for incremental
            )
            
            # Validate updated model
            validation_results = await self.model_validator.validate_model(
                model_name=model_name,
                symbols=self.config.symbols
            )
            
            results['updated_models'][model_name] = {
                'training_results': training_results,
                'validation_results': validation_results
            }
            
            self.stats['models_trained'] += 1
        
        logger.info("Incremental training completed")
        return results
    
    async def _run_validation_only(self) -> Dict[str, Any]:
        """Run validation only on existing models"""
        logger.info("Running validation-only mode...")
        
        # Get existing models
        existing_models = await self.model_registry.list_models()
        
        if not existing_models:
            raise TrainingError("No existing models found for validation")
        
        results = {'validation_results': {}}
        
        # Validate each model
        for model_name in existing_models:
            logger.info(f"Validating {model_name}...")
            
            validation_results = await self.model_validator.validate_model(
                model_name=model_name,
                symbols=self.config.symbols
            )
            
            results['validation_results'][model_name] = validation_results
        
        # Select best performing model
        best_model = await self.model_selector.select_best_model(existing_models)
        results['best_model'] = best_model
        
        logger.info("Validation-only mode completed")
        return results
    
    async def _run_comprehensive_validation(self, training_results: Dict[str, Any]) -> Dict[str, Any]:
        """Run comprehensive validation on all trained models"""
        logger.info("Running comprehensive validation...")
        
        validation_results = {}
        
        # Validate individual models
        for model_name, model_results in training_results['models'].items():
            validation_results[model_name] = await self.model_validator.comprehensive_validation(
                model_name=model_results['model_name'],
                symbols=self.config.symbols
            )
        
        # Validate ensemble if available
        if training_results.get('ensemble_model'):
            ensemble_name = training_results['ensemble_model']['ensemble_name']
            validation_results[ensemble_name] = await self.model_validator.comprehensive_validation(
                model_name=ensemble_name,
                symbols=self.config.symbols
            )
        
        logger.info("Comprehensive validation completed")
        return validation_results
    
    async def _auto_deploy_models(self, training_results: Dict[str, Any]):
        """Auto-deploy models that meet deployment criteria"""
        logger.info("Evaluating models for auto-deployment...")
        
        # Check best model for deployment
        best_model = training_results.get('best_model')
        if best_model:
            sharpe_ratio = best_model.get('validation_results', {}).get('sharpe_ratio', 0.0)
            
            if sharpe_ratio >= self.config.deployment_threshold:
                logger.info(f"Deploying best model: {best_model['model_name']}")
                
                await self.model_registry.deploy_model(
                    model_name=best_model['model_name'],
                    deployment_type='production'
                )
                
                # Send deployment notification
                await self.alert_system.send_alert(
                    level='info',
                    message=f'Model {best_model["model_name"]} auto-deployed to production',
                    context={
                        'model_name': best_model['model_name'],
                        'sharpe_ratio': sharpe_ratio,
                        'training_id': self.training_id
                    }
                )
        
        # Check ensemble model for deployment
        ensemble_model = training_results.get('ensemble_model')
        if ensemble_model:
            sharpe_ratio = ensemble_model.get('validation_results', {}).get('sharpe_ratio', 0.0)
            
            if sharpe_ratio >= self.config.deployment_threshold:
                logger.info(f"Deploying ensemble model: {ensemble_model['ensemble_name']}")
                
                await self.model_registry.deploy_model(
                    model_name=ensemble_model['ensemble_name'],
                    deployment_type='production'
                )
    
    async def _generate_training_report(self, training_results: Dict[str, Any]):
        """Generate comprehensive training report"""
        logger.info("Generating training report...")
        
        report = {
            'training_id': self.training_id,
            'timestamp': datetime.now().isoformat(),
            'configuration': {
                'training_mode': self.config.training_mode,
                'symbols': self.config.symbols,
                'model_types': self.config.model_types,
                'epochs': self.config.epochs,
                'batch_size': self.config.batch_size
            },
            'results': training_results,
            'statistics': self.stats,
            'performance_summary': self._generate_performance_summary(training_results)
        }
        
        # Save report
        report_dir = project_root / "reports" / "training"
        report_dir.mkdir(parents=True, exist_ok=True)
        
        report_file = report_dir / f"training_report_{self.training_id}.json"
        
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        # Track performance
        await self.performance_tracker.record_metrics('model_training', report)
        
        logger.info(f"Training report saved: {report_file}")
    
    def _generate_performance_summary(self, training_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate performance summary"""
        summary = {
            'total_models_trained': self.stats['models_trained'],
            'best_sharpe_ratio': 0.0,
            'average_accuracy': 0.0,
            'training_duration': self.stats['total_training_time']
        }
        
        # Calculate best Sharpe ratio
        sharpe_ratios = []
        accuracies = []
        
        for model_results in training_results.get('models', {}).values():
            validation = model_results.get('validation_results', {})
            if 'sharpe_ratio' in validation:
                sharpe_ratios.append(validation['sharpe_ratio'])
            if 'accuracy' in validation:
                accuracies.append(validation['accuracy'])
        
        if sharpe_ratios:
            summary['best_sharpe_ratio'] = max(sharpe_ratios)
        
        if accuracies:
            summary['average_accuracy'] = np.mean(accuracies)
        
        return summary
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        logger.info(f"Received signal {signum}, stopping training...")
        self.is_running = False
    
    async def _cleanup(self):
        """Cleanup resources"""
        logger.info("Cleaning up training resources...")
        
        self.is_running = False
        
        # Close connections
        if self.polygon_client:
            await self.polygon_client.disconnect()
        
        if self.message_broker:
            await self.message_broker.stop()
        
        if self.health_checker:
            await self.health_checker.stop()
        
        if self.alert_system:
            await self.alert_system.stop()
        
        if self.performance_tracker:
            await self.performance_tracker.stop()
        
        logger.info("Training cleanup completed")

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Enhanced Model Training Script')
    
    # Training mode
    parser.add_argument('--mode', choices=['full', 'incremental', 'validation_only'], default='full',
                       help='Training mode')
    
    # Data configuration
    parser.add_argument('--symbols', nargs='+', default=['SPY', 'QQQ', 'IWM'],
                       help='Symbols to train on')
    parser.add_argument('--lookback-days', type=int, default=252,
                       help='Number of days of historical data')
    
    # Model configuration
    parser.add_argument('--model-types', nargs='+', default=['lstm', 'transformer'],
                       choices=['lstm', 'transformer', 'ensemble'],
                       help='Model types to train')
    parser.add_argument('--no-ensemble', action='store_true',
                       help='Disable ensemble training')
    parser.add_argument('--ensemble-size', type=int, default=5,
                       help='Ensemble size')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int,
                       help='Batch size (auto-detected if not specified)')
    parser.add_argument('--learning-rate', type=float, default=0.001,
                       help='Learning rate')
    
    # Optimization
    parser.add_argument('--no-hyperparameter-tuning', action='store_true',
                       help='Disable hyperparameter tuning')
    parser.add_argument('--tuning-trials', type=int, default=50,
                       help='Number of hyperparameter tuning trials')
    parser.add_argument('--enable-distributed', action='store_true',
                       help='Enable distributed training')
    parser.add_argument('--world-size', type=int, default=1,
                       help='World size for distributed training')
    
    # Validation
    parser.add_argument('--no-cross-validation', action='store_true',
                       help='Disable cross-validation')
    parser.add_argument('--cv-folds', type=int, default=5,
                       help='Number of cross-validation folds')
    
    # Deployment
    parser.add_argument('--auto-deploy', action='store_true',
                       help='Enable auto-deployment')
    parser.add_argument('--deployment-threshold', type=float, default=0.5,
                       help='Minimum Sharpe ratio for deployment')
    
    # Performance
    parser.add_argument('--max-workers', type=int,
                       help='Maximum number of workers')
    parser.add_argument('--no-mixed-precision', action='store_true',
                       help='Disable mixed precision training')
    
    # Options
    parser.add_argument('--verbose', action='store_true',
                       help='Verbose logging')
    parser.add_argument('--config-file',
                       help='Custom configuration file')
    
    return parser.parse_args()

async def main():
    """Main training function"""
    args = parse_arguments()
    
    # Setup logging
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Load custom configuration if provided
    custom_config = {}
    if args.config_file:
        with open(args.config_file, 'r') as f:
            custom_config = yaml.safe_load(f)
    
    # Create configuration
    config = TrainingConfig(
        training_mode=args.mode,
        symbols=args.symbols,
        lookback_days=args.lookback_days,
        model_types=args.model_types,
        enable_ensemble=not args.no_ensemble,
        ensemble_size=args.ensemble_size,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        enable_hyperparameter_tuning=not args.no_hyperparameter_tuning,
        tuning_trials=args.tuning_trials,
        enable_distributed=args.enable_distributed,
        world_size=args.world_size,
        enable_cross_validation=not args.no_cross_validation,
        cv_folds=args.cv_folds,
        auto_deploy=args.auto_deploy,
        deployment_threshold=args.deployment_threshold,
        max_workers=args.max_workers,
        enable_mixed_precision=not args.no_mixed_precision,
        **custom_config
    )
    
    # Initialize and start training
    engine = ModelTrainingEngine(config)
    
    try:
        print(f"Starting model training...")
        print(f"Mode: {config.training_mode}")
        print(f"Symbols: {', '.join(config.symbols)}")
        print(f"Model Types: {', '.join(config.model_types)}")
        print(f"Epochs: {config.epochs}")
        print(f"Ensemble: {'enabled' if config.enable_ensemble else 'disabled'}")
        print(f"Hyperparameter Tuning: {'enabled' if config.enable_hyperparameter_tuning else 'disabled'}")
        
        result = await engine.start_training()
        
        # Print summary
        print(f"\n=== Training Results ===")
        print(f"Status: {result['status']}")
        print(f"Training ID: {result['training_id']}")
        print(f"Models Trained: {result['stats']['models_trained']}")
        print(f"Best Performance: {result['stats']['best_model_performance']:.4f}")
        print(f"Training Duration: {result['stats']['total_training_time']:.2f}s")
        print(f"Data Points Processed: {result['stats']['data_points_processed']:,}")
        print(f"Validation Accuracy: {result['stats']['validation_accuracy']:.4f}")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())