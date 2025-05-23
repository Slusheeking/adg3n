"""
Distributed Training System for Deep Momentum Trading Models
Supports multi-GPU and multi-node training with ARM64 optimizations.
"""

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import os
import time
import asyncio
from typing import Dict, Any, Optional, List, Tuple
import logging
from datetime import datetime

from deep_momentum_trading.src.training.trainer import Trainer
from deep_momentum_trading.src.models.model_registry import ModelRegistry
from deep_momentum_trading.src.models.arm64_optimizations import ARM64Optimizer
from deep_momentum_trading.src.training.training_data_loader import TrainingDataLoader
from deep_momentum_trading.src.communication.enhanced_zmq_hub import EnhancedZMQHub
from deep_momentum_trading.src.training.training_monitor import TrainingMonitor
from deep_momentum_trading.src.utils.logger import get_logger

logger = get_logger(__name__)

class DistributedTrainer:
    """
    Distributed training coordinator with ARM64 optimizations and real-time monitoring.
    Supports both data parallel and model parallel training strategies.
    """
    
    def __init__(self,
                 model_config: Dict[str, Any],
                 training_config: Dict[str, Any],
                 model_registry: ModelRegistry,
                 arm64_optimizer: ARM64Optimizer,
                 data_loader: TrainingDataLoader,
                 zmq_hub: EnhancedZMQHub,
                 training_monitor: TrainingMonitor):
        """
        Initialize distributed trainer.
        
        Args:
            model_config (Dict[str, Any]): Model configuration
            training_config (Dict[str, Any]): Training configuration
            model_registry (ModelRegistry): Model registry instance
            arm64_optimizer (ARM64Optimizer): ARM64 optimizer instance
            data_loader (TrainingDataLoader): Data loader instance
            zmq_hub (EnhancedZMQHub): ZMQ communication hub
            training_monitor (TrainingMonitor): Training monitor instance
        """
        self.model_config = model_config
        self.training_config = training_config
        self.model_registry = model_registry
        self.arm64_optimizer = arm64_optimizer
        self.data_loader = data_loader
        self.zmq_hub = zmq_hub
        self.training_monitor = training_monitor
        
        # Distributed training settings
        self.world_size = training_config['distributed']['world_size']
        self.backend = training_config['distributed']['backend']
        self.find_unused_parameters = training_config['distributed']['find_unused_parameters']
        self.gradient_as_bucket_view = training_config['distributed']['gradient_as_bucket_view']
        
        # Training state
        self.rank = None
        self.local_rank = None
        self.device = None
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.scaler = None
        
        logger.info(f"DistributedTrainer initialized for world_size: {self.world_size}")
    
    def setup_process_group(self, rank: int, world_size: int):
        """
        Setup distributed process group.
        
        Args:
            rank (int): Process rank
            world_size (int): Total number of processes
        """
        self.rank = rank
        self.world_size = world_size
        self.local_rank = rank % torch.cuda.device_count()
        
        # Set device
        if torch.cuda.is_available():
            self.device = torch.device(f'cuda:{self.local_rank}')
            torch.cuda.set_device(self.local_rank)
        else:
            self.device = torch.device('cpu')
        
        # Initialize process group
        if not dist.is_initialized():
            dist.init_process_group(
                backend=self.backend,
                rank=rank,
                world_size=world_size
            )
        
        logger.info(f"Process group setup complete - Rank: {rank}, Local Rank: {self.local_rank}, Device: {self.device}")
    
    def setup_model(self, model_name: str):
        """
        Setup distributed model with ARM64 optimizations.
        
        Args:
            model_name (str): Name of the model to create
        """
        # Create model
        self.model = self.model_registry.create_model(model_name).to(self.device)
        
        # Apply ARM64 optimizations
        if self.arm64_optimizer:
            self.model = self.arm64_optimizer.optimize_model(
                self.model,
                sample_input=self._get_sample_input(),
                optimization_level=self.training_config.get('arm64', {}).get('optimization_level', 'moderate')
            )
        
        # Wrap with DistributedDataParallel
        self.model = DDP(
            self.model,
            device_ids=[self.local_rank] if torch.cuda.is_available() else None,
            find_unused_parameters=self.find_unused_parameters,
            gradient_as_bucket_view=self.gradient_as_bucket_view
        )
        
        logger.info(f"Distributed model setup complete on rank {self.rank}")
    
    def setup_optimizer_and_scheduler(self):
        """Setup optimizer and learning rate scheduler."""
        # Optimizer
        optimizer_config = self.training_config['optimizer']
        optimizer_class = getattr(torch.optim, optimizer_config['type'])
        
        self.optimizer = optimizer_class(
            self.model.parameters(),
            lr=optimizer_config['lr'],
            weight_decay=optimizer_config.get('weight_decay', 0.0),
            **{k: v for k, v in optimizer_config.items() if k not in ['type', 'lr', 'weight_decay']}
        )
        
        # Scheduler
        scheduler_config = self.training_config.get('scheduler', {})
        if scheduler_config.get('type'):
            scheduler_class = getattr(torch.optim.lr_scheduler, scheduler_config['type'])
            self.scheduler = scheduler_class(
                self.optimizer,
                **{k: v for k, v in scheduler_config.items() if k != 'type'}
            )
        
        # Mixed precision scaler
        if self.training_config.get('arm64', {}).get('use_mixed_precision', False):
            self.scaler = torch.cuda.amp.GradScaler()
        
        logger.info(f"Optimizer and scheduler setup complete on rank {self.rank}")
    
    def setup_data_loaders(self):
        """Setup distributed data loaders."""
        train_loader = self.data_loader.get_train_loader()
        val_loader = self.data_loader.get_val_loader()
        
        # Create distributed samplers
        train_sampler = DistributedSampler(
            train_loader.dataset,
            num_replicas=self.world_size,
            rank=self.rank,
            shuffle=True
        )
        
        val_sampler = DistributedSampler(
            val_loader.dataset,
            num_replicas=self.world_size,
            rank=self.rank,
            shuffle=False
        )
        
        # Create new data loaders with distributed samplers
        self.train_loader = torch.utils.data.DataLoader(
            train_loader.dataset,
            batch_size=train_loader.batch_size,
            sampler=train_sampler,
            num_workers=train_loader.num_workers,
            pin_memory=train_loader.pin_memory,
            prefetch_factor=getattr(train_loader, 'prefetch_factor', 2)
        )
        
        self.val_loader = torch.utils.data.DataLoader(
            val_loader.dataset,
            batch_size=val_loader.batch_size,
            sampler=val_sampler,
            num_workers=val_loader.num_workers,
            pin_memory=val_loader.pin_memory,
            prefetch_factor=getattr(val_loader, 'prefetch_factor', 2)
        )
        
        logger.info(f"Distributed data loaders setup complete on rank {self.rank}")
    
    async def run_training(self, model_name: str) -> Dict[str, Any]:
        """
        Run distributed training session.
        
        Args:
            model_name (str): Name of the model to train
            
        Returns:
            Dict[str, Any]: Training results
        """
        logger.info(f"Starting distributed training for model: {model_name} on rank {self.rank}")
        
        # Setup components
        self.setup_model(model_name)
        self.setup_optimizer_and_scheduler()
        self.setup_data_loaders()
        
        # Training loop
        training_results = await self._training_loop(model_name)
        
        # Cleanup
        self._cleanup()
        
        logger.info(f"Distributed training completed for model: {model_name} on rank {self.rank}")
        return training_results
    
    async def _training_loop(self, model_name: str) -> Dict[str, Any]:
        """Main distributed training loop."""
        epochs = self.training_config['epochs']
        best_val_loss = float('inf')
        patience_counter = 0
        patience = self.training_config.get('early_stopping_patience', 10)
        
        training_metrics = {
            'train_losses': [],
            'val_losses': [],
            'learning_rates': [],
            'epoch_times': []
        }
        
        for epoch in range(epochs):
            epoch_start_time = time.time()
            
            # Set epoch for distributed sampler
            self.train_loader.sampler.set_epoch(epoch)
            
            # Training phase
            train_loss = await self._train_epoch(epoch)
            
            # Validation phase
            val_loss = await self._validate_epoch(epoch)
            
            # Learning rate scheduling
            if self.scheduler:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()
            
            # Record metrics
            epoch_time = time.time() - epoch_start_time
            current_lr = self.optimizer.param_groups[0]['lr']
            
            training_metrics['train_losses'].append(train_loss)
            training_metrics['val_losses'].append(val_loss)
            training_metrics['learning_rates'].append(current_lr)
            training_metrics['epoch_times'].append(epoch_time)
            
            # Early stopping check (only on rank 0)
            if self.rank == 0:
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    
                    # Save best model
                    await self._save_checkpoint(model_name, epoch, val_loss, is_best=True)
                else:
                    patience_counter += 1
                
                # Broadcast early stopping decision
                should_stop = patience_counter >= patience
                stop_tensor = torch.tensor([1 if should_stop else 0], device=self.device)
            else:
                stop_tensor = torch.tensor([0], device=self.device)
            
            # Synchronize early stopping decision across all ranks
            dist.broadcast(stop_tensor, src=0)
            
            if stop_tensor.item() == 1:
                logger.info(f"Early stopping triggered at epoch {epoch} on rank {self.rank}")
                break
            
            # Log progress (only on rank 0)
            if self.rank == 0:
                await self._log_epoch_progress(epoch, train_loss, val_loss, current_lr, epoch_time)
        
        # Final results
        results = {
            'model_name': model_name,
            'best_val_loss': best_val_loss,
            'total_epochs': epoch + 1,
            'training_metrics': training_metrics,
            'final_lr': current_lr
        }
        
        return results
    
    async def _train_epoch(self, epoch: int) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch_idx, (features, targets, prev_positions) in enumerate(self.train_loader):
            features = features.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)
            prev_positions = prev_positions.to(self.device, non_blocking=True)
            
            self.optimizer.zero_grad()
            
            # Forward pass with mixed precision
            if self.scaler:
                with torch.cuda.amp.autocast():
                    positions, confidence, volatility_pred = self.model(features)
                    loss_output = self._compute_loss(positions, targets, confidence, prev_positions, volatility_pred)
                    loss = loss_output['total_loss']
                
                # Backward pass with gradient scaling
                self.scaler.scale(loss).backward()
                
                # Gradient clipping
                if self.training_config.get('gradient_clip_value'):
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.training_config['gradient_clip_value']
                    )
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                positions, confidence, volatility_pred = self.model(features)
                loss_output = self._compute_loss(positions, targets, confidence, prev_positions, volatility_pred)
                loss = loss_output['total_loss']
                
                loss.backward()
                
                # Gradient clipping
                if self.training_config.get('gradient_clip_value'):
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.training_config['gradient_clip_value']
                    )
                
                self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            # Log batch progress
            if batch_idx % self.training_config.get('log_interval', 100) == 0 and self.rank == 0:
                logger.debug(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.6f}")
        
        # Average loss across all processes
        avg_loss = total_loss / num_batches
        loss_tensor = torch.tensor([avg_loss], device=self.device)
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
        avg_loss = loss_tensor.item() / self.world_size
        
        return avg_loss
    
    async def _validate_epoch(self, epoch: int) -> float:
        """Validate for one epoch."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for features, targets, prev_positions in self.val_loader:
                features = features.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)
                prev_positions = prev_positions.to(self.device, non_blocking=True)
                
                if self.scaler:
                    with torch.cuda.amp.autocast():
                        positions, confidence, volatility_pred = self.model(features)
                        loss_output = self._compute_loss(positions, targets, confidence, prev_positions, volatility_pred)
                        loss = loss_output['total_loss']
                else:
                    positions, confidence, volatility_pred = self.model(features)
                    loss_output = self._compute_loss(positions, targets, confidence, prev_positions, volatility_pred)
                    loss = loss_output['total_loss']
                
                total_loss += loss.item()
                num_batches += 1
        
        # Average loss across all processes
        avg_loss = total_loss / num_batches
        loss_tensor = torch.tensor([avg_loss], device=self.device)
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
        avg_loss = loss_tensor.item() / self.world_size
        
        return avg_loss
    
    def _compute_loss(self, positions, targets, confidence, prev_positions, volatility_pred):
        """Compute loss using the model's loss function."""
        # This should use the actual loss function from the model
        # For now, using a simple MSE loss as placeholder
        loss = torch.nn.functional.mse_loss(positions, targets)
        return {'total_loss': loss}
    
    async def _save_checkpoint(self, model_name: str, epoch: int, val_loss: float, is_best: bool = False):
        """Save model checkpoint (only on rank 0)."""
        if self.rank != 0:
            return
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.module.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss,
            'model_config': self.model_config,
            'training_config': self.training_config
        }
        
        if self.scheduler:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        if self.scaler:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        # Save via model registry
        self.model_registry.save_model_checkpoint(
            model_name=model_name,
            model=self.model.module,
            optimizer=self.optimizer,
            epoch=epoch,
            loss=val_loss,
            additional_data=checkpoint
        )
        
        logger.info(f"Checkpoint saved for {model_name} at epoch {epoch}")
    
    async def _log_epoch_progress(self, epoch: int, train_loss: float, val_loss: float, lr: float, epoch_time: float):
        """Log epoch progress and send to monitoring system."""
        log_msg = (f"Epoch {epoch}: Train Loss: {train_loss:.6f}, "
                  f"Val Loss: {val_loss:.6f}, LR: {lr:.2e}, Time: {epoch_time:.2f}s")
        logger.info(log_msg)
        
        # Send metrics to monitoring system
        if self.training_monitor:
            await self.training_monitor.log_epoch_metrics({
                'epoch': epoch,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'learning_rate': lr,
                'epoch_time': epoch_time,
                'rank': self.rank
            })
    
    def _get_sample_input(self) -> torch.Tensor:
        """Get sample input for ARM64 optimization."""
        # This should match the expected input shape for the model
        batch_size = self.training_config.get('batch_size', 32)
        sequence_length = self.data_loader.config.get('sequence_length', 60)
        input_size = self.model_config['parameters'].get('input_size', 200)
        
        return torch.randn(batch_size, sequence_length, input_size, device=self.device)
    
    def _cleanup(self):
        """Cleanup distributed training resources."""
        if dist.is_initialized():
            dist.destroy_process_group()
        
        logger.info(f"Distributed training cleanup complete on rank {self.rank}")

def run_distributed_training(rank: int, world_size: int, trainer_config: Dict[str, Any]):
    """
    Entry point for distributed training process.
    
    Args:
        rank (int): Process rank
        world_size (int): Total number of processes
        trainer_config (Dict[str, Any]): Trainer configuration
    """
    # Setup environment
    os.environ['RANK'] = str(rank)
    os.environ['WORLD_SIZE'] = str(world_size)
    
    # Create trainer instance
    trainer = DistributedTrainer(**trainer_config)
    
    # Setup process group
    trainer.setup_process_group(rank, world_size)
    
    # Run training
    asyncio.run(trainer.run_training(trainer_config['model_name']))

if __name__ == "__main__":
    # Example usage for testing
    import torch.multiprocessing as mp
    
    def test_distributed_training():
        world_size = 2
        
        # Mock configuration
        trainer_config = {
            'model_config': {'model_type': 'deep_momentum_lstm', 'parameters': {'input_size': 200}},
            'training_config': {
                'distributed': {'world_size': world_size, 'backend': 'nccl'},
                'epochs': 5,
                'optimizer': {'type': 'Adam', 'lr': 0.001},
                'batch_size': 32
            },
            'model_name': 'test_model'
        }
        
        # Launch distributed training
        mp.spawn(
            run_distributed_training,
            args=(world_size, trainer_config),
            nprocs=world_size,
            join=True
        )
    
    # Uncomment to test
    # test_distributed_training()