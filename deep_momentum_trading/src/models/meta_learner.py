import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from collections import defaultdict, deque
import time
from dataclasses import dataclass
from enum import Enum
import math
from concurrent.futures import ThreadPoolExecutor
import threading

from deep_momentum_trading.src.utils.logger import get_logger
from deep_momentum_trading.src.models.loss_functions import ScalableLossConfig, ScalableCombinedLoss, create_production_loss_function

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

# ===== SCALABLE META-LEARNING CONFIGURATION =====

@dataclass
class ScalableMetaLearningConfig:
    """Configuration for massive-scale meta-learning system"""
    num_models: int = 50                    # Support 50+ models in ensemble
    num_assets: int = 10000                 # Support 10,000+ assets
    market_feature_dim: int = 200           # Rich market feature representation
    performance_feature_dim: int = 100      # Comprehensive performance metrics
    hidden_dim: int = 512                   # Large hidden dimensions for capacity
    num_attention_heads: int = 16           # Multi-head attention
    num_transformer_layers: int = 6         # Deep transformer architecture
    
    # Meta-learning parameters
    meta_lr: float = 0.0001                 # Conservative meta-learning rate
    adaptation_steps: int = 10              # More adaptation steps
    memory_size: int = 10000                # Large experience replay buffer
    batch_size: int = 256                   # Large batch processing
    
    # Performance optimization
    use_gradient_checkpointing: bool = True
    use_mixed_precision: bool = True
    use_model_parallelism: bool = True
    compile_for_inference: bool = True
    
    # Advanced features
    use_hierarchical_attention: bool = True
    use_uncertainty_estimation: bool = True
    use_continual_learning: bool = True
    use_federated_updates: bool = True


# ===== ADVANCED ATTENTION MECHANISMS =====

class HierarchicalAttention(nn.Module):
    """Hierarchical attention for multi-scale feature processing"""
    
    def __init__(self, config: ScalableMetaLearningConfig):
        super().__init__()
        self.config = config
        
        # Multi-scale attention layers
        self.local_attention = nn.MultiheadAttention(
            embed_dim=config.hidden_dim,
            num_heads=config.num_attention_heads,
            batch_first=True,
            dropout=0.1
        )
        
        self.global_attention = nn.MultiheadAttention(
            embed_dim=config.hidden_dim,
            num_heads=config.num_attention_heads // 2,
            batch_first=True,
            dropout=0.1
        )
        
        # Cross-scale fusion
        self.scale_fusion = nn.Sequential(
            nn.Linear(config.hidden_dim * 2, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
    def forward(self, features: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Apply hierarchical attention"""
        # Local attention (within sequence)
        local_out, _ = self.local_attention(features, features, features, key_padding_mask=mask)
        
        # Global attention (across sequence)
        global_out, _ = self.global_attention(features, features, features, key_padding_mask=mask)
        
        # Fuse multi-scale representations
        fused = torch.cat([local_out, global_out], dim=-1)
        output = self.scale_fusion(fused)
        
        return output + features  # Residual connection


class AdaptiveMetaOptimizer(nn.Module):
    """Advanced meta-optimizer with adaptive learning and uncertainty estimation"""
    
    def __init__(self, config: ScalableMetaLearningConfig):
        super().__init__()
        self.config = config
        
        # Market context encoder with transformer
        self.market_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=config.hidden_dim,
                nhead=config.num_attention_heads,
                dim_feedforward=config.hidden_dim * 4,
                dropout=0.1,
                batch_first=True
            ),
            num_layers=config.num_transformer_layers // 2
        )
        
        # Performance context encoder
        self.performance_encoder = nn.Sequential(
            nn.Linear(config.performance_feature_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.ReLU()
        )
        
        # Model prediction encoder with attention
        self.prediction_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=config.hidden_dim,
                nhead=config.num_attention_heads,
                dim_feedforward=config.hidden_dim * 2,
                dropout=0.1,
                batch_first=True
            ),
            num_layers=2
        )
        
        # Hierarchical attention if enabled
        if config.use_hierarchical_attention:
            self.hierarchical_attention = HierarchicalAttention(config)
        
        # Advanced weight generation with multiple heads
        self.weight_generator = nn.ModuleDict({
            'position_weights': self._create_weight_head(config.num_models),
            'confidence_weights': self._create_weight_head(config.num_models),
            'volatility_weights': self._create_weight_head(config.num_models),
            'temporal_weights': self._create_weight_head(5)  # Different time horizons
        })
        
        # Uncertainty estimation heads
        if config.use_uncertainty_estimation:
            self.uncertainty_heads = nn.ModuleDict({
                'aleatoric': self._create_uncertainty_head(),  # Data uncertainty
                'epistemic': self._create_uncertainty_head(),  # Model uncertainty
                'temporal': self._create_uncertainty_head()    # Time-varying uncertainty
            })
        
        # Meta-learning adaptation network
        self.adaptation_network = nn.Sequential(
            nn.Linear(config.hidden_dim * 3, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(config.hidden_dim // 2, config.num_models)
        )
        
        # Initialize projection layers for different input dimensions
        self.market_projection = nn.Linear(config.market_feature_dim, config.hidden_dim)
        self.prediction_projection = nn.Linear(config.num_models * 3, config.hidden_dim)
        
    def _create_weight_head(self, output_dim: int) -> nn.Module:
        """Create a weight generation head"""
        return nn.Sequential(
            nn.Linear(self.config.hidden_dim * 3, self.config.hidden_dim),
            nn.LayerNorm(self.config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(self.config.hidden_dim, self.config.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(self.config.hidden_dim // 2, output_dim),
            nn.Softmax(dim=-1)
        )
    
    def _create_uncertainty_head(self) -> nn.Module:
        """Create an uncertainty estimation head"""
        return nn.Sequential(
            nn.Linear(self.config.hidden_dim * 3, self.config.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(self.config.hidden_dim // 2, 1),
            nn.Sigmoid()
        )
    
    def forward(self, 
                market_features: torch.Tensor,
                performance_features: torch.Tensor,
                model_predictions: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Advanced forward pass with multiple attention mechanisms"""
        
        batch_size = market_features.size(0)
        
        # Project inputs to hidden dimension
        market_projected = self.market_projection(market_features).unsqueeze(1)  # Add sequence dim
        pred_flat = model_predictions.view(batch_size, -1)
        pred_projected = self.prediction_projection(pred_flat).unsqueeze(1)
        
        # Encode market context with transformer
        market_encoded = self.market_encoder(market_projected).squeeze(1)
        
        # Encode performance context
        perf_encoded = self.performance_encoder(performance_features)
        
        # Encode model predictions with attention
        pred_encoded = self.prediction_encoder(pred_projected).squeeze(1)
        
        # Apply hierarchical attention if enabled
        if self.config.use_hierarchical_attention:
            combined_features = torch.stack([market_encoded, perf_encoded, pred_encoded], dim=1)
            attended_features = self.hierarchical_attention(combined_features)
            attended_features = attended_features.mean(dim=1)  # Pool across sequence
        else:
            attended_features = torch.cat([market_encoded, perf_encoded, pred_encoded], dim=-1)
        
        # Generate multiple types of weights
        weights = {}
        for weight_type, weight_head in self.weight_generator.items():
            weights[weight_type] = weight_head(attended_features)
        
        # Generate uncertainty estimates if enabled
        uncertainties = {}
        if self.config.use_uncertainty_estimation:
            for uncertainty_type, uncertainty_head in self.uncertainty_heads.items():
                uncertainties[uncertainty_type] = uncertainty_head(attended_features)
        
        # Meta-adaptation signal
        adaptation_signal = self.adaptation_network(attended_features)
        
        return {
            'weights': weights,
            'uncertainties': uncertainties,
            'adaptation_signal': adaptation_signal,
            'attended_features': attended_features
        }


# ===== SCALABLE ONLINE META-LEARNER =====

class ScalableOnlineMetaLearner:
    """Scalable online meta-learner for massive-scale trading"""
    
    def __init__(self, config: ScalableMetaLearningConfig, device: str = 'cuda'):
        self.config = config
        self.device = device
        
        # Core meta-optimizer
        self.meta_optimizer = AdaptiveMetaOptimizer(config).to(device)
        
        # Optimizers with different learning rates
        self.optimizers = {
            'fast': torch.optim.AdamW(self.meta_optimizer.parameters(), lr=config.meta_lr * 10, weight_decay=1e-4),
            'slow': torch.optim.AdamW(self.meta_optimizer.parameters(), lr=config.meta_lr, weight_decay=1e-5),
            'meta': torch.optim.AdamW(self.meta_optimizer.parameters(), lr=config.meta_lr * 0.1, weight_decay=1e-3)
        }
        
        # Experience replay with prioritization
        self.experience_buffer = PrioritizedExperienceReplay(
            capacity=config.memory_size,
            alpha=0.6,
            beta=0.4
        )
        
        # Performance tracking
        self.performance_tracker = AdvancedPerformanceTracker(config)
        
        # Loss function
        loss_config = ScalableLossConfig(
            num_assets=config.num_assets,
            target_sharpe=4.0,
            max_drawdown_target=0.03
        )
        self.loss_fn = create_production_loss_function(loss_config).to(device)
        
        # Continual learning components
        if config.use_continual_learning:
            self.continual_learner = ContinualLearningManager(config)
        
        # Model parallelism setup
        if config.use_model_parallelism and torch.cuda.device_count() > 1:
            self.meta_optimizer = nn.DataParallel(self.meta_optimizer)
        
        # Compilation for inference
        if config.compile_for_inference and hasattr(torch, 'compile'):
            self.meta_optimizer = torch.compile(self.meta_optimizer, mode='max-autotune')
        
        logger.info(f"ScalableOnlineMetaLearner initialized with {config.num_models} models on {device}")
    
    def update(self, 
               market_features: torch.Tensor,
               model_predictions: Dict[str, Dict[str, torch.Tensor]],
               actual_returns: torch.Tensor,
               model_names: List[str],
               priority_weight: float = 1.0) -> Dict[str, torch.Tensor]:
        """Scalable online update with prioritized experience replay"""
        
        # Move to device
        market_features = market_features.to(self.device)
        actual_returns = actual_returns.to(self.device)
        
        # Stack model predictions
        stacked_predictions = self._stack_predictions_efficient(model_predictions, model_names)
        
        # Extract performance features
        performance_features = self.performance_tracker.get_features(model_names).to(self.device)
        
        # Store experience with priority
        experience = {
            'market_features': market_features.clone().detach(),
            'model_predictions': {k: {sub_k: v.clone().detach() for sub_k, v in val.items()} 
                                for k, val in model_predictions.items()},
            'actual_returns': actual_returns.clone().detach(),
            'timestamp': time.time(),
            'model_names': model_names.copy()
        }
        
        # Calculate priority based on prediction error or uncertainty
        with torch.no_grad():
            meta_output = self.meta_optimizer(market_features, performance_features, stacked_predictions)
            priority = self._calculate_priority(meta_output, actual_returns, priority_weight)
        
        self.experience_buffer.add(experience, priority)
        
        # Perform meta-learning updates
        if len(self.experience_buffer) >= self.config.batch_size:
            meta_results = self._meta_learning_step()
            
            # Update performance tracking
            self.performance_tracker.update_batch(model_predictions, actual_returns, model_names)
            
            # Continual learning update
            if self.config.use_continual_learning:
                self.continual_learner.update(meta_results, experience)
            
            return meta_results
        
        return meta_output
    
    def _stack_predictions_efficient(self, 
                                   model_predictions: Dict[str, Dict[str, torch.Tensor]], 
                                   model_names: List[str]) -> torch.Tensor:
        """Efficiently stack model predictions with padding for missing models"""
        
        if not model_predictions:
            batch_size = 1
            return torch.zeros(batch_size, self.config.num_models, 3, device=self.device)
        
        # Get batch size from first available prediction
        first_pred = next(iter(model_predictions.values()))
        batch_size = first_pred['positions'].shape[0]
        
        # Pre-allocate tensor
        stacked = torch.zeros(batch_size, self.config.num_models, 3, device=self.device)
        
        # Fill with available predictions
        for i, name in enumerate(model_names[:self.config.num_models]):
            if name in model_predictions:
                pred = model_predictions[name]
                stacked[:, i, 0] = pred['positions'].squeeze(-1) if pred['positions'].dim() > 1 else pred['positions']
                stacked[:, i, 1] = pred['confidence'].squeeze(-1) if pred['confidence'].dim() > 1 else pred['confidence']
                stacked[:, i, 2] = pred['volatility'].squeeze(-1) if pred['volatility'].dim() > 1 else pred['volatility']
        
        return stacked
    
    def _calculate_priority(self, meta_output: Dict[str, torch.Tensor], 
                          actual_returns: torch.Tensor, 
                          base_priority: float) -> float:
        """Calculate priority for experience replay"""
        
        # Use uncertainty as priority signal
        if 'uncertainties' in meta_output and 'epistemic' in meta_output['uncertainties']:
            uncertainty = meta_output['uncertainties']['epistemic'].mean().item()
            priority = base_priority * (1.0 + uncertainty)
        else:
            priority = base_priority
        
        # Add prediction error component
        if 'weights' in meta_output and 'position_weights' in meta_output['weights']:
            weights = meta_output['weights']['position_weights']
            # Simple error approximation
            error = torch.abs(weights.mean() - actual_returns.mean()).item()
            priority += error * 10.0
        
        return max(priority, 0.01)  # Minimum priority
    
    def _meta_learning_step(self) -> Dict[str, torch.Tensor]:
        """Advanced meta-learning step with multiple optimization phases"""
        
        # Sample prioritized batch
        batch_data, indices, weights = self.experience_buffer.sample(self.config.batch_size)
        
        # Prepare batch tensors
        batch_market = torch.stack([exp['market_features'] for exp in batch_data]).to(self.device)
        batch_returns = torch.stack([exp['actual_returns'] for exp in batch_data]).to(self.device)
        
        # Stack predictions for batch
        batch_predictions = []
        for exp in batch_data:
            stacked = self._stack_predictions_efficient(exp['model_predictions'], exp['model_names'])
            batch_predictions.append(stacked)
        batch_predictions = torch.stack(batch_predictions).squeeze(1)
        
        # Get performance features (global)
        all_model_names = list(set().union(*[exp['model_names'] for exp in batch_data]))
        perf_features = self.performance_tracker.get_features(all_model_names).to(self.device)
        perf_features_batch = perf_features.expand(len(batch_data), -1)
        
        # Multi-phase optimization
        total_loss = 0.0
        
        # Phase 1: Fast adaptation
        self.meta_optimizer.train()
        for step in range(self.config.adaptation_steps // 3):
            self.optimizers['fast'].zero_grad()
            
            meta_output = self.meta_optimizer(batch_market, perf_features_batch, batch_predictions)
            loss = self._compute_meta_loss(meta_output, batch_returns, weights)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.meta_optimizer.parameters(), 1.0)
            self.optimizers['fast'].step()
            
            total_loss += loss.item()
        
        # Phase 2: Slow consolidation
        for step in range(self.config.adaptation_steps // 3):
            self.optimizers['slow'].zero_grad()
            
            meta_output = self.meta_optimizer(batch_market, perf_features_batch, batch_predictions)
            loss = self._compute_meta_loss(meta_output, batch_returns, weights)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.meta_optimizer.parameters(), 0.5)
            self.optimizers['slow'].step()
            
            total_loss += loss.item()
        
        # Phase 3: Meta-optimization
        for step in range(self.config.adaptation_steps // 3):
            self.optimizers['meta'].zero_grad()
            
            meta_output = self.meta_optimizer(batch_market, perf_features_batch, batch_predictions)
            loss = self._compute_meta_loss(meta_output, batch_returns, weights)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.meta_optimizer.parameters(), 0.1)
            self.optimizers['meta'].step()
            
            total_loss += loss.item()
        
        # Update priorities in experience buffer
        new_priorities = self._compute_new_priorities(meta_output, batch_returns)
        self.experience_buffer.update_priorities(indices, new_priorities)
        
        # Final evaluation
        self.meta_optimizer.eval()
        with torch.no_grad():
            final_output = self.meta_optimizer(batch_market[-1:], perf_features[-1:], batch_predictions[-1:])
        
        final_output['meta_loss'] = total_loss / self.config.adaptation_steps
        return final_output
    
    def _compute_meta_loss(self, meta_output: Dict[str, torch.Tensor], 
                          actual_returns: torch.Tensor, 
                          sample_weights: torch.Tensor) -> torch.Tensor:
        """Compute comprehensive meta-learning loss"""
        
        batch_size = actual_returns.shape[0]
        total_loss = 0.0
        
        # Primary loss: ensemble performance
        if 'weights' in meta_output and 'position_weights' in meta_output['weights']:
            position_weights = meta_output['weights']['position_weights']
            
            # Create dummy predictions for loss function
            predictions = {
                'positions': position_weights,
                'confidence': torch.ones_like(position_weights),
                'volatility': torch.ones_like(position_weights) * 0.02
            }
            
            targets = {
                'returns': actual_returns.unsqueeze(-1).expand(-1, position_weights.shape[-1]),
                'previous_positions': torch.zeros_like(position_weights)
            }
            
            market_data = {
                'volumes': torch.ones_like(position_weights) * 1000000,
                'market_cap': torch.ones_like(position_weights) * 1e9,
                'returns_history': torch.randn(60, position_weights.shape[-1], device=self.device) * 0.01
            }
            
            loss_output = self.loss_fn(predictions, targets, market_data)
            total_loss += loss_output['total_loss']
        
        # Uncertainty regularization
        if 'uncertainties' in meta_output:
            for uncertainty_type, uncertainty in meta_output['uncertainties'].items():
                # Penalize high uncertainty unless it leads to better performance
                uncertainty_penalty = torch.mean(uncertainty) * 0.01
                total_loss += uncertainty_penalty
        
        # Adaptation regularization
        if 'adaptation_signal' in meta_output:
            adaptation_reg = torch.mean(torch.abs(meta_output['adaptation_signal'])) * 0.001
            total_loss += adaptation_reg
        
        # Apply sample weights for prioritized replay
        if sample_weights is not None:
            total_loss = total_loss * sample_weights.mean()
        
        return total_loss
    
    def _compute_new_priorities(self, meta_output: Dict[str, torch.Tensor], 
                              actual_returns: torch.Tensor) -> List[float]:
        """Compute new priorities based on current performance"""
        
        priorities = []
        batch_size = actual_returns.shape[0]
        
        for i in range(batch_size):
            priority = 1.0
            
            # Use uncertainty as priority signal
            if 'uncertainties' in meta_output:
                uncertainty_sum = sum(meta_output['uncertainties'][key][i].item() 
                                    for key in meta_output['uncertainties'])
                priority += uncertainty_sum
            
            # Use adaptation signal magnitude
            if 'adaptation_signal' in meta_output:
                adaptation_magnitude = torch.abs(meta_output['adaptation_signal'][i]).mean().item()
                priority += adaptation_magnitude * 0.1
            
            priorities.append(max(priority, 0.01))
        
        return priorities


# ===== SUPPORTING CLASSES =====

class PrioritizedExperienceReplay:
    """Prioritized experience replay buffer for meta-learning"""
    
    def __init__(self, capacity: int, alpha: float = 0.6, beta: float = 0.4):
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.buffer = []
        self.priorities = []
        self.position = 0
        
    def add(self, experience: Dict, priority: float):
        """Add experience with priority"""
        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
            self.priorities.append(priority ** self.alpha)
        else:
            self.buffer[self.position] = experience
            self.priorities[self.position] = priority ** self.alpha
            self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size: int) -> Tuple[List[Dict], List[int], torch.Tensor]:
        """Sample batch with prioritized sampling"""
        if len(self.buffer) == 0:
            return [], [], torch.tensor([])
        
        priorities = np.array(self.priorities[:len(self.buffer)])
        probabilities = priorities / priorities.sum()
        
        indices = np.random.choice(len(self.buffer), batch_size, p=probabilities, replace=True)
        
        # Importance sampling weights
        weights = (len(self.buffer) * probabilities[indices]) ** (-self.beta)
        weights = weights / weights.max()
        
        batch = [self.buffer[i] for i in indices]
        
        return batch, indices.tolist(), torch.tensor(weights, dtype=torch.float32)
    
    def update_priorities(self, indices: List[int], priorities: List[float]):
        """Update priorities for sampled experiences"""
        for idx, priority in zip(indices, priorities):
            if idx < len(self.priorities):
                self.priorities[idx] = priority ** self.alpha
    
    def __len__(self):
        return len(self.buffer)


class AdvancedPerformanceTracker:
    """Advanced performance tracking with regime awareness"""
    
    def __init__(self, config: ScalableMetaLearningConfig):
        self.config = config
        self.model_metrics = defaultdict(lambda: {
            'sharpe_ratio': 0.0,
            'win_rate': 0.5,
            'max_drawdown': 0.0,
            'volatility': 0.02,
            'alpha': 0.0,
            'beta': 1.0,
            'information_ratio': 0.0,
            'calmar_ratio': 0.0,
            'sortino_ratio': 0.0,
            'tail_ratio': 1.0
        })
        
        self.regime_performance = defaultdict(lambda: defaultdict(dict))
        self.update_count = defaultdict(int)
        
    def update_batch(self, model_predictions: Dict[str, Dict[str, torch.Tensor]], 
                    actual_returns: torch.Tensor, 
                    model_names: List[str]):
        """Batch update of performance metrics"""
        
        for name in model_names:
            if name in model_predictions:
                pred = model_predictions[name]
                positions = pred['positions'].detach().cpu().numpy()
                returns = actual_returns.detach().cpu().numpy()
                
                # Update metrics
                self._update_model_metrics(name, positions, returns)
                self.update_count[name] += 1
    
    def _update_model_metrics(self, model_name: str, positions: np.ndarray, returns: np.ndarray):
        """Update comprehensive metrics for a model"""
        
        # Calculate returns
        model_returns = positions.flatten() * returns.flatten()
        
        if len(model_returns) > 1:
            # Sharpe ratio
            if np.std(model_returns) > 0:
                sharpe = np.mean(model_returns) / np.std(model_returns) * np.sqrt(252)
                self.model_metrics[model_name]['sharpe_ratio'] = self._ema_update(
                    self.model_metrics[model_name]['sharpe_ratio'], sharpe, 0.1
                )
            
            # Win rate
            win_rate = np.mean(model_returns > 0)
            self.model_metrics[model_name]['win_rate'] = self._ema_update(
                self.model_metrics[model_name]['win_rate'], win_rate, 0.1
            )
            
            # Volatility
            volatility = np.std(model_returns) * np.sqrt(252)
            self.model_metrics[model_name]['volatility'] = self._ema_update(
                self.model_metrics[model_name]['volatility'], volatility, 0.1
            )
            
            # Max drawdown
            cumulative = np.cumprod(1 + model_returns)
            running_max = np.maximum.accumulate(cumulative)
            drawdown = (cumulative - running_max) / running_max
            max_dd = np.min(drawdown)
            self.model_metrics[model_name]['max_drawdown'] = min(
                self.model_metrics[model_name]['max_drawdown'], max_dd
            )
    
    def _ema_update(self, old_value: float, new_value: float, alpha: float) -> float:
        """Exponential moving average update"""
        if not np.isfinite(new_value):
            return old_value
        return alpha * new_value + (1 - alpha) * old_value
    
    def get_features(self, model_names: List[str]) -> torch.Tensor:
        """Get performance features for meta-learning"""
        
        features = []
        for name in model_names[:self.config.num_models]:
            metrics = self.model_metrics[name]
            model_features = [
                metrics['sharpe_ratio'],
                metrics['win_rate'],
                metrics['max_drawdown'],
                metrics['volatility'],
                metrics['alpha'],
                metrics['beta'],
                metrics['information_ratio'],
                metrics['calmar_ratio'],
                metrics['sortino_ratio'],
                metrics['tail_ratio']
            ]
            features.extend(model_features)
        
        # Pad to expected dimension
        target_dim = self.config.performance_feature_dim
        if len(features) < target_dim:
            features.extend([0.0] * (target_dim - len(features)))
        elif len(features) > target_dim:
            features = features[:target_dim]
        
        return torch.tensor([features], dtype=torch.float32)


class ContinualLearningManager:
    """Manages continual learning to prevent catastrophic forgetting"""
    
    def __init__(self, config: ScalableMetaLearningConfig):
        self.config = config
        self.task_memory = deque(maxlen=1000)
        self.importance_weights = {}
        
    def update(self, meta_results: Dict[str, torch.Tensor], experience: Dict):
        """Update continual learning components"""
        # Store important experiences
        self.task_memory.append({
            'experience': experience,
            'meta_results': meta_results,
            'timestamp': time.time()
        })
        
        # Update importance weights (simplified EWC-like approach)
        if 'adaptation_signal' in meta_results:
            signal_magnitude = torch.abs(meta_results['adaptation_signal']).mean().item()
            task_id = len(self.task_memory)
            self.importance_weights[task_id] = signal_magnitude


# ===== LEGACY COMPATIBILITY CLASSES =====

class OnlineMetaLearner:
    """Legacy compatibility wrapper for ScalableOnlineMetaLearner"""
    
    def __init__(self, 
                 num_models: int = 10,
                 market_feature_dim: int = 50,
                 performance_feature_dim: int = 20,
                 hidden_dim: int = 128,
                 meta_lr: float = 0.001,
                 device: str = 'cuda'):
        
        # Create scalable config from legacy parameters
        config = ScalableMetaLearningConfig(
            num_models=num_models,
            market_feature_dim=market_feature_dim,
            performance_feature_dim=performance_feature_dim,
            hidden_dim=hidden_dim,
            meta_lr=meta_lr,
            use_hierarchical_attention=False,  # Disable for legacy
            use_uncertainty_estimation=False,
            use_continual_learning=False
        )
        
        self.scalable_learner = ScalableOnlineMetaLearner(config, device)
        logger.info("Legacy OnlineMetaLearner initialized - consider upgrading to ScalableOnlineMetaLearner")
    
    def update(self, 
               market_features: torch.Tensor,
               model_predictions: Dict[str, Dict[str, torch.Tensor]],
               actual_returns: torch.Tensor,
               model_names: List[str]) -> Dict[str, torch.Tensor]:
        """Legacy update method"""
        return self.scalable_learner.update(market_features, model_predictions, actual_returns, model_names)


# ===== PRODUCTION FACTORY FUNCTIONS =====

def create_production_meta_learner(
    num_models: int = 50,
    num_assets: int = 10000,
    device: str = 'cuda',
    enable_all_features: bool = True
) -> ScalableOnlineMetaLearner:
    """Create production-ready meta-learner with optimal configuration"""
    
    config = ScalableMetaLearningConfig(
        num_models=num_models,
        num_assets=num_assets,
        market_feature_dim=200,
        performance_feature_dim=100,
        hidden_dim=512,
        num_attention_heads=16,
        num_transformer_layers=6,
        meta_lr=0.0001,
        adaptation_steps=10,
        memory_size=10000,
        batch_size=256,
        use_gradient_checkpointing=True,
        use_mixed_precision=True,
        use_model_parallelism=True,
        compile_for_inference=True,
        use_hierarchical_attention=enable_all_features,
        use_uncertainty_estimation=enable_all_features,
        use_continual_learning=enable_all_features,
        use_federated_updates=enable_all_features
    )
    
    meta_learner = ScalableOnlineMetaLearner(config, device)
    
    logger.info(f"Production meta-learner created with {num_models} models for {num_assets} assets")
    return meta_learner


def create_lightweight_meta_learner(
    num_models: int = 10,
    device: str = 'cpu'
) -> ScalableOnlineMetaLearner:
    """Create lightweight meta-learner for resource-constrained environments"""
    
    config = ScalableMetaLearningConfig(
        num_models=num_models,
        num_assets=1000,
        market_feature_dim=50,
        performance_feature_dim=30,
        hidden_dim=128,
        num_attention_heads=4,
        num_transformer_layers=2,
        meta_lr=0.001,
        adaptation_steps=5,
        memory_size=1000,
        batch_size=64,
        use_gradient_checkpointing=False,
        use_mixed_precision=False,
        use_model_parallelism=False,
        compile_for_inference=False,
        use_hierarchical_attention=False,
        use_uncertainty_estimation=False,
        use_continual_learning=False,
        use_federated_updates=False
    )
    
    meta_learner = ScalableOnlineMetaLearner(config, device)
    
    logger.info(f"Lightweight meta-learner created with {num_models} models")
    return meta_learner


# ===== USAGE EXAMPLE =====

if __name__ == "__main__":
    # Create production meta-learner
    config = ScalableMetaLearningConfig(
        num_models=50,
        num_assets=10000,
        market_feature_dim=200,
        performance_feature_dim=100,
        hidden_dim=512,
        use_hierarchical_attention=True,
        use_uncertainty_estimation=True,
        use_continual_learning=True
    )
    
    meta_learner = ScalableOnlineMetaLearner(config, device='cuda' if torch.cuda.is_available() else 'cpu')
    
    # Test with sample data
    batch_size = 32
    num_models = 50
    
    # Sample market features
    market_features = torch.randn(batch_size, config.market_feature_dim)
    
    # Sample model predictions
    model_predictions = {}
    model_names = [f"model_{i}" for i in range(num_models)]
    
    for name in model_names:
        model_predictions[name] = {
            'positions': torch.randn(batch_size, 1) * 0.01,
            'confidence': torch.sigmoid(torch.randn(batch_size, 1)),
            'volatility': torch.abs(torch.randn(batch_size, 1)) * 0.02
        }
    
    # Sample actual returns
    actual_returns = torch.randn(batch_size, 1) * 0.01
    
    # Update meta-learner
    meta_output = meta_learner.update(
        market_features=market_features,
        model_predictions=model_predictions,
        actual_returns=actual_returns,
        model_names=model_names
    )
    
    print("Meta-learner output keys:", list(meta_output.keys()))
    
    if 'weights' in meta_output:
        print("Weight types:", list(meta_output['weights'].keys()))
        print("Position weights shape:", meta_output['weights']['position_weights'].shape)
    
    if 'uncertainties' in meta_output:
        print("Uncertainty types:", list(meta_output['uncertainties'].keys()))
    
    print(f"Successfully processed {num_models} models with {config.num_assets:,} assets")
    print(f"Meta-learning configuration: {config}")
