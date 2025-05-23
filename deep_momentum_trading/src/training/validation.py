import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.cuda.amp import autocast
from typing import Dict, Any, Optional, List, Tuple, Union, Callable
import numpy as np
import pandas as pd
import math
import time
from pathlib import Path
from dataclasses import dataclass, asdict
import json
import warnings
from concurrent.futures import ThreadPoolExecutor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

from deep_momentum_trading.src.utils.logger import get_logger
from deep_momentum_trading.src.models.loss_functions import (
    SharpeOptimizedLoss, TurnoverRegularization, RiskAdjustedLoss,
    InformationRatioLoss, CalmarRatioLoss, SortinoRatioLoss
)
from deep_momentum_trading.src.models.arm64_optimizations import ARM64Optimizer

logger = get_logger(__name__)

@dataclass
class ValidationMetrics:
    """Container for comprehensive validation metrics."""
    # Basic metrics
    average_loss: float
    mse: float
    mae: float
    
    # Financial performance metrics
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    information_ratio: float
    
    # Return metrics
    total_return: float
    annualized_return: float
    annualized_volatility: float
    
    # Risk metrics
    max_drawdown: float
    var_95: float
    var_99: float
    expected_shortfall: float
    downside_deviation: float
    
    # Trading metrics
    turnover: float
    transaction_costs: float
    hit_ratio: float
    profit_factor: float
    
    # Statistical metrics
    skewness: float
    kurtosis: float
    tracking_error: float
    
    # Model-specific metrics
    confidence_accuracy: float
    volatility_prediction_error: float
    position_consistency: float
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return asdict(self)
    
    def save_json(self, filepath: str) -> None:
        """Save metrics to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

@dataclass
class CrossValidationResult:
    """Results from cross-validation."""
    fold_metrics: List[ValidationMetrics]
    mean_metrics: ValidationMetrics
    std_metrics: Dict[str, float]
    best_fold: int
    worst_fold: int
    stability_score: float
    
    def save_json(self, filepath: str) -> None:
        """Save cross-validation results to JSON file."""
        result_dict = {
            'fold_metrics': [m.to_dict() for m in self.fold_metrics],
            'mean_metrics': self.mean_metrics.to_dict(),
            'std_metrics': self.std_metrics,
            'best_fold': self.best_fold,
            'worst_fold': self.worst_fold,
            'stability_score': self.stability_score
        }
        
        with open(filepath, 'w') as f:
            json.dump(result_dict, f, indent=2)

class AdvancedModelValidator:
    """
    Advanced model validation with ARM64 optimizations, comprehensive metrics,
    and sophisticated validation techniques.
    """

    def __init__(self, 
                 device: torch.device = None,
                 use_arm64: bool = True,
                 use_mixed_precision: bool = True,
                 validation_config: Dict[str, Any] = None):
        """
        Initialize the Advanced Model Validator.

        Args:
            device: Device for computation
            use_arm64: Whether to use ARM64 optimizations
            use_mixed_precision: Whether to use mixed precision
            validation_config: Configuration for validation
        """
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.use_arm64 = use_arm64
        self.use_mixed_precision = use_mixed_precision
        self.config = validation_config or {}
        
        # ARM64 optimizations
        if self.use_arm64:
            self.arm64_optimizer = ARM64Optimizer()
        
        # Validation parameters
        self.confidence_threshold = self.config.get("confidence_threshold", 0.5)
        self.risk_free_rate = self.config.get("risk_free_rate", 0.02)
        self.transaction_cost_rate = self.config.get("transaction_cost_rate", 0.001)
        self.benchmark_return = self.config.get("benchmark_return", 0.0)
        
        logger.info("AdvancedModelValidator initialized")
        logger.info(f"Device: {self.device}")
        logger.info(f"ARM64 optimizations: {'enabled' if self.use_arm64 else 'disabled'}")

    def evaluate_model(self,
                      model: nn.Module,
                      data_loader: DataLoader,
                      loss_function: nn.Module,
                      return_predictions: bool = False) -> Union[ValidationMetrics, Tuple[ValidationMetrics, Dict[str, torch.Tensor]]]:
        """
        Comprehensive model evaluation with advanced metrics.

        Args:
            model: The trained neural network model
            data_loader: DataLoader for evaluation dataset
            loss_function: Loss function for evaluation
            return_predictions: Whether to return predictions

        Returns:
            Validation metrics and optionally predictions
        """
        logger.info("Starting comprehensive model evaluation")
        
        model.eval()
        
        # Apply ARM64 optimizations for inference
        if self.use_arm64:
            model = self.arm64_optimizer.optimize_for_inference(model)
        
        # Collect predictions and targets
        all_predictions = {
            'positions': [],
            'confidence': [],
            'volatility': [],
            'returns': [],
            'prev_positions': []
        }
        
        total_loss = 0.0
        inference_times = []
        
        with torch.no_grad():
            for batch_idx, batch_data in enumerate(data_loader):
                start_time = time.time()
                
                # Handle different batch formats
                if len(batch_data) == 3:
                    features, targets, prev_positions = batch_data
                else:
                    features, targets = batch_data[:2]
                    prev_positions = torch.zeros(features.shape[0], targets.shape[-1], device=self.device)
                
                features = features.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)
                prev_positions = prev_positions.to(self.device, non_blocking=True)
                
                # Forward pass with mixed precision
                with autocast(enabled=self.use_mixed_precision):
                    outputs = model(features)
                    
                    # Handle different model output formats
                    if isinstance(outputs, tuple):
                        positions, confidence, volatility = outputs
                    else:
                        positions = outputs
                        confidence = torch.ones_like(positions)
                        volatility = torch.ones_like(positions) * 0.1
                    
                    # Prepare targets for loss calculation
                    if targets.dim() == 2 and targets.shape[1] == 1 and positions.shape[1] > 1:
                        targets_for_loss = targets.expand(-1, positions.shape[1])
                    else:
                        targets_for_loss = targets
                    
                    # Calculate loss
                    loss_output = loss_function(
                        positions=positions,
                        returns=targets_for_loss,
                        confidence=confidence,
                        prev_positions=prev_positions,
                        volatility_pred=volatility
                    )
                    
                    loss = loss_output['total_loss']
                
                total_loss += loss.item()
                
                # Store predictions
                all_predictions['positions'].append(positions.cpu())
                all_predictions['confidence'].append(confidence.cpu())
                all_predictions['volatility'].append(volatility.cpu())
                all_predictions['returns'].append(targets_for_loss.cpu())
                all_predictions['prev_positions'].append(prev_positions.cpu())
                
                # Track inference time
                inference_time = time.time() - start_time
                inference_times.append(inference_time)
        
        # Concatenate all predictions
        predictions = {
            key: torch.cat(values, dim=0) for key, values in all_predictions.items()
        }
        
        # Calculate comprehensive metrics
        metrics = self._calculate_comprehensive_metrics(
            predictions, total_loss / len(data_loader), inference_times
        )
        
        logger.info("Model evaluation completed")
        logger.info(f"Average inference time: {np.mean(inference_times):.4f}s")
        
        if return_predictions:
            return metrics, predictions
        else:
            return metrics

    def _calculate_comprehensive_metrics(self,
                                       predictions: Dict[str, torch.Tensor],
                                       avg_loss: float,
                                       inference_times: List[float]) -> ValidationMetrics:
        """Calculate comprehensive validation metrics."""
        positions = predictions['positions']
        confidence = predictions['confidence']
        volatility = predictions['volatility']
        returns = predictions['returns']
        prev_positions = predictions['prev_positions']
        
        # Portfolio returns
        weighted_positions = positions * confidence
        portfolio_returns = torch.sum(weighted_positions * returns, dim=1).numpy()
        
        # Basic metrics
        mse = mean_squared_error(returns.numpy().flatten(), positions.numpy().flatten())
        mae = mean_absolute_error(returns.numpy().flatten(), positions.numpy().flatten())
        
        # Financial performance metrics
        sharpe_ratio = self._calculate_sharpe_ratio(portfolio_returns)
        sortino_ratio = self._calculate_sortino_ratio(portfolio_returns)
        calmar_ratio = self._calculate_calmar_ratio(portfolio_returns)
        information_ratio = self._calculate_information_ratio(portfolio_returns)
        
        # Return metrics
        total_return = np.sum(portfolio_returns)
        annualized_return = np.mean(portfolio_returns) * 252
        annualized_volatility = np.std(portfolio_returns) * np.sqrt(252)
        
        # Risk metrics
        max_drawdown = self._calculate_max_drawdown(portfolio_returns)
        var_95 = np.percentile(portfolio_returns, 5)
        var_99 = np.percentile(portfolio_returns, 1)
        expected_shortfall = np.mean(portfolio_returns[portfolio_returns <= var_95])
        downside_returns = portfolio_returns[portfolio_returns < 0]
        downside_deviation = np.std(downside_returns) if len(downside_returns) > 0 else 0.0
        
        # Trading metrics
        turnover = self._calculate_turnover(positions, prev_positions)
        transaction_costs = turnover * self.transaction_cost_rate
        hit_ratio = self._calculate_hit_ratio(portfolio_returns)
        profit_factor = self._calculate_profit_factor(portfolio_returns)
        
        # Statistical metrics
        skewness = stats.skew(portfolio_returns)
        kurtosis = stats.kurtosis(portfolio_returns)
        tracking_error = np.std(portfolio_returns - self.benchmark_return)
        
        # Model-specific metrics
        confidence_accuracy = self._calculate_confidence_accuracy(positions, returns, confidence)
        volatility_prediction_error = self._calculate_volatility_prediction_error(
            portfolio_returns, volatility
        )
        position_consistency = self._calculate_position_consistency(positions, prev_positions)
        
        return ValidationMetrics(
            average_loss=avg_loss,
            mse=mse,
            mae=mae,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            calmar_ratio=calmar_ratio,
            information_ratio=information_ratio,
            total_return=total_return,
            annualized_return=annualized_return,
            annualized_volatility=annualized_volatility,
            max_drawdown=max_drawdown,
            var_95=var_95,
            var_99=var_99,
            expected_shortfall=expected_shortfall,
            downside_deviation=downside_deviation,
            turnover=turnover,
            transaction_costs=transaction_costs,
            hit_ratio=hit_ratio,
            profit_factor=profit_factor,
            skewness=skewness,
            kurtosis=kurtosis,
            tracking_error=tracking_error,
            confidence_accuracy=confidence_accuracy,
            volatility_prediction_error=volatility_prediction_error,
            position_consistency=position_consistency
        )

    def walk_forward_validation(self,
                               model_builder: Callable,
                               data: Dict[str, torch.Tensor],
                               loss_function: nn.Module,
                               training_params: Dict[str, Any],
                               n_splits: int = 5,
                               train_ratio: float = 0.6,
                               val_ratio: float = 0.2) -> CrossValidationResult:
        """
        Perform walk-forward validation with time series splits.

        Args:
            model_builder: Function that returns a new model instance
            data: Dictionary containing features, targets, and prev_positions
            loss_function: Loss function for training and evaluation
            training_params: Parameters for training
            n_splits: Number of validation splits
            train_ratio: Ratio of data for training
            val_ratio: Ratio of data for validation

        Returns:
            Cross-validation results
        """
        logger.info(f"Starting walk-forward validation with {n_splits} splits")
        
        features = data['features']
        targets = data['targets']
        prev_positions = data['prev_positions']
        
        n_samples = features.shape[0]
        fold_metrics = []
        
        for fold in range(n_splits):
            logger.info(f"Processing fold {fold + 1}/{n_splits}")
            
            # Calculate split indices for this fold
            start_idx = int(fold * n_samples / n_splits)
            end_idx = min(int((fold + 2) * n_samples / n_splits), n_samples)
            
            fold_data = features[start_idx:end_idx]
            fold_targets = targets[start_idx:end_idx]
            fold_prev_pos = prev_positions[start_idx:end_idx]
            
            fold_size = fold_data.shape[0]
            train_size = int(fold_size * train_ratio)
            val_size = int(fold_size * val_ratio)
            
            # Create datasets for this fold
            train_dataset = TensorDataset(
                fold_data[:train_size],
                fold_targets[:train_size],
                fold_prev_pos[:train_size]
            )
            val_dataset = TensorDataset(
                fold_data[train_size:train_size + val_size],
                fold_targets[train_size:train_size + val_size],
                fold_prev_pos[train_size:train_size + val_size]
            )
            test_dataset = TensorDataset(
                fold_data[train_size + val_size:],
                fold_targets[train_size + val_size:],
                fold_prev_pos[train_size + val_size:]
            )
            
            # Create data loaders
            batch_size = training_params.get("batch_size", 32)
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
            
            # Train model for this fold
            model = model_builder().to(self.device)
            fold_metrics_result = self._train_and_evaluate_fold(
                model, train_loader, val_loader, test_loader, 
                loss_function, training_params
            )
            
            fold_metrics.append(fold_metrics_result)
        
        # Calculate aggregated results
        cv_result = self._aggregate_cv_results(fold_metrics)
        
        logger.info("Walk-forward validation completed")
        return cv_result

    def time_series_cross_validation(self,
                                   model_builder: Callable,
                                   data: Dict[str, torch.Tensor],
                                   loss_function: nn.Module,
                                   training_params: Dict[str, Any],
                                   initial_train_size: int = 1000,
                                   step_size: int = 100,
                                   max_train_size: Optional[int] = None) -> CrossValidationResult:
        """
        Perform time series cross-validation with expanding window.

        Args:
            model_builder: Function that returns a new model instance
            data: Dictionary containing time series data
            loss_function: Loss function for training and evaluation
            training_params: Parameters for training
            initial_train_size: Initial training set size
            step_size: Step size for expanding window
            max_train_size: Maximum training set size

        Returns:
            Cross-validation results
        """
        logger.info("Starting time series cross-validation")
        
        features = data['features']
        targets = data['targets']
        prev_positions = data['prev_positions']
        
        n_samples = features.shape[0]
        fold_metrics = []
        
        current_train_size = initial_train_size
        
        while current_train_size + step_size < n_samples:
            test_start = current_train_size
            test_end = min(current_train_size + step_size, n_samples)
            
            logger.info(f"Training on samples 0:{current_train_size}, testing on {test_start}:{test_end}")
            
            # Limit training size if specified
            if max_train_size and current_train_size > max_train_size:
                train_start = current_train_size - max_train_size
            else:
                train_start = 0
            
            # Create datasets
            train_dataset = TensorDataset(
                features[train_start:current_train_size],
                targets[train_start:current_train_size],
                prev_positions[train_start:current_train_size]
            )
            test_dataset = TensorDataset(
                features[test_start:test_end],
                targets[test_start:test_end],
                prev_positions[test_start:test_end]
            )
            
            # Create data loaders
            batch_size = training_params.get("batch_size", 32)
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
            
            # Train and evaluate
            model = model_builder().to(self.device)
            fold_metrics_result = self._train_and_evaluate_fold(
                model, train_loader, None, test_loader,
                loss_function, training_params
            )
            
            fold_metrics.append(fold_metrics_result)
            current_train_size += step_size
        
        # Calculate aggregated results
        cv_result = self._aggregate_cv_results(fold_metrics)
        
        logger.info("Time series cross-validation completed")
        return cv_result

    def _train_and_evaluate_fold(self,
                                model: nn.Module,
                                train_loader: DataLoader,
                                val_loader: Optional[DataLoader],
                                test_loader: DataLoader,
                                loss_function: nn.Module,
                                training_params: Dict[str, Any]) -> ValidationMetrics:
        """Train and evaluate model for a single fold."""
        # Import here to avoid circular dependency
        from deep_momentum_trading.src.training.trainer import AdvancedTrainer
        
        # Create optimizer
        optimizer_config = training_params.get("optimizer", {"type": "AdamW", "lr": 0.001})
        optimizer_class = getattr(torch.optim, optimizer_config.get("type", "AdamW"))
        optimizer_params = {k: v for k, v in optimizer_config.items() if k != "type"}
        optimizer = optimizer_class(model.parameters(), **optimizer_params)
        
        # Create trainer
        trainer = AdvancedTrainer(
            model=model,
            optimizer=optimizer,
            loss_function=loss_function,
            train_loader=train_loader,
            val_loader=val_loader or test_loader,
            device=self.device,
            training_params=training_params
        )
        
        # Train model
        trainer.run_training()
        
        # Evaluate on test set
        metrics = self.evaluate_model(model, test_loader, loss_function)
        
        return metrics

    def _aggregate_cv_results(self, fold_metrics: List[ValidationMetrics]) -> CrossValidationResult:
        """Aggregate cross-validation results."""
        # Calculate mean metrics
        metric_names = list(fold_metrics[0].to_dict().keys())
        mean_values = {}
        std_values = {}
        
        for metric_name in metric_names:
            values = [getattr(fold_metric, metric_name) for fold_metric in fold_metrics]
            mean_values[metric_name] = np.mean(values)
            std_values[metric_name] = np.std(values)
        
        mean_metrics = ValidationMetrics(**mean_values)
        
        # Find best and worst folds (based on Sharpe ratio)
        sharpe_ratios = [fold_metric.sharpe_ratio for fold_metric in fold_metrics]
        best_fold = int(np.argmax(sharpe_ratios))
        worst_fold = int(np.argmin(sharpe_ratios))
        
        # Calculate stability score (inverse of coefficient of variation)
        cv_sharpe = np.std(sharpe_ratios) / np.mean(sharpe_ratios) if np.mean(sharpe_ratios) != 0 else float('inf')
        stability_score = 1.0 / (1.0 + cv_sharpe)
        
        return CrossValidationResult(
            fold_metrics=fold_metrics,
            mean_metrics=mean_metrics,
            std_metrics=std_values,
            best_fold=best_fold,
            worst_fold=worst_fold,
            stability_score=stability_score
        )

    # Helper methods for metric calculations
    def _calculate_sharpe_ratio(self, returns: np.ndarray) -> float:
        """Calculate Sharpe ratio."""
        excess_returns = returns - self.risk_free_rate / 252
        return np.mean(excess_returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0.0

    def _calculate_sortino_ratio(self, returns: np.ndarray) -> float:
        """Calculate Sortino ratio."""
        excess_returns = returns - self.risk_free_rate / 252
        downside_returns = returns[returns < 0]
        downside_std = np.std(downside_returns) if len(downside_returns) > 0 else 1e-8
        return np.mean(excess_returns) / downside_std * np.sqrt(252)

    def _calculate_calmar_ratio(self, returns: np.ndarray) -> float:
        """Calculate Calmar ratio."""
        annual_return = np.mean(returns) * 252
        max_dd = self._calculate_max_drawdown(returns)
        return annual_return / max_dd if max_dd > 0 else 0.0

    def _calculate_information_ratio(self, returns: np.ndarray) -> float:
        """Calculate Information ratio."""
        excess_returns = returns - self.benchmark_return
        tracking_error = np.std(excess_returns)
        return np.mean(excess_returns) / tracking_error if tracking_error > 0 else 0.0

    def _calculate_max_drawdown(self, returns: np.ndarray) -> float:
        """Calculate maximum drawdown."""
        cumulative = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        return abs(np.min(drawdown))

    def _calculate_turnover(self, positions: torch.Tensor, prev_positions: torch.Tensor) -> float:
        """Calculate portfolio turnover."""
        position_changes = torch.abs(positions - prev_positions)
        return torch.mean(torch.sum(position_changes, dim=1)).item()

    def _calculate_hit_ratio(self, returns: np.ndarray) -> float:
        """Calculate hit ratio (percentage of positive returns)."""
        return np.mean(returns > 0)

    def _calculate_profit_factor(self, returns: np.ndarray) -> float:
        """Calculate profit factor."""
        profits = returns[returns > 0]
        losses = returns[returns < 0]
        
        total_profit = np.sum(profits) if len(profits) > 0 else 0
        total_loss = abs(np.sum(losses)) if len(losses) > 0 else 1e-8
        
        return total_profit / total_loss

    def _calculate_confidence_accuracy(self, 
                                     positions: torch.Tensor, 
                                     returns: torch.Tensor, 
                                     confidence: torch.Tensor) -> float:
        """Calculate confidence prediction accuracy."""
        # High confidence should correspond to better predictions
        high_conf_mask = confidence > self.confidence_threshold
        
        if torch.sum(high_conf_mask) == 0:
            return 0.0
        
        # Calculate prediction accuracy for high confidence predictions
        high_conf_positions = positions[high_conf_mask]
        high_conf_returns = returns[high_conf_mask]
        
        # Check if position direction matches return direction
        correct_predictions = torch.sign(high_conf_positions) == torch.sign(high_conf_returns)
        return torch.mean(correct_predictions.float()).item()

    def _calculate_volatility_prediction_error(self, 
                                             portfolio_returns: np.ndarray, 
                                             volatility_pred: torch.Tensor) -> float:
        """Calculate volatility prediction error."""
        actual_vol = np.std(portfolio_returns)
        predicted_vol = torch.mean(volatility_pred).item()
        return abs(actual_vol - predicted_vol) / actual_vol if actual_vol > 0 else 0.0

    def _calculate_position_consistency(self, 
                                      positions: torch.Tensor, 
                                      prev_positions: torch.Tensor) -> float:
        """Calculate position consistency (how much positions change)."""
        position_changes = torch.abs(positions - prev_positions)
        max_possible_change = torch.abs(positions) + torch.abs(prev_positions)
        
        # Avoid division by zero
        consistency = 1.0 - torch.mean(position_changes / (max_possible_change + 1e-8))
        return consistency.item()

# Legacy compatibility functions
def evaluate_model(model: nn.Module,
                  data_loader: DataLoader,
                  loss_function: nn.Module,
                  device: torch.device) -> Dict[str, float]:
    """Legacy function for backward compatibility."""
    validator = AdvancedModelValidator(device=device)
    metrics = validator.evaluate_model(model, data_loader, loss_function)
    return metrics.to_dict()

def walk_forward_validation(model_builder: Callable,
                           data: Dict[str, Any],
                           loss_function: nn.Module,
                           device: torch.device,
                           split_dates: list,
                           training_params: Dict[str, Any]) -> Dict[str, Any]:
    """Legacy function for backward compatibility."""
    validator = AdvancedModelValidator(device=device)
    
    # Convert data format if needed
    if 'features' not in data:
        # Assume old format
        features = data.get('features', torch.randn(1000, 60, 200))
        targets = data.get('targets', torch.randn(1000, 100))
        prev_positions = data.get('prev_positions', torch.randn(1000, 100))
        
        data = {
            'features': features,
            'targets': targets,
            'prev_positions': prev_positions
        }
    
    cv_result = validator.walk_forward_validation(
        model_builder=model_builder,
        data=data,
        loss_function=loss_function,
        training_params=training_params,
        n_splits=len(split_dates) if split_dates else 3
    )
    
    # Convert to legacy format
    return {
        f"avg_{k}": v for k, v in cv_result.mean_metrics.to_dict().items()
    }

if __name__ == "__main__":
    # Example usage with ARM64 optimizations
    from deep_momentum_trading.src.models.deep_momentum_lstm import DeepMomentumLSTM
    from deep_momentum_trading.src.models.loss_functions import CombinedLoss
    
    # Configuration
    validation_config = {
        "confidence_threshold": 0.6,
        "risk_free_rate": 0.02,
        "transaction_cost_rate": 0.001,
        "benchmark_return": 0.0
    }
    
    # Create validator
    validator = AdvancedModelValidator(
        use_arm64=True,
        use_mixed_precision=True,
        validation_config=validation_config
    )
    
    # Dummy data
    input_size = 200
    sequence_length = 60
    num_assets = 100
    num_samples = 2000
    batch_size = 32
    
    features = torch.randn(num_samples, sequence_length, input_size)
    targets = torch.randn(num_samples, num_assets) * 0.01
    prev_positions = torch.randn(num_samples, num_assets) * 0.01
    
    test_dataset = TensorDataset(features, targets, prev_positions)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Model setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DeepMomentumLSTM(
        input_size=input_size, 
        hidden_size=256, 
        num_layers=2, 
        dropout=0.1,
        num_assets=num_assets
    ).to(device)
    
    loss_function = CombinedLoss(sharpe_weight=1.0, turnover_weight=0.05).to(device)
    
    print("=== Advanced Model Validation Test ===")
    
    # Test comprehensive evaluation
    print("\n1. Comprehensive Model Evaluation:")
    metrics = validator.evaluate_model(model, test_loader, loss_function)
    
    print(f"   Sharpe Ratio: {metrics.sharpe_ratio:.4f}")
    print(f"   Max Drawdown: {metrics.max_drawdown:.4f}")
    print(f"   Hit Ratio: {metrics.hit_ratio:.4f}")
    print(f"   Confidence Accuracy: {metrics.confidence_accuracy:.4f}")
    
    # Test walk-forward validation
    print("\n2. Walk-Forward Validation:")
    
    def build_model():
        return DeepMomentumLSTM(
            input_size=input_size,
            hidden_size=128,  # Smaller for faster training
            num_layers=1,
            dropout=0.1,
            num_assets=num_assets
        )
    
    training_params = {
        "epochs": 2,  # Reduced for testing
        "batch_size": 32,
        "optimizer": {"type": "AdamW", "lr": 0.001}
    }
    
    data = {
        'features': features,
        'targets': targets,
        'prev_positions': prev_positions
    }
    
    cv_result = validator.walk_forward_validation(
        model_builder=build_model,
        data=data,
        loss_function=loss_function,
        training_params=training_params,
        n_splits=3
    )
    
    print(f"   Mean Sharpe Ratio: {cv_result.mean_metrics.sharpe_ratio:.4f}")
    print(f"   Stability Score: {cv_result.stability_score:.4f}")
    print(f"   Best Fold: {cv_result.best_fold + 1}")
    
    # Save results
    metrics.save_json("test_validation_metrics.json")
    cv_result.save_json("test_cv_results.json")
    
    # Cleanup
    import os
    if os.path.exists("test_validation_metrics.json"):
        os.remove("test_validation_metrics.json")
    if os.path.exists("test_cv_results.json"):
        os.remove("test_cv_results.json")
    
    print("\n=== Advanced Model Validation Test Complete ===")
