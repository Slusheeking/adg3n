import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Any, Optional, Tuple, List, Union
import optuna
from optuna.samplers import TPESampler, CmaEsSampler, RandomSampler
from optuna.pruners import MedianPruner, HyperbandPruner
import logging
import numpy as np
import json
import time
import os
import pickle
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp
from dataclasses import dataclass, asdict
import warnings
from pathlib import Path

from deep_momentum_trading.src.utils.logger import get_logger
from deep_momentum_trading.src.models.model_registry import ModelRegistry, ModelConfig
from deep_momentum_trading.src.models.loss_functions import CombinedLoss
from deep_momentum_trading.src.training.trainer import Trainer
from deep_momentum_trading.src.training.validation import evaluate_model
from deep_momentum_trading.src.models.arm64_optimizations import ARM64Optimizer

logger = get_logger(__name__)

@dataclass
class TuningResult:
    """Container for hyperparameter tuning results."""
    best_value: float
    best_params: Dict[str, Any]
    best_trial_number: int
    study_name: str
    direction: str
    n_trials: int
    duration_seconds: float
    convergence_history: List[float]
    pruned_trials: int
    failed_trials: int
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)
    
    def save(self, filepath: str) -> None:
        """Save results to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, filepath: str) -> 'TuningResult':
        """Load results from JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        return cls(**data)

class AdvancedHyperparameterTuner:
    """
    Advanced hyperparameter optimization for deep momentum trading models
    with ARM64 optimizations, multi-objective optimization, and ensemble tuning.
    """

    def __init__(self,
                 model_registry: ModelRegistry,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 test_loader: Optional[DataLoader] = None,
                 loss_function: nn.Module = None,
                 device: torch.device = None,
                 tuning_config: Dict[str, Any] = None):
        """
        Initialize the Advanced HyperparameterTuner.

        Args:
            model_registry: The model registry instance
            train_loader: DataLoader for training dataset
            val_loader: DataLoader for validation dataset
            test_loader: Optional DataLoader for test dataset
            loss_function: The loss function to optimize
            device: The device (CPU or GPU) to train on
            tuning_config: Dictionary of tuning parameters
        """
        self.model_registry = model_registry
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.loss_function = loss_function or CombinedLoss()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tuning_config = tuning_config or {}
        
        # ARM64 optimizations
        self.arm64_optimizer = ARM64Optimizer()
        self.use_arm64 = self.tuning_config.get("use_arm64_optimizations", True)
        
        # Study configuration
        self.study_name = self.tuning_config.get("study_name", "momentum_model_tuning")
        self.n_trials = self.tuning_config.get("n_trials", 100)
        self.timeout = self.tuning_config.get("timeout", None)
        self.direction = self.tuning_config.get("direction", "maximize")
        
        # Advanced sampling and pruning
        sampler_type = self.tuning_config.get("sampler", "tpe")
        self.sampler = self._create_sampler(sampler_type)
        
        pruner_type = self.tuning_config.get("pruner", "median")
        self.pruner = self._create_pruner(pruner_type)
        
        # Search space and constraints
        self.search_space = self.tuning_config.get("search_space", {})
        self.fixed_params = self.tuning_config.get("fixed_params", {})
        self.constraints = self.tuning_config.get("constraints", [])
        
        # Multi-objective optimization
        self.multi_objective = self.tuning_config.get("multi_objective", False)
        self.objectives = self.tuning_config.get("objectives", ["validation_loss"])
        
        # Parallel execution
        self.n_jobs = self.tuning_config.get("n_jobs", 1)
        self.parallel_backend = self.tuning_config.get("parallel_backend", "thread")
        
        # Early stopping and convergence
        self.early_stopping_rounds = self.tuning_config.get("early_stopping_rounds", 20)
        self.convergence_threshold = self.tuning_config.get("convergence_threshold", 1e-6)
        
        # Results tracking
        self.results_dir = Path(self.tuning_config.get("results_dir", "tuning_results"))
        self.results_dir.mkdir(exist_ok=True)
        
        # Performance monitoring
        self.trial_times = []
        self.memory_usage = []
        self.convergence_history = []
        
        logger.info(f"AdvancedHyperparameterTuner initialized for study '{self.study_name}'")
        logger.info(f"Using {sampler_type} sampler with {pruner_type} pruner")
        logger.info(f"ARM64 optimizations: {'enabled' if self.use_arm64 else 'disabled'}")

    def _create_sampler(self, sampler_type: str) -> optuna.samplers.BaseSampler:
        """Create Optuna sampler based on configuration."""
        seed = self.tuning_config.get("seed", 42)
        
        if sampler_type == "tpe":
            return TPESampler(
                seed=seed,
                n_startup_trials=self.tuning_config.get("n_startup_trials", 10),
                n_ei_candidates=self.tuning_config.get("n_ei_candidates", 24)
            )
        elif sampler_type == "cmaes":
            return CmaEsSampler(seed=seed)
        elif sampler_type == "random":
            return RandomSampler(seed=seed)
        else:
            logger.warning(f"Unknown sampler type: {sampler_type}, using TPE")
            return TPESampler(seed=seed)

    def _create_pruner(self, pruner_type: str) -> optuna.pruners.BasePruner:
        """Create Optuna pruner based on configuration."""
        if pruner_type == "median":
            return MedianPruner(
                n_startup_trials=self.tuning_config.get("pruner_startup_trials", 5),
                n_warmup_steps=self.tuning_config.get("pruner_warmup_steps", 10)
            )
        elif pruner_type == "hyperband":
            return HyperbandPruner(
                min_resource=self.tuning_config.get("min_resource", 1),
                max_resource=self.tuning_config.get("max_resource", 100)
            )
        elif pruner_type == "none":
            return optuna.pruners.NopPruner()
        else:
            logger.warning(f"Unknown pruner type: {pruner_type}, using median")
            return MedianPruner()

    def _suggest_hyperparameters(self, trial: optuna.Trial) -> Dict[str, Any]:
        """Suggest hyperparameters for a trial."""
        params = {}
        
        # Model architecture parameters
        if "hidden_size" in self.search_space:
            space = self.search_space["hidden_size"]
            if isinstance(space, list):
                params["hidden_size"] = trial.suggest_categorical("hidden_size", space)
            else:
                params["hidden_size"] = trial.suggest_int("hidden_size", *space)
        
        if "num_layers" in self.search_space:
            params["num_layers"] = trial.suggest_int("num_layers", *self.search_space["num_layers"])
        
        if "dropout" in self.search_space:
            params["dropout"] = trial.suggest_float("dropout", *self.search_space["dropout"])
        
        if "attention_heads" in self.search_space:
            params["attention_heads"] = trial.suggest_categorical("attention_heads", self.search_space["attention_heads"])
        
        # Optimizer parameters
        if "optimizer_type" in self.search_space:
            params["optimizer_type"] = trial.suggest_categorical("optimizer_type", self.search_space["optimizer_type"])
        
        if "learning_rate" in self.search_space:
            lr_space = self.search_space["learning_rate"]
            if len(lr_space) == 3 and lr_space[2]:  # log scale
                params["learning_rate"] = trial.suggest_float("learning_rate", lr_space[0], lr_space[1], log=True)
            else:
                params["learning_rate"] = trial.suggest_float("learning_rate", lr_space[0], lr_space[1])
        
        if "weight_decay" in self.search_space:
            wd_space = self.search_space["weight_decay"]
            params["weight_decay"] = trial.suggest_float("weight_decay", wd_space[0], wd_space[1], log=True)
        
        if "momentum" in self.search_space:
            params["momentum"] = trial.suggest_float("momentum", *self.search_space["momentum"])
        
        # Training parameters
        if "batch_size" in self.search_space:
            params["batch_size"] = trial.suggest_categorical("batch_size", self.search_space["batch_size"])
        
        if "gradient_clip_value" in self.search_space:
            params["gradient_clip_value"] = trial.suggest_float("gradient_clip_value", *self.search_space["gradient_clip_value"])
        
        # Loss function parameters
        if "sharpe_weight" in self.search_space:
            params["sharpe_weight"] = trial.suggest_float("sharpe_weight", *self.search_space["sharpe_weight"])
        
        if "turnover_weight" in self.search_space:
            params["turnover_weight"] = trial.suggest_float("turnover_weight", *self.search_space["turnover_weight"])
        
        # ARM64 specific parameters
        if self.use_arm64 and "arm64_compile_mode" in self.search_space:
            params["arm64_compile_mode"] = trial.suggest_categorical("arm64_compile_mode", self.search_space["arm64_compile_mode"])
        
        return params

    def _check_constraints(self, params: Dict[str, Any]) -> bool:
        """Check if parameters satisfy constraints."""
        for constraint in self.constraints:
            if not eval(constraint, {"params": params}):
                return False
        return True

    def _objective(self, trial: optuna.Trial) -> Union[float, List[float]]:
        """Objective function for Optuna optimization."""
        start_time = time.time()
        
        try:
            # Suggest hyperparameters
            suggested_params = self._suggest_hyperparameters(trial)
            
            # Check constraints
            if not self._check_constraints(suggested_params):
                raise optuna.exceptions.TrialPruned()
            
            # Combine with fixed parameters
            all_params = {**self.fixed_params, **suggested_params}
            
            # Create model configuration
            model_type = all_params.get("model_type", "deep_momentum_lstm")
            model_params = {
                "input_size": all_params.get("input_size", 200),
                "hidden_size": all_params.get("hidden_size", 256),
                "num_layers": all_params.get("num_layers", 2),
                "dropout": all_params.get("dropout", 0.1),
                "num_assets": all_params.get("num_assets", 100)
            }
            
            # Add transformer-specific parameters if applicable
            if "transformer" in model_type.lower():
                model_params.update({
                    "attention_heads": all_params.get("attention_heads", 8),
                    "feedforward_dim": all_params.get("feedforward_dim", 512)
                })
            
            # Create model
            model_config = ModelConfig(
                model_type=model_type,
                parameters=model_params,
                description=f"Trial {trial.number} model",
                version=f"trial_{trial.number}"
            )
            
            trial_model_name = f"{model_type}_trial_{trial.number}"
            self.model_registry.register_model(trial_model_name, model_config)
            model = self.model_registry.create_model(trial_model_name).to(self.device)
            
            # Apply ARM64 optimizations
            if self.use_arm64:
                compile_mode = all_params.get("arm64_compile_mode", "default")
                model = self.arm64_optimizer.optimize_model(model, compile_mode)
            
            # Create optimizer
            optimizer_type = all_params.get("optimizer_type", "Adam")
            optimizer_params = {
                "lr": all_params.get("learning_rate", 1e-3),
                "weight_decay": all_params.get("weight_decay", 1e-5)
            }
            
            if optimizer_type in ["SGD", "RMSprop"]:
                optimizer_params["momentum"] = all_params.get("momentum", 0.9)
            
            optimizer = getattr(torch.optim, optimizer_type)(model.parameters(), **optimizer_params)
            
            # Create loss function
            loss_params = {
                "sharpe_weight": all_params.get("sharpe_weight", 1.0),
                "turnover_weight": all_params.get("turnover_weight", 0.05)
            }
            loss_function = CombinedLoss(**loss_params).to(self.device)
            
            # Training parameters
            training_params = {
                "epochs": all_params.get("epochs", 10),
                "batch_size": all_params.get("batch_size", 64),
                "gradient_clip_value": all_params.get("gradient_clip_value", 1.0),
                "early_stopping_patience": all_params.get("early_stopping_patience", 5),
                "log_interval": all_params.get("log_interval", 100)
            }
            
            # Create trainer
            trainer = Trainer(
                model=model,
                optimizer=optimizer,
                loss_function=loss_function,
                train_loader=self.train_loader,
                val_loader=self.val_loader,
                device=self.device,
                training_params=training_params
            )
            
            # Train model with intermediate reporting for pruning
            best_val_loss = float('inf')
            metrics_history = []
            
            for epoch in range(training_params["epochs"]):
                # Train one epoch
                train_loss = trainer.train_epoch()
                val_metrics = trainer.validate()
                
                # Collect metrics
                val_loss = val_metrics.get("loss", float('inf'))
                sharpe_ratio = val_metrics.get("sharpe_ratio", 0.0)
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                
                metrics_history.append({
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "sharpe_ratio": sharpe_ratio
                })
                
                # Report intermediate value for pruning
                if self.multi_objective:
                    intermediate_values = [val_loss, -sharpe_ratio]  # Minimize loss, maximize Sharpe
                else:
                    intermediate_value = -sharpe_ratio if self.direction == "maximize" else val_loss
                    trial.report(intermediate_value, epoch)
                
                # Check if trial should be pruned
                if trial.should_prune():
                    raise optuna.exceptions.TrialPruned()
            
            # Final evaluation
            final_metrics = evaluate_model(model, self.val_loader, loss_function, self.device)
            
            # Calculate objectives
            if self.multi_objective:
                objectives = []
                for obj_name in self.objectives:
                    if obj_name == "validation_loss":
                        objectives.append(best_val_loss)
                    elif obj_name == "sharpe_ratio":
                        objectives.append(-final_metrics.get("sharpe_ratio", 0.0))  # Minimize negative Sharpe
                    elif obj_name == "max_drawdown":
                        objectives.append(final_metrics.get("max_drawdown", 1.0))
                    elif obj_name == "volatility":
                        objectives.append(final_metrics.get("volatility", 1.0))
                return objectives
            else:
                if self.direction == "maximize":
                    return final_metrics.get("sharpe_ratio", 0.0)
                else:
                    return best_val_loss
                    
        except optuna.exceptions.TrialPruned:
            raise
        except Exception as e:
            logger.error(f"Trial {trial.number} failed: {e}")
            return float('inf') if self.direction == "minimize" else float('-inf')
        finally:
            # Record trial time
            trial_time = time.time() - start_time
            self.trial_times.append(trial_time)
            
            # Clean up trial model
            if f"{model_type}_trial_{trial.number}" in self.model_registry.models:
                del self.model_registry.models[f"{model_type}_trial_{trial.number}"]

    def run_tuning(self) -> TuningResult:
        """Run hyperparameter tuning study."""
        logger.info(f"Starting hyperparameter tuning with {self.n_trials} trials")
        
        # Set up Optuna logging
        optuna.logging.set_verbosity(optuna.logging.INFO)
        
        # Create study
        if self.multi_objective:
            study = optuna.create_study(
                study_name=self.study_name,
                directions=["minimize"] * len(self.objectives),
                sampler=self.sampler,
                pruner=self.pruner,
                load_if_exists=True
            )
        else:
            study = optuna.create_study(
                study_name=self.study_name,
                direction=self.direction,
                sampler=self.sampler,
                pruner=self.pruner,
                load_if_exists=True
            )
        
        start_time = time.time()
        
        try:
            if self.n_jobs > 1:
                # Parallel optimization
                if self.parallel_backend == "process":
                    with ProcessPoolExecutor(max_workers=self.n_jobs) as executor:
                        study.optimize(
                            self._objective,
                            n_trials=self.n_trials,
                            timeout=self.timeout,
                            callbacks=[self._trial_callback],
                            n_jobs=self.n_jobs
                        )
                else:
                    study.optimize(
                        self._objective,
                        n_trials=self.n_trials,
                        timeout=self.timeout,
                        callbacks=[self._trial_callback],
                        n_jobs=self.n_jobs
                    )
            else:
                # Sequential optimization
                study.optimize(
                    self._objective,
                    n_trials=self.n_trials,
                    timeout=self.timeout,
                    callbacks=[self._trial_callback]
                )
                
        except KeyboardInterrupt:
            logger.info("Tuning interrupted by user")
        except Exception as e:
            logger.error(f"Error during tuning: {e}", exc_info=True)
        
        duration = time.time() - start_time
        
        # Analyze results
        completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        pruned_trials = len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])
        failed_trials = len([t for t in study.trials if t.state == optuna.trial.TrialState.FAIL])
        
        logger.info(f"Tuning completed in {duration:.2f} seconds")
        logger.info(f"Completed trials: {len(completed_trials)}")
        logger.info(f"Pruned trials: {pruned_trials}")
        logger.info(f"Failed trials: {failed_trials}")
        
        if completed_trials:
            if self.multi_objective:
                # For multi-objective, use the first Pareto optimal solution
                best_trial = study.best_trials[0] if study.best_trials else completed_trials[0]
                best_value = best_trial.values[0] if best_trial.values else float('inf')
            else:
                best_trial = study.best_trial
                best_value = best_trial.value
            
            logger.info(f"Best trial: {best_trial.number}")
            logger.info(f"Best value: {best_value}")
            logger.info(f"Best parameters: {best_trial.params}")
            
            # Save best model
            self._save_best_model(study, best_trial)
            
            # Create results
            result = TuningResult(
                best_value=best_value,
                best_params=best_trial.params,
                best_trial_number=best_trial.number,
                study_name=self.study_name,
                direction=self.direction,
                n_trials=len(completed_trials),
                duration_seconds=duration,
                convergence_history=self.convergence_history,
                pruned_trials=pruned_trials,
                failed_trials=failed_trials
            )
            
            # Save results
            results_file = self.results_dir / f"{self.study_name}_results.json"
            result.save(str(results_file))
            
            # Save study
            study_file = self.results_dir / f"{self.study_name}_study.pkl"
            with open(study_file, 'wb') as f:
                pickle.dump(study, f)
            
            return result
        else:
            logger.warning("No completed trials found")
            return None

    def _trial_callback(self, study: optuna.Study, trial: optuna.Trial) -> None:
        """Callback function called after each trial."""
        if trial.state == optuna.trial.TrialState.COMPLETE:
            if self.multi_objective:
                value_str = f"values={trial.values}"
            else:
                value_str = f"value={trial.value:.6f}"
            
            logger.info(f"Trial {trial.number} completed: {value_str}")
            
            # Update convergence history
            if not self.multi_objective:
                self.convergence_history.append(trial.value)
                
                # Check for convergence
                if len(self.convergence_history) >= self.early_stopping_rounds:
                    recent_values = self.convergence_history[-self.early_stopping_rounds:]
                    if self.direction == "minimize":
                        improvement = max(recent_values) - min(recent_values)
                    else:
                        improvement = max(recent_values) - min(recent_values)
                    
                    if improvement < self.convergence_threshold:
                        logger.info(f"Early stopping: convergence detected after {trial.number} trials")
                        study.stop()
        
        elif trial.state == optuna.trial.TrialState.PRUNED:
            logger.info(f"Trial {trial.number} pruned")
        elif trial.state == optuna.trial.TrialState.FAIL:
            logger.warning(f"Trial {trial.number} failed")

    def _save_best_model(self, study: optuna.Study, best_trial: optuna.Trial) -> None:
        """Save the best model to the registry."""
        try:
            model_type = self.fixed_params.get("model_type", "deep_momentum_lstm")
            best_model_name = f"{model_type}_best_tuned_{self.study_name}"
            
            # Create model with best parameters
            best_params = {**self.fixed_params, **best_trial.params}
            model_params = {
                "input_size": best_params.get("input_size", 200),
                "hidden_size": best_params.get("hidden_size", 256),
                "num_layers": best_params.get("num_layers", 2),
                "dropout": best_params.get("dropout", 0.1),
                "num_assets": best_params.get("num_assets", 100)
            }
            
            model_config = ModelConfig(
                model_type=model_type,
                parameters=model_params,
                description=f"Best model from hyperparameter tuning study {self.study_name}",
                version=f"tuned_v{int(time.time())}",
                tags=["tuned", "best", self.study_name]
            )
            
            self.model_registry.register_model(best_model_name, model_config)
            self.model_registry.save_registry()
            
            logger.info(f"Best model saved as '{best_model_name}' in registry")
            
        except Exception as e:
            logger.error(f"Failed to save best model: {e}")

    def analyze_study(self, study_file: str) -> Dict[str, Any]:
        """Analyze completed study results."""
        try:
            with open(study_file, 'rb') as f:
                study = pickle.load(f)
            
            analysis = {
                "study_name": study.study_name,
                "n_trials": len(study.trials),
                "best_value": study.best_value if hasattr(study, 'best_value') else None,
                "best_params": study.best_params if hasattr(study, 'best_params') else None,
                "parameter_importance": {},
                "optimization_history": []
            }
            
            # Parameter importance
            try:
                importance = optuna.importance.get_param_importances(study)
                analysis["parameter_importance"] = importance
            except Exception as e:
                logger.warning(f"Could not calculate parameter importance: {e}")
            
            # Optimization history
            for trial in study.trials:
                if trial.state == optuna.trial.TrialState.COMPLETE:
                    analysis["optimization_history"].append({
                        "trial": trial.number,
                        "value": trial.value if hasattr(trial, 'value') else None,
                        "params": trial.params
                    })
            
            return analysis
            
        except Exception as e:
            logger.error(f"Failed to analyze study: {e}")
            return {}

# Legacy compatibility
HyperparameterTuner = AdvancedHyperparameterTuner

if __name__ == "__main__":
    # Example usage with ARM64 optimizations
    import os
    from deep_momentum_trading.src.models.deep_momentum_lstm import DeepMomentumLSTM
    from torch.utils.data import TensorDataset
    
    # Configuration
    config = {
        "study_name": "advanced_momentum_tuning",
        "n_trials": 20,
        "timeout": 1800,  # 30 minutes
        "direction": "maximize",
        "use_arm64_optimizations": True,
        "sampler": "tpe",
        "pruner": "median",
        "n_jobs": 1,
        "search_space": {
            "hidden_size": [128, 256, 512],
            "num_layers": [1, 4],
            "dropout": [0.0, 0.3],
            "learning_rate": [1e-5, 1e-2, True],  # log scale
            "optimizer_type": ["Adam", "AdamW", "RMSprop"],
            "batch_size": [32, 64, 128],
            "sharpe_weight": [0.5, 2.0],
            "turnover_weight": [0.01, 0.1]
        },
        "fixed_params": {
            "model_type": "deep_momentum_lstm",
            "input_size": 200,
            "num_assets": 100,
            "epochs": 5
        }
    }
    
    # Dummy data
    num_samples = 1000
    features = torch.randn(num_samples, 60, 200)
    targets = torch.randn(num_samples, 100) * 0.01
    prev_positions = torch.randn(num_samples, 100) * 0.01
    
    train_dataset = TensorDataset(features[:800], targets[:800], prev_positions[:800])
    val_dataset = TensorDataset(features[800:], targets[800:], prev_positions[800:])
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # Initialize components
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    registry = ModelRegistry(registry_path="test_advanced_tuning_registry.json")
    
    # Create tuner
    tuner = AdvancedHyperparameterTuner(
        model_registry=registry,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        tuning_config=config
    )
    
    print("Starting advanced hyperparameter tuning...")
    result = tuner.run_tuning()
    
    if result:
        print(f"\nTuning completed successfully!")
        print(f"Best value: {result.best_value:.6f}")
        print(f"Best parameters: {result.best_params}")
        print(f"Duration: {result.duration_seconds:.2f} seconds")
    else:
        print("\nTuning failed or no results found")
    
    # Cleanup
    if os.path.exists("test_advanced_tuning_registry.json"):
        os.remove("test_advanced_tuning_registry.json")
    
    print("\nAdvanced hyperparameter tuning example complete!")
