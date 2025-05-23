import torch
import torch.nn as nn
from typing import Dict, Any, List, Optional, Tuple, Union, Callable
import logging
import numpy as np
import os
import shutil
import json
import time
from pathlib import Path
from dataclasses import dataclass, asdict
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
import pickle
import hashlib
from concurrent.futures import ThreadPoolExecutor
import warnings

from deep_momentum_trading.src.utils.logger import get_logger
from deep_momentum_trading.src.models.model_registry import ModelRegistry, ModelConfig
from deep_momentum_trading.src.models.arm64_optimizations import ARM64Optimizer
from deep_momentum_trading.src.training.validation import evaluate_model

logger = get_logger(__name__)

@dataclass
class SelectionCriteria:
    """Criteria for model selection."""
    primary_metric: str = "sharpe_ratio"
    secondary_metrics: List[str] = None
    min_metric_value: float = 0.0
    max_drawdown_limit: Optional[float] = None
    min_samples: int = 100
    stability_threshold: float = 0.1
    diversity_weight: float = 0.2
    performance_weight: float = 0.8
    
    def __post_init__(self):
        if self.secondary_metrics is None:
            self.secondary_metrics = ["calmar_ratio", "sortino_ratio"]

@dataclass
class ModelScore:
    """Model scoring result."""
    model_name: str
    performance_score: float
    stability_score: float
    diversity_score: float
    composite_score: float
    metrics: Dict[str, float]
    rank: int = 0

class AdvancedModelSelector:
    """
    Advanced model selection with ARM64 optimizations, ensemble strategies,
    and comprehensive model management capabilities.
    """

    def __init__(self, 
                 model_registry: ModelRegistry,
                 selection_config: Dict[str, Any] = None):
        """
        Initialize the Advanced Model Selector.

        Args:
            model_registry: The model registry instance
            selection_config: Configuration for model selection
        """
        self.model_registry = model_registry
        self.config = selection_config or {}
        
        # ARM64 optimizations
        self.arm64_optimizer = ARM64Optimizer()
        self.use_arm64 = self.config.get("use_arm64_optimizations", True)
        
        # Selection parameters
        self.default_criteria = SelectionCriteria(
            primary_metric=self.config.get("primary_metric", "sharpe_ratio"),
            secondary_metrics=self.config.get("secondary_metrics", ["calmar_ratio", "sortino_ratio"]),
            min_metric_value=self.config.get("min_metric_value", 0.0),
            max_drawdown_limit=self.config.get("max_drawdown_limit", None),
            stability_threshold=self.config.get("stability_threshold", 0.1),
            diversity_weight=self.config.get("diversity_weight", 0.2),
            performance_weight=self.config.get("performance_weight", 0.8)
        )
        
        # Caching
        self.cache_dir = Path(self.config.get("cache_dir", "model_selection_cache"))
        self.cache_dir.mkdir(exist_ok=True)
        self.cache_ttl = self.config.get("cache_ttl", 3600)  # 1 hour
        
        # Parallel processing
        self.n_jobs = self.config.get("n_jobs", 1)
        
        logger.info("AdvancedModelSelector initialized")
        logger.info(f"ARM64 optimizations: {'enabled' if self.use_arm64 else 'disabled'}")

    def select_best_model(self,
                         criteria: Optional[SelectionCriteria] = None,
                         model_type: Optional[str] = None,
                         tags: Optional[List[str]] = None,
                         exclude_tags: Optional[List[str]] = None) -> Optional[Dict[str, Any]]:
        """
        Select the best model based on comprehensive criteria.

        Args:
            criteria: Selection criteria
            model_type: Filter by model type
            tags: Required tags
            exclude_tags: Tags to exclude

        Returns:
            Best model information or None
        """
        criteria = criteria or self.default_criteria
        
        # Get candidate models
        candidates = self._get_candidate_models(
            model_type=model_type,
            tags=tags,
            exclude_tags=exclude_tags
        )
        
        if not candidates:
            logger.warning("No candidate models found")
            return None
        
        # Score models
        scored_models = self._score_models(candidates, criteria)
        
        if not scored_models:
            logger.warning("No models passed scoring criteria")
            return None
        
        # Select best model
        best_model = max(scored_models, key=lambda x: x.composite_score)
        
        logger.info(f"Selected best model: {best_model.model_name}")
        logger.info(f"Composite score: {best_model.composite_score:.4f}")
        logger.info(f"Performance score: {best_model.performance_score:.4f}")
        logger.info(f"Stability score: {best_model.stability_score:.4f}")
        
        # Return full model information
        model_info = next(m for m in candidates if m["name"] == best_model.model_name)
        model_info["selection_score"] = asdict(best_model)
        
        return model_info

    def select_ensemble_models(self,
                              criteria: Optional[SelectionCriteria] = None,
                              num_models: int = 5,
                              diversity_method: str = "correlation",
                              model_type: Optional[str] = None,
                              tags: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Select models for ensemble based on performance and diversity.

        Args:
            criteria: Selection criteria
            num_models: Number of models to select
            diversity_method: Method for diversity calculation
            model_type: Filter by model type
            tags: Required tags

        Returns:
            List of selected models for ensemble
        """
        criteria = criteria or self.default_criteria
        
        # Get candidate models
        candidates = self._get_candidate_models(
            model_type=model_type,
            tags=tags,
            exclude_tags=["deprecated", "failed"]
        )
        
        if len(candidates) < num_models:
            logger.warning(f"Only {len(candidates)} candidates available, requested {num_models}")
            num_models = len(candidates)
        
        # Score models
        scored_models = self._score_models(candidates, criteria)
        
        if len(scored_models) < num_models:
            logger.warning(f"Only {len(scored_models)} models passed criteria")
            num_models = len(scored_models)
        
        # Select diverse ensemble
        ensemble_models = self._select_diverse_ensemble(
            scored_models, 
            candidates, 
            num_models, 
            diversity_method
        )
        
        logger.info(f"Selected {len(ensemble_models)} models for ensemble")
        for i, model in enumerate(ensemble_models):
            logger.info(f"  {i+1}. {model['name']} (score: {model['selection_score']['composite_score']:.4f})")
        
        return ensemble_models

    def deploy_model(self, 
                    model_name: str, 
                    deployment_config: Dict[str, Any] = None) -> bool:
        """
        Deploy a model to production with ARM64 optimizations.

        Args:
            model_name: Name of the model to deploy
            deployment_config: Deployment configuration

        Returns:
            True if deployment successful, False otherwise
        """
        config = deployment_config or {}
        production_path = config.get("production_path", "deep_momentum_trading/data/models/production/")
        optimize_for_inference = config.get("optimize_for_inference", True)
        create_backup = config.get("create_backup", True)
        
        model_metadata = self.model_registry.models.get(model_name)
        
        if not model_metadata:
            logger.error(f"Model '{model_name}' not found in registry")
            return False
        
        if not model_metadata.checkpoint_path or not os.path.exists(model_metadata.checkpoint_path):
            logger.error(f"Checkpoint for '{model_name}' not found")
            return False
        
        try:
            # Ensure production directory exists
            os.makedirs(production_path, exist_ok=True)
            
            # Create backup if requested
            if create_backup:
                self._create_deployment_backup(production_path)
            
            # Load and optimize model for production
            model = self.model_registry.load_model(model_name)
            
            if optimize_for_inference and self.use_arm64:
                logger.info("Applying ARM64 optimizations for inference")
                model = self.arm64_optimizer.optimize_for_inference(model)
            
            # Save optimized model
            production_file = os.path.join(production_path, f"{model_name}_production.pth")
            
            if self.use_arm64:
                # Use TorchScript for ARM64 optimization
                scripted_model = torch.jit.script(model)
                scripted_model.save(production_file)
            else:
                torch.save(model.state_dict(), production_file)
            
            # Update model metadata
            model_metadata.model_path = production_file
            if "production" not in model_metadata.config.tags:
                model_metadata.config.tags.append("production")
            
            # Add deployment metadata
            deployment_info = {
                "deployed_at": time.time(),
                "deployment_version": int(time.time()),
                "optimized_for_arm64": self.use_arm64,
                "optimization_level": "inference" if optimize_for_inference else "standard"
            }
            
            if not hasattr(model_metadata, 'deployment_info'):
                model_metadata.deployment_info = {}
            model_metadata.deployment_info.update(deployment_info)
            
            self.model_registry.save_registry()
            
            logger.info(f"Model '{model_name}' successfully deployed to {production_file}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to deploy model '{model_name}': {e}", exc_info=True)
            return False

    def compare_models(self, 
                      model_names: List[str],
                      metrics: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Compare multiple models across various metrics.

        Args:
            model_names: List of model names to compare
            metrics: Metrics to compare (if None, use all available)

        Returns:
            Comparison results
        """
        if not model_names:
            return {}
        
        comparison = {
            "models": model_names,
            "metrics": {},
            "rankings": {},
            "summary": {}
        }
        
        # Get model information
        models_info = []
        for name in model_names:
            model_info = self.model_registry.get_model_info(name)
            if model_info:
                models_info.append(model_info)
            else:
                logger.warning(f"Model '{name}' not found in registry")
        
        if not models_info:
            return comparison
        
        # Determine metrics to compare
        if metrics is None:
            all_metrics = set()
            for model in models_info:
                if "performance" in model:
                    all_metrics.update(model["performance"].keys())
            metrics = list(all_metrics)
        
        # Extract metric values
        for metric in metrics:
            comparison["metrics"][metric] = {}
            values = []
            
            for model in models_info:
                value = model.get("performance", {}).get(metric)
                comparison["metrics"][metric][model["name"]] = value
                if value is not None:
                    values.append((model["name"], value))
            
            # Rank models for this metric
            if values:
                # Assume higher is better for most metrics (except loss-type metrics)
                reverse = not any(word in metric.lower() for word in ["loss", "error", "drawdown"])
                ranked = sorted(values, key=lambda x: x[1], reverse=reverse)
                comparison["rankings"][metric] = [name for name, _ in ranked]
        
        # Calculate summary statistics
        comparison["summary"] = self._calculate_comparison_summary(comparison["metrics"])
        
        return comparison

    def get_model_recommendations(self,
                                 use_case: str = "trading",
                                 risk_tolerance: str = "medium",
                                 performance_target: Optional[float] = None) -> List[Dict[str, Any]]:
        """
        Get model recommendations based on use case and requirements.

        Args:
            use_case: Use case ("trading", "backtesting", "research")
            risk_tolerance: Risk tolerance ("low", "medium", "high")
            performance_target: Target performance metric value

        Returns:
            List of recommended models
        """
        # Define criteria based on use case and risk tolerance
        criteria_map = {
            ("trading", "low"): SelectionCriteria(
                primary_metric="calmar_ratio",
                max_drawdown_limit=0.05,
                stability_threshold=0.05
            ),
            ("trading", "medium"): SelectionCriteria(
                primary_metric="sharpe_ratio",
                max_drawdown_limit=0.10,
                stability_threshold=0.10
            ),
            ("trading", "high"): SelectionCriteria(
                primary_metric="total_return",
                max_drawdown_limit=0.20,
                stability_threshold=0.15
            ),
            ("backtesting", "low"): SelectionCriteria(
                primary_metric="sharpe_ratio",
                stability_threshold=0.05
            ),
            ("backtesting", "medium"): SelectionCriteria(
                primary_metric="sharpe_ratio",
                stability_threshold=0.10
            ),
            ("backtesting", "high"): SelectionCriteria(
                primary_metric="total_return",
                stability_threshold=0.15
            ),
            ("research", "low"): SelectionCriteria(
                primary_metric="information_ratio",
                stability_threshold=0.05
            ),
            ("research", "medium"): SelectionCriteria(
                primary_metric="information_ratio",
                stability_threshold=0.10
            ),
            ("research", "high"): SelectionCriteria(
                primary_metric="alpha",
                stability_threshold=0.15
            )
        }
        
        criteria = criteria_map.get((use_case, risk_tolerance), self.default_criteria)
        
        if performance_target:
            criteria.min_metric_value = performance_target
        
        # Get candidates
        candidates = self._get_candidate_models()
        scored_models = self._score_models(candidates, criteria)
        
        # Sort by composite score
        scored_models.sort(key=lambda x: x.composite_score, reverse=True)
        
        # Return top recommendations with explanations
        recommendations = []
        for i, scored_model in enumerate(scored_models[:5]):  # Top 5
            model_info = next(m for m in candidates if m["name"] == scored_model.model_name)
            
            recommendation = {
                **model_info,
                "recommendation_rank": i + 1,
                "selection_score": asdict(scored_model),
                "recommendation_reason": self._generate_recommendation_reason(
                    scored_model, use_case, risk_tolerance
                )
            }
            recommendations.append(recommendation)
        
        return recommendations

    def _get_candidate_models(self,
                             model_type: Optional[str] = None,
                             tags: Optional[List[str]] = None,
                             exclude_tags: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Get candidate models based on filters."""
        all_models = self.model_registry.list_models(model_type=model_type, tags=tags)
        
        if exclude_tags:
            all_models = [
                m for m in all_models 
                if not any(tag in m.get("tags", []) for tag in exclude_tags)
            ]
        
        # Filter models with performance metrics
        candidates = [
            m for m in all_models 
            if "performance" in m and m["performance"]
        ]
        
        return candidates

    def _score_models(self, 
                     candidates: List[Dict[str, Any]], 
                     criteria: SelectionCriteria) -> List[ModelScore]:
        """Score models based on criteria."""
        scored_models = []
        
        for model in candidates:
            performance = model.get("performance", {})
            
            # Check minimum requirements
            primary_value = performance.get(criteria.primary_metric)
            if primary_value is None or primary_value < criteria.min_metric_value:
                continue
            
            if criteria.max_drawdown_limit:
                drawdown = performance.get("max_drawdown", performance.get("calmar_max_drawdown"))
                if drawdown and drawdown > criteria.max_drawdown_limit:
                    continue
            
            # Calculate scores
            performance_score = self._calculate_performance_score(performance, criteria)
            stability_score = self._calculate_stability_score(model, criteria)
            diversity_score = 0.0  # Will be calculated in ensemble selection
            
            composite_score = (
                criteria.performance_weight * performance_score +
                (1 - criteria.performance_weight) * stability_score
            )
            
            scored_model = ModelScore(
                model_name=model["name"],
                performance_score=performance_score,
                stability_score=stability_score,
                diversity_score=diversity_score,
                composite_score=composite_score,
                metrics=performance
            )
            
            scored_models.append(scored_model)
        
        return scored_models

    def _calculate_performance_score(self, 
                                   performance: Dict[str, float], 
                                   criteria: SelectionCriteria) -> float:
        """Calculate normalized performance score."""
        primary_value = performance.get(criteria.primary_metric, 0.0)
        
        # Normalize primary metric (simple min-max scaling)
        primary_score = max(0.0, min(1.0, primary_value / max(1.0, criteria.min_metric_value * 2)))
        
        # Add secondary metrics
        secondary_scores = []
        for metric in criteria.secondary_metrics:
            value = performance.get(metric, 0.0)
            # Simple normalization - could be improved with historical data
            normalized = max(0.0, min(1.0, value / 2.0))
            secondary_scores.append(normalized)
        
        # Weighted combination
        if secondary_scores:
            secondary_avg = np.mean(secondary_scores)
            return 0.7 * primary_score + 0.3 * secondary_avg
        else:
            return primary_score

    def _calculate_stability_score(self, 
                                 model: Dict[str, Any], 
                                 criteria: SelectionCriteria) -> float:
        """Calculate model stability score."""
        # This is a simplified stability calculation
        # In practice, you'd want to analyze performance over time
        
        performance = model.get("performance", {})
        
        # Use volatility as a proxy for stability
        volatility = performance.get("volatility", 0.2)
        stability = max(0.0, 1.0 - volatility / 0.5)  # Normalize assuming max vol of 0.5
        
        # Consider drawdown stability
        max_drawdown = performance.get("max_drawdown", 0.1)
        drawdown_stability = max(0.0, 1.0 - max_drawdown / 0.3)  # Normalize assuming max dd of 0.3
        
        return 0.6 * stability + 0.4 * drawdown_stability

    def _select_diverse_ensemble(self,
                               scored_models: List[ModelScore],
                               candidates: List[Dict[str, Any]],
                               num_models: int,
                               diversity_method: str) -> List[Dict[str, Any]]:
        """Select diverse ensemble of models."""
        if len(scored_models) <= num_models:
            return [next(m for m in candidates if m["name"] == sm.model_name) for sm in scored_models]
        
        # Start with the best model
        selected = [scored_models[0]]
        remaining = scored_models[1:]
        
        # Greedily select diverse models
        while len(selected) < num_models and remaining:
            best_candidate = None
            best_score = -float('inf')
            
            for candidate in remaining:
                # Calculate diversity score with already selected models
                diversity = self._calculate_diversity_score(
                    candidate, selected, candidates, diversity_method
                )
                
                # Combined score: performance + diversity
                combined_score = 0.7 * candidate.composite_score + 0.3 * diversity
                
                if combined_score > best_score:
                    best_score = combined_score
                    best_candidate = candidate
            
            if best_candidate:
                selected.append(best_candidate)
                remaining.remove(best_candidate)
        
        # Return full model information
        ensemble_models = []
        for scored_model in selected:
            model_info = next(m for m in candidates if m["name"] == scored_model.model_name)
            model_info["selection_score"] = asdict(scored_model)
            ensemble_models.append(model_info)
        
        return ensemble_models

    def _calculate_diversity_score(self,
                                 candidate: ModelScore,
                                 selected: List[ModelScore],
                                 candidates: List[Dict[str, Any]],
                                 method: str) -> float:
        """Calculate diversity score for ensemble selection."""
        if not selected:
            return 1.0
        
        if method == "correlation":
            # Simplified correlation-based diversity
            # In practice, you'd calculate actual prediction correlations
            candidate_metrics = candidate.metrics
            
            correlations = []
            for selected_model in selected:
                # Simple metric-based correlation proxy
                correlation = self._calculate_metric_correlation(
                    candidate_metrics, selected_model.metrics
                )
                correlations.append(abs(correlation))
            
            # Lower average correlation = higher diversity
            avg_correlation = np.mean(correlations)
            return 1.0 - avg_correlation
        
        elif method == "clustering":
            # Use model parameters for clustering-based diversity
            return self._calculate_clustering_diversity(candidate, selected, candidates)
        
        else:
            # Default: random diversity
            return np.random.random()

    def _calculate_metric_correlation(self, 
                                    metrics1: Dict[str, float], 
                                    metrics2: Dict[str, float]) -> float:
        """Calculate correlation between model metrics."""
        common_metrics = set(metrics1.keys()) & set(metrics2.keys())
        
        if len(common_metrics) < 2:
            return 0.0
        
        values1 = [metrics1[m] for m in common_metrics]
        values2 = [metrics2[m] for m in common_metrics]
        
        return np.corrcoef(values1, values2)[0, 1] if len(values1) > 1 else 0.0

    def _calculate_clustering_diversity(self,
                                      candidate: ModelScore,
                                      selected: List[ModelScore],
                                      candidates: List[Dict[str, Any]]) -> float:
        """Calculate clustering-based diversity."""
        # Simplified implementation
        # In practice, you'd use model architecture features
        return np.random.random()  # Placeholder

    def _create_deployment_backup(self, production_path: str) -> None:
        """Create backup of current production models."""
        backup_dir = os.path.join(production_path, "backups", str(int(time.time())))
        
        if os.path.exists(production_path):
            os.makedirs(backup_dir, exist_ok=True)
            
            for file in os.listdir(production_path):
                if file.endswith(('.pth', '.pt', '.pkl')):
                    src = os.path.join(production_path, file)
                    dst = os.path.join(backup_dir, file)
                    shutil.copy2(src, dst)
            
            logger.info(f"Created deployment backup at {backup_dir}")

    def _calculate_comparison_summary(self, metrics: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
        """Calculate summary statistics for model comparison."""
        summary = {}
        
        for metric, values in metrics.items():
            valid_values = [v for v in values.values() if v is not None]
            
            if valid_values:
                summary[metric] = {
                    "mean": np.mean(valid_values),
                    "std": np.std(valid_values),
                    "min": np.min(valid_values),
                    "max": np.max(valid_values),
                    "range": np.max(valid_values) - np.min(valid_values)
                }
        
        return summary

    def _generate_recommendation_reason(self,
                                      scored_model: ModelScore,
                                      use_case: str,
                                      risk_tolerance: str) -> str:
        """Generate explanation for model recommendation."""
        reasons = []
        
        if scored_model.performance_score > 0.8:
            reasons.append("excellent performance metrics")
        elif scored_model.performance_score > 0.6:
            reasons.append("good performance metrics")
        
        if scored_model.stability_score > 0.8:
            reasons.append("high stability")
        elif scored_model.stability_score > 0.6:
            reasons.append("moderate stability")
        
        if use_case == "trading" and risk_tolerance == "low":
            reasons.append("suitable for conservative trading")
        elif use_case == "trading" and risk_tolerance == "high":
            reasons.append("suitable for aggressive trading")
        
        return f"Recommended due to {', '.join(reasons)}"

# Legacy compatibility
ModelSelector = AdvancedModelSelector

if __name__ == "__main__":
    # Example usage with ARM64 optimizations
    import os
    import shutil
    from deep_momentum_trading.src.models.deep_momentum_lstm import DeepMomentumLSTM
    from torch.utils.data import TensorDataset, DataLoader
    
    # Setup test registry
    test_registry_path = "test_advanced_model_selection_registry.json"
    if os.path.exists(test_registry_path):
        os.remove(test_registry_path)
    
    registry = ModelRegistry(test_registry_path)
    
    # Configuration
    config = {
        "use_arm64_optimizations": True,
        "primary_metric": "sharpe_ratio",
        "cache_dir": "test_selection_cache",
        "n_jobs": 1
    }
    
    selector = AdvancedModelSelector(registry, config)
    
    # Create test models
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_size = 200
    num_assets = 100
    
    # Model 1: High performance, high stability
    model_config_1 = ModelConfig(
        model_type="deep_momentum_lstm",
        parameters={"input_size": input_size, "hidden_size": 256, "num_layers": 2, "dropout": 0.1},
        description="High Performance LSTM",
        tags=["test", "stable"]
    )
    registry.register_model("model_hp", model_config_1)
    registry.update_model_performance("model_hp", {
        "sharpe_ratio": 2.1, "calmar_ratio": 1.8, "sortino_ratio": 2.3,
        "max_drawdown": 0.06, "volatility": 0.15, "total_return": 0.25
    })
    
    # Model 2: Medium performance, very stable
    model_config_2 = ModelConfig(
        model_type="deep_momentum_lstm",
        parameters={"input_size": input_size, "hidden_size": 128, "num_layers": 1, "dropout": 0.05},
        description="Stable LSTM",
        tags=["test", "conservative"]
    )
    registry.register_model("model_stable", model_config_2)
    registry.update_model_performance("model_stable", {
        "sharpe_ratio": 1.5, "calmar_ratio": 1.6, "sortino_ratio": 1.7,
        "max_drawdown": 0.03, "volatility": 0.08, "total_return": 0.15
    })
    
    # Model 3: High performance, less stable
    model_config_3 = ModelConfig(
        model_type="transformer_momentum",
        parameters={"input_size": input_size, "hidden_size": 512, "num_layers": 4, "dropout": 0.2},
        description="High Risk Transformer",
        tags=["test", "aggressive"]
    )
    registry.register_model("model_risky", model_config_3)
    registry.update_model_performance("model_risky", {
        "sharpe_ratio": 2.5, "calmar_ratio": 1.2, "sortino_ratio": 2.8,
        "max_drawdown": 0.15, "volatility": 0.25, "total_return": 0.35
    })
    
    print("=== Advanced Model Selection Test ===")
    
    # Test best model selection
    print("\n1. Best Model Selection:")
    best_model = selector.select_best_model()
    if best_model:
        print(f"   Best model: {best_model['name']}")
        print(f"   Composite score: {best_model['selection_score']['composite_score']:.4f}")
    
    # Test ensemble selection
    print("\n2. Ensemble Selection:")
    ensemble = selector.select_ensemble_models(num_models=2)
    for i, model in enumerate(ensemble):
        print(f"   {i+1}. {model['name']} (score: {model['selection_score']['composite_score']:.4f})")
    
    # Test model comparison
    print("\n3. Model Comparison:")
    comparison = selector.compare_models(["model_hp", "model_stable", "model_risky"])
    for metric, rankings in comparison["rankings"].items():
        print(f"   {metric}: {' > '.join(rankings)}")
    
    # Test recommendations
    print("\n4. Recommendations for Conservative Trading:")
    recommendations = selector.get_model_recommendations(
        use_case="trading",
        risk_tolerance="low"
    )
    for i, rec in enumerate(recommendations[:2]):
        print(f"   {i+1}. {rec['name']}: {rec['recommendation_reason']}")
    
    # Test deployment
    print("\n5. Model Deployment:")
    success = selector.deploy_model("model_hp", {
        "optimize_for_inference": True,
        "create_backup": False
    })
    print(f"   Deployment successful: {success}")
    
    # Cleanup
    if os.path.exists(test_registry_path):
        os.remove(test_registry_path)
    
    if os.path.exists("test_selection_cache"):
        shutil.rmtree("test_selection_cache")
    
    production_dir = "deep_momentum_trading/data/models/production/"
    if os.path.exists(production_dir):
        shutil.rmtree(production_dir)
    
    print("\n=== Advanced Model Selection Test Complete ===")
