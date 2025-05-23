"""
Scalable portfolio optimization system for Deep Momentum Trading System with massive-scale support.

This module provides enterprise-grade portfolio optimization capabilities for 10,000+ assets
with ARM64-specific optimizations for NVIDIA GH200 Grace Hopper platform, advanced risk models,
hierarchical clustering, and distributed optimization strategies.
"""

import numpy as np
import pandas as pd
import threading
import time
import asyncio
import multiprocessing as mp
from typing import Dict, List, Tuple, Optional, Any, Union, Set, Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from collections import defaultdict, deque
import warnings
import platform
import psutil
import gc
from functools import lru_cache, partial
import pickle
import json
import logging

# Scientific computing imports
try:
    import cupy as cp
    import cupyx.scipy.optimize as cp_optimize
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False

try:
    import torch
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# Optimization libraries
from scipy.optimize import minimize, differential_evolution, NonlinearConstraint
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from scipy.linalg import cholesky, LinAlgError
from sklearn.covariance import LedoitWolf, OAS, ShrunkCovariance
from sklearn.cluster import KMeans, AgglomerativeClustering

# Internal imports
from ..utils.logger import get_logger, PerformanceLogger
from ..utils.decorators import performance_monitor, retry_with_backoff, async_performance_monitor
from ..utils.exceptions import RiskError, ValidationError, OptimizationError, ScalingError
from ..utils.validators import validate_numeric_data, validate_batch_data
from ..utils.shared_memory import create_shared_array, create_shared_dict, SharedArray, SharedDict
from ..utils.shared_memory import create_shared_array, create_shared_dict, SharedArray, SharedDict
from ..utils.memory_pool import MemoryPool, UnifiedMemoryPool
from ..utils.profiler import SystemProfiler, PerformanceProfiler

# ARM64 and platform detection
IS_ARM64 = platform.machine().lower() in ['arm64', 'aarch64']
IS_NVIDIA_GH200 = IS_ARM64 and any('nvidia' in line.lower() for line in open('/proc/cpuinfo', 'r').readlines() if 'model name' in line.lower()) if platform.system() == 'Linux' else False

logger = get_logger(__name__)
perf_logger = PerformanceLogger(logger)

class OptimizationMethod(Enum):
    """Advanced portfolio optimization methods for massive-scale operations."""
    MEAN_VARIANCE = "mean_variance"
    RISK_PARITY = "risk_parity"
    HIERARCHICAL_RISK_PARITY = "hierarchical_risk_parity"
    BLACK_LITTERMAN = "black_litterman"
    MINIMUM_VARIANCE = "minimum_variance"
    MAXIMUM_DIVERSIFICATION = "maximum_diversification"
    EQUAL_WEIGHT = "equal_weight"
    MOMENTUM_WEIGHTED = "momentum_weighted"
    CRITICAL_LINE_ALGORITHM = "critical_line_algorithm"
    NESTED_CLUSTERED_OPTIMIZATION = "nested_clustered_optimization"
    MULTI_OBJECTIVE_OPTIMIZATION = "multi_objective_optimization"
    ROBUST_OPTIMIZATION = "robust_optimization"
    FACTOR_RISK_PARITY = "factor_risk_parity"
    ADAPTIVE_ALLOCATION = "adaptive_allocation"

class RiskModel(Enum):
    """Advanced risk model types for massive-scale portfolios."""
    SAMPLE_COVARIANCE = "sample_covariance"
    SHRINKAGE_COVARIANCE = "shrinkage_covariance"
    LEDOIT_WOLF = "ledoit_wolf"
    OAS_ESTIMATOR = "oas_estimator"
    FACTOR_MODEL = "factor_model"
    ROBUST_COVARIANCE = "robust_covariance"
    DYNAMIC_CONDITIONAL_CORRELATION = "dcc"
    GARCH_COVARIANCE = "garch_covariance"
    RANDOM_MATRIX_THEORY = "rmt_denoising"

class OptimizationObjective(Enum):
    """Optimization objectives for different strategies."""
    MAXIMIZE_SHARPE = "maximize_sharpe"
    MINIMIZE_VARIANCE = "minimize_variance"
    MAXIMIZE_RETURN = "maximize_return"
    MINIMIZE_DRAWDOWN = "minimize_drawdown"
    MAXIMIZE_DIVERSIFICATION = "maximize_diversification"
    RISK_PARITY = "risk_parity"
    UTILITY_MAXIMIZATION = "utility_maximization"

@dataclass
class ScalableOptimizerConfig:
    """Enhanced configuration for massive-scale portfolio optimization."""
    # Scale parameters
    max_assets: int = 10000
    batch_size: int = 2000
    optimization_batch_size: int = 1000
    max_concurrent_optimizations: int = min(32, mp.cpu_count())
    
    # Basic optimization settings
    risk_aversion: float = 1.0
    target_sharpe_ratio: float = 6.0
    max_position_concentration: float = 0.001  # 0.1% max per position for massive diversification
    max_sector_concentration: float = 0.15     # 15% max per sector
    enable_massive_diversification: bool = True
    
    # Optimization methods and objectives
    primary_method: OptimizationMethod = OptimizationMethod.NESTED_CLUSTERED_OPTIMIZATION
    fallback_method: OptimizationMethod = OptimizationMethod.EQUAL_WEIGHT
    risk_model: RiskModel = RiskModel.LEDOIT_WOLF
    optimization_objective: OptimizationObjective = OptimizationObjective.MAXIMIZE_SHARPE
    
    # ARM64 and GPU optimizations
    enable_arm64_optimizations: bool = True
    enable_gpu_acceleration: bool = True
    enable_mixed_precision: bool = True
    enable_cuda_graphs: bool = True
    enable_vectorized_calculations: bool = True
    enable_parallel_processing: bool = True
    
    # Performance settings
    optimization_timeout_seconds: float = 600.0  # 10 minutes for massive portfolios
    max_iterations: int = 5000
    convergence_tolerance: float = 1e-8
    enable_caching: bool = True
    cache_ttl_seconds: int = 1800  # 30 minutes
    enable_warm_start: bool = True
    
    # Advanced risk constraints
    max_portfolio_volatility: float = 0.15  # 15% annual
    min_expected_return: float = 0.08       # 8% annual
    max_drawdown_constraint: float = 0.08   # 8%
    enable_turnover_constraint: bool = True
    max_turnover: float = 0.30              # 30% turnover limit
    enable_leverage_constraint: bool = True
    max_leverage: float = 1.0               # No leverage
    
    # Massive diversification settings
    massive_diversification_threshold: int = 500
    hierarchical_clustering_enabled: bool = True
    cluster_risk_budget_method: str = "risk_parity"  # "equal", "inverse_variance", "risk_parity"
    max_clusters: int = 50
    min_cluster_size: int = 10
    
    # Advanced optimization features
    enable_robust_optimization: bool = True
    uncertainty_set_size: float = 0.05      # 5% uncertainty
    enable_transaction_costs: bool = True
    transaction_cost_bps: float = 5.0       # 5 basis points
    enable_factor_exposure_constraints: bool = True
    
    # Memory management
    enable_shared_memory: bool = True
    enable_memory_pool: bool = True
    shared_memory_size: int = 500000000     # 500MB
    memory_pool_size: int = 1000000000      # 1GB
    
    # Monitoring and alerting
    enable_real_time_monitoring: bool = True
    enable_performance_attribution: bool = True
    enable_risk_decomposition: bool = True

@dataclass
class ScalableOptimizationResult:
    """Comprehensive optimization result for massive-scale portfolios."""
    target_weights: Dict[str, float]
    expected_return: float
    expected_volatility: float
    sharpe_ratio: float
    optimization_method: OptimizationMethod
    
    # Enhanced metrics
    risk_contributions: Dict[str, float]
    factor_exposures: Dict[str, float]
    sector_allocations: Dict[str, float]
    cluster_allocations: Dict[str, float]
    
    # Performance metrics
    optimization_time_ms: float
    convergence_achieved: bool
    constraints_satisfied: bool
    turnover: float
    diversification_ratio: float
    effective_number_of_assets: float
    
    # Risk metrics
    portfolio_var_95: float
    portfolio_cvar_95: float
    maximum_drawdown_estimate: float
    tracking_error: float
    information_ratio: float
    
    # Advanced analytics
    factor_risk_contributions: Dict[str, float]
    concentration_metrics: Dict[str, float]
    stress_test_results: Dict[str, float]
    
    # Metadata
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    optimization_id: str = field(default_factory=lambda: str(int(time.time() * 1000000)))
    assets_count: int = 0
    clusters_used: int = 0

@dataclass
class ScalablePortfolioMetrics:
    """Comprehensive portfolio performance metrics for massive-scale analysis."""
    # Return metrics
    total_return: float
    annualized_return: float
    excess_return: float
    
    # Risk metrics
    volatility: float
    downside_volatility: float
    tracking_error: float
    
    # Risk-adjusted metrics
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    information_ratio: float
    treynor_ratio: float
    
    # Drawdown metrics
    max_drawdown: float
    avg_drawdown: float
    drawdown_duration: float
    
    # Risk measures
    var_95: float
    var_99: float
    cvar_95: float
    cvar_99: float
    
    # Market exposure
    beta: float
    alpha: float
    correlation_to_benchmark: float
    
    # Diversification metrics
    diversification_ratio: float
    effective_number_of_assets: float
    concentration_index: float
    
    # Factor exposures
    factor_loadings: Dict[str, float] = field(default_factory=dict)
    sector_exposures: Dict[str, float] = field(default_factory=dict)
    
    # Performance attribution
    selection_effect: float = 0.0
    allocation_effect: float = 0.0
    interaction_effect: float = 0.0

class ScalablePortfolioOptimizer:
    """Enterprise-grade portfolio optimizer for massive-scale trading operations."""
    
    def __init__(self, config: Optional[ScalableOptimizerConfig] = None):
        """Initialize scalable portfolio optimizer with massive-scale support."""
        self.config = config or ScalableOptimizerConfig()
        self.is_arm64 = IS_ARM64
        self.is_nvidia_gh200 = IS_NVIDIA_GH200
        
        # Apply platform-specific optimizations
        if self.is_arm64 and self.config.enable_arm64_optimizations:
            self._apply_arm64_optimizations()
        
        # Initialize GPU support
        self.gpu_available = CUPY_AVAILABLE and self.config.enable_gpu_acceleration
        self.torch_available = TORCH_AVAILABLE
        
        # Data storage with massive-scale support
        self.expected_returns: Dict[str, float] = {}
        self.covariance_matrix: Optional[pd.DataFrame] = None
        self.factor_loadings: Dict[str, Dict[str, float]] = {}
        self.sector_mapping: Dict[str, str] = {}
        self.market_caps: Dict[str, float] = {}
        self.asset_universe: Set[str] = set()
        
        # Optimization state
        self.optimization_history: deque = deque(maxlen=1000)
        self.performance_metrics: Dict[str, Any] = {}
        self.cluster_assignments: Dict[str, int] = {}
        self.factor_models: Dict[str, Any] = {}
        
        # Performance tracking
        self.calculation_stats = {
            "total_optimizations": 0,
            "successful_optimizations": 0,
            "avg_optimization_time_ms": 0.0,
            "peak_memory_usage_mb": 0.0,
            "cache_hits": 0,
            "cache_misses": 0,
            "gpu_optimizations": 0,
            "cpu_optimizations": 0,
            "total_assets_optimized": 0
        }
        
        # Threading and async
        self._lock = threading.RLock()
        self._running = False
        self._background_tasks: List[threading.Thread] = []
        self._executor = ThreadPoolExecutor(max_workers=self.config.max_concurrent_optimizations)
        self._process_executor = ProcessPoolExecutor(max_workers=min(8, mp.cpu_count()))
        
        # Memory management
        self._setup_memory_management()
        
        # Caching system
        self._optimization_cache: Dict[str, Tuple[ScalableOptimizationResult, datetime]] = {}
        self._lru_cache_size = min(1000, self.config.max_assets)
        
        # Profiling
        self.system_profiler = SystemProfiler()
        self.performance_profiler = PerformanceProfiler()
        
        # Initialize GPU resources
        if self.gpu_available:
            self._initialize_gpu_resources()
        
        logger.info(f"ScalablePortfolioOptimizer initialized for {self.config.max_assets} assets "
                   f"(ARM64: {self.is_arm64}, GH200: {self.is_nvidia_gh200}, GPU: {self.gpu_available})")
    
    def _apply_arm64_optimizations(self):
        """Apply comprehensive ARM64 optimizations for NVIDIA GH200 platform."""
        # Optimize for Grace Hopper architecture
        if self.is_nvidia_gh200:
            # Ultra-fast optimization for GH200
            self.config.optimization_timeout_seconds = max(300.0, self.config.optimization_timeout_seconds * 0.5)
            
            # Maximize parallel workers for Grace CPU
            self.config.max_concurrent_optimizations = min(72, mp.cpu_count())  # Grace has up to 144 cores
            
            # Optimize batch sizes for unified memory
            self.config.batch_size = min(5000, self.config.batch_size * 2)
            self.config.optimization_batch_size = min(2000, self.config.optimization_batch_size * 2)
            
            # Enable all advanced features
            self.config.enable_mixed_precision = True
            self.config.enable_cuda_graphs = True
            self.config.enable_memory_pool = True
            
        else:
            # Standard ARM64 optimizations
            self.config.optimization_timeout_seconds = max(400.0, self.config.optimization_timeout_seconds * 0.7)
            self.config.max_concurrent_optimizations = min(16, self.config.max_concurrent_optimizations * 2)
        
        # Enable all vectorized operations
        self.config.enable_vectorized_calculations = True
        self.config.enable_parallel_processing = True
        
        logger.info(f"Applied ARM64 optimizations: timeout={self.config.optimization_timeout_seconds}s, "
                   f"workers={self.config.max_concurrent_optimizations}, GH200={self.is_nvidia_gh200}")
    
    def _setup_memory_management(self):
        """Setup advanced memory management for massive-scale operations."""
        try:
            # Unified memory pool for ARM64/GH200
            if self.config.enable_memory_pool:
                self.memory_pool = UnifiedMemoryPool(
                    size=self.config.memory_pool_size,
                    enable_gpu=self.gpu_available and self.is_nvidia_gh200
                )
            
            # Shared memory for optimization results
            if self.config.enable_shared_memory:
                self.shared_weights = create_shared_array(
                    name="portfolio_weights_scalable",
                    size=self.config.max_assets,
                    dtype=np.float32 if self.config.enable_mixed_precision else np.float64
                )
                
                self.shared_metadata = create_shared_dict(
                    name="portfolio_metadata_scalable",
                    max_items=10000
                )
                
                # Additional shared arrays for covariance matrices
                matrix_size = min(self.config.max_assets * self.config.max_assets, self.config.shared_memory_size)
                self.shared_covariance = create_shared_array(
                    name="covariance_matrix_scalable",
                    size=matrix_size,
                    dtype=np.float32 if self.config.enable_mixed_precision else np.float64
                )
            
        except Exception as e:
            logger.warning(f"Failed to setup advanced memory management: {e}")
            self.memory_pool = None
            self.shared_weights = None
            self.shared_metadata = None
            self.shared_covariance = None
    
    def _initialize_gpu_resources(self):
        """Initialize GPU resources for optimization calculations."""
        if not self.gpu_available:
            return
        
        try:
            # Initialize CuPy memory pool
            cp.cuda.MemoryPool().set_limit(size=self.config.memory_pool_size // 2)
            
            # Pre-allocate GPU arrays for common operations
            max_batch = self.config.optimization_batch_size
            self.gpu_optimization_workspace = cp.zeros((max_batch, max_batch), dtype=cp.float32)
            self.gpu_weights_buffer = cp.zeros(max_batch, dtype=cp.float32)
            
            # Initialize CUDA graphs if available
            if self.config.enable_cuda_graphs and hasattr(cp.cuda, 'Graph'):
                self._setup_cuda_graphs()
            
            logger.info("GPU resources initialized for portfolio optimization")
            
        except Exception as e:
            logger.warning(f"Failed to initialize GPU resources: {e}")
            self.gpu_available = False
    
    def _setup_cuda_graphs(self):
        """Setup CUDA graphs for optimized portfolio calculations."""
        try:
            # Create CUDA graph for optimization
            self.optimization_graph = cp.cuda.Graph()
            
            with self.optimization_graph.capture():
                # Dummy optimization calculation to capture the graph
                dummy_cov = cp.random.randn(100, 100)
                dummy_cov = dummy_cov @ dummy_cov.T
                cp.linalg.inv(dummy_cov)
            
            logger.debug("CUDA graphs initialized for portfolio optimization")
            
        except Exception as e:
            logger.warning(f"Failed to setup CUDA graphs: {e}")
            self.optimization_graph = None
    
    @performance_monitor
    def optimize_portfolio_scalable(self,
                                  expected_returns: Dict[str, float],
                                  covariance_matrix: Union[Dict[Tuple[str, str], float], pd.DataFrame, np.ndarray],
                                  current_positions: Dict[str, float],
                                  total_portfolio_value: float,
                                  sector_mapping: Optional[Dict[str, str]] = None,
                                  market_caps: Optional[Dict[str, float]] = None,
                                  factor_loadings: Optional[Dict[str, Dict[str, float]]] = None,
                                  benchmark_weights: Optional[Dict[str, float]] = None) -> ScalableOptimizationResult:
        """
        Optimize massive-scale portfolio with advanced methods and ARM64 optimizations.
        
        Args:
            expected_returns: Expected returns for each asset
            covariance_matrix: Covariance matrix (various formats supported)
            current_positions: Current portfolio positions
            total_portfolio_value: Total portfolio value
            sector_mapping: Asset to sector mapping
            market_caps: Market capitalizations
            factor_loadings: Factor loadings for each asset
            benchmark_weights: Benchmark weights for tracking error optimization
            
        Returns:
            ScalableOptimizationResult with comprehensive metrics
        """
        start_time = time.time()
        
        try:
            # Validate and prepare inputs
            assets = self._validate_and_prepare_inputs_scalable(
                expected_returns, covariance_matrix, current_positions,
                sector_mapping, market_caps, factor_loadings
            )
            
            if not assets:
                raise OptimizationError("No valid assets for optimization")
            
            # Check cache
            cache_key = self._generate_cache_key_scalable(expected_returns, current_positions, assets)
            if self.config.enable_caching and cache_key in self._optimization_cache:
                cached_result, cache_time = self._optimization_cache[cache_key]
                if (datetime.now(timezone.utc) - cache_time).total_seconds() < self.config.cache_ttl_seconds:
                    self.calculation_stats["cache_hits"] += 1
                    return cached_result
            
            # Prepare optimization data
            mu = np.array([expected_returns[asset] for asset in assets])
            sigma = self._prepare_covariance_matrix_scalable(assets)
            
            # Choose optimization method based on portfolio size and complexity
            num_assets = len(assets)
            method = self._select_optimization_method(num_assets)
            
            # Perform optimization with massive-scale support
            if num_assets >= self.config.massive_diversification_threshold:
                target_weights = self._optimize_massive_scale(method, mu, sigma, assets, current_positions)
            else:
                target_weights = self._optimize_standard_scale(method, mu, sigma, assets, current_positions)
            
            # Calculate comprehensive portfolio metrics
            portfolio_metrics = self._calculate_portfolio_metrics_scalable(
                target_weights, mu, sigma, assets, benchmark_weights
            )
            
            # Validate constraints
            constraints_satisfied = self._validate_constraints_scalable(target_weights, assets)
            
            # Create comprehensive result
            optimization_time_ms = (time.time() - start_time) * 1000
            
            result = ScalableOptimizationResult(
                target_weights=target_weights,
                expected_return=portfolio_metrics["expected_return"],
                expected_volatility=portfolio_metrics["expected_volatility"],
                sharpe_ratio=portfolio_metrics["sharpe_ratio"],
                optimization_method=method,
                risk_contributions=portfolio_metrics["risk_contributions"],
                factor_exposures=portfolio_metrics["factor_exposures"],
                sector_allocations=portfolio_metrics["sector_allocations"],
                cluster_allocations=portfolio_metrics["cluster_allocations"],
                optimization_time_ms=optimization_time_ms,
                convergence_achieved=True,
                constraints_satisfied=constraints_satisfied,
                turnover=self._calculate_turnover_scalable(target_weights, current_positions, total_portfolio_value),
                diversification_ratio=portfolio_metrics["diversification_ratio"],
                effective_number_of_assets=portfolio_metrics["effective_number_of_assets"],
                portfolio_var_95=portfolio_metrics["var_95"],
                portfolio_cvar_95=portfolio_metrics["cvar_95"],
                maximum_drawdown_estimate=portfolio_metrics["max_drawdown_estimate"],
                tracking_error=portfolio_metrics["tracking_error"],
                information_ratio=portfolio_metrics["information_ratio"],
                factor_risk_contributions=portfolio_metrics["factor_risk_contributions"],
                concentration_metrics=portfolio_metrics["concentration_metrics"],
                stress_test_results=portfolio_metrics["stress_test_results"],
                assets_count=num_assets,
                clusters_used=portfolio_metrics.get("clusters_used", 0)
            )
            
            # Cache result
            if self.config.enable_caching:
                self._optimization_cache[cache_key] = (result, datetime.now(timezone.utc))
                self.calculation_stats["cache_misses"] += 1
            
            # Update statistics
            self._update_calculation_stats(optimization_time_ms, num_assets)
            
            # Store in history
            self.optimization_history.append(result)
            
            # Update shared memory
            self._update_shared_memory_scalable(result)
            
            perf_logger.log_latency("portfolio_optimization_scalable", optimization_time_ms)
            
            logger.info(f"Scalable portfolio optimization completed: {num_assets} assets, "
                       f"method={method.value}, Sharpe={result.sharpe_ratio:.3f}, "
                       f"time={optimization_time_ms:.1f}ms")
            
            return result
            
        except Exception as e:
            logger.error(f"Scalable portfolio optimization failed: {e}")
            self.calculation_stats["total_optimizations"] += 1
            
            # Return fallback result
            return self._create_fallback_result_scalable(list(expected_returns.keys()), start_time)
    
    def _validate_and_prepare_inputs_scalable(self, expected_returns, covariance_matrix,
                                            current_positions, sector_mapping, market_caps,
                                            factor_loadings) -> List[str]:
        """Validate and prepare optimization inputs for massive-scale operations."""
        # Validate expected returns
        if not expected_returns:
            raise ValidationError("Expected returns cannot be empty")
        
        assets = []
        for symbol, ret in expected_returns.items():
            if validate_numeric_data(ret, allow_negative=True):
                assets.append(symbol)
                self.asset_universe.add(symbol)
            else:
                logger.warning(f"Invalid expected return for {symbol}: {ret}")
        
        if len(assets) < 2:
            raise ValidationError("Need at least 2 valid assets for optimization")
        
        # Limit to maximum assets for performance
        if len(assets) > self.config.max_assets:
            # Prioritize by data quality and market cap
            assets = self._select_top_assets_for_optimization(assets, self.config.max_assets)
        
        # Store data
        self.expected_returns = {asset: expected_returns[asset] for asset in assets}
        self.sector_mapping = sector_mapping or {}
        self.market_caps = market_caps or {}
        self.factor_loadings = factor_loadings or {}
        
        # Prepare covariance matrix
        self.covariance_matrix = self._prepare_covariance_input(covariance_matrix, assets)
        
        return assets
    
    def _select_top_assets_for_optimization(self, assets: List[str], max_count: int) -> List[str]:
        """Select top assets for optimization based on multiple criteria."""
        asset_scores = []
        
        for asset in assets:
            score = 0.0
            
            # Expected return score (normalized)
            expected_return = self.expected_returns.get(asset, 0.0)
            return_score = max(0.0, expected_return) * 10.0
            
            # Market cap score (normalized)
            market_cap = self.market_caps.get(asset, 0.0)
            market_cap_score = np.log1p(market_cap) / 30.0
            
            # Sector diversification bonus
            sector = self.sector_mapping.get(asset, "Unknown")
            sector_count = sum(1 for a in assets if self.sector_mapping.get(a, "Unknown") == sector)
            diversification_bonus = 1.0 / np.sqrt(sector_count) if sector_count > 0 else 1.0
            
            # Combined score
            total_score = return_score * 0.4 + market_cap_score * 0.4 + diversification_bonus * 0.2
            asset_scores.append((asset, total_score))
        
        # Sort by score and return top assets
        asset_scores.sort(key=lambda x: x[1], reverse=True)
        return [asset for asset, _ in asset_scores[:max_count]]
    
    def _prepare_covariance_input(self, covariance_matrix, assets: List[str]) -> pd.DataFrame:
        """Prepare covariance matrix from various input formats."""
        if isinstance(covariance_matrix, dict):
            return self._dict_to_dataframe_cov_scalable(covariance_matrix, assets)
        elif isinstance(covariance_matrix, pd.DataFrame):
            return covariance_matrix.loc[assets, assets]
        elif isinstance(covariance_matrix, np.ndarray):
            return pd.DataFrame(covariance_matrix, index=assets, columns=assets)
        else:
            raise ValidationError("Invalid covariance matrix format")
    
    def _dict_to_dataframe_cov_scalable(self, cov_dict: Dict[Tuple[str, str], float], assets: List[str]) -> pd.DataFrame:
        """Convert covariance dictionary to DataFrame with massive-scale optimizations."""
        n = len(assets)
        cov_matrix = np.zeros((n, n), dtype=np.float32 if self.config.enable_mixed_precision else np.float64)
        
        # Vectorized approach for massive-scale
        asset_to_idx = {asset: i for i, asset in enumerate(assets)}
        
        for (asset1, asset2), value in cov_dict.items():
            if asset1 in asset_to_idx and asset2 in asset_to_idx:
                i, j = asset_to_idx[asset1], asset_to_idx[asset2]
                cov_matrix[i, j] = value
                if i != j:  # Ensure symmetry
                    cov_matrix[j, i] = value
        
        # Fill diagonal with default variance if missing
        for i in range(n):
            if cov_matrix[i, i] == 0:
                cov_matrix[i, i] = 0.0001  # Default variance
        
        return pd.DataFrame(cov_matrix, index=assets, columns=assets)
    
    def _prepare_covariance_matrix_scalable(self, assets: List[str]) -> np.ndarray:
        """Prepare covariance matrix with advanced risk model adjustments."""
        if self.covariance_matrix is None:
            raise ValidationError("Covariance matrix not available")
        
        sigma = self.covariance_matrix.loc[assets, assets].values
        
        # Apply advanced risk model adjustments
        if self.config.risk_model == RiskModel.LEDOIT_WOLF:
            sigma = self._apply_ledoit_wolf_shrinkage(sigma)
        elif self.config.risk_model == RiskModel.OAS_ESTIMATOR:
            sigma = self._apply_oas_shrinkage(sigma)
        elif self.config.risk_model == RiskModel.ROBUST_COVARIANCE:
            sigma = self._apply_robust_estimator_scalable(sigma)
        elif self.config.risk_model == RiskModel.RANDOM_MATRIX_THEORY:
            sigma = self._apply_rmt_denoising(sigma)
        
        # Ensure numerical stability and positive definiteness
        sigma = self._ensure_positive_definite(sigma)
        
        return sigma
    
    def _apply_ledoit_wolf_shrinkage(self, sigma: np.ndarray) -> np.ndarray:
        """Apply Ledoit-Wolf shrinkage estimator for massive-scale covariance."""
        try:
            # Use sklearn's implementation for efficiency
            lw = LedoitWolf()
            # Create dummy data for fitting (simplified approach)
            n_samples = max(100, sigma.shape[0] * 2)
            dummy_data = np.random.multivariate_normal(np.zeros(sigma.shape[0]), sigma, n_samples)
            
            lw.fit(dummy_data)
            return lw.covariance_
            
        except Exception as e:
            logger.warning(f"Ledoit-Wolf shrinkage failed: {e}")
            return self._apply_simple_shrinkage(sigma)
    
    def _apply_oas_shrinkage(self, sigma: np.ndarray) -> np.ndarray:
        """Apply Oracle Approximating Shrinkage estimator."""
        try:
            oas = OAS()
            n_samples = max(100, sigma.shape[0] * 2)
            dummy_data = np.random.multivariate_normal(np.zeros(sigma.shape[0]), sigma, n_samples)
            
            oas.fit(dummy_data)
            return oas.covariance_
            
        except Exception as e:
            logger.warning(f"OAS shrinkage failed: {e}")
            return self._apply_simple_shrinkage(sigma)
    
    def _apply_simple_shrinkage(self, sigma: np.ndarray, shrinkage: float = 0.1) -> np.ndarray:
        """Apply simple shrinkage to identity matrix."""
        n = sigma.shape[0]
        target = np.eye(n) * np.trace(sigma) / n
        return (1 - shrinkage) * sigma + shrinkage * target
    
    def _apply_robust_estimator_scalable(self, sigma: np.ndarray) -> np.ndarray:
        """Apply robust covariance estimator for massive-scale portfolios."""
        try:
            # Winsorize extreme correlations
            std_devs = np.sqrt(np.diag(sigma))
            corr = sigma / np.outer(std_devs, std_devs)
            
            # Clip extreme correlations
            corr = np.clip(corr, -0.95, 0.95)
            
            # Ensure diagonal is 1
            np.fill_diagonal(corr, 1.0)
            
            # Reconstruct covariance
            sigma_robust = corr * np.outer(std_devs, std_devs)
            
            return sigma_robust
            
        except Exception as e:
            logger.warning(f"Robust estimation failed: {e}")
            return sigma
    
    def _apply_rmt_denoising(self, sigma: np.ndarray) -> np.ndarray:
        """Apply Random Matrix Theory denoising for massive portfolios."""
        try:
            # Eigenvalue decomposition
            eigenvals, eigenvecs = np.linalg.eigh(sigma)
            
            # Marchenko-Pastur distribution parameters
            n, p = sigma.shape[0], sigma.shape[0]
            q = p / n if n > 0 else 1.0
            
            # Theoretical bounds for noise eigenvalues
            lambda_plus = (1 + np.sqrt(q))**2
            lambda_minus = (1 - np.sqrt(q))**2
            
            # Filter eigenvalues (keep only signal)
            signal_mask = (eigenvals > lambda_plus) | (eigenvals < lambda_minus)
            
            # Reconstruct with filtered eigenvalues
            eigenvals_filtered = eigenvals.copy()
            eigenvals_filtered[~signal_mask] = np.median(eigenvals[~signal_mask])
            
            # Ensure positive eigenvalues
            eigenvals_filtered = np.maximum(eigenvals_filtered, 1e-8)
            
            sigma_denoised = eigenvecs @ np.diag(eigenvals_filtered) @ eigenvecs.T
            
            return sigma_denoised
            
        except Exception as e:
            logger.warning(f"RMT denoising failed: {e}")
            return sigma
    
    def _ensure_positive_definite(self, sigma: np.ndarray) -> np.ndarray:
        """Ensure covariance matrix is positive definite."""
        try:
            # Add small regularization to diagonal
            sigma += np.eye(sigma.shape[0]) * 1e-8
            
            # Check if already positive definite
            try:
                np.linalg.cholesky(sigma)
                return sigma
            except np.linalg.LinAlgError:
                pass
            
            # Make positive definite using eigenvalue decomposition
            eigenvals, eigenvecs = np.linalg.eigh(sigma)
            eigenvals = np.maximum(eigenvals, 1e-8)
            
            return eigenvecs @ np.diag(eigenvals) @ eigenvecs.T
            
        except Exception as e:
            logger.error(f"Failed to ensure positive definiteness: {e}")
            # Return identity matrix as last resort
            return np.eye(sigma.shape[0]) * 0.01
    
    def _select_optimization_method(self, num_assets: int) -> OptimizationMethod:
        """Select optimal optimization method based on portfolio size."""
        if num_assets >= self.config.massive_diversification_threshold:
            if self.config.hierarchical_clustering_enabled:
                return OptimizationMethod.NESTED_CLUSTERED_OPTIMIZATION
            else:
                return OptimizationMethod.HIERARCHICAL_RISK_PARITY
        elif num_assets >= 100:
            return OptimizationMethod.RISK_PARITY
        else:
            return self.config.primary_method
    
    def _optimize_massive_scale(self, method: OptimizationMethod, mu: np.ndarray,
                              sigma: np.ndarray, assets: List[str],
                              current_positions: Dict[str, float]) -> Dict[str, float]:
        """Optimize portfolio for massive-scale (1000+ assets)."""
        try:
            if method == OptimizationMethod.NESTED_CLUSTERED_OPTIMIZATION:
                return self._optimize_nested_clustered(mu, sigma, assets)
            elif method == OptimizationMethod.HIERARCHICAL_RISK_PARITY:
                return self._optimize_hierarchical_risk_parity_scalable(mu, sigma, assets)
            else:
                # Fall back to batch optimization
                return self._optimize_batch_processing(method, mu, sigma, assets)
                
        except Exception as e:
            logger.error(f"Massive-scale optimization failed: {e}")
            return self._optimize_equal_weight_scalable(assets)
    
    def _optimize_standard_scale(self, method: OptimizationMethod, mu: np.ndarray,
                                sigma: np.ndarray, assets: List[str],
                                current_positions: Dict[str, float]) -> Dict[str, float]:
        """Optimize portfolio for standard scale (<1000 assets)."""
        try:
            if method == OptimizationMethod.MEAN_VARIANCE:
                return self._optimize_mean_variance_scalable(mu, sigma, assets)
            elif method == OptimizationMethod.RISK_PARITY:
                return self._optimize_risk_parity_scalable(sigma, assets)
            elif method == OptimizationMethod.MINIMUM_VARIANCE:
                return self._optimize_minimum_variance_scalable(sigma, assets)
            elif method == OptimizationMethod.MAXIMUM_DIVERSIFICATION:
                return self._optimize_maximum_diversification_scalable(sigma, assets)
            elif method == OptimizationMethod.BLACK_LITTERMAN:
                return self._optimize_black_litterman(mu, sigma, assets)
            else:
                return self._optimize_equal_weight_scalable(assets)
                
        except Exception as e:
            logger.error(f"Standard-scale optimization failed: {e}")
            return self._optimize_equal_weight_scalable(assets)
    
    def _optimize_nested_clustered(self, mu: np.ndarray, sigma: np.ndarray, assets: List[str]) -> Dict[str, float]:
        """Nested Clustered Optimization for massive-scale portfolios."""
        try:
            n_assets = len(assets)
            
            # Step 1: Hierarchical clustering
            clusters = self._perform_hierarchical_clustering(sigma, assets)
            
            # Step 2: Optimize between clusters
            cluster_weights = self._optimize_cluster_allocation(mu, sigma, clusters, assets)
            
            # Step 3: Optimize within each cluster
            final_weights = np.zeros(n_assets)
            
            for cluster_id, cluster_assets in clusters.items():
                if len(cluster_assets) == 0:
                    continue
                
                cluster_indices = [assets.index(asset) for asset in cluster_assets]
                cluster_mu = mu[cluster_indices]
                cluster_sigma = sigma[np.ix_(cluster_indices, cluster_indices)]
                
                # Optimize within cluster
                if len(cluster_assets) > 1:
                    intra_weights = self._optimize_risk_parity_scalable(cluster_sigma, cluster_assets)
                    intra_weight_array = np.array([intra_weights[asset] for asset in cluster_assets])
                else:
                    intra_weight_array = np.array([1.0])
                
                # Scale by cluster weight
                cluster_weight = cluster_weights.get(cluster_id, 0.0)
                for i, asset_idx in enumerate(cluster_indices):
                    final_weights[asset_idx] = cluster_weight * intra_weight_array[i]
            
            # Normalize
            final_weights = final_weights / np.sum(final_weights)
            
            return {assets[i]: final_weights[i] for i in range(n_assets)}
            
        except Exception as e:
            logger.error(f"Nested clustered optimization failed: {e}")
            return self._optimize_equal_weight_scalable(assets)
    
    def _perform_hierarchical_clustering(self, sigma: np.ndarray, assets: List[str]) -> Dict[int, List[str]]:
        """Perform hierarchical clustering for massive-scale portfolios."""
        try:
            # Convert covariance to correlation for clustering
            std_devs = np.sqrt(np.diag(sigma))
            corr_matrix = sigma / np.outer(std_devs, std_devs)
            
            # Distance matrix for clustering
            distance_matrix = np.sqrt(0.5 * (1 - corr_matrix))
            
            # Hierarchical clustering
            condensed_distances = squareform(distance_matrix, checks=False)
            linkage_matrix = linkage(condensed_distances, method='ward')
            
            # Determine optimal number of clusters
            n_clusters = min(self.config.max_clusters, max(2, len(assets) // self.config.min_cluster_size))
            
            cluster_labels = fcluster(linkage_matrix, n_clusters, criterion='maxclust')
            
            # Group assets by cluster
            clusters = defaultdict(list)
            for i, asset in enumerate(assets):
                cluster_id = cluster_labels[i]
                clusters[cluster_id].append(asset)
                self.cluster_assignments[asset] = cluster_id
            
            return dict(clusters)
            
        except Exception as e:
            logger.error(f"Hierarchical clustering failed: {e}")
            # Return single cluster as fallback
            return {1: assets}
    
    def _optimize_cluster_allocation(self, mu: np.ndarray, sigma: np.ndarray,
                                   clusters: Dict[int, List[str]], assets: List[str]) -> Dict[int, float]:
        """Optimize allocation between clusters."""
        try:
            n_clusters = len(clusters)
            cluster_returns = np.zeros(n_clusters)
            cluster_variances = np.zeros(n_clusters)
            
            # Calculate cluster-level statistics
            for i, (cluster_id, cluster_assets) in enumerate(clusters.items()):
                cluster_indices = [assets.index(asset) for asset in cluster_assets]
                
                # Equal-weighted cluster return and variance
                cluster_mu = mu[cluster_indices]
                cluster_sigma = sigma[np.ix_(cluster_indices, cluster_indices)]
                
                equal_weights = np.ones(len(cluster_assets)) / len(cluster_assets)
                cluster_returns[i] = np.dot(equal_weights, cluster_mu)
                cluster_variances[i] = np.dot(equal_weights, np.dot(cluster_sigma, equal_weights))
            
            # Optimize cluster allocation using risk parity
            cluster_weights = self._risk_parity_optimization(cluster_variances)
            
            # Map back to cluster IDs
            cluster_allocation = {}
            for i, cluster_id in enumerate(clusters.keys()):
                cluster_allocation[cluster_id] = cluster_weights[i]
            
            return cluster_allocation
            
        except Exception as e:
            logger.error(f"Cluster allocation optimization failed: {e}")
            # Return equal allocation
            n_clusters = len(clusters)
            return {cluster_id: 1.0/n_clusters for cluster_id in clusters.keys()}
    
    def _risk_parity_optimization(self, variances: np.ndarray) -> np.ndarray:
        """Simple risk parity optimization for cluster allocation."""
        try:
            # Inverse variance weighting
            inv_vol = 1.0 / np.sqrt(variances)
            weights = inv_vol / np.sum(inv_vol)
            return weights
            
        except Exception as e:
            logger.warning(f"Risk parity optimization failed: {e}")
            return np.ones(len(variances)) / len(variances)
    
    def _optimize_hierarchical_risk_parity_scalable(self, mu: np.ndarray, sigma: np.ndarray, assets: List[str]) -> Dict[str, float]:
        """Scalable Hierarchical Risk Parity optimization."""
        try:
            if len(assets) < 10:
                return self._optimize_risk_parity_scalable(sigma, assets)
            
            # Perform clustering
            clusters = self._perform_hierarchical_clustering(sigma, assets)
            
            # Allocate weights hierarchically
            weights = np.zeros(len(assets))
            
            # Step 1: Allocate between clusters using risk parity
            cluster_weights = self._allocate_cluster_weights_scalable(sigma, clusters, assets)
            
            # Step 2: Allocate within clusters using risk parity
            for cluster_id, cluster_assets in clusters.items():
                if len(cluster_assets) == 0:
                    continue
                
                cluster_indices = [assets.index(asset) for asset in cluster_assets]
                cluster_sigma = sigma[np.ix_(cluster_indices, cluster_indices)]
                
                # Risk parity within cluster
                if len(cluster_assets) > 1:
                    intra_weights = self._optimize_risk_parity_scalable(cluster_sigma, cluster_assets)
                    intra_weight_array = np.array([intra_weights[asset] for asset in cluster_assets])
                else:
                    intra_weight_array = np.array([1.0])
                
                # Scale by cluster weight
                cluster_weight = cluster_weights.get(cluster_id, 0.0)
                for i, asset_idx in enumerate(cluster_indices):
                    weights[asset_idx] = cluster_weight * intra_weight_array[i]
            
            # Normalize
            weights = weights / np.sum(weights)
            
            return {assets[i]: weights[i] for i in range(len(assets))}
            
        except Exception as e:
            logger.error(f"Hierarchical risk parity failed: {e}")
            return self._optimize_equal_weight_scalable(assets)
    
    def _allocate_cluster_weights_scalable(self, sigma: np.ndarray, clusters: Dict[int, List[str]], assets: List[str]) -> Dict[int, float]:
        """Allocate weights between clusters using advanced risk measures."""
        try:
            cluster_weights = {}
            cluster_risks = {}
            
            # Calculate cluster risk contributions
            for cluster_id, cluster_assets in clusters.items():
                if len(cluster_assets) == 0:
                    cluster_risks[cluster_id] = 1.0
                    continue
                
                cluster_indices = [assets.index(asset) for asset in cluster_assets]
                cluster_sigma = sigma[np.ix_(cluster_indices, cluster_indices)]
                
                # Calculate cluster risk (average variance)
                cluster_risk = np.mean(np.diag(cluster_sigma))
                cluster_risks[cluster_id] = cluster_risk
            
            # Inverse risk weighting
            total_inv_risk = sum(1.0 / risk for risk in cluster_risks.values())
            
            for cluster_id, risk in cluster_risks.items():
                cluster_weights[cluster_id] = (1.0 / risk) / total_inv_risk
            
            return cluster_weights
            
        except Exception as e:
            logger.error(f"Cluster weight allocation failed: {e}")
            n_clusters = len(clusters)
            return {cluster_id: 1.0/n_clusters for cluster_id in clusters.keys()}
    
    def _optimize_batch_processing(self, method: OptimizationMethod, mu: np.ndarray,
                                 sigma: np.ndarray, assets: List[str]) -> Dict[str, float]:
        """Optimize portfolio using batch processing for massive-scale."""
        try:
            batch_size = self.config.optimization_batch_size
            n_assets = len(assets)
            
            if n_assets <= batch_size:
                return self._optimize_standard_scale(method, mu, sigma, assets, {})
            
            # Split into batches
            batches = []
            for i in range(0, n_assets, batch_size):
                end_idx = min(i + batch_size, n_assets)
                batch_assets = assets[i:end_idx]
                batch_indices = list(range(i, end_idx))
                batches.append((batch_assets, batch_indices))
            
            # Optimize each batch
            batch_results = []
            for batch_assets, batch_indices in batches:
                batch_mu = mu[batch_indices]
                batch_sigma = sigma[np.ix_(batch_indices, batch_indices)]
                
                batch_weights = self._optimize_standard_scale(method, batch_mu, batch_sigma, batch_assets, {})
                batch_results.append(batch_weights)
            
            # Combine batch results
            final_weights = {}
            total_batches = len(batches)
            
            for i, (batch_assets, _) in enumerate(batches):
                batch_weight_scale = 1.0 / total_batches  # Equal batch weighting
                
                for asset, weight in batch_results[i].items():
                    final_weights[asset] = weight * batch_weight_scale
            
            return final_weights
            
        except Exception as e:
            logger.error(f"Batch processing optimization failed: {e}")
            return self._optimize_equal_weight_scalable(assets)
    
    def _optimize_mean_variance_scalable(self, mu: np.ndarray, sigma: np.ndarray, assets: List[str]) -> Dict[str, float]:
        """Scalable mean-variance optimization with GPU acceleration."""
        try:
            n = len(assets)
            
            # Use GPU acceleration for large portfolios
            if self.gpu_available and n > 500:
                return self._optimize_mean_variance_gpu(mu, sigma, assets)
            
            # CPU optimization with ARM64 enhancements
            def objective(weights):
                portfolio_return = np.dot(weights, mu)
                portfolio_variance = np.dot(weights, np.dot(sigma, weights))
                return -(portfolio_return - 0.5 * self.config.risk_aversion * portfolio_variance)
            
            # Enhanced constraints for massive-scale
            constraints = [
                {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0},  # Weights sum to 1
            ]
            
            # Tighter bounds for massive diversification
            bounds = [(0, self.config.max_position_concentration) for _ in range(n)]
            
            # Initial guess with better starting point
            x0 = np.ones(n) / n
            
            # Use differential evolution for ARM64 or large portfolios
            if self.is_arm64 or n > 200:
                result = differential_evolution(
                    objective, bounds, seed=42, maxiter=self.config.max_iterations//20,
                    workers=min(4, mp.cpu_count())
                )
                weights = result.x
            else:
                result = minimize(
                    objective, x0, method='SLSQP', bounds=bounds,
                    constraints=constraints, options={'maxiter': self.config.max_iterations}
                )
                weights = result.x
            
            # Apply constraints and normalize
            weights = np.maximum(weights, 0)
            weights = np.minimum(weights, self.config.max_position_concentration)
            weights = weights / np.sum(weights)
            
            return {assets[i]: weights[i] for i in range(n)}
            
        except Exception as e:
            logger.error(f"Scalable mean-variance optimization failed: {e}")
            return self._optimize_equal_weight_scalable(assets)
    
    def _optimize_mean_variance_gpu(self, mu: np.ndarray, sigma: np.ndarray, assets: List[str]) -> Dict[str, float]:
        """GPU-accelerated mean-variance optimization."""
        try:
            # Transfer to GPU
            mu_gpu = cp.asarray(mu)
            sigma_gpu = cp.asarray(sigma)
            n = len(assets)
            
            # GPU-optimized objective function
            def objective_gpu(weights_cpu):
                weights_gpu = cp.asarray(weights_cpu)
                portfolio_return = cp.dot(weights_gpu, mu_gpu)
                portfolio_variance = cp.dot(weights_gpu, cp.dot(sigma_gpu, weights_gpu))
                utility = portfolio_return - 0.5 * self.config.risk_aversion * portfolio_variance
                return -float(cp.asnumpy(utility))
            
            # Use differential evolution with GPU objective
            bounds = [(0, self.config.max_position_concentration) for _ in range(n)]
            
            result = differential_evolution(
                objective_gpu, bounds, seed=42, maxiter=self.config.max_iterations//10,
                workers=1  # GPU doesn't benefit from multiple workers
            )
            
            weights = result.x
            weights = weights / np.sum(weights)
            
            self.calculation_stats["gpu_optimizations"] += 1
            
            return {assets[i]: weights[i] for i in range(n)}
            
        except Exception as e:
            logger.warning(f"GPU mean-variance optimization failed, falling back to CPU: {e}")
            return self._optimize_mean_variance_scalable(mu, sigma, assets)
    
    def _optimize_risk_parity_scalable(self, sigma: np.ndarray, assets: List[str]) -> Dict[str, float]:
        """Scalable risk parity optimization with enhanced convergence."""
        try:
            n = len(assets)
            
            def risk_budget_objective(weights):
                weights = np.maximum(weights, 1e-8)  # Avoid division by zero
                portfolio_vol = np.sqrt(np.dot(weights, np.dot(sigma, weights)))
                
                if portfolio_vol < 1e-10:
                    return 1e10
                
                marginal_contrib = np.dot(sigma, weights) / portfolio_vol
                contrib = weights * marginal_contrib
                target_contrib = np.ones(n) / n  # Equal risk contribution
                
                # Use relative error for better scaling
                relative_error = (contrib - target_contrib * np.sum(contrib)) / (target_contrib * np.sum(contrib) + 1e-8)
                return np.sum(relative_error**2)
            
            # Enhanced constraints and bounds
            constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}]
            bounds = [(1e-6, self.config.max_position_concentration) for _ in range(n)]
            
            # Better initial guess
            x0 = np.ones(n) / n
            
            # Use multiple optimization attempts for robustness
            best_result = None
            best_objective = float('inf')
            
            for attempt in range(3):
                try:
                    if attempt == 0:
                        # Standard SLSQP
                        result = minimize(
                            risk_budget_objective, x0, method='SLSQP',
                            bounds=bounds, constraints=constraints,
                            options={'maxiter': self.config.max_iterations, 'ftol': self.config.convergence_tolerance}
                        )
                    elif attempt == 1:
                        # Trust-constr for better handling of constraints
                        result = minimize(
                            risk_budget_objective, x0, method='trust-constr',
                            bounds=bounds, constraints=constraints,
                            options={'maxiter': self.config.max_iterations}
                        )
                    else:
                        # Differential evolution as fallback
                        result = differential_evolution(
                            risk_budget_objective, bounds, seed=42,
                            maxiter=self.config.max_iterations//10
                        )
                    
                    if result.fun < best_objective:
                        best_result = result
                        best_objective = result.fun
                        
                except Exception as e:
                    logger.warning(f"Risk parity optimization attempt {attempt} failed: {e}")
                    continue
            
            if best_result is None:
                raise OptimizationError("All risk parity optimization attempts failed")
            
            weights = best_result.x
            weights = np.maximum(weights, 0)
            weights = weights / np.sum(weights)
            
            return {assets[i]: weights[i] for i in range(n)}
            
        except Exception as e:
            logger.error(f"Scalable risk parity optimization failed: {e}")
            return self._optimize_equal_weight_scalable(assets)
    
    def _optimize_minimum_variance_scalable(self, sigma: np.ndarray, assets: List[str]) -> Dict[str, float]:
        """Scalable minimum variance optimization."""
        try:
            n = len(assets)
            
            # For large portfolios, use analytical solution when possible
            if n > 1000:
                return self._optimize_minimum_variance_analytical(sigma, assets)
            
            def objective(weights):
                return np.dot(weights, np.dot(sigma, weights))
            
            constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}]
            bounds = [(0, self.config.max_position_concentration) for _ in range(n)]
            
            x0 = np.ones(n) / n
            
            result = minimize(
                objective, x0, method='SLSQP', bounds=bounds, constraints=constraints,
                options={'maxiter': self.config.max_iterations}
            )
            
            weights = result.x
            weights = weights / np.sum(weights)
            
            return {assets[i]: weights[i] for i in range(n)}
            
        except Exception as e:
            logger.error(f"Scalable minimum variance optimization failed: {e}")
            return self._optimize_equal_weight_scalable(assets)
    
    def _optimize_minimum_variance_analytical(self, sigma: np.ndarray, assets: List[str]) -> Dict[str, float]:
        """Analytical minimum variance optimization for massive portfolios."""
        try:
            # Analytical solution: w = (Σ^-1 * 1) / (1^T * Σ^-1 * 1)
            ones = np.ones(sigma.shape[0])
            
            # Use pseudo-inverse for numerical stability
            sigma_inv = pinv(sigma)
            
            weights = sigma_inv @ ones
            weights = weights / np.sum(weights)
            
            # Apply concentration constraints
            weights = np.minimum(weights, self.config.max_position_concentration)
            weights = weights / np.sum(weights)
            
            return {assets[i]: weights[i] for i in range(len(assets))}
            
        except Exception as e:
            logger.error(f"Analytical minimum variance optimization failed: {e}")
            return self._optimize_equal_weight_scalable(assets)
    
    def _optimize_maximum_diversification_scalable(self, sigma: np.ndarray, assets: List[str]) -> Dict[str, float]:
        """Scalable maximum diversification optimization."""
        try:
            n = len(assets)
            std_devs = np.sqrt(np.diag(sigma))
            
            def objective(weights):
                weighted_avg_vol = np.dot(weights, std_devs)
                portfolio_vol = np.sqrt(np.dot(weights, np.dot(sigma, weights)))
                
                if portfolio_vol < 1e-10:
                    return -1e10
                
                return -weighted_avg_vol / portfolio_vol  # Maximize diversification ratio
            
            constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}]
            bounds = [(0, self.config.max_position_concentration) for _ in range(n)]
            
            x0 = np.ones(n) / n
            
            result = minimize(
                objective, x0, method='SLSQP', bounds=bounds, constraints=constraints,
                options={'maxiter': self.config.max_iterations}
            )
            
            weights = result.x
            weights = weights / np.sum(weights)
            
            return {assets[i]: weights[i] for i in range(n)}
            
        except Exception as e:
            logger.error(f"Scalable maximum diversification optimization failed: {e}")
            return self._optimize_equal_weight_scalable(assets)
    
    def _optimize_black_litterman(self, mu: np.ndarray, sigma: np.ndarray, assets: List[str]) -> Dict[str, float]:
        """Black-Litterman optimization with scalable implementation."""
        try:
            # Simplified Black-Litterman implementation
            # Market capitalization weights as prior
            market_caps = np.array([self.market_caps.get(asset, 1.0) for asset in assets])
            market_weights = market_caps / np.sum(market_caps)
            
            # Implied returns from market weights
            risk_aversion = self.config.risk_aversion
            implied_returns = risk_aversion * sigma @ market_weights
            
            # Combine with views (simplified - use expected returns as views)
            tau = 0.025  # Scaling factor
            omega = np.eye(len(assets)) * 0.01  # View uncertainty
            
            # Black-Litterman formula
            sigma_inv = pinv(sigma)
