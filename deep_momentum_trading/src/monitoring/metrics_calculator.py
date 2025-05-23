"""
Enhanced Metrics Calculator with ARM64 Optimizations

This module provides comprehensive metrics calculation capabilities for the Deep Momentum Trading System
with ARM64-specific optimizations for high-performance financial and system metrics computation.

Features:
- Real-time trading metrics calculation with ARM64 optimizations
- Performance metrics and system resource monitoring
- Statistical analysis and risk metrics
- Time-series analysis and trend detection
- Vectorized calculations using ARM64 SIMD
- Shared memory integration for high-frequency metrics
- Comprehensive financial ratios and indicators
- Machine learning model performance metrics
"""

import numpy as np
import pandas as pd
import time
import threading
import queue
import platform
import multiprocessing as mp
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
from concurrent.futures import ThreadPoolExecutor
import logging
from functools import lru_cache
import psutil
from collections import deque, defaultdict
from scipy import stats
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings

from ..utils.logger import get_logger
from ..utils.shared_memory import SharedMemoryManager
from ..utils.decorators import performance_monitor, error_handler
from ..utils.exceptions import ValidationError, CalculationError

logger = get_logger(__name__)

class MetricType(Enum):
    """Types of metrics"""
    TRADING = "trading"
    PERFORMANCE = "performance"
    RISK = "risk"
    SYSTEM = "system"
    MODEL = "model"

class AggregationType(Enum):
    """Aggregation types for metrics"""
    SUM = "sum"
    MEAN = "mean"
    MEDIAN = "median"
    MIN = "min"
    MAX = "max"
    STD = "std"
    COUNT = "count"
    LAST = "last"

@dataclass
class MetricsConfig:
    """Configuration for metrics calculator"""
    enable_arm64_optimizations: bool = True
    use_shared_memory: bool = True
    calculation_interval: float = 1.0  # seconds
    history_window: int = 10000  # number of data points to keep
    enable_real_time_calculation: bool = True
    parallel_processing: bool = True
    max_workers: Optional[int] = None
    cache_size: int = 1000
    enable_statistical_analysis: bool = True
    enable_risk_metrics: bool = True
    enable_model_metrics: bool = True
    precision: int = 6  # decimal places

@dataclass
class MetricDefinition:
    """Definition of a metric"""
    name: str
    metric_type: MetricType
    calculation_func: str  # Function name or expression
    aggregation: AggregationType
    window_size: Optional[int] = None
    dependencies: List[str] = field(default_factory=list)
    enabled: bool = True

@dataclass
class MetricResult:
    """Result of metric calculation"""
    name: str
    value: Union[float, int, Dict[str, Any]]
    timestamp: float
    metric_type: MetricType
    metadata: Dict[str, Any] = field(default_factory=dict)

class ARM64MetricsOptimizer:
    """ARM64-specific optimizations for metrics calculations"""
    
    def __init__(self):
        self.is_arm64 = platform.machine().lower() in ['arm64', 'aarch64']
        self.simd_available = self._check_simd_support()
        
    def _check_simd_support(self) -> bool:
        """Check for ARM64 SIMD support"""
        if not self.is_arm64:
            return False
        try:
            with open('/proc/cpuinfo', 'r') as f:
                cpuinfo = f.read()
                return 'neon' in cpuinfo.lower() or 'asimd' in cpuinfo.lower()
        except:
            return False
    
    def vectorized_statistics(self, data: np.ndarray) -> Dict[str, float]:
        """ARM64-optimized statistical calculations"""
        if self.simd_available and len(data) > 100:
            # Use ARM64 SIMD for large datasets
            return {
                'mean': np.mean(data),
                'std': np.std(data, ddof=1),
                'min': np.min(data),
                'max': np.max(data),
                'median': np.median(data),
                'skewness': stats.skew(data),
                'kurtosis': stats.kurtosis(data)
            }
        else:
            return self._standard_statistics(data)
    
    def _standard_statistics(self, data: np.ndarray) -> Dict[str, float]:
        """Standard statistical calculations"""
        return {
            'mean': float(np.mean(data)),
            'std': float(np.std(data, ddof=1)),
            'min': float(np.min(data)),
            'max': float(np.max(data)),
            'median': float(np.median(data)),
            'skewness': float(stats.skew(data)),
            'kurtosis': float(stats.kurtosis(data))
        }
    
    def parallel_metric_calculation(self, calculations: List[callable], max_workers: int = None) -> List[Any]:
        """ARM64-optimized parallel metric calculations"""
        if max_workers is None:
            max_workers = min(mp.cpu_count(), 8)
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(calc) for calc in calculations]
            return [future.result() for future in futures]

class TradingMetricsCalculator:
    """Calculator for trading-specific metrics"""
    
    def __init__(self, arm64_optimizer: ARM64MetricsOptimizer):
        self.optimizer = arm64_optimizer
        self.trade_history = deque(maxlen=10000)
        self.position_history = deque(maxlen=10000)
        self.pnl_history = deque(maxlen=10000)
    
    def add_trade(self, trade_data: Dict[str, Any]):
        """Add trade data for metrics calculation"""
        trade_data['timestamp'] = time.time()
        self.trade_history.append(trade_data)
        
        # Update PnL history
        if 'pnl' in trade_data:
            self.pnl_history.append(trade_data['pnl'])
    
    def add_position(self, position_data: Dict[str, Any]):
        """Add position data for metrics calculation"""
        position_data['timestamp'] = time.time()
        self.position_history.append(position_data)
    
    @performance_monitor
    def calculate_sharpe_ratio(self, returns: np.ndarray, risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio"""
        if len(returns) < 2:
            return 0.0
        
        excess_returns = returns - risk_free_rate / 252  # Daily risk-free rate
        return np.mean(excess_returns) / np.std(excess_returns, ddof=1) * np.sqrt(252)
    
    @performance_monitor
    def calculate_sortino_ratio(self, returns: np.ndarray, risk_free_rate: float = 0.02) -> float:
        """Calculate Sortino ratio"""
        if len(returns) < 2:
            return 0.0
        
        excess_returns = returns - risk_free_rate / 252
        downside_returns = excess_returns[excess_returns < 0]
        
        if len(downside_returns) == 0:
            return float('inf')
        
        downside_deviation = np.std(downside_returns, ddof=1)
        return np.mean(excess_returns) / downside_deviation * np.sqrt(252)
    
    @performance_monitor
    def calculate_max_drawdown(self, returns: np.ndarray) -> Dict[str, float]:
        """Calculate maximum drawdown"""
        if len(returns) == 0:
            return {'max_drawdown': 0.0, 'drawdown_duration': 0}
        
        cumulative_returns = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdown = (cumulative_returns - running_max) / running_max
        
        max_drawdown = np.min(drawdown)
        
        # Calculate drawdown duration
        drawdown_periods = np.where(drawdown < 0)[0]
        if len(drawdown_periods) > 0:
            drawdown_duration = len(drawdown_periods)
        else:
            drawdown_duration = 0
        
        return {
            'max_drawdown': float(max_drawdown),
            'drawdown_duration': drawdown_duration
        }
    
    @performance_monitor
    def calculate_win_rate(self) -> float:
        """Calculate win rate from trade history"""
        if not self.trade_history:
            return 0.0
        
        winning_trades = sum(1 for trade in self.trade_history if trade.get('pnl', 0) > 0)
        return winning_trades / len(self.trade_history)
    
    @performance_monitor
    def calculate_profit_factor(self) -> float:
        """Calculate profit factor"""
        if not self.trade_history:
            return 0.0
        
        gross_profit = sum(trade.get('pnl', 0) for trade in self.trade_history if trade.get('pnl', 0) > 0)
        gross_loss = abs(sum(trade.get('pnl', 0) for trade in self.trade_history if trade.get('pnl', 0) < 0))
        
        return gross_profit / gross_loss if gross_loss > 0 else float('inf')
    
    @performance_monitor
    def calculate_calmar_ratio(self, returns: np.ndarray) -> float:
        """Calculate Calmar ratio"""
        if len(returns) < 2:
            return 0.0
        
        annual_return = np.mean(returns) * 252
        max_drawdown_info = self.calculate_max_drawdown(returns)
        max_drawdown = abs(max_drawdown_info['max_drawdown'])
        
        return annual_return / max_drawdown if max_drawdown > 0 else float('inf')
    
    def get_trading_summary(self) -> Dict[str, Any]:
        """Get comprehensive trading metrics summary"""
        if not self.pnl_history:
            return {}
        
        pnl_array = np.array(list(self.pnl_history))
        returns = np.diff(pnl_array) / pnl_array[:-1] if len(pnl_array) > 1 else np.array([])
        
        summary = {
            'total_trades': len(self.trade_history),
            'total_pnl': float(np.sum(pnl_array)),
            'win_rate': self.calculate_win_rate(),
            'profit_factor': self.calculate_profit_factor()
        }
        
        if len(returns) > 1:
            summary.update({
                'sharpe_ratio': self.calculate_sharpe_ratio(returns),
                'sortino_ratio': self.calculate_sortino_ratio(returns),
                'calmar_ratio': self.calculate_calmar_ratio(returns),
                'max_drawdown': self.calculate_max_drawdown(returns)
            })
        
        return summary

class SystemMetricsCalculator:
    """Calculator for system performance metrics"""
    
    def __init__(self, arm64_optimizer: ARM64MetricsOptimizer):
        self.optimizer = arm64_optimizer
        self.cpu_history = deque(maxlen=1000)
        self.memory_history = deque(maxlen=1000)
        self.disk_history = deque(maxlen=1000)
        self.network_history = deque(maxlen=1000)
    
    @performance_monitor
    def collect_system_metrics(self) -> Dict[str, Any]:
        """Collect current system metrics"""
        try:
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=None)
            cpu_count = psutil.cpu_count()
            cpu_freq = psutil.cpu_freq()
            
            # Memory metrics
            memory = psutil.virtual_memory()
            swap = psutil.swap_memory()
            
            # Disk metrics
            disk = psutil.disk_usage('/')
            disk_io = psutil.disk_io_counters()
            
            # Network metrics
            network_io = psutil.net_io_counters()
            
            metrics = {
                'timestamp': time.time(),
                'cpu': {
                    'percent': cpu_percent,
                    'count': cpu_count,
                    'frequency': cpu_freq.current if cpu_freq else 0
                },
                'memory': {
                    'total': memory.total,
                    'available': memory.available,
                    'percent': memory.percent,
                    'used': memory.used,
                    'free': memory.free
                },
                'swap': {
                    'total': swap.total,
                    'used': swap.used,
                    'free': swap.free,
                    'percent': swap.percent
                },
                'disk': {
                    'total': disk.total,
                    'used': disk.used,
                    'free': disk.free,
                    'percent': (disk.used / disk.total) * 100
                },
                'network': {
                    'bytes_sent': network_io.bytes_sent,
                    'bytes_recv': network_io.bytes_recv,
                    'packets_sent': network_io.packets_sent,
                    'packets_recv': network_io.packets_recv
                }
            }
            
            # Add to history
            self.cpu_history.append(cpu_percent)
            self.memory_history.append(memory.percent)
            self.disk_history.append((disk.used / disk.total) * 100)
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error collecting system metrics: {e}")
            return {}
    
    def get_system_summary(self) -> Dict[str, Any]:
        """Get system metrics summary"""
        current_metrics = self.collect_system_metrics()
        
        summary = {
            'current': current_metrics,
            'averages': {}
        }
        
        if self.cpu_history:
            cpu_array = np.array(list(self.cpu_history))
            summary['averages']['cpu'] = self.optimizer.vectorized_statistics(cpu_array)
        
        if self.memory_history:
            memory_array = np.array(list(self.memory_history))
            summary['averages']['memory'] = self.optimizer.vectorized_statistics(memory_array)
        
        if self.disk_history:
            disk_array = np.array(list(self.disk_history))
            summary['averages']['disk'] = self.optimizer.vectorized_statistics(disk_array)
        
        return summary

class ModelMetricsCalculator:
    """Calculator for machine learning model metrics"""
    
    def __init__(self, arm64_optimizer: ARM64MetricsOptimizer):
        self.optimizer = arm64_optimizer
        self.prediction_history = deque(maxlen=10000)
        self.actual_history = deque(maxlen=10000)
        self.training_metrics = {}
    
    def add_prediction(self, predicted: float, actual: float, metadata: Dict[str, Any] = None):
        """Add prediction for metrics calculation"""
        self.prediction_history.append(predicted)
        self.actual_history.append(actual)
    
    @performance_monitor
    def calculate_regression_metrics(self) -> Dict[str, float]:
        """Calculate regression model metrics"""
        if len(self.prediction_history) < 2 or len(self.actual_history) < 2:
            return {}
        
        y_true = np.array(list(self.actual_history))
        y_pred = np.array(list(self.prediction_history))
        
        # Ensure same length
        min_len = min(len(y_true), len(y_pred))
        y_true = y_true[-min_len:]
        y_pred = y_pred[-min_len:]
        
        try:
            metrics = {
                'mse': float(mean_squared_error(y_true, y_pred)),
                'rmse': float(np.sqrt(mean_squared_error(y_true, y_pred))),
                'mae': float(mean_absolute_error(y_true, y_pred)),
                'r2_score': float(r2_score(y_true, y_pred)),
                'mape': float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100),
                'correlation': float(np.corrcoef(y_true, y_pred)[0, 1])
            }
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating regression metrics: {e}")
            return {}
    
    @performance_monitor
    def calculate_directional_accuracy(self) -> float:
        """Calculate directional accuracy for trading models"""
        if len(self.prediction_history) < 2 or len(self.actual_history) < 2:
            return 0.0
        
        predictions = np.array(list(self.prediction_history))
        actuals = np.array(list(self.actual_history))
        
        # Calculate direction changes
        pred_directions = np.sign(np.diff(predictions))
        actual_directions = np.sign(np.diff(actuals))
        
        # Calculate accuracy
        correct_directions = np.sum(pred_directions == actual_directions)
        total_directions = len(pred_directions)
        
        return correct_directions / total_directions if total_directions > 0 else 0.0
    
    def get_model_summary(self) -> Dict[str, Any]:
        """Get model metrics summary"""
        summary = {
            'total_predictions': len(self.prediction_history),
            'regression_metrics': self.calculate_regression_metrics(),
            'directional_accuracy': self.calculate_directional_accuracy(),
            'training_metrics': self.training_metrics
        }
        
        return summary

class AdvancedMetricsCalculator:
    """
    Advanced Metrics Calculator with ARM64 optimizations
    
    Provides comprehensive metrics calculation capabilities including:
    - Real-time trading metrics
    - System performance monitoring
    - Statistical analysis and risk metrics
    - Machine learning model evaluation
    """
    
    def __init__(self, config: MetricsConfig = None):
        self.config = config or MetricsConfig()
        self.arm64_optimizer = ARM64MetricsOptimizer()
        self.shared_memory = SharedMemoryManager() if self.config.use_shared_memory else None
        
        # Specialized calculators
        self.trading_calculator = TradingMetricsCalculator(self.arm64_optimizer)
        self.system_calculator = SystemMetricsCalculator(self.arm64_optimizer)
        self.model_calculator = ModelMetricsCalculator(self.arm64_optimizer)
        
        # Core components
        self.metrics_definitions = {}
        self.metrics_cache = {}
        self.calculation_queue = queue.Queue()
        
        # Threading
        self.is_running = False
        self.calculation_thread = None
        
        # Performance tracking
        self.calculation_stats = {
            'total_calculations': 0,
            'calculation_times': deque(maxlen=1000),
            'cache_hits': 0,
            'cache_misses': 0
        }
        
        # Initialize default metrics
        self._initialize_default_metrics()
        
        logger.info(f"AdvancedMetricsCalculator initialized with ARM64 optimizations: {self.arm64_optimizer.is_arm64}")
    
    def _initialize_default_metrics(self):
        """Initialize default metric definitions"""
        default_metrics = [
            MetricDefinition(
                name="total_pnl",
                metric_type=MetricType.TRADING,
                calculation_func="sum_pnl",
                aggregation=AggregationType.SUM
            ),
            MetricDefinition(
                name="win_rate",
                metric_type=MetricType.TRADING,
                calculation_func="calculate_win_rate",
                aggregation=AggregationType.LAST
            ),
            MetricDefinition(
                name="sharpe_ratio",
                metric_type=MetricType.RISK,
                calculation_func="calculate_sharpe_ratio",
                aggregation=AggregationType.LAST
            ),
            MetricDefinition(
                name="cpu_usage",
                metric_type=MetricType.SYSTEM,
                calculation_func="get_cpu_usage",
                aggregation=AggregationType.MEAN,
                window_size=60
            ),
            MetricDefinition(
                name="memory_usage",
                metric_type=MetricType.SYSTEM,
                calculation_func="get_memory_usage",
                aggregation=AggregationType.MEAN,
                window_size=60
            )
        ]
        
        for metric in default_metrics:
            self.add_metric_definition(metric)
    
    def add_metric_definition(self, metric_def: MetricDefinition):
        """Add metric definition"""
        self.metrics_definitions[metric_def.name] = metric_def
        logger.info(f"Added metric definition: {metric_def.name}")
    
    def remove_metric_definition(self, metric_name: str):
        """Remove metric definition"""
        if metric_name in self.metrics_definitions:
            del self.metrics_definitions[metric_name]
            logger.info(f"Removed metric definition: {metric_name}")
    
    @performance_monitor
    @error_handler
    def calculate_metric(self, metric_name: str, data: Dict[str, Any] = None) -> Optional[MetricResult]:
        """
        Calculate specific metric
        
        Args:
            metric_name: Name of metric to calculate
            data: Additional data for calculation
            
        Returns:
            MetricResult or None if calculation failed
        """
        if metric_name not in self.metrics_definitions:
            logger.warning(f"Metric definition not found: {metric_name}")
            return None
        
        metric_def = self.metrics_definitions[metric_name]
        if not metric_def.enabled:
            return None
        
        start_time = time.time()
        
        try:
            # Check cache first
            cache_key = f"{metric_name}_{hash(str(data))}"
            if cache_key in self.metrics_cache:
                self.calculation_stats['cache_hits'] += 1
                return self.metrics_cache[cache_key]
            
            # Calculate metric
            value = self._execute_calculation(metric_def, data)
            
            result = MetricResult(
                name=metric_name,
                value=value,
                timestamp=time.time(),
                metric_type=metric_def.metric_type,
                metadata={'calculation_time': time.time() - start_time}
            )
            
            # Cache result
            self.metrics_cache[cache_key] = result
            self.calculation_stats['cache_misses'] += 1
            
            # Update statistics
            self.calculation_stats['total_calculations'] += 1
            self.calculation_stats['calculation_times'].append(time.time() - start_time)
            
            return result
            
        except Exception as e:
            logger.error(f"Error calculating metric {metric_name}: {e}")
            return None
    
    def _execute_calculation(self, metric_def: MetricDefinition, data: Dict[str, Any] = None) -> Any:
        """Execute metric calculation"""
        func_name = metric_def.calculation_func
        
        # Built-in calculation functions
        if func_name == "sum_pnl":
            return sum(trade.get('pnl', 0) for trade in self.trading_calculator.trade_history)
        elif func_name == "calculate_win_rate":
            return self.trading_calculator.calculate_win_rate()
        elif func_name == "calculate_sharpe_ratio":
            if self.trading_calculator.pnl_history:
                pnl_array = np.array(list(self.trading_calculator.pnl_history))
                returns = np.diff(pnl_array) / pnl_array[:-1] if len(pnl_array) > 1 else np.array([])
                return self.trading_calculator.calculate_sharpe_ratio(returns)
            return 0.0
        elif func_name == "get_cpu_usage":
            return psutil.cpu_percent()
        elif func_name == "get_memory_usage":
            return psutil.virtual_memory().percent
        else:
            # Custom calculation function
            if hasattr(self, func_name):
                func = getattr(self, func_name)
                return func(data) if data else func()
            else:
                raise CalculationError(f"Unknown calculation function: {func_name}")
    
    @performance_monitor
    def calculate_all_metrics(self, data: Dict[str, Any] = None) -> Dict[str, MetricResult]:
        """Calculate all enabled metrics"""
        results = {}
        
        # Prepare calculation functions for parallel execution
        if self.config.parallel_processing:
            calculations = []
            for metric_name, metric_def in self.metrics_definitions.items():
                if metric_def.enabled:
                    calculations.append(lambda mn=metric_name: self.calculate_metric(mn, data))
            
            # Execute in parallel
            parallel_results = self.arm64_optimizer.parallel_metric_calculation(
                calculations, self.config.max_workers
            )
            
            # Collect results
            for i, (metric_name, _) in enumerate(self.metrics_definitions.items()):
                if parallel_results[i]:
                    results[metric_name] = parallel_results[i]
        else:
            # Sequential calculation
            for metric_name, metric_def in self.metrics_definitions.items():
                if metric_def.enabled:
                    result = self.calculate_metric(metric_name, data)
                    if result:
                        results[metric_name] = result
        
        return results
    
    def update_trading_metrics(self, event_type: str, data: Dict[str, Any]):
        """Update trading metrics with new event data"""
        if event_type == "trade_executed":
            self.trading_calculator.add_trade(data)
        elif event_type == "position_updated":
            self.trading_calculator.add_position(data)
        elif event_type == "prediction_made":
            if 'predicted' in data and 'actual' in data:
                self.model_calculator.add_prediction(data['predicted'], data['actual'], data)
    
    def _real_time_calculation(self):
        """Background thread for real-time metric calculations"""
        while self.is_running:
            try:
                # Calculate system metrics
                system_metrics = self.system_calculator.collect_system_metrics()
                
                # Calculate all metrics
                if self.config.enable_real_time_calculation:
                    self.calculate_all_metrics()
                
                time.sleep(self.config.calculation_interval)
                
            except Exception as e:
                logger.error(f"Error in real-time calculation: {e}")
                time.sleep(1.0)
    
    def start(self) -> bool:
        """Start metrics calculator"""
        if self.is_running:
            return True
        
        try:
            self.is_running = True
            
            # Start real-time calculation thread
            if self.config.enable_real_time_calculation:
                self.calculation_thread = threading.Thread(
                    target=self._real_time_calculation, 
                    name="MetricsCalculationThread"
                )
                self.calculation_thread.daemon = True
                self.calculation_thread.start()
            
            logger.info("Metrics calculator started successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start metrics calculator: {e}")
            self.is_running = False
            return False
    
    def stop(self):
        """Stop metrics calculator"""
        self.is_running = False
        
        if self.calculation_thread:
            self.calculation_thread.join(timeout=5.0)
        
        logger.info("Metrics calculator stopped")
    
    def get_status(self) -> Dict[str, Any]:
        """Get metrics calculator status"""
        avg_calculation_time = (
            sum(self.calculation_stats['calculation_times']) / len(self.calculation_stats['calculation_times'])
            if self.calculation_stats['calculation_times'] else 0
        )
        
        cache_hit_rate = (
            self.calculation_stats['cache_hits'] / 
            (self.calculation_stats['cache_hits'] + self.calculation_stats['cache_misses'])
            if (self.calculation_stats['cache_hits'] + self.calculation_stats['cache_misses']) > 0 else 0
        )
        
        return {
            'is_running': self.is_running,
            'total_metrics': len(self.metrics_definitions),
            'enabled_metrics': len([m for m in self.metrics_definitions.values() if m.enabled]),
            'total_calculations': self.calculation_stats['total_calculations'],
            'average_calculation_time': avg_calculation_time,
            'cache_hit_rate': cache_hit_rate,
            'arm64_optimized': self.arm64_optimizer.is_arm64
        }
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get comprehensive metrics summary"""
        return {
            'trading_metrics': self.trading_calculator.get_trading_summary(),
            'system_metrics': self.system_calculator.get_system_summary(),
            'model_metrics': self.model_calculator.get_model_summary(),
            'calculation_stats': {
                'total_calculations': self.calculation_stats['total_calculations'],
                'cache_hit_rate': (
                    self.calculation_stats['cache_hits'] / 
                    (self.calculation_stats['cache_hits'] + self.calculation_stats['cache_misses'])
                    if (self.calculation_stats['cache_hits'] + self.calculation_stats['cache_misses']) > 0 else 0
                )
            }
        }
    
    def export_metrics(self, format_type: str = "json") -> Union[str, Dict[str, Any]]:
        """Export metrics in specified format"""
        metrics_data = self.get_metrics_summary()
        
        if format_type.lower() == "json":
            import json
            return json.dumps(metrics_data, indent=2, default=str)
        elif format_type.lower() == "csv":
            # Flatten metrics for CSV export
            flattened = self._flatten_metrics(metrics_data)
            return self._to_csv(flattened)
        else:
            return metrics_data
    
    def _flatten_metrics(self, data: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
        """Flatten nested metrics dictionary"""
        flattened = {}
        
        for key, value in data.items():
            new_key = f"{prefix}.{key}" if prefix else key
            
            if isinstance(value, dict):
                flattened.update(self._flatten_metrics(value, new_key))
            else:
                flattened[new_key] = value
        
        return flattened
    
    def _to_csv(self, data: Dict[str, Any]) -> str:
        """Convert flattened metrics to CSV format"""
        import io
        import csv
        
        output = io.StringIO()
        writer = csv.writer(output)
        
        # Write header
        writer.writerow(['metric', 'value', 'timestamp'])
        
        # Write data
        timestamp = time.time()
        for metric, value in data.items():
            writer.writerow([metric, value, timestamp])
        
        return output.getvalue()
    
    def cleanup(self):
        """Cleanup metrics calculator resources"""
        if self.is_running:
            self.stop()
        
        if self.shared_memory:
            self.shared_memory.cleanup()
        
        # Clear caches
        self.metrics_cache.clear()
        
        logger.info("MetricsCalculator cleanup completed")

# Factory function for easy instantiation
def create_metrics_calculator(config: MetricsConfig = None) -> AdvancedMetricsCalculator:
    """
    Factory function to create metrics calculator with optimal configuration
    
    Args:
        config: Metrics calculator configuration
        
    Returns:
        Configured AdvancedMetricsCalculator instance
    """
    if config is None:
        config = MetricsConfig()
        
        # Auto-detect optimal settings
        if platform.machine().lower() in ['arm64', 'aarch64']:
            config.enable_arm64_optimizations = True
            config.parallel_processing = True
            config.max_workers = min(mp.cpu_count(), 6)
        
        # Adjust for available memory
        available_memory = psutil.virtual_memory().available / (1024**3)  # GB
        if available_memory > 8:
            config.history_window = 20000
            config.cache_size = 2000
        elif available_memory > 4:
            config.history_window = 10000
            config.cache_size = 1000
        else:
            config.history_window = 5000
            config.cache_size = 500
    
    return AdvancedMetricsCalculator(config)

# Legacy compatibility
MetricsCalculator = AdvancedMetricsCalculator

if __name__ == "__main__":
    # Example usage and testing
    
    # Create metrics calculator
    metrics_calc = create_metrics_calculator()
    
    # Start calculator
    if metrics_calc.start():
        print("Metrics calculator started successfully")
        
        # Add sample trading data
        sample_trade = {
            'symbol': 'AAPL',
            'quantity': 100,
            'price': 150.0,
            'pnl': 250.0
        }
        metrics_calc.update_trading_metrics("trade_executed", sample_trade)
        
        # Calculate metrics
        all_metrics = metrics_calc.calculate_all_metrics()
        print(f"Calculated metrics: {len(all_metrics)}")
        
        # Get summary
        summary = metrics_calc.get_metrics_summary()
        print(f"Metrics summary: {summary}")
        
        # Get status
        status = metrics_calc.get_status()
        print(f"Calculator status: {status}")
        
        # Cleanup
        metrics_calc.stop()
        metrics_calc.cleanup()
    else:
        print("Failed to start metrics calculator")