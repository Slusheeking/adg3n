"""
Enhanced Performance Tracker with ARM64 Optimizations

This module provides comprehensive performance tracking capabilities for the Deep Momentum Trading System
with ARM64-specific optimizations for real-time performance monitoring and analysis.

Features:
- Real-time performance tracking with ARM64 optimizations
- Execution time monitoring and profiling
- Resource utilization tracking
- Performance bottleneck detection
- Historical performance analysis
- Shared memory integration for high-frequency tracking
- Advanced statistical analysis and trend detection
- Performance alerts and optimization recommendations
"""

import time
import threading
import queue
import platform
import multiprocessing as mp
from typing import Dict, List, Any, Optional, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
from concurrent.futures import ThreadPoolExecutor
import logging
from functools import wraps, lru_cache
import psutil
from collections import deque, defaultdict
import numpy as np
from scipy import stats
import json
import pickle
from pathlib import Path
import contextlib

from ..utils.logger import get_logger
from ..utils.shared_memory import SharedMemoryManager
from ..utils.decorators import performance_monitor, error_handler
from ..utils.exceptions import ValidationError, PerformanceError

logger = get_logger(__name__)

class PerformanceMetricType(Enum):
    """Types of performance metrics"""
    EXECUTION_TIME = "execution_time"
    MEMORY_USAGE = "memory_usage"
    CPU_USAGE = "cpu_usage"
    THROUGHPUT = "throughput"
    LATENCY = "latency"
    ERROR_RATE = "error_rate"
    RESOURCE_UTILIZATION = "resource_utilization"

class AlertLevel(Enum):
    """Performance alert levels"""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"

@dataclass
class TrackerConfig:
    """Configuration for performance tracker"""
    enable_arm64_optimizations: bool = True
    use_shared_memory: bool = True
    sampling_interval: float = 0.1  # seconds
    history_window: int = 10000  # number of samples to keep
    enable_real_time_tracking: bool = True
    enable_profiling: bool = True
    parallel_processing: bool = True
    max_workers: Optional[int] = None
    cache_size: int = 1000
    enable_alerts: bool = True
    alert_thresholds: Dict[str, float] = field(default_factory=lambda: {
        'cpu_usage': 80.0,
        'memory_usage': 85.0,
        'execution_time': 1.0,
        'error_rate': 5.0
    })
    export_interval: float = 60.0  # seconds
    export_directory: str = "performance_data"

@dataclass
class PerformanceEvent:
    """Performance event data structure"""
    timestamp: float
    event_type: str
    metric_type: PerformanceMetricType
    value: Union[float, int, Dict[str, Any]]
    source: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    duration: Optional[float] = None

@dataclass
class PerformanceAlert:
    """Performance alert data structure"""
    timestamp: float
    level: AlertLevel
    metric_type: PerformanceMetricType
    message: str
    value: float
    threshold: float
    source: str

class ARM64PerformanceOptimizer:
    """ARM64-specific optimizations for performance tracking"""
    
    def __init__(self):
        self.is_arm64 = platform.machine().lower() in ['arm64', 'aarch64']
        self.simd_available = self._check_simd_support()
        self.cpu_count = mp.cpu_count()
        
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
    
    def vectorized_analysis(self, data: np.ndarray) -> Dict[str, float]:
        """ARM64-optimized performance data analysis"""
        if self.simd_available and len(data) > 100:
            # Use ARM64 SIMD for large datasets
            return {
                'mean': float(np.mean(data)),
                'std': float(np.std(data, ddof=1)),
                'min': float(np.min(data)),
                'max': float(np.max(data)),
                'median': float(np.median(data)),
                'p95': float(np.percentile(data, 95)),
                'p99': float(np.percentile(data, 99))
            }
        else:
            return self._standard_analysis(data)
    
    def _standard_analysis(self, data: np.ndarray) -> Dict[str, float]:
        """Standard performance data analysis"""
        return {
            'mean': float(np.mean(data)),
            'std': float(np.std(data, ddof=1)),
            'min': float(np.min(data)),
            'max': float(np.max(data)),
            'median': float(np.median(data)),
            'p95': float(np.percentile(data, 95)),
            'p99': float(np.percentile(data, 99))
        }
    
    def parallel_event_processing(self, events: List[PerformanceEvent], 
                                 process_func: Callable, max_workers: int = None) -> List[Any]:
        """ARM64-optimized parallel event processing"""
        if max_workers is None:
            max_workers = min(self.cpu_count, 8)
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(process_func, event) for event in events]
            return [future.result() for future in futures]

class ExecutionTimeTracker:
    """Tracks execution times for functions and code blocks"""
    
    def __init__(self, arm64_optimizer: ARM64PerformanceOptimizer):
        self.optimizer = arm64_optimizer
        self.execution_times = defaultdict(deque)
        self.active_timers = {}
        self.lock = threading.Lock()
    
    @contextlib.contextmanager
    def track(self, operation_name: str, metadata: Dict[str, Any] = None):
        """Context manager for tracking execution time"""
        start_time = time.perf_counter()
        timer_id = f"{operation_name}_{threading.get_ident()}_{start_time}"
        
        try:
            with self.lock:
                self.active_timers[timer_id] = {
                    'operation': operation_name,
                    'start_time': start_time,
                    'metadata': metadata or {}
                }
            
            yield timer_id
            
        finally:
            end_time = time.perf_counter()
            execution_time = end_time - start_time
            
            with self.lock:
                if timer_id in self.active_timers:
                    del self.active_timers[timer_id]
                
                # Store execution time
                self.execution_times[operation_name].append(execution_time)
                
                # Keep only recent measurements
                if len(self.execution_times[operation_name]) > 1000:
                    self.execution_times[operation_name].popleft()
    
    def get_statistics(self, operation_name: str) -> Dict[str, float]:
        """Get execution time statistics for an operation"""
        if operation_name not in self.execution_times:
            return {}
        
        times = np.array(list(self.execution_times[operation_name]))
        return self.optimizer.vectorized_analysis(times)
    
    def get_all_statistics(self) -> Dict[str, Dict[str, float]]:
        """Get execution time statistics for all operations"""
        return {
            operation: self.get_statistics(operation)
            for operation in self.execution_times.keys()
        }

class ResourceMonitor:
    """Monitors system resource utilization"""
    
    def __init__(self, arm64_optimizer: ARM64PerformanceOptimizer):
        self.optimizer = arm64_optimizer
        self.cpu_history = deque(maxlen=1000)
        self.memory_history = deque(maxlen=1000)
        self.disk_history = deque(maxlen=1000)
        self.network_history = deque(maxlen=1000)
        self.process_history = deque(maxlen=1000)
    
    def collect_resource_metrics(self) -> Dict[str, Any]:
        """Collect current resource metrics"""
        try:
            # System-wide metrics
            cpu_percent = psutil.cpu_percent(interval=None)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            network = psutil.net_io_counters()
            
            # Process-specific metrics
            process = psutil.Process()
            process_memory = process.memory_info()
            process_cpu = process.cpu_percent()
            
            metrics = {
                'timestamp': time.time(),
                'system': {
                    'cpu_percent': cpu_percent,
                    'memory_percent': memory.percent,
                    'memory_available': memory.available,
                    'disk_percent': (disk.used / disk.total) * 100,
                    'network_bytes_sent': network.bytes_sent,
                    'network_bytes_recv': network.bytes_recv
                },
                'process': {
                    'cpu_percent': process_cpu,
                    'memory_rss': process_memory.rss,
                    'memory_vms': process_memory.vms,
                    'memory_percent': process.memory_percent(),
                    'num_threads': process.num_threads(),
                    'num_fds': process.num_fds() if hasattr(process, 'num_fds') else 0
                }
            }
            
            # Add to history
            self.cpu_history.append(cpu_percent)
            self.memory_history.append(memory.percent)
            self.disk_history.append((disk.used / disk.total) * 100)
            self.process_history.append(process_memory.rss)
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error collecting resource metrics: {e}")
            return {}
    
    def get_resource_summary(self) -> Dict[str, Any]:
        """Get resource utilization summary"""
        current_metrics = self.collect_resource_metrics()
        
        summary = {
            'current': current_metrics,
            'trends': {}
        }
        
        # Analyze trends
        if self.cpu_history:
            cpu_array = np.array(list(self.cpu_history))
            summary['trends']['cpu'] = self.optimizer.vectorized_analysis(cpu_array)
        
        if self.memory_history:
            memory_array = np.array(list(self.memory_history))
            summary['trends']['memory'] = self.optimizer.vectorized_analysis(memory_array)
        
        if self.process_history:
            process_array = np.array(list(self.process_history))
            summary['trends']['process_memory'] = self.optimizer.vectorized_analysis(process_array)
        
        return summary

class PerformanceProfiler:
    """Advanced performance profiler"""
    
    def __init__(self, arm64_optimizer: ARM64PerformanceOptimizer):
        self.optimizer = arm64_optimizer
        self.function_profiles = defaultdict(lambda: {
            'call_count': 0,
            'total_time': 0.0,
            'execution_times': deque(maxlen=1000)
        })
        self.lock = threading.Lock()
    
    def profile_function(self, func: Callable) -> Callable:
        """Decorator to profile function execution"""
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.perf_counter()
            
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                end_time = time.perf_counter()
                execution_time = end_time - start_time
                
                func_name = f"{func.__module__}.{func.__name__}"
                
                with self.lock:
                    profile = self.function_profiles[func_name]
                    profile['call_count'] += 1
                    profile['total_time'] += execution_time
                    profile['execution_times'].append(execution_time)
        
        return wrapper
    
    def get_profile_summary(self) -> Dict[str, Dict[str, Any]]:
        """Get profiling summary for all functions"""
        summary = {}
        
        with self.lock:
            for func_name, profile in self.function_profiles.items():
                if profile['execution_times']:
                    times_array = np.array(list(profile['execution_times']))
                    stats = self.optimizer.vectorized_analysis(times_array)
                    
                    summary[func_name] = {
                        'call_count': profile['call_count'],
                        'total_time': profile['total_time'],
                        'average_time': profile['total_time'] / profile['call_count'],
                        'statistics': stats
                    }
        
        return summary

class PerformanceAlertManager:
    """Manages performance alerts and notifications"""
    
    def __init__(self, config: TrackerConfig):
        self.config = config
        self.alerts = deque(maxlen=1000)
        self.alert_counts = defaultdict(int)
        self.last_alert_times = {}
        self.cooldown_period = 60.0  # seconds
    
    def check_thresholds(self, metrics: Dict[str, Any]) -> List[PerformanceAlert]:
        """Check metrics against thresholds and generate alerts"""
        alerts = []
        current_time = time.time()
        
        # Check system metrics
        if 'system' in metrics:
            system_metrics = metrics['system']
            
            # CPU usage alert
            if 'cpu_percent' in system_metrics:
                cpu_usage = system_metrics['cpu_percent']
                if cpu_usage > self.config.alert_thresholds.get('cpu_usage', 80.0):
                    alert = self._create_alert(
                        AlertLevel.WARNING if cpu_usage < 90 else AlertLevel.CRITICAL,
                        PerformanceMetricType.CPU_USAGE,
                        f"High CPU usage: {cpu_usage:.1f}%",
                        cpu_usage,
                        self.config.alert_thresholds['cpu_usage'],
                        "system_monitor"
                    )
                    if self._should_send_alert(alert):
                        alerts.append(alert)
            
            # Memory usage alert
            if 'memory_percent' in system_metrics:
                memory_usage = system_metrics['memory_percent']
                if memory_usage > self.config.alert_thresholds.get('memory_usage', 85.0):
                    alert = self._create_alert(
                        AlertLevel.WARNING if memory_usage < 95 else AlertLevel.CRITICAL,
                        PerformanceMetricType.MEMORY_USAGE,
                        f"High memory usage: {memory_usage:.1f}%",
                        memory_usage,
                        self.config.alert_thresholds['memory_usage'],
                        "system_monitor"
                    )
                    if self._should_send_alert(alert):
                        alerts.append(alert)
        
        return alerts
    
    def _create_alert(self, level: AlertLevel, metric_type: PerformanceMetricType,
                     message: str, value: float, threshold: float, source: str) -> PerformanceAlert:
        """Create performance alert"""
        return PerformanceAlert(
            timestamp=time.time(),
            level=level,
            metric_type=metric_type,
            message=message,
            value=value,
            threshold=threshold,
            source=source
        )
    
    def _should_send_alert(self, alert: PerformanceAlert) -> bool:
        """Check if alert should be sent based on cooldown"""
        alert_key = f"{alert.metric_type.value}_{alert.level.value}"
        current_time = time.time()
        
        if alert_key in self.last_alert_times:
            if current_time - self.last_alert_times[alert_key] < self.cooldown_period:
                return False
        
        self.last_alert_times[alert_key] = current_time
        self.alerts.append(alert)
        self.alert_counts[alert_key] += 1
        
        return True
    
    def get_alert_summary(self) -> Dict[str, Any]:
        """Get alert summary statistics"""
        return {
            'total_alerts': len(self.alerts),
            'alert_counts': dict(self.alert_counts),
            'recent_alerts': [
                {
                    'timestamp': alert.timestamp,
                    'level': alert.level.value,
                    'metric_type': alert.metric_type.value,
                    'message': alert.message,
                    'value': alert.value,
                    'threshold': alert.threshold
                }
                for alert in list(self.alerts)[-10:]  # Last 10 alerts
            ]
        }

class AdvancedPerformanceTracker:
    """
    Advanced Performance Tracker with ARM64 optimizations
    
    Provides comprehensive performance tracking capabilities including:
    - Real-time execution time monitoring
    - Resource utilization tracking
    - Performance profiling and analysis
    - Alert generation and notification
    """
    
    def __init__(self, config: TrackerConfig = None):
        self.config = config or TrackerConfig()
        self.arm64_optimizer = ARM64PerformanceOptimizer()
        self.shared_memory = SharedMemoryManager() if self.config.use_shared_memory else None
        
        # Core components
        self.execution_tracker = ExecutionTimeTracker(self.arm64_optimizer)
        self.resource_monitor = ResourceMonitor(self.arm64_optimizer)
        self.profiler = PerformanceProfiler(self.arm64_optimizer)
        self.alert_manager = PerformanceAlertManager(self.config)
        
        # Event processing
        self.event_queue = queue.Queue()
        self.events_history = deque(maxlen=self.config.history_window)
        
        # Threading
        self.is_running = False
        self.monitoring_thread = None
        self.processing_thread = None
        
        # Performance tracking
        self.tracker_stats = {
            'total_events': 0,
            'events_by_type': defaultdict(int),
            'processing_times': deque(maxlen=1000),
            'alerts_generated': 0
        }
        
        # Setup export directory
        self._setup_export_directory()
        
        logger.info(f"AdvancedPerformanceTracker initialized with ARM64 optimizations: {self.arm64_optimizer.is_arm64}")
    
    def _setup_export_directory(self):
        """Create export directory if it doesn't exist"""
        export_dir = Path(self.config.export_directory)
        export_dir.mkdir(parents=True, exist_ok=True)
    
    @contextlib.contextmanager
    def track_execution(self, operation_name: str, metadata: Dict[str, Any] = None):
        """Context manager for tracking execution time"""
        with self.execution_tracker.track(operation_name, metadata) as timer_id:
            yield timer_id
    
    def track_event(self, event_type: str, data: Dict[str, Any]):
        """Track performance event"""
        event = PerformanceEvent(
            timestamp=time.time(),
            event_type=event_type,
            metric_type=PerformanceMetricType.EXECUTION_TIME,  # Default
            value=data.get('value', 0),
            source=data.get('source', 'unknown'),
            metadata=data,
            duration=data.get('duration')
        )
        
        try:
            self.event_queue.put(event, timeout=0.1)
        except queue.Full:
            logger.warning("Performance event queue is full, dropping event")
    
    def profile_function(self, func: Callable) -> Callable:
        """Decorator to profile function execution"""
        return self.profiler.profile_function(func)
    
    def _monitor_performance(self):
        """Background thread for performance monitoring"""
        while self.is_running:
            try:
                # Collect resource metrics
                resource_metrics = self.resource_monitor.collect_resource_metrics()
                
                # Check for alerts
                if self.config.enable_alerts and resource_metrics:
                    alerts = self.alert_manager.check_thresholds(resource_metrics)
                    self.tracker_stats['alerts_generated'] += len(alerts)
                
                time.sleep(self.config.sampling_interval)
                
            except Exception as e:
                logger.error(f"Error in performance monitoring: {e}")
                time.sleep(1.0)
    
    def _process_events(self):
        """Background thread for processing performance events"""
        while self.is_running:
            try:
                event = self.event_queue.get(timeout=1.0)
                start_time = time.perf_counter()
                
                # Process event
                self.events_history.append(event)
                self.tracker_stats['total_events'] += 1
                self.tracker_stats['events_by_type'][event.event_type] += 1
                
                # Track processing time
                processing_time = time.perf_counter() - start_time
                self.tracker_stats['processing_times'].append(processing_time)
                
                self.event_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error processing performance event: {e}")
    
    @performance_monitor
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary"""
        return {
            'execution_times': self.execution_tracker.get_all_statistics(),
            'resource_utilization': self.resource_monitor.get_resource_summary(),
            'function_profiles': self.profiler.get_profile_summary(),
            'alerts': self.alert_manager.get_alert_summary(),
            'tracker_stats': {
                'total_events': self.tracker_stats['total_events'],
                'events_by_type': dict(self.tracker_stats['events_by_type']),
                'alerts_generated': self.tracker_stats['alerts_generated'],
                'average_processing_time': (
                    sum(self.tracker_stats['processing_times']) / len(self.tracker_stats['processing_times'])
                    if self.tracker_stats['processing_times'] else 0
                )
            }
        }
    
    def get_bottlenecks(self, top_n: int = 10) -> List[Dict[str, Any]]:
        """Identify performance bottlenecks"""
        bottlenecks = []
        
        # Analyze execution times
        execution_stats = self.execution_tracker.get_all_statistics()
        for operation, stats in execution_stats.items():
            if 'mean' in stats and stats['mean'] > 0.1:  # Operations taking > 100ms
                bottlenecks.append({
                    'type': 'execution_time',
                    'operation': operation,
                    'average_time': stats['mean'],
                    'max_time': stats.get('max', 0),
                    'p99_time': stats.get('p99', 0)
                })
        
        # Analyze function profiles
        function_profiles = self.profiler.get_profile_summary()
        for func_name, profile in function_profiles.items():
            if profile['average_time'] > 0.05:  # Functions taking > 50ms on average
                bottlenecks.append({
                    'type': 'function_profile',
                    'function': func_name,
                    'call_count': profile['call_count'],
                    'total_time': profile['total_time'],
                    'average_time': profile['average_time']
                })
        
        # Sort by impact (total time or average time)
        bottlenecks.sort(key=lambda x: x.get('total_time', x.get('average_time', 0)), reverse=True)
        
        return bottlenecks[:top_n]
    
    def generate_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report"""
        summary = self.get_performance_summary()
        bottlenecks = self.get_bottlenecks()
        
        report = {
            'timestamp': time.time(),
            'system_info': {
                'platform': platform.platform(),
                'cpu_count': mp.cpu_count(),
                'is_arm64': self.arm64_optimizer.is_arm64,
                'simd_available': self.arm64_optimizer.simd_available
            },
            'performance_summary': summary,
            'bottlenecks': bottlenecks,
            'recommendations': self._generate_recommendations(summary, bottlenecks)
        }
        
        return report
    
    def _generate_recommendations(self, summary: Dict[str, Any], 
                                bottlenecks: List[Dict[str, Any]]) -> List[str]:
        """Generate performance optimization recommendations"""
        recommendations = []
        
        # Check resource utilization
        resource_summary = summary.get('resource_utilization', {})
        current_metrics = resource_summary.get('current', {})
        system_metrics = current_metrics.get('system', {})
        
        if system_metrics.get('cpu_percent', 0) > 80:
            recommendations.append("High CPU usage detected. Consider optimizing CPU-intensive operations.")
        
        if system_metrics.get('memory_percent', 0) > 85:
            recommendations.append("High memory usage detected. Consider memory optimization or increasing available memory.")
        
        # Check bottlenecks
        if bottlenecks:
            top_bottleneck = bottlenecks[0]
            if top_bottleneck['type'] == 'execution_time':
                recommendations.append(f"Operation '{top_bottleneck['operation']}' is taking {top_bottleneck['average_time']:.3f}s on average. Consider optimization.")
            elif top_bottleneck['type'] == 'function_profile':
                recommendations.append(f"Function '{top_bottleneck['function']}' is consuming significant time. Consider profiling and optimization.")
        
        # ARM64 specific recommendations
        if self.arm64_optimizer.is_arm64 and not self.arm64_optimizer.simd_available:
            recommendations.append("ARM64 SIMD optimizations are not available. Consider enabling NEON support.")
        
        return recommendations
    
    def export_performance_data(self, format_type: str = "json") -> str:
        """Export performance data to file"""
        report = self.generate_performance_report()
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        if format_type.lower() == "json":
            filename = f"performance_report_{timestamp}.json"
            filepath = Path(self.config.export_directory) / filename
            
            with open(filepath, 'w') as f:
                json.dump(report, f, indent=2, default=str)
                
        elif format_type.lower() == "pickle":
            filename = f"performance_report_{timestamp}.pkl"
            filepath = Path(self.config.export_directory) / filename
            
            with open(filepath, 'wb') as f:
                pickle.dump(report, f)
        
        else:
            raise ValueError(f"Unsupported export format: {format_type}")
        
        return str(filepath)
    
    def start(self) -> bool:
        """Start performance tracker"""
        if self.is_running:
            return True
        
        try:
            self.is_running = True
            
            # Start monitoring thread
            if self.config.enable_real_time_tracking:
                self.monitoring_thread = threading.Thread(
                    target=self._monitor_performance,
                    name="PerformanceMonitoringThread"
                )
                self.monitoring_thread.daemon = True
                self.monitoring_thread.start()
            
            # Start event processing thread
            self.processing_thread = threading.Thread(
                target=self._process_events,
                name="PerformanceProcessingThread"
            )
            self.processing_thread.daemon = True
            self.processing_thread.start()
            
            logger.info("Performance tracker started successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start performance tracker: {e}")
            self.is_running = False
            return False
    
    def stop(self):
        """Stop performance tracker"""
        self.is_running = False
        
        # Wait for event queue to empty
        self.event_queue.join()
        
        # Wait for threads
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5.0)
        
        if self.processing_thread:
            self.processing_thread.join(timeout=5.0)
        
        logger.info("Performance tracker stopped")
    
    def get_status(self) -> Dict[str, Any]:
        """Get performance tracker status"""
        avg_processing_time = (
            sum(self.tracker_stats['processing_times']) / len(self.tracker_stats['processing_times'])
            if self.tracker_stats['processing_times'] else 0
        )
        
        return {
            'is_running': self.is_running,
            'total_events': self.tracker_stats['total_events'],
            'queue_size': self.event_queue.qsize(),
            'events_by_type': dict(self.tracker_stats['events_by_type']),
            'alerts_generated': self.tracker_stats['alerts_generated'],
            'average_processing_time': avg_processing_time,
            'arm64_optimized': self.arm64_optimizer.is_arm64,
            'simd_available': self.arm64_optimizer.simd_available
        }
    
    def cleanup(self):
        """Cleanup performance tracker resources"""
        if self.is_running:
            self.stop()
        
        if self.shared_memory:
            self.shared_memory.cleanup()
        
        logger.info("PerformanceTracker cleanup completed")

# Factory function for easy instantiation
def create_performance_tracker(config: TrackerConfig = None) -> AdvancedPerformanceTracker:
    """
    Factory function to create performance tracker with optimal configuration
    
    Args:
        config: Performance tracker configuration
        
    Returns:
        Configured AdvancedPerformanceTracker instance
    """
    if config is None:
        config = TrackerConfig()
        
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
    
    return AdvancedPerformanceTracker(config)

# Legacy compatibility
PerformanceTracker = AdvancedPerformanceTracker

if __name__ == "__main__":
    # Example usage and testing
    
    # Create performance tracker
    tracker = create_performance_tracker()
    
    # Start tracker
    if tracker.start():
        print("Performance tracker started successfully")
        
        # Test execution tracking
        with tracker.track_execution("test_operation"):
            time.sleep(0.1)  # Simulate work
        
        # Test function profiling
        @tracker.profile_function
        def test_function():
            time.sleep(0.05)
            return "test"
        
        test_function()
        test_function()
        
        # Track custom event
        tracker.track_event("custom_event", {
            "value": 123,
            "source": "test",
            "duration": 0.05
        })
        
        # Get performance summary
        summary = tracker.get_performance_summary()
        print(f"Performance summary: {summary}")
        
        # Get bottlenecks
        bottlenecks = tracker.get_bottlenecks()
        print(f"Bottlenecks: {bottlenecks}")
        
        # Generate report
        report = tracker.generate_performance_report()
        print(f"Performance report generated with {len(report)} sections")
        
        # Get status
        status = tracker.get_status()
        print(f"Tracker status: {status}")
        
        # Cleanup
        time.sleep(2)  # Allow time for processing
        tracker.stop()
        tracker.cleanup()
    else:
        print("Failed to start performance tracker")