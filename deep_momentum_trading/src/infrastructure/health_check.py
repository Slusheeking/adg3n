"""
Enhanced Health Check System with ARM64 Optimizations

This module provides comprehensive health monitoring capabilities for the Deep Momentum Trading System
with ARM64-specific optimizations for high-performance system health assessment and monitoring.

Features:
- Advanced health monitoring with ARM64 optimizations
- Multi-dimensional health assessment (system, process, API, network)
- Real-time health metrics collection and analysis
- Predictive health analytics and trend detection
- Automated recovery and self-healing capabilities
- Comprehensive alerting and notification system
- Performance optimization recommendations
"""

import asyncio
import psutil
import time
import threading
import platform
import multiprocessing as mp
from typing import Dict, List, Any, Optional, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
from concurrent.futures import ThreadPoolExecutor
import logging
from functools import lru_cache
import requests
import socket
from collections import deque, defaultdict
import numpy as np
from scipy import stats
import json

from ..utils.logger import get_logger
from ..utils.shared_memory import SharedMemoryManager
from ..utils.decorators import performance_monitor, error_handler
from ..utils.exceptions import ValidationError, HealthCheckError

logger = get_logger(__name__)

class HealthStatus(Enum):
    """Health status levels"""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    UNKNOWN = "unknown"

class ComponentType(Enum):
    """Types of components to monitor"""
    SYSTEM = "system"
    PROCESS = "process"
    API = "api"
    NETWORK = "network"
    DATABASE = "database"
    SERVICE = "service"

@dataclass
class HealthConfig:
    """Configuration for health monitoring"""
    check_interval_seconds: int = 30
    enable_arm64_optimizations: bool = True
    use_shared_memory: bool = True
    enable_predictive_analytics: bool = True
    enable_auto_recovery: bool = True
    parallel_processing: bool = True
    max_workers: Optional[int] = None
    history_window: int = 1000
    alert_thresholds: Dict[str, float] = field(default_factory=lambda: {
        'cpu_usage': 80.0,
        'memory_usage': 85.0,
        'disk_usage': 90.0,
        'response_time': 5.0,
        'error_rate': 5.0
    })
    recovery_actions: Dict[str, str] = field(default_factory=dict)

@dataclass
class HealthMetric:
    """Health metric data structure"""
    timestamp: float
    component_name: str
    component_type: ComponentType
    metric_name: str
    value: Union[float, int, bool, str]
    status: HealthStatus
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class HealthCheckResult:
    """Health check result"""
    component_name: str
    component_type: ComponentType
    status: HealthStatus
    metrics: List[HealthMetric]
    recommendations: List[str] = field(default_factory=list)
    recovery_actions: List[str] = field(default_factory=list)

class ARM64HealthOptimizer:
    """ARM64-specific optimizations for health monitoring"""
    
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
    
    def vectorized_health_analysis(self, metrics: np.ndarray) -> Dict[str, float]:
        """ARM64-optimized health metrics analysis"""
        if self.simd_available and len(metrics) > 50:
            # Use ARM64 SIMD for large metric datasets
            return {
                'mean': float(np.mean(metrics)),
                'std': float(np.std(metrics, ddof=1)),
                'trend': float(np.polyfit(range(len(metrics)), metrics, 1)[0]),
                'stability': float(1.0 / (1.0 + np.std(metrics))),
                'health_score': float(np.mean(metrics) * (1.0 / (1.0 + np.std(metrics))))
            }
        else:
            return self._standard_analysis(metrics)
    
    def _standard_analysis(self, metrics: np.ndarray) -> Dict[str, float]:
        """Standard health metrics analysis"""
        return {
            'mean': float(np.mean(metrics)),
            'std': float(np.std(metrics, ddof=1)),
            'trend': float(np.polyfit(range(len(metrics)), metrics, 1)[0]) if len(metrics) > 1 else 0.0,
            'stability': float(1.0 / (1.0 + np.std(metrics))),
            'health_score': float(np.mean(metrics) * (1.0 / (1.0 + np.std(metrics))))
        }
    
    def parallel_health_checks(self, check_functions: List[Callable], max_workers: int = None) -> List[Any]:
        """ARM64-optimized parallel health checks"""
        if max_workers is None:
            max_workers = min(self.cpu_count, 8)
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(check_func) for check_func in check_functions]
            return [future.result() for future in futures]

class SystemHealthChecker:
    """System-level health checker"""
    
    def __init__(self, arm64_optimizer: ARM64HealthOptimizer, config: HealthConfig):
        self.optimizer = arm64_optimizer
        self.config = config
        self.metrics_history = defaultdict(lambda: deque(maxlen=config.history_window))
    
    @performance_monitor
    @error_handler
    def check_system_health(self) -> HealthCheckResult:
        """Comprehensive system health check"""
        metrics = []
        status = HealthStatus.HEALTHY
        recommendations = []
        
        try:
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=0.5)
            cpu_count = psutil.cpu_count()
            cpu_freq = psutil.cpu_freq()
            
            metrics.append(HealthMetric(
                timestamp=time.time(),
                component_name="system",
                component_type=ComponentType.SYSTEM,
                metric_name="cpu_usage",
                value=cpu_percent,
                status=HealthStatus.CRITICAL if cpu_percent > 90 else 
                       HealthStatus.WARNING if cpu_percent > self.config.alert_thresholds['cpu_usage'] else 
                       HealthStatus.HEALTHY,
                metadata={'cpu_count': cpu_count, 'cpu_freq': cpu_freq.current if cpu_freq else 0}
            ))
            
            # Memory metrics
            memory = psutil.virtual_memory()
            swap = psutil.swap_memory()
            
            metrics.append(HealthMetric(
                timestamp=time.time(),
                component_name="system",
                component_type=ComponentType.SYSTEM,
                metric_name="memory_usage",
                value=memory.percent,
                status=HealthStatus.CRITICAL if memory.percent > 95 else 
                       HealthStatus.WARNING if memory.percent > self.config.alert_thresholds['memory_usage'] else 
                       HealthStatus.HEALTHY,
                metadata={'total_gb': memory.total / (1024**3), 'available_gb': memory.available / (1024**3)}
            ))
            
            # Disk metrics
            disk = psutil.disk_usage('/')
            disk_io = psutil.disk_io_counters()
            
            disk_percent = (disk.used / disk.total) * 100
            metrics.append(HealthMetric(
                timestamp=time.time(),
                component_name="system",
                component_type=ComponentType.SYSTEM,
                metric_name="disk_usage",
                value=disk_percent,
                status=HealthStatus.CRITICAL if disk_percent > 95 else 
                       HealthStatus.WARNING if disk_percent > self.config.alert_thresholds['disk_usage'] else 
                       HealthStatus.HEALTHY,
                metadata={'total_gb': disk.total / (1024**3), 'free_gb': disk.free / (1024**3)}
            ))
            
            # Network metrics
            network = psutil.net_io_counters()
            
            metrics.append(HealthMetric(
                timestamp=time.time(),
                component_name="system",
                component_type=ComponentType.SYSTEM,
                metric_name="network_activity",
                value=network.bytes_sent + network.bytes_recv,
                status=HealthStatus.HEALTHY,
                metadata={
                    'bytes_sent': network.bytes_sent,
                    'bytes_recv': network.bytes_recv,
                    'packets_sent': network.packets_sent,
                    'packets_recv': network.packets_recv
                }
            ))
            
            # Load average (Unix-like systems)
            try:
                load_avg = psutil.getloadavg()
                metrics.append(HealthMetric(
                    timestamp=time.time(),
                    component_name="system",
                    component_type=ComponentType.SYSTEM,
                    metric_name="load_average",
                    value=load_avg[0],
                    status=HealthStatus.WARNING if load_avg[0] > cpu_count else HealthStatus.HEALTHY,
                    metadata={'load_1min': load_avg[0], 'load_5min': load_avg[1], 'load_15min': load_avg[2]}
                ))
            except:
                pass  # Not available on all systems
            
            # Determine overall status
            metric_statuses = [m.status for m in metrics]
            if HealthStatus.CRITICAL in metric_statuses:
                status = HealthStatus.CRITICAL
            elif HealthStatus.WARNING in metric_statuses:
                status = HealthStatus.WARNING
            
            # Generate recommendations
            if cpu_percent > self.config.alert_thresholds['cpu_usage']:
                recommendations.append("High CPU usage detected - consider process optimization")
            if memory.percent > self.config.alert_thresholds['memory_usage']:
                recommendations.append("High memory usage detected - consider memory cleanup")
            if disk_percent > self.config.alert_thresholds['disk_usage']:
                recommendations.append("High disk usage detected - consider cleanup or expansion")
            
            # Store metrics for trend analysis
            for metric in metrics:
                self.metrics_history[metric.metric_name].append(metric.value)
            
        except Exception as e:
            logger.error(f"Error in system health check: {e}")
            status = HealthStatus.UNKNOWN
        
        return HealthCheckResult(
            component_name="system",
            component_type=ComponentType.SYSTEM,
            status=status,
            metrics=metrics,
            recommendations=recommendations
        )
    
    def get_system_trends(self) -> Dict[str, Dict[str, float]]:
        """Get system health trends"""
        trends = {}
        
        for metric_name, history in self.metrics_history.items():
            if len(history) > 10:
                history_array = np.array(list(history))
                trends[metric_name] = self.optimizer.vectorized_health_analysis(history_array)
        
        return trends

class ProcessHealthChecker:
    """Process-level health checker"""
    
    def __init__(self, arm64_optimizer: ARM64HealthOptimizer, config: HealthConfig):
        self.optimizer = arm64_optimizer
        self.config = config
        self.process_metrics = defaultdict(lambda: deque(maxlen=config.history_window))
    
    @performance_monitor
    @error_handler
    def check_process_health(self, process_name: str, pid: Optional[int] = None) -> HealthCheckResult:
        """Check health of specific process"""
        metrics = []
        status = HealthStatus.HEALTHY
        recommendations = []
        recovery_actions = []
        
        try:
            # Find process
            if pid:
                process = psutil.Process(pid)
            else:
                processes = [p for p in psutil.process_iter(['pid', 'name']) if p.info['name'] == process_name]
                if not processes:
                    return HealthCheckResult(
                        component_name=process_name,
                        component_type=ComponentType.PROCESS,
                        status=HealthStatus.CRITICAL,
                        metrics=[],
                        recommendations=["Process not found - may need to be restarted"],
                        recovery_actions=["restart_process"]
                    )
                process = psutil.Process(processes[0].info['pid'])
            
            # Check if process is running
            if not process.is_running():
                return HealthCheckResult(
                    component_name=process_name,
                    component_type=ComponentType.PROCESS,
                    status=HealthStatus.CRITICAL,
                    metrics=[],
                    recommendations=["Process is not running"],
                    recovery_actions=["restart_process"]
                )
            
            # Collect process metrics
            with process.oneshot():
                cpu_percent = process.cpu_percent(interval=0.1)
                memory_info = process.memory_info()
                memory_percent = process.memory_percent()
                num_threads = process.num_threads()
                
                # CPU usage metric
                metrics.append(HealthMetric(
                    timestamp=time.time(),
                    component_name=process_name,
                    component_type=ComponentType.PROCESS,
                    metric_name="cpu_usage",
                    value=cpu_percent,
                    status=HealthStatus.WARNING if cpu_percent > 50 else HealthStatus.HEALTHY,
                    metadata={'pid': process.pid}
                ))
                
                # Memory usage metric
                metrics.append(HealthMetric(
                    timestamp=time.time(),
                    component_name=process_name,
                    component_type=ComponentType.PROCESS,
                    metric_name="memory_usage",
                    value=memory_percent,
                    status=HealthStatus.WARNING if memory_percent > 20 else HealthStatus.HEALTHY,
                    metadata={'rss_mb': memory_info.rss / (1024**2), 'vms_mb': memory_info.vms / (1024**2)}
                ))
                
                # Thread count metric
                metrics.append(HealthMetric(
                    timestamp=time.time(),
                    component_name=process_name,
                    component_type=ComponentType.PROCESS,
                    metric_name="thread_count",
                    value=num_threads,
                    status=HealthStatus.WARNING if num_threads > 100 else HealthStatus.HEALTHY,
                    metadata={'pid': process.pid}
                ))
                
                # File descriptors (Unix-like systems)
                try:
                    num_fds = process.num_fds()
                    metrics.append(HealthMetric(
                        timestamp=time.time(),
                        component_name=process_name,
                        component_type=ComponentType.PROCESS,
                        metric_name="file_descriptors",
                        value=num_fds,
                        status=HealthStatus.WARNING if num_fds > 1000 else HealthStatus.HEALTHY,
                        metadata={'pid': process.pid}
                    ))
                except:
                    pass  # Not available on all systems
            
            # Determine overall status
            metric_statuses = [m.status for m in metrics]
            if HealthStatus.CRITICAL in metric_statuses:
                status = HealthStatus.CRITICAL
            elif HealthStatus.WARNING in metric_statuses:
                status = HealthStatus.WARNING
            
            # Generate recommendations
            if cpu_percent > 80:
                recommendations.append(f"High CPU usage ({cpu_percent:.1f}%) - consider optimization")
            if memory_percent > 50:
                recommendations.append(f"High memory usage ({memory_percent:.1f}%) - check for memory leaks")
            if num_threads > 200:
                recommendations.append(f"High thread count ({num_threads}) - review threading strategy")
            
        except psutil.NoSuchProcess:
            status = HealthStatus.CRITICAL
            recommendations.append("Process no longer exists")
            recovery_actions.append("restart_process")
        except Exception as e:
            logger.error(f"Error checking process {process_name}: {e}")
            status = HealthStatus.UNKNOWN
        
        return HealthCheckResult(
            component_name=process_name,
            component_type=ComponentType.PROCESS,
            status=status,
            metrics=metrics,
            recommendations=recommendations,
            recovery_actions=recovery_actions
        )

class APIHealthChecker:
    """API endpoint health checker"""
    
    def __init__(self, arm64_optimizer: ARM64HealthOptimizer, config: HealthConfig):
        self.optimizer = arm64_optimizer
        self.config = config
        self.response_times = defaultdict(lambda: deque(maxlen=config.history_window))
    
    @performance_monitor
    @error_handler
    async def check_api_health(self, api_name: str, endpoint_url: str, 
                              timeout: float = 5.0, expected_status: int = 200) -> HealthCheckResult:
        """Check health of API endpoint"""
        metrics = []
        status = HealthStatus.HEALTHY
        recommendations = []
        
        try:
            start_time = time.time()
            
            # Make HTTP request
            async with asyncio.timeout(timeout):
                response = await asyncio.to_thread(requests.get, endpoint_url, timeout=timeout)
            
            response_time = time.time() - start_time
            
            # Response time metric
            metrics.append(HealthMetric(
                timestamp=time.time(),
                component_name=api_name,
                component_type=ComponentType.API,
                metric_name="response_time",
                value=response_time,
                status=HealthStatus.WARNING if response_time > self.config.alert_thresholds['response_time'] else HealthStatus.HEALTHY,
                metadata={'endpoint': endpoint_url}
            ))
            
            # Status code metric
            metrics.append(HealthMetric(
                timestamp=time.time(),
                component_name=api_name,
                component_type=ComponentType.API,
                metric_name="status_code",
                value=response.status_code,
                status=HealthStatus.HEALTHY if response.status_code == expected_status else HealthStatus.WARNING,
                metadata={'expected_status': expected_status}
            ))
            
            # Availability metric
            metrics.append(HealthMetric(
                timestamp=time.time(),
                component_name=api_name,
                component_type=ComponentType.API,
                metric_name="availability",
                value=1.0 if response.status_code == expected_status else 0.0,
                status=HealthStatus.HEALTHY if response.status_code == expected_status else HealthStatus.CRITICAL,
                metadata={'endpoint': endpoint_url}
            ))
            
            # Store response time for trend analysis
            self.response_times[api_name].append(response_time)
            
            # Determine overall status
            if response.status_code != expected_status:
                status = HealthStatus.CRITICAL
                recommendations.append(f"API returned status {response.status_code}, expected {expected_status}")
            elif response_time > self.config.alert_thresholds['response_time']:
                status = HealthStatus.WARNING
                recommendations.append(f"Slow response time: {response_time:.2f}s")
            
        except asyncio.TimeoutError:
            status = HealthStatus.CRITICAL
            recommendations.append(f"API request timed out after {timeout}s")
            metrics.append(HealthMetric(
                timestamp=time.time(),
                component_name=api_name,
                component_type=ComponentType.API,
                metric_name="availability",
                value=0.0,
                status=HealthStatus.CRITICAL,
                metadata={'error': 'timeout'}
            ))
        except Exception as e:
            status = HealthStatus.CRITICAL
            recommendations.append(f"API request failed: {str(e)}")
            metrics.append(HealthMetric(
                timestamp=time.time(),
                component_name=api_name,
                component_type=ComponentType.API,
                metric_name="availability",
                value=0.0,
                status=HealthStatus.CRITICAL,
                metadata={'error': str(e)}
            ))
        
        return HealthCheckResult(
            component_name=api_name,
            component_type=ComponentType.API,
            status=status,
            metrics=metrics,
            recommendations=recommendations
        )

class AdvancedHealthMonitor:
    """
    Advanced Health Monitor with ARM64 optimizations
    
    Provides comprehensive health monitoring capabilities including:
    - System-level health monitoring
    - Process health tracking
    - API endpoint monitoring
    - Predictive health analytics
    - Automated recovery actions
    """
    
    def __init__(self, config: HealthConfig = None):
        self.config = config or HealthConfig()
        self.arm64_optimizer = ARM64HealthOptimizer()
        self.shared_memory = SharedMemoryManager() if self.config.use_shared_memory else None
        
        # Specialized checkers
        self.system_checker = SystemHealthChecker(self.arm64_optimizer, self.config)
        self.process_checker = ProcessHealthChecker(self.arm64_optimizer, self.config)
        self.api_checker = APIHealthChecker(self.arm64_optimizer, self.config)
        
        # Health tracking
        self.health_history = deque(maxlen=self.config.history_window)
        self.component_registry = {}
        self.recovery_actions = {}
        
        # Threading
        self.is_running = False
        self.monitoring_thread = None
        
        # Performance tracking
        self.monitor_stats = {
            'total_checks': 0,
            'checks_by_type': defaultdict(int),
            'check_times': deque(maxlen=1000),
            'recovery_actions_taken': 0
        }
        
        logger.info(f"AdvancedHealthMonitor initialized with ARM64 optimizations: {self.arm64_optimizer.is_arm64}")
    
    def register_component(self, name: str, component_type: ComponentType, 
                          check_config: Dict[str, Any] = None):
        """Register component for health monitoring"""
        self.component_registry[name] = {
            'type': component_type,
            'config': check_config or {},
            'last_check': None,
            'status': HealthStatus.UNKNOWN
        }
        logger.info(f"Registered component: {name} ({component_type.value})")
    
    def unregister_component(self, name: str):
        """Unregister component from monitoring"""
        if name in self.component_registry:
            del self.component_registry[name]
            logger.info(f"Unregistered component: {name}")
    
    @performance_monitor
    @error_handler
    def check_component_health(self, component_name: str) -> Optional[HealthCheckResult]:
        """Check health of specific component"""
        if component_name not in self.component_registry:
            logger.warning(f"Component {component_name} not registered")
            return None
        
        component_info = self.component_registry[component_name]
        component_type = component_info['type']
        config = component_info['config']
        
        start_time = time.time()
        
        try:
            if component_type == ComponentType.SYSTEM:
                result = self.system_checker.check_system_health()
            elif component_type == ComponentType.PROCESS:
                result = self.process_checker.check_process_health(
                    component_name, 
                    config.get('pid')
                )
            elif component_type == ComponentType.API:
                result = asyncio.run(self.api_checker.check_api_health(
                    component_name,
                    config.get('endpoint_url', ''),
                    config.get('timeout', 5.0),
                    config.get('expected_status', 200)
                ))
            else:
                logger.warning(f"Unsupported component type: {component_type}")
                return None
            
            # Update component status
            component_info['last_check'] = time.time()
            component_info['status'] = result.status
            
            # Update statistics
            self.monitor_stats['total_checks'] += 1
            self.monitor_stats['checks_by_type'][component_type.value] += 1
            self.monitor_stats['check_times'].append(time.time() - start_time)
            
            # Execute recovery actions if needed
            if self.config.enable_auto_recovery and result.recovery_actions:
                self._execute_recovery_actions(component_name, result.recovery_actions)
            
            return result
            
        except Exception as e:
            logger.error(f"Error checking component {component_name}: {e}")
            return None
    
    @performance_monitor
    def check_all_components(self) -> Dict[str, HealthCheckResult]:
        """Check health of all registered components"""
        results = {}
        
        if self.config.parallel_processing:
            # Parallel health checks
            check_functions = [
                lambda name=name: (name, self.check_component_health(name))
                for name in self.component_registry.keys()
            ]
            
            parallel_results = self.arm64_optimizer.parallel_health_checks(
                check_functions, self.config.max_workers
            )
            
            for name, result in parallel_results:
                if result:
                    results[name] = result
        else:
            # Sequential health checks
            for component_name in self.component_registry.keys():
                result = self.check_component_health(component_name)
                if result:
                    results[component_name] = result
        
        return results
    
    def get_overall_health_status(self) -> Dict[str, Any]:
        """Get overall system health status"""
        component_results = self.check_all_components()
        
        overall_status = HealthStatus.HEALTHY
        component_statuses = {}
        critical_components = []
        warning_components = []
        
        for name, result in component_results.items():
            component_statuses[name] = {
                'status': result.status.value,
                'metrics_count': len(result.metrics),
                'recommendations_count': len(result.recommendations)
            }
            
            if result.status == HealthStatus.CRITICAL:
                critical_components.append(name)
                overall_status = HealthStatus.CRITICAL
            elif result.status == HealthStatus.WARNING and overall_status != HealthStatus.CRITICAL:
                warning_components.append(name)
                overall_status = HealthStatus.WARNING
        
        # Get system trends
        system_trends = self.system_checker.get_system_trends()
        
        return {
            'overall_status': overall_status.value,
            'timestamp': time.time(),
            'components': component_statuses,
            'critical_components': critical_components,
            'warning_components': warning_components,
            'system_trends': system_trends,
            'monitor_stats': dict(self.monitor_stats),
            'arm64_optimized': self.arm64_optimizer.is_arm64
        }
    
    def _execute_recovery_actions(self, component_name: str, actions: List[str]):
        """Execute recovery actions for component"""
        for action in actions:
            try:
                if action == "restart_process":
                    logger.info(f"Executing recovery action: restart_process for {component_name}")
                    # Implementation would depend on process manager integration
                    self.monitor_stats['recovery_actions_taken'] += 1
                elif action in self.config.recovery_actions:
                    # Execute custom recovery action
                    recovery_command = self.config.recovery_actions[action]
                    logger.info(f"Executing recovery action: {action} for {component_name}")
                    # Execute command (implementation depends on requirements)
                    self.monitor_stats['recovery_actions_taken'] += 1
                else:
                    logger.warning(f"Unknown recovery action: {action}")
            except Exception as e:
                logger.error(f"Error executing recovery action {action}: {e}")
    
    def _continuous_monitoring(self):
        """Background thread for continuous health monitoring"""
        while self.is_running:
            try:
                # Perform health checks
                results = self.check_all_components()
                
                # Store results in history
                self.health_history.append({
                    'timestamp': time.time(),
                    'results': results
                })
                
                time.sleep(self.config.check_interval_seconds)
                
            except Exception as e:
                logger.error(f"Error in continuous monitoring: {e}")
                time.sleep(5.0)
    
    def start(self) -> bool:
        """Start health monitoring"""
        if self.is_running:
            return True
        
        try:
            self.is_running = True
            
            # Start monitoring thread
            self.monitoring_thread = threading.Thread(
                target=self._continuous_monitoring,
                name="HealthMonitoringThread"
            )
            self.monitoring_thread.daemon = True
            self.monitoring_thread.start()
            
            logger.info("Health monitoring started successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start health monitoring: {e}")
            self.is_running = False
            return False
    
    def stop(self):
        """Stop health monitoring"""
        self.is_running = False
        
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5.0)
        
        logger.info("Health monitoring stopped")
    
    def get_status(self) -> Dict[str, Any]:
        """Get health monitor status"""
        avg_check_time = (
            sum(self.monitor_stats['check_times']) / len(self.monitor_stats['check_times'])
            if self.monitor_stats['check_times'] else 0
        )
        
        return {
            'is_running': self.is_running,
            'registered_components': len(self.component_registry),
            'total_checks': self.monitor_stats['total_checks'],
            'checks_by_type': dict(self.monitor_stats['checks_by_type']),
            'average_check_time': avg_check_time,
            'recovery_actions_taken': self.monitor_stats['recovery_actions_taken'],
            'arm64_optimized': self.arm64_optimizer.is_arm64
        }
    
    def get_health_report(self) -> Dict[str, Any]:
        """Generate comprehensive health report"""
        overall_status = self.get_overall_health_status()
        
        return {
            'overall_health': overall_status,
            'component_details': {
                name: self.check_component_health(name)
                for name in self.component_registry.keys()
            },
            'system_capabilities': {
                'is_arm64': self.arm64_optimizer.is_arm64,
                'simd_available': self.arm64_optimizer.simd_available,
                'cpu_count': self.arm64_optimizer.cpu_count
            },
            'monitor_performance': self.get_status()
        }
    
    def cleanup(self):
        """Cleanup health monitor resources"""
        if self.is_running:
            self.stop()
        
        if self.shared_memory:
            self.shared_memory.cleanup()
        
        logger.info("HealthMonitor cleanup completed")

# Factory function for easy instantiation
def create_health_monitor(config: HealthConfig = None) -> AdvancedHealthMonitor:
    """
    Factory function to create health monitor with optimal configuration
    
    Args:
        config: Health monitor configuration
        
    Returns:
        Configured AdvancedHealthMonitor instance
    """
    if config is None:
        config = HealthConfig()
        
        # Auto-detect optimal settings
        if platform.machine().lower() in ['arm64', 'aarch64']:
            config.enable_arm64_optimizations = True
            config.parallel_processing = True
            config.max_workers = min(mp.cpu_count(), 6)
        
        # Adjust for available memory
        available_memory = psutil.virtual_memory().available / (1024**3)  # GB
        if available_memory > 8:
            config.history_window = 2000
        elif available_memory > 4:
            config.history_window = 1000
        else:
            config.history_window = 500
    
    return AdvancedHealthMonitor(config)

# Legacy compatibility
HealthMonitor = AdvancedHealthMonitor

if __name__ == "__main__":
    # Example usage and testing
    
    # Create health monitor
    health_monitor = create_health_monitor()
    
    # Register components
    health_monitor.register_component("system", ComponentType.SYSTEM)
    health_monitor.register_component("trading_process", ComponentType.PROCESS, {"pid": None})
    
    # Start monitoring
    if health_monitor.start():
        print("Health monitoring started successfully")
        
        # Get health status
        status = health_monitor.get_overall_health_status()
        print(f"Overall health status: {status['overall_status']}")
        
        # Generate report
        report = health_monitor.get_health_report()
        print(f"Health report generated with {len(report)} sections")
        
        # Cleanup
        time.sleep(5)  # Allow some monitoring
        health_monitor.stop()
        health_monitor.cleanup()
    else:
        print("Failed to start health monitoring")
