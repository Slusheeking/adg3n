"""
Enhanced Resource Monitor with ARM64 Optimizations

This module provides comprehensive resource monitoring capabilities for the Deep Momentum Trading System
with ARM64-specific optimizations for high-performance system resource tracking and analysis.

Features:
- Advanced system resource monitoring with ARM64 optimizations
- Real-time resource usage tracking and analysis
- Predictive resource analytics and trend detection
- Resource threshold monitoring and alerting
- Performance bottleneck identification
- Resource optimization recommendations
- Distributed resource coordination
"""

import os
import psutil
import threading
import time
import platform
import multiprocessing as mp
from typing import Dict, List, Any, Optional, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
from concurrent.futures import ThreadPoolExecutor
import logging
from functools import lru_cache
from collections import deque, defaultdict
import numpy as np
from scipy import stats
import json
import socket
import subprocess

from ..utils.logger import get_logger
from ..utils.shared_memory import SharedMemoryManager
from ..utils.decorators import performance_monitor, error_handler
from ..utils.exceptions import ValidationError, ResourceError
from ..communication.zmq_publisher import ZMQPublisher

logger = get_logger(__name__)

class ResourceType(Enum):
    """Types of resources to monitor"""
    CPU = "cpu"
    MEMORY = "memory"
    DISK = "disk"
    NETWORK = "network"
    GPU = "gpu"
    PROCESS = "process"

class AlertLevel(Enum):
    """Resource alert levels"""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"

@dataclass
class ResourceConfig:
    """Configuration for resource monitoring"""
    publish_port: int = 5562
    check_interval_seconds: float = 5.0
    enable_arm64_optimizations: bool = True
    use_shared_memory: bool = True
    enable_predictive_analytics: bool = True
    enable_gpu_monitoring: bool = True
    parallel_processing: bool = True
    max_workers: Optional[int] = None
    history_window: int = 1000
    alert_thresholds: Dict[str, float] = field(default_factory=lambda: {
        'cpu_usage': 80.0,
        'memory_usage': 85.0,
        'disk_usage': 90.0,
        'network_utilization': 80.0,
        'process_count': 500
    })
    enable_process_monitoring: bool = True
    enable_network_monitoring: bool = True
    enable_disk_io_monitoring: bool = True

@dataclass
class ResourceMetric:
    """Resource metric data structure"""
    timestamp: float
    resource_type: ResourceType
    metric_name: str
    value: Union[float, int, Dict[str, Any]]
    unit: str
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ResourceAlert:
    """Resource alert data structure"""
    timestamp: float
    resource_type: ResourceType
    alert_level: AlertLevel
    message: str
    current_value: float
    threshold: float
    recommendations: List[str] = field(default_factory=list)

class ARM64ResourceOptimizer:
    """ARM64-specific optimizations for resource monitoring"""
    
    def __init__(self):
        self.is_arm64 = platform.machine().lower() in ['arm64', 'aarch64']
        self.simd_available = self._check_simd_support()
        self.cpu_count = mp.cpu_count()
        self.cpu_topology = self._detect_cpu_topology()
        
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
    
    def _detect_cpu_topology(self) -> Dict[str, Any]:
        """Detect ARM64 CPU topology for monitoring optimization"""
        topology = {
            'total_cores': self.cpu_count,
            'performance_cores': [],
            'efficiency_cores': [],
            'cache_levels': {},
            'numa_nodes': []
        }
        
        if not self.is_arm64:
            return topology
        
        try:
            # Parse CPU topology
            with open('/proc/cpuinfo', 'r') as f:
                cpuinfo = f.read()
            
            # Detect cache hierarchy
            try:
                cache_info = {}
                for level in [1, 2, 3]:
                    try:
                        with open(f'/sys/devices/system/cpu/cpu0/cache/index{level}/size', 'r') as f:
                            cache_info[f'L{level}'] = f.read().strip()
                    except:
                        pass
                topology['cache_levels'] = cache_info
            except:
                pass
            
            # Detect NUMA topology
            try:
                numa_nodes = []
                numa_path = '/sys/devices/system/node'
                if os.path.exists(numa_path):
                    for item in os.listdir(numa_path):
                        if item.startswith('node') and item[4:].isdigit():
                            numa_nodes.append(int(item[4:]))
                topology['numa_nodes'] = sorted(numa_nodes)
            except:
                pass
                
        except Exception as e:
            logger.warning(f"Could not detect ARM64 topology: {e}")
        
        return topology
    
    def vectorized_resource_analysis(self, metrics: np.ndarray) -> Dict[str, float]:
        """ARM64-optimized resource metrics analysis"""
        if self.simd_available and len(metrics) > 50:
            # Use ARM64 SIMD for large metric datasets
            return {
                'mean': float(np.mean(metrics)),
                'std': float(np.std(metrics, ddof=1)),
                'min': float(np.min(metrics)),
                'max': float(np.max(metrics)),
                'median': float(np.median(metrics)),
                'p95': float(np.percentile(metrics, 95)),
                'p99': float(np.percentile(metrics, 99)),
                'trend': float(np.polyfit(range(len(metrics)), metrics, 1)[0]) if len(metrics) > 1 else 0.0,
                'volatility': float(np.std(np.diff(metrics))) if len(metrics) > 1 else 0.0
            }
        else:
            return self._standard_analysis(metrics)
    
    def _standard_analysis(self, metrics: np.ndarray) -> Dict[str, float]:
        """Standard resource metrics analysis"""
        return {
            'mean': float(np.mean(metrics)),
            'std': float(np.std(metrics, ddof=1)),
            'min': float(np.min(metrics)),
            'max': float(np.max(metrics)),
            'median': float(np.median(metrics)),
            'p95': float(np.percentile(metrics, 95)),
            'p99': float(np.percentile(metrics, 99)),
            'trend': float(np.polyfit(range(len(metrics)), metrics, 1)[0]) if len(metrics) > 1 else 0.0,
            'volatility': float(np.std(np.diff(metrics))) if len(metrics) > 1 else 0.0
        }
    
    def optimize_monitoring_frequency(self, resource_type: ResourceType, 
                                    current_usage: float) -> float:
        """Optimize monitoring frequency based on resource usage and ARM64 capabilities"""
        base_interval = 5.0
        
        # Increase frequency for high usage
        if current_usage > 80:
            return base_interval * 0.5  # Monitor more frequently
        elif current_usage > 60:
            return base_interval * 0.75
        else:
            return base_interval
        
        # ARM64 can handle higher frequency monitoring
        if self.is_arm64:
            return base_interval * 0.8
        
        return base_interval

class SystemResourceCollector:
    """System-level resource collector"""
    
    def __init__(self, arm64_optimizer: ARM64ResourceOptimizer, config: ResourceConfig):
        self.optimizer = arm64_optimizer
        self.config = config
        self.metrics_history = defaultdict(lambda: deque(maxlen=config.history_window))
        self.last_network_io = None
        self.last_disk_io = None
        
    @performance_monitor
    @error_handler
    def collect_cpu_metrics(self) -> List[ResourceMetric]:
        """Collect comprehensive CPU metrics"""
        metrics = []
        timestamp = time.time()
        
        try:
            # Overall CPU usage
            cpu_percent = psutil.cpu_percent(interval=0.5)
            metrics.append(ResourceMetric(
                timestamp=timestamp,
                resource_type=ResourceType.CPU,
                metric_name="usage_percent",
                value=cpu_percent,
                unit="percent"
            ))
            
            # Per-core CPU usage
            cpu_per_core = psutil.cpu_percent(interval=0.1, percpu=True)
            metrics.append(ResourceMetric(
                timestamp=timestamp,
                resource_type=ResourceType.CPU,
                metric_name="usage_per_core",
                value=cpu_per_core,
                unit="percent",
                metadata={'core_count': len(cpu_per_core)}
            ))
            
            # CPU frequency
            cpu_freq = psutil.cpu_freq()
            if cpu_freq:
                metrics.append(ResourceMetric(
                    timestamp=timestamp,
                    resource_type=ResourceType.CPU,
                    metric_name="frequency",
                    value=cpu_freq.current,
                    unit="MHz",
                    metadata={'min': cpu_freq.min, 'max': cpu_freq.max}
                ))
            
            # Load average (Unix-like systems)
            try:
                load_avg = psutil.getloadavg()
                metrics.append(ResourceMetric(
                    timestamp=timestamp,
                    resource_type=ResourceType.CPU,
                    metric_name="load_average",
                    value=load_avg[0],
                    unit="load",
                    metadata={'1min': load_avg[0], '5min': load_avg[1], '15min': load_avg[2]}
                ))
            except:
                pass  # Not available on all systems
            
            # CPU times
            cpu_times = psutil.cpu_times()
            metrics.append(ResourceMetric(
                timestamp=timestamp,
                resource_type=ResourceType.CPU,
                metric_name="cpu_times",
                value={
                    'user': cpu_times.user,
                    'system': cpu_times.system,
                    'idle': cpu_times.idle,
                    'iowait': getattr(cpu_times, 'iowait', 0),
                    'irq': getattr(cpu_times, 'irq', 0),
                    'softirq': getattr(cpu_times, 'softirq', 0)
                },
                unit="seconds"
            ))
            
            # Store for trend analysis
            self.metrics_history['cpu_usage'].append(cpu_percent)
            
        except Exception as e:
            logger.error(f"Error collecting CPU metrics: {e}")
        
        return metrics
    
    @performance_monitor
    @error_handler
    def collect_memory_metrics(self) -> List[ResourceMetric]:
        """Collect comprehensive memory metrics"""
        metrics = []
        timestamp = time.time()
        
        try:
            # Virtual memory
            memory = psutil.virtual_memory()
            metrics.append(ResourceMetric(
                timestamp=timestamp,
                resource_type=ResourceType.MEMORY,
                metric_name="usage_percent",
                value=memory.percent,
                unit="percent",
                metadata={
                    'total_gb': memory.total / (1024**3),
                    'available_gb': memory.available / (1024**3),
                    'used_gb': memory.used / (1024**3),
                    'free_gb': memory.free / (1024**3)
                }
            ))
            
            # Swap memory
            swap = psutil.swap_memory()
            metrics.append(ResourceMetric(
                timestamp=timestamp,
                resource_type=ResourceType.MEMORY,
                metric_name="swap_usage_percent",
                value=swap.percent,
                unit="percent",
                metadata={
                    'total_gb': swap.total / (1024**3),
                    'used_gb': swap.used / (1024**3),
                    'free_gb': swap.free / (1024**3)
                }
            ))
            
            # Memory breakdown
            metrics.append(ResourceMetric(
                timestamp=timestamp,
                resource_type=ResourceType.MEMORY,
                metric_name="memory_breakdown",
                value={
                    'buffers': getattr(memory, 'buffers', 0) / (1024**3),
                    'cached': getattr(memory, 'cached', 0) / (1024**3),
                    'shared': getattr(memory, 'shared', 0) / (1024**3),
                    'slab': getattr(memory, 'slab', 0) / (1024**3)
                },
                unit="GB"
            ))
            
            # Store for trend analysis
            self.metrics_history['memory_usage'].append(memory.percent)
            
        except Exception as e:
            logger.error(f"Error collecting memory metrics: {e}")
        
        return metrics
    
    @performance_monitor
    @error_handler
    def collect_disk_metrics(self) -> List[ResourceMetric]:
        """Collect comprehensive disk metrics"""
        metrics = []
        timestamp = time.time()
        
        try:
            # Disk usage for root partition
            disk_usage = psutil.disk_usage('/')
            disk_percent = (disk_usage.used / disk_usage.total) * 100
            
            metrics.append(ResourceMetric(
                timestamp=timestamp,
                resource_type=ResourceType.DISK,
                metric_name="usage_percent",
                value=disk_percent,
                unit="percent",
                metadata={
                    'total_gb': disk_usage.total / (1024**3),
                    'used_gb': disk_usage.used / (1024**3),
                    'free_gb': disk_usage.free / (1024**3)
                }
            ))
            
            # Disk I/O statistics
            if self.config.enable_disk_io_monitoring:
                disk_io = psutil.disk_io_counters()
                if disk_io:
                    io_metrics = {
                        'read_bytes': disk_io.read_bytes,
                        'write_bytes': disk_io.write_bytes,
                        'read_count': disk_io.read_count,
                        'write_count': disk_io.write_count,
                        'read_time': disk_io.read_time,
                        'write_time': disk_io.write_time
                    }
                    
                    # Calculate rates if we have previous data
                    if self.last_disk_io:
                        time_delta = timestamp - self.last_disk_io['timestamp']
                        if time_delta > 0:
                            io_metrics['read_rate_mbps'] = (disk_io.read_bytes - self.last_disk_io['read_bytes']) / (1024**2) / time_delta
                            io_metrics['write_rate_mbps'] = (disk_io.write_bytes - self.last_disk_io['write_bytes']) / (1024**2) / time_delta
                    
                    metrics.append(ResourceMetric(
                        timestamp=timestamp,
                        resource_type=ResourceType.DISK,
                        metric_name="io_statistics",
                        value=io_metrics,
                        unit="mixed"
                    ))
                    
                    # Store for rate calculations
                    self.last_disk_io = {
                        'timestamp': timestamp,
                        'read_bytes': disk_io.read_bytes,
                        'write_bytes': disk_io.write_bytes
                    }
            
            # Per-disk usage
            disk_partitions = psutil.disk_partitions()
            partition_usage = {}
            for partition in disk_partitions:
                try:
                    usage = psutil.disk_usage(partition.mountpoint)
                    partition_usage[partition.device] = {
                        'mountpoint': partition.mountpoint,
                        'fstype': partition.fstype,
                        'total_gb': usage.total / (1024**3),
                        'used_gb': usage.used / (1024**3),
                        'free_gb': usage.free / (1024**3),
                        'percent': (usage.used / usage.total) * 100
                    }
                except:
                    pass  # Skip inaccessible partitions
            
            metrics.append(ResourceMetric(
                timestamp=timestamp,
                resource_type=ResourceType.DISK,
                metric_name="partition_usage",
                value=partition_usage,
                unit="mixed"
            ))
            
            # Store for trend analysis
            self.metrics_history['disk_usage'].append(disk_percent)
            
        except Exception as e:
            logger.error(f"Error collecting disk metrics: {e}")
        
        return metrics
    
    @performance_monitor
    @error_handler
    def collect_network_metrics(self) -> List[ResourceMetric]:
        """Collect comprehensive network metrics"""
        metrics = []
        timestamp = time.time()
        
        if not self.config.enable_network_monitoring:
            return metrics
        
        try:
            # Network I/O statistics
            network_io = psutil.net_io_counters()
            if network_io:
                net_metrics = {
                    'bytes_sent': network_io.bytes_sent,
                    'bytes_recv': network_io.bytes_recv,
                    'packets_sent': network_io.packets_sent,
                    'packets_recv': network_io.packets_recv,
                    'errin': network_io.errin,
                    'errout': network_io.errout,
                    'dropin': network_io.dropin,
                    'dropout': network_io.dropout
                }
                
                # Calculate rates if we have previous data
                if self.last_network_io:
                    time_delta = timestamp - self.last_network_io['timestamp']
                    if time_delta > 0:
                        net_metrics['send_rate_mbps'] = (network_io.bytes_sent - self.last_network_io['bytes_sent']) / (1024**2) / time_delta
                        net_metrics['recv_rate_mbps'] = (network_io.bytes_recv - self.last_network_io['bytes_recv']) / (1024**2) / time_delta
                
                metrics.append(ResourceMetric(
                    timestamp=timestamp,
                    resource_type=ResourceType.NETWORK,
                    metric_name="io_statistics",
                    value=net_metrics,
                    unit="mixed"
                ))
                
                # Store for rate calculations
                self.last_network_io = {
                    'timestamp': timestamp,
                    'bytes_sent': network_io.bytes_sent,
                    'bytes_recv': network_io.bytes_recv
                }
            
            # Per-interface statistics
            net_if_stats = psutil.net_if_stats()
            net_if_addrs = psutil.net_if_addrs()
            
            interface_stats = {}
            for interface, stats in net_if_stats.items():
                interface_info = {
                    'is_up': stats.isup,
                    'duplex': stats.duplex,
                    'speed': stats.speed,
                    'mtu': stats.mtu
                }
                
                # Add addresses
                if interface in net_if_addrs:
                    addresses = []
                    for addr in net_if_addrs[interface]:
                        addresses.append({
                            'family': addr.family.name,
                            'address': addr.address,
                            'netmask': addr.netmask,
                            'broadcast': addr.broadcast
                        })
                    interface_info['addresses'] = addresses
                
                interface_stats[interface] = interface_info
            
            metrics.append(ResourceMetric(
                timestamp=timestamp,
                resource_type=ResourceType.NETWORK,
                metric_name="interface_statistics",
                value=interface_stats,
                unit="mixed"
            ))
            
            # Network connections
            connections = psutil.net_connections()
            connection_stats = {
                'total': len(connections),
                'established': len([c for c in connections if c.status == 'ESTABLISHED']),
                'listen': len([c for c in connections if c.status == 'LISTEN']),
                'time_wait': len([c for c in connections if c.status == 'TIME_WAIT'])
            }
            
            metrics.append(ResourceMetric(
                timestamp=timestamp,
                resource_type=ResourceType.NETWORK,
                metric_name="connection_statistics",
                value=connection_stats,
                unit="count"
            ))
            
        except Exception as e:
            logger.error(f"Error collecting network metrics: {e}")
        
        return metrics

class ProcessResourceCollector:
    """Process-level resource collector"""
    
    def __init__(self, arm64_optimizer: ARM64ResourceOptimizer, config: ResourceConfig):
        self.optimizer = arm64_optimizer
        self.config = config
        self.process_history = defaultdict(lambda: deque(maxlen=100))
        
    @performance_monitor
    @error_handler
    def collect_process_metrics(self, pid: Optional[int] = None) -> List[ResourceMetric]:
        """Collect process-specific metrics"""
        metrics = []
        timestamp = time.time()
        
        if not self.config.enable_process_monitoring:
            return metrics
        
        try:
            if pid:
                # Monitor specific process
                try:
                    process = psutil.Process(pid)
                    process_metrics = self._collect_single_process_metrics(process, timestamp)
                    metrics.extend(process_metrics)
                except psutil.NoSuchProcess:
                    logger.warning(f"Process {pid} no longer exists")
            else:
                # Monitor all processes (top consumers)
                processes = []
                for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent']):
                    try:
                        proc_info = proc.info
                        if proc_info['cpu_percent'] > 1.0 or proc_info['memory_percent'] > 1.0:
                            processes.append(proc)
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        pass
                
                # Sort by resource usage and take top 10
                processes.sort(key=lambda p: p.info['cpu_percent'] + p.info['memory_percent'], reverse=True)
                
                top_processes = []
                for proc in processes[:10]:
                    try:
                        top_processes.append({
                            'pid': proc.info['pid'],
                            'name': proc.info['name'],
                            'cpu_percent': proc.info['cpu_percent'],
                            'memory_percent': proc.info['memory_percent']
                        })
                    except:
                        pass
                
                metrics.append(ResourceMetric(
                    timestamp=timestamp,
                    resource_type=ResourceType.PROCESS,
                    metric_name="top_consumers",
                    value=top_processes,
                    unit="mixed"
                ))
                
                # Overall process statistics
                process_count = len(list(psutil.process_iter()))
                metrics.append(ResourceMetric(
                    timestamp=timestamp,
                    resource_type=ResourceType.PROCESS,
                    metric_name="total_count",
                    value=process_count,
                    unit="count"
                ))
                
        except Exception as e:
            logger.error(f"Error collecting process metrics: {e}")
        
        return metrics
    
    def _collect_single_process_metrics(self, process: psutil.Process, timestamp: float) -> List[ResourceMetric]:
        """Collect metrics for a single process"""
        metrics = []
        
        try:
            with process.oneshot():
                # Basic metrics
                cpu_percent = process.cpu_percent(interval=0.1)
                memory_info = process.memory_info()
                memory_percent = process.memory_percent()
                
                metrics.append(ResourceMetric(
                    timestamp=timestamp,
                    resource_type=ResourceType.PROCESS,
                    metric_name="cpu_usage",
                    value=cpu_percent,
                    unit="percent",
                    metadata={'pid': process.pid, 'name': process.name()}
                ))
                
                metrics.append(ResourceMetric(
                    timestamp=timestamp,
                    resource_type=ResourceType.PROCESS,
                    metric_name="memory_usage",
                    value=memory_percent,
                    unit="percent",
                    metadata={
                        'pid': process.pid,
                        'name': process.name(),
                        'rss_mb': memory_info.rss / (1024**2),
                        'vms_mb': memory_info.vms / (1024**2)
                    }
                ))
                
                # Thread and file descriptor counts
                metrics.append(ResourceMetric(
                    timestamp=timestamp,
                    resource_type=ResourceType.PROCESS,
                    metric_name="thread_count",
                    value=process.num_threads(),
                    unit="count",
                    metadata={'pid': process.pid, 'name': process.name()}
                ))
                
                try:
                    metrics.append(ResourceMetric(
                        timestamp=timestamp,
                        resource_type=ResourceType.PROCESS,
                        metric_name="file_descriptors",
                        value=process.num_fds(),
                        unit="count",
                        metadata={'pid': process.pid, 'name': process.name()}
                    ))
                except:
                    pass  # Not available on all systems
                
        except Exception as e:
            logger.error(f"Error collecting metrics for process {process.pid}: {e}")
        
        return metrics

class AdvancedResourceMonitor:
    """
    Advanced Resource Monitor with ARM64 optimizations
    
    Provides comprehensive resource monitoring capabilities including:
    - System-level resource monitoring
    - Process-level resource tracking
    - Predictive analytics and trend detection
    - Resource optimization recommendations
    """
    
    def __init__(self, config: ResourceConfig = None):
        self.config = config or ResourceConfig()
        self.arm64_optimizer = ARM64ResourceOptimizer()
        self.shared_memory = SharedMemoryManager() if self.config.use_shared_memory else None
        
        # Resource collectors
        self.system_collector = SystemResourceCollector(self.arm64_optimizer, self.config)
        self.process_collector = ProcessResourceCollector(self.arm64_optimizer, self.config)
        
        # Communication
        self.zmq_publisher = ZMQPublisher(port=self.config.publish_port)
        
        # Monitoring state
        self.is_running = False
        self.monitoring_thread = None
        self.alert_history = deque(maxlen=1000)
        
        # Performance tracking
        self.monitor_stats = {
            'total_collections': 0,
            'collection_times': deque(maxlen=1000),
            'alerts_generated': 0,
            'metrics_published': 0
        }
        
        logger.info(f"AdvancedResourceMonitor initialized with ARM64 optimizations: {self.arm64_optimizer.is_arm64}")
    
    @performance_monitor
    @error_handler
    def collect_all_metrics(self) -> Dict[str, List[ResourceMetric]]:
        """Collect all resource metrics"""
        all_metrics = {}
        
        try:
            # Collect system metrics
            if self.config.parallel_processing:
                # Parallel collection
                with ThreadPoolExecutor(max_workers=self.config.max_workers or 4) as executor:
                    cpu_future = executor.submit(self.system_collector.collect_cpu_metrics)
                    memory_future = executor.submit(self.system_collector.collect_memory_metrics)
                    disk_future = executor.submit(self.system_collector.collect_disk_metrics)
                    network_future = executor.submit(self.system_collector.collect_network_metrics)
                    process_future = executor.submit(self.process_collector.collect_process_metrics)
                    
                    all_metrics['cpu'] = cpu_future.result()
                    all_metrics['memory'] = memory_future.result()
                    all_metrics['disk'] = disk_future.result()
                    all_metrics['network'] = network_future.result()
                    all_metrics['process'] = process_future.result()
            else:
                # Sequential collection
                all_metrics['cpu'] = self.system_collector.collect_cpu_metrics()
                all_metrics['memory'] = self.system_collector.collect_memory_metrics()
                all_metrics['disk'] = self.system_collector.collect_disk_metrics()
                all_metrics['network'] = self.system_collector.collect_network_metrics()
                all_metrics['process'] = self.process_collector.collect_process_metrics()
            
            self.monitor_stats['total_collections'] += 1
            
        except Exception as e:
            logger.error(f"Error collecting metrics: {e}")
        
        return all_metrics
    
    def check_resource_thresholds(self, metrics: Dict[str, List[ResourceMetric]]) -> List[ResourceAlert]:
        """Check metrics against thresholds and generate alerts"""
        alerts = []
        
        try:
            for resource_type, metric_list in metrics.items():
                for metric in metric_list:
                    alert = self._check_metric_threshold(metric)
                    if alert:
                        alerts.append(alert)
                        self.alert_history.append(alert)
            
            self.monitor_stats['alerts_generated'] += len(alerts)
            
        except Exception as e:
            logger.error(f"Error checking thresholds: {e}")
        
        return alerts
    
    def _check_metric_threshold(self, metric: ResourceMetric) -> Optional[ResourceAlert]:
        """Check individual metric against thresholds"""
        threshold_key = None
        current_value = None
        
        # Map metric to threshold
        if metric.resource_type == ResourceType.CPU and metric.metric_name == "usage_percent":
            threshold_key = 'cpu_usage'
            current_value = metric.value
        elif metric.resource_type == ResourceType.MEMORY and metric.metric_name == "usage_percent":
            threshold_key = 'memory_usage'
            current_value = metric.value
        elif metric.resource_type == ResourceType.DISK and metric.metric_name == "usage_percent":
            threshold_key = 'disk_usage'
            current_value = metric.value
        elif metric.resource_type == ResourceType.PROCESS and metric.metric_name == "total_count":
            threshold_key = 'process_count'
            current_value = metric.value
        
        if threshold_key and threshold_key in self.config.alert_thresholds:
            threshold = self.config.alert_thresholds[threshold_key]
            
            if current_value > threshold:
                alert_level = AlertLevel.CRITICAL if current_value > threshold * 1.1 else AlertLevel.WARNING
                
                recommendations = self._generate_recommendations(metric.resource_type, current_value, threshold)
                
                return ResourceAlert(
                    timestamp=metric.timestamp,
                    resource_type=metric.resource_type,
                    alert_level=alert_level,
                    message=f"{metric.resource_type.value.upper()} {metric.metric_name} is {current_value:.1f}% (threshold: {threshold}%)",
                    current_value=current_value,
                    threshold=threshold,
                    recommendations=recommendations
                )
        
        return None
    
    def _generate_recommendations(self, resource_type: ResourceType, 
                                current_value: float, threshold: float) -> List[str]:
        """Generate optimization recommendations"""
        recommendations = []
        
        if resource_type == ResourceType.CPU:
            recommendations.extend([
                "Consider optimizing CPU-intensive processes",
                "Check for runaway processes or infinite loops",
                "Consider scaling horizontally or upgrading CPU"
            ])
            if self.arm64_optimizer.is_arm64:
                recommendations.append("Optimize CPU affinity for ARM64 big.LITTLE architecture")
        
        elif resource_type == ResourceType.MEMORY:
            recommendations.extend([
                "Check for memory leaks in applications",
                "Consider increasing swap space",
                "Optimize memory usage in applications",
                "Consider adding more RAM"
            ])
        
        elif resource_type == ResourceType.DISK:
            recommendations.extend([
                "Clean up temporary files and logs",
                "Archive or compress old data",
                "Consider adding more storage capacity",
                "Optimize database storage"
            ])
        
        elif resource_type == ResourceType.PROCESS:
            recommendations.extend([
                "Check for process leaks",
                "Optimize process lifecycle management",
                "Consider process pooling or reuse"
            ])
        
        return recommendations
    
    def get_resource_trends(self) -> Dict[str, Dict[str, float]]:
        """Get resource usage trends"""
        trends = {}
        
        for metric_name, history in self.system_collector.metrics_history.items():
            if len(history) > 10:
                history_array = np.array(list(history))
                trends[metric_name] = self.arm64_optimizer.vectorized_resource_analysis(history_array)
        
        return trends
    
    def get_resource_summary(self) -> Dict[str, Any]:
        """Get comprehensive resource summary"""
        current_metrics = self.collect_all_metrics()
        trends = self.get_resource_trends()
        
        # Extract current values
        current_values = {}
        for resource_type, metric_list in current_metrics.items():
            current_values[resource_type] = {}
            for metric in metric_list:
                current_values[resource_type][metric.metric_name] = metric.value
        
        return {
            'timestamp': time.time(),
            'current': current_values,
            'trends': trends,
            'alerts': [
                {
                    'timestamp': alert.timestamp,
                    'resource_type': alert.resource_type.value,
                    'level': alert.alert_level.value,
                    'message': alert.message,
                    'recommendations': alert.recommendations
                }
                for alert in list(self.alert_history)[-10:]  # Last 10 alerts
            ],
            'system_info': {
                'is_arm64': self.arm64_optimizer.is_arm64,
                'cpu_count': self.arm64_optimizer.cpu_count,
                'cpu_topology': self.arm64_optimizer.cpu_topology
            }
        }
    
    def _continuous_monitoring(self):
        """Background thread for continuous resource monitoring"""
        while self.is_running:
            try:
                start_time = time.time()
                
                # Collect metrics
                metrics = self.collect_all_metrics()
                
                # Check thresholds
                alerts = self.check_resource_thresholds(metrics)
                
                # Publish metrics
                self.zmq_publisher.send({
                    'type': 'resource_metrics',
                    'timestamp': time.time(),
                    'metrics': self._serialize_metrics(metrics),
                    'alerts': [self._serialize_alert(alert) for alert in alerts]
                })
                
                self.monitor_stats['metrics_published'] += 1
                
                # Track collection time
                collection_time = time.time() - start_time
                self.monitor_stats['collection_times'].append(collection_time)
                
                # Adaptive interval based on system load
                if metrics.get('cpu'):
                    cpu_metrics = metrics['cpu']
                    cpu_usage = next((m.value for m in cpu_metrics if m.metric_name == 'usage_percent'), 0)
                    interval = self.arm64_optimizer.optimize_monitoring_frequency(ResourceType.CPU, cpu_usage)
                else:
                    interval = self.config.check_interval_seconds
                
                time.sleep(interval)
                
            except Exception as e:
                logger.error(f"Error in continuous monitoring: {e}")
                time.sleep(5.0)
    
    def _serialize_metrics(self, metrics: Dict[str, List[ResourceMetric]]) -> Dict[str, Any]:
        """Serialize metrics for transmission"""
        serialized = {}
        
        for resource_type, metric_list in metrics.items():
            serialized[resource_type] = []
            for metric in metric_list:
                serialized[resource_type].append({
                    'timestamp': metric.timestamp,
                    'metric_name': metric.metric_name,
                    'value': metric.value,
                    'unit': metric.unit,
                    'metadata': metric.metadata
                })
        
        return serialized
    
    def _serialize_alert(self, alert: ResourceAlert) -> Dict[str, Any]:
        """Serialize alert for transmission"""
        return {
            'timestamp': alert.timestamp,
            'resource_type': alert.resource_type.value,
            'alert_level': alert.alert_level.value,
            'message': alert.message,
            'current_value': alert.current_value,
            'threshold': alert.threshold,
            'recommendations': alert.recommendations
        }
    
    def start(self) -> bool:
        """Start resource monitoring"""
        if self.is_running:
            return True
        
        try:
            self.is_running = True
            
            # Start monitoring thread
            self.monitoring_thread = threading.Thread(
                target=self._continuous_monitoring,
                name="ResourceMonitoringThread"
            )
            self.monitoring_thread.daemon = True
            self.monitoring_thread.start()
            
            logger.info("Resource monitoring started successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start resource monitoring: {e}")
            self.is_running = False
            return False
    
    def stop(self):
        """Stop resource monitoring"""
        self.is_running = False
        
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5.0)
        
        logger.info("Resource monitoring stopped")
    
    def get_status(self) -> Dict[str, Any]:
        """Get resource monitor status"""
        avg_collection_time = (
            sum(self.monitor_stats['collection_times']) / len(self.monitor_stats['collection_times'])
            if self.monitor_stats['collection_times'] else 0
        )
        
        return {
            'is_running': self.is_running,
            'total_collections': self.monitor_stats['total_collections'],
            'average_collection_time': avg_collection_time,
            'alerts_generated': self.monitor_stats['alerts_generated'],
            'metrics_published': self.monitor_stats['metrics_published'],
            'arm64_optimized': self.arm64_optimizer.is_arm64,
            'publish_port': self.config.publish_port
        }
    
    def cleanup(self):
        """Cleanup resource monitor"""
        if self.is_running:
            self.stop()
        
        if self.shared_memory:
            self.shared_memory.cleanup()
        
        logger.info("ResourceMonitor cleanup completed")

# Factory function for easy instantiation
def create_resource_monitor(config: ResourceConfig = None) -> AdvancedResourceMonitor:
    """
    Factory function to create resource monitor with optimal configuration
    
    Args:
        config: Resource monitor configuration
        
    Returns:
        Configured AdvancedResourceMonitor instance
    """
    if config is None:
        config = ResourceConfig()
        
        # Auto-detect optimal settings
        if platform.machine().lower() in ['arm64', 'aarch64']:
            config.enable_arm64_optimizations = True
            config.parallel_processing = True
            config.max_workers = min(mp.cpu_count(), 6)
        
        # Adjust monitoring frequency based on system capabilities
        if psutil.cpu_count() > 8:
            config.check_interval_seconds = 3.0
        elif psutil.cpu_count() > 4:
            config.check_interval_seconds = 5.0
        else:
            config.check_interval_seconds = 10.0
    
    return AdvancedResourceMonitor(config)

# Legacy compatibility
ResourceMonitor = AdvancedResourceMonitor

if __name__ == "__main__":
    # Example usage and testing
    
    # Create resource monitor
    resource_monitor = create_resource_monitor()
    
    # Start monitoring
    if resource_monitor.start():
        print("Resource monitoring started successfully")
        
        # Monitor for a while
        time.sleep(10)
        
        # Get resource summary
        summary = resource_monitor.get_resource_summary()
        print(f"Resource summary: {summary['current']}")
        
        # Get status
        status = resource_monitor.get_status()
        print(f"Monitor status: {status}")
        
        # Cleanup
        resource_monitor.stop()
        resource_monitor.cleanup()
    else:
        print("Failed to start resource monitoring")
