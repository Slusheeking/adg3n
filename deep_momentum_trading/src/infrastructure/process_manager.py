"""
Enhanced Process Manager with ARM64 Optimizations

This module provides comprehensive process management capabilities for the Deep Momentum Trading System
with ARM64-specific optimizations for high-performance process orchestration and monitoring.

Features:
- Advanced process lifecycle management with ARM64 optimizations
- Intelligent process monitoring and auto-recovery
- CPU affinity optimization for ARM64 architectures
- Process resource management and throttling
- Distributed process coordination
- Performance monitoring and optimization
- Fault tolerance and resilience mechanisms
"""

import multiprocessing as mp
import threading
import time
import signal
import os
import platform
import psutil
from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import logging
from functools import wraps
import queue
import subprocess
import json
from pathlib import Path
import yaml

from ..utils.logger import get_logger
from ..utils.shared_memory import SharedMemoryManager
from ..utils.decorators import performance_monitor, error_handler
from ..utils.exceptions import ValidationError, ProcessError

logger = get_logger(__name__)

class ProcessState(Enum):
    """Process states"""
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    STOPPED = "stopped"
    FAILED = "failed"
    UNKNOWN = "unknown"

class ProcessPriority(Enum):
    """Process priority levels"""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class ProcessConfig:
    """Configuration for process management"""
    enable_arm64_optimizations: bool = True
    use_shared_memory: bool = True
    enable_auto_recovery: bool = True
    enable_resource_monitoring: bool = True
    parallel_processing: bool = True
    max_workers: Optional[int] = None
    monitoring_interval: float = 5.0
    restart_delay: float = 2.0
    max_restart_attempts: int = 3
    cpu_affinity_optimization: bool = True
    memory_limit_mb: Optional[int] = None
    cpu_limit_percent: Optional[float] = None
    process_timeout: float = 30.0

@dataclass
class ProcessDefinition:
    """Process definition"""
    name: str
    target: Union[Callable, str]  # Function or command
    args: tuple = field(default_factory=tuple)
    kwargs: Dict[str, Any] = field(default_factory=dict)
    priority: ProcessPriority = ProcessPriority.NORMAL
    auto_restart: bool = True
    cpu_affinity: Optional[List[int]] = None
    memory_limit_mb: Optional[int] = None
    environment: Optional[Dict[str, str]] = None
    working_directory: Optional[str] = None
    dependencies: List[str] = field(default_factory=list)

@dataclass
class ProcessInfo:
    """Process information"""
    definition: ProcessDefinition
    process: Optional[mp.Process] = None
    subprocess: Optional[subprocess.Popen] = None
    state: ProcessState = ProcessState.STOPPED
    pid: Optional[int] = None
    start_time: Optional[float] = None
    restart_count: int = 0
    last_restart: Optional[float] = None
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    error_count: int = 0
    last_error: Optional[str] = None

class ARM64ProcessOptimizer:
    """ARM64-specific optimizations for process management"""
    
    def __init__(self):
        self.is_arm64 = platform.machine().lower() in ['arm64', 'aarch64']
        self.cpu_count = mp.cpu_count()
        self.cpu_topology = self._detect_cpu_topology()
        
    def _detect_cpu_topology(self) -> Dict[str, Any]:
        """Detect ARM64 CPU topology"""
        topology = {
            'total_cores': self.cpu_count,
            'performance_cores': [],
            'efficiency_cores': [],
            'numa_nodes': []
        }
        
        if not self.is_arm64:
            return topology
        
        try:
            # Try to detect big.LITTLE architecture
            with open('/proc/cpuinfo', 'r') as f:
                cpuinfo = f.read()
                
            # Parse CPU information for ARM64
            cores_info = []
            current_core = {}
            
            for line in cpuinfo.split('\n'):
                if line.startswith('processor'):
                    if current_core:
                        cores_info.append(current_core)
                    current_core = {'id': int(line.split(':')[1].strip())}
                elif line.startswith('CPU implementer'):
                    current_core['implementer'] = line.split(':')[1].strip()
                elif line.startswith('CPU part'):
                    current_core['part'] = line.split(':')[1].strip()
                elif line.startswith('CPU variant'):
                    current_core['variant'] = line.split(':')[1].strip()
            
            if current_core:
                cores_info.append(current_core)
            
            # Classify cores (simplified heuristic)
            for core in cores_info:
                core_id = core['id']
                # This is a simplified classification - real implementation would be more sophisticated
                if core_id < self.cpu_count // 2:
                    topology['efficiency_cores'].append(core_id)
                else:
                    topology['performance_cores'].append(core_id)
                    
        except Exception as e:
            logger.warning(f"Could not detect ARM64 CPU topology: {e}")
            # Fallback: assume all cores are equivalent
            topology['performance_cores'] = list(range(self.cpu_count))
        
        return topology
    
    def optimize_cpu_affinity(self, process_priority: ProcessPriority) -> Optional[List[int]]:
        """Optimize CPU affinity based on ARM64 topology and process priority"""
        if not self.is_arm64:
            return None
        
        topology = self.cpu_topology
        
        if process_priority == ProcessPriority.CRITICAL:
            # Use performance cores for critical processes
            return topology['performance_cores'] if topology['performance_cores'] else None
        elif process_priority == ProcessPriority.HIGH:
            # Use mix of performance and efficiency cores
            cores = topology['performance_cores'] + topology['efficiency_cores'][:2]
            return cores if cores else None
        elif process_priority == ProcessPriority.LOW:
            # Use efficiency cores for low priority processes
            return topology['efficiency_cores'] if topology['efficiency_cores'] else None
        else:
            # Normal priority - use all cores
            return None
    
    def optimize_process_scheduling(self, process: mp.Process, priority: ProcessPriority):
        """Optimize process scheduling for ARM64"""
        if not self.is_arm64 or not process.pid:
            return
        
        try:
            # Set CPU affinity
            affinity = self.optimize_cpu_affinity(priority)
            if affinity:
                psutil.Process(process.pid).cpu_affinity(affinity)
                logger.debug(f"Set CPU affinity for process {process.pid}: {affinity}")
            
            # Set process priority
            priority_map = {
                ProcessPriority.LOW: psutil.BELOW_NORMAL_PRIORITY_CLASS if hasattr(psutil, 'BELOW_NORMAL_PRIORITY_CLASS') else 10,
                ProcessPriority.NORMAL: psutil.NORMAL_PRIORITY_CLASS if hasattr(psutil, 'NORMAL_PRIORITY_CLASS') else 0,
                ProcessPriority.HIGH: psutil.ABOVE_NORMAL_PRIORITY_CLASS if hasattr(psutil, 'ABOVE_NORMAL_PRIORITY_CLASS') else -10,
                ProcessPriority.CRITICAL: psutil.HIGH_PRIORITY_CLASS if hasattr(psutil, 'HIGH_PRIORITY_CLASS') else -20
            }
            
            if hasattr(psutil.Process(process.pid), 'nice'):
                psutil.Process(process.pid).nice(priority_map.get(priority, 0))
                
        except Exception as e:
            logger.warning(f"Could not optimize process scheduling: {e}")

class ProcessMonitor:
    """Process monitoring and resource tracking"""
    
    def __init__(self, config: ProcessConfig):
        self.config = config
        self.monitoring_data = {}
        self.alert_thresholds = {
            'cpu_usage': 80.0,
            'memory_usage': 85.0,
            'error_rate': 5.0
        }
    
    def monitor_process(self, process_info: ProcessInfo) -> Dict[str, Any]:
        """Monitor process resources and health"""
        if not process_info.pid:
            return {}
        
        try:
            process = psutil.Process(process_info.pid)
            
            # Collect metrics
            cpu_percent = process.cpu_percent(interval=0.1)
            memory_info = process.memory_info()
            memory_percent = process.memory_percent()
            
            metrics = {
                'timestamp': time.time(),
                'cpu_percent': cpu_percent,
                'memory_rss_mb': memory_info.rss / (1024**2),
                'memory_vms_mb': memory_info.vms / (1024**2),
                'memory_percent': memory_percent,
                'num_threads': process.num_threads(),
                'status': process.status(),
                'create_time': process.create_time()
            }
            
            # Update process info
            process_info.cpu_usage = cpu_percent
            process_info.memory_usage = memory_percent
            
            # Store monitoring data
            if process_info.name not in self.monitoring_data:
                self.monitoring_data[process_info.name] = []
            
            self.monitoring_data[process_info.name].append(metrics)
            
            # Keep only recent data
            if len(self.monitoring_data[process_info.name]) > 1000:
                self.monitoring_data[process_info.name] = self.monitoring_data[process_info.name][-500:]
            
            return metrics
            
        except psutil.NoSuchProcess:
            logger.warning(f"Process {process_info.pid} no longer exists")
            return {}
        except Exception as e:
            logger.error(f"Error monitoring process {process_info.name}: {e}")
            return {}
    
    def check_resource_limits(self, process_info: ProcessInfo) -> List[str]:
        """Check if process exceeds resource limits"""
        violations = []
        
        if process_info.definition.memory_limit_mb:
            if process_info.memory_usage > process_info.definition.memory_limit_mb:
                violations.append(f"Memory usage ({process_info.memory_usage:.1f}MB) exceeds limit ({process_info.definition.memory_limit_mb}MB)")
        
        if self.config.cpu_limit_percent:
            if process_info.cpu_usage > self.config.cpu_limit_percent:
                violations.append(f"CPU usage ({process_info.cpu_usage:.1f}%) exceeds limit ({self.config.cpu_limit_percent}%)")
        
        return violations
    
    def get_process_statistics(self, process_name: str) -> Dict[str, Any]:
        """Get statistical analysis of process performance"""
        if process_name not in self.monitoring_data:
            return {}
        
        data = self.monitoring_data[process_name]
        if not data:
            return {}
        
        # Extract metrics
        cpu_values = [d['cpu_percent'] for d in data]
        memory_values = [d['memory_percent'] for d in data]
        
        import numpy as np
        
        return {
            'cpu_stats': {
                'mean': float(np.mean(cpu_values)),
                'std': float(np.std(cpu_values)),
                'min': float(np.min(cpu_values)),
                'max': float(np.max(cpu_values)),
                'p95': float(np.percentile(cpu_values, 95))
            },
            'memory_stats': {
                'mean': float(np.mean(memory_values)),
                'std': float(np.std(memory_values)),
                'min': float(np.min(memory_values)),
                'max': float(np.max(memory_values)),
                'p95': float(np.percentile(memory_values, 95))
            },
            'sample_count': len(data),
            'time_range': data[-1]['timestamp'] - data[0]['timestamp'] if len(data) > 1 else 0
        }

class AdvancedProcessManager:
    """
    Advanced Process Manager with ARM64 optimizations
    
    Provides comprehensive process management capabilities including:
    - Process lifecycle management
    - Resource monitoring and optimization
    - Auto-recovery and fault tolerance
    - ARM64-specific optimizations
    """
    
    def __init__(self, config: ProcessConfig = None):
        self.config = config or ProcessConfig()
        self.arm64_optimizer = ARM64ProcessOptimizer()
        self.shared_memory = SharedMemoryManager() if self.config.use_shared_memory else None
        
        # Process management
        self.processes: Dict[str, ProcessInfo] = {}
        self.process_monitor = ProcessMonitor(self.config)
        self.dependency_graph = {}
        
        # Threading
        self.is_running = False
        self.monitoring_thread = None
        self.recovery_thread = None
        
        # Performance tracking
        self.manager_stats = {
            'total_processes': 0,
            'active_processes': 0,
            'restart_count': 0,
            'error_count': 0,
            'monitoring_cycles': 0
        }
        
        # Setup signal handlers
        self._setup_signal_handlers()
        
        logger.info(f"AdvancedProcessManager initialized with ARM64 optimizations: {self.arm64_optimizer.is_arm64}")
    
    def _setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown"""
        def signal_handler(signum, frame):
            logger.info(f"Received signal {signum}, shutting down processes...")
            self.stop_all_processes()
        
        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)
    
    def register_process(self, definition: ProcessDefinition):
        """Register process definition"""
        if definition.name in self.processes:
            raise ValidationError(f"Process {definition.name} already registered")
        
        process_info = ProcessInfo(definition=definition)
        self.processes[definition.name] = process_info
        
        # Build dependency graph
        self.dependency_graph[definition.name] = definition.dependencies
        
        logger.info(f"Registered process: {definition.name}")
    
    def unregister_process(self, name: str):
        """Unregister process"""
        if name in self.processes:
            # Stop process if running
            self.stop_process(name)
            del self.processes[name]
            
            # Remove from dependency graph
            if name in self.dependency_graph:
                del self.dependency_graph[name]
            
            logger.info(f"Unregistered process: {name}")
    
    @performance_monitor
    @error_handler
    def start_process(self, name: str, wait_for_dependencies: bool = True) -> bool:
        """Start process with dependency resolution"""
        if name not in self.processes:
            raise ValidationError(f"Process {name} not registered")
        
        process_info = self.processes[name]
        
        if process_info.state == ProcessState.RUNNING:
            logger.warning(f"Process {name} is already running")
            return True
        
        # Check dependencies
        if wait_for_dependencies:
            for dep in process_info.definition.dependencies:
                if dep not in self.processes:
                    logger.error(f"Dependency {dep} not found for process {name}")
                    return False
                
                if self.processes[dep].state != ProcessState.RUNNING:
                    logger.info(f"Starting dependency {dep} for process {name}")
                    if not self.start_process(dep, wait_for_dependencies=True):
                        logger.error(f"Failed to start dependency {dep}")
                        return False
        
        try:
            process_info.state = ProcessState.STARTING
            
            # Determine if it's a function or command
            if callable(process_info.definition.target):
                # Start as multiprocessing.Process
                process = mp.Process(
                    target=process_info.definition.target,
                    args=process_info.definition.args,
                    kwargs=process_info.definition.kwargs,
                    name=name
                )
                process.start()
                process_info.process = process
                process_info.pid = process.pid
                
                # Apply ARM64 optimizations
                if self.config.cpu_affinity_optimization:
                    self.arm64_optimizer.optimize_process_scheduling(
                        process, process_info.definition.priority
                    )
                
            else:
                # Start as subprocess
                env = os.environ.copy()
                if process_info.definition.environment:
                    env.update(process_info.definition.environment)
                
                cmd = process_info.definition.target
                if process_info.definition.args:
                    cmd = f"{cmd} {' '.join(map(str, process_info.definition.args))}"
                
                subprocess_proc = subprocess.Popen(
                    cmd,
                    shell=True,
                    env=env,
                    cwd=process_info.definition.working_directory,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )
                
                process_info.subprocess = subprocess_proc
                process_info.pid = subprocess_proc.pid
            
            process_info.state = ProcessState.RUNNING
            process_info.start_time = time.time()
            
            self.manager_stats['total_processes'] += 1
            self.manager_stats['active_processes'] += 1
            
            logger.info(f"Started process {name} with PID {process_info.pid}")
            return True
            
        except Exception as e:
            process_info.state = ProcessState.FAILED
            process_info.last_error = str(e)
            process_info.error_count += 1
            self.manager_stats['error_count'] += 1
            logger.error(f"Failed to start process {name}: {e}")
            return False
    
    @performance_monitor
    @error_handler
    def stop_process(self, name: str, timeout: float = None) -> bool:
        """Stop process gracefully"""
        if name not in self.processes:
            logger.warning(f"Process {name} not found")
            return False
        
        process_info = self.processes[name]
        
        if process_info.state != ProcessState.RUNNING:
            logger.warning(f"Process {name} is not running")
            return True
        
        timeout = timeout or self.config.process_timeout
        
        try:
            process_info.state = ProcessState.STOPPING
            
            if process_info.process:
                # Multiprocessing.Process
                process_info.process.terminate()
                process_info.process.join(timeout=timeout)
                
                if process_info.process.is_alive():
                    logger.warning(f"Process {name} did not terminate gracefully, killing...")
                    process_info.process.kill()
                    process_info.process.join()
                
            elif process_info.subprocess:
                # Subprocess
                process_info.subprocess.terminate()
                try:
                    process_info.subprocess.wait(timeout=timeout)
                except subprocess.TimeoutExpired:
                    logger.warning(f"Process {name} did not terminate gracefully, killing...")
                    process_info.subprocess.kill()
                    process_info.subprocess.wait()
            
            process_info.state = ProcessState.STOPPED
            process_info.pid = None
            self.manager_stats['active_processes'] -= 1
            
            logger.info(f"Stopped process {name}")
            return True
            
        except Exception as e:
            process_info.state = ProcessState.FAILED
            process_info.last_error = str(e)
            logger.error(f"Error stopping process {name}: {e}")
            return False
    
    def restart_process(self, name: str) -> bool:
        """Restart process"""
        if name not in self.processes:
            return False
        
        process_info = self.processes[name]
        
        # Check restart limits
        if process_info.restart_count >= self.config.max_restart_attempts:
            logger.error(f"Process {name} has exceeded maximum restart attempts")
            return False
        
        logger.info(f"Restarting process {name}")
        
        # Stop process
        self.stop_process(name)
        
        # Wait before restart
        time.sleep(self.config.restart_delay)
        
        # Start process
        success = self.start_process(name)
        
        if success:
            process_info.restart_count += 1
            process_info.last_restart = time.time()
            self.manager_stats['restart_count'] += 1
        
        return success
    
    def start_all_processes(self) -> Dict[str, bool]:
        """Start all registered processes in dependency order"""
        results = {}
        
        # Topological sort for dependency resolution
        sorted_processes = self._topological_sort()
        
        for process_name in sorted_processes:
            results[process_name] = self.start_process(process_name, wait_for_dependencies=False)
        
        return results
    
    def stop_all_processes(self) -> Dict[str, bool]:
        """Stop all processes in reverse dependency order"""
        results = {}
        
        # Reverse topological sort
        sorted_processes = list(reversed(self._topological_sort()))
        
        for process_name in sorted_processes:
            if self.processes[process_name].state == ProcessState.RUNNING:
                results[process_name] = self.stop_process(process_name)
        
        return results
    
    def _topological_sort(self) -> List[str]:
        """Topological sort for dependency resolution"""
        visited = set()
        temp_visited = set()
        result = []
        
        def visit(node):
            if node in temp_visited:
                raise ValidationError(f"Circular dependency detected involving {node}")
            if node in visited:
                return
            
            temp_visited.add(node)
            for dependency in self.dependency_graph.get(node, []):
                if dependency in self.dependency_graph:
                    visit(dependency)
            
            temp_visited.remove(node)
            visited.add(node)
            result.append(node)
        
        for process_name in self.processes.keys():
            if process_name not in visited:
                visit(process_name)
        
        return result
    
    def get_process_status(self, name: str) -> Dict[str, Any]:
        """Get detailed process status"""
        if name not in self.processes:
            return {'status': 'not_found'}
        
        process_info = self.processes[name]
        
        status = {
            'name': name,
            'state': process_info.state.value,
            'pid': process_info.pid,
            'start_time': process_info.start_time,
            'restart_count': process_info.restart_count,
            'last_restart': process_info.last_restart,
            'cpu_usage': process_info.cpu_usage,
            'memory_usage': process_info.memory_usage,
            'error_count': process_info.error_count,
            'last_error': process_info.last_error,
            'priority': process_info.definition.priority.value,
            'auto_restart': process_info.definition.auto_restart,
            'dependencies': process_info.definition.dependencies
        }
        
        # Add monitoring data
        if self.config.enable_resource_monitoring:
            monitoring_data = self.process_monitor.monitor_process(process_info)
            status['monitoring'] = monitoring_data
            
            # Check resource violations
            violations = self.process_monitor.check_resource_limits(process_info)
            status['resource_violations'] = violations
        
        return status
    
    def get_all_process_statuses(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all processes"""
        return {name: self.get_process_status(name) for name in self.processes.keys()}
    
    def _continuous_monitoring(self):
        """Background thread for continuous process monitoring"""
        while self.is_running:
            try:
                for name, process_info in self.processes.items():
                    if process_info.state == ProcessState.RUNNING:
                        # Monitor process
                        if self.config.enable_resource_monitoring:
                            self.process_monitor.monitor_process(process_info)
                        
                        # Check if process is still alive
                        is_alive = False
                        if process_info.process:
                            is_alive = process_info.process.is_alive()
                        elif process_info.subprocess:
                            is_alive = process_info.subprocess.poll() is None
                        
                        if not is_alive:
                            logger.warning(f"Process {name} has died")
                            process_info.state = ProcessState.FAILED
                            self.manager_stats['active_processes'] -= 1
                
                self.manager_stats['monitoring_cycles'] += 1
                time.sleep(self.config.monitoring_interval)
                
            except Exception as e:
                logger.error(f"Error in process monitoring: {e}")
                time.sleep(5.0)
    
    def _auto_recovery(self):
        """Background thread for automatic process recovery"""
        while self.is_running:
            try:
                for name, process_info in self.processes.items():
                    if (process_info.state == ProcessState.FAILED and 
                        process_info.definition.auto_restart and
                        process_info.restart_count < self.config.max_restart_attempts):
                        
                        # Check if enough time has passed since last restart
                        if (not process_info.last_restart or 
                            time.time() - process_info.last_restart > self.config.restart_delay * 2):
                            
                            logger.info(f"Auto-recovering process {name}")
                            self.restart_process(name)
                
                time.sleep(self.config.monitoring_interval * 2)
                
            except Exception as e:
                logger.error(f"Error in auto-recovery: {e}")
                time.sleep(10.0)
    
    def start(self) -> bool:
        """Start process manager"""
        if self.is_running:
            return True
        
        try:
            self.is_running = True
            
            # Start monitoring thread
            if self.config.enable_resource_monitoring:
                self.monitoring_thread = threading.Thread(
                    target=self._continuous_monitoring,
                    name="ProcessMonitoringThread"
                )
                self.monitoring_thread.daemon = True
                self.monitoring_thread.start()
            
            # Start recovery thread
            if self.config.enable_auto_recovery:
                self.recovery_thread = threading.Thread(
                    target=self._auto_recovery,
                    name="ProcessRecoveryThread"
                )
                self.recovery_thread.daemon = True
                self.recovery_thread.start()
            
            logger.info("Process manager started successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start process manager: {e}")
            self.is_running = False
            return False
    
    def stop(self):
        """Stop process manager"""
        self.is_running = False
        
        # Stop all processes
        self.stop_all_processes()
        
        # Wait for threads
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5.0)
        
        if self.recovery_thread:
            self.recovery_thread.join(timeout=5.0)
        
        logger.info("Process manager stopped")
    
    def get_status(self) -> Dict[str, Any]:
        """Get process manager status"""
        return {
            'is_running': self.is_running,
            'total_processes': len(self.processes),
            'active_processes': sum(1 for p in self.processes.values() if p.state == ProcessState.RUNNING),
            'failed_processes': sum(1 for p in self.processes.values() if p.state == ProcessState.FAILED),
            'manager_stats': self.manager_stats,
            'arm64_optimized': self.arm64_optimizer.is_arm64,
            'cpu_topology': self.arm64_optimizer.cpu_topology
        }
    
    def generate_process_report(self) -> Dict[str, Any]:
        """Generate comprehensive process report"""
        return {
            'manager_status': self.get_status(),
            'process_statuses': self.get_all_process_statuses(),
            'process_statistics': {
                name: self.process_monitor.get_process_statistics(name)
                for name in self.processes.keys()
            },
            'dependency_graph': self.dependency_graph,
            'system_info': {
                'platform': platform.platform(),
                'cpu_count': mp.cpu_count(),
                'is_arm64': self.arm64_optimizer.is_arm64
            }
        }
    
    def optimize_cpu_affinity(self):
        """Optimize CPU affinity for all running processes"""
        if not self.config.cpu_affinity_optimization:
            return
        
        for name, process_info in self.processes.items():
            if process_info.state == ProcessState.RUNNING and process_info.process:
                self.arm64_optimizer.optimize_process_scheduling(
                    process_info.process, process_info.definition.priority
                )
    
    def cleanup(self):
        """Cleanup process manager resources"""
        if self.is_running:
            self.stop()
        
        if self.shared_memory:
            self.shared_memory.cleanup()
        
        logger.info("ProcessManager cleanup completed")

# Factory function for easy instantiation
def create_process_manager(config: ProcessConfig = None) -> AdvancedProcessManager:
    """
    Factory function to create process manager with optimal configuration
    
    Args:
        config: Process manager configuration
        
    Returns:
        Configured AdvancedProcessManager instance
    """
    if config is None:
        config = ProcessConfig()
        
        # Auto-detect optimal settings
        if platform.machine().lower() in ['arm64', 'aarch64']:
            config.enable_arm64_optimizations = True
            config.cpu_affinity_optimization = True
            config.max_workers = min(mp.cpu_count(), 8)
        
        # Adjust for available memory
        available_memory = psutil.virtual_memory().available / (1024**3)  # GB
        if available_memory > 8:
            config.memory_limit_mb = 2048
        elif available_memory > 4:
            config.memory_limit_mb = 1024
        else:
            config.memory_limit_mb = 512
    
    return AdvancedProcessManager(config)

# Legacy compatibility
ProcessManager = AdvancedProcessManager

if __name__ == "__main__":
    # Example usage and testing
    
    def sample_worker(name: str, duration: int = 10):
        """Sample worker function"""
        print(f"Worker {name} starting...")
        time.sleep(duration)
        print(f"Worker {name} finished")
    
    # Create process manager
    process_manager = create_process_manager()
    
    # Register processes
    process_def = ProcessDefinition(
        name="sample_worker",
        target=sample_worker,
        args=("test_worker", 5),
        priority=ProcessPriority.NORMAL,
        auto_restart=True
    )
    
    process_manager.register_process(process_def)
    
    # Start process manager
    if process_manager.start():
        print("Process manager started successfully")
        
        # Start process
        if process_manager.start_process("sample_worker"):
            print("Sample worker started")
            
            # Monitor for a while
            time.sleep(3)
            
            # Get status
            status = process_manager.get_process_status("sample_worker")
            print(f"Process status: {status}")
            
            # Generate report
            report = process_manager.generate_process_report()
            print(f"Process report generated with {len(report)} sections")
        
        # Cleanup
        process_manager.stop()
        process_manager.cleanup()
    else:
        print("Failed to start process manager")
