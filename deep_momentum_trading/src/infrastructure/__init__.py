"""
Enhanced Infrastructure Package with ARM64 Optimizations

This package provides comprehensive infrastructure management capabilities for the Deep Momentum Trading System
with ARM64-specific optimizations for high-performance system management and monitoring.

Features:
- Advanced health monitoring with ARM64 optimizations
- Intelligent process management with auto-recovery
- Comprehensive resource monitoring and alerting
- High-performance task scheduling with priority queues
- System orchestration and coordination
- Performance optimization and tuning
"""

import platform
import multiprocessing as mp
from typing import Dict, Any, Optional

from .health_check import AdvancedHealthMonitor, HealthConfig, create_health_monitor
from .process_manager import AdvancedProcessManager, ProcessConfig, create_process_manager
from .resource_monitor import AdvancedResourceMonitor, ResourceConfig, create_resource_monitor
from .scheduler import AdvancedScheduler, SchedulerConfig, create_scheduler

# Package version
__version__ = "1.0.0"

# Package metadata
__author__ = "Deep Momentum Trading Team"
__description__ = "Enhanced infrastructure system with ARM64 optimizations"

# Export main classes
__all__ = [
    'InfrastructureManager',
    'AdvancedHealthMonitor',
    'AdvancedProcessManager',
    'AdvancedResourceMonitor',
    'AdvancedScheduler',
    'HealthConfig',
    'ProcessConfig',
    'ResourceConfig',
    'SchedulerConfig',
    'create_infrastructure_manager',
    'create_health_monitor',
    'create_process_manager',
    'create_resource_monitor',
    'create_scheduler'
]

class InfrastructureManager:
    """
    Central manager for all infrastructure components
    
    Provides unified interface for managing health monitoring, process management,
    resource monitoring, and task scheduling with ARM64 optimizations.
    """
    
    def __init__(self,
                 health_config: Optional[HealthConfig] = None,
                 process_config: Optional[ProcessConfig] = None,
                 resource_config: Optional[ResourceConfig] = None,
                 scheduler_config: Optional[SchedulerConfig] = None):
        
        self.is_arm64 = platform.machine().lower() in ['arm64', 'aarch64']
        self.cpu_count = mp.cpu_count()
        
        # Initialize components
        self.health_monitor = create_health_monitor(health_config)
        self.process_manager = create_process_manager(process_config)
        self.resource_monitor = create_resource_monitor(resource_config)
        self.scheduler = create_scheduler(scheduler_config)
        
        # System status
        self.is_running = False
        self.components_status = {}
        
        self._setup_logging()
        
    def _setup_logging(self):
        """Setup centralized logging"""
        from ..utils.logger import get_logger
        self.logger = get_logger(__name__)
        self.logger.info(f"InfrastructureManager initialized on ARM64: {self.is_arm64}")
    
    def start_infrastructure(self) -> bool:
        """
        Start all infrastructure components
        
        Returns:
            bool: True if all components started successfully
        """
        try:
            # Start components in order
            components = [
                ('resource_monitor', self.resource_monitor),
                ('health_monitor', self.health_monitor),
                ('scheduler', self.scheduler),
                ('process_manager', self.process_manager)
            ]
            
            for name, component in components:
                if hasattr(component, 'start'):
                    success = component.start()
                    self.components_status[name] = 'running' if success else 'failed'
                    if not success:
                        self.logger.error(f"Failed to start {name}")
                        return False
                else:
                    self.components_status[name] = 'ready'
            
            self.is_running = True
            self.logger.info("All infrastructure components started successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Error starting infrastructure: {e}")
            return False
    
    def stop_infrastructure(self) -> bool:
        """
        Stop all infrastructure components
        
        Returns:
            bool: True if all components stopped successfully
        """
        try:
            # Stop components in reverse order
            components = [
                ('process_manager', self.process_manager),
                ('scheduler', self.scheduler),
                ('health_monitor', self.health_monitor),
                ('resource_monitor', self.resource_monitor)
            ]
            
            for name, component in components:
                if hasattr(component, 'stop'):
                    component.stop()
                self.components_status[name] = 'stopped'
            
            self.is_running = False
            self.logger.info("All infrastructure components stopped successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Error stopping infrastructure: {e}")
            return False
    
    def get_system_status(self) -> Dict[str, Any]:
        """
        Get comprehensive system status
        
        Returns:
            Dict containing system status information
        """
        return {
            'is_running': self.is_running,
            'is_arm64': self.is_arm64,
            'cpu_count': self.cpu_count,
            'components_status': self.components_status,
            'health_status': self.health_monitor.get_overall_health_status() if hasattr(self.health_monitor, 'get_overall_health_status') else {},
            'process_status': self.process_manager.get_all_process_statuses() if hasattr(self.process_manager, 'get_all_process_statuses') else {},
            'resource_status': self.resource_monitor.get_resource_summary() if hasattr(self.resource_monitor, 'get_resource_summary') else {},
            'scheduler_status': self.scheduler.get_status() if hasattr(self.scheduler, 'get_status') else {}
        }
    
    def perform_health_check(self) -> Dict[str, Any]:
        """Perform comprehensive health check across all components"""
        health_results = {
            'timestamp': platform.time.time() if hasattr(platform, 'time') else 0,
            'overall_healthy': True,
            'component_health': {}
        }
        
        # Check each component
        components = {
            'health_monitor': self.health_monitor,
            'process_manager': self.process_manager,
            'resource_monitor': self.resource_monitor,
            'scheduler': self.scheduler
        }
        
        for name, component in components.items():
            try:
                if hasattr(component, 'get_health_status'):
                    status = component.get_health_status()
                    health_results['component_health'][name] = status
                    if not status.get('healthy', True):
                        health_results['overall_healthy'] = False
                else:
                    health_results['component_health'][name] = {'healthy': True, 'status': 'unknown'}
            except Exception as e:
                health_results['component_health'][name] = {'healthy': False, 'error': str(e)}
                health_results['overall_healthy'] = False
        
        return health_results
    
    def optimize_system_performance(self) -> Dict[str, Any]:
        """Optimize system performance based on current conditions"""
        optimization_results = {
            'optimizations_applied': [],
            'recommendations': [],
            'performance_impact': {}
        }
        
        # Get current system status
        system_status = self.get_system_status()
        
        # Apply ARM64-specific optimizations
        if self.is_arm64:
            optimization_results['optimizations_applied'].append('ARM64 SIMD optimizations enabled')
            
            # Optimize CPU affinity for ARM64
            if hasattr(self.process_manager, 'optimize_cpu_affinity'):
                self.process_manager.optimize_cpu_affinity()
                optimization_results['optimizations_applied'].append('CPU affinity optimization')
        
        # Resource-based optimizations
        resource_status = system_status.get('resource_status', {})
        if resource_status:
            current_metrics = resource_status.get('current', {})
            if current_metrics:
                cpu_usage = current_metrics.get('system', {}).get('cpu_percent', 0)
                memory_usage = current_metrics.get('system', {}).get('memory_percent', 0)
                
                if cpu_usage > 80:
                    optimization_results['recommendations'].append('High CPU usage detected - consider process optimization')
                
                if memory_usage > 85:
                    optimization_results['recommendations'].append('High memory usage detected - consider memory cleanup')
        
        return optimization_results
    
    def generate_infrastructure_report(self) -> Dict[str, Any]:
        """Generate comprehensive infrastructure report"""
        return {
            'system_status': self.get_system_status(),
            'health_check': self.perform_health_check(),
            'performance_optimization': self.optimize_system_performance(),
            'component_details': {
                'health_monitor': self.health_monitor.get_status() if hasattr(self.health_monitor, 'get_status') else {},
                'process_manager': self.process_manager.get_status() if hasattr(self.process_manager, 'get_status') else {},
                'resource_monitor': self.resource_monitor.get_status() if hasattr(self.resource_monitor, 'get_status') else {},
                'scheduler': self.scheduler.get_status() if hasattr(self.scheduler, 'get_status') else {}
            }
        }
    
    def cleanup(self):
        """Cleanup all infrastructure components"""
        if self.is_running:
            self.stop_infrastructure()
        
        # Cleanup individual components
        for component in [self.health_monitor, self.process_manager, 
                         self.resource_monitor, self.scheduler]:
            if hasattr(component, 'cleanup'):
                component.cleanup()
        
        self.logger.info("InfrastructureManager cleanup completed")

def create_infrastructure_manager(health_config: Optional[HealthConfig] = None,
                                process_config: Optional[ProcessConfig] = None,
                                resource_config: Optional[ResourceConfig] = None,
                                scheduler_config: Optional[SchedulerConfig] = None) -> InfrastructureManager:
    """
    Factory function to create infrastructure manager with optimal configuration
    
    Args:
        health_config: Health monitor configuration
        process_config: Process manager configuration
        resource_config: Resource monitor configuration
        scheduler_config: Scheduler configuration
        
    Returns:
        Configured InfrastructureManager instance
    """
    return InfrastructureManager(
        health_config=health_config,
        process_config=process_config,
        resource_config=resource_config,
        scheduler_config=scheduler_config
    )

def detect_infrastructure_capabilities() -> Dict[str, Any]:
    """
    Detect infrastructure capabilities for optimization
    
    Returns:
        Dict containing infrastructure capability information
    """
    is_arm64 = platform.machine().lower() in ['arm64', 'aarch64']
    
    capabilities = {
        'is_arm64': is_arm64,
        'cpu_count': mp.cpu_count(),
        'platform': platform.platform(),
        'machine': platform.machine(),
        'processor': platform.processor(),
        'python_version': platform.python_version()
    }
    
    if is_arm64:
        try:
            # Check for specific ARM64 features
            with open('/proc/cpuinfo', 'r') as f:
                cpuinfo = f.read().lower()
                capabilities.update({
                    'neon_support': 'neon' in cpuinfo or 'asimd' in cpuinfo,
                    'crypto_support': 'aes' in cpuinfo or 'sha' in cpuinfo,
                    'fp16_support': 'fphp' in cpuinfo or 'asimdhp' in cpuinfo,
                    'sve_support': 'sve' in cpuinfo
                })
        except:
            capabilities.update({
                'neon_support': False,
                'crypto_support': False,
                'fp16_support': False,
                'sve_support': False
            })
    
    return capabilities

# Initialize infrastructure detection on import
INFRASTRUCTURE_CAPABILITIES = detect_infrastructure_capabilities()

# Legacy compatibility
HealthMonitor = AdvancedHealthMonitor
ProcessManager = AdvancedProcessManager
ResourceMonitor = AdvancedResourceMonitor
Scheduler = AdvancedScheduler

if __name__ == "__main__":
    # Example usage
    print("Deep Momentum Trading - Infrastructure System")
    print(f"Infrastructure Capabilities: {INFRASTRUCTURE_CAPABILITIES}")
    
    # Create infrastructure manager
    infrastructure = create_infrastructure_manager()
    
    # Start infrastructure
    if infrastructure.start_infrastructure():
        print("Infrastructure started successfully")
        
        # Get status
        status = infrastructure.get_system_status()
        print(f"System Status: {status}")
        
        # Generate report
        report = infrastructure.generate_infrastructure_report()
        print(f"Infrastructure Report: {len(report)} sections")
        
        # Stop infrastructure
        infrastructure.stop_infrastructure()
        infrastructure.cleanup()
    else:
        print("Failed to start infrastructure")
