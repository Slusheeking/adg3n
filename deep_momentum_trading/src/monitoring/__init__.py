"""
Enhanced Monitoring Package with ARM64 Optimizations

This package provides comprehensive monitoring capabilities for the Deep Momentum Trading System,
including real-time performance tracking, alerting, metrics calculation, and logging with
ARM64-specific optimizations.

Features:
- Real-time performance monitoring with ARM64 optimizations
- Advanced alerting system with multiple notification channels
- Comprehensive metrics calculation and analysis
- High-performance file logging with structured data
- System resource monitoring and optimization
- Trading performance analytics and reporting
"""

import platform
import multiprocessing as mp
from typing import Dict, Any, Optional

from .alert_system import AlertSystem, AlertConfig, create_alert_system
from .file_logger import FileLogger, LoggerConfig, create_file_logger
from .metrics_calculator import MetricsCalculator, MetricsConfig, create_metrics_calculator
from .performance_tracker import PerformanceTracker, TrackerConfig, create_performance_tracker

# Package version
__version__ = "1.0.0"

# Package metadata
__author__ = "Deep Momentum Trading Team"
__description__ = "Enhanced monitoring system with ARM64 optimizations"

# Export main classes
__all__ = [
    'MonitoringSystemManager',
    'AlertSystem',
    'FileLogger', 
    'MetricsCalculator',
    'PerformanceTracker',
    'AlertConfig',
    'LoggerConfig',
    'MetricsConfig',
    'TrackerConfig',
    'create_monitoring_system',
    'create_alert_system',
    'create_file_logger',
    'create_metrics_calculator',
    'create_performance_tracker'
]

class MonitoringSystemManager:
    """
    Central manager for all monitoring components
    
    Provides unified interface for managing alerts, logging, metrics,
    and performance tracking with ARM64 optimizations.
    """
    
    def __init__(self, 
                 alert_config: Optional[AlertConfig] = None,
                 logger_config: Optional[LoggerConfig] = None,
                 metrics_config: Optional[MetricsConfig] = None,
                 tracker_config: Optional[TrackerConfig] = None):
        
        self.is_arm64 = platform.machine().lower() in ['arm64', 'aarch64']
        self.cpu_count = mp.cpu_count()
        
        # Initialize components
        self.alert_system = create_alert_system(alert_config)
        self.file_logger = create_file_logger(logger_config)
        self.metrics_calculator = create_metrics_calculator(metrics_config)
        self.performance_tracker = create_performance_tracker(tracker_config)
        
        # System status
        self.is_running = False
        self.components_status = {}
        
        self._setup_logging()
        
    def _setup_logging(self):
        """Setup centralized logging"""
        from ..utils.logger import get_logger
        self.logger = get_logger(__name__)
        self.logger.info(f"MonitoringSystemManager initialized on ARM64: {self.is_arm64}")
        
    def start_monitoring(self) -> bool:
        """
        Start all monitoring components
        
        Returns:
            bool: True if all components started successfully
        """
        try:
            # Start components in order
            components = [
                ('file_logger', self.file_logger),
                ('metrics_calculator', self.metrics_calculator),
                ('performance_tracker', self.performance_tracker),
                ('alert_system', self.alert_system)
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
            self.logger.info("All monitoring components started successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Error starting monitoring system: {e}")
            return False
    
    def stop_monitoring(self) -> bool:
        """
        Stop all monitoring components
        
        Returns:
            bool: True if all components stopped successfully
        """
        try:
            # Stop components in reverse order
            components = [
                ('alert_system', self.alert_system),
                ('performance_tracker', self.performance_tracker),
                ('metrics_calculator', self.metrics_calculator),
                ('file_logger', self.file_logger)
            ]
            
            for name, component in components:
                if hasattr(component, 'stop'):
                    component.stop()
                self.components_status[name] = 'stopped'
            
            self.is_running = False
            self.logger.info("All monitoring components stopped successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Error stopping monitoring system: {e}")
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
            'alert_system_status': self.alert_system.get_status() if hasattr(self.alert_system, 'get_status') else {},
            'logger_status': self.file_logger.get_status() if hasattr(self.file_logger, 'get_status') else {},
            'metrics_status': self.metrics_calculator.get_status() if hasattr(self.metrics_calculator, 'get_status') else {},
            'tracker_status': self.performance_tracker.get_status() if hasattr(self.performance_tracker, 'get_status') else {}
        }
    
    def log_trading_event(self, event_type: str, data: Dict[str, Any]):
        """Log trading event across all relevant components"""
        # Log to file
        self.file_logger.log_trading_event(event_type, data)
        
        # Update metrics
        self.metrics_calculator.update_trading_metrics(event_type, data)
        
        # Track performance
        self.performance_tracker.track_event(event_type, data)
        
        # Check for alerts
        self.alert_system.check_trading_alerts(event_type, data)
    
    def generate_monitoring_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive monitoring report
        
        Returns:
            Dict containing monitoring report
        """
        return {
            'system_status': self.get_system_status(),
            'performance_metrics': self.performance_tracker.get_performance_summary(),
            'trading_metrics': self.metrics_calculator.get_metrics_summary(),
            'alert_summary': self.alert_system.get_alert_summary(),
            'log_summary': self.file_logger.get_log_summary()
        }
    
    def cleanup(self):
        """Cleanup all monitoring components"""
        if self.is_running:
            self.stop_monitoring()
        
        # Cleanup individual components
        for component in [self.alert_system, self.file_logger, 
                         self.metrics_calculator, self.performance_tracker]:
            if hasattr(component, 'cleanup'):
                component.cleanup()
        
        self.logger.info("MonitoringSystemManager cleanup completed")

def create_monitoring_system(alert_config: Optional[AlertConfig] = None,
                           logger_config: Optional[LoggerConfig] = None,
                           metrics_config: Optional[MetricsConfig] = None,
                           tracker_config: Optional[TrackerConfig] = None) -> MonitoringSystemManager:
    """
    Factory function to create monitoring system with optimal configuration
    
    Args:
        alert_config: Alert system configuration
        logger_config: File logger configuration
        metrics_config: Metrics calculator configuration
        tracker_config: Performance tracker configuration
        
    Returns:
        Configured MonitoringSystemManager instance
    """
    return MonitoringSystemManager(
        alert_config=alert_config,
        logger_config=logger_config,
        metrics_config=metrics_config,
        tracker_config=tracker_config
    )

def detect_arm64_capabilities() -> Dict[str, Any]:
    """
    Detect ARM64 capabilities for monitoring optimization
    
    Returns:
        Dict containing ARM64 capability information
    """
    is_arm64 = platform.machine().lower() in ['arm64', 'aarch64']
    
    capabilities = {
        'is_arm64': is_arm64,
        'cpu_count': mp.cpu_count(),
        'platform': platform.platform(),
        'machine': platform.machine(),
        'processor': platform.processor()
    }
    
    if is_arm64:
        try:
            # Check for specific ARM64 features
            with open('/proc/cpuinfo', 'r') as f:
                cpuinfo = f.read().lower()
                capabilities.update({
                    'neon_support': 'neon' in cpuinfo or 'asimd' in cpuinfo,
                    'crypto_support': 'aes' in cpuinfo or 'sha' in cpuinfo,
                    'fp16_support': 'fphp' in cpuinfo or 'asimdhp' in cpuinfo
                })
        except:
            capabilities.update({
                'neon_support': False,
                'crypto_support': False,
                'fp16_support': False
            })
    
    return capabilities

# Initialize ARM64 detection on import
ARM64_CAPABILITIES = detect_arm64_capabilities()

if __name__ == "__main__":
    # Example usage
    print("Deep Momentum Trading - Monitoring System")
    print(f"ARM64 Capabilities: {ARM64_CAPABILITIES}")
    
    # Create monitoring system
    monitoring = create_monitoring_system()
    
    # Start monitoring
    if monitoring.start_monitoring():
        print("Monitoring system started successfully")
        
        # Get status
        status = monitoring.get_system_status()
        print(f"System Status: {status}")
        
        # Stop monitoring
        monitoring.stop_monitoring()
        monitoring.cleanup()
    else:
        print("Failed to start monitoring system")