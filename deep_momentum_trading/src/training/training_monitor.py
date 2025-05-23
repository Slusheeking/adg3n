"""
Training Monitor for MLOps and Real-Time Training Monitoring
Provides comprehensive monitoring, alerting, and visualization for training processes.
"""

import asyncio
import time
import threading
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime, timedelta
import json
import numpy as np
import logging
from collections import defaultdict, deque
from dataclasses import dataclass, asdict
import psutil
import torch

from deep_momentum_trading.src.communication.enhanced_zmq_hub import EnhancedZMQHub
from deep_momentum_trading.src.storage.training_database import TrainingDatabase
from deep_momentum_trading.src.utils.logger import get_logger

logger = get_logger(__name__)

@dataclass
class TrainingMetrics:
    """Data class for training metrics."""
    timestamp: float
    epoch: int
    train_loss: float
    val_loss: float
    learning_rate: float
    epoch_time: float
    memory_usage_mb: float
    gpu_memory_mb: float
    cpu_percent: float
    gpu_utilization: float
    batch_size: int
    samples_per_second: float

@dataclass
class SystemMetrics:
    """Data class for system metrics."""
    timestamp: float
    cpu_percent: float
    memory_percent: float
    memory_available_gb: float
    disk_usage_percent: float
    gpu_count: int
    gpu_memory_used_mb: List[float]
    gpu_memory_total_mb: List[float]
    gpu_utilization: List[float]
    gpu_temperature: List[float]
    network_io_mb: Dict[str, float]

@dataclass
class Alert:
    """Data class for training alerts."""
    timestamp: float
    alert_type: str
    severity: str  # 'info', 'warning', 'error', 'critical'
    message: str
    metrics: Dict[str, Any]
    training_id: str

class TrainingMonitor:
    """
    Comprehensive training monitor with real-time metrics collection,
    alerting, and visualization capabilities.
    """
    
    def __init__(self,
                 training_id: str,
                 zmq_hub: Optional[EnhancedZMQHub] = None,
                 database: Optional[TrainingDatabase] = None,
                 monitoring_config: Optional[Dict[str, Any]] = None):
        """
        Initialize training monitor.
        
        Args:
            training_id (str): Unique training session ID
            zmq_hub (Optional[EnhancedZMQHub]): ZMQ communication hub
            database (Optional[TrainingDatabase]): Database for storing metrics
            monitoring_config (Optional[Dict[str, Any]]): Monitoring configuration
        """
        self.training_id = training_id
        self.zmq_hub = zmq_hub
        self.database = database
        self.config = monitoring_config or self._default_config()
        
        # Monitoring state
        self.is_monitoring = False
        self.start_time = None
        self.monitoring_thread = None
        
        # Metrics storage
        self.training_metrics_buffer = deque(maxlen=1000)
        self.system_metrics_buffer = deque(maxlen=1000)
        self.alerts_buffer = deque(maxlen=100)
        
        # Alert thresholds
        self.alert_thresholds = self.config.get('alert_thresholds', {})
        
        # Callbacks for custom monitoring
        self.metric_callbacks: List[Callable] = []
        self.alert_callbacks: List[Callable] = []
        
        # Performance tracking
        self.performance_tracker = PerformanceTracker()
        
        logger.info(f"TrainingMonitor initialized for training_id: {training_id}")
    
    def _default_config(self) -> Dict[str, Any]:
        """Get default monitoring configuration."""
        return {
            'metrics_interval_seconds': 30,
            'system_metrics_enabled': True,
            'gpu_monitoring_enabled': True,
            'alert_thresholds': {
                'memory_percent': 90,
                'gpu_memory_percent': 95,
                'loss_divergence_factor': 10.0,
                'training_stall_minutes': 30,
                'gpu_temperature_celsius': 85
            },
            'visualization': {
                'enabled': True,
                'update_interval_seconds': 60,
                'save_plots': True
            }
        }
    
    async def start_monitoring(self):
        """Start the monitoring process."""
        if self.is_monitoring:
            logger.warning("Monitoring already started")
            return
        
        self.is_monitoring = True
        self.start_time = time.time()
        
        # Start monitoring thread
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop)
        self.monitoring_thread.daemon = True
        self.monitoring_thread.start()
        
        # Send start event
        if self.zmq_hub:
            await self.zmq_hub.publish_system_event({
                'event_type': 'training_monitoring_started',
                'training_id': self.training_id,
                'timestamp': self.start_time
            })
        
        logger.info(f"Training monitoring started for {self.training_id}")
    
    async def stop_monitoring(self):
        """Stop the monitoring process."""
        if not self.is_monitoring:
            logger.warning("Monitoring not running")
            return
        
        self.is_monitoring = False
        
        # Wait for monitoring thread to finish
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5.0)
        
        # Send stop event
        if self.zmq_hub:
            await self.zmq_hub.publish_system_event({
                'event_type': 'training_monitoring_stopped',
                'training_id': self.training_id,
                'timestamp': time.time(),
                'duration_seconds': time.time() - self.start_time if self.start_time else 0
            })
        
        # Save final metrics to database
        await self._save_final_metrics()
        
        logger.info(f"Training monitoring stopped for {self.training_id}")
    
    def _monitoring_loop(self):
        """Main monitoring loop running in separate thread."""
        while self.is_monitoring:
            try:
                # Collect system metrics
                if self.config['system_metrics_enabled']:
                    system_metrics = self._collect_system_metrics()
                    self.system_metrics_buffer.append(system_metrics)
                    
                    # Check for system alerts
                    self._check_system_alerts(system_metrics)
                
                # Process any queued training metrics
                self._process_training_metrics()
                
                # Update performance tracker
                self.performance_tracker.update()
                
                # Sleep until next collection
                time.sleep(self.config['metrics_interval_seconds'])
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}", exc_info=True)
    
    def _collect_system_metrics(self) -> SystemMetrics:
        """Collect system performance metrics."""
        # CPU and Memory
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        # Network I/O
        network_io = psutil.net_io_counters()
        network_io_mb = {
            'bytes_sent': network_io.bytes_sent / (1024 * 1024),
            'bytes_recv': network_io.bytes_recv / (1024 * 1024)
        }
        
        # GPU metrics
        gpu_memory_used = []
        gpu_memory_total = []
        gpu_utilization = []
        gpu_temperature = []
        gpu_count = 0
        
        if torch.cuda.is_available() and self.config['gpu_monitoring_enabled']:
            gpu_count = torch.cuda.device_count()
            
            for i in range(gpu_count):
                try:
                    # Memory info
                    memory_info = torch.cuda.memory_stats(i)
                    allocated = memory_info.get('allocated_bytes.all.current', 0) / (1024 * 1024)
                    total = torch.cuda.get_device_properties(i).total_memory / (1024 * 1024)
                    
                    gpu_memory_used.append(allocated)
                    gpu_memory_total.append(total)
                    
                    # Utilization (placeholder - would need nvidia-ml-py for real data)
                    gpu_utilization.append(0.0)
                    gpu_temperature.append(0.0)
                    
                except Exception as e:
                    logger.warning(f"Failed to collect GPU {i} metrics: {e}")
                    gpu_memory_used.append(0.0)
                    gpu_memory_total.append(0.0)
                    gpu_utilization.append(0.0)
                    gpu_temperature.append(0.0)
        
        return SystemMetrics(
            timestamp=time.time(),
            cpu_percent=cpu_percent,
            memory_percent=memory.percent,
            memory_available_gb=memory.available / (1024**3),
            disk_usage_percent=disk.percent,
            gpu_count=gpu_count,
            gpu_memory_used_mb=gpu_memory_used,
            gpu_memory_total_mb=gpu_memory_total,
            gpu_utilization=gpu_utilization,
            gpu_temperature=gpu_temperature,
            network_io_mb=network_io_mb
        )
    
    def _check_system_alerts(self, metrics: SystemMetrics):
        """Check system metrics against alert thresholds."""
        alerts = []
        
        # Memory usage alert
        if metrics.memory_percent > self.alert_thresholds.get('memory_percent', 90):
            alerts.append(Alert(
                timestamp=metrics.timestamp,
                alert_type='high_memory_usage',
                severity='warning',
                message=f"High memory usage: {metrics.memory_percent:.1f}%",
                metrics={'memory_percent': metrics.memory_percent},
                training_id=self.training_id
            ))
        
        # GPU memory alerts
        for i, (used, total) in enumerate(zip(metrics.gpu_memory_used_mb, metrics.gpu_memory_total_mb)):
            if total > 0:
                gpu_percent = (used / total) * 100
                if gpu_percent > self.alert_thresholds.get('gpu_memory_percent', 95):
                    alerts.append(Alert(
                        timestamp=metrics.timestamp,
                        alert_type='high_gpu_memory',
                        severity='warning',
                        message=f"High GPU {i} memory usage: {gpu_percent:.1f}%",
                        metrics={'gpu_id': i, 'gpu_memory_percent': gpu_percent},
                        training_id=self.training_id
                    ))
        
        # GPU temperature alerts
        for i, temp in enumerate(metrics.gpu_temperature):
            if temp > self.alert_thresholds.get('gpu_temperature_celsius', 85):
                alerts.append(Alert(
                    timestamp=metrics.timestamp,
                    alert_type='high_gpu_temperature',
                    severity='error',
                    message=f"High GPU {i} temperature: {temp:.1f}°C",
                    metrics={'gpu_id': i, 'temperature': temp},
                    training_id=self.training_id
                ))
        
        # Process alerts
        for alert in alerts:
            self._process_alert(alert)
    
    async def log_epoch_metrics(self, metrics: Dict[str, Any]):
        """Log training metrics for an epoch."""
        # Create training metrics object
        training_metrics = TrainingMetrics(
            timestamp=time.time(),
            epoch=metrics.get('epoch', 0),
            train_loss=metrics.get('train_loss', 0.0),
            val_loss=metrics.get('val_loss', 0.0),
            learning_rate=metrics.get('learning_rate', 0.0),
            epoch_time=metrics.get('epoch_time', 0.0),
            memory_usage_mb=metrics.get('memory_usage_mb', 0.0),
            gpu_memory_mb=metrics.get('gpu_memory_mb', 0.0),
            cpu_percent=metrics.get('cpu_percent', 0.0),
            gpu_utilization=metrics.get('gpu_utilization', 0.0),
            batch_size=metrics.get('batch_size', 0),
            samples_per_second=metrics.get('samples_per_second', 0.0)
        )
        
        # Add to buffer
        self.training_metrics_buffer.append(training_metrics)
        
        # Check for training alerts
        self._check_training_alerts(training_metrics)
        
        # Publish metrics via ZMQ
        if self.zmq_hub:
            await self.zmq_hub.publish_training_metrics({
                'training_id': self.training_id,
                'metrics': asdict(training_metrics)
            })
        
        # Save to database
        if self.database:
            await self.database.log_training_metrics(self.training_id, asdict(training_metrics))
        
        # Call custom callbacks
        for callback in self.metric_callbacks:
            try:
                callback(training_metrics)
            except Exception as e:
                logger.error(f"Error in metric callback: {e}")
    
    def _check_training_alerts(self, metrics: TrainingMetrics):
        """Check training metrics for alerts."""
        alerts = []
        
        # Loss divergence check
        if len(self.training_metrics_buffer) >= 2:
            prev_metrics = self.training_metrics_buffer[-2]
            loss_ratio = metrics.train_loss / (prev_metrics.train_loss + 1e-8)
            
            if loss_ratio > self.alert_thresholds.get('loss_divergence_factor', 10.0):
                alerts.append(Alert(
                    timestamp=metrics.timestamp,
                    alert_type='loss_divergence',
                    severity='error',
                    message=f"Training loss diverged: {loss_ratio:.2f}x increase",
                    metrics={'loss_ratio': loss_ratio, 'current_loss': metrics.train_loss},
                    training_id=self.training_id
                ))
        
        # Training stall check
        stall_threshold = self.alert_thresholds.get('training_stall_minutes', 30) * 60
        if (time.time() - metrics.timestamp) > stall_threshold:
            alerts.append(Alert(
                timestamp=time.time(),
                alert_type='training_stall',
                severity='warning',
                message=f"Training appears stalled for {stall_threshold/60:.1f} minutes",
                metrics={'last_update': metrics.timestamp},
                training_id=self.training_id
            ))
        
        # Process alerts
        for alert in alerts:
            self._process_alert(alert)
    
    def _process_alert(self, alert: Alert):
        """Process and distribute an alert."""
        self.alerts_buffer.append(alert)
        
        # Log alert
        log_level = {
            'info': logging.INFO,
            'warning': logging.WARNING,
            'error': logging.ERROR,
            'critical': logging.CRITICAL
        }.get(alert.severity, logging.WARNING)
        
        logger.log(log_level, f"Training Alert [{alert.alert_type}]: {alert.message}")
        
        # Send via ZMQ
        if self.zmq_hub:
            asyncio.create_task(self.zmq_hub.publish_system_event({
                'event_type': 'training_alert',
                'alert': asdict(alert)
            }))
        
        # Save to database
        if self.database:
            asyncio.create_task(self.database.log_training_alert(asdict(alert)))
        
        # Call custom callbacks
        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                logger.error(f"Error in alert callback: {e}")
    
    def _process_training_metrics(self):
        """Process any queued training metrics."""
        # This could include aggregation, trend analysis, etc.
        if len(self.training_metrics_buffer) >= 10:
            # Calculate moving averages
            recent_metrics = list(self.training_metrics_buffer)[-10:]
            avg_train_loss = np.mean([m.train_loss for m in recent_metrics])
            avg_val_loss = np.mean([m.val_loss for m in recent_metrics])
            
            # Update performance tracker
            self.performance_tracker.update_averages(avg_train_loss, avg_val_loss)
    
    async def _save_final_metrics(self):
        """Save final metrics summary to database."""
        if not self.database or not self.training_metrics_buffer:
            return
        
        # Calculate summary statistics
        metrics_list = list(self.training_metrics_buffer)
        
        summary = {
            'training_id': self.training_id,
            'total_epochs': len(metrics_list),
            'final_train_loss': metrics_list[-1].train_loss if metrics_list else 0.0,
            'final_val_loss': metrics_list[-1].val_loss if metrics_list else 0.0,
            'best_val_loss': min(m.val_loss for m in metrics_list) if metrics_list else 0.0,
            'total_training_time': time.time() - self.start_time if self.start_time else 0.0,
            'avg_epoch_time': np.mean([m.epoch_time for m in metrics_list]) if metrics_list else 0.0,
            'total_alerts': len(self.alerts_buffer)
        }
        
        await self.database.log_training_summary(summary)
        logger.info(f"Final training metrics saved for {self.training_id}")
    
    def add_metric_callback(self, callback: Callable[[TrainingMetrics], None]):
        """Add custom metric callback."""
        self.metric_callbacks.append(callback)
    
    def add_alert_callback(self, callback: Callable[[Alert], None]):
        """Add custom alert callback."""
        self.alert_callbacks.append(callback)
    
    def get_current_metrics(self) -> Dict[str, Any]:
        """Get current training metrics summary."""
        if not self.training_metrics_buffer:
            return {}
        
        latest = self.training_metrics_buffer[-1]
        return {
            'training_id': self.training_id,
            'current_epoch': latest.epoch,
            'current_train_loss': latest.train_loss,
            'current_val_loss': latest.val_loss,
            'current_lr': latest.learning_rate,
            'uptime_seconds': time.time() - self.start_time if self.start_time else 0,
            'total_alerts': len(self.alerts_buffer),
            'is_monitoring': self.is_monitoring
        }

class PerformanceTracker:
    """Tracks training performance trends and anomalies."""
    
    def __init__(self):
        self.loss_history = deque(maxlen=100)
        self.lr_history = deque(maxlen=100)
        self.time_history = deque(maxlen=100)
        
        self.moving_avg_window = 10
        self.anomaly_threshold = 2.0  # Standard deviations
    
    def update(self):
        """Update performance tracking."""
        # This would contain more sophisticated performance analysis
        pass
    
    def update_averages(self, train_loss: float, val_loss: float):
        """Update moving averages."""
        self.loss_history.append({'train': train_loss, 'val': val_loss})
    
    def detect_anomalies(self) -> List[Dict[str, Any]]:
        """Detect performance anomalies."""
        anomalies = []
        
        if len(self.loss_history) < self.moving_avg_window:
            return anomalies
        
        # Simple anomaly detection based on loss trends
        recent_losses = [entry['train'] for entry in list(self.loss_history)[-self.moving_avg_window:]]
        mean_loss = np.mean(recent_losses)
        std_loss = np.std(recent_losses)
        
        if std_loss > 0:
            latest_loss = recent_losses[-1]
            z_score = abs(latest_loss - mean_loss) / std_loss
            
            if z_score > self.anomaly_threshold:
                anomalies.append({
                    'type': 'loss_anomaly',
                    'z_score': z_score,
                    'current_loss': latest_loss,
                    'mean_loss': mean_loss
                })
        
        return anomalies

if __name__ == "__main__":
    # Example usage
    import asyncio
    
    async def test_training_monitor():
        monitor = TrainingMonitor(training_id="test_training_123")
        
        # Start monitoring
        await monitor.start_monitoring()
        
        # Simulate training epochs
        for epoch in range(5):
            await monitor.log_epoch_metrics({
                'epoch': epoch,
                'train_loss': 1.0 - epoch * 0.1,
                'val_loss': 1.1 - epoch * 0.1,
                'learning_rate': 0.001,
                'epoch_time': 30.0,
                'batch_size': 32
            })
            
            await asyncio.sleep(1)  # Simulate time between epochs
        
        # Stop monitoring
        await monitor.stop_monitoring()
        
        print("Training monitor test completed")
    
    asyncio.run(test_training_monitor())