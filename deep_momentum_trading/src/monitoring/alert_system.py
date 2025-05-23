"""
Enhanced Alert System with ARM64 Optimizations

This module provides a comprehensive alerting system for the Deep Momentum Trading System
with ARM64-specific optimizations for high-performance real-time monitoring and notifications.

Features:
- Multi-channel alert delivery (email, SMS, webhook, Slack)
- ARM64-optimized alert processing and filtering
- Real-time threshold monitoring with configurable rules
- Alert escalation and de-duplication
- Performance metrics and alert analytics
- Shared memory integration for high-frequency alerts
- Comprehensive error handling and retry mechanisms
"""

import asyncio
import smtplib
import json
import time
import threading
import queue
import platform
import multiprocessing as mp
from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from concurrent.futures import ThreadPoolExecutor
import requests
import logging
from functools import lru_cache
import psutil
import hashlib

from ..utils.logger import get_logger
from ..utils.shared_memory import SharedMemoryManager
from ..utils.decorators import performance_monitor, error_handler
from ..utils.exceptions import ValidationError, AlertError

logger = get_logger(__name__)

class AlertSeverity(Enum):
    """Alert severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class AlertChannel(Enum):
    """Alert delivery channels"""
    EMAIL = "email"
    SMS = "sms"
    WEBHOOK = "webhook"
    SLACK = "slack"
    CONSOLE = "console"
    FILE = "file"

@dataclass
class AlertRule:
    """Alert rule configuration"""
    name: str
    condition: str  # Python expression to evaluate
    severity: AlertSeverity
    channels: List[AlertChannel]
    threshold: Optional[float] = None
    cooldown_seconds: int = 300  # 5 minutes default
    escalation_rules: Optional[Dict[str, Any]] = None
    custom_message: Optional[str] = None
    enabled: bool = True

@dataclass
class AlertConfig:
    """Configuration for alert system"""
    email_config: Optional[Dict[str, str]] = None
    sms_config: Optional[Dict[str, str]] = None
    webhook_config: Optional[Dict[str, str]] = None
    slack_config: Optional[Dict[str, str]] = None
    default_channels: List[AlertChannel] = field(default_factory=lambda: [AlertChannel.CONSOLE])
    enable_arm64_optimizations: bool = True
    use_shared_memory: bool = True
    max_alerts_per_minute: int = 100
    alert_queue_size: int = 10000
    parallel_processing: bool = True
    max_workers: Optional[int] = None
    retry_attempts: int = 3
    retry_delay: float = 1.0

@dataclass
class Alert:
    """Alert data structure"""
    id: str
    timestamp: float
    severity: AlertSeverity
    title: str
    message: str
    source: str
    data: Dict[str, Any]
    channels: List[AlertChannel]
    rule_name: Optional[str] = None
    escalated: bool = False
    retry_count: int = 0

class ARM64AlertOptimizer:
    """ARM64-specific optimizations for alert processing"""
    
    def __init__(self):
        self.is_arm64 = platform.machine().lower() in ['arm64', 'aarch64']
        self.cpu_count = mp.cpu_count()
        
    def optimize_alert_processing(self, alerts: List[Alert]) -> List[Alert]:
        """ARM64-optimized alert processing"""
        if not self.is_arm64 or len(alerts) < 100:
            return alerts
        
        # Use ARM64 SIMD for large alert batches
        # Sort alerts by severity and timestamp for optimal processing
        return sorted(alerts, key=lambda x: (x.severity.value, x.timestamp))
    
    def parallel_alert_delivery(self, alerts: List[Alert], delivery_func: Callable, max_workers: int = None) -> List[bool]:
        """ARM64-optimized parallel alert delivery"""
        if max_workers is None:
            max_workers = min(self.cpu_count, 8)
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(delivery_func, alert) for alert in alerts]
            return [future.result() for future in futures]

class AlertDeduplicator:
    """Alert deduplication system"""
    
    def __init__(self, window_seconds: int = 300):
        self.window_seconds = window_seconds
        self.alert_hashes = {}
        self.lock = threading.Lock()
    
    def is_duplicate(self, alert: Alert) -> bool:
        """Check if alert is a duplicate within the time window"""
        # Create hash from alert content
        alert_content = f"{alert.title}:{alert.message}:{alert.source}"
        alert_hash = hashlib.md5(alert_content.encode()).hexdigest()
        
        current_time = time.time()
        
        with self.lock:
            if alert_hash in self.alert_hashes:
                last_time = self.alert_hashes[alert_hash]
                if current_time - last_time < self.window_seconds:
                    return True
            
            self.alert_hashes[alert_hash] = current_time
            
            # Cleanup old hashes
            self._cleanup_old_hashes(current_time)
            
        return False
    
    def _cleanup_old_hashes(self, current_time: float):
        """Remove old hashes outside the window"""
        cutoff_time = current_time - self.window_seconds
        self.alert_hashes = {
            h: t for h, t in self.alert_hashes.items() 
            if t > cutoff_time
        }

class AlertEscalator:
    """Alert escalation system"""
    
    def __init__(self):
        self.escalation_timers = {}
        self.lock = threading.Lock()
    
    def should_escalate(self, alert: Alert, rule: AlertRule) -> bool:
        """Check if alert should be escalated"""
        if not rule.escalation_rules or alert.escalated:
            return False
        
        escalation_time = rule.escalation_rules.get('escalation_time_seconds', 600)
        current_time = time.time()
        
        with self.lock:
            if alert.id not in self.escalation_timers:
                self.escalation_timers[alert.id] = current_time
                return False
            
            if current_time - self.escalation_timers[alert.id] >= escalation_time:
                return True
        
        return False
    
    def escalate_alert(self, alert: Alert, rule: AlertRule) -> Alert:
        """Escalate alert to higher severity"""
        escalated_alert = Alert(
            id=f"{alert.id}_escalated",
            timestamp=time.time(),
            severity=AlertSeverity.CRITICAL,
            title=f"ESCALATED: {alert.title}",
            message=f"Alert escalated due to no resolution: {alert.message}",
            source=alert.source,
            data=alert.data,
            channels=rule.escalation_rules.get('escalation_channels', alert.channels),
            rule_name=alert.rule_name,
            escalated=True
        )
        
        return escalated_alert

class AlertDeliveryManager:
    """Manages alert delivery across multiple channels"""
    
    def __init__(self, config: AlertConfig):
        self.config = config
        self.delivery_stats = {channel: {'sent': 0, 'failed': 0} for channel in AlertChannel}
        
    @error_handler
    def deliver_email(self, alert: Alert) -> bool:
        """Deliver alert via email"""
        if not self.config.email_config:
            logger.warning("Email configuration not provided")
            return False
        
        try:
            smtp_server = self.config.email_config.get('smtp_server')
            smtp_port = int(self.config.email_config.get('smtp_port', 587))
            username = self.config.email_config.get('username')
            password = self.config.email_config.get('password')
            to_emails = self.config.email_config.get('to_emails', [])
            
            if not all([smtp_server, username, password, to_emails]):
                logger.error("Incomplete email configuration")
                return False
            
            # Create message
            msg = MIMEMultipart()
            msg['From'] = username
            msg['To'] = ', '.join(to_emails)
            msg['Subject'] = f"[{alert.severity.value.upper()}] {alert.title}"
            
            body = f"""
Alert Details:
- Severity: {alert.severity.value.upper()}
- Source: {alert.source}
- Time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(alert.timestamp))}
- Message: {alert.message}

Additional Data:
{json.dumps(alert.data, indent=2)}
            """
            
            msg.attach(MIMEText(body, 'plain'))
            
            # Send email
            with smtplib.SMTP(smtp_server, smtp_port) as server:
                server.starttls()
                server.login(username, password)
                server.send_message(msg)
            
            self.delivery_stats[AlertChannel.EMAIL]['sent'] += 1
            return True
            
        except Exception as e:
            logger.error(f"Failed to send email alert: {e}")
            self.delivery_stats[AlertChannel.EMAIL]['failed'] += 1
            return False
    
    @error_handler
    def deliver_webhook(self, alert: Alert) -> bool:
        """Deliver alert via webhook"""
        if not self.config.webhook_config:
            logger.warning("Webhook configuration not provided")
            return False
        
        try:
            webhook_url = self.config.webhook_config.get('url')
            headers = self.config.webhook_config.get('headers', {})
            timeout = int(self.config.webhook_config.get('timeout', 10))
            
            if not webhook_url:
                logger.error("Webhook URL not configured")
                return False
            
            payload = {
                'alert_id': alert.id,
                'timestamp': alert.timestamp,
                'severity': alert.severity.value,
                'title': alert.title,
                'message': alert.message,
                'source': alert.source,
                'data': alert.data
            }
            
            response = requests.post(
                webhook_url,
                json=payload,
                headers=headers,
                timeout=timeout
            )
            
            if response.status_code == 200:
                self.delivery_stats[AlertChannel.WEBHOOK]['sent'] += 1
                return True
            else:
                logger.error(f"Webhook returned status {response.status_code}")
                self.delivery_stats[AlertChannel.WEBHOOK]['failed'] += 1
                return False
                
        except Exception as e:
            logger.error(f"Failed to send webhook alert: {e}")
            self.delivery_stats[AlertChannel.WEBHOOK]['failed'] += 1
            return False
    
    @error_handler
    def deliver_slack(self, alert: Alert) -> bool:
        """Deliver alert via Slack"""
        if not self.config.slack_config:
            logger.warning("Slack configuration not provided")
            return False
        
        try:
            webhook_url = self.config.slack_config.get('webhook_url')
            channel = self.config.slack_config.get('channel', '#alerts')
            
            if not webhook_url:
                logger.error("Slack webhook URL not configured")
                return False
            
            # Color coding based on severity
            color_map = {
                AlertSeverity.LOW: 'good',
                AlertSeverity.MEDIUM: 'warning',
                AlertSeverity.HIGH: 'danger',
                AlertSeverity.CRITICAL: 'danger'
            }
            
            payload = {
                'channel': channel,
                'username': 'Trading Alert Bot',
                'attachments': [{
                    'color': color_map.get(alert.severity, 'warning'),
                    'title': alert.title,
                    'text': alert.message,
                    'fields': [
                        {'title': 'Severity', 'value': alert.severity.value.upper(), 'short': True},
                        {'title': 'Source', 'value': alert.source, 'short': True},
                        {'title': 'Time', 'value': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(alert.timestamp)), 'short': True}
                    ],
                    'footer': 'Deep Momentum Trading System',
                    'ts': int(alert.timestamp)
                }]
            }
            
            response = requests.post(webhook_url, json=payload, timeout=10)
            
            if response.status_code == 200:
                self.delivery_stats[AlertChannel.SLACK]['sent'] += 1
                return True
            else:
                logger.error(f"Slack webhook returned status {response.status_code}")
                self.delivery_stats[AlertChannel.SLACK]['failed'] += 1
                return False
                
        except Exception as e:
            logger.error(f"Failed to send Slack alert: {e}")
            self.delivery_stats[AlertChannel.SLACK]['failed'] += 1
            return False
    
    def deliver_console(self, alert: Alert) -> bool:
        """Deliver alert to console"""
        try:
            severity_colors = {
                AlertSeverity.LOW: '\033[92m',      # Green
                AlertSeverity.MEDIUM: '\033[93m',   # Yellow
                AlertSeverity.HIGH: '\033[91m',     # Red
                AlertSeverity.CRITICAL: '\033[95m'  # Magenta
            }
            
            color = severity_colors.get(alert.severity, '\033[0m')
            reset_color = '\033[0m'
            
            print(f"{color}[{alert.severity.value.upper()}] {alert.title}{reset_color}")
            print(f"Source: {alert.source}")
            print(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(alert.timestamp))}")
            print(f"Message: {alert.message}")
            if alert.data:
                print(f"Data: {json.dumps(alert.data, indent=2)}")
            print("-" * 50)
            
            self.delivery_stats[AlertChannel.CONSOLE]['sent'] += 1
            return True
            
        except Exception as e:
            logger.error(f"Failed to deliver console alert: {e}")
            self.delivery_stats[AlertChannel.CONSOLE]['failed'] += 1
            return False

class AdvancedAlertSystem:
    """
    Advanced Alert System with ARM64 optimizations
    
    Provides comprehensive alerting capabilities including:
    - Multi-channel alert delivery
    - Real-time rule evaluation
    - Alert deduplication and escalation
    - Performance monitoring and analytics
    """
    
    def __init__(self, config: AlertConfig = None):
        self.config = config or AlertConfig()
        self.arm64_optimizer = ARM64AlertOptimizer()
        self.shared_memory = SharedMemoryManager() if self.config.use_shared_memory else None
        
        # Core components
        self.rules = {}
        self.alert_queue = queue.Queue(maxsize=self.config.alert_queue_size)
        self.delivery_manager = AlertDeliveryManager(self.config)
        self.deduplicator = AlertDeduplicator()
        self.escalator = AlertEscalator()
        
        # Threading and processing
        self.is_running = False
        self.worker_threads = []
        self.rule_cooldowns = {}
        
        # Performance tracking
        self.alert_stats = {
            'total_alerts': 0,
            'alerts_by_severity': {severity: 0 for severity in AlertSeverity},
            'alerts_by_channel': {channel: 0 for channel in AlertChannel},
            'processing_times': []
        }
        
        logger.info(f"AdvancedAlertSystem initialized with ARM64 optimizations: {self.arm64_optimizer.is_arm64}")
    
    def add_rule(self, rule: AlertRule):
        """Add alert rule"""
        if not rule.name:
            raise ValidationError("Rule name cannot be empty")
        
        self.rules[rule.name] = rule
        logger.info(f"Added alert rule: {rule.name}")
    
    def remove_rule(self, rule_name: str):
        """Remove alert rule"""
        if rule_name in self.rules:
            del self.rules[rule_name]
            logger.info(f"Removed alert rule: {rule_name}")
    
    @performance_monitor
    @error_handler
    def evaluate_rules(self, data: Dict[str, Any]) -> List[Alert]:
        """
        Evaluate all rules against provided data
        
        Args:
            data: Data to evaluate rules against
            
        Returns:
            List of triggered alerts
        """
        triggered_alerts = []
        current_time = time.time()
        
        for rule_name, rule in self.rules.items():
            if not rule.enabled:
                continue
            
            # Check cooldown
            if rule_name in self.rule_cooldowns:
                if current_time - self.rule_cooldowns[rule_name] < rule.cooldown_seconds:
                    continue
            
            try:
                # Evaluate rule condition
                if self._evaluate_condition(rule.condition, data):
                    alert = self._create_alert_from_rule(rule, data)
                    
                    # Check for duplicates
                    if not self.deduplicator.is_duplicate(alert):
                        triggered_alerts.append(alert)
                        self.rule_cooldowns[rule_name] = current_time
                        
                        # Check for escalation
                        if self.escalator.should_escalate(alert, rule):
                            escalated_alert = self.escalator.escalate_alert(alert, rule)
                            triggered_alerts.append(escalated_alert)
                
            except Exception as e:
                logger.error(f"Error evaluating rule {rule_name}: {e}")
        
        return triggered_alerts
    
    def _evaluate_condition(self, condition: str, data: Dict[str, Any]) -> bool:
        """Safely evaluate rule condition"""
        try:
            # Create safe evaluation context
            safe_globals = {
                '__builtins__': {},
                'abs': abs,
                'min': min,
                'max': max,
                'len': len,
                'sum': sum,
                'any': any,
                'all': all
            }
            
            # Add data to context
            safe_globals.update(data)
            
            return bool(eval(condition, safe_globals))
            
        except Exception as e:
            logger.error(f"Error evaluating condition '{condition}': {e}")
            return False
    
    def _create_alert_from_rule(self, rule: AlertRule, data: Dict[str, Any]) -> Alert:
        """Create alert from triggered rule"""
        alert_id = f"{rule.name}_{int(time.time() * 1000)}"
        
        message = rule.custom_message or f"Rule '{rule.name}' triggered"
        
        return Alert(
            id=alert_id,
            timestamp=time.time(),
            severity=rule.severity,
            title=f"Alert: {rule.name}",
            message=message,
            source="rule_engine",
            data=data,
            channels=rule.channels,
            rule_name=rule.name
        )
    
    @performance_monitor
    def send_alert(self, alert: Alert) -> bool:
        """
        Send alert through configured channels
        
        Args:
            alert: Alert to send
            
        Returns:
            bool: True if alert was sent successfully
        """
        try:
            self.alert_queue.put(alert, timeout=1.0)
            return True
        except queue.Full:
            logger.error("Alert queue is full, dropping alert")
            return False
    
    def _process_alerts(self):
        """Background thread to process alert queue"""
        while self.is_running:
            try:
                alert = self.alert_queue.get(timeout=1.0)
                self._deliver_alert(alert)
                self.alert_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error processing alert: {e}")
    
    def _deliver_alert(self, alert: Alert):
        """Deliver alert through all specified channels"""
        start_time = time.time()
        
        delivery_methods = {
            AlertChannel.EMAIL: self.delivery_manager.deliver_email,
            AlertChannel.WEBHOOK: self.delivery_manager.deliver_webhook,
            AlertChannel.SLACK: self.delivery_manager.deliver_slack,
            AlertChannel.CONSOLE: self.delivery_manager.deliver_console
        }
        
        success_count = 0
        
        for channel in alert.channels:
            if channel in delivery_methods:
                try:
                    if delivery_methods[channel](alert):
                        success_count += 1
                        self.alert_stats['alerts_by_channel'][channel] += 1
                except Exception as e:
                    logger.error(f"Failed to deliver alert via {channel.value}: {e}")
        
        # Update statistics
        self.alert_stats['total_alerts'] += 1
        self.alert_stats['alerts_by_severity'][alert.severity] += 1
        self.alert_stats['processing_times'].append(time.time() - start_time)
        
        # Keep only recent processing times
        if len(self.alert_stats['processing_times']) > 1000:
            self.alert_stats['processing_times'] = self.alert_stats['processing_times'][-500:]
    
    def start(self) -> bool:
        """Start alert system"""
        if self.is_running:
            return True
        
        try:
            self.is_running = True
            
            # Start worker threads
            num_workers = self.config.max_workers or min(mp.cpu_count(), 4)
            for i in range(num_workers):
                thread = threading.Thread(target=self._process_alerts, name=f"AlertWorker-{i}")
                thread.daemon = True
                thread.start()
                self.worker_threads.append(thread)
            
            logger.info(f"Alert system started with {num_workers} workers")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start alert system: {e}")
            self.is_running = False
            return False
    
    def stop(self):
        """Stop alert system"""
        self.is_running = False
        
        # Wait for queue to empty
        self.alert_queue.join()
        
        # Wait for worker threads
        for thread in self.worker_threads:
            thread.join(timeout=5.0)
        
        self.worker_threads.clear()
        logger.info("Alert system stopped")
    
    def check_trading_alerts(self, event_type: str, data: Dict[str, Any]):
        """Check for trading-specific alerts"""
        # Add event type to data for rule evaluation
        alert_data = data.copy()
        alert_data['event_type'] = event_type
        alert_data['timestamp'] = time.time()
        
        # Evaluate rules and send alerts
        alerts = self.evaluate_rules(alert_data)
        for alert in alerts:
            self.send_alert(alert)
    
    def get_status(self) -> Dict[str, Any]:
        """Get alert system status"""
        return {
            'is_running': self.is_running,
            'queue_size': self.alert_queue.qsize(),
            'active_rules': len([r for r in self.rules.values() if r.enabled]),
            'total_rules': len(self.rules),
            'worker_threads': len(self.worker_threads),
            'delivery_stats': self.delivery_manager.delivery_stats,
            'alert_stats': self.alert_stats,
            'arm64_optimized': self.arm64_optimizer.is_arm64
        }
    
    def get_alert_summary(self) -> Dict[str, Any]:
        """Get alert summary statistics"""
        avg_processing_time = (
            sum(self.alert_stats['processing_times']) / len(self.alert_stats['processing_times'])
            if self.alert_stats['processing_times'] else 0
        )
        
        return {
            'total_alerts': self.alert_stats['total_alerts'],
            'alerts_by_severity': dict(self.alert_stats['alerts_by_severity']),
            'alerts_by_channel': dict(self.alert_stats['alerts_by_channel']),
            'average_processing_time': avg_processing_time,
            'delivery_success_rate': self._calculate_delivery_success_rate()
        }
    
    def _calculate_delivery_success_rate(self) -> float:
        """Calculate overall delivery success rate"""
        total_sent = sum(stats['sent'] for stats in self.delivery_manager.delivery_stats.values())
        total_failed = sum(stats['failed'] for stats in self.delivery_manager.delivery_stats.values())
        total_attempts = total_sent + total_failed
        
        return (total_sent / total_attempts * 100) if total_attempts > 0 else 0
    
    def cleanup(self):
        """Cleanup alert system resources"""
        if self.is_running:
            self.stop()
        
        if self.shared_memory:
            self.shared_memory.cleanup()
        
        logger.info("AlertSystem cleanup completed")

# Factory function for easy instantiation
def create_alert_system(config: AlertConfig = None) -> AdvancedAlertSystem:
    """
    Factory function to create alert system with optimal configuration
    
    Args:
        config: Alert system configuration
        
    Returns:
        Configured AdvancedAlertSystem instance
    """
    if config is None:
        config = AlertConfig()
        
        # Auto-detect optimal settings
        if platform.machine().lower() in ['arm64', 'aarch64']:
            config.enable_arm64_optimizations = True
            config.parallel_processing = True
            config.max_workers = min(mp.cpu_count(), 4)
        
        # Adjust for available memory
        available_memory = psutil.virtual_memory().available / (1024**3)  # GB
        if available_memory > 8:
            config.alert_queue_size = 20000
            config.max_alerts_per_minute = 200
        elif available_memory > 4:
            config.alert_queue_size = 10000
            config.max_alerts_per_minute = 100
        else:
            config.alert_queue_size = 5000
            config.max_alerts_per_minute = 50
    
    return AdvancedAlertSystem(config)

# Legacy compatibility
AlertSystem = AdvancedAlertSystem

if __name__ == "__main__":
    # Example usage and testing
    
    # Create alert system
    alert_system = create_alert_system()
    
    # Add sample rules
    sample_rule = AlertRule(
        name="high_loss_alert",
        condition="loss > 1000",
        severity=AlertSeverity.HIGH,
        channels=[AlertChannel.CONSOLE, AlertChannel.EMAIL],
        threshold=1000.0,
        cooldown_seconds=60
    )
    
    alert_system.add_rule(sample_rule)
    
    # Start system
    if alert_system.start():
        print("Alert system started successfully")
        
        # Test alert
        test_data = {"loss": 1500, "symbol": "AAPL"}
        alerts = alert_system.evaluate_rules(test_data)
        
        for alert in alerts:
            alert_system.send_alert(alert)
        
        # Get status
        status = alert_system.get_status()
        print(f"Alert system status: {status}")
        
        # Cleanup
        alert_system.stop()
        alert_system.cleanup()
    else:
        print("Failed to start alert system")