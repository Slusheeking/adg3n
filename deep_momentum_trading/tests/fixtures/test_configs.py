"""
Test configuration fixtures for Deep Momentum Trading System.

This module provides comprehensive configuration objects for testing
all components of the trading system with various scenarios.
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from datetime import datetime, timezone
import os

@dataclass
class TestModelConfig:
    """Test configuration for deep learning models."""
    input_size: int = 200
    hidden_size: int = 256  # Smaller for testing
    num_layers: int = 2     # Fewer layers for testing
    dropout: float = 0.1
    device: str = 'cpu'
    enable_arm64_optimizations: bool = True
    batch_size: int = 16    # Smaller batch for testing
    sequence_length: int = 30  # Shorter sequences for testing
    learning_rate: float = 0.001
    weight_decay: float = 1e-5
    gradient_clip_norm: float = 1.0

@dataclass
class TestRiskConfig:
    """Test configuration for risk management."""
    max_portfolio_var: float = 0.02
    max_portfolio_volatility: float = 0.20
    max_drawdown_limit: float = 0.10
    max_position_concentration: float = 0.005
    max_sector_concentration: float = 0.25
    max_illiquid_percentage: float = 10.0
    max_simultaneous_positions: int = 100  # Smaller for testing
    daily_capital_limit: float = 10000.0   # Smaller for testing
    min_position_value: float = 5.0
    max_position_value: float = 500.0      # Smaller for testing
    enable_real_time_monitoring: bool = False  # Disabled for testing
    risk_check_interval_seconds: float = 1.0   # Faster for testing
    enable_stress_testing: bool = False        # Disabled for testing
    enable_arm64_optimizations: bool = True
    enable_parallel_processing: bool = False  # Disabled for testing
    calculation_timeout_seconds: float = 5.0  # Shorter for testing
    enable_emergency_stop: bool = True
    emergency_var_threshold: float = 0.05
    emergency_drawdown_threshold: float = 0.15

@dataclass
class TestFeatureConfig:
    """Test configuration for feature engineering."""
    zmq_subscriber_port: int = 15555  # Different ports for testing
    zmq_publisher_port: int = 15556
    memory_cache_max_gb: float = 1.0  # Smaller for testing
    enable_arm64_optimizations: bool = True
    enable_parallel_processing: bool = False  # Disabled for testing
    max_workers: int = 2  # Fewer workers for testing
    chunk_size: int = 100  # Smaller chunks for testing
    enable_performance_monitoring: bool = False  # Disabled for testing
    enable_caching: bool = True
    cache_size: int = 10  # Smaller cache for testing
    feature_calculation_timeout: float = 5.0  # Shorter for testing
    enable_advanced_features: bool = True
    enable_real_time_features: bool = False  # Disabled for testing

@dataclass
class TestTradingConfig:
    """Test configuration for trading components."""
    # Alpaca configuration
    alpaca_api_key: str = "test_alpaca_key"
    alpaca_secret_key: str = "test_alpaca_secret"
    alpaca_base_url: str = "https://paper-api.alpaca.markets"
    enable_paper_trading: bool = True
    
    # Position management
    max_positions: int = 50  # Smaller for testing
    position_size_limit: float = 1000.0  # Smaller for testing
    enable_fractional_shares: bool = True
    
    # Order management
    default_order_type: str = "market"
    order_timeout_seconds: float = 30.0
    max_order_retries: int = 3
    enable_order_validation: bool = True
    
    # Risk controls
    enable_position_limits: bool = True
    enable_daily_loss_limit: bool = True
    daily_loss_limit: float = 500.0  # Smaller for testing
    
    # Performance
    enable_arm64_optimizations: bool = True
    enable_async_operations: bool = False  # Disabled for testing

@dataclass
class TestDataConfig:
    """Test configuration for data components."""
    # Polygon configuration
    polygon_api_key: str = "test_polygon_key"
    enable_real_time_data: bool = False  # Disabled for testing
    enable_historical_data: bool = True
    
    # Data storage
    storage_type: str = "memory"  # Use memory storage for testing
    database_url: str = "sqlite:///:memory:"
    redis_url: str = "redis://localhost:6379/15"  # Different DB for testing
    
    # Data processing
    enable_data_validation: bool = True
    enable_data_cleaning: bool = True
    max_missing_data_ratio: float = 0.1
    
    # Performance
    enable_caching: bool = False  # Disabled for testing
    cache_ttl_seconds: int = 60  # Shorter for testing
    enable_compression: bool = False  # Disabled for testing

@dataclass
class TestCommunicationConfig:
    """Test configuration for communication components."""
    # ZMQ configuration
    zmq_base_port: int = 15000  # Different base port for testing
    zmq_timeout_ms: int = 1000  # Shorter timeout for testing
    enable_encryption: bool = False  # Disabled for testing
    
    # Message handling
    max_message_size: int = 1024 * 1024  # 1MB for testing
    message_queue_size: int = 100  # Smaller queue for testing
    enable_message_persistence: bool = False  # Disabled for testing
    
    # Performance
    enable_compression: bool = False  # Disabled for testing
    compression_level: int = 1
    enable_batching: bool = False  # Disabled for testing

@dataclass
class TestMonitoringConfig:
    """Test configuration for monitoring components."""
    # Logging
    log_level: str = "WARNING"  # Reduce log noise during testing
    enable_file_logging: bool = False  # Disabled for testing
    enable_console_logging: bool = False  # Disabled for testing
    
    # Metrics
    enable_metrics_collection: bool = False  # Disabled for testing
    metrics_interval_seconds: float = 10.0  # Longer for testing
    
    # Alerts
    enable_alerts: bool = False  # Disabled for testing
    alert_cooldown_seconds: float = 60.0
    
    # Performance monitoring
    enable_performance_tracking: bool = False  # Disabled for testing
    performance_sample_rate: float = 0.1  # Lower rate for testing

class TestConfigManager:
    """Manager for test configurations."""
    
    def __init__(self):
        self.model_config = TestModelConfig()
        self.risk_config = TestRiskConfig()
        self.feature_config = TestFeatureConfig()
        self.trading_config = TestTradingConfig()
        self.data_config = TestDataConfig()
        self.communication_config = TestCommunicationConfig()
        self.monitoring_config = TestMonitoringConfig()
    
    def get_model_config(self, **overrides) -> TestModelConfig:
        """Get model configuration with optional overrides."""
        config = TestModelConfig()
        for key, value in overrides.items():
            if hasattr(config, key):
                setattr(config, key, value)
        return config
    
    def get_risk_config(self, **overrides) -> TestRiskConfig:
        """Get risk configuration with optional overrides."""
        config = TestRiskConfig()
        for key, value in overrides.items():
            if hasattr(config, key):
                setattr(config, key, value)
        return config
    
    def get_feature_config(self, **overrides) -> TestFeatureConfig:
        """Get feature configuration with optional overrides."""
        config = TestFeatureConfig()
        for key, value in overrides.items():
            if hasattr(config, key):
                setattr(config, key, value)
        return config
    
    def get_trading_config(self, **overrides) -> TestTradingConfig:
        """Get trading configuration with optional overrides."""
        config = TestTradingConfig()
        for key, value in overrides.items():
            if hasattr(config, key):
                setattr(config, key, value)
        return config
    
    def get_data_config(self, **overrides) -> TestDataConfig:
        """Get data configuration with optional overrides."""
        config = TestDataConfig()
        for key, value in overrides.items():
            if hasattr(config, key):
                setattr(config, key, value)
        return config
    
    def get_communication_config(self, **overrides) -> TestCommunicationConfig:
        """Get communication configuration with optional overrides."""
        config = TestCommunicationConfig()
        for key, value in overrides.items():
            if hasattr(config, key):
                setattr(config, key, value)
        return config
    
    def get_monitoring_config(self, **overrides) -> TestMonitoringConfig:
        """Get monitoring configuration with optional overrides."""
        config = TestMonitoringConfig()
        for key, value in overrides.items():
            if hasattr(config, key):
                setattr(config, key, value)
        return config
    
    def get_full_config(self) -> Dict[str, Any]:
        """Get complete configuration dictionary."""
        return {
            'model': self.model_config.__dict__,
            'risk': self.risk_config.__dict__,
            'feature': self.feature_config.__dict__,
            'trading': self.trading_config.__dict__,
            'data': self.data_config.__dict__,
            'communication': self.communication_config.__dict__,
            'monitoring': self.monitoring_config.__dict__
        }

# Predefined test scenarios
class TestScenarios:
    """Predefined test scenario configurations."""
    
    @staticmethod
    def get_minimal_config() -> TestConfigManager:
        """Get minimal configuration for basic testing."""
        config_manager = TestConfigManager()
        
        # Minimal model config
        config_manager.model_config.hidden_size = 64
        config_manager.model_config.num_layers = 1
        config_manager.model_config.batch_size = 4
        config_manager.model_config.sequence_length = 10
        
        # Minimal risk config
        config_manager.risk_config.max_simultaneous_positions = 10
        config_manager.risk_config.daily_capital_limit = 1000.0
        config_manager.risk_config.enable_real_time_monitoring = False
        
        # Minimal feature config
        config_manager.feature_config.memory_cache_max_gb = 0.1
        config_manager.feature_config.enable_parallel_processing = False
        config_manager.feature_config.cache_size = 5
        
        return config_manager
    
    @staticmethod
    def get_performance_config() -> TestConfigManager:
        """Get configuration for performance testing."""
        config_manager = TestConfigManager()
        
        # Performance model config
        config_manager.model_config.hidden_size = 512
        config_manager.model_config.num_layers = 4
        config_manager.model_config.batch_size = 32
        config_manager.model_config.sequence_length = 60
        config_manager.model_config.enable_arm64_optimizations = True
        
        # Performance risk config
        config_manager.risk_config.enable_parallel_processing = True
        config_manager.risk_config.max_simultaneous_positions = 500
        config_manager.risk_config.daily_capital_limit = 50000.0
        
        # Performance feature config
        config_manager.feature_config.enable_parallel_processing = True
        config_manager.feature_config.max_workers = 4
        config_manager.feature_config.chunk_size = 1000
        
        return config_manager
    
    @staticmethod
    def get_integration_config() -> TestConfigManager:
        """Get configuration for integration testing."""
        config_manager = TestConfigManager()
        
        # Integration-specific ports to avoid conflicts
        config_manager.feature_config.zmq_subscriber_port = 16555
        config_manager.feature_config.zmq_publisher_port = 16556
        config_manager.communication_config.zmq_base_port = 16000
        
        # Enable components needed for integration
        config_manager.risk_config.enable_real_time_monitoring = True
        config_manager.feature_config.enable_real_time_features = True
        config_manager.data_config.enable_real_time_data = True
        
        # Shorter timeouts for faster integration tests
        config_manager.risk_config.risk_check_interval_seconds = 0.5
        config_manager.communication_config.zmq_timeout_ms = 500
        
        return config_manager
    
    @staticmethod
    def get_stress_test_config() -> TestConfigManager:
        """Get configuration for stress testing."""
        config_manager = TestConfigManager()
        
        # Stress test model config
        config_manager.model_config.batch_size = 64
        config_manager.model_config.sequence_length = 120
        
        # Stress test risk config
        config_manager.risk_config.max_simultaneous_positions = 1000
        config_manager.risk_config.daily_capital_limit = 100000.0
        config_manager.risk_config.enable_stress_testing = True
        
        # Stress test feature config
        config_manager.feature_config.memory_cache_max_gb = 5.0
        config_manager.feature_config.chunk_size = 10000
        config_manager.feature_config.enable_parallel_processing = True
        config_manager.feature_config.max_workers = 8
        
        return config_manager

# Environment-specific configurations
class TestEnvironments:
    """Environment-specific test configurations."""
    
    @staticmethod
    def get_ci_config() -> TestConfigManager:
        """Get configuration for CI/CD environment."""
        config_manager = TestConfigManager()
        
        # CI-specific settings
        config_manager.model_config.device = 'cpu'  # No GPU in CI
        config_manager.model_config.batch_size = 8   # Smaller for CI
        
        # Disable resource-intensive features
        config_manager.risk_config.enable_parallel_processing = False
        config_manager.feature_config.enable_parallel_processing = False
        config_manager.data_config.enable_caching = False
        
        # Shorter timeouts for CI
        config_manager.feature_config.feature_calculation_timeout = 10.0
        config_manager.risk_config.calculation_timeout_seconds = 10.0
        
        return config_manager
    
    @staticmethod
    def get_local_dev_config() -> TestConfigManager:
        """Get configuration for local development."""
        config_manager = TestConfigManager()
        
        # Local development settings
        config_manager.monitoring_config.enable_console_logging = True
        config_manager.monitoring_config.log_level = "INFO"
        
        # Enable performance features for local testing
        config_manager.model_config.enable_arm64_optimizations = True
        config_manager.risk_config.enable_parallel_processing = True
        config_manager.feature_config.enable_parallel_processing = True
        
        return config_manager
    
    @staticmethod
    def get_docker_config() -> TestConfigManager:
        """Get configuration for Docker environment."""
        config_manager = TestConfigManager()
        
        # Docker-specific settings
        config_manager.data_config.redis_url = "redis://redis:6379/15"
        config_manager.communication_config.zmq_base_port = 15000
        
        # Resource constraints for Docker
        config_manager.feature_config.memory_cache_max_gb = 0.5
        config_manager.feature_config.max_workers = 2
        config_manager.risk_config.max_parallel_workers = 2
        
        return config_manager

@dataclass
class TestInfrastructureConfig:
    """Test configuration for infrastructure components."""
    # Health check settings
    health_check_interval_seconds: float = 5.0
    health_check_timeout_seconds: float = 2.0
    enable_health_monitoring: bool = False  # Disabled for testing
    
    # Process management
    max_processes: int = 2  # Fewer processes for testing
    process_restart_delay_seconds: float = 1.0
    enable_process_monitoring: bool = False  # Disabled for testing
    
    # Resource monitoring
    memory_threshold_percent: float = 80.0
    cpu_threshold_percent: float = 80.0
    disk_threshold_percent: float = 90.0
    enable_resource_alerts: bool = False  # Disabled for testing
    
    # Scheduler settings
    scheduler_tick_interval_seconds: float = 1.0
    max_concurrent_tasks: int = 5  # Fewer for testing
    enable_task_persistence: bool = False  # Disabled for testing

@dataclass
class TestStorageConfig:
    """Test configuration for storage components."""
    # HDF5 settings
    hdf5_compression: str = "gzip"
    hdf5_compression_level: int = 1  # Lower for testing
    hdf5_chunk_size: int = 1000  # Smaller for testing
    
    # Parquet settings
    parquet_compression: str = "snappy"
    parquet_row_group_size: int = 1000  # Smaller for testing
    
    # SQLite settings
    sqlite_journal_mode: str = "WAL"
    sqlite_synchronous: str = "NORMAL"
    sqlite_cache_size: int = 1000  # Smaller for testing
    
    # Memory storage settings
    memory_max_size_mb: int = 100  # Smaller for testing
    memory_eviction_policy: str = "LRU"
    
    # General settings
    enable_compression: bool = False  # Disabled for testing
    enable_encryption: bool = False  # Disabled for testing
    backup_enabled: bool = False  # Disabled for testing

@dataclass
class TestTrainingConfig:
    """Test configuration for training components."""
    # Training settings
    max_epochs: int = 5  # Fewer epochs for testing
    early_stopping_patience: int = 2  # Shorter patience for testing
    validation_frequency: int = 1  # More frequent for testing
    
    # Distributed training
    enable_distributed_training: bool = False  # Disabled for testing
    world_size: int = 1
    rank: int = 0
    
    # Hyperparameter tuning
    enable_hyperparameter_tuning: bool = False  # Disabled for testing
    max_trials: int = 5  # Fewer trials for testing
    tuning_timeout_seconds: float = 60.0  # Shorter for testing
    
    # Model selection
    enable_model_selection: bool = False  # Disabled for testing
    cross_validation_folds: int = 3  # Fewer folds for testing
    
    # Performance
    enable_mixed_precision: bool = False  # Disabled for testing
    enable_gradient_checkpointing: bool = False  # Disabled for testing
    dataloader_num_workers: int = 0  # No workers for testing

# Enhanced TestConfigManager with new components
class EnhancedTestConfigManager(TestConfigManager):
    """Enhanced test configuration manager with additional components."""
    
    def __init__(self):
        super().__init__()
        self.infrastructure_config = TestInfrastructureConfig()
        self.storage_config = TestStorageConfig()
        self.training_config = TestTrainingConfig()
    
    def get_infrastructure_config(self, **overrides) -> TestInfrastructureConfig:
        """Get infrastructure configuration with optional overrides."""
        config = TestInfrastructureConfig()
        for key, value in overrides.items():
            if hasattr(config, key):
                setattr(config, key, value)
        return config
    
    def get_storage_config(self, **overrides) -> TestStorageConfig:
        """Get storage configuration with optional overrides."""
        config = TestStorageConfig()
        for key, value in overrides.items():
            if hasattr(config, key):
                setattr(config, key, value)
        return config
    
    def get_training_config(self, **overrides) -> TestTrainingConfig:
        """Get training configuration with optional overrides."""
        config = TestTrainingConfig()
        for key, value in overrides.items():
            if hasattr(config, key):
                setattr(config, key, value)
        return config
    
    def get_full_config(self) -> Dict[str, Any]:
        """Get complete configuration dictionary."""
        base_config = super().get_full_config()
        base_config.update({
            'infrastructure': self.infrastructure_config.__dict__,
            'storage': self.storage_config.__dict__,
            'training': self.training_config.__dict__
        })
        return base_config

# Configuration validation
def validate_test_config(config: TestConfigManager) -> List[str]:
    """Validate test configuration and return list of issues."""
    issues = []
    
    # Model config validation
    if config.model_config.input_size <= 0:
        issues.append("Model input_size must be positive")
    if config.model_config.hidden_size <= 0:
        issues.append("Model hidden_size must be positive")
    if config.model_config.num_layers <= 0:
        issues.append("Model num_layers must be positive")
    if not 0 <= config.model_config.dropout <= 1:
        issues.append("Model dropout must be between 0 and 1")
    
    # Risk config validation
    if config.risk_config.max_portfolio_var <= 0:
        issues.append("Risk max_portfolio_var must be positive")
    if config.risk_config.daily_capital_limit <= 0:
        issues.append("Risk daily_capital_limit must be positive")
    if config.risk_config.min_position_value >= config.risk_config.max_position_value:
        issues.append("Risk min_position_value must be less than max_position_value")
    
    # Feature config validation
    if config.feature_config.memory_cache_max_gb <= 0:
        issues.append("Feature memory_cache_max_gb must be positive")
    if config.feature_config.max_workers <= 0:
        issues.append("Feature max_workers must be positive")
    
    # Port conflicts validation
    ports_used = [
        config.feature_config.zmq_subscriber_port,
        config.feature_config.zmq_publisher_port,
        config.communication_config.zmq_base_port
    ]
    if len(ports_used) != len(set(ports_used)):
        issues.append("Port conflicts detected in configuration")
    
    return issues

# Export main classes and functions
__all__ = [
    'TestModelConfig',
    'TestRiskConfig',
    'TestFeatureConfig',
    'TestTradingConfig',
    'TestDataConfig',
    'TestCommunicationConfig',
    'TestMonitoringConfig',
    'TestInfrastructureConfig',
    'TestStorageConfig',
    'TestTrainingConfig',
    'TestConfigManager',
    'EnhancedTestConfigManager',
    'TestScenarios',
    'TestEnvironments',
    'validate_test_config'
]
