"""
Enhanced risk management package for Deep Momentum Trading System with ARM64 optimizations.

This package provides comprehensive risk management capabilities including:
- Real-time correlation monitoring and analysis
- Liquidity risk assessment and monitoring
- Portfolio optimization with ARM64 acceleration
- Advanced risk management and position sizing
- Stress testing and scenario analysis
- Value-at-Risk (VaR) calculations with multiple methodologies

All components are optimized for ARM64 architectures while maintaining compatibility
with x86_64 systems, providing enterprise-grade risk management for high-frequency
trading operations.
"""

import platform
from typing import Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum

# Core risk management components
from .correlation_monitor import CorrelationMonitor, CorrelationConfig
from .liquidity_monitor import LiquidityMonitor, LiquidityConfig
from .portfolio_optimizer import PortfolioOptimizer, OptimizerConfig
from .risk_manager import RiskManager, RiskConfig
from .stress_testing import StressTester, StressTestConfig
from .var_calculator import VaRCalculator, VaRConfig

# Utilities
from ..utils.logger import get_logger
from ..utils.exceptions import RiskError, ConfigurationError

# ARM64 detection
IS_ARM64 = platform.machine().lower() in ['arm64', 'aarch64']

logger = get_logger(__name__)

class RiskSystemMode(Enum):
    """Risk system operating modes."""
    CONSERVATIVE = "conservative"
    MODERATE = "moderate"
    AGGRESSIVE = "aggressive"
    CUSTOM = "custom"

@dataclass
class RiskSystemConfig:
    """Comprehensive risk system configuration."""
    # System settings
    mode: RiskSystemMode = RiskSystemMode.MODERATE
    enable_arm64_optimizations: bool = True
    enable_real_time_monitoring: bool = True
    enable_shared_memory: bool = True
    
    # Risk limits
    max_portfolio_var: float = 0.02  # 2% daily VaR
    max_position_concentration: float = 0.10  # 10% max single position
    max_sector_concentration: float = 0.25  # 25% max sector exposure
    max_correlation_threshold: float = 0.8  # Maximum allowed correlation
    min_liquidity_score: float = 0.3  # Minimum liquidity requirement
    
    # Monitoring intervals
    correlation_update_seconds: float = 30.0
    liquidity_update_seconds: float = 60.0
    var_calculation_seconds: float = 300.0  # 5 minutes
    stress_test_hours: float = 24.0  # Daily stress tests
    
    # Performance settings
    max_concurrent_calculations: int = 8
    calculation_timeout_seconds: float = 30.0
    enable_caching: bool = True
    cache_ttl_seconds: int = 300

class RiskSystemManager:
    """Centralized risk system manager with ARM64 optimizations."""
    
    def __init__(self, config: Optional[RiskSystemConfig] = None):
        self.config = config or RiskSystemConfig()
        self.is_arm64 = IS_ARM64
        
        # Apply ARM64 optimizations
        if self.is_arm64 and self.config.enable_arm64_optimizations:
            self._apply_arm64_optimizations()
        
        # Initialize components
        self._initialize_components()
        
        # System state
        self._is_running = False
        self._components_initialized = False
        
        logger.info(f"RiskSystemManager initialized (ARM64: {self.is_arm64}, Mode: {self.config.mode.value})")
    
    def _apply_arm64_optimizations(self):
        """Apply ARM64-specific optimizations."""
        # Optimize update intervals for ARM64 performance
        self.config.correlation_update_seconds = max(15.0, self.config.correlation_update_seconds * 0.5)
        self.config.liquidity_update_seconds = max(30.0, self.config.liquidity_update_seconds * 0.75)
        self.config.var_calculation_seconds = max(120.0, self.config.var_calculation_seconds * 0.6)
        
        # Increase concurrent calculations for ARM64
        self.config.max_concurrent_calculations = min(16, self.config.max_concurrent_calculations * 2)
        
        logger.debug(f"Applied ARM64 optimizations: correlation={self.config.correlation_update_seconds}s, "
                    f"liquidity={self.config.liquidity_update_seconds}s, var={self.config.var_calculation_seconds}s")
    
    def _initialize_components(self):
        """Initialize all risk management components."""
        try:
            # Correlation monitor configuration
            correlation_config = CorrelationConfig(
                update_interval_seconds=self.config.correlation_update_seconds,
                max_correlation_threshold=self.config.max_correlation_threshold,
                enable_arm64_optimizations=self.config.enable_arm64_optimizations,
                enable_shared_memory=self.config.enable_shared_memory
            )
            self.correlation_monitor = CorrelationMonitor(correlation_config)
            
            # Liquidity monitor configuration
            liquidity_config = LiquidityConfig(
                update_interval_seconds=self.config.liquidity_update_seconds,
                min_liquidity_score=self.config.min_liquidity_score,
                enable_arm64_optimizations=self.config.enable_arm64_optimizations,
                enable_real_time_monitoring=self.config.enable_real_time_monitoring
            )
            self.liquidity_monitor = LiquidityMonitor(liquidity_config)
            
            # Portfolio optimizer configuration
            optimizer_config = OptimizerConfig(
                max_position_concentration=self.config.max_position_concentration,
                max_sector_concentration=self.config.max_sector_concentration,
                enable_arm64_optimizations=self.config.enable_arm64_optimizations,
                max_concurrent_optimizations=self.config.max_concurrent_calculations
            )
            self.portfolio_optimizer = PortfolioOptimizer(optimizer_config)
            
            # Risk manager configuration
            risk_config = RiskConfig(
                max_portfolio_var=self.config.max_portfolio_var,
                enable_real_time_monitoring=self.config.enable_real_time_monitoring,
                enable_arm64_optimizations=self.config.enable_arm64_optimizations,
                calculation_timeout_seconds=self.config.calculation_timeout_seconds
            )
            self.risk_manager = RiskManager(risk_config)
            
            # Stress testing configuration
            stress_config = StressTestConfig(
                test_interval_hours=self.config.stress_test_hours,
                enable_arm64_optimizations=self.config.enable_arm64_optimizations,
                max_concurrent_tests=self.config.max_concurrent_calculations
            )
            self.stress_tester = StressTester(stress_config)
            
            # VaR calculator configuration
            var_config = VaRConfig(
                calculation_interval_seconds=self.config.var_calculation_seconds,
                enable_arm64_optimizations=self.config.enable_arm64_optimizations,
                enable_caching=self.config.enable_caching,
                cache_ttl_seconds=self.config.cache_ttl_seconds
            )
            self.var_calculator = VaRCalculator(var_config)
            
            self._components_initialized = True
            logger.info("All risk management components initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize risk components: {e}")
            raise ConfigurationError(f"Risk system initialization failed: {e}")
    
    def start(self):
        """Start all risk management components."""
        if not self._components_initialized:
            raise RiskError("Components not initialized")
        
        if self._is_running:
            logger.warning("Risk system is already running")
            return
        
        try:
            # Start all components
            self.correlation_monitor.start()
            self.liquidity_monitor.start()
            self.portfolio_optimizer.start()
            self.risk_manager.start()
            self.stress_tester.start()
            self.var_calculator.start()
            
            self._is_running = True
            logger.info("Risk management system started successfully")
            
        except Exception as e:
            logger.error(f"Failed to start risk system: {e}")
            raise RiskError(f"Risk system startup failed: {e}")
    
    def stop(self):
        """Stop all risk management components."""
        if not self._is_running:
            logger.warning("Risk system is not running")
            return
        
        try:
            # Stop all components
            self.correlation_monitor.stop()
            self.liquidity_monitor.stop()
            self.portfolio_optimizer.stop()
            self.risk_manager.stop()
            self.stress_tester.stop()
            self.var_calculator.stop()
            
            self._is_running = False
            logger.info("Risk management system stopped successfully")
            
        except Exception as e:
            logger.error(f"Error stopping risk system: {e}")
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive risk system status."""
        if not self._components_initialized:
            return {"status": "not_initialized"}
        
        return {
            "status": "running" if self._is_running else "stopped",
            "mode": self.config.mode.value,
            "arm64_optimized": self.is_arm64 and self.config.enable_arm64_optimizations,
            "components": {
                "correlation_monitor": self.correlation_monitor.get_status(),
                "liquidity_monitor": self.liquidity_monitor.get_status(),
                "portfolio_optimizer": self.portfolio_optimizer.get_status(),
                "risk_manager": self.risk_manager.get_status(),
                "stress_tester": self.stress_tester.get_status(),
                "var_calculator": self.var_calculator.get_status()
            },
            "configuration": {
                "max_portfolio_var": self.config.max_portfolio_var,
                "max_position_concentration": self.config.max_position_concentration,
                "max_correlation_threshold": self.config.max_correlation_threshold,
                "min_liquidity_score": self.config.min_liquidity_score
            }
        }
    
    def get_risk_metrics(self) -> Dict[str, Any]:
        """Get current risk metrics from all components."""
        if not self._is_running:
            return {"error": "Risk system not running"}
        
        try:
            return {
                "correlation_metrics": self.correlation_monitor.get_current_correlations(),
                "liquidity_metrics": self.liquidity_monitor.get_liquidity_scores(),
                "portfolio_metrics": self.portfolio_optimizer.get_optimization_results(),
                "risk_metrics": self.risk_manager.get_risk_metrics(),
                "stress_test_results": self.stress_tester.get_latest_results(),
                "var_metrics": self.var_calculator.get_current_var()
            }
        except Exception as e:
            logger.error(f"Error getting risk metrics: {e}")
            return {"error": str(e)}

# Export all public components
__all__ = [
    # Core components
    "CorrelationMonitor",
    "LiquidityMonitor", 
    "PortfolioOptimizer",
    "RiskManager",
    "StressTester",
    "VaRCalculator",
    
    # Configuration classes
    "CorrelationConfig",
    "LiquidityConfig",
    "OptimizerConfig", 
    "RiskConfig",
    "StressTestConfig",
    "VaRConfig",
    
    # System management
    "RiskSystemManager",
    "RiskSystemConfig",
    "RiskSystemMode",
    
    # Constants
    "IS_ARM64"
]
