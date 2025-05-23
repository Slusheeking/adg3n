"""
Enhanced trading module for Deep Momentum Trading System with ARM64 optimizations.

This module provides comprehensive trading capabilities including order execution,
position management, risk management, and real-time trading operations with
ARM64-specific optimizations for high-performance trading systems.
"""

import platform
from typing import Dict, Any, Optional, List
import logging

# ARM64 detection
IS_ARM64 = platform.machine().lower() in ['arm64', 'aarch64']

# Core trading components
from .alpaca_client import AlpacaClient, AlpacaConfig
from .execution_engine import ExecutionEngine, ExecutionConfig
from .order_manager import OrderManager, OrderConfig
from .position_manager import PositionManager, PositionConfig
from .trade_logger import TradeLogger, TradeLoggerConfig
from .trading_engine import TradingEngine, TradingConfig

# Trading exceptions
from .exceptions import (
    TradingError,
    OrderError,
    PositionError,
    ExecutionError,
    RiskError,
    BrokerError
)

# Trading data models
from .models import (
    Order,
    Position,
    Trade,
    OrderStatus,
    OrderType,
    PositionSide,
    RiskMetrics
)

# Performance monitoring
from ..utils.logger import get_logger
from ..utils.decorators import performance_monitor
from ..utils.constants import TRADING_CONSTANTS

# Initialize module logger
logger = get_logger(__name__)

class TradingSystemManager:
    """Central manager for the trading system with ARM64 optimizations."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.is_arm64 = IS_ARM64
        self._components = {}
        self._initialized = False
        
        logger.info(f"Initializing Trading System Manager (ARM64: {self.is_arm64})")
    
    @performance_monitor
    def initialize(self) -> bool:
        """Initialize all trading components."""
        try:
            # Initialize components with ARM64 optimizations
            self._components['alpaca_client'] = AlpacaClient(
                config=AlpacaConfig(**self.config.get('alpaca', {}))
            )
            
            self._components['order_manager'] = OrderManager(
                config=OrderConfig(**self.config.get('order_manager', {}))
            )
            
            self._components['position_manager'] = PositionManager(
                config=PositionConfig(**self.config.get('position_manager', {}))
            )
            
            self._components['execution_engine'] = ExecutionEngine(
                broker_client=self._components['alpaca_client'],
                order_manager=self._components['order_manager'],
                config=ExecutionConfig(**self.config.get('execution', {}))
            )
            
            self._components['trade_logger'] = TradeLogger(
                config=TradeLoggerConfig(**self.config.get('trade_logger', {}))
            )
            
            self._components['trading_engine'] = TradingEngine(
                execution_engine=self._components['execution_engine'],
                position_manager=self._components['position_manager'],
                trade_logger=self._components['trade_logger'],
                config=TradingConfig(**self.config.get('trading_engine', {}))
            )
            
            self._initialized = True
            logger.info("Trading system components initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize trading system: {e}")
            return False
    
    def get_component(self, name: str) -> Any:
        """Get trading system component by name."""
        if not self._initialized:
            raise RuntimeError("Trading system not initialized")
        
        return self._components.get(name)
    
    def get_trading_engine(self) -> TradingEngine:
        """Get the main trading engine."""
        return self.get_component('trading_engine')
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        if not self._initialized:
            return {"status": "not_initialized", "components": {}}
        
        status = {
            "status": "initialized",
            "arm64_optimized": self.is_arm64,
            "components": {}
        }
        
        for name, component in self._components.items():
            try:
                if hasattr(component, 'get_status'):
                    status["components"][name] = component.get_status()
                else:
                    status["components"][name] = {"status": "active"}
            except Exception as e:
                status["components"][name] = {"status": "error", "error": str(e)}
        
        return status
    
    def shutdown(self):
        """Shutdown all trading components."""
        logger.info("Shutting down trading system...")
        
        for name, component in self._components.items():
            try:
                if hasattr(component, 'shutdown'):
                    component.shutdown()
                logger.debug(f"Component {name} shutdown successfully")
            except Exception as e:
                logger.error(f"Error shutting down component {name}: {e}")
        
        self._components.clear()
        self._initialized = False
        logger.info("Trading system shutdown completed")

def get_trading_constants() -> Dict[str, Any]:
    """Get trading-specific constants with ARM64 optimizations."""
    constants = TRADING_CONSTANTS.copy()
    
    if IS_ARM64:
        # ARM64-specific optimizations
        constants.update({
            "ARM64_OPTIMIZED": True,
            "BATCH_SIZE_OPTIMIZATION": 64,  # Power of 2 for ARM64 SIMD
            "MEMORY_ALIGNMENT": 64,  # ARM64 cache line size
            "SIMD_VECTOR_SIZE": 16,  # ARM64 NEON vector size
        })
    
    return constants

def create_trading_system(config: Optional[Dict[str, Any]] = None) -> TradingSystemManager:
    """Create and initialize a complete trading system."""
    manager = TradingSystemManager(config)
    
    if not manager.initialize():
        raise RuntimeError("Failed to initialize trading system")
    
    return manager

def get_system_info() -> Dict[str, Any]:
    """Get trading system information."""
    return {
        "module": "trading",
        "version": "2.0.0",
        "arm64_support": IS_ARM64,
        "components": [
            "AlpacaClient",
            "ExecutionEngine", 
            "OrderManager",
            "PositionManager",
            "TradeLogger",
            "TradingEngine"
        ],
        "features": [
            "ARM64 optimizations",
            "Real-time execution",
            "Risk management",
            "Performance monitoring",
            "Comprehensive logging"
        ]
    }

# Log module initialization
logger.info(f"Trading module initialized (ARM64: {IS_ARM64})")

# Export all public components
__all__ = [
    # Core components
    "AlpacaClient",
    "AlpacaConfig",
    "ExecutionEngine", 
    "ExecutionConfig",
    "OrderManager",
    "OrderConfig",
    "PositionManager",
    "PositionConfig",
    "TradeLogger",
    "TradeLoggerConfig",
    "TradingEngine",
    "TradingConfig",
    
    # Exceptions
    "TradingError",
    "OrderError",
    "PositionError", 
    "ExecutionError",
    "RiskError",
    "BrokerError",
    
    # Data models
    "Order",
    "Position",
    "Trade",
    "OrderStatus",
    "OrderType",
    "PositionSide",
    "RiskMetrics",
    
    # System management
    "TradingSystemManager",
    "create_trading_system",
    "get_trading_constants",
    "get_system_info",
    
    # Constants
    "IS_ARM64"
]
