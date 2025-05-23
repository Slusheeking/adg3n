#!/usr/bin/env python3
"""
Enhanced Risk Monitor Startup Script with ARM64 Optimizations

This script provides comprehensive risk monitoring capabilities for the Deep Momentum Trading System
with ARM64-specific optimizations, real-time risk assessment, and automated alert systems.

Features:
- Real-time portfolio risk monitoring
- ARM64-optimized risk calculations
- Multi-dimensional risk metrics (VaR, CVaR, correlation, liquidity)
- Automated alert and circuit breaker systems
- Stress testing and scenario analysis
- Performance monitoring and reporting
"""

import os
import sys
import argparse
import asyncio
import signal
import time
import platform
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional
import logging
import json
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import multiprocessing as mp

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.risk.risk_manager import RiskManager
from src.risk.var_calculator import VaRCalculator
from src.risk.correlation_monitor import CorrelationMonitor
from src.risk.liquidity_monitor import LiquidityMonitor
from src.risk.stress_testing import StressTesting
from src.risk.portfolio_optimizer import PortfolioOptimizer
from src.trading.position_manager import PositionManager
from src.communication.zmq_subscriber import ZMQSubscriber
from src.communication.message_broker import MessageBroker
from src.infrastructure.health_check import HealthChecker
from src.monitoring.alert_system import AlertSystem
from src.monitoring.performance_tracker import PerformanceTracker
from src.utils.logger import get_logger
from src.utils.decorators import performance_monitor, error_handler
from src.utils.exceptions import RiskMonitorError
from config.settings import config_manager

logger = get_logger(__name__)

class ARM64RiskOptimizer:
    """ARM64-specific optimizations for risk calculations"""
    
    def __init__(self):
        self.is_arm64 = platform.machine().lower() in ['arm64', 'aarch64']
        self.cpu_count = mp.cpu_count()
        self.optimal_workers = self._calculate_optimal_workers()
        self.matrix_block_size = self._calculate_matrix_block_size()
        
    def _calculate_optimal_workers(self) -> int:
        """Calculate optimal number of workers for ARM64"""
        if self.is_arm64:
            # ARM64 benefits from moderate parallelism for matrix operations
            return min(self.cpu_count, 6)
        return min(self.cpu_count, 8)
    
    def _calculate_matrix_block_size(self) -> int:
        """Calculate optimal matrix block size for ARM64"""
        if self.is_arm64:
            # ARM64 cache-friendly block sizes for matrix operations
            return 64
        return 128

class RiskMonitorConfig:
    """Risk monitor configuration"""
    
    def __init__(self, **kwargs):
        # Risk limits
        self.max_portfolio_var = kwargs.get('max_portfolio_var', 0.02)  # 2% daily VaR
        self.max_position_weight = kwargs.get('max_position_weight', 0.10)  # 10% max position
        self.max_sector_concentration = kwargs.get('max_sector_concentration', 0.25)  # 25% max sector
        self.max_correlation_threshold = kwargs.get('max_correlation_threshold', 0.8)
        self.min_liquidity_score = kwargs.get('min_liquidity_score', 0.5)
        
        # Monitoring intervals
        self.risk_check_interval = kwargs.get('risk_check_interval', 30.0)  # 30 seconds
        self.var_calculation_interval = kwargs.get('var_calculation_interval', 300.0)  # 5 minutes
        self.correlation_update_interval = kwargs.get('correlation_update_interval', 600.0)  # 10 minutes
        self.stress_test_interval = kwargs.get('stress_test_interval', 3600.0)  # 1 hour
        
        # Data sources
        self.data_feed_port = kwargs.get('data_feed_port', 5555)
        self.position_feed_port = kwargs.get('position_feed_port', 5556)
        
        # Alert settings
        self.enable_alerts = kwargs.get('enable_alerts', True)
        self.alert_cooldown = kwargs.get('alert_cooldown', 300.0)  # 5 minutes
        self.enable_circuit_breakers = kwargs.get('enable_circuit_breakers', True)
        
        # Performance
        self.max_workers = kwargs.get('max_workers', None)
        self.enable_caching = kwargs.get('enable_caching', True)
        self.cache_ttl = kwargs.get('cache_ttl', 300.0)  # 5 minutes
        
        # Reporting
        self.enable_reporting = kwargs.get('enable_reporting', True)
        self.report_interval = kwargs.get('report_interval', 3600.0)  # 1 hour
        self.save_reports = kwargs.get('save_reports', True)

class RiskMonitorEngine:
    """
    Enhanced risk monitoring engine with ARM64 optimizations
    """
    
    def __init__(self, config: RiskMonitorConfig):
        self.config = config
        self.optimizer = ARM64RiskOptimizer()
        
        # Initialize risk components
        self.risk_manager = RiskManager()
        self.var_calculator = VaRCalculator()
        self.correlation_monitor = CorrelationMonitor()
        self.liquidity_monitor = LiquidityMonitor()
        self.stress_testing = StressTesting()
        self.portfolio_optimizer = PortfolioOptimizer()
        self.position_manager = PositionManager()
        
        # Initialize infrastructure
        self.health_checker = HealthChecker()
        self.alert_system = AlertSystem()
        self.performance_tracker = PerformanceTracker()
        self.message_broker = MessageBroker()
        
        # Data subscribers
        self.data_subscriber = ZMQSubscriber(port=config.data_feed_port)
        self.position_subscriber = ZMQSubscriber(port=config.position_feed_port)
        
        # State management
        self.is_running = False
        self.current_positions = {}
        self.market_data = {}
        self.risk_metrics = {}
        self.alert_history = {}
        self.last_calculations = {}
        
        # Statistics
        self.stats = {
            'risk_checks': 0,
            'var_calculations': 0,
            'alerts_sent': 0,
            'circuit_breakers_triggered': 0,
            'start_time': None
        }
        
        # Signal handling
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        logger.info(f"RiskMonitorEngine initialized with ARM64 optimizations: {self.optimizer.is_arm64}")
    
    @performance_monitor
    @error_handler
    async def start_risk_monitor(self) -> Dict[str, Any]:
        """
        Start comprehensive risk monitoring
        
        Returns:
            Dict containing monitoring results
        """
        logger.info("Starting enhanced risk monitor...")
        self.stats['start_time'] = time.time()
        
        try:
            # Initialize connections
            await self._initialize_connections()
            
            # Start data feeds
            await self._start_data_feeds()
            
            # Start monitoring loops
            await self._start_monitoring_loops()
            
            # Start reporting
            if self.config.enable_reporting:
                await self._start_reporting()
            
            self.is_running = True
            logger.info("Risk monitor started successfully")
            
            # Keep running until stopped
            await self._run_main_loop()
            
            return {
                'status': 'stopped',
                'stats': self.stats,
                'uptime': time.time() - self.stats['start_time']
            }
            
        except Exception as e:
            logger.error(f"Risk monitor startup failed: {e}")
            await self._cleanup()
            raise RiskMonitorError(f"Risk monitor startup failed: {e}")
    
    async def _initialize_connections(self):
        """Initialize connections"""
        logger.info("Initializing risk monitor connections...")
        
        # Initialize message broker
        await self.message_broker.start()
        
        # Initialize data subscribers
        await self.data_subscriber.start()
        await self.position_subscriber.start()
        
        # Initialize risk components
        await self.risk_manager.initialize()
        await self.var_calculator.initialize()
        await self.correlation_monitor.initialize()
        await self.liquidity_monitor.initialize()
        
        logger.info("Risk monitor connections initialized")
    
    async def _start_data_feeds(self):
        """Start data feed subscriptions"""
        logger.info("Starting risk monitor data feeds...")
        
        # Subscribe to market data
        await self.data_subscriber.subscribe('trades', self._handle_trade_data)
        await self.data_subscriber.subscribe('quotes', self._handle_quote_data)
        await self.data_subscriber.subscribe('bars', self._handle_bar_data)
        
        # Subscribe to position updates
        await self.position_subscriber.subscribe('positions', self._handle_position_update)
        await self.position_subscriber.subscribe('orders', self._handle_order_update)
        
        logger.info("Risk monitor data feeds started")
    
    async def _start_monitoring_loops(self):
        """Start monitoring loops"""
        logger.info("Starting risk monitoring loops...")
        
        # Start main risk check loop
        asyncio.create_task(self._risk_check_loop())
        
        # Start VaR calculation loop
        asyncio.create_task(self._var_calculation_loop())
        
        # Start correlation monitoring loop
        asyncio.create_task(self._correlation_monitoring_loop())
        
        # Start stress testing loop
        asyncio.create_task(self._stress_testing_loop())
        
        # Start health monitoring
        asyncio.create_task(self._health_monitoring_loop())
        
        logger.info("Risk monitoring loops started")
    
    async def _handle_trade_data(self, trade_data: Dict[str, Any]):
        """Handle incoming trade data"""
        try:
            symbol = trade_data.get('symbol')
            if symbol:
                if symbol not in self.market_data:
                    self.market_data[symbol] = {'trades': [], 'quotes': [], 'bars': []}
                
                self.market_data[symbol]['trades'].append(trade_data)
                
                # Update liquidity metrics
                await self.liquidity_monitor.update_trade_data(symbol, trade_data)
                
        except Exception as e:
            logger.error(f"Error handling trade data: {e}")
    
    async def _handle_quote_data(self, quote_data: Dict[str, Any]):
        """Handle incoming quote data"""
        try:
            symbol = quote_data.get('symbol')
            if symbol:
                if symbol not in self.market_data:
                    self.market_data[symbol] = {'trades': [], 'quotes': [], 'bars': []}
                
                self.market_data[symbol]['quotes'].append(quote_data)
                
                # Update liquidity metrics
                await self.liquidity_monitor.update_quote_data(symbol, quote_data)
                
        except Exception as e:
            logger.error(f"Error handling quote data: {e}")
    
    async def _handle_bar_data(self, bar_data: Dict[str, Any]):
        """Handle incoming bar data"""
        try:
            symbol = bar_data.get('symbol')
            if symbol:
                if symbol not in self.market_data:
                    self.market_data[symbol] = {'trades': [], 'quotes': [], 'bars': []}
                
                self.market_data[symbol]['bars'].append(bar_data)
                
                # Update correlation data
                await self.correlation_monitor.update_price_data(symbol, bar_data)
                
        except Exception as e:
            logger.error(f"Error handling bar data: {e}")
    
    async def _handle_position_update(self, position_data: Dict[str, Any]):
        """Handle position updates"""
        try:
            symbol = position_data.get('symbol')
            if symbol:
                self.current_positions[symbol] = position_data
                
                # Update position manager
                await self.position_manager.update_position(symbol, position_data)
                
                # Trigger immediate risk check for position changes
                await self._check_position_risk(symbol, position_data)
                
        except Exception as e:
            logger.error(f"Error handling position update: {e}")
    
    async def _handle_order_update(self, order_data: Dict[str, Any]):
        """Handle order updates"""
        try:
            # Update order tracking for risk calculations
            await self.risk_manager.update_pending_orders(order_data)
            
        except Exception as e:
            logger.error(f"Error handling order update: {e}")
    
    async def _risk_check_loop(self):
        """Main risk checking loop"""
        while self.is_running:
            try:
                await self._perform_risk_check()
                self.stats['risk_checks'] += 1
                
                await asyncio.sleep(self.config.risk_check_interval)
                
            except Exception as e:
                logger.error(f"Error in risk check loop: {e}")
                await asyncio.sleep(30)
    
    async def _perform_risk_check(self):
        """Perform comprehensive risk check"""
        try:
            # Calculate portfolio metrics
            portfolio_metrics = await self._calculate_portfolio_metrics()
            
            # Check position limits
            position_violations = await self._check_position_limits()
            
            # Check concentration limits
            concentration_violations = await self._check_concentration_limits()
            
            # Check correlation limits
            correlation_violations = await self._check_correlation_limits()
            
            # Check liquidity requirements
            liquidity_violations = await self._check_liquidity_requirements()
            
            # Aggregate risk metrics
            self.risk_metrics = {
                'timestamp': time.time(),
                'portfolio_metrics': portfolio_metrics,
                'position_violations': position_violations,
                'concentration_violations': concentration_violations,
                'correlation_violations': correlation_violations,
                'liquidity_violations': liquidity_violations
            }
            
            # Handle violations
            await self._handle_risk_violations()
            
        except Exception as e:
            logger.error(f"Error performing risk check: {e}")
    
    async def _calculate_portfolio_metrics(self) -> Dict[str, Any]:
        """Calculate portfolio-level risk metrics"""
        try:
            if not self.current_positions:
                return {}
            
            # Calculate portfolio value
            total_value = sum(
                pos.get('market_value', 0) for pos in self.current_positions.values()
            )
            
            # Calculate position weights
            weights = {}
            for symbol, position in self.current_positions.items():
                market_value = position.get('market_value', 0)
                weights[symbol] = market_value / total_value if total_value > 0 else 0
            
            # Calculate portfolio beta
            portfolio_beta = await self._calculate_portfolio_beta(weights)
            
            # Calculate portfolio volatility
            portfolio_volatility = await self._calculate_portfolio_volatility(weights)
            
            return {
                'total_value': total_value,
                'position_count': len(self.current_positions),
                'weights': weights,
                'beta': portfolio_beta,
                'volatility': portfolio_volatility
            }
            
        except Exception as e:
            logger.error(f"Error calculating portfolio metrics: {e}")
            return {}
    
    async def _calculate_portfolio_beta(self, weights: Dict[str, float]) -> float:
        """Calculate portfolio beta"""
        try:
            # Simplified beta calculation - would use actual beta data in production
            weighted_beta = 0.0
            for symbol, weight in weights.items():
                # Default beta of 1.0 for stocks, 0.0 for cash
                beta = 1.0 if weight > 0 else 0.0
                weighted_beta += weight * beta
            
            return weighted_beta
            
        except Exception as e:
            logger.error(f"Error calculating portfolio beta: {e}")
            return 1.0
    
    async def _calculate_portfolio_volatility(self, weights: Dict[str, float]) -> float:
        """Calculate portfolio volatility"""
        try:
            # Get correlation matrix
            correlation_matrix = await self.correlation_monitor.get_correlation_matrix(
                list(weights.keys())
            )
            
            if correlation_matrix is None:
                return 0.0
            
            # Calculate portfolio variance using ARM64 optimizations
            portfolio_variance = await self._calculate_portfolio_variance_arm64(
                weights, correlation_matrix
            )
            
            return np.sqrt(portfolio_variance)
            
        except Exception as e:
            logger.error(f"Error calculating portfolio volatility: {e}")
            return 0.0
    
    async def _calculate_portfolio_variance_arm64(self, weights: Dict[str, float], 
                                                correlation_matrix: np.ndarray) -> float:
        """Calculate portfolio variance with ARM64 optimizations"""
        try:
            symbols = list(weights.keys())
            weight_vector = np.array([weights[symbol] for symbol in symbols])
            
            # Use ARM64-optimized matrix operations
            if self.optimizer.is_arm64:
                # Block-wise matrix multiplication for better cache performance
                block_size = self.optimizer.matrix_block_size
                portfolio_variance = 0.0
                
                for i in range(0, len(weight_vector), block_size):
                    for j in range(0, len(weight_vector), block_size):
                        i_end = min(i + block_size, len(weight_vector))
                        j_end = min(j + block_size, len(weight_vector))
                        
                        w_block_i = weight_vector[i:i_end]
                        w_block_j = weight_vector[j:j_end]
                        corr_block = correlation_matrix[i:i_end, j:j_end]
                        
                        portfolio_variance += np.dot(w_block_i, np.dot(corr_block, w_block_j))
            else:
                # Standard calculation
                portfolio_variance = np.dot(weight_vector, np.dot(correlation_matrix, weight_vector))
            
            return portfolio_variance
            
        except Exception as e:
            logger.error(f"Error calculating portfolio variance: {e}")
            return 0.0
    
    async def _check_position_limits(self) -> List[Dict[str, Any]]:
        """Check position size limits"""
        violations = []
        
        try:
            for symbol, position in self.current_positions.items():
                market_value = position.get('market_value', 0)
                total_portfolio_value = sum(
                    pos.get('market_value', 0) for pos in self.current_positions.values()
                )
                
                if total_portfolio_value > 0:
                    weight = abs(market_value) / total_portfolio_value
                    
                    if weight > self.config.max_position_weight:
                        violations.append({
                            'type': 'position_limit',
                            'symbol': symbol,
                            'current_weight': weight,
                            'limit': self.config.max_position_weight,
                            'severity': 'high' if weight > self.config.max_position_weight * 1.5 else 'medium'
                        })
            
        except Exception as e:
            logger.error(f"Error checking position limits: {e}")
        
        return violations
    
    async def _check_concentration_limits(self) -> List[Dict[str, Any]]:
        """Check sector/asset concentration limits"""
        violations = []
        
        try:
            # Group positions by sector (simplified - would use actual sector data)
            sector_exposure = {}
            total_value = sum(
                abs(pos.get('market_value', 0)) for pos in self.current_positions.values()
            )
            
            for symbol, position in self.current_positions.items():
                # Simplified sector mapping
                sector = self._get_symbol_sector(symbol)
                market_value = abs(position.get('market_value', 0))
                
                if sector not in sector_exposure:
                    sector_exposure[sector] = 0
                sector_exposure[sector] += market_value
            
            # Check sector limits
            for sector, exposure in sector_exposure.items():
                if total_value > 0:
                    concentration = exposure / total_value
                    
                    if concentration > self.config.max_sector_concentration:
                        violations.append({
                            'type': 'concentration_limit',
                            'sector': sector,
                            'current_concentration': concentration,
                            'limit': self.config.max_sector_concentration,
                            'severity': 'high' if concentration > self.config.max_sector_concentration * 1.2 else 'medium'
                        })
            
        except Exception as e:
            logger.error(f"Error checking concentration limits: {e}")
        
        return violations
    
    def _get_symbol_sector(self, symbol: str) -> str:
        """Get sector for symbol (simplified mapping)"""
        # Simplified sector mapping - would use actual sector data in production
        tech_symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'QQQ']
        finance_symbols = ['JPM', 'BAC', 'WFC', 'GS', 'MS']
        
        if symbol in tech_symbols:
            return 'Technology'
        elif symbol in finance_symbols:
            return 'Financials'
        elif symbol in ['SPY', 'IWM']:
            return 'Broad Market'
        else:
            return 'Other'
    
    async def _check_correlation_limits(self) -> List[Dict[str, Any]]:
        """Check correlation limits"""
        violations = []
        
        try:
            symbols = list(self.current_positions.keys())
            if len(symbols) < 2:
                return violations
            
            correlation_matrix = await self.correlation_monitor.get_correlation_matrix(symbols)
            if correlation_matrix is None:
                return violations
            
            # Check pairwise correlations
            for i, symbol1 in enumerate(symbols):
                for j, symbol2 in enumerate(symbols[i+1:], i+1):
                    correlation = correlation_matrix[i, j]
                    
                    if abs(correlation) > self.config.max_correlation_threshold:
                        violations.append({
                            'type': 'correlation_limit',
                            'symbol1': symbol1,
                            'symbol2': symbol2,
                            'correlation': correlation,
                            'limit': self.config.max_correlation_threshold,
                            'severity': 'medium'
                        })
            
        except Exception as e:
            logger.error(f"Error checking correlation limits: {e}")
        
        return violations
    
    async def _check_liquidity_requirements(self) -> List[Dict[str, Any]]:
        """Check liquidity requirements"""
        violations = []
        
        try:
            for symbol, position in self.current_positions.items():
                liquidity_score = await self.liquidity_monitor.get_liquidity_score(symbol)
                
                if liquidity_score < self.config.min_liquidity_score:
                    violations.append({
                        'type': 'liquidity_requirement',
                        'symbol': symbol,
                        'liquidity_score': liquidity_score,
                        'requirement': self.config.min_liquidity_score,
                        'severity': 'high' if liquidity_score < self.config.min_liquidity_score * 0.5 else 'medium'
                    })
            
        except Exception as e:
            logger.error(f"Error checking liquidity requirements: {e}")
        
        return violations
    
    async def _handle_risk_violations(self):
        """Handle risk violations"""
        try:
            all_violations = []
            
            # Collect all violations
            for violation_type in ['position_violations', 'concentration_violations', 
                                 'correlation_violations', 'liquidity_violations']:
                violations = self.risk_metrics.get(violation_type, [])
                all_violations.extend(violations)
            
            if not all_violations:
                return
            
            # Group by severity
            high_severity = [v for v in all_violations if v.get('severity') == 'high']
            medium_severity = [v for v in all_violations if v.get('severity') == 'medium']
            
            # Handle high severity violations
            if high_severity and self.config.enable_circuit_breakers:
                await self._trigger_circuit_breaker(high_severity)
            
            # Send alerts
            if self.config.enable_alerts:
                await self._send_risk_alerts(all_violations)
            
        except Exception as e:
            logger.error(f"Error handling risk violations: {e}")
    
    async def _trigger_circuit_breaker(self, violations: List[Dict[str, Any]]):
        """Trigger circuit breaker for high severity violations"""
        try:
            logger.warning(f"Triggering circuit breaker for {len(violations)} high severity violations")
            
            # Send emergency alert
            await self.alert_system.send_alert(
                level='critical',
                message=f'Circuit breaker triggered: {len(violations)} high severity risk violations',
                context={'violations': violations}
            )
            
            # Notify trading system to halt
            await self.message_broker.publish('risk_events', {
                'type': 'circuit_breaker',
                'timestamp': time.time(),
                'violations': violations,
                'action': 'halt_trading'
            })
            
            self.stats['circuit_breakers_triggered'] += 1
            
        except Exception as e:
            logger.error(f"Error triggering circuit breaker: {e}")
    
    async def _send_risk_alerts(self, violations: List[Dict[str, Any]]):
        """Send risk alerts"""
        try:
            current_time = time.time()
            
            for violation in violations:
                violation_key = f"{violation['type']}_{violation.get('symbol', 'portfolio')}"
                
                # Check alert cooldown
                last_alert = self.alert_history.get(violation_key, 0)
                if current_time - last_alert < self.config.alert_cooldown:
                    continue
                
                # Send alert
                await self.alert_system.send_alert(
                    level=violation.get('severity', 'medium'),
                    message=f"Risk violation: {violation['type']}",
                    context=violation
                )
                
                self.alert_history[violation_key] = current_time
                self.stats['alerts_sent'] += 1
            
        except Exception as e:
            logger.error(f"Error sending risk alerts: {e}")
    
    async def _check_position_risk(self, symbol: str, position_data: Dict[str, Any]):
        """Check risk for specific position change"""
        try:
            # Immediate position size check
            market_value = position_data.get('market_value', 0)
            total_portfolio_value = sum(
                pos.get('market_value', 0) for pos in self.current_positions.values()
            )
            
            if total_portfolio_value > 0:
                weight = abs(market_value) / total_portfolio_value
                
                if weight > self.config.max_position_weight:
                    await self.alert_system.send_alert(
                        level='high',
                        message=f'Position limit exceeded for {symbol}',
                        context={
                            'symbol': symbol,
                            'weight': weight,
                            'limit': self.config.max_position_weight
                        }
                    )
            
        except Exception as e:
            logger.error(f"Error checking position risk for {symbol}: {e}")
    
    async def _var_calculation_loop(self):
        """VaR calculation loop"""
        while self.is_running:
            try:
                await self._calculate_var()
                self.stats['var_calculations'] += 1
                
                await asyncio.sleep(self.config.var_calculation_interval)
                
            except Exception as e:
                logger.error(f"Error in VaR calculation loop: {e}")
                await asyncio.sleep(300)
    
    async def _calculate_var(self):
        """Calculate Value at Risk"""
        try:
            if not self.current_positions:
                return
            
            # Calculate portfolio VaR
            portfolio_var = await self.var_calculator.calculate_portfolio_var(
                self.current_positions,
                confidence_level=0.95,
                time_horizon=1
            )
            
            # Check VaR limit
            if portfolio_var > self.config.max_portfolio_var:
                await self.alert_system.send_alert(
                    level='high',
                    message=f'Portfolio VaR limit exceeded: {portfolio_var:.4f}',
                    context={
                        'current_var': portfolio_var,
                        'limit': self.config.max_portfolio_var
                    }
                )
            
            # Store VaR metrics
            self.last_calculations['var'] = {
                'timestamp': time.time(),
                'portfolio_var': portfolio_var
            }
            
        except Exception as e:
            logger.error(f"Error calculating VaR: {e}")
    
    async def _correlation_monitoring_loop(self):
        """Correlation monitoring loop"""
        while self.is_running:
            try:
                await self.correlation_monitor.update_correlations()
                
                await asyncio.sleep(self.config.correlation_update_interval)
                
            except Exception as e:
                logger.error(f"Error in correlation monitoring loop: {e}")
                await asyncio.sleep(600)
    
    async def _stress_testing_loop(self):
        """Stress testing loop"""
        while self.is_running:
            try:
                await self._perform_stress_tests()
                
                await asyncio.sleep(self.config.stress_test_interval)
                
            except Exception as e:
                logger.error(f"Error in stress testing loop: {e}")
                await asyncio.sleep(1800)
    
    async def _perform_stress_tests(self):
        """Perform stress tests"""
        try:
            if not self.current_positions:
                return
            
            # Market crash scenario
            crash_results = await self.stress_testing.market_crash_scenario(
                self.current_positions,
                market_drop=0.20
            )
            
            # Interest rate shock
            rate_shock_results = await self.stress_testing.interest_rate_shock(
                self.current_positions,
                rate_change=0.02
            )
            
            # Store stress test results
            self.last_calculations['stress_tests'] = {
                'timestamp': time.time(),
                'market_crash': crash_results,
                'rate_shock': rate_shock_results
            }
            
        except Exception as e:
            logger.error(f"Error performing stress tests: {e}")
    
    async def _health_monitoring_loop(self):
        """Health monitoring loop"""
        while self.is_running:
            try:
                # Check component health
                health_status = await self._check_component_health()
                
                if not health_status['overall_healthy']:
                    await self.alert_system.send_alert(
                        level='warning',
                        message='Risk monitor health check failed',
                        context=health_status
                    )
                
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"Error in health monitoring loop: {e}")
                await asyncio.sleep(120)
    
    async def _check_component_health(self) -> Dict[str, Any]:
        """Check health of risk monitor components"""
        try:
            health_status = {
                'risk_manager': await self.risk_manager.health_check(),
                'var_calculator': await self.var_calculator.health_check(),
                'correlation_monitor': await self.correlation_monitor.health_check(),
                'liquidity_monitor': await self.liquidity_monitor.health_check(),
                'data_subscriber': await self.data_subscriber.health_check(),
                'position_subscriber': await self.position_subscriber.health_check()
            }
            
            health_status['overall_healthy'] = all(health_status.values())
            
            return health_status
            
        except Exception as e:
            logger.error(f"Error checking component health: {e}")
            return {'overall_healthy': False, 'error': str(e)}
    
    async def _start_reporting(self):
        """Start risk reporting"""
        logger.info("Starting risk reporting...")
        
        asyncio.create_task(self._reporting_loop())
        
        logger.info("Risk reporting started")
    
    async def _reporting_loop(self):
        """Risk reporting loop"""
        while self.is_running:
            try:
                await self._generate_risk_report()
                
                await asyncio.sleep(self.config.report_interval)
                
            except Exception as e:
                logger.error(f"Error in reporting loop: {e}")
                await asyncio.sleep(1800)
    
    async def _generate_risk_report(self):
        """Generate comprehensive risk report"""
        try:
            report = {
                'timestamp': time.time(),
                'portfolio_metrics': self.risk_metrics.get('portfolio_metrics', {}),
                'var_metrics': self.last_calculations.get('var', {}),
                'stress_test_results': self.last_calculations.get('stress_tests', {}),
                'violation_summary': self._summarize_violations(),
                'performance_stats': self.stats.copy()
            }
            
            # Save report if enabled
            if self.config.save_reports:
                await self._save_risk_report(report)
            
            # Track performance
            await self.performance_tracker.record_metrics('risk_monitor', report)
            
        except Exception as e:
            logger.error(f"Error generating risk report: {e}")
    
    def _summarize_violations(self) -> Dict[str, Any]:
        """Summarize current violations"""
        try:
            summary = {
                'total_violations': 0,
                'by_type': {},
                'by_severity': {'high': 0, 'medium': 0, 'low': 0}
            }
            
            for violation_type in ['position_violations', 'concentration_violations', 
                                 'correlation_violations', 'liquidity_violations']:
                violations = self.risk_metrics.get(violation_type, [])
                summary['total_violations'] += len(violations)
                summary['by_type'][violation_type] = len(violations)
                
                for violation in violations:
                    severity = violation.get('severity', 'medium')
                    summary['by_severity'][severity] += 1
            
            return summary
            
        except Exception as e:
            logger.error(f"Error summarizing violations: {e}")
            return {}
    
    async def _save_risk_report(self, report: Dict[str, Any]):
        """Save risk report to file"""
        try:
            timestamp = datetime.fromtimestamp(report['timestamp'])
            filename = f"risk_report_{timestamp.strftime('%Y%m%d_%H%M%S')}.json"
            filepath = Path("reports") / "risk" / filename
            
            # Create directory if it doesn't exist
            filepath.parent.mkdir(parents=True, exist_ok=True)
            
            with open(filepath, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
        except Exception as e:
            logger.error(f"Error saving risk report: {e}")
    
    async def _run_main_loop(self):
        """Main event loop"""
        logger.info("Risk monitor running... Press Ctrl+C to stop")
        
        try:
            while self.is_running:
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            logger.info("Shutdown signal received")
        finally:
            await self._cleanup()
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        logger.info(f"Received signal {signum}, shutting down...")
        self.is_running = False
    
    async def _cleanup(self):
        """Cleanup resources"""
        logger.info("Cleaning up risk monitor...")
        
        self.is_running = False
        
        # Close subscribers
        if self.data_subscriber:
            await self.data_subscriber.stop()
        
        if self.position_subscriber:
            await self.position_subscriber.stop()
        
        # Close message broker
        if self.message_broker:
            await self.message_broker.stop()
        
        logger.info("Risk monitor cleanup completed")

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Enhanced Risk Monitor Startup Script')
    
    # Risk limits
    parser.add_argument('--max-var', type=float, default=0.02, help='Maximum portfolio VaR')
    parser.add_argument('--max-position', type=float, default=0.10, help='Maximum position weight')
    parser.add_argument('--max-sector', type=float, default=0.25, help='Maximum sector concentration')
    parser.add_argument('--max-correlation', type=float, default=0.8, help='Maximum correlation threshold')
    parser.add_argument('--min-liquidity', type=float, default=0.5, help='Minimum liquidity score')
    
    # Monitoring intervals
    parser.add_argument('--risk-interval', type=float, default=30.0, help='Risk check interval (seconds)')
    parser.add_argument('--var-interval', type=float, default=300.0, help='VaR calculation interval (seconds)')
    parser.add_argument('--correlation-interval', type=float, default=600.0, help='Correlation update interval (seconds)')
    parser.add_argument('--stress-interval', type=float, default=3600.0, help='Stress test interval (seconds)')
    
    # Data sources
    parser.add_argument('--data-port', type=int, default=5555, help='Data feed port')
    parser.add_argument('--position-port', type=int, default=5556, help='Position feed port')
    
    # Options
    parser.add_argument('--no-alerts', action='store_true', help='Disable alerts')
    parser.add_argument('--no-circuit-breakers', action='store_true', help='Disable circuit breakers')
    parser.add_argument('--no-reporting', action='store_true', help='Disable reporting')
    parser.add_argument('--max-workers', type=int, help='Maximum number of workers')
    parser.add_argument('--verbose', action='store_true', help='Verbose logging')
    
    return parser.parse_args()

async def main():
    """Main risk monitor function"""
    args = parse_arguments()
    
    # Setup logging
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Create configuration
    config = RiskMonitorConfig(
        max_portfolio_var=args.max_var,
        max_position_weight=args.max_position,
        max_sector_concentration=args.max_sector,
        max_correlation_threshold=args.max_correlation,
        min_liquidity_score=args.min_liquidity,
        risk_check_interval=args.risk_interval,
        var_calculation_interval=args.var_interval,
        correlation_update_interval=args.correlation_interval,
        stress_test_interval=args.stress_interval,
        data_feed_port=args.data_port,
        position_feed_port=args.position_port,
        enable_alerts=not args.no_alerts,
        enable_circuit_breakers=not args.no_circuit_breakers,
        enable_reporting=not args.no_reporting,
        max_workers=args.max_workers
    )
    
    # Initialize and start risk monitor
    engine = RiskMonitorEngine(config)
    
    try:
        print(f"Starting risk monitor...")
        print(f"Max Portfolio VaR: {config.max_portfolio_var:.2%}")
        print(f"Max Position Weight: {config.max_position_weight:.2%}")
        print(f"Alerts: {'enabled' if config.enable_alerts else 'disabled'}")
        print(f"Circuit Breakers: {'enabled' if config.enable_circuit_breakers else 'disabled'}")
        
        result = await engine.start_risk_monitor()
        
        # Print summary
        print(f"\n=== Risk Monitor Results ===")
        print(f"Status: {result['status']}")
        print(f"Uptime: {result['uptime']:.2f}s")
        print(f"Risk Checks: {result['stats']['risk_checks']:,}")
        print(f"VaR Calculations: {result['stats']['var_calculations']:,}")
        print(f"Alerts Sent: {result['stats']['alerts_sent']}")
        print(f"Circuit Breakers: {result['stats']['circuit_breakers_triggered']}")
        
    except Exception as e:
        logger.error(f"Risk monitor failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())