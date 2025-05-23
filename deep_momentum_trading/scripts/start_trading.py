#!/usr/bin/env python3
"""
Enhanced Trading System Startup Script with ARM64 Optimizations

This script provides comprehensive trading system startup capabilities for the Deep Momentum Trading System
with ARM64-specific optimizations, orchestrated component management, and production-ready features.

Features:
- Complete trading system orchestration
- ARM64-optimized execution engine
- Multi-component coordination (data, models, risk, execution)
- Health monitoring and automatic recovery
- Performance optimization and monitoring
- Production deployment support
"""

import os
import sys
import argparse
import asyncio
import signal
import time
import platform
import threading
import subprocess
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional
import logging
import json
import yaml
from concurrent.futures import ThreadPoolExecutor
import multiprocessing as mp

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.trading.trading_engine import TradingEngine
from src.trading.execution_engine import ExecutionEngine
from src.trading.order_manager import OrderManager
from src.trading.position_manager import PositionManager
from src.trading.alpaca_client import AlpacaClient
from src.models.model_registry import ModelRegistry
from src.models.ensemble_system import EnsembleSystem
from src.data.data_manager import DataManager
from src.risk.risk_manager import RiskManager
from src.communication.message_broker import MessageBroker
from src.infrastructure.process_manager import ProcessManager
from src.infrastructure.health_check import HealthChecker
from src.infrastructure.scheduler import Scheduler
from src.monitoring.alert_system import AlertSystem
from src.monitoring.performance_tracker import PerformanceTracker
from src.utils.logger import get_logger
from src.utils.decorators import performance_monitor, error_handler
from src.utils.exceptions import TradingSystemError
from config.settings import config_manager

logger = get_logger(__name__)

class ARM64TradingOptimizer:
    """ARM64-specific optimizations for trading system"""
    
    def __init__(self):
        self.is_arm64 = platform.machine().lower() in ['arm64', 'aarch64']
        self.cpu_count = mp.cpu_count()
        self.optimal_workers = self._calculate_optimal_workers()
        self.memory_optimization = self._get_memory_optimization()
        
    def _calculate_optimal_workers(self) -> int:
        """Calculate optimal number of workers for ARM64"""
        if self.is_arm64:
            # ARM64 benefits from moderate parallelism for trading operations
            return min(self.cpu_count, 4)
        return min(self.cpu_count, 6)
    
    def _get_memory_optimization(self) -> Dict[str, Any]:
        """Get ARM64 memory optimization settings"""
        if self.is_arm64:
            return {
                'batch_size': 32,
                'cache_size': 1024,
                'buffer_size': 4096
            }
        return {
            'batch_size': 64,
            'cache_size': 2048,
            'buffer_size': 8192
        }

class TradingSystemConfig:
    """Trading system configuration"""
    
    def __init__(self, **kwargs):
        # Trading parameters
        self.trading_mode = kwargs.get('trading_mode', 'live')  # live, paper, simulation
        self.symbols = kwargs.get('symbols', ['SPY', 'QQQ', 'IWM'])
        self.max_positions = kwargs.get('max_positions', 10)
        self.position_size = kwargs.get('position_size', 10000)  # USD
        
        # Model configuration
        self.model_ensemble = kwargs.get('model_ensemble', True)
        self.model_update_interval = kwargs.get('model_update_interval', 3600.0)  # 1 hour
        self.prediction_threshold = kwargs.get('prediction_threshold', 0.6)
        
        # Risk management
        self.enable_risk_management = kwargs.get('enable_risk_management', True)
        self.max_daily_loss = kwargs.get('max_daily_loss', 0.02)  # 2%
        self.max_position_risk = kwargs.get('max_position_risk', 0.01)  # 1%
        
        # Execution
        self.execution_delay = kwargs.get('execution_delay', 0.1)  # seconds
        self.order_timeout = kwargs.get('order_timeout', 30.0)  # seconds
        self.enable_smart_routing = kwargs.get('enable_smart_routing', True)
        
        # Data feeds
        self.enable_data_feed = kwargs.get('enable_data_feed', True)
        self.data_feed_port = kwargs.get('data_feed_port', 5555)
        
        # Monitoring
        self.enable_monitoring = kwargs.get('enable_monitoring', True)
        self.health_check_interval = kwargs.get('health_check_interval', 60.0)
        self.performance_tracking = kwargs.get('performance_tracking', True)
        
        # Infrastructure
        self.enable_scheduler = kwargs.get('enable_scheduler', True)
        self.auto_recovery = kwargs.get('auto_recovery', True)
        self.max_recovery_attempts = kwargs.get('max_recovery_attempts', 3)
        
        # Performance
        self.max_workers = kwargs.get('max_workers', None)
        self.enable_caching = kwargs.get('enable_caching', True)
        self.optimization_level = kwargs.get('optimization_level', 'high')

class TradingSystemOrchestrator:
    """
    Enhanced trading system orchestrator with ARM64 optimizations
    """
    
    def __init__(self, config: TradingSystemConfig):
        self.config = config
        self.optimizer = ARM64TradingOptimizer()
        
        # Core trading components
        self.trading_engine = TradingEngine()
        self.execution_engine = ExecutionEngine()
        self.order_manager = OrderManager()
        self.position_manager = PositionManager()
        self.alpaca_client = AlpacaClient()
        
        # Model components
        self.model_registry = ModelRegistry()
        self.ensemble_system = EnsembleSystem()
        
        # Data and risk
        self.data_manager = DataManager()
        self.risk_manager = RiskManager()
        
        # Infrastructure
        self.message_broker = MessageBroker()
        self.process_manager = ProcessManager()
        self.health_checker = HealthChecker()
        self.scheduler = Scheduler()
        self.alert_system = AlertSystem()
        self.performance_tracker = PerformanceTracker()
        
        # Component processes
        self.component_processes = {}
        self.component_health = {}
        
        # State management
        self.is_running = False
        self.startup_time = None
        self.last_health_check = None
        self.recovery_attempts = {}
        
        # Statistics
        self.stats = {
            'trades_executed': 0,
            'orders_placed': 0,
            'positions_opened': 0,
            'positions_closed': 0,
            'total_pnl': 0.0,
            'uptime': 0.0,
            'component_restarts': 0,
            'start_time': None
        }
        
        # Signal handling
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        logger.info(f"TradingSystemOrchestrator initialized with ARM64 optimizations: {self.optimizer.is_arm64}")
    
    @performance_monitor
    @error_handler
    async def start_trading_system(self) -> Dict[str, Any]:
        """
        Start comprehensive trading system
        
        Returns:
            Dict containing startup results
        """
        logger.info("Starting enhanced trading system...")
        self.stats['start_time'] = time.time()
        self.startup_time = time.time()
        
        try:
            # Pre-startup validation
            await self._validate_system_requirements()
            
            # Initialize core infrastructure
            await self._initialize_infrastructure()
            
            # Start external components
            await self._start_external_components()
            
            # Initialize trading components
            await self._initialize_trading_components()
            
            # Start trading engine
            await self._start_trading_engine()
            
            # Start monitoring and health checks
            await self._start_monitoring()
            
            # Start scheduler if enabled
            if self.config.enable_scheduler:
                await self._start_scheduler()
            
            self.is_running = True
            logger.info("Trading system started successfully")
            
            # Keep running until stopped
            await self._run_main_loop()
            
            return {
                'status': 'stopped',
                'stats': self.stats,
                'uptime': time.time() - self.stats['start_time']
            }
            
        except Exception as e:
            logger.error(f"Trading system startup failed: {e}")
            await self._cleanup()
            raise TradingSystemError(f"Trading system startup failed: {e}")
    
    async def _validate_system_requirements(self):
        """Validate system requirements before startup"""
        logger.info("Validating system requirements...")
        
        # Check configuration files
        required_configs = [
            'config/trading_config.yaml',
            'config/risk_config.yaml',
            'config/model_config.yaml'
        ]
        
        for config_file in required_configs:
            config_path = project_root / config_file
            if not config_path.exists():
                raise TradingSystemError(f"Required configuration file not found: {config_file}")
        
        # Check model files
        model_registry_path = project_root / "models"
        if not model_registry_path.exists():
            logger.warning("Model registry directory not found, creating...")
            model_registry_path.mkdir(parents=True, exist_ok=True)
        
        # Check data directory
        data_path = project_root / "data"
        if not data_path.exists():
            logger.warning("Data directory not found, creating...")
            data_path.mkdir(parents=True, exist_ok=True)
        
        # Validate trading mode
        if self.config.trading_mode not in ['live', 'paper', 'simulation']:
            raise TradingSystemError(f"Invalid trading mode: {self.config.trading_mode}")
        
        # Check broker credentials for live trading
        if self.config.trading_mode == 'live':
            if not os.getenv('ALPACA_API_KEY') or not os.getenv('ALPACA_SECRET_KEY'):
                raise TradingSystemError("Alpaca API credentials required for live trading")
        
        logger.info("System requirements validated")
    
    async def _initialize_infrastructure(self):
        """Initialize core infrastructure"""
        logger.info("Initializing core infrastructure...")
        
        # Initialize message broker
        await self.message_broker.start()
        
        # Initialize process manager
        await self.process_manager.start()
        
        # Initialize health checker
        await self.health_checker.start()
        
        # Initialize alert system
        await self.alert_system.start()
        
        # Initialize performance tracker
        await self.performance_tracker.start()
        
        logger.info("Core infrastructure initialized")
    
    async def _start_external_components(self):
        """Start external components (data feed, risk monitor)"""
        logger.info("Starting external components...")
        
        # Start data feed if enabled
        if self.config.enable_data_feed:
            await self._start_data_feed_component()
        
        # Start risk monitor if enabled
        if self.config.enable_risk_management:
            await self._start_risk_monitor_component()
        
        # Wait for components to be ready
        await self._wait_for_components_ready()
        
        logger.info("External components started")
    
    async def _start_data_feed_component(self):
        """Start data feed component"""
        try:
            script_path = project_root / "scripts" / "start_data_feed.py"
            
            cmd = [
                sys.executable, str(script_path),
                '--symbols'] + self.config.symbols + [
                '--zmq-port', str(self.config.data_feed_port),
                '--data-types', 'trades', 'quotes', 'bars'
            ]
            
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            self.component_processes['data_feed'] = process
            logger.info("Data feed component started")
            
        except Exception as e:
            logger.error(f"Failed to start data feed component: {e}")
            raise
    
    async def _start_risk_monitor_component(self):
        """Start risk monitor component"""
        try:
            script_path = project_root / "scripts" / "start_risk_monitor.py"
            
            cmd = [
                sys.executable, str(script_path),
                '--data-port', str(self.config.data_feed_port),
                '--max-var', str(self.config.max_daily_loss),
                '--risk-interval', '30'
            ]
            
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            self.component_processes['risk_monitor'] = process
            logger.info("Risk monitor component started")
            
        except Exception as e:
            logger.error(f"Failed to start risk monitor component: {e}")
            raise
    
    async def _wait_for_components_ready(self):
        """Wait for external components to be ready"""
        logger.info("Waiting for components to be ready...")
        
        max_wait_time = 60.0  # 60 seconds
        start_time = time.time()
        
        while time.time() - start_time < max_wait_time:
            all_ready = True
            
            # Check if data feed is ready
            if 'data_feed' in self.component_processes:
                if not await self._check_component_health('data_feed'):
                    all_ready = False
            
            # Check if risk monitor is ready
            if 'risk_monitor' in self.component_processes:
                if not await self._check_component_health('risk_monitor'):
                    all_ready = False
            
            if all_ready:
                logger.info("All components are ready")
                return
            
            await asyncio.sleep(2)
        
        raise TradingSystemError("Components failed to become ready within timeout")
    
    async def _check_component_health(self, component_name: str) -> bool:
        """Check health of a component"""
        try:
            process = self.component_processes.get(component_name)
            if not process:
                return False
            
            # Check if process is still running
            if process.returncode is not None:
                return False
            
            # Additional health checks could be added here
            # For now, just check if process is alive
            return True
            
        except Exception as e:
            logger.error(f"Error checking health of {component_name}: {e}")
            return False
    
    async def _initialize_trading_components(self):
        """Initialize trading components"""
        logger.info("Initializing trading components...")
        
        # Initialize data manager
        await self.data_manager.initialize()
        
        # Initialize broker client
        await self.alpaca_client.initialize(
            mode=self.config.trading_mode,
            api_key=os.getenv('ALPACA_API_KEY'),
            secret_key=os.getenv('ALPACA_SECRET_KEY')
        )
        
        # Initialize position manager
        await self.position_manager.initialize(self.alpaca_client)
        
        # Initialize order manager
        await self.order_manager.initialize(
            broker_client=self.alpaca_client,
            position_manager=self.position_manager
        )
        
        # Initialize execution engine
        await self.execution_engine.initialize(
            order_manager=self.order_manager,
            optimization_level=self.config.optimization_level
        )
        
        # Initialize model components
        await self.model_registry.initialize()
        
        if self.config.model_ensemble:
            await self.ensemble_system.initialize(self.model_registry)
        
        # Initialize risk manager
        if self.config.enable_risk_management:
            await self.risk_manager.initialize(
                position_manager=self.position_manager,
                max_daily_loss=self.config.max_daily_loss,
                max_position_risk=self.config.max_position_risk
            )
        
        logger.info("Trading components initialized")
    
    async def _start_trading_engine(self):
        """Start the main trading engine"""
        logger.info("Starting trading engine...")
        
        # Configure trading engine
        engine_config = {
            'symbols': self.config.symbols,
            'max_positions': self.config.max_positions,
            'position_size': self.config.position_size,
            'prediction_threshold': self.config.prediction_threshold,
            'execution_delay': self.config.execution_delay,
            'enable_smart_routing': self.config.enable_smart_routing
        }
        
        await self.trading_engine.initialize(
            execution_engine=self.execution_engine,
            model_system=self.ensemble_system if self.config.model_ensemble else None,
            risk_manager=self.risk_manager if self.config.enable_risk_management else None,
            data_manager=self.data_manager,
            config=engine_config
        )
        
        # Start trading engine
        asyncio.create_task(self.trading_engine.start())
        
        logger.info("Trading engine started")
    
    async def _start_monitoring(self):
        """Start monitoring and health checks"""
        logger.info("Starting monitoring...")
        
        # Start health monitoring loop
        asyncio.create_task(self._health_monitoring_loop())
        
        # Start performance monitoring
        if self.config.performance_tracking:
            asyncio.create_task(self._performance_monitoring_loop())
        
        # Start component monitoring
        asyncio.create_task(self._component_monitoring_loop())
        
        logger.info("Monitoring started")
    
    async def _start_scheduler(self):
        """Start scheduler for automated tasks"""
        logger.info("Starting scheduler...")
        
        # Schedule model updates
        if self.config.model_ensemble:
            await self.scheduler.schedule_recurring(
                'model_update',
                self._update_models,
                interval=self.config.model_update_interval
            )
        
        # Schedule daily reports
        await self.scheduler.schedule_daily(
            'daily_report',
            self._generate_daily_report,
            hour=17, minute=0  # 5 PM
        )
        
        # Schedule weekly model retraining
        await self.scheduler.schedule_weekly(
            'model_retrain',
            self._retrain_models,
            day=6, hour=20, minute=0  # Sunday 8 PM
        )
        
        await self.scheduler.start()
        
        logger.info("Scheduler started")
    
    async def _health_monitoring_loop(self):
        """Health monitoring loop"""
        while self.is_running:
            try:
                await self._perform_health_check()
                
                await asyncio.sleep(self.config.health_check_interval)
                
            except Exception as e:
                logger.error(f"Error in health monitoring loop: {e}")
                await asyncio.sleep(60)
    
    async def _perform_health_check(self):
        """Perform comprehensive health check"""
        try:
            self.last_health_check = time.time()
            
            # Check trading engine health
            trading_healthy = await self.trading_engine.health_check()
            
            # Check execution engine health
            execution_healthy = await self.execution_engine.health_check()
            
            # Check broker connection
            broker_healthy = await self.alpaca_client.health_check()
            
            # Check external components
            components_healthy = await self._check_all_components_health()
            
            # Overall health status
            overall_healthy = all([
                trading_healthy,
                execution_healthy,
                broker_healthy,
                components_healthy
            ])
            
            self.component_health = {
                'trading_engine': trading_healthy,
                'execution_engine': execution_healthy,
                'broker_connection': broker_healthy,
                'external_components': components_healthy,
                'overall': overall_healthy,
                'timestamp': time.time()
            }
            
            # Handle unhealthy components
            if not overall_healthy:
                await self._handle_health_issues()
            
        except Exception as e:
            logger.error(f"Error performing health check: {e}")
    
    async def _check_all_components_health(self) -> bool:
        """Check health of all external components"""
        try:
            for component_name in self.component_processes:
                if not await self._check_component_health(component_name):
                    logger.warning(f"Component {component_name} is unhealthy")
                    
                    # Attempt recovery if enabled
                    if self.config.auto_recovery:
                        await self._recover_component(component_name)
                    
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error checking component health: {e}")
            return False
    
    async def _handle_health_issues(self):
        """Handle health issues"""
        try:
            health_issues = []
            
            for component, healthy in self.component_health.items():
                if component != 'overall' and component != 'timestamp' and not healthy:
                    health_issues.append(component)
            
            if health_issues:
                await self.alert_system.send_alert(
                    level='warning',
                    message=f'Trading system health issues detected: {", ".join(health_issues)}',
                    context=self.component_health
                )
            
        except Exception as e:
            logger.error(f"Error handling health issues: {e}")
    
    async def _recover_component(self, component_name: str):
        """Recover a failed component"""
        try:
            logger.info(f"Attempting to recover component: {component_name}")
            
            # Track recovery attempts
            if component_name not in self.recovery_attempts:
                self.recovery_attempts[component_name] = 0
            
            self.recovery_attempts[component_name] += 1
            
            # Check if max recovery attempts exceeded
            if self.recovery_attempts[component_name] > self.config.max_recovery_attempts:
                logger.error(f"Max recovery attempts exceeded for {component_name}")
                await self.alert_system.send_alert(
                    level='critical',
                    message=f'Component {component_name} failed to recover after {self.config.max_recovery_attempts} attempts',
                    context={'component': component_name, 'attempts': self.recovery_attempts[component_name]}
                )
                return
            
            # Terminate existing process
            process = self.component_processes.get(component_name)
            if process:
                try:
                    process.terminate()
                    await asyncio.wait_for(process.wait(), timeout=10.0)
                except asyncio.TimeoutError:
                    process.kill()
                    await process.wait()
            
            # Restart component
            if component_name == 'data_feed':
                await self._start_data_feed_component()
            elif component_name == 'risk_monitor':
                await self._start_risk_monitor_component()
            
            # Wait for component to be ready
            await asyncio.sleep(5)
            
            if await self._check_component_health(component_name):
                logger.info(f"Component {component_name} recovered successfully")
                self.recovery_attempts[component_name] = 0  # Reset counter on success
                self.stats['component_restarts'] += 1
            else:
                logger.error(f"Component {component_name} recovery failed")
            
        except Exception as e:
            logger.error(f"Error recovering component {component_name}: {e}")
    
    async def _performance_monitoring_loop(self):
        """Performance monitoring loop"""
        while self.is_running:
            try:
                await self._collect_performance_metrics()
                
                await asyncio.sleep(300)  # Every 5 minutes
                
            except Exception as e:
                logger.error(f"Error in performance monitoring loop: {e}")
                await asyncio.sleep(300)
    
    async def _collect_performance_metrics(self):
        """Collect performance metrics"""
        try:
            # Update uptime
            if self.startup_time:
                self.stats['uptime'] = time.time() - self.startup_time
            
            # Get trading metrics
            trading_metrics = await self.trading_engine.get_metrics()
            self.stats.update(trading_metrics)
            
            # Get execution metrics
            execution_metrics = await self.execution_engine.get_metrics()
            
            # Get position metrics
            position_metrics = await self.position_manager.get_metrics()
            
            # Combine all metrics
            combined_metrics = {
                **self.stats,
                'execution_metrics': execution_metrics,
                'position_metrics': position_metrics,
                'health_status': self.component_health,
                'timestamp': time.time()
            }
            
            # Track performance
            await self.performance_tracker.record_metrics('trading_system', combined_metrics)
            
            # Log key metrics
            logger.info(f"Trading System Metrics - Uptime: {self.stats['uptime']:.1f}s, "
                       f"Trades: {self.stats['trades_executed']}, "
                       f"PnL: ${self.stats['total_pnl']:.2f}")
            
        except Exception as e:
            logger.error(f"Error collecting performance metrics: {e}")
    
    async def _component_monitoring_loop(self):
        """Component monitoring loop"""
        while self.is_running:
            try:
                # Monitor component resource usage
                for component_name, process in self.component_processes.items():
                    if process and process.returncode is None:
                        # Could add CPU/memory monitoring here
                        pass
                
                await asyncio.sleep(60)  # Every minute
                
            except Exception as e:
                logger.error(f"Error in component monitoring loop: {e}")
                await asyncio.sleep(60)
    
    async def _update_models(self):
        """Update models (scheduled task)"""
        try:
            logger.info("Updating models...")
            
            if self.config.model_ensemble:
                await self.ensemble_system.update_models()
            
            logger.info("Models updated successfully")
            
        except Exception as e:
            logger.error(f"Error updating models: {e}")
    
    async def _generate_daily_report(self):
        """Generate daily report (scheduled task)"""
        try:
            logger.info("Generating daily report...")
            
            # Get daily performance
            daily_metrics = await self.performance_tracker.get_daily_metrics()
            
            # Get position summary
            position_summary = await self.position_manager.get_position_summary()
            
            # Generate report
            report = {
                'date': datetime.now().strftime('%Y-%m-%d'),
                'performance': daily_metrics,
                'positions': position_summary,
                'system_stats': self.stats.copy()
            }
            
            # Save report
            report_path = project_root / "reports" / "daily" / f"trading_report_{datetime.now().strftime('%Y%m%d')}.json"
            report_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            logger.info(f"Daily report saved: {report_path}")
            
        except Exception as e:
            logger.error(f"Error generating daily report: {e}")
    
    async def _retrain_models(self):
        """Retrain models (scheduled task)"""
        try:
            logger.info("Starting model retraining...")
            
            # Run training script
            script_path = project_root / "scripts" / "train_models.py"
            
            cmd = [
                sys.executable, str(script_path),
                '--mode', 'scheduled',
                '--symbols'] + self.config.symbols
            
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            if process.returncode == 0:
                logger.info("Model retraining completed successfully")
            else:
                logger.error(f"Model retraining failed: {stderr.decode()}")
            
        except Exception as e:
            logger.error(f"Error retraining models: {e}")
    
    async def _run_main_loop(self):
        """Main event loop"""
        logger.info("Trading system running... Press Ctrl+C to stop")
        
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
        logger.info("Cleaning up trading system...")
        
        self.is_running = False
        
        # Stop trading engine
        if self.trading_engine:
            await self.trading_engine.stop()
        
        # Stop scheduler
        if self.scheduler:
            await self.scheduler.stop()
        
        # Close positions if in live mode
        if self.config.trading_mode == 'live':
            logger.info("Closing all positions...")
            await self.position_manager.close_all_positions()
        
        # Stop external components
        for component_name, process in self.component_processes.items():
            if process and process.returncode is None:
                logger.info(f"Stopping {component_name}...")
                process.terminate()
                try:
                    await asyncio.wait_for(process.wait(), timeout=10.0)
                except asyncio.TimeoutError:
                    process.kill()
                    await process.wait()
        
        # Stop infrastructure
        if self.message_broker:
            await self.message_broker.stop()
        
        if self.process_manager:
            await self.process_manager.stop()
        
        if self.health_checker:
            await self.health_checker.stop()
        
        if self.alert_system:
            await self.alert_system.stop()
        
        if self.performance_tracker:
            await self.performance_tracker.stop()
        
        logger.info("Trading system cleanup completed")

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Enhanced Trading System Startup Script')
    
    # Trading parameters
    parser.add_argument('--mode', choices=['live', 'paper', 'simulation'], default='paper',
                       help='Trading mode')
    parser.add_argument('--symbols', nargs='+', default=['SPY', 'QQQ', 'IWM'],
                       help='Symbols to trade')
    parser.add_argument('--max-positions', type=int, default=10,
                       help='Maximum number of positions')
    parser.add_argument('--position-size', type=float, default=10000,
                       help='Position size in USD')
    
    # Model configuration
    parser.add_argument('--no-ensemble', action='store_true',
                       help='Disable model ensemble')
    parser.add_argument('--prediction-threshold', type=float, default=0.6,
                       help='Prediction confidence threshold')
    
    # Risk management
    parser.add_argument('--no-risk-management', action='store_true',
                       help='Disable risk management')
    parser.add_argument('--max-daily-loss', type=float, default=0.02,
                       help='Maximum daily loss (as fraction)')
    parser.add_argument('--max-position-risk', type=float, default=0.01,
                       help='Maximum position risk (as fraction)')
    
    # Infrastructure
    parser.add_argument('--no-data-feed', action='store_true',
                       help='Disable data feed')
    parser.add_argument('--no-scheduler', action='store_true',
                       help='Disable scheduler')
    parser.add_argument('--no-auto-recovery', action='store_true',
                       help='Disable auto recovery')
    parser.add_argument('--no-monitoring', action='store_true',
                       help='Disable monitoring')
    
    # Performance
    parser.add_argument('--max-workers', type=int,
                       help='Maximum number of workers')
    parser.add_argument('--optimization-level', choices=['low', 'medium', 'high'], default='high',
                       help='Optimization level')
    
    # Options
    parser.add_argument('--verbose', action='store_true',
                       help='Verbose logging')
    parser.add_argument('--config-file',
                       help='Custom configuration file')
    
    return parser.parse_args()

async def main():
    """Main trading system function"""
    args = parse_arguments()
    
    # Setup logging
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Load custom configuration if provided
    custom_config = {}
    if args.config_file:
        with open(args.config_file, 'r') as f:
            custom_config = yaml.safe_load(f)
    
    # Create configuration
    config = TradingSystemConfig(
        trading_mode=args.mode,
        symbols=args.symbols,
        max_positions=args.max_positions,
        position_size=args.position_size,
        model_ensemble=not args.no_ensemble,
        prediction_threshold=args.prediction_threshold,
        enable_risk_management=not args.no_risk_management,
        max_daily_loss=args.max_daily_loss,
        max_position_risk=args.max_position_risk,
        enable_data_feed=not args.no_data_feed,
        enable_scheduler=not args.no_scheduler,
        auto_recovery=not args.no_auto_recovery,
        enable_monitoring=not args.no_monitoring,
        max_workers=args.max_workers,
        optimization_level=args.optimization_level,
        **custom_config
    )
    
    # Initialize and start trading system
    orchestrator = TradingSystemOrchestrator(config)
    
    try:
        print(f"Starting trading system...")
        print(f"Mode: {config.trading_mode}")
        print(f"Symbols: {', '.join(config.symbols)}")
        print(f"Max Positions: {config.max_positions}")
        print(f"Position Size: ${config.position_size:,.0f}")
        print(f"Model Ensemble: {'enabled' if config.model_ensemble else 'disabled'}")
        print(f"Risk Management: {'enabled' if config.enable_risk_management else 'disabled'}")
        
        result = await orchestrator.start_trading_system()
        
        # Print summary
        print(f"\n=== Trading System Results ===")
        print(f"Status: {result['status']}")
        print(f"Uptime: {result['uptime']:.2f}s")
        print(f"Trades Executed: {result['stats']['trades_executed']:,}")
        print(f"Orders Placed: {result['stats']['orders_placed']:,}")
        print(f"Total PnL: ${result['stats']['total_pnl']:.2f}")
        print(f"Component Restarts: {result['stats']['component_restarts']}")
        
    except Exception as e:
        logger.error(f"Trading system failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())