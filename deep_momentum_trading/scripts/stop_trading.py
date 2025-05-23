#!/usr/bin/env python3
"""
Enhanced Trading System Shutdown Script with ARM64 Optimizations

This script provides comprehensive trading system shutdown capabilities for the Deep Momentum Trading System
with ARM64-specific optimizations, graceful component termination, and safety mechanisms.

Features:
- Graceful trading system shutdown
- Position management and liquidation options
- Component termination with safety checks
- Data preservation and backup
- Emergency shutdown capabilities
- ARM64-optimized cleanup operations
"""

import os
import sys
import argparse
import asyncio
import signal
import time
import platform
import psutil
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
from src.trading.position_manager import PositionManager
from src.trading.order_manager import OrderManager
from src.trading.alpaca_client import AlpacaClient
from src.data.data_manager import DataManager
from src.risk.risk_manager import RiskManager
from src.communication.message_broker import MessageBroker
from src.infrastructure.process_manager import ProcessManager
from src.infrastructure.health_check import HealthChecker
from src.monitoring.alert_system import AlertSystem
from src.monitoring.performance_tracker import PerformanceTracker
from src.utils.logger import get_logger
from src.utils.decorators import performance_monitor, error_handler
from src.utils.exceptions import TradingSystemError
from config.settings import config_manager

logger = get_logger(__name__)

class ARM64ShutdownOptimizer:
    """ARM64-specific optimizations for shutdown operations"""
    
    def __init__(self):
        self.is_arm64 = platform.machine().lower() in ['arm64', 'aarch64']
        self.cpu_count = mp.cpu_count()
        self.optimal_workers = self._calculate_optimal_workers()
        
    def _calculate_optimal_workers(self) -> int:
        """Calculate optimal number of workers for ARM64"""
        if self.is_arm64:
            # ARM64 benefits from moderate parallelism for cleanup operations
            return min(self.cpu_count, 4)
        return min(self.cpu_count, 6)

class ShutdownConfig:
    """Shutdown configuration"""
    
    def __init__(self, **kwargs):
        # Shutdown mode
        self.shutdown_mode = kwargs.get('shutdown_mode', 'graceful')  # graceful, immediate, emergency
        
        # Position handling
        self.close_positions = kwargs.get('close_positions', True)
        self.cancel_orders = kwargs.get('cancel_orders', True)
        self.liquidation_timeout = kwargs.get('liquidation_timeout', 300.0)  # 5 minutes
        self.force_liquidation = kwargs.get('force_liquidation', False)
        
        # Component shutdown
        self.component_timeout = kwargs.get('component_timeout', 30.0)  # 30 seconds
        self.force_kill_timeout = kwargs.get('force_kill_timeout', 60.0)  # 1 minute
        self.preserve_data = kwargs.get('preserve_data', True)
        
        # Safety checks
        self.require_confirmation = kwargs.get('require_confirmation', True)
        self.backup_data = kwargs.get('backup_data', True)
        self.generate_report = kwargs.get('generate_report', True)
        
        # Recovery preparation
        self.save_state = kwargs.get('save_state', True)
        self.prepare_restart = kwargs.get('prepare_restart', False)
        
        # Performance
        self.max_workers = kwargs.get('max_workers', None)
        self.parallel_shutdown = kwargs.get('parallel_shutdown', True)

class TradingSystemShutdown:
    """
    Enhanced trading system shutdown with ARM64 optimizations
    """
    
    def __init__(self, config: ShutdownConfig):
        self.config = config
        self.optimizer = ARM64ShutdownOptimizer()
        
        # Initialize components for shutdown
        self.position_manager = None
        self.order_manager = None
        self.alpaca_client = None
        self.trading_engine = None
        self.data_manager = None
        self.risk_manager = None
        self.message_broker = None
        self.process_manager = None
        self.health_checker = None
        self.alert_system = None
        self.performance_tracker = None
        
        # Shutdown state
        self.shutdown_start_time = None
        self.components_status = {}
        self.positions_closed = []
        self.orders_cancelled = []
        self.errors = []
        
        # Statistics
        self.stats = {
            'shutdown_duration': 0.0,
            'positions_closed': 0,
            'orders_cancelled': 0,
            'components_stopped': 0,
            'errors_encountered': 0,
            'data_preserved': False,
            'backup_created': False
        }
        
        logger.info(f"TradingSystemShutdown initialized with ARM64 optimizations: {self.optimizer.is_arm64}")
    
    @performance_monitor
    @error_handler
    async def shutdown_trading_system(self) -> Dict[str, Any]:
        """
        Shutdown comprehensive trading system
        
        Returns:
            Dict containing shutdown results
        """
        logger.info(f"Starting {self.config.shutdown_mode} trading system shutdown...")
        self.shutdown_start_time = time.time()
        
        try:
            # Pre-shutdown validation
            if self.config.require_confirmation and not await self._confirm_shutdown():
                return {'status': 'cancelled', 'reason': 'User cancelled shutdown'}
            
            # Initialize components
            await self._initialize_components()
            
            # Send shutdown alerts
            await self._send_shutdown_alerts()
            
            # Execute shutdown based on mode
            if self.config.shutdown_mode == 'emergency':
                await self._emergency_shutdown()
            elif self.config.shutdown_mode == 'immediate':
                await self._immediate_shutdown()
            else:
                await self._graceful_shutdown()
            
            # Generate final report
            if self.config.generate_report:
                await self._generate_shutdown_report()
            
            # Calculate final statistics
            self.stats['shutdown_duration'] = time.time() - self.shutdown_start_time
            
            logger.info("Trading system shutdown completed successfully")
            
            return {
                'status': 'completed',
                'mode': self.config.shutdown_mode,
                'stats': self.stats,
                'duration': self.stats['shutdown_duration']
            }
            
        except Exception as e:
            logger.error(f"Trading system shutdown failed: {e}")
            self.errors.append(str(e))
            self.stats['errors_encountered'] += 1
            
            # Attempt emergency shutdown if graceful failed
            if self.config.shutdown_mode != 'emergency':
                logger.warning("Attempting emergency shutdown due to failure...")
                await self._emergency_shutdown()
            
            return {
                'status': 'failed',
                'error': str(e),
                'stats': self.stats,
                'errors': self.errors
            }
    
    async def _confirm_shutdown(self) -> bool:
        """Confirm shutdown with user"""
        try:
            print(f"\n=== Trading System Shutdown Confirmation ===")
            print(f"Shutdown Mode: {self.config.shutdown_mode}")
            print(f"Close Positions: {'Yes' if self.config.close_positions else 'No'}")
            print(f"Cancel Orders: {'Yes' if self.config.cancel_orders else 'No'}")
            print(f"Backup Data: {'Yes' if self.config.backup_data else 'No'}")
            
            if self.config.shutdown_mode == 'emergency':
                print("\n⚠️  WARNING: Emergency shutdown will forcefully terminate all processes!")
                print("This may result in data loss and incomplete transactions.")
            
            response = input("\nProceed with shutdown? (yes/no): ").lower().strip()
            return response in ['yes', 'y']
            
        except Exception as e:
            logger.error(f"Error confirming shutdown: {e}")
            return False
    
    async def _initialize_components(self):
        """Initialize components for shutdown"""
        logger.info("Initializing components for shutdown...")
        
        try:
            # Initialize broker client
            self.alpaca_client = AlpacaClient()
            await self.alpaca_client.initialize()
            
            # Initialize position manager
            self.position_manager = PositionManager()
            await self.position_manager.initialize(self.alpaca_client)
            
            # Initialize order manager
            self.order_manager = OrderManager()
            await self.order_manager.initialize(
                broker_client=self.alpaca_client,
                position_manager=self.position_manager
            )
            
            # Initialize other components
            self.data_manager = DataManager()
            self.risk_manager = RiskManager()
            self.message_broker = MessageBroker()
            self.process_manager = ProcessManager()
            self.health_checker = HealthChecker()
            self.alert_system = AlertSystem()
            self.performance_tracker = PerformanceTracker()
            
            # Start essential components
            await self.message_broker.start()
            await self.alert_system.start()
            
            logger.info("Components initialized for shutdown")
            
        except Exception as e:
            logger.error(f"Error initializing components: {e}")
            # Continue with shutdown even if initialization fails
    
    async def _send_shutdown_alerts(self):
        """Send shutdown alerts"""
        try:
            if self.alert_system:
                await self.alert_system.send_alert(
                    level='warning' if self.config.shutdown_mode == 'graceful' else 'critical',
                    message=f'Trading system {self.config.shutdown_mode} shutdown initiated',
                    context={
                        'shutdown_mode': self.config.shutdown_mode,
                        'close_positions': self.config.close_positions,
                        'cancel_orders': self.config.cancel_orders,
                        'timestamp': time.time()
                    }
                )
            
        except Exception as e:
            logger.error(f"Error sending shutdown alerts: {e}")
    
    async def _graceful_shutdown(self):
        """Perform graceful shutdown"""
        logger.info("Performing graceful shutdown...")
        
        # Step 1: Stop accepting new orders
        await self._stop_new_orders()
        
        # Step 2: Cancel pending orders
        if self.config.cancel_orders:
            await self._cancel_pending_orders()
        
        # Step 3: Close positions
        if self.config.close_positions:
            await self._close_all_positions()
        
        # Step 4: Backup data
        if self.config.backup_data:
            await self._backup_system_data()
        
        # Step 5: Save system state
        if self.config.save_state:
            await self._save_system_state()
        
        # Step 6: Stop components gracefully
        await self._stop_components_gracefully()
        
        # Step 7: Stop external processes
        await self._stop_external_processes()
        
        logger.info("Graceful shutdown completed")
    
    async def _immediate_shutdown(self):
        """Perform immediate shutdown"""
        logger.info("Performing immediate shutdown...")
        
        # Step 1: Cancel all orders immediately
        if self.config.cancel_orders:
            await self._cancel_all_orders_immediately()
        
        # Step 2: Close positions with shorter timeout
        if self.config.close_positions:
            await self._close_positions_immediately()
        
        # Step 3: Quick data backup
        if self.config.backup_data:
            await self._quick_backup()
        
        # Step 4: Stop components with shorter timeout
        await self._stop_components_immediately()
        
        # Step 5: Terminate external processes
        await self._terminate_external_processes()
        
        logger.info("Immediate shutdown completed")
    
    async def _emergency_shutdown(self):
        """Perform emergency shutdown"""
        logger.warning("Performing emergency shutdown...")
        
        # Step 1: Force kill all external processes
        await self._force_kill_all_processes()
        
        # Step 2: Emergency position liquidation (if possible)
        if self.config.close_positions and self.config.force_liquidation:
            await self._emergency_liquidation()
        
        # Step 3: Force stop all components
        await self._force_stop_components()
        
        # Step 4: Emergency data preservation
        await self._emergency_data_preservation()
        
        logger.warning("Emergency shutdown completed")
    
    async def _stop_new_orders(self):
        """Stop accepting new orders"""
        try:
            logger.info("Stopping new order acceptance...")
            
            if self.trading_engine:
                await self.trading_engine.stop_new_orders()
            
            # Send message to all components to stop trading
            if self.message_broker:
                await self.message_broker.publish('system_events', {
                    'type': 'stop_trading',
                    'timestamp': time.time(),
                    'reason': 'shutdown_initiated'
                })
            
            logger.info("New order acceptance stopped")
            
        except Exception as e:
            logger.error(f"Error stopping new orders: {e}")
            self.errors.append(f"Stop new orders: {e}")
    
    async def _cancel_pending_orders(self):
        """Cancel all pending orders"""
        try:
            logger.info("Cancelling pending orders...")
            
            if not self.order_manager:
                return
            
            # Get all pending orders
            pending_orders = await self.order_manager.get_pending_orders()
            
            if not pending_orders:
                logger.info("No pending orders to cancel")
                return
            
            # Cancel orders in parallel for ARM64 optimization
            if self.optimizer.is_arm64 and self.config.parallel_shutdown:
                await self._cancel_orders_parallel(pending_orders)
            else:
                await self._cancel_orders_sequential(pending_orders)
            
            self.stats['orders_cancelled'] = len(self.orders_cancelled)
            logger.info(f"Cancelled {len(self.orders_cancelled)} pending orders")
            
        except Exception as e:
            logger.error(f"Error cancelling pending orders: {e}")
            self.errors.append(f"Cancel orders: {e}")
    
    async def _cancel_orders_parallel(self, orders: List[Dict[str, Any]]):
        """Cancel orders in parallel (ARM64 optimized)"""
        try:
            max_workers = self.config.max_workers or self.optimizer.optimal_workers
            
            async def cancel_order(order):
                try:
                    result = await self.order_manager.cancel_order(order['id'])
                    if result:
                        self.orders_cancelled.append(order)
                    return result
                except Exception as e:
                    logger.error(f"Error cancelling order {order['id']}: {e}")
                    return False
            
            # Process orders in batches
            batch_size = max_workers
            for i in range(0, len(orders), batch_size):
                batch = orders[i:i + batch_size]
                tasks = [cancel_order(order) for order in batch]
                await asyncio.gather(*tasks, return_exceptions=True)
            
        except Exception as e:
            logger.error(f"Error in parallel order cancellation: {e}")
    
    async def _cancel_orders_sequential(self, orders: List[Dict[str, Any]]):
        """Cancel orders sequentially"""
        try:
            for order in orders:
                try:
                    result = await self.order_manager.cancel_order(order['id'])
                    if result:
                        self.orders_cancelled.append(order)
                except Exception as e:
                    logger.error(f"Error cancelling order {order['id']}: {e}")
            
        except Exception as e:
            logger.error(f"Error in sequential order cancellation: {e}")
    
    async def _cancel_all_orders_immediately(self):
        """Cancel all orders immediately"""
        try:
            logger.info("Cancelling all orders immediately...")
            
            if self.order_manager:
                result = await self.order_manager.cancel_all_orders()
                self.stats['orders_cancelled'] = result.get('cancelled_count', 0)
            
            logger.info(f"Immediately cancelled {self.stats['orders_cancelled']} orders")
            
        except Exception as e:
            logger.error(f"Error cancelling orders immediately: {e}")
            self.errors.append(f"Immediate cancel orders: {e}")
    
    async def _close_all_positions(self):
        """Close all positions gracefully"""
        try:
            logger.info("Closing all positions...")
            
            if not self.position_manager:
                return
            
            # Get all open positions
            positions = await self.position_manager.get_all_positions()
            
            if not positions:
                logger.info("No positions to close")
                return
            
            # Close positions with timeout
            start_time = time.time()
            
            for position in positions:
                try:
                    if time.time() - start_time > self.config.liquidation_timeout:
                        logger.warning("Liquidation timeout reached, forcing remaining closures")
                        break
                    
                    result = await self.position_manager.close_position(
                        position['symbol'],
                        reason='shutdown'
                    )
                    
                    if result:
                        self.positions_closed.append(position)
                    
                except Exception as e:
                    logger.error(f"Error closing position {position['symbol']}: {e}")
            
            self.stats['positions_closed'] = len(self.positions_closed)
            logger.info(f"Closed {len(self.positions_closed)} positions")
            
        except Exception as e:
            logger.error(f"Error closing positions: {e}")
            self.errors.append(f"Close positions: {e}")
    
    async def _close_positions_immediately(self):
        """Close positions immediately with shorter timeout"""
        try:
            logger.info("Closing positions immediately...")
            
            if self.position_manager:
                result = await self.position_manager.close_all_positions(
                    timeout=30.0,  # 30 second timeout
                    force=True
                )
                self.stats['positions_closed'] = result.get('closed_count', 0)
            
            logger.info(f"Immediately closed {self.stats['positions_closed']} positions")
            
        except Exception as e:
            logger.error(f"Error closing positions immediately: {e}")
            self.errors.append(f"Immediate close positions: {e}")
    
    async def _emergency_liquidation(self):
        """Emergency position liquidation"""
        try:
            logger.warning("Performing emergency liquidation...")
            
            if self.position_manager:
                result = await self.position_manager.emergency_liquidation()
                self.stats['positions_closed'] = result.get('liquidated_count', 0)
            
            logger.warning(f"Emergency liquidated {self.stats['positions_closed']} positions")
            
        except Exception as e:
            logger.error(f"Error in emergency liquidation: {e}")
            self.errors.append(f"Emergency liquidation: {e}")
    
    async def _backup_system_data(self):
        """Backup system data"""
        try:
            logger.info("Backing up system data...")
            
            backup_dir = project_root / "backups" / f"shutdown_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            backup_dir.mkdir(parents=True, exist_ok=True)
            
            # Backup configuration files
            config_files = [
                'config/trading_config.yaml',
                'config/risk_config.yaml',
                'config/model_config.yaml',
                'config/settings.py'
            ]
            
            for config_file in config_files:
                source = project_root / config_file
                if source.exists():
                    dest = backup_dir / config_file
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    
                    import shutil
                    shutil.copy2(source, dest)
            
            # Backup recent data
            if self.data_manager:
                await self.data_manager.backup_recent_data(backup_dir / "data")
            
            # Backup logs
            log_dir = project_root / "logs"
            if log_dir.exists():
                import shutil
                shutil.copytree(log_dir, backup_dir / "logs", dirs_exist_ok=True)
            
            self.stats['backup_created'] = True
            logger.info(f"System data backed up to: {backup_dir}")
            
        except Exception as e:
            logger.error(f"Error backing up system data: {e}")
            self.errors.append(f"Backup data: {e}")
    
    async def _quick_backup(self):
        """Quick backup of essential data"""
        try:
            logger.info("Performing quick backup...")
            
            backup_dir = project_root / "backups" / f"quick_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            backup_dir.mkdir(parents=True, exist_ok=True)
            
            # Backup only essential files
            essential_files = [
                'config/trading_config.yaml',
                '.env'
            ]
            
            for file_path in essential_files:
                source = project_root / file_path
                if source.exists():
                    dest = backup_dir / file_path
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    
                    import shutil
                    shutil.copy2(source, dest)
            
            self.stats['backup_created'] = True
            logger.info(f"Quick backup completed: {backup_dir}")
            
        except Exception as e:
            logger.error(f"Error in quick backup: {e}")
            self.errors.append(f"Quick backup: {e}")
    
    async def _save_system_state(self):
        """Save system state for recovery"""
        try:
            logger.info("Saving system state...")
            
            state = {
                'timestamp': time.time(),
                'shutdown_mode': self.config.shutdown_mode,
                'positions_at_shutdown': self.positions_closed,
                'orders_cancelled': self.orders_cancelled,
                'component_status': self.components_status,
                'stats': self.stats.copy()
            }
            
            state_file = project_root / "state" / "shutdown_state.json"
            state_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(state_file, 'w') as f:
                json.dump(state, f, indent=2, default=str)
            
            logger.info(f"System state saved: {state_file}")
            
        except Exception as e:
            logger.error(f"Error saving system state: {e}")
            self.errors.append(f"Save state: {e}")
    
    async def _stop_components_gracefully(self):
        """Stop components gracefully"""
        try:
            logger.info("Stopping components gracefully...")
            
            components = [
                ('trading_engine', self.trading_engine),
                ('risk_manager', self.risk_manager),
                ('data_manager', self.data_manager),
                ('order_manager', self.order_manager),
                ('position_manager', self.position_manager),
                ('performance_tracker', self.performance_tracker),
                ('health_checker', self.health_checker),
                ('message_broker', self.message_broker)
            ]
            
            for name, component in components:
                if component:
                    try:
                        await asyncio.wait_for(
                            component.stop(),
                            timeout=self.config.component_timeout
                        )
                        self.components_status[name] = 'stopped'
                        self.stats['components_stopped'] += 1
                        logger.info(f"Component {name} stopped gracefully")
                    except asyncio.TimeoutError:
                        logger.warning(f"Component {name} stop timeout")
                        self.components_status[name] = 'timeout'
                    except Exception as e:
                        logger.error(f"Error stopping component {name}: {e}")
                        self.components_status[name] = 'error'
            
            logger.info("Component shutdown completed")
            
        except Exception as e:
            logger.error(f"Error stopping components: {e}")
            self.errors.append(f"Stop components: {e}")
    
    async def _stop_components_immediately(self):
        """Stop components immediately"""
        try:
            logger.info("Stopping components immediately...")
            
            # Use shorter timeout
            timeout = min(self.config.component_timeout, 10.0)
            
            components = [
                ('trading_engine', self.trading_engine),
                ('risk_manager', self.risk_manager),
                ('data_manager', self.data_manager),
                ('message_broker', self.message_broker)
            ]
            
            # Stop components in parallel
            tasks = []
            for name, component in components:
                if component:
                    task = asyncio.create_task(self._stop_component_with_timeout(name, component, timeout))
                    tasks.append(task)
            
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
            
            logger.info("Immediate component shutdown completed")
            
        except Exception as e:
            logger.error(f"Error stopping components immediately: {e}")
            self.errors.append(f"Immediate stop components: {e}")
    
    async def _stop_component_with_timeout(self, name: str, component: Any, timeout: float):
        """Stop component with timeout"""
        try:
            await asyncio.wait_for(component.stop(), timeout=timeout)
            self.components_status[name] = 'stopped'
            self.stats['components_stopped'] += 1
        except asyncio.TimeoutError:
            self.components_status[name] = 'timeout'
        except Exception as e:
            self.components_status[name] = 'error'
            logger.error(f"Error stopping {name}: {e}")
    
    async def _force_stop_components(self):
        """Force stop all components"""
        try:
            logger.warning("Force stopping all components...")
            
            # Force stop without waiting
            components = [
                ('trading_engine', self.trading_engine),
                ('risk_manager', self.risk_manager),
                ('data_manager', self.data_manager),
                ('message_broker', self.message_broker)
            ]
            
            for name, component in components:
                if component:
                    try:
                        if hasattr(component, 'force_stop'):
                            await component.force_stop()
                        else:
                            await component.stop()
                        self.components_status[name] = 'force_stopped'
                        self.stats['components_stopped'] += 1
                    except Exception as e:
                        logger.error(f"Error force stopping {name}: {e}")
                        self.components_status[name] = 'error'
            
            logger.warning("Force stop completed")
            
        except Exception as e:
            logger.error(f"Error force stopping components: {e}")
            self.errors.append(f"Force stop components: {e}")
    
    async def _stop_external_processes(self):
        """Stop external processes gracefully"""
        try:
            logger.info("Stopping external processes...")
            
            # Find trading system processes
            processes = self._find_trading_processes()
            
            for proc in processes:
                try:
                    # Send SIGTERM for graceful shutdown
                    proc.terminate()
                    
                    # Wait for process to terminate
                    try:
                        proc.wait(timeout=self.config.component_timeout)
                        logger.info(f"Process {proc.pid} terminated gracefully")
                    except psutil.TimeoutExpired:
                        logger.warning(f"Process {proc.pid} did not terminate, killing...")
                        proc.kill()
                        proc.wait()
                    
                except psutil.NoSuchProcess:
                    # Process already terminated
                    pass
                except Exception as e:
                    logger.error(f"Error stopping process {proc.pid}: {e}")
            
            logger.info("External processes stopped")
            
        except Exception as e:
            logger.error(f"Error stopping external processes: {e}")
            self.errors.append(f"Stop external processes: {e}")
    
    async def _terminate_external_processes(self):
        """Terminate external processes immediately"""
        try:
            logger.info("Terminating external processes...")
            
            processes = self._find_trading_processes()
            
            for proc in processes:
                try:
                    proc.kill()
                    proc.wait(timeout=5.0)
                    logger.info(f"Process {proc.pid} killed")
                except psutil.NoSuchProcess:
                    pass
                except Exception as e:
                    logger.error(f"Error killing process {proc.pid}: {e}")
            
            logger.info("External processes terminated")
            
        except Exception as e:
            logger.error(f"Error terminating external processes: {e}")
            self.errors.append(f"Terminate external processes: {e}")
    
    async def _force_kill_all_processes(self):
        """Force kill all trading system processes"""
        try:
            logger.warning("Force killing all trading system processes...")
            
            processes = self._find_trading_processes()
            
            for proc in processes:
                try:
                    proc.kill()
                    logger.warning(f"Force killed process {proc.pid}")
                except psutil.NoSuchProcess:
                    pass
                except Exception as e:
                    logger.error(f"Error force killing process {proc.pid}: {e}")
            
            logger.warning("Force kill completed")
            
        except Exception as e:
            logger.error(f"Error force killing processes: {e}")
            self.errors.append(f"Force kill processes: {e}")
    
    def _find_trading_processes(self) -> List[psutil.Process]:
        """Find trading system processes"""
        try:
            processes = []
            
            # Look for processes with trading system script names
            script_names = [
                'start_trading.py',
                'start_data_feed.py',
                'start_risk_monitor.py',
                'train_models.py'
            ]
            
            for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                try:
                    cmdline = proc.info['cmdline']
                    if cmdline:
                        cmdline_str = ' '.join(cmdline)
                        if any(script in cmdline_str for script in script_names):
                            processes.append(proc)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
            
            return processes
            
        except Exception as e:
            logger.error(f"Error finding trading processes: {e}")
            return []
    
    async def _emergency_data_preservation(self):
        """Emergency data preservation"""
        try:
            logger.warning("Performing emergency data preservation...")
            
            # Save critical data only
            emergency_dir = project_root / "emergency" / f"shutdown_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            emergency_dir.mkdir(parents=True, exist_ok=True)
            
            # Save shutdown state
            state = {
                'timestamp': time.time(),
                'emergency_shutdown': True,
                'errors': self.errors,
                'stats': self.stats
            }
            
            with open(emergency_dir / "emergency_state.json", 'w') as f:
                json.dump(state, f, indent=2, default=str)
            
            self.stats['data_preserved'] = True
            logger.warning(f"Emergency data preserved: {emergency_dir}")
            
        except Exception as e:
            logger.error(f"Error in emergency data preservation: {e}")
            self.errors.append(f"Emergency data preservation: {e}")
    
    async def _generate_shutdown_report(self):
        """Generate shutdown report"""
        try:
            logger.info("Generating shutdown report...")
            
            report = {
                'shutdown_summary': {
                    'mode': self.config.shutdown_mode,
                    'start_time': self.shutdown_start_time,
                    'duration': time.time() - self.shutdown_start_time,
                    'status': 'completed' if not self.errors else 'completed_with_errors'
                },
                'statistics': self.stats,
                'positions_closed': self.positions_closed,
                'orders_cancelled': self.orders_cancelled,
                'component_status': self.components_status,
                'errors': self.errors,
                'configuration': {
                    'close_positions': self.config.close_positions,
                    'cancel_orders': self.config.cancel_orders,
                    'backup_data': self.config.backup_data,
                    'save_state': self.config.save_state
                }
            }
            
            # Save report
            report_dir = project_root / "reports" / "shutdown"
            report_dir.mkdir(parents=True, exist_ok=True)
            
            report_file = report_dir / f"shutdown_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
            with open(report_file, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            logger.info(f"Shutdown report saved: {report_file}")
            
        except Exception as e:
            logger.error(f"Error generating shutdown report: {e}")
            self.errors.append(f"Generate report: {e}")

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Enhanced Trading System Shutdown Script')
    
    # Shutdown mode
    parser.add_argument('--mode', choices=['graceful', 'immediate', 'emergency'], default='graceful',
                       help='Shutdown mode')
    
    # Position handling
    parser.add_argument('--no-close-positions', action='store_true',
                       help='Do not close positions')
    parser.add_argument('--no-cancel-orders', action='store_true',
                       help='Do not cancel orders')
    parser.add_argument('--force-liquidation', action='store_true',
                       help='Force position liquidation')
    parser.add_argument('--liquidation-timeout', type=float, default=300.0,
                       help='Liquidation timeout in seconds')
    
    # Safety options
    parser.add_argument('--no-confirmation', action='store_true',
                       help='Skip confirmation prompt')
    parser.add_argument('--no-backup', action='store_true',
                       help='Skip data backup')
    parser.add_argument('--no-report', action='store_true',
                       help='Skip shutdown report')
    parser.add_argument('--no-save-state', action='store_true',
                       help='Skip saving system state')
    
    # Timeouts
    parser.add_argument('--component-timeout', type=float, default=30.0,
                       help='Component shutdown timeout')
    parser.add_argument('--force-kill-timeout', type=float, default=60.0,
                       help='Force kill timeout')
    
    # Performance
    parser.add_argument('--max-workers', type=int,
                       help='Maximum number of workers')
    parser.add_argument('--no-parallel', action='store_true',
                       help='Disable parallel shutdown')
    
    # Options
    parser.add_argument('--verbose', action='store_true',
                       help='Verbose logging')
    
    return parser.parse_args()

async def main():
    """Main shutdown function"""
    args = parse_arguments()
    
    # Setup logging
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Create configuration
    config = ShutdownConfig(
        shutdown_mode=args.mode,
        close_positions=not args.no_close_positions,
        cancel_orders=not args.no_cancel_orders,
        force_liquidation=args.force_liquidation,
        liquidation_timeout=args.liquidation_timeout,
        require_confirmation=not args.no_confirmation,
        backup_data=not args.no_backup,
        generate_report=not args.no_report,
        save_state=not args.no_save_state,
        component_timeout=args.component_timeout,
        force_kill_timeout=args.force_kill_timeout,
        max_workers=args.max_workers,
        parallel_shutdown=not args.no_parallel
    )
    
    # Initialize and start shutdown
    shutdown = TradingSystemShutdown(config)
    
    try:
        print(f"Initiating trading system shutdown...")
        print(f"Mode: {config.shutdown_mode}")
        print(f"Close Positions: {'Yes' if config.close_positions else 'No'}")
        print(f"Cancel Orders: {'Yes' if config.cancel_orders else 'No'}")
        print(f"Backup Data: {'Yes' if config.backup_data else 'No'}")
        
        result = await shutdown.shutdown_trading_system()
        
        # Print summary
        print(f"\n=== Shutdown Results ===")
        print(f"Status: {result['status']}")
        print(f"Mode: {result.get('mode', 'unknown')}")
        print(f"Duration: {result.get('duration', 0):.2f}s")
        
        if 'stats' in result:
            stats = result['stats']
            print(f"Positions Closed: {stats['positions_closed']}")
            print(f"Orders Cancelled: {stats['orders_cancelled']}")
            print(f"Components Stopped: {stats['components_stopped']}")
            print(f"Errors: {stats['errors_encountered']}")
            print(f"Data Preserved: {'Yes' if stats['data_preserved'] else 'No'}")
            print(f"Backup Created: {'Yes' if stats['backup_created'] else 'No'}")
        
        if result['status'] == 'failed':
            print(f"Error: {result.get('error', 'Unknown error')}")
            sys.exit(1)
        
    except Exception as e:
        logger.error(f"Shutdown failed: {e}")
        print(f"Shutdown failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())