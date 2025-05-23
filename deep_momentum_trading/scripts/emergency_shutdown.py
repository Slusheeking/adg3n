#!/usr/bin/env python3
"""
Enhanced Emergency Shutdown Script with ARM64 Optimizations

This script provides comprehensive emergency shutdown capabilities for the Deep Momentum Trading System
with ARM64-specific optimizations, graceful shutdown procedures, and safety mechanisms.

Features:
- Immediate and graceful shutdown modes
- Position liquidation and risk management
- Process termination and cleanup
- Data persistence and backup
- Alert notifications and logging
- Recovery state preservation
- ARM64-optimized shutdown procedures
"""

import os
import sys
import signal
import asyncio
import time
import platform
import subprocess
import threading
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
import logging
import json
import argparse
from concurrent.futures import ThreadPoolExecutor, TimeoutError
import psutil

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.trading.trading_engine import TradingEngine
from src.trading.position_manager import PositionManager
from src.trading.order_manager import OrderManager
from src.risk.risk_manager import RiskManager
from src.infrastructure.process_manager import ProcessManager
from src.infrastructure.health_check import HealthChecker
from src.monitoring.alert_system import AlertSystem
from src.data.data_manager import DataManager
from src.utils.logger import get_logger
from src.utils.decorators import performance_monitor, error_handler
from src.utils.exceptions import EmergencyShutdownError
from config.settings import config_manager

logger = get_logger(__name__)

class ShutdownMode:
    """Shutdown mode constants"""
    IMMEDIATE = "immediate"
    GRACEFUL = "graceful"
    EMERGENCY = "emergency"

class ARM64ShutdownOptimizer:
    """ARM64-specific shutdown optimizations"""
    
    def __init__(self):
        self.is_arm64 = platform.machine().lower() in ['arm64', 'aarch64']
        self.cpu_count = os.cpu_count()
        self.system_info = self._get_system_info()
        
    def _get_system_info(self) -> Dict[str, Any]:
        """Get system information for shutdown optimization"""
        return {
            'platform': platform.platform(),
            'architecture': platform.machine(),
            'cpu_count': self.cpu_count,
            'is_arm64': self.is_arm64,
            'memory_gb': psutil.virtual_memory().total / (1024**3),
            'boot_time': psutil.boot_time()
        }
    
    def get_optimal_shutdown_timeout(self, mode: str) -> int:
        """Get optimal shutdown timeout for ARM64"""
        base_timeouts = {
            ShutdownMode.IMMEDIATE: 5,
            ShutdownMode.GRACEFUL: 30,
            ShutdownMode.EMERGENCY: 2
        }
        
        timeout = base_timeouts.get(mode, 30)
        
        # ARM64 may need slightly longer for certain operations
        if self.is_arm64:
            timeout = int(timeout * 1.2)
        
        return timeout

class EmergencyShutdownEngine:
    """
    Enhanced emergency shutdown engine with ARM64 optimizations
    """
    
    def __init__(self, mode: str = ShutdownMode.GRACEFUL):
        self.mode = mode
        self.optimizer = ARM64ShutdownOptimizer()
        self.shutdown_timeout = self.optimizer.get_optimal_shutdown_timeout(mode)
        
        # Initialize components
        self.trading_engine = None
        self.position_manager = None
        self.order_manager = None
        self.risk_manager = None
        self.process_manager = ProcessManager()
        self.health_checker = HealthChecker()
        self.alert_system = AlertSystem()
        self.data_manager = DataManager()
        
        # Shutdown state
        self.shutdown_id = f"shutdown-{int(time.time())}"
        self.shutdown_start_time = None
        self.shutdown_steps = []
        self.shutdown_status = "initialized"
        self.emergency_triggered = False
        
        # Recovery state
        self.recovery_state = {
            'positions': {},
            'orders': {},
            'system_state': {},
            'shutdown_reason': '',
            'shutdown_time': None
        }
        
        logger.info(f"EmergencyShutdownEngine initialized in {mode} mode")
    
    @performance_monitor
    @error_handler
    async def execute_shutdown(self, reason: str = "Manual shutdown") -> Dict[str, Any]:
        """
        Execute emergency shutdown procedure
        
        Args:
            reason: Reason for shutdown
            
        Returns:
            Dict containing shutdown results
        """
        logger.critical(f"EMERGENCY SHUTDOWN INITIATED: {reason}")
        self.shutdown_start_time = time.time()
        self.shutdown_status = "in_progress"
        self.recovery_state['shutdown_reason'] = reason
        self.recovery_state['shutdown_time'] = datetime.now().isoformat()
        
        try:
            # Send immediate alert
            await self._send_shutdown_alert(reason)
            
            # Initialize trading components
            await self._initialize_components()
            
            # Execute shutdown sequence based on mode
            if self.mode == ShutdownMode.EMERGENCY:
                result = await self._emergency_shutdown()
            elif self.mode == ShutdownMode.IMMEDIATE:
                result = await self._immediate_shutdown()
            else:  # GRACEFUL
                result = await self._graceful_shutdown()
            
            # Save recovery state
            await self._save_recovery_state()
            
            # Final cleanup
            await self._final_cleanup()
            
            execution_time = time.time() - self.shutdown_start_time
            self.shutdown_status = "completed"
            
            logger.critical(f"Emergency shutdown completed in {execution_time:.2f} seconds")
            
            return {
                'shutdown_id': self.shutdown_id,
                'mode': self.mode,
                'reason': reason,
                'status': self.shutdown_status,
                'execution_time': execution_time,
                'steps_completed': self.shutdown_steps,
                'recovery_state': self.recovery_state,
                'system_info': self.optimizer.system_info
            }
            
        except Exception as e:
            self.shutdown_status = "failed"
            logger.error(f"Emergency shutdown failed: {e}")
            await self._handle_shutdown_failure(e)
            raise EmergencyShutdownError(f"Emergency shutdown failed: {e}")
    
    async def _initialize_components(self):
        """Initialize trading components for shutdown"""
        try:
            self.trading_engine = TradingEngine()
            self.position_manager = PositionManager()
            self.order_manager = OrderManager()
            self.risk_manager = RiskManager()
            
            self.shutdown_steps.append("components_initialized")
            logger.info("Trading components initialized for shutdown")
            
        except Exception as e:
            logger.warning(f"Failed to initialize some components: {e}")
            # Continue with available components
    
    async def _emergency_shutdown(self) -> Dict[str, Any]:
        """Execute emergency shutdown (fastest, minimal cleanup)"""
        logger.critical("Executing EMERGENCY shutdown - minimal cleanup")
        
        results = {
            'positions_liquidated': False,
            'orders_cancelled': False,
            'processes_terminated': False,
            'data_saved': False
        }
        
        try:
            # 1. Cancel all orders immediately (highest priority)
            if self.order_manager:
                await asyncio.wait_for(
                    self._cancel_all_orders(),
                    timeout=2
                )
                results['orders_cancelled'] = True
                self.shutdown_steps.append("orders_cancelled")
            
            # 2. Liquidate positions if possible
            if self.position_manager:
                try:
                    await asyncio.wait_for(
                        self._liquidate_positions(emergency=True),
                        timeout=3
                    )
                    results['positions_liquidated'] = True
                    self.shutdown_steps.append("positions_liquidated")
                except TimeoutError:
                    logger.error("Position liquidation timed out in emergency mode")
            
            # 3. Force terminate processes
            await self._force_terminate_processes()
            results['processes_terminated'] = True
            self.shutdown_steps.append("processes_terminated")
            
        except Exception as e:
            logger.error(f"Emergency shutdown error: {e}")
        
        return results
    
    async def _immediate_shutdown(self) -> Dict[str, Any]:
        """Execute immediate shutdown (fast with basic cleanup)"""
        logger.critical("Executing IMMEDIATE shutdown - basic cleanup")
        
        results = {
            'positions_liquidated': False,
            'orders_cancelled': False,
            'processes_terminated': False,
            'data_saved': False
        }
        
        try:
            # 1. Cancel all orders
            if self.order_manager:
                await asyncio.wait_for(
                    self._cancel_all_orders(),
                    timeout=5
                )
                results['orders_cancelled'] = True
                self.shutdown_steps.append("orders_cancelled")
            
            # 2. Liquidate positions
            if self.position_manager:
                await asyncio.wait_for(
                    self._liquidate_positions(emergency=False),
                    timeout=10
                )
                results['positions_liquidated'] = True
                self.shutdown_steps.append("positions_liquidated")
            
            # 3. Save critical data
            await asyncio.wait_for(
                self._save_critical_data(),
                timeout=5
            )
            results['data_saved'] = True
            self.shutdown_steps.append("data_saved")
            
            # 4. Terminate processes
            await self._terminate_processes()
            results['processes_terminated'] = True
            self.shutdown_steps.append("processes_terminated")
            
        except Exception as e:
            logger.error(f"Immediate shutdown error: {e}")
        
        return results
    
    async def _graceful_shutdown(self) -> Dict[str, Any]:
        """Execute graceful shutdown (comprehensive cleanup)"""
        logger.critical("Executing GRACEFUL shutdown - comprehensive cleanup")
        
        results = {
            'positions_liquidated': False,
            'orders_cancelled': False,
            'processes_terminated': False,
            'data_saved': False,
            'system_state_saved': False,
            'notifications_sent': False
        }
        
        try:
            # 1. Pause new trading
            if self.trading_engine:
                await self.trading_engine.pause_trading()
                self.shutdown_steps.append("trading_paused")
            
            # 2. Cancel all pending orders
            if self.order_manager:
                await self._cancel_all_orders()
                results['orders_cancelled'] = True
                self.shutdown_steps.append("orders_cancelled")
            
            # 3. Liquidate positions gracefully
            if self.position_manager:
                await self._liquidate_positions(emergency=False)
                results['positions_liquidated'] = True
                self.shutdown_steps.append("positions_liquidated")
            
            # 4. Save all data and state
            await self._save_all_data()
            results['data_saved'] = True
            self.shutdown_steps.append("data_saved")
            
            # 5. Save system state
            await self._save_system_state()
            results['system_state_saved'] = True
            self.shutdown_steps.append("system_state_saved")
            
            # 6. Send notifications
            await self._send_shutdown_notifications()
            results['notifications_sent'] = True
            self.shutdown_steps.append("notifications_sent")
            
            # 7. Gracefully terminate processes
            await self._graceful_terminate_processes()
            results['processes_terminated'] = True
            self.shutdown_steps.append("processes_terminated")
            
        except Exception as e:
            logger.error(f"Graceful shutdown error: {e}")
        
        return results
    
    async def _cancel_all_orders(self):
        """Cancel all pending orders"""
        logger.info("Cancelling all pending orders...")
        
        try:
            if self.order_manager:
                pending_orders = await self.order_manager.get_pending_orders()
                
                if pending_orders:
                    logger.warning(f"Cancelling {len(pending_orders)} pending orders")
                    
                    # Cancel orders in parallel for speed
                    cancel_tasks = []
                    for order in pending_orders:
                        task = self.order_manager.cancel_order(order['id'])
                        cancel_tasks.append(task)
                    
                    # Wait for all cancellations with timeout
                    await asyncio.wait_for(
                        asyncio.gather(*cancel_tasks, return_exceptions=True),
                        timeout=10
                    )
                    
                    # Store cancelled orders in recovery state
                    self.recovery_state['orders'] = {
                        'cancelled_orders': [order['id'] for order in pending_orders],
                        'cancellation_time': datetime.now().isoformat()
                    }
                    
                    logger.info(f"Successfully cancelled {len(pending_orders)} orders")
                else:
                    logger.info("No pending orders to cancel")
            
        except Exception as e:
            logger.error(f"Error cancelling orders: {e}")
            raise
    
    async def _liquidate_positions(self, emergency: bool = False):
        """Liquidate all open positions"""
        logger.info(f"Liquidating positions (emergency={emergency})...")
        
        try:
            if self.position_manager:
                positions = await self.position_manager.get_all_positions()
                
                if positions:
                    logger.warning(f"Liquidating {len(positions)} positions")
                    
                    liquidation_tasks = []
                    for position in positions:
                        if emergency:
                            # Market orders for immediate liquidation
                            task = self.position_manager.liquidate_position(
                                position['symbol'], 
                                order_type='market'
                            )
                        else:
                            # Limit orders for better execution
                            task = self.position_manager.liquidate_position(
                                position['symbol'], 
                                order_type='limit'
                            )
                        liquidation_tasks.append(task)
                    
                    # Execute liquidations
                    timeout = 5 if emergency else 20
                    await asyncio.wait_for(
                        asyncio.gather(*liquidation_tasks, return_exceptions=True),
                        timeout=timeout
                    )
                    
                    # Store position information in recovery state
                    self.recovery_state['positions'] = {
                        'liquidated_positions': [
                            {
                                'symbol': pos['symbol'],
                                'quantity': pos['quantity'],
                                'market_value': pos.get('market_value', 0)
                            }
                            for pos in positions
                        ],
                        'liquidation_time': datetime.now().isoformat(),
                        'emergency_liquidation': emergency
                    }
                    
                    logger.info(f"Successfully liquidated {len(positions)} positions")
                else:
                    logger.info("No positions to liquidate")
            
        except Exception as e:
            logger.error(f"Error liquidating positions: {e}")
            raise
    
    async def _save_critical_data(self):
        """Save critical data for recovery"""
        logger.info("Saving critical data...")
        
        try:
            if self.data_manager:
                # Save current market data
                await self.data_manager.save_current_state()
                
                # Save trading metrics
                if self.trading_engine:
                    metrics = await self.trading_engine.get_current_metrics()
                    self.recovery_state['system_state']['trading_metrics'] = metrics
                
                logger.info("Critical data saved successfully")
            
        except Exception as e:
            logger.error(f"Error saving critical data: {e}")
    
    async def _save_all_data(self):
        """Save all system data"""
        logger.info("Saving all system data...")
        
        try:
            # Save critical data first
            await self._save_critical_data()
            
            # Save additional data
            if self.data_manager:
                await self.data_manager.backup_all_data()
                
            logger.info("All data saved successfully")
            
        except Exception as e:
            logger.error(f"Error saving all data: {e}")
    
    async def _save_system_state(self):
        """Save complete system state"""
        logger.info("Saving system state...")
        
        try:
            # Get system health
            if self.health_checker:
                health_status = await self.health_checker.get_system_health()
                self.recovery_state['system_state']['health'] = health_status
            
            # Get process information
            running_processes = await self.process_manager.get_running_processes()
            self.recovery_state['system_state']['processes'] = running_processes
            
            # Get risk metrics
            if self.risk_manager:
                risk_metrics = await self.risk_manager.get_current_risk_metrics()
                self.recovery_state['system_state']['risk_metrics'] = risk_metrics
            
            logger.info("System state saved successfully")
            
        except Exception as e:
            logger.error(f"Error saving system state: {e}")
    
    async def _terminate_processes(self):
        """Terminate trading processes"""
        logger.info("Terminating trading processes...")
        
        try:
            # Get trading processes
            trading_processes = await self.process_manager.get_trading_processes()
            
            if trading_processes:
                logger.warning(f"Terminating {len(trading_processes)} trading processes")
                
                # Terminate processes
                for process in trading_processes:
                    try:
                        await self.process_manager.terminate_process(process['pid'])
                    except Exception as e:
                        logger.warning(f"Failed to terminate process {process['pid']}: {e}")
                
                logger.info("Trading processes terminated")
            else:
                logger.info("No trading processes to terminate")
            
        except Exception as e:
            logger.error(f"Error terminating processes: {e}")
    
    async def _graceful_terminate_processes(self):
        """Gracefully terminate processes"""
        logger.info("Gracefully terminating processes...")
        
        try:
            # Get all system processes
            all_processes = await self.process_manager.get_all_processes()
            
            # Graceful shutdown with SIGTERM first
            for process in all_processes:
                try:
                    await self.process_manager.graceful_shutdown(process['pid'])
                except Exception as e:
                    logger.warning(f"Graceful shutdown failed for process {process['pid']}: {e}")
            
            # Wait a bit for graceful shutdown
            await asyncio.sleep(2)
            
            # Force terminate any remaining processes
            await self._force_terminate_processes()
            
        except Exception as e:
            logger.error(f"Error in graceful process termination: {e}")
    
    async def _force_terminate_processes(self):
        """Force terminate all trading processes"""
        logger.info("Force terminating processes...")
        
        try:
            # Get all trading-related processes
            trading_processes = await self.process_manager.get_trading_processes()
            
            for process in trading_processes:
                try:
                    # Send SIGKILL
                    await self.process_manager.force_terminate(process['pid'])
                except Exception as e:
                    logger.warning(f"Force termination failed for process {process['pid']}: {e}")
            
            logger.info("Force termination completed")
            
        except Exception as e:
            logger.error(f"Error in force termination: {e}")
    
    async def _send_shutdown_alert(self, reason: str):
        """Send immediate shutdown alert"""
        try:
            await self.alert_system.send_alert(
                level='critical',
                message=f"EMERGENCY SHUTDOWN INITIATED: {reason}",
                context={
                    'shutdown_id': self.shutdown_id,
                    'mode': self.mode,
                    'reason': reason,
                    'timestamp': datetime.now().isoformat()
                }
            )
        except Exception as e:
            logger.error(f"Failed to send shutdown alert: {e}")
    
    async def _send_shutdown_notifications(self):
        """Send comprehensive shutdown notifications"""
        try:
            # Send detailed notification
            await self.alert_system.send_alert(
                level='critical',
                message=f"Trading system shutdown completed",
                context={
                    'shutdown_id': self.shutdown_id,
                    'mode': self.mode,
                    'steps_completed': self.shutdown_steps,
                    'recovery_state': self.recovery_state
                }
            )
            
            # Send SMS/email notifications if configured
            await self.alert_system.send_emergency_notifications(
                message=f"Deep Momentum Trading System shutdown completed. ID: {self.shutdown_id}"
            )
            
        except Exception as e:
            logger.error(f"Failed to send shutdown notifications: {e}")
    
    async def _save_recovery_state(self):
        """Save recovery state to file"""
        try:
            recovery_file = Path(f"recovery_state_{self.shutdown_id}.json")
            
            with open(recovery_file, 'w') as f:
                json.dump(self.recovery_state, f, indent=2, default=str)
            
            logger.info(f"Recovery state saved to {recovery_file}")
            
        except Exception as e:
            logger.error(f"Failed to save recovery state: {e}")
    
    async def _final_cleanup(self):
        """Perform final cleanup operations"""
        logger.info("Performing final cleanup...")
        
        try:
            # Clean up temporary files
            temp_files = Path("/tmp").glob("deep_momentum_*")
            for temp_file in temp_files:
                try:
                    temp_file.unlink()
                except Exception:
                    pass
            
            # Clean up shared memory
            # This would clean up any shared memory segments
            
            # Final log entry
            logger.critical(f"Emergency shutdown {self.shutdown_id} completed successfully")
            
        except Exception as e:
            logger.error(f"Error in final cleanup: {e}")
    
    async def _handle_shutdown_failure(self, error: Exception):
        """Handle shutdown failure"""
        logger.critical(f"SHUTDOWN FAILURE: {error}")
        
        try:
            # Send critical alert
            await self.alert_system.send_alert(
                level='critical',
                message=f"EMERGENCY SHUTDOWN FAILED: {error}",
                context={
                    'shutdown_id': self.shutdown_id,
                    'error': str(error),
                    'steps_completed': self.shutdown_steps
                }
            )
            
            # Force system shutdown as last resort
            if self.mode != ShutdownMode.EMERGENCY:
                logger.critical("Attempting emergency force shutdown...")
                await self._force_system_shutdown()
            
        except Exception as e:
            logger.critical(f"Failed to handle shutdown failure: {e}")
    
    async def _force_system_shutdown(self):
        """Force system shutdown as last resort"""
        logger.critical("EXECUTING FORCE SYSTEM SHUTDOWN")
        
        try:
            # Kill all Python processes related to trading
            subprocess.run(['pkill', '-f', 'deep_momentum'], check=False)
            
            # Force kill if needed
            subprocess.run(['pkill', '-9', '-f', 'deep_momentum'], check=False)
            
        except Exception as e:
            logger.critical(f"Force system shutdown error: {e}")

def signal_handler(signum, frame):
    """Handle shutdown signals"""
    logger.critical(f"Received signal {signum}, initiating emergency shutdown")
    
    # Create emergency shutdown engine
    engine = EmergencyShutdownEngine(mode=ShutdownMode.EMERGENCY)
    
    # Run shutdown in event loop
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # Schedule shutdown
            asyncio.create_task(engine.execute_shutdown(f"Signal {signum}"))
        else:
            # Run shutdown
            asyncio.run(engine.execute_shutdown(f"Signal {signum}"))
    except Exception as e:
        logger.critical(f"Signal handler shutdown failed: {e}")
        sys.exit(1)

def setup_signal_handlers():
    """Setup signal handlers for emergency shutdown"""
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)
    if hasattr(signal, 'SIGHUP'):
        signal.signal(signal.SIGHUP, signal_handler)

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Emergency Shutdown Script')
    
    parser.add_argument('--mode', choices=['immediate', 'graceful', 'emergency'], 
                       default='graceful', help='Shutdown mode')
    parser.add_argument('--reason', default='Manual shutdown', help='Shutdown reason')
    parser.add_argument('--timeout', type=int, help='Shutdown timeout in seconds')
    parser.add_argument('--force', action='store_true', help='Force shutdown without confirmation')
    parser.add_argument('--save-state', action='store_true', help='Save recovery state')
    
    return parser.parse_args()

async def main():
    """Main shutdown function"""
    args = parse_arguments()
    
    # Setup signal handlers
    setup_signal_handlers()
    
    # Confirmation for non-force shutdowns
    if not args.force and args.mode != ShutdownMode.EMERGENCY:
        response = input(f"Are you sure you want to perform a {args.mode} shutdown? (yes/no): ")
        if response.lower() != 'yes':
            print("Shutdown cancelled")
            return
    
    # Create shutdown engine
    engine = EmergencyShutdownEngine(mode=args.mode)
    
    if args.timeout:
        engine.shutdown_timeout = args.timeout
    
    try:
        print(f"Initiating {args.mode} shutdown...")
        result = await engine.execute_shutdown(args.reason)
        
        # Print summary
        print(f"\n=== Shutdown Results ===")
        print(f"Shutdown ID: {result['shutdown_id']}")
        print(f"Mode: {result['mode']}")
        print(f"Status: {result['status']}")
        print(f"Reason: {result['reason']}")
        print(f"Execution Time: {result['execution_time']:.2f}s")
        print(f"Steps Completed: {len(result['steps_completed'])}")
        
        if result['status'] == 'completed':
            print("✅ Shutdown completed successfully")
        else:
            print("❌ Shutdown failed or incomplete")
        
        if args.save_state:
            print(f"Recovery state saved for shutdown ID: {result['shutdown_id']}")
        
    except Exception as e:
        logger.critical(f"Emergency shutdown failed: {e}")
        print(f"❌ Emergency shutdown failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nShutdown interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"Fatal error: {e}")
        sys.exit(1)