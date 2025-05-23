#!/usr/bin/env python3
"""
Enhanced Data Feed Startup Script with ARM64 Optimizations

This script provides comprehensive data feed startup capabilities for the Deep Momentum Trading System
with ARM64-specific optimizations, Polygon.io integration, and real-time processing.

Features:
- Polygon.io real-time data feeds (WebSocket and REST)
- ARM64-optimized data processing
- Data validation and quality checks
- Automatic reconnection and failover
- Performance monitoring and metrics
- Distributed data feed architecture
"""

import os
import sys
import argparse
import asyncio
import signal
import time
import platform
import threading
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
import logging
import json
from concurrent.futures import ThreadPoolExecutor
import multiprocessing as mp

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.real_time_feed import RealTimeFeed
from src.data.polygon_client import PolygonClient
from src.data.data_manager import DataManager
from src.communication.zmq_publisher import ZMQPublisher
from src.communication.message_broker import MessageBroker
from src.infrastructure.health_check import HealthChecker
from src.infrastructure.process_manager import ProcessManager
from src.monitoring.alert_system import AlertSystem
from src.monitoring.performance_tracker import PerformanceTracker
from src.utils.logger import get_logger
from src.utils.decorators import performance_monitor, error_handler
from src.utils.exceptions import DataFeedError
from config.settings import config_manager

logger = get_logger(__name__)

class ARM64DataFeedOptimizer:
    """ARM64-specific optimizations for data feed processing"""
    
    def __init__(self):
        self.is_arm64 = platform.machine().lower() in ['arm64', 'aarch64']
        self.cpu_count = mp.cpu_count()
        self.optimal_workers = self._calculate_optimal_workers()
        self.buffer_size = self._calculate_optimal_buffer_size()
        
    def _calculate_optimal_workers(self) -> int:
        """Calculate optimal number of workers for ARM64"""
        if self.is_arm64:
            # ARM64 benefits from moderate parallelism for I/O operations
            return min(self.cpu_count, 8)
        return min(self.cpu_count, 12)
    
    def _calculate_optimal_buffer_size(self) -> int:
        """Calculate optimal buffer size for ARM64"""
        if self.is_arm64:
            # ARM64 cache-friendly buffer sizes
            return 8192
        return 16384

class DataFeedConfig:
    """Data feed configuration"""
    
    def __init__(self, **kwargs):
        # Data source (Polygon.io focused)
        self.primary_source = 'polygon'
        
        # Symbols and markets
        self.symbols = kwargs.get('symbols', ['SPY', 'QQQ', 'IWM'])
        self.markets = kwargs.get('markets', ['stocks', 'crypto', 'forex'])
        
        # Data types
        self.data_types = kwargs.get('data_types', ['trades', 'quotes', 'bars'])
        self.enable_level2 = kwargs.get('enable_level2', False)
        
        # Processing
        self.enable_preprocessing = kwargs.get('enable_preprocessing', True)
        self.enable_validation = kwargs.get('enable_validation', True)
        self.buffer_size = kwargs.get('buffer_size', 10000)
        self.batch_size = kwargs.get('batch_size', 100)
        
        # Performance
        self.max_workers = kwargs.get('max_workers', None)
        self.enable_compression = kwargs.get('enable_compression', True)
        self.enable_caching = kwargs.get('enable_caching', True)
        
        # Reliability
        self.enable_failover = kwargs.get('enable_failover', True)
        self.reconnect_attempts = kwargs.get('reconnect_attempts', 5)
        self.reconnect_delay = kwargs.get('reconnect_delay', 5.0)
        
        # Output
        self.publish_zmq = kwargs.get('publish_zmq', True)
        self.zmq_port = kwargs.get('zmq_port', 5555)
        self.save_to_storage = kwargs.get('save_to_storage', True)
        
        # Monitoring
        self.enable_monitoring = kwargs.get('enable_monitoring', True)
        self.metrics_interval = kwargs.get('metrics_interval', 60.0)

class DataFeedEngine:
    """
    Enhanced data feed engine with ARM64 optimizations
    """
    
    def __init__(self, config: DataFeedConfig):
        self.config = config
        self.optimizer = ARM64DataFeedOptimizer()
        
        # Initialize components
        self.polygon_client = PolygonClient()
        self.real_time_feed = RealTimeFeed()
        self.data_manager = DataManager()
        self.health_checker = HealthChecker()
        self.process_manager = ProcessManager()
        self.alert_system = AlertSystem()
        self.performance_tracker = PerformanceTracker()
        
        # Communication
        self.message_broker = MessageBroker()
        self.zmq_publisher = None
        if config.publish_zmq:
            self.zmq_publisher = ZMQPublisher(port=config.zmq_port)
        
        # State management
        self.is_running = False
        self.feed_threads = {}
        self.data_buffer = {}
        self.stats = {
            'messages_received': 0,
            'messages_processed': 0,
            'messages_published': 0,
            'errors': 0,
            'start_time': None
        }
        
        # Signal handling
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        logger.info(f"DataFeedEngine initialized with ARM64 optimizations: {self.optimizer.is_arm64}")
    
    @performance_monitor
    @error_handler
    async def start_data_feed(self) -> Dict[str, Any]:
        """
        Start comprehensive data feed
        
        Returns:
            Dict containing startup results
        """
        logger.info("Starting enhanced data feed...")
        self.stats['start_time'] = time.time()
        
        try:
            # Initialize connections
            await self._initialize_connections()
            
            # Start data feeds
            await self._start_feeds()
            
            # Start processing pipeline
            await self._start_processing_pipeline()
            
            # Start monitoring
            if self.config.enable_monitoring:
                await self._start_monitoring()
            
            self.is_running = True
            logger.info("Data feed started successfully")
            
            # Keep running until stopped
            await self._run_main_loop()
            
            return {
                'status': 'stopped',
                'stats': self.stats,
                'uptime': time.time() - self.stats['start_time']
            }
            
        except Exception as e:
            logger.error(f"Data feed startup failed: {e}")
            await self._cleanup()
            raise DataFeedError(f"Data feed startup failed: {e}")
    
    async def _initialize_connections(self):
        """Initialize data source connections"""
        logger.info("Initializing data connections...")
        
        # Initialize Polygon client
        await self.polygon_client.connect()
        
        # Initialize message broker
        await self.message_broker.start()
        
        # Initialize ZMQ publisher
        if self.zmq_publisher:
            await self.zmq_publisher.start()
        
        logger.info("Data connections initialized")
    
    async def _start_feeds(self):
        """Start data feeds for configured symbols"""
        logger.info(f"Starting data feeds for {len(self.config.symbols)} symbols...")
        
        # Start feeds based on data types
        for data_type in self.config.data_types:
            if data_type == 'trades':
                await self._start_trades_feed()
            elif data_type == 'quotes':
                await self._start_quotes_feed()
            elif data_type == 'bars':
                await self._start_bars_feed()
        
        logger.info("Data feeds started")
    
    async def _start_trades_feed(self):
        """Start trades data feed"""
        logger.info("Starting trades feed...")
        
        async def trades_handler(trade_data):
            await self._process_trade_data(trade_data)
        
        # Subscribe to trades for all symbols
        for symbol in self.config.symbols:
            await self.polygon_client.subscribe_trades(symbol, trades_handler)
        
        logger.info(f"Trades feed started for {len(self.config.symbols)} symbols")
    
    async def _start_quotes_feed(self):
        """Start quotes data feed"""
        logger.info("Starting quotes feed...")
        
        async def quotes_handler(quote_data):
            await self._process_quote_data(quote_data)
        
        # Subscribe to quotes for all symbols
        for symbol in self.config.symbols:
            await self.polygon_client.subscribe_quotes(symbol, quotes_handler)
        
        logger.info(f"Quotes feed started for {len(self.config.symbols)} symbols")
    
    async def _start_bars_feed(self):
        """Start bars data feed"""
        logger.info("Starting bars feed...")
        
        async def bars_handler(bar_data):
            await self._process_bar_data(bar_data)
        
        # Subscribe to minute bars for all symbols
        for symbol in self.config.symbols:
            await self.polygon_client.subscribe_bars(symbol, bars_handler, timespan='minute')
        
        logger.info(f"Bars feed started for {len(self.config.symbols)} symbols")
    
    async def _start_processing_pipeline(self):
        """Start data processing pipeline"""
        logger.info("Starting data processing pipeline...")
        
        # Start processing workers
        max_workers = self.config.max_workers or self.optimizer.optimal_workers
        
        self.processing_executor = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="DataProcessor"
        )
        
        # Start buffer processing
        asyncio.create_task(self._process_data_buffer())
        
        logger.info(f"Processing pipeline started with {max_workers} workers")
    
    async def _process_trade_data(self, trade_data: Dict[str, Any]):
        """Process incoming trade data"""
        try:
            self.stats['messages_received'] += 1
            
            # Validate data
            if self.config.enable_validation:
                if not self._validate_trade_data(trade_data):
                    self.stats['errors'] += 1
                    return
            
            # Preprocess data
            if self.config.enable_preprocessing:
                trade_data = await self._preprocess_trade_data(trade_data)
            
            # Add to buffer
            symbol = trade_data.get('symbol', 'UNKNOWN')
            if symbol not in self.data_buffer:
                self.data_buffer[symbol] = {'trades': [], 'quotes': [], 'bars': []}
            
            self.data_buffer[symbol]['trades'].append(trade_data)
            
            # Publish if enabled
            if self.zmq_publisher:
                await self.zmq_publisher.publish('trades', trade_data)
                self.stats['messages_published'] += 1
            
            self.stats['messages_processed'] += 1
            
        except Exception as e:
            logger.error(f"Error processing trade data: {e}")
            self.stats['errors'] += 1
    
    async def _process_quote_data(self, quote_data: Dict[str, Any]):
        """Process incoming quote data"""
        try:
            self.stats['messages_received'] += 1
            
            # Validate data
            if self.config.enable_validation:
                if not self._validate_quote_data(quote_data):
                    self.stats['errors'] += 1
                    return
            
            # Preprocess data
            if self.config.enable_preprocessing:
                quote_data = await self._preprocess_quote_data(quote_data)
            
            # Add to buffer
            symbol = quote_data.get('symbol', 'UNKNOWN')
            if symbol not in self.data_buffer:
                self.data_buffer[symbol] = {'trades': [], 'quotes': [], 'bars': []}
            
            self.data_buffer[symbol]['quotes'].append(quote_data)
            
            # Publish if enabled
            if self.zmq_publisher:
                await self.zmq_publisher.publish('quotes', quote_data)
                self.stats['messages_published'] += 1
            
            self.stats['messages_processed'] += 1
            
        except Exception as e:
            logger.error(f"Error processing quote data: {e}")
            self.stats['errors'] += 1
    
    async def _process_bar_data(self, bar_data: Dict[str, Any]):
        """Process incoming bar data"""
        try:
            self.stats['messages_received'] += 1
            
            # Validate data
            if self.config.enable_validation:
                if not self._validate_bar_data(bar_data):
                    self.stats['errors'] += 1
                    return
            
            # Preprocess data
            if self.config.enable_preprocessing:
                bar_data = await self._preprocess_bar_data(bar_data)
            
            # Add to buffer
            symbol = bar_data.get('symbol', 'UNKNOWN')
            if symbol not in self.data_buffer:
                self.data_buffer[symbol] = {'trades': [], 'quotes': [], 'bars': []}
            
            self.data_buffer[symbol]['bars'].append(bar_data)
            
            # Publish if enabled
            if self.zmq_publisher:
                await self.zmq_publisher.publish('bars', bar_data)
                self.stats['messages_published'] += 1
            
            self.stats['messages_processed'] += 1
            
        except Exception as e:
            logger.error(f"Error processing bar data: {e}")
            self.stats['errors'] += 1
    
    def _validate_trade_data(self, trade_data: Dict[str, Any]) -> bool:
        """Validate trade data"""
        required_fields = ['symbol', 'price', 'size', 'timestamp']
        
        for field in required_fields:
            if field not in trade_data:
                logger.warning(f"Missing required field in trade data: {field}")
                return False
        
        # Validate price and size
        if trade_data['price'] <= 0 or trade_data['size'] <= 0:
            logger.warning("Invalid price or size in trade data")
            return False
        
        return True
    
    def _validate_quote_data(self, quote_data: Dict[str, Any]) -> bool:
        """Validate quote data"""
        required_fields = ['symbol', 'bid', 'ask', 'bid_size', 'ask_size', 'timestamp']
        
        for field in required_fields:
            if field not in quote_data:
                logger.warning(f"Missing required field in quote data: {field}")
                return False
        
        # Validate bid/ask
        if quote_data['bid'] <= 0 or quote_data['ask'] <= 0:
            logger.warning("Invalid bid or ask in quote data")
            return False
        
        if quote_data['bid'] >= quote_data['ask']:
            logger.warning("Bid >= Ask in quote data")
            return False
        
        return True
    
    def _validate_bar_data(self, bar_data: Dict[str, Any]) -> bool:
        """Validate bar data"""
        required_fields = ['symbol', 'open', 'high', 'low', 'close', 'volume', 'timestamp']
        
        for field in required_fields:
            if field not in bar_data:
                logger.warning(f"Missing required field in bar data: {field}")
                return False
        
        # Validate OHLC
        ohlc = [bar_data['open'], bar_data['high'], bar_data['low'], bar_data['close']]
        if any(price <= 0 for price in ohlc):
            logger.warning("Invalid OHLC prices in bar data")
            return False
        
        if bar_data['high'] < max(bar_data['open'], bar_data['close']):
            logger.warning("High price inconsistent in bar data")
            return False
        
        if bar_data['low'] > min(bar_data['open'], bar_data['close']):
            logger.warning("Low price inconsistent in bar data")
            return False
        
        return True
    
    async def _preprocess_trade_data(self, trade_data: Dict[str, Any]) -> Dict[str, Any]:
        """Preprocess trade data"""
        # Add processing timestamp
        trade_data['processed_at'] = time.time()
        
        # Normalize symbol
        trade_data['symbol'] = trade_data['symbol'].upper()
        
        # Add derived fields
        trade_data['value'] = trade_data['price'] * trade_data['size']
        
        return trade_data
    
    async def _preprocess_quote_data(self, quote_data: Dict[str, Any]) -> Dict[str, Any]:
        """Preprocess quote data"""
        # Add processing timestamp
        quote_data['processed_at'] = time.time()
        
        # Normalize symbol
        quote_data['symbol'] = quote_data['symbol'].upper()
        
        # Add derived fields
        quote_data['spread'] = quote_data['ask'] - quote_data['bid']
        quote_data['mid_price'] = (quote_data['bid'] + quote_data['ask']) / 2
        
        return quote_data
    
    async def _preprocess_bar_data(self, bar_data: Dict[str, Any]) -> Dict[str, Any]:
        """Preprocess bar data"""
        # Add processing timestamp
        bar_data['processed_at'] = time.time()
        
        # Normalize symbol
        bar_data['symbol'] = bar_data['symbol'].upper()
        
        # Add derived fields
        bar_data['vwap'] = bar_data.get('vwap', bar_data['close'])  # Use close if VWAP not available
        bar_data['typical_price'] = (bar_data['high'] + bar_data['low'] + bar_data['close']) / 3
        
        return bar_data
    
    async def _process_data_buffer(self):
        """Process data buffer periodically"""
        while self.is_running:
            try:
                if self.config.save_to_storage:
                    await self._save_buffered_data()
                
                # Clear processed data from buffer
                await self._cleanup_buffer()
                
                await asyncio.sleep(10)  # Process every 10 seconds
                
            except Exception as e:
                logger.error(f"Error processing data buffer: {e}")
                await asyncio.sleep(5)
    
    async def _save_buffered_data(self):
        """Save buffered data to storage"""
        for symbol, data in self.data_buffer.items():
            try:
                # Save trades
                if data['trades']:
                    await self.data_manager.save_trades(symbol, data['trades'])
                
                # Save quotes
                if data['quotes']:
                    await self.data_manager.save_quotes(symbol, data['quotes'])
                
                # Save bars
                if data['bars']:
                    await self.data_manager.save_bars(symbol, data['bars'])
                
            except Exception as e:
                logger.error(f"Error saving data for {symbol}: {e}")
    
    async def _cleanup_buffer(self):
        """Cleanup old data from buffer"""
        current_time = time.time()
        
        for symbol in self.data_buffer:
            for data_type in ['trades', 'quotes', 'bars']:
                # Keep only recent data (last 5 minutes)
                self.data_buffer[symbol][data_type] = [
                    item for item in self.data_buffer[symbol][data_type]
                    if current_time - item.get('timestamp', 0) < 300
                ]
    
    async def _start_monitoring(self):
        """Start monitoring and metrics collection"""
        logger.info("Starting monitoring...")
        
        asyncio.create_task(self._metrics_loop())
        
        # Start health checks
        asyncio.create_task(self._health_check_loop())
        
        logger.info("Monitoring started")
    
    async def _metrics_loop(self):
        """Metrics collection loop"""
        while self.is_running:
            try:
                # Calculate metrics
                uptime = time.time() - self.stats['start_time']
                messages_per_second = self.stats['messages_processed'] / max(uptime, 1)
                error_rate = self.stats['errors'] / max(self.stats['messages_received'], 1)
                
                metrics = {
                    'uptime': uptime,
                    'messages_per_second': messages_per_second,
                    'error_rate': error_rate,
                    'buffer_size': sum(
                        len(data['trades']) + len(data['quotes']) + len(data['bars'])
                        for data in self.data_buffer.values()
                    ),
                    **self.stats
                }
                
                # Track performance
                await self.performance_tracker.record_metrics('data_feed', metrics)
                
                # Log metrics
                logger.info(f"Data Feed Metrics - MPS: {messages_per_second:.1f}, "
                          f"Error Rate: {error_rate:.3f}, Buffer: {metrics['buffer_size']}")
                
                await asyncio.sleep(self.config.metrics_interval)
                
            except Exception as e:
                logger.error(f"Error in metrics loop: {e}")
                await asyncio.sleep(30)
    
    async def _health_check_loop(self):
        """Health check loop"""
        while self.is_running:
            try:
                # Check Polygon connection
                polygon_healthy = await self.polygon_client.health_check()
                
                # Check message broker
                broker_healthy = await self.message_broker.health_check()
                
                # Check overall health
                overall_healthy = polygon_healthy and broker_healthy
                
                if not overall_healthy:
                    logger.warning("Data feed health check failed")
                    await self.alert_system.send_alert(
                        level='warning',
                        message='Data feed health check failed',
                        context={
                            'polygon_healthy': polygon_healthy,
                            'broker_healthy': broker_healthy
                        }
                    )
                
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logger.error(f"Error in health check loop: {e}")
                await asyncio.sleep(60)
    
    async def _run_main_loop(self):
        """Main event loop"""
        logger.info("Data feed running... Press Ctrl+C to stop")
        
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
        logger.info("Cleaning up data feed...")
        
        self.is_running = False
        
        # Close connections
        if self.polygon_client:
            await self.polygon_client.disconnect()
        
        if self.message_broker:
            await self.message_broker.stop()
        
        if self.zmq_publisher:
            await self.zmq_publisher.stop()
        
        # Shutdown executors
        if hasattr(self, 'processing_executor'):
            self.processing_executor.shutdown(wait=True)
        
        # Save final buffer data
        if self.config.save_to_storage:
            await self._save_buffered_data()
        
        logger.info("Data feed cleanup completed")

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Enhanced Data Feed Startup Script')
    
    # Symbols
    parser.add_argument('--symbols', nargs='+', default=['SPY', 'QQQ', 'IWM'], 
                       help='Symbols to subscribe to')
    parser.add_argument('--symbol-file', help='File containing symbols (one per line)')
    
    # Data types
    parser.add_argument('--data-types', nargs='+', default=['trades', 'quotes', 'bars'], 
                       choices=['trades', 'quotes', 'bars'], help='Data types to subscribe to')
    parser.add_argument('--enable-level2', action='store_true', help='Enable Level 2 data')
    
    # Processing
    parser.add_argument('--batch-size', type=int, default=100, help='Processing batch size')
    parser.add_argument('--buffer-size', type=int, default=10000, help='Data buffer size')
    parser.add_argument('--max-workers', type=int, help='Maximum number of workers')
    
    # Output
    parser.add_argument('--zmq-port', type=int, default=5555, help='ZMQ publisher port')
    parser.add_argument('--no-zmq', action='store_true', help='Disable ZMQ publishing')
    parser.add_argument('--no-storage', action='store_true', help='Disable data storage')
    
    # Options
    parser.add_argument('--no-validation', action='store_true', help='Disable data validation')
    parser.add_argument('--no-preprocessing', action='store_true', help='Disable data preprocessing')
    parser.add_argument('--no-monitoring', action='store_true', help='Disable monitoring')
    parser.add_argument('--verbose', action='store_true', help='Verbose logging')
    
    return parser.parse_args()

async def main():
    """Main data feed function"""
    args = parse_arguments()
    
    # Setup logging
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Prepare symbols
    symbols = args.symbols.copy()
    
    if args.symbol_file:
        with open(args.symbol_file, 'r') as f:
            file_symbols = [line.strip().upper() for line in f if line.strip()]
        symbols.extend(file_symbols)
    
    # Remove duplicates
    symbols = list(set(symbols))
    
    # Create configuration
    config = DataFeedConfig(
        symbols=symbols,
        data_types=args.data_types,
        enable_level2=args.enable_level2,
        batch_size=args.batch_size,
        buffer_size=args.buffer_size,
        max_workers=args.max_workers,
        publish_zmq=not args.no_zmq,
        zmq_port=args.zmq_port,
        save_to_storage=not args.no_storage,
        enable_validation=not args.no_validation,
        enable_preprocessing=not args.no_preprocessing,
        enable_monitoring=not args.no_monitoring
    )
    
    # Initialize and start data feed
    engine = DataFeedEngine(config)
    
    try:
        print(f"Starting data feed for {len(symbols)} symbols...")
        print(f"Data types: {', '.join(args.data_types)}")
        print(f"ZMQ publishing: {'enabled' if config.publish_zmq else 'disabled'}")
        print(f"Data storage: {'enabled' if config.save_to_storage else 'disabled'}")
        
        result = await engine.start_data_feed()
        
        # Print summary
        print(f"\n=== Data Feed Results ===")
        print(f"Status: {result['status']}")
        print(f"Uptime: {result['uptime']:.2f}s")
        print(f"Messages Processed: {result['stats']['messages_processed']:,}")
        print(f"Messages Published: {result['stats']['messages_published']:,}")
        print(f"Errors: {result['stats']['errors']}")
        
    except Exception as e:
        logger.error(f"Data feed failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())