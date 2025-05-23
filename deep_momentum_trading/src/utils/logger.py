"""
Enhanced logging system for Deep Momentum Trading System with ARM64 optimizations.

This module provides comprehensive logging capabilities with structured logging,
performance monitoring, distributed logging, and ARM64-specific optimizations.
"""

import logging
import logging.handlers
import os
import sys
import time
import threading
import json
import gzip
from typing import Dict, Any, Optional, List, Union
from datetime import datetime, timezone
from dataclasses import dataclass, asdict
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import queue
import atexit
from enum import Enum
import traceback
import psutil

# Global logger registry
_loggers: Dict[str, logging.Logger] = {}
_logger_lock = threading.RLock()

class LogLevel(Enum):
    """Enhanced log levels."""
    TRACE = 5
    DEBUG = 10
    INFO = 20
    WARNING = 30
    ERROR = 40
    CRITICAL = 50
    PERFORMANCE = 25  # Custom level for performance metrics
    TRADING = 35      # Custom level for trading events

# Add custom log levels
logging.addLevelName(LogLevel.TRACE.value, "TRACE")
logging.addLevelName(LogLevel.PERFORMANCE.value, "PERFORMANCE")
logging.addLevelName(LogLevel.TRADING.value, "TRADING")

@dataclass
class LogConfig:
    """Enhanced logging configuration."""
    name: str = "deep_momentum_trading"
    level: Union[str, int] = LogLevel.INFO.value
    log_dir: str = "deep_momentum_trading/logs"
    log_file: str = "system.log"
    max_bytes: int = 50 * 1024 * 1024  # 50MB
    backup_count: int = 10
    console_output: bool = True
    json_format: bool = False
    enable_compression: bool = True
    enable_performance_logging: bool = True
    enable_trading_logging: bool = True
    enable_distributed_logging: bool = False
    distributed_endpoint: Optional[str] = None
    buffer_size: int = 1000
    flush_interval: float = 5.0
    enable_arm64_optimizations: bool = True
    log_rotation_time: str = "midnight"  # midnight, hourly
    utc_timestamps: bool = True
    include_process_info: bool = True
    include_thread_info: bool = True
    include_memory_info: bool = False
    max_message_size: int = 10000  # Max characters per log message
    
class StructuredFormatter(logging.Formatter):
    """Enhanced structured formatter with ARM64 optimizations."""
    
    def __init__(self, 
                 config: LogConfig,
                 include_extra_fields: bool = True):
        self.config = config
        self.include_extra_fields = include_extra_fields
        self.hostname = os.uname().nodename
        self.pid = os.getpid()
        
        # ARM64 detection
        import platform
        self.is_arm64 = platform.machine().lower() in ['arm64', 'aarch64']
        
        super().__init__()
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record with structured data."""
        # Create base log entry
        log_entry = {
            "timestamp": self._format_timestamp(record.created),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage()[:self.config.max_message_size],
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno
        }
        
        # Add process/thread info if enabled
        if self.config.include_process_info:
            log_entry.update({
                "hostname": self.hostname,
                "pid": self.pid,
                "process_name": record.processName
            })
        
        if self.config.include_thread_info:
            log_entry.update({
                "thread_id": record.thread,
                "thread_name": record.threadName
            })
        
        # Add memory info if enabled
        if self.config.include_memory_info:
            try:
                process = psutil.Process()
                memory_info = process.memory_info()
                log_entry["memory_mb"] = round(memory_info.rss / 1024 / 1024, 2)
            except:
                pass
        
        # Add ARM64 info
        if self.config.enable_arm64_optimizations:
            log_entry["arch"] = "arm64" if self.is_arm64 else "x86_64"
        
        # Add exception info if present
        if record.exc_info:
            log_entry["exception"] = {
                "type": record.exc_info[0].__name__,
                "message": str(record.exc_info[1]),
                "traceback": traceback.format_exception(*record.exc_info)
            }
        
        # Add extra fields from record
        if self.include_extra_fields:
            extra_fields = {}
            for key, value in record.__dict__.items():
                if key not in ['name', 'msg', 'args', 'levelname', 'levelno', 
                              'pathname', 'filename', 'module', 'lineno', 
                              'funcName', 'created', 'msecs', 'relativeCreated',
                              'thread', 'threadName', 'processName', 'process',
                              'getMessage', 'exc_info', 'exc_text', 'stack_info']:
                    try:
                        # Only include JSON-serializable values
                        json.dumps(value)
                        extra_fields[key] = value
                    except (TypeError, ValueError):
                        extra_fields[key] = str(value)
            
            if extra_fields:
                log_entry["extra"] = extra_fields
        
        # Format output
        if self.config.json_format:
            return json.dumps(log_entry, default=str)
        else:
            # Human-readable format
            timestamp = log_entry["timestamp"]
            level = log_entry["level"]
            logger_name = log_entry["logger"]
            message = log_entry["message"]
            location = f"{log_entry['module']}:{log_entry['function']}:{log_entry['line']}"
            
            formatted = f"{timestamp} - {level:>11} - {logger_name} - {location} - {message}"
            
            # Add exception info
            if "exception" in log_entry:
                formatted += f"\nException: {log_entry['exception']['type']}: {log_entry['exception']['message']}"
            
            return formatted
    
    def _format_timestamp(self, created: float) -> str:
        """Format timestamp with timezone support."""
        if self.config.utc_timestamps:
            dt = datetime.fromtimestamp(created, tz=timezone.utc)
            return dt.isoformat()
        else:
            dt = datetime.fromtimestamp(created)
            return dt.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

class AsyncFileHandler(logging.Handler):
    """Asynchronous file handler with ARM64 optimizations."""
    
    def __init__(self, 
                 filename: str,
                 config: LogConfig):
        super().__init__()
        self.filename = filename
        self.config = config
        
        # Create async queue and worker
        self.log_queue = queue.Queue(maxsize=config.buffer_size)
        self.worker_thread = threading.Thread(target=self._worker, daemon=True)
        self.shutdown_event = threading.Event()
        
        # File handler for actual writing
        self.file_handler = self._create_file_handler()
        
        # Start worker thread
        self.worker_thread.start()
        
        # Register cleanup
        atexit.register(self.close)
    
    def _create_file_handler(self) -> logging.Handler:
        """Create appropriate file handler based on configuration."""
        if self.config.log_rotation_time == "midnight":
            handler = logging.handlers.TimedRotatingFileHandler(
                filename=self.filename,
                when='midnight',
                interval=1,
                backupCount=self.config.backup_count,
                encoding='utf-8',
                utc=self.config.utc_timestamps
            )
        else:
            handler = logging.handlers.RotatingFileHandler(
                filename=self.filename,
                maxBytes=self.config.max_bytes,
                backupCount=self.config.backup_count,
                encoding='utf-8'
            )
        
        # Add compression if enabled
        if self.config.enable_compression:
            handler.rotator = self._compress_rotator
        
        return handler
    
    def _compress_rotator(self, source: str, dest: str):
        """Compress rotated log files."""
        try:
            with open(source, 'rb') as f_in:
                with gzip.open(f"{dest}.gz", 'wb') as f_out:
                    f_out.writelines(f_in)
            os.remove(source)
        except Exception as e:
            # Fallback to regular rotation
            os.rename(source, dest)
    
    def emit(self, record: logging.LogRecord):
        """Emit log record to async queue."""
        try:
            self.log_queue.put_nowait(record)
        except queue.Full:
            # Drop oldest record and add new one
            try:
                self.log_queue.get_nowait()
                self.log_queue.put_nowait(record)
            except queue.Empty:
                pass
    
    def _worker(self):
        """Worker thread for async log writing."""
        records_buffer = []
        last_flush = time.time()
        
        while not self.shutdown_event.is_set():
            try:
                # Get record with timeout
                try:
                    record = self.log_queue.get(timeout=0.1)
                    records_buffer.append(record)
                except queue.Empty:
                    pass
                
                # Flush buffer if needed
                current_time = time.time()
                should_flush = (
                    len(records_buffer) >= 100 or  # Buffer size threshold
                    (records_buffer and current_time - last_flush >= self.config.flush_interval)
                )
                
                if should_flush:
                    self._flush_buffer(records_buffer)
                    records_buffer.clear()
                    last_flush = current_time
                
            except Exception as e:
                # Log to stderr to avoid infinite recursion
                print(f"Error in async log worker: {e}", file=sys.stderr)
        
        # Flush remaining records on shutdown
        if records_buffer:
            self._flush_buffer(records_buffer)
    
    def _flush_buffer(self, records: List[logging.LogRecord]):
        """Flush buffer of records to file."""
        for record in records:
            try:
                self.file_handler.emit(record)
            except Exception as e:
                print(f"Error writing log record: {e}", file=sys.stderr)
        
        try:
            self.file_handler.flush()
        except Exception as e:
            print(f"Error flushing log handler: {e}", file=sys.stderr)
    
    def close(self):
        """Close handler and cleanup resources."""
        self.shutdown_event.set()
        if self.worker_thread.is_alive():
            self.worker_thread.join(timeout=5.0)
        
        try:
            self.file_handler.close()
        except:
            pass
        
        super().close()

class PerformanceLogger:
    """Specialized logger for performance metrics."""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.metrics_buffer = []
        self.buffer_lock = threading.Lock()
    
    def log_performance(self, 
                       metric_name: str,
                       value: float,
                       unit: str = "ms",
                       tags: Optional[Dict[str, str]] = None):
        """Log performance metric."""
        metric_data = {
            "metric_name": metric_name,
            "value": value,
            "unit": unit,
            "timestamp": time.time(),
            "tags": tags or {}
        }
        
        with self.buffer_lock:
            self.metrics_buffer.append(metric_data)
        
        # Log to performance level
        self.logger.log(
            LogLevel.PERFORMANCE.value,
            f"Performance metric: {metric_name}={value}{unit}",
            extra={"performance_metric": metric_data}
        )
    
    def log_latency(self, operation: str, latency_ms: float, **kwargs):
        """Log latency metric."""
        self.log_performance(f"latency_{operation}", latency_ms, "ms", kwargs)
    
    def log_throughput(self, operation: str, ops_per_sec: float, **kwargs):
        """Log throughput metric."""
        self.log_performance(f"throughput_{operation}", ops_per_sec, "ops/s", kwargs)
    
    def log_memory_usage(self, component: str, memory_mb: float, **kwargs):
        """Log memory usage metric."""
        self.log_performance(f"memory_{component}", memory_mb, "MB", kwargs)

class TradingLogger:
    """Specialized logger for trading events."""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
    
    def log_order(self, order_data: Dict[str, Any]):
        """Log order event."""
        self.logger.log(
            LogLevel.TRADING.value,
            f"Order: {order_data.get('action', 'unknown')} {order_data.get('symbol', 'unknown')}",
            extra={"trading_event": "order", "order_data": order_data}
        )
    
    def log_trade(self, trade_data: Dict[str, Any]):
        """Log trade execution."""
        self.logger.log(
            LogLevel.TRADING.value,
            f"Trade: {trade_data.get('symbol', 'unknown')} @ {trade_data.get('price', 'unknown')}",
            extra={"trading_event": "trade", "trade_data": trade_data}
        )
    
    def log_position(self, position_data: Dict[str, Any]):
        """Log position update."""
        self.logger.log(
            LogLevel.TRADING.value,
            f"Position: {position_data.get('symbol', 'unknown')} size={position_data.get('size', 'unknown')}",
            extra={"trading_event": "position", "position_data": position_data}
        )
    
    def log_risk_event(self, risk_data: Dict[str, Any]):
        """Log risk management event."""
        self.logger.log(
            LogLevel.TRADING.value,
            f"Risk: {risk_data.get('event_type', 'unknown')} - {risk_data.get('message', '')}",
            extra={"trading_event": "risk", "risk_data": risk_data}
        )

def setup_logging(config: LogConfig) -> None:
    """Setup global logging configuration."""
    # Create log directory
    log_dir = Path(config.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Suppress noisy third-party loggers
    noisy_loggers = [
        "websockets", "asyncio", "h5py", "numba", "urllib3", 
        "httpx", "zmq", "matplotlib", "PIL", "requests"
    ]
    
    for logger_name in noisy_loggers:
        logging.getLogger(logger_name).setLevel(logging.WARNING)
    
    # Set root logger level
    logging.getLogger().setLevel(config.level)

def get_logger(name: str, config: Optional[LogConfig] = None) -> logging.Logger:
    """
    Get or create enhanced logger instance.
    
    Args:
        name: Logger name
        config: Optional logging configuration
        
    Returns:
        Configured logger instance
    """
    with _logger_lock:
        if name in _loggers:
            return _loggers[name]
        
        # Use provided config or create default
        if config is None:
            config = LogConfig(name=name)
        
        # Create logger
        logger = logging.getLogger(name)
        logger.setLevel(config.level)
        logger.propagate = False
        
        # Clear existing handlers
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
        
        # Create formatter
        formatter = StructuredFormatter(config)
        
        # Setup file logging
        log_file_path = Path(config.log_dir) / config.log_file
        
        if config.enable_arm64_optimizations:
            # Use async handler for better ARM64 performance
            file_handler = AsyncFileHandler(str(log_file_path), config)
        else:
            # Use standard rotating file handler
            file_handler = logging.handlers.RotatingFileHandler(
                filename=str(log_file_path),
                maxBytes=config.max_bytes,
                backupCount=config.backup_count,
                encoding='utf-8'
            )
        
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        # Setup console logging
        if config.console_output:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)
        
        # Add custom methods to logger
        logger.trace = lambda msg, *args, **kwargs: logger.log(LogLevel.TRACE.value, msg, *args, **kwargs)
        logger.performance = lambda msg, *args, **kwargs: logger.log(LogLevel.PERFORMANCE.value, msg, *args, **kwargs)
        logger.trading = lambda msg, *args, **kwargs: logger.log(LogLevel.TRADING.value, msg, *args, **kwargs)
        
        # Add specialized loggers
        logger.perf = PerformanceLogger(logger)
        logger.trading_events = TradingLogger(logger)
        
        # Store logger
        _loggers[name] = logger
        
        logger.info(f"Enhanced logger '{name}' initialized with ARM64 optimizations: {config.enable_arm64_optimizations}")
        
        return logger

def get_performance_logger(name: str) -> PerformanceLogger:
    """Get performance logger instance."""
    logger = get_logger(name)
    return logger.perf

def get_trading_logger(name: str) -> TradingLogger:
    """Get trading logger instance."""
    logger = get_logger(name)
    return logger.trading_events

def configure_distributed_logging(endpoint: str, 
                                 api_key: Optional[str] = None) -> None:
    """Configure distributed logging to external service."""
    # Implementation would depend on specific logging service
    # (e.g., ELK stack, Splunk, CloudWatch, etc.)
    pass

def log_system_info(logger: logging.Logger) -> None:
    """Log comprehensive system information."""
    try:
        import platform
        
        system_info = {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "python_version": platform.python_version(),
            "cpu_count": os.cpu_count(),
        }
        
        # Add memory info
        try:
            memory = psutil.virtual_memory()
            system_info.update({
                "total_memory_gb": round(memory.total / 1024**3, 2),
                "available_memory_gb": round(memory.available / 1024**3, 2),
                "memory_percent": memory.percent
            })
        except:
            pass
        
        # Add ARM64 specific info
        if platform.machine().lower() in ['arm64', 'aarch64']:
            system_info["arm64_optimizations"] = True
            
            # Check for NEON support
            try:
                with open('/proc/cpuinfo', 'r') as f:
                    cpuinfo = f.read()
                    system_info["neon_support"] = 'neon' in cpuinfo.lower()
            except:
                pass
        
        logger.info("System information", extra={"system_info": system_info})
        
    except Exception as e:
        logger.warning(f"Failed to log system info: {e}")

def cleanup_loggers() -> None:
    """Cleanup all loggers and handlers."""
    with _logger_lock:
        for logger in _loggers.values():
            for handler in logger.handlers[:]:
                try:
                    handler.close()
                except:
                    pass
                logger.removeHandler(handler)
        
        _loggers.clear()

# Register cleanup on exit
atexit.register(cleanup_loggers)

# Export all public components
__all__ = [
    "LogLevel",
    "LogConfig", 
    "StructuredFormatter",
    "AsyncFileHandler",
    "PerformanceLogger",
    "TradingLogger",
    "setup_logging",
    "get_logger",
    "get_performance_logger",
    "get_trading_logger",
    "configure_distributed_logging",
    "log_system_info",
    "cleanup_loggers"
]

if __name__ == "__main__":
    # Example usage with ARM64 optimizations
    config = LogConfig(
        name="test_logger",
        level=LogLevel.DEBUG.value,
        log_file="test_enhanced.log",
        json_format=True,
        enable_arm64_optimizations=True,
        enable_performance_logging=True,
        enable_trading_logging=True
    )
    
    logger = get_logger("test_enhanced", config)
    
    # Log system info
    log_system_info(logger)
    
    # Test different log levels
    logger.trace("This is a trace message")
    logger.debug("This is a debug message")
    logger.info("This is an info message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")
    logger.critical("This is a critical message")
    
    # Test performance logging
    logger.perf.log_latency("database_query", 15.5, table="trades")
    logger.perf.log_throughput("order_processing", 1250.0, symbol="AAPL")
    logger.perf.log_memory_usage("model_inference", 512.0, model="LSTM")
    
    # Test trading logging
    logger.trading_events.log_order({
        "symbol": "AAPL",
        "action": "BUY",
        "quantity": 100,
        "price": 150.25,
        "order_id": "ORD123"
    })
    
    logger.trading_events.log_trade({
        "symbol": "AAPL", 
        "price": 150.30,
        "quantity": 100,
        "side": "BUY"
    })
    
    # Test exception logging
    try:
        1 / 0
    except ZeroDivisionError:
        logger.exception("Division by zero error occurred")
    
    print(f"Enhanced logging test completed. Check logs in: {config.log_dir}/{config.log_file}")
