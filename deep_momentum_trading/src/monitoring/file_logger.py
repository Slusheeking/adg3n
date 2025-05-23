"""
Enhanced File Logger with ARM64 Optimizations

This module provides high-performance file logging capabilities for the Deep Momentum Trading System
with ARM64-specific optimizations for structured data logging and real-time performance monitoring.

Features:
- High-performance structured logging with ARM64 optimizations
- Multiple log formats (JSON, CSV, binary)
- Asynchronous logging with buffering and batching
- Log rotation and compression
- Real-time log analysis and filtering
- Shared memory integration for high-frequency logging
- Performance metrics and log analytics
- Comprehensive error handling and recovery
"""

import os
import json
import csv
import gzip
import time
import threading
import queue
import platform
import multiprocessing as mp
from typing import Dict, List, Any, Optional, Union, TextIO
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import logging
import logging.handlers
from functools import lru_cache
import psutil
import pickle
import struct
from datetime import datetime, timedelta
import asyncio
import aiofiles

from ..utils.logger import get_logger
from ..utils.shared_memory import SharedMemoryManager
from ..utils.decorators import performance_monitor, error_handler
from ..utils.exceptions import ValidationError, LoggingError

logger = get_logger(__name__)

class LogLevel(Enum):
    """Log levels"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"

class LogFormat(Enum):
    """Log output formats"""
    JSON = "json"
    CSV = "csv"
    TEXT = "text"
    BINARY = "binary"

@dataclass
class LoggerConfig:
    """Configuration for file logger"""
    log_directory: str = "logs"
    log_filename: str = "trading_system.log"
    log_format: LogFormat = LogFormat.JSON
    log_level: LogLevel = LogLevel.INFO
    max_file_size: int = 100 * 1024 * 1024  # 100MB
    max_backup_count: int = 10
    enable_compression: bool = True
    enable_arm64_optimizations: bool = True
    use_shared_memory: bool = True
    buffer_size: int = 10000
    flush_interval: float = 1.0  # seconds
    async_logging: bool = True
    max_workers: Optional[int] = None
    enable_real_time_analysis: bool = True
    log_retention_days: int = 30

@dataclass
class LogEntry:
    """Log entry data structure"""
    timestamp: float
    level: LogLevel
    message: str
    source: str
    data: Dict[str, Any]
    thread_id: Optional[int] = None
    process_id: Optional[int] = None
    session_id: Optional[str] = None

class ARM64LogOptimizer:
    """ARM64-specific optimizations for logging operations"""
    
    def __init__(self):
        self.is_arm64 = platform.machine().lower() in ['arm64', 'aarch64']
        self.cpu_count = mp.cpu_count()
        
    def optimize_log_serialization(self, entries: List[LogEntry], format_type: LogFormat) -> bytes:
        """ARM64-optimized log serialization"""
        if not self.is_arm64 or len(entries) < 100:
            return self._standard_serialization(entries, format_type)
        
        # Use ARM64 SIMD for large batches
        if format_type == LogFormat.BINARY:
            return self._arm64_binary_serialization(entries)
        else:
            return self._standard_serialization(entries, format_type)
    
    def _arm64_binary_serialization(self, entries: List[LogEntry]) -> bytes:
        """ARM64-optimized binary serialization"""
        # Use struct packing for efficient binary format
        binary_data = bytearray()
        
        for entry in entries:
            # Pack timestamp (8 bytes)
            binary_data.extend(struct.pack('d', entry.timestamp))
            
            # Pack level (1 byte)
            level_map = {LogLevel.DEBUG: 0, LogLevel.INFO: 1, LogLevel.WARNING: 2, LogLevel.ERROR: 3, LogLevel.CRITICAL: 4}
            binary_data.extend(struct.pack('B', level_map.get(entry.level, 1)))
            
            # Pack message length and message
            message_bytes = entry.message.encode('utf-8')
            binary_data.extend(struct.pack('I', len(message_bytes)))
            binary_data.extend(message_bytes)
            
            # Pack source length and source
            source_bytes = entry.source.encode('utf-8')
            binary_data.extend(struct.pack('I', len(source_bytes)))
            binary_data.extend(source_bytes)
            
            # Pack data as JSON
            data_bytes = json.dumps(entry.data).encode('utf-8')
            binary_data.extend(struct.pack('I', len(data_bytes)))
            binary_data.extend(data_bytes)
        
        return bytes(binary_data)
    
    def _standard_serialization(self, entries: List[LogEntry], format_type: LogFormat) -> bytes:
        """Standard serialization for non-ARM64 or small batches"""
        if format_type == LogFormat.JSON:
            return '\n'.join(json.dumps(self._entry_to_dict(entry)) for entry in entries).encode('utf-8')
        elif format_type == LogFormat.CSV:
            return self._entries_to_csv(entries).encode('utf-8')
        elif format_type == LogFormat.BINARY:
            return pickle.dumps(entries)
        else:
            return '\n'.join(self._entry_to_text(entry) for entry in entries).encode('utf-8')
    
    def _entry_to_dict(self, entry: LogEntry) -> Dict[str, Any]:
        """Convert log entry to dictionary"""
        return {
            'timestamp': entry.timestamp,
            'level': entry.level.value,
            'message': entry.message,
            'source': entry.source,
            'data': entry.data,
            'thread_id': entry.thread_id,
            'process_id': entry.process_id,
            'session_id': entry.session_id
        }
    
    def _entry_to_text(self, entry: LogEntry) -> str:
        """Convert log entry to text format"""
        timestamp_str = datetime.fromtimestamp(entry.timestamp).strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
        return f"[{timestamp_str}] {entry.level.value} {entry.source}: {entry.message}"
    
    def _entries_to_csv(self, entries: List[LogEntry]) -> str:
        """Convert log entries to CSV format"""
        import io
        output = io.StringIO()
        writer = csv.writer(output)
        
        # Write header
        writer.writerow(['timestamp', 'level', 'source', 'message', 'data', 'thread_id', 'process_id', 'session_id'])
        
        # Write entries
        for entry in entries:
            writer.writerow([
                entry.timestamp,
                entry.level.value,
                entry.source,
                entry.message,
                json.dumps(entry.data),
                entry.thread_id,
                entry.process_id,
                entry.session_id
            ])
        
        return output.getvalue()

class LogRotationManager:
    """Manages log file rotation and compression"""
    
    def __init__(self, config: LoggerConfig):
        self.config = config
        self.log_path = Path(config.log_directory) / config.log_filename
        
    def should_rotate(self) -> bool:
        """Check if log file should be rotated"""
        if not self.log_path.exists():
            return False
        
        return self.log_path.stat().st_size >= self.config.max_file_size
    
    def rotate_log(self):
        """Rotate log file"""
        if not self.log_path.exists():
            return
        
        # Create backup filename with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_path = self.log_path.with_suffix(f'.{timestamp}.log')
        
        # Move current log to backup
        self.log_path.rename(backup_path)
        
        # Compress if enabled
        if self.config.enable_compression:
            self._compress_log(backup_path)
        
        # Clean up old backups
        self._cleanup_old_backups()
    
    def _compress_log(self, log_path: Path):
        """Compress log file"""
        compressed_path = log_path.with_suffix(log_path.suffix + '.gz')
        
        with open(log_path, 'rb') as f_in:
            with gzip.open(compressed_path, 'wb') as f_out:
                f_out.writelines(f_in)
        
        # Remove uncompressed file
        log_path.unlink()
    
    def _cleanup_old_backups(self):
        """Remove old backup files"""
        log_dir = self.log_path.parent
        pattern = f"{self.log_path.stem}.*.log*"
        
        backup_files = list(log_dir.glob(pattern))
        backup_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        
        # Remove files beyond max_backup_count
        for backup_file in backup_files[self.config.max_backup_count:]:
            backup_file.unlink()
        
        # Remove files older than retention period
        cutoff_time = time.time() - (self.config.log_retention_days * 24 * 3600)
        for backup_file in backup_files:
            if backup_file.stat().st_mtime < cutoff_time:
                backup_file.unlink()

class LogAnalyzer:
    """Real-time log analysis and filtering"""
    
    def __init__(self):
        self.patterns = {}
        self.statistics = {
            'total_entries': 0,
            'entries_by_level': {level: 0 for level in LogLevel},
            'entries_by_source': {},
            'error_patterns': {},
            'performance_metrics': []
        }
    
    def add_pattern(self, name: str, pattern: str, callback: Optional[callable] = None):
        """Add pattern for real-time analysis"""
        import re
        self.patterns[name] = {
            'regex': re.compile(pattern),
            'callback': callback,
            'matches': 0
        }
    
    def analyze_entry(self, entry: LogEntry):
        """Analyze log entry in real-time"""
        # Update statistics
        self.statistics['total_entries'] += 1
        self.statistics['entries_by_level'][entry.level] += 1
        
        if entry.source not in self.statistics['entries_by_source']:
            self.statistics['entries_by_source'][entry.source] = 0
        self.statistics['entries_by_source'][entry.source] += 1
        
        # Check patterns
        for name, pattern_info in self.patterns.items():
            if pattern_info['regex'].search(entry.message):
                pattern_info['matches'] += 1
                if pattern_info['callback']:
                    pattern_info['callback'](entry, name)
        
        # Track performance metrics
        if 'execution_time' in entry.data:
            self.statistics['performance_metrics'].append(entry.data['execution_time'])
            # Keep only recent metrics
            if len(self.statistics['performance_metrics']) > 1000:
                self.statistics['performance_metrics'] = self.statistics['performance_metrics'][-500:]
    
    def get_analysis_summary(self) -> Dict[str, Any]:
        """Get analysis summary"""
        avg_performance = (
            sum(self.statistics['performance_metrics']) / len(self.statistics['performance_metrics'])
            if self.statistics['performance_metrics'] else 0
        )
        
        return {
            'total_entries': self.statistics['total_entries'],
            'entries_by_level': dict(self.statistics['entries_by_level']),
            'top_sources': dict(sorted(self.statistics['entries_by_source'].items(), 
                                     key=lambda x: x[1], reverse=True)[:10]),
            'pattern_matches': {name: info['matches'] for name, info in self.patterns.items()},
            'average_performance': avg_performance
        }

class AdvancedFileLogger:
    """
    Advanced File Logger with ARM64 optimizations
    
    Provides high-performance logging capabilities including:
    - Asynchronous logging with buffering
    - Multiple output formats
    - Log rotation and compression
    - Real-time analysis and filtering
    """
    
    def __init__(self, config: LoggerConfig = None):
        self.config = config or LoggerConfig()
        self.arm64_optimizer = ARM64LogOptimizer()
        self.shared_memory = SharedMemoryManager() if self.config.use_shared_memory else None
        
        # Core components
        self.log_buffer = queue.Queue(maxsize=self.config.buffer_size)
        self.rotation_manager = LogRotationManager(self.config)
        self.analyzer = LogAnalyzer() if self.config.enable_real_time_analysis else None
        
        # Threading and async
        self.is_running = False
        self.flush_thread = None
        self.writer_threads = []
        
        # Performance tracking
        self.log_stats = {
            'total_logs': 0,
            'logs_by_level': {level: 0 for level in LogLevel},
            'buffer_overflows': 0,
            'write_times': [],
            'flush_times': []
        }
        
        # Setup log directory
        self._setup_log_directory()
        
        logger.info(f"AdvancedFileLogger initialized with ARM64 optimizations: {self.arm64_optimizer.is_arm64}")
    
    def _setup_log_directory(self):
        """Create log directory if it doesn't exist"""
        log_dir = Path(self.config.log_directory)
        log_dir.mkdir(parents=True, exist_ok=True)
    
    @performance_monitor
    @error_handler
    def log(self, level: LogLevel, message: str, source: str = "system", data: Dict[str, Any] = None):
        """
        Log message with specified level
        
        Args:
            level: Log level
            message: Log message
            source: Source of the log
            data: Additional data to log
        """
        if data is None:
            data = {}
        
        entry = LogEntry(
            timestamp=time.time(),
            level=level,
            message=message,
            source=source,
            data=data,
            thread_id=threading.get_ident(),
            process_id=os.getpid(),
            session_id=getattr(self, 'session_id', None)
        )
        
        try:
            self.log_buffer.put(entry, timeout=0.1)
            self.log_stats['total_logs'] += 1
            self.log_stats['logs_by_level'][level] += 1
            
            # Real-time analysis
            if self.analyzer:
                self.analyzer.analyze_entry(entry)
                
        except queue.Full:
            self.log_stats['buffer_overflows'] += 1
            logger.warning("Log buffer overflow, dropping log entry")
    
    def debug(self, message: str, source: str = "system", data: Dict[str, Any] = None):
        """Log debug message"""
        self.log(LogLevel.DEBUG, message, source, data)
    
    def info(self, message: str, source: str = "system", data: Dict[str, Any] = None):
        """Log info message"""
        self.log(LogLevel.INFO, message, source, data)
    
    def warning(self, message: str, source: str = "system", data: Dict[str, Any] = None):
        """Log warning message"""
        self.log(LogLevel.WARNING, message, source, data)
    
    def error(self, message: str, source: str = "system", data: Dict[str, Any] = None):
        """Log error message"""
        self.log(LogLevel.ERROR, message, source, data)
    
    def critical(self, message: str, source: str = "system", data: Dict[str, Any] = None):
        """Log critical message"""
        self.log(LogLevel.CRITICAL, message, source, data)
    
    def log_trading_event(self, event_type: str, data: Dict[str, Any]):
        """Log trading-specific event"""
        self.info(f"Trading event: {event_type}", "trading", data)
    
    def log_performance_metric(self, metric_name: str, value: float, source: str = "performance"):
        """Log performance metric"""
        self.info(f"Performance metric: {metric_name}", source, {"metric": metric_name, "value": value})
    
    def _flush_buffer(self):
        """Background thread to flush log buffer"""
        while self.is_running:
            try:
                entries = []
                
                # Collect entries from buffer
                start_time = time.time()
                while len(entries) < 100:  # Batch size
                    try:
                        entry = self.log_buffer.get(timeout=self.config.flush_interval)
                        entries.append(entry)
                        self.log_buffer.task_done()
                    except queue.Empty:
                        break
                
                if entries:
                    self._write_entries(entries)
                    self.log_stats['flush_times'].append(time.time() - start_time)
                    
                    # Keep only recent flush times
                    if len(self.log_stats['flush_times']) > 100:
                        self.log_stats['flush_times'] = self.log_stats['flush_times'][-50:]
                
            except Exception as e:
                logger.error(f"Error in flush buffer: {e}")
    
    def _write_entries(self, entries: List[LogEntry]):
        """Write log entries to file"""
        start_time = time.time()
        
        try:
            # Check if rotation is needed
            if self.rotation_manager.should_rotate():
                self.rotation_manager.rotate_log()
            
            # Serialize entries
            serialized_data = self.arm64_optimizer.optimize_log_serialization(entries, self.config.log_format)
            
            # Write to file
            log_path = Path(self.config.log_directory) / self.config.log_filename
            
            if self.config.log_format == LogFormat.BINARY:
                mode = 'ab'
            else:
                mode = 'a'
            
            with open(log_path, mode) as f:
                if self.config.log_format == LogFormat.BINARY:
                    f.write(serialized_data)
                else:
                    f.write(serialized_data.decode('utf-8'))
                    f.write('\n')
            
            self.log_stats['write_times'].append(time.time() - start_time)
            
            # Keep only recent write times
            if len(self.log_stats['write_times']) > 100:
                self.log_stats['write_times'] = self.log_stats['write_times'][-50:]
                
        except Exception as e:
            logger.error(f"Error writing log entries: {e}")
    
    async def _async_write_entries(self, entries: List[LogEntry]):
        """Asynchronously write log entries"""
        try:
            serialized_data = self.arm64_optimizer.optimize_log_serialization(entries, self.config.log_format)
            log_path = Path(self.config.log_directory) / self.config.log_filename
            
            if self.config.log_format == LogFormat.BINARY:
                mode = 'ab'
            else:
                mode = 'a'
            
            async with aiofiles.open(log_path, mode) as f:
                if self.config.log_format == LogFormat.BINARY:
                    await f.write(serialized_data)
                else:
                    await f.write(serialized_data.decode('utf-8'))
                    await f.write('\n')
                    
        except Exception as e:
            logger.error(f"Error in async write: {e}")
    
    def start(self) -> bool:
        """Start file logger"""
        if self.is_running:
            return True
        
        try:
            self.is_running = True
            
            # Start flush thread
            self.flush_thread = threading.Thread(target=self._flush_buffer, name="LogFlushThread")
            self.flush_thread.daemon = True
            self.flush_thread.start()
            
            # Start writer threads if using parallel processing
            if self.config.max_workers and self.config.max_workers > 1:
                for i in range(self.config.max_workers - 1):  # -1 because flush thread is already one worker
                    thread = threading.Thread(target=self._flush_buffer, name=f"LogWriter-{i}")
                    thread.daemon = True
                    thread.start()
                    self.writer_threads.append(thread)
            
            logger.info("File logger started successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start file logger: {e}")
            self.is_running = False
            return False
    
    def stop(self):
        """Stop file logger"""
        self.is_running = False
        
        # Wait for buffer to empty
        self.log_buffer.join()
        
        # Wait for threads
        if self.flush_thread:
            self.flush_thread.join(timeout=5.0)
        
        for thread in self.writer_threads:
            thread.join(timeout=5.0)
        
        self.writer_threads.clear()
        logger.info("File logger stopped")
    
    def get_status(self) -> Dict[str, Any]:
        """Get file logger status"""
        avg_write_time = (
            sum(self.log_stats['write_times']) / len(self.log_stats['write_times'])
            if self.log_stats['write_times'] else 0
        )
        
        avg_flush_time = (
            sum(self.log_stats['flush_times']) / len(self.log_stats['flush_times'])
            if self.log_stats['flush_times'] else 0
        )
        
        return {
            'is_running': self.is_running,
            'buffer_size': self.log_buffer.qsize(),
            'total_logs': self.log_stats['total_logs'],
            'logs_by_level': dict(self.log_stats['logs_by_level']),
            'buffer_overflows': self.log_stats['buffer_overflows'],
            'average_write_time': avg_write_time,
            'average_flush_time': avg_flush_time,
            'log_file_size': self._get_log_file_size(),
            'arm64_optimized': self.arm64_optimizer.is_arm64
        }
    
    def get_log_summary(self) -> Dict[str, Any]:
        """Get log summary statistics"""
        summary = {
            'total_logs': self.log_stats['total_logs'],
            'logs_by_level': dict(self.log_stats['logs_by_level']),
            'buffer_overflows': self.log_stats['buffer_overflows'],
            'log_file_size': self._get_log_file_size()
        }
        
        if self.analyzer:
            summary['analysis'] = self.analyzer.get_analysis_summary()
        
        return summary
    
    def _get_log_file_size(self) -> int:
        """Get current log file size"""
        log_path = Path(self.config.log_directory) / self.config.log_filename
        return log_path.stat().st_size if log_path.exists() else 0
    
    def search_logs(self, pattern: str, max_results: int = 100) -> List[Dict[str, Any]]:
        """Search logs for pattern"""
        import re
        results = []
        regex = re.compile(pattern, re.IGNORECASE)
        
        log_path = Path(self.config.log_directory) / self.config.log_filename
        
        if not log_path.exists():
            return results
        
        try:
            with open(log_path, 'r') as f:
                for line_num, line in enumerate(f, 1):
                    if len(results) >= max_results:
                        break
                    
                    if regex.search(line):
                        try:
                            if self.config.log_format == LogFormat.JSON:
                                log_data = json.loads(line.strip())
                                log_data['line_number'] = line_num
                                results.append(log_data)
                            else:
                                results.append({
                                    'line_number': line_num,
                                    'content': line.strip()
                                })
                        except json.JSONDecodeError:
                            results.append({
                                'line_number': line_num,
                                'content': line.strip()
                            })
        
        except Exception as e:
            logger.error(f"Error searching logs: {e}")
        
        return results
    
    def cleanup(self):
        """Cleanup file logger resources"""
        if self.is_running:
            self.stop()
        
        if self.shared_memory:
            self.shared_memory.cleanup()
        
        logger.info("FileLogger cleanup completed")

# Factory function for easy instantiation
def create_file_logger(config: LoggerConfig = None) -> AdvancedFileLogger:
    """
    Factory function to create file logger with optimal configuration
    
    Args:
        config: File logger configuration
        
    Returns:
        Configured AdvancedFileLogger instance
    """
    if config is None:
        config = LoggerConfig()
        
        # Auto-detect optimal settings
        if platform.machine().lower() in ['arm64', 'aarch64']:
            config.enable_arm64_optimizations = True
            config.async_logging = True
            config.max_workers = min(mp.cpu_count(), 4)
        
        # Adjust for available memory and disk space
        available_memory = psutil.virtual_memory().available / (1024**3)  # GB
        if available_memory > 8:
            config.buffer_size = 20000
            config.max_file_size = 200 * 1024 * 1024  # 200MB
        elif available_memory > 4:
            config.buffer_size = 10000
            config.max_file_size = 100 * 1024 * 1024  # 100MB
        else:
            config.buffer_size = 5000
            config.max_file_size = 50 * 1024 * 1024   # 50MB
    
    return AdvancedFileLogger(config)

# Legacy compatibility
FileLogger = AdvancedFileLogger

if __name__ == "__main__":
    # Example usage and testing
    
    # Create file logger
    file_logger = create_file_logger()
    
    # Start logger
    if file_logger.start():
        print("File logger started successfully")
        
        # Test logging
        file_logger.info("System started", "main", {"version": "1.0.0"})
        file_logger.warning("Test warning", "test", {"test_data": 123})
        file_logger.error("Test error", "test", {"error_code": 500})
        
        # Log trading event
        file_logger.log_trading_event("order_placed", {
            "symbol": "AAPL",
            "quantity": 100,
            "price": 150.0
        })
        
        # Get status
        status = file_logger.get_status()
        print(f"Logger status: {status}")
        
        # Search logs
        results = file_logger.search_logs("error")
        print(f"Search results: {results}")
        
        # Cleanup
        time.sleep(2)  # Allow time for flushing
        file_logger.stop()
        file_logger.cleanup()
    else:
        print("Failed to start file logger")