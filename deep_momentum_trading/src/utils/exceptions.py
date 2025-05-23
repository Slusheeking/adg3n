"""
Enhanced exception classes for Deep Momentum Trading System.

This module provides comprehensive exception handling with detailed error information,
context tracking, and ARM64-specific error handling capabilities.
"""

import traceback
import time
import threading
from typing import Any, Dict, Optional, List, Union
from dataclasses import dataclass, field
from enum import Enum
import json

class ErrorSeverity(Enum):
    """Error severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class ErrorCategory(Enum):
    """Error categories for classification."""
    SYSTEM = "system"
    TRADING = "trading"
    MODEL = "model"
    DATA = "data"
    NETWORK = "network"
    VALIDATION = "validation"
    CONFIGURATION = "configuration"
    PERFORMANCE = "performance"
    SECURITY = "security"
    ARM64 = "arm64"

@dataclass
class ErrorContext:
    """Enhanced error context with detailed information."""
    timestamp: float = field(default_factory=time.time)
    thread_id: int = field(default_factory=lambda: threading.get_ident())
    function_name: Optional[str] = None
    module_name: Optional[str] = None
    line_number: Optional[int] = None
    file_path: Optional[str] = None
    local_variables: Optional[Dict[str, Any]] = None
    stack_trace: Optional[str] = None
    system_info: Optional[Dict[str, Any]] = None
    performance_metrics: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "timestamp": self.timestamp,
            "thread_id": self.thread_id,
            "function_name": self.function_name,
            "module_name": self.module_name,
            "line_number": self.line_number,
            "file_path": self.file_path,
            "local_variables": self.local_variables,
            "stack_trace": self.stack_trace,
            "system_info": self.system_info,
            "performance_metrics": self.performance_metrics
        }

class TradingSystemError(Exception):
    """Base exception class for all trading system errors."""
    
    def __init__(self, 
                 message: str,
                 error_code: Optional[str] = None,
                 severity: ErrorSeverity = ErrorSeverity.MEDIUM,
                 category: ErrorCategory = ErrorCategory.SYSTEM,
                 context: Optional[ErrorContext] = None,
                 cause: Optional[Exception] = None,
                 recoverable: bool = True,
                 retry_after: Optional[float] = None,
                 metadata: Optional[Dict[str, Any]] = None):
        """
        Initialize trading system error.
        
        Args:
            message: Error message
            error_code: Unique error code
            severity: Error severity level
            category: Error category
            context: Error context information
            cause: Original exception that caused this error
            recoverable: Whether the error is recoverable
            retry_after: Suggested retry delay in seconds
            metadata: Additional error metadata
        """
        super().__init__(message)
        
        self.message = message
        self.error_code = error_code or self._generate_error_code()
        self.severity = severity
        self.category = category
        self.context = context or self._create_context()
        self.cause = cause
        self.recoverable = recoverable
        self.retry_after = retry_after
        self.metadata = metadata or {}
        
        # Chain the original exception
        if cause:
            self.__cause__ = cause
    
    def _generate_error_code(self) -> str:
        """Generate unique error code."""
        import hashlib
        import uuid
        
        # Create unique code based on class name and timestamp
        data = f"{self.__class__.__name__}_{time.time()}_{uuid.uuid4()}"
        return hashlib.md5(data.encode()).hexdigest()[:8].upper()
    
    def _create_context(self) -> ErrorContext:
        """Create error context from current execution state."""
        import inspect
        import sys
        
        context = ErrorContext()
        
        try:
            # Get current frame information
            frame = inspect.currentframe()
            if frame and frame.f_back and frame.f_back.f_back:
                caller_frame = frame.f_back.f_back
                
                context.function_name = caller_frame.f_code.co_name
                context.module_name = caller_frame.f_globals.get('__name__')
                context.line_number = caller_frame.f_lineno
                context.file_path = caller_frame.f_code.co_filename
                
                # Capture local variables (safely)
                try:
                    local_vars = {}
                    for key, value in caller_frame.f_locals.items():
                        try:
                            # Only capture serializable values
                            json.dumps(value)
                            local_vars[key] = value
                        except (TypeError, ValueError):
                            local_vars[key] = str(type(value))
                    
                    context.local_variables = local_vars
                except:
                    pass
            
            # Capture stack trace
            context.stack_trace = traceback.format_exc()
            
            # System information
            try:
                import platform
                import psutil
                
                context.system_info = {
                    "platform": platform.platform(),
                    "python_version": platform.python_version(),
                    "cpu_count": psutil.cpu_count(),
                    "memory_usage": psutil.virtual_memory().percent,
                    "cpu_usage": psutil.cpu_percent()
                }
            except:
                pass
                
        except Exception:
            # Don't let context creation fail the original error
            pass
        
        return context
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for logging/serialization."""
        return {
            "error_code": self.error_code,
            "message": self.message,
            "severity": self.severity.value,
            "category": self.category.value,
            "recoverable": self.recoverable,
            "retry_after": self.retry_after,
            "metadata": self.metadata,
            "context": self.context.to_dict() if self.context else None,
            "cause": str(self.cause) if self.cause else None,
            "exception_type": self.__class__.__name__
        }
    
    def __str__(self) -> str:
        """String representation with error code."""
        return f"[{self.error_code}] {self.message}"
    
    def __repr__(self) -> str:
        """Detailed representation."""
        return (f"{self.__class__.__name__}(error_code='{self.error_code}', "
                f"message='{self.message}', severity={self.severity.value}, "
                f"category={self.category.value})")

class TradingError(TradingSystemError):
    """Trading-specific errors."""
    
    def __init__(self, message: str, **kwargs):
        kwargs.setdefault('category', ErrorCategory.TRADING)
        super().__init__(message, **kwargs)

class OrderError(TradingError):
    """Order execution errors."""
    
    def __init__(self, 
                 message: str,
                 order_id: Optional[str] = None,
                 symbol: Optional[str] = None,
                 order_type: Optional[str] = None,
                 **kwargs):
        super().__init__(message, **kwargs)
        self.metadata.update({
            "order_id": order_id,
            "symbol": symbol,
            "order_type": order_type
        })

class PositionError(TradingError):
    """Position management errors."""
    
    def __init__(self,
                 message: str,
                 symbol: Optional[str] = None,
                 position_size: Optional[float] = None,
                 **kwargs):
        super().__init__(message, **kwargs)
        self.metadata.update({
            "symbol": symbol,
            "position_size": position_size
        })

class RiskManagementError(TradingError):
    """Risk management errors."""
    
    def __init__(self,
                 message: str,
                 risk_type: Optional[str] = None,
                 current_value: Optional[float] = None,
                 limit_value: Optional[float] = None,
                 **kwargs):
        kwargs.setdefault('severity', ErrorSeverity.HIGH)
        super().__init__(message, **kwargs)
        self.metadata.update({
            "risk_type": risk_type,
            "current_value": current_value,
            "limit_value": limit_value
        })

class ModelError(TradingSystemError):
    """Machine learning model errors."""
    
    def __init__(self, message: str, **kwargs):
        kwargs.setdefault('category', ErrorCategory.MODEL)
        super().__init__(message, **kwargs)

class ModelTrainingError(ModelError):
    """Model training errors."""
    
    def __init__(self,
                 message: str,
                 model_name: Optional[str] = None,
                 epoch: Optional[int] = None,
                 loss: Optional[float] = None,
                 **kwargs):
        super().__init__(message, **kwargs)
        self.metadata.update({
            "model_name": model_name,
            "epoch": epoch,
            "loss": loss
        })

class ModelInferenceError(ModelError):
    """Model inference errors."""
    
    def __init__(self,
                 message: str,
                 model_name: Optional[str] = None,
                 input_shape: Optional[tuple] = None,
                 **kwargs):
        super().__init__(message, **kwargs)
        self.metadata.update({
            "model_name": model_name,
            "input_shape": input_shape
        })

class ModelValidationError(ModelError):
    """Model validation errors."""
    
    def __init__(self,
                 message: str,
                 validation_metric: Optional[str] = None,
                 expected_value: Optional[float] = None,
                 actual_value: Optional[float] = None,
                 **kwargs):
        super().__init__(message, **kwargs)
        self.metadata.update({
            "validation_metric": validation_metric,
            "expected_value": expected_value,
            "actual_value": actual_value
        })

class DataError(TradingSystemError):
    """Data-related errors."""
    
    def __init__(self, message: str, **kwargs):
        kwargs.setdefault('category', ErrorCategory.DATA)
        super().__init__(message, **kwargs)

class DataValidationError(DataError):
    """Data validation errors."""
    
    def __init__(self,
                 message: str,
                 field_name: Optional[str] = None,
                 expected_type: Optional[str] = None,
                 actual_type: Optional[str] = None,
                 **kwargs):
        super().__init__(message, **kwargs)
        self.metadata.update({
            "field_name": field_name,
            "expected_type": expected_type,
            "actual_type": actual_type
        })

class DataSourceError(DataError):
    """Data source connection/retrieval errors."""
    
    def __init__(self,
                 message: str,
                 source_name: Optional[str] = None,
                 endpoint: Optional[str] = None,
                 **kwargs):
        kwargs.setdefault('recoverable', True)
        kwargs.setdefault('retry_after', 5.0)
        super().__init__(message, **kwargs)
        self.metadata.update({
            "source_name": source_name,
            "endpoint": endpoint
        })

class DataIntegrityError(DataError):
    """Data integrity and consistency errors."""
    
    def __init__(self,
                 message: str,
                 table_name: Optional[str] = None,
                 record_count: Optional[int] = None,
                 **kwargs):
        kwargs.setdefault('severity', ErrorSeverity.HIGH)
        super().__init__(message, **kwargs)
        self.metadata.update({
            "table_name": table_name,
            "record_count": record_count
        })

class NetworkError(TradingSystemError):
    """Network-related errors."""
    
    def __init__(self, message: str, **kwargs):
        kwargs.setdefault('category', ErrorCategory.NETWORK)
        kwargs.setdefault('recoverable', True)
        kwargs.setdefault('retry_after', 1.0)
        super().__init__(message, **kwargs)

class ConnectionError(NetworkError):
    """Connection errors."""
    
    def __init__(self,
                 message: str,
                 host: Optional[str] = None,
                 port: Optional[int] = None,
                 protocol: Optional[str] = None,
                 **kwargs):
        super().__init__(message, **kwargs)
        self.metadata.update({
            "host": host,
            "port": port,
            "protocol": protocol
        })

class TimeoutError(NetworkError):
    """Timeout errors."""
    
    def __init__(self,
                 message: str,
                 timeout_seconds: Optional[float] = None,
                 operation: Optional[str] = None,
                 **kwargs):
        super().__init__(message, **kwargs)
        self.metadata.update({
            "timeout_seconds": timeout_seconds,
            "operation": operation
        })

class ValidationError(TradingSystemError):
    """Validation errors."""
    
    def __init__(self, message: str, **kwargs):
        kwargs.setdefault('category', ErrorCategory.VALIDATION)
        kwargs.setdefault('recoverable', False)
        super().__init__(message, **kwargs)

class ConfigurationError(TradingSystemError):
    """Configuration errors."""
    
    def __init__(self, message: str, **kwargs):
        kwargs.setdefault('category', ErrorCategory.CONFIGURATION)
        kwargs.setdefault('severity', ErrorSeverity.HIGH)
        kwargs.setdefault('recoverable', False)
        super().__init__(message, **kwargs)

class PerformanceError(TradingSystemError):
    """Performance-related errors."""
    
    def __init__(self,
                 message: str,
                 metric_name: Optional[str] = None,
                 current_value: Optional[float] = None,
                 threshold_value: Optional[float] = None,
                 **kwargs):
        kwargs.setdefault('category', ErrorCategory.PERFORMANCE)
        super().__init__(message, **kwargs)
        self.metadata.update({
            "metric_name": metric_name,
            "current_value": current_value,
            "threshold_value": threshold_value
        })

class SecurityError(TradingSystemError):
    """Security-related errors."""
    
    def __init__(self, message: str, **kwargs):
        kwargs.setdefault('category', ErrorCategory.SECURITY)
        kwargs.setdefault('severity', ErrorSeverity.CRITICAL)
        kwargs.setdefault('recoverable', False)
        super().__init__(message, **kwargs)

class AuthenticationError(SecurityError):
    """Authentication errors."""
    
    def __init__(self,
                 message: str,
                 username: Optional[str] = None,
                 **kwargs):
        super().__init__(message, **kwargs)
        self.metadata.update({
            "username": username
        })

class AuthorizationError(SecurityError):
    """Authorization errors."""
    
    def __init__(self,
                 message: str,
                 resource: Optional[str] = None,
                 required_permission: Optional[str] = None,
                 **kwargs):
        super().__init__(message, **kwargs)
        self.metadata.update({
            "resource": resource,
            "required_permission": required_permission
        })

class ARM64Error(TradingSystemError):
    """ARM64-specific errors."""
    
    def __init__(self, message: str, **kwargs):
        kwargs.setdefault('category', ErrorCategory.ARM64)
        super().__init__(message, **kwargs)

class ARM64OptimizationError(ARM64Error):
    """ARM64 optimization errors."""
    
    def __init__(self,
                 message: str,
                 optimization_type: Optional[str] = None,
                 **kwargs):
        super().__init__(message, **kwargs)
        self.metadata.update({
            "optimization_type": optimization_type
        })

class ARM64SIMDError(ARM64Error):
    """ARM64 SIMD operation errors."""
    
    def __init__(self,
                 message: str,
                 operation: Optional[str] = None,
                 vector_size: Optional[int] = None,
                 **kwargs):
        super().__init__(message, **kwargs)
        self.metadata.update({
            "operation": operation,
            "vector_size": vector_size
        })

# Exception handling utilities
class ExceptionHandler:
    """Centralized exception handling and logging."""
    
    def __init__(self):
        self.error_counts = {}
        self.error_history = []
        self.max_history = 1000
        self.lock = threading.Lock()
    
    def handle_exception(self, 
                        exception: Exception,
                        context: Optional[Dict[str, Any]] = None,
                        log_level: str = "error") -> None:
        """Handle and log exception with context."""
        from .logger import get_logger
        
        logger = get_logger(__name__)
        
        with self.lock:
            # Count errors by type
            error_type = type(exception).__name__
            self.error_counts[error_type] = self.error_counts.get(error_type, 0) + 1
            
            # Add to history
            error_record = {
                "timestamp": time.time(),
                "exception_type": error_type,
                "message": str(exception),
                "context": context
            }
            
            self.error_history.append(error_record)
            
            # Trim history if needed
            if len(self.error_history) > self.max_history:
                self.error_history = self.error_history[-self.max_history:]
        
        # Log the exception
        if isinstance(exception, TradingSystemError):
            error_dict = exception.to_dict()
            if context:
                error_dict["additional_context"] = context
            
            if log_level == "critical":
                logger.critical(f"Critical error: {error_dict}")
            elif log_level == "error":
                logger.error(f"Error: {error_dict}")
            elif log_level == "warning":
                logger.warning(f"Warning: {error_dict}")
        else:
            # Standard exception
            logger.error(f"Unhandled exception: {exception}", exc_info=True)
    
    def get_error_stats(self) -> Dict[str, Any]:
        """Get error statistics."""
        with self.lock:
            total_errors = sum(self.error_counts.values())
            
            return {
                "total_errors": total_errors,
                "error_counts": self.error_counts.copy(),
                "recent_errors": len([e for e in self.error_history 
                                    if time.time() - e["timestamp"] < 3600]),  # Last hour
                "history_size": len(self.error_history)
            }
    
    def clear_history(self) -> None:
        """Clear error history."""
        with self.lock:
            self.error_history.clear()
            self.error_counts.clear()

# Global exception handler instance
exception_handler = ExceptionHandler()

# Utility functions
def create_error_response(exception: Exception, 
                         include_traceback: bool = False) -> Dict[str, Any]:
    """Create standardized error response."""
    if isinstance(exception, TradingSystemError):
        response = exception.to_dict()
    else:
        response = {
            "error_code": "UNKNOWN",
            "message": str(exception),
            "exception_type": type(exception).__name__,
            "severity": ErrorSeverity.MEDIUM.value,
            "category": ErrorCategory.SYSTEM.value,
            "recoverable": True
        }
    
    if include_traceback:
        response["traceback"] = traceback.format_exc()
    
    return response

def is_recoverable_error(exception: Exception) -> bool:
    """Check if an error is recoverable."""
    if isinstance(exception, TradingSystemError):
        return exception.recoverable
    
    # Default recovery logic for standard exceptions
    recoverable_types = (
        ConnectionError,
        TimeoutError,
        OSError,
        IOError
    )
    
    return isinstance(exception, recoverable_types)

def get_retry_delay(exception: Exception) -> Optional[float]:
    """Get suggested retry delay for an exception."""
    if isinstance(exception, TradingSystemError):
        return exception.retry_after
    
    # Default retry delays for standard exceptions
    if isinstance(exception, (ConnectionError, TimeoutError)):
        return 1.0
    elif isinstance(exception, OSError):
        return 0.5
    
    return None

# Export all exceptions and utilities
__all__ = [
    # Enums
    "ErrorSeverity",
    "ErrorCategory",
    
    # Base classes
    "TradingSystemError",
    "ErrorContext",
    
    # Trading errors
    "TradingError",
    "OrderError", 
    "PositionError",
    "RiskManagementError",
    
    # Model errors
    "ModelError",
    "ModelTrainingError",
    "ModelInferenceError",
    "ModelValidationError",
    
    # Data errors
    "DataError",
    "DataValidationError",
    "DataSourceError",
    "DataIntegrityError",
    
    # Network errors
    "NetworkError",
    "ConnectionError",
    "TimeoutError",
    
    # System errors
    "ValidationError",
    "ConfigurationError",
    "PerformanceError",
    
    # Security errors
    "SecurityError",
    "AuthenticationError",
    "AuthorizationError",
    
    # ARM64 errors
    "ARM64Error",
    "ARM64OptimizationError",
    "ARM64SIMDError",
    
    # Utilities
    "ExceptionHandler",
    "exception_handler",
    "create_error_response",
    "is_recoverable_error",
    "get_retry_delay"
]