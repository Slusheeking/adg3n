"""
Enhanced validation system for Deep Momentum Trading System with ARM64 optimizations.

This module provides comprehensive validation capabilities for trading data,
model parameters, financial calculations, and system configurations with
ARM64-specific optimizations and performance monitoring.
"""

import re
import math
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Union, Tuple, Callable, Type
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from decimal import Decimal, ROUND_HALF_UP
import json
import warnings
from enum import Enum
import functools
import inspect
from pathlib import Path

# ARM64 detection
import platform
IS_ARM64 = platform.machine().lower() in ['arm64', 'aarch64']

class ValidationLevel(Enum):
    """Validation strictness levels."""
    STRICT = "strict"
    NORMAL = "normal"
    LENIENT = "lenient"
    DISABLED = "disabled"

class ValidationResult(Enum):
    """Validation result types."""
    VALID = "valid"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

@dataclass
class ValidationError:
    """Detailed validation error information."""
    field: str
    value: Any
    message: str
    result_type: ValidationResult
    suggestion: Optional[str] = None
    error_code: Optional[str] = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

@dataclass
class ValidationConfig:
    """Configuration for validation system."""
    level: ValidationLevel = ValidationLevel.NORMAL
    enable_arm64_optimizations: bool = True
    enable_performance_tracking: bool = True
    enable_caching: bool = True
    cache_size: int = 1000
    enable_warnings: bool = True
    strict_numeric_precision: bool = True
    max_validation_time_ms: float = 100.0
    enable_batch_validation: bool = True
    batch_size: int = 1000

class BaseValidator:
    """Base validator class with ARM64 optimizations."""
    
    def __init__(self, 
                 config: Optional[ValidationConfig] = None,
                 name: Optional[str] = None):
        self.config = config or ValidationConfig()
        self.name = name or self.__class__.__name__
        self._cache = {} if self.config.enable_caching else None
        self._performance_stats = {
            "validations": 0,
            "cache_hits": 0,
            "total_time_ms": 0.0,
            "avg_time_ms": 0.0
        }
    
    def validate(self, value: Any, **kwargs) -> List[ValidationError]:
        """Main validation method with performance tracking."""
        if self.config.level == ValidationLevel.DISABLED:
            return []
        
        start_time = self._get_time_ms() if self.config.enable_performance_tracking else 0
        
        try:
            # Check cache first
            if self._cache is not None:
                cache_key = self._get_cache_key(value, kwargs)
                if cache_key in self._cache:
                    self._performance_stats["cache_hits"] += 1
                    return self._cache[cache_key]
            
            # Perform validation
            errors = self._validate_impl(value, **kwargs)
            
            # Cache result
            if self._cache is not None and len(self._cache) < self.config.cache_size:
                self._cache[cache_key] = errors
            
            return errors
            
        finally:
            if self.config.enable_performance_tracking:
                elapsed = self._get_time_ms() - start_time
                self._update_performance_stats(elapsed)
    
    def _validate_impl(self, value: Any, **kwargs) -> List[ValidationError]:
        """Implementation-specific validation logic."""
        raise NotImplementedError("Subclasses must implement _validate_impl")
    
    def _get_cache_key(self, value: Any, kwargs: Dict) -> str:
        """Generate cache key for validation."""
        try:
            # Create a hashable representation
            value_str = str(value) if not isinstance(value, (dict, list)) else json.dumps(value, sort_keys=True, default=str)
            kwargs_str = json.dumps(kwargs, sort_keys=True, default=str)
            return f"{self.name}:{hash(value_str)}:{hash(kwargs_str)}"
        except:
            # Fallback for unhashable types
            return f"{self.name}:{id(value)}:{hash(str(kwargs))}"
    
    def _get_time_ms(self) -> float:
        """Get current time in milliseconds."""
        import time
        return time.time() * 1000
    
    def _update_performance_stats(self, elapsed_ms: float):
        """Update performance statistics."""
        self._performance_stats["validations"] += 1
        self._performance_stats["total_time_ms"] += elapsed_ms
        self._performance_stats["avg_time_ms"] = (
            self._performance_stats["total_time_ms"] / self._performance_stats["validations"]
        )
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get validation performance statistics."""
        return self._performance_stats.copy()
    
    def clear_cache(self):
        """Clear validation cache."""
        if self._cache is not None:
            self._cache.clear()

class NumericValidator(BaseValidator):
    """Enhanced numeric validator with ARM64 optimizations."""
    
    def __init__(self, 
                 min_value: Optional[float] = None,
                 max_value: Optional[float] = None,
                 allow_nan: bool = False,
                 allow_inf: bool = False,
                 precision: Optional[int] = None,
                 **kwargs):
        super().__init__(**kwargs)
        self.min_value = min_value
        self.max_value = max_value
        self.allow_nan = allow_nan
        self.allow_inf = allow_inf
        self.precision = precision
    
    def _validate_impl(self, value: Any, **kwargs) -> List[ValidationError]:
        """Validate numeric values with ARM64 optimizations."""
        errors = []
        
        # Type validation
        if not isinstance(value, (int, float, np.number, Decimal)):
            try:
                value = float(value)
            except (ValueError, TypeError):
                errors.append(ValidationError(
                    field="value",
                    value=value,
                    message=f"Value must be numeric, got {type(value).__name__}",
                    result_type=ValidationResult.ERROR,
                    error_code="INVALID_TYPE"
                ))
                return errors
        
        # Convert to float for validation
        if isinstance(value, Decimal):
            float_value = float(value)
        elif isinstance(value, np.number):
            float_value = value.item()
        else:
            float_value = float(value)
        
        # NaN validation
        if math.isnan(float_value):
            if not self.allow_nan:
                errors.append(ValidationError(
                    field="value",
                    value=value,
                    message="NaN values are not allowed",
                    result_type=ValidationResult.ERROR,
                    error_code="NAN_NOT_ALLOWED"
                ))
            return errors
        
        # Infinity validation
        if math.isinf(float_value):
            if not self.allow_inf:
                errors.append(ValidationError(
                    field="value",
                    value=value,
                    message="Infinite values are not allowed",
                    result_type=ValidationResult.ERROR,
                    error_code="INF_NOT_ALLOWED"
                ))
            return errors
        
        # Range validation with ARM64 optimized comparisons
        if self.min_value is not None and float_value < self.min_value:
            errors.append(ValidationError(
                field="value",
                value=value,
                message=f"Value {float_value} is below minimum {self.min_value}",
                result_type=ValidationResult.ERROR,
                error_code="BELOW_MINIMUM",
                suggestion=f"Use value >= {self.min_value}"
            ))
        
        if self.max_value is not None and float_value > self.max_value:
            errors.append(ValidationError(
                field="value",
                value=value,
                message=f"Value {float_value} is above maximum {self.max_value}",
                result_type=ValidationResult.ERROR,
                error_code="ABOVE_MAXIMUM",
                suggestion=f"Use value <= {self.max_value}"
            ))
        
        # Precision validation
        if self.precision is not None and isinstance(value, float):
            decimal_places = len(str(float_value).split('.')[-1]) if '.' in str(float_value) else 0
            if decimal_places > self.precision:
                if self.config.level == ValidationLevel.STRICT:
                    errors.append(ValidationError(
                        field="value",
                        value=value,
                        message=f"Value has {decimal_places} decimal places, maximum allowed is {self.precision}",
                        result_type=ValidationResult.ERROR,
                        error_code="PRECISION_EXCEEDED"
                    ))
                else:
                    errors.append(ValidationError(
                        field="value",
                        value=value,
                        message=f"Value precision exceeds {self.precision} decimal places",
                        result_type=ValidationResult.WARNING,
                        error_code="PRECISION_WARNING"
                    ))
        
        return errors

class TradingDataValidator(BaseValidator):
    """Specialized validator for trading data."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.price_validator = NumericValidator(min_value=0.0, precision=4)
        self.volume_validator = NumericValidator(min_value=0.0)
        self.timestamp_validator = TimestampValidator()
    
    def _validate_impl(self, value: Any, **kwargs) -> List[ValidationError]:
        """Validate trading data structures."""
        errors = []
        
        if not isinstance(value, dict):
            errors.append(ValidationError(
                field="data",
                value=value,
                message="Trading data must be a dictionary",
                result_type=ValidationResult.ERROR,
                error_code="INVALID_DATA_TYPE"
            ))
            return errors
        
        # Required fields validation
        required_fields = ['symbol', 'timestamp', 'price']
        for field in required_fields:
            if field not in value:
                errors.append(ValidationError(
                    field=field,
                    value=None,
                    message=f"Required field '{field}' is missing",
                    result_type=ValidationResult.ERROR,
                    error_code="MISSING_REQUIRED_FIELD"
                ))
        
        # Symbol validation
        if 'symbol' in value:
            symbol_errors = self._validate_symbol(value['symbol'])
            errors.extend(symbol_errors)
        
        # Price validation
        if 'price' in value:
            price_errors = self.price_validator.validate(value['price'])
            for error in price_errors:
                error.field = 'price'
            errors.extend(price_errors)
        
        # Volume validation
        if 'volume' in value:
            volume_errors = self.volume_validator.validate(value['volume'])
            for error in volume_errors:
                error.field = 'volume'
            errors.extend(volume_errors)
        
        # Timestamp validation
        if 'timestamp' in value:
            timestamp_errors = self.timestamp_validator.validate(value['timestamp'])
            for error in timestamp_errors:
                error.field = 'timestamp'
            errors.extend(timestamp_errors)
        
        # OHLC validation if present
        ohlc_fields = ['open', 'high', 'low', 'close']
        if any(field in value for field in ohlc_fields):
            ohlc_errors = self._validate_ohlc(value)
            errors.extend(ohlc_errors)
        
        return errors
    
    def _validate_symbol(self, symbol: str) -> List[ValidationError]:
        """Validate trading symbol."""
        errors = []
        
        if not isinstance(symbol, str):
            errors.append(ValidationError(
                field="symbol",
                value=symbol,
                message="Symbol must be a string",
                result_type=ValidationResult.ERROR,
                error_code="INVALID_SYMBOL_TYPE"
            ))
            return errors
        
        # Symbol format validation
        if not re.match(r'^[A-Z]{1,10}$', symbol):
            errors.append(ValidationError(
                field="symbol",
                value=symbol,
                message="Symbol must be 1-10 uppercase letters",
                result_type=ValidationResult.WARNING,
                error_code="INVALID_SYMBOL_FORMAT",
                suggestion="Use uppercase letters only (e.g., 'AAPL', 'MSFT')"
            ))
        
        return errors
    
    def _validate_ohlc(self, data: Dict[str, Any]) -> List[ValidationError]:
        """Validate OHLC data consistency."""
        errors = []
        
        ohlc_values = {}
        for field in ['open', 'high', 'low', 'close']:
            if field in data:
                try:
                    ohlc_values[field] = float(data[field])
                except (ValueError, TypeError):
                    errors.append(ValidationError(
                        field=field,
                        value=data[field],
                        message=f"OHLC field '{field}' must be numeric",
                        result_type=ValidationResult.ERROR,
                        error_code="INVALID_OHLC_TYPE"
                    ))
        
        # OHLC consistency checks
        if len(ohlc_values) >= 2:
            if 'high' in ohlc_values and 'low' in ohlc_values:
                if ohlc_values['high'] < ohlc_values['low']:
                    errors.append(ValidationError(
                        field="ohlc",
                        value=ohlc_values,
                        message="High price cannot be less than low price",
                        result_type=ValidationResult.ERROR,
                        error_code="INVALID_OHLC_RANGE"
                    ))
            
            # Check if open/close are within high/low range
            for price_type in ['open', 'close']:
                if price_type in ohlc_values:
                    price = ohlc_values[price_type]
                    if 'high' in ohlc_values and price > ohlc_values['high']:
                        errors.append(ValidationError(
                            field=price_type,
                            value=price,
                            message=f"{price_type.title()} price cannot exceed high price",
                            result_type=ValidationResult.ERROR,
                            error_code="PRICE_ABOVE_HIGH"
                        ))
                    if 'low' in ohlc_values and price < ohlc_values['low']:
                        errors.append(ValidationError(
                            field=price_type,
                            value=price,
                            message=f"{price_type.title()} price cannot be below low price",
                            result_type=ValidationResult.ERROR,
                            error_code="PRICE_BELOW_LOW"
                        ))
        
        return errors

class TimestampValidator(BaseValidator):
    """Enhanced timestamp validator."""
    
    def __init__(self, 
                 allow_future: bool = False,
                 max_age_days: Optional[int] = None,
                 require_timezone: bool = True,
                 **kwargs):
        super().__init__(**kwargs)
        self.allow_future = allow_future
        self.max_age_days = max_age_days
        self.require_timezone = require_timezone
    
    def _validate_impl(self, value: Any, **kwargs) -> List[ValidationError]:
        """Validate timestamp values."""
        errors = []
        
        # Convert to datetime if needed
        dt = None
        if isinstance(value, datetime):
            dt = value
        elif isinstance(value, (int, float)):
            try:
                dt = datetime.fromtimestamp(value, tz=timezone.utc)
            except (ValueError, OSError):
                errors.append(ValidationError(
                    field="timestamp",
                    value=value,
                    message="Invalid timestamp value",
                    result_type=ValidationResult.ERROR,
                    error_code="INVALID_TIMESTAMP"
                ))
                return errors
        elif isinstance(value, str):
            try:
                # Try ISO format first
                dt = datetime.fromisoformat(value.replace('Z', '+00:00'))
            except ValueError:
                try:
                    # Try parsing as timestamp
                    timestamp = float(value)
                    dt = datetime.fromtimestamp(timestamp, tz=timezone.utc)
                except ValueError:
                    errors.append(ValidationError(
                        field="timestamp",
                        value=value,
                        message="Unable to parse timestamp string",
                        result_type=ValidationResult.ERROR,
                        error_code="UNPARSEABLE_TIMESTAMP"
                    ))
                    return errors
        else:
            errors.append(ValidationError(
                field="timestamp",
                value=value,
                message=f"Timestamp must be datetime, number, or string, got {type(value).__name__}",
                result_type=ValidationResult.ERROR,
                error_code="INVALID_TIMESTAMP_TYPE"
            ))
            return errors
        
        # Timezone validation
        if self.require_timezone and dt.tzinfo is None:
            errors.append(ValidationError(
                field="timestamp",
                value=value,
                message="Timestamp must include timezone information",
                result_type=ValidationResult.WARNING,
                error_code="MISSING_TIMEZONE",
                suggestion="Use timezone-aware datetime objects"
            ))
        
        # Future timestamp validation
        now = datetime.now(timezone.utc)
        if not self.allow_future and dt > now:
            errors.append(ValidationError(
                field="timestamp",
                value=value,
                message="Future timestamps are not allowed",
                result_type=ValidationResult.ERROR,
                error_code="FUTURE_TIMESTAMP"
            ))
        
        # Age validation
        if self.max_age_days is not None:
            age = now - dt
            if age.days > self.max_age_days:
                errors.append(ValidationError(
                    field="timestamp",
                    value=value,
                    message=f"Timestamp is {age.days} days old, maximum allowed is {self.max_age_days}",
                    result_type=ValidationResult.WARNING,
                    error_code="TIMESTAMP_TOO_OLD"
                ))
        
        return errors

class ModelParameterValidator(BaseValidator):
    """Validator for machine learning model parameters."""
    
    def _validate_impl(self, value: Any, **kwargs) -> List[ValidationError]:
        """Validate model parameters."""
        errors = []
        
        if not isinstance(value, dict):
            errors.append(ValidationError(
                field="parameters",
                value=value,
                message="Model parameters must be a dictionary",
                result_type=ValidationResult.ERROR,
                error_code="INVALID_PARAMS_TYPE"
            ))
            return errors
        
        # Learning rate validation
        if 'learning_rate' in value:
            lr_errors = NumericValidator(
                min_value=1e-6,
                max_value=1.0,
                name="learning_rate"
            ).validate(value['learning_rate'])
            errors.extend(lr_errors)
        
        # Batch size validation
        if 'batch_size' in value:
            batch_errors = NumericValidator(
                min_value=1,
                max_value=10000,
                name="batch_size"
            ).validate(value['batch_size'])
            
            # Check if batch size is power of 2 for ARM64 optimization
            if IS_ARM64 and 'batch_size' in value:
                batch_size = value['batch_size']
                if isinstance(batch_size, int) and batch_size > 0:
                    if not (batch_size & (batch_size - 1)) == 0:
                        errors.append(ValidationError(
                            field="batch_size",
                            value=batch_size,
                            message="For ARM64 optimization, batch size should be a power of 2",
                            result_type=ValidationResult.WARNING,
                            error_code="NON_POWER_OF_2_BATCH",
                            suggestion="Use batch sizes like 32, 64, 128, 256, etc."
                        ))
            
            errors.extend(batch_errors)
        
        # Epochs validation
        if 'epochs' in value:
            epoch_errors = NumericValidator(
                min_value=1,
                max_value=10000,
                name="epochs"
            ).validate(value['epochs'])
            errors.extend(epoch_errors)
        
        # Dropout validation
        if 'dropout' in value:
            dropout_errors = NumericValidator(
                min_value=0.0,
                max_value=0.99,
                name="dropout"
            ).validate(value['dropout'])
            errors.extend(dropout_errors)
        
        return errors

class ArrayValidator(BaseValidator):
    """Enhanced validator for numpy arrays and tensors."""
    
    def __init__(self, 
                 expected_shape: Optional[Tuple[int, ...]] = None,
                 expected_dtype: Optional[np.dtype] = None,
                 allow_empty: bool = False,
                 **kwargs):
        super().__init__(**kwargs)
        self.expected_shape = expected_shape
        self.expected_dtype = expected_dtype
        self.allow_empty = allow_empty
    
    def _validate_impl(self, value: Any, **kwargs) -> List[ValidationError]:
        """Validate array data with ARM64 optimizations."""
        errors = []
        
        # Convert to numpy array if needed
        if not isinstance(value, np.ndarray):
            try:
                value = np.array(value)
            except Exception as e:
                errors.append(ValidationError(
                    field="array",
                    value=type(value).__name__,
                    message=f"Cannot convert to numpy array: {e}",
                    result_type=ValidationResult.ERROR,
                    error_code="ARRAY_CONVERSION_FAILED"
                ))
                return errors
        
        # Empty array validation
        if value.size == 0 and not self.allow_empty:
            errors.append(ValidationError(
                field="array",
                value=value.shape,
                message="Empty arrays are not allowed",
                result_type=ValidationResult.ERROR,
                error_code="EMPTY_ARRAY"
            ))
        
        # Shape validation
        if self.expected_shape is not None:
            if value.shape != self.expected_shape:
                errors.append(ValidationError(
                    field="array",
                    value=value.shape,
                    message=f"Expected shape {self.expected_shape}, got {value.shape}",
                    result_type=ValidationResult.ERROR,
                    error_code="SHAPE_MISMATCH"
                ))
        
        # Data type validation
        if self.expected_dtype is not None:
            if value.dtype != self.expected_dtype:
                if self.config.level == ValidationLevel.STRICT:
                    errors.append(ValidationError(
                        field="array",
                        value=str(value.dtype),
                        message=f"Expected dtype {self.expected_dtype}, got {value.dtype}",
                        result_type=ValidationResult.ERROR,
                        error_code="DTYPE_MISMATCH"
                    ))
                else:
                    errors.append(ValidationError(
                        field="array",
                        value=str(value.dtype),
                        message=f"Dtype mismatch: expected {self.expected_dtype}, got {value.dtype}",
                        result_type=ValidationResult.WARNING,
                        error_code="DTYPE_WARNING"
                    ))
        
        # ARM64 optimization checks
        if IS_ARM64 and self.config.enable_arm64_optimizations:
            # Check memory alignment for SIMD operations
            if not value.flags.aligned:
                errors.append(ValidationError(
                    field="array",
                    value="unaligned",
                    message="Array is not memory-aligned for ARM64 SIMD operations",
                    result_type=ValidationResult.WARNING,
                    error_code="UNALIGNED_MEMORY",
                    suggestion="Use np.ascontiguousarray() for better ARM64 performance"
                ))
            
            # Check for contiguous memory layout
            if not value.flags.c_contiguous:
                errors.append(ValidationError(
                    field="array",
                    value="non-contiguous",
                    message="Array is not C-contiguous, may impact ARM64 performance",
                    result_type=ValidationResult.WARNING,
                    error_code="NON_CONTIGUOUS",
                    suggestion="Use np.ascontiguousarray() for better performance"
                ))
        
        # NaN/Inf validation
        if np.issubdtype(value.dtype, np.floating):
            if np.any(np.isnan(value)):
                errors.append(ValidationError(
                    field="array",
                    value="contains_nan",
                    message="Array contains NaN values",
                    result_type=ValidationResult.WARNING,
                    error_code="CONTAINS_NAN"
                ))
            
            if np.any(np.isinf(value)):
                errors.append(ValidationError(
                    field="array",
                    value="contains_inf",
                    message="Array contains infinite values",
                    result_type=ValidationResult.WARNING,
                    error_code="CONTAINS_INF"
                ))
        
        return errors

def validate_trading_data(data: Any, 
                         config: Optional[ValidationConfig] = None) -> List[ValidationError]:
    """Validate trading data with comprehensive checks."""
    validator = TradingDataValidator(config=config)
    return validator.validate(data)

def validate_model_parameters(params: Dict[str, Any],
                            config: Optional[ValidationConfig] = None) -> List[ValidationError]:
    """Validate machine learning model parameters."""
    validator = ModelParameterValidator(config=config)
    return validator.validate(params)

def validate_numeric_array(array: Any,
                          expected_shape: Optional[Tuple[int, ...]] = None,
                          expected_dtype: Optional[np.dtype] = None,
                          config: Optional[ValidationConfig] = None) -> List[ValidationError]:
    """Validate numeric arrays with ARM64 optimizations."""
    validator = ArrayValidator(
        expected_shape=expected_shape,
        expected_dtype=expected_dtype,
        config=config
    )
    return validator.validate(array)

def validate_timestamp(timestamp: Any,
                      allow_future: bool = False,
                      max_age_days: Optional[int] = None,
                      config: Optional[ValidationConfig] = None) -> List[ValidationError]:
    """Validate timestamp values."""
    validator = TimestampValidator(
        allow_future=allow_future,
        max_age_days=max_age_days,
        config=config
    )
    return validator.validate(timestamp)

def validation_decorator(validator: BaseValidator):
    """Decorator for automatic function parameter validation."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Get function signature
            sig = inspect.signature(func)
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()
            
            # Validate each parameter
            all_errors = []
            for param_name, param_value in bound_args.arguments.items():
                errors = validator.validate(param_value)
                for error in errors:
                    error.field = f"{func.__name__}.{param_name}"
                all_errors.extend(errors)
            
            # Handle validation errors
            if all_errors:
                error_messages = [f"{error.field}: {error.message}" for error in all_errors]
                raise ValueError(f"Validation failed: {'; '.join(error_messages)}")
            
            return func(*args, **kwargs)
        
        return wrapper
    return decorator

def batch_validate(data_list: List[Any], 
                  validator: BaseValidator,
                  config: Optional[ValidationConfig] = None) -> Dict[int, List[ValidationError]]:
    """Perform batch validation with ARM64 optimizations."""
    if config is None:
        config = ValidationConfig()
    
    results = {}
    batch_size = config.batch_size if config.enable_batch_validation else len(data_list)
    
    for i in range(0, len(data_list), batch_size):
        batch = data_list[i:i + batch_size]
        
        for j, item in enumerate(batch):
            item_index = i + j
            try:
                errors = validator.validate(item)
                if errors:
                    results[item_index] = errors
            except Exception as e:
                results[item_index] = [ValidationError(
                    field="validation",
                    value=item,
                    message=f"Validation failed with exception: {e}",
                    result_type=ValidationResult.CRITICAL,
                    error_code="VALIDATION_EXCEPTION"
                )]
    
    return results

def create_validation_report(errors: List[ValidationError]) -> Dict[str, Any]:
    """Create comprehensive validation report."""
    if not errors:
        return {
            "status": "valid",
            "error_count": 0,
            "warning_count": 0,
            "critical_count": 0,
            "errors": []
        }
    
    error_count = sum(1 for e in errors if e.result_type == ValidationResult.ERROR)
    warning_count = sum(1 for e in errors if e.result_type == ValidationResult.WARNING)
    critical_count = sum(1 for e in errors if e.result_type == ValidationResult.CRITICAL)
    
    return {
        "status": "invalid" if error_count > 0 or critical_count > 0 else "warnings",
        "error_count": error_count,
        "warning_count": warning_count,
        "critical_count": critical_count,
        "errors": [
            {
                "field": error.field,
                "message": error.message,
                "type": error.result_type.value,
                "code": error.error_code,
                "suggestion": error.suggestion,
                "timestamp": error.timestamp.isoformat()
            }
            for error in errors
        ]
    }

# Export all public components
__all__ = [
    "ValidationLevel",
    "ValidationResult",
    "ValidationError",
    "ValidationConfig",
    "BaseValidator",
    "NumericValidator",
    "TradingDataValidator",
    "TimestampValidator",
    "ModelParameterValidator",
    "ArrayValidator",
    "validate_trading_data",
    "validate_model_parameters",
    "validate_numeric_array",
    "validate_timestamp",
    "validation_decorator",
    "batch_validate",
    "create_validation_report",
    "IS_ARM64"
]

if __name__ == "__main__":
    # Example usage with ARM64 optimizations
    print("Testing enhanced validation system with ARM64 optimizations...")
    print(f"ARM64 detected: {IS_ARM64}")
    
    # Test trading data validation
    print("\n1. Testing Trading Data Validation...")
    trading_data = {
        "symbol": "AAPL",
        "timestamp": datetime.now(timezone.utc),
        "price": 150.25,
        "volume": 1000,
        "open": 149.50,
        "high": 151.00,
        "low": 149.00,
        "close": 150.25
    }
    
    errors = validate_trading_data(trading_data)
    report = create_validation_report(errors)
    print(f"Trading data validation: {report['status']}")
    if report['errors']:
        for error in report['errors']:
            print(f"  - {error['field']}: {error['message']}")
    
    # Test model parameters validation
    print("\n2. Testing Model Parameters Validation...")
    model_params = {
        "learning_rate": 0.001,
        "batch_size": 64,  # Power of 2 for ARM64
        "epochs": 100,
        "dropout": 0.2
    }
    
    errors = validate_model_parameters(model_params)
    report = create_validation_report(errors)
    print(f"Model parameters validation: {report['status']}")
    if report['errors']:
        for error in report['errors']:
            print(f"  - {error['field']}: {error['message']}")
    
    # Test array validation with ARM64 optimizations
    print("\n3. Testing Array Validation...")
    test_array = np.random.random((100, 10)).astype(np.float32)
    
    errors = validate_numeric_array(
        test_array,
        expected_shape=(100, 10),
        expected_dtype=np.float32
    )
    report = create_validation_report(errors)
    print(f"Array validation: {report['status']}")
    if report['errors']:
        for error in report['errors']:
            print(f"  - {error['field']}: {error['message']}")
    
    # Test batch validation
    print("\n4. Testing Batch Validation...")
    data_batch = [
        {"symbol": "AAPL", "price": 150.0, "timestamp": datetime.now(timezone.utc)},
        {"symbol": "MSFT", "price": 300.0, "timestamp": datetime.now(timezone.utc)},
        {"symbol": "invalid", "price": -10.0, "timestamp": "invalid"}  # Invalid data
    ]
    
    validator = TradingDataValidator()
    batch_results = batch_validate(data_batch, validator)
    
    print(f"Batch validation completed. {len(batch_results)} items with errors:")
    for index, errors in batch_results.items():
        print(f"  Item {index}: {len(errors)} errors")
        for error in errors[:2]:  # Show first 2 errors
            print(f"    - {error.field}: {error.message}")
    
    print("\nEnhanced validation system testing completed!")