"""
Enhanced stress testing system for Deep Momentum Trading System with ARM64 optimizations.

This module provides comprehensive stress testing capabilities including scenario analysis,
Monte Carlo simulations, historical stress tests, and custom scenario modeling with
ARM64-specific optimizations for high-performance calculations.
"""

import numpy as np
import pandas as pd
import threading
import time
import asyncio
from typing import Dict, List, Optional, Any, Tuple, Union, Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
import platform
from collections import defaultdict, deque
import warnings
from scipy import stats
from concurrent.futures import ThreadPoolExecutor, as_completed

# Internal imports
from ..utils.logger import get_logger, PerformanceLogger
from ..utils.decorators import performance_monitor, retry_with_backoff
from ..utils.exceptions import RiskError, ValidationError
from ..utils.validators import validate_numeric_data
from ..utils.shared_memory import create_shared_array, create_shared_dict, SharedArray, SharedDict

# ARM64 detection
IS_ARM64 = platform.machine().lower() in ['arm64', 'aarch64']

logger = get_logger(__name__)
perf_logger = PerformanceLogger(logger)

class StressTestType(Enum):
    """Stress test types."""
    MARKET_CRASH = "market_crash"
    LIQUIDITY_CRISIS = "liquidity_crisis"
    SECTOR_SHOCK = "sector_shock"
    CORRELATION_BREAKDOWN = "correlation_breakdown"
    VOLATILITY_SPIKE = "volatility_spike"
    INTEREST_RATE_SHOCK = "interest_rate_shock"
    CURRENCY_CRISIS = "currency_crisis"
    MONTE_CARLO = "monte_carlo"
    HISTORICAL_SCENARIO = "historical_scenario"
    CUSTOM_SCENARIO = "custom_scenario"

class StressSeverity(Enum):
    """Stress test severity levels."""
    MILD = "mild"
    MODERATE = "moderate"
    SEVERE = "severe"
    EXTREME = "extreme"

@dataclass
class StressTestConfig:
    """Enhanced stress testing configuration."""
    # Basic settings
    test_interval_hours: float = 24.0
    enable_real_time_testing: bool = True
    enable_monte_carlo: bool = True
    monte_carlo_simulations: int = 10000
    
    # Scenario parameters
    market_crash_severity: float = -0.20  # 20% drop
    liquidity_crisis_spread_multiplier: float = 3.0
    volatility_spike_multiplier: float = 2.5
    correlation_shock_level: float = 0.9
    
    # ARM64 optimizations
    enable_arm64_optimizations: bool = True
    enable_vectorized_calculations: bool = True
    enable_parallel_processing: bool = True
    max_concurrent_tests: int = 4
    
    # Performance settings
    test_timeout_seconds: float = 300.0  # 5 minutes
    enable_caching: bool = True
    cache_ttl_seconds: int = 3600  # 1 hour
    enable_shared_memory: bool = True
    shared_memory_size: int = 5000
    
    # Historical scenarios
    enable_historical_scenarios: bool = True
    historical_lookback_years: int = 20

@dataclass
class StressTestResult:
    """Stress test result."""
    test_type: StressTestType
    severity: StressSeverity
    portfolio_impact_percent: float
    portfolio_impact_value: float
    var_impact: float
    liquidity_impact: float
    sector_impacts: Dict[str, float]
    asset_impacts: Dict[str, float]
    stressed_portfolio_value: float
    stressed_positions: Dict[str, float]
    stressed_market_data: Dict[str, Any]
    confidence_level: float
    test_duration_ms: float
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

@dataclass
class MonteCarloResult:
    """Monte Carlo simulation result."""
    simulations: int
    percentile_impacts: Dict[str, float]  # 1%, 5%, 95%, 99%
    expected_impact: float
    worst_case_impact: float
    best_case_impact: float
    var_estimates: Dict[str, float]
    confidence_intervals: Dict[str, Tuple[float, float]]

class StressTester:
    """Enhanced stress testing system with ARM64 optimizations."""
    
    def __init__(self, config: Optional[StressTestConfig] = None):
        self.config = config or StressTestConfig()
        self.is_arm64 = IS_ARM64
        
        # Apply ARM64 optimizations
        if self.is_arm64 and self.config.enable_arm64_optimizations:
            self._apply_arm64_optimizations()
        
        # Scenario registry
        self.scenarios: Dict[StressTestType, Callable] = {
            StressTestType.MARKET_CRASH: self._scenario_market_crash,
            StressTestType.LIQUIDITY_CRISIS: self._scenario_liquidity_crisis,
            StressTestType.SECTOR_SHOCK: self._scenario_sector_shock,
            StressTestType.CORRELATION_BREAKDOWN: self._scenario_correlation_breakdown,
            StressTestType.VOLATILITY_SPIKE: self._scenario_volatility_spike,
            StressTestType.INTEREST_RATE_SHOCK: self._scenario_interest_rate_shock,
            StressTestType.MONTE_CARLO: self._scenario_monte_carlo,
            StressTestType.HISTORICAL_SCENARIO: self._scenario_historical,
            StressTestType.CUSTOM_SCENARIO: self._scenario_custom
        }
        
        # Test history and caching
        self.test_history: deque = deque(maxlen=1000)
        self.test_cache: Dict[str, Tuple[StressTestResult, datetime]] = {}
        
        # Performance tracking
        self.performance_stats = {
            "total_tests": 0,
            "successful_tests": 0,
            "avg_test_time_ms": 0.0,
            "cache_hits": 0,
            "cache_misses": 0
        }
        
        # Threading
        self._lock = threading.RLock()
        self._running = False
        self._background_task = None
        
        # Shared memory setup
        self._setup_shared_memory()
        
        logger.info(f"StressTester initialized (ARM64: {self.is_arm64}, "
                   f"monte_carlo_sims={self.config.monte_carlo_simulations:,})")
    
    def _apply_arm64_optimizations(self):
        """Apply ARM64-specific optimizations."""
        # Increase concurrent tests for ARM64
        self.config.max_concurrent_tests = min(8, self.config.max_concurrent_tests * 2)
        
        # Increase Monte Carlo simulations for ARM64
        self.config.monte_carlo_simulations = min(50000, self.config.monte_carlo_simulations * 2)
        
        # Reduce timeout for faster ARM64 processing
        self.config.test_timeout_seconds = max(60.0, self.config.test_timeout_seconds * 0.5)
        
        logger.debug(f"Applied ARM64 optimizations: concurrent_tests={self.config.max_concurrent_tests}, "
                    f"monte_carlo_sims={self.config.monte_carlo_simulations}")
    
    def _setup_shared_memory(self):
        """Setup shared memory for high-performance data sharing."""
        if self.config.enable_shared_memory:
            try:
                self.shared_results = create_shared_array(
                    name="stress_test_results",
                    size=self.config.shared_memory_size,
                    dtype=np.float64
                )
                self.shared_metadata = create_shared_dict(
                    name="stress_test_metadata",
                    max_items=1000
                )
            except Exception as e:
                logger.warning(f"Failed to setup shared memory: {e}")
                self.shared_results = None
                self.shared_metadata = None
        else:
            self.shared_results = None
            self.shared_metadata = None
    
    @performance_monitor
    def run_stress_test(self, 
                       test_type: StressTestType,
                       portfolio_value: float,
                       positions: Dict[str, float],
                       market_data: Dict[str, Any],
                       severity: StressSeverity = StressSeverity.MODERATE,
                       **kwargs) -> StressTestResult:
        """Run a comprehensive stress test."""
        start_time = time.time()
        
        try:
            # Validate inputs
            if not validate_numeric_data(portfolio_value, min_value=0.0):
                raise ValidationError(f"Invalid portfolio value: {portfolio_value}")
            
            if not positions:
                raise ValidationError("No positions provided for stress test")
            
            # Check cache
            cache_key = self._generate_cache_key(test_type, portfolio_value, positions, severity)
            if self.config.enable_caching and cache_key in self.test_cache:
                cached_result, cache_time = self.test_cache[cache_key]
                if (datetime.now(timezone.utc) - cache_time).total_seconds() < self.config.cache_ttl_seconds:
                    self.performance_stats["cache_hits"] += 1
                    return cached_result
            
            # Run stress test scenario
            scenario_func = self.scenarios.get(test_type)
            if not scenario_func:
                raise ValidationError(f"Unknown stress test type: {test_type}")
            
            logger.info(f"Running stress test: {test_type.value} ({severity.value})")
            
            result = scenario_func(
                portfolio_value=portfolio_value,
                positions=positions,
                market_data=market_data,
                severity=severity,
                **kwargs
            )
            
            # Calculate test duration
            test_duration_ms = (time.time() - start_time) * 1000
            result.test_duration_ms = test_duration_ms
            
            # Cache result
            if self.config.enable_caching:
                self.test_cache[cache_key] = (result, datetime.now(timezone.utc))
                self.performance_stats["cache_misses"] += 1
            
            # Update performance stats
            self.performance_stats["total_tests"] += 1
            self.performance_stats["successful_tests"] += 1
            total_tests = self.performance_stats["total_tests"]
            current_avg = self.performance_stats["avg_test_time_ms"]
            self.performance_stats["avg_test_time_ms"] = (
                (current_avg * (total_tests - 1) + test_duration_ms) / total_tests
            )
            
            # Store in history
            self.test_history.append(result)
            
            # Update shared memory
            self._update_shared_memory(result)
            
            perf_logger.log_latency("stress_test", test_duration_ms)
            
            logger.info(f"Stress test completed: {test_type.value}, "
                       f"impact={result.portfolio_impact_percent:.2f}%, "
                       f"time={test_duration_ms:.1f}ms")
            
            return result
            
        except Exception as e:
            logger.error(f"Stress test failed: {e}")
            self.performance_stats["total_tests"] += 1
            
            # Return minimal result
            return StressTestResult(
                test_type=test_type,
                severity=severity,
                portfolio_impact_percent=0.0,
                portfolio_impact_value=0.0,
                var_impact=0.0,
                liquidity_impact=0.0,
                sector_impacts={},
                asset_impacts={},
                stressed_portfolio_value=portfolio_value,
                stressed_positions=positions,
                stressed_market_data=market_data,
                confidence_level=0.0,
                test_duration_ms=(time.time() - start_time) * 1000
            )
    
    def _scenario_market_crash(self, portfolio_value: float, positions: Dict[str, float],
                              market_data: Dict[str, Any], severity: StressSeverity,
                              **kwargs) -> StressTestResult:
        """Simulate market crash scenario."""
        # Determine crash severity
        severity_multipliers = {
            StressSeverity.MILD: -0.10,      # 10% drop
            StressSeverity.MODERATE: -0.20,  # 20% drop
            StressSeverity.SEVERE: -0.35,    # 35% drop
            StressSeverity.EXTREME: -0.50    # 50% drop
        }
        
        crash_multiplier = severity_multipliers[severity]
        
        # Apply crash to all assets
        stressed_positions = positions.copy()
        stressed_market_data = {}
        asset_impacts = {}
        total_impact_value = 0.0
        
        for symbol, data in market_data.items():
            if symbol in positions:
                original_price = data.get("last_price", 100.0)
                stressed_price = original_price * (1 + crash_multiplier)
                
                # Calculate impact for this asset
                position_qty = positions[symbol]
                original_value = position_qty * original_price
                stressed_value = position_qty * stressed_price
                impact_value = original_value - stressed_value
                
                asset_impacts[symbol] = impact_value
                total_impact_value += impact_value
                
                # Update stressed market data
                stressed_market_data[symbol] = {
                    "last_price": stressed_price,
                    "bid": stressed_price * 0.999,
                    "ask": stressed_price * 1.001,
                    "daily_volume": data.get("daily_volume", 1000000) * 1.5  # Volume spike
                }
        
        stressed_portfolio_value = portfolio_value - total_impact_value
        impact_percent = (total_impact_value / portfolio_value * 100) if portfolio_value > 0 else 0.0
        
        return StressTestResult(
            test_type=StressTestType.MARKET_CRASH,
            severity=severity,
            portfolio_impact_percent=impact_percent,
            portfolio_impact_value=total_impact_value,
            var_impact=abs(crash_multiplier),
            liquidity_impact=0.0,  # Market crash doesn't directly affect liquidity
            sector_impacts={},  # Would calculate if sector data available
            asset_impacts=asset_impacts,
            stressed_portfolio_value=stressed_portfolio_value,
            stressed_positions=stressed_positions,
            stressed_market_data=stressed_market_data,
            confidence_level=0.95,
            test_duration_ms=0.0  # Will be set by caller
        )
    
    def _scenario_liquidity_crisis(self, portfolio_value: float, positions: Dict[str, float],
                                  market_data: Dict[str, Any], severity: StressSeverity,
                                  **kwargs) -> StressTestResult:
        """Simulate liquidity crisis scenario."""
        # Determine liquidity impact based on severity
        severity_multipliers = {
            StressSeverity.MILD: 2.0,      # 2x spread widening
            StressSeverity.MODERATE: 3.0,  # 3x spread widening
            StressSeverity.SEVERE: 5.0,    # 5x spread widening
            StressSeverity.EXTREME: 10.0   # 10x spread widening
        }
        
        spread_multiplier = severity_multipliers[severity]
        volume_reduction = 1.0 / spread_multiplier  # Inverse relationship
        
        stressed_positions = positions.copy()
        stressed_market_data = {}
        asset_impacts = {}
        total_liquidity_impact = 0.0
        
        for symbol, data in market_data.items():
            if symbol in positions:
                original_price = data.get("last_price", 100.0)
                original_bid = data.get("bid", original_price * 0.999)
                original_ask = data.get("ask", original_price * 1.001)
                original_spread = original_ask - original_bid
                
                # Widen spreads
                new_spread = original_spread * spread_multiplier
                stressed_bid = original_price - new_spread / 2
                stressed_ask = original_price + new_spread / 2
                
                # Calculate liquidity impact (cost of immediate liquidation)
                position_qty = positions[symbol]
                if position_qty > 0:  # Long position - sell at bid
                    liquidation_value = position_qty * stressed_bid
                    fair_value = position_qty * original_price
                else:  # Short position - buy at ask
                    liquidation_value = abs(position_qty) * stressed_ask
                    fair_value = abs(position_qty) * original_price
                
                liquidity_impact = fair_value - liquidation_value
                asset_impacts[symbol] = liquidity_impact
                total_liquidity_impact += liquidity_impact
                
                # Update stressed market data
                stressed_market_data[symbol] = {
                    "last_price": original_price,  # Price doesn't change immediately
                    "bid": stressed_bid,
                    "ask": stressed_ask,
                    "daily_volume": data.get("daily_volume", 1000000) * volume_reduction
                }
        
        impact_percent = (total_liquidity_impact / portfolio_value * 100) if portfolio_value > 0 else 0.0
        
        return StressTestResult(
            test_type=StressTestType.LIQUIDITY_CRISIS,
            severity=severity,
            portfolio_impact_percent=impact_percent,
            portfolio_impact_value=total_liquidity_impact,
            var_impact=0.0,  # No direct price impact
            liquidity_impact=total_liquidity_impact,
            sector_impacts={},
            asset_impacts=asset_impacts,
            stressed_portfolio_value=portfolio_value - total_liquidity_impact,
            stressed_positions=stressed_positions,
            stressed_market_data=stressed_market_data,
            confidence_level=0.90,
            test_duration_ms=0.0
        )
    
    def _scenario_sector_shock(self, portfolio_value: float, positions: Dict[str, float],
                              market_data: Dict[str, Any], severity: StressSeverity,
                              sector_assets: Optional[List[str]] = None,
                              **kwargs) -> StressTestResult:
        """Simulate sector-specific shock scenario."""
        if not sector_assets:
            # Default to first 20% of assets as "sector"
            sector_assets = list(positions.keys())[:max(1, len(positions) // 5)]
        
        # Determine shock severity
        severity_multipliers = {
            StressSeverity.MILD: -0.15,      # 15% sector drop
            StressSeverity.MODERATE: -0.25,  # 25% sector drop
            StressSeverity.SEVERE: -0.40,    # 40% sector drop
            StressSeverity.EXTREME: -0.60    # 60% sector drop
        }
        
        shock_multiplier = severity_multipliers[severity]
        
        stressed_positions = positions.copy()
        stressed_market_data = market_data.copy()
        asset_impacts = {}
        sector_impact_value = 0.0
        
        for symbol in sector_assets:
            if symbol in positions and symbol in market_data:
                original_price = market_data[symbol].get("last_price", 100.0)
                stressed_price = original_price * (1 + shock_multiplier)
                
                # Calculate impact
                position_qty = positions[symbol]
                original_value = position_qty * original_price
                stressed_value = position_qty * stressed_price
                impact_value = original_value - stressed_value
                
                asset_impacts[symbol] = impact_value
                sector_impact_value += impact_value
                
                # Update market data
                stressed_market_data[symbol] = {
                    "last_price": stressed_price,
                    "bid": stressed_price * 0.998,
                    "ask": stressed_price * 1.002,
                    "daily_volume": market_data[symbol].get("daily_volume", 1000000) * 2.0
                }
        
        impact_percent = (sector_impact_value / portfolio_value * 100) if portfolio_value > 0 else 0.0
        
        return StressTestResult(
            test_type=StressTestType.SECTOR_SHOCK,
            severity=severity,
            portfolio_impact_percent=impact_percent,
            portfolio_impact_value=sector_impact_value,
            var_impact=abs(shock_multiplier),
            liquidity_impact=0.0,
            sector_impacts={"affected_sector": sector_impact_value},
            asset_impacts=asset_impacts,
            stressed_portfolio_value=portfolio_value - sector_impact_value,
            stressed_positions=stressed_positions,
            stressed_market_data=stressed_market_data,
            confidence_level=0.85,
            test_duration_ms=0.0
        )
    
    def _scenario_correlation_breakdown(self, portfolio_value: float, positions: Dict[str, float],
                                       market_data: Dict[str, Any], severity: StressSeverity,
                                       **kwargs) -> StressTestResult:
        """Simulate correlation breakdown scenario."""
        # In correlation breakdown, diversification benefits disappear
        # Assets that were negatively correlated become positively correlated
        
        # Generate random shocks with high correlation
        np.random.seed(42)  # For reproducibility
        n_assets = len(positions)
        
        # Base shock factor
        severity_base_shocks = {
            StressSeverity.MILD: 0.10,
            StressSeverity.MODERATE: 0.15,
            StressSeverity.SEVERE: 0.25,
            StressSeverity.EXTREME: 0.40
        }
        
        base_shock = severity_base_shocks[severity]
        
        # Generate correlated shocks (all negative in this scenario)
        common_factor = np.random.normal(-base_shock, base_shock * 0.3)
        
        stressed_positions = positions.copy()
        stressed_market_data = {}
        asset_impacts = {}
        total_impact_value = 0.0
        
        for i, (symbol, data) in enumerate(market_data.items()):
            if symbol in positions:
                # Individual shock + common factor (high correlation)
                individual_shock = np.random.normal(0, base_shock * 0.2)
                total_shock = 0.8 * common_factor + 0.2 * individual_shock
                
                original_price = data.get("last_price", 100.0)
                stressed_price = original_price * (1 + total_shock)
                
                # Calculate impact
                position_qty = positions[symbol]
                original_value = position_qty * original_price
                stressed_value = position_qty * stressed_price
                impact_value = original_value - stressed_value
                
                asset_impacts[symbol] = impact_value
                total_impact_value += impact_value
                
                # Update market data
                stressed_market_data[symbol] = {
                    "last_price": stressed_price,
                    "bid": stressed_price * 0.999,
                    "ask": stressed_price * 1.001,
                    "daily_volume": data.get("daily_volume", 1000000)
                }
        
        impact_percent = (total_impact_value / portfolio_value * 100) if portfolio_value > 0 else 0.0
        
        return StressTestResult(
            test_type=StressTestType.CORRELATION_BREAKDOWN,
            severity=severity,
            portfolio_impact_percent=impact_percent,
            portfolio_impact_value=total_impact_value,
            var_impact=base_shock,
            liquidity_impact=0.0,
            sector_impacts={},
            asset_impacts=asset_impacts,
            stressed_portfolio_value=portfolio_value - total_impact_value,
            stressed_positions=stressed_positions,
            stressed_market_data=stressed_market_data,
            confidence_level=0.80,
            test_duration_ms=0.0
        )
    
    def _scenario_volatility_spike(self, portfolio_value: float, positions: Dict[str, float],
                                  market_data: Dict[str, Any], severity: StressSeverity,
                                  **kwargs) -> StressTestResult:
        """Simulate volatility spike scenario."""
        # Volatility spike affects option values and risk metrics more than spot prices
        severity_vol_multipliers = {
            StressSeverity.MILD: 1.5,
            StressSeverity.MODERATE: 2.0,
            StressSeverity.SEVERE: 3.0,
            StressSeverity.EXTREME: 5.0
        }
        
        vol_multiplier = severity_vol_multipliers[severity]
        
        # Generate random price movements with higher volatility
        np.random.seed(42)
        
        stressed_positions = positions.copy()
        stressed_market_data = {}
        asset_impacts = {}
        total_impact_value = 0.0
        
        for symbol, data in market_data.items():
            if symbol in positions:
                original_price = data.get("last_price", 100.0)
                
                # Higher volatility random walk
                base_vol = 0.02  # 2% daily volatility
                stressed_vol = base_vol * vol_multiplier
                price_shock = np.random.normal(0, stressed_vol)
                
                stressed_price = original_price * (1 + price_shock)
                
                # Calculate impact
                position_qty = positions[symbol]
                original_value = position_qty * original_price
                stressed_value = position_qty * stressed_price
                impact_value = original_value - stressed_value
                
                asset_impacts[symbol] = impact_value
                total_impact_value += impact_value
                
                # Wider spreads due to higher volatility
                spread_multiplier = 1 + (vol_multiplier - 1) * 0.5
                original_spread = original_price * 0.002  # 0.2% spread
                new_spread = original_spread * spread_multiplier
                
                stressed_market_data[symbol] = {
                    "last_price": stressed_price,
                    "bid": stressed_price - new_spread / 2,
                    "ask": stressed_price + new_spread / 2,
                    "daily_volume": data.get("daily_volume", 1000000) * 1.2  # Slight volume increase
                }
        
        impact_percent = (total_impact_value / portfolio_value * 100) if portfolio_value > 0 else 0.0
        
        return StressTestResult(
            test_type=StressTestType.VOLATILITY_SPIKE,
            severity=severity,
            portfolio_impact_percent=impact_percent,
            portfolio_impact_value=total_impact_value,
            var_impact=stressed_vol * vol_multiplier,
            liquidity_impact=0.0,
            sector_impacts={},
            asset_impacts=asset_impacts,
            stressed_portfolio_value=portfolio_value - total_impact_value,
            stressed_positions=stressed_positions,
            stressed_market_data=stressed_market_data,
            confidence_level=0.75,
            test_duration_ms=0.0
        )
    
    def _scenario_interest_rate_shock(self, portfolio_value: float, positions: Dict[str, float],
                                     market_data: Dict[str, Any], severity: StressSeverity,
                                     **kwargs) -> StressTestResult:
        """Simulate interest rate shock scenario."""
        # Interest rate shocks affect different assets differently
        # Growth stocks are more sensitive than value stocks
        
        severity_rate_shocks = {
            StressSeverity.MILD: 0.005,      # 50 bps increase
            StressSeverity.MODERATE: 0.01,   # 100 bps increase
            StressSeverity.SEVERE: 0.02,     # 200 bps increase
            StressSeverity.EXTREME: 0.04     # 400 bps increase
        }
        
        rate_shock = severity_rate_shocks[severity]
        
        stressed_positions = positions.copy()
        stressed_market_data = {}
        asset_impacts = {}
        total_impact_value = 0.0
        
        for symbol, data in market_data.items():
            if symbol in positions:
                original_price = data.get("last_price", 100.0)
                
                # Simplified duration-based impact (assuming average duration of 5 years)
                # Higher growth stocks have higher duration sensitivity
                duration_sensitivity = np.random.uniform(3, 8)  # 3-8 years duration
                price_impact = -duration_sensitivity * rate_shock
                
                stressed_price = original_price * (1 + price_impact)
                
                # Calculate impact
                position_qty = positions[symbol]
                original_value = position_qty * original_price
                stressed_value = position_qty * stressed_price
                impact_value = original_value - stressed_value
                
                asset_impacts[symbol] = impact_value
                total_impact_value += impact_value
                
                stressed_market_data[symbol] = {
                    "last_price": stressed_price,
                    "bid": stressed_price * 0.999,
                    "ask": stressed_price * 1.001,
                    "daily_volume": data.get("daily_volume", 1000000)
                }
        
        impact_percent = (total_impact_value / portfolio_value * 100) if portfolio_value > 0 else 0.0
        
        return StressTestResult(
            test_type=StressTestType.INTEREST_RATE_SHOCK,
            severity=severity,
            portfolio_impact_percent=impact_percent,
            portfolio_impact_value=total_impact_value,
            var_impact=rate_shock * 5,  # Approximate duration impact
            liquidity_impact=0.0,
            sector_impacts={},
            asset_impacts=asset_impacts,
            stressed_portfolio_value=portfolio_value - total_impact_value,
            stressed_positions=stressed_positions,
            stressed_market_data=stressed_market_data,
            confidence_level=0.85,
            test_duration_ms=0.0
        )
    
    def _scenario_monte_carlo(self, portfolio_value: float, positions: Dict[str, float],
                             market_data: Dict[str, Any], severity: StressSeverity,
                             **kwargs) -> StressTestResult:
        """Run Monte Carlo simulation."""
        n_simulations = self.config.monte_carlo_simulations
        
        # Generate random scenarios
        np.random.seed(42)
        
        simulation_results = []
        
        for _ in range(n_simulations):
            scenario_impact = 0.0
            
            for symbol, data in market_data.items():
                if symbol in positions:
                    original_price = data.get("last_price", 100.0)
                    
                    # Random return from normal distribution
                    daily_return = np.random.normal(0, 0.02)  # 2% daily volatility
                    stressed_price = original_price * (1 + daily_return)
                    
                    # Calculate impact
                    position_qty = positions[symbol]
                    original_value = position_qty * original_price
                    stressed_value = position_qty * stressed_price
                    impact_value = original_value - stressed_value
                    
                    scenario_impact += impact_value
            
            simulation_results.append(scenario_impact)
        
        # Calculate statistics
        simulation_results = np.array(simulation_results)
        percentiles = {
            "1%": np.percentile(simulation_results, 1),
            "5%": np.percentile(simulation_results, 5),
            "95%": np.percentile(simulation_results, 95),
            "99%": np.percentile(simulation_results, 99)
        }
        
        expected_impact = np.mean(simulation_results)
        worst_case = np.min(simulation_results)
        best_case = np.max(simulation_results)
        
        # Use worst case for stress test result
        impact_percent = (abs(worst_case) / portfolio_value * 100) if portfolio_value > 0 else 0.0
        
        return StressTestResult(
            test_type=StressTestType.MONTE_CARLO,
            severity=severity,
            portfolio_impact_percent=impact_percent,
            portfolio_impact_value=abs(worst_case),
            var_impact=abs(percentiles["5%"]) / portfolio_value if portfolio_value > 0 else 0.0,
            liquidity_impact=0.0,
            sector_impacts={},
            asset_impacts={},
            stressed_portfolio_value=portfolio_value + worst_case,
            stressed_positions=positions,
            stressed_market_data=market_data,
            confidence_level=0.99,
            test_duration_ms=0.0
        )
    
    def _scenario_historical(self, portfolio_value: float, positions: Dict[str, float],
                            market_data: Dict[str, Any], severity: StressSeverity,
                            historical_date: Optional[str] = None,
                            **kwargs) -> StressTestResult:
        """Simulate historical scenario (simplified)."""
        # This would typically use actual historical data
        # For now, we'll simulate known historical events
        
        historical_scenarios = {
            "2008_financial_crisis": -0.37,  # 37% drop
            "2020_covid_crash": -0.34,       # 34% drop
            "2000_dotcom_crash": -0.49,      # 49% drop
            "1987_black_monday": -0.22       # 22% drop
        }
        
        scenario_name = historical_date or "2008_financial_crisis"
        historical_impact = historical_scenarios.get(scenario_name, -0.20)
        
        # Apply historical scenario
        stressed_positions = positions.copy()
        stressed_market_data = {}
        asset_impacts = {}
        total_impact_value = 0.0
        
        for symbol, data in market_data.items():
            if symbol in positions:
                original_price = data.get("last_price", 100.0)
                stressed_price = original_price * (1 + historical_impact)
                
                # Calculate impact
                position_qty = positions[symbol]
                original_value = position_qty * original_price
                stressed_value = position_qty * stressed_price
                impact_value = original_value - stressed_value
                
                asset_impacts[symbol] = impact_value
                total_impact_value += impact_value
                
                stressed_market_data[symbol] = {
                    "last_price": stressed_price,
                    "bid": stressed_price * 0.995,  # Wider spreads in crisis
                    "ask": stressed_price * 1.005,
                    "daily_volume": data.get("daily_volume", 1000000) * 3.0  # Volume spike
                }
        
        impact_percent = (total_impact_value / portfolio_value * 100) if portfolio_value > 0 else 0.0
        
        return StressTestResult(
            test_type=StressTestType.HISTORICAL_SCENARIO,
            severity=severity,
            portfolio_impact_percent=impact_percent,
            portfolio_impact_value=total_impact_value,
            var_impact=abs(historical_impact),
            liquidity_impact=0.0,
            sector_impacts={},
            asset_impacts=asset_impacts,
            stressed_portfolio_value=portfolio_value - total_impact_value,
            stressed_positions=stressed_positions,
            stressed_market_data=stressed_market_data,
            confidence_level=1.0,  # Historical events have 100% confidence
            test_duration_ms=0.0
        )
    
    def _scenario_custom(self, portfolio_value: float, positions: Dict[str, float],
                        market_data: Dict[str, Any], severity: StressSeverity,
                        custom_shocks: Optional[Dict[str, float]] = None,
                        **kwargs) -> StressTestResult:
        """Run custom scenario with user-defined shocks."""
        if not custom_shocks:
            # Default to mild market-wide shock
            custom_shocks = {symbol: -0.05 for symbol in positions.keys()}
        
        stressed_positions = positions.copy()
        stressed_market_data = {}
        asset_impacts = {}
        total_impact_value = 0.0
        
        for symbol, data in market_data.items():
            if symbol in positions:
                original_price = data.get("last_price", 100.0)
                shock = custom_shocks.get(symbol, 0.0)
                stressed_price = original_price * (1 + shock)
                
                # Calculate impact
                position_qty = positions[symbol]
                original_value = position_qty * original_price
                stressed_value = position_qty * stressed_price
                impact_value = original_value - stressed_value
                
                asset_impacts[symbol] = impact_value
                total_impact_value += impact_value
                
                stressed_market_data[symbol] = {
                    "last_price": stressed_price,
                    "bid": stressed_price * 0.999,
                    "ask": stressed_price * 1.001,
                    "daily_volume": data.get("daily_volume", 1000000)
                }
        
        impact_percent = (total_impact_value / portfolio_value * 100) if portfolio_value > 0 else 0.0
        
        return StressTestResult(
            test_type=StressTestType.CUSTOM_SCENARIO,
            severity=severity,
            portfolio_impact_percent=impact_percent,
            portfolio_impact_value=total_impact_value,
            var_impact=np.mean(list(custom_shocks.values())) if custom_shocks else 0.0,
            liquidity_impact=0.0,
            sector_impacts={},
            asset_impacts=asset_impacts,
            stressed_portfolio_value=portfolio_value - total_impact_value,
            stressed_positions=stressed_positions,
            stressed_market_data=stressed_market_data,
            confidence_level=0.90,
            test_duration_ms=0.0
        )
    
    def run_comprehensive_stress_tests(self, portfolio_value: float, positions: Dict[str, float],
                                     market_data: Dict[str, Any]) -> Dict[StressTestType, StressTestResult]:
        """Run comprehensive suite of stress tests."""
        results = {}
        
        # Standard scenarios to run
        scenarios_to_run = [
            (StressTestType.MARKET_CRASH, StressSeverity.MODERATE),
            (StressTestType.LIQUIDITY_CRISIS, StressSeverity.MODERATE),
            (StressTestType.SECTOR_SHOCK, StressSeverity.MODERATE),
            (StressTestType.CORRELATION_BREAKDOWN, StressSeverity.MODERATE),
            (StressTestType.VOLATILITY_SPIKE, StressSeverity.MODERATE)
        ]
        
        if self.config.enable_parallel_processing and self.config.max_concurrent_tests > 1:
            # Run tests in parallel
            with ThreadPoolExecutor(max_workers=self.config.max_concurrent_tests) as executor:
                future_to_scenario = {
                    executor.submit(
                        self.run_stress_test, test_type, portfolio_value, positions, market_data, severity
                    ): test_type
                    for test_type, severity in scenarios_to_run
                }
                
                for future in as_completed(future_to_scenario):
                    test_type = future_to_scenario[future]
                    try:
                        result = future.result()
                        results[test_type] = result
                    except Exception as e:
                        logger.error(f"Parallel stress test {test_type.value} failed: {e}")
        else:
            # Run tests sequentially
            for test_type, severity in scenarios_to_run:
                try:
                    result = self.run_stress_test(test_type, portfolio_value, positions, market_data, severity)
                    results[test_type] = result
                except Exception as e:
                    logger.error(f"Stress test {test_type.value} failed: {e}")
        
        logger.info(f"Comprehensive stress tests completed: {len(results)} scenarios")
        return results
    
    def _generate_cache_key(self, test_type: StressTestType, portfolio_value: float,
                           positions: Dict[str, float], severity: StressSeverity) -> str:
        """Generate cache key for stress test."""
        try:
            import hashlib
            
            key_data = f"{test_type.value}_{portfolio_value}_{len(positions)}_{severity.value}"
            return hashlib.md5(key_data.encode()).hexdigest()
        except Exception:
            return str(time.time())
    
    def _update_shared_memory(self, result: StressTestResult):
        """Update shared memory with stress test results."""
        if not self.shared_results or not self.shared_metadata:
            return
        
        try:
            # Store key metrics in shared array
            with self.shared_results.write_lock() as array:
                array[0] = result.portfolio_impact_percent
                array[1] = result.portfolio_impact_value
                array[2] = result.var_impact
                array[3] = result.liquidity_impact
                array[4] = result.confidence_level
            
            # Store metadata
            metadata = {
                "test_type": result.test_type.value,
                "severity": result.severity.value,
                "portfolio_impact_percent": result.portfolio_impact_percent,
                "timestamp": result.timestamp.isoformat(),
                "test_duration_ms": result.test_duration_ms
            }
            
            import json
            self.shared_metadata.put("latest_stress_test", json.dumps(metadata).encode())
            
        except Exception as e:
            logger.warning(f"Failed to update shared memory: {e}")
    
    def start(self):
        """Start background stress testing."""
        if self._running:
            logger.warning("StressTester is already running")
            return
        
        self._running = True
        if self.config.enable_real_time_testing:
            self._background_task = threading.Thread(target=self._background_testing, daemon=True)
            self._background_task.start()
        
        logger.info("StressTester started")
    
    def stop(self):
        """Stop background stress testing."""
        self._running = False
        if self._background_task and self._background_task.is_alive():
            self._background_task.join(timeout=5.0)
        logger.info("StressTester stopped")
    
    def _background_testing(self):
        """Background stress testing loop."""
        while self._running:
            try:
                # This would run periodic stress tests
                # For now, just sleep
                time.sleep(self.config.test_interval_hours * 3600)
                
            except Exception as e:
                logger.error(f"Error in background stress testing: {e}")
                time.sleep(3600)  # Sleep 1 hour on error
    
    def get_latest_results(self) -> Dict[str, Any]:
        """Get latest stress test results."""
        if not self.test_history:
            return {"status": "no_tests_run"}
        
        latest = self.test_history[-1]
        
        return {
            "latest_result": {
                "test_type": latest.test_type.value,
                "severity": latest.severity.value,
                "portfolio_impact_percent": latest.portfolio_impact_percent,
                "portfolio_impact_value": latest.portfolio_impact_value,
                "confidence_level": latest.confidence_level,
                "test_duration_ms": latest.test_duration_ms,
                "timestamp": latest.timestamp.isoformat()
            },
            "performance_stats": self.performance_stats,
            "total_tests_run": len(self.test_history)
        }
    
    def get_status(self) -> Dict[str, Any]:
        """Get stress tester status."""
        return {
            "running": self._running,
            "arm64_optimized": self.is_arm64 and self.config.enable_arm64_optimizations,
            "monte_carlo_simulations": self.config.monte_carlo_simulations,
            "max_concurrent_tests": self.config.max_concurrent_tests,
            "available_scenarios": [t.value for t in StressTestType],
            "performance_stats": self.performance_stats,
            "tests_in_history": len(self.test_history),
            "cache_hit_rate": (
                self.performance_stats["cache_hits"] / 
                (self.performance_stats["cache_hits"] + self.performance_stats["cache_misses"])
                if (self.performance_stats["cache_hits"] + self.performance_stats["cache_misses"]) > 0 else 0.0
            )
        }
    
    def get_available_scenarios(self) -> List[str]:
        """Get list of available stress test scenarios."""
        return [test_type.value for test_type in StressTestType]

# Export all public components
__all__ = [
    "StressTester",
    "StressTestConfig",
    "StressTestResult",
    "MonteCarloResult",
    "StressTestType",
    "StressSeverity",
    "IS_ARM64"
]
