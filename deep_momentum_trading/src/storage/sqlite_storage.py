import sqlite3
import pandas as pd
import numpy as np
import threading
import time
import os
import asyncio
import aiosqlite
from typing import List, Dict, Optional, Tuple, Union, Any, Callable
from contextlib import contextmanager, asynccontextmanager
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor
import json
import pickle
import warnings
from pathlib import Path
import queue
import gc
from functools import lru_cache
import weakref

from deep_momentum_trading.src.utils.logger import get_logger
from deep_momentum_trading.config.settings import config_manager

logger = get_logger(__name__)

@dataclass
class SQLiteConfig:
    """Configuration for SQLite storage with ARM64 optimizations."""
    use_arm64_optimizations: bool = True
    journal_mode: str = "WAL"  # WAL, DELETE, TRUNCATE, PERSIST, MEMORY, OFF
    synchronous: str = "NORMAL"  # OFF, NORMAL, FULL, EXTRA
    cache_size: int = -64000  # Negative means KB, positive means pages
    temp_store: str = "MEMORY"  # DEFAULT, FILE, MEMORY
    mmap_size: int = 268435456  # 256MB
    page_size: int = 4096
    auto_vacuum: str = "INCREMENTAL"  # NONE, FULL, INCREMENTAL
    wal_autocheckpoint: int = 1000
    busy_timeout: int = 30000  # 30 seconds
    enable_fts: bool = True  # Full-text search
    enable_rtree: bool = True  # R-tree spatial index
    connection_pool_size: int = 10
    enable_connection_pooling: bool = True
    enable_prepared_statements: bool = True
    enable_batch_operations: bool = True
    batch_size: int = 1000

@dataclass
class TradeRecord:
    """Enhanced trade record with additional fields."""
    timestamp: int  # Unix timestamp in nanoseconds
    symbol: str
    side: str  # 'buy' or 'sell'
    quantity: float
    price: float
    commission: float = 0.0
    order_id: Optional[str] = None
    strategy: Optional[str] = None
    exchange: Optional[str] = None
    order_type: Optional[str] = None  # market, limit, stop, etc.
    execution_venue: Optional[str] = None
    slippage: Optional[float] = None
    market_impact: Optional[float] = None
    metadata: Optional[Dict] = None

@dataclass
class PerformanceMetric:
    """Enhanced performance metric record."""
    timestamp: int
    strategy: str
    metric_name: str
    metric_value: float
    period: Optional[str] = None  # daily, weekly, monthly, etc.
    benchmark: Optional[str] = None
    confidence_interval: Optional[Tuple[float, float]] = None
    metadata: Optional[Dict] = None

@dataclass
class ModelMetadata:
    """Enhanced model metadata with versioning and lineage."""
    model_name: str
    version: str
    model_type: Optional[str] = None  # LSTM, Transformer, etc.
    parent_version: Optional[str] = None  # For model lineage
    parameters: Optional[Dict] = None
    hyperparameters: Optional[Dict] = None
    performance_metrics: Optional[Dict] = None
    training_data_hash: Optional[str] = None
    feature_importance: Optional[Dict] = None
    model_size_bytes: Optional[int] = None
    inference_time_ms: Optional[float] = None
    created_at: Optional[int] = None
    created_by: Optional[str] = None

@dataclass
class SQLiteMetrics:
    """SQLite performance metrics."""
    total_connections: int = 0
    active_connections: int = 0
    total_queries: int = 0
    avg_query_time: float = 0.0
    cache_hit_ratio: float = 0.0
    wal_size_bytes: int = 0
    database_size_bytes: int = 0
    page_count: int = 0
    freelist_count: int = 0

class ConnectionPool:
    """Thread-safe connection pool for SQLite."""
    
    def __init__(self, db_path: str, config: SQLiteConfig, max_connections: int = 10):
        self.db_path = db_path
        self.config = config
        self.max_connections = max_connections
        self.pool = queue.Queue(maxsize=max_connections)
        self.active_connections = 0
        self.lock = threading.Lock()
        
        # Pre-populate pool
        for _ in range(min(3, max_connections)):
            conn = self._create_connection()
            self.pool.put(conn)
    
    def _create_connection(self) -> sqlite3.Connection:
        """Create optimized SQLite connection."""
        conn = sqlite3.connect(
            self.db_path,
            timeout=self.config.busy_timeout / 1000.0,
            check_same_thread=False
        )
        
        # Apply ARM64 optimizations
        self._optimize_connection(conn)
        
        with self.lock:
            self.active_connections += 1
        
        return conn
    
    def _optimize_connection(self, conn: sqlite3.Connection) -> None:
        """Apply ARM64 and performance optimizations."""
        cursor = conn.cursor()
        
        # Core performance settings
        cursor.execute(f"PRAGMA journal_mode={self.config.journal_mode}")
        cursor.execute(f"PRAGMA synchronous={self.config.synchronous}")
        cursor.execute(f"PRAGMA cache_size={self.config.cache_size}")
        cursor.execute(f"PRAGMA temp_store={self.config.temp_store}")
        cursor.execute(f"PRAGMA mmap_size={self.config.mmap_size}")
        cursor.execute(f"PRAGMA page_size={self.config.page_size}")
        cursor.execute(f"PRAGMA auto_vacuum={self.config.auto_vacuum}")
        cursor.execute(f"PRAGMA wal_autocheckpoint={self.config.wal_autocheckpoint}")
        
        # ARM64-specific optimizations
        if self.config.use_arm64_optimizations:
            # Enable optimizations for ARM64 architecture
            cursor.execute("PRAGMA optimize")
            cursor.execute("PRAGMA analysis_limit=1000")
            cursor.execute("PRAGMA threads=4")  # Use multiple threads
        
        # Enable extensions if available
        if self.config.enable_fts:
            try:
                cursor.execute("SELECT load_extension('fts5')")
            except:
                pass
        
        if self.config.enable_rtree:
            try:
                cursor.execute("SELECT load_extension('rtree')")
            except:
                pass
        
        conn.commit()
    
    @contextmanager
    def get_connection(self):
        """Get connection from pool."""
        conn = None
        try:
            # Try to get from pool
            try:
                conn = self.pool.get_nowait()
            except queue.Empty:
                # Create new connection if pool is empty and under limit
                with self.lock:
                    if self.active_connections < self.max_connections:
                        conn = self._create_connection()
                    else:
                        # Wait for connection to become available
                        conn = self.pool.get(timeout=5.0)
            
            yield conn
            
        finally:
            if conn:
                # Return to pool
                try:
                    self.pool.put_nowait(conn)
                except queue.Full:
                    # Pool is full, close connection
                    conn.close()
                    with self.lock:
                        self.active_connections -= 1
    
    def close_all(self):
        """Close all connections in pool."""
        while not self.pool.empty():
            try:
                conn = self.pool.get_nowait()
                conn.close()
            except queue.Empty:
                break
        
        with self.lock:
            self.active_connections = 0

class AdvancedSQLiteStorage:
    """
    Advanced SQLite storage with ARM64 optimizations, connection pooling,
    batch operations, and comprehensive transaction management.
    """

    def __init__(self, db_path: str = "data/storage/trading_history.db", config: Optional[SQLiteConfig] = None):
        """
        Initialize Advanced SQLite Storage.

        Args:
            db_path: Path to SQLite database file (defaults to config if not provided)
            config: SQLite configuration (loads from config manager if not provided)
        """
        # Load configuration from config manager if not provided
        if config is None:
            config_data = config_manager.get('storage_config.sqlite', {})
            self.config = SQLiteConfig(**config_data) if config_data else SQLiteConfig()
        else:
            self.config = config
            
        # Use default path from config if not explicitly provided
        if db_path == "data/storage/trading_history.db":
            db_path = config_manager.get('storage_config.paths.trading_history_db', db_path)
            
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Connection management
        if self.config.enable_connection_pooling:
            self.connection_pool = ConnectionPool(str(self.db_path), self.config)
        else:
            self.connection_pool = None
        
        # Thread safety
        self.lock = threading.RLock()
        self.local = threading.local()
        
        # Performance tracking
        self.metrics = SQLiteMetrics()
        self.query_times = []
        self.prepared_statements = {}
        
        # Batch operation queues
        self.batch_queues = {
            'trades': [],
            'performance_metrics': [],
            'model_metadata': []
        }
        self.batch_locks = {
            'trades': threading.Lock(),
            'performance_metrics': threading.Lock(),
            'model_metadata': threading.Lock()
        }
        
        # Initialize database
        self._initialize_database()
        
        # Setup ARM64 optimizations
        self._setup_arm64_optimizations()
        
        logger.info(f"AdvancedSQLiteStorage initialized at {self.db_path}")
        logger.info(f"ARM64 optimizations: {'enabled' if self.config.use_arm64_optimizations else 'disabled'}")
        logger.info(f"Connection pooling: {'enabled' if self.config.enable_connection_pooling else 'disabled'}")

    def _setup_arm64_optimizations(self) -> None:
        """Setup ARM64-specific optimizations."""
        if not self.config.use_arm64_optimizations:
            return
            
        try:
            # Set ARM64-specific environment variables
            os.environ.setdefault("SQLITE_ENABLE_COLUMN_METADATA", "1")
            os.environ.setdefault("SQLITE_ENABLE_FTS5", "1")
            os.environ.setdefault("SQLITE_ENABLE_RTREE", "1")
            
            logger.info("ARM64 optimizations applied for SQLite storage")
            
        except Exception as e:
            logger.warning(f"Failed to apply ARM64 optimizations: {e}")

    @contextmanager
    def get_connection(self):
        """Get database connection (pooled or direct)."""
        if self.connection_pool:
            with self.connection_pool.get_connection() as conn:
                yield conn
        else:
            # Direct connection for non-pooled mode
            if not hasattr(self.local, 'connection') or self.local.connection is None:
                self.local.connection = sqlite3.connect(
                    str(self.db_path),
                    timeout=self.config.busy_timeout / 1000.0,
                    check_same_thread=False
                )
                self._optimize_connection(self.local.connection)
            
            yield self.local.connection

    def _optimize_connection(self, conn: sqlite3.Connection) -> None:
        """Apply connection optimizations."""
        if self.connection_pool:
            # Already optimized in pool
            return
        
        cursor = conn.cursor()
        
        # Apply all optimizations
        cursor.execute(f"PRAGMA journal_mode={self.config.journal_mode}")
        cursor.execute(f"PRAGMA synchronous={self.config.synchronous}")
        cursor.execute(f"PRAGMA cache_size={self.config.cache_size}")
        cursor.execute(f"PRAGMA temp_store={self.config.temp_store}")
        cursor.execute(f"PRAGMA mmap_size={self.config.mmap_size}")
        cursor.execute(f"PRAGMA page_size={self.config.page_size}")
        cursor.execute(f"PRAGMA auto_vacuum={self.config.auto_vacuum}")
        cursor.execute(f"PRAGMA wal_autocheckpoint={self.config.wal_autocheckpoint}")
        
        if self.config.use_arm64_optimizations:
            cursor.execute("PRAGMA optimize")
            cursor.execute("PRAGMA analysis_limit=1000")
            cursor.execute("PRAGMA threads=4")
        
        conn.commit()

    def _initialize_database(self):
        """Initialize database schema with optimizations."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Enhanced trades table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp INTEGER NOT NULL,
                    symbol TEXT NOT NULL,
                    side TEXT NOT NULL CHECK(side IN ('buy', 'sell')),
                    quantity REAL NOT NULL CHECK(quantity > 0),
                    price REAL NOT NULL CHECK(price > 0),
                    commission REAL DEFAULT 0 CHECK(commission >= 0),
                    order_id TEXT UNIQUE,
                    strategy TEXT,
                    exchange TEXT,
                    order_type TEXT,
                    execution_venue TEXT,
                    slippage REAL,
                    market_impact REAL,
                    metadata TEXT,
                    created_at INTEGER DEFAULT (strftime('%s', 'now') * 1000000000)
                )
            """)
            
            # Enhanced performance metrics table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS performance_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp INTEGER NOT NULL,
                    strategy TEXT NOT NULL,
                    metric_name TEXT NOT NULL,
                    metric_value REAL NOT NULL,
                    period TEXT,
                    benchmark TEXT,
                    confidence_lower REAL,
                    confidence_upper REAL,
                    metadata TEXT,
                    created_at INTEGER DEFAULT (strftime('%s', 'now') * 1000000000)
                )
            """)
            
            # Enhanced model metadata table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS model_metadata (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_name TEXT NOT NULL,
                    version TEXT NOT NULL,
                    model_type TEXT,
                    parent_version TEXT,
                    parameters TEXT,
                    hyperparameters TEXT,
                    performance_metrics TEXT,
                    training_data_hash TEXT,
                    feature_importance TEXT,
                    model_size_bytes INTEGER,
                    inference_time_ms REAL,
                    created_at INTEGER DEFAULT (strftime('%s', 'now') * 1000000000),
                    created_by TEXT,
                    UNIQUE(model_name, version)
                )
            """)
            
            # Create optimized indexes
            self._create_indexes(cursor)
            
            # Create full-text search tables if enabled
            if self.config.enable_fts:
                self._create_fts_tables(cursor)
            
            conn.commit()
            logger.info("SQLite database schema initialized")

    def _create_indexes(self, cursor: sqlite3.Cursor):
        """Create optimized indexes for ARM64 performance."""
        indexes = [
            # Trades indexes
            "CREATE INDEX IF NOT EXISTS idx_trades_timestamp ON trades(timestamp)",
            "CREATE INDEX IF NOT EXISTS idx_trades_symbol ON trades(symbol)",
            "CREATE INDEX IF NOT EXISTS idx_trades_strategy ON trades(strategy)",
            "CREATE INDEX IF NOT EXISTS idx_trades_symbol_timestamp ON trades(symbol, timestamp)",
            "CREATE INDEX IF NOT EXISTS idx_trades_strategy_timestamp ON trades(strategy, timestamp)",
            
            # Performance metrics indexes
            "CREATE INDEX IF NOT EXISTS idx_performance_timestamp ON performance_metrics(timestamp)",
            "CREATE INDEX IF NOT EXISTS idx_performance_strategy ON performance_metrics(strategy)",
            "CREATE INDEX IF NOT EXISTS idx_performance_metric_name ON performance_metrics(metric_name)",
            "CREATE INDEX IF NOT EXISTS idx_performance_strategy_metric ON performance_metrics(strategy, metric_name)",
            
            # Model metadata indexes
            "CREATE INDEX IF NOT EXISTS idx_model_name_version ON model_metadata(model_name, version)",
            "CREATE INDEX IF NOT EXISTS idx_model_type ON model_metadata(model_type)",
            "CREATE INDEX IF NOT EXISTS idx_model_created_at ON model_metadata(created_at)",
        ]
        
        for index_sql in indexes:
            cursor.execute(index_sql)

    def _create_fts_tables(self, cursor: sqlite3.Cursor):
        """Create full-text search tables."""
        try:
            # FTS for trades
            cursor.execute("""
                CREATE VIRTUAL TABLE IF NOT EXISTS trades_fts USING fts5(
                    symbol, strategy, order_id, metadata,
                    content='trades',
                    content_rowid='id'
                )
            """)
            
            # FTS for model metadata
            cursor.execute("""
                CREATE VIRTUAL TABLE IF NOT EXISTS models_fts USING fts5(
                    model_name, model_type, created_by,
                    content='model_metadata',
                    content_rowid='id'
                )
            """)
            
        except Exception as e:
            logger.warning(f"Failed to create FTS tables: {e}")

    def insert_trade(self, trade: TradeRecord) -> bool:
        """Insert single trade record."""
        if self.config.enable_batch_operations:
            return self._add_to_batch('trades', trade)
        else:
            return self._insert_trade_direct(trade)

    def _insert_trade_direct(self, trade: TradeRecord) -> bool:
        """Insert trade directly to database."""
        start_time = time.time()
        
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    INSERT INTO trades (
                        timestamp, symbol, side, quantity, price, commission,
                        order_id, strategy, exchange, order_type, execution_venue,
                        slippage, market_impact, metadata
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    trade.timestamp, trade.symbol, trade.side, trade.quantity,
                    trade.price, trade.commission, trade.order_id, trade.strategy,
                    trade.exchange, trade.order_type, trade.execution_venue,
                    trade.slippage, trade.market_impact,
                    json.dumps(trade.metadata) if trade.metadata else None
                ))
                
                conn.commit()
                
                # Update metrics
                query_time = time.time() - start_time
                self.query_times.append(query_time)
                self.metrics.total_queries += 1
                
                logger.debug(f"Inserted trade: {trade.symbol} {trade.side} {trade.quantity} @ {trade.price}")
                return True
                
        except sqlite3.IntegrityError as e:
            logger.warning(f"Trade integrity error: {e}")
            return False
        except Exception as e:
            logger.error(f"Error inserting trade: {e}")
            return False

    def insert_performance_metric(self, metric: PerformanceMetric) -> bool:
        """Insert performance metric."""
        if self.config.enable_batch_operations:
            return self._add_to_batch('performance_metrics', metric)
        else:
            return self._insert_metric_direct(metric)

    def _insert_metric_direct(self, metric: PerformanceMetric) -> bool:
        """Insert metric directly to database."""
        start_time = time.time()
        
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                confidence_lower = None
                confidence_upper = None
                if metric.confidence_interval:
                    confidence_lower, confidence_upper = metric.confidence_interval
                
                cursor.execute("""
                    INSERT INTO performance_metrics (
                        timestamp, strategy, metric_name, metric_value,
                        period, benchmark, confidence_lower, confidence_upper, metadata
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    metric.timestamp, metric.strategy, metric.metric_name,
                    metric.metric_value, metric.period, metric.benchmark,
                    confidence_lower, confidence_upper,
                    json.dumps(metric.metadata) if metric.metadata else None
                ))
                
                conn.commit()
                
                # Update metrics
                query_time = time.time() - start_time
                self.query_times.append(query_time)
                self.metrics.total_queries += 1
                
                return True
                
        except Exception as e:
            logger.error(f"Error inserting performance metric: {e}")
            return False

    def insert_model_metadata(self, metadata: ModelMetadata) -> bool:
        """Insert or update model metadata."""
        if self.config.enable_batch_operations:
            return self._add_to_batch('model_metadata', metadata)
        else:
            return self._insert_model_direct(metadata)

    def _insert_model_direct(self, metadata: ModelMetadata) -> bool:
        """Insert model metadata directly to database."""
        start_time = time.time()
        
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    INSERT OR REPLACE INTO model_metadata (
                        model_name, version, model_type, parent_version,
                        parameters, hyperparameters, performance_metrics,
                        training_data_hash, feature_importance, model_size_bytes,
                        inference_time_ms, created_at, created_by
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    metadata.model_name, metadata.version, metadata.model_type,
                    metadata.parent_version,
                    json.dumps(metadata.parameters) if metadata.parameters else None,
                    json.dumps(metadata.hyperparameters) if metadata.hyperparameters else None,
                    json.dumps(metadata.performance_metrics) if metadata.performance_metrics else None,
                    metadata.training_data_hash,
                    json.dumps(metadata.feature_importance) if metadata.feature_importance else None,
                    metadata.model_size_bytes, metadata.inference_time_ms,
                    metadata.created_at or int(time.time() * 1e9),
                    metadata.created_by
                ))
                
                conn.commit()
                
                # Update metrics
                query_time = time.time() - start_time
                self.query_times.append(query_time)
                self.metrics.total_queries += 1
                
                return True
                
        except Exception as e:
            logger.error(f"Error inserting model metadata: {e}")
            return False

    def _add_to_batch(self, table: str, record: Any) -> bool:
        """Add record to batch queue."""
        try:
            with self.batch_locks[table]:
                self.batch_queues[table].append(record)
                
                # Auto-flush if batch is full
                if len(self.batch_queues[table]) >= self.config.batch_size:
                    self._flush_batch(table)
            
            return True
            
        except Exception as e:
            logger.error(f"Error adding to batch {table}: {e}")
            return False

    def _flush_batch(self, table: str) -> bool:
        """Flush batch queue to database."""
        with self.batch_locks[table]:
            if not self.batch_queues[table]:
                return True
            
            records = self.batch_queues[table].copy()
            self.batch_queues[table].clear()
        
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                if table == 'trades':
                    self._batch_insert_trades(cursor, records)
                elif table == 'performance_metrics':
                    self._batch_insert_metrics(cursor, records)
                elif table == 'model_metadata':
                    self._batch_insert_models(cursor, records)
                
                conn.commit()
                logger.debug(f"Flushed {len(records)} records to {table}")
                return True
                
        except Exception as e:
            logger.error(f"Error flushing batch {table}: {e}")
            # Put records back in queue
            with self.batch_locks[table]:
                self.batch_queues[table] = records + self.batch_queues[table]
            return False

    def _batch_insert_trades(self, cursor: sqlite3.Cursor, trades: List[TradeRecord]):
        """Batch insert trades."""
        data = []
        for trade in trades:
            data.append((
                trade.timestamp, trade.symbol, trade.side, trade.quantity,
                trade.price, trade.commission, trade.order_id, trade.strategy,
                trade.exchange, trade.order_type, trade.execution_venue,
                trade.slippage, trade.market_impact,
                json.dumps(trade.metadata) if trade.metadata else None
            ))
        
        cursor.executemany("""
            INSERT OR IGNORE INTO trades (
                timestamp, symbol, side, quantity, price, commission,
                order_id, strategy, exchange, order_type, execution_venue,
                slippage, market_impact, metadata
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, data)

    def _batch_insert_metrics(self, cursor: sqlite3.Cursor, metrics: List[PerformanceMetric]):
        """Batch insert performance metrics."""
        data = []
        for metric in metrics:
            confidence_lower = None
            confidence_upper = None
            if metric.confidence_interval:
                confidence_lower, confidence_upper = metric.confidence_interval
            
            data.append((
                metric.timestamp, metric.strategy, metric.metric_name,
                metric.metric_value, metric.period, metric.benchmark,
                confidence_lower, confidence_upper,
                json.dumps(metric.metadata) if metric.metadata else None
            ))
        
        cursor.executemany("""
            INSERT INTO performance_metrics (
                timestamp, strategy, metric_name, metric_value,
                period, benchmark, confidence_lower, confidence_upper, metadata
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, data)

    def _batch_insert_models(self, cursor: sqlite3.Cursor, models: List[ModelMetadata]):
        """Batch insert model metadata."""
        data = []
        for metadata in models:
            data.append((
                metadata.model_name, metadata.version, metadata.model_type,
                metadata.parent_version,
                json.dumps(metadata.parameters) if metadata.parameters else None,
                json.dumps(metadata.hyperparameters) if metadata.hyperparameters else None,
                json.dumps(metadata.performance_metrics) if metadata.performance_metrics else None,
                metadata.training_data_hash,
                json.dumps(metadata.feature_importance) if metadata.feature_importance else None,
                metadata.model_size_bytes, metadata.inference_time_ms,
                metadata.created_at or int(time.time() * 1e9),
                metadata.created_by
            ))
        
        cursor.executemany("""
            INSERT OR REPLACE INTO model_metadata (
                model_name, version, model_type, parent_version,
                parameters, hyperparameters, performance_metrics,
                training_data_hash, feature_importance, model_size_bytes,
                inference_time_ms, created_at, created_by
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, data)

    def flush_all_batches(self) -> bool:
        """Flush all batch queues."""
        success = True
        for table in self.batch_queues.keys():
            if not self._flush_batch(table):
                success = False
        return success

    def get_trades(self, 
                  symbol: Optional[str] = None,
                  strategy: Optional[str] = None,
                  start_timestamp: Optional[int] = None,
                  end_timestamp: Optional[int] = None,
                  limit: Optional[int] = None) -> pd.DataFrame:
        """Get trades with advanced filtering."""
        start_time = time.time()
        
        try:
            with self.get_connection() as conn:
                query = "SELECT * FROM trades WHERE 1=1"
                params = []
                
                if symbol:
                    query += " AND symbol = ?"
                    params.append(symbol)
                
                if strategy:
                    query += " AND strategy = ?"
                    params.append(strategy)
                
                if start_timestamp:
                    query += " AND timestamp >= ?"
                    params.append(start_timestamp)
                
                if end_timestamp:
                    query += " AND timestamp <= ?"
                    params.append(end_timestamp)
                
                query += " ORDER BY timestamp DESC"
                
                if limit:
                    query += f" LIMIT {limit}"
                
                df = pd.read_sql_query(query, conn, params=params)
                
                if not df.empty:
                    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ns')
                    if 'created_at' in df.columns:
                        df['created_at'] = pd.to_datetime(df['created_at'], unit='ns')
                
                # Update metrics
                query_time = time.time() - start_time
                self.query_times.append(query_time)
                self.metrics.total_queries += 1
                
                logger.info(f"Retrieved {len(df)} trades ({query_time:.3f}s)")
                return df
                
        except Exception as e:
            logger.error(f"Error retrieving trades: {e}")
            return pd.DataFrame()

    def get_portfolio_positions(self, as_of_timestamp: Optional[int] = None) -> Dict[str, float]:
        """Calculate portfolio positions with ARM64 optimizations."""
        start_time = time.time()
        
        try:
            with self.get_connection() as conn:
                if as_of_timestamp:
                    query = """
                        SELECT symbol, 
                               SUM(CASE WHEN side='buy' THEN quantity ELSE -quantity END) as net_position
                        FROM trades 
                        WHERE timestamp <= ?
                        GROUP BY symbol
                        HAVING ABS(net_position) > 1e-6
                    """
                    params = (as_of_timestamp,)
                else:
                    query = """
                        SELECT symbol, 
                               SUM(CASE WHEN side='buy' THEN quantity ELSE -quantity END) as net_position
                        FROM trades 
                        GROUP BY symbol
                        HAVING ABS(net_position) > 1e-6
                    """
                    params = ()
                
                cursor = conn.execute(query, params)
                positions = {row[0]: row[1] for row in cursor.fetchall()}
                
                # Update metrics
                query_time = time.time() - start_time
                self.query_times.append(query_time)
                self.metrics.total_queries += 1
                
                logger.info(f"Calculated {len(positions)} positions ({query_time:.3f}s)")
                return positions
                
        except Exception as e:
            logger.error(f"Error calculating positions: {e}")
            return {}

    def get_performance_metrics(self, 
                              strategy: Optional[str] = None,
                              metric_name: Optional[str] = None,
                              start_timestamp: Optional[int] = None,
                              end_timestamp: Optional[int] = None) -> pd.DataFrame:
        """Get performance metrics with filtering."""
        start_time = time.time()
        
        try:
            with self.get_connection() as conn:
                query = "SELECT * FROM performance_metrics WHERE 1=1"
                params = []
                
                if strategy:
                    query += " AND strategy = ?"
                    params.append(strategy)
                
                if metric_name:
                    query += " AND metric_name = ?"
                    params.append(metric_name)
                
                if start_timestamp:
                    query += " AND timestamp >= ?"
                    params.append(start_timestamp)
                
                if end_timestamp:
                    query += " AND timestamp <= ?"
                    params.append(end_timestamp)
                
                query += " ORDER BY timestamp DESC"
                
                df = pd.read_sql_query(query, conn, params=params)
                
                if not df.empty:
                    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ns')
                    if 'created_at' in df.columns:
                        df['created_at'] = pd.to_datetime(df['created_at'], unit='ns')
                
                # Update metrics
                query_time = time.time() - start_time
                self.query_times.append(query_time)
                self.metrics.total_queries += 1
                
                logger.info(f"Retrieved {len(df)} performance metrics ({query_time:.3f}s)")
                return df
                
        except Exception as e:
            logger.error(f"Error retrieving performance metrics: {e}")
            return pd.DataFrame()

    def get_model_metadata(self, 
                          model_name: Optional[str] = None,
                          version: Optional[str] = None,
                          model_type: Optional[str] = None) -> pd.DataFrame:
        """Get model metadata with filtering."""
        start_time = time.time()
        
        try:
            with self.get_connection() as conn:
                query = "SELECT * FROM model_metadata WHERE 1=1"
                params = []
                
                if model_name:
                    query += " AND model_name = ?"
                    params.append(model_name)
                
                if version:
                    query += " AND version = ?"
                    params.append(version)
                
                if model_type:
                    query += " AND model_type = ?"
                    params.append(model_type)
                
                query += " ORDER BY created_at DESC"
                
                df = pd.read_sql_query(query, conn, params=params)
                
                if not df.empty:
                    # Parse JSON columns
                    json_columns = ['parameters', 'hyperparameters', 'performance_metrics', 'feature_importance']
                    for col in json_columns:
                        if col in df.columns:
                            df[col] = df[col].apply(lambda x: json.loads(x) if x else None)
                    
                    if 'created_at' in df.columns:
                        df['created_at'] = pd.to_datetime(df['created_at'], unit='ns')
                
                # Update metrics
                query_time = time.time() - start_time
                self.query_times.append(query_time)
                self.metrics.total_queries += 1
                
                logger.info(f"Retrieved {len(df)} model metadata records ({query_time:.3f}s)")
                return df
                
        except Exception as e:
            logger.error(f"Error retrieving model metadata: {e}")
            return pd.DataFrame()

    async def get_trades_async(self, **kwargs) -> pd.DataFrame:
        """Async version of get_trades."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.get_trades, **kwargs)

    async def insert_trade_async(self, trade: TradeRecord) -> bool:
        """Async version of insert_trade."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.insert_trade, trade)

    def get_metrics(self) -> SQLiteMetrics:
        """Get comprehensive storage metrics."""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                # Database size
                cursor.execute("PRAGMA page_count")
                page_count = cursor.fetchone()[0]
                
                cursor.execute("PRAGMA page_size")
                page_size = cursor.fetchone()[0]
                
                cursor.execute("PRAGMA freelist_count")
                freelist_count = cursor.fetchone()[0]
                
                self.metrics.database_size_bytes = page_count * page_size
                self.metrics.page_count = page_count
                self.metrics.freelist_count = freelist_count
                
                # WAL size
                wal_path = Path(str(self.db_path) + "-wal")
                if wal_path.exists():
                    self.metrics.wal_size_bytes = wal_path.stat().st_size
                
                # Query performance
                if self.query_times:
                    self.metrics.avg_query_time = sum(self.query_times) / len(self.query_times)
                
                # Connection pool stats
                if self.connection_pool:
                    self.metrics.active_connections = self.connection_pool.active_connections
                    self.metrics.total_connections = self.connection_pool.max_connections
                
        except Exception as e:
            logger.warning(f"Failed to collect metrics: {e}")
        
        return self.metrics

    def optimize_database(self) -> None:
        """Optimize database performance."""
        logger.info("Optimizing SQLite database")
        
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                # Analyze tables for better query planning
                cursor.execute("ANALYZE")
                
                # Optimize database
                cursor.execute("PRAGMA optimize")
                
                # Vacuum if needed
                cursor.execute("PRAGMA freelist_count")
                freelist_count = cursor.fetchone()[0]
                
                if freelist_count > 1000:  # Threshold for vacuum
                    cursor.execute("VACUUM")
                    logger.info("Database vacuumed")
                
                # Update statistics
                cursor.execute("PRAGMA wal_checkpoint(TRUNCATE)")
                
                conn.commit()
                
            logger.info("Database optimization completed")
            
        except Exception as e:
            logger.error(f"Database optimization failed: {e}")

    def backup_database(self, backup_path: str) -> bool:
        """Create database backup."""
        try:
            backup_path = Path(backup_path)
            backup_path.parent.mkdir(parents=True, exist_ok=True)
            
            with self.get_connection() as conn:
                backup_conn = sqlite3.connect(str(backup_path))
                conn.backup(backup_conn)
                backup_conn.close()
            
            logger.info(f"Database backed up to {backup_path}")
            return True
            
        except Exception as e:
            logger.error(f"Database backup failed: {e}")
            return False

    def close(self) -> None:
        """Close storage and cleanup resources."""
        # Flush any remaining batches
        self.flush_all_batches()
        
        # Close connection pool
        if self.connection_pool:
            self.connection_pool.close_all()
        
        # Close thread-local connections
        if hasattr(self.local, 'connection') and self.local.connection:
            self.local.connection.close()
        
        logger.info("SQLite storage closed")

# Legacy compatibility
SQLiteTransactionStorage = AdvancedSQLiteStorage

if __name__ == "__main__":
    # Example usage with ARM64 optimizations
    config = SQLiteConfig(
        use_arm64_optimizations=True,
        enable_connection_pooling=True,
        enable_batch_operations=True,
        batch_size=100,
        cache_size=-128000,  # 128MB cache
        mmap_size=268435456  # 256MB mmap
    )
    
    storage = AdvancedSQLiteStorage("temp_trading_test.db", config)
    
    print("=== Advanced SQLite Storage Test ===")
    
    # Test trade insertion
    print("\n1. Trade Insertion Test:")
    trades = []
    for i in range(1000):
        trade = TradeRecord(
            timestamp=int(time.time() * 1e9) + i * 1000000,  # Microsecond intervals
            symbol=f"STOCK{i % 10}",
            side="buy" if i % 2 == 0 else "sell",
            quantity=float(10 + i % 100),
            price=float(100 + i % 50),
            commission=0.01,
            order_id=f"ORD{i:06d}",
            strategy="TestStrategy",
            exchange="NYSE",
            order_type="market"
        )
        trades.append(trade)
        storage.insert_trade(trade)
    
    # Flush batches
    storage.flush_all_batches()
    print(f"   Inserted {len(trades)} trades")
    
    # Test performance metrics
    print("\n2. Performance Metrics Test:")
    for i in range(100):
        metric = PerformanceMetric(
            timestamp=int(time.time() * 1e9) + i * 1000000,
            strategy="TestStrategy",
            metric_name=f"metric_{i % 5}",
            metric_value=float(i * 0.01),
            period="daily",
            confidence_interval=(float(i * 0.005), float(i * 0.015))
        )
        storage.insert_performance_metric(metric)
    
    storage.flush_all_batches()
    print("   Inserted 100 performance metrics")
    
    # Test model metadata
    print("\n3. Model Metadata Test:")
    for i in range(10):
        metadata = ModelMetadata(
            model_name=f"Model_{i % 3}",
            version=f"1.{i}.0",
            model_type="LSTM",
            parameters={"layers": 3 + i, "units": 64 + i * 16},
            performance_metrics={"accuracy": 0.9 + i * 0.001},
            model_size_bytes=1024 * 1024 * (10 + i),
            inference_time_ms=float(10 + i * 0.5)
        )
        storage.insert_model_metadata(metadata)
    
    storage.flush_all_batches()
    print("   Inserted 10 model metadata records")
    
    # Test queries
    print("\n4. Query Performance Test:")
    
    # Get trades
    start_time = time.time()
    trades_df = storage.get_trades(symbol="STOCK0", limit=100)
    query_time = time.time() - start_time
    print(f"   Retrieved {len(trades_df)} trades in {query_time:.3f}s")
    
    # Get positions
    start_time = time.time()
    positions = storage.get_portfolio_positions()
    query_time = time.time() - start_time
    print(f"   Calculated {len(positions)} positions in {query_time:.3f}s")
    
    # Get performance metrics
    start_time = time.time()
    metrics_df = storage.get_performance_metrics(strategy="TestStrategy")
    query_time = time.time() - start_time
    print(f"   Retrieved {len(metrics_df)} metrics in {query_time:.3f}s")
    
    # Test async operations
    print("\n5. Async Operations Test:")
    async def test_async():
        trade = TradeRecord(
            timestamp=int(time.time() * 1e9),
            symbol="ASYNC_TEST",
            side="buy",
            quantity=100.0,
            price=50.0,
            order_id="ASYNC001"
        )
        
        success = await storage.insert_trade_async(trade)
        trades = await storage.get_trades_async(symbol="ASYNC_TEST")
        return success, len(trades)
    
    import asyncio
    async_success, async_count = asyncio.run(test_async())
    print(f"   Async operations: {'✓' if async_success else '✗'}, {async_count} records")
    
    # Show metrics
    print("\n6. Storage Metrics:")
    metrics = storage.get_metrics()
    print(f"   Database size: {metrics.database_size_bytes / 1024**2:.1f} MB")
    print(f"   WAL size: {metrics.wal_size_bytes / 1024:.1f} KB")
    print(f"   Avg query time: {metrics.avg_query_time:.3f}s")
    print(f"   Total queries: {metrics.total_queries}")
    print(f"   Active connections: {metrics.active_connections}")
    
    # Test optimization
    print("\n7. Database Optimization:")
    storage.optimize_database()
    print("   Database optimized")
    
    # Cleanup
    storage.close()
    
    import os
    os.remove("temp_trading_test.db")
    
    print("\n=== Advanced SQLite Storage Test Complete ===")
