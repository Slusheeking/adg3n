"""
Enhanced Scheduler with ARM64 Optimizations

This module provides comprehensive task scheduling capabilities for the Deep Momentum Trading System
with ARM64-specific optimizations for high-performance task orchestration and execution.

Features:
- Advanced task scheduling with ARM64 optimizations
- Priority-based task queuing and execution
- Distributed task coordination and load balancing
- Real-time task monitoring and performance tracking
- Fault tolerance and automatic retry mechanisms
- Resource-aware task scheduling
- Cron-like scheduling with advanced patterns
- Task dependency management and execution graphs
"""

import asyncio
import time
import threading
import platform
import multiprocessing as mp
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Callable, Awaitable
from dataclasses import dataclass, field
from enum import Enum
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, Future
import logging
from functools import wraps
import queue
import heapq
from collections import defaultdict, deque
import croniter
import uuid
import json
from pathlib import Path

from ..utils.logger import get_logger
from ..utils.shared_memory import SharedMemoryManager
from ..utils.decorators import performance_monitor, error_handler
from ..utils.exceptions import ValidationError, SchedulerError

logger = get_logger(__name__)

class TaskPriority(Enum):
    """Task priority levels"""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4

class TaskStatus(Enum):
    """Task execution status"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    RETRYING = "retrying"

class ScheduleType(Enum):
    """Types of task schedules"""
    ONCE = "once"
    INTERVAL = "interval"
    CRON = "cron"
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"

@dataclass
class SchedulerConfig:
    """Configuration for task scheduler"""
    enable_arm64_optimizations: bool = True
    use_shared_memory: bool = True
    enable_distributed_scheduling: bool = True
    parallel_processing: bool = True
    max_workers: Optional[int] = None
    max_concurrent_tasks: int = 100
    task_timeout: float = 300.0  # 5 minutes
    retry_attempts: int = 3
    retry_delay: float = 5.0
    enable_task_persistence: bool = True
    persistence_file: str = "scheduler_state.json"
    enable_performance_monitoring: bool = True
    cleanup_interval: float = 3600.0  # 1 hour

@dataclass
class TaskDefinition:
    """Task definition"""
    id: str
    name: str
    func: Union[Callable, str]
    args: tuple = field(default_factory=tuple)
    kwargs: Dict[str, Any] = field(default_factory=dict)
    priority: TaskPriority = TaskPriority.NORMAL
    schedule_type: ScheduleType = ScheduleType.ONCE
    schedule_value: Optional[Union[str, int, float]] = None
    max_retries: int = 3
    timeout: Optional[float] = None
    dependencies: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class TaskExecution:
    """Task execution instance"""
    task_id: str
    execution_id: str
    definition: TaskDefinition
    scheduled_time: float
    status: TaskStatus = TaskStatus.PENDING
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    result: Any = None
    error: Optional[str] = None
    retry_count: int = 0
    execution_time: Optional[float] = None
    resource_usage: Dict[str, Any] = field(default_factory=dict)

class ARM64SchedulerOptimizer:
    """ARM64-specific optimizations for task scheduling"""
    
    def __init__(self):
        self.is_arm64 = platform.machine().lower() in ['arm64', 'aarch64']
        self.cpu_count = mp.cpu_count()
        self.cpu_topology = self._detect_cpu_topology()
        
    def _detect_cpu_topology(self) -> Dict[str, Any]:
        """Detect ARM64 CPU topology for optimal task scheduling"""
        topology = {
            'total_cores': self.cpu_count,
            'performance_cores': [],
            'efficiency_cores': [],
            'numa_nodes': []
        }
        
        if not self.is_arm64:
            return topology
        
        try:
            # Detect big.LITTLE architecture
            with open('/proc/cpuinfo', 'r') as f:
                cpuinfo = f.read()
            
            # Parse CPU information
            cores_info = []
            current_core = {}
            
            for line in cpuinfo.split('\n'):
                if line.startswith('processor'):
                    if current_core:
                        cores_info.append(current_core)
                    current_core = {'id': int(line.split(':')[1].strip())}
                elif line.startswith('CPU part'):
                    current_core['part'] = line.split(':')[1].strip()
            
            if current_core:
                cores_info.append(current_core)
            
            # Classify cores based on CPU part (simplified)
            for core in cores_info:
                core_id = core['id']
                cpu_part = core.get('part', '')
                
                # This is a simplified heuristic - real implementation would be more sophisticated
                if 'x1' in cpu_part.lower() or 'a78' in cpu_part.lower():
                    topology['performance_cores'].append(core_id)
                else:
                    topology['efficiency_cores'].append(core_id)
            
            # Fallback if classification fails
            if not topology['performance_cores'] and not topology['efficiency_cores']:
                mid_point = self.cpu_count // 2
                topology['efficiency_cores'] = list(range(mid_point))
                topology['performance_cores'] = list(range(mid_point, self.cpu_count))
                
        except Exception as e:
            logger.warning(f"Could not detect ARM64 CPU topology: {e}")
            # Fallback: assume all cores are equivalent
            topology['performance_cores'] = list(range(self.cpu_count))
        
        return topology
    
    def optimize_task_placement(self, task_priority: TaskPriority) -> Optional[List[int]]:
        """Optimize task placement based on ARM64 topology and priority"""
        if not self.is_arm64:
            return None
        
        topology = self.cpu_topology
        
        if task_priority == TaskPriority.CRITICAL:
            # Use performance cores for critical tasks
            return topology['performance_cores'] if topology['performance_cores'] else None
        elif task_priority == TaskPriority.HIGH:
            # Use mix of performance and efficiency cores
            cores = topology['performance_cores'] + topology['efficiency_cores'][:2]
            return cores if cores else None
        elif task_priority == TaskPriority.LOW:
            # Use efficiency cores for low priority tasks
            return topology['efficiency_cores'] if topology['efficiency_cores'] else None
        else:
            # Normal priority - use all cores
            return None
    
    def calculate_optimal_workers(self, task_priority: TaskPriority) -> int:
        """Calculate optimal number of workers based on ARM64 capabilities"""
        base_workers = self.cpu_count
        
        if task_priority == TaskPriority.CRITICAL:
            return min(base_workers, 8)  # Limit to avoid oversubscription
        elif task_priority == TaskPriority.HIGH:
            return min(base_workers, 6)
        elif task_priority == TaskPriority.LOW:
            return min(base_workers // 2, 4)
        else:
            return min(base_workers, 4)

class TaskQueue:
    """Priority-based task queue with ARM64 optimizations"""
    
    def __init__(self, arm64_optimizer: ARM64SchedulerOptimizer):
        self.optimizer = arm64_optimizer
        self.queue = []
        self.queue_lock = threading.Lock()
        self.condition = threading.Condition(self.queue_lock)
        
    def put(self, task_execution: TaskExecution):
        """Add task to priority queue"""
        with self.condition:
            # Use negative priority for max-heap behavior
            priority = -task_execution.definition.priority.value
            heapq.heappush(self.queue, (priority, task_execution.scheduled_time, task_execution))
            self.condition.notify()
    
    def get(self, timeout: Optional[float] = None) -> Optional[TaskExecution]:
        """Get highest priority task from queue"""
        with self.condition:
            while not self.queue:
                if not self.condition.wait(timeout):
                    return None
            
            _, _, task_execution = heapq.heappop(self.queue)
            return task_execution
    
    def peek(self) -> Optional[TaskExecution]:
        """Peek at next task without removing it"""
        with self.queue_lock:
            if self.queue:
                return self.queue[0][2]
            return None
    
    def size(self) -> int:
        """Get queue size"""
        with self.queue_lock:
            return len(self.queue)
    
    def clear(self):
        """Clear all tasks from queue"""
        with self.condition:
            self.queue.clear()
            self.condition.notify_all()

class TaskExecutor:
    """Task executor with ARM64 optimizations"""
    
    def __init__(self, arm64_optimizer: ARM64SchedulerOptimizer, config: SchedulerConfig):
        self.optimizer = arm64_optimizer
        self.config = config
        self.thread_executor = None
        self.process_executor = None
        self.active_tasks = {}
        self.task_lock = threading.Lock()
        
    def start(self):
        """Start task executors"""
        max_workers = self.config.max_workers or self.optimizer.calculate_optimal_workers(TaskPriority.NORMAL)
        
        self.thread_executor = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="TaskExecutor"
        )
        
        if self.config.parallel_processing:
            self.process_executor = ProcessPoolExecutor(
                max_workers=min(max_workers, mp.cpu_count())
            )
    
    def stop(self):
        """Stop task executors"""
        if self.thread_executor:
            self.thread_executor.shutdown(wait=True)
        
        if self.process_executor:
            self.process_executor.shutdown(wait=True)
    
    @performance_monitor
    @error_handler
    def execute_task(self, task_execution: TaskExecution) -> TaskExecution:
        """Execute a single task"""
        task_execution.status = TaskStatus.RUNNING
        task_execution.start_time = time.time()
        
        try:
            with self.task_lock:
                self.active_tasks[task_execution.execution_id] = task_execution
            
            # Determine execution method
            if callable(task_execution.definition.func):
                # Execute function
                if asyncio.iscoroutinefunction(task_execution.definition.func):
                    # Async function
                    result = asyncio.run(task_execution.definition.func(
                        *task_execution.definition.args,
                        **task_execution.definition.kwargs
                    ))
                else:
                    # Sync function
                    result = task_execution.definition.func(
                        *task_execution.definition.args,
                        **task_execution.definition.kwargs
                    )
            else:
                # Execute command
                import subprocess
                result = subprocess.run(
                    task_execution.definition.func,
                    shell=True,
                    capture_output=True,
                    text=True,
                    timeout=task_execution.definition.timeout or self.config.task_timeout
                )
                result = {
                    'returncode': result.returncode,
                    'stdout': result.stdout,
                    'stderr': result.stderr
                }
            
            task_execution.result = result
            task_execution.status = TaskStatus.COMPLETED
            
        except Exception as e:
            task_execution.error = str(e)
            task_execution.status = TaskStatus.FAILED
            logger.error(f"Task {task_execution.task_id} failed: {e}")
        
        finally:
            task_execution.end_time = time.time()
            task_execution.execution_time = task_execution.end_time - task_execution.start_time
            
            with self.task_lock:
                if task_execution.execution_id in self.active_tasks:
                    del self.active_tasks[task_execution.execution_id]
        
        return task_execution
    
    def submit_task(self, task_execution: TaskExecution) -> Future:
        """Submit task for execution"""
        executor = self.thread_executor
        
        # Use process executor for CPU-intensive tasks
        if (self.process_executor and 
            task_execution.definition.priority in [TaskPriority.HIGH, TaskPriority.CRITICAL]):
            executor = self.process_executor
        
        return executor.submit(self.execute_task, task_execution)
    
    def get_active_tasks(self) -> Dict[str, TaskExecution]:
        """Get currently active tasks"""
        with self.task_lock:
            return self.active_tasks.copy()

class ScheduleManager:
    """Manages task schedules and triggers"""
    
    def __init__(self):
        self.schedules = {}
        self.next_runs = {}
        
    def add_schedule(self, task_def: TaskDefinition) -> bool:
        """Add task schedule"""
        try:
            if task_def.schedule_type == ScheduleType.ONCE:
                # One-time execution
                if isinstance(task_def.schedule_value, (int, float)):
                    self.next_runs[task_def.id] = task_def.schedule_value
                else:
                    self.next_runs[task_def.id] = time.time()
                    
            elif task_def.schedule_type == ScheduleType.INTERVAL:
                # Interval-based execution
                interval = task_def.schedule_value or 60  # Default 1 minute
                self.next_runs[task_def.id] = time.time() + interval
                
            elif task_def.schedule_type == ScheduleType.CRON:
                # Cron-based execution
                if task_def.schedule_value:
                    cron = croniter.croniter(task_def.schedule_value, time.time())
                    self.next_runs[task_def.id] = cron.get_next()
                else:
                    return False
                    
            elif task_def.schedule_type == ScheduleType.DAILY:
                # Daily execution at specified time
                target_time = task_def.schedule_value or "00:00"
                hour, minute = map(int, target_time.split(':'))
                
                now = datetime.now()
                next_run = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
                
                if next_run <= now:
                    next_run += timedelta(days=1)
                
                self.next_runs[task_def.id] = next_run.timestamp()
                
            self.schedules[task_def.id] = task_def
            return True
            
        except Exception as e:
            logger.error(f"Error adding schedule for task {task_def.id}: {e}")
            return False
    
    def remove_schedule(self, task_id: str):
        """Remove task schedule"""
        if task_id in self.schedules:
            del self.schedules[task_id]
        if task_id in self.next_runs:
            del self.next_runs[task_id]
    
    def get_ready_tasks(self) -> List[TaskDefinition]:
        """Get tasks ready for execution"""
        ready_tasks = []
        current_time = time.time()
        
        for task_id, next_run_time in list(self.next_runs.items()):
            if current_time >= next_run_time:
                if task_id in self.schedules:
                    task_def = self.schedules[task_id]
                    ready_tasks.append(task_def)
                    
                    # Update next run time for recurring tasks
                    self._update_next_run(task_def)
        
        return ready_tasks
    
    def _update_next_run(self, task_def: TaskDefinition):
        """Update next run time for recurring tasks"""
        if task_def.schedule_type == ScheduleType.ONCE:
            # Remove one-time tasks
            self.remove_schedule(task_def.id)
            
        elif task_def.schedule_type == ScheduleType.INTERVAL:
            interval = task_def.schedule_value or 60
            self.next_runs[task_def.id] = time.time() + interval
            
        elif task_def.schedule_type == ScheduleType.CRON:
            if task_def.schedule_value:
                cron = croniter.croniter(task_def.schedule_value, time.time())
                self.next_runs[task_def.id] = cron.get_next()
                
        elif task_def.schedule_type == ScheduleType.DAILY:
            # Add one day
            self.next_runs[task_def.id] += 24 * 3600

class AdvancedScheduler:
    """
    Advanced Task Scheduler with ARM64 optimizations
    
    Provides comprehensive task scheduling capabilities including:
    - Priority-based task queuing
    - Multiple scheduling patterns (cron, interval, daily)
    - Task dependency management
    - Fault tolerance and retry mechanisms
    - Performance monitoring and optimization
    """
    
    def __init__(self, config: SchedulerConfig = None):
        self.config = config or SchedulerConfig()
        self.arm64_optimizer = ARM64SchedulerOptimizer()
        self.shared_memory = SharedMemoryManager() if self.config.use_shared_memory else None
        
        # Core components
        self.task_queue = TaskQueue(self.arm64_optimizer)
        self.task_executor = TaskExecutor(self.arm64_optimizer, self.config)
        self.schedule_manager = ScheduleManager()
        
        # Task management
        self.task_definitions = {}
        self.task_executions = {}
        self.execution_history = deque(maxlen=10000)
        
        # Threading
        self.is_running = False
        self.scheduler_thread = None
        self.cleanup_thread = None
        
        # Performance tracking
        self.scheduler_stats = {
            'total_tasks_scheduled': 0,
            'total_tasks_executed': 0,
            'successful_executions': 0,
            'failed_executions': 0,
            'average_execution_time': 0.0,
            'queue_size': 0
        }
        
        logger.info(f"AdvancedScheduler initialized with ARM64 optimizations: {self.arm64_optimizer.is_arm64}")
    
    def add_task(self, task_def: TaskDefinition) -> bool:
        """Add task definition to scheduler"""
        try:
            if not task_def.id:
                task_def.id = str(uuid.uuid4())
            
            self.task_definitions[task_def.id] = task_def
            
            # Add to schedule manager
            if not self.schedule_manager.add_schedule(task_def):
                logger.error(f"Failed to add schedule for task {task_def.id}")
                return False
            
            logger.info(f"Added task: {task_def.name} ({task_def.id})")
            return True
            
        except Exception as e:
            logger.error(f"Error adding task {task_def.name}: {e}")
            return False
    
    def remove_task(self, task_id: str) -> bool:
        """Remove task from scheduler"""
        try:
            if task_id in self.task_definitions:
                del self.task_definitions[task_id]
            
            self.schedule_manager.remove_schedule(task_id)
            
            logger.info(f"Removed task: {task_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error removing task {task_id}: {e}")
            return False
    
    def schedule_task_now(self, task_id: str) -> Optional[str]:
        """Schedule task for immediate execution"""
        if task_id not in self.task_definitions:
            logger.error(f"Task {task_id} not found")
            return None
        
        task_def = self.task_definitions[task_id]
        execution_id = str(uuid.uuid4())
        
        task_execution = TaskExecution(
            task_id=task_id,
            execution_id=execution_id,
            definition=task_def,
            scheduled_time=time.time()
        )
        
        self.task_queue.put(task_execution)
        self.task_executions[execution_id] = task_execution
        self.scheduler_stats['total_tasks_scheduled'] += 1
        
        logger.info(f"Scheduled task {task_def.name} for immediate execution")
        return execution_id
    
    def get_task_status(self, execution_id: str) -> Optional[TaskExecution]:
        """Get task execution status"""
        return self.task_executions.get(execution_id)
    
    def get_task_history(self, task_id: Optional[str] = None, limit: int = 100) -> List[TaskExecution]:
        """Get task execution history"""
        history = list(self.execution_history)
        
        if task_id:
            history = [exec for exec in history if exec.task_id == task_id]
        
        return history[-limit:]
    
    def cancel_task(self, execution_id: str) -> bool:
        """Cancel task execution"""
        if execution_id in self.task_executions:
            task_execution = self.task_executions[execution_id]
            if task_execution.status == TaskStatus.PENDING:
                task_execution.status = TaskStatus.CANCELLED
                logger.info(f"Cancelled task execution {execution_id}")
                return True
        
        return False
    
    def _scheduler_loop(self):
        """Main scheduler loop"""
        while self.is_running:
            try:
                # Check for ready tasks
                ready_tasks = self.schedule_manager.get_ready_tasks()
                
                for task_def in ready_tasks:
                    # Check dependencies
                    if self._check_dependencies(task_def):
                        execution_id = str(uuid.uuid4())
                        
                        task_execution = TaskExecution(
                            task_id=task_def.id,
                            execution_id=execution_id,
                            definition=task_def,
                            scheduled_time=time.time()
                        )
                        
                        self.task_queue.put(task_execution)
                        self.task_executions[execution_id] = task_execution
                        self.scheduler_stats['total_tasks_scheduled'] += 1
                
                # Process task queue
                self._process_task_queue()
                
                # Update queue size stat
                self.scheduler_stats['queue_size'] = self.task_queue.size()
                
                time.sleep(1.0)  # Check every second
                
            except Exception as e:
                logger.error(f"Error in scheduler loop: {e}")
                time.sleep(5.0)
    
    def _process_task_queue(self):
        """Process tasks from queue"""
        while self.task_queue.size() > 0:
            # Check if we're at max concurrent tasks
            active_tasks = self.task_executor.get_active_tasks()
            if len(active_tasks) >= self.config.max_concurrent_tasks:
                break
            
            task_execution = self.task_queue.get(timeout=0.1)
            if not task_execution:
                break
            
            # Submit task for execution
            future = self.task_executor.submit_task(task_execution)
            
            # Add callback to handle completion
            future.add_done_callback(lambda f, te=task_execution: self._task_completed(f, te))
    
    def _task_completed(self, future: Future, task_execution: TaskExecution):
        """Handle task completion"""
        try:
            completed_execution = future.result()
            
            # Update statistics
            self.scheduler_stats['total_tasks_executed'] += 1
            
            if completed_execution.status == TaskStatus.COMPLETED:
                self.scheduler_stats['successful_executions'] += 1
            elif completed_execution.status == TaskStatus.FAILED:
                self.scheduler_stats['failed_executions'] += 1
                
                # Handle retry logic
                if (completed_execution.retry_count < completed_execution.definition.max_retries):
                    self._retry_task(completed_execution)
                    return
            
            # Update average execution time
            if completed_execution.execution_time:
                current_avg = self.scheduler_stats['average_execution_time']
                total_executed = self.scheduler_stats['total_tasks_executed']
                
                self.scheduler_stats['average_execution_time'] = (
                    (current_avg * (total_executed - 1) + completed_execution.execution_time) / total_executed
                )
            
            # Add to history
            self.execution_history.append(completed_execution)
            
            logger.info(f"Task {completed_execution.definition.name} completed with status {completed_execution.status.value}")
            
        except Exception as e:
            logger.error(f"Error handling task completion: {e}")
    
    def _retry_task(self, task_execution: TaskExecution):
        """Retry failed task"""
        task_execution.retry_count += 1
        task_execution.status = TaskStatus.RETRYING
        
        # Calculate retry delay with exponential backoff
        delay = self.config.retry_delay * (2 ** (task_execution.retry_count - 1))
        
        # Schedule retry
        retry_execution = TaskExecution(
            task_id=task_execution.task_id,
            execution_id=str(uuid.uuid4()),
            definition=task_execution.definition,
            scheduled_time=time.time() + delay,
            retry_count=task_execution.retry_count
        )
        
        self.task_queue.put(retry_execution)
        self.task_executions[retry_execution.execution_id] = retry_execution
        
        logger.info(f"Retrying task {task_execution.definition.name} (attempt {task_execution.retry_count})")
    
    def _check_dependencies(self, task_def: TaskDefinition) -> bool:
        """Check if task dependencies are satisfied"""
        for dep_task_id in task_def.dependencies:
            # Check if dependency task has completed successfully recently
            recent_executions = [
                exec for exec in self.execution_history
                if exec.task_id == dep_task_id and 
                   exec.status == TaskStatus.COMPLETED and
                   time.time() - exec.end_time < 3600  # Within last hour
            ]
            
            if not recent_executions:
                logger.debug(f"Dependency {dep_task_id} not satisfied for task {task_def.id}")
                return False
        
        return True
    
    def _cleanup_loop(self):
        """Cleanup old task executions and history"""
        while self.is_running:
            try:
                current_time = time.time()
                
                # Clean up old task executions
                expired_executions = []
                for execution_id, task_execution in self.task_executions.items():
                    if (task_execution.end_time and 
                        current_time - task_execution.end_time > 3600):  # 1 hour old
                        expired_executions.append(execution_id)
                
                for execution_id in expired_executions:
                    del self.task_executions[execution_id]
                
                if expired_executions:
                    logger.debug(f"Cleaned up {len(expired_executions)} old task executions")
                
                time.sleep(self.config.cleanup_interval)
                
            except Exception as e:
                logger.error(f"Error in cleanup loop: {e}")
                time.sleep(300)  # 5 minutes
    
    def start(self) -> bool:
        """Start the scheduler"""
        if self.is_running:
            return True
        
        try:
            self.is_running = True
            
            # Start task executor
            self.task_executor.start()
            
            # Start scheduler thread
            self.scheduler_thread = threading.Thread(
                target=self._scheduler_loop,
                name="SchedulerThread"
            )
            self.scheduler_thread.daemon = True
            self.scheduler_thread.start()
            
            # Start cleanup thread
            self.cleanup_thread = threading.Thread(
                target=self._cleanup_loop,
                name="SchedulerCleanupThread"
            )
            self.cleanup_thread.daemon = True
            self.cleanup_thread.start()
            
            logger.info("Scheduler started successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start scheduler: {e}")
            self.is_running = False
            return False
    
    def stop(self):
        """Stop the scheduler"""
        self.is_running = False
        
        # Clear task queue
        self.task_queue.clear()
        
        # Stop task executor
        self.task_executor.stop()
        
        # Wait for threads
        if self.scheduler_thread:
            self.scheduler_thread.join(timeout=5.0)
        
        if self.cleanup_thread:
            self.cleanup_thread.join(timeout=5.0)
        
        logger.info("Scheduler stopped")
    
    def get_status(self) -> Dict[str, Any]:
        """Get scheduler status"""
        active_tasks = self.task_executor.get_active_tasks()
        
        return {
            'is_running': self.is_running,
            'total_tasks': len(self.task_definitions),
            'active_tasks': len(active_tasks),
            'queue_size': self.task_queue.size(),
            'statistics': self.scheduler_stats,
            'arm64_optimized': self.arm64_optimizer.is_arm64,
            'cpu_topology': self.arm64_optimizer.cpu_topology
        }
    
    def generate_scheduler_report(self) -> Dict[str, Any]:
        """Generate comprehensive scheduler report"""
        return {
            'status': self.get_status(),
            'task_definitions': {
                task_id: {
                    'name': task_def.name,
                    'priority': task_def.priority.value,
                    'schedule_type': task_def.schedule_type.value,
                    'schedule_value': task_def.schedule_value,
                    'dependencies': task_def.dependencies,
                    'tags': task_def.tags
                }
                for task_id, task_def in self.task_definitions.items()
            },
            'recent_executions': [
                {
                    'task_id': exec.task_id,
                    'execution_id': exec.execution_id,
                    'status': exec.status.value,
                    'execution_time': exec.execution_time,
                    'scheduled_time': exec.scheduled_time,
                    'start_time': exec.start_time,
                    'end_time': exec.end_time
                }
                for exec in list(self.execution_history)[-50:]  # Last 50 executions
            ],
            'system_info': {
                'platform': platform.platform(),
                'cpu_count': mp.cpu_count(),
                'is_arm64': self.arm64_optimizer.is_arm64
            }
        }
    
    def cleanup(self):
        """Cleanup scheduler resources"""
        if self.is_running:
            self.stop()
        
        if self.shared_memory:
            self.shared_memory.cleanup()
        
        logger.info("Scheduler cleanup completed")

# Factory function for easy instantiation
def create_scheduler(config: SchedulerConfig = None) -> AdvancedScheduler:
    """
    Factory function to create scheduler with optimal configuration
    
    Args:
        config: Scheduler configuration
        
    Returns:
        Configured AdvancedScheduler instance
    """
    if config is None:
        config = SchedulerConfig()
        
        # Auto-detect optimal settings
        if platform.machine().lower() in ['arm64', 'aarch64']:
            config.enable_arm64_optimizations = True
            config.parallel_processing = True
            config.max_workers = min(mp.cpu_count(), 8)
        
        # Adjust for available resources
        cpu_count = mp.cpu_count()
        if cpu_count > 8:
            config.max_concurrent_tasks = 200
        elif cpu_count > 4:
            config.max_concurrent_tasks = 100
        else:
            config.max_concurrent_tasks = 50
    
    return AdvancedScheduler(config)

# Legacy compatibility
Scheduler = AdvancedScheduler

if __name__ == "__main__":
    # Example usage and testing
    
    def sample_task(name: str, duration: int = 2):
        """Sample task function"""
        print(f"Task {name} starting...")
        time.sleep(duration)
        print(f"Task {name} completed")
        return f"Result from {name}"
    
    # Create scheduler
    scheduler = create_scheduler()
    
    # Add sample tasks
    task_def = TaskDefinition(
        id="sample_task_1",
        name="Sample Task 1",
        func=sample_task,
        args=("test_task", 3),
        priority=TaskPriority.NORMAL,
        schedule_type=ScheduleType.INTERVAL,
        schedule_value=10  # Every 10 seconds
    )
    
    scheduler.add_task(task_def)
    
    # Start scheduler
    if scheduler.start():
        print("Scheduler started successfully")
        
        # Schedule a task immediately
        execution_id = scheduler.schedule_task_now("sample_task_1")
        if execution_id:
            print(f"Task scheduled with execution ID: {execution_id}")
            
            # Monitor task status
            time.sleep(1)
            status = scheduler.get_task_status(execution_id)
            if status:
                print(f"Task status: {status.status.value}")
        
        # Get scheduler status
        status = scheduler.get_status()
        print(f"Scheduler status: {status}")
        
        # Generate report
        report = scheduler.generate_scheduler_report()
        print(f"Scheduler report generated with {len(report)} sections")
        
        # Let it run for a bit
        time.sleep(5)
        
        # Cleanup
        scheduler.stop()
        scheduler.cleanup()
    else:
        print("Failed to start scheduler")
