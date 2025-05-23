# Communication Package - ARM64 Optimized ZeroMQ System
# Implements high-performance communication architecture for Deep Momentum Trading
# Optimized for NVIDIA GH200 ARM64 platform with sub-millisecond latency

# Core messaging components
from .event_dispatcher import EventDispatcher, Event, EventPriority
from .message_broker import MessageBroker
from .zmq_publisher import ZMQPublisher, MarketDataMessage, TradingSignalMessage
from .zmq_subscriber import ZMQSubscriber

# ARM64-specific modules (to be implemented)
try:
    from .arm64_data_publisher import ARM64DataPublisher
    from .torchscript_subscriber import TorchScriptModelSubscriber 
    from .optuna_coordinator import OptunaCoordinator
    from .ray_tune_worker import RayTuneWorker
    from .arm64_zmq_config import ARM64ZMQConfig
    from .arm64_serialization import ARM64MessageSerializer
    from .arm64_connection_manager import ARM64ConnectionManager
    from .arm64_comm_monitor import ARM64CommunicationMonitor
    ARM64_MODULES_AVAILABLE = True
except ImportError:
    ARM64_MODULES_AVAILABLE = False

__all__ = [
    'EventDispatcher', 'Event', 'EventPriority',
    'MessageBroker', 
    'ZMQPublisher', 'MarketDataMessage', 'TradingSignalMessage',
    'ZMQSubscriber'
]

# Add ARM64 modules to exports if available
if ARM64_MODULES_AVAILABLE:
    __all__.extend([
        'ARM64DataPublisher',
        'TorchScriptModelSubscriber', 
        'OptunaCoordinator',
        'RayTuneWorker',
        'ARM64ZMQConfig',
        'ARM64MessageSerializer',
        'ARM64ConnectionManager',
        'ARM64CommunicationMonitor'
    ])

# Version and platform info
__version__ = '1.0.0'
__platform__ = 'ARM64-optimized'

# ARM64 optimization status
import platform
IS_ARM64 = platform.machine().lower() in ['arm64', 'aarch64']
ARM64_OPTIMIZATIONS_ENABLED = IS_ARM64 and ARM64_MODULES_AVAILABLE
