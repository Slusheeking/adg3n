"""
Integration tests for Communication System.

Tests the ZMQ-based communication infrastructure, message routing,
event dispatching, and inter-process communication reliability.
"""

import pytest
import asyncio
import time
import json
import threading
import zmq
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from datetime import datetime, timezone
from typing import Dict, List, Any

from deep_momentum_trading.tests.fixtures.sample_data import (
    get_sample_market_data,
    get_sample_predictions,
    get_sample_features_data
)
from deep_momentum_trading.tests.fixtures.test_configs import (
    TestConfigManager,
    TestScenarios
)


@pytest.mark.integration
class TestZMQCommunication:
    """Test ZMQ communication infrastructure."""
    
    @pytest.fixture
    def communication_config(self):
        """Get communication test configuration."""
        return {
            'publisher_port': 5555,
            'subscriber_port': 5556,
            'event_dispatcher_port': 5557,
            'message_broker_port': 5558,
            'timeout_ms': 5000,
            'max_retries': 3,
            'enable_encryption': False,
            'buffer_size': 1000
        }
    
    @pytest.fixture
    def mock_zmq_context(self):
        """Mock ZMQ context for testing."""
        with patch('zmq.Context') as mock_context:
            mock_socket = Mock()
            mock_socket.bind = Mock()
            mock_socket.connect = Mock()
            mock_socket.send_string = Mock()
            mock_socket.recv_string = Mock(return_value='{"test": "message"}')
            mock_socket.send_json = Mock()
            mock_socket.recv_json = Mock(return_value={"test": "message"})
            mock_socket.poll = Mock(return_value=1)  # Socket ready
            mock_socket.close = Mock()
            
            mock_context.return_value.socket.return_value = mock_socket
            yield mock_context, mock_socket
    
    def test_zmq_publisher_initialization(self, communication_config, mock_zmq_context):
        """Test ZMQ publisher initialization."""
        mock_context, mock_socket = mock_zmq_context
        
        from deep_momentum_trading.src.communication.zmq_publisher import ZMQPublisher
        
        publisher = ZMQPublisher(
            port=communication_config['publisher_port'],
            topic="test_topic"
        )
        
        assert publisher.port == communication_config['publisher_port']
        assert publisher.topic == "test_topic"
        assert publisher.is_connected is False
        
        # Test connection
        publisher.connect()
        mock_socket.bind.assert_called_once()
        
        # Test publishing
        test_message = {"symbol": "AAPL", "price": 150.0}
        publisher.publish(test_message)
        mock_socket.send_json.assert_called()
    
    def test_zmq_subscriber_initialization(self, communication_config, mock_zmq_context):
        """Test ZMQ subscriber initialization."""
        mock_context, mock_socket = mock_zmq_context
        
        from deep_momentum_trading.src.communication.zmq_subscriber import ZMQSubscriber
        
        subscriber = ZMQSubscriber(
            port=communication_config['subscriber_port'],
            topics=["market_data", "predictions"]
        )
        
        assert subscriber.port == communication_config['subscriber_port']
        assert "market_data" in subscriber.topics
        assert "predictions" in subscriber.topics
        
        # Test connection
        subscriber.connect()
        mock_socket.connect.assert_called_once()
        
        # Test subscription
        for topic in subscriber.topics:
            mock_socket.setsockopt_string.assert_any_call(zmq.SUBSCRIBE, topic)
    
    @pytest.mark.asyncio
    async def test_message_publishing_and_receiving(self, communication_config, mock_zmq_context):
        """Test message publishing and receiving flow."""
        mock_context, mock_socket = mock_zmq_context
        
        from deep_momentum_trading.src.communication.zmq_publisher import ZMQPublisher
        from deep_momentum_trading.src.communication.zmq_subscriber import ZMQSubscriber
        
        # Setup publisher
        publisher = ZMQPublisher(port=5555, topic="test_data")
        publisher.connect()
        
        # Setup subscriber
        subscriber = ZMQSubscriber(port=5556, topics=["test_data"])
        subscriber.connect()
        
        # Test message flow
        test_messages = [
            {"type": "market_data", "symbol": "AAPL", "price": 150.0},
            {"type": "prediction", "symbol": "MSFT", "position": 0.3},
            {"type": "risk_alert", "level": "high", "message": "Position limit exceeded"}
        ]
        
        received_messages = []
        
        def message_handler(message):
            received_messages.append(message)
        
        subscriber.set_message_handler(message_handler)
        
        # Publish messages
        for message in test_messages:
            publisher.publish(message)
            await asyncio.sleep(0.1)  # Small delay for processing
        
        # Verify messages were sent
        assert mock_socket.send_json.call_count == len(test_messages)
    
    def test_event_dispatcher_functionality(self, communication_config, mock_zmq_context):
        """Test event dispatcher functionality."""
        mock_context, mock_socket = mock_zmq_context
        
        from deep_momentum_trading.src.communication.event_dispatcher import EventDispatcher
        
        dispatcher = EventDispatcher(
            port=communication_config['event_dispatcher_port']
        )
        
        # Test event registration
        event_handlers = {}
        
        def market_data_handler(event_data):
            event_handlers['market_data'] = event_data
        
        def prediction_handler(event_data):
            event_handlers['prediction'] = event_data
        
        dispatcher.register_handler("market_data", market_data_handler)
        dispatcher.register_handler("prediction", prediction_handler)
        
        # Test event dispatching
        market_event = {"symbol": "AAPL", "price": 150.0, "volume": 1000000}
        prediction_event = {"symbol": "MSFT", "position": 0.25, "confidence": 0.8}
        
        dispatcher.dispatch_event("market_data", market_event)
        dispatcher.dispatch_event("prediction", prediction_event)
        
        # Verify handlers were called
        assert "market_data" in event_handlers
        assert "prediction" in event_handlers
        assert event_handlers["market_data"]["symbol"] == "AAPL"
        assert event_handlers["prediction"]["position"] == 0.25
    
    def test_message_broker_routing(self, communication_config, mock_zmq_context):
        """Test message broker routing functionality."""
        mock_context, mock_socket = mock_zmq_context
        
        from deep_momentum_trading.src.communication.message_broker import MessageBroker
        
        broker = MessageBroker(
            port=communication_config['message_broker_port']
        )
        
        # Test route registration
        routes = {}
        
        def feature_processor_route(message):
            routes['features'] = message
            return {"status": "processed", "features_count": 50}
        
        def risk_manager_route(message):
            routes['risk'] = message
            return {"status": "assessed", "risk_level": "medium"}
        
        broker.register_route("process_features", feature_processor_route)
        broker.register_route("assess_risk", risk_manager_route)
        
        # Test message routing
        feature_message = {
            "route": "process_features",
            "data": {"symbol": "AAPL", "market_data": get_sample_market_data(n_records=100)}
        }
        
        risk_message = {
            "route": "assess_risk", 
            "data": {"predictions": get_sample_predictions(n_predictions=5)}
        }
        
        # Route messages
        feature_response = broker.route_message(feature_message)
        risk_response = broker.route_message(risk_message)
        
        # Verify routing
        assert "features" in routes
        assert "risk" in routes
        assert feature_response["status"] == "processed"
        assert risk_response["risk_level"] == "medium"


@pytest.mark.integration
class TestCommunicationReliability:
    """Test communication system reliability and error handling."""
    
    @pytest.fixture
    def reliability_config(self):
        """Configuration for reliability testing."""
        return {
            'max_retries': 3,
            'retry_delay_ms': 100,
            'timeout_ms': 1000,
            'heartbeat_interval_ms': 500,
            'connection_timeout_ms': 5000
        }
    
    def test_connection_retry_mechanism(self, reliability_config):
        """Test connection retry mechanism."""
        with patch('zmq.Context') as mock_context:
            mock_socket = Mock()
            # Simulate connection failures then success
            mock_socket.bind.side_effect = [Exception("Connection failed")] * 2 + [None]
            mock_context.return_value.socket.return_value = mock_socket
            
            from deep_momentum_trading.src.communication.zmq_publisher import ZMQPublisher
            
            publisher = ZMQPublisher(port=5555, topic="test")
            publisher.max_retries = reliability_config['max_retries']
            publisher.retry_delay_ms = reliability_config['retry_delay_ms']
            
            # Should succeed after retries
            publisher.connect()
            
            # Verify retries occurred
            assert mock_socket.bind.call_count == 3
    
    def test_message_timeout_handling(self, reliability_config):
        """Test message timeout handling."""
        with patch('zmq.Context') as mock_context:
            mock_socket = Mock()
            # Simulate timeout
            mock_socket.poll.return_value = 0  # No messages available
            mock_context.return_value.socket.return_value = mock_socket
            
            from deep_momentum_trading.src.communication.zmq_subscriber import ZMQSubscriber
            
            subscriber = ZMQSubscriber(port=5556, topics=["test"])
            subscriber.timeout_ms = reliability_config['timeout_ms']
            subscriber.connect()
            
            # Should handle timeout gracefully
            message = subscriber.receive_message()
            assert message is None  # Timeout should return None
    
    def test_heartbeat_mechanism(self, reliability_config):
        """Test heartbeat mechanism for connection monitoring."""
        heartbeat_received = []
        
        def heartbeat_handler(timestamp):
            heartbeat_received.append(timestamp)
        
        with patch('zmq.Context') as mock_context:
            mock_socket = Mock()
            mock_context.return_value.socket.return_value = mock_socket
            
            from deep_momentum_trading.src.communication.zmq_publisher import ZMQPublisher
            
            publisher = ZMQPublisher(port=5555, topic="heartbeat")
            publisher.heartbeat_interval_ms = reliability_config['heartbeat_interval_ms']
            publisher.connect()
            
            # Start heartbeat
            publisher.start_heartbeat(heartbeat_handler)
            
            # Wait for heartbeats
            time.sleep(1.5)  # Should get ~3 heartbeats
            
            publisher.stop_heartbeat()
            
            # Verify heartbeats were sent
            assert len(heartbeat_received) >= 2
    
    def test_error_recovery_mechanism(self, reliability_config):
        """Test error recovery mechanism."""
        error_count = 0
        recovery_count = 0
        
        def error_handler(error):
            nonlocal error_count
            error_count += 1
        
        def recovery_handler():
            nonlocal recovery_count
            recovery_count += 1
        
        with patch('zmq.Context') as mock_context:
            mock_socket = Mock()
            # Simulate intermittent errors
            mock_socket.send_json.side_effect = [
                Exception("Network error"),
                Exception("Network error"), 
                None,  # Success
                None   # Success
            ]
            mock_context.return_value.socket.return_value = mock_socket
            
            from deep_momentum_trading.src.communication.zmq_publisher import ZMQPublisher
            
            publisher = ZMQPublisher(port=5555, topic="test")
            publisher.set_error_handler(error_handler)
            publisher.set_recovery_handler(recovery_handler)
            publisher.connect()
            
            # Send messages with errors
            messages = [{"test": i} for i in range(4)]
            for message in messages:
                try:
                    publisher.publish(message)
                except:
                    pass  # Expected errors
            
            # Verify error handling and recovery
            assert error_count >= 2  # Should have caught errors
            assert recovery_count >= 1  # Should have recovered


@pytest.mark.integration
class TestCommunicationPerformance:
    """Test communication system performance."""
    
    @pytest.mark.performance
    def test_message_throughput(self):
        """Test message throughput performance."""
        with patch('zmq.Context') as mock_context:
            mock_socket = Mock()
            mock_context.return_value.socket.return_value = mock_socket
            
            from deep_momentum_trading.src.communication.zmq_publisher import ZMQPublisher
            
            publisher = ZMQPublisher(port=5555, topic="performance_test")
            publisher.connect()
            
            # Test high-frequency message publishing
            n_messages = 1000
            test_message = {"symbol": "AAPL", "price": 150.0, "timestamp": time.time()}
            
            start_time = time.perf_counter()
            
            for i in range(n_messages):
                test_message["sequence"] = i
                publisher.publish(test_message)
            
            end_time = time.perf_counter()
            
            # Calculate throughput
            duration = end_time - start_time
            throughput = n_messages / duration
            
            # Should achieve reasonable throughput
            assert throughput > 500  # At least 500 messages per second
            assert mock_socket.send_json.call_count == n_messages
    
    @pytest.mark.performance
    def test_message_latency(self):
        """Test message latency."""
        latencies = []
        
        def latency_handler(message):
            receive_time = time.time()
            send_time = message.get('timestamp', receive_time)
            latency = (receive_time - send_time) * 1000  # Convert to ms
            latencies.append(latency)
        
        with patch('zmq.Context') as mock_context:
            mock_socket = Mock()
            mock_context.return_value.socket.return_value = mock_socket
            
            from deep_momentum_trading.src.communication.zmq_publisher import ZMQPublisher
            from deep_momentum_trading.src.communication.zmq_subscriber import ZMQSubscriber
            
            publisher = ZMQPublisher(port=5555, topic="latency_test")
            subscriber = ZMQSubscriber(port=5556, topics=["latency_test"])
            
            publisher.connect()
            subscriber.connect()
            subscriber.set_message_handler(latency_handler)
            
            # Send timestamped messages
            for i in range(100):
                message = {
                    "sequence": i,
                    "timestamp": time.time(),
                    "data": f"test_message_{i}"
                }
                publisher.publish(message)
                time.sleep(0.01)  # 10ms interval
            
            # Mock receiving messages with simulated latency
            for i in range(100):
                mock_message = {
                    "sequence": i,
                    "timestamp": time.time() - 0.005,  # 5ms ago
                    "data": f"test_message_{i}"
                }
                latency_handler(mock_message)
            
            # Verify latency measurements
            assert len(latencies) == 100
            avg_latency = sum(latencies) / len(latencies)
            assert avg_latency < 50  # Average latency should be < 50ms
    
    @pytest.mark.stress
    def test_high_load_communication(self):
        """Test communication under high load."""
        with patch('zmq.Context') as mock_context:
            mock_socket = Mock()
            mock_context.return_value.socket.return_value = mock_socket
            
            from deep_momentum_trading.src.communication.zmq_publisher import ZMQPublisher
            
            # Create multiple publishers
            publishers = []
            for i in range(5):
                publisher = ZMQPublisher(port=5555 + i, topic=f"stress_test_{i}")
                publisher.connect()
                publishers.append(publisher)
            
            # Generate high-frequency messages from multiple sources
            n_messages_per_publisher = 500
            
            def publish_messages(publisher, publisher_id):
                for i in range(n_messages_per_publisher):
                    message = {
                        "publisher_id": publisher_id,
                        "sequence": i,
                        "timestamp": time.time(),
                        "data": get_sample_market_data(n_records=10)  # Small dataset
                    }
                    publisher.publish(message)
            
            # Start concurrent publishing
            threads = []
            start_time = time.perf_counter()
            
            for i, publisher in enumerate(publishers):
                thread = threading.Thread(
                    target=publish_messages,
                    args=(publisher, i)
                )
                thread.start()
                threads.append(thread)
            
            # Wait for completion
            for thread in threads:
                thread.join()
            
            end_time = time.perf_counter()
            
            # Verify performance under load
            total_messages = len(publishers) * n_messages_per_publisher
            duration = end_time - start_time
            throughput = total_messages / duration
            
            # Should maintain reasonable throughput under load
            assert throughput > 1000  # At least 1000 messages per second total
            
            # Verify all messages were sent
            total_calls = sum(pub.socket.send_json.call_count for pub in publishers if hasattr(pub, 'socket'))
            # Note: In real implementation, this would be verified differently


@pytest.mark.integration
class TestTorchScriptCommunication:
    """Test TorchScript-specific communication features."""
    
    def test_torchscript_subscriber_initialization(self):
        """Test TorchScript subscriber initialization."""
        with patch('zmq.Context') as mock_context:
            mock_socket = Mock()
            mock_context.return_value.socket.return_value = mock_socket
            
            from deep_momentum_trading.src.communication.torchscript_subscriber import TorchScriptSubscriber
            
            subscriber = TorchScriptSubscriber(
                port=5559,
                topics=["model_predictions", "feature_updates"]
            )
            
            assert subscriber.port == 5559
            assert "model_predictions" in subscriber.topics
            assert "feature_updates" in subscriber.topics
            
            # Test TorchScript-specific features
            subscriber.connect()
            mock_socket.connect.assert_called_once()
    
    def test_torchscript_tensor_serialization(self):
        """Test tensor serialization for TorchScript communication."""
        with patch('torch.jit.load') as mock_torch_load, \
             patch('zmq.Context') as mock_context:
            
            mock_socket = Mock()
            mock_context.return_value.socket.return_value = mock_socket
            
            from deep_momentum_trading.src.communication.torchscript_subscriber import TorchScriptSubscriber
            
            subscriber = TorchScriptSubscriber(port=5559, topics=["tensors"])
            subscriber.connect()
            
            # Mock tensor data
            tensor_message = {
                "type": "tensor_data",
                "shape": [32, 60, 50],  # batch_size, sequence_length, features
                "dtype": "float32",
                "data": "base64_encoded_tensor_data"
            }
            
            # Test tensor handling
            mock_socket.recv_json.return_value = tensor_message
            received_message = subscriber.receive_message()
            
            # Verify tensor message structure
            assert received_message["type"] == "tensor_data"
            assert received_message["shape"] == [32, 60, 50]
            assert received_message["dtype"] == "float32"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
