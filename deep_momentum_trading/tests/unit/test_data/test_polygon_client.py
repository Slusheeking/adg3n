import pytest
import asyncio
import os
import websockets # Added import
import httpx # Added import
from unittest.mock import AsyncMock, patch
from deep_momentum_trading.src.data.polygon_client import PolygonClient

# Mock environment variable for API key
@pytest.fixture(autouse=True)
def mock_polygon_api_key():
    with patch.dict(os.environ, {'POLYGON_API_KEY': 'TEST_API_KEY'}):
        yield

@pytest.fixture
def polygon_client():
    """Provides a PolygonClient instance for testing."""
    return PolygonClient()

@pytest.mark.asyncio
async def test_polygon_client_init_success(mock_polygon_api_key):
    """Test successful initialization of PolygonClient."""
    client = PolygonClient()
    assert client.api_key == 'TEST_API_KEY'
    assert not client.is_connected

@pytest.mark.asyncio
async def test_polygon_client_init_no_api_key():
    """Test initialization fails without API key."""
    with patch.dict(os.environ, {}, clear=True): # Clear env vars
        with pytest.raises(ValueError, match="API key not provided"):
            PolygonClient()

@pytest.mark.asyncio
async def test_connect_websocket_success(polygon_client):
    """Test successful WebSocket connection and authentication."""
    with patch('websockets.connect', new_callable=AsyncMock) as mock_connect:
        mock_ws = AsyncMock()
        mock_connect.return_value = mock_ws
        mock_ws.recv.return_value = '{"status": "auth_success"}'

        await polygon_client._connect_websocket()

        mock_connect.assert_awaited_once_with(polygon_client.WS_URL)
        mock_ws.send.assert_awaited_once_with('{"action": "auth", "params": "TEST_API_KEY"}')
        assert polygon_client.is_connected
        assert polygon_client.reconnect_attempts == 0

@pytest.mark.asyncio
async def test_connect_websocket_auth_fail(polygon_client):
    """Test WebSocket connection with authentication failure."""
    with patch('websockets.connect', new_callable=AsyncMock) as mock_connect:
        mock_ws = AsyncMock()
        mock_connect.return_value = mock_ws
        mock_ws.recv.return_value = '{"status": "auth_failed"}'

        await polygon_client._connect_websocket()

        assert not polygon_client.is_connected
        mock_ws.close.assert_awaited_once() # Should close on auth failure

@pytest.mark.asyncio
async def test_handle_reconnect_attempts(polygon_client):
    """Test reconnection logic with multiple attempts."""
    polygon_client.max_reconnect_attempts = 2
    polygon_client.reconnect_delay = 0.01 # Shorten delay for test

    with patch('websockets.connect', new_callable=AsyncMock) as mock_connect:
        mock_connect.side_effect = [
            websockets.exceptions.ConnectionClosedOK, # Simulate first failure
            AsyncMock(), # Simulate second attempt success
        ]
        mock_connect.return_value.recv.return_value = '{"status": "auth_success"}' # For successful connect

        await polygon_client._connect_websocket() # First attempt fails
        assert polygon_client.reconnect_attempts == 1
        assert not polygon_client.is_connected # Still not connected after first failure

        # The _connect_websocket calls _handle_reconnect which then calls _connect_websocket again
        # So we need to call _connect_websocket once more to trigger the second attempt
        await polygon_client._connect_websocket() 
        
        assert polygon_client.reconnect_attempts == 2
        assert polygon_client.is_connected # Should be connected after second attempt

@pytest.mark.asyncio
async def test_stream_real_time_data_success(polygon_client):
    """Test successful streaming of real-time data."""
    mock_handler = AsyncMock()
    symbols = ["AAPL", "MSFT"]

    with patch('websockets.connect', new_callable=AsyncMock) as mock_connect:
        mock_ws = AsyncMock()
        mock_connect.return_value = mock_ws
        mock_ws.recv.side_effect = [
            '{"status": "auth_success"}', # Auth response
            '[{"ev":"T","sym":"AAPL","p":100}]', # Data message 1
            '[{"ev":"Q","sym":"MSFT","bp":200}]', # Data message 2
            asyncio.CancelledError # Stop streaming
        ]

        streaming_task = asyncio.create_task(polygon_client.stream_real_time_data(symbols, mock_handler))
        
        try:
            await streaming_task
        except asyncio.CancelledError:
            pass # Expected

        mock_ws.send.assert_any_call('{"action": "auth", "params": "TEST_API_KEY"}')
        mock_ws.send.assert_any_call('{"action": "subscribe", "params": "T.AAPL,Q.AAPL,AM.AAPL,T.MSFT,Q.MSFT,AM.MSFT"}')
        
        assert mock_handler.call_count == 2
        mock_handler.assert_any_call({"ev": "T", "sym": "AAPL", "p": 100})
        mock_handler.assert_any_call({"ev": "Q", "sym": "MSFT", "bp": 200})

@pytest.mark.asyncio
async def test_get_historical_data_rest_success(polygon_client):
    """Test successful retrieval of historical data via REST."""
    with patch('httpx.AsyncClient.get', new_callable=AsyncMock) as mock_get:
        mock_response = AsyncMock()
        mock_response.json.return_value = {"results": [{"o": 100, "c": 101}]}
        mock_response.raise_for_status.return_value = None
        mock_get.return_value.__aenter__.return_value = mock_response # For async with

        data = await polygon_client.get_historical_data_rest(
            symbol="AAPL", from_date="2023-01-01", to_date="2023-01-02"
        )
        assert data is not None
        assert data["results"][0]["o"] == 100

@pytest.mark.asyncio
async def test_get_historical_data_rest_http_error(polygon_client):
    """Test historical data retrieval with HTTP error."""
    with patch('httpx.AsyncClient.get', new_callable=AsyncMock) as mock_get:
        mock_response = AsyncMock()
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "404 Not Found", request=httpx.Request("GET", "http://test.com"), response=httpx.Response(404)
        )
        mock_get.return_value.__aenter__.return_value = mock_response

        data = await polygon_client.get_historical_data_rest(
            symbol="AAPL", from_date="2023-01-01", to_date="2023-01-02"
        )
        assert data is None

@pytest.mark.asyncio
async def test_close_websocket(polygon_client):
    """Test closing the WebSocket connection."""
    polygon_client.is_connected = True
    polygon_client.websocket = AsyncMock()

    await polygon_client.close()

    polygon_client.websocket.close.assert_awaited_once()
    assert not polygon_client.is_connected
    assert polygon_client.websocket is None
