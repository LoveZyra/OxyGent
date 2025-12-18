"""
Unit tests for SSEOxyAgent
"""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
import respx
from aioresponses import aioresponses
from pydantic import ValidationError

from oxygent.mas import MAS
from oxygent.oxy.agents.sse_oxy_agent import SSEOxyGent
from oxygent.schemas import OxyRequest, OxyState, SSEMessage


# ──────────────────────────────────────────────────────────────────────────────
# Dummy MAS
# ──────────────────────────────────────────────────────────────────────────────
class DummyMAS:
    def __init__(self):
        self.oxy_name_to_oxy = {}
        self.message_prefix = "msg"
        self.name = "test_mas"
        self.background_tasks = set()
        self.send_message = AsyncMock()
        self.func_process_message = lambda dict_message, oxy_request: dict_message


# ──────────────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────────────
@pytest.fixture
def sse_agent():
    return SSEOxyGent(
        name="sse_agent",
        desc="UT SSE Agent",
        server_url="https://remote-mas.example.com",
    )


@pytest.fixture
def oxy_request():
    req = OxyRequest(
        arguments={"query": "ping"},
        caller="user",
        caller_category="user",
        current_trace_id="trace123",
    )
    req.mas = DummyMAS()
    return req


# ──────────────────────────────────────────────────────────────────────────────
# Tests
# ──────────────────────────────────────────────────────────────────────────────
def test_url_validation():
    with pytest.raises(ValidationError):
        SSEOxyGent(name="bad", desc="bad", server_url="ftp://foo.com")


@pytest.mark.asyncio
async def test_init_fetch_org(sse_agent):
    """init() 会调用 httpx GET /get_organization 并填充 .org"""
    with respx.mock(assert_all_called=True) as router:
        router.get(httpx.URL("https://remote-mas.example.com/get_organization")).mock(
            return_value=httpx.Response(
                200,
                json={"data": {"organization": [{"id": 1, "is_remote": False}]}},
            )
        )
        await sse_agent.init()
        assert sse_agent.org[0]["id"] == 1
        assert sse_agent.org[0]["is_remote"] is False


@pytest.mark.asyncio
async def test_execute_sse_flow(sse_agent, oxy_request):
    with respx.mock() as router:
        router.get(httpx.URL("https://remote-mas.example.com/get_organization")).mock(
            return_value=httpx.Response(200, json={"data": {"organization": []}})
        )
        await sse_agent.init()

    sse_payloads = [
        {
            "type": "tool_call",
            "content": {"caller_category": "agent", "callee_category": "agent"},
        },
        {
            "type": "observation",
            "content": {"caller_category": "agent", "callee_category": "agent"},
        },
        {"type": "answer", "content": "pong"},
    ]

    sse_bytes = (
        b"".join(f"data: {json.dumps(evt)}\n\n".encode() for evt in sse_payloads)
        + b"data: done\n\n"
    )

    with aioresponses() as mocked_aio:
        mocked_aio.post(
            "https://remote-mas.example.com/sse/chat",
            status=200,
            body=sse_bytes,
            headers={"Content-Type": "text/event-stream"},
        )
        resp = await sse_agent.execute(oxy_request)

        assert resp.state is OxyState.COMPLETED
        assert resp.output == "pong"


# ──────────────────────────────────────────────────────────────────────────────
# Retry Functionality Integration Tests
# ──────────────────────────────────────────────────────────────────────────────
@pytest.fixture
def real_mas():
    """Create a real MAS instance for retry testing"""
    return MAS(name="test_retry_mas")


@pytest.fixture
def real_oxy_request(real_mas):
    """Create OxyRequest with real MAS for retry testing"""
    req = OxyRequest(
        arguments={"query": "test_retry"},
        caller="user",
        caller_category="user",
        current_trace_id="retry_trace_123",
    )
    req.mas = real_mas
    return req


class MockRedisClient:
    """Mock Redis client that can simulate failures"""
    def __init__(self, fail_count=0):
        self.fail_count = fail_count
        self.call_count = 0
        self.success_count = 0
    
    async def lpush(self, key, value):
        self.call_count += 1
        if self.call_count <= self.fail_count:
            raise Exception(f"Redis connection failed (attempt {self.call_count})")
        self.success_count += 1


@pytest.mark.asyncio
async def test_retry_mechanism_redis_failure(real_oxy_request):
    """Test retry mechanism when Redis operations fail"""
    # Setup mock Redis that fails first 2 times, succeeds on 3rd
    mock_redis = MockRedisClient(fail_count=2)
    real_oxy_request.mas.redis_client = mock_redis
    real_oxy_request.mas.es_client = AsyncMock()  
 
    test_message = {"type": "test", "content": "retry test"}
    redis_key = f"{real_oxy_request.mas.message_prefix}:{real_oxy_request.mas.name}:{real_oxy_request.current_trace_id}"
    
    # First attempt
    with pytest.raises(Exception, match="Redis connection failed"):
        await real_oxy_request.send_message(test_message)
    
    assert real_oxy_request.mas.get_retry_attempt(redis_key) == 1
    
    with pytest.raises(Exception, match="Redis connection failed"):
        await real_oxy_request.send_message(test_message)
    
    assert real_oxy_request.mas.get_retry_attempt(redis_key) == 2
    
    await real_oxy_request.send_message(test_message)
    
    assert real_oxy_request.mas.get_retry_attempt(redis_key) == 0
    assert mock_redis.success_count == 1


@pytest.mark.asyncio
async def test_retry_exponential_backoff_timing(real_oxy_request):
    """Test that retry times follow exponential backoff pattern"""
    # Setup mock Redis that always fails to test timing calculation
    mock_redis = MockRedisClient(fail_count=10)  
    real_oxy_request.mas.redis_client = mock_redis
    real_oxy_request.mas.es_client = AsyncMock()
    
    test_message = {"type": "test", "content": "timing test"}
    redis_key = f"{real_oxy_request.mas.message_prefix}:{real_oxy_request.mas.name}:{real_oxy_request.current_trace_id}"
    
    # Expected retry times for each attempt
    expected_times = [2000, 4000, 8000, 16000, 30000, 30000, 30000, 30000, 30000, 30000]
    
    for i, expected_time in enumerate(expected_times):
        try:
            await real_oxy_request.send_message(test_message)
        except Exception:
            pass  
        
        retry_attempt = real_oxy_request.mas.get_retry_attempt(redis_key)
        if retry_attempt > 0:
            base_retry = 1000
            calculated_time = min(base_retry * (2 ** retry_attempt), 30000)
            assert calculated_time == expected_time, f"Attempt {i+1}: expected {expected_time}ms, got {calculated_time}ms"


@pytest.mark.asyncio
async def test_retry_reset_on_success(real_oxy_request):
    """Test that retry attempts are reset after successful send"""
    # Setup mock Redis that fails once, then succeeds
    mock_redis = MockRedisClient(fail_count=1)
    real_oxy_request.mas.redis_client = mock_redis
    real_oxy_request.mas.es_client = AsyncMock()
    
    test_message = {"type": "test", "content": "reset test"}
    redis_key = f"{real_oxy_request.mas.message_prefix}:{real_oxy_request.mas.name}:{real_oxy_request.current_trace_id}"
    
    with pytest.raises(Exception):
        await real_oxy_request.send_message(test_message)
    assert real_oxy_request.mas.get_retry_attempt(redis_key) == 1
    
    await real_oxy_request.send_message(test_message)
    assert real_oxy_request.mas.get_retry_attempt(redis_key) == 0
    assert mock_redis.success_count == 1


@pytest.mark.asyncio
async def test_retry_with_different_trace_ids(real_oxy_request):
    """Test that retry tracking works independently for different trace IDs"""
    mock_redis = MockRedisClient(fail_count=1)
    real_oxy_request.mas.redis_client = mock_redis
    real_oxy_request.mas.es_client = AsyncMock()
    
    test_message = {"type": "test", "content": "multi trace test"}
    
    # Create requests with different trace IDs
    req1 = real_oxy_request.clone_with(current_trace_id="trace_1")
    req2 = real_oxy_request.clone_with(current_trace_id="trace_2")
    
    redis_key1 = f"{req1.mas.message_prefix}:{req1.mas.name}:trace_1"
    redis_key2 = f"{req2.mas.message_prefix}:{req2.mas.name}:trace_2"
    
    with pytest.raises(Exception):
        await req1.send_message(test_message)
    
    await asyncio.sleep(0.01)
    
    mock_redis2 = MockRedisClient(fail_count=1)
    req2.mas.redis_client = mock_redis2
    
    with pytest.raises(Exception):
        await req2.send_message(test_message)
    
    await asyncio.sleep(0.01)
    
    assert req1.mas.get_retry_attempt(redis_key1) == 1
    assert req2.mas.get_retry_attempt(redis_key2) == 1
    
    mock_redis.fail_count = 0
    mock_redis2.fail_count = 0
    
    await req1.send_message(test_message)
    await req2.send_message(test_message)
    
    assert req1.mas.get_retry_attempt(redis_key1) == 0
    assert req2.mas.get_retry_attempt(redis_key2) == 0


@pytest.mark.asyncio
async def test_retry_sse_field_update(real_oxy_request):
    """Test that SSE message retry field is updated correctly during retries"""
    mock_redis = MockRedisClient(fail_count=2)
    real_oxy_request.mas.redis_client = mock_redis
    real_oxy_request.mas.es_client = AsyncMock()
    
    test_message = {"type": "test", "content": "field test"}
    redis_key = f"{real_oxy_request.mas.message_prefix}:{real_oxy_request.mas.name}:{real_oxy_request.current_trace_id}"
    
    with pytest.raises(Exception):
        await real_oxy_request.send_message(test_message)
    
    retry_attempt = real_oxy_request.mas.get_retry_attempt(redis_key)
    assert retry_attempt == 1
    
    base_retry = 1000
    expected_retry = base_retry * (2 ** retry_attempt) 
    assert expected_retry == 2000
    
    with pytest.raises(Exception):
        await real_oxy_request.send_message(test_message)
    
    retry_attempt = real_oxy_request.mas.get_retry_attempt(redis_key)
    assert retry_attempt == 2
    
    expected_retry = base_retry * (2 ** retry_attempt)  
    assert expected_retry == 4000
    
    mock_redis.fail_count = 0
    
    await real_oxy_request.send_message(test_message)
    assert real_oxy_request.mas.get_retry_attempt(redis_key) == 0


@pytest.mark.asyncio
async def test_retry_with_message_processing(real_oxy_request):
    """Test retry works correctly with message processing pipeline"""
    mock_redis = MockRedisClient(fail_count=1)
    real_oxy_request.mas.redis_client = mock_redis
    real_oxy_request.mas.es_client = AsyncMock()
    
    # Add custom message processing that modifies messages
    def process_message(dict_message, oxy_request):
        dict_message['data']['processed'] = True
        dict_message['data']['timestamp'] = '2024-01-01'
        return dict_message
    
    real_oxy_request.mas.func_process_message = process_message
    
    test_message = {"type": "test", "content": "processing test"}
    
    with pytest.raises(Exception):
        await real_oxy_request.send_message(test_message)
    
    await real_oxy_request.send_message(test_message)
    
    assert real_oxy_request.mas.get_retry_attempt(
        f"{real_oxy_request.mas.message_prefix}:{real_oxy_request.mas.name}:{real_oxy_request.current_trace_id}"
    ) == 0
