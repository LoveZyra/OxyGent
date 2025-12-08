"""
Unit tests for FunctionHub
"""

import asyncio
import time
import concurrent.futures

import pytest

from oxygent.oxy.function_tools.function_hub import FunctionHub
from oxygent.oxy.function_tools.function_tool import FunctionTool
from oxygent.schemas import OxyResponse, OxyState


# ────────────────────────────────────────────────────────────────────────────
# Dummy MAS
# ────────────────────────────────────────────────────────────────────────────
class DummyMAS:
    def __init__(self):
        self.oxy_name_to_oxy = {}

    def add_oxy(self, oxy):
        self.oxy_name_to_oxy[oxy.name] = oxy


# ────────────────────────────────────────────────────────────────────────────
# Fixtures
# ────────────────────────────────────────────────────────────────────────────
@pytest.fixture
def mas_env():
    return DummyMAS()


@pytest.fixture
def func_hub(mas_env):
    hub = FunctionHub(name="hub", desc="UT FunctionHub")
    hub.set_mas(mas_env)
    return hub


# ────────────────────────────────────────────────────────────────────────────
# Tests
# ────────────────────────────────────────────────────────────────────────────
def test_decorator_registers_functions(func_hub):
    @func_hub.tool("sync add")
    def add(a: int, b: int):
        return a + b

    @func_hub.tool("async mul")
    async def mul(a: int, b: int):
        return a * b

    assert "add" in func_hub.func_dict
    assert "mul" in func_hub.func_dict
    desc, async_fn = func_hub.func_dict["add"]
    assert asyncio.iscoroutinefunction(async_fn)
    assert desc == "sync add"


@pytest.mark.asyncio
async def test_init_converts_to_function_tools(func_hub, mas_env):
    @func_hub.tool("echo")
    def echo(msg: str):
        return msg

    await func_hub.init()

    assert "echo" in mas_env.oxy_name_to_oxy
    tool = mas_env.oxy_name_to_oxy["echo"]
    assert isinstance(tool, FunctionTool)
    assert tool.desc == "echo"
    from oxygent.schemas import OxyRequest

    oxy_req = OxyRequest(
        arguments={"msg": "hello"},
        caller="tester",
        caller_category="agent",
        current_trace_id="trace123",
    )

    resp: OxyResponse = await tool._execute(oxy_req)
    assert resp.state is OxyState.COMPLETED
    assert resp.output == "hello"


def test_sync_function_wrapped_async(func_hub):
    @func_hub.tool("inc")
    def inc(x: int):
        return x + 1

    _, async_inc = func_hub.func_dict["inc"]
    assert asyncio.iscoroutinefunction(async_inc)

    result = asyncio.run(async_inc(41))
    assert result == 42


# ────────────────────────────────────────────────────────────────────────────
# Thread Pool Tests
# ────────────────────────────────────────────────────────────────────────────
def test_thread_pool_lazy_initialization(func_hub):
    """Test that thread pool is lazily initialized."""
    # Thread pool should be None initially
    assert func_hub._thread_pool is None
    
    # Accessing thread_pool property should initialize it
    pool = func_hub.thread_pool
    assert isinstance(pool, concurrent.futures.ThreadPoolExecutor)
    assert func_hub._thread_pool is not None
    
    # Second access should return the same instance
    pool2 = func_hub.thread_pool
    assert pool is pool2


@pytest.mark.asyncio
async def test_sync_function_execution_with_thread_pool(func_hub):
    """Test that sync functions are executed in thread pool."""
    execution_info = {"thread_id": None, "main_thread_id": None}
    
    @func_hub.tool("test sync function")
    def blocking_function(duration: float):
        """Simulate blocking operation."""
        import threading
        execution_info["thread_id"] = threading.current_thread().ident
        time.sleep(duration)
        return f"completed in thread {execution_info['thread_id']}"
    
    # Get main thread ID
    import threading
    execution_info["main_thread_id"] = threading.current_thread().ident
    
    # Execute the function
    _, async_func = func_hub.func_dict["blocking_function"]
    result = await async_func(0.1)
    
    # Verify function executed in different thread
    assert execution_info["thread_id"] is not None
    assert execution_info["thread_id"] != execution_info["main_thread_id"]
    assert "completed in thread" in result


@pytest.mark.asyncio
async def test_sync_function_with_kwargs_in_thread_pool(func_hub):
    """Test sync functions with kwargs are executed in thread pool."""
    @func_hub.tool("test function with kwargs")
    def function_with_kwargs(a: int, b: int, multiplier: float = 1.0):
        """Function that uses kwargs."""
        time.sleep(0.01)  # Small delay to simulate work
        return (a + b) * multiplier
    
    # Execute function with kwargs
    _, async_func = func_hub.func_dict["function_with_kwargs"]
    result = await async_func(2, 3, multiplier=2.0)
    
    assert result == 10.0  # (2 + 3) * 2.0 = 10.0


@pytest.mark.asyncio
async def test_cleanup_shuts_down_thread_pool(func_hub):
    """Test that cleanup properly shuts down thread pool."""
    # Initialize thread pool by accessing it
    pool = func_hub.thread_pool
    assert isinstance(pool, concurrent.futures.ThreadPoolExecutor)
    
    # Verify pool is active
    assert func_hub._thread_pool is not None
    
    # Cleanup should shut down the thread pool
    await func_hub.cleanup()
    
    # Thread pool should be None after cleanup
    assert func_hub._thread_pool is None


@pytest.mark.asyncio
async def test_multiple_cleanup_calls_safe(func_hub):
    """Test that multiple cleanup calls are safe."""
    # Initialize thread pool
    func_hub.thread_pool
    
    # First cleanup
    await func_hub.cleanup()
    assert func_hub._thread_pool is None
    
    # Second cleanup should not raise error
    await func_hub.cleanup()
    assert func_hub._thread_pool is None


@pytest.mark.asyncio
async def test_cleanup_without_thread_pool_initialization(func_hub):
    """Test cleanup when thread pool was never initialized."""
    # Ensure thread pool is not initialized
    assert func_hub._thread_pool is None
    
    # Cleanup should work without errors
    await func_hub.cleanup()
    assert func_hub._thread_pool is None


@pytest.mark.asyncio
async def test_concurrent_sync_function_execution(func_hub):
    """Test concurrent execution of multiple sync functions."""
    results = []
    
    @func_hub.tool("concurrent task")
    def concurrent_task(task_id: int, duration: float):
        """Simulate concurrent blocking operation."""
        time.sleep(duration)
        return f"task_{task_id}_completed"
    
    # Execute multiple tasks concurrently
    _, async_func = func_hub.func_dict["concurrent_task"]
    tasks = [
        async_func(1, 0.1),
        async_func(2, 0.15),
        async_func(3, 0.05),
    ]
    
    start_time = time.time()
    results = await asyncio.gather(*tasks)
    total_time = time.time() - start_time
    
    # Verify all tasks completed
    assert len(results) == 3
    assert "task_1_completed" in results
    assert "task_2_completed" in results
    assert "task_3_completed" in results
    
    # Total time should be less than sum of individual times (due to concurrency)
    assert total_time < 0.3  # Should be around 0.15s (max duration)
