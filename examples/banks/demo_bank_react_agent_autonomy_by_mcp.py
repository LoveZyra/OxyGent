import asyncio
import os

from oxygent import MAS, oxy

oxy_space = [
    oxy.HttpLLM(
        name="default_llm",
        api_key=os.getenv("DEFAULT_LLM_API_KEY"),
        base_url=os.getenv("DEFAULT_LLM_BASE_URL"),
        model_name=os.getenv("DEFAULT_LLM_MODEL_NAME"),
        llm_params={"temperature": 0.01},
        semaphore=4,
    ),
    oxy.StdioMCPClient(
        name="time_tools",
        params={
            "command": "uvx",
            "args": ["mcp-server-time", "--local-timezone=Asia/Shanghai"],
        },
    ),
    oxy.ReActAgent(
        name="qa_agent",
        llm_model="default_llm",
        tools=["time_tools", "remote_user_profile_banks"],
    ),
    oxy.SSEMCPClient(
        name="remote_user_profile_banks",
        sse_url="http://127.0.0.1:8000/sse",
    ),
]


def func_filter(payload):
    payload["group_data"] = {"user_pin": "002"}
    return payload


async def main():
    async with MAS(oxy_space=oxy_space, func_filter=func_filter) as mas:
        await mas.start_web_service(first_query="Who I am")


if __name__ == "__main__":
    asyncio.run(main())
