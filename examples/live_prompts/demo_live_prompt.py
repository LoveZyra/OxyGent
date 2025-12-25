import asyncio
import os

from oxygent import MAS, oxy

oxy_space = [
    oxy.HttpLLM(
        name="default_llm",
        api_key=os.getenv("DEFAULT_LLM_API_KEY"),
        base_url=os.getenv("DEFAULT_LLM_BASE_URL"),
        model_name=os.getenv("DEFAULT_LLM_MODEL_NAME"),
    ),
    oxy.ChatAgent(
        name="chat_agent1",
        prompt="You are a helpful assistant.",
        prompt_key="my_prompt",
    ),
    oxy.ChatAgent(
        name="chat_agent2",
        prompt="You are a helpful assistant.",
        prompt_key="my_prompt",
    ),
    oxy.ChatAgent(
        name="chat_agent3",
        prompt="You are a helpful assistant.",
        use_live_prompt=False,
    ),
]


async def main():
    async with MAS(oxy_space=oxy_space) as mas:
        await mas.start_web_service(first_query="hello")


if __name__ == "__main__":
    asyncio.run(main())
