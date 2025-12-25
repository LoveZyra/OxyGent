import asyncio

from oxygent import MAS, oxy

oxy_space = [
    oxy.LocalLLM(
        name="default_llm",
        model_path="/path/to/your_model",
    ),
    oxy.ChatAgent(
        name="master_agent",
        llm_model="default_llm",
    ),
]


async def main():
    async with MAS(oxy_space=oxy_space) as mas:
        await mas.start_web_service(first_query="hello")


if __name__ == "__main__":
    asyncio.run(main())
