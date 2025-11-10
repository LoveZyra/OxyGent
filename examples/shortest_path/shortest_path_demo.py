import os
from oxygent import MAS, Config, oxy
from oxygent.shortest_path.shortest_path import shortest_path_tools

Config.set_agent_llm_model("default_llm")

def create_optimal_agent():
    return oxy.ReActAgent(
        name="shortest_path_agent",
        desc="Agent for computing shortest path between different cities",
        category="agent",
        class_name="ReActAgent",
        tools=["shortest_path_tools"],
        llm_model="default_llm",
        is_entrance=False,
        is_permission_required=False,
        is_save_data=True,
        timeout=30,
        retries=3,
        delay=1,
        is_multimodal_supported=False,
        semaphore=2,
    )
oxy_space = [
    oxy.HttpLLM(
        name="default_llm",
        api_key=os.getenv("DEFAULT_LLM_API_KEY"),
        base_url=os.getenv("DEFAULT_LLM_BASE_URL"),
        model_name=os.getenv("DEFAULT_LLM_MODEL_NAME"),
    ),
    shortest_path_tools,
    create_optimal_agent(),
    oxy.ReActAgent(
        name="excel_agent",
        desc="A tool that can read file information based on the Excel file path.",
        tools=["shortest_path_tools"],
    ),
    oxy.ReActAgent(
        is_master=True,
        name="master_agent",
        sub_agents=["excel_agent","shortest_path_agent"],
    ),
]

async def main():
    async with MAS(oxy_space=oxy_space) as mas:
        await mas.start_web_service(first_query="")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())