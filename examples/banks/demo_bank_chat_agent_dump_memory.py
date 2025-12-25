import asyncio
import json
import os

from oxygent import MAS, OxyResponse, oxy


async def dump_memory(oxy_response: OxyResponse) -> OxyResponse:
    oxy_request = oxy_response.oxy_request
    history = {
        "query": oxy_request.get_query(),
        "answer": oxy_response.output,
    }

    await oxy_request.call_async(
        callee="user_profile_deposit",
        arguments={"content": json.dumps(history, ensure_ascii=False)},
        is_send_message=False,
    )
    return oxy_response


oxy_space = [
    oxy.HttpLLM(
        name="default_llm",
        api_key=os.getenv("DEFAULT_LLM_API_KEY"),
        base_url=os.getenv("DEFAULT_LLM_BASE_URL"),
        model_name=os.getenv("DEFAULT_LLM_MODEL_NAME"),
        llm_params={"temperature": 0.01},
        semaphore=4,
    ),
    oxy.ChatAgent(
        name="qa_agent",
        llm_model="default_llm",
        prompt="You can refer to the following information to answer the question:\n${preceding_text}",
        banks=["remote_user_profile_banks"],
        preceding_oxy=["user_profile_retrieve"],
        preceding_placeholder="preceding_text",
        func_process_output=dump_memory,
    ),
    oxy.BankClient(
        name="remote_user_profile_banks",
        server_url="http://127.0.0.1:8090",
    ),
]


def func_filter(payload):
    payload["group_data"] = {"user_pin": "002"}
    return payload


async def main():
    async with MAS(
        oxy_space=oxy_space, func_filter=func_filter, name="temp_app"
    ) as mas:
        await mas.start_web_service(first_query="Who I am")


if __name__ == "__main__":
    asyncio.run(main())
