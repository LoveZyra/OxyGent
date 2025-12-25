import logging

from pydantic import Field

from ...config import Config
from ...schemas import OxyRequest, OxyResponse, OxyState
from .base_llm import BaseLLM

logger = logging.getLogger(__name__)


class LocalLLM(BaseLLM):
    model_path: str = Field("")
    device_map: str = Field("auto")
    dtype: str = Field("bfloat16")

    async def init(self):
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as e:
            raise ImportError(
                "LocalLLM requires 'torch' and 'transformers' packages."
                "Please install them using 'pip install torch transformers einops transformers_stream_generator accelerate'"
            ) from e

        await super().init()
        # Load model directly
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_path, device_map=self.device_map, dtype=self.dtype
        )
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_path)

    async def _execute(self, oxy_request: OxyRequest) -> OxyResponse:
        payload = Config.get_llm_config()
        for k, v in self.llm_params.items():
            payload[k] = v
        for k, v in oxy_request.arguments.items():
            if k == "messages":
                continue
            payload[k] = v

        replace_dict = {
            "max_tokens": "max_new_tokens",
            "stream": "",
        }
        for k, v in replace_dict.items():
            if k in payload:
                if v:
                    payload[v] = payload[k]
                del payload[k]

        messages = oxy_request.arguments["messages"]

        input_text = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        input_ids = self._tokenizer.encode(input_text, return_tensors="pt")
        input_ids = input_ids.to(self._model.device)
        outputs = self._model.generate(input_ids=input_ids, **payload)[0]
        outputs = outputs[len(input_ids[0]) :]

        return OxyResponse(
            state=OxyState.COMPLETED,
            output=self._tokenizer.decode(outputs, skip_special_tokens=True),
        )
