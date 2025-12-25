import logging
from typing import Dict, Literal

import httpx
from pydantic import AnyUrl, Field

from ...schemas import OxyRequest, OxyResponse, OxyState
from .base_bank import BaseBank

logger = logging.getLogger(__name__)


class BankTool(BaseBank):
    server_url: AnyUrl = Field("")
    method: Literal["GET", "POST"] = Field("GET")
    is_permission_required: bool = Field(True, description="")
    headers: Dict[str, str] = Field(
        default_factory=dict, description="Extra HTTP headers"
    )
    is_retrievable: bool = Field(False, description="")

    async def _execute(self, oxy_request: OxyRequest) -> OxyResponse:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                str(self.server_url),
                headers=self.headers,
                timeout=self.timeout,
                json=oxy_request.arguments,
            )
            return OxyResponse(
                state=OxyState.COMPLETED,
                output=response.text,
            )
