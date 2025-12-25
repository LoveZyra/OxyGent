"""Base Bank module for Large Language Model implementations.

This module provides a base class for implementing bank modules for large language models.
"""

import logging

from pydantic import Field

from ...schemas import OxyRequest, OxyResponse
from ..base_tool import BaseTool

logger = logging.getLogger(__name__)


class BaseBank(BaseTool):
    category: str = Field("bank", description="")

    async def _execute(self, oxy_request: OxyRequest) -> OxyResponse:
        raise NotImplementedError("This method is not yet implemented")
