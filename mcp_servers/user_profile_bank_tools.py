from mcp.server.fastmcp import FastMCP
from pydantic import Field

mcp = FastMCP(port=8000)


@mcp.tool(description="A tool for querying user profile")
def user_profile_retrieve(
    query: str = Field(description="query"),
    user_pin: str = Field(description="SystemArg.agent_pin"),
    agent_pin: str = Field(description="SystemArg.user_pin"),
) -> str:
    user_profile_dict = {
        "001": "Arlen, a student, likes music",
        "002": "Tom, a programmer, likes sports",
    }
    portrait = user_profile_dict.get(user_pin, "Nothing")
    return f"The current user profile is: {portrait}"


@mcp.tool(description="A tool for updating user profile")
def user_profile_deposit(
    content: str = Field(description="content"),
    user_pin: str = Field(description="SystemArg.agent_pin"),
    agent_pin: str = Field(description="SystemArg.user_pin"),
) -> str:
    print(agent_pin, user_pin, content)
    return "updated user_profile"


if __name__ == "__main__":
    mcp.run(transport="sse")
