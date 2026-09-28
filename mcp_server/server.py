"""MCP server exposing the forecast agent's tools to any MCP client.

Why this exists: the seventeen tools in ``agent.tools`` are the useful part of
this project, and inside ``agent.agent`` they are reachable only by this repo's
own agent loop. Over the Model Context Protocol the same tools become callable
from Claude Desktop, an IDE, or any other MCP host, with no second
implementation to keep in sync.

The tools stay the single source of truth. This module walks ``ALL_TOOLS`` and
registers each one's underlying function, reusing the name and description the
Anthropic SDK already generated from the signature and docstring. A tool added
to ``agent/tools.py`` therefore appears over MCP automatically and cannot drift
out of step with the agent's own view of it.

No API key is needed. These tools are pure lookups over the JSON in ``data/``:
the model lives on the client side of the protocol, not here.

Run:
    python -m mcp_server.server

Register with an MCP host (Claude Desktop, ``claude_desktop_config.json``):

    {
      "mcpServers": {
        "ask-the-forecast": {
          "command": "python",
          "args": ["-m", "mcp_server.server"],
          "cwd": "/path/to/ask-the-forecast"
        }
      }
    }
"""

from __future__ import annotations

from mcp.server.mcpserver import MCPServer

from agent.tools import ALL_TOOLS

SERVER_NAME = "ask-the-forecast"

INSTRUCTIONS = """Read-only tools over a demand-forecasting backtest (Walmart M5
data, 263-day window). Every tool is a lookup over precomputed JSON: none of
them models, extrapolates, or infers. Do not do arithmetic on the returned
figures yourself. If no tool answers the question, say so rather than
estimating, and note that no forward-looking forecast exists in this data."""


def build_server() -> MCPServer:
    """Register every agent tool on a fresh MCPServer and return it."""
    server = MCPServer(SERVER_NAME, instructions=INSTRUCTIONS)
    for tool in ALL_TOOLS:
        # tool.func is the original typed function; the SDK derives the schema
        # from its annotations, so the MCP contract matches the agent's.
        server.add_tool(tool.func, name=tool.name, description=tool.description)
    return server


server = build_server()


def main() -> None:
    server.run(transport="stdio")


if __name__ == "__main__":
    main()
