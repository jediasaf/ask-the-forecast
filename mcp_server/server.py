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


@server.custom_route("/healthz", methods=["GET"], include_in_schema=False)
async def healthz(_request):
    """Liveness and readiness for the HTTP transport.

    It reports the tool count rather than a bare 200, so a container that
    started but failed to load its data is caught by the probe instead of by
    the first caller.
    """
    from starlette.responses import JSONResponse

    return JSONResponse({"status": "ok", "tools": len(ALL_TOOLS)})


def main(argv: list[str] | None = None) -> None:
    """stdio by default, for desktop hosts. HTTP when it runs as a service.

    The HTTP transport is stateless and answers in plain JSON. These tools are
    pure lookups with nothing to remember between calls, so keeping sessions
    would buy nothing and would pin each client to one replica.
    """
    import argparse
    import os

    ap = argparse.ArgumentParser(description="ask-the-forecast MCP server")
    ap.add_argument("--transport", choices=["stdio", "http"],
                    default=os.environ.get("MCP_TRANSPORT", "stdio"))
    ap.add_argument("--host", default=os.environ.get("MCP_HOST", "127.0.0.1"))
    ap.add_argument("--port", type=int, default=int(os.environ.get("MCP_PORT", "8000")))
    args = ap.parse_args(argv)

    if args.transport == "stdio":
        server.run(transport="stdio")
        return

    from mcp.server.transport_security import TransportSecuritySettings

    # Host header checking stays on. The allowed names come from the
    # environment because only the deployment knows what it is reached as.
    allowed = [h.strip() for h in os.environ.get("MCP_ALLOWED_HOSTS", "").split(",") if h.strip()]
    security = TransportSecuritySettings(
        enable_dns_rebinding_protection=True,
        allowed_hosts=allowed or [f"127.0.0.1:{args.port}", f"localhost:{args.port}"],
    )
    server.run(
        transport="streamable-http",
        host=args.host,
        port=args.port,
        stateless_http=True,
        json_response=True,
        transport_security=security,
    )


if __name__ == "__main__":
    main()
