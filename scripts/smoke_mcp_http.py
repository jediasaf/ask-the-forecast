"""Smoke test for the MCP server running over HTTP.

Used three ways, all with the same assertions: against a local process, against
the container, and against the Service inside a Kubernetes cluster. It speaks
the protocol the way a real client does rather than curling the health route,
because a server can be healthy and still advertise the wrong tools.

    python -m scripts.smoke_mcp_http http://127.0.0.1:8000

Exit code 0 only if the server lists every tool the agent defines and a real
tool call returns the figure that is in the data.
"""

from __future__ import annotations

import asyncio
import json
import sys
import urllib.request

from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

from agent.tools import ALL_TOOLS, VALID_DEPARTMENTS


def _direct(dept: str) -> dict:
    """The same call made straight to the tool function, bypassing the network."""
    tool = next(t for t in ALL_TOOLS if t.name == "get_accuracy")
    return json.loads(tool.func(dept))


async def check(base: str) -> None:
    with urllib.request.urlopen(f"{base}/healthz", timeout=10) as r:
        health = json.loads(r.read())
    assert health == {"status": "ok", "tools": len(ALL_TOOLS)}, health

    async with streamable_http_client(f"{base}/mcp") as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            listed = {t.name for t in (await session.list_tools()).tools}
            expected = {t.name for t in ALL_TOOLS}
            assert listed == expected, f"server drifted: {sorted(listed ^ expected)}"

            dept = VALID_DEPARTMENTS[0]
            result = await session.call_tool("get_accuracy", {"dept": dept})
            assert not result.is_error, result
            payload = json.loads(result.content[0].text)
            assert payload == _direct(dept), payload

    print(f"ok: {base} serves {len(listed)} tools, get_accuracy({dept}) = {payload}")


if __name__ == "__main__":
    asyncio.run(check(sys.argv[1].rstrip("/") if len(sys.argv) > 1 else "http://127.0.0.1:8000"))
