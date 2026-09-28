"""Tests for the MCP adapter.

The adapter's whole job is to forward the agent's tools without re-declaring
them. These assert exactly that: every tool appears, dispatch reaches the right
function, and a bad argument still comes back as readable data rather than an
exception, because an MCP client has no other way to recover.
"""

from __future__ import annotations

import asyncio
import json

from agent.tools import ALL_TOOLS, VALID_DEPARTMENTS
from mcp_server.server import build_server

SERVER = build_server()


def _run(coro):
    return asyncio.run(coro)


def _text(result) -> str:
    """Pull the text payload out of whatever shape the SDK returns."""
    content = result.content if hasattr(result, "content") else result
    first = content[0] if isinstance(content, list) else content
    return first.text


def test_every_agent_tool_is_advertised():
    listed = _run(SERVER.list_tools())
    assert {t.name for t in listed} == {t.name for t in ALL_TOOLS}
    assert len(listed) == 17


def test_descriptions_are_forwarded_unchanged():
    """If these drift, the MCP client and the agent disagree about the tool."""
    listed = {t.name: t for t in _run(SERVER.list_tools())}
    for tool in ALL_TOOLS:
        assert listed[tool.name].description == tool.description


def test_every_advertised_tool_has_an_object_schema():
    for t in _run(SERVER.list_tools()):
        assert t.input_schema["type"] == "object"


def test_call_tool_returns_the_tool_payload():
    out = _run(SERVER.call_tool("get_accuracy", {"dept": VALID_DEPARTMENTS[0]}))
    payload = json.loads(_text(out))
    assert payload["dept"] == VALID_DEPARTMENTS[0]
    assert "forecast_accuracy_pct" in payload


def test_call_tool_with_no_arguments():
    out = _run(SERVER.call_tool("get_overall", {}))
    assert json.loads(_text(out))


def test_bad_argument_comes_back_as_data_not_an_exception():
    """The tools answer a bad department with an error payload and a hint. That
    behaviour has to survive the MCP hop, or a client has nothing to act on."""
    out = _run(SERVER.call_tool("get_accuracy", {"dept": "NOT_A_DEPARTMENT"}))
    assert "error" in json.loads(_text(out))
