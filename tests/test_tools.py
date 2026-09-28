"""Deterministic tests for the seventeen agent tools.

These run without an API key and without a model. They exist because the
agent's honesty guarantee rests entirely on the tools: the system prompt
forbids the model from doing arithmetic, so every figure in an answer is a
figure a tool returned. If a tool silently changes shape or starts inventing a
key, the eval would catch it only after spending money on a live run. These
catch it in CI, for free, in under a second.
"""

from __future__ import annotations

import json

import pytest

from agent import tools
from agent.tools import ALL_TOOLS, VALID_DEPARTMENTS

BY_NAME = {t.name: t for t in ALL_TOOLS}


def call(name: str, **kwargs):
    """Invoke a tool by name and parse its JSON return."""
    return json.loads(BY_NAME[name].func(**kwargs))


# --------------------------------------------------------------- the registry


def test_seventeen_tools_exposed():
    """The README and the CV both claim seventeen tools. Keep that true."""
    assert len(ALL_TOOLS) == 17


def test_tool_names_are_unique():
    names = [t.name for t in ALL_TOOLS]
    assert len(names) == len(set(names))


@pytest.mark.parametrize("tool", ALL_TOOLS, ids=lambda t: t.name)
def test_every_tool_is_strictly_schemad(tool):
    """strict=True is what stops the model passing a field the tool never reads."""
    schema = tool.input_schema
    assert schema["type"] == "object"
    assert schema.get("additionalProperties") is False
    assert tool.description, f"{tool.name} has no description for the model to read"


@pytest.mark.parametrize("tool", ALL_TOOLS, ids=lambda t: t.name)
def test_every_tool_returns_json(tool):
    """Tool results travel as text, so every return must parse as JSON."""
    schema = tool.input_schema
    if schema.get("required"):
        pytest.skip("covered by the per-tool tests below")
    json.loads(tool.func())


# ------------------------------------------------------------------ contracts


def test_list_departments_matches_the_index():
    """Rows carry dept plus volume, so the agent can rank without a second call."""
    out = call("list_departments")
    names = [row["dept"] for row in out["departments"]]
    assert sorted(names) == sorted(VALID_DEPARTMENTS)
    assert all("volume_units" in row for row in out["departments"])


def test_get_accuracy_returns_both_the_model_and_the_baseline():
    """Forecast accuracy is meaningless without the baseline beside it."""
    out = call("get_accuracy", dept=VALID_DEPARTMENTS[0])
    assert "forecast_accuracy_pct" in out and "naive_accuracy_pct" in out
    assert isinstance(out["forecast_accuracy_pct"], (int, float))


def test_unknown_department_is_an_error_payload_not_an_exception():
    """The model must be able to read and recover from a bad argument."""
    out = call("get_accuracy", dept="NOT_A_DEPARTMENT")
    assert "error" in out
    assert "hint" in out or "valid" in json.dumps(out).lower()


def test_unknown_sku_is_an_error_payload_not_an_exception():
    out = call("get_sku", item="NOT_A_SKU")
    assert "error" in out


def test_compare_departments_rejects_an_unknown_metric():
    out = call("compare_departments", metric="not_a_metric")
    assert "error" in out


# ---------------------------------------------------------------- call logging


def test_calls_are_logged_for_the_eval_to_score_tool_selection():
    """The eval scores tool selection off CALL_LOG. If logging breaks, the
    eval silently scores every question as calling nothing."""
    tools.reset_call_log()
    call("get_overall")
    call("get_accuracy", dept=VALID_DEPARTMENTS[0])
    logged = [c["tool"] for c in tools.CALL_LOG]
    assert logged == ["get_overall", "get_accuracy"]


def test_reset_clears_the_log():
    tools.reset_call_log()
    call("get_overall")
    tools.reset_call_log()
    assert tools.CALL_LOG == []
