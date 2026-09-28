"""The graph, tested without a model.

A scripted chat model stands in for Claude and returns the messages a real one
would: a tool call, then an answer. That is enough to test everything the graph
itself is responsible for: routing, running tools, recording what they
returned, parsing the answer, and the guard.
"""

from __future__ import annotations

import json

import pytest

pytest.importorskip("langgraph")

from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage

from agent.tools import ALL_TOOLS
from agent_graph.graph import ask, build_graph, tool_specs, untraced_figures

TRUE_FA = json.loads(next(t for t in ALL_TOOLS if t.name == "get_accuracy").func("FOODS_1"))[
    "forecast_accuracy_pct"
]


def call(name: str, args: dict, call_id: str = "c1") -> AIMessage:
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": call_id}])


def answer(value: float, unanswerable: bool = False) -> AIMessage:
    report = {
        "answer": f"FOODS_1 forecast accuracy is {value}%.",
        "departments_referenced": ["FOODS_1"],
        "figures_cited": [] if unanswerable else [
            {"metric": "forecast_accuracy_pct", "dept": "FOODS_1", "value": value, "date": None}
        ],
        "confidence": "high",
        "unanswerable": unanswerable,
    }
    return AIMessage(content=json.dumps(report))


def scripted(*messages: AIMessage):
    return FakeMessagesListChatModel(responses=list(messages))


def test_tool_contracts_match_the_loop_engine():
    specs = tool_specs()
    assert [s["name"] for s in specs] == [t.name for t in ALL_TOOLS]
    assert all(s["input_schema"]["additionalProperties"] is False for s in specs)


def test_graph_has_the_five_nodes():
    nodes = set(build_graph(scripted(answer(1.0))).get_graph().nodes)
    assert {"retrieve", "model", "tools", "parse", "guard"} <= nodes


def test_tool_call_then_answer():
    llm = scripted(call("get_accuracy", {"dept": "FOODS_1"}), answer(TRUE_FA))
    result = ask("How accurate is FOODS_1?", llm=llm)
    assert result.error is None
    assert [c["tool"] for c in result.tool_calls] == ["get_accuracy"]
    assert result.tool_calls[0]["output"]["forecast_accuracy_pct"] == TRUE_FA
    assert result.report.figures_cited[0].value == TRUE_FA
    assert result.usage["iterations"] == 2
    assert result.usage["corrections"] == 0


def test_guard_sends_an_untraced_figure_back_once():
    llm = scripted(
        call("get_accuracy", {"dept": "FOODS_1"}),
        answer(TRUE_FA + 11.1),   # a figure no tool returned
        answer(TRUE_FA),          # corrected on the second attempt
    )
    result = ask("How accurate is FOODS_1?", llm=llm)
    assert result.error is None
    assert result.report.figures_cited[0].value == TRUE_FA
    assert result.usage["corrections"] == 1


def test_guard_fails_an_answer_that_is_wrong_twice():
    llm = scripted(
        call("get_accuracy", {"dept": "FOODS_1"}),
        answer(TRUE_FA + 11.1),
        answer(TRUE_FA + 22.2),
    )
    result = ask("How accurate is FOODS_1?", llm=llm)
    assert result.report is None
    assert result.error.startswith("unverified_figures")


def test_guard_rejects_a_figure_cited_with_no_tool_call_at_all():
    llm = scripted(answer(TRUE_FA), answer(0.0, unanswerable=True))
    result = ask("How accurate is FOODS_1?", llm=llm)
    # The figure is true, but nothing returned it in this conversation.
    assert result.usage["corrections"] == 1
    assert result.report.unanswerable is True


def test_unknown_tool_is_reported_to_the_model_not_raised():
    llm = scripted(call("get_revenue", {"dept": "FOODS_1"}), answer(0.0, unanswerable=True))
    result = ask("What was revenue?", llm=llm)
    assert result.error is None
    assert "Unknown tool" in result.tool_calls[0]["output"]["error"]


def test_answer_that_is_not_json_is_a_schema_error():
    result = ask("anything", llm=scripted(AIMessage(content="FOODS_1 is fine.")))
    assert result.error.startswith("schema_invalid")


def test_untraced_figures_tolerates_float_noise():
    calls = [{"tool": "get_accuracy", "input": {}, "output": {"fa": 75.0}}]
    report = {"figures_cited": [{"metric": "fa", "dept": "FOODS_1", "value": 75.00001}]}
    assert untraced_figures(report, calls) == []
