"""The same agent, as a LangGraph state machine.

``agent.agent`` lets the SDK's tool runner drive the loop. This module spells
the loop out as a graph, so each step is a node that can be traced, tested on
its own, and replaced:

    retrieve -> model <-> tools
                  |
                parse -> guard -> END
                           |
                           +--> model   (one correction, then it must stand)

Three things are held constant so the two engines can be compared on the same
question sets: the system prompt, the seventeen tool contracts, and the
structured output schema. They are imported, not copied.

One thing is new. ``guard`` is a deterministic check inside the graph: every
figure the answer cites must appear in a tool return from this conversation.
The loop engine relies on the prompt for that and on the eval to catch a miss
afterwards. Here a miss is caught before the answer leaves, sent back once with
the offending figures named, and reported as an error if it happens again.
"""

from __future__ import annotations

import json
import time
from typing import Annotated, Any, TypedDict

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from agent import tools as agent_tools
from agent.agent import (
    MAX_ITERATIONS,
    MAX_TOKENS,
    MODEL,
    PRICING,
    SYSTEM_PROMPT,
    AgentResult,
)
from agent.retrieval import format_entries, retrieve
from agent.schema import AnswerReport, output_format

MAX_CORRECTIONS = 1

_TOOLS = {t.name: t for t in agent_tools.ALL_TOOLS}


class State(TypedDict, total=False):
    messages: Annotated[list, add_messages]
    question: str
    calls: list[dict]          # every tool call made, with its input and output
    model_turns: int
    corrections: int
    report: dict | None
    raw_text: str | None
    error: str | None
    unverified: list[str]      # figures the guard could not trace to a tool return
    flagged: list[str]         # everything the guard ever sent back, kept for the record
    usage: dict


# ------------------------------------------------------------------ the model


def tool_specs() -> list[dict]:
    """The tool contracts exactly as the loop engine sends them."""
    return [t.to_dict() for t in agent_tools.ALL_TOOLS]


def make_model(model: str = MODEL, effort: str = "high"):
    """ChatAnthropic bound to the seventeen tools and the answer schema."""
    from langchain_anthropic import ChatAnthropic

    llm = ChatAnthropic(
        model=model,
        max_tokens=MAX_TOKENS,
        thinking={"type": "adaptive"},
        output_config={"effort": effort, **output_format()},
        max_retries=2,
    )
    return llm.bind_tools(tool_specs())


# ---------------------------------------------------------------------- nodes


def retrieve_node(state: State) -> dict:
    question = state["question"]
    system: list[dict] = [
        {"type": "text", "text": SYSTEM_PROMPT, "cache_control": {"type": "ephemeral"}}
    ]
    glossary = format_entries(retrieve(question, k=3))
    if glossary:
        system.append({"type": "text", "text": glossary})
    return {
        "messages": [SystemMessage(content=system), HumanMessage(content=question)],
        "calls": [],
        "model_turns": 0,
        "corrections": 0,
        "usage": {"input_tokens": 0, "output_tokens": 0,
                  "cache_read_tokens": 0, "cache_write_tokens": 0},
    }


def _add_usage(totals: dict, message: AIMessage) -> dict:
    meta = getattr(message, "usage_metadata", None) or {}
    details = meta.get("input_token_details") or {}
    cache_read = details.get("cache_read", 0) or 0
    cache_write = details.get("cache_creation", 0) or 0
    out = dict(totals)
    # LangChain reports input_tokens with the cached portions folded in. The
    # loop engine prices them separately, so they are taken back out here.
    out["input_tokens"] += max(0, (meta.get("input_tokens", 0) or 0) - cache_read - cache_write)
    out["output_tokens"] += meta.get("output_tokens", 0) or 0
    out["cache_read_tokens"] += cache_read
    out["cache_write_tokens"] += cache_write
    return out


def make_model_node(llm):
    def model_node(state: State) -> dict:
        message = llm.invoke(state["messages"])
        return {
            "messages": [message],
            "model_turns": state.get("model_turns", 0) + 1,
            "usage": _add_usage(state["usage"], message),
        }

    return model_node


def tools_node(state: State) -> dict:
    """Run the requested tools in order and record each call with its return.

    Sequential on purpose. The record of what each tool returned is what the
    guard checks the answer against, so it has to be exact.
    """
    last = state["messages"][-1]
    calls = list(state.get("calls", []))
    replies = []
    for call in last.tool_calls:
        name, args = call["name"], call.get("args") or {}
        tool = _TOOLS.get(name)
        if tool is None:
            text = json.dumps({"error": f"Unknown tool '{name}'."})
        else:
            try:
                text = tool.func(**args)
            except TypeError as e:  # wrong or missing arguments
                text = json.dumps({"error": f"Bad arguments for {name}: {e}"})
        try:
            output: Any = json.loads(text)
        except ValueError:
            output = text
        calls.append({"tool": name, "input": args, "output": output})
        replies.append(ToolMessage(content=text, tool_call_id=call["id"], name=name))
    return {"messages": replies, "calls": calls}


def _text_of(message: AIMessage) -> str | None:
    content = message.content
    if isinstance(content, str):
        return content or None
    parts = [b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text"]
    text = "".join(parts)
    return text or None


def parse_node(state: State) -> dict:
    text = _text_of(state["messages"][-1])
    if text is None:
        return {"error": "no_text_block", "report": None, "raw_text": None}
    try:
        report = AnswerReport.model_validate(json.loads(text))
    except Exception as e:  # json or schema failure, the eval counts these
        return {"error": f"schema_invalid: {e}", "report": None, "raw_text": text}
    return {"error": None, "report": report.model_dump(), "raw_text": text}


def _numbers(node, out: set[float]) -> None:
    if isinstance(node, bool):
        return
    if isinstance(node, (int, float)):
        out.add(float(node))
    elif isinstance(node, list):
        for item in node:
            _numbers(item, out)
    elif isinstance(node, dict):
        for value in node.values():
            _numbers(value, out)


def untraced_figures(report: dict, calls: list[dict]) -> list[str]:
    """Figures the answer cites that no tool returned in this conversation."""
    returned: set[float] = set()
    for call in calls:
        _numbers(call.get("output"), returned)
    missing = []
    for fig in report.get("figures_cited", []):
        value = float(fig["value"])
        tol = max(0.05, abs(value) * 1e-4)
        if not any(abs(value - r) <= tol for r in returned):
            missing.append(f"{fig['metric']}={fig['value']} ({fig.get('dept') or 'overall'})")
    return missing


def guard_node(state: State) -> dict:
    report = state.get("report")
    if state.get("error") or report is None:
        return {"unverified": []}
    missing = untraced_figures(report, state.get("calls", []))
    if not missing:
        return {"unverified": []}
    if state.get("corrections", 0) >= MAX_CORRECTIONS:
        return {"unverified": missing, "error": f"unverified_figures: {missing}"}
    note = (
        "Your answer cites figures that no tool returned in this conversation: "
        + "; ".join(missing)
        + ". Every figure must come from a tool return. Call the tool that returns "
        "each one, or remove it, or mark the question unanswerable. Then answer again."
    )
    return {
        "unverified": missing,
        "flagged": list(state.get("flagged", [])) + missing,
        "corrections": state.get("corrections", 0) + 1,
        "messages": [HumanMessage(content=note)],
    }


# -------------------------------------------------------------------- routing


def after_model(state: State) -> str:
    last = state["messages"][-1]
    if getattr(last, "tool_calls", None):
        if state.get("model_turns", 0) >= MAX_ITERATIONS:
            return "parse"  # out of turns: whatever it said last has to stand
        return "tools"
    return "parse"


def after_guard(state: State) -> str:
    if state.get("error") or not state.get("unverified"):
        return END
    return "model"


def build_graph(llm):
    g = StateGraph(State)
    g.add_node("retrieve", retrieve_node)
    g.add_node("model", make_model_node(llm))
    g.add_node("tools", tools_node)
    g.add_node("parse", parse_node)
    g.add_node("guard", guard_node)
    g.add_edge(START, "retrieve")
    g.add_edge("retrieve", "model")
    g.add_conditional_edges("model", after_model, {"tools": "tools", "parse": "parse"})
    g.add_edge("tools", "model")
    g.add_edge("parse", "guard")
    g.add_conditional_edges("guard", after_guard, {"model": "model", END: END})
    return g.compile()


# ----------------------------------------------------------------- entry point


def _cost_usd(totals: dict, model: str) -> float | None:
    if model not in PRICING:
        return None
    in_price, out_price = PRICING[model]
    return round(
        (
            totals["input_tokens"] * in_price
            + totals["cache_read_tokens"] * in_price * 0.1
            + totals["cache_write_tokens"] * in_price * 1.25
            + totals["output_tokens"] * out_price
        )
        / 1_000_000,
        4,
    )


def ask(
    question: str,
    model: str = MODEL,
    effort: str = "high",
    llm=None,
    run_metadata: dict | None = None,
) -> AgentResult:
    """Answer one question. Returns the same AgentResult the loop engine does,
    so the eval harness scores both without knowing which one ran."""
    agent_tools.reset_call_log()
    started = time.monotonic()
    try:
        graph = build_graph(llm if llm is not None else make_model(model, effort))
        final = graph.invoke(
            {"question": question},
            config={
                "recursion_limit": 4 * MAX_ITERATIONS + 8,
                # Picked up by LangSmith when LANGSMITH_TRACING is set, and
                # ignored otherwise. Nothing here needs a key to run.
                "run_name": "ask-the-forecast",
                "tags": ["engine:langgraph", f"model:{model}", f"effort:{effort}"],
                "metadata": run_metadata or {},
            },
        )
    except Exception as e:  # credentials, network, rate limits
        kind = type(e).__name__
        return AgentResult(None, [], error=f"{kind}: {e}")

    totals = final.get("usage") or {}
    usage = {
        **totals,
        "cost_usd": _cost_usd(totals, model) if totals else None,
        "seconds": round(time.monotonic() - started, 1),
        "iterations": final.get("model_turns", 0),
        "corrections": final.get("corrections", 0),
        "guard_flagged": final.get("flagged", []),
    }
    calls = final.get("calls", [])
    if final.get("error"):
        return AgentResult(None, calls, error=final["error"],
                           raw_text=final.get("raw_text"), usage=usage)
    return AgentResult(
        AnswerReport.model_validate(final["report"]), calls,
        raw_text=final.get("raw_text"), usage=usage,
    )
