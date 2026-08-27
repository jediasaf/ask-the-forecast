"""The agent loop.

One entry point: ``ask(question)``. It retrieves the relevant glossary
entries, hands Claude the tool set, lets the SDK tool runner drive the
tool-call loop, and validates the final message against the structured
output schema. The model never computes a number — every figure it cites
must come from a tool return, and the eval harness checks that it does.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field

import anthropic
from anthropic import Anthropic

from . import tools
from .retrieval import format_entries, retrieve
from .schema import AnswerReport, output_format

MODEL = "claude-opus-5"  # exact string, no date suffix
MAX_TOKENS = 16000
MAX_ITERATIONS = 8

SYSTEM_PROMPT = """\
You are a forecast-performance analyst answering questions about one fixed
backtest: a demand-forecast model for 7 departments of a Walmart store,
measured against a seasonal-naive baseline over 263 days.

Hard rules:
- Every number in your answer MUST come from a tool return in this
  conversation. Never compute, estimate, extrapolate, or recall a figure.
  No arithmetic — not even differences or averages of tool-returned numbers.
  If a question needs a number no tool returns, it is unanswerable.
- If the data cannot answer the question (future forecasts, revenue, prices,
  headcount, store or SKU-level detail), set "unanswerable": true and say
  briefly what is missing. Do not offer a substitute number as if it answered
  the question.
- If a question is ambiguous, answer the most reasonable scoped reading and
  say explicitly how you scoped it (e.g. "overall, pooled across
  departments"). Do not answer a broader question than the data supports.
- List every figure you rely on in "figures_cited", with the department it
  belongs to (null for pooled figures), exactly as the tool returned it.
- Keep answers short and concrete: the finding first, then the numbers that
  support it.
"""


@dataclass
class AgentResult:
    report: AnswerReport | None
    tool_calls: list[dict] = field(default_factory=list)
    error: str | None = None
    stop_reason: str | None = None
    raw_text: str | None = None


def _build_system(question: str) -> list[dict]:
    system: list[dict] = [
        {
            "type": "text",
            "text": SYSTEM_PROMPT,
            "cache_control": {"type": "ephemeral"},
        }
    ]
    glossary = format_entries(retrieve(question, k=2))
    if glossary:
        system.append({"type": "text", "text": glossary})
    return system


def ask(
    question: str,
    client: Anthropic | None = None,
    model: str = MODEL,
    effort: str = "high",
) -> AgentResult:
    if client is None:
        try:
            client = Anthropic()  # resolves ANTHROPIC_API_KEY / ant auth profile
        except TypeError:  # the SDK raises TypeError when no credential source exists
            return AgentResult(
                None,
                error="no_credentials: export ANTHROPIC_API_KEY or run `ant auth login`",
            )
    tools.reset_call_log()

    try:
        runner = client.beta.messages.tool_runner(
            model=model,
            max_tokens=MAX_TOKENS,
            thinking={"type": "adaptive"},
            output_config={"effort": effort, **output_format()},
            max_iterations=MAX_ITERATIONS,
            system=_build_system(question),
            tools=tools.ALL_TOOLS,
            messages=[{"role": "user", "content": question}],
        )
        final = runner.until_done()
    except anthropic.NotFoundError as e:
        return AgentResult(None, list(tools.CALL_LOG), error=f"not_found: {e.message}")
    except anthropic.RateLimitError as e:
        return AgentResult(None, list(tools.CALL_LOG), error=f"rate_limited: {e.message}")
    except anthropic.APIStatusError as e:
        return AgentResult(
            None, list(tools.CALL_LOG), error=f"api_error_{e.status_code}: {e.message}"
        )
    except anthropic.APIConnectionError as e:
        return AgentResult(None, list(tools.CALL_LOG), error=f"connection_error: {e}")
    except TypeError:  # the SDK raises TypeError at request time when no credential resolves
        return AgentResult(
            None,
            list(tools.CALL_LOG),
            error="no_credentials: export ANTHROPIC_API_KEY or run `ant auth login`",
        )

    call_log = list(tools.CALL_LOG)

    if final.stop_reason == "refusal":
        return AgentResult(None, call_log, error="refusal", stop_reason="refusal")

    text = next((b.text for b in final.content if b.type == "text"), None)
    if text is None:
        return AgentResult(
            None, call_log, error="no_text_block", stop_reason=final.stop_reason
        )

    try:
        report = AnswerReport.model_validate(json.loads(text))
    except Exception as e:  # json or schema failure — the eval counts these
        return AgentResult(
            None,
            call_log,
            error=f"schema_invalid: {e}",
            stop_reason=final.stop_reason,
            raw_text=text,
        )

    return AgentResult(report, call_log, stop_reason=final.stop_reason, raw_text=text)


def main() -> None:
    parser = argparse.ArgumentParser(description="Ask the forecast a question.")
    parser.add_argument("question", help="natural-language question about the backtest")
    args = parser.parse_args()

    result = ask(args.question)
    print(f"tools called: {[c['tool'] for c in result.tool_calls]}")
    if result.error:
        print(f"error: {result.error}", file=sys.stderr)
        if result.raw_text:
            print(result.raw_text, file=sys.stderr)
        sys.exit(1)
    print(json.dumps(result.report.model_dump(), indent=2))


if __name__ == "__main__":
    main()
