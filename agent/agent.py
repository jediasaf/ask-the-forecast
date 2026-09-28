"""The agent loop.

One entry point: ``ask(question)``. It retrieves the relevant glossary
entries, hands Claude the tool set, lets the SDK tool runner drive the
tool-call loop, and validates the final message against the structured
output schema. The model never computes a number — every figure it cites
must come from a tool return, and the eval harness checks that it does.
"""

from __future__ import annotations

import argparse
import base64
import json
import sys
import time
from dataclasses import dataclass, field

import anthropic
from anthropic import Anthropic

from . import tools
from .retrieval import format_entries, retrieve
from .schema import AnswerReport, output_format

MODEL = "claude-opus-5"  # exact string, no date suffix

# USD per million tokens: (input, output). Cache reads bill at 0.1x input,
# cache writes at 1.25x input.
PRICING = {
    "claude-opus-5": (5.00, 25.00),
    "claude-sonnet-5": (2.00, 10.00),
}
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
  "value" is strictly numeric; if a figure belongs to a specific day, put
  the ISO date in the "date" field — never encode a date as a number.
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
    usage: dict | None = None  # tokens, cost_usd, seconds, iterations


def _accumulate_usage(totals: dict, usage) -> None:
    totals["input_tokens"] += usage.input_tokens or 0
    totals["output_tokens"] += usage.output_tokens or 0
    totals["cache_read_tokens"] += getattr(usage, "cache_read_input_tokens", 0) or 0
    totals["cache_write_tokens"] += getattr(usage, "cache_creation_input_tokens", 0) or 0


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


def _build_system(question: str, has_image: bool = False) -> list[dict]:
    system: list[dict] = [
        {
            "type": "text",
            "text": SYSTEM_PROMPT,
            "cache_control": {"type": "ephemeral"},
        }
    ]
    # The vision rules go in their own block, after the cached prefix, so
    # attaching an image does not invalidate the cache on the main prompt.
    if has_image:
        system.append({"type": "text", "text": VISION_RULES})
    glossary = format_entries(retrieve(question, k=3))
    if glossary:
        system.append({"type": "text", "text": glossary})
    return system


VISION_RULES = """\

The user has attached an image, normally a chart or a slide from a review deck.
Treat it as a CLAIM, not as evidence. It was produced by someone else and may be
wrong, stale, or edited.

Your job is verification, in this order:
1. Read every figure the image states, and say which department each belongs to.
2. Call the tools to get the same figures from the system of record.
3. Compare them. Report any figure where the image and the data disagree, giving
   both numbers and naming the source of truth.
4. If they all agree, say so plainly. Do not invent a discrepancy to seem useful.

Never cite a number read off an image as if it were data. Every figure in
figures_cited must come from a tool return, exactly as it does for any other
question. A number that exists only in the image is a claim about data, not data.
"""


def _content_blocks(question: str, image: bytes | None) -> list[dict] | str:
    """Build the user turn, with an image block first when one is attached.

    The image goes before the text because the question almost always refers to
    it, and a model reads the turn in order.
    """
    if image is None:
        return question
    return [
        {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/png",
                "data": base64.standard_b64encode(image).decode(),
            },
        },
        {"type": "text", "text": question},
    ]


def ask(
    question: str,
    image: bytes | None = None,
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
    totals = {"input_tokens": 0, "output_tokens": 0, "cache_read_tokens": 0, "cache_write_tokens": 0}
    iterations = 0
    started = time.monotonic()

    try:
        runner = client.beta.messages.tool_runner(
            model=model,
            max_tokens=MAX_TOKENS,
            thinking={"type": "adaptive"},
            output_config={"effort": effort, **output_format()},
            max_iterations=MAX_ITERATIONS,
            system=_build_system(question, image is not None),
            tools=tools.ALL_TOOLS,
            messages=[{"role": "user", "content": _content_blocks(question, image)}],
        )
        final = None
        for message in runner:  # equivalent to until_done, but captures per-turn usage
            final = message
            iterations += 1
            if message.usage is not None:
                _accumulate_usage(totals, message.usage)
        if final is None:
            return AgentResult(None, list(tools.CALL_LOG), error="runner_returned_nothing")
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
    usage = {
        **totals,
        "cost_usd": _cost_usd(totals, model),
        "seconds": round(time.monotonic() - started, 1),
        "iterations": iterations,
    }

    if final.stop_reason == "refusal":
        return AgentResult(None, call_log, error="refusal", stop_reason="refusal", usage=usage)

    text = next((b.text for b in final.content if b.type == "text"), None)
    if text is None:
        return AgentResult(
            None, call_log, error="no_text_block", stop_reason=final.stop_reason, usage=usage
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
            usage=usage,
        )

    return AgentResult(report, call_log, stop_reason=final.stop_reason, raw_text=text, usage=usage)


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
