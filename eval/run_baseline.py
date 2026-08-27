"""The no-tools baseline: prompt-stuffing instead of tool calls.

Answers the same eval questions by pasting all six data files into the
prompt and asking for the same structured output — no tools at all. This is
the architecture the agent exists to beat; running it makes "why tools?" an
empirical question instead of a design opinion.

Scored with the same harness, minus tool selection (there are no tools):
numeric accuracy, refusal accuracy, and fabrication all still apply.

Usage:
    python -m eval.run_baseline [--model MODEL] [--effort LEVEL] [--limit N]
Writes eval/results-baseline-<model-shortname>.md / .json.
"""

from __future__ import annotations

import argparse
import json
import time

import anthropic
from anthropic import Anthropic

from agent.agent import MAX_TOKENS, MODEL, PRICING, AgentResult, _cost_usd
from agent.schema import AnswerReport, output_format
from agent.tools import DATA_DIR

from .run_eval import QUESTIONS, data_universe, score_question, write_results

BASELINE_RULES = """\
You are a forecast-performance analyst answering questions about one fixed
backtest: a demand-forecast model for 7 departments of a Walmart store,
measured against a seasonal-naive baseline over 263 days. The complete data
is in the DATA section of this prompt.

Hard rules:
- Every number in your answer MUST appear verbatim in the DATA section.
  Never compute, estimate, extrapolate, or recall a figure. No arithmetic —
  not even differences or averages of numbers in the data. If a question
  needs a number the data does not contain verbatim, it is unanswerable.
- If the data cannot answer the question (future forecasts, revenue, prices,
  headcount, store or SKU-level detail), set "unanswerable": true and say
  briefly what is missing. Do not offer a substitute number as if it answered
  the question.
- If a question is ambiguous, answer the most reasonable scoped reading and
  say explicitly how you scoped it. Do not answer a broader question than
  the data supports.
- List every figure you rely on in "figures_cited", with the department it
  belongs to (null for pooled figures), exactly as it appears in the data.
- Keep answers short and concrete: the finding first, then the numbers that
  support it.
"""


def _data_block() -> str:
    parts = []
    for path in sorted(DATA_DIR.glob("*.json")):
        parts.append(f"### {path.name}\n{path.read_text()}")
    return "DATA:\n\n" + "\n\n".join(parts)


def ask_baseline(question: str, client: Anthropic, model: str, effort: str) -> AgentResult:
    started = time.monotonic()
    try:
        response = client.messages.create(
            model=model,
            max_tokens=MAX_TOKENS,
            thinking={"type": "adaptive"},
            output_config={"effort": effort, **output_format()},
            system=[
                {"type": "text", "text": BASELINE_RULES},
                # the big block is cached so 37 questions don't re-bill it
                {"type": "text", "text": _data_block(), "cache_control": {"type": "ephemeral"}},
            ],
            messages=[{"role": "user", "content": question}],
        )
    except anthropic.RateLimitError as e:
        return AgentResult(None, error=f"rate_limited: {e.message}")
    except anthropic.APIStatusError as e:
        return AgentResult(None, error=f"api_error_{e.status_code}: {e.message}")
    except anthropic.APIConnectionError as e:
        return AgentResult(None, error=f"connection_error: {e}")

    totals = {
        "input_tokens": response.usage.input_tokens or 0,
        "output_tokens": response.usage.output_tokens or 0,
        "cache_read_tokens": response.usage.cache_read_input_tokens or 0,
        "cache_write_tokens": response.usage.cache_creation_input_tokens or 0,
    }
    usage = {
        **totals,
        "cost_usd": _cost_usd(totals, model),
        "seconds": round(time.monotonic() - started, 1),
        "iterations": 1,
    }

    if response.stop_reason == "refusal":
        return AgentResult(None, error="refusal", stop_reason="refusal", usage=usage)
    text = next((b.text for b in response.content if b.type == "text"), None)
    if text is None:
        return AgentResult(None, error="no_text_block", usage=usage)
    try:
        report = AnswerReport.model_validate(json.loads(text))
    except Exception as e:
        return AgentResult(None, error=f"schema_invalid: {e}", raw_text=text, usage=usage)
    return AgentResult(report, usage=usage, stop_reason=response.stop_reason, raw_text=text)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=MODEL, choices=sorted(PRICING))
    parser.add_argument("--effort", default="high", choices=["low", "medium", "high", "xhigh", "max"])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--questions", default=None, help="path to a questions .jsonl")
    parser.add_argument("--tag", default=None, help="override the results-file suffix")
    args = parser.parse_args()

    client = Anthropic()
    universe = data_universe()
    from pathlib import Path
    qpath = Path(args.questions) if args.questions else QUESTIONS
    questions = [json.loads(line) for line in qpath.read_text().splitlines() if line.strip()]
    if args.limit:
        questions = questions[: args.limit]

    rows = []
    for i, q in enumerate(questions, 1):
        print(f"[{i}/{len(questions)}] {q['id']}: {q['q']}")
        result = ask_baseline(q["q"], client, args.model, args.effort)
        # no tools exist in this mode, so tool selection cannot be scored
        q_scored = {**q, "expect_tools": []}
        row = score_question(q_scored, result, universe)
        status = "PASS" if row["pass"] else "FAIL — " + "; ".join(row["failures"])
        print(f"  {status}")
        rows.append(row)

    short = args.model.replace("claude-", "")
    write_results(rows, args.model, args.effort, tag=args.tag or f"baseline-{short}")


if __name__ == "__main__":
    main()
