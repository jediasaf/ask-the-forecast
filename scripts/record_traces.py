"""Record full agent traces for the portfolio's replay view.

Runs a curated set of questions through the real agent and saves everything
the replay needs: the question, each tool call with its input and output,
the structured answer, and the run's cost/latency. The portfolio view plays
these back verbatim — recorded runs, clearly labeled as such, so the live
site needs no API key.

Usage: python -m scripts.record_traces  →  scripts/agentTraces.json
"""

import json
from pathlib import Path

from agent.agent import MODEL, ask

QUESTIONS = [
    ("bias", "What's the bias on FOODS_3?"),
    ("beat-naive", "Which departments beat the naive baseline?"),
    ("losing-volume", "Of the departments losing to the naive baseline, which has the highest volume?"),
    ("folds", "Across the three walk-forward folds, is FOODS_1's forecast value add improving or deteriorating?"),
    ("december", "In December 2013, which departments actually beat the naive baseline?"),
    ("worst-day", "Which department had the largest single-day absolute forecast error anywhere in the backtest, and what happened that day?"),
    ("trap", "What's the revenue per head?"),
    ("future", "What's the forecast for next quarter?"),
]

def main() -> None:
    traces = []
    for key, q in QUESTIONS:
        print(f"recording {key}: {q}")
        r = ask(q)
        if r.error:
            print(f"  ERROR {r.error} — skipping")
            continue
        traces.append(
            {
                "key": key,
                "question": q,
                "model": MODEL,
                "calls": r.tool_calls,
                "report": r.report.model_dump(),
                "usage": r.usage,
            }
        )
        print(f"  {len(r.tool_calls)} calls, ${r.usage['cost_usd']}, {r.usage['seconds']}s")
    out = Path(__file__).resolve().parent / "agentTraces.json"
    out.write_text(json.dumps(traces, indent=1))
    print(f"wrote {out} ({len(traces)} traces)")

if __name__ == "__main__":
    main()
