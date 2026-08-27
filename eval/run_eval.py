"""Run the eval set against the agent and write the failure analysis.

Scores four things per question:
  - tool selection: did the agent call the expected tool(s)?
  - numeric accuracy: does every expected figure appear in figures_cited,
    for the right department, at the tool-returned value?
  - refusal accuracy: did it decline the unanswerable ones — and only those?
  - fabrication: is every figure it cited actually present in the data?

Usage:
    python -m eval.run_eval [--limit N] [--only ID_OR_CATEGORY] [--model MODEL] [--effort LEVEL]

Writes eval/results.md (the failure analysis) and eval/results.json (raw).
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from datetime import date
from pathlib import Path

from agent.agent import MODEL, AgentResult, ask
from agent.tools import DATA_DIR

EVAL_DIR = Path(__file__).resolve().parent
QUESTIONS = EVAL_DIR / "questions.jsonl"

# ------------------------------------------------------------- data universe


def _collect_numbers(node, out: set[float]) -> None:
    if isinstance(node, bool):
        return
    if isinstance(node, (int, float)):
        out.add(float(node))
    elif isinstance(node, list):
        for item in node:
            _collect_numbers(item, out)
    elif isinstance(node, dict):
        for value in node.values():
            _collect_numbers(value, out)


def data_universe() -> set[float]:
    """Every number that exists anywhere in the data files.

    A cited figure not in this set cannot have come from a tool return.
    The set is dense (thousands of series values), so this check can only
    catch fabrications, not prove provenance — tool-selection and numeric
    checks carry that weight.
    """
    universe: set[float] = set()
    for path in DATA_DIR.glob("*.json"):
        with open(path) as f:
            _collect_numbers(json.load(f), universe)
    universe.add(3049.0)  # total SKU count, stated in tool returns
    return universe


# ----------------------------------------------------------------- scoring


def _tol(expected: float) -> float:
    return max(0.05, abs(expected) * 1e-4)


def score_question(q: dict, result: AgentResult, universe: set[float]) -> dict:
    called = {c["tool"] for c in result.tool_calls}
    row = {
        "id": q["id"],
        "category": q["category"],
        "q": q["q"],
        "tools_called": sorted(called),
        "error": result.error,
        "checks": {},
        "failures": [],
    }

    if result.error or result.report is None:
        row["checks"] = {k: False for k in ("tool_selection", "numeric", "refusal", "fabrication")}
        row["failures"].append(f"agent error: {result.error}")
        row["pass"] = False
        row["usage"] = result.usage
        return row

    report = result.report

    # --- tool selection
    expected = set(q.get("expect_tools", []))
    if not expected:
        tool_ok = True
    elif q.get("tool_match", "all") == "any":
        tool_ok = bool(expected & called)
    else:
        tool_ok = expected <= called
    if not tool_ok:
        row["failures"].append(
            f"tool selection: expected {sorted(expected)} ({q.get('tool_match', 'all')}), called {sorted(called)}"
        )

    # --- numeric accuracy: every expected figure present in figures_cited
    numeric_ok = True
    cited = [(f.dept.upper() if f.dept else None, f.value, f.metric) for f in report.figures_cited]
    for exp in q.get("expect_values", []):
        exp_dept = exp["dept"].upper() if exp["dept"] else None
        hit = any(
            (exp_dept == "ANY" or dept == exp_dept)
            and abs(value - exp["value"]) <= _tol(exp["value"])
            for dept, value, _ in cited
        )
        if not hit:
            numeric_ok = False
            row["failures"].append(
                f"numeric: expected {exp['metric']}={exp['value']} for {exp['dept'] or 'overall'}, not cited"
            )
    # expected departments named in the answer
    for dept in q.get("expect_depts", []):
        if dept.upper() not in {d.upper() for d in report.departments_referenced}:
            numeric_ok = False
            row["failures"].append(f"numeric: {dept} not in departments_referenced")

    # --- refusal
    refusal_ok = report.unanswerable == q["expect_unanswerable"]
    if not refusal_ok:
        verb = "refused an answerable" if report.unanswerable else "answered an unanswerable"
        row["failures"].append(f"refusal: {verb} question")

    # --- fabrication: every cited value must appear in a tool return from
    # this conversation (strongest check), or failing that in the raw data
    returned: set[float] = set()
    for call in result.tool_calls:
        if call.get("output") is not None:
            _collect_numbers(call["output"], returned)
    provenance = returned | universe
    fabricated = [
        f"{metric}={value} ({dept or 'overall'})"
        for dept, value, metric in cited
        if not any(abs(value - u) <= _tol(value) for u in provenance)
    ]
    fabrication_ok = not fabricated
    if fabricated:
        row["failures"].append(f"fabrication: cited value(s) not in the data: {fabricated}")

    row["checks"] = {
        "tool_selection": tool_ok,
        "numeric": numeric_ok,
        "refusal": refusal_ok,
        "fabrication": fabrication_ok,
    }
    row["pass"] = all(row["checks"].values())
    row["usage"] = result.usage
    row["trace"] = result.tool_calls
    row["answer"] = report.answer
    row["unanswerable"] = report.unanswerable
    row["confidence"] = report.confidence
    return row


# ------------------------------------------------------------------ running


def run_one(q: dict, model: str, effort: str, retries: int = 2) -> AgentResult:
    for attempt in range(retries + 1):
        result = ask(q["q"], model=model, effort=effort)
        if result.error and result.error.startswith(("rate_limited", "connection_error", "api_error_5")):
            wait = 15 * (attempt + 1)
            print(f"  {result.error.split(':')[0]}, retrying in {wait}s")
            time.sleep(wait)
            continue
        return result
    return result


def pct(num: int, den: int) -> str:
    return f"{100 * num / den:.0f}%" if den else "n/a"


def _usage_summary(rows: list[dict]) -> dict | None:
    used = [r["usage"] for r in rows if r.get("usage")]
    if not used:
        return None
    n = len(used)
    total_cost = sum(u["cost_usd"] or 0 for u in used)
    return {
        "questions_with_usage": n,
        "total_cost_usd": round(total_cost, 2),
        "mean_cost_usd": round(total_cost / n, 4),
        "mean_seconds": round(sum(u["seconds"] for u in used) / n, 1),
        "mean_output_tokens": round(sum(u["output_tokens"] for u in used) / n),
        "mean_iterations": round(sum(u["iterations"] for u in used) / n, 1),
    }


def write_results(rows: list[dict], model: str, effort: str, tag: str = "") -> None:
    n = len(rows)
    checks = ("tool_selection", "numeric", "refusal", "fabrication")
    totals = {c: sum(r["checks"][c] for r in rows) for c in checks}
    passed = sum(r["pass"] for r in rows)

    by_cat: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_cat[r["category"]].append(r)

    lines = [
        "# Eval results",
        "",
        f"Generated by `python -m eval.run_eval` on {date.today().isoformat()}, model `{model}`, effort `{effort}`, {n} questions.",
        "",
        "## Headline",
        "",
        "| Check | Score |",
        "|---|---|",
        f"| Overall pass (all checks) | **{pct(passed, n)}** ({passed}/{n}) |",
        f"| Tool selection | {pct(totals['tool_selection'], n)} |",
        f"| Numeric accuracy | {pct(totals['numeric'], n)} |",
        f"| Refusal accuracy | {pct(totals['refusal'], n)} |",
        f"| No fabricated figures | {pct(totals['fabrication'], n)} |",
        "",
        *(
            [
                "## Cost and latency",
                "",
                "| Total cost | Mean cost / question | Mean latency | Mean output tokens | Mean tool-loop turns |",
                "|---|---|---|---|---|",
                f"| ${u['total_cost_usd']} | ${u['mean_cost_usd']} | {u['mean_seconds']}s | {u['mean_output_tokens']} | {u['mean_iterations']} |",
                "",
            ]
            if (u := _usage_summary(rows))
            else []
        ),
        "## By category",
        "",
        "| Category | Passed | Tool | Numeric | Refusal |",
        "|---|---|---|---|---|",
    ]
    for cat in ("single_lookup", "comparison", "multi_hop", "ambiguous", "unanswerable", "trap"):
        cat_rows = by_cat.get(cat, [])
        if not cat_rows:
            continue
        m = len(cat_rows)
        lines.append(
            f"| {cat} | {sum(r['pass'] for r in cat_rows)}/{m} "
            f"| {pct(sum(r['checks']['tool_selection'] for r in cat_rows), m)} "
            f"| {pct(sum(r['checks']['numeric'] for r in cat_rows), m)} "
            f"| {pct(sum(r['checks']['refusal'] for r in cat_rows), m)} |"
        )

    failures = [r for r in rows if not r["pass"]]
    lines += ["", "## Failures", ""]
    if not failures:
        lines.append("None on this run.")
    for r in failures:
        lines.append(f"### {r['id']} ({r['category']})")
        lines.append(f"> {r['q']}")
        lines.append("")
        for f in r["failures"]:
            lines.append(f"- {f}")
        if r.get("answer"):
            lines.append(f"- answer given: {r['answer']}")
        lines.append("")

    suffix = f"-{tag}" if tag else ""
    (EVAL_DIR / f"results{suffix}.md").write_text("\n".join(lines) + "\n")
    (EVAL_DIR / f"results{suffix}.json").write_text(json.dumps(rows, indent=1))
    print(f"\nwrote eval/results{suffix}.md and eval/results{suffix}.json — {passed}/{n} passed")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--only", default=None, help="question id or category")
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--effort", default="high", choices=["low", "medium", "high", "xhigh", "max"])
    parser.add_argument("--questions", default=None, help="path to a questions .jsonl (default: eval/questions.jsonl)")
    parser.add_argument("--tag", default="", help="suffix for the results files, e.g. opus-high")
    args = parser.parse_args()

    qpath = Path(args.questions) if args.questions else QUESTIONS
    questions = [json.loads(line) for line in qpath.read_text().splitlines() if line.strip()]
    if args.only:
        questions = [q for q in questions if args.only in (q["id"], q["category"])]
    if args.limit:
        questions = questions[: args.limit]

    universe = data_universe()
    rows = []
    for i, q in enumerate(questions, 1):
        print(f"[{i}/{len(questions)}] {q['id']}: {q['q']}")
        result = run_one(q, args.model, args.effort)
        row = score_question(q, result, universe)
        status = "PASS" if row["pass"] else "FAIL — " + "; ".join(row["failures"])
        print(f"  {status}")
        rows.append(row)

    write_results(rows, args.model, args.effort, args.tag)


if __name__ == "__main__":
    main()
