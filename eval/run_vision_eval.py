"""Score the vision path: can the agent catch a chart that disagrees with the data?

The cases are generated, not written, because that is the only way the ground
truth is exact. Each case renders a chart from the same JSON the tools serve.
Half are faithful. Half have one bar's label altered by a known amount, in a
known department, with the underlying data untouched.

Three things are scored, and the third is the one most evals skip:

    detection      on a tampered chart, did it flag the altered figure?
    correctness    did it report the real value from a tool, not the image's?
    false alarms   on a FAITHFUL chart, did it stay quiet?

A detector that shouts "discrepancy" at everything scores 100% on detection and
is useless. Precision only means something if the clean cases are scored too.

Usage:
    python -m eval.run_vision_eval [--limit N] [--model MODEL] [--effort LEVEL]

Writes eval/results-vision.md and eval/results-vision.json.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import date
from pathlib import Path

from agent.agent import MODEL, ask
from agent.charts import render, truth

EVAL_DIR = Path(__file__).resolve().parent

QUESTION = (
    "This chart is from our monthly forecast review deck. Does every figure on "
    "it match the system of record? If anything disagrees, say which department, "
    "what the slide claims, and what the data actually says."
)

# (case id, metric, tamper) - tamper None means the chart is faithful.
# The altered values are large enough to be unambiguous and small enough to be
# plausible on a real slide: this tests verification, not anomaly spotting.
CASES: list[tuple[str, str, tuple[str, float] | None]] = [
    ("fa-clean",        "fa",      None),
    ("fa-foods3",       "fa",      ("FOODS_3", 91.4)),
    ("fa-hobbies2",     "fa",      ("HOBBIES_2", 55.0)),
    ("naive-clean",     "naiveFa", None),
    ("naive-household1","naiveFa", ("HOUSEHOLD_1", 70.2)),
    ("fva-clean",       "fva",     None),
    ("fva-foods1",      "fva",     ("FOODS_1", 12.5)),
    ("bias-clean",      "bias",    None),
    ("bias-foods2",     "bias",    ("FOODS_2", -18.0)),
    ("volume-foods3",   "volume",  ("FOODS_3", 415655.0)),
]


# Words that appear when a mismatch is being asserted. Substring matching alone
# is not enough: the FIRST live run flagged a correct answer as a false alarm
# because "No discrepancies" contains "discrepanc". Scoring the clean cases is
# what surfaced that, which is the argument for scoring them at all.
_MISMATCH = ("disagree", "discrepanc", "mismatch", "does not match", "doesn't match",
             "not match", "incorrect", "differs", "wrong")
_NEGATORS = ("no ", "not ", "n't", "none", "zero", "without", "nothing", "never",
             "all match", "everything match")


def _claims_mismatch(answer: str) -> bool:
    """Does the answer actually assert a mismatch, rather than deny one?

    Crude but checkable: for every mismatch word, look back a short window for a
    negator. If every occurrence is negated, no claim was made. This will
    mis-score a sufficiently convoluted sentence; it is documented rather than
    hidden, and the failure mode it does catch is the common one.
    """
    low = answer.lower()
    for word in _MISMATCH:
        start = 0
        while (i := low.find(word, start)) != -1:
            window = low[max(0, i - 28):i]
            if not any(neg in window for neg in _NEGATORS):
                return True
            start = i + len(word)
    return False


def _mentions(text: str, value: float, metric: str) -> bool:
    """Is this figure present in the answer, in any of its plausible spellings?"""
    forms = {f"{value:,.1f}", f"{value:.1f}", f"{value:,.0f}", f"{value:.0f}"}
    if metric == "volume":
        forms |= {f"{int(value):,}", str(int(value))}
    return any(form in text for form in forms)


def score_case(case_id: str, metric: str, tamper, model: str, effort: str) -> dict:
    image = render(metric=metric, tamper=tamper)
    started = time.monotonic()
    result = ask(QUESTION, image=image, model=model, effort=effort)
    elapsed = round(time.monotonic() - started, 1)

    row = {
        "id": case_id,
        "metric": metric,
        "tampered": tamper is not None,
        "seconds": elapsed,
        "tools_called": [c["tool"] for c in (result.tool_calls or [])],
        "error": result.error,
    }
    if result.error or result.report is None:
        row["checks"] = {"detection": False, "correctness": False, "no_false_alarm": False}
        row["pass"] = False
        return row

    report = result.report
    answer = json.dumps(report.model_dump() if hasattr(report, "model_dump") else report)
    row["answer"] = answer[:1200]

    if tamper:
        dept, shown = tamper
        real = truth(metric)[dept]
        # Detection: it named the department AND quoted the figure the slide showed.
        detected = dept in answer and _mentions(answer, shown, metric)
        # Correctness: it also gave the real figure, which can only come from a tool.
        correct = detected and _mentions(answer, real, metric)
        checks = {"detection": detected, "correctness": correct, "no_false_alarm": True}
        row["expected"] = {"dept": dept, "slide_says": shown, "data_says": real}
    else:
        checks = {"detection": True, "correctness": True,
                  "no_false_alarm": not _claims_mismatch(answer)}

    row["checks"] = checks
    row["pass"] = all(checks.values())
    return row


def write_report(rows: list[dict], model: str, effort: str) -> str:
    tampered = [r for r in rows if r["tampered"]]
    clean = [r for r in rows if not r["tampered"]]

    def pct(n, d):
        return f"{n}/{d}" + (f" ({n / d:.0%})" if d else "")

    lines = [
        f"# Vision eval — {model}, effort {effort}",
        "",
        f"_{date.today().isoformat()} · {len(rows)} charts generated from the same data the tools serve._",
        "",
        "| Check | Result |",
        "| --- | --- |",
        f"| Detected the planted error | {pct(sum(r['checks']['detection'] for r in tampered), len(tampered))} |",
        f"| Reported the real figure too | {pct(sum(r['checks']['correctness'] for r in tampered), len(tampered))} |",
        f"| Stayed quiet on a clean chart | {pct(sum(r['checks']['no_false_alarm'] for r in clean), len(clean))} |",
        f"| Passed everything | {pct(sum(r['pass'] for r in rows), len(rows))} |",
        "",
        "## Cases",
        "",
        "| Case | Tampered | Detected | Correct | No false alarm | Tools |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for r in rows:
        c = r["checks"]
        tick = lambda b: "yes" if b else "**no**"  # noqa: E731
        lines.append(
            f"| `{r['id']}` | {'yes' if r['tampered'] else 'no'} | {tick(c['detection'])} "
            f"| {tick(c['correctness'])} | {tick(c['no_false_alarm'])} | {len(r['tools_called'])} |"
        )
    failures = [r for r in rows if not r["pass"]]
    if failures:
        lines += ["", "## Failures", ""]
        for r in failures:
            lines.append(f"- `{r['id']}`: {r.get('expected') or 'clean chart'} "
                         f"→ {r.get('error') or 'see results-vision.json'}")
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int)
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--effort", default="high")
    args = ap.parse_args()

    cases = CASES[: args.limit] if args.limit else CASES
    rows = []
    for case_id, metric, tamper in cases:
        row = score_case(case_id, metric, tamper, args.model, args.effort)
        rows.append(row)
        mark = "ok  " if row["pass"] else "FAIL"
        print(f"  {mark} {case_id:18s} {'tampered' if row['tampered'] else 'clean   '} "
              f"{row['seconds']:>5.1f}s")

    (EVAL_DIR / "results-vision.json").write_text(json.dumps(rows, indent=2) + "\n")
    (EVAL_DIR / "results-vision.md").write_text(write_report(rows, args.model, args.effort))
    passed = sum(r["pass"] for r in rows)
    print(f"\n{passed}/{len(rows)} passed · wrote eval/results-vision.md")


if __name__ == "__main__":
    main()
