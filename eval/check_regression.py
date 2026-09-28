"""Fail the build when a committed eval result drops below its baseline.

Why this is separate from ``run_eval``: running the real eval costs money and
needs an API key, so it cannot sensibly fire on every push. What CAN run on
every push is a check that the results already committed to the repo still meet
the bar they claim. That catches the two things that actually go wrong between
live runs: a results file edited or regenerated downward without anyone
noticing, and a scoring change that quietly makes the numbers look better.

Baselines live in ``eval/baselines.json`` as the minimum each committed run must
still hit. They are floors, not targets: raise one only when a live run has
genuinely beaten it.

    python -m eval.check_regression            # check every baseline
    python -m eval.check_regression --update   # rewrite floors from current results
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent
BASELINES = EVAL_DIR / "baselines.json"

CHECKS = ("tool_selection", "numeric", "refusal", "fabrication")


def score(path: Path) -> dict:
    """Return pass counts and question count for one results JSON."""
    rows = json.loads(path.read_text())
    n = len(rows)
    out = {"questions": n, "passed": sum(1 for r in rows if r.get("pass"))}
    for check in CHECKS:
        out[check] = sum(1 for r in rows if r.get("checks", {}).get(check))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--update", action="store_true",
                    help="rewrite the floors from the current results files")
    args = ap.parse_args()

    if args.update:
        floors = {}
        for path in sorted(EVAL_DIR.glob("results-*.json")):
            floors[path.name] = score(path)
        BASELINES.write_text(json.dumps(floors, indent=2) + "\n")
        print(f"wrote {len(floors)} baselines to {BASELINES.name}")
        return 0

    if not BASELINES.exists():
        print("no baselines.json; run with --update first", file=sys.stderr)
        return 1

    floors = json.loads(BASELINES.read_text())
    failures: list[str] = []

    for name, floor in sorted(floors.items()):
        path = EVAL_DIR / name
        if not path.exists():
            failures.append(f"{name}: results file is gone but a baseline still claims it")
            continue
        actual = score(path)
        if actual["questions"] != floor["questions"]:
            failures.append(
                f"{name}: question count changed {floor['questions']} -> {actual['questions']}; "
                "a shrinking eval set is how a score improves by accident"
            )
        for key in ("passed", *CHECKS):
            if actual[key] < floor[key]:
                failures.append(
                    f"{name}: {key} regressed {floor[key]} -> {actual[key]} "
                    f"of {actual['questions']}"
                )
        status = "ok" if not any(name in f for f in failures) else "FAIL"
        print(f"  {status:4s} {name}  passed {actual['passed']}/{actual['questions']}")

    if failures:
        print("\nregressions:", file=sys.stderr)
        for f in failures:
            print(f"  - {f}", file=sys.stderr)
        return 1

    print(f"\nall {len(floors)} committed eval runs still meet their floors")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
