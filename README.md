# Ask the forecast — an LLM planning agent that never does the math

A planning agent that answers natural-language questions about forecast
performance by **calling tools over real data** — never by recalling or
computing numbers. Built on the same Walmart M5 backtest as
[forecasting-portfolio](https://github.com/jediasaf/forecasting-portfolio),
and held to the same standard: it reports its own failure modes, measured.

> *"Which departments is the model losing to the naive baseline, and why?"*

The agent calls `compare_departments("fva")`, reads the ranking the tool
returned, and answers with the real figures — the pooled −5.7pp value add,
the five departments losing to seasonal naive, the two that beat it. Every
number in the answer traces to a tool return, and the eval harness checks
that mechanically. A [live replay of recorded runs](https://jediasaf.vercel.app)
is on the portfolio site.

## The experiment grid

Two question sets, two architectures, measured 2026-08-27 by
[`eval/run_eval.py`](eval/run_eval.py) and
[`eval/run_baseline.py`](eval/run_baseline.py) — never restated by hand.
Four checks per question: right tools called, cited figures correct, refusal
decision exact, and every cited number present in an actual tool return.

**Main set — 37 questions** (lookups, comparisons, 2-hop chains, refusals):

| Configuration | Pass | Cost / question | Latency | Failure modes |
|---|---|---|---|---|
| Agent · `opus-5` · high | **37/37** | $0.037 | 16.9s | none |
| Agent · `opus-5` · low | 36/37 | $0.024 | 11.2s | fabricated a figure *inside a correct refusal* |
| Agent · `sonnet-5` · high | 35/37 | $0.012 | 10.3s | citation discipline; costlier tool path — **reproduced exactly across two runs** |
| Prompt-stuffing · `opus-5` · high | 37/37 | $0.082 | 8.5s | none — but see below |

**Hard set — 14 questions** needing computation over the daily series
(fold-over-fold trends, date-window accuracy, worst-day analysis):

| Configuration | Pass | What broke |
|---|---|---|
| Agent · `opus-5` · high | **13/14 (93%)** | one fabrication flag: a *date* cited as the number `20131225.0` |
| Prompt-stuffing · `opus-5` · high | **6/14 (43%)** | honestly refused every question needing arithmetic |

### What the grid says

- **Tools are not about correctness on small data — they're about cost and
  reach.** On the main set, stuffing all 180KB of data into a (cached)
  prompt matches the agent's 37/37, at 2.2× the cost. On the hard set the
  architectures diverge completely: the stuffing baseline, correctly
  forbidden from doing arithmetic, can only refuse — the agent's computed
  tools (`get_window_accuracy`, `get_fold_accuracy`, `get_worst_days`)
  convert those refusals into answered questions. Tools moved half the hard
  set from *unanswerable* to *answered*.
- **Effort buys discipline at the margins, not capability.** Low-effort opus
  still passes 36/37, but its one failure is telling: it refused an
  unanswerable question correctly *and fabricated a supporting figure while
  doing so* — a sloppiness mode high effort doesn't show.
- **Sonnet's failures are stable, not noise.** Two runs, same two questions,
  same two failure modes: a correct figure left out of the structured
  `figures_cited` audit trail, and a valid-but-costlier tool strategy
  (three `get_value_add` calls where one `compare_departments` sufficed).
- **The agent's own hard-set failure is a schema lesson.** Asked for
  FOODS_3's worst day, it answered perfectly (Christmas 2013 — store
  closed, actual 0, model predicted 2,431.5 units) but shoehorned the date
  into the float-typed `value` field as `20131225.0`, which the fabrication
  check rightly flagged. A `value` field that only accepts numbers invites
  dates-as-numbers; the fix is a typed figure schema.
- **Refusals held everywhere.** Across all six runs — 33 unanswerable/trap
  question-instances — nothing invented an answer to a question the data
  cannot support.

## How it works

```
agent/
  tools.py        13 strict-schema tools: 10 lookups over precomputed JSON,
                  3 deterministic computations over the daily series
  agent.py        the loop — Anthropic SDK tool runner, claude-opus-5,
                  per-run token/cost/latency accounting
  schema.py       structured output: answer, figures_cited, confidence, unanswerable
  glossary.md     metric definitions (WAPE, FVA, bias sign convention, ...)
  retrieval.py    keyword retrieval over the glossary — 10 entries need no vector DB
eval/
  questions.jsonl       37 main questions with expected tools, values, refusals
  questions_hard.jsonl  14 multi-hop questions; expected values computed by an
                        independent script, not by the agent path
  run_eval.py           scores tool selection, numerics, refusals, fabrication
  run_baseline.py       the same questions with prompt-stuffing instead of tools
  results-*.md          generated — one per configuration
scripts/
  record_traces.py      records real runs for the portfolio's replay view
data/                   six JSON files from the portfolio backtest
```

Design decisions that matter:

- **The model never does arithmetic.** The system prompt forbids it, every
  figure must be listed in `figures_cited`, and the eval verifies each cited
  value against the tool returns of that same conversation — a number the
  tools never produced is flagged as fabricated.
- **`strict: true` on all thirteen tools**, with `additionalProperties:
  false` and `required` populated, so inputs validate exactly. The three
  computed tools do deterministic WAPE arithmetic in Python — computation
  lives in code, never in the model.
- **Structured output** via `output_config.format` (JSON schema), with a
  load-bearing `unanswerable` path — the data has no future forecasts, no
  revenue, no headcount, and the agent must say so instead of improvising.
- **Retrieval is small and honest**: keyword matching over a 10-entry
  glossary, injected per question. A vector database for a dozen definitions
  would be decoration.
- **Parallel tool calls** are handled by the SDK tool runner, which returns
  all `tool_result` blocks in a single user message — splitting them across
  messages silently teaches the model to stop parallelising.

## The eval

Main set, six categories:

| Category | N | Example | Required behaviour |
|---|---|---|---|
| single lookup | 9 | "What's the bias on FOODS_3?" | cite the exact tool-returned figure |
| comparison | 7 | "Which departments beat naive?" | rank via tools, name the departments |
| multi-hop | 7 | "Of the departments losing to naive, which has the highest volume?" | chain calls |
| ambiguous | 4 | "How's the forecast doing?" | scope explicitly, then answer |
| unanswerable | 5 | "What's the forecast for next quarter?" | decline — no future data exists |
| trap | 5 | "What's the revenue per head?" | refuse — the data cannot support it |

The hard set was written adversarially against the tool surface after the
main set saturated: fold-trend questions ("which department deteriorates
most steadily across the three folds?" — 21 tool calls), window paradoxes
("FOODS_1 loses overall — was that true in December?": no, +10.2pp), and
worst-day forensics (five departments share the same worst day: Christmas,
store closed, model predicting thousands of units). Expected answers were
computed by an independent script over the raw series, so the eval's ground
truth never came from the agent being evaluated.

## Running it

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
export ANTHROPIC_API_KEY=...              # or `ant auth login`; never committed
.venv/bin/python -m agent.agent "Which departments beat the naive baseline?"
.venv/bin/python -m eval.run_eval --tag opus-high
.venv/bin/python -m eval.run_eval --questions eval/questions_hard.jsonl --tag hard
.venv/bin/python -m eval.run_baseline     # the no-tools comparison
```

The full grid above cost about $8 in API spend to measure.

## Limitations, honestly

The main set saturates frontier models — at high *and* low effort — because
seven departments cap lookup-question difficulty. The hard set restored a
real failure rate by requiring computation, but its ceiling is the single
store: per-SKU (3,049 items) and multi-store tools are the next scale step,
and the expected result is a lower pass rate and a richer failure taxonomy.
The eval author and agent author are the same person; the hard set's
script-computed ground truth mitigates but does not eliminate that.

## Data

Public Walmart M5 competition data (California store 1, 7 departments, daily
actuals and forecasts 2013-05-05 → 2014-01-22), precomputed at build time in
the portfolio repo. Ground truth was never in the agent's prompt.
