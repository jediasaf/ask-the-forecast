# Ask the forecast — an LLM planning agent that never does the math

A planning agent that answers natural-language questions about forecast
performance by **calling tools over real data** — never by recalling or
computing numbers. Built on the same Walmart M5 backtest as
[forecasting-portfolio](https://github.com/jediasaf/forecasting-portfolio),
and held to the same standard: it reports its own failure modes, measured.

> *"Which departments is the model losing to the naive baseline, and why?"*

The agent calls `compare_departments("fva")`, reads the ranking the tool
returned, and answers with the real figures — the pooled −5.7pp value add, the
five departments losing to seasonal naive, the two (HOBBIES_1 +2.8pp,
HOBBIES_2 +10.6pp) that beat it. Every number in the answer traces to a tool
return, and the eval harness checks that mechanically.

## Measured results

37 questions, four checks per question, measured 2026-08-27 by
[`eval/run_eval.py`](eval/run_eval.py) — never restated by hand:

| Run | Overall | Tool selection | Numeric | Refusal | Fabricated figures |
|---|---|---|---|---|---|
| `claude-opus-5` · effort high | **37/37 (100%)** | 100% | 100% | 100% | 0 |
| `claude-sonnet-5` · effort high | 35/37 (95%) | 97% | 97% | 100% | 0 |
| `claude-opus-5` · effort low | 37/37 (100%) | 100% | 100% | 100% | 0 |

Full breakdowns: [opus/high](eval/results-opus.md) ·
[sonnet/high](eval/results-sonnet.md) · [opus/low](eval/results-opus-low.md).

### Failure analysis

The headline run is perfect, which is why the comparison rows exist — a
perfect score is only meaningful if the harness can be shown to fail things.
It can:

- **Citation discipline** (`sonnet-5`, single-08): asked for Prophet's WAPE on
  HOBBIES_2, the model gave the correct figure *and* the Elastic Net caveat in
  prose — but left the second figure out of the structured `figures_cited`.
  The number was right; the audit trail was incomplete. That distinction is
  exactly what the structured-output contract exists to catch.
- **Valid-but-unexpected tool paths** (`sonnet-5`, comp-05): asked to rank the
  three FOODS departments, it called `get_value_add` three times instead of
  `compare_departments` once. The answer was correct; the tool-selection
  metric scores strategy, not just correctness, and flags the costlier path.
- **Effort is not the binding constraint** (`opus-5` at effort low): dropping
  reasoning effort to minimum changed nothing — still 37/37. The eval's
  difficulty ceiling, not model capability, is what saturates.
- The **mocked-failure tests** used to develop the scorer (wrong tool, wrong
  value, fabricated figure, wrongly-answered unanswerable) are all caught —
  the harness was proven able to fail answers before any live run.

The honest limitation: at high effort, this eval no longer separates frontier
models. Seven departments and ten tools give a two-hop ceiling on question
difficulty. The next iteration that would restore a meaningful failure rate is
a larger tool surface — per-SKU accuracy, multi-store comparison — where
chains get long enough to break.

## How it works

```
agent/
  tools.py        10 strict-schema tools over the backtest JSON — pure lookups
  agent.py        the loop — Anthropic SDK tool runner, claude-opus-5
  schema.py       structured output: answer, figures_cited, confidence, unanswerable
  glossary.md     metric definitions (WAPE, FVA, bias sign convention, ...)
  retrieval.py    keyword retrieval over the glossary — 10 entries need no vector DB
eval/
  questions.jsonl 37 questions with expected tools, values, and refusals
  run_eval.py     scores tool selection, numeric accuracy, refusals, fabrication
  results*.md     generated — one per model × effort configuration
data/             six JSON files from the portfolio backtest (ground truth)
```

Design decisions that matter:

- **The model never does arithmetic.** The system prompt forbids it, every
  figure must be listed in `figures_cited`, and the eval cross-checks each
  cited value against the data files — a number that exists nowhere in the
  data is flagged as fabricated.
- **`strict: true` on all ten tools**, with `additionalProperties: false` and
  `required` populated, so tool inputs validate exactly. Inputs are parsed
  from the `tool_use` blocks by the SDK — never string-matched.
- **Structured output** via `output_config.format` (JSON schema): `answer`,
  `departments_referenced`, `figures_cited`, `confidence`, `unanswerable`.
  The `unanswerable` path is load-bearing — the data has no future forecasts,
  no revenue, no headcount, and the agent must say so instead of improvising.
  Both models refused all 10 unanswerable/trap questions, and invented
  nothing on any run.
- **Retrieval is small and honest**: keyword matching over a 10-entry
  glossary, injected per question. A vector database for a dozen definitions
  would be decoration.
- **Parallel tool calls** are handled by the SDK tool runner, which returns
  all `tool_result` blocks in a single user message — splitting them across
  messages silently teaches the model to stop parallelising.

## The eval

| Category | N | Example | Required behaviour |
|---|---|---|---|
| single lookup | 9 | "What's the bias on FOODS_3?" | cite the exact tool-returned figure |
| comparison | 7 | "Which departments beat naive?" | rank via tools, name the departments |
| multi-hop | 7 | "Of the departments losing to naive, which has the highest volume?" | chain calls across tools |
| ambiguous | 4 | "How's the forecast doing?" | scope explicitly, then answer |
| unanswerable | 5 | "What's the forecast for next quarter?" | decline — no future data exists |
| trap | 5 | "What's the revenue per head?" | refuse — the data cannot support it |

Four checks per question: were the expected tools called; does every expected
figure appear in `figures_cited` for the right department; was the refusal
decision exactly right; and does every cited number exist in the data at all.

## Running it

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
export ANTHROPIC_API_KEY=...             # or `ant auth login`; never committed
.venv/bin/python -m agent.agent "Which departments beat the naive baseline?"
.venv/bin/python -m eval.run_eval        # full eval → eval/results.md
.venv/bin/python -m eval.run_eval --model claude-sonnet-5 --effort low
```

## Data

Public Walmart M5 competition data (California store 1, 7 departments, daily
actuals and forecasts 2013-05-05 → 2014-01-22), with all metrics precomputed
at build time in the portfolio repo. The agent's tools are lookups over those
files; ground truth was never in the prompt.
