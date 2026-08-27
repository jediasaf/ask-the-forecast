# Ask the forecast — an LLM planning agent that never does the math

A planning agent that answers natural-language questions about forecast
performance by **calling tools over real data** — never by recalling or
computing numbers. Built on the same Walmart M5 backtest as
[forecasting-portfolio](https://github.com/jediasaf/forecasting-portfolio),
and held to the same standard: it reports its own failure modes, measured.

> *"Which departments is the model losing to the naive baseline, and why?"*

The agent calls `compare_departments("fva")`, reads the ranking a tool
returned, and answers with the real figures — the pooled −5.7pp value add, the
five departments losing to seasonal naive, the two (HOBBIES_1 +2.8pp,
HOBBIES_2 +10.6pp) that beat it. Every number in the answer traces to a tool
return; the eval harness checks that mechanically.

## Why this exists

"Built solutions with LLMs" has to mean the LLM is a **component of a system
you designed** — the prompts, the tools it can call, the output schema, and
the evaluation that tells you when it is wrong. A wrapper that pastes JSON
into a prompt and asks for prose is none of that. The plumbing here is the
point, not the chat.

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
  results.md      generated — the failure analysis
data/             six JSON files from the portfolio backtest (ground truth)
```

Design decisions that matter:

- **The model never does arithmetic.** The system prompt forbids it, every
  figure must be listed in `figures_cited`, and the eval cross-checks each
  cited value against the data files. A number that exists nowhere in the
  data is flagged as fabricated.
- **`strict: true` on all ten tools**, with `additionalProperties: false` and
  `required` populated, so tool inputs validate exactly. Inputs are parsed by
  the SDK from the `tool_use` blocks — never string-matched.
- **Structured output** via `output_config.format` (JSON schema): `answer`,
  `departments_referenced`, `figures_cited`, `confidence`, and `unanswerable`.
  The `unanswerable` path is load-bearing — the data has no future forecasts,
  no revenue, no headcount, and the agent must say so instead of improvising.
- **Retrieval is small and honest**: keyword matching over a 10-entry
  glossary, injected per question. A vector database for a dozen definitions
  would be decoration.
- **Parallel tool calls** are handled by the SDK tool runner, which returns
  all `tool_result` blocks in a single user message — splitting them across
  messages silently teaches the model to stop parallelising.

## The eval — what makes it real

Anyone can demo a working question. `eval/questions.jsonl` covers 37
questions in six deliberate categories:

| Category | Example | What it tests |
|---|---|---|
| single_lookup (9) | "What's the bias on FOODS_3?" | baseline |
| comparison (7) | "Which departments beat naive?" | multi-call |
| multi_hop (7) | "Of the departments losing to naive, which has the highest volume?" | chaining |
| ambiguous (4) | "How's the forecast doing?" | must scope, not sprawl |
| unanswerable (5) | "What's the forecast for next quarter?" | must decline, not invent |
| trap (5) | "What's the revenue per head?" | data can't support it — must refuse |

`run_eval.py` scores four things per question — did it call the right tools,
are the cited figures correct, did it refuse exactly the unanswerable ones,
and is every cited number actually present in the data — then writes the
failure breakdown to [`eval/results.md`](eval/results.md).

## Measured results

Run 2026-08-27, `claude-opus-5`, high effort: **37/37 passed — 100% on tool
selection, numeric accuracy, refusal accuracy, and zero fabricated figures.**
Full breakdown in [`eval/results.md`](eval/results.md), written by the harness
so the numbers can never drift from what was measured.

A perfect score deserves suspicion, so two things keep it honest. First, the
scorer's ability to fail answers is itself tested — mocked wrong-tool,
wrong-value, and fabricated-figure answers are all caught (that's how the
harness was developed, before any live run). Second, the same 37 questions
were re-run on `claude-sonnet-5` as a discriminant control — see
[`eval/results-sonnet.md`](eval/results-sonnet.md). The honest reading today:
this eval set no longer separates frontier models on a 7-department dataset;
the next hard step is scaling the tool surface (per-SKU, multi-store) until
the failure rate is nonzero again.

## Running it

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
export ANTHROPIC_API_KEY=...          # or `ant auth login`; never committed
.venv/bin/python -m agent.agent "Which departments beat the naive baseline?"
.venv/bin/python -m eval.run_eval     # full eval → eval/results.md
```

## Data

Public Walmart M5 competition data (California store 1, 7 departments,
2013-05-05 → 2014-01-22), with all metrics precomputed at build time in the
portfolio repo. The agent's tools are lookups over those files; ground truth
was never in the prompt.
