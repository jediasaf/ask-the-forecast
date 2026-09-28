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
| Agent · `opus-5` · high · schema v1 | 13/14 (93%) | one fabrication flag: a *date* cited as the number `20131225.0` |
| Agent · `opus-5` · high · **schema v2** | **14/14 (100%)** | nothing — the fix below, re-measured |
| Prompt-stuffing · `opus-5` · high | **6/14 (43%)** | honestly refused every question needing arithmetic |

**Scale set — 12 questions** over the raw-M5 layer (ten stores, 2,652 SKUs):

| Configuration | Pass | Notes |
|---|---|---|
| Agent · `opus-5` · high | **12/12** | $0.029/question — includes cross-granularity traps and model-vs-baseline scope refusals |

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
- **The agent's one hard-set failure was found, fixed, and re-measured.**
  Asked for FOODS_3's worst day, it answered perfectly (Christmas 2013 —
  store closed, actual 0, model predicted 2,431.5 units) but shoehorned the
  date into the float-typed `value` field as `20131225.0`, which the
  fabrication check rightly flagged. The fix — a dedicated `date` field on
  cited figures, plus one system-prompt line — took the hard set from 13/14
  to 14/14 on re-run. The full loop (eval catches → diagnose as schema
  design → fix → re-measure) is in the commit history.
- **Once, the eval broke before the agent did.** A rebuild of the scale
  data shifted two store volumes by ~3k units; the first scale run "failed"
  two questions where the agent cited the *correct new* figures against my
  *stale* expectations. The failure signature — numeric check failing while
  the fabrication check passed — localizes the bug to the eval, not the
  agent. Expected values are data; they version with the data.
- **Refusals held everywhere.** Across all nine runs — 44 unanswerable/trap
  question-instances — nothing invented an answer to a question the data
  cannot support.

## How it works

```
agent/
  tools.py        17 strict-schema tools: 10 lookups over precomputed JSON,
                  3 computations over the daily series, 4 over the raw-M5
                  scale layer (10 stores, 2,652 SKUs)
  agent.py        the loop — Anthropic SDK tool runner, claude-opus-5,
                  per-run token/cost/latency accounting
  schema.py       structured output: answer, figures_cited, confidence, unanswerable
  glossary.md     metric definitions (WAPE, FVA, bias sign convention, ...)
  retrieval.py    keyword retrieval over the glossary — 11 entries need no vector DB
eval/
  questions.jsonl       37 main questions with expected tools, values, refusals
  questions_hard.jsonl  14 multi-hop questions; expected values computed by an
                        independent script, not by the agent path
  run_eval.py           scores tool selection, numerics, refusals, fabrication
  run_baseline.py       the same questions with prompt-stuffing instead of tools
  results-*.md          generated — one per configuration
scripts/
  record_traces.py      records real runs for the portfolio's replay view
  build_scale_data.py   builds data/scale/ from raw M5 (datasetsforecast mirror)
data/                   six JSON files from the portfolio backtest
data/scale/             store/SKU aggregates computed from raw M5
```

Design decisions that matter:

- **The model never does arithmetic.** The system prompt forbids it, every
  figure must be listed in `figures_cited`, and the eval verifies each cited
  value against the tool returns of that same conversation — a number the
  tools never produced is flagged as fabricated.
- **`strict: true` on all seventeen tools**, with `additionalProperties:
  false` and `required` populated, so inputs validate exactly. The three
  computed tools do deterministic WAPE arithmetic in Python — computation
  lives in code, never in the model.
- **Structured output** via `output_config.format` (JSON schema), with a
  load-bearing `unanswerable` path — the data has no future forecasts, no
  revenue, no headcount, and the agent must say so instead of improvising.
- **Retrieval is small and honest**: keyword matching over an 11-entry
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

The scale layer is scoped honestly: the portfolio model's forecasts exist
only for CA_1's seven departments, so store- and SKU-level tools serve
actuals and seasonal-naive metrics computed from raw M5 — and the agent must
refuse "model accuracy for store X". Two cross-pipeline checks anchor the
computation: CA_1's per-department naive accuracies computed from raw M5
reproduce the portfolio's stored figures exactly (FOODS_3 87.1, HOBBIES_2
54.9, all seven), and CA_1's window volume lands on planningOverall's
1,153,225 to the unit. Store metrics are computed on aggregated daily
series; SKU metrics on each item's own series — the first build pooled
SKU-day errors into "store accuracy" of 16%, a granularity mistake the
aggregation now prevents and the glossary warns the agent about.

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
.venv/bin/python -m eval.run_eval --questions eval/questions_scale.jsonl --tag scale
.venv/bin/python -m eval.run_baseline     # the no-tools comparison
.venv/bin/python -m scripts.build_scale_data   # rebuild data/scale from raw M5
```

The full grid above cost about $10 in API spend to measure.

## Reading a chart, and checking it

The agent also takes an image. It is not asked to read the chart; it is asked to
**verify** it.

```python
from agent.agent import ask
from agent.charts import render

ask("Does every figure on this slide match the system of record?",
    image=render(metric="fa"))
```

A chart from a review deck is a claim someone else made. It can be stale, or
edited, or simply wrong. So the vision rules tell the model to treat the image
as a claim and the tools as the record: read what the slide states, call the
tools for the same figures, and report any disagreement with both numbers and
which one is authoritative. A figure read off an image may never enter
`figures_cited` — that field is for numbers a tool returned, and a number that
exists only in a picture is a claim about data, not data.

### How it is scored

`agent/charts.py` renders charts **from the same JSON the tools serve**, so the
true value behind every bar is known exactly. `tamper=("FOODS_3", 91.4)` then
alters one bar's label and leaves the underlying data untouched. Nothing marks
the altered bar.

```bash
.venv/bin/python -m eval.run_vision_eval
```

Ten cases, half faithful and half with a planted error, scoring three things:

| | |
| --- | --- |
| **detection** | on a tampered chart, did it flag the altered figure? |
| **correctness** | did it report the real value, which can only come from a tool? |
| **false alarms** | on a **clean** chart, did it stay quiet? |

The third is the one that is usually missing. A detector that shouts
"discrepancy" at every chart scores 100% on detection and is useless; precision
only means anything if the clean cases are scored too.

It earned its place on the first run. A correct answer — *"Every number on the
slide matches the system of record. No discrepancies."* — was failed as a false
alarm, because the keyword check saw "discrepanc" inside "No discrepancies".
The bug was in the scorer, not the agent. It is negation-aware now and
`tests/test_charts.py` locks that in.

### Result

`claude-opus-5`, effort high, 2026-09-28:

| Check | Result |
| --- | --- |
| Detected the planted error | 6/6 |
| Reported the real figure too | 6/6 |
| Stayed quiet on a clean chart | 4/4 |
| **Passed everything** | **10/10** |

Full breakdown in [`eval/results-vision.md`](eval/results-vision.md).

**Calibrate that properly.** Ten cases is a small set and a clean sweep on it is
encouraging, not conclusive. The planted errors are deliberately unambiguous —
a bar relabelled by several points, not by a rounding — because the task under
test is *verification against a system of record*, not anomaly spotting. A
harder set would move a figure by less than a point, alter a total rather than a
bar, or tamper with a chart whose metric has no tool behind it at all. The
interesting number would be where it starts to fail, and this set does not find
that edge.

## Use the tools from any MCP client

The seventeen tools are served over the [Model Context
Protocol](https://modelcontextprotocol.io), so any MCP host can call them and
bring its own model. **No API key is needed to run the server** — every tool is
a lookup over bundled data, and the model lives on the client side.

Add this to `claude_desktop_config.json` (Claude Desktop → Settings → Developer
→ Edit Config), restart, and the tools appear:

```json
{
  "mcpServers": {
    "ask-the-forecast": {
      "command": "uvx",
      "args": ["--from", "git+https://github.com/jediasaf/ask-the-forecast",
               "ask-the-forecast-mcp"]
    }
  }
}
```

Nothing to clone and nothing to build: `uvx` fetches and runs it. The same
command works for any MCP client that speaks stdio.

Then ask it things like:

- *Which departments beat the naive baseline?*
- *What's the bias on FOODS_3, and is it getting worse across the folds?*
- *Which SKUs at CA_1 have the worst accuracy at meaningful volume?*

It will refuse questions the data cannot answer — there is no forward forecast
in here, and it will say so rather than produce one.

If you would rather install it properly:

```bash
pip install git+https://github.com/jediasaf/ask-the-forecast
ask-the-forecast-mcp        # stdio
```

The adapter reads `ALL_TOOLS` and forwards each tool's name, description and
schema, so the MCP contract and the agent's own contract cannot drift apart.

## CI

Two workflows, split by what they cost.

`ci.yml` runs on every push and pull request and is free and deterministic: no
API key, no model call, nothing that can flake. It runs the tool contract tests,
asserts the MCP adapter still advertises every agent tool, and runs the eval
regression gate.

```bash
.venv/bin/python -m pytest tests/ -q
.venv/bin/python -m eval.check_regression
```

The gate reads `eval/baselines.json`, a floor for each committed results file,
and fails if any run drops below it **or if a question set shrinks** — a
shrinking eval is how a score improves by accident. Raise a floor only when a
live run has genuinely beaten it (`--update` rewrites them).

`eval-live.yml` is the real thing and is manual only, because it calls a paid
API. Trigger it after a change to the agent, the prompt or the tools; it needs
the `ANTHROPIC_API_KEY` repository secret.

## Limitations, honestly

After the schema fix, `opus-5` at high effort passes every set — main,
hard, and scale. The discriminants that keep the grid honest are sonnet's
reproducible discipline failures, low-effort's fabrication-inside-a-refusal,
the schema-v1 flaw (caught, fixed, re-measured), and the stuffing baseline's
collapse on computation. The remaining ceilings: single-turn questions only
(no conversation state), one person wrote both the agent and the eval (the
script-computed ground truth mitigates but does not eliminate that), and
SKU-level tools cover CA_1 only — the model's forecasts don't exist
elsewhere, and inventing them would be exactly what this project refuses
to do.

## Data

Public Walmart M5 competition data. The planning layer (CA_1, 7 departments,
2013-05-05 → 2014-01-22) is precomputed in the portfolio repo; the scale
layer (10 stores, 2,652 SKUs) is computed from the raw M5 files by
`scripts/build_scale_data.py` over the same window. Ground truth was never
in the agent's prompt.
