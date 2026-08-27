# Metric glossary

Retrieval source for the agent. Each `##` section is one entry; the first line
after the heading is a comma-separated keyword list used for matching.

## forecast accuracy (FA)
keywords: accuracy, fa, forecast accuracy, accurate, wape, error rate, how good

Forecast accuracy is 100 minus WAPE, in percent. It measures how close daily
unit forecasts were to actual sales over the backtest window. An FA of 79.5%
means the summed absolute daily errors were 20.5% of total actual units.
FA alone is close to meaningless without a baseline beside it — always read it
next to the seasonal-naive accuracy.

## WAPE
keywords: wape, weighted absolute percentage error, error, mape

Weighted Absolute Percentage Error: the sum of absolute errors divided by the
sum of actuals, in percent. Lower is better. Unlike MAPE it does not explode
on near-zero days, which matters for intermittent demand. FA = 100 − WAPE.

## forecast value add (FVA)
keywords: fva, value add, beat, baseline, better than naive, worth, losing, winning, earn

Forecast value add = model accuracy minus seasonal-naive accuracy, in
percentage points (pp). Positive means the model beats the baseline; negative
means simply repeating last week would have been more accurate. FVA is the
headline metric of the planning view: a model with negative FVA is costing
accuracy, whatever its standalone FA says.

## bias (sign convention)
keywords: bias, over-forecast, under-forecast, systematic, skew, direction

Signed bias in percent: (total forecast − total actual) / total actual.
Positive = over-forecasting (predicting more than sold, risking excess stock).
Negative = under-forecasting (risking stock-outs). Bias near zero can hide
large offsetting daily errors — read it together with FA, never instead of it.

## seasonal naive baseline
keywords: naive, baseline, seasonal naive, last week, same weekday, benchmark

The baseline every model is measured against: predict the same weekday one
week earlier. It costs nothing to run and encodes weekly seasonality for free,
which makes it a demanding baseline for stable retail demand.

## intermittent demand
keywords: intermittent, sparse, zeros, spiky, low volume, hobbies_2, croston

Demand that is mostly zeros with occasional spikes, typical of low-volume
departments. Smooth trend-plus-seasonality models fit it poorly, and the
seasonal naive is also a poor guess — which is why intermittent series are
where a model can most easily add value. Croston's method is the classical
tool for such series.

## volume
keywords: volume, units, size, biggest, smallest, largest, matters

Total units sold per department over the 263-day backtest window. Use it to
weight conclusions: a 9pp FVA loss on a 615k-unit department matters far more
than a 10pp win on a 6k-unit one.

## Prophet vs Elastic Net comparison
keywords: prophet, elastic net, elasticnet, model comparison, which model, daily, monthly

A separate benchmark from the planning metrics: Prophet forecast daily units,
Elastic Net forecast monthly aggregates. Monthly aggregation cancels noise,
so the two WAPEs are not comparable head-to-head — the study reports both and
says so. Do not present Elastic Net as "4x better" than Prophet.

## revenue concentration (Pareto)
keywords: pareto, concentration, revenue, sku, top items, 80/20, long tail

The share of realised revenue covered by the top N% of the 3,049 SKUs, ranked
by a revenue-weighted seasonal priority score. Stored as a curve at 5%-grid
points, e.g. the top 20% of SKUs cover 56% of revenue.

## data scope and limits
keywords: scope, cannot, unavailable, future, next quarter, predict, revenue per head, store, price, margin, headcount, customers

The data is a fixed historical backtest: public Walmart M5 data, California
store 1, 7 departments, daily actuals and forecasts for 263 days
(2013-05-05 → 2014-01-22), plus 38 weekly accuracy rows. There are NO future
forecasts, NO prices, margins or dollar revenue per department, NO headcount
or customer counts, and NO store- or SKU-level accuracy metrics. Questions
needing any of those are unanswerable from this data and must be declined,
not estimated.
