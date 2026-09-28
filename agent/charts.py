"""Render the planning data as chart images, optionally with a planted error.

This exists so the vision path can be *evaluated* rather than demonstrated.
Every image is generated from the same JSON the tools serve, so the true value
behind every bar is known exactly. ``tamper`` then changes one displayed number
without changing the underlying data, which gives a labelled test case: the
chart now disagrees with the source of truth by a known amount, in a known
place.

That is the whole point. An agent that reads a chart and repeats what it says
is doing OCR. An agent that reads a chart, checks it against the system of
record, and says "this slide claims 80.1 but the data says 74.5" is doing the
job. Only the second can be scored, and only if the planted errors are known.

    render(metric="fa")                      -> a faithful chart
    render(metric="fa", tamper=("FOODS_3", 91.4))  -> one bar is a lie
"""

from __future__ import annotations

import io
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no display needed, and none exists in CI
import matplotlib.pyplot as plt  # noqa: E402

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# What each chartable metric is called on the slide, and its axis label.
METRICS = {
    "fa": ("Forecast accuracy by department", "Forecast accuracy (%)"),
    "naiveFa": ("Seasonal-naive accuracy by department", "Naive accuracy (%)"),
    "fva": ("Forecast value add by department", "FVA (percentage points)"),
    "bias": ("Forecast bias by department", "Bias (%)"),
    "volume": ("Volume by department", "Units sold"),
}


def _depts() -> list[dict]:
    with open(DATA_DIR / "planningDepts.json") as f:
        return json.load(f)


def truth(metric: str) -> dict[str, float]:
    """The real value per department, straight from the data the tools serve."""
    return {row["dept"]: row[metric] for row in _depts()}


def render(
    metric: str = "fa",
    tamper: tuple[str, float] | None = None,
    title: str | None = None,
) -> bytes:
    """Return a PNG of one metric by department.

    ``tamper=(dept, shown_value)`` draws and labels that department at
    ``shown_value`` while every other bar stays truthful. The returned image is
    indistinguishable from a real slide: nothing marks the altered bar.
    """
    if metric not in METRICS:
        raise ValueError(f"unknown metric {metric!r}; expected one of {sorted(METRICS)}")

    default_title, ylabel = METRICS[metric]
    values = truth(metric)
    if tamper:
        dept, shown = tamper
        if dept not in values:
            raise ValueError(f"unknown department {dept!r}")
        values = {**values, dept: shown}

    order = sorted(values, key=lambda d: values[d], reverse=True)
    heights = [values[d] for d in order]

    fig, ax = plt.subplots(figsize=(9, 5), dpi=110)
    bars = ax.bar(order, heights, color="#4C78A8")
    ax.set_title(title or default_title, fontsize=14)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=30)
    ax.spines[["top", "right"]].set_visible(False)

    # Value labels: the figure a reader would quote, and the figure the agent
    # has to check. Without them this is a shape-reading task, not a claim-
    # verification task.
    for bar, value in zip(bars, heights):
        ax.annotate(
            f"{value:,.1f}" if metric != "volume" else f"{value:,.0f}",
            (bar.get_x() + bar.get_width() / 2, bar.get_height()),
            ha="center",
            va="bottom" if bar.get_height() >= 0 else "top",
            fontsize=10,
        )

    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    return buf.getvalue()


def save(path: str | Path, **kwargs) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(render(**kwargs))
    return out
