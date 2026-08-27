"""Structured output definition for the agent's final answer.

The answer is constrained to a JSON schema (``output_config.format``) so it is
parseable, not prose — and so the eval harness can score the cited figures
mechanically. ``unanswerable`` is part of the contract: the agent must be able
to say the data cannot answer a question instead of inventing something.
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict


class FigureCited(BaseModel):
    """One number the answer relies on, traced to a tool return.

    ``value`` is strictly numeric; a date that supports the answer goes in
    ``date`` (ISO string). This split exists because the first hard-set run
    showed the model shoehorning 2013-12-25 into ``value`` as 20131225.0 —
    a schema that only offers a float field invites dates-as-numbers.
    """

    model_config = ConfigDict(extra="forbid")

    metric: str
    dept: Optional[str]  # null for pooled/overall figures
    value: float
    date: Optional[str] = None  # ISO date this figure belongs to, if any


class AnswerReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    answer: str
    departments_referenced: list[str]
    figures_cited: list[FigureCited]
    confidence: Literal["high", "medium", "low"]
    unanswerable: bool


def output_format() -> dict:
    """The ``output_config`` value enforcing AnswerReport on the final message."""
    return {
        "format": {
            "type": "json_schema",
            "schema": AnswerReport.model_json_schema(),
        }
    }
