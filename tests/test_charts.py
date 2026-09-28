"""Tests for the chart renderer and the vision plumbing.

These run without an API key. They cover the two things that would silently
invalidate the vision eval: a tamper that does not actually change the image,
and an image that never reaches the model.
"""

from __future__ import annotations

import pytest

from agent import charts
from agent.agent import _build_system, _content_blocks


# --------------------------------------------------------------- the renderer


def test_renders_a_png():
    png = charts.render(metric="fa")
    assert png[:8] == b"\x89PNG\r\n\x1a\n"


def test_truth_comes_from_the_same_data_the_tools_serve():
    t = charts.truth("fa")
    assert len(t) == 7
    assert all(isinstance(v, (int, float)) for v in t.values())


def test_tampering_actually_changes_the_image():
    """If a tampered chart rendered identically, every planted error would be
    undetectable and the eval would be scoring nothing."""
    clean = charts.render(metric="fa")
    dirty = charts.render(metric="fa", tamper=("FOODS_3", 91.4))
    assert clean != dirty


def test_tampering_does_not_touch_the_underlying_data():
    """The chart lies; the source of truth must not. This is what makes the
    case checkable: the agent has to reach past the image to a tool."""
    before = charts.truth("fa")["FOODS_3"]
    charts.render(metric="fa", tamper=("FOODS_3", 91.4))
    assert charts.truth("fa")["FOODS_3"] == before


def test_unknown_metric_and_department_are_rejected_loudly():
    with pytest.raises(ValueError):
        charts.render(metric="not_a_metric")
    with pytest.raises(ValueError):
        charts.render(metric="fa", tamper=("NOT_A_DEPT", 1.0))


# ------------------------------------------------------------ the vision path


def test_no_image_sends_a_plain_string():
    assert _content_blocks("hello", None) == "hello"


def test_an_image_is_sent_before_the_question():
    blocks = _content_blocks("hello", b"\x89PNG-bytes")
    assert [b["type"] for b in blocks] == ["image", "text"]
    assert blocks[0]["source"]["media_type"] == "image/png"
    assert blocks[0]["source"]["type"] == "base64"


def test_vision_rules_are_added_only_when_an_image_is_attached():
    """Attaching the rules unconditionally would change the cached prompt for
    every text question and pay for it on every call."""
    plain = _build_system("q", False)
    with_image = _build_system("q", True)
    assert len(with_image) == len(plain) + 1
    assert "verification" in with_image[1]["text"].lower()


def test_the_cached_system_prefix_is_unchanged_by_an_image():
    """The vision rules must come AFTER the cache breakpoint, or every image
    question invalidates the cache on the main prompt."""
    plain = _build_system("q", False)
    with_image = _build_system("q", True)
    assert with_image[0] == plain[0]
    assert with_image[0].get("cache_control") == {"type": "ephemeral"}


# ------------------------------------------------- the false-alarm scorer

from eval.run_vision_eval import _claims_mismatch  # noqa: E402


@pytest.mark.parametrize(
    "answer,expected",
    [
        # The first live run failed a correct answer because "No discrepancies"
        # contains "discrepanc". These lock that fix in.
        ("Every number on the slide matches the system of record. No discrepancies.", False),
        ("All seven bars are correct. Nothing differs from the data.", False),
        ("There were no mismatches.", False),
        ("FOODS_3 is incorrect: the slide claims 91.4 but the data says 80.1.", True),
        ("There is a discrepancy in HOBBIES_2.", True),
        ("The chart does not match the system of record.", True),
    ],
)
def test_false_alarm_detection_is_negation_aware(answer, expected):
    assert _claims_mismatch(answer) is expected
