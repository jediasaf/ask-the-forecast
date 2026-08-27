"""Keyword retrieval over the metric glossary.

Deliberately small and honest: a dozen entries do not need a vector database.
Each glossary section declares its own keyword list; a question is scored by
keyword and heading-word overlap and the top entries are injected into the
system prompt.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

GLOSSARY_PATH = Path(__file__).resolve().parent / "glossary.md"

_WORD = re.compile(r"[a-z0-9_]+")


@dataclass
class Entry:
    title: str
    keywords: set[str]
    body: str


def _tokenize(text: str) -> set[str]:
    return set(_WORD.findall(text.lower()))


def load_entries(path: Path = GLOSSARY_PATH) -> list[Entry]:
    text = path.read_text()
    entries = []
    # split on ## headings; skip the preamble before the first one
    for chunk in re.split(r"^## ", text, flags=re.M)[1:]:
        lines = chunk.strip().splitlines()
        title = lines[0].strip()
        body_lines = lines[1:]
        keywords: set[str] = set()
        if body_lines and body_lines[0].startswith("keywords:"):
            raw = body_lines[0][len("keywords:") :]
            keywords = {k.strip().lower() for k in raw.split(",") if k.strip()}
            body_lines = body_lines[1:]
        body = "\n".join(body_lines).strip()
        entries.append(Entry(title=title, keywords=keywords, body=body))
    return entries


def retrieve(question: str, k: int = 2, entries: list[Entry] | None = None) -> list[Entry]:
    """Top-k glossary entries for a question, by keyword overlap.

    Multi-word keywords match as substrings of the question; single words
    match on token overlap. Heading words count double. Entries scoring zero
    are never returned.
    """
    entries = entries if entries is not None else load_entries()
    q_lower = question.lower()
    q_tokens = _tokenize(question)

    scored = []
    for entry in entries:
        score = 0.0
        for kw in entry.keywords:
            if " " in kw:
                if kw in q_lower:
                    score += 2.0
            elif kw in q_tokens:
                score += 1.0
        score += 2.0 * len(_tokenize(entry.title) & q_tokens)
        if score > 0:
            scored.append((score, entry))

    scored.sort(key=lambda pair: -pair[0])
    return [entry for _, entry in scored[:k]]


def format_entries(entries: list[Entry]) -> str:
    if not entries:
        return ""
    parts = [f"### {e.title}\n{e.body}" for e in entries]
    return "Relevant metric definitions:\n\n" + "\n\n".join(parts)
