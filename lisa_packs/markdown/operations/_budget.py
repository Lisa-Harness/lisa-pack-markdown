"""Target-length → token-budget translation for markdown pack operations.

Phase B of the markdown pack hardening: when a caller supplies a
``target_length`` (approximate word count), the operation derives the
first-pass ``max_tokens`` from it instead of using the flat 50k ceiling.
This tells the model — via both the token budget and an explicit prompt
line — how long the document should be, so mid-stream truncation becomes
rare and the continuation machinery shrinks from a primary path to a
safety net.

This module owns the single words→tokens translation so both the
``markdown_forge_document`` operation and (potentially) other callers
share one heuristic. The factor defaults to 1.3 (the standard English
heuristic) and is overridable per-workspace via the
``forge_tokens_per_word`` ``pack_config`` key.
"""

from __future__ import annotations

import logging
import math

logger = logging.getLogger(__name__)

__all__ = ["target_words_to_max_tokens"]

# Hard floor and ceiling on the derived budget. The floor keeps a tiny
# target from producing an unscannable document; the ceiling is the
# operation's existing ``_MAX_COMPLETION_TOKENS`` so a huge target cannot
# exceed the legacy ceiling (the continuation path already assumes it).
_MIN_MAX_TOKENS = 512
_MAX_MAX_TOKENS = 50_000


def target_words_to_max_tokens(
    words: int,
    *,
    tokens_per_word: float = 1.3,
    floor: int = _MIN_MAX_TOKENS,
    ceiling: int = _MAX_MAX_TOKENS,
) -> int:
    """Return a ``max_tokens`` budget for an approximate ``words`` target.

    ``words * tokens_per_word``, clamped to ``[floor, ceiling]`` and rounded
    up. ``tokens_per_word`` defaults to 1.3 (English heuristic); non-English
    content may need ~2.0 (override via ``forge_tokens_per_word``).
    """
    if words <= 0:
        return ceiling
    raw = words * max(float(tokens_per_word), 0.1)
    budget = int(math.ceil(raw))
    if budget < floor:
        return floor
    if budget > ceiling:
        return ceiling
    return budget