"""Shared truncation/continuation machinery for markdown pack operations.

Both ``markdown_forge_document`` (single-doc forge) and
``markdown_consolidate_sources`` (multi-source consolidation) forge long
markdown documents and must handle the case where the model stops
mid-stream because it hit its token budget. This module owns that logic so
it is not duplicated across the two operations' per-path call sites.

The continuation contract (see
``/memories/repo/forge-document-truncation-contract.md``):

- Only truncate-flagged first passes (``finish_reason`` in
  ``{"length", "unknown", None}``) trigger continuation; a clean ``stop``
  returns the document as-is unless a structural-incompleteness override
  applies.
- A continuation that returns empty or duplicates the existing tail is
  treated as "model has nothing more to add" — stop and mark truncated.
- A merged document that ends on a clean boundary (``_looks_complete``)
  AND whose continuation reported a non-truncating ``finish_reason`` is
  considered complete; otherwise we keep going up to the budget.

``_load_text`` lives here because every continuation pass loads an authored
continue prompt via ``ctx.read_authored_text_fn``; the caller supplies the
exact ``continue_prompt_path`` so each operation can use its own prompt.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from kernel.models import OperationExecutionContext

logger = logging.getLogger(__name__)

__all__ = [
    "continue_until_complete",
    "extract_markdown_block",
    "is_truncating_finish",
    "coerce_finish_reason",
]

# Finish-reason values that signal the model stopped before completing
# the document. ``"length"`` is the OpenAI-standard "hit max_tokens"
# signal. ``"unknown"`` / ``None`` cover local backends (Ollama) that do
# not populate ``finish_reason`` — empirically these stop mid-stream on
# long documents. ``"stop"`` and ``"tool_calls"`` are NOT truncation.
_TRUNCATING_FINISH_REASONS = frozenset({"length", "unknown", None})


def _load_text(ctx: OperationExecutionContext, path: str) -> str:
    """Load authored markdown via the SDK content-registry loader.

    Fails loudly when the loader is not wired, returns empty/None, or raises.
    Per the pack-content rule: no fallbacks to missing content.
    """
    if not callable(ctx.read_authored_text_fn):
        raise RuntimeError(
            f"read_authored_text_fn is not wired; cannot load authored content at {path}"
        )
    try:
        loaded = str(ctx.read_authored_text_fn(path) or "").strip()
    except Exception as exc:  # pylint: disable=broad-except
        raise RuntimeError(f"read_authored_text_fn failed for {path}: {exc}") from exc
    if not loaded:
        raise RuntimeError(f"Authored text at {path} is empty or missing")
    return loaded


def coerce_finish_reason(raw: Any) -> str | None:
    """Normalize a provider finish_reason into a comparable string or None.

    Providers occasionally return ``""`` or whitespace for "no signal"; coerce
    those to ``None`` so they fall into the truncation set (a missing signal
    on a long document is empirically a truncation, not a clean stop).
    """
    if raw is None:
        return None
    text = str(raw).strip().lower()
    if not text:
        return None
    return text


def is_truncating_finish(finish_reason: str | None) -> bool:
    """Return True when the finish reason signals mid-stream truncation."""
    return finish_reason in _TRUNCATING_FINISH_REASONS


def extract_markdown_block(text: str) -> str:
    """Extract the first fenced ```markdown ...``` block, else return text.

    Accepts fenced blocks tagged ``markdown`` or untagged. Falls back to the
    raw text when no fence is present so the outline tool can still assess it.
    """
    match = re.search(r"```(?:markdown|md)?\s*\n(.*?)```", text, flags=re.DOTALL)
    if match:
        return match.group(1).strip()
    return text.strip()


def _looks_complete(document: str) -> bool:
    """Heuristic: does the document end on a clean paragraph/list boundary?

    Used after a continuation pass to decide whether the merged document is
    plausibly finished or whether we should try another continuation. A
    document ending mid-word, mid-sentence, or with a dangling list marker
    is treated as still-truncated.

    A heading as the very last content line is treated as TRUNCATED: the
    model opened a section ("### 5.1 ...") and stopped before writing its
    body.
    """
    if not document:
        return False
    stripped = document.rstrip()
    if not stripped:
        return False
    # Dangling list marker (e.g. "- " or "1. " at end of file) = truncated.
    if re.search(r"[\-\*\+] +$", stripped):
        return False
    # Dangling fenced code block (odd number of fences) = truncated.
    fence_count = len(re.findall(r"^```", stripped, flags=re.MULTILINE))
    if fence_count % 2 == 1:
        return False
    last_line = stripped.splitlines()[-1].rstrip()
    if not last_line:
        return True  # ends on a blank line — clean
    if last_line.endswith(("```", ".", "!", "?", ":", ";", ">", ")")):
        return True
    if re.match(r"^\s*[\-\*\+] +\S", last_line) or re.match(r"^\s*\d+\. +\S", last_line):
        return True
    # A heading as the LAST content line is a truncation signal: the model
    # opened a section but wrote no body.
    if last_line.lstrip().startswith("#"):
        return False
    return False


def _ends_on_dangling_heading(document: str) -> bool:
    """Return True when the document's last content line is a heading.

    Narrow structural-incompleteness detector used to override a clean
    provider ``finish_reason="stop"``: a section heading with no body
    following it is incomplete regardless of the provider's finish signal.
    Kept narrow (headings only) so a clean stop on a table row or other
    unconventional ending does not spuriously trigger a continuation.
    """
    if not document:
        return False
    stripped = document.rstrip()
    if not stripped:
        return False
    lines = [ln for ln in stripped.splitlines() if ln.strip()]
    if not lines:
        return False
    last = lines[-1].rstrip()
    return bool(re.match(r"^#{1,6}\s+\S", last.lstrip()))


def _merge_continuation(document_so_far: str, tail: str) -> str:
    """Concatenate the document-so-far with a continuation tail.

    The continuation prompt asks for ONLY the remaining body, but local
    models sometimes re-emit a leading heading or blank line. This helper
    collapses the join boundary to exactly one blank line and strips any
    leading blank lines from the tail.
    """
    head = document_so_far.rstrip()
    tail_body = tail.lstrip("\n").rstrip()
    if not tail_body:
        return document_so_far
    if not head:
        return tail_body + "\n"
    return head + "\n\n" + tail_body + "\n"


def _tail_is_duplicate(document_so_far: str, tail: str) -> bool:
    """Return True when the continuation just re-emits the existing tail.

    Some local models, when asked to "continue", repeat the last paragraph
    or the entire document-so-far. Detecting this prevents the merge from
    producing a doubled or looped document. The check is conservative: it
    only flags tails whose non-empty lines are a contiguous suffix of the
    existing document.
    """
    head_lines = [ln for ln in document_so_far.splitlines() if ln.strip()]
    tail_lines = [ln for ln in tail.splitlines() if ln.strip()]
    if not tail_lines:
        return True
    if len(tail_lines) > len(head_lines):
        # The continuation is longer than the entire document-so-far — it
        # must contain new content, so not a pure duplicate.
        return False
    suffix = head_lines[-len(tail_lines):]
    return suffix == tail_lines


async def _continue_llm(
    ctx: OperationExecutionContext,
    user_input: str,
    document_so_far: str,
    model_def: Any,
    *,
    max_tokens: int,
    continue_prompt_path: str,
) -> tuple[str, str | None]:
    """Issue one continuation pass to finish a truncated document.

    Appends a continuation instruction to the original message list (so the
    model has full context) and asks for ONLY the remaining tail. Returns
    ``(continuation_content, finish_reason)``.
    """
    if ctx.complete_fn is None:
        raise RuntimeError("markdown requires complete_fn to continue a truncated document")

    continue_template = _load_text(ctx, continue_prompt_path)
    continue_prompt = continue_template.replace("{user_input}", user_input).replace(
        "{document_so_far}", document_so_far
    )
    messages = [
        {"role": "user", "content": continue_prompt},
    ]
    completion = await ctx.complete_fn(
        messages=messages,
        model_def=model_def,
        max_tokens=max_tokens,
        temperature=0.2,
        tools=None,
        response_format=None,
    )
    content = getattr(completion, "content", "") or ""
    finish_reason = coerce_finish_reason(getattr(completion, "finish_reason", None))
    return content.strip(), finish_reason


async def continue_until_complete(
    ctx: OperationExecutionContext,
    *,
    user_input: str,
    document: str,
    model_def: Any,
    first_finish_reason: str | None,
    max_continue_attempts: int,
    max_continue_tokens: int,
    continue_prompt_path: str,
    structural_override: bool = True,
) -> tuple[str, bool, int, str | None]:
    """Run continuation passes until the document looks complete or budget is hit.

    Returns ``(document, was_truncated, attempts, final_finish_reason)``.

    ``structural_override`` defaults to True (final single-pass / merge
    documents benefit from it). Per-source sections set it False: their
    completeness is governed by the per-source density floor, and a
    heading-only section is the density floor's job to catch, not the
    continuation path's (otherwise the continuation consumes the retry
    budget duplicating a heading-only tail before the density retry runs).
    """
    structurally_incomplete = _ends_on_dangling_heading(document) if structural_override else False
    if not is_truncating_finish(first_finish_reason) and not structurally_incomplete:
        return document, False, 0, first_finish_reason
    if structurally_incomplete and not is_truncating_finish(first_finish_reason):
        logger.warning(
            "forge_document: first pass finish_reason=%r but document ends "
            "on a dangling heading (section opened with no body); attempting "
            "continuation despite clean stop",
            first_finish_reason,
        )

    logger.warning(
        "forge_document: first pass finish_reason=%r; attempting continuation",
        first_finish_reason,
    )
    attempts = 0
    final_finish_reason = first_finish_reason
    current = document
    for attempt in range(1, max_continue_attempts + 1):
        attempts = attempt
        continuation_text, cont_finish_reason = await _continue_llm(
            ctx,
            user_input=user_input,
            document_so_far=current,
            model_def=model_def,
            max_tokens=max_continue_tokens,
            continue_prompt_path=continue_prompt_path,
        )
        final_finish_reason = cont_finish_reason
        if not continuation_text or not continuation_text.strip():
            logger.warning(
                "forge_document: continuation pass %d returned empty; giving up",
                attempt,
            )
            break
        tail = extract_markdown_block(continuation_text)
        # If the fenced block extracted was empty but the raw continuation
        # had content, the model emitted prose outside the fence (or an
        # empty fence). Either way, there is no usable markdown tail to
        # append.
        if not tail.strip():
            logger.warning(
                "forge_document: continuation pass %d yielded no markdown tail; giving up",
                attempt,
            )
            break
        if _tail_is_duplicate(current, tail):
            logger.warning(
                "forge_document: continuation pass %d duplicated existing tail; giving up",
                attempt,
            )
            break
        current = _merge_continuation(current, tail)
        if _looks_complete(current) and not is_truncating_finish(cont_finish_reason):
            logger.info(
                "forge_document: continuation pass %d completed the document", attempt,
            )
            return current, False, attempts, cont_finish_reason
        if attempt >= max_continue_attempts:
            logger.warning(
                "forge_document: exhausted %d continuation attempts; "
                "persisting with truncated annotation (final finish_reason=%r)",
                attempt, cont_finish_reason,
            )
    return current, True, attempts, final_finish_reason
