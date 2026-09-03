"""Render a single prior turn (or a range) from the lossless thread store as Markdown.

This is the trivial "turn a previous turn into markdown" operation. It reads
the entry by global turn index from ``sdk.thread.ThreadStore`` (the same
lossless, segmented, zstd-sealed store the host writes to every turn) and
renders a self-contained Markdown block — user query, Lisa's full response,
and metadata (mode, timestamp, handoff, write receipts).

No LLM calls, no memory recall, no storage writes. The operation is a pure
read-and-render: the thread log is the ground truth, and this operation
makes any turn in it addressable as a clean Markdown document.

The model can call this after consulting the thread manifest (injected in
system prompt each turn) to find a turn number worth rendering.

Dependency direction: ``render_turn`` → ``sdk.thread`` (read-only),
``kernel.models`` (``OperationExecutionContext`` / ``OperationResult``).
No host imports — the thread store is SDK-level and constructible from
``ctx.data_dir`` + ``ctx.user_slug``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from kernel.models import OperationExecutionContext, OperationResult
from pydantic import BaseModel, Field
from sdk.thread import ThreadStore

logger = logging.getLogger(__name__)

__all__ = ["Parameters", "execute"]


class Parameters(BaseModel):
    """Parameter surface for callers invoking the operation.

    The operation reads ``ctx.params`` directly; this model documents and
    validates the accepted inputs. Declared as a ``BaseModel`` so the SDK
    catalog builder can assign it to ``OperationDefinition.parameters_schema``
    (which requires a ``type[BaseModel]``).
    """

    turn: int = Field(
        ge=0,
        description="The global (0-based) turn index to render.",
    )
    end_turn: int | None = Field(
        default=None,
        ge=0,
        description="Optional inclusive upper bound. When set, the operation renders the "
        "inclusive range [turn, end_turn] as one Markdown document. None (the "
        "default) renders a single turn.",
    )


def _thread_store(ctx: OperationExecutionContext) -> ThreadStore:
    """Build a ``ThreadStore`` from the context's data_dir + user_slug.

    The thread base path is ``<data_dir>/threads/<user_slug>`` — the same
    derivation ``host.runtime.HostRuntime.thread_path`` uses. When
    ``user_slug`` is empty, the store reads from ``<data_dir>/threads``
    directly (single-user mode, matching the ``lisa-thread`` MCP server).
    """
    data_dir = Path(ctx.data_dir) if ctx.data_dir else Path(".")
    thread_dir = data_dir / "threads"
    user_slug = str(ctx.user_slug or "").strip()
    base = thread_dir / user_slug if user_slug else thread_dir
    return ThreadStore(base)


def _render_entry(entry: dict[str, Any], turn: int) -> str:
    """Render a single thread entry as a self-contained Markdown block.

    Returns the rendered Markdown string. The response_text is included in
    full (not truncated) — this is the "give me the turn as markdown" use
    case, not a prompt-injection excerpt.
    """
    ts = str(entry.get("timestamp_utc") or "")
    mode = str(entry.get("mode_id") or "-")
    query = str(entry.get("query") or "").strip()
    response = str(entry.get("response_text") or "").strip()
    handoff = str(entry.get("handoff_to") or entry.get("transitioned_to") or "")
    receipts = entry.get("effect_receipts") or []

    lines: list[str] = [
        f"## Turn {turn} — {mode}",
        "",
    ]
    if ts:
        lines.append(f"_Captured {ts}_")
        lines.append("")
    lines.append("### User")
    lines.append("")
    lines.append(query or "_(no query)_")
    lines.append("")
    lines.append("### Lisa")
    lines.append("")
    lines.append(response or "_(no response)_")

    if handoff:
        lines += ["", f"**Handoff:** {handoff}"]

    if isinstance(receipts, list) and receipts:
        write_caps = sorted(
            {
                str(r.get("capability_id") or "")
                for r in receipts
                if isinstance(r, dict)
                and str(r.get("effect", "")) in ("write", "external_write")
            }
        )
        if write_caps:
            lines += ["", f"**Writes:** {', '.join(write_caps)}"]

    return "\n".join(lines)


def _extract_turn(params: dict[str, Any]) -> tuple[int, int | None]:
    """Return ``(turn, end_turn)`` from the operation params.

    ``end_turn`` is ``None`` for a single-turn render. Raises
    ``ValueError`` when ``turn`` is missing or not a non-negative integer.
    """
    raw_turn = params.get("turn")
    if raw_turn is None:
        raise ValueError("render_turn: 'turn' parameter is required (global turn index)")
    try:
        turn = int(raw_turn)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"render_turn: 'turn' must be an integer, got {raw_turn!r}") from exc
    if turn < 0:
        raise ValueError(f"render_turn: 'turn' must be non-negative, got {turn}")

    raw_end = params.get("end_turn")
    end_turn: int | None = None
    if raw_end is not None:
        try:
            end_turn = int(raw_end)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"render_turn: 'end_turn' must be an integer, got {raw_end!r}"
            ) from exc
        if end_turn < turn:
            raise ValueError(
                f"render_turn: 'end_turn' ({end_turn}) must be >= 'turn' ({turn})"
            )

    return turn, end_turn


async def execute(ctx: OperationExecutionContext) -> OperationResult:
    """Render one or more prior turns from the thread store as Markdown.

    Reads ``ctx.params["turn"]`` (required, 0-based global turn index) and
    ``ctx.params["end_turn"]`` (optional, inclusive upper bound for a range
    render). Returns the rendered Markdown as ``OperationResult.output_text``.

    Returns an empty ``output_text`` when the requested turn(s) are not
    found in the thread store (e.g. the turn index is out of range or the
    store has no entries yet).
    """
    params = ctx.params or {}
    try:
        turn, end_turn = _extract_turn(params)
    except ValueError as exc:
        return OperationResult(output_text=str(exc))

    store = _thread_store(ctx)
    if end_turn is not None:
        entries = store.load_range(start_turn=turn, end_turn=end_turn)
    else:
        entries = store.load_range(start_turn=turn, end_turn=turn)

    if not entries:
        return OperationResult(output_text="")

    # For a range render, join the individual turn blocks with a separator.
    # ``load_range`` returns entries in chronological order; derive each
    # entry's global turn from its stored ``turn`` field when available,
    # falling back to the requested range's sequential indexing.
    blocks: list[str] = []
    for offset, entry in enumerate(entries):
        stored_turn = entry.get("turn")
        if isinstance(stored_turn, int):
            entry_turn = stored_turn
        else:
            entry_turn = turn + offset
        blocks.append(_render_entry(entry, entry_turn))

    return OperationResult(output_text="\n\n---\n\n".join(blocks))