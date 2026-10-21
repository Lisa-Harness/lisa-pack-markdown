"""Tests for the markdown_render_turn operation.

The operation is a pure read-and-render: it reads entries from
``sdk.thread.ThreadStore`` by global turn index and renders them as
self-contained Markdown blocks. No LLM calls, no memory recall, no storage
writes. Tests write synthetic entries to a real ``ThreadStore`` in a tmp
directory and verify the rendered output.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from kernel.models import OperationExecutionContext

from lisa_packs.markdown.operations.render_turn.render_turn import execute


def _write_entry(
    base: Path,
    *,
    mode_id: str = "default",
    query: str = "test query",
    response_text: str = "test response",
    turn: int = 0,
    effect_receipts: list[dict[str, Any]] | None = None,
    handoff_to: str = "",
) -> None:
    """Write a synthetic entry directly to the active segment."""
    base.parent.mkdir(parents=True, exist_ok=True)
    entry: dict[str, Any] = {
        "timestamp_utc": "2026-08-21T12:00:00+00:00",
        "user_slug": "test",
        "mode_id": mode_id,
        "query": query,
        "response_text": response_text,
        "turn": turn,
    }
    if effect_receipts:
        entry["effect_receipts"] = effect_receipts
    if handoff_to:
        entry["handoff_to"] = handoff_to
    current = Path(f"{base}.current.jsonl")
    with current.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, ensure_ascii=False) + "\n")


def _ctx(
    tmp_path: Path,
    *,
    params: dict[str, Any] | None = None,
    user_slug: str = "test",
) -> OperationExecutionContext:
    """Build a minimal OperationExecutionContext pointing at a tmp thread store."""
    return OperationExecutionContext(
        data_dir=str(tmp_path),
        user_slug=user_slug,
        params=params or {},
    )


@pytest.mark.asyncio
async def test_single_turn_renders_query_and_response(tmp_path: Path) -> None:
    base = tmp_path / "threads" / "test"
    _write_entry(
        base,
        mode_id="coding",
        query="fix the bug",
        response_text="Fixed the bug in main.py.",
        turn=0,
    )
    result = await execute(_ctx(tmp_path, params={"turn": 0}))
    text = result.output_text
    assert "## Turn 0 — coding" in text
    assert "### User" in text
    assert "fix the bug" in text
    assert "### Lisa" in text
    assert "Fixed the bug in main.py." in text
    assert "_Captured 2026-08-21T12:00:00+00:00_" in text


@pytest.mark.asyncio
async def test_missing_turn_returns_empty(tmp_path: Path) -> None:
    base = tmp_path / "threads" / "test"
    _write_entry(base, turn=0)
    result = await execute(_ctx(tmp_path, params={"turn": 99}))
    assert result.output_text == ""


@pytest.mark.asyncio
async def test_empty_store_returns_empty(tmp_path: Path) -> None:
    result = await execute(_ctx(tmp_path, params={"turn": 0}))
    assert result.output_text == ""


@pytest.mark.asyncio
async def test_turn_with_receipts_renders_writes(tmp_path: Path) -> None:
    base = tmp_path / "threads" / "test"
    # Write a first entry (turn 0 by position) so the receipt entry is at
    # position turn 1 — load_range computes the global turn from segment
    # position, not the stored ``turn`` field.
    _write_entry(base, mode_id="search", query="search", response_text="found.", turn=0)
    _write_entry(
        base,
        mode_id="markdown",
        query="write the doc",
        response_text="Document written.",
        turn=1,
        effect_receipts=[
            {"capability_id": "markdown_forge_document", "effect": "write", "artifact_id": "doc-1"},
            {"capability_id": "memory.search", "effect": "read", "artifact_id": ""},
        ],
        handoff_to="default",
    )
    result = await execute(_ctx(tmp_path, params={"turn": 1}))
    text = result.output_text
    assert "**Writes:** markdown_forge_document" in text
    # Read-only receipts are not listed in the Writes block.
    assert "memory.search" not in text
    assert "**Handoff:** default" in text


@pytest.mark.asyncio
async def test_range_renders_multiple_turns(tmp_path: Path) -> None:
    base = tmp_path / "threads" / "test"
    _write_entry(base, mode_id="search", query="search for x", response_text="Found x.", turn=0)
    _write_entry(base, mode_id="coding", query="implement x", response_text="Done.", turn=1)
    _write_entry(base, mode_id="markdown", query="document x", response_text="Documented.", turn=2)
    result = await execute(_ctx(tmp_path, params={"turn": 0, "end_turn": 2}))
    text = result.output_text
    assert "## Turn 0 — search" in text
    assert "## Turn 1 — coding" in text
    assert "## Turn 2 — markdown" in text
    # Separator between blocks.
    assert "\n\n---\n\n" in text


@pytest.mark.asyncio
async def test_missing_turn_param_returns_error(tmp_path: Path) -> None:
    result = await execute(_ctx(tmp_path, params={}))
    assert "required" in result.output_text.lower()


@pytest.mark.asyncio
async def test_negative_turn_returns_error(tmp_path: Path) -> None:
    result = await execute(_ctx(tmp_path, params={"turn": -1}))
    assert "non-negative" in result.output_text.lower()


@pytest.mark.asyncio
async def test_end_turn_less_than_turn_returns_error(tmp_path: Path) -> None:
    result = await execute(_ctx(tmp_path, params={"turn": 5, "end_turn": 3}))
    assert ">=" in result.output_text


@pytest.mark.asyncio
async def test_empty_user_slug_uses_thread_dir_directly(tmp_path: Path) -> None:
    # When user_slug is empty, the store reads from <data_dir>/threads
    # directly (single-user mode, matching the lisa-thread MCP server).
    base = tmp_path / "threads"
    _write_entry(base, mode_id="default", query="hello", response_text="hi", turn=0)
    result = await execute(
        OperationExecutionContext(
            data_dir=str(tmp_path),
            user_slug="",
            params={"turn": 0},
        )
    )
    assert "## Turn 0 — default" in text if (text := result.output_text) else False
