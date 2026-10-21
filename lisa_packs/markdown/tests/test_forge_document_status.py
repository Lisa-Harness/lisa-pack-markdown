"""Tests for the ``OperationResult.status`` contract in forge_document.

Phase 2 of the structured error-handling contract: the markdown pack
adopts the ``status`` / ``error_detail`` fields added to
``kernel.models.OperationResult`` in Phase 1.

- A clean (non-truncated) forge returns ``status="success"``.
- A truncated forge (continuation exhausted / empty / char-cap) returns
  ``status="degraded"`` with ``error_detail`` and a structured
  ``extra["continuation"]`` block carrying ``was_truncated``,
  ``attempts``, ``stop_reason``, and the finish reasons.
- A no-outline document (0 headings) returns ``status="degraded"``.

These run against the lisa repo's installed ``kernel`` package and the
markdown pack's real authored prompts (same fixture wiring as
``test_forge_document.py``).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
from kernel.models import OperationExecutionContext, ToolResult

from lisa_packs.markdown.operations.forge_document.forge_document import execute

_CONTENT_ROOT = Path(__file__).resolve().parents[1]


def _make_loader() -> Any:
    """Return a ``read_authored_text`` callable that reads real prompt files."""

    def read_authored_text(relative_path: str) -> str:
        full_path = _CONTENT_ROOT / relative_path
        if not full_path.exists():
            raise FileNotFoundError(f"Authored text not found: {full_path}")
        return full_path.read_text(encoding="utf-8")

    return read_authored_text


@dataclass
class _FakeModel:
    provider: str = "openai"
    model_id: str = "gpt-chat-latest"
    display_name: str = "GPT Chat"
    reasoning_model: bool = False
    thinking_model: bool = False


@dataclass
class _FakeCompletion:
    content: str
    provider: str = "openai"
    model_id: str = "gpt-chat-latest"
    display_name: str = "GPT Chat"
    request_id: str = "test"
    tool_calls: list[Any] | None = None
    usage: Any = None
    finish_reason: str | None = "stop"
    truncated: bool | None = False


_VALID_DOCUMENT = (
    "# Runbook: Restart a Stuck VM\n"
    "\n"
    "## Symptoms\n"
    "- VM is unresponsive to SSH.\n"
    "- Agent reports `NotReady`.\n"
    "\n"
    "## Steps\n"
    "1. Confirm the VM state via `az vm show`.\n"
    "2. Restart with `az vm restart -g rg -n vm`.\n"
    "3. Verify health probes return 200.\n"
)

_PARTIAL_SWOT = (
    "# SWOT Analysis\n\n## Strengths\n- Strong technical foundation.\n- Experienced team.\n"
)

_SWOT_TAIL = (
    "## Weaknesses\n"
    "- Limited market presence.\n"
    "- Small budget.\n"
    "\n"
    "## Opportunities\n"
    "- Growing demand for AI tooling.\n"
    "- Enterprise adoption.\n"
)


def _make_tool_executor() -> tuple[Any, dict[str, Any]]:
    """Return a fake ``execute_tool_fn`` and a captures dict."""
    captures: dict[str, Any] = {"writes": [], "memory_query": None, "outline_markdown": None}

    async def execute_tool_fn(tool_id: str, ctx: Any) -> ToolResult:
        params = getattr(ctx, "parameters", {}) or {}
        if tool_id == "memory.search":
            captures["memory_query"] = params.get("query")
            return ToolResult(tool_id=tool_id, output_text="No relevant memories found.")
        if tool_id == "markdown_generate":
            captures["outline_markdown"] = params.get("markdown")
            md_text = str(params.get("markdown") or "")
            heading_count = sum(1 for line in md_text.splitlines() if line.lstrip().startswith("#"))
            return ToolResult(
                tool_id=tool_id,
                output_text=f"Outline: {heading_count} headings.",
                output_json={"heading_count": heading_count, "headings": []},
            )
        if tool_id == "read_file":
            return ToolResult(
                tool_id=tool_id,
                output_text=json.dumps({"error": "file not found: docs/forge.md"}),
            )
        if tool_id == "write_file":
            path = str(params.get("path") or "")
            content = str(params.get("content") or "")
            captures["writes"].append({"path": path, "content": content})
            return ToolResult(
                tool_id=tool_id,
                output_text=json.dumps(
                    {"ok": True, "path": path, "bytes": len(content.encode("utf-8"))}
                ),
            )
        return ToolResult(tool_id=tool_id, error=f"unknown tool: {tool_id}")

    return execute_tool_fn, captures


def _make_ctx(
    *,
    query: str = "forge a runbook",
    complete_fn: Any = None,
    params: dict[str, Any] | None = None,
) -> OperationExecutionContext:
    from kernel.seams.host_adapter import _CallableHost

    host = _CallableHost(
        complete_fn=complete_fn,
        resolve_model_fn=lambda _alias: _FakeModel(),
        read_authored_text_fn=_make_loader(),
        execute_tool_fn=_make_tool_executor()[0],
    )
    return OperationExecutionContext(
        workspace_id="test",
        user_id="test-user",
        user_slug="test-user",
        surface_type="personal_dm",
        surface_id="test",
        memory_scope="personal",
        mode_id="markdown",
        query=query,
        host=host,
        params=params or {},
    )


def _fake_complete_returning(document: str) -> Any:
    async def _complete(**_kwargs: Any) -> _FakeCompletion:
        return _FakeCompletion(content=f"```markdown\n{document}\n```")

    return _complete


def _fake_complete_sequence(
    *,
    documents: list[str],
    finish_reasons: list[str | None] | None = None,
    truncated_flags: list[bool | None] | None = None,
) -> Any:
    """Return a complete_fn that yields each document in sequence."""
    call_count = {"n": 0}
    n = len(documents)
    fr_list = finish_reasons or ["stop"] * n
    tf_list = truncated_flags or [False] * n

    async def _complete(**_kwargs: Any) -> _FakeCompletion:
        i = call_count["n"]
        call_count["n"] += 1
        doc = documents[min(i, n - 1)]
        fr = fr_list[min(i, len(fr_list) - 1)]
        tf = tf_list[min(i, len(tf_list) - 1)]
        return _FakeCompletion(
            content=f"```markdown\n{doc}\n```",
            finish_reason=fr,
            truncated=tf,
        )

    return _complete


# ── Tests ──────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_clean_forge_returns_success_status() -> None:
    """A non-truncated forge returns ``status="success"`` with no error_detail."""
    ctx = _make_ctx(complete_fn=_fake_complete_returning(_VALID_DOCUMENT))
    result = await execute(ctx)
    assert result.status == "success"
    assert result.is_success is True
    assert result.error_detail is None
    # No continuation block when not truncated.
    assert (
        "continuation" not in result.extra or result.extra["continuation"]["was_truncated"] is False
    )


@pytest.mark.asyncio
async def test_truncated_forge_returns_degraded_status() -> None:
    """A truncated forge (continuation exhausted) returns ``status="degraded"``."""
    # First pass truncates; continuation also truncates → budget exhausted.
    complete_fn = _fake_complete_sequence(
        documents=[_PARTIAL_SWOT, "## Weaknesses\n- Limited market presence and the\n"],
        finish_reasons=["length", "length"],
        truncated_flags=[True, True],
    )
    ctx = _make_ctx(query="forge a SWOT analysis", complete_fn=complete_fn)
    result = await execute(ctx)
    assert result.status == "degraded"
    assert result.is_success is False
    assert result.error_detail is not None
    assert (
        "continuation" in result.error_detail.lower() or "exhausted" in result.error_detail.lower()
    )


@pytest.mark.asyncio
async def test_truncated_forge_has_continuation_block_in_extra() -> None:
    """The ``extra["continuation"]`` block carries structured truncation context."""
    complete_fn = _fake_complete_sequence(
        documents=[_PARTIAL_SWOT, "## Weaknesses\n- Limited market presence and the\n"],
        finish_reasons=["length", "length"],
        truncated_flags=[True, True],
    )
    ctx = _make_ctx(query="forge a SWOT analysis", complete_fn=complete_fn)
    result = await execute(ctx)
    cont = result.extra.get("continuation")
    assert isinstance(cont, dict)
    assert cont["was_truncated"] is True
    assert cont["attempts"] >= 1
    assert "stop_reason" in cont
    assert cont["stop_reason"] in ("exhausted", "empty", "no_markdown_tail", "duplicate_tail")
    assert cont["first_finish_reason"] == "length"
    assert cont["final_finish_reason"] == "length"


@pytest.mark.asyncio
async def test_continuation_completed_returns_success_status() -> None:
    """When a continuation pass completes the document, status is ``success``."""
    complete_fn = _fake_complete_sequence(
        documents=[_PARTIAL_SWOT, _SWOT_TAIL],
        finish_reasons=["length", "stop"],
        truncated_flags=[True, False],
    )
    ctx = _make_ctx(query="forge a SWOT analysis", complete_fn=complete_fn)
    result = await execute(ctx)
    assert result.status == "success"
    cont: dict[str, Any] = result.extra["continuation"]
    assert cont["was_truncated"] is False
    assert cont["stop_reason"] == "completed"
    assert cont["attempts"] == 1


@pytest.mark.asyncio
async def test_empty_continuation_returns_degraded_with_stop_reason_empty() -> None:
    """An empty continuation pass surfaces ``stop_reason="empty"`` in the extra block."""
    call_count = {"n": 0}

    async def _complete(**_kwargs: Any) -> _FakeCompletion:
        call_count["n"] += 1
        if call_count["n"] == 1:
            return _FakeCompletion(
                content=f"```markdown\n{_PARTIAL_SWOT}\n```",
                finish_reason="length",
                truncated=True,
            )
        # Truly empty content (no fence) so the empty check fires.
        return _FakeCompletion(content="", finish_reason="stop", truncated=False)

    ctx = _make_ctx(query="forge a SWOT analysis", complete_fn=_complete)
    result = await execute(ctx)
    assert result.status == "degraded"
    cont: dict[str, Any] = result.extra["continuation"]
    assert cont["stop_reason"] == "empty"
    assert cont["was_truncated"] is True


@pytest.mark.asyncio
async def test_no_outline_document_returns_degraded_status() -> None:
    """A document with 0 headings returns ``status="degraded"`` (persisted with warning)."""
    no_headings = "This is just prose with no headings at all.\n\nAnother paragraph."
    ctx = _make_ctx(complete_fn=_fake_complete_returning(no_headings))
    result = await execute(ctx)
    assert result.status == "degraded"
    assert result.is_success is False
    assert "no headings" in (result.error_detail or "").lower()
    cont = result.extra.get("continuation")
    assert isinstance(cont, dict)


@pytest.mark.asyncio
async def test_char_cap_truncation_sets_stop_reason_char_cap() -> None:
    """When the document exceeds the char cap, ``stop_reason`` is ``char_cap``."""
    # A valid document with headings, but very long, exceeding max_document_chars.
    # Use a low max_document_chars via pack_config to trigger the cap.
    long_body = "# Title\n\n" + ("x" * 2000) + "\n"
    ctx = _make_ctx(
        complete_fn=_fake_complete_returning(long_body),
        params={},
    )
    # Override pack_config with a tiny char cap.
    ctx.pack_config = {"forge_max_document_chars": 100}
    result = await execute(ctx)
    # The char cap sets was_truncated=True → status="degraded".
    assert result.status == "degraded"
    cont = result.extra.get("continuation")
    if cont and cont.get("was_truncated"):
        assert cont["stop_reason"] == "char_cap"
