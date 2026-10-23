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


# ── 2026-10-22 char-cap incident fixes ────────────────────────────────


_LONG_REVISION = (
    "# Plan\n\n"
    "## Zero\n\n" + ("z" * 80) + "\n\n"
    "## One\n\n" + ("1" * 80) + "\n\n"
    "## Two\n\n" + ("2" * 80) + "\n\n"
    "## Three\n\n" + ("3" * 400) + "\n"
)


def _make_recording_ctx(
    *,
    complete_fn: Any,
    params: dict[str, Any] | None = None,
    pack_config: dict[str, Any] | None = None,
    query: str = "forge a runbook",
) -> tuple[OperationExecutionContext, list[dict[str, Any]]]:
    """Return a ctx whose write_file calls are captured for assertions."""
    captured: list[dict[str, Any]] = []

    async def executor(tool_id: str, tctx: Any) -> ToolResult:
        params_inner = getattr(tctx, "parameters", {}) or {}
        if tool_id == "write_file":
            content = str(params_inner.get("content") or "")
            path = str(params_inner.get("path") or "")
            captured.append({"path": path, "content": content})
            return ToolResult(
                tool_id=tool_id,
                output_text=json.dumps(
                    {"ok": True, "path": path, "bytes": len(content.encode("utf-8"))}
                ),
            )
        if tool_id == "markdown_generate":
            md = str(params_inner.get("markdown") or "")
            heading_count = sum(1 for line in md.splitlines() if line.lstrip().startswith("#"))
            return ToolResult(
                tool_id=tool_id,
                output_text=f"Outline: {heading_count} headings.",
                output_json={"heading_count": heading_count, "headings": []},
            )
        if tool_id == "memory.search":
            return ToolResult(tool_id=tool_id, output_text="No relevant memories found.")
        return ToolResult(
            tool_id=tool_id,
            output_text=json.dumps({"error": "file not found: docs/forge.md"}),
        )

    from kernel.seams.host_adapter import _CallableHost

    host = _CallableHost(
        complete_fn=complete_fn,
        resolve_model_fn=lambda _alias: _FakeModel(),
        read_authored_text_fn=_make_loader(),
        execute_tool_fn=executor,
    )
    ctx = OperationExecutionContext(
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
        pack_config=pack_config,
    )
    return ctx, captured


@pytest.mark.asyncio
async def test_revision_not_clamped_below_existing_document_size() -> None:
    """A revision is never clamped below the size of the document it revises.

    The 2026-10-22 incident: a 32.9k-char plan was revised cleanly to
    46.6k chars, then the flat 20k post-hoc clamp destroyed 26.6k chars of
    approved content and degraded the result into an unverified write. The
    cap now floors at the existing document's length, so a legitimate
    growth revision persists whole.
    """
    existing = "# Plan\n\noriginal approved content. " * 400  # ~13.3k chars
    ctx, captured = _make_recording_ctx(
        complete_fn=_fake_complete_returning(_LONG_REVISION),
        params={"existing_document": existing, "document_path": "docs/plan.md"},
        pack_config={"forge_max_document_chars": 100},
    )
    result = await execute(ctx)
    assert result.status == "success"
    assert len(captured) == 1
    content = captured[0]["content"]
    # Every planned section survived the (already floored-away) cap.
    for section in ("## Zero", "## One", "## Two", "## Three"):
        assert section in content
    assert ("3" * 400) in content
    assert "<!-- forge: capped -->" not in content
    assert "<!-- truncated -->" not in content


@pytest.mark.asyncio
async def test_char_cap_cuts_at_heading_boundary_with_omitted_list() -> None:
    """A char-cap cut lands on a heading boundary and names omitted sections.

    The old mid-sentence slice left documents ending mid-word with no
    signal of what was dropped (the 2026-10-22 artifact ended
    ``"...delay for the s``). The cut now backs off to the last heading
    that fits and appends an explicit omitted-sections comment.
    """
    ctx, captured = _make_recording_ctx(
        complete_fn=_fake_complete_returning(_LONG_REVISION),
        pack_config={"forge_max_document_chars": 200},
    )
    result = await execute(ctx)
    assert captured, "write_file must have been called"
    content = captured[0]["content"]
    assert "<!-- forge: capped -->" in content
    # The note names the dropped downstream headings.
    assert "omitted" in content
    assert "## Three" in content
    # The retained body ends at a heading boundary: the last retained
    # line is one of the section bodies, not a cut mid-word.
    body = content.split("<!-- forge: capped -->")[0].rstrip()
    assert body.endswith(("z", "1", "2"))
    assert result.status == "degraded"
    assert result.extra["continuation"]["stop_reason"] == "char_cap"


@pytest.mark.asyncio
async def test_revision_prompt_carries_size_headroom_guidance() -> None:
    """A revision prompt tells the model its char budget before it writes.

    The model planned 46.6k chars against an invisible 20k clamp. The
    guidance line surfaces the cap and the headroom so it can plan inside
    the budget (and preserve approved content) instead of discovering the
    clamp post-generation.
    """
    prompts: list[str] = []

    async def complete(**kwargs: Any) -> _FakeCompletion:
        prompts.append(kwargs["messages"][1]["content"])
        return _FakeCompletion(content=f"```markdown\n{_LONG_REVISION}\n```")

    from kernel.seams.host_adapter import _CallableHost

    existing = "# Plan\n\n" + ("approved " * 500)  # ~4k chars
    ctx = OperationExecutionContext(
        workspace_id="test",
        user_id="test-user",
        user_slug="test-user",
        surface_type="personal_dm",
        surface_id="test",
        memory_scope="personal",
        mode_id="markdown",
        query="revise",
        host=_CallableHost(
            complete_fn=complete,
            resolve_model_fn=lambda _alias: _FakeModel(),
            read_authored_text_fn=_make_loader(),
            execute_tool_fn=_make_tool_executor()[0],
        ),
        params={"existing_document": existing},
        pack_config={"forge_max_document_chars": 6000},
    )
    await execute(ctx)
    assert prompts, "completion must have been called"
    assert "persistence budget" in prompts[0]
    # The op strips the loaded document, so the guidance reports the
    # stripped length (4507 here, not the 4508 with the trailing space).
    assert str(len(existing.strip())) in prompts[0]


@pytest.mark.asyncio
async def test_fresh_forge_prompt_stays_byte_identical() -> None:
    """No existing document → no size-guidance line (legacy prompt shape)."""
    prompts: list[str] = []

    async def complete(**kwargs: Any) -> _FakeCompletion:
        prompts.append(kwargs["messages"][1]["content"])
        return _FakeCompletion(content=f"```markdown\n{_VALID_DOCUMENT}\n```")

    from kernel.seams.host_adapter import _CallableHost

    ctx = OperationExecutionContext(
        workspace_id="test",
        user_id="test-user",
        user_slug="test-user",
        surface_type="personal_dm",
        surface_id="test",
        memory_scope="personal",
        mode_id="markdown",
        query="forge a runbook",
        host=_CallableHost(
            complete_fn=complete,
            resolve_model_fn=lambda _alias: _FakeModel(),
            read_authored_text_fn=_make_loader(),
            execute_tool_fn=_make_tool_executor()[0],
        ),
        params={},
    )
    await execute(ctx)
    assert prompts and "persistence budget" not in prompts[0]
    assert "Target length" not in prompts[0]
