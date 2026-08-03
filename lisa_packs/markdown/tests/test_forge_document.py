"""Tests for the markdown_forge_document operation.

Tests inject a real content-registry loader (`read_authored_text_fn`) that
reads the actual authored prompt files from the pack's content root, a fake
`complete_fn` that returns a canned markdown document, and a fake
`execute_tool_fn` that simulates the `memory.search`, `markdown_generate`,
and `write_file` tools. Per the pack-content rule, the operation has no
fallbacks — tests must wire both callbacks exactly as production hosts do.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from kernel.models import OperationExecutionContext, ToolResult
from lisa_packs.markdown.operations.forge_document.forge_document import (
    Parameters,
    execute,
)

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
    # Default to a clean "stop" so existing tests do not trip the new
    # truncation/continuation path. Tests that exercise truncation pass
    # ``finish_reason="length"`` or ``"unknown"`` explicitly.
    finish_reason: str | None = "stop"


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


def _make_tool_executor(
    *,
    memory_records: list[str] | None = None,
    outline: dict[str, Any] | None = None,
    write_ok: bool = True,
    stored_documents: dict[str, str] | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Return a fake ``execute_tool_fn`` and a captures dict.

    ``captures`` records the arguments of each ``write_file`` call so tests
    can assert what the operation persisted. ``stored_documents`` seeds the
    fake storage with existing document text keyed by path; ``read_file``
    reads from it and ``write_file`` updates it.
    """
    store: dict[str, str] = dict(stored_documents or {})
    captures: dict[str, Any] = {
        "writes": [],
        "memory_query": None,
        "outline_markdown": None,
        "read_paths": [],
    }

    async def execute_tool_fn(tool_id: str, ctx: Any) -> ToolResult:
        params = getattr(ctx, "parameters", {}) or {}
        if tool_id == "memory.search":
            captures["memory_query"] = params.get("query")
            if memory_records is None:
                return ToolResult(tool_id=tool_id, output_text="No relevant memories found.")
            lines = [f"- {r}" for r in memory_records]
            return ToolResult(tool_id=tool_id, output_text="\n".join(lines))
        if tool_id == "markdown_generate":
            captures["outline_markdown"] = params.get("markdown")
            if outline is not None:
                return ToolResult(
                    tool_id=tool_id,
                    output_text="Outline from injected fixture.",
                    output_json=outline,
                )
            # Compute a minimal outline from the supplied markdown so the
            # operation's heading-count gate can be exercised realistically.
            md_text = str(params.get("markdown") or "")
            heading_count = sum(
                1 for line in md_text.splitlines() if line.lstrip().startswith("#")
            )
            return ToolResult(
                tool_id=tool_id,
                output_text=f"Outline: {heading_count} headings.",
                output_json={"heading_count": heading_count, "headings": []},
            )
        if tool_id == "read_file":
            path = str(params.get("path") or "")
            captures["read_paths"].append(path)
            if path in store:
                return ToolResult(tool_id=tool_id, output_text=store[path])
            return ToolResult(
                tool_id=tool_id,
                output_text=json.dumps({"error": f"file not found: {path}"}),
            )
        if tool_id == "write_file":
            path = str(params.get("path") or "")
            content = str(params.get("content") or "")
            captures["writes"].append({"path": path, "content": content})
            if not write_ok:
                return ToolResult(
                    tool_id=tool_id,
                    output_text=json.dumps({"error": "server is read-only"}),
                )
            store[path] = content
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
    query: str = "forge a runbook for restarting a stuck VM",
    params: dict[str, Any] | None = None,
    complete_fn: Any = None,
    resolve_model_fn: Any = None,
    execute_tool_fn: Any = None,
    pack_config: dict[str, Any] | None = None,
) -> OperationExecutionContext:
    return OperationExecutionContext(
        workspace_id="test",
        user_id="test-user",
        user_slug="test-user",
        surface_type="personal_dm",
        surface_id="test",
        memory_scope="personal",
        mode_id="markdown",
        query=query,
        complete_fn=complete_fn,
        resolve_model_fn=resolve_model_fn,
        read_authored_text_fn=_make_loader(),
        execute_tool_fn=execute_tool_fn,
        params=params or {},
        pack_config=pack_config,
    )


def _fake_complete_returning(document: str) -> Any:
    async def _complete(**_kwargs: Any) -> _FakeCompletion:
        return _FakeCompletion(content=f"```markdown\n{document}\n```")

    return _complete


def _fake_resolve_model() -> Any:
    def _resolve(_alias: str) -> _FakeModel:
        return _FakeModel()

    return _resolve


# ── Operation tests ────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_empty_input_returns_no_input_message() -> None:
    ctx = _make_ctx(query="", execute_tool_fn=(lambda *_a, **_kw: None))
    result = await execute(ctx)
    assert "No instruction" in result.output_text


@pytest.mark.asyncio
async def test_forges_and_persists_document() -> None:
    tool_fn, captures = _make_tool_executor(memory_records=["Incident 42: VM stuck, restart fixed it."])
    ctx = _make_ctx(
        complete_fn=_fake_complete_returning(_VALID_DOCUMENT),
        resolve_model_fn=_fake_resolve_model(),
        execute_tool_fn=tool_fn,
        params={"document_path": "docs/runbook.md"},
    )
    result = await execute(ctx)
    assert "Document forged and stored" in result.output_text
    assert "docs/runbook.md" in result.output_text
    # The operation must have recalled memory with the user query.
    assert captures["memory_query"] == "forge a runbook for restarting a stuck VM"
    # The operation must have validated the document via the outline tool.
    assert captures["outline_markdown"] is not None
    assert "Restart a Stuck VM" in str(captures["outline_markdown"])
    # The operation must have written the document to storage.
    assert len(captures["writes"]) == 1
    write = captures["writes"][0]
    assert write["path"] == "docs/runbook.md"
    assert "Restart a Stuck VM" in write["content"]


@pytest.mark.asyncio
async def test_uses_default_path_when_none_supplied() -> None:
    tool_fn, captures = _make_tool_executor()
    ctx = _make_ctx(
        complete_fn=_fake_complete_returning(_VALID_DOCUMENT),
        resolve_model_fn=_fake_resolve_model(),
        execute_tool_fn=tool_fn,
    )
    result = await execute(ctx)
    assert "docs/forge.md" in result.output_text
    assert captures["writes"][0]["path"] == "docs/forge.md"


@pytest.mark.asyncio
async def test_limit_zero_is_accepted_and_coerced_to_default() -> None:
    """``limit: 0`` must not be rejected by the Parameters schema gate.

    Regression for the 2026-08-02 incident where the model called
    ``markdown_forge_document`` with ``limit: 0`` and the kernel rejected
    the whole call with ``Input should be greater than or equal to 1``
    before ``execute`` ran. The schema now allows 0; ``execute`` coerces
    a falsy limit to the configured default (``forge_max_recall_records``)
    so the forge proceeds and still recalls memory.
    """
    # The schema gate (kernel _validate_parameters runs Parameters.model_validate)
    # must accept limit=0 — this is the actual regression: the unit tests call
    # execute() directly with a dict and never exercise the schema, so a too-
    # strict Field constraint shipped unnoticed.
    Parameters.model_validate({"document_path": "docs/runbook.md", "limit": 0})
    tool_fn, captures = _make_tool_executor(memory_records=["a recalled fact"])
    ctx = _make_ctx(
        complete_fn=_fake_complete_returning(_VALID_DOCUMENT),
        resolve_model_fn=_fake_resolve_model(),
        execute_tool_fn=tool_fn,
        params={"document_path": "docs/runbook.md", "limit": 0},
    )
    result = await execute(ctx)
    assert "Document forged and stored" in result.output_text
    # Memory recall must still have happened (0 coerced to the default,
    # not treated as "disable recall").
    assert captures["memory_query"] is not None


@pytest.mark.asyncio
async def test_rejects_path_with_parent_traversal() -> None:
    tool_fn, _ = _make_tool_executor()
    ctx = _make_ctx(
        complete_fn=_fake_complete_returning(_VALID_DOCUMENT),
        resolve_model_fn=_fake_resolve_model(),
        execute_tool_fn=tool_fn,
        params={"document_path": "../escape.md"},
    )
    result = await execute(ctx)
    assert ".." in result.output_text or "document_path" in result.output_text


@pytest.mark.asyncio
async def test_empty_llm_response_returns_error() -> None:
    tool_fn, _ = _make_tool_executor()

    async def _empty_complete(**_kwargs: Any) -> _FakeCompletion:
        return _FakeCompletion(content="")

    ctx = _make_ctx(
        complete_fn=_empty_complete,
        resolve_model_fn=_fake_resolve_model(),
        execute_tool_fn=tool_fn,
    )
    result = await execute(ctx)
    assert "empty response" in result.output_text.lower()


@pytest.mark.asyncio
async def test_document_with_no_headings_returns_no_outline_error() -> None:
    tool_fn, _ = _make_tool_executor()
    no_headings_doc = "Just a paragraph. No headings here.\n"
    ctx = _make_ctx(
        complete_fn=_fake_complete_returning(no_headings_doc),
        resolve_model_fn=_fake_resolve_model(),
        execute_tool_fn=tool_fn,
    )
    result = await execute(ctx)
    assert "no headings" in result.output_text.lower()


@pytest.mark.asyncio
async def test_storage_failure_returns_storage_failed_message() -> None:
    tool_fn, _ = _make_tool_executor(write_ok=False)
    ctx = _make_ctx(
        complete_fn=_fake_complete_returning(_VALID_DOCUMENT),
        resolve_model_fn=_fake_resolve_model(),
        execute_tool_fn=tool_fn,
    )
    result = await execute(ctx)
    assert "Storage Failed" in result.output_text
    assert "read-only" in result.output_text


@pytest.mark.asyncio
async def test_outlines_injected_fixture_with_zero_headings_fails() -> None:
    tool_fn, _ = _make_tool_executor(outline={"heading_count": 0, "headings": []})
    ctx = _make_ctx(
        complete_fn=_fake_complete_returning(_VALID_DOCUMENT),
        resolve_model_fn=_fake_resolve_model(),
        execute_tool_fn=tool_fn,
    )
    result = await execute(ctx)
    assert "no headings" in result.output_text.lower()


@pytest.mark.asyncio
async def test_memory_context_passed_to_llm() -> None:
    """The recalled memory context must reach the LLM user prompt."""
    tool_fn, _ = _make_tool_executor(memory_records=["Prior incident: VM restart fixed it."])
    captured_messages: list[dict[str, Any]] = []

    async def _capturing_complete(**kwargs: Any) -> _FakeCompletion:
        captured_messages.extend(kwargs.get("messages", []))
        return _FakeCompletion(content=f"```markdown\n{_VALID_DOCUMENT}\n```")

    ctx = _make_ctx(
        complete_fn=_capturing_complete,
        resolve_model_fn=_fake_resolve_model(),
        execute_tool_fn=tool_fn,
    )
    await execute(ctx)
    user_prompt = next(m for m in captured_messages if m["role"] == "user")
    assert "Prior incident: VM restart fixed it." in user_prompt["content"]


@pytest.mark.asyncio
async def test_reads_existing_document_from_storage_when_omitted() -> None:
    """When existing_document is not supplied, the operation reads it from storage."""
    existing = "# Existing Doc\n\nOld content.\n"
    tool_fn, captures = _make_tool_executor(stored_documents={"docs/forge.md": existing})

    async def _complete(**_kwargs: Any) -> _FakeCompletion:
        return _FakeCompletion(content=f"```markdown\n{_VALID_DOCUMENT}\n```")

    ctx = _make_ctx(
        complete_fn=_complete,
        resolve_model_fn=_fake_resolve_model(),
        execute_tool_fn=tool_fn,
    )
    await execute(ctx)
    # The operation must have read the existing document from storage.
    assert "docs/forge.md" in captures["read_paths"]


@pytest.mark.asyncio
async def test_existing_document_passed_to_llm() -> None:
    """When existing_document is supplied, it must reach the LLM user prompt."""
    existing = "# Existing Doc\n\nOld content.\n"
    tool_fn, _ = _make_tool_executor()
    captured_messages: list[dict[str, Any]] = []

    async def _capturing_complete(**kwargs: Any) -> _FakeCompletion:
        captured_messages.extend(kwargs.get("messages", []))
        return _FakeCompletion(content=f"```markdown\n{_VALID_DOCUMENT}\n```")

    ctx = _make_ctx(
        complete_fn=_capturing_complete,
        resolve_model_fn=_fake_resolve_model(),
        execute_tool_fn=tool_fn,
        params={"existing_document": existing},
    )
    await execute(ctx)
    user_prompt = next(m for m in captured_messages if m["role"] == "user")
    assert "Existing Doc" in user_prompt["content"]
    assert "Old content." in user_prompt["content"]


@pytest.mark.asyncio
async def test_iteration_revises_and_persists() -> None:
    """A second call with existing_document revises the stored document."""
    original = "# Runbook\n\n## Steps\n1. Old step.\n"
    revised = "# Runbook\n\n## Steps\n1. New step.\n2. Added step.\n"
    tool_fn, captures = _make_tool_executor(stored_documents={"docs/runbook.md": original})

    async def _complete(**_kwargs: Any) -> _FakeCompletion:
        return _FakeCompletion(content=f"```markdown\n{revised}\n```")

    ctx = _make_ctx(
        query="add a step to verify the VM is healthy after restart",
        complete_fn=_complete,
        resolve_model_fn=_fake_resolve_model(),
        execute_tool_fn=tool_fn,
        params={"document_path": "docs/runbook.md"},
    )
    result = await execute(ctx)
    assert "Document forged and stored" in result.output_text
    assert "docs/runbook.md" in result.output_text
    # The operation must have read the existing document before revising.
    assert "docs/runbook.md" in captures["read_paths"]
    # The persisted content must be the revised document.
    assert len(captures["writes"]) == 1
    assert "New step." in captures["writes"][0]["content"]
    assert "Added step." in captures["writes"][0]["content"]


@pytest.mark.asyncio
async def test_no_existing_document_starts_fresh() -> None:
    """When storage has no existing document, the operation starts fresh."""
    tool_fn, captures = _make_tool_executor()  # no stored_documents

    async def _complete(**_kwargs: Any) -> _FakeCompletion:
        return _FakeCompletion(content=f"```markdown\n{_VALID_DOCUMENT}\n```")

    ctx = _make_ctx(
        complete_fn=_complete,
        resolve_model_fn=_fake_resolve_model(),
        execute_tool_fn=tool_fn,
    )
    result = await execute(ctx)
    assert "Document forged and stored" in result.output_text
    # The operation attempted a read but found nothing (not-found is normal).
    assert "docs/forge.md" in captures["read_paths"]


@pytest.mark.asyncio
async def test_document_annotated_with_model_by_default() -> None:
    """The forged document includes a non-rendering model annotation by default."""
    tool_fn, captures = _make_tool_executor()
    ctx = _make_ctx(
        complete_fn=_fake_complete_returning(_VALID_DOCUMENT),
        resolve_model_fn=_fake_resolve_model(),
        execute_tool_fn=tool_fn,
    )
    await execute(ctx)
    persisted = captures["writes"][0]["content"]
    assert "<!--" in persisted
    assert "generated by lisa" in persisted
    # The fake model id and provider must appear in the annotation.
    assert "gpt-chat-latest" in persisted
    assert "openai" in persisted


@pytest.mark.asyncio
async def test_annotation_suppressed_by_param() -> None:
    """``annotate_model: False`` suppresses the model annotation."""
    tool_fn, captures = _make_tool_executor()
    ctx = _make_ctx(
        complete_fn=_fake_complete_returning(_VALID_DOCUMENT),
        resolve_model_fn=_fake_resolve_model(),
        execute_tool_fn=tool_fn,
        params={"annotate_model": False},
    )
    await execute(ctx)
    persisted = captures["writes"][0]["content"]
    assert "generated by lisa" not in persisted


@pytest.mark.asyncio
async def test_annotation_suppressed_by_user_phrase() -> None:
    """A natural-language opt-out phrase suppresses the annotation."""
    tool_fn, captures = _make_tool_executor()
    ctx = _make_ctx(
        query="forge a runbook, but don't annotate the model",
        complete_fn=_fake_complete_returning(_VALID_DOCUMENT),
        resolve_model_fn=_fake_resolve_model(),
        execute_tool_fn=tool_fn,
    )
    await execute(ctx)
    persisted = captures["writes"][0]["content"]
    assert "generated by lisa" not in persisted


@pytest.mark.asyncio
async def test_document_formatted_by_mdformat() -> None:
    """The persisted document is normalized (single trailing newline, clean spacing)."""
    messy = (
        "# Runbook\n"
        "\n\n\n"
        "## Steps\n"
        "\n"
        "-  step one\n"
        "-   step two\n"
        "\n\n\n"
    )
    tool_fn, captures = _make_tool_executor()
    ctx = _make_ctx(
        complete_fn=_fake_complete_returning(messy),
        resolve_model_fn=_fake_resolve_model(),
        execute_tool_fn=tool_fn,
    )
    await execute(ctx)
    persisted = captures["writes"][0]["content"]
    # mdformat collapses 3+ blank lines to one and ensures a single trailing newline.
    body = persisted.split("<!--")[0]
    assert "\n\n\n" not in body
    assert persisted.endswith("\n")


# ── Truncation / continuation tests ────────────────────────────────────
#
# These exercise the finish_reason-aware continuation contract added
# 2026-07-18 (see /memories/repo/forge-document-truncation-contract.md).
# The first pass returns a partial document with finish_reason="length"
# (or "unknown"/None), and a second fake complete_fn returns the tail.
# The operation must merge the two and persist a complete document with
# NO truncated annotation.


def _fake_complete_sequence(documents: list[str], finish_reasons: list[str | None]) -> Any:
    """Return a ``complete_fn`` that returns each document/finish_reason in order.

    The first call returns (documents[0], finish_reasons[0]); the second call
    returns (documents[1], finish_reasons[1]); and so on. Used to simulate a
    truncated first pass followed by a continuation pass.
    """
    state = {"index": 0}

    async def _complete(**_kwargs: Any) -> _FakeCompletion:
        idx = state["index"]
        state["index"] += 1
        doc = documents[idx] if idx < len(documents) else ""
        finish = finish_reasons[idx] if idx < len(finish_reasons) else "stop"
        return _FakeCompletion(
            content=f"```markdown\n{doc}\n```",
            finish_reason=finish,
        )

    return _complete


_PARTIAL_SWOT = (
    "# SWOT Analysis\n"
    "\n"
    "## Strengths\n"
    "- Deterministic state control.\n"
    "\n"
    "## Weaknesses\n"
    "- Steeper learning curve.\n"
)

_SWOT_TAIL = (
    "## Opportunities\n"
    "- Enterprise adoption.\n"
    "\n"
    "## Threats\n"
    "- Integrated IDE dominance.\n"
)

_FULL_SWOT = _PARTIAL_SWOT + "\n" + _SWOT_TAIL


@pytest.mark.asyncio
async def test_truncated_first_pass_continues_and_completes() -> None:
    """finish_reason='length' triggers a continuation pass that completes the doc."""
    tool_fn, captures = _make_tool_executor()
    complete_fn = _fake_complete_sequence(
        documents=[_PARTIAL_SWOT, _SWOT_TAIL],
        finish_reasons=["length", "stop"],
    )
    ctx = _make_ctx(
        query="forge a SWOT analysis",
        complete_fn=complete_fn,
        resolve_model_fn=_fake_resolve_model(),
        execute_tool_fn=tool_fn,
    )
    result = await execute(ctx)
    assert "Document forged and stored" in result.output_text
    # The summary must NOT carry a truncation warning.
    assert "truncated" not in result.output_text.lower()
    assert result.output_json["truncated"] is False
    assert result.output_json["continue_attempts"] == 1
    persisted = captures["writes"][0]["content"]
    # Both the partial head and the continuation tail must be present.
    assert "Strengths" in persisted
    assert "Weaknesses" in persisted
    assert "Opportunities" in persisted
    assert "Threats" in persisted
    # The annotation must NOT mark the document as truncated.
    assert "truncated by model" not in persisted


@pytest.mark.asyncio
async def test_unknown_finish_reason_triggers_continuation() -> None:
    """finish_reason='unknown' (Ollama-style) is treated as truncation."""
    tool_fn, captures = _make_tool_executor()
    complete_fn = _fake_complete_sequence(
        documents=[_PARTIAL_SWOT, _SWOT_TAIL],
        finish_reasons=["unknown", "stop"],
    )
    ctx = _make_ctx(
        query="forge a SWOT analysis",
        complete_fn=complete_fn,
        resolve_model_fn=_fake_resolve_model(),
        execute_tool_fn=tool_fn,
    )
    result = await execute(ctx)
    assert result.output_json["truncated"] is False
    assert result.output_json["continue_attempts"] == 1
    persisted = captures["writes"][0]["content"]
    assert "Opportunities" in persisted
    assert "truncated by model" not in persisted


@pytest.mark.asyncio
async def test_none_finish_reason_triggers_continuation() -> None:
    """A missing finish_reason (None) is treated as truncation on a long doc."""
    tool_fn, captures = _make_tool_executor()
    complete_fn = _fake_complete_sequence(
        documents=[_PARTIAL_SWOT, _SWOT_TAIL],
        finish_reasons=[None, "stop"],
    )
    ctx = _make_ctx(
        query="forge a SWOT analysis",
        complete_fn=complete_fn,
        resolve_model_fn=_fake_resolve_model(),
        execute_tool_fn=tool_fn,
    )
    result = await execute(ctx)
    assert result.output_json["truncated"] is False
    assert result.output_json["continue_attempts"] == 1


@pytest.mark.asyncio
async def test_continuation_also_truncates_persists_with_truncated_annotation() -> None:
    """When the continuation also truncates, the artifact is marked truncated.

    Both passes return finish_reason='length'; the budget is exhausted; the
    persisted document must carry a 'truncated by model' annotation and the
    OperationResult must surface the truncation in output_json + the summary.
    """
    tool_fn, captures = _make_tool_executor()
    call_count = {"n": 0}

    async def _complete(**_kwargs: Any) -> _FakeCompletion:
        call_count["n"] += 1
        if call_count["n"] == 1:
            return _FakeCompletion(
                content=f"```markdown\n{_PARTIAL_SWOT}\n```",
                finish_reason="length",
            )
        # Continuation passes: always return a tail that ends mid-sentence
        # AND reports 'length', so the merge never looks complete and the
        # continuation budget is exhausted.
        return _FakeCompletion(
            content="```markdown\n## Opportunities\n- Enterprise adoption and the\n```",
            finish_reason="length",
        )

    ctx = _make_ctx(
        query="forge a SWOT analysis",
        complete_fn=_complete,
        resolve_model_fn=_fake_resolve_model(),
        execute_tool_fn=tool_fn,
    )
    result = await execute(ctx)
    assert result.output_json["truncated"] is True
    assert result.output_json["continue_attempts"] >= 1
    assert "truncated" in result.output_text.lower()
    persisted = captures["writes"][0]["content"]
    assert "truncated by model" in persisted
    assert "finish_reason: length" in persisted
    # The merged body is still persisted (head + tail), just marked.
    assert "Strengths" in persisted
    assert "Opportunities" in persisted


@pytest.mark.asyncio
async def test_clean_stop_does_not_trigger_continuation() -> None:
    """finish_reason='stop' returns the document as-is with no continuation."""
    tool_fn, captures = _make_tool_executor()
    call_count = {"n": 0}

    async def _complete(**_kwargs: Any) -> _FakeCompletion:
        call_count["n"] += 1
        return _FakeCompletion(
            content=f"```markdown\n{_VALID_DOCUMENT}\n```",
            finish_reason="stop",
        )

    ctx = _make_ctx(
        complete_fn=_complete,
        resolve_model_fn=_fake_resolve_model(),
        execute_tool_fn=tool_fn,
    )
    result = await execute(ctx)
    assert result.output_json["truncated"] is False
    assert result.output_json["continue_attempts"] == 0
    # Only one completion call — no continuation pass.
    assert call_count["n"] == 1
    persisted = captures["writes"][0]["content"]
    assert "truncated by model" not in persisted


@pytest.mark.asyncio
async def test_clean_stop_on_trailing_heading_triggers_continuation() -> None:
    """A clean ``stop`` ending on a dangling heading is structurally incomplete.

    Regression for the 2026-08-02 "short doc ending on a heading" incident:
    the model emitted a section heading ("### 5.1 Hook on ...") with no
    body and returned ``finish_reason="stop"``. The old heuristic treated
    a trailing heading as complete, so no continuation ran and the
    half-written section was persisted. The fix: a trailing heading with
    no body is a truncation signal regardless of finish_reason; the op
    attempts a continuation that supplies the missing body.
    """
    tool_fn, captures = _make_tool_executor()
    # First pass: ends on a heading with no body, clean stop.
    # Continuation pass: supplies the body and ends on a clean sentence.
    heading_only_doc = (
        "# Spec\n"
        "\n"
        "## Design\n"
        "Body of the design section.\n"
        "\n"
        "### 5.1 Hook on OperationExecutionContext\n"
    )
    continuation_body = (
        "A new callable attribute `delegate_fn` is added to the context.\n"
        "It captures the parent's tool-call loop return address.\n"
    )
    complete_fn = _fake_complete_sequence(
        documents=[heading_only_doc, continuation_body],
        finish_reasons=["stop", "stop"],
    )
    ctx = _make_ctx(
        query="forge a delegate design spec",
        complete_fn=complete_fn,
        resolve_model_fn=_fake_resolve_model(),
        execute_tool_fn=tool_fn,
    )
    result = await execute(ctx)
    assert "Document forged and stored" in result.output_text
    # A continuation must have been attempted despite the clean stop.
    assert result.output_json["continue_attempts"] == 1
    # The continuation supplied the body, so the final artifact is NOT marked truncated.
    assert result.output_json["truncated"] is False
    persisted = captures["writes"][0]["content"]
    # The missing body must be present in the persisted document.
    assert "delegate_fn" in persisted
    assert "truncated by model" not in persisted


@pytest.mark.asyncio
async def test_continuation_empty_gives_up_and_marks_truncated() -> None:
    """A continuation that returns empty content stops and marks the doc truncated."""
    tool_fn, captures = _make_tool_executor()
    call_count = {"n": 0}

    async def _complete(**_kwargs: Any) -> _FakeCompletion:
        call_count["n"] += 1
        if call_count["n"] == 1:
            return _FakeCompletion(
                content=f"```markdown\n{_PARTIAL_SWOT}\n```",
                finish_reason="length",
            )
        # Continuation pass: truly empty content (not fenced).
        return _FakeCompletion(content="", finish_reason="stop")

    ctx = _make_ctx(
        query="forge a SWOT analysis",
        complete_fn=_complete,
        resolve_model_fn=_fake_resolve_model(),
        execute_tool_fn=tool_fn,
    )
    result = await execute(ctx)
    assert result.output_json["truncated"] is True
    assert result.output_json["continue_attempts"] == 1
    persisted = captures["writes"][0]["content"]
    assert "truncated by model" in persisted
    # The partial head is still persisted.
    assert "Strengths" in persisted
    # The tail that never arrived is absent.
    assert "Opportunities" not in persisted


@pytest.mark.asyncio
async def test_continuation_duplicate_tail_gives_up_and_marks_truncated() -> None:
    """A continuation that just re-emits the existing tail is rejected as a duplicate."""
    tool_fn, captures = _make_tool_executor()
    # The continuation re-emits the partial head verbatim — a known local-model
    # failure mode where "continue" produces a loop instead of new content.
    complete_fn = _fake_complete_sequence(
        documents=[_PARTIAL_SWOT, _PARTIAL_SWOT],
        finish_reasons=["length", "stop"],
    )
    ctx = _make_ctx(
        query="forge a SWOT analysis",
        complete_fn=complete_fn,
        resolve_model_fn=_fake_resolve_model(),
        execute_tool_fn=tool_fn,
    )
    result = await execute(ctx)
    assert result.output_json["truncated"] is True
    assert result.output_json["continue_attempts"] == 1
    persisted = captures["writes"][0]["content"]
    assert "truncated by model" in persisted
    # The duplicate tail must not have been merged — Opportunities still absent.
    assert "Opportunities" not in persisted

# ── Phase A: advisory lint gate tests ──────────────────────────────────


def _fake_complete_spy() -> tuple[Any, list[dict[str, Any]]]:
    """Return a ``complete_fn`` that records every call's kwargs and returns a valid doc."""
    calls: list[dict[str, Any]] = []

    async def _complete(**kwargs: Any) -> _FakeCompletion:
        calls.append(kwargs)
        return _FakeCompletion(content=f"```markdown\n{_VALID_DOCUMENT}\n```")

    return _complete, calls


@pytest.mark.asyncio
async def test_clean_document_passes_lint_no_annotation() -> None:
    """A clean document produces no lint comment in the persisted artifact."""
    tool_fn, captures = _make_tool_executor()
    ctx = _make_ctx(
        complete_fn=_fake_complete_returning(_VALID_DOCUMENT),
        resolve_model_fn=_fake_resolve_model(),
        execute_tool_fn=tool_fn,
    )
    result = await execute(ctx)
    assert "Document forged" in result.output_text
    persisted = captures["writes"][0]["content"]
    assert "lint warnings" not in persisted
    assert result.output_json["lint_warnings"] == 0
    assert result.output_json["lint_repair_used"] is False


@pytest.mark.asyncio
async def test_lint_issues_trigger_repair_pass() -> None:
    """A document with a skipped heading level triggers one lint repair call."""
    malformed = (
        "# Title\n"
        "\n"
        "#### Skipped\n"  # MD001: h1 -> h4 skips levels
        "\n"
        "Body.\n"
    )
    calls: list[dict[str, Any]] = []

    async def _complete(**kwargs: Any) -> _FakeCompletion:
        calls.append(kwargs)
        # First call returns the malformed doc; the repair call returns a fixed doc.
        if len(calls) == 1:
            return _FakeCompletion(content=f"```markdown\n{malformed}\n```", finish_reason="stop")
        return _FakeCompletion(content=f"```markdown\n{_VALID_DOCUMENT}\n```", finish_reason="stop")

    tool_fn, captures = _make_tool_executor()
    ctx = _make_ctx(
        complete_fn=_complete,
        resolve_model_fn=_fake_resolve_model(),
        execute_tool_fn=tool_fn,
    )
    result = await execute(ctx)
    assert result.output_json["lint_repair_used"] is True
    # Two complete_fn calls: initial forge + one lint repair.
    assert len(calls) == 2
    # The repair call's user prompt contains the lint issues block.
    repair_user = calls[1]["messages"][1]["content"]
    assert "Structural lint issues to fix" in repair_user


@pytest.mark.asyncio
async def test_lint_repair_disabled_by_config() -> None:
    """When ``forge_lint_repair`` is False, issues persist as an advisory comment, no repair call."""
    malformed = "# Title\n\n#### Skipped\n\nBody.\n"
    calls: list[dict[str, Any]] = []

    async def _complete(**kwargs: Any) -> _FakeCompletion:
        calls.append(kwargs)
        return _FakeCompletion(content=f"```markdown\n{malformed}\n```", finish_reason="stop")

    tool_fn, captures = _make_tool_executor()
    ctx = _make_ctx(
        complete_fn=_complete,
        resolve_model_fn=_fake_resolve_model(),
        execute_tool_fn=tool_fn,
        pack_config={"forge_lint_repair": False},
    )
    result = await execute(ctx)
    assert result.output_json["lint_repair_used"] is False
    assert result.output_json["lint_warnings"] >= 1
    # Only the initial forge call — no repair pass.
    assert len(calls) == 1
    persisted = captures["writes"][0]["content"]
    assert "lint warnings" in persisted


@pytest.mark.asyncio
async def test_pymarkdownlnt_missing_silently_skips() -> None:
    """When the linter is unavailable, the op still succeeds (soft dependency)."""
    malformed = "# Title\n\n#### Skipped\n\nBody.\n"
    tool_fn, captures = _make_tool_executor()
    import lisa_packs.markdown.operations.forge_document.forge_document as fd_mod

    real_lint = fd_mod.lint_markdown

    def _empty_lint(document: str, *, rules_csv: str | None = None) -> list:
        return []

    fd_mod.lint_markdown = _empty_lint  # type: ignore[assignment]
    try:
        ctx = _make_ctx(
            complete_fn=_fake_complete_returning(malformed),
            resolve_model_fn=_fake_resolve_model(),
            execute_tool_fn=tool_fn,
        )
        result = await execute(ctx)
    finally:
        fd_mod.lint_markdown = real_lint  # type: ignore[assignment]
    assert "Document forged" in result.output_text
    assert result.output_json["lint_warnings"] == 0
    persisted = captures["writes"][0]["content"]
    assert "lint warnings" not in persisted


@pytest.mark.asyncio
async def test_lint_repair_introduces_no_new_truncation() -> None:
    """A repair pass that returns finish_reason=stop does not trigger continuation."""
    malformed = "# Title\n\n#### Skipped\n\nBody.\n"
    calls: list[dict[str, Any]] = []

    async def _complete(**kwargs: Any) -> _FakeCompletion:
        calls.append(kwargs)
        if len(calls) == 1:
            return _FakeCompletion(content=f"```markdown\n{malformed}\n```", finish_reason="stop")
        return _FakeCompletion(content=f"```markdown\n{_VALID_DOCUMENT}\n```", finish_reason="stop")

    tool_fn, captures = _make_tool_executor()
    ctx = _make_ctx(
        complete_fn=_complete,
        resolve_model_fn=_fake_resolve_model(),
        execute_tool_fn=tool_fn,
    )
    result = await execute(ctx)
    assert result.output_json["lint_repair_used"] is True
    assert result.output_json["truncated"] is False
    assert result.output_json["continue_attempts"] == 0


# ── Phase B: target-length-driven budgeting tests ──────────────────────


@pytest.mark.asyncio
async def test_target_length_scales_max_tokens() -> None:
    """A target_length derives max_tokens from the word target, not the flat 50k."""
    _complete, calls = _fake_complete_spy()
    tool_fn, _captures = _make_tool_executor()
    ctx = _make_ctx(
        complete_fn=_complete,
        resolve_model_fn=_fake_resolve_model(),
        execute_tool_fn=tool_fn,
        params={"target_length": 500},
    )
    await execute(ctx)
    import math
    expected = min(50_000, max(512, math.ceil(500 * 1.3)))
    assert calls[0]["max_tokens"] == expected


@pytest.mark.asyncio
async def test_target_length_none_preserves_flat_budget() -> None:
    """No target_length keeps the legacy flat 50k max_tokens (no regression)."""
    _complete, calls = _fake_complete_spy()
    tool_fn, _captures = _make_tool_executor()
    ctx = _make_ctx(
        complete_fn=_complete,
        resolve_model_fn=_fake_resolve_model(),
        execute_tool_fn=tool_fn,
    )
    await execute(ctx)
    assert calls[0]["max_tokens"] == 50_000


@pytest.mark.asyncio
async def test_target_length_zero_coerced_to_default() -> None:
    """target_length=0 coerces to the configured default target word count."""
    _complete, calls = _fake_complete_spy()
    tool_fn, _captures = _make_tool_executor()
    ctx = _make_ctx(
        complete_fn=_complete,
        resolve_model_fn=_fake_resolve_model(),
        execute_tool_fn=tool_fn,
        params={"target_length": 0},
    )
    await execute(ctx)
    import math
    from lisa_packs.markdown.operations.forge_document.forge_document import _DEFAULT_TARGET_WORDS

    expected = min(50_000, max(512, math.ceil(_DEFAULT_TARGET_WORDS * 1.3)))
    assert calls[0]["max_tokens"] == expected


@pytest.mark.asyncio
async def test_target_length_exceeds_char_cap_is_clamped() -> None:
    """A target whose char equivalent exceeds max_document_chars is clamped down."""
    _complete, calls = _fake_complete_spy()
    tool_fn, _captures = _make_tool_executor()
    # 10000 words * ~5 chars = 50000 chars, well over the 20000 default char cap.
    ctx = _make_ctx(
        complete_fn=_complete,
        resolve_model_fn=_fake_resolve_model(),
        execute_tool_fn=tool_fn,
        params={"target_length": 10_000},
    )
    await execute(ctx)
    # Clamped to 20000 // 5 = 4000 words -> ceil(4000 * 1.3) = 5200 tokens.
    import math
    expected = min(50_000, max(512, math.ceil(4000 * 1.3)))
    assert calls[0]["max_tokens"] == expected


@pytest.mark.asyncio
async def test_target_length_guidance_in_prompt() -> None:
    """The target length guidance line appears in the user prompt only when set."""
    _complete, calls = _fake_complete_spy()
    tool_fn, _captures = _make_tool_executor()
    ctx = _make_ctx(
        complete_fn=_complete,
        resolve_model_fn=_fake_resolve_model(),
        execute_tool_fn=tool_fn,
        params={"target_length": 800},
    )
    await execute(ctx)
    user_prompt = calls[0]["messages"][1]["content"]
    assert "Target length: approximately 800 words" in user_prompt


@pytest.mark.asyncio
async def test_target_length_none_omits_guidance_in_prompt() -> None:
    """No target_length means no guidance line in the prompt (byte-identical to legacy)."""
    _complete, calls = _fake_complete_spy()
    tool_fn, _captures = _make_tool_executor()
    ctx = _make_ctx(
        complete_fn=_complete,
        resolve_model_fn=_fake_resolve_model(),
        execute_tool_fn=tool_fn,
    )
    await execute(ctx)
    user_prompt = calls[0]["messages"][1]["content"]
    assert "Target length" not in user_prompt


@pytest.mark.asyncio
async def test_target_length_scales_continue_budget() -> None:
    """A truncating first pass uses a continuation budget scaled to the target."""
    partial = "# Title\n\n## Section\n- item\n"  # will be detected as truncated via finish_reason
    full = _VALID_DOCUMENT
    complete_fn = _fake_complete_sequence(
        documents=[partial, full],
        finish_reasons=["length", "stop"],
    )
    tool_fn, _captures = _make_tool_executor()
    ctx = _make_ctx(
        complete_fn=complete_fn,
        resolve_model_fn=_fake_resolve_model(),
        execute_tool_fn=tool_fn,
        params={"target_length": 500},
    )
    result = await execute(ctx)
    # Continue budget = min(50000, first_pass*2). first_pass = ceil(500*1.3) = 650.
    assert result.output_json["truncated"] is False
    assert result.output_json["continue_attempts"] == 1
