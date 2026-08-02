"""Tests for the source-aware consolidation path of markdown_forge_document.

Covers the 2026-07-30 contract extension:
  - source_paths parsing (bare strings, dicts with NL priority, globs)
  - source reading via read_file + glob expansion via list_files
  - per-source priority notes reaching the LLM prompt
  - the density floor (anti-abridgment gate) tripping and triggering a retry
  - OKF provenance capture (okf_emit_concept + okf_log)
  - the extended non-rendering footer carrying sources/processing/OKF pointer
  - the single-doc/memory-only path unchanged when source_paths is absent
    (regression guard — see test_forge_document.py for the full single-doc
    suite)

These tests inject a content-registry loader reading the real authored
prompt files, a fake complete_fn, and a fake execute_tool_fn simulating
list_files/read_file/memory.search/markdown_generate/write_file/
okf_emit_concept/okf_log.
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
    def read_authored_text(relative_path: str) -> str:
        full_path = _CONTENT_ROOT / relative_path
        if not full_path.exists():
            raise FileNotFoundError(f"Authored text not found: {full_path}")
        return full_path.read_text(encoding="utf-8")

    return read_authored_text


@dataclass
class _FakeModel:
    provider: str = "ollama"
    model_id: str = "glm-5.2:cloud"
    display_name: str = "GLM"
    reasoning_model: bool = False
    thinking_model: bool = False


@dataclass
class _FakeCompletion:
    content: str
    provider: str = "ollama"
    model_id: str = "glm-5.2:cloud"
    display_name: str = "GLM"
    request_id: str = "test"
    tool_calls: list[Any] | None = None
    usage: Any = None
    finish_reason: str | None = "stop"


def _make_tool_executor(
    *,
    stored_sources: dict[str, str] | None = None,
    list_files_map: dict[str, list[str]] | None = None,
    memory_records: list[str] | None = None,
    write_ok: bool = True,
    okf_emit_ok: bool = True,
) -> tuple[Any, dict[str, Any]]:
    """Fake execute_tool_fn with list_files + OKF support.

    ``stored_sources`` seeds read_file responses (path -> text).
    ``list_files_map`` seeds list_files responses (prefix -> list of paths).
    ``captures`` records write_file, okf_emit_concept, and okf_log calls.
    """
    store: dict[str, str] = dict(stored_sources or {})
    list_map: dict[str, list[str]] = dict(list_files_map or {})
    captures: dict[str, Any] = {
        "writes": [],
        "okf_emits": [],
        "okf_logs": [],
        "read_paths": [],
        "list_prefixes": [],
        "memory_query": None,
        "outline_markdown": None,
    }

    async def execute_tool_fn(tool_id: str, ctx: Any) -> ToolResult:
        params = getattr(ctx, "parameters", {}) or {}
        if tool_id == "memory.search":
            captures["memory_query"] = params.get("query")
            if memory_records is None:
                return ToolResult(tool_id=tool_id, output_text="(no memories)")
            return ToolResult(
                tool_id=tool_id,
                output_text="\n".join(f"- {r}" for r in memory_records),
            )
        if tool_id == "markdown_generate":
            captures["outline_markdown"] = params.get("markdown")
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
        if tool_id == "list_files":
            prefix = str(params.get("prefix") or "")
            captures["list_prefixes"].append(prefix)
            matches = list_map.get(prefix, [])
            return ToolResult(
                tool_id=tool_id,
                output_text=json.dumps(matches),
            )
        if tool_id == "write_file":
            path = str(params.get("path") or "")
            content = str(params.get("content") or "")
            captures["writes"].append({"path": path, "content": content})
            if not write_ok:
                return ToolResult(
                    tool_id=tool_id,
                    output_text=json.dumps({"error": "read-only"}),
                )
            store[path] = content
            return ToolResult(
                tool_id=tool_id,
                output_text=json.dumps(
                    {"ok": True, "path": path, "bytes": len(content.encode("utf-8"))}
                ),
            )
        if tool_id == "okf_emit_concept":
            captures["okf_emits"].append(dict(params))
            if not okf_emit_ok:
                return ToolResult(
                    tool_id=tool_id,
                    output_text="OKF emit failed: storage error",
                )
            full_path = f"{params.get('bundle_path')}/{params.get('concept_path')}"
            return ToolResult(
                tool_id=tool_id,
                output_text=f"OKF concept emitted: {full_path}",
            )
        if tool_id == "okf_log":
            captures["okf_logs"].append(dict(params))
            return ToolResult(tool_id=tool_id, output_text="OKF log appended")
        return ToolResult(tool_id=tool_id, error=f"unknown tool: {tool_id}")

    return execute_tool_fn, captures


def _make_ctx(
    *,
    query: str = "consolidate the source docs into one narrative",
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
        pack_config=pack_config or {},
    )


def _single_pass_pack_config() -> dict[str, Any]:
    """Pack config that forces the single-pass path (raises both thresholds).

    Used by the density-floor tests so they exercise the single-pass
    re-forge contract, which only runs when the map-reduce path is not
    selected. With byte threshold far above the test source size and
    source threshold above the test's source count, the op takes the
    single-pass branch.
    """
    return {
        "forge_map_reduce_byte_threshold": 10_000_000,
        "forge_map_reduce_source_threshold": 100,
    }


def _map_reduce_pack_config() -> dict[str, Any]:
    """Pack config that forces the map-reduce path (low thresholds)."""
    return {
        "forge_map_reduce_byte_threshold": 0,
        "forge_map_reduce_source_threshold": 0,
    }


def _fake_resolve_model() -> Any:
    def _resolve(_alias: str) -> _FakeModel:
        return _FakeModel()

    return _resolve


def _fake_complete_returning(document: str, finish_reason: str | None = "stop") -> Any:
    async def _complete(**_kwargs: Any) -> _FakeCompletion:
        return _FakeCompletion(
            content=f"```markdown\n{document}\n```",
            finish_reason=finish_reason,
        )

    return _complete


# A consolidation output that is deliberately small relative to the sources,
# to exercise the density floor. ~120 bytes of output vs ~3000 bytes source.
_ABRIDGED_OUTPUT = (
    "# Framework Narrative\n\nLisa is an MCP-native agent harness. "
    "It has strengths and weaknesses. See the SWOT.\n"
)

# A consolidation output that preserves substance — comparable size to sources.
_FULL_OUTPUT = (
    "# Framework Narrative\n\n"
    "## Strengths\n- MCP-native architecture\n- 6 first-party MCP servers\n"
    "- Fork mechanism\n- IPSAR FSM\n- OpenTelemetry\n- Pack ecosystem\n\n"
    "## Weaknesses\n- Steeper learning curve\n- No A2A protocol\n"
    "- No code sandbox\n- No browser automation\n\n"
    "## Opportunities\n- Enterprise adoption\n- OKF as interchange format\n\n"
    "## Threats\n- Integrated IDE dominance\n- Framework fatigue\n\n"
    "## Comparison\n| Framework | MCP-native | FSM |\n| --- | --- | --- |\n"
    "| Lisa | yes | yes |\n| LangChain | no | partial |\n"
)


_SOURCE_A = "# Source A: SWOT\n\n## Strengths\n- MCP-native architecture\n- 6 first-party MCP servers\n- Fork mechanism\n- IPSAR FSM\n- OpenTelemetry\n- Pack ecosystem\n\n## Weaknesses\n- Steeper learning curve\n- No A2A protocol\n- No code sandbox\n- No browser automation\n"
_SOURCE_B = "# Source B: Comparison\n\n## Comparison\n| Framework | MCP-native | FSM |\n| --- | --- | --- |\n| Lisa | yes | yes |\n| LangChain | no | partial |\n| smolagents | no | no |\n\n## Opportunities\n- Enterprise adoption\n- OKF as interchange format\n"


# ── source_paths parsing + reading ─────────────────────────────────────


@pytest.mark.asyncio
async def test_source_paths_reads_each_source_and_injects_into_prompt() -> None:
    """Each source is read and appears in the LLM user prompt with its priority."""
    tool_fn, captures = _make_tool_executor(
        stored_sources={"a.md": _SOURCE_A, "b.md": _SOURCE_B},
    )
    captured_messages: list[dict[str, Any]] = []

    async def _capturing_complete(**kwargs: Any) -> _FakeCompletion:
        captured_messages.extend(kwargs.get("messages", []))
        return _FakeCompletion(content=f"```markdown\n{_FULL_OUTPUT}\n```")

    ctx = _make_ctx(
        complete_fn=_capturing_complete,
        resolve_model_fn=_fake_resolve_model(),
        execute_tool_fn=tool_fn,
        params={
            "document_path": "narrative.md",
            "source_paths": [
                {"path": "a.md", "priority": "primary — keep section structure"},
                {"path": "b.md", "priority": "secondary — dedupe, keep unique"},
            ],
        },
        pack_config=_single_pass_pack_config(),
    )
    await execute(ctx)
    # Both sources must have been read.
    assert "a.md" in captures["read_paths"]
    assert "b.md" in captures["read_paths"]
    # The system prompt must be the source-aware (preserve-detail) one.
    system_msg = next(m for m in captured_messages if m["role"] == "system")
    assert "PRESERVE DETAIL" in system_msg["content"]
    assert "DO NOT SUMMARIZE" in system_msg["content"]
    # The user prompt must contain the source block with both sources and
    # their verbatim priority notes.
    user_msg = next(m for m in captured_messages if m["role"] == "user")
    assert "## Source Documents" in user_msg["content"]
    assert "Source A: SWOT" in user_msg["content"]
    assert "Source B: Comparison" in user_msg["content"]
    assert "primary — keep section structure" in user_msg["content"]
    assert "secondary — dedupe, keep unique" in user_msg["content"]


@pytest.mark.asyncio
async def test_source_paths_bare_string_defaults_to_normal_priority() -> None:
    """A bare path string (no dict) gets the default 'normal' priority."""
    tool_fn, _ = _make_tool_executor(stored_sources={"a.md": _SOURCE_A})
    captured_messages: list[dict[str, Any]] = []

    async def _capturing_complete(**kwargs: Any) -> _FakeCompletion:
        captured_messages.extend(kwargs.get("messages", []))
        return _FakeCompletion(content=f"```markdown\n{_FULL_OUTPUT}\n```")

    ctx = _make_ctx(
        complete_fn=_capturing_complete,
        resolve_model_fn=_fake_resolve_model(),
        execute_tool_fn=tool_fn,
        params={"document_path": "n.md", "source_paths": ["a.md"]},
    )
    await execute(ctx)
    user_msg = next(m for m in captured_messages if m["role"] == "user")
    assert "(priority: normal)" in user_msg["content"]


@pytest.mark.asyncio
async def test_source_paths_glob_expands_via_list_files() -> None:
    """A glob entry is expanded via list_files; matches inherit the glob's priority."""
    tool_fn, captures = _make_tool_executor(
        stored_sources={"a.md": _SOURCE_A, "b.md": _SOURCE_B},
        list_files_map={"": ["a.md", "b.md"]},
    )

    async def _complete(**_kwargs: Any) -> _FakeCompletion:
        return _FakeCompletion(content=f"```markdown\n{_FULL_OUTPUT}\n```")

    ctx = _make_ctx(
        complete_fn=_complete,
        resolve_model_fn=_fake_resolve_model(),
        execute_tool_fn=tool_fn,
        params={
            "document_path": "n.md",
            "source_paths": [{"path": "*.md", "priority": "primary — all of them"}],
        },
    )
    result = await execute(ctx)
    assert "Document forged and stored" in result.output_text
    # list_files must have been called with the glob prefix.
    assert "" in captures["list_prefixes"]
    # Both expanded files must have been read.
    assert "a.md" in captures["read_paths"]
    assert "b.md" in captures["read_paths"]


@pytest.mark.asyncio
async def test_source_paths_rejects_parent_traversal() -> None:
    """A source_paths entry with '..' is rejected before any tool call."""
    tool_fn, _ = _make_tool_executor()
    ctx = _make_ctx(
        complete_fn=_fake_complete_returning(_FULL_OUTPUT),
        resolve_model_fn=_fake_resolve_model(),
        execute_tool_fn=tool_fn,
        params={"source_paths": ["../escape.md"]},
    )
    result = await execute(ctx)
    assert ".." in result.output_text or "source_paths" in result.output_text


@pytest.mark.asyncio
async def test_source_paths_all_unreadable_returns_error() -> None:
    """When every source fails to read, the operation reports it and stops."""
    tool_fn, _ = _make_tool_executor(stored_sources={})  # nothing stored
    ctx = _make_ctx(
        complete_fn=_fake_complete_returning(_FULL_OUTPUT),
        resolve_model_fn=_fake_resolve_model(),
        execute_tool_fn=tool_fn,
        params={"source_paths": ["missing.md"]},
    )
    result = await execute(ctx)
    assert "unreadable" in result.output_text.lower() or "no source" in result.output_text.lower()


# ── Density floor ──────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_density_floor_trips_on_abridged_output_and_reforges() -> None:
    """A sub-floor output triggers a re-forge pass; the retry output is used."""
    tool_fn, _ = _make_tool_executor(
        stored_sources={"a.md": _SOURCE_A, "b.md": _SOURCE_B},
    )
    call_count = {"n": 0}

    async def _complete(**_kwargs: Any) -> _FakeCompletion:
        call_count["n"] += 1
        # First pass: abridged (small). Retry: full (preserves substance).
        doc = _ABRIDGED_OUTPUT if call_count["n"] == 1 else _FULL_OUTPUT
        return _FakeCompletion(content=f"```markdown\n{doc}\n```")

    ctx = _make_ctx(
        complete_fn=_complete,
        resolve_model_fn=_fake_resolve_model(),
        execute_tool_fn=tool_fn,
        params={
            "document_path": "n.md",
            "source_paths": [
                {"path": "a.md", "priority": "primary"},
                {"path": "b.md", "priority": "secondary"},
            ],
        },
        pack_config=_single_pass_pack_config(),
    )
    result = await execute(ctx)
    # The retry must have happened (two completion calls).
    assert call_count["n"] == 2
    # Density check must report passed after the retry.
    assert result.output_json["density_passed"] is True
    assert result.output_json["density_retried"] is True
    assert "density FAILED" not in result.output_text


@pytest.mark.asyncio
async def test_density_floor_failure_persists_with_warning() -> None:
    """When the retry still abridges, the artifact is persisted with a density-FAILED warning."""
    tool_fn, captures = _make_tool_executor(
        stored_sources={"a.md": _SOURCE_A, "b.md": _SOURCE_B},
    )
    call_count = {"n": 0}

    async def _complete(**_kwargs: Any) -> _FakeCompletion:
        call_count["n"] += 1
        # Both passes return the abridged output — retry does not help.
        return _FakeCompletion(content=f"```markdown\n{_ABRIDGED_OUTPUT}\n```")

    ctx = _make_ctx(
        complete_fn=_complete,
        resolve_model_fn=_fake_resolve_model(),
        execute_tool_fn=tool_fn,
        params={
            "document_path": "n.md",
            "source_paths": ["a.md", "b.md"],
        },
        pack_config=_single_pass_pack_config(),
    )
    result = await execute(ctx)
    # The retry happened but did not pass the floor.
    assert result.output_json["density_retried"] is True
    assert result.output_json["density_passed"] is False
    assert "density floor FAILED" in result.output_text
    persisted = captures["writes"][0]["content"]
    # The persisted document must carry the density-FAILED marker in the footer.
    assert "density FAILED" in persisted


@pytest.mark.asyncio
async def test_density_floor_not_applied_without_source_paths() -> None:
    """The single-doc path has no density floor (regression guard)."""
    tool_fn, _ = _make_tool_executor()
    ctx = _make_ctx(
        complete_fn=_fake_complete_returning(_ABRIDGED_OUTPUT),
        resolve_model_fn=_fake_resolve_model(),
        execute_tool_fn=tool_fn,
        params={"document_path": "n.md"},  # no source_paths
    )
    result = await execute(ctx)
    assert result.output_json.get("density_passed") is True
    assert result.output_json.get("source_count", 0) == 0


# ── Map-reduce consolidation ───────────────────────────────────────────


@pytest.mark.asyncio
async def test_map_reduce_forges_per_source_then_merges() -> None:
    """When thresholds trip, one section is forged per source then merged.

    Regression for the 2026-08-01 chunked-consolidation change (Phase 2):
    a multi-source consolidation above the map-reduce threshold issues
    N per-source completion calls + 1 merge call, rather than one giant
    single-pass call. Each per-source call uses the per-source system
    prompt; the merge uses the merge system prompt.
    """
    tool_fn, captures = _make_tool_executor(
        stored_sources={"a.md": _SOURCE_A, "b.md": _SOURCE_B},
    )
    call_log: list[dict[str, Any]] = []

    async def _complete(**kwargs: Any) -> _FakeCompletion:
        messages = kwargs.get("messages", [])
        system_content = next((m["content"] for m in messages if m["role"] == "system"), "")
        # Per-source calls return a section; the merge returns the full doc.
        if "Per-Source Consolidation Section" in system_content:
            call_log.append({"phase": "per-source", "system": system_content})
            return _FakeCompletion(
                content="```markdown\n# Section\n\nforged detail for one source\n```"
            )
        if "Section Merge" in system_content:
            call_log.append({"phase": "merge", "system": system_content})
            return _FakeCompletion(content=f"```markdown\n{_FULL_OUTPUT}\n```")
        call_log.append({"phase": "unknown", "system": system_content})
        return _FakeCompletion(content=f"```markdown\n{_FULL_OUTPUT}\n```")

    ctx = _make_ctx(
        complete_fn=_complete,
        resolve_model_fn=_fake_resolve_model(),
        execute_tool_fn=tool_fn,
        params={
            "document_path": "n.md",
            "source_paths": [
                {"path": "a.md", "priority": "primary"},
                {"path": "b.md", "priority": "secondary"},
            ],
        },
        pack_config=_map_reduce_pack_config(),
    )
    result = await execute(ctx)
    assert "Document forged and stored" in result.output_text
    # Two per-source calls + one merge call.
    per_source_calls = [c for c in call_log if c["phase"] == "per-source"]
    merge_calls = [c for c in call_log if c["phase"] == "merge"]
    assert len(per_source_calls) == 2
    assert len(merge_calls) == 1
    # Both sources were read.
    assert "a.md" in captures["read_paths"]
    assert "b.md" in captures["read_paths"]


@pytest.mark.asyncio
async def test_map_reduce_default_threshold_takes_chunked_path() -> None:
    """With default config, 2+ sources take the map-reduce path (no single-pass re-forge)."""
    tool_fn, _ = _make_tool_executor(
        stored_sources={"a.md": _SOURCE_A, "b.md": _SOURCE_B},
    )
    call_count = {"n": 0}

    async def _complete(**_kwargs: Any) -> _FakeCompletion:
        call_count["n"] += 1
        return _FakeCompletion(content=f"```markdown\n{_FULL_OUTPUT}\n```")

    ctx = _make_ctx(
        complete_fn=_complete,
        resolve_model_fn=_fake_resolve_model(),
        execute_tool_fn=tool_fn,
        params={
            "document_path": "n.md",
            "source_paths": ["a.md", "b.md"],
        },
        # No pack_config → defaults → n_readable(2) >= source_threshold(2) → map-reduce.
    )
    result = await execute(ctx)
    assert "Document forged and stored" in result.output_text
    # Map-reduce issues 2 per-source + 1 merge = 3 calls (no single-pass re-forge).
    assert call_count["n"] == 3
    # The map-reduce path does not set density_retried (no single-pass re-forge).
    assert result.output_json.get("density_retried", False) is False


# ── OKF provenance + extended footer ───────────────────────────────────


@pytest.mark.asyncio
async def test_okf_provenance_concept_written_and_logged() -> None:
    """Source-aware forges write an OKF forge-provenance concept + log entry."""
    tool_fn, captures = _make_tool_executor(
        stored_sources={"a.md": _SOURCE_A, "b.md": _SOURCE_B},
    )
    ctx = _make_ctx(
        complete_fn=_fake_complete_returning(_FULL_OUTPUT),
        resolve_model_fn=_fake_resolve_model(),
        execute_tool_fn=tool_fn,
        params={
            "document_path": "narrative.md",
            "source_paths": ["a.md", "b.md"],
            "okf_bundle_path": "okf/forge-provenance",
        },
    )
    result = await execute(ctx)
    # okf_emit_concept must have been called with the provenance type.
    assert len(captures["okf_emits"]) == 1
    emit = captures["okf_emits"][0]
    assert emit["type"] == "forge-provenance"
    assert emit["bundle_path"] == "okf/forge-provenance"
    assert "narrative.forge" in emit["concept_path"]
    assert "Forge Provenance" in emit["body"]
    assert "## Sources" in emit["body"]
    # okf_log must have been called with action=forge.
    assert len(captures["okf_logs"]) == 1
    assert captures["okf_logs"][0]["action"] == "forge"
    # The result summary must reference the OKF concept path.
    assert "okf/forge-provenance/narrative.forge.md" in result.output_text


@pytest.mark.asyncio
async def test_okf_provenance_skipped_when_bundle_path_empty() -> None:
    """An empty okf_bundle_path suppresses OKF capture (footer still written)."""
    tool_fn, captures = _make_tool_executor(
        stored_sources={"a.md": _SOURCE_A},
    )
    ctx = _make_ctx(
        complete_fn=_fake_complete_returning(_FULL_OUTPUT),
        resolve_model_fn=_fake_resolve_model(),
        execute_tool_fn=tool_fn,
        params={
            "document_path": "n.md",
            "source_paths": ["a.md"],
            "okf_bundle_path": "",
        },
    )
    result = await execute(ctx)
    assert captures["okf_emits"] == []
    assert result.output_json.get("okf_provenance_path") is None


@pytest.mark.asyncio
async def test_extended_footer_present_in_persisted_document() -> None:
    """The persisted document carries the extended non-rendering footer."""
    tool_fn, captures = _make_tool_executor(
        stored_sources={"a.md": _SOURCE_A, "b.md": _SOURCE_B},
    )
    ctx = _make_ctx(
        complete_fn=_fake_complete_returning(_FULL_OUTPUT),
        resolve_model_fn=_fake_resolve_model(),
        execute_tool_fn=tool_fn,
        params={
            "document_path": "n.md",
            "source_paths": [
                {"path": "a.md", "priority": "primary — keep structure"},
                {"path": "b.md", "priority": "secondary — dedupe"},
            ],
            "okf_bundle_path": "okf/fp",
        },
    )
    await execute(ctx)
    persisted = captures["writes"][0]["content"]
    # The footer is an HTML comment block.
    assert "<!--" in persisted and "-->" in persisted
    # It must carry the model attribution.
    assert "generated by lisa" in persisted
    assert "glm-5.2:cloud" in persisted
    # It must summarize the sources.
    assert "sources: 2" in persisted
    # It must reference the OKF provenance concept.
    assert "provenance: okf/fp/" in persisted
    # The density line must be present (sources supplied).
    assert "density:" in persisted


@pytest.mark.asyncio
async def test_single_doc_path_keeps_original_one_line_annotation() -> None:
    """Without source_paths, the footer is the original single-line annotation (regression)."""
    tool_fn, captures = _make_tool_executor()
    valid_doc = "# Doc\n\n## Steps\n1. Step.\n"
    ctx = _make_ctx(
        complete_fn=_fake_complete_returning(valid_doc),
        resolve_model_fn=_fake_resolve_model(),
        execute_tool_fn=tool_fn,
        params={"document_path": "n.md"},
    )
    await execute(ctx)
    persisted = captures["writes"][0]["content"]
    assert "generated by lisa" in persisted
    # The single-doc footer must NOT carry the consolidation fields.
    assert "sources:" not in persisted
    assert "density:" not in persisted
    assert "provenance:" not in persisted


@pytest.mark.asyncio
async def test_okf_provenance_failure_is_non_fatal() -> None:
    """If okf_emit_concept fails, the forge still succeeds and persists."""
    tool_fn, captures = _make_tool_executor(
        stored_sources={"a.md": _SOURCE_A},
        okf_emit_ok=False,
    )
    ctx = _make_ctx(
        complete_fn=_fake_complete_returning(_FULL_OUTPUT),
        resolve_model_fn=_fake_resolve_model(),
        execute_tool_fn=tool_fn,
        params={
            "document_path": "n.md",
            "source_paths": ["a.md"],
            "okf_bundle_path": "okf/fp",
        },
    )
    result = await execute(ctx)
    # The document must still be persisted.
    assert "Document forged and stored" in result.output_text
    assert len(captures["writes"]) == 1
    # The OKF concept path must be None (emit failed).
    assert result.output_json.get("okf_provenance_path") is None


# ── Regression: single-doc path unchanged ──────────────────────────────


@pytest.mark.asyncio
async def test_no_source_paths_uses_original_system_prompt() -> None:
    """Without source_paths, the original (non-preserve-detail) system prompt is used."""
    tool_fn, _ = _make_tool_executor()
    captured_messages: list[dict[str, Any]] = []

    async def _capturing_complete(**kwargs: Any) -> _FakeCompletion:
        captured_messages.extend(kwargs.get("messages", []))
        return _FakeCompletion(content="```markdown\n# Doc\n\n## S\n1. x\n```")

    ctx = _make_ctx(
        complete_fn=_capturing_complete,
        resolve_model_fn=_fake_resolve_model(),
        execute_tool_fn=tool_fn,
        params={"document_path": "n.md"},
    )
    await execute(ctx)
    system_msg = next(m for m in captured_messages if m["role"] == "system")
    assert "PRESERVE DETAIL" not in system_msg["content"]
    user_msg = next(m for m in captured_messages if m["role"] == "user")
    assert "## Source Documents" not in user_msg["content"]