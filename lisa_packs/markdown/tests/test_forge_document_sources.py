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
from lisa_packs.markdown.operations.consolidate.consolidate import execute

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
    stat_sizes: dict[str, int] | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Fake execute_tool_fn with list_files + OKF support.

    ``stored_sources`` seeds read_file responses (path -> text).
    ``list_files_map`` seeds list_files responses (prefix -> list of paths).
    ``stat_sizes`` seeds stat_file responses (path -> byte size); when a
    path is absent, stat falls back to the stored text length (or 0).
    ``captures`` records write_file, okf_emit_concept, okf_log, stat,
    read_file_range, and append_file calls.
    """
    store: dict[str, str] = dict(stored_sources or {})
    list_map: dict[str, list[str]] = dict(list_files_map or {})
    stat_map: dict[str, int] = dict(stat_sizes or {})
    appends: dict[str, str] = {}
    captures: dict[str, Any] = {
        "writes": [],
        "okf_emits": [],
        "okf_logs": [],
        "read_paths": [],
        "list_prefixes": [],
        "memory_query": None,
        "outline_markdown": None,
        "stat_paths": [],
        "read_file_ranges": [],
        "append_file_calls": [],
        "appended_text": appends,
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
        if tool_id == "stat_file":
            path = str(params.get("path") or "")
            captures["stat_paths"].append(path)
            if path in stat_map:
                size = stat_map[path]
            elif path in store:
                size = len(store[path].encode("utf-8"))
            else:
                return ToolResult(
                    tool_id=tool_id,
                    output_text=json.dumps({"error": f"not found: {path}"}),
                )
            return ToolResult(
                tool_id=tool_id,
                output_text=json.dumps(
                    {"path": path, "size": size, "modified": "2026-08-02T00:00:00+00:00", "is_dir": False}
                ),
            )
        if tool_id == "read_file_range":
            path = str(params.get("path") or "")
            start = int(params.get("start") or 0)
            length = int(params.get("length") or 4096)
            captures["read_file_ranges"].append({"path": path, "start": start, "length": length})
            if path not in store:
                return ToolResult(
                    tool_id=tool_id,
                    output_text=json.dumps({"error": f"file not found: {path}"}),
                )
            text = store[path]
            total = len(text)
            content = text[start:start + length]
            end = start + length
            eof = end >= total
            return ToolResult(
                tool_id=tool_id,
                output_text=json.dumps(
                    {"content": content, "start": start, "length": len(content), "total": total, "eof": eof}
                ),
            )
        if tool_id == "append_file":
            path = str(params.get("path") or "")
            content = str(params.get("content") or "")
            captures["append_file_calls"].append({"path": path, "content": content})
            appends[path] = appends.get(path, "") + content
            return ToolResult(
                tool_id=tool_id,
                output_text=json.dumps(
                    {"ok": True, "path": path, "bytes": len(appends[path].encode("utf-8"))}
                ),
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
    assert "Document consolidated and stored" in result.output_text
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
    assert "Document consolidated and stored" in result.output_text
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
    assert "Document consolidated and stored" in result.output_text
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
    assert "Document consolidated and stored" in result.output_text
    assert len(captures["writes"]) == 1
    # The OKF concept path must be None (emit failed).
    assert result.output_json.get("okf_provenance_path") is None


# ── Regression: single-doc path unchanged ──────────────────────────────


# ── Per-source density floor (map-reduce path, Task 1) ────────────────
# These pin the 2026-08-02 per-source density-floor contract: a section
# implausibly tiny relative to ITS source trips one retry; if the retry
# still trips, the section is kept but annotated inline and the source
# path surfaces in ``density_failed_sources``. The single-pass density
# floor tests above are NOT modified — they pin _single_pass_pack_config()
# and must stay green.


# A section deliberately tiny relative to _SOURCE_A (~300 bytes): a
# one-line stub is well under the 10% per-source floor (the floor trips
# when section_bytes < 0.10 * source_bytes; ~300B source → ~30B floor).
_ABRIDGED_SECTION = "# S\n"

# A section preserving _SOURCE_A's substance (comparable size).
_FULL_SECTION_A = (
    "# Source A: SWOT\n\n"
    "## Strengths\n- MCP-native architecture\n- 6 first-party MCP servers\n"
    "- Fork mechanism\n- IPSAR FSM\n- OpenTelemetry\n- Pack ecosystem\n\n"
    "## Weaknesses\n- Steeper learning curve\n- No A2A protocol\n"
    "- No code sandbox\n- No browser automation\n"
)


@pytest.mark.asyncio
async def test_map_reduce_per_source_density_failure_surfaces_failed_paths() -> None:
    """One source returns a tiny section; retry stays tiny → failed path + inline marker."""
    tool_fn, captures = _make_tool_executor(
        stored_sources={"a.md": _SOURCE_A, "b.md": _SOURCE_B},
    )
    call_log: list[dict[str, Any]] = []

    async def _complete(**kwargs: Any) -> _FakeCompletion:
        messages = kwargs.get("messages", [])
        system_content = next((m["content"] for m in messages if m["role"] == "system"), "")
        if "Per-Source Consolidation Section" in system_content:
            # Inspect the source index in the user prompt to decide which source.
            user_content = next((m["content"] for m in messages if m["role"] == "user"), "")
            call_log.append({"phase": "per-source", "user": user_content})
            # Source 1 (a.md): always tiny (retry stays tiny too).
            if "Source 1" in user_content:
                return _FakeCompletion(
                    content=f"```markdown\n{_ABRIDGED_SECTION}\n```",
                    finish_reason="stop",
                )
            # Source 2 (b.md): full section.
            return _FakeCompletion(
                content=f"```markdown\n{_FULL_SECTION_A}\n```",
                finish_reason="stop",
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
            "source_paths": ["a.md", "b.md"],
        },
        pack_config=_map_reduce_pack_config(),
    )
    result = await execute(ctx)
    assert "Document consolidated and stored" in result.output_text
    # The abridged source must surface in density_failed_sources.
    failed = result.output_json.get("density_failed_sources") or []
    assert "a.md" in failed
    assert "b.md" not in failed
    # The persisted document must carry the inline section density marker.
    persisted = captures["writes"][0]["content"]
    assert "section density floor failed" in persisted
    # Source 1 must have been forged twice (initial + density retry); the
    # retry used the per-source density-retry instruction.
    per_source_calls = [c for c in call_log if c["phase"] == "per-source"]
    source_one_calls = [c for c in per_source_calls if "Source 1" in c["user"]]
    assert len(source_one_calls) == 2


@pytest.mark.asyncio
async def test_map_reduce_per_source_density_retry_recovers() -> None:
    """First per-source call tiny, retry full → density_failed_sources is empty."""
    tool_fn, captures = _make_tool_executor(
        stored_sources={"a.md": _SOURCE_A, "b.md": _SOURCE_B},
    )
    call_log: list[dict[str, Any]] = []
    # Track call index per source to flip the response on retry.
    source_one_index = {"n": 0}

    async def _complete(**kwargs: Any) -> _FakeCompletion:
        messages = kwargs.get("messages", [])
        system_content = next((m["content"] for m in messages if m["role"] == "system"), "")
        if "Per-Source Consolidation Section" in system_content:
            user_content = next((m["content"] for m in messages if m["role"] == "user"), "")
            call_log.append({"phase": "per-source", "user": user_content})
            if "Source 1" in user_content:
                source_one_index["n"] += 1
                # First call tiny; retry (second call) full.
                doc = _ABRIDGED_SECTION if source_one_index["n"] == 1 else _FULL_SECTION_A
                return _FakeCompletion(
                    content=f"```markdown\n{doc}\n```",
                    finish_reason="stop",
                )
            return _FakeCompletion(
                content=f"```markdown\n{_FULL_SECTION_A}\n```",
                finish_reason="stop",
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
            "source_paths": ["a.md", "b.md"],
        },
        pack_config=_map_reduce_pack_config(),
    )
    result = await execute(ctx)
    assert "Document consolidated and stored" in result.output_text
    # The retry recovered → no failed sources and no inline marker.
    failed = result.output_json.get("density_failed_sources") or []
    assert failed == []
    persisted = captures["writes"][0]["content"]
    assert "section density floor failed" not in persisted
    # Source 1 must have been forged twice (initial tiny + retry full).
    per_source_calls = [c for c in call_log if c["phase"] == "per-source"]
    source_one_calls = [c for c in per_source_calls if "Source 1" in c["user"]]
    assert len(source_one_calls) == 2


@pytest.mark.asyncio
async def test_okf_provenance_map_reduce_carries_passes_and_density_columns() -> None:
    """The map-reduce OKF concept body carries the Passes and Density columns."""
    tool_fn, captures = _make_tool_executor(
        stored_sources={"a.md": _SOURCE_A, "b.md": _SOURCE_B},
    )

    async def _complete(**kwargs: Any) -> _FakeCompletion:
        messages = kwargs.get("messages", [])
        system_content = next((m["content"] for m in messages if m["role"] == "system"), "")
        if "Per-Source Consolidation Section" in system_content:
            return _FakeCompletion(
                content=f"```markdown\n{_FULL_SECTION_A}\n```",
                finish_reason="stop",
            )
        if "Section Merge" in system_content:
            return _FakeCompletion(content=f"```markdown\n{_FULL_OUTPUT}\n```")
        return _FakeCompletion(content=f"```markdown\n{_FULL_OUTPUT}\n```")

    ctx = _make_ctx(
        complete_fn=_complete,
        resolve_model_fn=_fake_resolve_model(),
        execute_tool_fn=tool_fn,
        params={
            "document_path": "narrative.md",
            "source_paths": ["a.md", "b.md"],
            "okf_bundle_path": "okf/forge-provenance",
        },
        pack_config=_map_reduce_pack_config(),
    )
    await execute(ctx)
    assert len(captures["okf_emits"]) == 1
    body = captures["okf_emits"][0]["body"]
    # The header must carry the two new columns.
    assert "| # | Path | Priority | Bytes | Passes | Density | Treatment |" in body
    # Each readable row must carry per-source pass/density values (1 pass,
    # "pass" density — no retry needed when the section is full-size).
    assert "| 1 | `a.md` |" in body
    assert "| pass |" in body
    assert "| 2 | `b.md` |" in body


@pytest.mark.asyncio
async def test_okf_provenance_single_pass_emits_na_density_columns() -> None:
    """The single-pass OKF concept emits 0 passes and n/a density (stable table shape)."""
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
        pack_config=_single_pass_pack_config(),
    )
    await execute(ctx)
    body = captures["okf_emits"][0]["body"]
    # The header must carry the new columns even on the single-pass path.
    assert "| # | Path | Priority | Bytes | Passes | Density | Treatment |" in body
    # Single-pass path emits 0 passes and n/a density per source.
    assert "| 0 | n/a |" in body
    assert "| pass |" not in body


# ── Phase 3: rolling accumulation (gated, default off) ────────────────
# These pin the 2026-08-02 rolling-accumulation path: a single oversize
# source (>MAX_PER_SOURCE_CHARS) is chunked via read_file_range and forged
# chunk-by-chunk into a running document — but only when the
# forge_rolling_accumulation gate is opted in. Default off → map-reduce
# still handles the same source with no read_file_range calls.

from lisa_packs.markdown.operations.consolidate._source_paths import (  # noqa: E402
    MAX_PER_SOURCE_CHARS,
    read_source_chunks,
)
from lisa_packs.markdown.operations.consolidate._source_paths import (  # noqa: E402
    SourceSpec as _SourceSpec,
)


# A source deliberately larger than MAX_PER_SOURCE_CHARS so the rolling
# gate trips. Built by repeating _SOURCE_A well past the 32K cap.
_OVERSIZE_SOURCE = ("# Big Source\n\n## Section\n\n" + ("detail line.\n" * 4000))
assert len(_OVERSIZE_SOURCE) > MAX_PER_SOURCE_CHARS


def _rolling_pack_config() -> dict[str, Any]:
    """Pack config that opts into the rolling-accumulation gate + map-reduce."""
    cfg = _map_reduce_pack_config()
    cfg["forge_rolling_accumulation"] = True
    return cfg


@pytest.mark.asyncio
async def test_read_source_chunks_plans_chunks_from_stat() -> None:
    """read_source_chunks stats the file and returns ordered chunks until eof."""
    tool_fn, captures = _make_tool_executor(
        stored_sources={"big.md": _OVERSIZE_SOURCE},
        stat_sizes={"big.md": len(_OVERSIZE_SOURCE.encode("utf-8"))},
    )
    ctx = _make_ctx(
        execute_tool_fn=tool_fn,
        params={"document_path": "n.md"},
    )
    spec = _SourceSpec(path="big.md", priority="primary")
    chunks = await read_source_chunks(spec, ctx=ctx, chunk_chars=8000)
    # stat_file must have been called once.
    assert captures["stat_paths"] == ["big.md"]
    # Multiple read_file_range calls, advancing start by chunk_chars.
    ranges = captures["read_file_ranges"]
    assert len(ranges) >= 2
    # Chunks are ordered, contiguous, and the last carries chunk_eof.
    assert chunks[0].extra["chunk_index"] == 0
    assert chunks[-1].extra.get("chunk_eof") is True
    assert all(c.path == "big.md" for c in chunks)
    # chunk_total must be set on every chunk and match the count.
    assert all(c.extra["chunk_total"] == len(chunks) for c in chunks)
    # The concatenated chunk content covers the planned read range.
    joined = "".join(c.text for c in chunks)
    assert len(joined) > 0


@pytest.mark.asyncio
async def test_rolling_accumulation_extends_running_doc() -> None:
    """A single oversize source with the gate on takes the rolling path."""
    tool_fn, captures = _make_tool_executor(
        stored_sources={"big.md": _OVERSIZE_SOURCE},
        stat_sizes={"big.md": len(_OVERSIZE_SOURCE.encode("utf-8"))},
    )
    call_log: list[dict[str, Any]] = []
    # Each rolling chunk call returns a short extension referencing its index.
    async def _complete(**kwargs: Any) -> _FakeCompletion:
        messages = kwargs.get("messages", [])
        system_content = next((m["content"] for m in messages if m["role"] == "system"), "")
        if "Rolling Accumulation" in system_content:
            user_content = next((m["content"] for m in messages if m["role"] == "user"), "")
            # Extract chunk index from the template marker.
            import re as _re
            m = _re.search(r"chunk (\d+) of", user_content)
            idx = m.group(1) if m else "?"
            call_log.append({"phase": "rolling", "chunk": idx})
            return _FakeCompletion(
                content=f"```markdown\n## Chunk {idx} section\n\ndetail from chunk {idx}.\n```",
                finish_reason="stop",
            )
        if "Section Merge" in system_content:
            call_log.append({"phase": "merge"})
            return _FakeCompletion(content=f"```markdown\n{_FULL_OUTPUT}\n```")
        if "Per-Source Consolidation Section" in system_content:
            call_log.append({"phase": "per-source"})
            return _FakeCompletion(content="```markdown\n# Section\n\nforged.\n```")
        # Phase A: lint repair pass — the user prompt contains the lint
        # issues block. Return the document with an H1 prepended so the
        # MD041 violation is resolved and the chunk content survives.
        user_content = next((m["content"] for m in messages if m["role"] == "user"), "")
        if "Structural lint issues to fix" in user_content:
            call_log.append({"phase": "lint-repair"})
            return _FakeCompletion(
                content="```markdown\n# Consolidated Document\n\n"
                + "## Chunk 1 section\n\ndetail from chunk 1.\n"
                + "## Chunk 2 section\n\ndetail from chunk 2.\n```",
                finish_reason="stop",
            )
        call_log.append({"phase": "unknown"})
        return _FakeCompletion(content=f"```markdown\n{_FULL_OUTPUT}\n```")

    ctx = _make_ctx(
        complete_fn=_complete,
        resolve_model_fn=_fake_resolve_model(),
        execute_tool_fn=tool_fn,
        params={
            "document_path": "n.md",
            "source_paths": ["big.md"],
        },
        pack_config=_rolling_pack_config(),
    )
    result = await execute(ctx)
    assert "Document consolidated and stored" in result.output_text
    # The rolling path must have been taken (multiple rolling completion calls).
    rolling_calls = [c for c in call_log if c["phase"] == "rolling"]
    assert len(rolling_calls) >= 2
    # read_file_range must have been called to chunk the source.
    assert len(captures["read_file_ranges"]) >= 2
    # The op result must report rolling_accumulation_used=True.
    assert result.output_json.get("rolling_accumulation_used") is True
    # The persisted document must contain content from multiple chunks.
    persisted = captures["writes"][0]["content"]
    assert "Chunk" in persisted


@pytest.mark.asyncio
async def test_rolling_accumulation_default_off_uses_map_reduce() -> None:
    """Same oversize source, default config → map-reduce path (no read_file_range)."""
    tool_fn, captures = _make_tool_executor(
        stored_sources={"big.md": _OVERSIZE_SOURCE},
        stat_sizes={"big.md": len(_OVERSIZE_SOURCE.encode("utf-8"))},
    )
    call_log: list[dict[str, Any]] = []

    async def _complete(**kwargs: Any) -> _FakeCompletion:
        messages = kwargs.get("messages", [])
        system_content = next((m["content"] for m in messages if m["role"] == "system"), "")
        if "Rolling Accumulation" in system_content:
            call_log.append({"phase": "rolling"})
            return _FakeCompletion(content="```markdown\n# rolling\n```")
        if "Per-Source Consolidation Section" in system_content:
            call_log.append({"phase": "per-source"})
            return _FakeCompletion(content=f"```markdown\n{_FULL_OUTPUT}\n```")
        if "Section Merge" in system_content:
            call_log.append({"phase": "merge"})
            return _FakeCompletion(content=f"```markdown\n{_FULL_OUTPUT}\n```")
        call_log.append({"phase": "unknown"})
        return _FakeCompletion(content=f"```markdown\n{_FULL_OUTPUT}\n```")

    ctx = _make_ctx(
        complete_fn=_complete,
        resolve_model_fn=_fake_resolve_model(),
        execute_tool_fn=tool_fn,
        params={
            "document_path": "n.md",
            "source_paths": ["big.md"],
        },
        # Default map-reduce config — rolling gate NOT opted in.
        pack_config=_map_reduce_pack_config(),
    )
    result = await execute(ctx)
    assert "Document consolidated and stored" in result.output_text
    # No rolling calls and no read_file_range calls (default off → map-reduce).
    assert all(c["phase"] != "rolling" for c in call_log)
    assert captures["read_file_ranges"] == []
    # The op result must report rolling_accumulation_used=False.
    assert result.output_json.get("rolling_accumulation_used") is False


# ── Phase A: advisory lint gate (consolidate) ──────────────────────────


@pytest.mark.asyncio
async def test_consolidate_lint_advisory_annotation() -> None:
    """Consolidate persists a lint advisory comment when issues remain after repair is disabled."""
    # A merged document that starts with an H2 (no H1) trips MD041.
    no_h1_doc = (
        "## Strengths\n"
        "- MCP-native architecture\n"
        "\n"
        "## Weaknesses\n"
        "- Steeper learning curve\n"
    )
    tool_fn, captures = _make_tool_executor(
        stored_sources={"a.md": _SOURCE_A},
    )

    async def _complete(**kwargs: Any) -> _FakeCompletion:
        return _FakeCompletion(content=f"```markdown\n{no_h1_doc}\n```", finish_reason="stop")

    ctx = _make_ctx(
        complete_fn=_complete,
        resolve_model_fn=_fake_resolve_model(),
        execute_tool_fn=tool_fn,
        params={
            "document_path": "narrative.md",
            "source_paths": ["a.md"],
        },
        # Force single-pass and disable the lint repair pass so the advisory
        # comment is the expected outcome.
        pack_config={
            **_single_pass_pack_config(),
            "forge_lint_repair": False,
        },
    )
    result = await execute(ctx)
    assert "Document consolidated and stored" in result.output_text
    assert result.output_json["lint_warnings"] >= 1
    assert result.output_json["lint_repair_used"] is False
    persisted = captures["writes"][0]["content"]
    assert "lint warnings" in persisted
