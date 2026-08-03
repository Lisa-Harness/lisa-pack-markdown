"""Consolidate-sources operation — multi-source preserve-detail markdown merging.

Owns the source-aware consolidation path that used to live inside
``markdown_forge_document``. Splitting it out lets the forge operation stay a
narrow single-document writer while this operation owns the distinct concerns
of multi-source ingestion, preserve-detail consolidation, anti-abridgment
density floors, map-reduce sectioning, and OKF provenance.

Pipeline:

1. Parse + expand + read every source via ``lisa-storage`` tools
   (``list_files`` for globs, ``read_file`` for content) through
   ``ctx.execute_tool_fn``.
2. Recall memory context via the ``memory.search`` MCP alias.
3. Forge the consolidated document. For small aggregates a single
   preserve-detail pass suffices; for aggregates at/above the map-reduce
   thresholds the operation forges one bounded-context section per source
   then merges. A single oversize source (with ``forge_rolling_accumulation``
   opted in) is chunked and extended incrementally.
4. Enforce a density floor against lossy summarization (global + per-source)
   with a sharpened re-forge retry.
5. Persist via ``write_file`` and capture full OKF provenance.

The operation never touches the filesystem directly — all reads (sources,
memory) and writes (document, provenance) go through ``ctx.execute_tool_fn``.
Authored prompts are loaded via ``ctx.read_authored_text_fn``; per the
pack-content rule, a missing/empty loader raises rather than silently
falling back.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, Field

from kernel.models import OperationExecutionContext, OperationResult, ToolExecutionContext

from lisa_packs.markdown.operations._config import cfg_bool, cfg_float, cfg_int, cfg_str
from lisa_packs.markdown.operations._lint import (
    LintIssue,
    format_lint_issues,
    lint_markdown,
    lint_summary_comment,
)
from lisa_packs.markdown.operations._continuation import (
    coerce_finish_reason,
    continue_until_complete,
    extract_markdown_block,
)
from lisa_packs.markdown.operations.consolidate._source_paths import (
    SourceSpec,
    expand_globs,
    failed_specs,
    has_sources,
    parse_source_paths,
    read_source_chunks,
    read_sources,
    readable_count,
    render_one_source_block,
    render_priorities_block,
    render_source_block,
    stat_source_chars,
    total_source_bytes,
)

logger = logging.getLogger(__name__)

__all__ = ["Parameters", "execute"]

# Tool ids — declared in the operation YAML ``tool_ids`` and routed through
# ``ctx.execute_tool_fn``.
_OUTLINE_TOOL_ID = "markdown_generate"
_WRITE_TOOL_ID = "write_file"
_READ_TOOL_ID = "read_file"
_APPEND_TOOL_ID = "append_file"
_MEMORY_SEARCH_TOOL_ID = "memory.search"
_OKF_EMIT_TOOL_ID = "okf_emit_concept"
_OKF_LOG_TOOL_ID = "okf_log"

# Authored-text paths — loaded via ctx.read_authored_text_fn (SDK loader).
_PROMPT_DIR = "operations/consolidate/prompts"
_SYSTEM_PROMPT_SOURCES_PATH = f"{_PROMPT_DIR}/consolidate_system_sources.md"
_USER_PROMPT_SOURCES_PATH = f"{_PROMPT_DIR}/consolidate_user_sources.md"
_DENSITY_RETRY_PATH = f"{_PROMPT_DIR}/consolidate_density_retry.md"
_SYSTEM_PROMPT_PER_SOURCE_PATH = f"{_PROMPT_DIR}/consolidate_system_per_source.md"
_USER_PROMPT_PER_SOURCE_PATH = f"{_PROMPT_DIR}/consolidate_user_per_source.md"
_PER_SOURCE_DENSITY_RETRY_PATH = f"{_PROMPT_DIR}/consolidate_per_source_density_retry.md"
_SYSTEM_PROMPT_MERGE_PATH = f"{_PROMPT_DIR}/consolidate_system_merge.md"
_USER_PROMPT_MERGE_PATH = f"{_PROMPT_DIR}/consolidate_user_merge.md"
_SYSTEM_PROMPT_ROLLING_PATH = f"{_PROMPT_DIR}/consolidate_system_rolling.md"
_USER_PROMPT_ROLLING_PATH = f"{_PROMPT_DIR}/consolidate_user_rolling.md"
_CONTINUE_PROMPT_PATH = f"{_PROMPT_DIR}/consolidate_continue.md"
_EMPTY_RESPONSE_PATH = f"{_PROMPT_DIR}/consolidate_empty_response.md"
_EMPTY_DOCUMENT_PATH = f"{_PROMPT_DIR}/consolidate_empty_document.md"
_NO_OUTLINE_PATH = f"{_PROMPT_DIR}/consolidate_no_outline.md"
_STORAGE_FAILED_PATH = f"{_PROMPT_DIR}/consolidate_storage_failed.md"
_LINT_REPAIR_PROMPT_PATH = f"{_PROMPT_DIR}/consolidate_lint_repair.md"

# User-instruction phrases that opt out of model attribution.
_NO_ANNOTATE_PHRASES = (
    "don't annotate the model",
    "do not annotate the model",
    "no model annotation",
    "no model attribution",
    "without model annotation",
)

# Fallback defaults for the operation's configurable budgets. The
# authoritative source is the session manifest's
# ``resources.packs.configuration.markdown.<key>`` block, surfaced to the
# op at runtime as ``ctx.pack_config``. These constants are the last-resort
# fallback when no manifest config is supplied so the op still works in a
# bare host.
_MAX_COMPLETION_TOKENS = 50_000
_MAX_CONTINUE_TOKENS = _MAX_COMPLETION_TOKENS
_MAX_CONTINUE_ATTEMPTS = 4
_MAX_RECALL_RECORDS = 25
# Per-source token cap (map-reduce). A single preserve-detail section from
# one bounded source rarely needs the full 50K; capping per-source passes
# keeps every model call's context small enough for a local backend to
# satisfy the preserve-detail directive without truncating. This is the
# budget reconciliation that removes the whole-document truncation driver.
_PER_SOURCE_MAX_TOKENS = 16_000
_MAX_DOCUMENT_CHARS = 1_500_000
_MAX_QUERY_CHARS = 20_000
# Map-reduce consolidation thresholds.
_MAP_REDUCE_BYTE_THRESHOLD = 8_000
_MAP_REDUCE_SOURCE_THRESHOLD = 2
# Density floor (consolidation anti-abridgment gate).
_DENSITY_FLOOR_MANY_SOURCES_RATIO = 0.15   # 5+ readable sources
_DENSITY_FLOOR_FEW_SOURCES_RATIO = 0.30    # 2-4 readable sources
_DENSITY_FLOOR_MANY_SOURCES_THRESHOLD = 5
_DENSITY_RETRY_ATTEMPTS = 2
# Per-source density floor (map-reduce path).
_PER_SOURCE_MIN_RATIO = 0.10
# Phase A: advisory lint gate defaults (mirrors forge_document).
_DEFAULT_LINT_RULES = ""
_DEFAULT_LINT_REPAIR = True

# Minimum structural bar for a forged document: at least one heading.
_MIN_HEADINGS = 1


def _load_text(ctx: OperationExecutionContext, path: str) -> str:
    """Load authored markdown via the SDK content-registry loader."""
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


def _format_template(template: str, **kwargs: Any) -> str:
    """Substitute ``{placeholder}`` tokens in a loaded template."""
    try:
        return template.format_map(_SafeDict(kwargs))
    except Exception:  # pylint: disable=broad-except
        return template


class _SafeDict(dict[str, str]):
    """dict subclass that returns ``{key}`` for missing keys in str.format_map."""

    def __missing__(self, key: str) -> str:
        return "{" + key + "}"


class Parameters(BaseModel):
    """Parameter surface for the consolidate operation."""

    document_path: str = Field(
        default="docs/consolidated.md",
        description="Relative path within the lisa-storage files sandbox where the consolidated document is persisted.",
    )
    source_paths: list[str | dict[str, str]] = Field(
        default_factory=list,
        description="Ordered list of source documents to consolidate. Each entry is either a path/glob string "
        "(default priority 'normal') or a dict with 'path' (required) and 'priority' (optional NL prose). When "
        "supplied, the operation consolidates the sources preserving detail rather than summarizing.",
    )
    query: str = Field(
        default="",
        description="Natural-language instruction for the document. Falls back to ctx.query when empty.",
    )
    okf_bundle_path: str = Field(
        default="okf/forge-provenance",
        description="Relative path of the OKF bundle directory where the provenance concept is written. "
        "Set to empty to skip OKF provenance capture.",
    )
    annotate_model: bool = Field(
        default=True,
        description="When True, append a non-rendering HTML comment to the forged document noting the "
        "provider/model that wrote it. Set False to suppress unconditionally.",
    )


def _extract_document_path(params: dict[str, Any]) -> str:
    """Return the target storage path, sanitized to a safe relative form."""
    raw = str(params.get("document_path") or "").strip()
    if not raw:
        return "docs/consolidated.md"
    normalized = raw.replace("\\", "/").lstrip("/")
    if ".." in normalized.split("/"):
        raise RuntimeError("document_path may not contain '..' segments")
    return normalized


@dataclass(frozen=True)
class _ResolvedSettings:
    """Resolved operation budgets: ``ctx.pack_config`` overrides module fallbacks."""

    max_completion_tokens: int
    max_continue_tokens: int
    max_continue_attempts: int
    max_recall_records: int
    max_document_chars: int
    max_query_chars: int
    map_reduce_byte_threshold: int
    map_reduce_source_threshold: int
    per_source_min_ratio: float
    per_source_max_tokens: int
    forge_rolling_accumulation: bool
    forge_rolling_incremental_persist: bool
    density_retry_attempts: int
    # Phase A: advisory lint gate
    lint_rules: str
    lint_repair: bool


def _resolve_settings(ctx: OperationExecutionContext) -> _ResolvedSettings:
    """Resolve all operation budgets from ``ctx.pack_config`` with fallbacks."""
    pc = ctx.pack_config
    return _ResolvedSettings(
        max_completion_tokens=cfg_int(pc, "forge_max_completion_tokens", _MAX_COMPLETION_TOKENS),
        max_continue_tokens=cfg_int(pc, "forge_max_continue_tokens", _MAX_CONTINUE_TOKENS),
        max_continue_attempts=cfg_int(pc, "forge_max_continue_attempts", _MAX_CONTINUE_ATTEMPTS),
        max_recall_records=cfg_int(pc, "forge_max_recall_records", _MAX_RECALL_RECORDS),
        max_document_chars=cfg_int(pc, "forge_max_document_chars_with_sources", _MAX_DOCUMENT_CHARS),
        max_query_chars=cfg_int(pc, "forge_max_query_chars", _MAX_QUERY_CHARS),
        map_reduce_byte_threshold=cfg_int(pc, "forge_map_reduce_byte_threshold", _MAP_REDUCE_BYTE_THRESHOLD),
        map_reduce_source_threshold=cfg_int(pc, "forge_map_reduce_source_threshold", _MAP_REDUCE_SOURCE_THRESHOLD),
        per_source_min_ratio=cfg_float(pc, "forge_per_source_min_ratio", _PER_SOURCE_MIN_RATIO),
        per_source_max_tokens=cfg_int(pc, "forge_per_source_max_tokens", _PER_SOURCE_MAX_TOKENS),
        forge_rolling_accumulation=cfg_bool(pc, "forge_rolling_accumulation", False),
        forge_rolling_incremental_persist=cfg_bool(pc, "forge_rolling_incremental_persist", False),
        density_retry_attempts=cfg_int(pc, "forge_density_retry_attempts", _DENSITY_RETRY_ATTEMPTS),
        lint_rules=cfg_str(pc, "forge_lint_rules", _DEFAULT_LINT_RULES),
        lint_repair=cfg_bool(pc, "forge_lint_repair", _DEFAULT_LINT_REPAIR),
    )


def _document_stem(document_path: str) -> str:
    """Return the filename stem (no extension) of a storage path."""
    normalized = document_path.replace("\\", "/").rstrip("/")
    base = normalized.rsplit("/", 1)[-1]
    if "." in base:
        base = base.rsplit(".", 1)[0]
    return base or "consolidated"


def _extract_query(ctx: OperationExecutionContext, params: dict[str, Any], max_query_chars: int) -> str:
    """Return the user-supplied instruction, bounded in length."""
    raw = str(params.get("query") or ctx.query or "").strip()
    if len(raw) > max_query_chars:
        raw = raw[:max_query_chars] + "…"
    return raw


async def _recall_memory(
    ctx: OperationExecutionContext,
    query: str,
    *,
    limit: int,
) -> str:
    """Recall relevant memories via the ``memory.search`` MCP alias."""
    if not callable(ctx.execute_tool_fn):
        raise RuntimeError("markdown requires execute_tool_fn to call memory/storage tools")

    recall_limit = max(1, min(int(limit), 20))
    tool_result = await ctx.execute_tool_fn(
        _MEMORY_SEARCH_TOOL_ID,
        ToolExecutionContext(
            tool_id=_MEMORY_SEARCH_TOOL_ID,
            workspace_id=ctx.workspace_id,
            user_id=ctx.user_id,
            user_slug=ctx.user_slug,
            surface_type=ctx.surface_type,
            surface_id=ctx.surface_id,
            memory_scope=ctx.memory_scope,
            mode_id=ctx.mode_id,
            query=query,
            parameters={
                "query": query,
                "user_slug": ctx.user_slug,
                "workspace_id": ctx.workspace_id,
                "memory_scope": ctx.memory_scope,
                "surface_id": ctx.surface_id,
                "limit": recall_limit,
            },
        ),
    )
    error_text = getattr(tool_result, "error", "") or ""
    if error_text:
        logger.warning("markdown: memory.search returned error: %s", error_text)
        return ""
    output_text = getattr(tool_result, "output_text", "") or ""
    try:
        data = json.loads(output_text)
    except (json.JSONDecodeError, TypeError):
        return output_text.strip()
    if isinstance(data, list):
        lines: list[str] = []
        for item in data:
            if isinstance(item, dict):
                content = str(item.get("content") or "").strip()
                if content:
                    lines.append(f"- {content}")
            elif isinstance(item, str) and item.strip():
                lines.append(f"- {item.strip()}")
        return "\n".join(lines)
    if isinstance(data, dict) and "results" in data and isinstance(data["results"], list):
        lines = []
        for item in data["results"]:
            if isinstance(item, dict):
                content = str(item.get("content") or "").strip()
                if content:
                    lines.append(f"- {content}")
        return "\n".join(lines)
    return output_text.strip()


async def _call_llm(
    ctx: OperationExecutionContext,
    user_input: str,
    memory_context: str,
    source_block: str,
    *,
    max_tokens: int,
) -> tuple[str, Any, str | None]:
    """Call the host's completion function with the source-aware consolidate prompt."""
    if ctx.complete_fn is None or ctx.resolve_model_fn is None:
        raise RuntimeError("markdown requires complete_fn and resolve_model_fn")

    preferred_model_id = ctx.preferred_model_id or ctx.model_id or "default"
    model_def = ctx.resolve_model_fn(preferred_model_id)
    if model_def is None:
        raise RuntimeError(f"markdown could not resolve model '{preferred_model_id}'")

    system_prompt = _load_text(ctx, _SYSTEM_PROMPT_SOURCES_PATH)
    user_template = _load_text(ctx, _USER_PROMPT_SOURCES_PATH)
    user_prompt = _format_template(
        user_template,
        user_input=user_input,
        memory_context=memory_context or "(no recalled context)",
        source_block=source_block,
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
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
    finish_reason = getattr(completion, "finish_reason", None)
    return content.strip(), model_def, coerce_finish_reason(finish_reason)


def _user_asked_not_to_annotate(user_input: str) -> bool:
    """Return True when the user instruction explicitly opts out of attribution."""
    lowered = user_input.lower()
    return any(phrase in lowered for phrase in _NO_ANNOTATE_PHRASES)


async def _lint_repair_pass(
    ctx: OperationExecutionContext,
    *,
    user_input: str,
    document: str,
    model_def: Any,
    issues: list[LintIssue],
    max_tokens: int,
) -> str:
    """Issue one advisory lint repair pass feeding issues back to the model.

    Uses the consolidate source-aware system prompt so the repair respects
    the preserve-detail directive. Returns the repaired document text or
    the original ``document`` on any failure. Never raises; lint is advisory.
    """
    if not issues:
        return document
    if ctx.complete_fn is None:
        return document
    try:
        repair_prompt = _load_text(ctx, _LINT_REPAIR_PROMPT_PATH)
        system_prompt = _load_text(ctx, _SYSTEM_PROMPT_SOURCES_PATH)
    except RuntimeError as exc:
        logger.warning("consolidate: lint repair prompt unavailable: %s", exc)
        return document
    user_prompt = _format_template(
        repair_prompt,
        user_input=user_input,
        lint_issues=format_lint_issues(issues),
        document_so_far=document,
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    try:
        completion = await ctx.complete_fn(
            messages=messages,
            model_def=model_def,
            max_tokens=max_tokens,
            temperature=0.2,
            tools=None,
            response_format=None,
        )
    except Exception as exc:  # pylint: disable=broad-except
        logger.warning("consolidate: lint repair pass raised: %s", exc)
        return document
    content = (getattr(completion, "content", "") or "").strip()
    if not content:
        return document
    repaired = extract_markdown_block(content)
    if not repaired.strip():
        return document
    return repaired


async def _validate_outline(ctx: OperationExecutionContext, document: str) -> dict[str, Any]:
    """Run the ``markdown_generate`` tool and return its outline dict."""
    tool_result = await ctx.execute_tool_fn(
        _OUTLINE_TOOL_ID,
        ToolExecutionContext(
            tool_id=_OUTLINE_TOOL_ID,
            workspace_id=ctx.workspace_id,
            user_id=ctx.user_id,
            user_slug=ctx.user_slug,
            surface_type=ctx.surface_type,
            surface_id=ctx.surface_id,
            memory_scope=ctx.memory_scope,
            mode_id=ctx.mode_id,
            parameters={"markdown": document},
        ),
    )
    error_text = getattr(tool_result, "error", "") or ""
    if error_text:
        raise RuntimeError(f"{_OUTLINE_TOOL_ID} failed: {error_text}")
    output_json = getattr(tool_result, "output_json", {}) or {}
    if not output_json:
        output_text = getattr(tool_result, "output_text", "") or ""
        try:
            output_json = json.loads(output_text)
        except (json.JSONDecodeError, TypeError):
            output_json = {}
    return output_json if isinstance(output_json, dict) else {}


async def _persist_document(
    ctx: OperationExecutionContext,
    document: str,
    path: str,
) -> dict[str, Any]:
    """Write the document to lisa-storage via the ``write_file`` tool."""
    tool_result = await ctx.execute_tool_fn(
        _WRITE_TOOL_ID,
        ToolExecutionContext(
            tool_id=_WRITE_TOOL_ID,
            workspace_id=ctx.workspace_id,
            user_id=ctx.user_id,
            user_slug=ctx.user_slug,
            surface_type=ctx.surface_type,
            surface_id=ctx.surface_id,
            memory_scope=ctx.memory_scope,
            mode_id=ctx.mode_id,
            parameters={"path": path, "content": document},
        ),
    )
    error_text = getattr(tool_result, "error", "") or ""
    if error_text:
        raise RuntimeError(f"{_WRITE_TOOL_ID} failed: {error_text}")
    output_text = getattr(tool_result, "output_text", "") or str(tool_result)
    try:
        data = json.loads(output_text)
    except (json.JSONDecodeError, TypeError):
        data = {"ok": False, "raw": output_text}
    if isinstance(data, dict) and data.get("error"):
        raise RuntimeError(f"{_WRITE_TOOL_ID} failed: {data['error']}")
    return data if isinstance(data, dict) else {}


def _density_floor_ratio(source_count: int) -> float:
    """Return the minimum output/source byte ratio that is NOT a lossy abridgment."""
    if source_count >= _DENSITY_FLOOR_MANY_SOURCES_THRESHOLD:
        return _DENSITY_FLOOR_MANY_SOURCES_RATIO
    if source_count >= 2:
        return _DENSITY_FLOOR_FEW_SOURCES_RATIO
    return 0.0


def _density_floor_check(
    *,
    source_total_bytes: int,
    output_bytes: int,
    source_count: int,
) -> str | None:
    """Return a violation message if the output looks like a lossy abridgment."""
    if source_total_bytes <= 0 or source_count < 2:
        return None
    min_ratio = _density_floor_ratio(source_count)
    if min_ratio <= 0.0:
        return None
    ratio = output_bytes / source_total_bytes if source_total_bytes else 1.0
    if ratio >= min_ratio:
        return None
    return (
        f"Output is {output_bytes}B vs {source_total_bytes}B of source "
        f"({ratio:.0%}, {source_count} sources) — below the {min_ratio:.0%} "
        f"preservation floor. Re-forge preserving source detail."
    )


def _per_source_density_check(
    *,
    section: str,
    source_bytes: int,
    min_ratio: float,
    priority: str,
) -> str | None:
    """Return a violation message if a per-source section is abridged."""
    if source_bytes <= 0 or min_ratio <= 0.0:
        return None
    lowered = (priority or "").lower()
    latitude_tokens = ("background", "one paragraph", "brief", "summary", "condense", "verbatim")
    if any(tok in lowered for tok in latitude_tokens):
        return None
    section_bytes = len(section.encode("utf-8"))
    ratio = section_bytes / source_bytes if source_bytes else 1.0
    if ratio >= min_ratio:
        return None
    return (
        f"Per-source section for is {section_bytes}B vs {source_bytes}B source "
        f"({ratio:.0%}, floor {min_ratio:.0%}) — below the per-source "
        f"preservation floor. The section summarized instead of preserving."
    )


async def _reforge_for_density(
    ctx: OperationExecutionContext,
    *,
    user_input: str,
    memory_context: str,
    source_block: str,
    model_def: Any,
    violation: str,
    source_total_bytes: int,
    output_bytes: int,
    source_count: int,
    max_tokens: int,
) -> tuple[str, str | None]:
    """Issue one density-retry forge pass with a sharpened instruction."""
    if ctx.complete_fn is None:
        return "", None
    system_prompt = _load_text(ctx, _SYSTEM_PROMPT_SOURCES_PATH)
    user_template = _load_text(ctx, _USER_PROMPT_SOURCES_PATH)
    base_user_prompt = _format_template(
        user_template,
        user_input=user_input,
        memory_context=memory_context or "(no recalled context)",
        source_block=source_block,
    )
    retry_template = _load_text(ctx, _DENSITY_RETRY_PATH)
    min_ratio = _density_floor_ratio(source_count)
    min_output_bytes = max(1, int(min_ratio * source_total_bytes)) if source_total_bytes else 0
    min_output_words = max(1, min_output_bytes // 6) if min_output_bytes else 0
    retry_instruction = _format_template(
        retry_template,
        output_bytes=output_bytes,
        source_total_bytes=source_total_bytes,
        ratio=(output_bytes / source_total_bytes) if source_total_bytes else 0.0,
        source_count=source_count,
        min_ratio=min_ratio,
        min_output_bytes=min_output_bytes,
        min_output_words=min_output_words,
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": base_user_prompt},
        {"role": "user", "content": retry_instruction},
    ]
    logger.warning("consolidate: density floor violated — %s", violation)
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


async def _reforge_one_source_for_density(
    ctx: OperationExecutionContext,
    *,
    spec: SourceSpec,
    idx: int,
    user_input: str,
    memory_context: str,
    model_def: Any,
    violation: str,
    section_bytes: int,
    source_bytes: int,
    min_ratio: float,
    max_tokens: int,
    max_continue_attempts: int,
    max_continue_tokens: int,
) -> tuple[str, str | None]:
    """Issue one per-source density-retry forge pass with a sharpened instruction.

    ``max_continue_attempts``/``max_continue_tokens`` are accepted for
    signature symmetry with the other forge helpers but are not used here —
    continuation handling for the retry section is the caller's job.
    """
    if ctx.complete_fn is None:
        return "", None
    system_prompt = _load_text(ctx, _SYSTEM_PROMPT_PER_SOURCE_PATH)
    user_template = _load_text(ctx, _USER_PROMPT_PER_SOURCE_PATH)
    source_text = render_one_source_block(spec)
    base_user_prompt = _format_template(
        user_template,
        user_input=user_input,
        source_index=idx,
        source_path=spec.path,
        priority=spec.priority,
        source_text=source_text,
        memory_context=memory_context or "(no recalled context)",
    )
    retry_template = _load_text(ctx, _PER_SOURCE_DENSITY_RETRY_PATH)
    min_section_bytes = max(1, int(min_ratio * source_bytes)) if source_bytes else 0
    min_section_words = max(1, min_section_bytes // 6) if min_section_bytes else 0
    retry_instruction = _format_template(
        retry_template,
        source_path=spec.path,
        priority=spec.priority,
        section_bytes=section_bytes,
        source_bytes=source_bytes,
        ratio=(section_bytes / source_bytes) if source_bytes else 0.0,
        min_ratio=min_ratio,
        min_section_bytes=min_section_bytes,
        min_section_words=min_section_words,
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": base_user_prompt},
        {"role": "user", "content": retry_instruction},
    ]
    logger.warning("consolidate: per-source density floor violated — %s", violation)
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
    section = extract_markdown_block(content)
    return section.strip(), finish_reason


async def _forge_one_source(
    ctx: OperationExecutionContext,
    *,
    spec: SourceSpec,
    idx: int,
    user_input: str,
    memory_context: str,
    model_def: Any,
    max_tokens: int,
    max_continue_attempts: int,
    max_continue_tokens: int,
    continue_prompt_path: str,
    per_source_min_ratio: float,
) -> tuple[str, str | None, bool, str | None, list[str], int, str]:
    """Forge one preserve-detail section from a single source.

    Returns ``(section, finish_reason, density_passed, density_reason,
    failed_paths, passes, density_marker)``.
    """
    if ctx.complete_fn is None:
        return "", None, True, None, [], 1, ""
    system_prompt = _load_text(ctx, _SYSTEM_PROMPT_PER_SOURCE_PATH)
    user_template = _load_text(ctx, _USER_PROMPT_PER_SOURCE_PATH)
    source_text = render_one_source_block(spec)
    user_prompt = _format_template(
        user_template,
        user_input=user_input,
        source_index=idx,
        source_path=spec.path,
        priority=spec.priority,
        source_text=source_text,
        memory_context=memory_context or "(no recalled context)",
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
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
    section = extract_markdown_block(content)
    passes = 1
    failed_paths: list[str] = []
    if section.strip():
        section, _truncated, _attempts, finish_reason = await continue_until_complete(
            ctx,
            user_input=user_input,
            document=section,
            model_def=model_def,
            first_finish_reason=finish_reason,
            max_continue_attempts=max_continue_attempts,
            max_continue_tokens=max_continue_tokens,
            continue_prompt_path=continue_prompt_path,
            structural_override=False,
        )

    density_reason: str | None = None
    density_passed = True
    density_marker = ""
    if section.strip() and per_source_min_ratio > 0.0 and spec.bytes > 0:
        density_reason = _per_source_density_check(
            section=section,
            source_bytes=spec.bytes,
            min_ratio=per_source_min_ratio,
            priority=spec.priority,
        )
        if density_reason is not None:
            retry_section, retry_finish = await _reforge_one_source_for_density(
                ctx,
                spec=spec,
                idx=idx,
                user_input=user_input,
                memory_context=memory_context,
                model_def=model_def,
                violation=density_reason,
                section_bytes=len(section.encode("utf-8")),
                source_bytes=spec.bytes,
                min_ratio=per_source_min_ratio,
                max_tokens=max_tokens,
                max_continue_attempts=max_continue_attempts,
                max_continue_tokens=max_continue_tokens,
            )
            passes += 1
            if retry_section.strip():
                retry_section, _t, _a, retry_finish = await continue_until_complete(
                    ctx,
                    user_input=user_input,
                    document=retry_section,
                    model_def=model_def,
                    first_finish_reason=retry_finish,
                    max_continue_attempts=max_continue_attempts,
                    max_continue_tokens=max_continue_tokens,
                    continue_prompt_path=continue_prompt_path,
                    structural_override=False,
                )
                retry_reason = _per_source_density_check(
                    section=retry_section,
                    source_bytes=spec.bytes,
                    min_ratio=per_source_min_ratio,
                    priority=spec.priority,
                )
                if retry_reason is None:
                    section = retry_section
                    finish_reason = retry_finish
                    density_passed = True
                    density_reason = None
                else:
                    if len(retry_section.encode("utf-8")) > len(section.encode("utf-8")):
                        section = retry_section
                        finish_reason = retry_finish
                    density_passed = False
                    density_reason = retry_reason
                    failed_paths.append(spec.path)
                    density_marker = (
                        f"<!-- section density floor failed: "
                        f"`{spec.path}` — {len(section.encode('utf-8'))}B "
                        f"vs {spec.bytes}B source -->"
                    )
            else:
                density_passed = False
                failed_paths.append(spec.path)
                density_marker = (
                    f"<!-- section density floor failed: "
                    f"`{spec.path}` — {len(section.encode('utf-8'))}B "
                    f"vs {spec.bytes}B source -->"
                )
    return section, finish_reason, density_passed, density_reason, failed_paths, passes, density_marker


def _tail_headings(document: str, *, max_headings: int = 3) -> str:
    """Return the running tail of a document for a rolling-accumulation pass."""
    if not document or not document.strip():
        return "(running document is empty — start fresh)"
    lines = document.splitlines()
    heading_indices: list[int] = []
    in_fence = False
    for i, line in enumerate(lines):
        stripped = line.lstrip()
        if stripped.startswith("```"):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        if re.match(r"^#{1,6}\s+\S", stripped):
            heading_indices.append(i)
    if not heading_indices:
        tail = document.rstrip()[-2000:]
        return tail if tail.strip() else "(running document has no headings yet)"
    start_idx = heading_indices[-min(max_headings, len(heading_indices))]
    tail = "\n".join(lines[start_idx:]).rstrip()
    return tail if tail.strip() else "(running tail is empty)"


async def _forge_rolling(
    ctx: OperationExecutionContext,
    chunks: list[SourceSpec],
    *,
    user_input: str,
    memory_context: str,
    model_def: Any,
    max_tokens: int,
    max_continue_attempts: int,
    max_continue_tokens: int,
    continue_prompt_path: str,
    document_path: str,
    incremental_persist: bool,
) -> tuple[str, list[str | None]]:
    """Forge a running document chunk-by-chunk (rolling accumulation)."""
    if ctx.complete_fn is None:
        return "", []
    system_prompt = _load_text(ctx, _SYSTEM_PROMPT_ROLLING_PATH)
    user_template = _load_text(ctx, _USER_PROMPT_ROLLING_PATH)
    chunk_total = len(chunks)
    document = ""
    finish_reasons: list[str | None] = []
    for chunk in chunks:
        chunk_index = int(chunk.extra.get("chunk_index", 0))
        running_tail = _tail_headings(document, max_headings=3)
        chunk_text = chunk.text
        user_prompt = _format_template(
            user_template,
            user_input=user_input,
            running_tail=running_tail,
            chunk_text=chunk_text,
            chunk_index=chunk_index,
            chunk_total=chunk_total,
            source_path=chunk.path,
            priority=chunk.priority,
            memory_context=memory_context or "(no recalled context)",
            max_headings=3,
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
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
        extension = extract_markdown_block(content)
        if extension.strip():
            extension, _truncated, _attempts, finish_reason = await continue_until_complete(
                ctx,
                user_input=user_input,
                document=extension,
                model_def=model_def,
                first_finish_reason=finish_reason,
                max_continue_attempts=max_continue_attempts,
                max_continue_tokens=max_continue_tokens,
                continue_prompt_path=continue_prompt_path,
            )
        if extension.strip():
            if document:
                document = document.rstrip() + "\n\n" + extension.rstrip() + "\n"
            else:
                document = extension.rstrip() + "\n"
            if incremental_persist and callable(ctx.execute_tool_fn):
                try:
                    await ctx.execute_tool_fn(
                        _APPEND_TOOL_ID,
                        ToolExecutionContext(
                            tool_id=_APPEND_TOOL_ID,
                            workspace_id=ctx.workspace_id,
                            user_id=ctx.user_id,
                            user_slug=ctx.user_slug,
                            surface_type=ctx.surface_type,
                            surface_id=ctx.surface_id,
                            memory_scope=ctx.memory_scope,
                            mode_id=ctx.mode_id,
                            parameters={"path": document_path, "content": extension + "\n\n"},
                        ),
                    )
                except Exception as exc:  # pylint: disable=broad-except
                    logger.warning(
                        "consolidate: rolling incremental append failed (non-fatal): %s", exc
                    )
        finish_reasons.append(finish_reason)
    return document, finish_reasons


async def _merge_sections(
    ctx: OperationExecutionContext,
    *,
    sections: list[str],
    specs: list[SourceSpec],
    user_input: str,
    memory_context: str,
    model_def: Any,
    max_tokens: int,
    max_continue_attempts: int,
    max_continue_tokens: int,
    continue_prompt_path: str,
) -> tuple[str, str | None]:
    """Merge pre-forged sections into one coherent document."""
    if ctx.complete_fn is None:
        return "", None
    system_prompt = _load_text(ctx, _SYSTEM_PROMPT_MERGE_PATH)
    user_template = _load_text(ctx, _USER_PROMPT_MERGE_PATH)
    rendered: list[str] = []
    for idx, (section, spec) in enumerate(zip(sections, specs), start=1):
        if not section.strip():
            continue
        rendered.append(
            f"### Section {idx} — `{spec.path}` (priority: {spec.priority})\n\n"
            f"```markdown\n{section.rstrip()}\n```"
        )
    sections_block = "\n\n".join(rendered) or "(no sections were forged)"
    priorities_block = render_priorities_block(specs)
    user_prompt = _format_template(
        user_template,
        user_input=user_input,
        sections_block=sections_block,
        priorities_block=priorities_block,
        memory_context=memory_context or "(no recalled context)",
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
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
    document = extract_markdown_block(content)
    if not document.strip():
        return content.strip(), finish_reason
    document, _truncated, _attempts, finish_reason = await continue_until_complete(
        ctx,
        user_input=user_input,
        document=document,
        model_def=model_def,
        first_finish_reason=finish_reason,
        max_continue_attempts=max_continue_attempts,
        max_continue_tokens=max_continue_tokens,
        continue_prompt_path=continue_prompt_path,
    )
    return document, finish_reason


def _build_extended_annotation(
    model_def: Any,
    *,
    sources: list[SourceSpec],
    processing_summary: str,
    density_line: str,
    okf_concept_path: str,
    truncated: bool = False,
    final_finish_reason: str | None = None,
    continue_attempts: int = 0,
) -> str:
    """Build the extended non-rendering provenance footer for a source-aware forge."""
    provider = str(getattr(model_def, "provider", "") or "").strip()
    model_id = str(getattr(model_def, "model_id", "") or "").strip()
    parts: list[str] = []
    attribution_bits: list[str] = []
    if provider:
        attribution_bits.append(f"provider: {provider}")
    if model_id:
        attribution_bits.append(f"model: {model_id}")
    if truncated:
        reason = final_finish_reason or "unknown"
        attribution_bits.append(f"truncated by model · finish_reason: {reason}")
        if continue_attempts:
            attribution_bits.append(f"continue_attempts: {continue_attempts}")
    timestamp = datetime.now(UTC).strftime("%Y-%m-%d")
    parts.append(f"generated by lisa · {' · '.join(attribution_bits)} · {timestamp}")

    readable = [s for s in sources if s.text]
    failed = failed_specs(sources)
    tier_summary = _summarize_source_tiers(readable)
    sources_line = f"sources: {len(readable)} readable"
    if failed:
        sources_line += f", {len(failed)} unreadable"
    if tier_summary:
        sources_line += f" ({tier_summary})"
    parts.append(sources_line)

    parts.append(f"processing: {processing_summary}")
    if density_line:
        parts.append(f"density: {density_line}")
    if okf_concept_path:
        parts.append(f"provenance: {okf_concept_path}")
    body = " · ".join(parts)
    return f"<!--\n  {body}\n  \n  Full provenance (per-source treatment, exact priorities, density math):\n  {okf_concept_path}\n-->"


def _summarize_source_tiers(readable: list[SourceSpec]) -> str:
    """Render a compact tier summary of readable sources by priority keyword."""
    if not readable:
        return ""
    tiers: dict[str, list[str]] = {}
    order: list[str] = []
    for spec in readable:
        key = (spec.priority or "normal").split("—")[0].split(":")[0].strip().lower() or "normal"
        if key not in tiers:
            tiers[key] = []
            order.append(key)
        tiers[key].append(spec.path)
    if len(order) == 1:
        paths = "; ".join(tiers[order[0]][:4])
        if len(tiers[order[0]]) > 4:
            paths += f"; +{len(tiers[order[0]]) - 4} more"
        return paths
    chunks: list[str] = []
    for key in order:
        paths = "; ".join(tiers[key][:3])
        if len(tiers[key]) > 3:
            paths += f"; +{len(tiers[key]) - 3} more"
        chunks.append(f"{key}: {paths}")
    return "; ".join(chunks)


def _build_okf_provenance_body(
    *,
    sources: list[SourceSpec],
    stored_path: str,
    output_bytes: int,
    heading_count: int,
    model_id: str,
    provider: str,
    passes: int,
    continue_attempts: int,
    density_ratio: float,
    density_floor: float,
    density_passed: bool,
    first_finish_reason: str | None,
    final_finish_reason: str | None,
    truncated: bool,
    forge_timestamp: str,
    per_source_meta: list[dict[str, Any]] | None = None,
) -> str:
    """Render the markdown body for the OKF forge-provenance concept."""
    lines: list[str] = ["# Forge Provenance", "", "## Sources", ""]
    readable = [s for s in sources if s.text]
    failed = failed_specs(sources)
    meta = per_source_meta or []
    if readable or failed:
        lines.append("| # | Path | Priority | Bytes | Passes | Density | Treatment |")
        lines.append("| --- | --- | --- | --- | --- | --- | --- |")
        for idx, spec in enumerate(readable, start=1):
            treatment = "preserved"
            if spec.truncated_to_fit:
                treatment = "truncated to per-source cap"
            if spec.expanded_from_glob:
                treatment += f"; glob-expanded from `{spec.origin_glob}`"
            entry = meta[idx - 1] if idx - 1 < len(meta) else {}
            src_passes = entry.get("passes", 0) if entry else 0
            src_density = entry.get("density", "n/a") if entry else "n/a"
            lines.append(
                f"| {idx} | `{spec.path}` | {spec.priority} | {spec.bytes} | "
                f"{src_passes} | {src_density} | {treatment} |"
            )
        for spec in failed:
            lines.append(
                f"| – | `{spec.path}` | {spec.priority} | 0 | 0 | n/a | "
                f"UNREADABLE: {spec.read_error} |"
            )
    else:
        lines.append("_(no sources supplied)_")
    lines.append("")
    lines.append("## Processing")
    lines.append("")
    lines.append(f"- **Model:** {provider}/{model_id}")
    lines.append(f"- **Passes:** {passes} primary"
                 + (f" + {continue_attempts} continuation" if continue_attempts else ""))
    lines.append(f"- **finish_reason:** first={first_finish_reason!r}, final={final_finish_reason!r}")
    lines.append(f"- **Density check:** output {output_bytes}B "
                 f"({density_ratio:.0%} of source bytes; floor {density_floor:.0%}) "
                 f"— {'passed' if density_passed else 'FAILED (lossy abridgment)'}")
    lines.append(f"- **Truncated:** {'yes' if truncated else 'no'}")
    lines.append(f"- **Forge timestamp:** {forge_timestamp}")
    lines.append("")
    lines.append("## Output")
    lines.append("")
    lines.append(f"- **Path:** `{stored_path}`")
    lines.append(f"- **Size:** {output_bytes} bytes")
    lines.append(f"- **Headings:** {heading_count}")
    lines.append("")
    return "\n".join(lines)


async def _write_okf_provenance(
    ctx: OperationExecutionContext,
    *,
    okf_bundle_path: str,
    document_stem: str,
    sources: list[SourceSpec],
    stored_path: str,
    output_bytes: int,
    heading_count: int,
    model_def: Any,
    passes: int,
    continue_attempts: int,
    density_ratio: float,
    density_floor: float,
    density_passed: bool,
    first_finish_reason: str | None,
    final_finish_reason: str | None,
    truncated: bool,
    per_source_meta: list[dict[str, Any]] | None = None,
) -> str | None:
    """Write the OKF forge-provenance concept + append a log entry."""
    if not okf_bundle_path:
        return None
    if not callable(ctx.execute_tool_fn):
        return None
    provider = str(getattr(model_def, "provider", "") or "").strip()
    model_id = str(getattr(model_def, "model_id", "") or "").strip()
    forge_timestamp = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    body = _build_okf_provenance_body(
        sources=sources,
        stored_path=stored_path,
        output_bytes=output_bytes,
        heading_count=heading_count,
        model_id=model_id,
        provider=provider,
        passes=passes,
        continue_attempts=continue_attempts,
        density_ratio=density_ratio,
        density_floor=density_floor,
        density_passed=density_passed,
        first_finish_reason=first_finish_reason,
        final_finish_reason=final_finish_reason,
        truncated=truncated,
        forge_timestamp=forge_timestamp,
        per_source_meta=per_source_meta,
    )
    concept_rel = f"{document_stem}.forge.md"
    try:
        emit_result = await ctx.execute_tool_fn(
            _OKF_EMIT_TOOL_ID,
            ToolExecutionContext(
                tool_id=_OKF_EMIT_TOOL_ID,
                workspace_id=ctx.workspace_id,
                user_id=ctx.user_id,
                user_slug=ctx.user_slug,
                surface_type=ctx.surface_type,
                surface_id=ctx.surface_id,
                memory_scope=ctx.memory_scope,
                mode_id=ctx.mode_id,
                parameters={
                    "bundle_path": okf_bundle_path,
                    "concept_path": concept_rel,
                    "type": "forge-provenance",
                    "title": f"Forge provenance for {stored_path}",
                    "description": (
                        f"Consolidated {readable_count(sources)} source(s) "
                        f"into {stored_path}"
                    ),
                    "tags": ["forge", "provenance", "markdown"],
                    "status": "stable",
                    "generated_by": f"lisa-pack-markdown/{model_id or 'unknown'}",
                    "resource": f"lisa-storage://{stored_path}",
                    "body": body,
                },
            ),
        )
        emit_text = getattr(emit_result, "output_text", "") or ""
        if "failed" in emit_text.lower() and "emitted" not in emit_text.lower():
            logger.warning("consolidate: OKF emit_concept failed: %s", emit_text)
            return None
    except Exception as exc:  # pylint: disable=broad-except
        logger.warning("consolidate: OKF emit_concept raised: %s", exc)
        return None
    full_path = f"{okf_bundle_path}/{concept_rel}"
    try:
        await ctx.execute_tool_fn(
            _OKF_LOG_TOOL_ID,
            ToolExecutionContext(
                tool_id=_OKF_LOG_TOOL_ID,
                workspace_id=ctx.workspace_id,
                user_id=ctx.user_id,
                user_slug=ctx.user_slug,
                surface_type=ctx.surface_type,
                surface_id=ctx.surface_id,
                memory_scope=ctx.memory_scope,
                mode_id=ctx.mode_id,
                parameters={
                    "bundle_path": okf_bundle_path,
                    "action": "forge",
                    "concept_path": concept_rel,
                    "by": f"lisa-pack-markdown/{model_id or 'unknown'}",
                    "note": (
                        f"consolidated {readable_count(sources)} source(s) "
                        f"into {stored_path}"
                    ),
                },
            ),
        )
    except Exception as exc:  # pylint: disable=broad-except
        logger.debug("consolidate: OKF log append failed (non-fatal): %s", exc)
    return full_path


def _format_markdown(document: str) -> str:
    """Lint/format the document via ``mdformat`` if available."""
    try:
        import mdformat  # type: ignore[import-not-found]
    except ImportError:
        return document
    try:
        return mdformat.text(document, options={"number": True}).rstrip() + "\n"
    except Exception as exc:  # pylint: disable=broad-except
        logger.warning("consolidate: mdformat failed (%s); using unformatted document", exc)
        return document


async def execute(ctx: OperationExecutionContext) -> OperationResult:
    """Consolidate multiple source documents into one markdown document, preserving detail."""
    params = ctx.params or {}
    cfg = _resolve_settings(ctx)
    user_input = _extract_query(ctx, params, cfg.max_query_chars)
    if not user_input:
        return OperationResult(output_text=_load_text(ctx, _EMPTY_RESPONSE_PATH))

    try:
        document_path = _extract_document_path(params)
    except RuntimeError as exc:
        return OperationResult(output_text=str(exc))

    if not callable(ctx.execute_tool_fn):
        raise RuntimeError("markdown_consolidate_sources requires execute_tool_fn")

    # ── Source ingestion ───────────────────────────────────────────────
    sources: list[SourceSpec] = []
    if has_sources(params):
        try:
            sources = parse_source_paths(params)
        except RuntimeError as exc:
            return OperationResult(output_text=str(exc))
        if not sources:
            return OperationResult(
                output_text=(
                    "source_paths was supplied but parsed to no entries "
                    "(all empty or invalid)."
                )
            )
        sources = await expand_globs(sources, ctx=ctx)
        sources = await read_sources(sources, ctx=ctx)
        readable = [s for s in sources if s.text]
        if not readable:
            failed_note = "; ".join(
                f"{s.path}: {s.read_error}" for s in failed_specs(sources)
            ) or "unknown"
            return OperationResult(
                output_text=(
                    "All source_paths entries were unreadable. No source "
                    f"material to consolidate. Failures: {failed_note}"
                )
            )
        source_block = render_source_block(sources)
    else:
        return OperationResult(
            output_text="markdown_consolidate_sources requires a non-empty source_paths."
        )

    memory_context = await _recall_memory(ctx, user_input, limit=cfg.max_recall_records)

    # ── Map-reduce decision ────────────────────────────────────────────
    source_total = total_source_bytes(sources)
    n_readable = readable_count(sources)
    use_map_reduce = (
        bool(sources)
        and (
            source_total >= cfg.map_reduce_byte_threshold
            or n_readable >= cfg.map_reduce_source_threshold
        )
    )

    model_def: Any = None
    first_finish_reason: str | None = None
    final_finish_reason: str | None = None
    was_truncated = False
    continue_attempts = 0
    document = ""
    density_passed = True
    density_violation: str | None = None
    density_ratio = 0.0
    density_floor_value = 0.0
    density_retried = False
    density_failed_sources: list[str] = []
    per_source_passes: list[int] = []
    per_source_density: list[str] = []
    per_source_meta: list[dict[str, Any]] | None = None
    rolling_used = False

    # Resolve the model once for all passes.
    if ctx.resolve_model_fn is None:
        raise RuntimeError("markdown requires resolve_model_fn")
    preferred_model_id = ctx.preferred_model_id or ctx.model_id or "default"
    model_def = ctx.resolve_model_fn(preferred_model_id)
    if model_def is None:
        raise RuntimeError(f"markdown could not resolve model '{preferred_model_id}'")

    if use_map_reduce:
        readable = [s for s in sources if s.text]
        sections: list[str] = []
        per_source_finish_reasons: list[str | None] = []
        density_markers: list[str] = []
        for idx, spec in enumerate(readable, start=1):
            use_rolling = False
            if cfg.forge_rolling_accumulation and len(readable) == 1:
                stat_size = await stat_source_chars(spec, ctx=ctx)
                if stat_size is not None and stat_size > 32_000:
                    use_rolling = True
            if use_rolling:
                rolling_used = True
                chunks = await read_source_chunks(spec, ctx=ctx)
                if chunks and chunks[0].read_error:
                    sections.append("")
                    per_source_finish_reasons.append(None)
                    per_source_passes.append(1)
                    per_source_density.append("n/a")
                    continue
                section, chunk_finishs = await _forge_rolling(
                    ctx,
                    chunks,
                    user_input=user_input,
                    memory_context=memory_context,
                    model_def=model_def,
                    max_tokens=cfg.per_source_max_tokens,
                    max_continue_attempts=cfg.max_continue_attempts,
                    max_continue_tokens=cfg.max_continue_tokens,
                    continue_prompt_path=_CONTINUE_PROMPT_PATH,
                    document_path=document_path,
                    incremental_persist=cfg.forge_rolling_incremental_persist,
                )
                sections.append(section)
                per_source_finish_reasons.append(
                    chunk_finishs[-1] if chunk_finishs else None
                )
                per_source_passes.append(max(1, len(chunks)))
                per_source_density.append("n/a")
                continue
            section, sec_finish, _passed, _reason, failed, passes, marker = await _forge_one_source(
                ctx,
                spec=spec,
                idx=idx,
                user_input=user_input,
                memory_context=memory_context,
                model_def=model_def,
                max_tokens=cfg.per_source_max_tokens,
                max_continue_attempts=cfg.max_continue_attempts,
                max_continue_tokens=cfg.max_continue_tokens,
                continue_prompt_path=_CONTINUE_PROMPT_PATH,
                per_source_min_ratio=cfg.per_source_min_ratio,
            )
            sections.append(section)
            per_source_finish_reasons.append(sec_finish)
            per_source_passes.append(passes)
            if failed:
                density_failed_sources.extend(failed)
                per_source_density.append("FAILED")
                if marker:
                    density_markers.append(marker)
            else:
                per_source_density.append("pass")
        first_finish_reason = per_source_finish_reasons[0] if per_source_finish_reasons else None
        if rolling_used and len(sections) == 1:
            merged = sections[0]
            merge_finish = per_source_finish_reasons[0] if per_source_finish_reasons else None
        else:
            merged, merge_finish = await _merge_sections(
                ctx,
                sections=sections,
                specs=readable,
                user_input=user_input,
                memory_context=memory_context,
                model_def=model_def,
                max_tokens=cfg.max_completion_tokens,
                max_continue_attempts=cfg.max_continue_attempts,
                max_continue_tokens=cfg.max_continue_tokens,
                continue_prompt_path=_CONTINUE_PROMPT_PATH,
            )
        if not merged or not merged.strip():
            return OperationResult(output_text=_load_text(ctx, _EMPTY_RESPONSE_PATH))
        document = extract_markdown_block(merged) if "```" in merged else merged
        if not document.strip():
            text = _format_template(
                _load_text(ctx, _EMPTY_DOCUMENT_PATH),
                response_preview=merged[:400],
            )
            return OperationResult(output_text=text)
        if density_markers:
            marker_block = "\n\n".join(density_markers)
            document = marker_block + "\n\n" + document
        final_finish_reason = merge_finish
    else:
        # ── Single-pass consolidation ─────────────────────────────────
        completion_text, model_def, first_finish_reason = await _call_llm(
            ctx,
            user_input,
            memory_context,
            source_block,
            max_tokens=cfg.max_completion_tokens,
        )
        if not completion_text or not completion_text.strip():
            return OperationResult(output_text=_load_text(ctx, _EMPTY_RESPONSE_PATH))
        document = extract_markdown_block(completion_text)
        if not document.strip():
            text = _format_template(
                _load_text(ctx, _EMPTY_DOCUMENT_PATH),
                response_preview=completion_text[:400],
            )
            return OperationResult(output_text=text)
        document, was_truncated, continue_attempts, final_finish_reason = (
            await continue_until_complete(
                ctx,
                user_input=user_input,
                document=document,
                model_def=model_def,
                first_finish_reason=first_finish_reason,
                max_continue_attempts=cfg.max_continue_attempts,
                max_continue_tokens=cfg.max_continue_tokens,
                continue_prompt_path=_CONTINUE_PROMPT_PATH,
            )
        )
        # ── Density floor ─────────────────────────────────────────────
        if sources and source_total > 0:
            density_floor_value = _density_floor_ratio(n_readable)
            density_violation = _density_floor_check(
                source_total_bytes=source_total,
                output_bytes=len(document.encode("utf-8")),
                source_count=n_readable,
            )
            if density_violation is not None:
                density_passed = False
                violation_msg = density_violation
                for _ in range(cfg.density_retry_attempts):
                    density_retried = True
                    retry_text, retry_finish = await _reforge_for_density(
                        ctx,
                        user_input=user_input,
                        memory_context=memory_context,
                        source_block=source_block,
                        model_def=model_def,
                        violation=violation_msg,
                        source_total_bytes=source_total,
                        output_bytes=len(document.encode("utf-8")),
                        source_count=n_readable,
                        max_tokens=cfg.max_completion_tokens,
                    )
                    if not retry_text or not retry_text.strip():
                        break
                    retry_document = extract_markdown_block(retry_text)
                    if not retry_document.strip():
                        break
                    retry_document, was_truncated, continue_attempts, final_finish_reason = (
                        await continue_until_complete(
                            ctx,
                            user_input=user_input,
                            document=retry_document,
                            model_def=model_def,
                            first_finish_reason=retry_finish,
                            max_continue_attempts=cfg.max_continue_attempts,
                            max_continue_tokens=cfg.max_continue_tokens,
                            continue_prompt_path=_CONTINUE_PROMPT_PATH,
                        )
                    )
                    document = retry_document
                    density_violation = _density_floor_check(
                        source_total_bytes=source_total,
                        output_bytes=len(document.encode("utf-8")),
                        source_count=n_readable,
                    )
                    density_passed = density_violation is None
                    if density_passed:
                        break
        density_ratio = (
            len(document.encode("utf-8")) / source_total if source_total else 1.0
        )

    # ── Map-reduce density assessment (post-merge) ────────────────────
    if use_map_reduce and sources and source_total > 0:
        density_floor_value = _density_floor_ratio(n_readable)
        density_violation = _density_floor_check(
            source_total_bytes=source_total,
            output_bytes=len(document.encode("utf-8")),
            source_count=n_readable,
        )
        density_passed = not density_failed_sources
        density_ratio = (
            len(document.encode("utf-8")) / source_total if source_total else 1.0
        )
        per_source_meta = [
            {"passes": per_source_passes[i] if i < len(per_source_passes) else 0,
             "density": per_source_density[i] if i < len(per_source_density) else "n/a"}
            for i in range(n_readable)
        ]
    else:
        per_source_meta = None

    # Lint/format the document.
    document = _format_markdown(document)

    # Char cap.
    if len(document) > cfg.max_document_chars:
        document = document[:cfg.max_document_chars] + "\n\n<!-- truncated -->\n"
        was_truncated = True

    # Annotate with the model that wrote it, unless the user opted out.
    annotate_model = bool(params.get("annotate_model", True))
    if annotate_model and not _user_asked_not_to_annotate(user_input):
        okf_bundle_path = str(params.get("okf_bundle_path") or "").strip()
        okf_concept_path = ""
        if okf_bundle_path:
            stem = _document_stem(document_path)
            okf_concept_path = f"{okf_bundle_path}/{stem}.forge.md"
        processing_summary = (
            "1 primary pass"
            + (f" + {continue_attempts} continuation" if continue_attempts else "")
            + ("; density passed" if density_passed else "; density FAILED"
               if sources and source_total > 0 else "")
            + ("; re-forged" if density_retried else "")
            + f" · finish_reason: {final_finish_reason!r}"
        )
        density_line = ""
        if sources and source_total > 0:
            density_line = (
                f"{len(document.encode('utf-8'))}B / {source_total}B "
                f"({density_ratio:.0%})"
            )
        annotation = _build_extended_annotation(
            model_def,
            sources=sources,
            processing_summary=processing_summary,
            density_line=density_line,
            okf_concept_path=okf_concept_path,
            truncated=was_truncated,
            final_finish_reason=final_finish_reason if was_truncated else None,
            continue_attempts=continue_attempts if was_truncated else 0,
        )
        if annotation:
            document = document.rstrip() + "\n\n" + annotation + "\n"

    try:
        outline = await _validate_outline(ctx, document)
    except RuntimeError as exc:
        return OperationResult(output_text=str(exc))

    heading_count = int(outline.get("heading_count", 0) or 0)
    if heading_count < _MIN_HEADINGS:
        text = _format_template(
            _load_text(ctx, _NO_OUTLINE_PATH),
            heading_count=heading_count,
        )
        return OperationResult(output_text=text)

    # Phase A: advisory lint gate (mirrors forge_document). Lint is never a
    # hard reject; the outline gate above is the only hard gate. One repair
    # pass when ``forge_lint_repair`` is set; residual issues are persisted
    # as a non-rendering HTML comment so the user knows the artifact has
    # warnings. The repair pass uses the source-aware system prompt so the
    # preserve-detail directive still applies.
    lint_issues = lint_markdown(document, rules_csv=cfg.lint_rules or None)
    lint_repair_used = False
    if lint_issues and cfg.lint_repair:
        repaired = await _lint_repair_pass(
            ctx,
            user_input=user_input,
            document=document,
            model_def=model_def,
            issues=lint_issues,
            max_tokens=cfg.max_completion_tokens,
        )
        if repaired != document:
            lint_repair_used = True
            document = _format_markdown(repaired)
            try:
                outline = await _validate_outline(ctx, document)
            except RuntimeError as exc:
                return OperationResult(output_text=str(exc))
            heading_count = int(outline.get("heading_count", 0) or 0)
            if heading_count < _MIN_HEADINGS:
                text = _format_template(
                    _load_text(ctx, _NO_OUTLINE_PATH),
                    heading_count=heading_count,
                )
                return OperationResult(output_text=text)
            lint_issues = lint_markdown(document, rules_csv=cfg.lint_rules or None)
    if lint_issues:
        comment = lint_summary_comment(lint_issues)
        if comment:
            document = document.rstrip() + "\n\n" + comment + "\n"

    try:
        write_response = await _persist_document(ctx, document, document_path)
    except RuntimeError as exc:
        text = _format_template(
            _load_text(ctx, _STORAGE_FAILED_PATH),
            error=str(exc),
        )
        return OperationResult(output_text=text)

    ok = bool(write_response.get("ok", False))
    if not ok:
        text = _format_template(
            _load_text(ctx, _STORAGE_FAILED_PATH),
            error=write_response.get("raw", "unknown storage error"),
        )
        return OperationResult(output_text=text)

    stored_path = str(write_response.get("path") or document_path)
    byte_count = int(write_response.get("bytes") or len(document.encode("utf-8")))

    okf_concept_path_written: str | None = None
    if sources and str(params.get("okf_bundle_path") or "").strip():
        okf_concept_path_written = await _write_okf_provenance(
            ctx,
            okf_bundle_path=str(params["okf_bundle_path"]).strip(),
            document_stem=_document_stem(document_path),
            sources=sources,
            stored_path=stored_path,
            output_bytes=byte_count,
            heading_count=heading_count,
            model_def=model_def,
            passes=1 + (1 if density_retried else 0),
            continue_attempts=continue_attempts,
            density_ratio=density_ratio,
            density_floor=density_floor_value,
            density_passed=density_passed,
            first_finish_reason=first_finish_reason,
            final_finish_reason=final_finish_reason,
            truncated=was_truncated,
            per_source_meta=per_source_meta,
        )

    summary = (
        f"Document consolidated and stored at `{stored_path}` "
        f"({byte_count} bytes, {heading_count} headings)."
    )
    if was_truncated:
        summary += (
            f" WARNING: persisted truncated after {continue_attempts} "
            f"continuation attempt(s) (final finish_reason="
            f"{final_finish_reason!r}). See annotation in document."
        )
    if sources and not density_passed:
        summary += (
            f" WARNING: density floor FAILED ({density_ratio:.0%} of "
            f"{source_total}B source; floor {density_floor_value:.0%}). "
            f"The persisted document may be a lossy abridgment — review it."
        )
    if density_failed_sources:
        summary += (
            f" WARNING: per-source density floor failed for "
            f"{len(density_failed_sources)} source(s): "
            f"{', '.join(density_failed_sources)}. See inline markers in "
            f"the persisted document."
        )
    if okf_concept_path_written:
        summary += f" Provenance: `{okf_concept_path_written}`"
    if lint_issues:
        summary += (
            f" Note: {len(lint_issues)} structural lint warning(s) remain "
            f"after {'a' if lint_repair_used else 'no'} repair pass; see lint comment in document."
        )
    return OperationResult(
        output_text=summary,
        output_json={
            "path": stored_path,
            "bytes": byte_count,
            "heading_count": heading_count,
            "outline": outline,
            "truncated": was_truncated,
            "continue_attempts": continue_attempts,
            "first_finish_reason": first_finish_reason,
            "final_finish_reason": final_finish_reason,
            "source_count": n_readable,
            "rolling_accumulation_used": rolling_used,
            "source_total_bytes": source_total,
            "density_ratio": density_ratio,
            "density_passed": density_passed,
            "density_retried": density_retried,
            "density_failed_sources": density_failed_sources,
            "okf_provenance_path": okf_concept_path_written,
            "lint_warnings": len(lint_issues),
            "lint_repair_used": lint_repair_used,
        },
        extra={
            "memory_context_lines": memory_context.count("\n") + 1 if memory_context else 0,
            "document_path": stored_path,
            "truncated": was_truncated,
            "continue_attempts": continue_attempts,
            "source_count": n_readable,
            "density_passed": density_passed,
            "lint_warnings": len(lint_issues),
        },
    )
