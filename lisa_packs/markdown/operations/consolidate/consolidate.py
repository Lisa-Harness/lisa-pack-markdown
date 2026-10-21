"""Consolidate-sources operation — multi-source preserve-detail markdown merging.

Owns the source-aware consolidation path that used to live inside
``markdown_forge_document``. Splitting it out lets the forge operation stay a
narrow single-document writer while this operation owns the distinct concerns
of multi-source ingestion, preserve-detail consolidation, anti-abridgment
density floors, map-reduce sectioning, and OKF provenance.

Pipeline:

1. Parse + expand + read every source via ``lisa-storage`` tools
   (``list_files`` for globs, ``read_file`` for content) through
   ``ctx.host.execute_tool``.
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
memory) and writes (document, provenance) go through ``ctx.host.execute_tool``.
Authored prompts are loaded via ``ctx.host.read_authored_text``; per the
pack-content rule, a missing/empty loader raises rather than silently
falling back.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from kernel.models import OperationExecutionContext, OperationResult, ToolExecutionContext
from lisa_packs.markdown.operations._config import cfg_bool, cfg_float, cfg_int, cfg_str
from lisa_packs.markdown.operations._continuation import (
    continue_until_complete,
    extract_markdown_block,
    repair_inline_fences,
)
from lisa_packs.markdown.operations._lint import (
    LintIssue,
    format_lint_issues,
    lint_markdown,
    lint_summary_comment,
)
from lisa_packs.markdown.operations._memory import recall_memory
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
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

__all__ = ["Parameters", "execute"]

# Tool ids — declared in the operation YAML ``tool_ids`` and routed through
# ``ctx.host.execute_tool``.
_OUTLINE_TOOL_ID = "markdown_generate"
_WRITE_TOOL_ID = "write_file"
_READ_TOOL_ID = "read_file"
_APPEND_TOOL_ID = "append_file"
_OKF_EMIT_TOOL_ID = "okf_emit_concept"
_OKF_LOG_TOOL_ID = "okf_log"

# Authored-text paths — loaded via ctx.host.read_authored_text (SDK loader).
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
_DENSITY_FLOOR_MANY_SOURCES_RATIO = 0.15  # 5+ readable sources
_DENSITY_FLOOR_FEW_SOURCES_RATIO = 0.30  # 2-4 readable sources
_DENSITY_FLOOR_MANY_SOURCES_THRESHOLD = 5
_DENSITY_FLOOR_FEW_SOURCES_THRESHOLD = 2
_DENSITY_RETRY_ATTEMPTS = 2
# Provenance tier summary: how many source paths to list per priority tier.
_SINGLE_TIER_PATH_LIMIT = 4
_MULTI_TIER_PATH_LIMIT = 3
# A single source above this many chars is accumulated in rolling mode.
_ROLLING_ACCUMULATION_CHAR_THRESHOLD = 32_000
# Per-source density floor (map-reduce path).
_PER_SOURCE_MIN_RATIO = 0.10
# Phase A: advisory lint gate defaults (mirrors forge_document).
_DEFAULT_LINT_RULES = ""
_DEFAULT_LINT_REPAIR = True

# Minimum structural bar for a forged document: at least one heading.
_MIN_HEADINGS = 1


def _load_text(ctx: OperationExecutionContext, path: str) -> str:
    """Load authored markdown via the SDK content-registry loader."""
    if ctx.host is None:
        raise RuntimeError(
            f"read_authored_text_fn is not wired; cannot load authored content at {path}"
        )
    try:
        loaded = str(ctx.host.read_authored_text(path) or "").strip()
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
        description="Relative path within the lisa-storage files sandbox where the consolidated "
        "document is persisted.",
    )
    source_paths: list[str | dict[str, str]] = Field(
        default_factory=list,
        description="Ordered list of source documents to consolidate. Each entry is either a "
        "path/glob string (default priority 'normal') or a dict with 'path' (required) and "
        "'priority' (optional NL prose). When supplied, the operation consolidates the sources "
        "preserving detail rather than summarizing.",
    )
    query: str = Field(
        default="",
        description="Natural-language instruction for the document. Falls back to ctx.query when "
        "empty.",
    )
    okf_bundle_path: str = Field(
        default="okf/forge-provenance",
        description="Relative path of the OKF bundle directory where the provenance concept is "
        "written. Set to empty to skip OKF provenance capture.",
    )
    annotate_model: bool = Field(
        default=True,
        description="When True, append a non-rendering HTML comment to the forged document noting "
        "the provider/model that wrote it. Set False to suppress unconditionally.",
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
        max_document_chars=cfg_int(
            pc, "forge_max_document_chars_with_sources", _MAX_DOCUMENT_CHARS
        ),
        max_query_chars=cfg_int(pc, "forge_max_query_chars", _MAX_QUERY_CHARS),
        map_reduce_byte_threshold=cfg_int(
            pc, "forge_map_reduce_byte_threshold", _MAP_REDUCE_BYTE_THRESHOLD
        ),
        map_reduce_source_threshold=cfg_int(
            pc, "forge_map_reduce_source_threshold", _MAP_REDUCE_SOURCE_THRESHOLD
        ),
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


def _extract_query(
    ctx: OperationExecutionContext, params: dict[str, Any], max_query_chars: int
) -> str:
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
    if ctx.host is None:
        raise RuntimeError("markdown requires execute_tool_fn to call memory/storage tools")
    return await recall_memory(ctx, query, limit=limit)


async def _call_llm(
    ctx: OperationExecutionContext,
    user_input: str,
    memory_context: str,
    source_block: str,
    *,
    max_tokens: int,
) -> tuple[str, Any, bool | None, str | None]:
    """Call the host's completion function with the source-aware consolidate prompt."""
    if ctx.host is None:
        raise RuntimeError("markdown requires complete_fn and resolve_model_fn")

    preferred_model_id = ctx.preferred_model_id or ctx.model_id or "default"
    model_def = ctx.host.resolve_model(preferred_model_id)
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
    completion = await ctx.host.complete(
        messages=messages,
        model_def=model_def,
        max_tokens=max_tokens,
        temperature=0.2,
        tools=None,
        response_format=None,
    )
    content = getattr(completion, "content", "") or ""
    finish_reason = getattr(completion, "finish_reason", None)
    truncated = getattr(completion, "truncated", None)
    return content.strip(), model_def, truncated, finish_reason


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
    if not issues or ctx.host is None:
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
        completion = await ctx.host.complete(
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
    tool_result = await ctx.host.execute_tool(
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
    tool_result = await ctx.host.execute_tool(
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
    if source_count >= _DENSITY_FLOOR_FEW_SOURCES_THRESHOLD:
        return _DENSITY_FLOOR_FEW_SOURCES_RATIO
    return 0.0


def _density_floor_check(
    *,
    source_total_bytes: int,
    output_bytes: int,
    source_count: int,
) -> str | None:
    """Return a violation message if the output looks like a lossy abridgment."""
    if source_total_bytes <= 0 or source_count < _DENSITY_FLOOR_FEW_SOURCES_THRESHOLD:
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
    if ctx.host is None:
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
    completion = await ctx.host.complete(
        messages=messages,
        model_def=model_def,
        max_tokens=max_tokens,
        temperature=0.2,
        tools=None,
        response_format=None,
    )
    content = getattr(completion, "content", "") or ""
    finish_reason = getattr(completion, "finish_reason", None)
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
    if ctx.host is None:
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
    completion = await ctx.host.complete(
        messages=messages,
        model_def=model_def,
        max_tokens=max_tokens,
        temperature=0.2,
        tools=None,
        response_format=None,
    )
    content = getattr(completion, "content", "") or ""
    finish_reason = getattr(completion, "finish_reason", None)
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
    if ctx.host is None:
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
    completion = await ctx.host.complete(
        messages=messages,
        model_def=model_def,
        max_tokens=max_tokens,
        temperature=0.2,
        tools=None,
        response_format=None,
    )
    content = getattr(completion, "content", "") or ""
    finish_reason = getattr(completion, "finish_reason", None)
    _truncated = getattr(completion, "truncated", None)
    section = extract_markdown_block(content)
    passes = 1
    failed_paths: list[str] = []
    if section.strip():
        section, _truncated, _attempts, finish_reason, _stop = await continue_until_complete(
            ctx,
            user_input=user_input,
            document=section,
            model_def=model_def,
            first_truncated=_truncated,
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
                retry_section, _t, _a, retry_finish, _stop = await continue_until_complete(
                    ctx,
                    user_input=user_input,
                    document=retry_section,
                    model_def=model_def,
                    first_truncated=False,
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
    return (
        section,
        finish_reason,
        density_passed,
        density_reason,
        failed_paths,
        passes,
        density_marker,
    )


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
    if ctx.host is None:
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
        completion = await ctx.host.complete(
            messages=messages,
            model_def=model_def,
            max_tokens=max_tokens,
            temperature=0.2,
            tools=None,
            response_format=None,
        )
        content = getattr(completion, "content", "") or ""
        finish_reason = getattr(completion, "finish_reason", None)
        _truncated = getattr(completion, "truncated", None)
        extension = extract_markdown_block(content)
        if extension.strip():
            extension, _truncated, _attempts, finish_reason, _stop = await continue_until_complete(
                ctx,
                user_input=user_input,
                document=extension,
                model_def=model_def,
                first_truncated=_truncated,
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
            if incremental_persist and ctx.host is not None:
                try:
                    await ctx.host.execute_tool(
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
    if ctx.host is None:
        return "", None
    system_prompt = _load_text(ctx, _SYSTEM_PROMPT_MERGE_PATH)
    user_template = _load_text(ctx, _USER_PROMPT_MERGE_PATH)
    rendered: list[str] = []
    for idx, (section, spec) in enumerate(zip(sections, specs, strict=False), start=1):
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
    completion = await ctx.host.complete(
        messages=messages,
        model_def=model_def,
        max_tokens=max_tokens,
        temperature=0.2,
        tools=None,
        response_format=None,
    )
    content = getattr(completion, "content", "") or ""
    finish_reason = getattr(completion, "finish_reason", None)
    _truncated = getattr(completion, "truncated", None)
    document = extract_markdown_block(content)
    if not document.strip():
        return content.strip(), finish_reason
    document, _truncated, _attempts, finish_reason, _stop = await continue_until_complete(
        ctx,
        user_input=user_input,
        document=document,
        model_def=model_def,
        first_truncated=_truncated,
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
    return (
        f"<!--\n  {body}\n  \n"
        "  Full provenance (per-source treatment, exact priorities, density math):\n"
        f"  {okf_concept_path}\n-->"
    )


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
        paths = "; ".join(tiers[order[0]][:_SINGLE_TIER_PATH_LIMIT])
        if len(tiers[order[0]]) > _SINGLE_TIER_PATH_LIMIT:
            paths += f"; +{len(tiers[order[0]]) - _SINGLE_TIER_PATH_LIMIT} more"
        return paths
    chunks: list[str] = []
    for key in order:
        paths = "; ".join(tiers[key][:_MULTI_TIER_PATH_LIMIT])
        if len(tiers[key]) > _MULTI_TIER_PATH_LIMIT:
            paths += f"; +{len(tiers[key]) - _MULTI_TIER_PATH_LIMIT} more"
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
    lines.append(
        f"- **Passes:** {passes} primary"
        + (f" + {continue_attempts} continuation" if continue_attempts else "")
    )
    lines.append(
        f"- **finish_reason:** first={first_finish_reason!r}, final={final_finish_reason!r}"
    )
    lines.append(
        f"- **Density check:** output {output_bytes}B "
        f"({density_ratio:.0%} of source bytes; floor {density_floor:.0%}) "
        f"— {'passed' if density_passed else 'FAILED (lossy abridgment)'}"
    )
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
    if ctx.host is None:
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
        emit_result = await ctx.host.execute_tool(
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
                        f"Consolidated {readable_count(sources)} source(s) into {stored_path}"
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
        await ctx.host.execute_tool(
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
                        f"consolidated {readable_count(sources)} source(s) into {stored_path}"
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
        import mdformat
    except ImportError:
        return document
    try:
        return mdformat.text(document, options={"number": True}).rstrip() + "\n"
    except Exception as exc:  # pylint: disable=broad-except
        logger.warning("consolidate: mdformat failed (%s); using unformatted document", exc)
        return document


# ── Consolidation phases ────────────────────────────────────────────
#
# ``execute`` is the pipeline's orchestrator, not its implementation. The
# work is split into five phases — prepare → forge → finalize → validate
# → persist — each in its own helper, so no single function carries the
# whole pipeline's complexity. ``_ConsolidationState`` is the shared
# mutable record the phases read and write: the map-reduce and single-pass
# paths fill in the same fields, so the shared post-merge phases (density
# assessment, annotation, provenance, summary) do not need to know which
# path ran.


@dataclass
class _MapReduceSections:
    """Per-source forge results for the map-reduce path."""

    sections: list[str] = field(default_factory=list)
    finish_reasons: list[str | None] = field(default_factory=list)
    passes: list[int] = field(default_factory=list)
    density: list[str] = field(default_factory=list)
    density_markers: list[str] = field(default_factory=list)
    density_failed_sources: list[str] = field(default_factory=list)
    rolling_used: bool = False


@dataclass
class _ConsolidationState:
    """Operation state threaded through the consolidation phases."""

    # ── prepared inputs ──
    params: dict[str, Any]
    cfg: _ResolvedSettings
    user_input: str = ""
    document_path: str = ""
    sources: list[SourceSpec] = field(default_factory=list)
    source_block: str = ""
    source_total: int = 0
    n_readable: int = 0
    use_map_reduce: bool = False

    # ── forge outputs ──
    document: str = ""
    model_def: Any = None
    memory_context: str = ""
    first_finish_reason: str | None = None
    final_finish_reason: str | None = None
    was_truncated: bool = False
    continue_attempts: int = 0

    # ── density telemetry ──
    density_passed: bool = True
    density_violation: str | None = None
    density_ratio: float = 0.0
    density_floor_value: float = 0.0
    density_retried: bool = False
    density_failed_sources: list[str] = field(default_factory=list)
    per_source_passes: list[int] = field(default_factory=list)
    per_source_density: list[str] = field(default_factory=list)
    per_source_meta: list[dict[str, Any]] | None = None
    rolling_used: bool = False


@dataclass
class _LintOutcome:
    """Result of the outline validation + advisory lint gate."""

    document: str
    outline: dict[str, Any]
    heading_count: int
    issues: list[LintIssue]
    repair_used: bool
    error: str | None = None


async def _ingest_sources(
    ctx: OperationExecutionContext,
    params: dict[str, Any],
) -> tuple[list[SourceSpec], str, str | None]:
    """Parse, expand, and read every source document.

    Returns ``(sources, source_block, error)``. ``error`` is the
    user-facing message when the sources cannot be consolidated; the
    sources list is empty in that case.
    """
    if not has_sources(params):
        return [], "", "markdown_consolidate_sources requires a non-empty source_paths."
    try:
        sources = parse_source_paths(params)
    except RuntimeError as exc:
        return [], "", str(exc)
    if not sources:
        return (
            [],
            "",
            "source_paths was supplied but parsed to no entries (all empty or invalid).",
        )
    sources = await expand_globs(sources, ctx=ctx)
    sources = await read_sources(sources, ctx=ctx)
    if not [s for s in sources if s.text]:
        failed_note = (
            "; ".join(f"{s.path}: {s.read_error}" for s in failed_specs(sources)) or "unknown"
        )
        return (
            sources,
            "",
            "All source_paths entries were unreadable. No source "
            f"material to consolidate. Failures: {failed_note}",
        )
    return sources, render_source_block(sources), None


async def _prepare_consolidation(
    ctx: OperationExecutionContext,
) -> tuple[_ConsolidationState, OperationResult | None]:
    """Phase 1 — resolve settings, the user instruction, and the sources.

    Returns ``(state, failure)``. ``failure`` is a terminal
    ``OperationResult`` when the operation cannot proceed (no
    instruction, an invalid document path, or unusable sources).
    """
    params = ctx.params or {}
    cfg = _resolve_settings(ctx)
    state = _ConsolidationState(params=params, cfg=cfg)
    user_input = _extract_query(ctx, params, cfg.max_query_chars)
    if not user_input:
        return state, OperationResult(output_text=_load_text(ctx, _EMPTY_RESPONSE_PATH))
    state.user_input = user_input
    try:
        state.document_path = _extract_document_path(params)
    except RuntimeError as exc:
        return state, OperationResult(output_text=str(exc))
    if ctx.host is None:
        raise RuntimeError("markdown_consolidate_sources requires execute_tool_fn")
    sources, source_block, error = await _ingest_sources(ctx, params)
    if error is not None:
        return state, OperationResult(output_text=error)
    state.sources = sources
    state.source_block = source_block
    return state, None


def _resolve_model(ctx: OperationExecutionContext) -> Any:
    """Resolve the operation's model definition once for all passes."""
    if ctx.host is None:
        raise RuntimeError("markdown requires resolve_model_fn")
    preferred_model_id = ctx.preferred_model_id or ctx.model_id or "default"
    model_def = ctx.host.resolve_model(preferred_model_id)
    if model_def is None:
        raise RuntimeError(f"markdown could not resolve model '{preferred_model_id}'")
    return model_def


async def _should_use_rolling(
    ctx: OperationExecutionContext,
    *,
    state: _ConsolidationState,
    spec: SourceSpec,
    readable: list[SourceSpec],
) -> bool:
    """Return True when this source must be forged in rolling mode.

    Rolling accumulation is opt-in (``forge_rolling_accumulation``) and
    applies only when a SINGLE source is oversized.
    """
    if not state.cfg.forge_rolling_accumulation or len(readable) != 1:
        return False
    stat_size = await stat_source_chars(spec, ctx=ctx)
    return stat_size is not None and stat_size > _ROLLING_ACCUMULATION_CHAR_THRESHOLD


async def _forge_rolling_source(
    ctx: OperationExecutionContext,
    *,
    state: _ConsolidationState,
    sections: _MapReduceSections,
    spec: SourceSpec,
) -> None:
    """Forge one source in rolling mode and record its per-source result."""
    cfg = state.cfg
    chunks = await read_source_chunks(spec, ctx=ctx)
    if chunks and chunks[0].read_error:
        sections.sections.append("")
        sections.finish_reasons.append(None)
        sections.passes.append(1)
        sections.density.append("n/a")
        return
    section, chunk_finishs = await _forge_rolling(
        ctx,
        chunks,
        user_input=state.user_input,
        memory_context=state.memory_context,
        model_def=state.model_def,
        max_tokens=cfg.per_source_max_tokens,
        max_continue_attempts=cfg.max_continue_attempts,
        max_continue_tokens=cfg.max_continue_tokens,
        continue_prompt_path=_CONTINUE_PROMPT_PATH,
        document_path=state.document_path,
        incremental_persist=cfg.forge_rolling_incremental_persist,
    )
    sections.sections.append(section)
    sections.finish_reasons.append(chunk_finishs[-1] if chunk_finishs else None)
    sections.passes.append(max(1, len(chunks)))
    sections.density.append("n/a")


async def _forge_plain_source(
    ctx: OperationExecutionContext,
    *,
    state: _ConsolidationState,
    sections: _MapReduceSections,
    spec: SourceSpec,
    idx: int,
) -> None:
    """Forge one source in bounded-context mode and record its result."""
    cfg = state.cfg
    section, sec_finish, _passed, _reason, failed, passes, marker = await _forge_one_source(
        ctx,
        spec=spec,
        idx=idx,
        user_input=state.user_input,
        memory_context=state.memory_context,
        model_def=state.model_def,
        max_tokens=cfg.per_source_max_tokens,
        max_continue_attempts=cfg.max_continue_attempts,
        max_continue_tokens=cfg.max_continue_tokens,
        continue_prompt_path=_CONTINUE_PROMPT_PATH,
        per_source_min_ratio=cfg.per_source_min_ratio,
    )
    sections.sections.append(section)
    sections.finish_reasons.append(sec_finish)
    sections.passes.append(passes)
    if failed:
        sections.density_failed_sources.extend(failed)
        sections.density.append("FAILED")
        if marker:
            sections.density_markers.append(marker)
    else:
        sections.density.append("pass")


async def _forge_source_section(
    ctx: OperationExecutionContext,
    *,
    state: _ConsolidationState,
    sections: _MapReduceSections,
    spec: SourceSpec,
    idx: int,
    readable: list[SourceSpec],
) -> None:
    """Forge one source's section, choosing the rolling or bounded path."""
    if await _should_use_rolling(ctx, state=state, spec=spec, readable=readable):
        sections.rolling_used = True
        await _forge_rolling_source(ctx, state=state, sections=sections, spec=spec)
        return
    await _forge_plain_source(ctx, state=state, sections=sections, spec=spec, idx=idx)


async def _merge_forged_sections(
    ctx: OperationExecutionContext,
    *,
    state: _ConsolidationState,
    sections: _MapReduceSections,
    readable: list[SourceSpec],
) -> tuple[str, str | None]:
    """Merge the per-source sections into one coherent document.

    A single rolling-mode section is already the whole document, so it is
    not merged again.
    """
    if state.rolling_used and len(sections.sections) == 1:
        finish = sections.finish_reasons[0] if sections.finish_reasons else None
        return sections.sections[0], finish
    cfg = state.cfg
    return await _merge_sections(
        ctx,
        sections=sections.sections,
        specs=readable,
        user_input=state.user_input,
        memory_context=state.memory_context,
        model_def=state.model_def,
        max_tokens=cfg.max_completion_tokens,
        max_continue_attempts=cfg.max_continue_attempts,
        max_continue_tokens=cfg.max_continue_tokens,
        continue_prompt_path=_CONTINUE_PROMPT_PATH,
    )


async def _run_map_reduce_pass(
    ctx: OperationExecutionContext,
    state: _ConsolidationState,
) -> OperationResult | None:
    """Phase 2 (map-reduce) — forge one section per source, then merge."""
    readable = [s for s in state.sources if s.text]
    sections = _MapReduceSections()
    for idx, spec in enumerate(readable, start=1):
        await _forge_source_section(
            ctx, state=state, sections=sections, spec=spec, idx=idx, readable=readable
        )
    state.first_finish_reason = sections.finish_reasons[0] if sections.finish_reasons else None
    state.per_source_passes = sections.passes
    state.per_source_density = sections.density
    state.density_failed_sources = sections.density_failed_sources
    state.rolling_used = sections.rolling_used
    merged, merge_finish = await _merge_forged_sections(
        ctx, state=state, sections=sections, readable=readable
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
    if sections.density_markers:
        marker_block = "\n\n".join(sections.density_markers)
        document = marker_block + "\n\n" + document
    state.document = document
    state.final_finish_reason = merge_finish
    return None


async def _enforce_density_floor(
    ctx: OperationExecutionContext,
    state: _ConsolidationState,
) -> None:
    """Apply the anti-abridgment density floor to a single-pass document.

    Issues up to ``density_retry_attempts`` sharpened re-forge passes while
    the output stays below the floor.
    """
    cfg = state.cfg
    if not (state.sources and state.source_total > 0):
        return
    state.density_floor_value = _density_floor_ratio(state.n_readable)
    state.density_violation = _density_floor_check(
        source_total_bytes=state.source_total,
        output_bytes=len(state.document.encode("utf-8")),
        source_count=state.n_readable,
    )
    if state.density_violation is None:
        return
    state.density_passed = False
    violation_msg = state.density_violation
    for _ in range(cfg.density_retry_attempts):
        state.density_retried = True
        retry_text, retry_finish = await _reforge_for_density(
            ctx,
            user_input=state.user_input,
            memory_context=state.memory_context,
            source_block=state.source_block,
            model_def=state.model_def,
            violation=violation_msg,
            source_total_bytes=state.source_total,
            output_bytes=len(state.document.encode("utf-8")),
            source_count=state.n_readable,
            max_tokens=cfg.max_completion_tokens,
        )
        if not retry_text or not retry_text.strip():
            break
        retry_document = extract_markdown_block(retry_text)
        if not retry_document.strip():
            break
        (
            retry_document,
            was_truncated,
            continue_attempts,
            final_finish_reason,
            _stop,
        ) = await continue_until_complete(
            ctx,
            user_input=state.user_input,
            document=retry_document,
            model_def=state.model_def,
            first_truncated=False,
            first_finish_reason=retry_finish,
            max_continue_attempts=cfg.max_continue_attempts,
            max_continue_tokens=cfg.max_continue_tokens,
            continue_prompt_path=_CONTINUE_PROMPT_PATH,
        )
        state.document = retry_document
        state.was_truncated = was_truncated
        state.continue_attempts = continue_attempts
        state.final_finish_reason = final_finish_reason
        state.density_violation = _density_floor_check(
            source_total_bytes=state.source_total,
            output_bytes=len(state.document.encode("utf-8")),
            source_count=state.n_readable,
        )
        state.density_passed = state.density_violation is None
        if state.density_passed:
            break


async def _run_single_pass(
    ctx: OperationExecutionContext,
    state: _ConsolidationState,
) -> OperationResult | None:
    """Phase 2 (single pass) — forge the whole document in one call."""
    cfg = state.cfg
    completion_text, resolved_model, first_truncated, first_finish_reason = await _call_llm(
        ctx,
        state.user_input,
        state.memory_context,
        state.source_block,
        max_tokens=cfg.max_completion_tokens,
    )
    state.model_def = resolved_model
    state.first_finish_reason = first_finish_reason
    if not completion_text or not completion_text.strip():
        return OperationResult(output_text=_load_text(ctx, _EMPTY_RESPONSE_PATH))
    document = extract_markdown_block(completion_text)
    if not document.strip():
        text = _format_template(
            _load_text(ctx, _EMPTY_DOCUMENT_PATH),
            response_preview=completion_text[:400],
        )
        return OperationResult(output_text=text)
    (
        document,
        was_truncated,
        continue_attempts,
        final_finish_reason,
        _stop,
    ) = await continue_until_complete(
        ctx,
        user_input=state.user_input,
        document=document,
        model_def=state.model_def,
        first_truncated=first_truncated,
        first_finish_reason=first_finish_reason,
        max_continue_attempts=cfg.max_continue_attempts,
        max_continue_tokens=cfg.max_continue_tokens,
        continue_prompt_path=_CONTINUE_PROMPT_PATH,
    )
    state.document = document
    state.was_truncated = was_truncated
    state.continue_attempts = continue_attempts
    state.final_finish_reason = final_finish_reason
    await _enforce_density_floor(ctx, state)
    state.density_ratio = (
        len(state.document.encode("utf-8")) / state.source_total if state.source_total else 1.0
    )
    return None


def _assess_map_reduce_density(state: _ConsolidationState) -> None:
    """Assess the merged document against the density floor (map-reduce path)."""
    if not (state.use_map_reduce and state.sources and state.source_total > 0):
        state.per_source_meta = None
        return
    state.density_floor_value = _density_floor_ratio(state.n_readable)
    state.density_violation = _density_floor_check(
        source_total_bytes=state.source_total,
        output_bytes=len(state.document.encode("utf-8")),
        source_count=state.n_readable,
    )
    state.density_passed = not state.density_failed_sources
    state.density_ratio = (
        len(state.document.encode("utf-8")) / state.source_total if state.source_total else 1.0
    )
    state.per_source_meta = [
        {
            "passes": state.per_source_passes[i] if i < len(state.per_source_passes) else 0,
            "density": state.per_source_density[i] if i < len(state.per_source_density) else "n/a",
        }
        for i in range(state.n_readable)
    ]


async def _forge_consolidation(
    ctx: OperationExecutionContext,
    state: _ConsolidationState,
) -> OperationResult | None:
    """Phase 2 — recall memory, choose the forge path, and forge the document."""
    cfg = state.cfg
    state.memory_context = await _recall_memory(ctx, state.user_input, limit=cfg.max_recall_records)
    state.source_total = total_source_bytes(state.sources)
    state.n_readable = readable_count(state.sources)
    state.use_map_reduce = bool(state.sources) and (
        state.source_total >= cfg.map_reduce_byte_threshold
        or state.n_readable >= cfg.map_reduce_source_threshold
    )
    state.model_def = _resolve_model(ctx)
    if state.use_map_reduce:
        failure = await _run_map_reduce_pass(ctx, state)
    else:
        failure = await _run_single_pass(ctx, state)
    if failure is not None:
        return failure
    _assess_map_reduce_density(state)
    return None


def _apply_annotation(
    *,
    state: _ConsolidationState,
    document: str,
) -> str:
    """Append the non-rendering provenance footer unless the user opted out."""
    params = state.params
    annotate_model = bool(params.get("annotate_model", True))
    if not annotate_model or _user_asked_not_to_annotate(state.user_input):
        return document
    okf_bundle_path = str(params.get("okf_bundle_path") or "").strip()
    okf_concept_path = ""
    if okf_bundle_path:
        stem = _document_stem(state.document_path)
        okf_concept_path = f"{okf_bundle_path}/{stem}.forge.md"
    processing_summary = (
        "1 primary pass"
        + (f" + {state.continue_attempts} continuation" if state.continue_attempts else "")
        + (
            "; density passed"
            if state.density_passed
            else "; density FAILED"
            if state.sources and state.source_total > 0
            else ""
        )
        + ("; re-forged" if state.density_retried else "")
        + f" · finish_reason: {state.final_finish_reason!r}"
    )
    density_line = ""
    if state.sources and state.source_total > 0:
        density_line = (
            f"{len(document.encode('utf-8'))}B / {state.source_total}B ({state.density_ratio:.0%})"
        )
    annotation = _build_extended_annotation(
        state.model_def,
        sources=state.sources,
        processing_summary=processing_summary,
        density_line=density_line,
        okf_concept_path=okf_concept_path,
        truncated=state.was_truncated,
        final_finish_reason=state.final_finish_reason if state.was_truncated else None,
        continue_attempts=state.continue_attempts if state.was_truncated else 0,
    )
    if not annotation:
        return document
    return document.rstrip() + "\n\n" + annotation + "\n"


def _finalize_document(ctx: OperationExecutionContext, state: _ConsolidationState) -> str:
    """Phase 3 — repair fences, format, cap, and annotate the document."""
    # Inline-fence repair runs BEFORE formatting: mdformat treats a closing
    # fence appended to the last code line as legal CommonMark and preserves
    # it, so the hygiene split must land first.
    document = repair_inline_fences(state.document)
    document = _format_markdown(document)
    if len(document) > state.cfg.max_document_chars:
        document = document[: state.cfg.max_document_chars] + "\n\n<!-- truncated -->\n"
        state.was_truncated = True
    return _apply_annotation(state=state, document=document)


async def _outline_for_document(
    ctx: OperationExecutionContext,
    document: str,
) -> tuple[dict[str, Any], str | None]:
    """Validate the document's outline. Returns ``(outline, error)``."""
    try:
        outline = await _validate_outline(ctx, document)
    except RuntimeError as exc:
        return {}, str(exc)
    return outline, None


async def _run_lint_gate(
    ctx: OperationExecutionContext,
    *,
    state: _ConsolidationState,
    document: str,
    outline: dict[str, Any],
    heading_count: int,
) -> _LintOutcome:
    """Phase 4b — advisory lint gate with one optional repair pass.

    Lint is never a hard reject; the outline gate is the only hard gate.
    One repair pass when ``forge_lint_repair`` is set; residual issues are
    persisted as a non-rendering HTML comment so the user knows the
    artifact has warnings.
    """
    cfg = state.cfg
    issues = lint_markdown(document, rules_csv=cfg.lint_rules or None)
    repair_used = False
    if issues and cfg.lint_repair:
        repaired = await _lint_repair_pass(
            ctx,
            user_input=state.user_input,
            document=document,
            model_def=state.model_def,
            issues=issues,
            max_tokens=cfg.max_completion_tokens,
        )
        if repaired != document:
            repair_used = True
            # The repair pass regenerated the text, so re-run the fence
            # hygiene split before formatting (same order rationale as
            # the primary pass).
            document = repair_inline_fences(repaired)
            document = _format_markdown(document)
            new_outline, failure = await _outline_for_document(ctx, document)
            if failure is not None:
                return _LintOutcome(
                    document=document,
                    outline=outline,
                    heading_count=heading_count,
                    issues=issues,
                    repair_used=repair_used,
                    error=failure,
                )
            outline = new_outline
            heading_count = int(outline.get("heading_count", 0) or 0)
            if heading_count < _MIN_HEADINGS:
                text = _format_template(
                    _load_text(ctx, _NO_OUTLINE_PATH),
                    heading_count=heading_count,
                )
                return _LintOutcome(
                    document=document,
                    outline=outline,
                    heading_count=heading_count,
                    issues=issues,
                    repair_used=repair_used,
                    error=text,
                )
            issues = lint_markdown(document, rules_csv=cfg.lint_rules or None)
    if issues:
        comment = lint_summary_comment(issues)
        if comment:
            document = document.rstrip() + "\n\n" + comment + "\n"
    return _LintOutcome(
        document=document,
        outline=outline,
        heading_count=heading_count,
        issues=issues,
        repair_used=repair_used,
    )


async def _validate_and_lint(
    ctx: OperationExecutionContext,
    state: _ConsolidationState,
    document: str,
) -> _LintOutcome:
    """Phase 4 — outline gate (hard) followed by the advisory lint gate."""
    outline, failure = await _outline_for_document(ctx, document)
    if failure is not None:
        return _LintOutcome(
            document=document,
            outline={},
            heading_count=0,
            issues=[],
            repair_used=False,
            error=failure,
        )
    heading_count = int(outline.get("heading_count", 0) or 0)
    if heading_count < _MIN_HEADINGS:
        text = _format_template(
            _load_text(ctx, _NO_OUTLINE_PATH),
            heading_count=heading_count,
        )
        return _LintOutcome(
            document=document,
            outline=outline,
            heading_count=heading_count,
            issues=[],
            repair_used=False,
            error=text,
        )
    return await _run_lint_gate(
        ctx, state=state, document=document, outline=outline, heading_count=heading_count
    )


async def _write_consolidation_provenance(
    ctx: OperationExecutionContext,
    *,
    state: _ConsolidationState,
    stored_path: str,
    byte_count: int,
    heading_count: int,
) -> str | None:
    """Write the OKF forge-provenance concept when a bundle path is set."""
    if not (state.sources and str(state.params.get("okf_bundle_path") or "").strip()):
        return None
    return await _write_okf_provenance(
        ctx,
        okf_bundle_path=str(state.params["okf_bundle_path"]).strip(),
        document_stem=_document_stem(state.document_path),
        sources=state.sources,
        stored_path=stored_path,
        output_bytes=byte_count,
        heading_count=heading_count,
        model_def=state.model_def,
        passes=1 + (1 if state.density_retried else 0),
        continue_attempts=state.continue_attempts,
        density_ratio=state.density_ratio,
        density_floor=state.density_floor_value,
        density_passed=state.density_passed,
        first_finish_reason=state.first_finish_reason,
        final_finish_reason=state.final_finish_reason,
        truncated=state.was_truncated,
        per_source_meta=state.per_source_meta,
    )


def _build_consolidation_summary(
    *,
    state: _ConsolidationState,
    stored_path: str,
    byte_count: int,
    heading_count: int,
    okf_concept_path_written: str | None,
    lint_issues: list[LintIssue],
    lint_repair_used: bool,
) -> str:
    """Build the user-facing summary of the consolidation result."""
    summary = (
        f"Document consolidated and stored at `{stored_path}` "
        f"({byte_count} bytes, {heading_count} headings)."
    )
    if state.was_truncated:
        summary += (
            f" WARNING: persisted truncated after {state.continue_attempts} "
            f"continuation attempt(s) (final finish_reason="
            f"{state.final_finish_reason!r}). See annotation in document."
        )
    if state.sources and not state.density_passed:
        summary += (
            f" WARNING: density floor FAILED ({state.density_ratio:.0%} of "
            f"{state.source_total}B source; floor {state.density_floor_value:.0%}). "
            f"The persisted document may be a lossy abridgment — review it."
        )
    if state.density_failed_sources:
        summary += (
            f" WARNING: per-source density floor failed for "
            f"{len(state.density_failed_sources)} source(s): "
            f"{', '.join(state.density_failed_sources)}. See inline markers in "
            f"the persisted document."
        )
    if okf_concept_path_written:
        summary += f" Provenance: `{okf_concept_path_written}`"
    if lint_issues:
        summary += (
            f" Note: {len(lint_issues)} structural lint warning(s) remain "
            f"after {'a' if lint_repair_used else 'no'} repair pass; see lint comment in document."
        )
    return summary


async def _persist_consolidation(
    ctx: OperationExecutionContext,
    state: _ConsolidationState,
    outcome: _LintOutcome,
) -> OperationResult:
    """Phase 5 — persist the document, capture provenance, and report."""
    document = outcome.document
    try:
        write_response = await _persist_document(ctx, document, state.document_path)
    except RuntimeError as exc:
        text = _format_template(
            _load_text(ctx, _STORAGE_FAILED_PATH),
            error=str(exc),
        )
        return OperationResult(output_text=text)
    if not bool(write_response.get("ok", False)):
        text = _format_template(
            _load_text(ctx, _STORAGE_FAILED_PATH),
            error=write_response.get("raw", "unknown storage error"),
        )
        return OperationResult(output_text=text)
    stored_path = str(write_response.get("path") or state.document_path)
    byte_count = int(write_response.get("bytes") or len(document.encode("utf-8")))
    okf_concept_path_written = await _write_consolidation_provenance(
        ctx,
        state=state,
        stored_path=stored_path,
        byte_count=byte_count,
        heading_count=outcome.heading_count,
    )
    summary = _build_consolidation_summary(
        state=state,
        stored_path=stored_path,
        byte_count=byte_count,
        heading_count=outcome.heading_count,
        okf_concept_path_written=okf_concept_path_written,
        lint_issues=outcome.issues,
        lint_repair_used=outcome.repair_used,
    )
    return OperationResult(
        output_text=summary,
        output_json={
            "path": stored_path,
            "bytes": byte_count,
            "heading_count": outcome.heading_count,
            "outline": outcome.outline,
            "truncated": state.was_truncated,
            "continue_attempts": state.continue_attempts,
            "first_finish_reason": state.first_finish_reason,
            "final_finish_reason": state.final_finish_reason,
            "source_count": state.n_readable,
            "rolling_accumulation_used": state.rolling_used,
            "source_total_bytes": state.source_total,
            "density_ratio": state.density_ratio,
            "density_passed": state.density_passed,
            "density_retried": state.density_retried,
            "density_failed_sources": state.density_failed_sources,
            "okf_provenance_path": okf_concept_path_written,
            "lint_warnings": len(outcome.issues),
            "lint_repair_used": outcome.repair_used,
        },
        extra={
            "memory_context_lines": state.memory_context.count("\n") + 1
            if state.memory_context
            else 0,
            "document_path": stored_path,
            "truncated": state.was_truncated,
            "continue_attempts": state.continue_attempts,
            "source_count": state.n_readable,
            "density_passed": state.density_passed,
            "lint_warnings": len(outcome.issues),
        },
    )


async def execute(ctx: OperationExecutionContext) -> OperationResult:
    """Consolidate multiple source documents into one markdown document, preserving detail.

    Orchestrates the five pipeline phases — prepare (parse + read the
    sources), forge (map-reduce or single-pass plus the density floor),
    finalize (format, cap, annotate), validate (outline + advisory lint),
    and persist (write + OKF provenance). Each phase lives in its own
    helper; this function only wires them together.
    """
    state, failure = await _prepare_consolidation(ctx)
    if failure is not None:
        return failure
    failure = await _forge_consolidation(ctx, state)
    if failure is not None:
        return failure
    document = _finalize_document(ctx, state)
    outcome = await _validate_and_lint(ctx, state, document)
    if outcome.error is not None:
        return OperationResult(output_text=outcome.error)
    return await _persist_consolidation(ctx, state, outcome)
