"""Forge-document operation — memory-grounded markdown synthesis.

Pipeline:

1. Recall context from long-term memory via the ``memory.search`` MCP alias
   (routed by the host to the configured memory backend).
2. Call an LLM with an authored system prompt + the user input + recalled
   context to synthesize a markdown document.
3. Validate the synthesized document's structure by calling the
   ``markdown_generate`` tool (backed by ``markdown-it-py``). The
   outline must contain at least one heading.
4. Persist the document to the ``lisa-storage`` MCP server's ``write_file``
   tool under a caller-supplied relative path (default ``docs/``).

The operation never touches the filesystem directly — all reads (memory)
and writes (storage) go through ``ctx.execute_tool_fn``. Authored prompts
are loaded via ``ctx.read_authored_text_fn``; per the pack-content rule,
a missing/empty loader raises rather than silently falling back.
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

from lisa_packs.markdown.operations.forge_document._source_paths import (
    SourceSpec,
    expand_globs,
    failed_specs,
    has_sources,
    parse_source_paths,
    read_sources,
    readable_count,
    render_source_block,
    total_source_bytes,
)

logger = logging.getLogger(__name__)

__all__ = ["Parameters", "execute"]

# Tool ids — declared in the operation YAML ``tool_ids`` and routed through
# ``ctx.execute_tool_fn``. ``markdown_generate`` is a pack-contributed
# in-process tool; ``write_file`` and ``memory.search`` are MCP tools served
# by ``lisa-storage`` and the memory backend respectively.
_OUTLINE_TOOL_ID = "markdown_generate"
_WRITE_TOOL_ID = "write_file"
_READ_TOOL_ID = "read_file"
_MEMORY_SEARCH_TOOL_ID = "memory.search"

# Tool ids used by the source-paths consolidation path (added 2026-07-30).
# ``list_files`` and ``read_file`` are served by the lisa-storage MCP server
# (same backend as _READ_TOOL_ID/_WRITE_TOOL_ID). ``okf_emit_concept`` and
# ``okf_log`` are served by the lisa-pack-okf pack's default-available
# operations, consumed here as tools via ctx.execute_tool_fn — the same
# pattern used for memory.search. See the consolidation design discussion.
_OKF_EMIT_TOOL_ID = "okf_emit_concept"
_OKF_LOG_TOOL_ID = "okf_log"

# User-instruction phrases that opt out of model attribution.
_NO_ANNOTATE_PHRASES = (
    "don't annotate the model",
    "do not annotate the model",
    "no model annotation",
    "no model attribution",
    "without model annotation",
)

# Authored-text paths — loaded via ctx.read_authored_text_fn (SDK loader).
_PROMPT_DIR = "operations/forge_document/prompts"
_SYSTEM_PROMPT_PATH = f"{_PROMPT_DIR}/forge_document_system.md"
_USER_PROMPT_PATH = f"{_PROMPT_DIR}/forge_document_user.md"
_USAGE_PATH = f"{_PROMPT_DIR}/forge_document_usage.md"
_NO_INPUT_PATH = f"{_PROMPT_DIR}/forge_document_no_input.md"
_EMPTY_RESPONSE_PATH = f"{_PROMPT_DIR}/forge_document_empty_response.md"
_EMPTY_DOCUMENT_PATH = f"{_PROMPT_DIR}/forge_document_empty_document.md"
_NO_OUTLINE_PATH = f"{_PROMPT_DIR}/forge_document_no_outline.md"
_STORAGE_FAILED_PATH = f"{_PROMPT_DIR}/forge_document_storage_failed.md"
_CONTINUE_PROMPT_PATH = f"{_PROMPT_DIR}/forge_document_continue.md"
# Source-aware consolidation prompts (added 2026-07-30). When the caller
# supplies ``source_paths``, the operation swaps to these prompts so the
# model receives the per-source priority notes and the "preserve detail,
# do not summarize" directive. The single-doc/memory-only path keeps the
# original prompts unchanged — no regression for existing callers.
_SYSTEM_PROMPT_SOURCES_PATH = f"{_PROMPT_DIR}/forge_document_system_sources.md"
_USER_PROMPT_SOURCES_PATH = f"{_PROMPT_DIR}/forge_document_user_sources.md"
_DENSITY_RETRY_PATH = f"{_PROMPT_DIR}/forge_document_density_retry.md"

# Fallback defaults for the operation's configurable budgets. The
# authoritative source is the session manifest's
# ``resources.packs.configuration.markdown.<key>`` block, surfaced to the
# op at runtime as ``ctx.pack_config`` (an immutable mapping loaded by
# ``kernel/pack_configuration.SessionPackConfigurationResolver`` before
# ``execute`` runs). These constants are only the last-resort fallback when
# no manifest config is supplied so the op still works in a bare host.
#
# Manifest keys (all under ``resources.packs.configuration.markdown``):
#   forge_max_completion_tokens, forge_max_continue_tokens,
#   forge_max_continue_attempts, forge_max_recall_records,
#   forge_max_document_chars, forge_max_document_chars_with_sources,
#   forge_max_query_chars.
#
# The YAML ``completion_max_tokens`` mirrors ``forge_max_completion_tokens``
# but is documentary-only for runtime ops (the kernel does not inject it
# into ``ctx`` for ``execution_kind: runtime``). See
# /memories/repo/operation-tunables-placement.md and
# /memories/repo/forge-document-truncation-contract.md.
_MAX_COMPLETION_TOKENS = 50_000
# Continuation pass uses the same cap. The continuation appends only the
# missing tail, so the same budget is sufficient for nearly all cases.
_MAX_CONTINUE_TOKENS = _MAX_COMPLETION_TOKENS
# Hard ceiling on continuation attempts. One retry covers the
# "model ran out of tokens one section early" case; two covers the
# rare "still truncated, one more chunk needed" case. Beyond this we
# give up and annotate the document as truncated rather than looping.
_MAX_CONTINUE_ATTEMPTS = 4
_MAX_RECALL_RECORDS = 25
# Hard ceiling on the persisted document. The original 20K cap (from the
# single-doc/memory era) forced summarization of anything larger and was a
# root cause of the 2026-07-30 "summarizing instead of writing" incident:
# a 130KB consolidation could not physically exceed ~20K. Source-aware
# consolidations raise the cap so the preservation directive is
# enforceable. The map-reduce path (future work) will handle the case
# where even this cap is too small for one pass.
_MAX_DOCUMENT_CHARS = 500_000
_MAX_DOCUMENT_CHARS_WITH_SOURCES = 1_500_000
_MAX_QUERY_CHARS = 20_000

# Density floor (consolidation anti-abridgment gate, added 2026-07-30).
# When the caller supplied source_paths, the output must preserve enough of
# the source substance to not be a lossy summary. The floor is a ratio of
# output bytes to source bytes, scaled by source count: more sources allow
# a lower ratio (legitimate de-duping), but a sub-floor output is presumed
# to be summarizing and triggers one re-forge pass. The thresholds are
# conservative starting points, tunable per-operation.
_DENSITY_FLOOR_MANY_SOURCES_RATIO = 0.15   # 5+ readable sources
_DENSITY_FLOOR_FEW_SOURCES_RATIO = 0.30    # 2-4 readable sources
_DENSITY_FLOOR_MANY_SOURCES_THRESHOLD = 5
# Allow one retry when the floor trips. The retry uses a sharpened prompt
# that names the deficit explicitly. If the retry still trips, we persist
# with the truncated annotation (honest failure) rather than looping.
_DENSITY_RETRY_ATTEMPTS = 1

# Minimum structural bar for a forged document: at least one heading.
_MIN_HEADINGS = 1

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
    """Parameter surface for callers invoking the operation as a tool.

    The operation reads ``ctx.params`` directly (it is not registered as a
    model-facing tool by default), but exposing the shape documents the
    accepted inputs.
    """

    document_path: str = Field(
        default="docs/forge.md",
        description="Relative path within the lisa-storage files sandbox where the forged document is persisted.",
    )
    query: str = Field(
        default="",
        description="Natural-language instruction for the document. Falls back to ctx.query when empty.",
    )
    existing_document: str = Field(
        default="",
        description="Current document text to revise. When non-empty, the operation "
        "reads it from storage via read_file and passes it to the LLM as the "
        "starting point. Omit for a new document.",
    )
    limit: int = Field(
        default=8,
        ge=1,
        le=20,
        description="Maximum number of recalled memory records to inject as context.",
    )
    annotate_model: bool = Field(
        default=True,
        description="When True, append a non-rendering HTML comment to the forged "
        "document noting the provider/model that wrote it. The user can also "
        "opt out by including phrases like 'don't annotate the model' in the "
        "instruction. Set False to suppress unconditionally.",
    )
    source_paths: list[Any] = Field(
        default_factory=list,
        description="Ordered list of source documents to consolidate. Each entry is "
        "either a path/glob string (default priority 'normal') or a dict with "
        "'path' (required) and 'priority' (optional NL prose, e.g. 'primary — "
        "keep section structure', 'secondary — dedupe against source 1', "
        "'background — one paragraph max'). Globs (paths with * or ?) are "
        "expanded via the list_files storage tool. When supplied, the "
        "operation switches to consolidation mode: it reads every source, "
        "injects them with their per-source priority notes, and instructs "
        "the model to PRESERVE DETAIL rather than summarize. A density floor "
        "validates the output is not a lossy abridgment. Provenance is "
        "captured to an OKF concept (see okf_bundle_path).",
    )
    okf_bundle_path: str = Field(
        default="okf/forge-provenance",
        description="Relative path of the OKF bundle directory where the forge "
        "provenance concept is written (only when source_paths is supplied). "
        "Set to empty to skip OKF provenance capture. The concept uses OKF "
        "type 'forge-provenance' and records the model, sources, processing "
        "passes, and density check result.",
    )


def _extract_document_path(params: dict[str, Any]) -> str:
    """Return the target storage path, sanitized to a safe relative form."""
    raw = str(params.get("document_path") or "").strip()
    if not raw:
        return "docs/forge.md"
    # sees them; the storage server re-validates, but we fail early with a
    # clearer error.
    normalized = raw.replace("\\", "/").lstrip("/")
    if ".." in normalized.split("/"):
        raise RuntimeError("document_path may not contain '..' segments")
    return normalized


@dataclass(frozen=True)
class _ResolvedSettings:
    """Resolved operation budgets: ``ctx.pack_config`` overrides module fallbacks.

    The authoritative source is the session manifest's
    ``resources.packs.configuration.markdown.<key>`` block, surfaced at
    runtime as ``ctx.pack_config`` (an immutable MappingProxyType populated
    by ``kernel/operations/registry._resolve_execution_configuration`` before
    ``execute`` runs). The module ``_MAX_*`` constants are last-resort
    fallbacks used only when the manifest supplies no value, so the op still
    works in a bare host with no manifest config.

    See /memories/repo/operation-tunables-placement.md.
    """

    max_completion_tokens: int
    max_continue_tokens: int
    max_continue_attempts: int
    max_recall_records: int
    max_document_chars: int
    max_document_chars_with_sources: int
    max_query_chars: int


def _cfg_int(pack_config: Any, key: str, default: int) -> int:
    """Read an int from ``ctx.pack_config`` with a module-constant fallback.

    Coerces non-int values to the default rather than raising, since pack
    config is session-authored and a malformed value should not crash the
    op mid-turn (the fallback is the safe, known-good default). The mapping
    is immutable (MappingProxyType), so we only read from it.
    """
    if pack_config is None:
        return default
    raw = pack_config.get(key)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except (TypeError, ValueError):
        logger.warning("forge_document: pack_config[%r]=%r is not an int; using default %d", key, raw, default)
        return default


def _resolve_settings(ctx: OperationExecutionContext) -> _ResolvedSettings:
    """Resolve all operation budgets from ``ctx.pack_config`` with fallbacks."""
    pc = ctx.pack_config
    return _ResolvedSettings(
        max_completion_tokens=_cfg_int(pc, "forge_max_completion_tokens", _MAX_COMPLETION_TOKENS),
        max_continue_tokens=_cfg_int(pc, "forge_max_continue_tokens", _MAX_CONTINUE_TOKENS),
        max_continue_attempts=_cfg_int(pc, "forge_max_continue_attempts", _MAX_CONTINUE_ATTEMPTS),
        max_recall_records=_cfg_int(pc, "forge_max_recall_records", _MAX_RECALL_RECORDS),
        max_document_chars=_cfg_int(pc, "forge_max_document_chars", _MAX_DOCUMENT_CHARS),
        max_document_chars_with_sources=_cfg_int(pc, "forge_max_document_chars_with_sources", _MAX_DOCUMENT_CHARS_WITH_SOURCES),
        max_query_chars=_cfg_int(pc, "forge_max_query_chars", _MAX_QUERY_CHARS),
    )


def _document_stem(document_path: str) -> str:
    """Return the filename stem (no extension) of a storage path.

    Used to derive the OKF provenance concept filename from the forged
    document path, e.g. ``docs/narrative.md`` → ``narrative``.
    """
    normalized = document_path.replace("\\", "/").rstrip("/")
    base = normalized.rsplit("/", 1)[-1]
    if "." in base:
        base = base.rsplit(".", 1)[0]
    return base or "forge"


def _extract_query(ctx: OperationExecutionContext, params: dict[str, Any], max_query_chars: int) -> str:
    """Return the user-supplied instruction, bounded in length."""
    raw = str(params.get("query") or ctx.query or "").strip()
    if len(raw) > max_query_chars:
        raw = raw[:max_query_chars] + "…"
    return raw


async def _read_existing_document(
    ctx: OperationExecutionContext,
    path: str,
) -> str:
    """Read the current document from lisa-storage via the ``read_file`` tool.

    Returns the document text, or an empty string when the document does not
    exist yet (a normal condition for a first forge). Raises ``RuntimeError``
    on any other tool failure.
    """
    tool_result = await ctx.execute_tool_fn(
        _READ_TOOL_ID,
        ToolExecutionContext(
            tool_id=_READ_TOOL_ID,
            workspace_id=ctx.workspace_id,
            user_id=ctx.user_id,
            user_slug=ctx.user_slug,
            surface_type=ctx.surface_type,
            surface_id=ctx.surface_id,
            memory_scope=ctx.memory_scope,
            mode_id=ctx.mode_id,
            parameters={"path": path},
        ),
    )
    error_text = getattr(tool_result, "error", "") or ""
    if error_text:
        raise RuntimeError(f"{_READ_TOOL_ID} failed: {error_text}")
    output_text = getattr(tool_result, "output_text", "") or ""
    # The storage server returns the raw file text on success, or a JSON
    # error object on failure. A JSON error with "not found" is a normal
    # first-forge condition — return "".
    try:
        data = json.loads(output_text)
    except (json.JSONDecodeError, TypeError):
        return output_text.strip()
    if isinstance(data, dict) and "error" in data:
        if "not found" in str(data["error"]):
            return ""
        raise RuntimeError(f"{_READ_TOOL_ID} failed: {data['error']}")
    return output_text.strip()


async def _recall_memory(
    ctx: OperationExecutionContext,
    query: str,
    *,
    limit: int,
) -> str:
    """Recall relevant memories via the ``memory.search`` MCP alias.

    Returns a newline-formatted context block (possibly empty). Raises
    ``RuntimeError`` only when the tool executor itself is missing.
    """
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
    # The memory backend returns newline-formatted text (or a JSON list).
    # If it's JSON, flatten to bullets; otherwise pass through.
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
    existing_document: str = "",
    source_block: str = "",
    *,
    max_tokens: int,
) -> tuple[str, Any, str | None]:
    """Call the host's completion function with the forging prompt.

    Returns a ``(content, model_def, finish_reason)`` triple so the caller
    can annotate the document with the resolved model and detect
    mid-document truncation (``finish_reason in {"length", "unknown", None}``).

    When ``source_block`` is non-empty, the operation swaps to the
    source-aware consolidation prompts (preserve-detail directive +
    per-source priority notes). The single-doc/memory-only path keeps
    the original prompts.
    """
    if ctx.complete_fn is None or ctx.resolve_model_fn is None:
        raise RuntimeError("markdown requires complete_fn and resolve_model_fn")

    preferred_model_id = ctx.preferred_model_id or ctx.model_id or "default"
    model_def = ctx.resolve_model_fn(preferred_model_id)
    if model_def is None:
        raise RuntimeError(
            f"markdown could not resolve model '{preferred_model_id}'"
        )

    if source_block:
        system_prompt = _load_text(ctx, _SYSTEM_PROMPT_SOURCES_PATH)
        user_template = _load_text(ctx, _USER_PROMPT_SOURCES_PATH)
    else:
        system_prompt = _load_text(ctx, _SYSTEM_PROMPT_PATH)
        user_template = _load_text(ctx, _USER_PROMPT_PATH)
    user_prompt = _format_template(
        user_template,
        user_input=user_input,
        memory_context=memory_context or "(no recalled context)",
        existing_document=existing_document or "(no existing document — start fresh)",
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
    finish_reason = _coerce_finish_reason(getattr(completion, "finish_reason", None))
    return content.strip(), model_def, finish_reason


def _coerce_finish_reason(raw: Any) -> str | None:
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


def _is_truncated(finish_reason: str | None) -> bool:
    """Return True when the finish reason signals mid-stream truncation."""
    normalized = finish_reason if finish_reason is not None else None
    return normalized in _TRUNCATING_FINISH_REASONS


async def _continue_llm(
    ctx: OperationExecutionContext,
    user_input: str,
    document_so_far: str,
    model_def: Any,
    *,
    max_tokens: int,
) -> tuple[str, str | None]:
    """Issue one continuation pass to finish a truncated document.

    Appends a continuation instruction to the original message list (so the
    model has full context) and asks for ONLY the remaining tail. Returns
    ``(continuation_content, finish_reason)`` where ``continuation_content``
    is the raw model output (caller extracts the fenced markdown block).
    """
    if ctx.complete_fn is None:
        raise RuntimeError("markdown requires complete_fn to continue a truncated document")

    continue_template = _load_text(ctx, _CONTINUE_PROMPT_PATH)
    continue_prompt = _format_template(
        continue_template,
        user_input=user_input,
        document_so_far=document_so_far,
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
    finish_reason = _coerce_finish_reason(getattr(completion, "finish_reason", None))
    return content.strip(), finish_reason


def _looks_complete(document: str) -> bool:
    """Heuristic: does the document end on a clean paragraph/list boundary?

    Used after a continuation pass to decide whether the merged document is
    plausibly finished or whether we should try another continuation. A
    document ending mid-word, mid-sentence, or with a dangling list marker
    is treated as still-truncated.
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
    # Ends with sentence-final punctuation, a heading, or a code fence.
    if last_line.endswith(("```", ".", "!", "?", ":", ";", ">", ")")):
        return True
    # Ends with a list item that has content after the marker.
    if re.match(r"^\s*[\-\*\+] +\S", last_line) or re.match(r"^\s*\d+\. +\S", last_line):
        return True
    # Ends with a heading line.
    if last_line.lstrip().startswith("#"):
        return True
    # Otherwise: mid-sentence — treat as truncated.
    return False


def _merge_continuation(document_so_far: str, tail: str) -> str:
    """Concatenate the document-so-far with a continuation tail.

    The continuation prompt asks for ONLY the remaining body, but local
    models sometimes re-emit a leading heading or blank line. This helper
    collapses the join boundary to exactly one blank line and strips any
    leading blank lines from the tail so the merge does not produce
    doubled paragraph breaks.
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

    Some local models, when asked to "continue", repeat the last paragraph or
    the entire document-so-far instead of appending new content. Detecting
    this prevents the merge from producing a doubled or looped document. The
    check is conservative: it only flags tails whose non-empty lines are a
    contiguous suffix of the existing document.
    """
    head_lines = [ln for ln in document_so_far.splitlines() if ln.strip()]
    tail_lines = [ln for ln in tail.splitlines() if ln.strip()]
    if not tail_lines:
        return True
    if len(tail_lines) > len(head_lines):
        # The continuation is longer than the entire document-so-far — it
        # must contain new content, so not a pure duplicate.
        return False
    # Compare the last len(tail_lines) lines of the head to the tail.
    suffix = head_lines[-len(tail_lines):]
    return suffix == tail_lines


def _extract_markdown_block(text: str) -> str:
    """Extract the first fenced ```markdown ...``` block, else return text.

    Accepts fenced blocks tagged ``markdown`` or untagged. Falls back to the
    raw text when no fence is present so the outline tool can still assess it.
    """
    match = re.search(r"```(?:markdown|md)?\s*\n(.*?)```", text, flags=re.DOTALL)
    if match:
        return match.group(1).strip()
    return text.strip()


def _format_markdown(document: str) -> str:
    """Lint/format the document via ``mdformat`` if available.

    ``mdformat`` is the canonical CommonMark formatter and is the natural
    companion to ``markdown-it-py`` (which the pack already depends on for
    outline validation). When installed it normalizes bullet markers,
    collapses redundant blank lines, fixes heading spacing, and guarantees a
    single trailing newline — producing clean, consistent output. When not
    installed the document passes through unchanged so the pack remains
    importable without the optional dependency.
    """
    try:
        import mdformat  # type: ignore[import-not-found]
    except ImportError:
        logger.debug("mdformat not installed; skipping markdown formatting")
        return document
    try:
        return mdformat.text(document, options={"number": True}).rstrip() + "\n"
    except Exception as exc:  # pylint: disable=broad-except
        logger.warning("forge_document: mdformat failed (%s); using unformatted document", exc)
        return document


def _build_model_annotation(
    model_def: Any,
    *,
    truncated: bool = False,
    final_finish_reason: str | None = None,
    continue_attempts: int = 0,
) -> str:
    """Build a trailing HTML-comment attribution line for the resolved model.

    The comment is non-rendering in every CommonMark renderer, so it annotates
    the source without polluting the visible document.

    When ``truncated`` is True, an explicit ``truncated by model`` marker is
    appended so a reader scanning the artifact knows the document is
    incomplete even though the operation persisted it. The final
    ``finish_reason`` and the number of continuation attempts are included
    for triage.
    """
    provider = str(getattr(model_def, "provider", "") or "").strip()
    model_id = str(getattr(model_def, "model_id", "") or "").strip()
    parts: list[str] = []
    if provider:
        parts.append(f"provider: {provider}")
    if model_id:
        parts.append(f"model: {model_id}")
    if truncated:
        reason = final_finish_reason or "unknown"
        parts.append(f"truncated by model · finish_reason: {reason}")
        if continue_attempts:
            parts.append(f"continue_attempts: {continue_attempts}")
    if not parts:
        return ""
    timestamp = datetime.now(UTC).strftime("%Y-%m-%d")
    return f"<!-- generated by lisa · {' · '.join(parts)} · {timestamp} -->"


def _user_asked_not_to_annotate(user_input: str) -> bool:
    """Return True when the user instruction explicitly opts out of attribution."""
    lowered = user_input.lower()
    return any(phrase in lowered for phrase in _NO_ANNOTATE_PHRASES)


async def _validate_outline(ctx: OperationExecutionContext, document: str) -> dict[str, Any]:
    """Run the ``markdown_generate`` tool and return its outline dict.

    Raises ``RuntimeError`` if the tool returns an error.
    """
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
        # Try to parse output_text as JSON for backends that only return text.
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
    """Write the document to lisa-storage via the ``write_file`` tool.

    Returns the parsed JSON response (with ``ok``/``path``/``bytes``).
    Raises ``RuntimeError`` on any tool failure.
    """
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


async def _continue_until_complete(
    ctx: OperationExecutionContext,
    *,
    user_input: str,
    document: str,
    model_def: Any,
    first_finish_reason: str | None,
    max_continue_attempts: int,
    max_continue_tokens: int,
) -> tuple[str, bool, int, str | None]:
    """Run continuation passes until the document looks complete or budget is hit.

    Returns ``(document, was_truncated, attempts, final_finish_reason)``.
    The caller annotates and persists ``document``; the other three values
    feed the truncated-annotation marker and the triage fields in
    ``OperationResult.output_json``.

    The contract is documented in
    /memories/repo/forge-document-truncation-contract.md:
    - Only truncate-flagged first passes (``length``/``unknown``/``None``)
      trigger continuation; clean ``stop`` returns the document as-is.
    - A continuation that returns empty or duplicates the existing tail is
      treated as "model has nothing more to add" — stop and mark truncated.
    - A merged document that ends on a clean boundary (``_looks_complete``)
      AND whose continuation reported a non-truncating finish_reason is
      considered complete; otherwise we keep going up to the budget.
    """
    if not _is_truncated(first_finish_reason):
        return document, False, 0, first_finish_reason

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
        )
        final_finish_reason = cont_finish_reason
        if not continuation_text or not continuation_text.strip():
            logger.warning(
                "forge_document: continuation pass %d returned empty; giving up",
                attempt,
            )
            break
        tail = _extract_markdown_block(continuation_text)
        # If the fenced block extracted was empty but the raw continuation
        # had content, the model emitted prose outside the fence (or an
        # empty fence). Either way, there is no usable markdown tail to
        # append — treat this as "continuation had nothing to add" and
        # stop. Persisting prose-as-tail would corrupt the document.
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
        if _looks_complete(current) and not _is_truncated(cont_finish_reason):
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


def _density_floor_ratio(source_count: int) -> float:
    """Return the minimum output/source byte ratio that is NOT a lossy abridgment.

    Scaled by source count: more sources allow a lower ratio (legitimate
    de-duping across many sources), but a sub-floor output is presumed to
    be summarizing. See _DENSITY_FLOOR_* constants for the thresholds.
    """
    if source_count >= _DENSITY_FLOOR_MANY_SOURCES_THRESHOLD:
        return _DENSITY_FLOOR_MANY_SOURCES_RATIO
    if source_count >= 2:
        return _DENSITY_FLOOR_FEW_SOURCES_RATIO
    # Single source or zero: no floor (the single-doc path already has the
    # revise-vs-replace contract; the floor only catches consolidation
    # abridgment, which requires multiple sources to be meaningful).
    return 0.0


def _density_floor_check(
    *,
    source_total_bytes: int,
    output_bytes: int,
    source_count: int,
) -> str | None:
    """Return a violation message if the output looks like a lossy abridgment.

    Returns ``None`` when the output passes the floor (or when no floor
    applies — zero/single source, or zero source bytes). The message is
    used both as a log signal and as input to the density-retry prompt
    so the re-forge pass can name the deficit explicitly.
    """
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


async def _reforge_for_density(
    ctx: OperationExecutionContext,
    *,
    user_input: str,
    memory_context: str,
    existing_document: str,
    source_block: str,
    model_def: Any,
    violation: str,
    source_total_bytes: int,
    output_bytes: int,
    source_count: int,
    max_tokens: int,
) -> tuple[str, str | None]:
    """Issue one density-retry forge pass with a sharpened instruction.

    Returns ``(content, finish_reason)``. The retry re-sends the full
    source-aware user prompt (so the model has the sources + priorities
    in context) followed by the density-retry instruction naming the
    deficit. The caller extracts the fenced markdown block and re-runs
    the density check; if it still fails, the caller persists with the
    truncated annotation (honest failure) rather than looping.
    """
    if ctx.complete_fn is None:
        return "", None
    system_prompt = _load_text(ctx, _SYSTEM_PROMPT_SOURCES_PATH)
    user_template = _load_text(ctx, _USER_PROMPT_SOURCES_PATH)
    base_user_prompt = _format_template(
        user_template,
        user_input=user_input,
        memory_context=memory_context or "(no recalled context)",
        existing_document=existing_document or "(no existing document — start fresh)",
        source_block=source_block,
    )
    retry_template = _load_text(ctx, _DENSITY_RETRY_PATH)
    min_ratio = _density_floor_ratio(source_count)
    retry_instruction = _format_template(
        retry_template,
        output_bytes=output_bytes,
        source_total_bytes=source_total_bytes,
        ratio=(output_bytes / source_total_bytes) if source_total_bytes else 0.0,
        source_count=source_count,
        min_ratio=min_ratio,
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": base_user_prompt},
        {"role": "user", "content": retry_instruction},
    ]
    logger.warning("forge_document: density floor violated — %s", violation)
    completion = await ctx.complete_fn(
        messages=messages,
        model_def=model_def,
        max_tokens=max_tokens,
        temperature=0.2,
        tools=None,
        response_format=None,
    )
    content = getattr(completion, "content", "") or ""
    finish_reason = _coerce_finish_reason(getattr(completion, "finish_reason", None))
    return content.strip(), finish_reason


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
    """Build the extended non-rendering provenance footer for a source-aware forge.

    The footer is a single HTML comment (non-rendering in every CommonMark
    renderer) with four logical lines:
      1. model attribution (provider · model · date)
      2. sources summary (count + the priority-tier breakdown, not all paths)
      3. processing summary (passes, finish_reason)
      4. OKF provenance concept pointer (the full per-source record lives there)

    The footer is the index card; the OKF concept is the filing cabinet.
    Keeping it one comment block means the visible document stays clean
    while the audit trail travels in-source.
    """
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
    # Summarize the priority tiers compactly: group paths by priority keyword.
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
    """Render a compact 'primary: a.md; secondary: b.md, c.md; background: d.md' summary.

    Groups readable sources by the leading word of their priority note
    (case-insensitive), preserving the order tiers first appear. Falls back
    to listing paths ungrouped if priorities are uniform or unparseable.
    """
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
        # All one tier — no point labeling; just list the paths (truncated).
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
) -> str:
    """Render the markdown body for the OKF forge-provenance concept.

    This is the machine-legible full record: the per-source table (path,
    priority, bytes, and a treatment column noting truncation/read errors),
    the processing pass log, and the density check result. The plain
    markdown footer carries only a summary; this is the complete audit
    trail an OKF consumer (or a later Lisa turn) can validate/diff.
    """
    lines: list[str] = ["# Forge Provenance", "", "## Sources", ""]
    readable = [s for s in sources if s.text]
    failed = failed_specs(sources)
    if readable or failed:
        lines.append("| # | Path | Priority | Bytes | Treatment |")
        lines.append("| --- | --- | --- | --- | --- |")
        for idx, spec in enumerate(readable, start=1):
            treatment = "preserved"
            if spec.truncated_to_fit:
                treatment = "truncated to per-source cap"
            if spec.expanded_from_glob:
                treatment += f"; glob-expanded from `{spec.origin_glob}`"
            lines.append(
                f"| {idx} | `{spec.path}` | {spec.priority} | {spec.bytes} | {treatment} |"
            )
        for spec in failed:
            lines.append(
                f"| – | `{spec.path}` | {spec.priority} | 0 | UNREADABLE: {spec.read_error} |"
            )
    else:
        lines.append("_(no sources supplied — single-doc/memory-only forge)_")
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
) -> str | None:
    """Write the OKF forge-provenance concept + append a log entry.

    Returns the sandbox-relative path of the written concept, or ``None``
    when the OKF tools are unavailable or the caller opted out (empty
    ``okf_bundle_path``). Failures to write provenance are non-fatal — the
    forge itself already succeeded and persisted the document; provenance
    is best-effort enrichment, not a gate. A warning is logged.
    """
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
            logger.warning("forge_document: OKF emit_concept failed: %s", emit_text)
            return None
    except Exception as exc:  # pylint: disable=broad-except
        logger.warning("forge_document: OKF emit_concept raised: %s", exc)
        return None
    full_path = f"{okf_bundle_path}/{concept_rel}"
    # Best-effort log entry (okf_log). Failure here is non-fatal.
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
        logger.debug("forge_document: OKF log append failed (non-fatal): %s", exc)
    return full_path


async def execute(ctx: OperationExecutionContext) -> OperationResult:
    """Forge a markdown document from user input + recalled memory and persist it."""
    params = ctx.params or {}
    cfg = _resolve_settings(ctx)
    user_input = _extract_query(ctx, params, cfg.max_query_chars)
    if not user_input:
        return OperationResult(output_text=_load_text(ctx, _NO_INPUT_PATH))

    try:
        document_path = _extract_document_path(params)
    except RuntimeError as exc:
        return OperationResult(output_text=str(exc))

    if not callable(ctx.execute_tool_fn):
        return OperationResult(output_text=_load_text(ctx, _USAGE_PATH))

    # ── Source-aware consolidation path (added 2026-07-30) ─────────────
    # When the caller supplies ``source_paths``, the operation switches to
    # consolidation mode: it reads every source (expanding globs), injects
    # them with per-source NL priority notes, uses the preserve-detail
    # system prompt, applies a density floor against lossy abridgment, and
    # captures provenance to an OKF concept. The single-doc/memory-only
    # path (no source_paths) is unchanged — no regression for existing
    # callers.
    sources: list[SourceSpec] = []
    source_block = ""
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

    recall_limit = int(params.get("limit") or cfg.max_recall_records)
    memory_context = await _recall_memory(ctx, user_input, limit=recall_limit)

    # Iteration: when the caller supplies an existing document, use it. When
    # the caller omits it but the target path already exists in storage,
    # read it so the LLM revises rather than replacing wholesale.
    existing_document = str(params.get("existing_document") or "").strip()
    if not existing_document:
        try:
            existing_document = await _read_existing_document(ctx, document_path)
        except RuntimeError as exc:
            return OperationResult(output_text=str(exc))

    completion = await _call_llm(
        ctx,
        user_input,
        memory_context,
        existing_document=existing_document,
        source_block=source_block,
        max_tokens=cfg.max_completion_tokens,
    )
    # _call_llm returns (content, model_def, finish_reason).
    completion_text, model_def, first_finish_reason = completion
    if not completion_text or not completion_text.strip():
        return OperationResult(output_text=_load_text(ctx, _EMPTY_RESPONSE_PATH))

    document = _extract_markdown_block(completion_text)
    if not document.strip():
        text = _format_template(
            _load_text(ctx, _EMPTY_DOCUMENT_PATH),
            response_preview=completion_text[:400],
        )
        return OperationResult(output_text=text)

    # Truncation handling: when the first pass stopped mid-document
    # (finish_reason in {length, unknown, None}), issue up to
    # _MAX_CONTINUE_ATTEMPTS continuation passes. Each pass appends only the
    # missing tail. If the merged document still looks truncated after the
    # budget is exhausted, mark it truncated and annotate the persisted
    # artifact so a reader knows it is incomplete. See
    # /memories/repo/forge-document-truncation-contract.md.
    document, was_truncated, continue_attempts, final_finish_reason = (
        await _continue_until_complete(
            ctx,
            user_input=user_input,
            document=document,
            model_def=model_def,
            first_finish_reason=first_finish_reason,
            max_continue_attempts=cfg.max_continue_attempts,
            max_continue_tokens=cfg.max_continue_tokens,
        )
    )

    # ── Density floor (consolidation anti-abridgment gate) ────────────
    # Only applies when source_paths was supplied. If the output is a
    # lossy abridgment (output bytes << source bytes, below the
    # source-count-scaled floor), issue one re-forge pass with a
    # sharpened "preserve detail" instruction. If the retry still trips,
    # we proceed and mark the artifact (honest failure) rather than
    # looping indefinitely.
    density_passed = True
    density_violation: str | None = None
    density_ratio = 0.0
    density_floor_value = 0.0
    source_total = total_source_bytes(sources)
    n_readable = readable_count(sources)
    density_retried = False
    if sources and source_total > 0:
        density_floor_value = _density_floor_ratio(n_readable)
        # Run the check before mdformat normalizes the byte count.
        density_violation = _density_floor_check(
            source_total_bytes=source_total,
            output_bytes=len(document.encode("utf-8")),
            source_count=n_readable,
        )
        if density_violation is not None:
            density_passed = False
            violation_msg = density_violation  # narrowed to str for the retry call
            for _ in range(_DENSITY_RETRY_ATTEMPTS):
                density_retried = True
                retry_text, retry_finish = await _reforge_for_density(
                    ctx,
                    user_input=user_input,
                    memory_context=memory_context,
                    existing_document=existing_document,
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
                retry_document = _extract_markdown_block(retry_text)
                if not retry_document.strip():
                    break
                # The retry may itself need continuation handling.
                retry_document, was_truncated, continue_attempts, final_finish_reason = (
                    await _continue_until_complete(
                        ctx,
                        user_input=user_input,
                        document=retry_document,
                        model_def=model_def,
                        first_finish_reason=retry_finish,
                        max_continue_attempts=cfg.max_continue_attempts,
                        max_continue_tokens=cfg.max_continue_tokens,
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

    # Lint/format the document so the persisted output is clean and consistent.
    document = _format_markdown(document)

    # Char cap: source-aware consolidations get the raised cap so the
    # preservation directive is physically enforceable. The single-doc path
    # keeps the original 20K cap. Both are session-configurable via
    # ``forge_max_document_chars`` / ``forge_max_document_chars_with_sources``.
    max_chars = (
        cfg.max_document_chars_with_sources if sources else cfg.max_document_chars
    )
    if len(document) > max_chars:
        document = document[:max_chars] + "\n\n<!-- truncated -->\n"
        # A char-budget truncation overrides the model-finish signal: the
        # artifact is definitely truncated.
        was_truncated = True

    # Annotate with the model that wrote it, unless the user opted out.
    annotate_model = bool(params.get("annotate_model", True))
    if annotate_model and not _user_asked_not_to_annotate(user_input):
        if sources:
            # Extended provenance footer for source-aware forges. The OKF
            # concept (written below, after persistence) holds the full
            # per-source record; the footer carries a summary + the OKF
            # pointer. We write the footer first with a placeholder OKF
            # path, then patch it once the concept path is known. To keep
            # the patch simple, we compute the OKF path deterministically
            # (it's derived from the document stem + bundle path) so the
            # footer and the concept agree without a second pass.
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
        else:
            annotation = _build_model_annotation(
                model_def,
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

    # ── OKF provenance capture (best-effort, source-aware only) ───────
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
        )

    summary = (
        f"Document forged and stored at `{stored_path}` "
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
    if okf_concept_path_written:
        summary += f" Provenance: `{okf_concept_path_written}`"
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
            "source_total_bytes": source_total,
            "density_ratio": density_ratio,
            "density_passed": density_passed,
            "density_retried": density_retried,
            "okf_provenance_path": okf_concept_path_written,
        },
        extra={
            "memory_context_lines": memory_context.count("\n") + 1 if memory_context else 0,
            "document_path": stored_path,
            "truncated": was_truncated,
            "continue_attempts": continue_attempts,
            "source_count": n_readable,
            "density_passed": density_passed,
        },
    )
