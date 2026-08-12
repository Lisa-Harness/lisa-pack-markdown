"""Forge-document operation — memory-grounded single markdown document synthesis.

Pipeline:

1. Recall context from long-term memory via the ``memory.search`` MCP alias
   (routed by the host to the configured memory backend).
2. Call an LLM with an authored system prompt + the user input + recalled
   context (and optionally the existing document) to synthesize or revise a
   markdown document.
3. Handle mid-stream truncation via continuation passes (shared
   ``operations/_continuation`` machinery).
4. Validate the synthesized document's structure by calling the
   ``markdown_generate`` tool (backed by ``markdown-it-py``). The outline
   must contain at least one heading.
5. Persist the document to the ``lisa-storage`` MCP server's ``write_file``
   tool under a caller-supplied relative path (default ``docs/``).

This is the original single-document purpose of the markdown pack. Multi-source
consolidation (source_paths, map-reduce, density floors, OKF provenance) lives
in the separate ``markdown_consolidate_sources`` operation — see
``operations/consolidate/consolidate.py``.

The operation never touches the filesystem directly — all reads (memory) and
writes (storage) go through ``ctx.execute_tool_fn``. Authored prompts are
loaded via ``ctx.read_authored_text_fn``; per the pack-content rule, a
missing/empty loader raises rather than silently falling back.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, Field

from kernel.models import OperationExecutionContext, OperationResult, ToolExecutionContext

from lisa_packs.markdown.operations._config import cfg_bool, cfg_float, cfg_int, cfg_str
from lisa_packs.markdown.operations._continuation import (
    continue_until_complete,
    extract_markdown_block,
)
from lisa_packs.markdown.operations._lint import (
    LintIssue,
    format_lint_issues,
    lint_markdown,
    lint_summary_comment,
)
from lisa_packs.markdown.operations._budget import target_words_to_max_tokens

logger = logging.getLogger(__name__)

__all__ = ["Parameters", "execute"]

# Tool ids — declared in the operation YAML ``tool_ids`` and routed through
# ``ctx.execute_tool_fn``. ``markdown_generate`` is a pack-contributed
# in-process tool; ``write_file`` / ``read_file`` and ``memory.search`` are
# MCP tools served by ``lisa-storage`` and the memory backend respectively.
_OUTLINE_TOOL_ID = "markdown_generate"
_WRITE_TOOL_ID = "write_file"
_READ_TOOL_ID = "read_file"
_MEMORY_SEARCH_TOOL_ID = "memory.search"

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
_LINT_REPAIR_PROMPT_PATH = f"{_PROMPT_DIR}/forge_document_lint_repair.md"

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
_MAX_DOCUMENT_CHARS = 20_000
_MAX_QUERY_CHARS = 20_000

# Phase B: target-length-driven budgeting defaults. When ``target_length`` is
# supplied, the first-pass ``max_tokens`` is derived from the word target
# instead of the flat 50k ceiling, so the model knows how long the document
# should be and truncation becomes rare. ``None`` (the default) preserves the
# legacy flat budget exactly — no behavior change on the default path.
_DEFAULT_TARGET_WORDS = 1200
_DEFAULT_TOKENS_PER_WORD = 1.3
_MIN_TARGET_WORDS = 100

# Phase A: advisory lint gate defaults. ``forge_lint_rules`` overrides the
# curated :data:`STRUCTURAL_RULES` subset; ``forge_lint_repair`` gates the
# single repair pass. Lint is advisory — the outline gate stays the only
# hard reject.
_DEFAULT_LINT_RULES = ""
_DEFAULT_LINT_REPAIR = True

# Minimum structural bar for a forged document: at least one heading.
_MIN_HEADINGS = 1


def _load_text(ctx: OperationExecutionContext, path: str) -> str:
    """Load authored markdown via the SDK content-registry loader.

    Fails loudly when the loader is not wired, returns empty/None, or raises.
    Per the pack-content rule: no fallbacks to missing content.
    """
    host = ctx.host
    if host is None:
        raise RuntimeError(
            f"OperationHost is not wired; cannot load authored content at {path}"
        )
    try:
        loaded = host.read_authored_text(path).strip()
    except Exception as exc:  # pylint: disable=broad-except
        raise RuntimeError(f"read_authored_text failed for {path}: {exc}") from exc
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
        ge=0,
        le=20,
        description="Maximum number of recalled memory records to inject as context. "
        "0 means 'use the default' (the operation coerces 0 to the configured "
        "default).",
    )
    annotate_model: bool = Field(
        default=True,
        description="When True, append a non-rendering HTML comment to the forged "
        "document noting the provider/model that wrote it. The user can also "
        "opt out by including phrases like 'don't annotate the model' in the "
        "instruction. Set False to suppress unconditionally.",
    )
    target_length: int | None = Field(
        default=None,
        ge=0,
        description="Optional target word count for the forged document. When set, "
        "the operation derives the first-pass max_tokens from this target "
        "(target_words * tokens_per_word) and tells the model the approximate "
        "length, so truncation becomes rare. 0 coerces to the default target. "
        "None (the default) preserves the legacy flat 50k-token budget exactly.",
    )


def _extract_document_path(params: dict[str, Any]) -> str:
    """Return the target storage path, sanitized to a safe relative form."""
    raw = str(params.get("document_path") or "").strip()
    if not raw:
        return "docs/forge.md"
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
    # Phase A: advisory lint gate
    lint_rules: str
    lint_repair: bool
    # Phase B: target-length-driven budgeting
    tokens_per_word: float


def _resolve_settings(ctx: OperationExecutionContext) -> _ResolvedSettings:
    """Resolve all operation budgets from ``ctx.pack_config`` with fallbacks."""
    pc = ctx.pack_config
    return _ResolvedSettings(
        max_completion_tokens=cfg_int(pc, "forge_max_completion_tokens", _MAX_COMPLETION_TOKENS),
        max_continue_tokens=cfg_int(pc, "forge_max_continue_tokens", _MAX_CONTINUE_TOKENS),
        max_continue_attempts=cfg_int(pc, "forge_max_continue_attempts", _MAX_CONTINUE_ATTEMPTS),
        max_recall_records=cfg_int(pc, "forge_max_recall_records", _MAX_RECALL_RECORDS),
        max_document_chars=cfg_int(pc, "forge_max_document_chars", _MAX_DOCUMENT_CHARS),
        max_query_chars=cfg_int(pc, "forge_max_query_chars", _MAX_QUERY_CHARS),
        lint_rules=cfg_str(pc, "forge_lint_rules", _DEFAULT_LINT_RULES),
        lint_repair=cfg_bool(pc, "forge_lint_repair", _DEFAULT_LINT_REPAIR),
        tokens_per_word=cfg_float(pc, "forge_tokens_per_word", _DEFAULT_TOKENS_PER_WORD),
    )


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
    tool_result = await ctx.host.execute_tool(
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
        # A "not found" from the storage MCP server is a normal first-forge
        # condition (the document does not exist yet). The error arrives via
        # ``tool_result.error`` when the MCP server wraps the not-found
        # condition as a tool-level error rather than a JSON payload. Treat
        # it the same as the JSON-path "not found" below: return empty string.
        if "not found" in error_text:
            return ""
        raise RuntimeError(f"{_READ_TOOL_ID} failed: {error_text}")
    output_text = getattr(tool_result, "output_text", "") or ""
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
    if ctx.host is None:
        raise RuntimeError("markdown requires OperationHost to call memory/storage tools")

    recall_limit = max(1, min(int(limit), 20))
    tool_result = await ctx.host.execute_tool(
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
    existing_document: str = "",
    *,
    max_tokens: int,
    target_length_guidance: str = "",
) -> tuple[str, Any, bool | None, str | None]:
    """Call the host's completion function with the forging prompt.

    Returns a ``(content, model_def, truncated, finish_reason)`` quadruple
    so the caller can annotate the document with the resolved model and
    detect mid-document truncation via the provider-owned ``truncated``
    signal. ``finish_reason`` is kept for telemetry only.

    ``target_length_guidance`` is injected into the user prompt so the model
    knows the approximate document length. Empty string (the default) leaves
    the rendered prompt byte-identical to the legacy behavior.
    """
    host = ctx.host
    if host is None:
        raise RuntimeError("markdown requires OperationHost.complete and OperationHost.resolve_model")

    preferred_model_id = ctx.preferred_model_id or ctx.model_id or "default"
    model_def = host.resolve_model(preferred_model_id)
    if model_def is None:
        raise RuntimeError(f"markdown could not resolve model '{preferred_model_id}'")

    system_prompt = _load_text(ctx, _SYSTEM_PROMPT_PATH)
    user_template = _load_text(ctx, _USER_PROMPT_PATH)
    user_prompt = _format_template(
        user_template,
        user_input=user_input,
        memory_context=memory_context or "(no recalled context)",
        existing_document=existing_document or "(no existing document — start fresh)",
        target_length_guidance=target_length_guidance,
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    completion = await host.complete(
        messages=messages,
        model_def=model_def,
        max_tokens=max_tokens,
        temperature=0.2,
        tools=None,
        response_format=None,
    )
    content = getattr(completion, "content", "") or ""
    truncated = getattr(completion, "truncated", None)
    finish_reason = getattr(completion, "finish_reason", None)
    return content.strip(), model_def, truncated, finish_reason


def _user_asked_not_to_annotate(user_input: str) -> bool:
    """Return True when the user instruction explicitly opts out of attribution."""
    lowered = user_input.lower()
    return any(phrase in lowered for phrase in _NO_ANNOTATE_PHRASES)


def _build_model_annotation(
    model_def: Any,
    *,
    truncated: bool = False,
    final_finish_reason: str | None = None,
    continue_attempts: int = 0,
) -> str:
    """Build a trailing HTML-comment attribution line for the resolved model.

    When ``truncated`` is True, an explicit ``truncated by model`` marker is
    appended so a reader scanning the artifact knows the document is
    incomplete even though the operation persisted it.
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


def _resolve_target_length(
    params: dict[str, Any],
    *,
    max_document_chars: int,
    tokens_per_word: float,
) -> tuple[int | None, str, int]:
    """Resolve the optional ``target_length`` into ``(words, guidance, max_tokens)``.

    Returns ``(None, "", cfg.max_completion_tokens-equivalent)`` when no
    target is supplied — the legacy flat-budget path. When a target is
    supplied, returns the resolved word count, a prompt guidance line, and
    a derived ``max_tokens`` budget (clamped to the document char cap so a
    huge target cannot exceed what the char-cap would clip anyway).
    """
    raw = params.get("target_length")
    if raw is None:
        return None, "", 0  # 0 is a sentinel: caller uses the flat budget
    try:
        words = int(raw)
    except (TypeError, ValueError):
        return None, "", 0
    if words == 0:
        words = _DEFAULT_TARGET_WORDS
    if words < _MIN_TARGET_WORDS:
        words = _MIN_TARGET_WORDS

    # Coherence: a target whose char equivalent exceeds the document char
    # cap would be silently clipped post-merge. Clamp it down.
    approx_chars = int(words * 5)  # ~5 chars/word English heuristic
    if approx_chars > max_document_chars:
        clamped = max(_MIN_TARGET_WORDS, max_document_chars // 5)
        logger.warning(
            "markdown: target_length=%d words (~%d chars) exceeds max_document_chars=%d; "
            "clamping to %d words",
            words,
            approx_chars,
            max_document_chars,
            clamped,
        )
        words = clamped

    max_tokens = target_words_to_max_tokens(words, tokens_per_word=tokens_per_word)
    guidance = (
        f"Target length: approximately {words} words "
        f"({max(2, words // 200)} sections suggested)."
    )
    return words, guidance, max_tokens


async def _lint_repair_pass(
    ctx: OperationExecutionContext,
    *,
    user_input: str,
    document: str,
    model_def: Any,
    issues: list[LintIssue],
    max_tokens: int,
) -> str:
    """Issue one repair pass feeding the lint issues back to the model.

    Returns the repaired document text (extracted from the fenced
    markdown block), or the original ``document`` on any failure. Never
    raises; advisory.
    """
    if not issues:
        return document
    if ctx.host is None:
        return document
    try:
        repair_prompt = _load_text(ctx, _LINT_REPAIR_PROMPT_PATH)
    except RuntimeError as exc:
        logger.warning("markdown: lint repair prompt unavailable: %s", exc)
        return document
    user_prompt = _format_template(
        repair_prompt,
        user_input=user_input,
        lint_issues=format_lint_issues(issues),
        document_so_far=document,
    )
    messages = [
        {"role": "system", "content": _load_text(ctx, _SYSTEM_PROMPT_PATH)},
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
        logger.warning("markdown: lint repair pass raised: %s", exc)
        return document
    content = (getattr(completion, "content", "") or "").strip()
    if not content:
        return document
    repaired = extract_markdown_block(content)
    if not repaired.strip():
        return document
    return repaired


async def _validate_outline(ctx: OperationExecutionContext, document: str) -> dict[str, Any]:
    """Run the ``markdown_gehost.execute_toolnd return its outline dict.

    Raises ``RuntimeError`` if the tool returns an error.
    """
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
    """Write the document to lisa-storage via the ``write_file`` tool.
host.execute_tool
    Returns the parsed JSON response (with ``ok``/``path``/``bytes``).
    Raises ``RuntimeError`` on any tool failure.
    """
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


def _format_markdown(document: str) -> str:
    """Lint/format the document via ``mdformat`` if available."""
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

    if ctx.host is None:
        return OperationResult(output_text=_load_text(ctx, _USAGE_PATH))

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

    # Phase B: resolve optional target_length into a derived max_tokens and
    # prompt guidance. ``target_words is None`` preserves the legacy flat
    # 50k-token budget exactly — no behavior change on the default path.
    target_words, target_guidance, derived_max_tokens = _resolve_target_length(
        params,
        max_document_chars=cfg.max_document_chars,
        tokens_per_word=cfg.tokens_per_word,
    )
    first_pass_max_tokens = (
        derived_max_tokens if target_words is not None else cfg.max_completion_tokens
    )
    # Scale the continuation budget with the target so a small target does
    # not retain the 50k continuation ceiling (which would re-truncate).
    first_pass_continue_tokens = (
        min(cfg.max_continue_tokens, first_pass_max_tokens * 2)
        if target_words is not None
        else cfg.max_continue_tokens
    )
    completion_text, model_def, first_truncated, first_finish_reason = await _call_llm(
        ctx,
        user_input,
        memory_context,
        existing_document=existing_document,
        max_tokens=first_pass_max_tokens,
        target_length_guidance=target_guidance,
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

    # Truncation handling: the provider-owned ``truncated`` signal drives
    # the decision. ``True`` → continue up to the budget; ``False`` → no
    # continuation (unless a dangling-heading structural override fires);
    # ``None`` → one safety-net pass, then stop.
    document, was_truncated, continue_attempts, final_finish_reason = (
        await continue_until_complete(
            ctx,
            user_input=user_input,
            document=document,
            model_def=model_def,
            first_truncated=first_truncated,
            first_finish_reason=first_finish_reason,
            max_continue_attempts=cfg.max_continue_attempts,
            max_continue_tokens=first_pass_continue_tokens,
            continue_prompt_path=_CONTINUE_PROMPT_PATH,
        )
    )

    # Lint/format the document so the persisted output is clean and consistent.
    document = _format_markdown(document)

    # Char cap.
    if len(document) > cfg.max_document_chars:
        document = document[:cfg.max_document_chars] + "\n\n<!-- truncated -->\n"
        was_truncated = True

    # Annotate with the model that wrote it, unless the user opted out.
    annotate_model = bool(params.get("annotate_model", True))
    if annotate_model and not _user_asked_not_to_annotate(user_input):
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

    # Phase A: advisory lint gate. Lint is never a hard reject — the outline
    # gate above is the only hard gate. When issues are found and
    # ``forge_lint_repair`` is set, issue one repair pass feeding the lint
    # issues back to the model; re-format and re-validate. Any issues that
    # remain (or when repair is disabled) are persisted as a non-rendering
    # HTML comment so the user knows the artifact has warnings.
    lint_issues = lint_markdown(document, rules_csv=cfg.lint_rules or None)
    lint_repair_used = False
    if lint_issues and cfg.lint_repair:
        repaired = await _lint_repair_pass(
            ctx,
            user_input=user_input,
            document=document,
            model_def=model_def,
            issues=lint_issues,
            max_tokens=first_pass_max_tokens,
        )
        if repaired != document:
            lint_repair_used = True
            document = _format_markdown(repaired)
            # Re-validate the repaired document's outline.
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
            "lint_warnings": len(lint_issues),
            "lint_repair_used": lint_repair_used,
            "target_words": target_words,
        },
        extra={
            "memory_context_lines": memory_context.count("\n") + 1 if memory_context else 0,
            "document_path": stored_path,
            "truncated": was_truncated,
            "continue_attempts": continue_attempts,
            "lint_warnings": len(lint_issues),
            "target_words": target_words,
        },
    )
