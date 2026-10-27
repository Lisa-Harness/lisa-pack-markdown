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
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from kernel.models import (
    OperationExecutionContext,
    OperationResult,
    ToolExecutionContext,
)
from lisa_packs.markdown.operations._budget import target_words_to_max_tokens
from lisa_packs.markdown.operations._config import cfg_bool, cfg_float, cfg_int, cfg_str
from lisa_packs.markdown.operations._continuation import (
    continue_until_complete,
    expand_missing_sections,
    extract_markdown_block,
    missing_planned_sections,
    parse_forge_plan,
    repair_inline_fences,
)
from lisa_packs.markdown.operations._lint import (
    LintIssue,
    format_lint_issues,
    lint_markdown,
    lint_summary_comment,
)
from lisa_packs.markdown.operations._memory import recall_memory
from pydantic import BaseModel, Field

from kernel import model_budgets

logger = logging.getLogger(__name__)

__all__ = ["Parameters", "execute", "lint_markdown"]

# Tool ids — declared in the operation YAML ``tool_ids`` and routed through
# ``ctx.execute_tool_fn``. ``markdown_generate`` is a pack-contributed
# in-process tool; ``write_file`` / ``read_file`` and ``memory.search`` are
# MCP tools served by ``lisa-storage`` and the memory backend respectively.
_OUTLINE_TOOL_ID = "markdown_generate"
_WRITE_TOOL_ID = "write_file"
_READ_TOOL_ID = "read_file"

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
_EXPAND_PROMPT_PATH = f"{_PROMPT_DIR}/forge_document_expand.md"
_LINT_REPAIR_PROMPT_PATH = f"{_PROMPT_DIR}/forge_document_lint_repair.md"
_UNNAMED_OVERWRITE_PATH = f"{_PROMPT_DIR}/forge_document_unnamed_overwrite.md"

#: Destination used when the caller names no ``document_path``. A call that
#: never named the path may not overwrite an existing document here — the
#: 2026-10-26 incident clobbered exactly this file from a zero-argument call.
_DEFAULT_DOCUMENT_PATH = "docs/forge.md"

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

# Outline contract (self-abridgment defense, 2026-10-16 incident): when the
# model declares a ``<!-- forge-plan: ... -->`` section plan and the finished
# document is missing planned sections, run bounded expansion passes. The
# gate is advisory — like lint, it never hard-rejects; a still-incomplete
# document persists with a degraded status and a structured telemetry block.
# Documents without a plan marker skip the gate entirely (legacy path).
_DEFAULT_OUTLINE_CONTRACT = True
_DEFAULT_OUTLINE_CONTRACT_ATTEMPTS = 2
_DEFAULT_LINT_REPAIR = True

# Reasoning-burnout rescue default. ``forge_burnout_rescue`` gates the
# empty-first-pass retry (see ``_burnout_retry_budget``); enabled by
# default because the failure it repairs is a budget-shape failure no
# caller refinement can fix.
_DEFAULT_BURNOUT_RESCUE = True

#: Minimum structural bar for a forged document: at least one heading.
_MIN_HEADINGS = 1


def _load_text(ctx: OperationExecutionContext, path: str) -> str:
    """Load authored markdown via the SDK content-registry loader.

    Fails loudly when the loader is not wired, returns empty/None, or raises.
    Per the pack-content rule: no fallbacks to missing content.
    """
    host = ctx.host
    if host is None:
        raise RuntimeError(f"OperationHost is not wired; cannot load authored content at {path}")
    try:
        loaded = str(host.read_authored_text(path) or "").strip()
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
        default=_DEFAULT_DOCUMENT_PATH,
        description="Relative path within the lisa-storage files sandbox where the forged document "
        "is persisted. Omitting it defaults to 'docs/forge.md', but an existing document at "
        "that default is not overwritten unless this field is set explicitly or "
        "existing_document is supplied.",
    )
    query: str = Field(
        default="",
        description="Natural-language instruction for the document. Falls back to ctx.query when "
        "empty.",
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
        return _DEFAULT_DOCUMENT_PATH
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
    # Reasoning-burnout rescue: whether the empty-first-pass retry fires.
    burnout_rescue: bool

    # Outline contract: expand passes for planned-but-missing sections.
    outline_contract: bool
    outline_contract_attempts: int


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
        outline_contract=cfg_bool(pc, "forge_outline_contract", _DEFAULT_OUTLINE_CONTRACT),
        outline_contract_attempts=cfg_int(
            pc, "forge_outline_contract_attempts", _DEFAULT_OUTLINE_CONTRACT_ATTEMPTS
        ),
        burnout_rescue=cfg_bool(pc, "forge_burnout_rescue", _DEFAULT_BURNOUT_RESCUE),
    )


def _extract_query(
    ctx: OperationExecutionContext, params: dict[str, Any], max_query_chars: int
) -> str:
    """Return the user-supplied instruction, bounded in length.

    An explicit empty ``query`` means "no instruction" and fails closed. It
    must not fall back to the ambient turn query: a call that passes
    ``query: ""`` is asking for nothing specific, and filling it from the
    turn's conversation forges a document the caller never requested. That
    fallback is how a zero-argument invocation produced 2,450 bytes of
    invented content (the 2026-10-26 forge incident).

    The ambient ``ctx.query`` remains the fallback only when the ``query``
    parameter is *absent*, which is the ordinary "the operation was invoked
    with the turn's own request" shape.
    """
    if "query" in params:
        raw = str(params.get("query") or "").strip()
    else:
        raw = str(ctx.query or "").strip()
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
    return await recall_memory(ctx, query, limit=limit)


async def _call_llm(
    ctx: OperationExecutionContext,
    user_input: str,
    memory_context: str,
    existing_document: str = "",
    *,
    max_tokens: int,
    target_length_guidance: str = "",
    max_document_chars: int = _MAX_DOCUMENT_CHARS,
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
        raise RuntimeError(
            "markdown requires OperationHost.complete and OperationHost.resolve_model"
        )

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
        target_length_guidance=(
            target_length_guidance or _size_headroom_guidance(existing_document, max_document_chars)
        ),
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


def _size_headroom_guidance(existing_document: str, max_document_chars: int) -> str:
    """Return the prompt guidance line describing the persistence char budget.

    The 2026-10-22 forge char-cap incident: a 32.9k-char plan revision grew
    to 46.6k chars, the post-hoc clamp silently cut it to 20k, and the
    degraded result negated the write. Telling the model the budget
    *before* it writes lets it plan inside the cap instead of discovering
    it after a multi-minute generation. Empty on the fresh-forge path
    (no existing document) so the legacy fresh prompt stays byte-identical
    even when the cap is generous.
    """
    if not existing_document:
        return ""
    existing_len = len(existing_document)
    headroom = max_document_chars - existing_len
    if headroom <= 0:
        return (
            f"Size guidance: the existing document is {existing_len} characters; the "
            f"persistence budget allows growth beyond it, so preserve ALL existing approved "
            f"content verbatim while adding the requested changes."
        )
    return (
        f"Size guidance: the existing document is {existing_len} characters and the "
        f"persistence budget is about {max_document_chars} characters ({headroom} characters "
        f"of headroom). Preserve ALL existing approved content verbatim and keep your "
        f"additions within the headroom."
    )


def _empty_response_diagnostics(finish_reason: Any, max_tokens: int) -> str:
    """Render the ``{diagnostics}`` placeholder of the empty-response prompt.

    The 2026-10-14 forge incident returned a bare "Empty Response" four turns
    in a row and the failure was misread as environmental. Surfacing the
    finish reason and the budget makes a reasoning-budget exhaustion — the
    budget-shape failure the rescue above repairs — distinguishable from a
    genuine provider-empty response in the tool result itself (the
    ``_empty_content_detail`` contract the declarative operations already
    follow).
    """
    parts: list[str] = []
    if finish_reason is not None:
        parts.append(f"finish_reason={finish_reason}")
    parts.append(f"max_tokens={max_tokens}")
    if str(finish_reason) == "length":
        parts.append("likely_reasoning_budget_exhaustion")
    return f" [{', '.join(parts)}]"


def _burnout_retry_budget(
    *,
    first_pass_max_tokens: int,
    model_def: Any,
    flat_ceiling: int,
) -> int | None:
    """Return the rescue budget for a budget-exhausted first pass.

    The 2026-10-14 forge incident: a ``target_length``-derived budget
    (words × tokens-per-word, e.g. 1300 tokens for a 1000-word target) is an
    *output* heuristic, but a reasoning model spends its chain-of-thought
    against the same budget — the deliberation alone exhausted it before
    any document text was emitted. The turn-loop rescues share this shape;
    a plain 2× of a derived budget is still too small when the deliberation
    is long (verified: 2 × 1300 = 2600 still burned out on the same query).

    The rescue therefore jumps to the operation's flat ceiling — the budget
    the *without-target* path already proves sufficient — when that
    exceeds the exhausted first-pass budget; ``None`` when raising cannot
    help (flat ceiling not above the base, or the model's declared
    ``max_completion_tokens`` caps below the base).
    """
    model_ceiling = model_budgets._model_completion_ceiling(model_def)
    candidates = [flat_ceiling]
    if model_ceiling is not None:
        candidates.append(model_ceiling)
    resolved = max(first_pass_max_tokens, min(candidates))
    if resolved <= first_pass_max_tokens:
        return None
    return resolved


async def _complete_raw(
    ctx: OperationExecutionContext,
    *,
    user_input: str,
    memory_context: str,
    existing_document: str,
    max_tokens: int,
    target_length_guidance: str,
    model_def: Any,
    max_document_chars: int = _MAX_DOCUMENT_CHARS,
) -> Any:
    """Issue one forge completion and return the raw host completion object.

    ``_call_llm`` reduces the completion to four scalars; the burnout
    classifier needs the spending shape (``usage``), so the forge first
    pass goes through this variant. Renders the same prompts byte-identical
    to ``_call_llm``.
    """
    system_prompt = _load_text(ctx, _SYSTEM_PROMPT_PATH)
    user_template = _load_text(ctx, _USER_PROMPT_PATH)
    user_prompt = _format_template(
        user_template,
        user_input=user_input,
        memory_context=memory_context or "(no recalled context)",
        existing_document=existing_document or "(no existing document — start fresh)",
        target_length_guidance=(
            target_length_guidance or _size_headroom_guidance(existing_document, max_document_chars)
        ),
    )
    return await ctx.host.complete(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        model_def=model_def,
        max_tokens=max_tokens,
        temperature=0.2,
        tools=None,
        response_format=None,
    )


def _completion_spent(completion: Any, *, requested: int) -> bool:
    """Return True when the completion visibly consumed its whole budget.

    A ``length`` stop is the provider's own exhaustion proof. A clean stop
    needs the completion-token spend to reach the shared burn share — the
    exact predicate the turn-loop rescues run.
    """
    if getattr(completion, "finish_reason", None) == "length":
        return True
    return model_budgets._is_reasoning_burnout_completion(completion, base_max_tokens=requested)


async def _call_llm_forge(
    ctx: OperationExecutionContext,
    user_input: str,
    memory_context: str,
    *,
    existing_document: str,
    max_tokens: int,
    target_length_guidance: str,
    flat_ceiling: int,
    burnout_rescue: bool,
    max_document_chars: int = _MAX_DOCUMENT_CHARS,
) -> tuple[str, Any, bool | None, str | None, int]:
    """Forge first pass plus one budget-shape rescue.

    Reasoning-capable models can exhaust a small derived budget on
    chain-of-thought before emitting anything (the 2026-10-14 forge Empty
    Response incident: four reproducible failures with
    ``finish_reason=length, content=""`` at a target-derived 1300-token
    budget). When the first pass consumed its budget and returned no
    visible content, the same call retries once at
    ``_burnout_retry_budget``'s usable ceiling. Any other empty response
    (a cheap provider-empty generation whose spend never approached the
    budget) returns unchanged — the model really had nothing, and retrying
    would only double an already-wasted call.

    Returns ``(content, model_def, truncated, finish_reason, max_tokens)``
    where ``max_tokens`` is the budget that produced the returned content
    (the rescue's ceiling when the retry ran).
    """
    host = ctx.host
    if host is None:
        raise RuntimeError(
            "markdown requires OperationHost.complete and OperationHost.resolve_model"
        )
    preferred_model_id = ctx.preferred_model_id or ctx.model_id or "default"
    model_def = host.resolve_model(preferred_model_id)
    if model_def is None:
        raise RuntimeError(f"markdown could not resolve model '{preferred_model_id}'")

    completion = await _complete_raw(
        ctx,
        user_input=user_input,
        memory_context=memory_context,
        existing_document=existing_document,
        max_tokens=max_tokens,
        target_length_guidance=target_length_guidance,
        model_def=model_def,
        max_document_chars=max_document_chars,
    )
    content = (getattr(completion, "content", "") or "").strip()
    truncated = getattr(completion, "truncated", None)
    finish_reason = getattr(completion, "finish_reason", None)
    # Gate on empty visible content, not on a budget diagnosis. A document
    # forge has no valid empty outcome, so "the model returned nothing" is
    # fatal whatever the finish reason — including the cheap clean stop that
    # the shared 90% burnout share deliberately declines to rescue (the
    # 2026-10-26 incident: finish_reason='stop', sparse spend, no rescue, and
    # the operation reported success with an empty artifact).
    #
    # The raw attribute is the gate, never ``completion_content``: that helper
    # folds ``reasoning_content`` into its return when content is empty, so
    # gating on it would persist the model's chain-of-thought as the document.
    if not content and burnout_rescue:
        diagnosis = model_budgets._burnout_or_truncation(completion, base_max_tokens=max_tokens)
        retry_tokens = _burnout_retry_budget(
            first_pass_max_tokens=max_tokens,
            model_def=model_def,
            flat_ceiling=flat_ceiling,
        )
        if retry_tokens is None:
            # Raising the ceiling cannot help, but a stochastic empty response
            # is still worth one more attempt at the same budget.
            retry_tokens = max_tokens
        logger.warning(
            "forge_document empty content: finish_reason=%s completion_tokens=%s "
            "max_tokens=%d diagnosis=%s; retrying once (model=%s, to=%d)",
            finish_reason,
            model_budgets._completion_usage_tokens(completion),
            max_tokens,
            diagnosis,
            getattr(model_def, "model_id", ""),
            retry_tokens,
        )
        retry_completion = await _complete_raw(
            ctx,
            user_input=user_input,
            memory_context=memory_context,
            existing_document=existing_document,
            max_tokens=retry_tokens,
            target_length_guidance=target_length_guidance,
            model_def=model_def,
            max_document_chars=max_document_chars,
        )
        return (
            (getattr(retry_completion, "content", "") or "").strip(),
            model_def,
            getattr(retry_completion, "truncated", None),
            getattr(retry_completion, "finish_reason", None),
            retry_tokens,
        )
    return content, model_def, truncated, finish_reason, max_tokens


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
    missing_sections: tuple[str, ...] = (),
) -> str:
    """Build a trailing HTML-comment attribution line for the resolved model.

    When ``truncated`` is True, an explicit ``truncated by model`` marker is
    appended so a reader scanning the artifact knows the document is
    incomplete even though the operation persisted it. ``missing_sections``
    is the outline-contract gap: planned sections the document still lacks.
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
        if missing_sections:
            parts.append("missing sections: " + ", ".join(missing_sections))
        else:
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
    existing_document: str = "",
) -> tuple[int | None, str, int]:
    """Resolve the optional ``target_length`` into ``(words, guidance, max_tokens)``.

    Returns ``(None, "", 0)`` when no target is supplied — the legacy
    flat-budget path. When a target is supplied, returns the resolved word
    count, a prompt guidance line, and a derived ``max_tokens`` budget
    (clamped to the document char cap so a huge target cannot exceed what
    the char-cap would clip anyway).
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
    words = max(words, _MIN_TARGET_WORDS)

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
        f"Target length: approximately {words} words ({max(2, words // 200)} sections suggested)."
    )
    return words, guidance, max_tokens


async def _request_lint_repair(
    ctx: OperationExecutionContext,
    *,
    user_input: str,
    document: str,
    model_def: Any,
    issues: list[LintIssue],
    max_tokens: int,
) -> str | None:
    """Ask the model to repair the lint issues. Returns ``None`` on failure.

    The ``None`` result covers every failure mode: no host, an unloadable
    repair prompt, a raised completion, an empty completion, and a
    completion whose fenced block yields no usable text. The caller
    treats ``None`` as "keep the original document".
    """
    if ctx.host is None:
        return None
    try:
        repair_prompt = _load_text(ctx, _LINT_REPAIR_PROMPT_PATH)
    except RuntimeError as exc:
        logger.warning("markdown: lint repair prompt unavailable: %s", exc)
        return None
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
        return None
    content = (getattr(completion, "content", "") or "").strip()
    if not content:
        return None
    repaired = extract_markdown_block(content)
    if not repaired.strip():
        return None
    return repaired


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
    repaired = await _request_lint_repair(
        ctx,
        user_input=user_input,
        document=document,
        model_def=model_def,
        issues=issues,
        max_tokens=max_tokens,
    )
    if repaired is None:
        return document
    return repaired


async def _persist_no_outline_document(
    ctx: OperationExecutionContext,
    document: str,
    document_path: str,
    *,
    heading_count: int,
    was_truncated: bool,
    continue_attempts: int,
    first_finish_reason: str | None,
    final_finish_reason: str | None,
    continuation_stop_reason: str,
    forge_plan: tuple[str, ...] = (),
    planned_missing: tuple[str, ...] = (),
    outline_contract_attempts: int = 0,
    outline_contract_stop_reason: str = "skipped",
) -> OperationResult:
    """Persist a zero-heading document with a warning comment instead of discarding it.

    The outline gate used to be a hard reject: a document with 0 headings was
    discarded and the operation returned only an error message, leaving no
    durable artifact. When the explainer loop re-called the operation, each
    iteration burned budget on a reject that produced nothing. Now the
    document is persisted with a non-rendering ``<!-- forge: N headings
    detected -->`` comment so the content survives and the user can refine.
    """
    warning = f"<!-- forge: {heading_count} headings detected -->"
    annotated = document.rstrip() + "\n\n" + warning + "\n"
    try:
        write_response = await _persist_document(ctx, annotated, document_path)
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
    byte_count = int(write_response.get("bytes") or len(annotated.encode("utf-8")))
    warning_text = _format_template(
        _load_text(ctx, _NO_OUTLINE_PATH),
        heading_count=heading_count,
    )
    summary = (
        f"Document forged at `{stored_path}` ({byte_count} bytes, "
        f"{heading_count} headings). WARNING: no headings detected; "
        f"persisted with a warning comment. {warning_text}"
    )
    return OperationResult(
        output_text=summary,
        output_json={
            "path": stored_path,
            "bytes": byte_count,
            "heading_count": heading_count,
            "outline": {},
            "truncated": was_truncated,
            "continue_attempts": continue_attempts,
            "first_finish_reason": first_finish_reason,
            "final_finish_reason": final_finish_reason,
            "lint_warnings": 0,
            "lint_repair_used": False,
            "target_words": None,
            "no_outline": True,
        },
        status="degraded",
        error_detail=(
            f"no headings detected (heading_count={heading_count})"
            + (
                f"; continuation {continuation_stop_reason}: "
                f"{continue_attempts} attempt(s), finish_reason={final_finish_reason!r}"
                if was_truncated
                else ""
            )
        ),
        extra={
            "document_path": stored_path,
            "truncated": was_truncated,
            "continue_attempts": continue_attempts,
            "no_outline": True,
            "outline_contract": {
                "planned": list(forge_plan),
                "missing_before": list(planned_missing),
                "missing_after": list(planned_missing),
                "attempts": outline_contract_attempts,
                "stop_reason": outline_contract_stop_reason,
            },
            "continuation": {
                "was_truncated": was_truncated,
                "attempts": continue_attempts,
                "stop_reason": continuation_stop_reason,
                "first_finish_reason": first_finish_reason,
                "final_finish_reason": final_finish_reason,
            },
        },
    )


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
        import mdformat
    except ImportError:
        logger.debug("mdformat not installed; skipping markdown formatting")
        return document
    try:
        return mdformat.text(document, options={"number": True}).rstrip() + "\n"
    except Exception as exc:  # pylint: disable=broad-except
        logger.warning("forge_document: mdformat failed (%s); using unformatted document", exc)
        return document


def _cap_document_at_heading(document: str, char_cap: int) -> str:
    """Cap ``document`` at ``char_cap`` chars without cutting mid-section.

    The old slice cut at an arbitrary character offset — mid-sentence,
    often mid-word — leaving a document that ended
    ``"...worth it; a two-week delay for the s`` with no closure and no
    signal of what was dropped. This helper instead backs off to the last
    heading boundary (a line-start ``#`` marker) that fits inside the
    cap and appends an explicit ``<!-- forge: capped -->`` comment that
    names every section heading beyond the cut, so the model and the
    reader can see exactly what was omitted and re-request it.

    Falls back to the legacy mid-text slice (with the same trailing
    comment) only when no heading boundary fits at all — a document with
    no earlier heading cannot be split safely.
    """
    stripped = document.rstrip()
    lines = stripped.splitlines(keepends=True)

    # Record the start offset of every line; heading lines are candidate cuts.
    offsets: list[tuple[int, str, bool]] = []
    offset = 0
    for line in lines:
        offsets.append((offset, line, bool(re.match(r"^#{1,6}\s", line))))
        offset += len(line)

    best_cut_line_count = 0
    for line_index, (line_offset, _line, is_heading) in enumerate(offsets):
        if is_heading and 0 < line_offset < char_cap:
            best_cut_line_count = line_index

    if best_cut_line_count:
        body = "".join(lines[:best_cut_line_count]).rstrip()
    else:
        body = stripped[:char_cap].rstrip()

    cut_at = offsets[best_cut_line_count][0] if best_cut_line_count else char_cap
    dropped_section_heads = [
        line.strip()
        for line_offset, line, is_heading in offsets
        if is_heading and line_offset >= cut_at
    ]
    comment = "<!-- forge: capped -->"
    if dropped_section_heads:
        note = (
            f"the document exceeded the {char_cap}-character persistence cap and was cut at a "
            "section boundary; the following section(s) were omitted: "
            + "; ".join(dropped_section_heads)
            + ". Re-run the forge with a higher ``forge_max_document_chars`` to keep them."
        )
    else:
        note = f"the document exceeded the {char_cap}-character persistence cap and was cut."
    return f"{body}\n\n{comment}\n{note}\n"


# ── Forging phases ──────────────────────────────────────────────────
#
# ``execute`` is the pipeline's orchestrator, not its implementation. The
# work is split into phases — prepare → forge → outline contract →
# finalize → validate → persist — each in its own helper. ``_ForgeState``
# is the shared record the phases read and write.


@dataclass
class _ForgeState:
    """Operation state threaded through the forging phases."""

    # ── prepared inputs ──
    params: dict[str, Any]
    cfg: _ResolvedSettings
    user_input: str = ""
    document_path: str = ""
    memory_context: str = ""
    existing_document: str = ""

    # ── target-length budgets ──
    target_words: int | None = None
    target_guidance: str = ""
    first_pass_max_tokens: int = 0
    first_pass_continue_tokens: int = 0

    # ── forge outputs ──
    document: str = ""
    model_def: Any = None
    first_truncated: bool | None = None
    first_finish_reason: str | None = None
    final_finish_reason: str | None = None
    was_truncated: bool = False
    continue_attempts: int = 0
    continuation_stop_reason: str = ""

    # ── outline contract ──
    forge_plan: tuple[str, ...] = ()
    planned_missing: tuple[str, ...] = ()
    missing_plan_sections: tuple[str, ...] = ()
    outline_contract_attempts: int = 0
    outline_contract_stop_reason: str = "skipped"


async def _prepare_forge(
    ctx: OperationExecutionContext,
) -> tuple[_ForgeState, OperationResult | None]:
    """Phase 1 — resolve settings, the instruction, and the existing document.

    Returns ``(state, failure)``. ``failure`` is a terminal
    ``OperationResult`` when the operation cannot proceed (no
    instruction, an invalid document path, or an unwired host).
    """
    params = ctx.params or {}
    cfg = _resolve_settings(ctx)
    state = _ForgeState(params=params, cfg=cfg)
    user_input = _extract_query(ctx, params, cfg.max_query_chars)
    if not user_input:
        # No instruction is a failure to report, not a successful forge: a
        # default ``status="success"`` here made the turn treat the operation
        # as satisfied while nothing was produced.
        return state, OperationResult(
            output_text=_load_text(ctx, _NO_INPUT_PATH),
            status="error",
            error_detail="no instruction was provided in query",
        )
    state.user_input = user_input
    try:
        state.document_path = _extract_document_path(params)
    except RuntimeError as exc:
        return state, OperationResult(output_text=str(exc))
    if ctx.host is None:
        return state, OperationResult(output_text=_load_text(ctx, _USAGE_PATH))
    return state, None


async def _load_existing_document(
    ctx: OperationExecutionContext,
    state: _ForgeState,
) -> OperationResult | None:
    """Read the document the caller wants revised, when one is not supplied.

    When the caller omits ``existing_document`` but the target path already
    exists in storage, read it so the LLM revises rather than replacing
    wholesale. This is the documented iteration contract, not an implicit
    overwrite.

    A document found at the *default* path, reached by a call that never named
    ``document_path``, is refused instead: the destination was not chosen by
    the caller, so replacing what is there would destroy content nobody asked
    to replace (the 2026-10-26 incident). A first forge at that path still
    proceeds — the guard is about overwriting, not about the default itself.

    Returns a terminal ``OperationResult`` when the read fails or the
    overwrite is refused.
    """
    existing_document = str(state.params.get("existing_document") or "").strip()
    if existing_document:
        # The caller has read the file it is replacing, so the overwrite is
        # demonstrably intentional.
        state.existing_document = existing_document
        return None
    try:
        state.existing_document = await _read_existing_document(ctx, state.document_path)
    except RuntimeError as exc:
        return OperationResult(output_text=str(exc), status="error")
    if (
        state.document_path == _DEFAULT_DOCUMENT_PATH
        and not _named_document_path(state)
        and state.existing_document
    ):
        # A first forge at the default path is fine; replacing a document the
        # caller never named the path of is not. Refused before any model call.
        return OperationResult(
            output_text=_format_template(
                _load_text(ctx, _UNNAMED_OVERWRITE_PATH),
                document_path=state.document_path,
            ),
            status="error",
            error_detail=(
                "refused to overwrite the default document path "
                f"'{state.document_path}' from a call that did not name it; "
                "pass document_path explicitly to overwrite"
            ),
        )
    return None


def _named_document_path(state: _ForgeState) -> bool:
    """Whether the caller named the destination instead of defaulting to it."""
    return str(state.params.get("document_path") or "").strip() != ""


def _resolve_forge_budgets(state: _ForgeState) -> None:
    """Resolve ``target_length`` into the first-pass and continuation budgets.

    ``target_words is None`` preserves the legacy flat 50k-token budget
    exactly — no behavior change on the default path.
    """
    cfg = state.cfg
    effective_char_cap = max(cfg.max_document_chars, len(state.existing_document))
    target_words, target_guidance, derived_max_tokens = _resolve_target_length(
        state.params,
        max_document_chars=effective_char_cap,
        tokens_per_word=cfg.tokens_per_word,
    )
    state.target_words = target_words
    state.target_guidance = target_guidance
    state.first_pass_max_tokens = (
        derived_max_tokens if target_words is not None else cfg.max_completion_tokens
    )
    # Scale the continuation budget with the target so a small target does
    # not retain the 50k continuation ceiling (which would re-truncate).
    state.first_pass_continue_tokens = (
        min(cfg.max_continue_tokens, state.first_pass_max_tokens * 2)
        if target_words is not None
        else cfg.max_continue_tokens
    )


async def _forge_document_body(
    ctx: OperationExecutionContext,
    state: _ForgeState,
) -> OperationResult | None:
    """Phase 2 — call the model and continue until the document is complete.

    Returns a terminal ``OperationResult`` when the model produced nothing
    usable.
    """
    cfg = state.cfg
    (
        completion_text,
        model_def,
        first_truncated,
        first_finish_reason,
        effective_max_tokens,
    ) = await _call_llm_forge(
        ctx,
        state.user_input,
        state.memory_context,
        existing_document=state.existing_document,
        max_tokens=state.first_pass_max_tokens,
        target_length_guidance=state.target_guidance,
        flat_ceiling=cfg.max_completion_tokens,
        burnout_rescue=cfg.burnout_rescue,
        max_document_chars=max(cfg.max_document_chars, len(state.existing_document)),
    )
    state.model_def = model_def
    state.first_truncated = first_truncated
    state.first_finish_reason = first_finish_reason
    if not completion_text or not completion_text.strip():
        # Never report success for an empty body. The operation's entire
        # purpose is to produce a document, so an empty one is a hard failure
        # regardless of finish_reason — the 2026-10-26 incident returned
        # ``finish_reason='stop'`` with zero content, and a default
        # ``status="success"`` made the turn treat the forge as satisfied and
        # hand the caller an empty artifact.
        return OperationResult(
            output_text=_format_template(
                _load_text(ctx, _EMPTY_RESPONSE_PATH),
                diagnostics=_empty_response_diagnostics(first_finish_reason, effective_max_tokens),
            ),
            status="error",
            error_detail=(
                f"model returned no content after the rescue pass "
                f"(finish_reason={first_finish_reason!r}, max_tokens={effective_max_tokens})"
            ),
        )
    document = extract_markdown_block(completion_text)
    if not document.strip():
        text = _format_template(
            _load_text(ctx, _EMPTY_DOCUMENT_PATH),
            response_preview=completion_text[:400],
        )
        return OperationResult(
            output_text=text,
            status="error",
            error_detail="completion contained no extractable markdown document",
        )
    # Outline contract: strip the declared section plan from the body and
    # remember it for the post-continuation completeness check. No marker →
    # no plan → the gate is skipped entirely (legacy path, no extra calls).
    document, forge_plan = parse_forge_plan(document)
    state.forge_plan = forge_plan
    # Truncation handling: the provider-owned ``truncated`` signal drives
    # the decision. ``True`` → continue up to the budget; ``False`` → no
    # continuation (unless a dangling-heading structural override fires);
    # ``None`` → one safety-net pass, then stop. The 5th return element
    # (``stop_reason``) is the structured outcome surfaced in
    # ``OperationResult.extra["continuation"]`` so the kernel and the
    # cap-pause diagnostic can see *why* the continuation stopped.
    (
        document,
        was_truncated,
        continue_attempts,
        final_finish_reason,
        continuation_stop_reason,
    ) = await continue_until_complete(
        ctx,
        user_input=state.user_input,
        document=document,
        model_def=model_def,
        first_truncated=first_truncated,
        first_finish_reason=first_finish_reason,
        max_continue_attempts=cfg.max_continue_attempts,
        max_continue_tokens=state.first_pass_continue_tokens,
        continue_prompt_path=_CONTINUE_PROMPT_PATH,
    )
    state.document = document
    state.was_truncated = was_truncated
    state.continue_attempts = continue_attempts
    state.final_finish_reason = final_finish_reason
    state.continuation_stop_reason = continuation_stop_reason
    return None


async def _run_outline_contract(
    ctx: OperationExecutionContext,
    state: _ForgeState,
) -> None:
    """Phase 3 — the outline-contract gate (self-abridgment defense).

    The model may stop cleanly yet quietly omit planned sections (the
    2026-10-16 self-abridgment incident — ``truncated=False`` was
    truthful, the doc was still short). When a plan was declared and
    sections are missing, run the same number of bounded expand passes as
    the continuation budget allows, at the same per-pass budget the
    continuation path uses. Advisory: a document that is still missing
    sections persists with degraded status and telemetry.
    """
    cfg = state.cfg
    if not cfg.outline_contract:
        state.outline_contract_stop_reason = "disabled"
        return
    if not state.forge_plan:
        return
    planned_missing = missing_planned_sections(state.forge_plan, state.document)
    state.planned_missing = planned_missing
    if not planned_missing:
        state.outline_contract_stop_reason = "satisfied"
        return
    try:
        (
            document,
            outline_contract_attempts,
            _contract_incomplete,
        ) = await expand_missing_sections(
            ctx,
            user_input=state.user_input,
            document=state.document,
            planned=state.forge_plan,
            missing=planned_missing,
            model_def=state.model_def,
            max_tokens=state.first_pass_continue_tokens,
            max_attempts=cfg.outline_contract_attempts,
            expand_prompt_path=_EXPAND_PROMPT_PATH,
        )
        state.document = document
        state.outline_contract_attempts = outline_contract_attempts
        missing_plan_sections = missing_planned_sections(state.forge_plan, document)
        if not missing_plan_sections:
            state.outline_contract_stop_reason = "expanded"
        elif missing_plan_sections != planned_missing:
            state.outline_contract_stop_reason = "partial"
        else:
            state.outline_contract_stop_reason = "still_missing"
    except RuntimeError as exc:
        logger.warning(
            "forge_document: outline-contract expansion failed (%s); persisting without it",
            exc,
        )
        state.outline_contract_stop_reason = "error"
        missing_plan_sections = planned_missing
    state.missing_plan_sections = missing_plan_sections
    if missing_plan_sections:
        logger.warning(
            "forge_document: document still missing planned section(s) after expansion: %s",
            ", ".join(missing_plan_sections),
        )
        state.was_truncated = True
        state.continuation_stop_reason = (
            state.continuation_stop_reason
            if state.continuation_stop_reason not in ("completed", "not_needed")
            else "outline_contract"
        )
    logger.info(
        "forge.outline_contract: attempts=%d planned=%d "
        "missing_before=%d missing_after=%d stop_reason=%s",
        state.outline_contract_attempts,
        len(state.forge_plan),
        len(state.planned_missing),
        len(state.missing_plan_sections),
        state.outline_contract_stop_reason,
    )


def _finalize_forged_document(ctx: OperationExecutionContext, state: _ForgeState) -> str:
    """Phase 4 — repair fences, format, cap, and annotate the document.

    Inline-fence repair runs BEFORE formatting: mdformat treats a closing
    fence appended to the last code line as legal CommonMark and preserves
    it, so the hygiene split must land first.

    The char cap floors at the existing document's size: a revision can
    never be smaller than what it revises. A 32.9k-char plan revision that
    synthesizes cleanly at 46.6k chars is legitimate *growth* — clamping
    it to a fixed 20k ceiling destroyed approved content mid-sentence and
    degraded the result to an unverified write (the 2026-10-22 forge
    char-cap incident). A fresh forge (no existing document) keeps the
    flat configured ceiling unchanged.
    """
    cfg = state.cfg
    document = repair_inline_fences(state.document)
    document = _format_markdown(document)
    effective_char_cap = max(cfg.max_document_chars, len(state.existing_document))
    if len(document) > effective_char_cap:
        document = _cap_document_at_heading(document, effective_char_cap)
        state.was_truncated = True
        state.continuation_stop_reason = "char_cap"
    annotate_model = bool(state.params.get("annotate_model", True))
    if annotate_model and not _user_asked_not_to_annotate(state.user_input):
        annotation = _build_model_annotation(
            state.model_def,
            truncated=state.was_truncated,
            final_finish_reason=state.final_finish_reason if state.was_truncated else None,
            continue_attempts=state.continue_attempts if state.was_truncated else 0,
            missing_sections=state.missing_plan_sections,
        )
        if annotation:
            document = document.rstrip() + "\n\n" + annotation + "\n"
    return document


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


async def _persist_zero_heading_document(
    ctx: OperationExecutionContext,
    state: _ForgeState,
    document: str,
    heading_count: int,
) -> OperationResult:
    """Diagnostic gate: persist a zero-heading document with a warning comment.

    The content survives so the user can refine; the explainer loop no
    longer burns iterations on a hard reject that leaves no durable
    artifact.
    """
    return await _persist_no_outline_document(
        ctx,
        document,
        state.document_path,
        heading_count=heading_count,
        was_truncated=state.was_truncated,
        continue_attempts=state.continue_attempts,
        first_finish_reason=state.first_finish_reason,
        final_finish_reason=state.final_finish_reason,
        continuation_stop_reason=state.continuation_stop_reason,
        forge_plan=state.forge_plan,
        planned_missing=state.planned_missing,
        outline_contract_attempts=state.outline_contract_attempts,
        outline_contract_stop_reason=state.outline_contract_stop_reason,
    )


async def _run_forge_lint_gate(
    ctx: OperationExecutionContext,
    *,
    state: _ForgeState,
    document: str,
    outline: dict[str, Any],
    heading_count: int,
) -> tuple[str, dict[str, Any], int, list[LintIssue], bool, OperationResult | None]:
    """Phase 5b — advisory lint gate with one optional repair pass.

    Lint is never a hard reject — the outline gate above is the only hard
    gate. When issues are found and ``forge_lint_repair`` is set, issue
    one repair pass feeding the lint issues back to the model; re-format
    and re-validate. Any issues that remain (or when repair is disabled)
    are persisted as a non-rendering HTML comment so the user knows the
    artifact has warnings.

    Returns ``(document, outline, heading_count, issues, repair_used,
    failure)``.
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
            max_tokens=state.first_pass_max_tokens,
        )
        if repaired != document:
            repair_used = True
            # The repair pass regenerated the text, so re-run the fence
            # hygiene split before formatting (same order rationale as
            # the primary pass above).
            document = repair_inline_fences(repaired)
            document = _format_markdown(document)
            # Re-validate the repaired document's outline.
            new_outline, outline_error = await _outline_for_document(ctx, document)
            if outline_error is not None:
                return (
                    document,
                    new_outline,
                    heading_count,
                    issues,
                    repair_used,
                    OperationResult(output_text=outline_error),
                )
            outline = new_outline
            heading_count = int(outline.get("heading_count", 0) or 0)
            if heading_count < _MIN_HEADINGS:
                zero_heading = await _persist_zero_heading_document(
                    ctx, state, document, heading_count
                )
                return document, outline, heading_count, issues, repair_used, zero_heading
            issues = lint_markdown(document, rules_csv=cfg.lint_rules or None)
    if issues:
        comment = lint_summary_comment(issues)
        if comment:
            document = document.rstrip() + "\n\n" + comment + "\n"
    return document, outline, heading_count, issues, repair_used, None


async def _validate_forged_document(
    ctx: OperationExecutionContext,
    state: _ForgeState,
    document: str,
) -> tuple[str, dict[str, Any], int, list[LintIssue], bool, OperationResult | None]:
    """Phase 5 — the hard outline gate followed by the advisory lint gate."""
    outline, outline_error = await _outline_for_document(ctx, document)
    if outline_error is not None:
        return document, outline, 0, [], False, OperationResult(output_text=outline_error)
    heading_count = int(outline.get("heading_count", 0) or 0)
    if heading_count < _MIN_HEADINGS:
        zero_heading = await _persist_zero_heading_document(ctx, state, document, heading_count)
        return document, outline, heading_count, [], False, zero_heading
    return await _run_forge_lint_gate(
        ctx, state=state, document=document, outline=outline, heading_count=heading_count
    )


def _build_forge_summary(
    *,
    state: _ForgeState,
    stored_path: str,
    byte_count: int,
    heading_count: int,
    issues: list[LintIssue],
    repair_used: bool,
) -> str:
    """Build the user-facing summary of the forge result.

    A degraded result leads with the warning. The earlier shape opened with
    "Document forged and stored" and appended " WARNING:" in the same
    sentence, so a caller reading the first clause — or a model skimming the
    result — saw success and stopped there. The 2026-10-26 incident dropped
    five of eight planned sections and the summary still began with
    "Document forged and stored".
    """
    if state.was_truncated:
        if state.missing_plan_sections:
            summary = (
                f"INCOMPLETE: document persisted at `{stored_path}` "
                f"({byte_count} bytes, {heading_count} headings) but is missing "
                f"planned section(s) after {state.outline_contract_attempts} "
                f"expansion attempt(s): {', '.join(state.missing_plan_sections)}. "
                f"See annotation in document."
            )
        else:
            summary = (
                f"INCOMPLETE: document persisted at `{stored_path}` "
                f"({byte_count} bytes, {heading_count} headings) but was truncated "
                f"after {state.continue_attempts} continuation attempt(s) "
                f"(final finish_reason={state.final_finish_reason!r}). "
                f"See annotation in document."
            )
    else:
        summary = (
            f"Document forged and stored at `{stored_path}` "
            f"({byte_count} bytes, {heading_count} headings)."
        )
    if issues:
        summary += (
            f" Note: {len(issues)} structural lint warning(s) remain "
            f"after {'a' if repair_used else 'no'} repair pass; see lint comment in document."
        )
    return summary


async def _persist_forged_document(
    ctx: OperationExecutionContext,
    state: _ForgeState,
    *,
    document: str,
    outline: dict[str, Any],
    heading_count: int,
    issues: list[LintIssue],
    repair_used: bool,
) -> OperationResult:
    """Phase 6 — persist the document and report the result."""
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
    summary = _build_forge_summary(
        state=state,
        stored_path=stored_path,
        byte_count=byte_count,
        heading_count=heading_count,
        issues=issues,
        repair_used=repair_used,
    )
    return OperationResult(
        output_text=summary,
        output_json={
            "path": stored_path,
            "bytes": byte_count,
            "heading_count": heading_count,
            "outline": outline,
            "truncated": state.was_truncated,
            "continue_attempts": state.continue_attempts,
            "first_finish_reason": state.first_finish_reason,
            "final_finish_reason": state.final_finish_reason,
            "lint_warnings": len(issues),
            "lint_repair_used": repair_used,
            "target_words": state.target_words,
            "missing_plan_sections": list(state.missing_plan_sections),
            # A caller must be able to branch on completeness without parsing
            # the human summary. ``truncated`` covers both a provider cut and
            # missing planned sections.
            "complete": not state.was_truncated,
        },
        status="degraded" if state.was_truncated else "success",
        error_detail=(
            f"continuation {state.continuation_stop_reason}: "
            f"{state.continue_attempts} attempt(s), finish_reason={state.final_finish_reason!r}"
            if state.was_truncated
            else None
        ),
        extra={
            "memory_context_lines": state.memory_context.count("\n") + 1
            if state.memory_context
            else 0,
            "document_path": stored_path,
            "truncated": state.was_truncated,
            "continue_attempts": state.continue_attempts,
            "lint_warnings": len(issues),
            "target_words": state.target_words,
            "outline_contract": {
                "planned": list(state.forge_plan),
                "missing_before": list(state.planned_missing) if state.forge_plan else [],
                "missing_after": list(state.missing_plan_sections),
                "attempts": state.outline_contract_attempts,
                "stop_reason": state.outline_contract_stop_reason,
            },
            "continuation": {
                "was_truncated": state.was_truncated,
                "attempts": state.continue_attempts,
                "stop_reason": state.continuation_stop_reason,
                "first_finish_reason": state.first_finish_reason,
                "final_finish_reason": state.final_finish_reason,
            },
        },
    )


#: Parameters that, together, identify what a caller wants forged. A call
#: carrying none of them is not a request for a document.
_FORGING_INTENT_PARAMS: tuple[str, ...] = (
    "query",
    "existing_document",
    "document_path",
    "target_length",
)


def _reject_unstated_invocation(
    ctx: OperationExecutionContext,
    params: dict[str, Any],
) -> OperationResult | None:
    """Refuse a call that names nothing to forge.

    Every ``Parameters`` field has a default, so a zero-argument call is
    schema-valid and used to fall through to the ambient ``ctx.query`` — which
    forged 2,450 bytes of invented content and overwrote ``docs/forge.md``
    with no one having asked for it (the 2026-10-26 forge incident).

    This is deliberately *not* an overwrite guard — overwriting is handled by
    ``_load_existing_document``, which refuses an unnamed default-path
    replacement. Revising an existing document by omitting
    ``existing_document`` is the documented iteration contract and still
    works: the auto-load reads the file first. What is refused here is an
    invocation that states no intent at all, because there is nothing to
    forge *from* and nothing the caller asked for.

    Returns a terminal ``OperationResult`` when the call states no intent.
    """
    if any(name in params for name in _FORGING_INTENT_PARAMS):
        return None
    if str(getattr(ctx, "query", "") or "").strip():
        # The host passed the turn's request as ambient context, which is the
        # ordinary "the operation serves this turn" shape.
        return None
    return OperationResult(
        output_text=_load_text(ctx, _NO_INPUT_PATH),
        status="error",
        error_detail=(
            "refused a forge invocation carrying no query, existing_document, "
            "document_path, or target_length and no ambient query"
        ),
    )


async def execute(ctx: OperationExecutionContext) -> OperationResult:
    """Forge a markdown document from user input + recalled memory and persist it.

    Orchestrates the six pipeline phases — prepare (resolve settings and
    the existing document), forge (call the model and continue until the
    document is complete), outline contract (expand planned-but-missing
    sections), finalize (format, cap, annotate), validate (outline +
    advisory lint), and persist. Each phase lives in its own helper; this
    function only wires them together.
    """
    unstated = _reject_unstated_invocation(ctx, ctx.params or {})
    if unstated is not None:
        return unstated
    state, failure = await _prepare_forge(ctx)
    if failure is not None:
        return failure
    failure = await _load_existing_document(ctx, state)
    if failure is not None:
        return failure
    recall_limit = int(state.params.get("limit") or state.cfg.max_recall_records)
    state.memory_context = await _recall_memory(ctx, state.user_input, limit=recall_limit)
    _resolve_forge_budgets(state)
    failure = await _forge_document_body(ctx, state)
    if failure is not None:
        return failure
    await _run_outline_contract(ctx, state)
    document = _finalize_forged_document(ctx, state)
    (
        document,
        outline,
        heading_count,
        issues,
        repair_used,
        failure,
    ) = await _validate_forged_document(ctx, state, document)
    if failure is not None:
        return failure
    return await _persist_forged_document(
        ctx,
        state,
        document=document,
        outline=outline,
        heading_count=heading_count,
        issues=issues,
        repair_used=repair_used,
    )
