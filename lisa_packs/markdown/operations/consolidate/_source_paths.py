"""Source-document ingestion for multi-source consolidation forging.

Parses the ``source_paths`` parameter into an ordered list of source
specs, expands globs via the ``list_files`` storage tool, reads each
source via ``read_file``, and renders them (with per-source NL priority)
into a context block for injection into the forge prompt.

Contract (see docs/lessons + the consolidation design, 2026-07-30):

- ``source_paths`` entries are either bare path/glob strings or dicts
  with ``path`` + free-text ``priority``.
- Priority is NL prose, passed through verbatim — no parsing, no
  normalization into an enum. The user expresses "keep verbatim",
  "dedupe against source N", "background, one paragraph max", etc.
- Globs (paths containing ``*`` or ``?``) are expanded via
  ``list_files(prefix)``; expanded entries inherit the entry's priority
  (or ``normal`` default when the entry was a bare glob string).
- Reading failures are per-source: a missing/unreadable source is
  recorded with a ``read_error`` and skipped from the rendered block,
  not fatal. The provenance record (OKF concept, commit 3) surfaces
  these so the user knows what was dropped.

This module never touches the filesystem directly — every read/list goes
through ``ctx.execute_tool_fn`` routed to the ``lisa-storage`` MCP server,
matching the pack boundary.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

from kernel.models import OperationExecutionContext, ToolExecutionContext, ToolResult

logger = logging.getLogger(__name__)

__all__ = [
    "MAX_PER_SOURCE_CHARS",
    "SourceSpec",
    "expand_globs",
    "has_sources",
    "parse_source_paths",
    "read_source_chunks",
    "read_sources",
    "render_one_source_block",
    "render_priorities_block",
    "render_source_block",
    "stat_source_chars",
    "total_source_bytes",
]

# Tool ids routed through ctx.execute_tool_fn (lisa-storage MCP server).
_LIST_FILES_TOOL_ID = "list_files"
_READ_FILE_TOOL_ID = "read_file"
# Storage tools shipped in Phase 1 (commits 126e7b5/2695234), used by the
# Phase 3 rolling-accumulation path to chunk a single oversize source.
_STAT_FILE_TOOL_ID = "stat_file"
_READ_FILE_RANGE_TOOL_ID = "read_file_range"

# Caps that keep a single runaway source from blowing the model context.
_MAX_PER_SOURCE_CHARS = 32_000
_MAX_TOTAL_SOURCE_CHARS = 240_000
_DEFAULT_PRIORITY = "normal"

# Public alias for the per-source char cap, re-exported so the
# rolling-accumulation gate (Phase 3) can single-source the threshold from
# this module rather than re-declaring it in forge_document.py.
MAX_PER_SOURCE_CHARS = _MAX_PER_SOURCE_CHARS


@dataclass
class SourceSpec:
    """One resolved source document entry.

    ``priority`` is the user's NL prose, verbatim. ``text``/``bytes`` are
    populated by :func:`read_sources`; ``read_error`` is set when a read
    failed (the spec is then excluded from the rendered block but kept in
    the list for provenance).
    """

    path: str
    priority: str = _DEFAULT_PRIORITY
    text: str = ""
    bytes: int = 0
    read_error: str = ""
    # True when this spec was produced by glob expansion (not an explicit
    # user entry). Used by provenance to distinguish "user listed X" from
    # "user globbed *.md and got X".
    expanded_from_glob: bool = False
    # The original glob string this spec was expanded from, when applicable.
    origin_glob: str = ""
    # Whether the source was truncated to fit _MAX_PER_SOURCE_CHARS.
    truncated_to_fit: bool = False
    # Extra metadata slots populated by later phases (commit 3 provenance).
    extra: dict[str, Any] = field(default_factory=dict)


def has_sources(params: dict[str, Any]) -> bool:
    """Return True when the caller supplied a non-empty ``source_paths``."""

    raw = params.get("source_paths")
    if raw is None:
        return False
    if isinstance(raw, str):
        return bool(raw.strip())
    if isinstance(raw, (list, tuple)):
        return any(_entry_path(e) for e in raw)
    return False


def parse_source_paths(params: dict[str, Any]) -> list[SourceSpec]:
    """Validate and normalize the ``source_paths`` parameter into specs.

    Accepts:
      - a list of strings (paths/globs, default priority ``normal``)
      - a list of dicts with ``path`` (required) and ``priority`` (optional NL)
      - a single path/glob string (convenience)

    Rejects absolute paths and parent-traversal segments early, mirroring
    :func:`_extract_document_path` in ``forge_document.py``. The storage
    server re-validates, but failing early yields a clearer error and keeps
    sandbox-escape attempts out of the model context.

    Order is preserved exactly as supplied — the user's relative
    prioritization is expressed by ordering + the per-entry ``priority``
    note; both reach the model verbatim.
    """

    raw = params.get("source_paths")
    if raw is None or raw == "":
        return []
    if isinstance(raw, str):
        raw = [raw]
    if not isinstance(raw, (list, tuple)):
        raise RuntimeError("source_paths must be a list of paths or {path, priority} dicts")

    specs: list[SourceSpec] = []
    for entry in raw:
        path = _entry_path(entry)
        if not path:
            continue
        _reject_unsafe_path(path)
        priority = _entry_priority(entry)
        specs.append(SourceSpec(path=path, priority=priority))
    return specs


def _entry_path(entry: Any) -> str:
    if isinstance(entry, str):
        return entry.strip()
    if isinstance(entry, dict):
        return str(entry.get("path") or "").strip()
    return ""


def _entry_priority(entry: Any) -> str:
    if isinstance(entry, dict):
        return str(entry.get("priority") or "").strip() or _DEFAULT_PRIORITY
    return _DEFAULT_PRIORITY


def _reject_unsafe_path(path: str) -> None:
    normalized = path.replace("\\", "/").lstrip("/")
    if normalized.startswith("/") or (":/" in normalized and not _is_relative_glob(normalized)):
        raise RuntimeError(f"source_paths entry may not be absolute: {path!r}")
    if ".." in normalized.split("/"):
        raise RuntimeError(f"source_paths entry may not contain '..' segments: {path!r}")


def _is_relative_glob(path: str) -> bool:
    """Heuristic: treat a path as a relative glob if it contains wildcard chars."""
    return bool(re.search(r"[*?\[\]]", path))


async def expand_globs(
    specs: list[SourceSpec],
    *,
    ctx: OperationExecutionContext,
) -> list[SourceSpec]:
    """Expand glob entries via the ``list_files`` storage tool.

    Bare paths pass through unchanged. Glob entries (containing ``*`` or
    ``?``) are expanded by calling ``list_files`` with the literal prefix
    up to the first wildcard. Expanded entries inherit the glob entry's
    ``priority`` and are flagged ``expanded_from_glob=True`` with
    ``origin_glob`` set.

    Failures to list a glob degrade gracefully: the glob is dropped with a
    warning rather than aborting the forge (a mis-typed glob should not
    block an otherwise-valid consolidation). The dropped glob is retained
    as a spec with ``read_error`` set so provenance can flag it.
    """

    if ctx.host is None:
        raise RuntimeError("source_paths requires execute_tool_fn to call list_files/read_file")

    resolved: list[SourceSpec] = []
    for spec in specs:
        if not _is_relative_glob(spec.path):
            resolved.append(spec)
            continue
        prefix = spec.path[: _first_wildcard_index(spec.path)]
        matches = await _list_files(ctx, prefix)
        if not matches:
            logger.warning(
                "source_paths: glob %r expanded to zero matches (prefix=%r)",
                spec.path,
                prefix,
            )
            resolved.append(spec)  # keep for provenance; read_sources will flag
            continue
        for match in matches:
            child = SourceSpec(
                path=match,
                priority=spec.priority,
                expanded_from_glob=True,
                origin_glob=spec.path,
            )
            resolved.append(child)
    return resolved


def _first_wildcard_index(path: str) -> int:
    for i, ch in enumerate(path):
        if ch in "*?[":
            return i
    return len(path)


async def _list_files(
    ctx: OperationExecutionContext,
    prefix: str,
) -> list[str]:
    """Call the ``list_files`` storage tool and return sorted match paths.

    Returns an empty list on any tool error (degrade gracefully). The
    storage server returns a JSON list of relative paths; non-JSON or
    unexpected shapes also degrade to empty.
    """

    import json

    try:
        result = await ctx.host.execute_tool(
            _LIST_FILES_TOOL_ID,
            ToolExecutionContext(
                tool_id=_LIST_FILES_TOOL_ID,
                workspace_id=ctx.workspace_id,
                user_id=ctx.user_id,
                user_slug=ctx.user_slug,
                surface_type=ctx.surface_type,
                surface_id=ctx.surface_id,
                memory_scope=ctx.memory_scope,
                mode_id=ctx.mode_id,
                parameters={"prefix": prefix},
            ),
        )
    except Exception as exc:  # pylint: disable=broad-except
        logger.warning("source_paths: list_files(prefix=%r) raised: %s", prefix, exc)
        return []

    error_text = getattr(result, "error", "") or ""
    if error_text:
        logger.warning("source_paths: list_files(prefix=%r) error: %s", prefix, error_text)
        return []
    output_text = getattr(result, "output_text", "") or ""
    try:
        data = json.loads(output_text)
    except (json.JSONDecodeError, TypeError):
        return []
    if isinstance(data, list):
        paths = [str(p) for p in data if isinstance(p, str) and p]
        return sorted(paths)
    return []


async def read_sources(
    specs: list[SourceSpec],
    *,
    ctx: OperationExecutionContext,
) -> list[SourceSpec]:
    """Read each source via ``read_file``, populating ``text``/``bytes``.

    Per-source failure is non-fatal: a missing/unreadable source keeps its
    place in the list with ``read_error`` set and ``text`` empty, so the
    provenance record can flag it. Sources exceeding
    ``_MAX_PER_SOURCE_CHARS`` are truncated to fit and flagged
    ``truncated_to_fit``.

    The aggregate is also capped at ``_MAX_TOTAL_SOURCE_CHARS``: once the
    running total exceeds the budget, remaining sources are marked with a
    "skipped: aggregate context budget exceeded" read_error and not read.
    This prevents a huge source set from blowing the model context window
    and silently producing a truncated forge. The cap is deliberately
    large (240K chars ≈ 60K tokens) so the typical consolidation fits in
    one pass; the map-reduce path (future work) handles the oversize case.
    """

    if ctx.host is None:
        raise RuntimeError("source_paths requires execute_tool_fn to call read_file")

    running_total = 0
    for spec in specs:
        if spec.read_error:
            continue
        if running_total >= _MAX_TOTAL_SOURCE_CHARS:
            spec.read_error = "skipped: aggregate source context budget exceeded"
            spec.extra["budget_skipped"] = True
            continue
        text, err = await _read_one(ctx, spec.path)
        if err is not None:
            spec.read_error = err
            continue
        if len(text) > _MAX_PER_SOURCE_CHARS:
            text = text[:_MAX_PER_SOURCE_CHARS] + "\n\n<!-- truncated to fit per-source cap -->\n"
            spec.truncated_to_fit = True
        spec.text = text
        spec.bytes = len(text.encode("utf-8"))
        running_total += spec.bytes
    return specs


async def _read_one(
    ctx: OperationExecutionContext,
    path: str,
) -> tuple[str, str | None]:
    """Read one file via ``read_file``. Returns ``(text, error_or_None)``."""

    try:
        result: ToolResult = await ctx.host.execute_tool(
            _READ_FILE_TOOL_ID,
            ToolExecutionContext(
                tool_id=_READ_FILE_TOOL_ID,
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
    except Exception as exc:  # pylint: disable=broad-except
        return "", f"read raised: {exc}"
    error_text = getattr(result, "error", "") or ""
    if error_text:
        return "", f"read_file error: {error_text}"
    output_text = getattr(result, "output_text", "") or ""
    # lisa-storage read_file returns either raw text or a JSON error object
    # ({"error": "..."}). Detect the JSON-error shape and surface it.
    import json

    stripped = output_text.strip()
    if stripped.startswith("{") and '"error"' in stripped:
        try:
            data = json.loads(stripped)
            if isinstance(data, dict) and data.get("error"):
                return "", f"read_file error: {data['error']}"
        except json.JSONDecodeError:
            pass
    return output_text, None


def render_source_block(specs: list[SourceSpec]) -> str:
    """Render the ``## Source Documents`` context block for the forge prompt.

    Only successfully-read sources are included (those with non-empty
    ``text``). Each is rendered with its 1-based index, its path, its
    priority note verbatim, and its full text in a fenced markdown block.
    Per-source truncation (to fit the per-source cap) is annotated inline
    so the model knows it is seeing a prefix, not the whole document.

    The structure is deliberately stable so the model can reference
    "Source 2" in its reasoning; the priority prose sits adjacent to the
    text so the preservation directive is scoped to exactly the source it
    governs.
    """

    readable = [s for s in specs if s.text]
    if not readable:
        return ""
    lines: list[str] = ["## Source Documents", ""]
    for idx, spec in enumerate(readable, start=1):
        lines.append(f"### Source {idx} — `{spec.path}` (priority: {spec.priority})")
        if spec.truncated_to_fit:
            lines.append(
                "> NOTE: this source was truncated to fit the per-source"
                " context cap; what follows is the leading portion only."
            )
        lines.append("")
        lines.append("```markdown")
        lines.append(spec.text.rstrip())
        lines.append("```")
        lines.append("")
    return "\n".join(lines)


def total_source_bytes(specs: list[SourceSpec]) -> int:
    """Sum the bytes of successfully-read sources (pre-budget-skip)."""
    return sum(s.bytes for s in specs if s.text)


def readable_count(specs: list[SourceSpec]) -> int:
    """Count sources that were actually read into the prompt."""
    return sum(1 for s in specs if s.text)


def failed_specs(specs: list[SourceSpec]) -> list[SourceSpec]:
    """Return sources that were supplied but could not be read."""
    return [s for s in specs if s.read_error]


def render_one_source_block(spec: SourceSpec) -> str:
    """Render ONE source's text block for the per-source forge prompt.

    Used by the map-reduce consolidation path (Phase 2): each source gets
    its own bounded-context forge pass. Unlike :func:`render_source_block`,
    this renders a single source's text in a fenced markdown block — the
    wrapper header (index, path, priority) is the prompt template's job,
    not this helper's.
    """

    if not spec.text:
        return ""
    lines: list[str] = []
    if spec.truncated_to_fit:
        lines.append(
            "> NOTE: this source was truncated to fit the per-source"
            " context cap; what follows is the leading portion only."
        )
        lines.append("")
    lines.append("```markdown")
    lines.append(spec.text.rstrip())
    lines.append("```")
    return "\n".join(lines)


def render_priorities_block(specs: list[SourceSpec]) -> str:
    """Render a compact per-source priority list for the merge prompt.

    Each readable source is one line: ``N. <path> — priority``. This
    travels into the merge pass so the model respects the priority notes
    when assembling the sections (e.g. "primary" sections anchor the
    structure, "background" sections stay condensed).
    """

    readable = [s for s in specs if s.text]
    if not readable:
        return "(no sources)"
    lines: list[str] = []
    for idx, spec in enumerate(readable, start=1):
        lines.append(f"{idx}. `{spec.path}` — {spec.priority}")
    return "\n".join(lines)


def _failed_chunk(spec: SourceSpec, error: str) -> list[SourceSpec]:
    """Build the single-element failure result for a chunked read."""
    failed = SourceSpec(path=spec.path, priority=spec.priority)
    failed.read_error = error
    return [failed]


def _single_chunk(spec: SourceSpec, text: str, *, truncated_to_fit: bool) -> list[SourceSpec]:
    """Build the single-chunk result used by the no-stat and empty-source paths."""
    chunk = SourceSpec(
        path=spec.path,
        priority=spec.priority,
        text=text,
        bytes=len(text.encode("utf-8")),
        truncated_to_fit=truncated_to_fit,
    )
    chunk.extra["chunk_index"] = 0
    chunk.extra["chunk_total"] = 1
    chunk.extra["chunk_eof"] = True
    return [chunk]


async def _read_whole_source_as_chunk(
    spec: SourceSpec,
    *,
    ctx: OperationExecutionContext,
    read_cap: int,
) -> list[SourceSpec]:
    """Degrade to one ``read_file`` chunk when ``stat_file`` is unavailable.

    Mirrors :func:`read_sources`'s per-source failure contract: an
    unreadable source yields a single-element list carrying the
    ``read_error`` so the caller can surface it rather than aborting the
    forge.
    """
    text, err = await _read_one(ctx, spec.path)
    if err is not None:
        return _failed_chunk(spec, err)
    if len(text) > read_cap:
        text = text[:read_cap] + "\n\n<!-- truncated to fit per-source cap -->\n"
    return _single_chunk(spec, text, truncated_to_fit=len(text) >= read_cap)


async def _read_chunk_windows(
    spec: SourceSpec,
    *,
    ctx: OperationExecutionContext,
    chunk_chars: int,
    read_cap: int,
    stat_total: int,
) -> list[SourceSpec]:
    """Read successive ``read_file_range`` windows until ``eof``.

    We read up to ``read_cap`` chars total; each window is ``chunk_chars``
    wide (the last may be shorter). ``read_file_range`` is char-based and
    reports ``eof`` when the window reaches/passes the total.
    """
    total_to_read = min(stat_total, read_cap)
    chunks: list[SourceSpec] = []
    offset = 0
    chunk_index = 0
    while offset < total_to_read:
        window = min(chunk_chars, total_to_read - offset)
        content, eof, err = await _read_file_range(ctx, spec.path, offset, window)
        if err is not None:
            return _failed_chunk(spec, err)
        if not content and chunk_index > 0:
            # No content returned mid-stream — stop to avoid an infinite loop.
            break
        chunk = SourceSpec(
            path=spec.path,
            priority=spec.priority,
            text=content,
            bytes=len(content.encode("utf-8")),
            truncated_to_fit=(offset + window) >= read_cap
            and not eof
            and total_to_read < stat_total,
        )
        chunk.extra["chunk_index"] = chunk_index
        chunk.extra["chunk_eof"] = eof
        chunks.append(chunk)
        chunk_index += 1
        if eof:
            break
        # Defensive: ensure forward progress (read_file_range always
        # returns >= 0 chars; a zero-length non-eof window would loop
        # forever, so stop if the backend stalls).
        if len(content) == 0:
            break
        offset += len(content)
    return chunks


async def read_source_chunks(
    spec: SourceSpec,
    *,
    ctx: OperationExecutionContext,
    chunk_chars: int = 8000,
) -> list[SourceSpec]:
    """Chunk ONE oversize source into bounded windows via ``read_file_range``.

    Phase 3 rolling-accumulation helper (added 2026-08-02). Calls
    ``stat_file`` to get the total char length, then plans chunk count from
    ``chunk_chars`` and reads successive windows via ``read_file_range``
    until ``eof``. Each window is returned as a ``SourceSpec`` reusing the
    original ``path``/``priority`` with the chunk text in ``text``, its byte
    count in ``bytes``, and a ``chunk_index`` (0-based) + ``chunk_total`` in
    ``extra``. The last chunk carries ``extra["chunk_eof"] = True``.

    Respects ``_MAX_PER_SOURCE_CHARS`` by default: windows are not read
    beyond the per-source cap unless ``chunk_chars`` is raised above it.

    This goes through ``ctx.execute_tool_fn`` routed to ``lisa-storage``
    (same pattern as :func:`read_sources`) — no direct filesystem access.
    Per the pack boundary, all chunking policy stays in the pack; the
    storage server only returns bytes/chars.

    Failures degrade gracefully: a ``stat_file`` or ``read_file_range``
    error returns a single-element list with the spec's ``read_error`` set
    (mirroring :func:`read_sources`'s per-source failure contract) so the
    caller can surface it rather than aborting the forge.
    """
    if ctx.host is None:
        raise RuntimeError(
            "read_source_chunks requires execute_tool_fn to call stat_file/read_file_range"
        )
    if chunk_chars <= 0:
        chunk_chars = 8000

    # Cap the total we will read at the per-source cap unless the caller
    # explicitly raised chunk_chars above it (opt-in to read more).
    read_cap = _MAX_PER_SOURCE_CHARS
    if chunk_chars > _MAX_PER_SOURCE_CHARS:
        read_cap = chunk_chars

    # Plan from stat_file (char total). Fall back to reading the whole
    # source via read_file if stat is unavailable (degrade gracefully).
    stat_total = await _stat_file_chars(ctx, spec.path)
    if stat_total is None:
        return await _read_whole_source_as_chunk(spec, ctx=ctx, read_cap=read_cap)

    if stat_total <= 0:
        # Empty source — one empty chunk with eof.
        return _single_chunk(spec, "", truncated_to_fit=False)

    chunks = await _read_chunk_windows(
        spec,
        ctx=ctx,
        chunk_chars=chunk_chars,
        read_cap=read_cap,
        stat_total=stat_total,
    )
    # Finalize chunk_total now that we know the count.
    chunk_total = len(chunks)
    for c in chunks:
        c.extra["chunk_total"] = chunk_total
    if chunks:
        chunks[-1].extra["chunk_eof"] = True
    return chunks


async def _stat_file_chars(
    ctx: OperationExecutionContext,
    path: str,
) -> int | None:
    """Call ``stat_file`` and return the file's char/byte size, or None on error.

    ``stat_file`` returns ``size`` in bytes; for text sources byte length ≈
    char length for the chunk-planning purpose (the chunk window is a soft
    cap, and read_file_range is char-based so a slight miscount just shifts
    the last window). Returns None on any tool/JSON error so the caller
    degrades to a single read_file chunk.
    """
    import json

    try:
        result = await ctx.host.execute_tool(
            _STAT_FILE_TOOL_ID,
            ToolExecutionContext(
                tool_id=_STAT_FILE_TOOL_ID,
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
    except Exception as exc:  # pylint: disable=broad-except
        logger.warning("source_paths: stat_file(path=%r) raised: %s", path, exc)
        return None
    error_text = getattr(result, "error", "") or ""
    if error_text:
        logger.warning("source_paths: stat_file(path=%r) error: %s", path, error_text)
        return None
    output_text = getattr(result, "output_text", "") or ""
    try:
        data = json.loads(output_text)
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(data, dict) or data.get("error"):
        return None
    try:
        return int(data.get("size") or 0)
    except (TypeError, ValueError):
        return None


async def stat_source_chars(
    spec: SourceSpec,
    *,
    ctx: OperationExecutionContext,
) -> int | None:
    """Public wrapper around :func:`_stat_file_chars` for a SourceSpec.

    Returns the file's byte size (≈ char count for text) via the
    ``stat_file`` storage tool, or ``None`` on any tool/JSON error. Used by
    the Phase 3 rolling-accumulation gate to decide whether a single source
    exceeds ``MAX_PER_SOURCE_CHARS`` (note: read_sources truncates
    ``spec.bytes`` to the cap, so the gate must stat the real file size
    rather than inspect ``spec.bytes``).
    """
    return await _stat_file_chars(ctx, spec.path)


async def _read_file_range(
    ctx: OperationExecutionContext,
    path: str,
    start: int,
    length: int,
) -> tuple[str, bool, str | None]:
    """Call ``read_file_range``. Returns ``(content, eof, error_or_None)``.

    ``read_file_range`` returns a JSON object with ``content``, ``start``,
    ``length``, ``total``, and ``eof``. We surface ``content`` and ``eof``
    to the chunk planner; ``error`` is set on any tool/JSON failure.
    """
    import json

    try:
        result = await ctx.host.execute_tool(
            _READ_FILE_RANGE_TOOL_ID,
            ToolExecutionContext(
                tool_id=_READ_FILE_RANGE_TOOL_ID,
                workspace_id=ctx.workspace_id,
                user_id=ctx.user_id,
                user_slug=ctx.user_slug,
                surface_type=ctx.surface_type,
                surface_id=ctx.surface_id,
                memory_scope=ctx.memory_scope,
                mode_id=ctx.mode_id,
                parameters={"path": path, "start": start, "length": length},
            ),
        )
    except Exception as exc:  # pylint: disable=broad-except
        return "", False, f"read_file_range raised: {exc}"
    error_text = getattr(result, "error", "") or ""
    if error_text:
        return "", False, f"read_file_range error: {error_text}"
    output_text = getattr(result, "output_text", "") or ""
    try:
        data = json.loads(output_text)
    except (json.JSONDecodeError, TypeError):
        return "", False, "read_file_range returned non-JSON"
    if isinstance(data, dict) and data.get("error"):
        return "", False, f"read_file_range error: {data['error']}"
    if not isinstance(data, dict):
        return "", False, "read_file_range returned non-object"
    content = str(data.get("content") or "")
    eof = bool(data.get("eof") or False)
    return content, eof, None
